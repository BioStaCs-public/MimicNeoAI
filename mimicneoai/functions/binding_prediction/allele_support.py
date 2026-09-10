"""Predictor allele support discovery from locally installed model data."""

from __future__ import annotations

import csv
import os
import pickle
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

from mimicneoai.functions.binding_prediction.adapters.base import (
    AdapterConfig,
    predictor_env,
)
from mimicneoai.functions.binding_prediction.schema import BindingTask


@dataclass(frozen=True)
class SupportCatalog:
    """Normalized allele keys and optional per-allele peptide lengths."""

    alleles: Optional[frozenset[str]]
    source: str
    lengths: Optional[Dict[str, frozenset[int]]] = None
    error: str = ""


class AlleleSupportMatrix:
    """Resolve support using the allele inventory shipped with each predictor.

    A catalog with ``alleles=None`` means support could not be discovered. In
    that case tasks remain runnable so an environment probe cannot silently
    remove biological candidates; the predictor adapter will retain any real
    execution failure in its normalized output.
    """

    def __init__(
        self,
        config: AdapterConfig,
        catalogs: Optional[Dict[str, SupportCatalog]] = None,
    ) -> None:
        self.config = config
        self._catalogs = dict(catalogs or {})

    def supports(self, task: BindingTask) -> Optional[bool]:
        if task.algorithm in {"MHCnuggetsI", "MHCnuggetsII"}:
            return self._mhcnuggets_supports(task)
        catalog = self.catalog(task.algorithm)
        if catalog.alleles is None:
            return None
        key = allele_key(task.hla_allele)
        if key not in catalog.alleles:
            return False
        if catalog.lengths is not None:
            allowed_lengths = catalog.lengths.get(key)
            if allowed_lengths is not None and task.peptide_length not in allowed_lengths:
                return False
        return True

    def catalog(self, algorithm: str) -> SupportCatalog:
        if algorithm not in self._catalogs:
            self._catalogs[algorithm] = self._load_catalog(algorithm)
        return self._catalogs[algorithm]

    def summary(self) -> Dict[str, Dict[str, object]]:
        result: Dict[str, Dict[str, object]] = {}
        for algorithm, catalog in sorted(self._catalogs.items()):
            result[algorithm] = {
                "source": catalog.source,
                "allele_count": len(catalog.alleles) if catalog.alleles is not None else None,
                "status": "ok" if catalog.alleles is not None else "unknown",
                "error": catalog.error,
            }
        return result

    def _load_catalog(self, algorithm: str) -> SupportCatalog:
        loaders: Dict[str, Callable[[], SupportCatalog]] = {
            "MHCflurry": self._load_mhcflurry,
            "MHCflurryEL": self._load_mhcflurry,
            "MHCnuggetsI": lambda: self._load_mhcnuggets("MHC-I"),
            "MHCnuggetsII": lambda: self._load_mhcnuggets("MHC-II"),
            "NetMHCpan": self._load_netmhcpan,
            "NetMHCpanEL": self._load_netmhcpan,
            "NetMHCIIpan": self._load_netmhciipan,
            "NetMHCIIpanEL": self._load_netmhciipan,
            "NNalign": self._load_nnalign,
            "SMM": lambda: self._load_iedb_mhci("smm"),
            "SMMPMBEC": lambda: self._load_iedb_mhci("smmpmbec"),
            "PickPocket": lambda: self._load_iedb_mhci("pickpocket"),
        }
        loader = loaders.get(algorithm)
        if loader is None:
            return SupportCatalog(None, "not_applicable", error="no allele catalog loader")
        try:
            return loader()
        except Exception as exc:  # noqa: BLE001 - discovery failure must be non-fatal
            return SupportCatalog(None, "discovery_failed", error=str(exc))

    def _load_mhcflurry(self) -> SupportCatalog:
        candidates = discover_mhcflurry_allele_files(
            self.config.mhcflurry_downloads_dir
        )
        if candidates:
            path = candidates[-1]
            with path.open(newline="") as handle:
                alleles = {
                    allele_key(row["allele"])
                    for row in csv.DictReader(handle)
                    if row.get("allele")
                }
            return SupportCatalog(frozenset(alleles), str(path))
        lines = run_inventory_command(
            [self.config.mhcflurry_predict_bin, "--list-supported-alleles"],
            env=predictor_env(self.config),
        )
        return catalog_from_lines(lines, f"{self.config.mhcflurry_predict_bin} --list-supported-alleles")

    def _load_netmhcpan(self) -> SupportCatalog:
        lines = run_inventory_command([self.config.netmhcpan_bin, "-listMHC"])
        return catalog_from_lines(lines, f"{self.config.netmhcpan_bin} -listMHC")

    def _load_netmhciipan(self) -> SupportCatalog:
        lines = run_inventory_command([self.config.netmhciipan_bin, "-list"])
        return catalog_from_lines(lines, f"{self.config.netmhciipan_bin} -list")

    def _load_mhcnuggets(self, mhc_class: str) -> SupportCatalog:
        root = Path(self.config.mhcnuggets_cwd) / "mhcnuggets"
        examples_path = root / "data" / "production" / "examples_per_allele.pkl"
        model_dir = root / "saves" / "production"
        examples = read_pickle(examples_path)
        model_keys = {
            allele_key(strip_mhcnuggets_model_suffix(path.stem))
            for path in model_dir.glob("*.h5")
        }
        all_alleles = {
            allele_key(value)
            for value in examples
            if value != "tuning_results" and allele_key(value) in model_keys
        }
        if mhc_class == "MHC-I":
            alleles = {value for value in all_alleles if re.match(r"^[ABCEFG]\d", value)}
        else:
            alleles = {value for value in all_alleles if value.startswith("D")}
        return SupportCatalog(frozenset(alleles), f"{examples_path};{model_dir}")

    def _mhcnuggets_supports(self, task: BindingTask) -> Optional[bool]:
        catalog = self.catalog(task.algorithm)
        if catalog.alleles is None:
            return None
        if task.algorithm == "MHCnuggetsI" and not 8 <= task.peptide_length <= 15:
            return False
        key = allele_key(task.hla_allele)
        if key in catalog.alleles:
            return True
        # MHCnuggets resolves untrained classical HLA loci to a closest trained
        # model. Non-classical loci without an exact model resolve to an empty
        # path in the upstream code and are therefore unsupported.
        if task.algorithm == "MHCnuggetsI":
            return bool(re.match(r"^[ABC]\d", key))
        return bool(re.match(r"^(DR|DPA|DQA)", key))

    def _load_nnalign(self) -> SupportCatalog:
        path = (
            Path(self.config.iedb_mhcii_cwd)
            / "methods"
            / "allele-info"
            / "allele_info"
            / "pickles"
            / "mhcii_info_dict.p"
        )
        data = read_pickle(path)
        alleles = data["method_allele_dict"]["nn_align-2.3"]
        return SupportCatalog(
            frozenset(allele_key(value) for value in alleles),
            f"{path}:method_allele_dict[nn_align-2.3]",
        )

    def _load_iedb_mhci(self, method: str) -> SupportCatalog:
        path = (
            Path(self.config.iedb_mhci_cwd)
            / "method"
            / "allele-info"
            / "allele_info"
            / "pickles"
            / "mhci_info_dict.p"
        )
        data = read_pickle(path)
        lengths = {
            allele_key(allele): frozenset(int(value) for value in allowed)
            for allele, allowed in data["method_allele_length_dict"][method].items()
        }
        return SupportCatalog(
            frozenset(lengths),
            f"{path}:method_allele_length_dict[{method}]",
            lengths=lengths,
        )


def allele_key(value: str) -> str:
    """Normalize predictor-specific HLA spelling to one comparison key."""

    normalized = str(value or "").strip().upper().replace("HLA-", "").replace("HLA_", "")
    return re.sub(r"[^A-Z0-9]", "", normalized)


def catalog_from_lines(lines: Iterable[str], source: str) -> SupportCatalog:
    alleles = set()
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#") or value.startswith("-") or " " in value:
            continue
        key = allele_key(value)
        if key:
            alleles.add(key)
    if not alleles:
        raise ValueError(f"no alleles parsed from {source}")
    return SupportCatalog(frozenset(alleles), source)


def run_inventory_command(
    command: list[str], env: Optional[dict[str, str]] = None
) -> list[str]:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"allele inventory command failed ({completed.returncode}): {' '.join(command)}")
    return completed.stdout.splitlines()


def discover_mhcflurry_allele_files(configured_root: str = "") -> list[Path]:
    roots = []
    configured_root = (
        configured_root or os.environ.get("MHCFLURRY_DOWNLOADS_DIR", "")
    ).strip()
    if configured_root:
        roots.append(Path(configured_root))
    roots.append(Path.home() / ".local" / "share" / "mhcflurry")
    paths = []
    for root in roots:
        paths.extend(root.glob("**/models_class1_presentation/models/affinity_predictor/allele_sequences.csv"))
    return sorted(set(paths), key=lambda path: (path.stat().st_mtime, str(path)))


def read_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted files from the local predictor installation


def strip_mhcnuggets_model_suffix(stem: str) -> str:
    for suffix in ("_BA_to_HLAp", "_BA"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem
