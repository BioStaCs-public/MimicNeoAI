"""Post hoc shared-HLA evidence from existing MimicNeoAI binding outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from ._io import read_tsv, require_tsv_columns, write_tsv
from ._models import InputEntry
from ._policy import (
    SHARED_CLASSICAL_HLA_I_BINDING_V1,
    SharedHlaBindingPolicy,
)


HLA_SUPPORT_FIELDS = [
    "candidate_id",
    "method_version",
    "patient_id",
    "source_pair",
    "source_a",
    "source_b",
    "peptide_a",
    "peptide_b",
    "peptide_length",
    "source_a_classical_hla_i_universe",
    "source_b_classical_hla_i_universe",
    "hla_input_contract_status",
    "source_a_supported_classical_hla_i",
    "source_b_supported_classical_hla_i",
    "shared_supported_classical_hla_i",
    "shared_supported_classical_hla_i_count",
    "binding_evidence_status",
    "binding_policy",
]


def _number(value: str) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _binding_columns(source: str) -> dict[str, str]:
    if source == "mutation-derived":
        return {
            "peptide": "MT Epitope Seq",
            "allele": "HLA Allele",
            "best_ic50": "Best MT IC50 Score",
            "median_ic50": "Median MT IC50 Score",
            "best_percentile": "Best MT Percentile",
            "median_percentile": "Median MT Percentile",
        }
    return {
        "peptide": "Epitope Seq",
        "allele": "HLA Allele",
        "best_ic50": "Best IC50 Score",
        "median_ic50": "Median IC50 Score",
        "best_percentile": "Best Percentile",
        "median_percentile": "Median Percentile",
    }


def _normalize_hla_i(value: str) -> str:
    allele = str(value).strip().upper().replace(" ", "")
    if allele.startswith(("A*", "B*", "C*")):
        allele = "HLA-" + allele
    return allele


def _passes_binding_policy(
    row: dict[str, str],
    columns: dict[str, str],
    policy: SharedHlaBindingPolicy,
) -> bool | None:
    values = {
        name: _number(row.get(column, ""))
        for name, column in columns.items()
        if name not in {"peptide", "allele"}
    }
    if any(value is None for value in values.values()):
        return None
    return bool(
        values["best_ic50"] < policy.maximum_best_ic50_nm
        and values["median_ic50"] < policy.maximum_median_ic50_nm
        and values["best_percentile"] < policy.maximum_best_percentile
        and values["median_percentile"] < policy.maximum_median_percentile
    )


def load_binding_support(
    entries: list[InputEntry],
    policy: SharedHlaBindingPolicy = SHARED_CLASSICAL_HLA_I_BINDING_V1,
) -> tuple[
    dict[tuple[str, str, str], set[str]],
    set[tuple[str, str, str]],
    dict[tuple[str, str], set[str]],
]:
    """Load passing alleles, evaluable peptides, and each source's HLA universe."""

    support: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    evaluable: set[tuple[str, str, str]] = set()
    hla_universes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entry in entries:
        if entry.binding_path is None:
            raise ValueError(
                f"binding_support.enabled requires binding_path for "
                f"{entry.patient_id}/{entry.antigen_source}"
            )
        columns = _binding_columns(entry.antigen_source)
        require_tsv_columns(entry.binding_path, columns.values())
        for _, row in read_tsv(entry.binding_path):
            peptide = str(row.get(columns["peptide"], "")).strip().upper()
            allele = _normalize_hla_i(row.get(columns["allele"], ""))
            if not peptide or not allele.startswith(policy.allowed_loci):
                continue
            hla_universes[(entry.patient_id, entry.antigen_source)].add(allele)
            passed = _passes_binding_policy(row, columns, policy)
            if passed is None:
                continue
            key = (entry.patient_id, entry.antigen_source, peptide)
            evaluable.add(key)
            if passed:
                support[key].add(allele)
    return dict(support), evaluable, dict(hla_universes)


def annotate_shared_hla_support(
    pair_path: Path,
    entries: list[InputEntry],
    output_path: Path,
    policy: SharedHlaBindingPolicy = SHARED_CLASSICAL_HLA_I_BINDING_V1,
) -> tuple[int, dict[str, int]]:
    """Annotate sequence pairs without changing their primary sequence call."""

    support, evaluable, hla_universes = load_binding_support(entries, policy)
    status_counts: dict[str, int] = defaultdict(int)

    def rows():
        for _, row in read_tsv(pair_path):
            key_a = (row["patient_id"], row["source_a"], row["peptide_a"])
            key_b = (row["patient_id"], row["source_b"], row["peptide_b"])
            universe_a = hla_universes.get((row["patient_id"], row["source_a"]), set())
            universe_b = hla_universes.get((row["patient_id"], row["source_b"]), set())
            hla_contract_status = (
                "classical_hla_i_universe_matched"
                if universe_a == universe_b
                else "classical_hla_i_universe_mismatch"
            )
            alleles_a = support.get(key_a, set())
            alleles_b = support.get(key_b, set())
            shared = alleles_a & alleles_b
            if hla_contract_status == "classical_hla_i_universe_mismatch":
                status = "binding_not_evaluable_hla_contract_mismatch"
            elif key_a not in evaluable or key_b not in evaluable:
                status = "binding_not_evaluable"
            elif shared:
                status = "binding_supported_sequence_mimicry_candidate"
            else:
                status = "no_shared_supported_classical_hla_i"
            status_counts[status] += 1
            yield {
                "candidate_id": row["candidate_id"],
                "method_version": row["method_version"],
                "patient_id": row["patient_id"],
                "source_pair": row["source_pair"],
                "source_a": row["source_a"],
                "source_b": row["source_b"],
                "peptide_a": row["peptide_a"],
                "peptide_b": row["peptide_b"],
                "peptide_length": row["peptide_length"],
                "source_a_classical_hla_i_universe": ";".join(sorted(universe_a)),
                "source_b_classical_hla_i_universe": ";".join(sorted(universe_b)),
                "hla_input_contract_status": hla_contract_status,
                "source_a_supported_classical_hla_i": ";".join(sorted(alleles_a)),
                "source_b_supported_classical_hla_i": ";".join(sorted(alleles_b)),
                "shared_supported_classical_hla_i": ";".join(sorted(shared)),
                "shared_supported_classical_hla_i_count": len(shared),
                "binding_evidence_status": status,
                "binding_policy": policy.name,
            }

    count = write_tsv(output_path, HLA_SUPPORT_FIELDS, rows())
    return count, dict(status_counts)
