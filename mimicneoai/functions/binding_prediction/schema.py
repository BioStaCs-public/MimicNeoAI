"""Common schemas and file helpers for HLA binding prediction."""

from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


PREDICTION_FIELDS = [
    "peptide",
    "hla_allele",
    "algorithm",
    "mhc_class",
    "peptide_length",
    "ic50",
    "ic50_type",
    "score",
    "score_type",
    "percentile",
    "percentile_type",
    "raw_rank",
    "raw_rank_type",
    "status",
    "raw_file",
    "error",
]


@dataclass(frozen=True)
class BindingTask:
    """One peptide-HLA-algorithm prediction request."""

    peptide: str
    hla_allele: str
    algorithm: str
    mhc_class: str

    @property
    def peptide_length(self) -> int:
        return len(self.peptide)


@dataclass(frozen=True)
class PredictionJob:
    """A chunk of homogeneous binding prediction tasks."""

    algorithm: str
    mhc_class: str
    hla_allele: str
    peptide_length: int
    chunk_index: int
    peptides: tuple[str, ...]
    outdir: str
    output_algorithms: Tuple[str, ...] = ()

    @property
    def key_dir(self) -> Path:
        return (
            Path(self.outdir)
            / self.output_label
            / sanitize_path_component(self.hla_allele)
            / str(self.peptide_length)
        )

    @property
    def output_label(self) -> str:
        if not self.output_algorithms or self.output_algorithms == (self.algorithm,):
            return self.algorithm
        return self.algorithm + "__" + sanitize_path_component(",".join(self.output_algorithms))

    @property
    def raw_path(self) -> Path:
        return self.key_dir / "raw" / f"chunk_{self.chunk_index:05d}.raw.tsv"

    @property
    def normalized_path(self) -> Path:
        return self.key_dir / "normalized" / f"chunk_{self.chunk_index:05d}.predictions.tsv"

    @property
    def log_path(self) -> Path:
        return self.key_dir / "logs" / f"chunk_{self.chunk_index:05d}.log"

    @property
    def input_path(self) -> Path:
        return self.key_dir / "inputs" / f"chunk_{self.chunk_index:05d}.peptides.txt"


@dataclass
class BindingPrediction:
    """Normalized predictor output."""

    peptide: str
    hla_allele: str
    algorithm: str
    mhc_class: str
    peptide_length: int
    ic50: str = ""
    ic50_type: str = ""
    score: str = ""
    score_type: str = ""
    percentile: str = ""
    percentile_type: str = ""
    raw_rank: str = ""
    raw_rank_type: str = ""
    status: str = "ok"
    raw_file: str = ""
    error: str = ""

    def to_row(self) -> dict[str, str]:
        row = asdict(self)
        return {field: stringify(row.get(field, "")) for field in PREDICTION_FIELDS}


def stringify(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def sanitize_path_component(value: str) -> str:
    """Make an HLA or algorithm string safe for nested output paths."""

    value = value.strip().replace("*", "_").replace(":", "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "NA"


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except ValueError:
        return None


def read_binding_tasks(path: Path, algorithms: Optional[set[str]] = None) -> list[BindingTask]:
    """Read the standard binding task table."""

    tasks: list[BindingTask] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"peptide", "hla_allele", "algorithm", "mhc_class"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            peptide = (row.get("peptide") or "").strip()
            if not peptide:
                continue
            algorithm = (row.get("algorithm") or "").strip()
            if algorithms and algorithm not in algorithms:
                continue
            tasks.append(
                BindingTask(
                    peptide=peptide,
                    hla_allele=(row.get("hla_allele") or "").strip(),
                    algorithm=algorithm,
                    mhc_class=(row.get("mhc_class") or "").strip(),
                )
            )
    return tasks


def write_prediction_rows(path: Path, predictions: Iterable[BindingPrediction]) -> None:
    """Write normalized prediction rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction.to_row())


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def chunked(values: list[str], chunk_size: int) -> Iterable[tuple[int, tuple[str, ...]]]:
    """Yield 1-based chunk indexes and tuple chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    for offset in range(0, len(values), chunk_size):
        yield offset // chunk_size + 1, tuple(values[offset : offset + chunk_size])
