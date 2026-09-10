"""Internal data models for the mimicry package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SUPPORTED_SOURCES = ("microbial", "cryptic", "mutation-derived")
SUPPORTED_SOURCE_PAIRS = (
    "microbial--mutation-derived",
    "microbial--cryptic",
    "cryptic--mutation-derived",
)


@dataclass(frozen=True)
class InputEntry:
    """One patient/source input declared in the input manifest."""

    patient_id: str
    antigen_source: str
    input_format: str
    peptide_path: Path
    provenance_path: Path | None = None
    run_manifest_path: Path | None = None
    binding_path: Path | None = None


@dataclass(frozen=True)
class PeptideOccurrence:
    """One source record contributing an eligible peptide sequence."""

    patient_id: str
    antigen_source: str
    peptide: str
    peptide_id: str
    occurrence_id: str
    source_record_id: str
    source_path: str
    source_row_number: int
    event_id: str = ""
    wt_peptide: str = ""
    wt_control_status: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class PairMetrics:
    """Primary and annotation-only metrics for one equal-length peptide pair."""

    central_positions: str
    central_matches: int
    central_total: int
    central_identity: float
    central_run_length: int
    central_run_sequence: str
    positional_match_count: int
    full_length_identity: float
    p2_exact_match: bool
    pomega_exact_match: bool
    blosum62_full_score: int
    blosum62_central_score: int
    legacy_lcs_length: int
    legacy_lcs_sequence: str

    @property
    def exact_p2_pomega(self) -> bool:
        return self.p2_exact_match and self.pomega_exact_match


@dataclass(frozen=True)
class MimicryRunConfig:
    """Resolved command-line configuration."""

    method: str
    input_manifest: Path
    output_dir: Path
    source_pairs: tuple[str, ...]
    workers: int
    shard_count: int
    binding_support_enabled: bool = False
