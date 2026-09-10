"""Versioned adapters for native and generic peptide candidate tables."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ._io import read_json, read_tsv, require_tsv_columns
from ._metrics import is_eligible_peptide, normalize_peptide
from ._models import InputEntry, PeptideOccurrence, SUPPORTED_SOURCES
from ._policy import SequenceMimicryPolicy


SUPPORTED_INPUT_FORMATS = (
    "generic_peptide_table_v1",
    "microbial_core_v1",
    "cryptic_final_core_v1_1",
    "mutation_epitope_windows_v1",
)

INPUT_REQUIRED_COLUMNS = {
    "generic_peptide_table_v1": {
        "peptide",
        "peptide_id",
        "mhc_class",
        "qc_status",
    },
    "microbial_core_v1": {
        "peptide",
        "peptide_id",
        "mhc_class",
        "peptide_qc_status",
    },
    "cryptic_final_core_v1_1": {
        "peptide",
        "peptide_record_id",
        "mhc_class",
        "peptide_core_status",
    },
    "mutation_epitope_windows_v1": {
        "event_id",
        "mhc_class",
        "mt_epitope_seq",
        "wt_epitope_seq",
        "wt_control_status",
        "covers_mutation",
    },
}


@dataclass(frozen=True)
class AdapterResult:
    """Eligible occurrences and input-QC counts for one manifest entry."""

    occurrences: tuple[PeptideOccurrence, ...]
    counts: dict[str, int]


def _resolve_path(value: str, manifest_path: Path) -> Path | None:
    value = str(value).strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def read_input_manifest(path: Path) -> list[InputEntry]:
    """Read and validate the patient/source input manifest."""

    required = {"patient_id", "antigen_source", "input_format", "peptide_path"}
    entries: list[InputEntry] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Input manifest is missing columns {sorted(missing)}: {path}")
        for row_number, row in enumerate(reader, start=2):
            patient_id = str(row.get("patient_id", "")).strip()
            antigen_source = str(row.get("antigen_source", "")).strip().lower().replace("_", "-")
            input_format = str(row.get("input_format", "")).strip()
            peptide_path = _resolve_path(str(row.get("peptide_path", "")), path)
            if not patient_id:
                raise ValueError(f"Empty patient_id at {path}:{row_number}")
            if antigen_source not in SUPPORTED_SOURCES:
                raise ValueError(
                    f"Unsupported antigen_source {antigen_source!r} at {path}:{row_number}"
                )
            if input_format not in SUPPORTED_INPUT_FORMATS:
                raise ValueError(
                    f"Unsupported input_format {input_format!r} at {path}:{row_number}"
                )
            if peptide_path is None or not peptide_path.is_file():
                raise FileNotFoundError(
                    f"Missing peptide table at {path}:{row_number}: {peptide_path}"
                )
            provenance_path = _resolve_path(str(row.get("provenance_path", "")), path)
            run_manifest_path = _resolve_path(str(row.get("run_manifest_path", "")), path)
            binding_path = _resolve_path(str(row.get("binding_path", "")), path)
            for label, optional_path in (
                ("provenance", provenance_path),
                ("run manifest", run_manifest_path),
                ("binding table", binding_path),
            ):
                if optional_path is not None and not optional_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {label} at {path}:{row_number}: {optional_path}"
                    )
            entries.append(
                InputEntry(
                    patient_id=patient_id,
                    antigen_source=antigen_source,
                    input_format=input_format,
                    peptide_path=peptide_path,
                    provenance_path=provenance_path,
                    run_manifest_path=run_manifest_path,
                    binding_path=binding_path,
                )
            )

    keys = [(entry.patient_id, entry.antigen_source) for entry in entries]
    if len(keys) != len(set(keys)):
        duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
        raise ValueError(f"Duplicate patient/source entries in {path}: {duplicates}")
    if not entries:
        raise ValueError(f"Input manifest contains no entries: {path}")
    return entries


def validate_upstream_manifest(entry: InputEntry) -> None:
    """Reject explicitly incomplete or non-binding-eligible native outputs."""

    if entry.run_manifest_path is None:
        return
    manifest = read_json(entry.run_manifest_path)
    status = str(manifest.get("run_status", manifest.get("status", ""))).strip().lower()
    if status not in {"complete", "completed"}:
        raise ValueError(
            "Upstream run is not complete for "
            f"{entry.patient_id}/{entry.antigen_source}: {status!r}"
        )
    if manifest.get("binding_eligible") is False:
        raise ValueError(
            f"Upstream Core is not formally eligible for downstream analysis: "
            f"{entry.patient_id}/{entry.antigen_source}"
        )


def _normalize_mhc_class(value: str) -> str:
    normalized = str(value).strip().upper().replace("_", "-")
    if normalized in {"I", "MHC-I", "HLA-I", "CLASS-I", "CLASS I"}:
        return "HLA-I"
    if normalized in {"II", "MHC-II", "HLA-II", "CLASS-II", "CLASS II"}:
        return "HLA-II"
    return normalized


def _adapter_columns(input_format: str) -> dict[str, str]:
    if input_format == "generic_peptide_table_v1":
        return {
            "peptide": "peptide",
            "peptide_id": "peptide_id",
            "status": "qc_status",
            "status_value": "core",
            "mhc_class": "mhc_class",
            "wt_peptide": "wt_peptide",
        }
    if input_format == "microbial_core_v1":
        return {
            "peptide": "peptide",
            "peptide_id": "peptide_id",
            "status": "peptide_qc_status",
            "status_value": "core",
            "mhc_class": "mhc_class",
        }
    if input_format == "cryptic_final_core_v1_1":
        return {
            "peptide": "peptide",
            "peptide_id": "peptide_record_id",
            "status": "peptide_core_status",
            "status_value": "core",
            "mhc_class": "mhc_class",
        }
    return {
        "peptide": "mt_epitope_seq",
        "peptide_id": "event_id",
        "mhc_class": "mhc_class",
        "wt_peptide": "wt_epitope_seq",
    }


def _row_is_core(row: dict[str, str], columns: dict[str, str], input_format: str) -> bool:
    if _normalize_mhc_class(row.get(columns["mhc_class"], "")) != "HLA-I":
        return False
    if input_format == "mutation_epitope_windows_v1":
        covers = str(row.get("covers_mutation", "YES")).strip().upper()
        return covers in {"YES", "TRUE", "1"}
    return str(row.get(columns["status"], "")).strip().lower() == columns["status_value"]


def _stable_occurrence_id(entry: InputEntry, row_number: int, row: dict[str, str]) -> str:
    key = "|".join(
        (
            entry.patient_id,
            entry.antigen_source,
            str(entry.provenance_path or entry.peptide_path),
            str(row_number),
            row.get("event_id", ""),
            row.get("parent_record_id", ""),
            row.get("source_parent_id", ""),
        )
    )
    return "occ-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _source_record_id(row: dict[str, str], peptide_id: str) -> str:
    for field in (
        "event_id",
        "parent_record_id",
        "source_parent_id",
        "source_group_id",
        "protein_accession",
    ):
        if str(row.get(field, "")).strip():
            return str(row[field]).strip()
    return peptide_id


def _occurrence_from_row(
    entry: InputEntry,
    row_number: int,
    row: dict[str, str],
    columns: dict[str, str],
    source_path: Path,
    fallback_peptide_id: str = "",
) -> PeptideOccurrence:
    peptide = normalize_peptide(row.get(columns["peptide"], ""))
    peptide_id = str(row.get(columns.get("peptide_id", ""), "")).strip()
    if not peptide_id:
        peptide_id = fallback_peptide_id
    if not peptide_id:
        peptide_id = f"{entry.antigen_source}:{row_number}"
    return PeptideOccurrence(
        patient_id=entry.patient_id,
        antigen_source=entry.antigen_source,
        peptide=peptide,
        peptide_id=peptide_id,
        occurrence_id=_stable_occurrence_id(entry, row_number, row),
        source_record_id=_source_record_id(row, peptide_id),
        source_path=str(source_path),
        source_row_number=row_number,
        event_id=str(row.get("event_id", "")).strip(),
        wt_peptide=normalize_peptide(row.get(columns.get("wt_peptide", ""), "")),
        wt_control_status=str(row.get("wt_control_status", "")).strip(),
        metadata=dict(row),
    )


def load_occurrences(
    entry: InputEntry,
    policy: SequenceMimicryPolicy,
) -> AdapterResult:
    """Load eligible HLA-I peptide occurrences from one versioned input."""

    validate_upstream_manifest(entry)
    require_tsv_columns(entry.peptide_path, INPUT_REQUIRED_COLUMNS[entry.input_format])
    columns = _adapter_columns(entry.input_format)
    counts: Counter[str] = Counter()
    eligible_rows: list[tuple[int, dict[str, str]]] = []
    eligible_peptides: set[str] = set()
    peptide_ids: dict[str, str] = {}

    for row_number, row in read_tsv(entry.peptide_path):
        counts["input_rows"] += 1
        if not _row_is_core(row, columns, entry.input_format):
            counts["excluded_non_core_or_non_hla_i_rows"] += 1
            continue
        peptide = normalize_peptide(row.get(columns["peptide"], ""))
        if not is_eligible_peptide(peptide, policy):
            counts["excluded_invalid_sequence_or_length_rows"] += 1
            continue
        eligible_rows.append((row_number, row))
        eligible_peptides.add(peptide)
        peptide_ids.setdefault(peptide, str(row.get(columns.get("peptide_id", ""), "")))

    source_path = entry.provenance_path or entry.peptide_path
    occurrences: list[PeptideOccurrence] = []
    if entry.provenance_path is None or entry.input_format == "mutation_epitope_windows_v1":
        for row_number, row in eligible_rows:
            occurrences.append(
                _occurrence_from_row(entry, row_number, row, columns, entry.peptide_path)
            )
    else:
        provenance_required = {"peptide"}
        if entry.input_format == "cryptic_final_core_v1_1":
            provenance_required.update({"peptide_record_id", "mhc_class", "parent_record_id"})
        require_tsv_columns(entry.provenance_path, provenance_required)
        mapped_peptides: set[str] = set()
        for row_number, row in read_tsv(entry.provenance_path):
            peptide = normalize_peptide(row.get(columns["peptide"], row.get("peptide", "")))
            if peptide not in eligible_peptides:
                continue
            mapped_peptides.add(peptide)
            provenance_row = dict(row)
            provenance_row[columns["peptide"]] = peptide
            occurrences.append(
                _occurrence_from_row(
                    entry,
                    row_number,
                    provenance_row,
                    columns,
                    entry.provenance_path,
                    peptide_ids.get(peptide, ""),
                )
            )
        missing = sorted(eligible_peptides - mapped_peptides)
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"Provenance table does not map {len(missing)} eligible peptides for "
                f"{entry.patient_id}/{entry.antigen_source}: {preview}"
            )

    if entry.antigen_source == "mutation-derived":
        missing_events = [row for row in occurrences if not row.event_id]
        if missing_events:
            raise ValueError(
                f"Mutation-derived input contains {len(missing_events)} rows without event_id: "
                f"{entry.peptide_path}"
            )

    occurrence_ids = [row.occurrence_id for row in occurrences]
    if len(occurrence_ids) != len(set(occurrence_ids)):
        raise ValueError(f"Duplicate occurrence IDs generated from {source_path}")
    counts["eligible_core_rows"] = len(eligible_rows)
    counts["eligible_occurrence_rows"] = len(occurrences)
    counts["eligible_unique_peptides"] = len({row.peptide for row in occurrences})
    return AdapterResult(tuple(occurrences), dict(counts))


def load_unique_peptides(
    entry: InputEntry,
    policy: SequenceMimicryPolicy,
    peptide_length: int | None = None,
) -> set[str]:
    """Read only the unique eligible sequences needed by the search index."""

    validate_upstream_manifest(entry)
    require_tsv_columns(entry.peptide_path, INPUT_REQUIRED_COLUMNS[entry.input_format])
    columns = _adapter_columns(entry.input_format)
    peptides: set[str] = set()
    for _, row in read_tsv(entry.peptide_path):
        if not _row_is_core(row, columns, entry.input_format):
            continue
        peptide = normalize_peptide(row.get(columns["peptide"], ""))
        if not is_eligible_peptide(peptide, policy):
            continue
        if peptide_length is None or len(peptide) == peptide_length:
            peptides.add(peptide)
    return peptides
