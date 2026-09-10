#!/usr/bin/env python3
"""Apply frozen external-normal resources to a Cryptic peptide Core.

This stage consumes 08b-cryptic_core_qc peptide Core outputs and materializes
the tumor-restricted primary Core used by downstream binding. Policy v1.0 uses
exact peptide sequence plus HLA-class matching; policy v1.1 keeps those rules
and adds normal smORF coordinate/frame evidence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from mimicneoai.cryptic_pipeline.scripts.cryptic_coordinate_utils import (
        GenomicBlock,
        blocks_to_json,
        blocks_to_text,
        canonical_chromosome,
        classify_coordinate_match,
        map_transcript_interval_to_genome,
        parse_json_blocks,
        transcript_order,
    )
    from mimicneoai.cryptic_pipeline.scripts.cryptic_junction_qc import annotate_peptide_junctions
except ImportError:  # pragma: no cover - direct script execution fallback
    from cryptic_coordinate_utils import (  # type: ignore
        GenomicBlock,
        blocks_to_json,
        blocks_to_text,
        canonical_chromosome,
        classify_coordinate_match,
        map_transcript_interval_to_genome,
        parse_json_blocks,
        transcript_order,
    )
    from cryptic_junction_qc import annotate_peptide_junctions  # type: ignore

POLICY_VERSION_V1 = "cryptic_external_normal_qc_v1.0"
POLICY_VERSION_V11 = "cryptic_external_normal_qc_v1.1"
POLICY_VERSION = POLICY_VERSION_V1
COORDINATE_UTILS_PATH = Path(__file__).with_name("cryptic_coordinate_utils.py")
EXPECTED_UPSTREAM_POLICY_VERSION_V1 = "cryptic_core_qc_v1.0"
EXPECTED_UPSTREAM_POLICY_VERSION_V11 = "cryptic_core_qc_v1.1"
EXPECTED_RESOURCE_FREEZE_VERSION = "external_normal_resources_v1.0_20260727"
EXPECTED_COORDINATE_RESOURCE_FREEZE_VERSION = "smorf_coordinate_resources_v1.1_20260727"
COORDINATE_STATUS = "not_evaluable_coordinates_not_materialized"
COORDINATE_CONCORDANT = "coordinate_frame_concordant"
SMORF_DETECTED = "normal_translation_detected_by_smorf_atlas"
SMORF_COORDINATE_DETECTED = "normal_translation_detected_by_smorf_coordinate_frame"
SMORF_NOT_DETECTED = "normal_translation_not_detected_in_evaluated_smorf_atlas"
SMORF_NOT_EVALUATED = "normal_translation_not_evaluated"
HLA_DETECTED = "normal_hla_presentation_detected"
HLA_NOT_DETECTED = "normal_hla_presentation_not_detected_in_evaluated_hla_ligand_atlas"
HLA_NOT_EVALUATED = "normal_hla_presentation_not_evaluated"
RETAINED_STATUS = "not_detected_in_evaluated_frozen_resources"
EXACT_EXCLUDED_STATUS = "external_normal_exact_match_excluded"
COORDINATE_EXCLUDED_STATUS = "external_normal_coordinate_frame_excluded"
BOTH_EXCLUDED_STATUS = "external_normal_exact_and_coordinate_frame_excluded"
NOT_EVALUATED_STATUS = "external_normal_resources_not_evaluated"
HUMAN_REFERENCE_MATCH = "human_reference_peptide_match"
HUMAN_REFERENCE_NOT_DETECTED = "human_reference_peptide_not_detected"
HUMAN_REFERENCE_NOT_EVALUATED = "not_evaluated"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
ALLOWED_LENGTHS = {
    "HLA-I": {8, 9, 10, 11},
    "HLA-II": {13, 14, 15, 16, 17},
}
COORDINATE_INDEX_BIN_SIZE = 100_000


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_tsv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    if len(set(fieldnames)) != len(fieldnames):
        duplicates = sorted({field for field in fieldnames if fieldnames.count(field) > 1})
        raise ValueError(f"TSV field list contains duplicate columns for {path.name}: {duplicates}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_fasta(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(f">{row['peptide_record_id']}\n{row['peptide']}\n")


def stable_refill_peptide_id(sample: str, mhc_class: str, peptide: str, ordinal: int) -> str:
    normalized_class = normalize_hla_class(mhc_class).replace("-", "")
    digest = sha256_text(f"{sample}|cryptic_refill|{normalized_class}|{len(peptide)}|{peptide}")[:12]
    return f"cryptic_refill_{normalized_class}_{ordinal:07d}_{digest}"


def read_fasta(path: Path) -> Iterable[tuple[str, str]]:
    header: Optional[str] = None
    chunks: list[str] = []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:  # type: ignore[arg-type]
        for line in handle:
            text = line.strip()
            if not text:
                continue
            if text.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = text[1:].split()[0]
                chunks = []
            else:
                chunks.append(text)
        if header is not None:
            yield header, "".join(chunks).upper()


def normalize_human_reference_sequence(sequence: str) -> str:
    seq = str(sequence).replace(" ", "").replace("\n", "").strip().upper()
    if seq.endswith("*"):
        seq = seq[:-1]
    return seq


def iter_canonical_segments(sequence: str) -> Iterable[str]:
    start: Optional[int] = None
    for index, aa in enumerate(sequence):
        if aa in STANDARD_AA:
            if start is None:
                start = index
        elif start is not None:
            yield sequence[start:index]
            start = None
    if start is not None:
        yield sequence[start:]


def load_human_reference_matches(
    human_proteome_fasta: Optional[Path],
    candidate_peptides: set[str],
) -> tuple[set[str], dict[str, object]]:
    if not human_proteome_fasta:
        return set(), {"status": "not_evaluated", "reason": "human_reference_proteome_not_configured"}
    if not human_proteome_fasta.exists():
        raise FileNotFoundError(f"human reference proteome FASTA not found: {human_proteome_fasta}")
    wanted = {str(peptide).upper() for peptide in candidate_peptides if peptide}
    matches: set[str] = set()
    records = 0
    windows_evaluated = 0
    wanted_lengths = sorted({len(peptide) for peptide in wanted})
    for _header, raw_seq in read_fasta(human_proteome_fasta):
        records += 1
        seq = normalize_human_reference_sequence(raw_seq)
        for segment in iter_canonical_segments(seq):
            for length in wanted_lengths:
                if length <= 0 or len(segment) < length:
                    continue
                windows_evaluated += len(segment) - length + 1
                for start in range(0, len(segment) - length + 1):
                    peptide = segment[start : start + length]
                    if peptide in wanted:
                        matches.add(peptide)
        if len(matches) == len(wanted):
            break
    return matches, {
        "status": "evaluated",
        "path": str(human_proteome_fasta),
        "sha256": sha256_file(human_proteome_fasta),
        "candidate_peptides": len(wanted),
        "matched_peptides": len(matches),
        "records_scanned": records,
        "standard_windows_evaluated": windows_evaluated,
    }


def iter_windows(sequence: str, lengths: Iterable[int]) -> Iterable[tuple[int, int, str]]:
    seq = str(sequence).strip().upper()
    for length in sorted({int(value) for value in lengths}):
        if length <= 0 or len(seq) < length:
            continue
        for start0 in range(0, len(seq) - length + 1):
            yield start0 + 1, length, seq[start0 : start0 + length]


def make_refill_parent_map_row(
    sample: str,
    parent_row: dict[str, object],
    mhc_class: str,
    start: int,
    length: int,
    peptide: str,
    peptide_record_id: str,
) -> dict[str, object]:
    return {
        "sample": sample,
        "peptide_record_id": peptide_record_id,
        "parent_record_id": parent_row["parent_record_id"],
        "source_parent_id": parent_row.get("source_parent_id", ""),
        "mhc_class": mhc_class,
        "peptide_start": start,
        "peptide_length": length,
        "peptide": peptide,
        "peptide_sequence_sha256": sha256_text(peptide),
        "peptide_core_status": "core",
        "peptide_qc_reasons": "",
        "human_reference_peptide_status": HUMAN_REFERENCE_NOT_DETECTED,
        "normal_hla_presentation_status": "normal_hla_atlas_not_evaluated",
        "normal_translation_status": "normal_translation_not_evaluable",
        "external_normal_status": "external_normal_not_evaluated",
        "prebinding_selection_origin": "deferred_refill",
    }


def output_signature(paths: list[Path]) -> dict[str, dict[str, object]]:
    return {path.name: file_identity(path) for path in paths}


def outputs_match_signature(signature: dict[str, object]) -> bool:
    for identity in signature.values():
        if not isinstance(identity, dict):
            return False
        path = Path(str(identity.get("path", "")))
        if file_identity(path) != identity:
            return False
    return True


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def normalize_hla_class(value: object) -> str:
    text = str(value).strip().upper().replace("_", "-")
    aliases = {
        "MHC-I": "HLA-I",
        "MHCI": "HLA-I",
        "CLASS-I": "HLA-I",
        "HLA-I": "HLA-I",
        "MHC-II": "HLA-II",
        "MHCII": "HLA-II",
        "CLASS-II": "HLA-II",
        "HLA-II": "HLA-II",
    }
    return aliases.get(text, text)


def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, low_memory=False)


def load_optional_table(path: Path) -> pd.DataFrame:
    if not path or not str(path).strip() or not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return load_table(path)


def manifest_processed_entry(resource_manifest: dict[str, object], rel_path: str) -> dict[str, object]:
    processed = resource_manifest.get("processed_files", {})
    if not isinstance(processed, dict) or rel_path not in processed:
        raise ValueError(f"resource manifest missing processed_files entry: {rel_path}")
    entry = processed[rel_path]
    if not isinstance(entry, dict):
        raise ValueError(f"resource manifest entry is invalid: {rel_path}")
    return entry


def resolve_resource_path(manifest_path: Path, explicit: str, rel_path: str) -> Path:
    if explicit:
        return Path(explicit)
    return manifest_path.parent / rel_path


def validate_resource_file(path: Path, manifest_entry: dict[str, object]) -> dict[str, object]:
    identity = file_identity(path)
    if not identity.get("exists"):
        raise FileNotFoundError(f"external-normal resource file not found: {path}")
    expected_size = int(manifest_entry.get("size_bytes", -1))
    expected_sha = str(manifest_entry.get("sha256", ""))
    if identity.get("size") != expected_size or identity.get("sha256") != expected_sha:
        raise ValueError(
            f"external-normal resource identity mismatch for {path}: "
            f"expected size={expected_size} sha256={expected_sha}, "
            f"observed size={identity.get('size')} sha256={identity.get('sha256')}"
        )
    return identity


def validate_resource_contract(resource_manifest: dict[str, object]) -> None:
    freeze_version = str(resource_manifest.get("freeze_version", ""))
    if freeze_version != EXPECTED_RESOURCE_FREEZE_VERSION:
        raise ValueError(
            f"unsupported external-normal resource freeze_version: {freeze_version}; "
            f"expected {EXPECTED_RESOURCE_FREEZE_VERSION}"
        )
    policy = resource_manifest.get("matching_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("resource manifest missing matching_policy")
    hla_i_lengths = [int(v) for v in policy.get("hla_i_lengths", [])]
    hla_ii_lengths = [int(v) for v in policy.get("hla_ii_lengths", [])]
    if hla_i_lengths != sorted(ALLOWED_LENGTHS["HLA-I"]):
        raise ValueError("resource manifest HLA-I length policy must be 8-11")
    if hla_ii_lengths != sorted(ALLOWED_LENGTHS["HLA-II"]):
        raise ValueError("resource manifest HLA-II length policy must be 13-17")
    smorf_policy = str(policy.get("smorf", "")).lower()
    hla_policy = str(policy.get("hla_ligand", "")).lower()
    if "exact peptide" not in smorf_policy:
        raise ValueError("resource manifest smORF policy must be exact peptide matching")
    if "exact peptide" not in hla_policy or "same hla class" not in hla_policy:
        raise ValueError("resource manifest HLA ligand policy must be exact peptide plus same HLA class")


def is_v11(policy_version: str) -> bool:
    return policy_version == POLICY_VERSION_V11


def validate_coordinate_resource_contract(resource_manifest: dict[str, object]) -> None:
    freeze_version = str(resource_manifest.get("freeze_version", ""))
    if freeze_version != EXPECTED_COORDINATE_RESOURCE_FREEZE_VERSION:
        raise ValueError(
            f"unsupported smORF coordinate resource freeze_version: {freeze_version}; "
            f"expected {EXPECTED_COORDINATE_RESOURCE_FREEZE_VERSION}"
        )
    source = resource_manifest.get("source", {})
    if not isinstance(source, dict) or str(source.get("reference_build", "")) != "GRCh38":
        raise ValueError("smORF coordinate resource must use reference_build=GRCh38")
    policy = resource_manifest.get("coordinate_policy", {})
    if not isinstance(policy, dict) or str(policy.get("normalized_internal_coordinates", "")) != "0_based_half_open":
        raise ValueError("smORF coordinate resource must use 0-based half-open normalized coordinates")
    qc = resource_manifest.get("qc", {})
    if not isinstance(qc, dict) or int(qc.get("formal_coordinate_qc_passes", -1)) != 7767:
        raise ValueError("smORF coordinate resource must have 7,767 formal coordinate QC pass records")


def coordinate_resource_identities(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, Path]]:
    manifest_value = getattr(args, "coordinate_resource_manifest", "")
    manifest_path = Path(manifest_value) if manifest_value else None
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError("coordinate_resource_manifest is required for CrypticExternalNormalQC v1.1")
    resource_manifest = read_json(manifest_path)
    validate_coordinate_resource_contract(resource_manifest)
    rel_paths = {
        "normal_smorf_coordinates": "processed/normal_smorfs_coordinates.tsv.gz",
        "normal_smorf_orfcds": "processed/normal_smorf_orfcds.tsv.gz",
    }
    explicit = {
        "normal_smorf_coordinates": getattr(args, "normal_smorf_coordinates", ""),
        "normal_smorf_orfcds": getattr(args, "normal_smorf_orfcds", ""),
    }
    identities: dict[str, object] = {
        "coordinate_resource_manifest": file_identity(manifest_path),
        "coordinate_resource_manifest_sha256": sha256_file(manifest_path),
        "coordinate_freeze_version": resource_manifest.get("freeze_version", ""),
        "coordinate_qc": resource_manifest.get("qc", {}),
        "coordinate_policy": resource_manifest.get("coordinate_policy", {}),
    }
    paths: dict[str, Path] = {}
    for key, rel_path in rel_paths.items():
        path = resolve_resource_path(manifest_path, str(explicit[key] or ""), rel_path)
        paths[key] = path
        identities[key] = validate_resource_file(path, manifest_processed_entry(resource_manifest, rel_path))
    return identities, paths


def resource_identities(args: argparse.Namespace) -> tuple[bool, dict[str, object], dict[str, Path]]:
    manifest_path = Path(args.resource_manifest) if args.resource_manifest else None
    if manifest_path is None or not manifest_path.exists():
        if args.allow_missing_external_normal_resources:
            return False, {
                "resource_manifest": {
                    "path": str(manifest_path) if manifest_path is not None else "",
                    "exists": False,
                }
            }, {}
        raise FileNotFoundError("resource_manifest is required for formal CrypticExternalNormalQC")

    resource_manifest = read_json(manifest_path)
    validate_resource_contract(resource_manifest)
    rel_paths = {
        "smorf_match_index": "processed/normal_smorf_peptide_match_index.tsv.gz",
        "smorf_parent_map": "processed/normal_smorf_peptide_parent_map.tsv.gz",
        "hla_ligand_match_index": "processed/normal_hla_peptide_match_index.tsv.gz",
        "hla_ligand_evidence": "processed/normal_hla_ligand_evidence.tsv.gz",
    }
    explicit = {
        "smorf_match_index": args.smorf_match_index,
        "smorf_parent_map": args.smorf_parent_map,
        "hla_ligand_match_index": args.hla_ligand_match_index,
        "hla_ligand_evidence": args.hla_ligand_evidence,
    }
    identities: dict[str, object] = {
        "resource_manifest": file_identity(manifest_path),
        "resource_manifest_sha256": sha256_file(manifest_path),
        "freeze_version": resource_manifest.get("freeze_version", ""),
    }
    paths: dict[str, Path] = {}
    try:
        for key, rel_path in rel_paths.items():
            path = resolve_resource_path(manifest_path, str(explicit[key] or ""), rel_path)
            paths[key] = path
            identities[key] = validate_resource_file(path, manifest_processed_entry(resource_manifest, rel_path))
    except Exception:
        if args.allow_missing_external_normal_resources:
            return False, identities, paths
        raise
    return True, identities, paths


def validate_signature_entry(actual_path: Path, entry: object, label: str) -> dict[str, object]:
    if not isinstance(entry, dict):
        raise ValueError(f"upstream output_signature entry for {label} is invalid")
    actual = file_identity(actual_path)
    if not actual.get("exists"):
        raise FileNotFoundError(f"required upstream 08b output does not exist: {actual_path}")
    if not bool(entry.get("exists", False)):
        raise ValueError(f"upstream output_signature says {label} does not exist")
    if actual.get("size") != entry.get("size") or actual.get("sha256") != entry.get("sha256"):
        raise ValueError(
            f"upstream output_signature mismatch for {label}: "
            f"expected size={entry.get('size')} sha256={entry.get('sha256')}, "
            f"observed size={actual.get('size')} sha256={actual.get('sha256')}"
        )
    return actual


def validate_upstream_08b(
    core_path: Path,
    parent_map_path: Path,
    manifest_path: Path,
    expected_policy_version: str,
    coordinate_sidecars: Optional[dict[str, Path]] = None,
) -> dict[str, object]:
    for label, path in [
        ("cryptic_peptide_core.tsv", core_path),
        ("cryptic_peptide_parent_map.tsv", parent_map_path),
        ("upstream run_manifest.json", manifest_path),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"required upstream 08b input missing: {label}: {path}")
    manifest = read_json(manifest_path)
    if manifest.get("run_status") != "complete":
        raise ValueError("upstream 08b run_manifest.json must have run_status=complete")
    if manifest.get("binding_eligible") is not True:
        raise ValueError("upstream 08b run_manifest.json must have binding_eligible=true")
    if manifest.get("policy_version") != expected_policy_version:
        raise ValueError(
            f"upstream 08b policy_version must be {expected_policy_version}; "
            f"observed {manifest.get('policy_version', '')}"
        )
    output_signature = manifest.get("output_signature", {})
    if not isinstance(output_signature, dict):
        raise ValueError("upstream 08b run_manifest.json missing output_signature")
    if not outputs_match_signature(output_signature):
        raise ValueError("upstream 08b output_signature does not match current output files")
    validate_signature_entry(core_path, output_signature.get("cryptic_peptide_core.tsv"), "cryptic_peptide_core.tsv")
    validate_signature_entry(
        parent_map_path,
        output_signature.get("cryptic_peptide_parent_map.tsv"),
        "cryptic_peptide_parent_map.tsv",
    )
    for label, path in (coordinate_sidecars or {}).items():
        validate_signature_entry(path, output_signature.get(label), label)
    return manifest


def add_key_columns(df: pd.DataFrame, peptide_col: str = "peptide_sequence") -> pd.DataFrame:
    out = df.copy()
    out["__hla_class"] = out["hla_class"].map(normalize_hla_class)
    out["__peptide"] = out[peptide_col].astype(str).str.upper()
    return out


def build_lookup(df: pd.DataFrame, peptide_col: str = "peptide_sequence") -> set[tuple[str, str]]:
    keyed = add_key_columns(df, peptide_col=peptide_col)
    return set(zip(keyed["__hla_class"], keyed["__peptide"]))


def group_rows_by_key(df: pd.DataFrame, peptide_col: str = "peptide_sequence") -> dict[tuple[str, str], list[dict[str, str]]]:
    keyed = add_key_columns(df, peptide_col=peptide_col)
    grouped: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in keyed.to_dict("records"):
        grouped[(str(row["__hla_class"]), str(row["__peptide"]))].append(row)
    return dict(grouped)


def required_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def parse_int(value: object, label: str) -> int:
    try:
        return int(str(value))
    except Exception as exc:
        raise ValueError(f"{label} must be an integer: {value}") from exc


def validate_peptide_row(row: dict[str, object], sample: str, table_label: str, row_id: str) -> None:
    observed_sample = str(row.get("sample", ""))
    if observed_sample != sample:
        raise ValueError(f"{table_label} sample mismatch for {row_id}: {observed_sample} != {sample}")
    hla_class = normalize_hla_class(row.get("mhc_class", ""))
    if hla_class not in ALLOWED_LENGTHS:
        raise ValueError(f"{table_label} has unsupported mhc_class for {row_id}: {row.get('mhc_class', '')}")
    peptide = str(row.get("peptide", ""))
    if not peptide or peptide.upper() != peptide or any(residue not in STANDARD_AA for residue in peptide):
        raise ValueError(f"{table_label} has non-standard peptide sequence for {row_id}: {peptide}")
    peptide_length = parse_int(row.get("peptide_length", ""), f"{table_label} peptide_length for {row_id}")
    if peptide_length != len(peptide):
        raise ValueError(
            f"{table_label} peptide_length does not match sequence length for {row_id}: "
            f"{peptide_length} != {len(peptide)}"
        )
    if peptide_length not in ALLOWED_LENGTHS[hla_class]:
        allowed = "-".join([str(min(ALLOWED_LENGTHS[hla_class])), str(max(ALLOWED_LENGTHS[hla_class]))])
        raise ValueError(f"{table_label} {hla_class} peptide length for {row_id} must be {allowed}: {peptide_length}")


def validate_core_parent_contract(core: pd.DataFrame, parent_map: pd.DataFrame, sample: str) -> None:
    if core["peptide_record_id"].duplicated().any():
        duplicates = core.loc[core["peptide_record_id"].duplicated(), "peptide_record_id"].head(5).tolist()
        raise ValueError(f"cryptic_peptide_core.tsv must contain unique peptide_record_id values: {duplicates}")

    core_records = {str(row["peptide_record_id"]): row for row in core.to_dict("records")}
    parent_records = parent_map.to_dict("records")
    parent_ids = {str(row.get("peptide_record_id", "")) for row in parent_records}
    core_ids = set(core_records)

    orphan_ids = sorted(parent_ids - core_ids)
    if orphan_ids:
        raise ValueError(f"cryptic_peptide_parent_map.tsv has orphan peptide_record_id values: {orphan_ids[:5]}")
    missing_parent_ids = sorted(core_ids - parent_ids)
    if missing_parent_ids:
        raise ValueError(f"cryptic_peptide_core.tsv peptides missing parent-map records: {missing_parent_ids[:5]}")

    for peptide_id, row in core_records.items():
        validate_peptide_row(row, sample, "cryptic_peptide_core.tsv", peptide_id)

    for idx, row in enumerate(parent_records, start=1):
        peptide_id = str(row.get("peptide_record_id", ""))
        validate_peptide_row(row, sample, "cryptic_peptide_parent_map.tsv", peptide_id or f"row{idx}")
        core_row = core_records.get(peptide_id)
        if core_row is None:
            continue
        mismatches = []
        for column in ["sample", "peptide", "peptide_length"]:
            if str(row.get(column, "")) != str(core_row.get(column, "")):
                mismatches.append(column)
        if normalize_hla_class(row.get("mhc_class", "")) != normalize_hla_class(core_row.get("mhc_class", "")):
            mismatches.append("mhc_class")
        if mismatches:
            raise ValueError(
                f"Core and parent-map disagree for peptide_record_id {peptide_id}: {', '.join(mismatches)}"
            )


def load_deferred_peptide_core(path: Path, sample: str, existing_keys: set[tuple[str, str]]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    deferred = load_table(path)
    if deferred.empty:
        return deferred
    required_columns(
        deferred,
        ["sample", "mhc_class", "peptide_length", "peptide"],
        "cryptic_peptide_deferred.tsv",
    )
    rows: list[dict[str, object]] = []
    seen_keys = set(existing_keys)
    for idx, row in enumerate(deferred.to_dict("records"), start=1):
        validate_peptide_row(row, sample, "cryptic_peptide_deferred.tsv", str(idx))
        key = (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out = dict(row)
        if not str(out.get("peptide_record_id", "")):
            out["peptide_record_id"] = stable_refill_peptide_id(
                sample,
                str(out["mhc_class"]),
                str(out["peptide"]),
                len(rows) + 1,
            )
        out["prebinding_selection_origin"] = "deferred_refill"
        rows.append(out)
    return pd.DataFrame(rows)


def validate_coordinate_sidecar_contract(
    core: pd.DataFrame,
    parent_map: pd.DataFrame,
    parent_coordinates: pd.DataFrame,
    parent_orfcds: pd.DataFrame,
    footprint: pd.DataFrame,
    sample: str,
) -> None:
    required_columns(
        parent_coordinates,
        [
            "sample", "parent_record_id", "block_count", "blocks_transcript_order",
            "coordinate_mapping_status",
        ],
        "cryptic_parent_coordinates",
    )
    required_columns(
        parent_orfcds,
        [
            "sample", "parent_record_id", "transcript_block_order", "chromosome",
            "strand", "start0", "end0", "block_length", "coordinate_mapping_status",
        ],
        "cryptic_parent_orfcds",
    )
    required_columns(
        footprint,
        [
            "sample", "peptide_record_id", "parent_record_id", "mhc_class", "peptide",
            "peptide_start", "peptide_length", "peptide_cds_start0", "peptide_cds_end0",
            "codon_blocks_transcript_order", "candidate_coordinate_status", "reference_build",
        ],
        "cryptic_peptide_genomic_footprint",
    )
    for label, table in [
        ("cryptic_parent_coordinates", parent_coordinates),
        ("cryptic_parent_orfcds", parent_orfcds),
        ("cryptic_peptide_genomic_footprint", footprint),
    ]:
        if not table.empty and not table["sample"].astype(str).eq(sample).all():
            raise ValueError(f"{label} sample must match --sample")

    core_by_id = {str(row["peptide_record_id"]): row for row in core.to_dict("records")}
    parent_ids_in_map = {str(row["parent_record_id"]) for row in parent_map.to_dict("records")}
    parent_ids_in_coords = set(parent_coordinates["parent_record_id"].astype(str))
    missing_parent_coords = sorted(parent_ids_in_map - parent_ids_in_coords)
    if missing_parent_coords:
        raise ValueError(
            "cryptic_parent_coordinates missing parent IDs used by parent-map: "
            + ", ".join(missing_parent_coords[:5])
        )

    coord_by_parent = {
        str(row["parent_record_id"]): row
        for row in parent_coordinates.to_dict("records")
    }
    orfcds_by_parent: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in parent_orfcds.to_dict("records"):
        orfcds_by_parent[str(row["parent_record_id"])].append(row)
    for parent_id, coord_row in coord_by_parent.items():
        block_count = parse_int(coord_row.get("block_count", 0), f"cryptic_parent_coordinates block_count for {parent_id}")
        orf_rows = sorted(
            orfcds_by_parent.get(parent_id, []),
            key=lambda row: parse_int(row.get("transcript_block_order", 0), f"orfCDS order for {parent_id}"),
        )
        if block_count != len(orf_rows):
            raise ValueError(
                f"cryptic_parent_orfcds block count mismatch for {parent_id}: "
                f"{len(orf_rows)} != {block_count}"
            )
        block_total = 0
        for row in orf_rows:
            start0 = parse_int(row.get("start0", ""), f"orfCDS start0 for {parent_id}")
            end0 = parse_int(row.get("end0", ""), f"orfCDS end0 for {parent_id}")
            length = parse_int(row.get("block_length", ""), f"orfCDS block_length for {parent_id}")
            if end0 - start0 != length:
                raise ValueError(f"cryptic_parent_orfcds block_length mismatch for {parent_id}")
            block_total += length
        coord_block_total = parse_int(coord_row.get("block_total_length", block_total), f"coordinate block_total_length for {parent_id}")
        if block_total != coord_block_total:
            raise ValueError(
                f"cryptic_parent_orfcds total length mismatch for {parent_id}: "
                f"{block_total} != {coord_block_total}"
            )

    parent_map_key_rows = [
        (
            str(row.get("peptide_record_id", "")),
            str(row.get("parent_record_id", "")),
            str(row.get("peptide_start", "")),
            str(row.get("peptide_length", "")),
        )
        for row in parent_map.to_dict("records")
    ]
    parent_map_keys = set(parent_map_key_rows)
    if len(parent_map_keys) != len(parent_map_key_rows):
        raise ValueError("cryptic_peptide_parent_map.tsv contains duplicate coordinate footprint keys")
    footprint_key_rows = [
        (
            str(row.get("peptide_record_id", "")),
            str(row.get("parent_record_id", "")),
            str(row.get("peptide_start", "")),
            str(row.get("peptide_length", "")),
        )
        for row in footprint.to_dict("records")
    ]
    footprint_keys = set(footprint_key_rows)
    if len(footprint_keys) != len(footprint_key_rows):
        raise ValueError("cryptic_peptide_genomic_footprint contains duplicate coordinate footprint keys")
    missing_footprints = sorted(parent_map_keys - footprint_keys)
    extra_footprints = sorted(footprint_keys - parent_map_keys)
    if missing_footprints or extra_footprints:
        raise ValueError(
            "cryptic_peptide_genomic_footprint keys must exactly match Core parent-map keys; "
            f"missing={missing_footprints[:5]} extra={extra_footprints[:5]}"
        )
    for idx, row in enumerate(footprint.to_dict("records"), start=1):
        peptide_id = str(row.get("peptide_record_id", ""))
        core_row = core_by_id.get(peptide_id)
        if core_row is None:
            raise ValueError(f"cryptic_peptide_genomic_footprint has orphan peptide_record_id: {peptide_id}")
        mismatches = []
        for column in ["peptide", "peptide_length"]:
            if str(row.get(column, "")) != str(core_row.get(column, "")):
                mismatches.append(column)
        if normalize_hla_class(row.get("mhc_class", "")) != normalize_hla_class(core_row.get("mhc_class", "")):
            mismatches.append("mhc_class")
        if mismatches:
            raise ValueError(
                f"cryptic_peptide_genomic_footprint disagrees with Core for {peptide_id}: "
                + ", ".join(mismatches)
            )
        key = (
            peptide_id,
            str(row.get("parent_record_id", "")),
            str(row.get("peptide_start", "")),
            str(row.get("peptide_length", "")),
        )
        if key not in parent_map_keys:
            raise ValueError(
                f"cryptic_peptide_genomic_footprint row {idx} is not represented in parent-map"
            )
        peptide_length = parse_int(row.get("peptide_length", ""), f"footprint peptide_length for {peptide_id}")
        cds_start0 = parse_int(row.get("peptide_cds_start0", ""), f"footprint peptide_cds_start0 for {peptide_id}")
        cds_end0 = parse_int(row.get("peptide_cds_end0", ""), f"footprint peptide_cds_end0 for {peptide_id}")
        if cds_end0 - cds_start0 != peptide_length * 3:
            raise ValueError(f"cryptic_peptide_genomic_footprint CDS interval length mismatch for {peptide_id}")
        blocks = parse_json_blocks(row.get("codon_blocks_transcript_order", ""))
        status = str(row.get("candidate_coordinate_status", ""))
        if status == "coordinate_evaluable":
            block_total = sum(block.length for block in blocks)
            if block_total != peptide_length * 3:
                raise ValueError(
                    f"cryptic_peptide_genomic_footprint codon block length mismatch for {peptide_id}: "
                    f"{block_total} != {peptide_length * 3}"
                )


def build_refill_peptide_footprints(
    sample: str,
    peptide_parent_rows: list[dict[str, object]],
    parent_coordinates: pd.DataFrame,
    parent_orfcds: pd.DataFrame,
) -> list[dict[str, object]]:
    coord_by_parent = {
        str(row.get("parent_record_id", "")): row
        for row in parent_coordinates.to_dict("records")
    }
    orfcds_by_parent: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in parent_orfcds.to_dict("records"):
        orfcds_by_parent[str(row.get("parent_record_id", ""))].append(row)

    footprints: list[dict[str, object]] = []
    for row in peptide_parent_rows:
        parent_id = str(row.get("parent_record_id", ""))
        coord = coord_by_parent.get(parent_id)
        if coord is None:
            raise ValueError(f"refill parent-map references parent without coordinate sidecar: {parent_id}")
        orf_rows = sorted(
            orfcds_by_parent.get(parent_id, []),
            key=lambda item: parse_int(item.get("transcript_block_order", 0), f"orfCDS order for {parent_id}"),
        )
        strand = str(coord.get("strand", ""))
        tx_blocks = [
            GenomicBlock(
                canonical_chromosome(item.get("chromosome", "")),
                str(item.get("strand", strand)),
                parse_int(item.get("start0", ""), f"orfCDS start0 for {parent_id}"),
                parse_int(item.get("end0", ""), f"orfCDS end0 for {parent_id}"),
            )
            for item in orf_rows
        ]
        status = str(coord.get("coordinate_mapping_status", "not_evaluable_candidate_coordinate_qc"))
        reasons = str(coord.get("coordinate_mapping_reasons", ""))
        peptide_start = parse_int(row.get("peptide_start", ""), f"refill peptide_start for {parent_id}")
        peptide_length = parse_int(row.get("peptide_length", ""), f"refill peptide_length for {parent_id}")
        cds_start0 = (peptide_start - 1) * 3
        cds_end0 = cds_start0 + peptide_length * 3
        footprint_blocks: list[GenomicBlock] = []
        if status == "coordinate_evaluable":
            try:
                footprint_blocks = [
                    block
                    for _segment_start, _segment_end, block in map_transcript_interval_to_genome(
                        tx_blocks,
                        strand,
                        cds_start0,
                        cds_end0,
                    )
                ]
            except Exception as exc:
                status = "not_evaluable_candidate_coordinate_qc"
                reasons = ";".join(filter(None, [reasons, f"peptide_footprint_mapping_failed:{exc}"]))
        footprints.append({
            "sample": sample,
            "peptide_record_id": row.get("peptide_record_id", ""),
            "parent_record_id": parent_id,
            "source_parent_id": row.get("source_parent_id", ""),
            "mhc_class": row.get("mhc_class", ""),
            "peptide": row.get("peptide", ""),
            "peptide_start": peptide_start,
            "peptide_length": peptide_length,
            "chromosome": coord.get("chromosome", ""),
            "strand": strand,
            "peptide_cds_start0": cds_start0,
            "peptide_cds_end0": cds_end0,
            "codon_blocks_transcript_order": blocks_to_json(footprint_blocks),
            "codon_blocks_text": blocks_to_text(footprint_blocks),
            "peptide_footprint_phase": cds_start0 % 3,
            "candidate_coordinate_status": status,
            "candidate_coordinate_reasons": reasons,
            "reference_build": coord.get("reference_build", "GRCh38"),
        })
    return footprints


def select_core_parent_map_rows(parent_map: pd.DataFrame, core_ids: set[str]) -> pd.DataFrame:
    if "peptide_core_status" not in parent_map.columns:
        return parent_map
    status = parent_map["peptide_core_status"].astype(str)
    core_parent_map = parent_map[status.eq("core")].copy()
    noncore_ids = {
        str(value)
        for value in parent_map.loc[~status.eq("core"), "peptide_record_id"].tolist()
        if str(value).strip()
    }
    conflicting_ids = sorted(noncore_ids & core_ids)
    if conflicting_ids:
        raise ValueError(f"non-core parent-map rows reference Core peptide_record_id values: {conflicting_ids[:5]}")
    return core_parent_map


def compact_values(rows: list[dict[str, str]], key: str) -> str:
    vals = sorted({str(row.get(key, "")).strip() for row in rows if str(row.get(key, "")).strip()})
    return ";".join(vals)


def compact_list(values: Iterable[object]) -> str:
    return ";".join(sorted({str(value).strip() for value in values if str(value).strip()}))


def load_normal_smorf_coordinate_blocks(
    coordinates_path: Path,
    orfcds_path: Path,
) -> dict[str, dict[str, object]]:
    coords = load_table(coordinates_path)
    orfcds = load_table(orfcds_path)
    required_columns(
        coords,
        ["iorf_id", "orf_id", "source_record_id", "reference_build", "coordinate_qc_status"],
        "normal_smorfs_coordinates",
    )
    required_columns(
        orfcds,
        ["iorf_id", "orf_id", "chromosome", "strand", "start0", "end0", "transcript_block_order"],
        "normal_smorf_orfcds",
    )
    if not coords["reference_build"].astype(str).eq("GRCh38").all():
        raise ValueError("normal_smorfs_coordinates reference_build must be GRCh38")
    if len(coords[coords["coordinate_qc_status"].astype(str).eq("pass")]) != 7767:
        raise ValueError("normal_smorfs_coordinates must contain 7,767 coordinate_qc_status=pass records")
    pass_iorfs = set(coords.loc[coords["coordinate_qc_status"].astype(str).eq("pass"), "iorf_id"].astype(str))
    coord_meta = coords.set_index("iorf_id", drop=False).to_dict("index")
    normal: dict[str, dict[str, object]] = {}
    for iorf_id, group in orfcds[orfcds["iorf_id"].astype(str).isin(pass_iorfs)].groupby("iorf_id", sort=False):
        rows = group.copy()
        rows["transcript_block_order"] = rows["transcript_block_order"].astype(int)
        rows = rows.sort_values("transcript_block_order")
        blocks = [
            GenomicBlock(
                canonical_chromosome(row["chromosome"]),
                str(row["strand"]),
                int(row["start0"]),
                int(row["end0"]),
            )
            for row in rows.to_dict("records")
        ]
        meta = coord_meta.get(str(iorf_id), {})
        normal[str(iorf_id)] = {
            "iorf_id": str(iorf_id),
            "orf_id": str(meta.get("orf_id", rows.iloc[0]["orf_id"])),
            "source_record_id": str(meta.get("source_record_id", "")),
            "chromosome": blocks[0].chrom if blocks else "",
            "strand": blocks[0].strand if blocks else "",
            "blocks": blocks,
            "blocks_text": blocks_to_text(blocks),
        }
    return normal


def choose_coordinate_status(statuses: Iterable[str]) -> str:
    priority = [
        COORDINATE_CONCORDANT,
        "coordinate_overlap_frame_discordant",
        "junction_chain_incompatible",
        "partial_coordinate_overlap",
        "ambiguous_candidate_mapping",
        "not_evaluable_candidate_coordinate_qc",
        "not_evaluable_reference_or_coordinate_mismatch",
        "no_coordinate_overlap",
    ]
    observed = set(statuses)
    for status in priority:
        if status in observed:
            return status
    return "no_coordinate_overlap"


def evaluate_coordinate_evidence(
    core: pd.DataFrame,
    footprint: pd.DataFrame,
    normal_blocks_by_iorf: dict[str, dict[str, object]],
    sample: str,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    if footprint.empty:
        return {}, []
    required_columns(
        footprint,
        [
            "sample", "peptide_record_id", "parent_record_id", "mhc_class", "peptide",
            "peptide_start", "peptide_length", "codon_blocks_transcript_order",
            "peptide_cds_start0", "candidate_coordinate_status", "reference_build",
        ],
        "cryptic_peptide_genomic_footprint",
    )
    core_ids = set(core["peptide_record_id"].astype(str))
    footprint_ids = set(footprint["peptide_record_id"].astype(str))
    orphan = sorted(footprint_ids - core_ids)
    if orphan:
        raise ValueError(f"cryptic_peptide_genomic_footprint has orphan peptide_record_id values: {orphan[:5]}")
    missing = sorted(core_ids - footprint_ids)
    if missing:
        raise ValueError(f"cryptic_peptide_genomic_footprint missing Core peptide_record_id values: {missing[:5]}")
    normal_index: defaultdict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for normal in normal_blocks_by_iorf.values():
        blocks = normal["blocks"]
        if blocks:
            seen_bins: set[tuple[str, str, int]] = set()
            for block in blocks:
                first_bin = block.start0 // COORDINATE_INDEX_BIN_SIZE
                last_bin = max(block.start0, block.end0 - 1) // COORDINATE_INDEX_BIN_SIZE
                for bin_id in range(first_bin, last_bin + 1):
                    key = (block.chrom, block.strand, bin_id)
                    if key not in seen_bins:
                        normal_index[key].append(normal)
                        seen_bins.add(key)

    summaries: dict[str, dict[str, object]] = {}
    evidence_rows: list[dict[str, object]] = []
    for peptide_id, group in footprint.groupby("peptide_record_id", sort=False):
        statuses: list[str] = []
        reasons: list[str] = []
        matched_iorfs: list[str] = []
        matched_orfs: list[str] = []
        concordant_count = 0
        overlap_count = 0
        evaluable_count = 0
        for row in group.to_dict("records"):
            if str(row.get("sample", "")) != sample:
                raise ValueError(f"cryptic_peptide_genomic_footprint sample mismatch for {peptide_id}")
            candidate_status = str(row.get("candidate_coordinate_status", ""))
            if candidate_status != "coordinate_evaluable":
                statuses.append(candidate_status or "not_evaluable_candidate_coordinate_qc")
                reasons.append(str(row.get("candidate_coordinate_reasons", "")))
                continue
            if str(row.get("reference_build", "")) != "GRCh38":
                statuses.append("not_evaluable_reference_or_coordinate_mismatch")
                reasons.append("candidate_reference_build_not_GRCh38")
                continue
            evaluable_count += 1
            blocks = parse_json_blocks(row.get("codon_blocks_transcript_order", ""))
            if not blocks:
                statuses.append("not_evaluable_candidate_coordinate_qc")
                reasons.append("missing_candidate_codon_blocks")
                continue
            candidate_by_iorf: dict[str, dict[str, object]] = {}
            for block in blocks:
                first_bin = block.start0 // COORDINATE_INDEX_BIN_SIZE
                last_bin = max(block.start0, block.end0 - 1) // COORDINATE_INDEX_BIN_SIZE
                for bin_id in range(first_bin, last_bin + 1):
                    for normal in normal_index.get((block.chrom, block.strand, bin_id), []):
                        candidate_by_iorf[str(normal["iorf_id"])] = normal
            normal_candidates = list(candidate_by_iorf.values())
            row_statuses: list[str] = []
            for normal in normal_candidates:
                status, reason = classify_coordinate_match(
                    blocks,
                    int(row.get("peptide_cds_start0", 0)),
                    normal["blocks"],
                )
                row_statuses.append(status)
                if status != "no_coordinate_overlap":
                    overlap_count += 1
                    evidence = {
                        "sample": sample,
                        "peptide_record_id": peptide_id,
                        "parent_record_id": row.get("parent_record_id", ""),
                        "mhc_class": row.get("mhc_class", ""),
                        "peptide": row.get("peptide", ""),
                        "candidate_coordinate_status": candidate_status,
                        "normal_smorf_coordinate_status": status,
                        "coordinate_matching_reason": reason,
                        "smorf_iorf_id": normal["iorf_id"],
                        "smorf_orf_id": normal["orf_id"],
                        "smorf_source_record_id": normal["source_record_id"],
                        "candidate_codon_blocks": blocks_to_text(blocks),
                        "normal_smorf_orfcds_blocks": normal["blocks_text"],
                        "reference_build": "GRCh38",
                    }
                    evidence_rows.append(evidence)
                if status == COORDINATE_CONCORDANT:
                    concordant_count += 1
                    matched_iorfs.append(str(normal["iorf_id"]))
                    matched_orfs.append(str(normal["orf_id"]))
            if row_statuses:
                statuses.append(choose_coordinate_status(row_statuses))
            else:
                statuses.append("no_coordinate_overlap")
        if not statuses:
            statuses.append("no_coordinate_overlap")
        summaries[str(peptide_id)] = {
            "candidate_coordinate_status": "coordinate_evaluable" if evaluable_count else choose_coordinate_status(statuses),
            "normal_smorf_coordinate_status": choose_coordinate_status(statuses),
            "coordinate_match_count": concordant_count,
            "matched_smorf_iorf_ids": compact_list(matched_iorfs),
            "matched_smorf_orf_ids": compact_list(matched_orfs),
            "coordinate_matching_reasons": compact_list(reasons + statuses),
            "coordinate_overlap_count": overlap_count,
            "coordinate_evaluable_parent_count": evaluable_count,
        }
    return summaries, evidence_rows


def base_output_row(row: dict[str, object], resource_freeze_version: str) -> dict[str, object]:
    out = dict(row)
    out["hla_class_normalized"] = normalize_hla_class(row.get("mhc_class", ""))
    peptide = str(out.get("peptide", ""))
    if peptide and not str(out.get("peptide_sequence_sha256", "")).strip():
        out["peptide_sequence_sha256"] = sha256_text(peptide)
    if not str(out.get("peptide_core_status", "")).strip():
        out["peptide_core_status"] = "core"
    if "peptide_qc_reasons" not in out:
        out["peptide_qc_reasons"] = ""
    out["normal_smorf_coordinate_status"] = COORDINATE_STATUS
    out["resource_freeze_version"] = resource_freeze_version
    return out


def make_source_evidence(row: dict[str, object], resource_freeze_version: str) -> dict[str, object]:
    out = base_output_row(row, resource_freeze_version)
    out.update({
        "evidence_type": "source_core_parent_map",
        "source_resource": "08b-cryptic_core_qc",
        "resource_record_id": str(row.get("parent_record_id", "")),
        "normal_translation_status": str(row.get("normal_translation_status", "")),
        "normal_hla_presentation_status": str(row.get("normal_hla_presentation_status", "")),
        "external_normal_status": str(row.get("external_normal_status", "")),
        "smorf_iorf_id": "",
        "smorf_orf_id": "",
        "smorf_peptide_start_1based": "",
        "hla_ligand_peptide_sequence_id": "",
        "hla_ligand_donor": "",
        "hla_ligand_tissue": "",
        "hla_ligand_donor_hla_genotype": "",
    })
    return out


def make_smorf_evidence(row: dict[str, object], evidence: dict[str, str], resource_freeze_version: str) -> dict[str, object]:
    out = base_output_row(row, resource_freeze_version)
    source_record_id = str(evidence.get("source_record_id", ""))
    out.update({
        "evidence_type": "normal_smorf_translation",
        "source_resource": str(evidence.get("source_resource", "Chothani_2022_smORF_atlas")),
        "resource_record_id": source_record_id or str(evidence.get("iorf_id", "")),
        "normal_translation_status": SMORF_DETECTED,
        "normal_hla_presentation_status": str(row.get("normal_hla_presentation_status", "")),
        "external_normal_status": EXACT_EXCLUDED_STATUS,
        "smorf_iorf_id": str(evidence.get("iorf_id", "")),
        "smorf_orf_id": str(evidence.get("orf_id", "")),
        "smorf_peptide_start_1based": str(evidence.get("peptide_start_1based", "")),
        "hla_ligand_peptide_sequence_id": "",
        "hla_ligand_donor": "",
        "hla_ligand_tissue": "",
        "hla_ligand_donor_hla_genotype": "",
    })
    return out


def make_hla_evidence(row: dict[str, object], evidence: dict[str, str], resource_freeze_version: str) -> dict[str, object]:
    out = base_output_row(row, resource_freeze_version)
    donor = str(evidence.get("donor", ""))
    tissue = str(evidence.get("tissue", ""))
    seq_id = str(evidence.get("peptide_sequence_id", ""))
    out.update({
        "evidence_type": "normal_hla_ligand",
        "source_resource": str(evidence.get("source_resource", "HLA_Ligand_Atlas_2020.12")),
        "resource_record_id": "|".join(token for token in [seq_id, donor, tissue] if token),
        "normal_translation_status": str(row.get("normal_translation_status", "")),
        "normal_hla_presentation_status": HLA_DETECTED,
        "external_normal_status": EXACT_EXCLUDED_STATUS,
        "smorf_iorf_id": "",
        "smorf_orf_id": "",
        "smorf_peptide_start_1based": "",
        "hla_ligand_peptide_sequence_id": seq_id,
        "hla_ligand_donor": donor,
        "hla_ligand_tissue": tissue,
        "hla_ligand_donor_hla_genotype": str(evidence.get("donor_hla_genotype", "")),
    })
    return out


FINAL_SIDECAR_FIELDS = [
    "sample",
    "peptide_record_id",
    "mhc_class",
    "peptide",
    "peptide_length",
    "peptide_start",
    "parent_record_id",
    "source_parent_id",
    "parent_sequence",
    "parent_sequence_sha256",
    "nc_class",
    "TPM_tumor",
    "TPM_ctrl",
    "log2FC",
    "expression_qc_status",
    "coordinate_mapping_status",
    "reference_translation_status",
    "chromosome",
    "strand",
    "block_chain",
    "junction_qc_status",
    "required_junction_count",
    "tumor_junction_support",
    "matched_normal_junction_annotation",
    "peptide_crosses_junction",
    "human_reference_peptide_status",
    "external_normal_status",
    "normal_translation_status",
    "normal_smorf_coordinate_status",
    "normal_hla_presentation_status",
    "selection_rank",
    "selection_origin",
    "supporting_parent_window_count",
    "primary_deferred_excluded_status",
    "primary_deferred_excluded_reason",
    "cryptic_discovery_fdr_status",
]


def row_key(row: dict[str, object]) -> tuple[str, str]:
    return (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper())


def occurrence_key(row: dict[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(row.get("peptide_record_id", "")),
        str(row.get("parent_record_id", "")),
        normalize_hla_class(row.get("mhc_class", "")),
        str(row.get("peptide", "")).upper(),
        str(row.get("peptide_start", "")),
        str(row.get("peptide_length", "")),
    )


def resolved_human_reference_status(source: dict[str, object], retained: dict[str, object]) -> str:
    status = str(source.get("human_reference_peptide_status", "")).strip()
    if not status or status == HUMAN_REFERENCE_NOT_EVALUATED:
        status = str(retained.get("human_reference_peptide_status", "")).strip()
    return status


def build_final_parent_sidecar(
    sample: str,
    retained_rows: list[dict[str, object]],
    core_parent_map: pd.DataFrame,
    deferred_parent_map: pd.DataFrame,
    parent_core: pd.DataFrame,
    parent_coordinates: pd.DataFrame,
    parent_ranked: pd.DataFrame,
    peptide_junction: pd.DataFrame,
    parent_junction_summary: pd.DataFrame,
) -> list[dict[str, object]]:
    if not retained_rows:
        return []
    retained_by_key = {row_key(row): row for row in retained_rows}
    retained_ids = {str(row.get("peptide_record_id", "")) for row in retained_rows}
    retained_id_by_key = {
        row_key(row): str(row.get("peptide_record_id", ""))
        for row in retained_rows
    }
    maps = [core_parent_map]
    if not deferred_parent_map.empty:
        maps.append(deferred_parent_map)
    parent_map = pd.concat(maps, ignore_index=True) if maps else pd.DataFrame()
    if parent_map.empty:
        return []

    parent_meta = {
        str(row.get("parent_record_id", "")): row
        for row in parent_core.to_dict("records")
    } if not parent_core.empty else {}
    coord_meta = {
        str(row.get("parent_record_id", "")): row
        for row in parent_coordinates.to_dict("records")
    } if not parent_coordinates.empty else {}
    rank_meta = {
        str(row.get("parent_record_id", "")): row
        for row in parent_ranked.to_dict("records")
    } if not parent_ranked.empty else {}
    junction_meta = {
        str(row.get("parent_record_id", "")): row
        for row in parent_junction_summary.to_dict("records")
    } if not parent_junction_summary.empty else {}
    peptide_junction_meta = {
        (
            str(row.get("peptide_record_id", "")),
            str(row.get("parent_record_id", "")),
            normalize_hla_class(row.get("mhc_class", "")),
            str(row.get("peptide", "")).upper(),
            str(row.get("peptide_start", "")),
            str(row.get("peptide_length", "")),
        ): row
        for row in peptide_junction.to_dict("records")
    } if not peptide_junction.empty else {}

    support_counts: defaultdict[str, int] = defaultdict(int)
    candidate_map_rows: list[dict[str, object]] = []
    for source in parent_map.to_dict("records"):
        key = row_key(source)
        retained = retained_by_key.get(key)
        if retained is None:
            continue
        row = dict(source)
        if not str(row.get("peptide_record_id", "")).strip():
            row["peptide_record_id"] = retained_id_by_key[key]
        if str(row.get("peptide_record_id", "")) not in retained_ids:
            row["peptide_record_id"] = retained_id_by_key[key]
        support_counts[str(row.get("peptide_record_id", ""))] += 1
        candidate_map_rows.append(row)

    sidecar_rows: list[dict[str, object]] = []
    for source in candidate_map_rows:
        key = row_key(source)
        retained = retained_by_key[key]
        peptide_id = str(source.get("peptide_record_id", ""))
        parent_id = str(source.get("parent_record_id", ""))
        parent = parent_meta.get(parent_id, {})
        coord = coord_meta.get(parent_id, {})
        rank = rank_meta.get(parent_id, {})
        parent_junction = junction_meta.get(parent_id, {})
        peptide_junction_row = peptide_junction_meta.get(
            (
                peptide_id,
                parent_id,
                normalize_hla_class(source.get("mhc_class", "")),
                str(source.get("peptide", "")).upper(),
                str(source.get("peptide_start", "")),
                str(source.get("peptide_length", "")),
            ),
            {},
        )
        normal_junction_annotation = ";".join(
            token
            for token in [
                f"any_ge1={parent_junction.get('normal_any_required_junction_unique_ge1', '')}",
                f"all_ge1={parent_junction.get('normal_all_required_junctions_unique_ge1', '')}",
                f"any_ge2={parent_junction.get('normal_any_required_junction_unique_ge2', '')}",
                f"all_ge2={parent_junction.get('normal_all_required_junctions_unique_ge2', '')}",
            ]
            if token.split("=", 1)[1] != ""
        )
        sidecar_rows.append({
            "sample": sample,
            "peptide_record_id": peptide_id,
            "mhc_class": source.get("mhc_class", retained.get("mhc_class", "")),
            "peptide": source.get("peptide", retained.get("peptide", "")),
            "peptide_length": source.get("peptide_length", retained.get("peptide_length", "")),
            "peptide_start": source.get("peptide_start", retained.get("peptide_start", "")),
            "parent_record_id": parent_id,
            "source_parent_id": source.get("source_parent_id", ""),
            "parent_sequence": parent.get("parent_sequence", ""),
            "parent_sequence_sha256": parent.get("parent_sequence_sha256", ""),
            "nc_class": parent.get("nc_class", ""),
            "TPM_tumor": parent.get("TPM_tumor", ""),
            "TPM_ctrl": parent.get("TPM_ctrl", ""),
            "log2FC": parent.get("log2FC", ""),
            "expression_qc_status": parent.get("expression_qc_status", ""),
            "coordinate_mapping_status": coord.get("coordinate_mapping_status", ""),
            "reference_translation_status": coord.get("reference_genomic_translation_status", ""),
            "chromosome": coord.get("chromosome", ""),
            "strand": coord.get("strand", ""),
            "block_chain": coord.get("blocks_transcript_order", ""),
            "junction_qc_status": parent_junction.get("junction_qc_status", parent.get("junction_support_status", "")),
            "required_junction_count": parent_junction.get("required_junction_count", ""),
            "tumor_junction_support": parent_junction.get("tumor_min_required_unique_reads", ""),
            "matched_normal_junction_annotation": normal_junction_annotation,
            "peptide_crosses_junction": peptide_junction_row.get("peptide_crosses_junction", ""),
            "human_reference_peptide_status": resolved_human_reference_status(source, retained),
            "external_normal_status": retained.get("external_normal_status", ""),
            "normal_translation_status": retained.get("normal_translation_status", ""),
            "normal_smorf_coordinate_status": retained.get("normal_smorf_coordinate_status", ""),
            "normal_hla_presentation_status": retained.get("normal_hla_presentation_status", ""),
            "selection_rank": rank.get("parent_rank", ""),
            "selection_origin": retained.get("prebinding_selection_origin", "initial_core"),
            "supporting_parent_window_count": support_counts[peptide_id],
            "primary_deferred_excluded_status": retained.get("external_normal_status", ""),
            "primary_deferred_excluded_reason": retained.get("external_normal_qc_reasons", ""),
            "cryptic_discovery_fdr_status": parent.get("cryptic_discovery_fdr_status", "not_estimable_single_pair_rule_based_qc"),
        })
    return sidecar_rows


def ranked_parent_stream(parent_core: pd.DataFrame, parent_ranked: pd.DataFrame) -> list[dict[str, object]]:
    if parent_core.empty or parent_ranked.empty:
        return []
    required_columns(parent_core, ["parent_record_id", "parent_sequence"], "cryptic_parent_core.tsv")
    required_columns(parent_ranked, ["parent_record_id", "parent_rank"], "cryptic_parent_ranked.tsv")
    core_by_parent = {
        str(row.get("parent_record_id", "")): row
        for row in parent_core.to_dict("records")
    }
    ranked_rows = sorted(
        parent_ranked.to_dict("records"),
        key=lambda row: parse_int(row.get("parent_rank", ""), "cryptic_parent_ranked parent_rank"),
    )
    stream: list[dict[str, object]] = []
    for rank_row in ranked_rows:
        parent_id = str(rank_row.get("parent_record_id", ""))
        parent = core_by_parent.get(parent_id)
        if parent is None:
            continue
        merged = dict(parent)
        merged["_parent_rank"] = parse_int(rank_row.get("parent_rank", ""), f"parent_rank for {parent_id}")
        stream.append(merged)
    return stream


def iter_refill_key_stream(
    sample: str,
    ranked_parents: list[dict[str, object]],
    seen_keys: set[tuple[str, str]],
    mhc_i_lengths: Iterable[int],
    mhc_ii_lengths: Iterable[int],
) -> Iterable[tuple[tuple[str, str], dict[str, object]]]:
    emitted = set(seen_keys)
    for parent in ranked_parents:
        sequence = str(parent.get("parent_sequence", "")).upper()
        for mhc_class, lengths in [("HLA-I", mhc_i_lengths), ("HLA-II", mhc_ii_lengths)]:
            for start, length, peptide in iter_windows(sequence, lengths):
                key = (mhc_class, peptide)
                if key in emitted:
                    continue
                emitted.add(key)
                yield key, {
                    "sample": sample,
                    "parent_record_id": parent.get("parent_record_id", ""),
                    "source_parent_id": parent.get("source_parent_id", ""),
                    "mhc_class": mhc_class,
                    "peptide_start": start,
                    "peptide_length": length,
                    "peptide": peptide,
                    "_parent_rank": parent.get("_parent_rank", ""),
                }


def upstream_configured_lengths(upstream_manifest: dict[str, object]) -> tuple[list[int], list[int]]:
    signature = upstream_manifest.get("input_signature", {})
    if not isinstance(signature, dict):
        raise ValueError("formal v1.1 ranked refill requires 08b input_signature")
    raw_i = signature.get("mhc_i_lengths")
    raw_ii = signature.get("mhc_ii_lengths")
    if not isinstance(raw_i, list) or not isinstance(raw_ii, list):
        raise ValueError("formal v1.1 ranked refill requires mhc_i_lengths and mhc_ii_lengths in 08b manifest")
    mhc_i = sorted({int(value) for value in raw_i})
    mhc_ii = sorted({int(value) for value in raw_ii})
    if not mhc_i or not mhc_ii:
        raise ValueError("formal v1.1 ranked refill requires non-empty configured MHC lengths")
    if not set(mhc_i).issubset(ALLOWED_LENGTHS["HLA-I"]):
        raise ValueError(f"08b MHC-I lengths are unsupported by 08c refill: {mhc_i}")
    if not set(mhc_ii).issubset(ALLOWED_LENGTHS["HLA-II"]):
        raise ValueError(f"08b MHC-II lengths are unsupported by 08c refill: {mhc_ii}")
    return mhc_i, mhc_ii


def build_parent_map_for_refill_keys(
    sample: str,
    source_rows_by_key: dict[tuple[str, str], dict[str, object]],
    peptide_id_by_key: dict[tuple[str, str], str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, peptide_id in peptide_id_by_key.items():
        source = source_rows_by_key.get(key)
        if source is None:
            raise ValueError(f"refill key missing materialized source row: {key}")
        rows.append(
            make_refill_parent_map_row(
                sample,
                source,
                key[0],
                parse_int(source.get("peptide_start", ""), f"refill peptide_start for {key}"),
                parse_int(source.get("peptide_length", ""), f"refill peptide_length for {key}"),
                key[1],
                peptide_id,
            )
        )
    return rows


def build_parent_map_for_target_keys(
    sample: str,
    ranked_parents: list[dict[str, object]],
    peptide_id_by_key: dict[tuple[str, str], str],
    mhc_i_lengths: Iterable[int],
    mhc_ii_lengths: Iterable[int],
    metadata_by_key: Optional[dict[tuple[str, str], dict[str, object]]] = None,
) -> list[dict[str, object]]:
    if not peptide_id_by_key:
        return []
    metadata_by_key = metadata_by_key or {}
    rows: list[dict[str, object]] = []
    for parent in ranked_parents:
        sequence = str(parent.get("parent_sequence", "")).upper()
        for mhc_class, lengths in [("HLA-I", mhc_i_lengths), ("HLA-II", mhc_ii_lengths)]:
            for start, length, peptide in iter_windows(sequence, lengths):
                key = (mhc_class, peptide)
                peptide_id = peptide_id_by_key.get(key)
                if not peptide_id:
                    continue
                row = make_refill_parent_map_row(sample, parent, mhc_class, start, length, peptide, peptide_id)
                metadata = metadata_by_key.get(key, {})
                for field in [
                    "peptide_core_status",
                    "peptide_qc_reasons",
                    "human_reference_peptide_status",
                    "normal_hla_presentation_status",
                    "normal_translation_status",
                    "external_normal_status",
                    "prebinding_selection_origin",
                ]:
                    if field in metadata:
                        row[field] = metadata[field]
                rows.append(row)
    return rows


def validate_final_core_contract(
    retained_rows: list[dict[str, object]],
    final_sidecar_rows: list[dict[str, object]],
    footprint_rows: list[dict[str, object]],
    junction_rows: list[dict[str, object]],
    mhc_i_lengths: Iterable[int],
    mhc_ii_lengths: Iterable[int],
    *,
    human_reference_evaluated: bool,
    formal_v11_ranked: bool,
) -> None:
    allowed_lengths = {
        "HLA-I": {int(value) for value in mhc_i_lengths},
        "HLA-II": {int(value) for value in mhc_ii_lengths},
    }
    for row in retained_rows:
        peptide_id = str(row.get("peptide_record_id", "")).strip()
        peptide = str(row.get("peptide", "")).strip().upper()
        if not peptide_id:
            raise AssertionError("final Core row missing peptide_record_id")
        if str(row.get("sample", "")).strip() == "":
            raise AssertionError(f"final Core row {peptide_id} missing sample")
        if str(row.get("peptide_core_status", "")) != "core":
            raise AssertionError(f"final Core row {peptide_id} has non-core peptide_core_status")
        if str(row.get("peptide_qc_reasons", "")).strip():
            raise AssertionError(f"final Core row {peptide_id} has non-empty peptide_qc_reasons")
        if str(row.get("peptide_sequence_sha256", "")) != sha256_text(peptide):
            raise AssertionError(f"final Core row {peptide_id} peptide_sequence_sha256 does not match peptide")
        hla_class = normalize_hla_class(row.get("mhc_class", ""))
        peptide_length = parse_int(row.get("peptide_length", ""), f"final Core peptide_length for {peptide_id}")
        if peptide_length != len(peptide):
            raise AssertionError(f"final Core row {peptide_id} peptide_length does not match peptide")
        if peptide_length not in allowed_lengths.get(hla_class, set()):
            raise AssertionError(
                f"final Core row {peptide_id} has {hla_class} length outside configured lengths: {peptide_length}"
            )
        if (
            human_reference_evaluated
            and str(row.get("human_reference_peptide_status", "")) != HUMAN_REFERENCE_NOT_DETECTED
        ):
            raise AssertionError(f"final Core row {peptide_id} has invalid human-reference pass status")

    if not formal_v11_ranked or not retained_rows:
        return

    sidecar_ids = {str(row.get("peptide_record_id", "")) for row in final_sidecar_rows}
    retained_ids = {str(row.get("peptide_record_id", "")) for row in retained_rows}
    missing_sidecar_ids = sorted(retained_ids - sidecar_ids)
    if missing_sidecar_ids:
        raise AssertionError(
            "final sidecar missing retained peptide parent mappings: " + ", ".join(missing_sidecar_ids[:5])
        )
    footprint_keys = {occurrence_key(row) for row in footprint_rows}
    junction_keys = {occurrence_key(row) for row in junction_rows}
    for idx, row in enumerate(final_sidecar_rows, start=1):
        key = occurrence_key(row)
        if key not in footprint_keys:
            raise AssertionError(f"final sidecar row {idx} missing matching genomic footprint evidence")
        if key not in junction_keys:
            raise AssertionError(f"final sidecar row {idx} missing matching junction evidence")
        if (
            human_reference_evaluated
            and str(row.get("human_reference_peptide_status", "")) != HUMAN_REFERENCE_NOT_DETECTED
        ):
            raise AssertionError(f"final sidecar row {idx} has invalid human-reference pass status")


def build_external_normal_qc(args: argparse.Namespace) -> dict[str, object]:
    started_at = ts()
    policy_version = str(getattr(args, "policy_version", POLICY_VERSION_V1))
    if policy_version not in {POLICY_VERSION_V1, POLICY_VERSION_V11}:
        raise ValueError(f"Unsupported CrypticExternalNormalQC policy_version: {policy_version}")
    coordinate_enabled = bool(getattr(args, "coordinate_matching_enabled", False))
    if policy_version == POLICY_VERSION_V1 and coordinate_enabled:
        raise ValueError("coordinate matching is not implemented in policy v1.0")
    if policy_version == POLICY_VERSION_V11 and not coordinate_enabled:
        raise ValueError("coordinate_matching_enabled=true is required for policy v1.1")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    script_sha256 = sha256_file(Path(__file__))

    core_path = Path(args.cryptic_peptide_core)
    parent_map_path = Path(args.cryptic_peptide_parent_map)
    deferred_path = Path(args.cryptic_peptide_deferred) if args.cryptic_peptide_deferred else Path("")
    deferred_parent_map_path = (
        Path(args.cryptic_peptide_deferred_parent_map)
        if getattr(args, "cryptic_peptide_deferred_parent_map", "")
        else Path("")
    )
    parent_core_path = Path(args.cryptic_parent_core) if getattr(args, "cryptic_parent_core", "") else Path("")
    parent_ranked_path = Path(args.cryptic_parent_ranked) if getattr(args, "cryptic_parent_ranked", "") else Path("")
    parent_junction_summary_path = (
        Path(args.cryptic_parent_junction_summary)
        if getattr(args, "cryptic_parent_junction_summary", "")
        else Path("")
    )
    peptide_junction_evidence_path = (
        Path(args.cryptic_peptide_junction_evidence)
        if getattr(args, "cryptic_peptide_junction_evidence", "")
        else Path("")
    )
    upstream_manifest_path = Path(args.upstream_manifest)
    coordinate_sidecars: dict[str, Path] = {}
    if is_v11(policy_version):
        coordinate_sidecars = {
            "cryptic_parent_coordinates.tsv": Path(getattr(args, "cryptic_parent_coordinates", "")),
            "cryptic_parent_orfcds.tsv": Path(getattr(args, "cryptic_parent_orfcds", "")),
            "cryptic_peptide_genomic_footprint.tsv": Path(getattr(args, "cryptic_peptide_genomic_footprint", "")),
        }
    expected_upstream = (
        EXPECTED_UPSTREAM_POLICY_VERSION_V11 if is_v11(policy_version) else EXPECTED_UPSTREAM_POLICY_VERSION_V1
    )
    upstream_manifest = validate_upstream_08b(
        core_path,
        parent_map_path,
        upstream_manifest_path,
        expected_upstream,
        coordinate_sidecars=coordinate_sidecars,
    )
    resources_evaluated, resource_identity, resource_paths = resource_identities(args)
    coordinate_resource_identity: dict[str, object] = {}
    coordinate_resource_paths: dict[str, Path] = {}
    if is_v11(policy_version):
        coordinate_resource_identity, coordinate_resource_paths = coordinate_resource_identities(args)
    freeze_version = str(resource_identity.get("freeze_version", "")) if resources_evaluated else ""
    coordinate_freeze_version = str(coordinate_resource_identity.get("coordinate_freeze_version", ""))

    upstream_selection = upstream_manifest.get("candidate_selection", {})
    upstream_selection_mode = str(upstream_selection.get("mode", "")) if isinstance(upstream_selection, dict) else ""
    formal_v11_ranked = is_v11(policy_version) and upstream_selection_mode == "ranked_cap"
    refill_mhc_i_lengths = sorted(ALLOWED_LENGTHS["HLA-I"])
    refill_mhc_ii_lengths = sorted(ALLOWED_LENGTHS["HLA-II"])
    human_reference_summary_from_08b = upstream_manifest.get("human_reference_summary", {})
    human_proteome_fasta = Path(str(getattr(args, "human_proteome_fasta", "") or ""))
    if (not str(human_proteome_fasta).strip() or str(human_proteome_fasta) == ".") and isinstance(
        human_reference_summary_from_08b, dict
    ):
        human_path = str(human_reference_summary_from_08b.get("path", "") or "").strip()
        human_proteome_fasta = Path(human_path) if human_path else Path("")
    if formal_v11_ranked:
        refill_mhc_i_lengths, refill_mhc_ii_lengths = upstream_configured_lengths(upstream_manifest)
        output_sig = upstream_manifest.get("output_signature", {})
        if not isinstance(output_sig, dict):
            raise ValueError("formal v1.1 ranked external-normal QC requires upstream output_signature")
        required_ranked_sidecars = {
            "cryptic_parent_core.tsv": parent_core_path,
            "cryptic_parent_ranked.tsv": parent_ranked_path,
            "cryptic_peptide_deferred_parent_map.tsv": deferred_parent_map_path,
            "cryptic_parent_junction_summary.tsv": parent_junction_summary_path,
            "cryptic_peptide_junction_evidence.tsv": peptide_junction_evidence_path,
        }
        for label, path in required_ranked_sidecars.items():
            if not str(path).strip() or not path.is_file():
                raise FileNotFoundError(f"formal v1.1 ranked external-normal QC requires {label}: {path}")
            validate_signature_entry(path, output_sig.get(label), label)
        if not str(human_proteome_fasta).strip() or not human_proteome_fasta.is_file():
            raise FileNotFoundError(
                "formal v1.1 ranked external-normal QC requires the 08b human reference proteome for refill-time QC"
            )

    input_signature = {
        "policy_version": policy_version,
        "script_sha256": script_sha256,
        "coordinate_utils_sha256": sha256_file(COORDINATE_UTILS_PATH) if is_v11(policy_version) else "",
        "sample": args.sample,
        "allow_missing_external_normal_resources": bool(args.allow_missing_external_normal_resources),
        "coordinate_matching_enabled": coordinate_enabled,
        "hla_class_normalization": "MHC-I/HLA-I->HLA-I; MHC-II/HLA-II->HLA-II",
        "exact_match_policy": "exact peptide sequence plus normalized HLA class",
        "coordinate_matching_status": "coordinate_frame_matching_enabled" if is_v11(policy_version) else COORDINATE_STATUS,
        "upstream_inputs": {
            "cryptic_peptide_core": file_identity(core_path),
            "cryptic_peptide_parent_map": file_identity(parent_map_path),
            "cryptic_peptide_deferred": file_identity(deferred_path) if args.cryptic_peptide_deferred else {},
            "cryptic_peptide_deferred_parent_map": file_identity(deferred_parent_map_path) if deferred_parent_map_path.is_file() else {},
            "cryptic_parent_core": file_identity(parent_core_path) if parent_core_path.is_file() else {},
            "cryptic_parent_ranked": file_identity(parent_ranked_path) if parent_ranked_path.is_file() else {},
            "cryptic_parent_junction_summary": file_identity(parent_junction_summary_path) if parent_junction_summary_path.is_file() else {},
            "cryptic_peptide_junction_evidence": file_identity(peptide_junction_evidence_path) if peptide_junction_evidence_path.is_file() else {},
            "human_reference_proteome_fasta_for_refill": (
                file_identity(human_proteome_fasta) if formal_v11_ranked else {}
            ),
            "upstream_manifest": file_identity(upstream_manifest_path),
            "coordinate_sidecars": {label: file_identity(path) for label, path in coordinate_sidecars.items()},
            "coordinate_utils": file_identity(COORDINATE_UTILS_PATH) if is_v11(policy_version) else {},
        },
        "upstream_08b": {
            "policy_version": upstream_manifest.get("policy_version", ""),
            "run_status": upstream_manifest.get("run_status", ""),
            "binding_eligible": upstream_manifest.get("binding_eligible", ""),
            "candidate_selection_policy_version": upstream_manifest.get("candidate_selection_policy_version", ""),
            "candidate_selection": upstream_manifest.get("candidate_selection", {}),
            "cryptic_peptide_core_output_signature": upstream_manifest.get("output_signature", {}).get(
                "cryptic_peptide_core.tsv", {}
            ),
            "cryptic_peptide_parent_map_output_signature": upstream_manifest.get("output_signature", {}).get(
                "cryptic_peptide_parent_map.tsv", {}
            ),
            "cryptic_peptide_deferred_output_signature": upstream_manifest.get("output_signature", {}).get(
                "cryptic_peptide_deferred.tsv", {}
            ),
        },
        "refill_policy": {
            "enabled": formal_v11_ranked or bool(args.cryptic_peptide_deferred),
            "max_hla_i_peptides": int(args.max_hla_i_peptides or 0),
            "max_hla_ii_peptides": int(args.max_hla_ii_peptides or 0),
            "mhc_i_lengths": refill_mhc_i_lengths if formal_v11_ranked else [],
            "mhc_ii_lengths": refill_mhc_ii_lengths if formal_v11_ranked else [],
        },
        "resource_identities": resource_identity,
        "coordinate_resource_identities": coordinate_resource_identity,
    }

    output_paths = [
        outdir / "cryptic_external_normal_annotated_core.tsv",
        outdir / "cryptic_external_normal_evidence.tsv",
        outdir / "cryptic_external_normal_excluded.tsv",
        outdir / "cryptic_tumor_restricted_primary_core.tsv",
        outdir / "cryptic_tumor_restricted_primary_core_hla_i.tsv",
        outdir / "cryptic_tumor_restricted_primary_core_hla_ii.tsv",
        outdir / "cryptic_tumor_restricted_primary_core_hla_i.fasta",
        outdir / "cryptic_tumor_restricted_primary_core_hla_ii.fasta",
        outdir / "cryptic_tumor_restricted_primary_core.fasta",
        outdir / "cryptic_external_normal_refill_deferred.tsv",
        outdir / "cryptic_external_normal_refill_parent_map.tsv",
        outdir / "cryptic_external_normal_refill_genomic_footprint.tsv",
        outdir / "cryptic_external_normal_refill_junction_evidence.tsv",
        outdir / "cryptic_external_normal_completed_parent_map.tsv",
        outdir / "cryptic_external_normal_completed_genomic_footprint.tsv",
        outdir / "cryptic_external_normal_completed_junction_evidence.tsv",
        outdir / "cryptic_final_peptide_parent_sidecar.tsv",
        outdir / "stagewise_qc.tsv",
    ]
    if is_v11(policy_version):
        output_paths.append(outdir / "cryptic_smorf_coordinate_evidence.tsv")
    manifest_path = outdir / "run_manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("input_signature") != input_signature:
            raise ValueError(
                f"Existing CrypticExternalNormalQC manifest does not match current inputs/config: {manifest_path}"
            )
        old_output_signature = old.get("output_signature")
        if isinstance(old_output_signature, dict) and outputs_match_signature(old_output_signature):
            return {**old, "reused": True}

    core = load_table(core_path)
    parent_map = load_table(parent_map_path)
    deferred_parent_map = load_optional_table(deferred_parent_map_path)
    parent_core = load_optional_table(parent_core_path)
    parent_ranked = load_optional_table(parent_ranked_path)
    parent_junction_summary = load_optional_table(parent_junction_summary_path)
    peptide_junction_evidence = load_optional_table(peptide_junction_evidence_path)
    required_columns(
        core,
        ["sample", "peptide_record_id", "mhc_class", "peptide_length", "peptide"],
        "cryptic_peptide_core.tsv",
    )
    required_columns(
        parent_map,
        ["sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class", "peptide_length", "peptide"],
        "cryptic_peptide_parent_map.tsv",
    )
    if core["peptide_record_id"].duplicated().any():
        raise ValueError("cryptic_peptide_core.tsv must contain unique peptide_record_id values")
    core_ids = {str(value) for value in core["peptide_record_id"].tolist()}
    core_parent_map = select_core_parent_map_rows(parent_map, core_ids)
    validate_core_parent_contract(core, core_parent_map, args.sample)
    core_keys = {
        (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper())
        for row in core.to_dict("records")
    }
    deferred = (
        load_deferred_peptide_core(deferred_path, args.sample, core_keys)
        if args.cryptic_peptide_deferred
        else pd.DataFrame()
    )
    if not formal_v11_ranked and not deferred.empty and deferred_parent_map.empty:
        deferred_parent_map = deferred.copy()
    cap_by_class = {
        "HLA-I": int(args.max_hla_i_peptides or 0),
        "HLA-II": int(args.max_hla_ii_peptides or 0),
    }
    upstream_selection = upstream_manifest.get("candidate_selection", {})
    if isinstance(upstream_selection, dict):
        cap_by_class["HLA-I"] = cap_by_class["HLA-I"] or int(upstream_selection.get("max_hla_i_peptides") or 0)
        cap_by_class["HLA-II"] = cap_by_class["HLA-II"] or int(upstream_selection.get("max_hla_ii_peptides") or 0)
    cap_by_class = {klass: (value if value > 0 else 10**18) for klass, value in cap_by_class.items()}
    coordinate_summaries: dict[str, dict[str, object]] = {}
    coordinate_evidence_rows: list[dict[str, object]] = []
    parent_coordinate_evaluable_count = 0
    parent_coordinate_not_evaluable_count = 0
    normal_blocks: dict[str, dict[str, object]] = {}
    ranked_parents_for_completion: list[dict[str, object]] = []
    completed_parent_map_rows: list[dict[str, object]] = []
    completed_footprint_rows: list[dict[str, object]] = []
    completed_junction_rows: list[dict[str, object]] = []
    if is_v11(policy_version):
        parent_coordinates = load_table(coordinate_sidecars["cryptic_parent_coordinates.tsv"])
        parent_orfcds = load_table(coordinate_sidecars["cryptic_parent_orfcds.tsv"])
        footprint = load_table(coordinate_sidecars["cryptic_peptide_genomic_footprint.tsv"])
        required_columns(
            parent_coordinates,
            ["sample", "parent_record_id", "coordinate_mapping_status"],
            "cryptic_parent_coordinates",
        )
        if not parent_coordinates["sample"].astype(str).eq(args.sample).all():
            raise ValueError("cryptic_parent_coordinates sample must match --sample")
        validate_coordinate_sidecar_contract(
            core,
            core_parent_map,
            parent_coordinates,
            parent_orfcds,
            footprint,
            args.sample,
        )
        parent_coordinate_evaluable_count = int(
            parent_coordinates["coordinate_mapping_status"].astype(str).eq("coordinate_evaluable").sum()
        )
        parent_coordinate_not_evaluable_count = int(len(parent_coordinates) - parent_coordinate_evaluable_count)
        normal_blocks = load_normal_smorf_coordinate_blocks(
            coordinate_resource_paths["normal_smorf_coordinates"],
            coordinate_resource_paths["normal_smorf_orfcds"],
        )
        if formal_v11_ranked:
            ranked_parents_for_completion = ranked_parent_stream(parent_core, parent_ranked)
            core_peptide_id_by_key = {
                (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper()):
                str(row.get("peptide_record_id", ""))
                for row in core.to_dict("records")
            }
            core_metadata_by_key = {
                (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper()): row
                for row in core.to_dict("records")
            }
            completed_parent_map_rows = build_parent_map_for_target_keys(
                args.sample,
                ranked_parents_for_completion,
                core_peptide_id_by_key,
                refill_mhc_i_lengths,
                refill_mhc_ii_lengths,
                core_metadata_by_key,
            )
            if completed_parent_map_rows:
                completed_core_parent_map = pd.DataFrame(completed_parent_map_rows)
                validate_core_parent_contract(core, completed_core_parent_map, args.sample)
                completed_footprint_rows = build_refill_peptide_footprints(
                    args.sample,
                    completed_parent_map_rows,
                    parent_coordinates,
                    parent_orfcds,
                )
                completed_junction_rows = annotate_peptide_junctions(
                    sample=args.sample,
                    peptide_parent_rows=completed_parent_map_rows,
                    peptide_footprint_rows=completed_footprint_rows,
                    parent_summary_rows=parent_junction_summary.to_dict("records"),
                )
                core_parent_map = completed_core_parent_map
                footprint = pd.DataFrame(completed_footprint_rows)
                peptide_junction_evidence = pd.DataFrame(completed_junction_rows)
        coordinate_summaries, coordinate_evidence_rows = evaluate_coordinate_evidence(
            core,
            footprint,
            normal_blocks,
            args.sample,
        )

    smorf_lookup: set[tuple[str, str]] = set()
    hla_lookup: set[tuple[str, str]] = set()
    smorf_evidence: dict[tuple[str, str], list[dict[str, str]]] = {}
    hla_evidence: dict[tuple[str, str], list[dict[str, str]]] = {}
    if resources_evaluated and (not core.empty or formal_v11_ranked):
        smorf_index = load_table(resource_paths["smorf_match_index"])
        hla_index = load_table(resource_paths["hla_ligand_match_index"])
        smorf_parent_map = load_table(resource_paths["smorf_parent_map"])
        hla_ligand_evidence = load_table(resource_paths["hla_ligand_evidence"])
        required_columns(smorf_index, ["hla_class", "peptide_sequence"], "normal_smorf_peptide_match_index")
        required_columns(hla_index, ["hla_class", "peptide_sequence"], "normal_hla_peptide_match_index")
        required_columns(smorf_parent_map, ["hla_class", "peptide_sequence"], "normal_smorf_peptide_parent_map")
        required_columns(hla_ligand_evidence, ["hla_class", "peptide_sequence"], "normal_hla_ligand_evidence")
        smorf_lookup = build_lookup(smorf_index)
        hla_lookup = build_lookup(hla_index)
        smorf_evidence = group_rows_by_key(smorf_parent_map)
        hla_evidence = group_rows_by_key(hla_ligand_evidence)

    candidate_records: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    excluded_rows: list[dict[str, object]] = []
    retained_rows: list[dict[str, object]] = []
    refill_deferred_rows: list[dict[str, object]] = []
    smorf_match_count = 0
    hla_match_count = 0
    both_match_count = 0
    exact_excluded_count = 0
    coordinate_concordant_count = 0
    coordinate_additional_excluded_count = 0
    refill_selected_count = 0
    retained_by_class: defaultdict[str, int] = defaultdict(int)

    source_core_records = core.to_dict("records")
    input_candidate_records: list[dict[str, object]] = []
    for record in source_core_records:
        row = dict(record)
        row["prebinding_selection_origin"] = "initial_core"
        input_candidate_records.append(row)
    if not formal_v11_ranked and not deferred.empty:
        input_candidate_records.extend(deferred.to_dict("records"))

    def annotate_and_route_candidate(record: dict[str, object]) -> None:
        nonlocal smorf_match_count
        nonlocal hla_match_count
        nonlocal both_match_count
        nonlocal exact_excluded_count
        nonlocal coordinate_concordant_count
        nonlocal coordinate_additional_excluded_count
        nonlocal refill_selected_count

        row = base_output_row(record, freeze_version)
        peptide_id = str(row.get("peptide_record_id", ""))
        origin = str(row.get("prebinding_selection_origin", "initial_core"))
        coordinate_summary = coordinate_summaries.get(peptide_id, {})
        coordinate_concordant = int(coordinate_summary.get("coordinate_match_count", 0) or 0) > 0
        row["candidate_coordinate_status"] = coordinate_summary.get("candidate_coordinate_status", "")
        row["normal_smorf_coordinate_status"] = (
            coordinate_summary.get("normal_smorf_coordinate_status", COORDINATE_STATUS)
            if is_v11(policy_version)
            else COORDINATE_STATUS
        )
        row["coordinate_match_count"] = int(coordinate_summary.get("coordinate_match_count", 0) or 0)
        row["matched_smorf_iorf_ids"] = str(coordinate_summary.get("matched_smorf_iorf_ids", ""))
        row["matched_smorf_orf_ids"] = str(coordinate_summary.get("matched_smorf_orf_ids", ""))
        row["coordinate_matching_reasons"] = str(coordinate_summary.get("coordinate_matching_reasons", ""))
        key = (str(row["hla_class_normalized"]), str(row["peptide"]).upper())
        smorf_match = resources_evaluated and key in smorf_lookup
        hla_match = resources_evaluated and key in hla_lookup

        if resources_evaluated:
            if smorf_match:
                row["normal_translation_status"] = SMORF_DETECTED
            elif coordinate_concordant:
                row["normal_translation_status"] = SMORF_COORDINATE_DETECTED
            else:
                row["normal_translation_status"] = SMORF_NOT_DETECTED
            row["normal_hla_presentation_status"] = HLA_DETECTED if hla_match else HLA_NOT_DETECTED
            row["normal_smorf_support_count"] = len(smorf_evidence.get(key, []))
            row["normal_hla_ligand_support_count"] = len(hla_evidence.get(key, []))
            if smorf_match or hla_match or coordinate_concordant:
                reasons = []
                if smorf_match:
                    reasons.append("normal_smorf_exact_peptide_match")
                    smorf_match_count += 1
                if hla_match:
                    reasons.append("normal_hla_ligand_exact_peptide_match")
                    hla_match_count += 1
                if smorf_match and hla_match:
                    both_match_count += 1
                exact_match = bool(smorf_match or hla_match)
                if exact_match:
                    exact_excluded_count += 1
                if coordinate_concordant:
                    reasons.append("normal_smorf_coordinate_frame_concordant")
                    coordinate_concordant_count += 1
                    if not exact_match:
                        coordinate_additional_excluded_count += 1
                if exact_match and coordinate_concordant:
                    row["external_normal_status"] = BOTH_EXCLUDED_STATUS
                elif coordinate_concordant:
                    row["external_normal_status"] = COORDINATE_EXCLUDED_STATUS
                else:
                    row["external_normal_status"] = EXACT_EXCLUDED_STATUS
                row["external_normal_qc_reasons"] = ";".join(reasons)
                excluded_rows.append(row)
            else:
                row["external_normal_status"] = RETAINED_STATUS
                row["external_normal_qc_reasons"] = ""
                hla_class = normalize_hla_class(row.get("mhc_class", ""))
                if retained_by_class[hla_class] < cap_by_class.get(hla_class, 10**18):
                    retained_rows.append(row)
                    retained_by_class[hla_class] += 1
                    if origin == "deferred_refill":
                        refill_selected_count += 1
                else:
                    row["external_normal_status"] = "not_selected_due_to_analysis_cap_after_external_normal_qc"
                    row["external_normal_qc_reasons"] = "not_selected_due_to_analysis_cap"
                    refill_deferred_rows.append(row)
        else:
            row["normal_translation_status"] = SMORF_NOT_EVALUATED
            row["normal_hla_presentation_status"] = HLA_NOT_EVALUATED
            row["normal_smorf_support_count"] = 0
            row["normal_hla_ligand_support_count"] = 0
            row["external_normal_status"] = NOT_EVALUATED_STATUS
            row["external_normal_qc_reasons"] = "external_normal_resources_not_evaluated"

        row["normal_smorf_iorf_ids"] = compact_values(smorf_evidence.get(key, []), "iorf_id") if smorf_match else ""
        row["normal_smorf_orf_ids"] = compact_values(smorf_evidence.get(key, []), "orf_id") if smorf_match else ""
        row["normal_hla_ligand_donors"] = compact_values(hla_evidence.get(key, []), "donor") if hla_match else ""
        row["normal_hla_ligand_tissues"] = compact_values(hla_evidence.get(key, []), "tissue") if hla_match else ""
        row["normal_hla_ligand_donor_hla_genotypes"] = (
            compact_values(hla_evidence.get(key, []), "donor_hla_genotype") if hla_match else ""
        )
        candidate_records.append(row)
        if smorf_match:
            for evidence in smorf_evidence.get(key, []):
                evidence_rows.append(make_smorf_evidence(row, evidence, freeze_version))
        if hla_match:
            for evidence in hla_evidence.get(key, []):
                evidence_rows.append(make_hla_evidence(row, evidence, freeze_version))

    for record in input_candidate_records:
        annotate_and_route_candidate(record)

    refill_parent_map_rows: list[dict[str, object]] = []
    refill_footprint_rows: list[dict[str, object]] = []
    refill_peptide_junction_rows: list[dict[str, object]] = []
    refill_human_reference_summaries: list[dict[str, object]] = []
    refill_exhausted_before_cap = {"HLA-I": False, "HLA-II": False}
    refill_materialized_batches = 0
    if formal_v11_ranked and resources_evaluated:
        ranked_parents = ranked_parents_for_completion or ranked_parent_stream(parent_core, parent_ranked)
        seen_refill_keys = {
            (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper())
            for row in candidate_records
        }
        refill_stream = iter_refill_key_stream(
            args.sample,
            ranked_parents,
            seen_refill_keys,
            refill_mhc_i_lengths,
            refill_mhc_ii_lengths,
        )
        refill_id_counter = 0
        parent_junction_records = parent_junction_summary.to_dict("records")
        while any(retained_by_class[klass] < cap_by_class.get(klass, 10**18) for klass in ["HLA-I", "HLA-II"]):
            missing = {
                klass: max(0, cap_by_class.get(klass, 10**18) - retained_by_class[klass])
                for klass in ["HLA-I", "HLA-II"]
            }
            pending: dict[tuple[str, str], dict[str, object]] = {}
            pending_by_class: defaultdict[str, int] = defaultdict(int)
            pending_source_by_key: dict[tuple[str, str], dict[str, object]] = {}
            for key, first_row in refill_stream:
                hla_class = key[0]
                if missing.get(hla_class, 0) <= 0:
                    continue
                pending[key] = first_row
                pending_source_by_key[key] = dict(first_row)
                pending_by_class[hla_class] += 1
                if all(
                    missing[klass] <= 0 or pending_by_class[klass] >= missing[klass]
                    for klass in ["HLA-I", "HLA-II"]
                ):
                    break
                if len(pending) >= 50000:
                    break
            if not pending:
                for klass in ["HLA-I", "HLA-II"]:
                    if retained_by_class[klass] < cap_by_class.get(klass, 10**18):
                        refill_exhausted_before_cap[klass] = True
                break

            refill_materialized_batches += 1
            human_matches, human_summary = load_human_reference_matches(
                human_proteome_fasta,
                {peptide for _klass, peptide in pending},
            )
            human_summary["refill_batch"] = refill_materialized_batches
            refill_human_reference_summaries.append(human_summary)

            peptide_id_by_key: dict[tuple[str, str], str] = {}
            unique_refill_rows: list[dict[str, object]] = []
            for key in sorted(
                pending,
                key=lambda item: (
                    parse_int(pending[item].get("_parent_rank", 0), "refill parent rank"),
                    item[0],
                    len(item[1]),
                    item[1],
                ),
            ):
                first = dict(pending[key])
                first.pop("_parent_rank", None)
                refill_id_counter += 1
                peptide_id = stable_refill_peptide_id(args.sample, key[0], key[1], refill_id_counter)
                if key[1] in human_matches:
                    row = base_output_row(first, freeze_version)
                    row["peptide_record_id"] = peptide_id
                    row["prebinding_selection_origin"] = "deferred_refill"
                    row["peptide_core_status"] = "excluded"
                    row["peptide_qc_reasons"] = "human_reference_peptide_match"
                    row["human_reference_peptide_status"] = HUMAN_REFERENCE_MATCH
                    row["normal_translation_status"] = SMORF_NOT_EVALUATED
                    row["normal_hla_presentation_status"] = HLA_NOT_EVALUATED
                    row["normal_smorf_support_count"] = 0
                    row["normal_hla_ligand_support_count"] = 0
                    row["supporting_window_count"] = ""
                    row["external_normal_status"] = "excluded_human_reference_match_before_external_normal_qc"
                    row["external_normal_qc_reasons"] = "human_reference_peptide_match"
                    candidate_records.append(row)
                    excluded_rows.append(row)
                    continue
                peptide_id_by_key[key] = peptide_id
                first["peptide_record_id"] = peptide_id
                first["prebinding_selection_origin"] = "deferred_refill"
                first["peptide_sequence_sha256"] = sha256_text(key[1])
                first["peptide_core_status"] = "core"
                first["peptide_qc_reasons"] = ""
                first["human_reference_peptide_status"] = HUMAN_REFERENCE_NOT_DETECTED
                first["normal_hla_presentation_status"] = "normal_hla_atlas_not_evaluated"
                first["normal_translation_status"] = "normal_translation_not_evaluable"
                first["external_normal_status"] = "external_normal_not_evaluated"
                unique_refill_rows.append(first)

            if not unique_refill_rows:
                continue

            batch_parent_map = build_parent_map_for_target_keys(
                args.sample,
                ranked_parents,
                peptide_id_by_key,
                refill_mhc_i_lengths,
                refill_mhc_ii_lengths,
                {
                    (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper()): row
                    for row in unique_refill_rows
                },
            )
            if not batch_parent_map:
                raise ValueError("refill selected candidate keys but generated no parent-map rows")
            support_count_by_key: defaultdict[tuple[str, str], int] = defaultdict(int)
            for row in batch_parent_map:
                support_count_by_key[
                    (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper())
                ] += 1
            for row in unique_refill_rows:
                row["supporting_window_count"] = support_count_by_key[
                    (normalize_hla_class(row.get("mhc_class", "")), str(row.get("peptide", "")).upper())
                ]
            batch_footprints = build_refill_peptide_footprints(
                args.sample,
                batch_parent_map,
                parent_coordinates,
                parent_orfcds,
            )
            batch_coordinate_summaries, batch_coordinate_evidence = evaluate_coordinate_evidence(
                pd.DataFrame(unique_refill_rows),
                pd.DataFrame(batch_footprints),
                normal_blocks,
                args.sample,
            )
            batch_junction_rows = annotate_peptide_junctions(
                sample=args.sample,
                peptide_parent_rows=batch_parent_map,
                peptide_footprint_rows=batch_footprints,
                parent_summary_rows=parent_junction_records,
            )
            coordinate_summaries.update(batch_coordinate_summaries)
            coordinate_evidence_rows.extend(batch_coordinate_evidence)
            refill_parent_map_rows.extend(batch_parent_map)
            refill_footprint_rows.extend(batch_footprints)
            refill_peptide_junction_rows.extend(batch_junction_rows)
            for row in unique_refill_rows:
                annotate_and_route_candidate(row)

        if refill_parent_map_rows:
            deferred_parent_map = pd.concat(
                [deferred_parent_map, pd.DataFrame(refill_parent_map_rows)],
                ignore_index=True,
            )
        if refill_peptide_junction_rows:
            peptide_junction_evidence = pd.concat(
                [peptide_junction_evidence, pd.DataFrame(refill_peptide_junction_rows)],
                ignore_index=True,
            )

    core_by_peptide_id = {str(record.get("peptide_record_id", "")): record for record in candidate_records}
    for row in core_parent_map.to_dict("records"):
        peptide_id = str(row.get("peptide_record_id", ""))
        enriched = core_by_peptide_id.get(peptide_id, {})
        source_row = dict(row)
        for key in [
            "normal_translation_status",
            "normal_hla_presentation_status",
            "normal_smorf_coordinate_status",
            "external_normal_status",
        ]:
            source_row[key] = enriched.get(key, "")
        evidence_rows.append(make_source_evidence(source_row, freeze_version))

    if not resources_evaluated:
        retained_rows = []
        refill_deferred_rows = []
        excluded_rows = list(candidate_records)

    hla_i_rows = [row for row in retained_rows if normalize_hla_class(row.get("mhc_class", "")) == "HLA-I"]
    hla_ii_rows = [row for row in retained_rows if normalize_hla_class(row.get("mhc_class", "")) == "HLA-II"]

    common_fields = list(core.columns)
    extra_fields = [
        "parent_record_id",
        "source_parent_id",
        "peptide_start",
        "peptide_sequence_sha256",
        "peptide_core_status",
        "peptide_qc_reasons",
        "human_reference_peptide_status",
        "supporting_window_count",
        "prebinding_selection_origin",
        "hla_class_normalized",
        "normal_translation_status",
        "normal_hla_presentation_status",
        "candidate_coordinate_status",
        "normal_smorf_coordinate_status",
        "coordinate_match_count",
        "matched_smorf_iorf_ids",
        "matched_smorf_orf_ids",
        "coordinate_matching_reasons",
        "normal_smorf_support_count",
        "normal_hla_ligand_support_count",
        "normal_smorf_iorf_ids",
        "normal_smorf_orf_ids",
        "normal_hla_ligand_donors",
        "normal_hla_ligand_tissues",
        "normal_hla_ligand_donor_hla_genotypes",
        "external_normal_status",
        "external_normal_qc_reasons",
        "resource_freeze_version",
    ]
    main_fields = common_fields + [field for field in extra_fields if field not in common_fields]
    evidence_fields = main_fields + [
        "evidence_type",
        "source_resource",
        "resource_record_id",
        "smorf_iorf_id",
        "smorf_orf_id",
        "smorf_peptide_start_1based",
        "hla_ligand_peptide_sequence_id",
        "hla_ligand_donor",
        "hla_ligand_tissue",
        "hla_ligand_donor_hla_genotype",
    ]

    if len(candidate_records) < len(input_candidate_records):
        raise AssertionError("annotated candidate row count must include all input candidate rows")
    if len(retained_rows) + len(excluded_rows) + len(refill_deferred_rows) != len(candidate_records):
        raise AssertionError("retained + excluded + deferred rows must equal annotated candidate rows")
    if is_v11(policy_version):
        human_reference_pre_external_excluded_count = sum(
            1
            for row in excluded_rows
            if str(row.get("external_normal_status", ""))
            == "excluded_human_reference_match_before_external_normal_qc"
        )
        if (
            exact_excluded_count
            + coordinate_additional_excluded_count
            + human_reference_pre_external_excluded_count
            + len(retained_rows)
            + len(refill_deferred_rows)
            != len(candidate_records)
        ):
            raise AssertionError(
                "annotated must equal exact_excluded + coordinate_additional_excluded + retained/deferred for v1.1"
            )

    write_tsv(candidate_records, outdir / "cryptic_external_normal_annotated_core.tsv", main_fields)
    write_tsv(evidence_rows, outdir / "cryptic_external_normal_evidence.tsv", evidence_fields)
    if is_v11(policy_version):
        coordinate_evidence_fields = [
            "sample", "peptide_record_id", "parent_record_id", "mhc_class", "peptide",
            "candidate_coordinate_status", "normal_smorf_coordinate_status",
            "coordinate_matching_reason", "smorf_iorf_id", "smorf_orf_id",
            "smorf_source_record_id", "candidate_codon_blocks", "normal_smorf_orfcds_blocks",
            "reference_build",
        ]
        write_tsv(
            coordinate_evidence_rows,
            outdir / "cryptic_smorf_coordinate_evidence.tsv",
            coordinate_evidence_fields,
        )
    write_tsv(excluded_rows, outdir / "cryptic_external_normal_excluded.tsv", main_fields)
    write_tsv(refill_deferred_rows, outdir / "cryptic_external_normal_refill_deferred.tsv", main_fields)
    refill_footprint_fields = [
        "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
        "peptide", "peptide_start", "peptide_length", "chromosome", "strand",
        "peptide_cds_start0", "peptide_cds_end0", "codon_blocks_transcript_order",
        "codon_blocks_text", "peptide_footprint_phase", "candidate_coordinate_status",
        "candidate_coordinate_reasons", "reference_build",
    ]
    refill_junction_fields = [
        "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
        "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
        "crossed_required_junction_ids", "parent_primary_core_status",
        "parent_junction_qc_status", "parent_required_junction_count",
    ]
    refill_parent_map_fields = list(parent_map.columns)
    if "prebinding_selection_origin" not in refill_parent_map_fields:
        refill_parent_map_fields.append("prebinding_selection_origin")
    write_tsv(
        refill_parent_map_rows,
        outdir / "cryptic_external_normal_refill_parent_map.tsv",
        refill_parent_map_fields,
    )
    write_tsv(
        refill_footprint_rows,
        outdir / "cryptic_external_normal_refill_genomic_footprint.tsv",
        refill_footprint_fields,
    )
    write_tsv(
        refill_peptide_junction_rows,
        outdir / "cryptic_external_normal_refill_junction_evidence.tsv",
        refill_junction_fields,
    )
    completed_parent_map_fields = list(parent_map.columns)
    if "prebinding_selection_origin" not in completed_parent_map_fields:
        completed_parent_map_fields.append("prebinding_selection_origin")
    write_tsv(
        completed_parent_map_rows,
        outdir / "cryptic_external_normal_completed_parent_map.tsv",
        completed_parent_map_fields,
    )
    write_tsv(
        completed_footprint_rows,
        outdir / "cryptic_external_normal_completed_genomic_footprint.tsv",
        refill_footprint_fields,
    )
    write_tsv(
        completed_junction_rows,
        outdir / "cryptic_external_normal_completed_junction_evidence.tsv",
        refill_junction_fields,
    )
    write_tsv(retained_rows, outdir / "cryptic_tumor_restricted_primary_core.tsv", main_fields)
    write_tsv(hla_i_rows, outdir / "cryptic_tumor_restricted_primary_core_hla_i.tsv", main_fields)
    write_tsv(hla_ii_rows, outdir / "cryptic_tumor_restricted_primary_core_hla_ii.tsv", main_fields)
    write_fasta(hla_i_rows, outdir / "cryptic_tumor_restricted_primary_core_hla_i.fasta")
    write_fasta(hla_ii_rows, outdir / "cryptic_tumor_restricted_primary_core_hla_ii.fasta")
    write_fasta(retained_rows, outdir / "cryptic_tumor_restricted_primary_core.fasta")
    final_sidecar_rows = build_final_parent_sidecar(
        args.sample,
        retained_rows,
        core_parent_map,
        deferred_parent_map,
        parent_core,
        parent_coordinates if is_v11(policy_version) else pd.DataFrame(),
        parent_ranked,
        peptide_junction_evidence,
        parent_junction_summary,
    )
    if retained_rows:
        retained_ids = {str(row.get("peptide_record_id", "")) for row in retained_rows}
        sidecar_ids = {str(row.get("peptide_record_id", "")) for row in final_sidecar_rows}
        missing_sidecar_ids = sorted(retained_ids - sidecar_ids)
        if missing_sidecar_ids:
            raise AssertionError(
                "final sidecar missing retained peptide parent mappings: "
                + ", ".join(missing_sidecar_ids[:5])
            )
        if formal_v11_ranked:
            required_sidecar_fields = [
                "parent_record_id",
                "parent_sequence",
                "coordinate_mapping_status",
                "reference_translation_status",
                "chromosome",
                "strand",
                "block_chain",
                "junction_qc_status",
                "required_junction_count",
                "peptide_crosses_junction",
                "peptide_start",
                "human_reference_peptide_status",
                "external_normal_status",
            ]
            missing_rows = [
                str(idx)
                for idx, row in enumerate(final_sidecar_rows, start=1)
                if any(str(row.get(field, "")).strip() == "" for field in required_sidecar_fields)
            ]
            if missing_rows:
                raise AssertionError(
                    "formal v1.1 ranked final sidecar has missing coordinate/junction/parent joins in rows: "
                    + ", ".join(missing_rows[:5])
                )
    validate_final_core_contract(
        retained_rows,
        final_sidecar_rows,
        completed_footprint_rows + refill_footprint_rows,
        completed_junction_rows + refill_peptide_junction_rows,
        refill_mhc_i_lengths if formal_v11_ranked else sorted(ALLOWED_LENGTHS["HLA-I"]),
        refill_mhc_ii_lengths if formal_v11_ranked else sorted(ALLOWED_LENGTHS["HLA-II"]),
        human_reference_evaluated=bool(human_reference_summary_from_08b.get("status") == "evaluated" or formal_v11_ranked),
        formal_v11_ranked=formal_v11_ranked,
    )
    write_tsv(
        final_sidecar_rows,
        outdir / "cryptic_final_peptide_parent_sidecar.tsv",
        FINAL_SIDECAR_FIELDS,
    )

    human_reference_refill_excluded_count = sum(
        1
        for row in excluded_rows
        if str(row.get("external_normal_status", "")) == "excluded_human_reference_match_before_external_normal_qc"
    )
    exact_only_excluded_count = sum(
        1 for row in excluded_rows if str(row.get("external_normal_status", "")) == EXACT_EXCLUDED_STATUS
    )
    coordinate_only_excluded_count = sum(
        1 for row in excluded_rows if str(row.get("external_normal_status", "")) == COORDINATE_EXCLUDED_STATUS
    )
    exact_and_coordinate_excluded_count = sum(
        1 for row in excluded_rows if str(row.get("external_normal_status", "")) == BOTH_EXCLUDED_STATUS
    )

    stage_counts = [
        {"stage": "source_core_unique_peptides", "count": len(source_core_records)},
        {"stage": "source_deferred_unique_peptides", "count": 0 if deferred.empty else len(deferred)},
        {"stage": "annotated_core_unique_peptides", "count": len(candidate_records)},
        {"stage": "external_normal_resources_evaluated", "count": int(resources_evaluated)},
        {"stage": "normal_smorf_exact_match_peptides", "count": smorf_match_count},
        {"stage": "normal_hla_ligand_exact_match_peptides", "count": hla_match_count},
        {"stage": "both_external_normal_evidence_peptides", "count": both_match_count},
        {"stage": "human_reference_refill_excluded_unique_peptides", "count": human_reference_refill_excluded_count},
        {"stage": "external_normal_exact_excluded_unique_peptides", "count": exact_only_excluded_count},
        {"stage": "external_normal_coordinate_frame_excluded_unique_peptides", "count": coordinate_only_excluded_count},
        {
            "stage": "external_normal_exact_and_coordinate_frame_excluded_unique_peptides",
            "count": exact_and_coordinate_excluded_count,
        },
        {"stage": "total_qc_excluded_unique_peptides", "count": len(excluded_rows)},
        {"stage": "external_normal_excluded_unique_peptides_legacy_combined", "count": len(excluded_rows)},
        {"stage": "external_normal_excluded_unique_peptides", "count": len(excluded_rows)},
        {
            "stage": "external_normal_not_evaluated_unique_peptides",
            "count": len(candidate_records) if not resources_evaluated else 0,
        },
        {"stage": "tumor_restricted_primary_core_unique_peptides", "count": len(retained_rows)},
        {"stage": "hla_i_tumor_restricted_primary_core_unique_peptides", "count": len(hla_i_rows)},
        {"stage": "hla_ii_tumor_restricted_primary_core_unique_peptides", "count": len(hla_ii_rows)},
        {"stage": "external_normal_refill_selected_unique_peptides", "count": refill_selected_count},
        {"stage": "external_normal_deferred_after_qc_unique_peptides", "count": len(refill_deferred_rows)},
        {"stage": "external_normal_refill_parent_map_rows", "count": len(refill_parent_map_rows)},
        {"stage": "external_normal_refill_genomic_footprint_rows", "count": len(refill_footprint_rows)},
        {"stage": "external_normal_refill_junction_evidence_rows", "count": len(refill_peptide_junction_rows)},
        {"stage": "external_normal_evidence_rows", "count": len(evidence_rows)},
        {"stage": "final_peptide_parent_sidecar_rows", "count": len(final_sidecar_rows)},
    ]
    if is_v11(policy_version):
        coordinate_status_by_peptide = {
            peptide_id: summary.get("normal_smorf_coordinate_status", "")
            for peptide_id, summary in coordinate_summaries.items()
        }
        coordinate_evaluable_peptides = sum(
            1
            for summary in coordinate_summaries.values()
            if summary.get("candidate_coordinate_status") == "coordinate_evaluable"
        )
        coordinate_overlap_peptides = sum(
            1
            for status in coordinate_status_by_peptide.values()
            if status
            and status
            not in {
                "no_coordinate_overlap",
                "not_evaluable_candidate_coordinate_qc",
                "ambiguous_candidate_mapping",
                "not_evaluable_reference_or_coordinate_mismatch",
            }
        )
        stage_counts.extend([
            {"stage": "candidate_parents_coordinate_evaluable", "count": parent_coordinate_evaluable_count},
            {"stage": "candidate_parents_coordinate_not_evaluable", "count": parent_coordinate_not_evaluable_count},
            {"stage": "peptides_coordinate_evaluable", "count": coordinate_evaluable_peptides},
            {"stage": "normal_smorf_coordinate_overlap_peptides", "count": coordinate_overlap_peptides},
            {"stage": "coordinate_frame_concordant_peptides", "count": coordinate_concordant_count},
            {
                "stage": "coordinate_frame_discordant_peptides",
                "count": sum(
                    1
                    for status in coordinate_status_by_peptide.values()
                    if status == "coordinate_overlap_frame_discordant"
                ),
            },
            {
                "stage": "coordinate_not_evaluable_peptides",
                "count": sum(
                    1
                    for status in coordinate_status_by_peptide.values()
                    if status
                    in {
                        "not_evaluable_candidate_coordinate_qc",
                        "ambiguous_candidate_mapping",
                        "not_evaluable_reference_or_coordinate_mismatch",
                    }
                ),
            },
            {"stage": "v1.0_exact_excluded_peptides", "count": exact_excluded_count},
            {"stage": "v1.1_coordinate_additional_excluded_peptides", "count": coordinate_additional_excluded_count},
            {"stage": "final_primary_core_unique_peptides", "count": len(retained_rows)},
            {"stage": "smorf_coordinate_evidence_rows", "count": len(coordinate_evidence_rows)},
        ])
    write_tsv(stage_counts, outdir / "stagewise_qc.tsv", ["stage", "count"])

    outputs_hash = output_signature(output_paths)
    final_binding_fasta = outdir / "cryptic_tumor_restricted_primary_core.fasta"
    final_binding_fasta_identity = file_identity(final_binding_fasta)
    binding_eligible = bool(resources_evaluated and upstream_manifest.get("binding_eligible") is True)
    if upstream_selection_mode == "ranked_cap":
        final_cap_reached_hla_i = len(hla_i_rows) >= cap_by_class.get("HLA-I", 10**18)
        final_cap_reached_hla_ii = len(hla_ii_rows) >= cap_by_class.get("HLA-II", 10**18)
        final_exhausted_before_cap_hla_i = (
            refill_exhausted_before_cap["HLA-I"] or not final_cap_reached_hla_i
        )
        final_exhausted_before_cap_hla_ii = (
            refill_exhausted_before_cap["HLA-II"] or not final_cap_reached_hla_ii
        )
        cap_refill_summary = {
            "source_core_unique_peptides": len(source_core_records),
            "source_deferred_unique_peptides": 0 if deferred.empty else len(deferred),
            "refill_selected_unique_peptides": refill_selected_count,
            "deferred_after_external_normal_qc_unique_peptides": len(refill_deferred_rows),
            "refill_materialized_batches": refill_materialized_batches,
            "refill_human_reference_summaries": refill_human_reference_summaries,
            "cap_hla_i": cap_by_class.get("HLA-I", 0),
            "cap_hla_ii": cap_by_class.get("HLA-II", 0),
            "final_cap_reached_hla_i": final_cap_reached_hla_i,
            "final_cap_reached_hla_ii": final_cap_reached_hla_ii,
            "final_exhausted_before_cap_hla_i": final_exhausted_before_cap_hla_i,
            "final_exhausted_before_cap_hla_ii": final_exhausted_before_cap_hla_ii,
            "exhausted_before_cap_hla_i": final_exhausted_before_cap_hla_i,
            "exhausted_before_cap_hla_ii": final_exhausted_before_cap_hla_ii,
        }
    else:
        final_cap_reached_hla_i = "not_applicable"
        final_cap_reached_hla_ii = "not_applicable"
        final_exhausted_before_cap_hla_i = "not_applicable"
        final_exhausted_before_cap_hla_ii = "not_applicable"
        cap_refill_summary = {
            "source_core_unique_peptides": len(source_core_records),
            "source_deferred_unique_peptides": 0,
            "refill_selected_unique_peptides": 0,
            "deferred_after_external_normal_qc_unique_peptides": 0,
            "cap_hla_i": "not_applicable",
            "cap_hla_ii": "not_applicable",
            "final_cap_reached_hla_i": "not_applicable",
            "final_cap_reached_hla_ii": "not_applicable",
            "final_exhausted_before_cap_hla_i": "not_applicable",
            "final_exhausted_before_cap_hla_ii": "not_applicable",
            "exhausted_before_cap_hla_i": "not_applicable",
            "exhausted_before_cap_hla_ii": "not_applicable",
        }

    manifest = {
        "sample": args.sample,
        "policy_version": policy_version,
        "run_status": "complete",
        "binding_eligible": binding_eligible,
        "upstream_binding_eligible": upstream_manifest.get("binding_eligible") is True,
        "started_at": started_at,
        "finished_at": ts(),
        "code_commit": git_commit(),
        "script_sha256": script_sha256,
        "input_signature": input_signature,
        "upstream_08b_policy_version": upstream_manifest.get("policy_version", ""),
        "external_normal_evaluation": "evaluated" if resources_evaluated else "not_evaluated",
        "resource_freeze_version": freeze_version,
        "coordinate_resource_freeze_version": coordinate_freeze_version,
        "coordinate_matching_status": "evaluated" if is_v11(policy_version) else COORDINATE_STATUS,
        "selection_policy_version": upstream_manifest.get("candidate_selection_policy_version", ""),
        "candidate_selection": upstream_manifest.get("candidate_selection", {}),
        "final_unique_peptide_counts": {
            "HLA-I": len(hla_i_rows),
            "HLA-II": len(hla_ii_rows),
            "total": len(retained_rows),
        },
        "final_cap_reached_hla_i": final_cap_reached_hla_i,
        "final_cap_reached_hla_ii": final_cap_reached_hla_ii,
        "final_exhausted_before_cap_hla_i": final_exhausted_before_cap_hla_i,
        "final_exhausted_before_cap_hla_ii": final_exhausted_before_cap_hla_ii,
        "cap_refill_summary": cap_refill_summary,
        "stage_count_notes": {
            "external_normal_excluded_unique_peptides": (
                "legacy combined count; includes human-reference refill exclusions, external-normal exact "
                "exclusions, coordinate/frame exclusions, and exact+coordinate exclusions"
            )
        },
        "final_binding_fasta_identity": final_binding_fasta_identity,
        "external_normal_resource_identities": resource_identity,
        "coordinate_resource_identities": coordinate_resource_identity,
        "outputs": {path.name: str(path) for path in output_paths},
        "output_signature": outputs_hash,
        "stage_counts": {row["stage"]: row["count"] for row in stage_counts},
        "binding_input_fasta": str(final_binding_fasta) if binding_eligible else "",
        "binding_input_mode": "peptide-core" if binding_eligible else "not_available",
        "reused": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument(
        "--policy-version",
        default=POLICY_VERSION_V1,
        choices=[POLICY_VERSION_V1, POLICY_VERSION_V11],
    )
    parser.add_argument("--cryptic-peptide-core", required=True)
    parser.add_argument("--cryptic-peptide-parent-map", required=True)
    parser.add_argument("--cryptic-peptide-deferred", default="")
    parser.add_argument("--cryptic-peptide-deferred-parent-map", default="")
    parser.add_argument("--cryptic-parent-core", default="")
    parser.add_argument("--cryptic-parent-ranked", default="")
    parser.add_argument("--cryptic-parent-junction-summary", default="")
    parser.add_argument("--cryptic-peptide-junction-evidence", default="")
    parser.add_argument("--upstream-manifest", required=True)
    parser.add_argument("--human-proteome-fasta", default="")
    parser.add_argument("--cryptic-parent-coordinates", default="")
    parser.add_argument("--cryptic-parent-orfcds", default="")
    parser.add_argument("--cryptic-peptide-genomic-footprint", default="")
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--resource-manifest", default="")
    parser.add_argument("--smorf-match-index", default="")
    parser.add_argument("--smorf-parent-map", default="")
    parser.add_argument("--hla-ligand-match-index", default="")
    parser.add_argument("--hla-ligand-evidence", default="")
    parser.add_argument("--coordinate-resource-manifest", default="")
    parser.add_argument("--normal-smorf-coordinates", default="")
    parser.add_argument("--normal-smorf-orfcds", default="")
    parser.add_argument("--allow-missing-external-normal-resources", action="store_true")
    parser.add_argument("--coordinate-matching-enabled", action="store_true")
    parser.add_argument("--max-hla-i-peptides", type=int, default=0)
    parser.add_argument("--max-hla-ii-peptides", type=int, default=0)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    manifest = build_external_normal_qc(build_parser().parse_args(argv))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
