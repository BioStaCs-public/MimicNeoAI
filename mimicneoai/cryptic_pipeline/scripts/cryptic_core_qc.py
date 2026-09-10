#!/usr/bin/env python3
"""Materialize a strict, pre-binding Cryptic Core.

This stage consumes aeSEP parent ORFs after ORF-genome filtering and writes
pre-tiled HLA-I/HLA-II peptide Core FASTA files. In v1.1 it can apply
coordinate/translation QC, cryptic_rna_variant_editing_qc_v1.0, and
junction_qc_v1.0 before peptide generation when those resources are configured.
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
        extract_transcript_cds_from_reference,
        genomic_order,
        load_bam_alignment_stats,
        map_transcript_interval_to_genome,
        open_reference_fasta,
        parse_bed12,
        reference_contig_lookup,
        read_fasta_records,
        transcript_order,
        translate_cds,
    )
    from mimicneoai.cryptic_pipeline.scripts.cryptic_junction_qc import (
        POLICY_VERSION as JUNCTION_POLICY_VERSION,
        PRIMARY_PASS as JUNCTION_PRIMARY_PASS,
        annotate_peptide_junctions,
        evaluate_parent_junctions,
        validate_junction_policy,
        validate_star_pair_contract,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from cryptic_coordinate_utils import (  # type: ignore
        GenomicBlock,
        blocks_to_json,
        blocks_to_text,
        canonical_chromosome,
        extract_transcript_cds_from_reference,
        genomic_order,
        load_bam_alignment_stats,
        map_transcript_interval_to_genome,
        open_reference_fasta,
        parse_bed12,
        reference_contig_lookup,
        read_fasta_records,
        transcript_order,
        translate_cds,
    )
    from cryptic_junction_qc import (  # type: ignore
        POLICY_VERSION as JUNCTION_POLICY_VERSION,
        PRIMARY_PASS as JUNCTION_PRIMARY_PASS,
        annotate_peptide_junctions,
        evaluate_parent_junctions,
        validate_junction_policy,
        validate_star_pair_contract,
    )

POLICY_VERSION_V1 = "cryptic_core_qc_v1.0"
POLICY_VERSION_V11 = "cryptic_core_qc_v1.1"
POLICY_VERSION = POLICY_VERSION_V1
COORDINATE_UTILS_PATH = Path(__file__).with_name("cryptic_coordinate_utils.py")
JUNCTION_QC_PATH = Path(__file__).with_name("cryptic_junction_qc.py")
RNA_VARIANT_QC_PATH = Path(__file__).with_name("cryptic_rna_variant_editing_qc.py")
RNA_VARIANT_POLICY_VERSION = "cryptic_rna_variant_editing_qc_v1.0"
RNA_VARIANT_MIN_MAPPING_QUALITY = 20.0
RNA_VARIANT_MIN_TOTAL_DEPTH = 10
MIN_VARIANT_ALLELE_FRACTION = 0.05
MIN_VARIANT_QUAL = 30.0
RNA_VARIANT_PRIMARY_MIN_ALT_READS = 3
RNA_VARIANT_SENSITIVITY_ALT_READS = (2, 3, 5)
RNA_VARIANT_EVENT_FIELDS: list[str] = []
RNA_VARIANT_PARENT_SUMMARY_FIELDS: list[str] = []
RNA_VARIANT_PEPTIDE_EVIDENCE_FIELDS: list[str] = []
_RNA_VARIANT_QC_LOADED = False
CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
ACCEPTED_SOURCE_CLASSES = {"novel", "noncoding"}
CANDIDATE_SELECTION_POLICY_VERSION = "ranked_prebinding_selection_v1.0"
CRYPTIC_RANKING_POLICY = "cryptic_parent_expression_v1.0"
ALL_MODE = "all"
RANKED_CAP_MODE = "ranked_cap"


def _missing_rna_variant_qc(*_args, **_kwargs):
    raise RuntimeError(
        "cryptic_rna_variant_editing_qc is required only when rna_variant_editing_qc is enabled"
    )


annotate_peptide_rna_variant_evidence = _missing_rna_variant_qc
evaluate_parent_rna_variants = _missing_rna_variant_qc
load_rediportal_events = _missing_rna_variant_qc
load_vcf_events = _missing_rna_variant_qc
validate_rna_variant_calling_manifest = _missing_rna_variant_qc
validate_rna_variant_policy = _missing_rna_variant_qc


def ensure_rna_variant_qc_loaded() -> None:
    global _RNA_VARIANT_QC_LOADED
    global RNA_VARIANT_EVENT_FIELDS
    global RNA_VARIANT_PARENT_SUMMARY_FIELDS
    global RNA_VARIANT_PEPTIDE_EVIDENCE_FIELDS
    global RNA_VARIANT_POLICY_VERSION
    global RNA_VARIANT_MIN_MAPPING_QUALITY
    global RNA_VARIANT_MIN_TOTAL_DEPTH
    global MIN_VARIANT_ALLELE_FRACTION
    global MIN_VARIANT_QUAL
    global RNA_VARIANT_PRIMARY_MIN_ALT_READS
    global RNA_VARIANT_SENSITIVITY_ALT_READS
    global annotate_peptide_rna_variant_evidence
    global evaluate_parent_rna_variants
    global load_rediportal_events
    global load_vcf_events
    global validate_rna_variant_calling_manifest
    global validate_rna_variant_policy

    if _RNA_VARIANT_QC_LOADED:
        return
    try:
        from mimicneoai.cryptic_pipeline.scripts import cryptic_rna_variant_editing_qc as module
    except ImportError:  # pragma: no cover - direct script execution fallback
        try:
            import cryptic_rna_variant_editing_qc as module  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "cryptic_rna_variant_editing_qc is not available; disable rna_variant_editing_qc "
                "or include the helper module in this code version"
            ) from exc

    RNA_VARIANT_EVENT_FIELDS = list(module.EVENT_FIELDS)
    RNA_VARIANT_PARENT_SUMMARY_FIELDS = list(module.PARENT_SUMMARY_FIELDS)
    RNA_VARIANT_PEPTIDE_EVIDENCE_FIELDS = list(module.PEPTIDE_EVIDENCE_FIELDS)
    RNA_VARIANT_POLICY_VERSION = str(module.POLICY_VERSION)
    RNA_VARIANT_MIN_MAPPING_QUALITY = float(module.MIN_MAPPING_QUALITY)
    RNA_VARIANT_MIN_TOTAL_DEPTH = int(module.MIN_TOTAL_DEPTH)
    MIN_VARIANT_ALLELE_FRACTION = float(module.MIN_VARIANT_ALLELE_FRACTION)
    MIN_VARIANT_QUAL = float(module.MIN_VARIANT_QUAL)
    RNA_VARIANT_PRIMARY_MIN_ALT_READS = int(module.PRIMARY_MIN_ALT_READS)
    RNA_VARIANT_SENSITIVITY_ALT_READS = tuple(module.SENSITIVITY_ALT_READS)
    annotate_peptide_rna_variant_evidence = module.annotate_peptide_rna_variant_evidence
    evaluate_parent_rna_variants = module.evaluate_parent_rna_variants
    load_rediportal_events = module.load_rediportal_events
    load_vcf_events = module.load_vcf_events
    validate_rna_variant_calling_manifest = module.validate_rna_variant_calling_manifest
    validate_rna_variant_policy = module.validate_policy
    _RNA_VARIANT_QC_LOADED = True


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_progress(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for token in str(value).replace(",", " ").split():
        token = token.strip()
        if token:
            out.append(int(token))
    if not out:
        raise ValueError("At least one peptide length is required")
    return sorted(set(out))


def validate_candidate_selection(
    mode: str,
    max_hla_i_peptides: Optional[int],
    max_hla_ii_peptides: Optional[int],
) -> tuple[str, Optional[int], Optional[int]]:
    mode = str(mode or ALL_MODE).strip().lower()
    if mode not in {ALL_MODE, RANKED_CAP_MODE}:
        raise ValueError(f"candidate_selection.mode must be '{ALL_MODE}' or '{RANKED_CAP_MODE}', got {mode!r}")
    if mode == ALL_MODE:
        return mode, None, None
    if max_hla_i_peptides is None or int(max_hla_i_peptides) <= 0:
        raise ValueError("ranked_cap requires positive max_hla_i_peptides")
    if max_hla_ii_peptides is None or int(max_hla_ii_peptides) <= 0:
        raise ValueError("ranked_cap requires positive max_hla_ii_peptides")
    return mode, int(max_hla_i_peptides), int(max_hla_ii_peptides)


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


def read_fasta(path: Path) -> Iterable[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    header: Optional[str] = None
    chunks: list[str] = []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def fasta_record_id(header: str) -> str:
    fields = str(header).strip().split()
    if not fields:
        raise ValueError("FASTA header does not contain a record id")
    return fields[0]


def source_parent_id(record_id: str) -> str:
    return record_id.rsplit(".p", 1)[0] if ".p" in record_id else record_id


def normalize_parent_sequence(sequence: str) -> tuple[str, list[str]]:
    seq = str(sequence).replace(" ", "").replace("\n", "").strip().upper()
    reasons: list[str] = []
    if not seq:
        return "", ["empty_sequence"]
    if seq.endswith("*"):
        seq = seq[:-1]
    if "*" in seq:
        reasons.append("internal_stop")
    if "-" in seq:
        reasons.append("gap_character")
    bad = sorted(set(seq).difference(CANONICAL_AA))
    if bad:
        reasons.append("noncanonical_amino_acid:" + ",".join(bad))
    if not seq:
        reasons.append("empty_after_terminal_stop_trim")
    return seq, reasons


def iter_windows(sequence: str, lengths: Iterable[int]) -> Iterable[tuple[int, int, str]]:
    for length in lengths:
        if len(sequence) < length:
            continue
        for start in range(0, len(sequence) - length + 1):
            yield start + 1, length, sequence[start : start + length]


def iter_canonical_segments(sequence: str) -> Iterable[str]:
    start: Optional[int] = None
    for index, aa in enumerate(sequence):
        if aa in CANONICAL_AA:
            if start is None:
                start = index
        elif start is not None:
            yield sequence[start:index]
            start = None
    if start is not None:
        yield sequence[start:]


def normalize_human_reference_sequence(sequence: str) -> str:
    seq = str(sequence).replace(" ", "").replace("\n", "").strip().upper()
    if seq.endswith("*"):
        seq = seq[:-1]
    return seq


def load_human_reference_matches(
    human_proteome_fasta: Optional[Path],
    candidate_peptides: set[str],
) -> tuple[set[str], dict[str, object]]:
    if not human_proteome_fasta:
        return set(), {"status": "not_evaluated", "path": ""}
    if not human_proteome_fasta.exists():
        raise FileNotFoundError(f"human reference proteome FASTA not found: {human_proteome_fasta}")
    matches: set[str] = set()
    lengths = sorted({len(peptide) for peptide in candidate_peptides})
    records_encountered = 0
    records_with_noncanonical_residues = 0
    standard_windows_evaluated = 0
    for _header, raw_seq in read_fasta(human_proteome_fasta):
        records_encountered += 1
        seq = normalize_human_reference_sequence(raw_seq)
        if not seq:
            continue
        if set(seq).difference(CANONICAL_AA):
            records_with_noncanonical_residues += 1
        for segment in iter_canonical_segments(seq):
            for length in lengths:
                if len(segment) < length:
                    continue
                standard_windows_evaluated += len(segment) - length + 1
                for start in range(0, len(segment) - length + 1):
                    pep = segment[start : start + length]
                    if pep in candidate_peptides:
                        matches.add(pep)
    return matches, {
        "status": "evaluated",
        "path": str(human_proteome_fasta),
        "sha256": sha256_file(human_proteome_fasta),
        "records_scanned": records_encountered,
        "records_encountered": records_encountered,
        "records_with_noncanonical_residues": records_with_noncanonical_residues,
        "standard_windows_evaluated": standard_windows_evaluated,
        "candidate_peptides": len(candidate_peptides),
        "matched_peptides": len(matches),
    }


def write_fasta(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(f">{row['parent_record_id']}\n{row['peptide']}\n")


def write_peptide_core_fasta(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(f">{row['peptide_record_id']}\n{row['peptide']}\n")


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


def read_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


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


def parse_bool(value: object) -> Optional[bool]:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def parse_float(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_expression(
    row: pd.Series,
    min_tpm_tumor: float,
    max_tpm_ctrl: float,
    min_log2fc: float,
) -> tuple[bool, list[str], dict[str, object]]:
    tumor = parse_float(row.get("TPM_tumor", ""))
    ctrl = parse_float(row.get("TPM_ctrl", ""))
    log2fc = parse_float(row.get("log2FC", ""))
    if tumor is None or ctrl is None or log2fc is None:
        return (
            False,
            ["expression_value_missing_or_invalid"],
            {
                "TPM_tumor": "" if tumor is None else tumor,
                "TPM_ctrl": "" if ctrl is None else ctrl,
                "log2FC": "" if log2fc is None else log2fc,
                "expression_qc_status": "fail",
                "expression_qc_reasons": "expression_value_missing_or_invalid",
                "upstream_is_aberrant_consistent": "",
            },
        )
    reasons: list[str] = []
    if tumor < min_tpm_tumor:
        reasons.append("tumor_tpm_below_threshold")
    if ctrl > max_tpm_ctrl:
        reasons.append("control_tpm_above_threshold")
    if log2fc < min_log2fc:
        reasons.append("log2fc_below_threshold")
    passed = not reasons
    upstream = parse_bool(row.get("is_aberrant", ""))
    upstream_consistent = "" if upstream is None else bool(upstream == passed)
    if upstream is not None and upstream != passed:
        reasons.append("upstream_is_aberrant_mismatch")
    return (
        passed,
        [reason for reason in reasons if reason != "upstream_is_aberrant_mismatch"],
        {
            "TPM_tumor": tumor,
            "TPM_ctrl": ctrl,
            "log2FC": log2fc,
            "expression_qc_status": "pass" if passed else "fail",
            "expression_qc_reasons": ";".join(
                reason for reason in reasons if reason != "upstream_is_aberrant_mismatch"
            ),
            "upstream_is_aberrant": "" if upstream is None else upstream,
            "upstream_is_aberrant_consistent": upstream_consistent,
        },
    )


def stable_peptide_id(sample: str, mhc_class: str, peptide: str, index: int) -> str:
    digest = sha256_text(f"{sample}|{mhc_class}|{peptide}")[:16]
    class_tag = "MHC-I" if mhc_class == "MHC-I" else "MHC-II"
    return f"cryptic_core_{class_tag}_{index:07d}_{digest}"


def stable_selection_digest(sample: str, parent_id: str, mhc_class: str = "", peptide: str = "") -> str:
    return sha256_text(f"{sample}|cryptic|{parent_id}|{mhc_class}|{len(peptide)}|{peptide}")


def sort_parent_core_for_ranked(rows: list[dict[str, object]], sample: str) -> list[dict[str, object]]:
    def key(row: dict[str, object]) -> tuple:
        tumor_tpm = parse_float(row.get("TPM_tumor", "")) or 0.0
        log2fc = parse_float(row.get("log2FC", "")) or 0.0
        ctrl_tpm = parse_float(row.get("TPM_ctrl", "")) or 0.0
        nc_class = str(row.get("nc_class", ""))
        nc_rank = 0 if nc_class == "novel" else 1 if nc_class == "noncoding" else 2
        sequence = str(row.get("parent_sequence", ""))
        parent_id = str(row.get("parent_record_id", ""))
        digest = stable_selection_digest(sample, parent_id)
        return (-tumor_tpm, -log2fc, ctrl_tpm, nc_rank, -len(sequence), parent_id, digest)

    return sorted(rows, key=key)


def make_candidate_window_row(
    sample: str,
    parent_row: dict[str, object],
    mhc_class: str,
    start: int,
    length: int,
    peptide: str,
) -> dict[str, object]:
    return {
        "sample": sample,
        "parent_record_id": parent_row["parent_record_id"],
        "source_parent_id": parent_row["source_parent_id"],
        "mhc_class": mhc_class,
        "peptide_start": start,
        "peptide_length": length,
        "peptide": peptide,
        "peptide_sequence_sha256": sha256_text(peptide),
        "peptide_core_status": "core",
        "peptide_qc_reasons": "",
        "human_reference_peptide_status": "not_evaluated",
        "normal_hla_presentation_status": "normal_hla_atlas_not_evaluated",
        "normal_translation_status": "normal_translation_not_evaluable",
        "external_normal_status": "external_normal_not_evaluated",
    }


def iter_parent_candidate_rows(
    sample: str,
    parent_row: dict[str, object],
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
) -> Iterable[dict[str, object]]:
    sequence = str(parent_row["parent_sequence"])
    for mhc_class, lengths in [("MHC-I", mhc_i_lengths), ("MHC-II", mhc_ii_lengths)]:
        for start, length, peptide in iter_windows(sequence, lengths):
            yield make_candidate_window_row(sample, parent_row, mhc_class, start, length, peptide)


def rebuild_selected_parent_windows(
    sample: str,
    parent_core_rows: list[dict[str, object]],
    selected_keys: set[tuple[str, str]],
    peptide_id_by_key: dict[tuple[str, str], str],
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
    metadata_by_key: Optional[dict[tuple[str, str], dict[str, object]]] = None,
) -> list[dict[str, object]]:
    if not selected_keys:
        return []
    metadata_by_key = metadata_by_key or {}
    rows: list[dict[str, object]] = []
    for parent_row in parent_core_rows:
        for row in iter_parent_candidate_rows(sample, parent_row, mhc_i_lengths, mhc_ii_lengths):
            key = (str(row["mhc_class"]), str(row["peptide"]))
            if key in selected_keys:
                metadata = metadata_by_key.get(key, {})
                for field in [
                    "peptide_record_id",
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
                row["peptide_record_id"] = peptide_id_by_key.get(key, row.get("peptide_record_id", ""))
                rows.append(row)
    return rows


def is_v11(policy_version: str) -> bool:
    return policy_version == POLICY_VERSION_V11


def require_coordinate_inputs(args: argparse.Namespace) -> None:
    required = [
        ("--orf-bed12", args.orf_bed12),
        ("--orf-bam", args.orf_bam),
        ("--orf-cds-fasta", args.orf_cds_fasta),
        ("--reference-genome-fasta", args.reference_genome_fasta),
    ]
    missing = [label for label, value in required if not str(value or "").strip()]
    if missing:
        raise ValueError(
            "cryptic_core_qc_v1.1 requires coordinate inputs: " + ", ".join(missing)
        )
    for label, value in required:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")


def summarize_alignment_loci(alignments: list[dict[str, object]]) -> str:
    values = []
    for aln in alignments:
        chrom = str(aln.get("chromosome", ""))
        strand = str(aln.get("strand", ""))
        start = str(aln.get("start0", ""))
        end = str(aln.get("end0", ""))
        mapq = str(aln.get("mapq", ""))
        cigar = str(aln.get("cigar", ""))
        values.append(f"{chrom}:{start}-{end}:{strand}:MAPQ{mapq}:{cigar}")
    return ";".join(values)


def record_rna_variant_evaluation(
    context: dict[str, object],
    parent_id: str,
    evaluation: dict[str, object],
) -> None:
    evaluated = context.setdefault("evaluated_parent_ids", set())
    if not isinstance(evaluated, set):
        raise TypeError("rna_variant_context evaluated_parent_ids must be a set")
    if parent_id in evaluated:
        return
    evaluated.add(parent_id)
    parent_summary = evaluation["parent_summary"]  # type: ignore[index]
    event_rows = evaluation["event_rows"]  # type: ignore[index]
    variant_positions = evaluation["variant_aa_positions"]  # type: ignore[index]
    threshold_parent_pass = evaluation["threshold_parent_pass"]  # type: ignore[index]
    parent_summary_rows = context.setdefault("parent_summary_rows", [])
    all_event_rows = context.setdefault("event_rows", [])
    variant_by_parent = context.setdefault("variant_aa_positions_by_parent", {})
    threshold_pass_counts = context.setdefault("threshold_parent_pass_counts", defaultdict(int))
    if not isinstance(parent_summary_rows, list) or not isinstance(all_event_rows, list):
        raise TypeError("rna_variant_context rows must be lists")
    if not isinstance(variant_by_parent, dict):
        raise TypeError("rna_variant_context variant_aa_positions_by_parent must be a dict")
    if not isinstance(threshold_pass_counts, defaultdict):
        raise TypeError("rna_variant_context threshold_parent_pass_counts must be a defaultdict")
    parent_summary_rows.append(parent_summary)
    all_event_rows.extend(event_rows)
    variant_by_parent[parent_id] = variant_positions
    for threshold, passed in dict(threshold_parent_pass).items():
        if passed:
            threshold_pass_counts[int(threshold)] += 1


def build_parent_status_from_coordinate_sidecars(
    parent_coordinates: list[dict[str, object]],
    parent_orfcds: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    blocks_by_parent: defaultdict[str, list[GenomicBlock]] = defaultdict(list)
    for row in sorted(
        parent_orfcds,
        key=lambda item: (
            str(item.get("parent_record_id", "")),
            int(item.get("transcript_block_order", 0) or 0),
        ),
    ):
        parent_id = str(row.get("parent_record_id", ""))
        if not parent_id:
            continue
        try:
            blocks_by_parent[parent_id].append(
                GenomicBlock(
                    str(row.get("chromosome", "")),
                    str(row.get("strand", "")),
                    int(row.get("start0", 0)),
                    int(row.get("end0", 0)),
                )
            )
        except (TypeError, ValueError):
            continue

    parent_status: dict[str, dict[str, object]] = {}
    for row in parent_coordinates:
        parent_id = str(row.get("parent_record_id", ""))
        if not parent_id:
            continue
        parent_status[parent_id] = {
            **row,
            "transcript_blocks": blocks_by_parent.get(parent_id, []),
        }
    return parent_status


def build_peptide_footprints_from_parent_status(
    sample: str,
    peptide_parent_rows: list[dict[str, object]],
    parent_status: dict[str, dict[str, object]],
    reference_build: str,
) -> list[dict[str, object]]:
    peptide_footprints: list[dict[str, object]] = []
    for row in peptide_parent_rows:
        parent_id = str(row.get("parent_record_id", ""))
        parent_info = parent_status.get(parent_id, {})
        status = str(parent_info.get("coordinate_mapping_status", "not_evaluable_candidate_coordinate_qc"))
        reasons = str(parent_info.get("coordinate_mapping_reasons", ""))
        tx_blocks = parent_info.get("transcript_blocks", [])
        peptide_start = int(row.get("peptide_start", 0))
        peptide_length = int(row.get("peptide_length", 0))
        cds_start0 = (peptide_start - 1) * 3
        cds_end0 = cds_start0 + peptide_length * 3
        footprint_blocks: list[GenomicBlock] = []
        if status == "coordinate_evaluable":
            try:
                footprint_blocks = [
                    block
                    for _segment_start, _segment_end, block in map_transcript_interval_to_genome(
                        tx_blocks, str(parent_info.get("strand", "")), cds_start0, cds_end0
                    )
                ]
            except Exception as exc:
                status = "not_evaluable_candidate_coordinate_qc"
                reasons = ";".join(filter(None, [reasons, f"peptide_footprint_mapping_failed:{exc}"]))
        peptide_footprints.append({
            "sample": sample,
            "peptide_record_id": row.get("peptide_record_id", ""),
            "parent_record_id": parent_id,
            "source_parent_id": row.get("source_parent_id", ""),
            "mhc_class": row.get("mhc_class", ""),
            "peptide": row.get("peptide", ""),
            "peptide_start": peptide_start,
            "peptide_length": peptide_length,
            "chromosome": parent_info.get("chromosome", ""),
            "strand": parent_info.get("strand", ""),
            "peptide_cds_start0": cds_start0,
            "peptide_cds_end0": cds_end0,
            "codon_blocks_transcript_order": blocks_to_json(footprint_blocks),
            "codon_blocks_text": blocks_to_text(footprint_blocks),
            "peptide_footprint_phase": cds_start0 % 3,
            "candidate_coordinate_status": status,
            "candidate_coordinate_reasons": reasons,
            "reference_build": reference_build,
        })
    return peptide_footprints


def build_coordinate_sidecars(
    args: argparse.Namespace,
    parent_core_rows: list[dict[str, object]],
    peptide_parent_rows: list[dict[str, object]],
    policy_version: str,
    rna_variant_context: Optional[dict[str, object]] = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not is_v11(policy_version):
        return [], [], [], {"status": "not_evaluated_policy_v1.0"}

    require_coordinate_inputs(args)
    bed12_entries = parse_bed12(Path(args.orf_bed12))
    bam_stats = load_bam_alignment_stats(Path(args.orf_bam))
    cds_records = read_fasta_records(Path(args.orf_cds_fasta))
    reference_fasta = open_reference_fasta(Path(args.reference_genome_fasta))
    reference_lookup = reference_contig_lookup(reference_fasta)
    min_mapq = int(args.coordinate_min_mapq)
    reference_build = str(args.reference_build or "GRCh38")

    parent_by_id = {str(row["parent_record_id"]): row for row in parent_core_rows}
    parent_coordinates: list[dict[str, object]] = []
    parent_orfcds: list[dict[str, object]] = []
    parent_status: dict[str, dict[str, object]] = {}

    try:
        for parent_id, parent_row in parent_by_id.items():
            stats = bam_stats.get(parent_id, {"primary": [], "secondary": [], "supplementary": []})
            primary = list(stats.get("primary", []))
            secondary = list(stats.get("secondary", []))
            supplementary = list(stats.get("supplementary", []))
            bed_entries = bed12_entries.get(parent_id, [])
            reasons: list[str] = []
            status = "coordinate_evaluable"
            chosen: dict[str, object] | None = primary[0] if len(primary) == 1 else None

            if len(primary) != 1:
                status = "ambiguous_candidate_mapping"
                reasons.append("primary_alignment_count_not_one")
            if secondary:
                status = "ambiguous_candidate_mapping"
                reasons.append("secondary_alignment_present")
            if supplementary:
                status = "ambiguous_candidate_mapping"
                reasons.append("supplementary_alignment_present")
            if chosen is None:
                chosen = primary[0] if primary else None

            blocks = [block for block in (chosen or {}).get("blocks", []) if isinstance(block, GenomicBlock)]
            chrom = str((chosen or {}).get("chromosome", ""))
            strand = str((chosen or {}).get("strand", ""))
            mapq_values = [int(aln.get("mapq", 0)) for aln in primary]
            mapq = mapq_values[0] if len(mapq_values) == 1 else ""
            if chosen is not None and mapq != "" and int(mapq) < min_mapq:
                status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                reasons.append(f"mapq_below_{min_mapq}")
            if chosen is not None and not canonical_chromosome(chrom):
                status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                reasons.append("noncanonical_contig")

            tx_blocks = transcript_order(blocks, strand) if blocks and strand in {"+", "-"} else []
            g_blocks = genomic_order(blocks)
            input_cds_seq = cds_records.get(parent_id, "")
            input_cds_length = len(input_cds_seq)
            block_total = sum(block.length for block in blocks)
            input_translated = translate_cds(input_cds_seq) if input_cds_seq else ""
            parent_sequence = str(parent_row.get("parent_sequence", ""))
            input_translation_status = "pass"
            if not input_cds_seq:
                input_translation_status = "missing_cds_fasta_record"
                status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                reasons.append("missing_cds_fasta_record")
            elif input_translated != parent_sequence:
                input_translation_status = "input_cds_translation_mismatch"
                status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                reasons.append("input_cds_translation_mismatch")
            if chosen is not None and input_cds_seq and block_total != input_cds_length:
                status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                reasons.append("input_cds_length_block_total_mismatch")

            reference_cds_seq = ""
            reference_reasons: list[str] = []
            reference_translation_status = "not_evaluable_candidate_coordinate_qc"
            rna_variant_coordinate_status = "not_evaluated"
            if chosen is None or not tx_blocks:
                reference_reasons.append("missing_trusted_primary_blocks")
            else:
                reference_cds_seq, reference_reasons = extract_transcript_cds_from_reference(
                    reference_fasta,
                    reference_lookup,
                    tx_blocks,
                )
            if reference_reasons:
                status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                reasons.extend(reference_reasons)
                reference_translation_status = "reference_translation_not_evaluable"
            else:
                reference_translated = translate_cds(reference_cds_seq)
                if reference_translated == parent_sequence:
                    reference_translation_status = "pass"
                    if rna_variant_context is not None:
                        rna_eval = evaluate_parent_rna_variants(
                            sample=args.sample,
                            parent_record_id=parent_id,
                            tx_blocks=tx_blocks,
                            strand=strand,
                            reference_cds_seq=reference_cds_seq,
                            candidate_cds_seq=input_cds_seq,
                            candidate_parent_sequence=parent_sequence,
                            vcf_events=rna_variant_context["vcf_events"],  # type: ignore[arg-type,index]
                            rediportal_events=rna_variant_context["rediportal_events"],  # type: ignore[arg-type,index]
                            rediportal_evaluation_status=str(rna_variant_context["rediportal_evaluation_status"]),
                            primary_min_alt_reads=int(rna_variant_context["primary_min_alt_reads"]),
                            sensitivity_alt_reads=tuple(rna_variant_context["sensitivity_alt_reads"]),  # type: ignore[arg-type,index]
                            min_variant_qual=float(rna_variant_context["min_variant_qual"]),
                            min_total_depth=int(rna_variant_context["min_total_depth"]),
                            min_variant_allele_fraction=float(rna_variant_context["min_variant_allele_fraction"]),
                            min_mapping_quality=float(rna_variant_context["min_mapping_quality"]),
                        )
                        record_rna_variant_evaluation(rna_variant_context, parent_id, rna_eval)
                        rna_variant_coordinate_status = "reference_concordant"
                else:
                    if rna_variant_context is None:
                        status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                        reasons.append("reference_translation_mismatch")
                        reference_translation_status = "reference_translation_mismatch"
                        rna_variant_coordinate_status = "RNA_variant_aware_not_evaluated"
                    else:
                        rna_eval = evaluate_parent_rna_variants(
                            sample=args.sample,
                            parent_record_id=parent_id,
                            tx_blocks=tx_blocks,
                            strand=strand,
                            reference_cds_seq=reference_cds_seq,
                            candidate_cds_seq=input_cds_seq,
                            candidate_parent_sequence=parent_sequence,
                            vcf_events=rna_variant_context["vcf_events"],  # type: ignore[arg-type,index]
                            rediportal_events=rna_variant_context["rediportal_events"],  # type: ignore[arg-type,index]
                            rediportal_evaluation_status=str(rna_variant_context["rediportal_evaluation_status"]),
                            primary_min_alt_reads=int(rna_variant_context["primary_min_alt_reads"]),
                            sensitivity_alt_reads=tuple(rna_variant_context["sensitivity_alt_reads"]),  # type: ignore[arg-type,index]
                            min_variant_qual=float(rna_variant_context["min_variant_qual"]),
                            min_total_depth=int(rna_variant_context["min_total_depth"]),
                            min_variant_allele_fraction=float(rna_variant_context["min_variant_allele_fraction"]),
                            min_mapping_quality=float(rna_variant_context["min_mapping_quality"]),
                        )
                        record_rna_variant_evaluation(rna_variant_context, parent_id, rna_eval)
                        parent_summary = rna_eval["parent_summary"]
                        rescue_status = str(parent_summary.get("rna_variant_rescue_status", ""))
                        rna_variant_coordinate_status = rescue_status
                        if rescue_status == "rna_variant_rescued":
                            reference_translation_status = "rna_variant_rescued"
                        else:
                            status = "not_evaluable_candidate_coordinate_qc" if status == "coordinate_evaluable" else status
                            reasons.append(rescue_status or "reference_translation_mismatch")
                            reference_translation_status = "reference_translation_mismatch"
            terminal_stop = (
                len(input_cds_seq) >= 3
                and len(input_cds_seq) % 3 == 0
                and input_cds_seq[-3:].upper() in {"TAA", "TAG", "TGA"}
            )
            parent_info = {
                "sample": args.sample,
                "parent_record_id": parent_id,
                "chromosome": chrom,
                "strand": strand,
                "genomic_start0": min((block.start0 for block in blocks), default=""),
                "genomic_end0": max((block.end0 for block in blocks), default=""),
                "block_count": len(blocks),
                "source_blocks_genomic_order": blocks_to_text(g_blocks),
                "blocks_transcript_order": blocks_to_text(tx_blocks),
                "primary_alignment_count": len(primary),
                "secondary_alignment_count": len(secondary),
                "supplementary_alignment_count": len(supplementary),
                "MAPQ": mapq,
                "cds_nucleotide_length": input_cds_length,
                "block_total_length": block_total,
                "terminal_stop_in_cds": terminal_stop,
                "reference_genomic_cds_length": len(reference_cds_seq),
                "input_cds_translation_status": input_translation_status,
                "reference_genomic_translation_status": reference_translation_status,
                "rna_variant_coordinate_status": rna_variant_coordinate_status,
                "reference_build": reference_build,
                "bed12_alignment_count": len(bed_entries),
                "all_alignment_loci": summarize_alignment_loci(primary + secondary + supplementary),
                "coordinate_mapping_status": status,
                "coordinate_mapping_reasons": ";".join(sorted(set(reasons))),
            }
            parent_coordinates.append(parent_info)
            parent_status[parent_id] = {**parent_info, "transcript_blocks": tx_blocks}

            for idx, block in enumerate(tx_blocks, start=1):
                cumulative_before = sum(prev.length for prev in tx_blocks[: idx - 1])
                parent_orfcds.append({
                    "sample": args.sample,
                    "parent_record_id": parent_id,
                    "transcript_block_order": idx,
                    "chromosome": block.chrom,
                    "strand": block.strand,
                    "start0": block.start0,
                    "end0": block.end0,
                    "block_length": block.length,
                    "derived_phase": cumulative_before % 3,
                    "phase_provenance": "derived_from_zero_phase_candidate_orf_start",
                    "coordinate_mapping_status": status,
                })
    finally:
        reference_fasta.close()

    peptide_footprints = build_peptide_footprints_from_parent_status(
        args.sample,
        peptide_parent_rows,
        parent_status,
        reference_build,
    )

    summary = {
        "status": "evaluated",
        "parent_rows": len(parent_coordinates),
        "parent_coordinate_evaluable": sum(
            1 for row in parent_coordinates if row["coordinate_mapping_status"] == "coordinate_evaluable"
        ),
        "parent_coordinate_not_evaluable": sum(
            1 for row in parent_coordinates if row["coordinate_mapping_status"] != "coordinate_evaluable"
        ),
        "parent_reference_translation_pass": sum(
            1 for row in parent_coordinates if row["reference_genomic_translation_status"] == "pass"
        ),
        "parent_reference_translation_rna_variant_rescued": sum(
            1 for row in parent_coordinates
            if row["reference_genomic_translation_status"] == "rna_variant_rescued"
        ),
        "parent_reference_translation_mismatch": sum(
            1 for row in parent_coordinates
            if row["reference_genomic_translation_status"] == "reference_translation_mismatch"
        ),
        "parent_reference_translation_not_evaluable": sum(
            1 for row in parent_coordinates
            if row["reference_genomic_translation_status"] == "reference_translation_not_evaluable"
        ),
        "peptide_footprint_rows": len(peptide_footprints),
        "peptide_coordinate_evaluable": sum(
            1 for row in peptide_footprints if row["candidate_coordinate_status"] == "coordinate_evaluable"
        ),
    }
    return parent_coordinates, parent_orfcds, peptide_footprints, summary


def build_all_candidate_rows(
    sample: str,
    parent_core_rows: list[dict[str, object]],
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parent_row in parent_core_rows:
        rows.extend(iter_parent_candidate_rows(sample, parent_row, mhc_i_lengths, mhc_ii_lengths))
    return rows


def build_ranked_candidate_rows(
    args: argparse.Namespace,
    parent_core_rows: list[dict[str, object]],
    human_proteome: Optional[Path],
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
    max_hla_i_peptides: int,
    max_hla_ii_peptides: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
]:
    ranked_parents = sort_parent_core_for_ranked(parent_core_rows, args.sample)
    caps = {"MHC-I": int(max_hla_i_peptides), "MHC-II": int(max_hla_ii_peptides)}
    selected_by_class: dict[str, dict[tuple[str, str], dict[str, object]]] = {"MHC-I": {}, "MHC-II": {}}
    selected_keys: set[tuple[str, str]] = set()
    deferred_keys: set[tuple[str, str]] = set()
    peptide_id_by_key: dict[tuple[str, str], str] = {}
    excluded_peptide_rows: list[dict[str, object]] = []
    deferred_peptide_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    deferred_parent_rows: list[dict[str, object]] = []
    parent_rank_rows: list[dict[str, object]] = []
    examined_keys: set[tuple[str, str]] = set()
    human_reference_summaries: list[dict[str, object]] = []
    boundary_parent = ""
    pending_rows_by_key: dict[tuple[str, str], dict[str, object]] = {}
    parent_rank_by_id: dict[str, dict[str, object]] = {}
    parent_status_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    unmaterialized_deferred_parent_count = 0
    materialized_parent_rows: list[dict[str, object]] = []
    materialized_parent_ids: set[str] = set()

    def caps_full() -> bool:
        return all(len(selected_by_class[mhc_class]) >= cap for mhc_class, cap in caps.items())

    def pending_count(mhc_class: str) -> int:
        return sum(1 for key in pending_rows_by_key if key[0] == mhc_class)

    def should_flush() -> bool:
        if not pending_rows_by_key:
            return False
        if all(len(selected_by_class[mhc]) + pending_count(mhc) >= caps[mhc] for mhc in caps):
            return True
        return len(pending_rows_by_key) >= 50000

    def add_evidence(row: dict[str, object], status: str, reason: str) -> None:
        parent_id = str(row["parent_record_id"])
        parent_status_counts[parent_id][status] += 1
        evidence_rows.append({
            "sample": args.sample,
            "parent_record_id": parent_id,
            "source_parent_id": row.get("source_parent_id", ""),
            "parent_rank": row.get("_parent_rank", ""),
            "mhc_class": row["mhc_class"],
            "peptide": row["peptide"],
            "peptide_length": row["peptide_length"],
            "selection_status": status,
            "selection_reason": reason,
            "selection_digest": stable_selection_digest(
                args.sample,
                parent_id,
                str(row["mhc_class"]),
                str(row["peptide"]),
            ),
        })

    def flush_pending() -> None:
        nonlocal boundary_parent
        if not pending_rows_by_key:
            return
        candidate_keys = set(pending_rows_by_key)
        log_progress(
            "Human reference scan started for ranked selection: "
            f"pending_unique_peptides={len(candidate_keys)}"
        )
        human_matches, human_summary = load_human_reference_matches(
            human_proteome,
            {peptide for _mhc, peptide in candidate_keys},
        )
        log_progress(
            "Human reference scan complete: "
            f"pending_unique_peptides={len(candidate_keys)}, "
            f"matched_peptides={human_summary.get('matched_peptides', 0)}"
        )
        human_reference_summaries.append(human_summary)
        ordered = sorted(
            candidate_keys,
            key=lambda key: (
                int(pending_rows_by_key[key].get("_parent_rank", 0)),
                key[0],
                len(key[1]),
                stable_selection_digest(
                    args.sample,
                    str(pending_rows_by_key[key]["parent_record_id"]),
                    key[0],
                    key[1],
                ),
            ),
        )
        for key in ordered:
            row = dict(pending_rows_by_key[key])
            row.pop("_parent_rank", None)
            reason = ""
            status = "selected_for_binding"
            if human_summary["status"] == "evaluated" and key[1] in human_matches:
                status = "excluded_human_reference"
                reason = "human_reference_peptide_match"
                row["peptide_core_status"] = "excluded"
                row["peptide_qc_reasons"] = reason
                row["human_reference_peptide_status"] = "human_reference_peptide_match"
                excluded_peptide_rows.append(row)
            elif len(selected_by_class[key[0]]) >= caps[key[0]]:
                status = "not_selected_due_to_analysis_cap"
                reason = "not_selected_due_to_analysis_cap"
                row["peptide_core_status"] = "deferred"
                row["peptide_qc_reasons"] = reason
                row["human_reference_peptide_status"] = (
                    "human_reference_peptide_not_detected"
                    if human_summary["status"] == "evaluated"
                    else "not_evaluated"
                )
                deferred_peptide_rows.append(row)
                deferred_keys.add(key)
                if not boundary_parent:
                    boundary_parent = str(row["parent_record_id"])
            else:
                row["human_reference_peptide_status"] = (
                    "human_reference_peptide_not_detected"
                    if human_summary["status"] == "evaluated"
                    else "not_evaluated"
                )
                selected_by_class[key[0]][key] = row
                selected_keys.add(key)
                peptide_id_by_key[key] = stable_peptide_id(args.sample, key[0], key[1], len(peptide_id_by_key) + 1)
            add_evidence(pending_rows_by_key[key], status, reason)
        pending_rows_by_key.clear()

    for rank, parent_row in enumerate(ranked_parents, start=1):
        parent_id = str(parent_row["parent_record_id"])
        parent_rank_record = {
            "sample": args.sample,
            "parent_record_id": parent_id,
            "source_parent_id": parent_row.get("source_parent_id", ""),
            "parent_rank": rank,
            "TPM_tumor": parent_row.get("TPM_tumor", ""),
            "TPM_ctrl": parent_row.get("TPM_ctrl", ""),
            "log2FC": parent_row.get("log2FC", ""),
            "nc_class": parent_row.get("nc_class", ""),
            "parent_sequence_length": len(str(parent_row.get("parent_sequence", ""))),
            "ranking_policy": CRYPTIC_RANKING_POLICY,
            "ranking_digest": stable_selection_digest(args.sample, parent_id),
            "selection_status": "not_evaluated",
            "selection_reason": "",
        }
        parent_rank_by_id[parent_id] = parent_rank_record
        parent_rank_rows.append(parent_rank_record)

        if caps_full():
            parent_rank_record["selection_status"] = "unmaterialized_deferred_parent"
            parent_rank_record["selection_reason"] = "cap_reached_before_parent_tiling"
            deferred_parent_rows.append(dict(parent_rank_record))
            unmaterialized_deferred_parent_count += 1
            continue

        if parent_id not in materialized_parent_ids:
            materialized_parent_rows.append(parent_row)
            materialized_parent_ids.add(parent_id)

        queued = 0
        for row in iter_parent_candidate_rows(args.sample, parent_row, mhc_i_lengths, mhc_ii_lengths):
            mhc_class = str(row["mhc_class"])
            if len(selected_by_class[mhc_class]) + pending_count(mhc_class) >= caps[mhc_class]:
                continue
            key = (str(row["mhc_class"]), str(row["peptide"]))
            if key in examined_keys:
                continue
            row["_parent_rank"] = rank
            pending_rows_by_key[key] = row
            examined_keys.add(key)
            queued += 1
        if not queued:
            parent_rank_record["selection_status"] = "no_new_candidate_before_cap"
        else:
            parent_rank_record["selection_status"] = "examined_for_selection"
        if should_flush():
            flush_pending()

    flush_pending()

    for parent_id, rank_record in parent_rank_by_id.items():
        if rank_record["selection_status"] not in {"examined_for_selection", "no_new_candidate_before_cap"}:
            continue
        counts = parent_status_counts.get(parent_id, {})
        if counts.get("selected_for_binding", 0):
            rank_record["selection_status"] = (
                "boundary_partially_selected"
                if counts.get("not_selected_due_to_analysis_cap", 0)
                else "selected_for_binding"
            )
        elif counts.get("not_selected_due_to_analysis_cap", 0):
            rank_record["selection_status"] = "not_selected_due_to_analysis_cap"
            rank_record["selection_reason"] = "not_selected_due_to_analysis_cap"
            deferred_parent_rows.append(dict(rank_record))
        elif counts:
            rank_record["selection_status"] = "examined_no_selected_peptides"

    selected_metadata_by_key = {
        key: row
        for class_rows in selected_by_class.values()
        for key, row in class_rows.items()
    }
    deferred_metadata_by_key = {
        (str(row["mhc_class"]), str(row["peptide"])): row
        for row in deferred_peptide_rows
    }

    selected_window_rows = rebuild_selected_parent_windows(
        args.sample,
        materialized_parent_rows,
        selected_keys,
        peptide_id_by_key,
        mhc_i_lengths,
        mhc_ii_lengths,
        selected_metadata_by_key,
    )
    deferred_window_rows = rebuild_selected_parent_windows(
        args.sample,
        materialized_parent_rows,
        deferred_keys,
        {},
        mhc_i_lengths,
        mhc_ii_lengths,
        deferred_metadata_by_key,
    )
    unique_rows_by_key: dict[tuple[str, str], dict[str, object]] = {}
    support_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for row in selected_window_rows:
        key = (str(row["mhc_class"]), str(row["peptide"]))
        support_counts[key] += 1
        if key not in unique_rows_by_key:
            unique = dict(row)
            unique["supporting_window_count"] = 0
            unique_rows_by_key[key] = unique
    for key, unique in unique_rows_by_key.items():
        unique["supporting_window_count"] = support_counts[key]
    unique_core_peptide_rows = [
        unique_rows_by_key[key]
        for key in sorted(unique_rows_by_key, key=lambda item: (item[0], len(item[1]), item[1]))
    ]
    for row in selected_window_rows:
        row["supporting_window_count"] = support_counts[(str(row["mhc_class"]), str(row["peptide"]))]
    for row in excluded_peptide_rows:
        row["peptide_record_id"] = ""
        row["supporting_window_count"] = ""
    for row in deferred_peptide_rows:
        row["peptide_record_id"] = ""
        row["supporting_window_count"] = ""
    for row in deferred_window_rows:
        row["peptide_record_id"] = ""
        row["supporting_window_count"] = ""

    human_reference_summary = {
        "status": "evaluated" if human_proteome else "not_evaluated",
        "path": str(human_proteome) if human_proteome else "",
        "selection_scans": len(human_reference_summaries),
        "candidate_peptides": len(examined_keys),
        "matched_peptides": sum(int(summary.get("matched_peptides", 0)) for summary in human_reference_summaries),
        "records_scanned": sum(int(summary.get("records_scanned", 0)) for summary in human_reference_summaries),
        "standard_windows_evaluated": sum(int(summary.get("standard_windows_evaluated", 0)) for summary in human_reference_summaries),
    }
    if human_proteome:
        human_reference_summary["sha256"] = sha256_file(human_proteome)
    selection_summary = {
        "mode": RANKED_CAP_MODE,
        "ranking_policy": CRYPTIC_RANKING_POLICY,
        "max_hla_i_peptides": max_hla_i_peptides,
        "max_hla_ii_peptides": max_hla_ii_peptides,
        "examined_candidate_peptides": len(examined_keys),
        "materialization_policy": "initial_cap_plus_ranked_parent_stream",
        "materialization_coverage": "selected_and_boundary_candidates",
        "evaluated_deferred_peptide_universe_complete": False,
        "materialized_parent_records": len(materialized_parent_rows),
        "unmaterialized_deferred_parent_records": unmaterialized_deferred_parent_count,
        "selected_hla_i": len(selected_by_class["MHC-I"]),
        "selected_hla_ii": len(selected_by_class["MHC-II"]),
        "cap_reached_hla_i": len(selected_by_class["MHC-I"]) >= caps["MHC-I"],
        "cap_reached_hla_ii": len(selected_by_class["MHC-II"]) >= caps["MHC-II"],
        "exhausted_before_cap_hla_i": len(selected_by_class["MHC-I"]) < caps["MHC-I"],
        "exhausted_before_cap_hla_ii": len(selected_by_class["MHC-II"]) < caps["MHC-II"],
        "boundary_parent_record_id": boundary_parent,
        "deferred_parent_records": len(deferred_parent_rows),
        "deferred_peptide_window_rows": len(deferred_peptide_rows),
        "deferred_peptide_parent_map_rows": len(deferred_window_rows),
    }
    return (
        selected_window_rows,
        excluded_peptide_rows,
        deferred_peptide_rows,
        unique_core_peptide_rows,
        parent_rank_rows,
        evidence_rows,
        deferred_parent_rows,
        deferred_window_rows,
        human_reference_summary,
        selection_summary,
    )


def build_core(args: argparse.Namespace) -> dict[str, object]:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_version = str(args.policy_version)
    if policy_version not in {POLICY_VERSION_V1, POLICY_VERSION_V11}:
        raise ValueError(f"Unsupported Cryptic Core QC policy_version: {policy_version}")
    if is_v11(policy_version):
        require_coordinate_inputs(args)

    script_sha256 = sha256_file(Path(__file__))
    mhc_i_lengths = parse_int_list(args.mhc_i_lengths)
    mhc_ii_lengths = parse_int_list(args.mhc_ii_lengths)
    if set(mhc_i_lengths).intersection(mhc_ii_lengths):
        raise ValueError("MHC-I and MHC-II peptide lengths must not overlap for peptide-core binding")
    candidate_selection_mode, max_hla_i_peptides, max_hla_ii_peptides = validate_candidate_selection(
        args.candidate_selection_mode,
        args.max_hla_i_peptides,
        args.max_hla_ii_peptides,
    )
    junction_qc_enabled = bool(getattr(args, "junction_qc_enabled", False))
    if junction_qc_enabled and not is_v11(policy_version):
        raise ValueError("junction_qc requires cryptic_core_qc_v1.1 coordinate sidecars")
    junction_policy_version = str(getattr(args, "junction_policy_version", JUNCTION_POLICY_VERSION))
    if junction_qc_enabled and junction_policy_version != JUNCTION_POLICY_VERSION:
        raise ValueError(f"Unsupported junction QC policy version: {junction_policy_version}")
    junction_thresholds = parse_int_list(str(getattr(args, "junction_sensitivity_thresholds", "1,2,3,5")))
    star_pair_validation: dict[str, object] | None = None
    if junction_qc_enabled:
        validate_junction_policy(
            policy_version=junction_policy_version,
            primary_min_tumor_unique_reads=int(getattr(args, "primary_min_tumor_unique_reads", 2)),
            sensitivity_thresholds=tuple(junction_thresholds),
        )
        if not str(getattr(args, "star_pair_inputs", "")).strip():
            raise ValueError("junction_qc requires --star-pair-inputs")
        star_pair_validation = validate_star_pair_contract(Path(args.star_pair_inputs), args.sample)

    rna_variant_qc_enabled = bool(getattr(args, "rna_variant_editing_qc_enabled", False))
    if rna_variant_qc_enabled:
        ensure_rna_variant_qc_loaded()
    rna_variant_policy_version = str(getattr(args, "rna_variant_qc_policy_version", RNA_VARIANT_POLICY_VERSION))
    rna_variant_sensitivity_alt_reads = parse_int_list(
        str(getattr(args, "rna_variant_sensitivity_alt_reads", "2,3,5"))
    )
    if rna_variant_qc_enabled and not is_v11(policy_version):
        raise ValueError("rna_variant_editing_qc requires cryptic_core_qc_v1.1 coordinate sidecars")
    rna_variant_context: dict[str, object] | None = None
    rna_variant_vcf_summary: dict[str, object] = {"status": "not_evaluated"}
    rna_variant_calling_manifest_summary: dict[str, object] = {"status": "not_evaluated"}
    rediportal_summary: dict[str, object] = {"status": "not_evaluated"}
    if rna_variant_qc_enabled:
        if float(getattr(args, "rna_variant_min_base_quality", 20.0)) != 20.0:
            raise ValueError(f"{RNA_VARIANT_POLICY_VERSION} requires min_base_quality=20")
        validate_rna_variant_policy(
            policy_version=rna_variant_policy_version,
            primary_min_alt_reads=int(getattr(args, "rna_variant_primary_min_alt_reads", RNA_VARIANT_PRIMARY_MIN_ALT_READS)),
            sensitivity_alt_reads=tuple(rna_variant_sensitivity_alt_reads),
            min_variant_qual=float(getattr(args, "rna_variant_min_variant_qual", MIN_VARIANT_QUAL)),
            min_total_depth=int(getattr(args, "rna_variant_min_total_depth", RNA_VARIANT_MIN_TOTAL_DEPTH)),
            min_variant_allele_fraction=float(
                getattr(args, "rna_variant_min_variant_allele_fraction", MIN_VARIANT_ALLELE_FRACTION)
            ),
            min_mapping_quality=float(getattr(args, "rna_variant_min_mapping_quality", RNA_VARIANT_MIN_MAPPING_QUALITY)),
        )
        if not str(getattr(args, "rna_variant_vcf", "")).strip():
            raise ValueError("rna_variant_editing_qc requires --rna-variant-vcf")
        allow_legacy_rna_variant_vcf = bool(getattr(args, "allow_legacy_rna_variant_vcf", False))
        calling_manifest_arg = str(getattr(args, "rna_variant_calling_manifest", "")).strip()
        calling_manifest_path = Path(calling_manifest_arg) if calling_manifest_arg else Path(args.rna_variant_vcf).parent / "rna.variant_calling.manifest.json"
        if calling_manifest_path.exists():
            rna_variant_calling_manifest_summary = validate_rna_variant_calling_manifest(
                calling_manifest_path,
                vcf_path=Path(args.rna_variant_vcf),
                expected_sample=args.sample,
            )
        elif allow_legacy_rna_variant_vcf:
            rna_variant_calling_manifest_summary = {
                "status": "not_evaluated_legacy_vcf_allowed",
                "manifest_path": str(calling_manifest_path),
            }
        else:
            raise FileNotFoundError(
                "Formal RNA variant/editing QC requires rna.variant_calling.manifest.json. "
                "Use --allow-legacy-rna-variant-vcf only for exploratory runs."
            )
        allow_legacy_duplicate_vcf = bool(getattr(args, "allow_legacy_duplicate_vcf", False))
        vcf_events, rna_variant_vcf_summary = load_vcf_events(
            Path(args.rna_variant_vcf),
            expected_sample=args.sample,
            require_mq=True,
            allow_legacy_duplicate_vcf=allow_legacy_duplicate_vcf,
        )
        if str(getattr(args, "rediportal_processed_table", "")).strip():
            if not str(getattr(args, "rediportal_resource_manifest", "")).strip():
                raise FileNotFoundError(
                    "Formal RNA variant/editing QC requires --rediportal-resource-manifest "
                    "with processed table size/SHA/build provenance."
                )
            rediportal_manifest = Path(args.rediportal_resource_manifest)
            rediportal_events, rediportal_summary = load_rediportal_events(
                Path(args.rediportal_processed_table),
                rediportal_manifest,
            )
        elif bool(getattr(args, "allow_missing_rediportal_resource", False)):
            rediportal_events = set()
            rediportal_summary = {
                "status": "not_evaluated_missing_resource_allowed",
                "path": "",
                "records": 0,
            }
        else:
            raise FileNotFoundError(
                "REDIportal processed table is required for formal RNA variant/editing QC. "
                "Use --allow-missing-rediportal-resource only for exploratory runs."
            )
        rna_variant_context = {
            "vcf_events": vcf_events,
            "rediportal_events": rediportal_events,
            "rediportal_evaluation_status": str(rediportal_summary.get("status", "not_evaluated")),
            "vcf_summary": rna_variant_vcf_summary,
            "calling_manifest_summary": rna_variant_calling_manifest_summary,
            "rediportal_summary": rediportal_summary,
            "policy_version": rna_variant_policy_version,
            "allow_legacy_rna_variant_vcf": allow_legacy_rna_variant_vcf,
            "allow_legacy_duplicate_vcf": allow_legacy_duplicate_vcf,
            "primary_min_alt_reads": int(getattr(args, "rna_variant_primary_min_alt_reads", RNA_VARIANT_PRIMARY_MIN_ALT_READS)),
            "sensitivity_alt_reads": tuple(rna_variant_sensitivity_alt_reads),
            "min_variant_qual": float(getattr(args, "rna_variant_min_variant_qual", MIN_VARIANT_QUAL)),
            "min_total_depth": int(getattr(args, "rna_variant_min_total_depth", RNA_VARIANT_MIN_TOTAL_DEPTH)),
            "min_variant_allele_fraction": float(
                getattr(args, "rna_variant_min_variant_allele_fraction", MIN_VARIANT_ALLELE_FRACTION)
            ),
            "min_mapping_quality": float(getattr(args, "rna_variant_min_mapping_quality", RNA_VARIANT_MIN_MAPPING_QUALITY)),
            "evaluated_parent_ids": set(),
            "event_rows": [],
            "parent_summary_rows": [],
            "variant_aa_positions_by_parent": {},
            "threshold_parent_pass_counts": defaultdict(int),
        }

    input_paths = {
        "aeseps_fasta": file_identity(Path(args.ae_seps_fasta)),
        "aeseps_annotation": file_identity(Path(args.aeseps_annotation)),
        "orf_filtered_fasta": file_identity(Path(args.orf_filtered_fasta)),
        "orf_final": file_identity(Path(args.orf_final)),
    }
    if is_v11(policy_version):
        input_paths["orf_bed12"] = file_identity(Path(args.orf_bed12))
        input_paths["orf_bam"] = file_identity(Path(args.orf_bam))
        input_paths["orf_cds_fasta"] = file_identity(Path(args.orf_cds_fasta))
        input_paths["coordinate_utils"] = file_identity(COORDINATE_UTILS_PATH)
    if junction_qc_enabled:
        input_paths["star_pair_inputs"] = file_identity(Path(args.star_pair_inputs))
        input_paths["junction_qc_script"] = file_identity(JUNCTION_QC_PATH)
    if rna_variant_qc_enabled:
        input_paths["rna_variant_vcf"] = file_identity(Path(args.rna_variant_vcf))
        input_paths["rna_variant_qc_script"] = file_identity(RNA_VARIANT_QC_PATH)
        calling_manifest_path_for_signature = str(getattr(args, "rna_variant_calling_manifest", "")).strip()
        if not calling_manifest_path_for_signature:
            calling_manifest_path_for_signature = str(Path(args.rna_variant_vcf).parent / "rna.variant_calling.manifest.json")
        input_paths["rna_variant_calling_manifest"] = file_identity(Path(calling_manifest_path_for_signature))
        if str(getattr(args, "rediportal_processed_table", "")).strip():
            input_paths["rediportal_processed_table"] = file_identity(Path(args.rediportal_processed_table))
        if str(getattr(args, "rediportal_resource_manifest", "")).strip():
            input_paths["rediportal_resource_manifest"] = file_identity(Path(args.rediportal_resource_manifest))
    human_proteome = Path(args.human_proteome_fasta) if args.human_proteome_fasta else None
    if human_proteome:
        input_paths["human_reference_proteome_fasta"] = file_identity(human_proteome)
    elif not bool(args.allow_missing_human_reference):
        raise FileNotFoundError(
            "human reference proteome FASTA is required for formal Cryptic Core QC. "
            "Use --allow-missing-human-reference only for exploratory runs."
        )
    for key, value in [
        ("reference_genome_fasta", args.reference_genome_fasta),
        ("reference_gtf", args.reference_gtf),
        ("reference_lnc_gtf", args.reference_lnc_gtf),
    ]:
        if value:
            input_paths[key] = file_identity(Path(value))

    signature = {
        "policy_version": policy_version,
        "script_sha256": script_sha256,
        "coordinate_utils_sha256": sha256_file(COORDINATE_UTILS_PATH) if is_v11(policy_version) else "",
        "sample": args.sample,
        "matched_control_sample": args.matched_control_sample,
        "accepted_source_classes": sorted(ACCEPTED_SOURCE_CLASSES),
        "min_tpm_tumor": float(args.min_tpm_tumor),
        "max_tpm_ctrl": float(args.max_tpm_ctrl),
        "min_log2fc": float(args.min_log2fc),
        "mhc_i_lengths": mhc_i_lengths,
        "mhc_ii_lengths": mhc_ii_lengths,
        "candidate_selection_policy_version": CANDIDATE_SELECTION_POLICY_VERSION,
        "candidate_selection": {
            "mode": candidate_selection_mode,
            "max_hla_i_peptides": max_hla_i_peptides,
            "max_hla_ii_peptides": max_hla_ii_peptides,
            "ranking_policy": CRYPTIC_RANKING_POLICY if candidate_selection_mode == RANKED_CAP_MODE else "",
        },
        "allow_missing_human_reference": bool(args.allow_missing_human_reference),
        "strandedness": args.strandedness,
        "coordinate_min_mapq": int(args.coordinate_min_mapq),
        "junction_qc": {
            "enabled": junction_qc_enabled,
            "policy_version": junction_policy_version if junction_qc_enabled else "",
            "junction_qc_sha256": sha256_file(JUNCTION_QC_PATH) if junction_qc_enabled else "",
            "primary_min_tumor_unique_reads": int(getattr(args, "primary_min_tumor_unique_reads", 2)),
            "sensitivity_thresholds": junction_thresholds if junction_qc_enabled else [],
            "intronless_policy": "retain_as_not_applicable" if junction_qc_enabled else "",
            "normal_junction_policy": "annotate_only" if junction_qc_enabled else "",
            "mapping_min_mapq": int(args.coordinate_min_mapq) if junction_qc_enabled else "",
            "star_pair_validation": {
                key: value
                for key, value in (star_pair_validation or {}).items()
                if key != "row"
            } if junction_qc_enabled else {},
        },
        "rna_variant_editing_qc": {
            "enabled": rna_variant_qc_enabled,
            "policy_version": rna_variant_policy_version if rna_variant_qc_enabled else "",
            "rna_variant_qc_sha256": sha256_file(RNA_VARIANT_QC_PATH) if rna_variant_qc_enabled else "",
            "min_read_mapping_quality": int(getattr(args, "rna_variant_min_mapping_quality", RNA_VARIANT_MIN_MAPPING_QUALITY)),
            "min_base_quality": 20 if rna_variant_qc_enabled else "",
            "min_variant_qual": float(getattr(args, "rna_variant_min_variant_qual", MIN_VARIANT_QUAL)),
            "min_total_depth": int(getattr(args, "rna_variant_min_total_depth", RNA_VARIANT_MIN_TOTAL_DEPTH)),
            "min_variant_allele_fraction": float(
                getattr(args, "rna_variant_min_variant_allele_fraction", MIN_VARIANT_ALLELE_FRACTION)
            ),
            "primary_min_alt_reads": int(getattr(args, "rna_variant_primary_min_alt_reads", RNA_VARIANT_PRIMARY_MIN_ALT_READS)),
            "sensitivity_alt_reads": rna_variant_sensitivity_alt_reads if rna_variant_qc_enabled else [],
            "known_editing_policy": "exclude" if rna_variant_qc_enabled else "",
            "complex_variant_policy": "not_evaluable" if rna_variant_qc_enabled else "",
            "allow_missing_rediportal_resource": bool(getattr(args, "allow_missing_rediportal_resource", False)),
            "allow_legacy_rna_variant_vcf": bool(getattr(args, "allow_legacy_rna_variant_vcf", False)),
            "allow_legacy_duplicate_vcf": bool(getattr(args, "allow_legacy_duplicate_vcf", False)),
            "vcf_summary": rna_variant_vcf_summary if rna_variant_qc_enabled else {},
            "calling_manifest_summary": rna_variant_calling_manifest_summary if rna_variant_qc_enabled else {},
            "rediportal_summary": rediportal_summary if rna_variant_qc_enabled else {},
        },
        "reference_build": str(args.reference_build or "GRCh38"),
        "input_paths": input_paths,
    }
    exploratory_reasons: list[str] = []
    if bool(args.allow_missing_human_reference):
        exploratory_reasons.append("missing_human_reference_allowed")
    if rna_variant_qc_enabled:
        if str(rediportal_summary.get("status", "")) != "evaluated":
            exploratory_reasons.append("rediportal_not_evaluated")
        if str(rna_variant_calling_manifest_summary.get("status", "")) != "validated":
            exploratory_reasons.append("legacy_rna_variant_calling_manifest_not_validated")
        if bool(getattr(args, "allow_legacy_rna_variant_vcf", False)):
            exploratory_reasons.append("legacy_rna_variant_vcf_allowed")
        if bool(getattr(args, "allow_legacy_duplicate_vcf", False)):
            exploratory_reasons.append("legacy_duplicate_vcf_allowed")
    run_status = "complete_exploratory" if exploratory_reasons else "complete"
    binding_eligible = not exploratory_reasons

    manifest_path = outdir / "run_manifest.json"
    required_outputs = [
        outdir / "cryptic_parent_records.tsv",
        outdir / "cryptic_parent_core.tsv",
        outdir / "cryptic_parent_excluded.tsv",
        outdir / "cryptic_parent_core.fasta",
        outdir / "cryptic_peptide_core_hla_i.fasta",
        outdir / "cryptic_peptide_core_hla_ii.fasta",
        outdir / "cryptic_peptide_core.fasta",
        outdir / "cryptic_peptide_core_hla_i.tsv",
        outdir / "cryptic_peptide_core_hla_ii.tsv",
        outdir / "cryptic_peptide_core.tsv",
        outdir / "cryptic_peptide_parent_map.tsv",
        outdir / "cryptic_normal_resource_evidence.tsv",
        outdir / "stagewise_qc.tsv",
    ]
    if candidate_selection_mode == RANKED_CAP_MODE:
        required_outputs.extend([
            outdir / "cryptic_parent_ranked.tsv",
            outdir / "cryptic_peptide_selection_evidence.tsv",
            outdir / "cryptic_deferred_parent.tsv",
            outdir / "cryptic_peptide_deferred.tsv",
            outdir / "cryptic_peptide_deferred_parent_map.tsv",
        ])
    if is_v11(policy_version):
        required_outputs.extend([
            outdir / "cryptic_parent_coordinates.tsv",
            outdir / "cryptic_parent_orfcds.tsv",
            outdir / "cryptic_peptide_genomic_footprint.tsv",
        ])
    if junction_qc_enabled:
        required_outputs.extend([
            outdir / "cryptic_parent_junctions.tsv",
            outdir / "cryptic_parent_junction_summary.tsv",
            outdir / "cryptic_peptide_junction_evidence.tsv",
            outdir / "junction_threshold_sensitivity.tsv",
            outdir / "junction_qc_stagewise.tsv",
            outdir / "junction_qc.manifest.json",
        ])
    if rna_variant_qc_enabled:
        required_outputs.extend([
            outdir / "cryptic_rna_variant_events.tsv",
            outdir / "cryptic_parent_rna_variant_summary.tsv",
            outdir / "cryptic_peptide_rna_variant_evidence.tsv",
            outdir / "cryptic_rna_editing_excluded.tsv",
            outdir / "rna_variant_threshold_sensitivity.tsv",
            outdir / "rna_variant_qc_stagewise.tsv",
            outdir / "rna_variant_qc.manifest.json",
        ])
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("input_signature") != signature:
            raise ValueError(
                f"Existing Cryptic Core manifest does not match current inputs/config: {manifest_path}"
            )
        old_output_signature = old.get("output_signature")
        if isinstance(old_output_signature, dict) and outputs_match_signature(old_output_signature):
            return {**old, "reused": True}

    log_progress(
        "Cryptic Core QC started: "
        f"sample={args.sample}, policy_version={policy_version}, "
        f"candidate_selection={candidate_selection_mode}"
    )
    annot = pd.read_csv(args.aeseps_annotation, low_memory=False)
    if "Name" not in annot.columns or "nc_class" not in annot.columns or "is_aberrant" not in annot.columns:
        raise ValueError("aeSEP annotation table must contain Name, nc_class, and is_aberrant columns")
    annot["Name"] = annot["Name"].astype(str)
    annot_by_name = annot.drop_duplicates("Name", keep="first").set_index("Name")

    orf_final = pd.read_csv(args.orf_final)
    if "TranscriptID" not in orf_final.columns:
        raise ValueError("orf_final table must contain TranscriptID")
    allowed_orf_ids = set(orf_final["TranscriptID"].astype(str))

    parent_rows: list[dict[str, object]] = []
    parent_core_rows: list[dict[str, object]] = []
    parent_excluded_rows: list[dict[str, object]] = []
    candidate_peptide_rows: list[dict[str, object]] = []
    parent_fasta_rows: list[tuple[str, str]] = []

    seen_records: set[str] = set()
    for header, raw_seq in read_fasta(Path(args.orf_filtered_fasta)):
        record_id = fasta_record_id(header)
        if record_id in seen_records:
            raise ValueError(f"Duplicate ORF record id in ORF-filtered aeSEP FASTA: {record_id}")
        seen_records.add(record_id)
        src_id = source_parent_id(record_id)
        sequence, seq_reasons = normalize_parent_sequence(raw_seq)
        reasons = list(seq_reasons)

        if record_id not in allowed_orf_ids:
            reasons.append("not_in_orf_final")
        if src_id not in annot_by_name.index:
            reasons.append("source_join_failure")
            nc_class = ""
            expression_info = {
                "TPM_tumor": "",
                "TPM_ctrl": "",
                "log2FC": "",
                "expression_qc_status": "fail",
                "expression_qc_reasons": "source_join_failure",
                "upstream_is_aberrant": "",
                "upstream_is_aberrant_consistent": "",
            }
            tpm_tumor = ""
            tpm_ctrl = ""
            log2fc = ""
        else:
            row = annot_by_name.loc[src_id]
            nc_class = str(row.get("nc_class", ""))
            expression_pass, expression_reasons, expression_info = evaluate_expression(
                row,
                min_tpm_tumor=float(args.min_tpm_tumor),
                max_tpm_ctrl=float(args.max_tpm_ctrl),
                min_log2fc=float(args.min_log2fc),
            )
            tpm_tumor = expression_info["TPM_tumor"]
            tpm_ctrl = expression_info["TPM_ctrl"]
            log2fc = expression_info["log2FC"]
            if nc_class not in ACCEPTED_SOURCE_CLASSES:
                reasons.append(f"source_not_core:{nc_class or 'missing'}")
            reasons.extend(expression_reasons)

        parent_status = "core" if not reasons else "excluded"
        parent_row = {
            "sample": args.sample,
            "parent_record_id": record_id,
            "source_parent_id": src_id,
            "nc_class": nc_class,
            "upstream_is_aberrant": expression_info["upstream_is_aberrant"],
            "upstream_is_aberrant_consistent": expression_info["upstream_is_aberrant_consistent"],
            "TPM_tumor": tpm_tumor,
            "TPM_ctrl": tpm_ctrl,
            "log2FC": log2fc,
            "expression_qc_status": expression_info["expression_qc_status"],
            "expression_qc_reasons": expression_info["expression_qc_reasons"],
            "parent_sequence": sequence,
            "parent_sequence_sha256": sha256_text(sequence) if sequence else "",
            "parent_qc_status": parent_status,
            "parent_qc_reasons": ";".join(reasons),
            "rna_variant_status": "not_evaluated",
            "rna_editing_qc_status": "editing_resource_not_evaluated",
            "junction_support_status": "not_evaluated",
            "primary_core_status": "primary_core_eligible" if parent_status == "core" else "excluded_upstream_qc",
            "primary_core_reasons": "",
            "rna_variant_rescue_status": "not_evaluated",
            "cryptic_discovery_fdr_status": "not_estimable_single_pair_rule_based_qc",
        }
        parent_rows.append(parent_row)
        if parent_status == "core":
            parent_core_rows.append(parent_row)
            parent_fasta_rows.append((record_id, sequence))
            if candidate_selection_mode == ALL_MODE and not junction_qc_enabled and not rna_variant_qc_enabled:
                candidate_peptide_rows.extend(
                    iter_parent_candidate_rows(args.sample, parent_row, mhc_i_lengths, mhc_ii_lengths)
                )
        else:
            parent_excluded_rows.append(parent_row)

    log_progress(
        "Parent QC complete: "
        f"parents={len(parent_rows)}, pre_coordinate_core={len(parent_core_rows)}, "
        f"excluded={len(parent_excluded_rows)}"
    )

    phase_a_parent_coordinate_rows: list[dict[str, object]] = []
    phase_a_parent_orfcds_rows: list[dict[str, object]] = []
    phase_a_coordinate_summary: dict[str, object] = {"status": "not_evaluated"}
    junction_result: dict[str, object] | None = None
    if junction_qc_enabled or rna_variant_qc_enabled:
        (
            phase_a_parent_coordinate_rows,
            phase_a_parent_orfcds_rows,
            _phase_a_peptide_footprints,
            phase_a_coordinate_summary,
        ) = build_coordinate_sidecars(args, parent_core_rows, [], policy_version, rna_variant_context)
        log_progress(
            "Coordinate phase-A QC complete: "
            f"parents={phase_a_coordinate_summary.get('parent_rows', 0)}, "
            f"coordinate_evaluable={phase_a_coordinate_summary.get('parent_coordinate_evaluable', 0)}, "
            f"coordinate_not_evaluable={phase_a_coordinate_summary.get('parent_coordinate_not_evaluable', 0)}"
        )

    if rna_variant_qc_enabled and rna_variant_context is not None:
        summaries = {
            str(row.get("parent_record_id", "")): row
            for row in rna_variant_context.get("parent_summary_rows", [])  # type: ignore[union-attr]
        }
        filtered_core_rows: list[dict[str, object]] = []
        filtered_fasta_rows: list[tuple[str, str]] = []
        for row in parent_core_rows:
            parent_id = str(row["parent_record_id"])
            summary = summaries.get(parent_id, {})
            rescue_status = str(summary.get("rna_variant_rescue_status", "rna_variant_not_evaluated"))
            eligibility = str(summary.get("primary_core_eligibility", "provisional_coordinate_not_evaluable"))
            row["rna_variant_status"] = rescue_status
            row["rna_variant_rescue_status"] = rescue_status
            if rescue_status == "reference_concordant":
                row["rna_editing_qc_status"] = "not_applicable_reference_concordant"
            elif str(rediportal_summary.get("status", "")) == "evaluated":
                row["rna_editing_qc_status"] = (
                    "known_rna_editing_excluded"
                    if int(summary.get("known_editing_count", 0))
                    else "no_known_editing_required_variant"
                )
            else:
                row["rna_editing_qc_status"] = str(rediportal_summary.get("status", "editing_resource_not_evaluated"))
            if eligibility == "primary_core_eligible":
                filtered_core_rows.append(row)
                filtered_fasta_rows.append((parent_id, str(row["parent_sequence"])))
            else:
                row["primary_core_status"] = eligibility
                row["parent_qc_status"] = eligibility
                reasons = [str(row.get("parent_qc_reasons", "")), rescue_status]
                row["parent_qc_reasons"] = ";".join(token for token in reasons if token)
                row["primary_core_reasons"] = rescue_status
                parent_excluded_rows.append(row)
        parent_core_rows = filtered_core_rows
        parent_fasta_rows = filtered_fasta_rows

        if candidate_selection_mode == ALL_MODE and not junction_qc_enabled:
            candidate_peptide_rows = build_all_candidate_rows(
                args.sample,
                parent_core_rows,
                mhc_i_lengths,
                mhc_ii_lengths,
            )
        log_progress(f"RNA variant/editing QC complete: retained_parent_core={len(parent_core_rows)}")

    if junction_qc_enabled:
        if star_pair_validation is None:
            raise ValueError("Internal error: junction_qc star pair validation was not initialized")
        star_pair = star_pair_validation["row"]  # type: ignore[index]
        junction_result = evaluate_parent_junctions(
            sample=args.sample,
            parent_coordinates=phase_a_parent_coordinate_rows,
            tumor_sj_path=Path(star_pair["tumor_sj_path"]),
            normal_sj_path=Path(star_pair["normal_sj_path"]),
            primary_min_tumor_unique_reads=int(getattr(args, "primary_min_tumor_unique_reads", 2)),
            sensitivity_thresholds=tuple(junction_thresholds),
            min_mapq=int(args.coordinate_min_mapq),
        )
        summary_by_parent = {
            str(row.get("parent_record_id", "")): row
            for row in junction_result["parent_summary_rows"]  # type: ignore[index]
        }
        filtered_core_rows: list[dict[str, object]] = []
        filtered_fasta_rows: list[tuple[str, str]] = []
        for row in parent_core_rows:
            parent_id = str(row["parent_record_id"])
            summary = summary_by_parent.get(parent_id, {})
            primary_status = str(summary.get("primary_core_status", "provisional_coordinate_not_evaluable"))
            row["primary_core_status"] = primary_status
            row["primary_core_reasons"] = summary.get("junction_qc_reasons", "")
            row["junction_support_status"] = summary.get("junction_qc_status", "")
            row["rna_variant_rescue_status"] = summary.get("rna_variant_rescue_status", "not_evaluated")
            if primary_status == JUNCTION_PRIMARY_PASS:
                filtered_core_rows.append(row)
                filtered_fasta_rows.append((parent_id, str(row["parent_sequence"])))
            else:
                row["parent_qc_status"] = primary_status
                reasons = [str(row.get("parent_qc_reasons", "")), str(summary.get("junction_qc_reasons", ""))]
                row["parent_qc_reasons"] = ";".join(token for token in reasons if token)
                parent_excluded_rows.append(row)
        parent_core_rows = filtered_core_rows
        parent_fasta_rows = filtered_fasta_rows
        if candidate_selection_mode == ALL_MODE:
            candidate_peptide_rows = build_all_candidate_rows(
                args.sample,
                parent_core_rows,
                mhc_i_lengths,
                mhc_ii_lengths,
            )
        log_progress(
            "Junction QC complete: "
            f"retained_parent_core={len(parent_core_rows)}, total_excluded_parents={len(parent_excluded_rows)}"
        )

    parent_rank_rows: list[dict[str, object]] = []
    peptide_selection_evidence_rows: list[dict[str, object]] = []
    deferred_parent_rows: list[dict[str, object]] = []
    deferred_peptide_rows: list[dict[str, object]] = []
    deferred_peptide_parent_map_rows: list[dict[str, object]] = []
    selection_summary: dict[str, object] = {
        "mode": candidate_selection_mode,
        "ranking_policy": CRYPTIC_RANKING_POLICY if candidate_selection_mode == RANKED_CAP_MODE else "",
        "max_hla_i_peptides": max_hla_i_peptides,
        "max_hla_ii_peptides": max_hla_ii_peptides,
    }

    if candidate_selection_mode == RANKED_CAP_MODE:
        log_progress(
            "Ranked peptide selection started: "
            f"parents={len(parent_core_rows)}, max_hla_i={max_hla_i_peptides}, max_hla_ii={max_hla_ii_peptides}"
        )
        (
            core_peptide_rows,
            excluded_peptide_rows,
            deferred_peptide_rows,
            unique_core_peptide_rows,
            parent_rank_rows,
            peptide_selection_evidence_rows,
            deferred_parent_rows,
            deferred_peptide_parent_map_rows,
            human_reference_summary,
            ranked_selection_summary,
        ) = build_ranked_candidate_rows(
            args,
            parent_core_rows,
            human_proteome,
            mhc_i_lengths,
            mhc_ii_lengths,
            int(max_hla_i_peptides),
            int(max_hla_ii_peptides),
        )
        candidate_peptide_rows = core_peptide_rows + excluded_peptide_rows + deferred_peptide_rows
        selection_summary.update(ranked_selection_summary)
        log_progress(
            "Ranked peptide selection complete: "
            f"selected_hla_i={selection_summary.get('selected_hla_i', 0)}, "
            f"selected_hla_ii={selection_summary.get('selected_hla_ii', 0)}, "
            f"examined_unique_peptides={selection_summary.get('examined_candidate_peptides', 0)}"
        )
    else:
        candidate_peptides = {str(row["peptide"]) for row in candidate_peptide_rows}
        human_matches, human_reference_summary = load_human_reference_matches(human_proteome, candidate_peptides)

        core_peptide_rows = []
        excluded_peptide_rows = []
        for row in candidate_peptide_rows:
            peptide = str(row["peptide"])
            if human_reference_summary["status"] == "evaluated":
                if peptide in human_matches:
                    row["peptide_core_status"] = "excluded"
                    row["peptide_qc_reasons"] = "human_reference_peptide_match"
                    row["human_reference_peptide_status"] = "human_reference_peptide_match"
                    excluded_peptide_rows.append(row)
                    continue
                row["human_reference_peptide_status"] = "human_reference_peptide_not_detected"
            core_peptide_rows.append(row)

        peptide_id_by_key: dict[tuple[str, str], str] = {}
        unique_core_peptide_rows = []
        support_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        for row in core_peptide_rows:
            key = (str(row["mhc_class"]), str(row["peptide"]))
            support_counts[key] += 1
            if key not in peptide_id_by_key:
                peptide_id = stable_peptide_id(args.sample, key[0], key[1], len(unique_core_peptide_rows) + 1)
                peptide_id_by_key[key] = peptide_id
                unique_row = dict(row)
                unique_row["peptide_record_id"] = peptide_id
                unique_row["supporting_window_count"] = 0
                unique_core_peptide_rows.append(unique_row)
            row["peptide_record_id"] = peptide_id_by_key[key]
        for row in unique_core_peptide_rows:
            key = (str(row["mhc_class"]), str(row["peptide"]))
            row["supporting_window_count"] = support_counts[key]
        for row in excluded_peptide_rows:
            row["peptide_record_id"] = ""
            row["supporting_window_count"] = ""

    hla_i_rows = [row for row in unique_core_peptide_rows if row["mhc_class"] == "MHC-I"]
    hla_ii_rows = [row for row in unique_core_peptide_rows if row["mhc_class"] == "MHC-II"]
    if junction_qc_enabled or rna_variant_qc_enabled:
        parent_coordinate_rows = phase_a_parent_coordinate_rows
        parent_orfcds_rows = phase_a_parent_orfcds_rows
        parent_status = build_parent_status_from_coordinate_sidecars(
            parent_coordinate_rows,
            parent_orfcds_rows,
        )
        peptide_footprint_rows = build_peptide_footprints_from_parent_status(
            args.sample,
            core_peptide_rows,
            parent_status,
            str(args.reference_build or "GRCh38"),
        )
        coordinate_summary = {
            **phase_a_coordinate_summary,
            "peptide_footprint_rows": len(peptide_footprint_rows),
            "peptide_coordinate_evaluable": sum(
                1 for row in peptide_footprint_rows if row["candidate_coordinate_status"] == "coordinate_evaluable"
            ),
        }
    else:
        parent_coordinate_rows, parent_orfcds_rows, peptide_footprint_rows, coordinate_summary = build_coordinate_sidecars(
            args,
            parent_core_rows,
            core_peptide_rows,
            policy_version,
            rna_variant_context,
        )
    log_progress(
        "Peptide coordinate footprint complete: "
        f"peptide_parent_rows={len(core_peptide_rows)}, footprint_rows={len(peptide_footprint_rows)}"
    )
    if junction_qc_enabled:
        peptide_junction_rows = annotate_peptide_junctions(
            sample=args.sample,
            peptide_parent_rows=core_peptide_rows,
            peptide_footprint_rows=peptide_footprint_rows,
            parent_summary_rows=junction_result["parent_summary_rows"] if junction_result else [],  # type: ignore[index]
        )
    else:
        peptide_junction_rows: list[dict[str, object]] = []

    with (outdir / "cryptic_parent_core.fasta").open("w") as handle:
        for record_id, sequence in parent_fasta_rows:
            handle.write(f">{record_id}\n{sequence}\n")
    write_peptide_core_fasta(hla_i_rows, outdir / "cryptic_peptide_core_hla_i.fasta")
    write_peptide_core_fasta(hla_ii_rows, outdir / "cryptic_peptide_core_hla_ii.fasta")
    write_peptide_core_fasta(unique_core_peptide_rows, outdir / "cryptic_peptide_core.fasta")

    parent_fields = [
        "sample", "parent_record_id", "source_parent_id", "nc_class",
        "upstream_is_aberrant", "upstream_is_aberrant_consistent",
        "TPM_tumor", "TPM_ctrl", "log2FC", "expression_qc_status", "expression_qc_reasons",
        "parent_sequence", "parent_sequence_sha256",
        "parent_qc_status", "parent_qc_reasons", "rna_variant_status",
        "rna_editing_qc_status", "junction_support_status", "primary_core_status",
        "primary_core_reasons", "rna_variant_rescue_status", "cryptic_discovery_fdr_status",
    ]
    peptide_fields = [
        "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class", "peptide_start",
        "peptide_length", "peptide", "peptide_sequence_sha256", "peptide_core_status",
        "peptide_qc_reasons", "human_reference_peptide_status",
        "normal_hla_presentation_status", "normal_translation_status", "external_normal_status",
        "supporting_window_count",
    ]
    write_tsv(parent_rows, outdir / "cryptic_parent_records.tsv", parent_fields)
    write_tsv(parent_core_rows, outdir / "cryptic_parent_core.tsv", parent_fields)
    write_tsv(parent_excluded_rows, outdir / "cryptic_parent_excluded.tsv", parent_fields)
    write_tsv(hla_i_rows, outdir / "cryptic_peptide_core_hla_i.tsv", peptide_fields)
    write_tsv(hla_ii_rows, outdir / "cryptic_peptide_core_hla_ii.tsv", peptide_fields)
    write_tsv(unique_core_peptide_rows, outdir / "cryptic_peptide_core.tsv", peptide_fields)
    write_tsv(core_peptide_rows + excluded_peptide_rows, outdir / "cryptic_peptide_parent_map.tsv", peptide_fields)
    write_tsv(core_peptide_rows + excluded_peptide_rows, outdir / "cryptic_normal_resource_evidence.tsv", peptide_fields)
    if candidate_selection_mode == RANKED_CAP_MODE:
        parent_rank_fields = [
            "sample", "parent_record_id", "source_parent_id", "parent_rank",
            "TPM_tumor", "TPM_ctrl", "log2FC", "nc_class", "parent_sequence_length",
            "ranking_policy", "ranking_digest", "selection_status", "selection_reason",
        ]
        peptide_selection_fields = [
            "sample", "parent_record_id", "source_parent_id", "parent_rank",
            "mhc_class", "peptide", "peptide_length", "selection_status",
            "selection_reason", "selection_digest",
        ]
        write_tsv(parent_rank_rows, outdir / "cryptic_parent_ranked.tsv", parent_rank_fields)
        write_tsv(peptide_selection_evidence_rows, outdir / "cryptic_peptide_selection_evidence.tsv", peptide_selection_fields)
        write_tsv(deferred_parent_rows, outdir / "cryptic_deferred_parent.tsv", parent_rank_fields)
        write_tsv(deferred_peptide_rows, outdir / "cryptic_peptide_deferred.tsv", peptide_fields)
        write_tsv(deferred_peptide_parent_map_rows, outdir / "cryptic_peptide_deferred_parent_map.tsv", peptide_fields)
    if is_v11(policy_version):
        parent_coordinate_fields = [
            "sample", "parent_record_id", "chromosome", "strand", "genomic_start0", "genomic_end0",
            "block_count", "source_blocks_genomic_order", "blocks_transcript_order",
            "primary_alignment_count", "secondary_alignment_count", "supplementary_alignment_count",
            "MAPQ", "cds_nucleotide_length", "block_total_length", "terminal_stop_in_cds",
            "reference_genomic_cds_length", "input_cds_translation_status",
            "reference_genomic_translation_status", "rna_variant_coordinate_status",
            "reference_build", "bed12_alignment_count", "all_alignment_loci",
            "coordinate_mapping_status", "coordinate_mapping_reasons",
        ]
        parent_orfcds_fields = [
            "sample", "parent_record_id", "transcript_block_order", "chromosome", "strand",
            "start0", "end0", "block_length", "derived_phase", "phase_provenance",
            "coordinate_mapping_status",
        ]
        peptide_footprint_fields = [
            "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
            "peptide", "peptide_start", "peptide_length", "chromosome", "strand",
            "peptide_cds_start0", "peptide_cds_end0", "codon_blocks_transcript_order",
            "codon_blocks_text", "peptide_footprint_phase", "candidate_coordinate_status",
            "candidate_coordinate_reasons", "reference_build",
        ]
        write_tsv(parent_coordinate_rows, outdir / "cryptic_parent_coordinates.tsv", parent_coordinate_fields)
        write_tsv(parent_orfcds_rows, outdir / "cryptic_parent_orfcds.tsv", parent_orfcds_fields)
        write_tsv(peptide_footprint_rows, outdir / "cryptic_peptide_genomic_footprint.tsv", peptide_footprint_fields)
    if junction_qc_enabled and junction_result is not None:
        parent_junction_fields = [
            "sample", "parent_record_id", "junction_id", "junction_order", "chromosome", "strand",
            "junction_start0", "junction_end0", "source_cigar", "tumor_unique_reads", "tumor_multi_reads",
            "tumor_unknown_strand_unique_reads", "tumor_max_overhang", "tumor_annotated",
            "normal_unique_reads", "normal_multi_reads", "normal_unknown_strand_unique_reads",
            "normal_max_overhang", "normal_annotated", "primary_support_pass",
        ]
        parent_junction_summary_fields = [
            "sample", "parent_record_id", "primary_core_status", "junction_qc_status", "junction_qc_reasons",
            "rna_variant_rescue_status", "required_junction_count", "tumor_min_required_unique_reads",
            "tumor_all_required_junctions_unique_ge1", "tumor_all_required_junctions_unique_ge2",
            "tumor_all_required_junctions_unique_ge3", "tumor_all_required_junctions_unique_ge5",
            "normal_any_required_junction_unique_ge1", "normal_all_required_junctions_unique_ge1",
            "normal_any_required_junction_unique_ge2", "normal_all_required_junctions_unique_ge2",
            "required_junction_ids", "normal_junction_policy",
        ]
        peptide_junction_fields = [
            "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
            "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
            "crossed_required_junction_ids", "parent_primary_core_status",
            "parent_junction_qc_status", "parent_required_junction_count",
        ]
        write_tsv(
            junction_result["parent_junction_rows"],  # type: ignore[index]
            outdir / "cryptic_parent_junctions.tsv",
            parent_junction_fields,
        )
        write_tsv(
            junction_result["parent_summary_rows"],  # type: ignore[index]
            outdir / "cryptic_parent_junction_summary.tsv",
            parent_junction_summary_fields,
        )
        write_tsv(peptide_junction_rows, outdir / "cryptic_peptide_junction_evidence.tsv", peptide_junction_fields)
        write_tsv(
            junction_result["sensitivity_rows"],  # type: ignore[index]
            outdir / "junction_threshold_sensitivity.tsv",
            [
                "sample", "threshold_tumor_unique_reads", "mapping_eligible_parents", "intronless_parents",
                "spliced_evaluable_parents", "parents_pass", "parents_fail", "spliced_parents_pass",
                "spliced_parents_fail", "total_parent_records",
            ],
        )
        junction_stage_counts = [
            {"stage": "parent_junction_rows", "count": len(junction_result["parent_junction_rows"])},  # type: ignore[arg-type,index]
            {"stage": "peptide_junction_evidence_rows", "count": len(peptide_junction_rows)},
        ] + [
            {"stage": f"parent_status_{status}", "count": count}
            for status, count in sorted(junction_result["stage_counts"].items())  # type: ignore[index,union-attr]
        ]
        write_tsv(junction_stage_counts, outdir / "junction_qc_stagewise.tsv", ["stage", "count"])
        write_json(
            outdir / "junction_qc.manifest.json",
            {
                "policy_version": JUNCTION_POLICY_VERSION,
                "sample": args.sample,
                "run_status": "complete",
                "star_pair_inputs": str(Path(args.star_pair_inputs)),
                "star_pair_validation": {
                    key: value
                    for key, value in (star_pair_validation or {}).items()
                    if key != "row"
                },
                "junction_qc_script_sha256": sha256_file(JUNCTION_QC_PATH),
                "primary_min_tumor_unique_reads": int(getattr(args, "primary_min_tumor_unique_reads", 2)),
                "sensitivity_thresholds": junction_thresholds,
                "tumor_sj_summary": junction_result["tumor_sj_summary"],
                "normal_sj_summary": junction_result["normal_sj_summary"],
                "stage_counts": {row["stage"]: row["count"] for row in junction_stage_counts},
            },
        )

    rna_variant_stage_counts: list[dict[str, object]] = []
    rna_variant_summary_payload: dict[str, object] = {"enabled": rna_variant_qc_enabled}
    if rna_variant_qc_enabled and rna_variant_context is not None:
        rna_event_rows = list(rna_variant_context.get("event_rows", []))  # type: ignore[arg-type]
        rna_parent_summary_rows = list(rna_variant_context.get("parent_summary_rows", []))  # type: ignore[arg-type]
        rna_parent_summary_by_id = {
            str(row.get("parent_record_id", "")): row
            for row in rna_parent_summary_rows
        }
        rna_variant_aa_positions = rna_variant_context.get("variant_aa_positions_by_parent", {})
        if not isinstance(rna_variant_aa_positions, dict):
            raise TypeError("rna_variant_context variant_aa_positions_by_parent must be a dict")
        peptide_rna_variant_rows = annotate_peptide_rna_variant_evidence(
            core_peptide_rows,
            rna_parent_summary_by_id,
            rna_variant_aa_positions,  # type: ignore[arg-type]
        )
        editing_excluded_rows = [
            row for row in rna_event_rows
            if str(row.get("rediportal_status", "")) == "known_rna_editing"
        ]
        threshold_pass_counts = rna_variant_context.get("threshold_parent_pass_counts", defaultdict(int))
        if not isinstance(threshold_pass_counts, defaultdict):
            threshold_pass_counts = defaultdict(int)
        reference_concordant_count = sum(
            1 for row in rna_parent_summary_rows
            if str(row.get("rna_variant_rescue_status", "")) == "reference_concordant"
        )
        variant_required_count = sum(
            1 for row in rna_parent_summary_rows
            if int(row.get("required_variant_count", 0)) > 0
        )
        sensitivity_rows = []
        for threshold in rna_variant_sensitivity_alt_reads:
            variant_supported = int(threshold_pass_counts[int(threshold)])
            sensitivity_rows.append({
                "sample": args.sample,
                "threshold_alt_reads": int(threshold),
                "evaluated_parent_records": len(rna_parent_summary_rows),
                "reference_concordant_parents": reference_concordant_count,
                "variant_required_parents": variant_required_count,
                "variant_supported_parents": variant_supported,
                "primary_core_eligible_parents": reference_concordant_count + variant_supported,
                "known_editing_parents": sum(1 for row in rna_parent_summary_rows if int(row.get("known_editing_count", 0)) > 0),
                "editing_not_evaluated_parents": sum(
                    1 for row in rna_parent_summary_rows if int(row.get("editing_not_evaluated_count", 0)) > 0
                ),
                "insufficient_support_parents": sum(
                    1 for row in rna_parent_summary_rows if int(row.get("insufficient_support_count", 0)) > 0
                ),
                "complex_variant_parents": sum(1 for row in rna_parent_summary_rows if int(row.get("complex_variant_count", 0)) > 0),
            })
        parent_status_counts = defaultdict(int)
        for row in rna_parent_summary_rows:
            parent_status_counts[str(row.get("rna_variant_rescue_status", ""))] += 1
        rna_variant_stage_counts = [
            {"stage": "rna_variant_qc_enabled", "count": 1},
            {"stage": "rna_vcf_events_loaded", "count": int(rna_variant_vcf_summary.get("events", 0))},
            {"stage": "rediportal_records_loaded", "count": int(rediportal_summary.get("records", 0))},
            {"stage": "rna_variant_event_rows", "count": len(rna_event_rows)},
            {"stage": "rna_variant_parent_summary_rows", "count": len(rna_parent_summary_rows)},
            {"stage": "rna_variant_peptide_evidence_rows", "count": len(peptide_rna_variant_rows)},
            {"stage": "rna_editing_excluded_events", "count": len(editing_excluded_rows)},
        ] + [
            {"stage": f"parent_rna_variant_status_{status}", "count": count}
            for status, count in sorted(parent_status_counts.items())
        ]
        write_tsv(rna_event_rows, outdir / "cryptic_rna_variant_events.tsv", RNA_VARIANT_EVENT_FIELDS)
        write_tsv(
            rna_parent_summary_rows,
            outdir / "cryptic_parent_rna_variant_summary.tsv",
            RNA_VARIANT_PARENT_SUMMARY_FIELDS,
        )
        write_tsv(
            peptide_rna_variant_rows,
            outdir / "cryptic_peptide_rna_variant_evidence.tsv",
            RNA_VARIANT_PEPTIDE_EVIDENCE_FIELDS,
        )
        write_tsv(editing_excluded_rows, outdir / "cryptic_rna_editing_excluded.tsv", RNA_VARIANT_EVENT_FIELDS)
        write_tsv(
            sensitivity_rows,
            outdir / "rna_variant_threshold_sensitivity.tsv",
            [
                "sample", "threshold_alt_reads", "evaluated_parent_records",
                "reference_concordant_parents", "variant_required_parents",
                "variant_supported_parents", "primary_core_eligible_parents",
                "known_editing_parents", "editing_not_evaluated_parents", "insufficient_support_parents",
                "complex_variant_parents",
            ],
        )
        write_tsv(rna_variant_stage_counts, outdir / "rna_variant_qc_stagewise.tsv", ["stage", "count"])
        rna_variant_outputs = [
            outdir / "cryptic_rna_variant_events.tsv",
            outdir / "cryptic_parent_rna_variant_summary.tsv",
            outdir / "cryptic_peptide_rna_variant_evidence.tsv",
            outdir / "cryptic_rna_editing_excluded.tsv",
            outdir / "rna_variant_threshold_sensitivity.tsv",
            outdir / "rna_variant_qc_stagewise.tsv",
        ]
        rna_variant_summary_payload = {
            "enabled": True,
            "policy_version": RNA_VARIANT_POLICY_VERSION,
            "run_status": "complete_exploratory" if any(
                reason in {
                    "rediportal_not_evaluated",
                    "legacy_rna_variant_calling_manifest_not_validated",
                    "legacy_rna_variant_vcf_allowed",
                    "legacy_duplicate_vcf_allowed",
                }
                for reason in exploratory_reasons
            ) else "complete",
            "exploratory_reasons": [
                reason
                for reason in exploratory_reasons
                if reason in {
                    "rediportal_not_evaluated",
                    "legacy_rna_variant_calling_manifest_not_validated",
                    "legacy_rna_variant_vcf_allowed",
                    "legacy_duplicate_vcf_allowed",
                }
            ],
            "binding_eligible": not any(
                reason in {
                    "rediportal_not_evaluated",
                    "legacy_rna_variant_calling_manifest_not_validated",
                    "legacy_rna_variant_vcf_allowed",
                    "legacy_duplicate_vcf_allowed",
                }
                for reason in exploratory_reasons
            ),
            "sample": args.sample,
            "vcf_summary": rna_variant_vcf_summary,
            "calling_manifest_summary": rna_variant_calling_manifest_summary,
            "rediportal_summary": rediportal_summary,
            "thresholds": signature["rna_variant_editing_qc"],
            "stage_counts": {row["stage"]: row["count"] for row in rna_variant_stage_counts},
            "input_signature": {
                "rna_variant_editing_qc": signature["rna_variant_editing_qc"],
                "input_paths": {
                    key: value
                    for key, value in input_paths.items()
                    if key in {
                        "rna_variant_vcf",
                        "rna_variant_calling_manifest",
                        "rediportal_processed_table",
                        "rna_variant_qc_script",
                        "coordinate_utils",
                    }
                    or key == "rediportal_resource_manifest"
                },
            },
            "output_signature": output_signature(rna_variant_outputs),
        }
        write_json(outdir / "rna_variant_qc.manifest.json", rna_variant_summary_payload)
    else:
        rna_variant_stage_counts = [{"stage": "rna_variant_qc_enabled", "count": 0}]

    stage_counts = [
        {"stage": "orf_filtered_parent_records", "count": len(parent_rows)},
        {"stage": "parent_core_records", "count": len(parent_core_rows)},
        {"stage": "parent_excluded_records", "count": len(parent_excluded_rows)},
        {"stage": "candidate_peptide_window_rows", "count": len(candidate_peptide_rows)},
        {"stage": "peptide_core_window_rows", "count": len(core_peptide_rows)},
        {"stage": "peptide_excluded_window_rows", "count": len(excluded_peptide_rows)},
        {"stage": "peptide_deferred_window_rows", "count": len(deferred_peptide_rows)},
        {"stage": "unique_peptide_core_records", "count": len(unique_core_peptide_rows)},
        {"stage": "hla_i_unique_peptide_core_records", "count": len(hla_i_rows)},
        {"stage": "hla_ii_unique_peptide_core_records", "count": len(hla_ii_rows)},
        {"stage": "candidate_selection_mode_ranked_cap", "count": 1 if candidate_selection_mode == RANKED_CAP_MODE else 0},
        {"stage": "candidate_selection_max_hla_i_peptides", "count": int(max_hla_i_peptides or 0)},
        {"stage": "candidate_selection_max_hla_ii_peptides", "count": int(max_hla_ii_peptides or 0)},
    ]
    if candidate_selection_mode == RANKED_CAP_MODE:
        stage_counts.extend([
            {
                "stage": "ranked_examined_candidate_peptides",
                "count": int(selection_summary.get("examined_candidate_peptides", 0)),
            },
            {
                "stage": "ranked_selected_hla_i",
                "count": int(selection_summary.get("selected_hla_i", 0)),
            },
            {
                "stage": "ranked_selected_hla_ii",
                "count": int(selection_summary.get("selected_hla_ii", 0)),
            },
            {
                "stage": "ranked_cap_reached_hla_i",
                "count": 1 if selection_summary.get("cap_reached_hla_i") else 0,
            },
            {
                "stage": "ranked_cap_reached_hla_ii",
                "count": 1 if selection_summary.get("cap_reached_hla_ii") else 0,
            },
            {
                "stage": "ranked_deferred_parent_records",
                "count": int(selection_summary.get("deferred_parent_records", 0)),
            },
            {
                "stage": "ranked_deferred_peptide_parent_map_rows",
                "count": int(selection_summary.get("deferred_peptide_parent_map_rows", 0)),
            },
            {
                "stage": "ranked_materialization_complete",
                "count": 1 if selection_summary.get("materialization_coverage") == "complete" else 0,
            },
            {
                "stage": "ranked_exhausted_before_cap_hla_i",
                "count": 1 if selection_summary.get("exhausted_before_cap_hla_i") else 0,
            },
            {
                "stage": "ranked_exhausted_before_cap_hla_ii",
                "count": 1 if selection_summary.get("exhausted_before_cap_hla_ii") else 0,
            },
        ])
    if junction_qc_enabled and junction_result is not None:
        junction_counts = junction_result["stage_counts"]  # type: ignore[index]
        stage_counts.extend([
            {"stage": "junction_qc_enabled", "count": 1},
            {
                "stage": "junction_parent_primary_core_eligible",
                "count": int(junction_counts.get("primary_core_eligible", 0)),  # type: ignore[union-attr]
            },
            {
                "stage": "junction_parent_provisional_low_support",
                "count": int(junction_counts.get("provisional_junction_low_support", 0)),  # type: ignore[union-attr]
            },
            {
                "stage": "junction_parent_provisional_mapping_ambiguous",
                "count": int(junction_counts.get("provisional_mapping_ambiguous", 0)),  # type: ignore[union-attr]
            },
            {
                "stage": "junction_parent_provisional_reference_translation_mismatch",
                "count": int(junction_counts.get("provisional_reference_translation_mismatch", 0)),  # type: ignore[union-attr]
            },
            {
                "stage": "junction_parent_provisional_coordinate_not_evaluable",
                "count": int(junction_counts.get("provisional_coordinate_not_evaluable", 0)),  # type: ignore[union-attr]
            },
        ])
    else:
        stage_counts.append({"stage": "junction_qc_enabled", "count": 0})
    stage_counts.extend(rna_variant_stage_counts)
    if is_v11(policy_version):
        stage_counts.extend([
            {
                "stage": "candidate_parents_coordinate_evaluable",
                "count": int(coordinate_summary.get("parent_coordinate_evaluable", 0)),
            },
            {
                "stage": "candidate_parents_coordinate_not_evaluable",
                "count": int(coordinate_summary.get("parent_coordinate_not_evaluable", 0)),
            },
            {
                "stage": "candidate_parents_reference_translation_pass",
                "count": int(coordinate_summary.get("parent_reference_translation_pass", 0)),
            },
            {
                "stage": "candidate_parents_reference_translation_rna_variant_rescued",
                "count": int(coordinate_summary.get("parent_reference_translation_rna_variant_rescued", 0)),
            },
            {
                "stage": "candidate_parents_reference_translation_mismatch",
                "count": int(coordinate_summary.get("parent_reference_translation_mismatch", 0)),
            },
            {
                "stage": "candidate_parents_reference_translation_not_evaluable",
                "count": int(coordinate_summary.get("parent_reference_translation_not_evaluable", 0)),
            },
            {
                "stage": "peptides_coordinate_evaluable",
                "count": int(coordinate_summary.get("peptide_coordinate_evaluable", 0)),
            },
        ])
    write_tsv(stage_counts, outdir / "stagewise_qc.tsv", ["stage", "count"])

    outputs_hash = output_signature(required_outputs)
    manifest = {
        "sample": args.sample,
        "matched_control_sample": args.matched_control_sample,
        "policy_version": policy_version,
        "run_status": run_status,
        "exploratory_reasons": exploratory_reasons,
        "binding_eligible": binding_eligible,
        "started_at": ts(),
        "finished_at": ts(),
        "code_commit": git_commit(),
        "script_sha256": script_sha256,
        "input_signature": signature,
        "human_reference_evaluation": human_reference_summary["status"],
        "human_reference_summary": human_reference_summary,
        "candidate_selection_policy_version": CANDIDATE_SELECTION_POLICY_VERSION,
        "candidate_selection": selection_summary,
        "candidate_coordinate_summary": coordinate_summary,
        "junction_qc": {
            "enabled": junction_qc_enabled,
            "policy_version": JUNCTION_POLICY_VERSION if junction_qc_enabled else "",
            "junction_qc_sha256": sha256_file(JUNCTION_QC_PATH) if junction_qc_enabled else "",
            "primary_min_tumor_unique_reads": int(getattr(args, "primary_min_tumor_unique_reads", 2)) if junction_qc_enabled else "",
            "sensitivity_thresholds": junction_thresholds if junction_qc_enabled else [],
            "manifest": str(outdir / "junction_qc.manifest.json") if junction_qc_enabled else "",
            "star_pair_validation": {
                key: value
                for key, value in (star_pair_validation or {}).items()
                if key != "row"
            } if junction_qc_enabled else {},
        },
        "rna_variant_editing_qc": rna_variant_summary_payload,
        "outputs": {path.name: str(path) for path in required_outputs},
        "output_signature": outputs_hash,
        "stage_counts": {row["stage"]: row["count"] for row in stage_counts},
        "binding_input_fasta": str(outdir / "cryptic_peptide_core.fasta"),
        "binding_input_mode": "peptide-core",
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
        help="Cryptic Core QC policy version. v1.1 writes candidate coordinate sidecars.",
    )
    parser.add_argument("--matched-control-sample", default="")
    parser.add_argument("--ae-seps-fasta", required=True)
    parser.add_argument("--aeseps-annotation", required=True)
    parser.add_argument("--orf-filtered-fasta", required=True)
    parser.add_argument("--orf-final", required=True)
    parser.add_argument("--orf-bed12", default="")
    parser.add_argument("--orf-bam", default="")
    parser.add_argument("--orf-cds-fasta", default="")
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--human-proteome-fasta", default="")
    parser.add_argument(
        "--allow-missing-human-reference",
        action="store_true",
        help="Allow exploratory Core generation without human reference exact peptide matching.",
    )
    parser.add_argument("--reference-genome-fasta", default="")
    parser.add_argument("--reference-gtf", default="")
    parser.add_argument("--reference-lnc-gtf", default="")
    parser.add_argument("--reference-build", default="GRCh38")
    parser.add_argument("--strandedness", default="")
    parser.add_argument("--min-tpm-tumor", type=float, default=5.0)
    parser.add_argument("--max-tpm-ctrl", type=float, default=0.5)
    parser.add_argument("--min-log2fc", type=float, default=4.0)
    parser.add_argument("--mhc-i-lengths", default="8,9,10,11")
    parser.add_argument("--mhc-ii-lengths", default="13,14,15,16,17")
    parser.add_argument(
        "--candidate-selection-mode",
        choices=[ALL_MODE, RANKED_CAP_MODE],
        default=ALL_MODE,
        help="Use all candidates or rank parent ORFs and cap unique HLA-I/HLA-II peptides before binding.",
    )
    parser.add_argument("--max-hla-i-peptides", type=int, default=None)
    parser.add_argument("--max-hla-ii-peptides", type=int, default=None)
    parser.add_argument("--coordinate-min-mapq", type=int, default=20)
    parser.add_argument("--junction-qc-enabled", action="store_true")
    parser.add_argument("--junction-policy-version", default=JUNCTION_POLICY_VERSION)
    parser.add_argument("--star-pair-inputs", default="")
    parser.add_argument("--primary-min-tumor-unique-reads", type=int, default=2)
    parser.add_argument("--junction-sensitivity-thresholds", default="1,2,3,5")
    parser.add_argument("--rna-variant-editing-qc-enabled", action="store_true")
    parser.add_argument("--rna-variant-qc-policy-version", default=RNA_VARIANT_POLICY_VERSION)
    parser.add_argument("--rna-variant-vcf", default="")
    parser.add_argument("--rna-variant-calling-manifest", default="")
    parser.add_argument("--rediportal-processed-table", default="")
    parser.add_argument("--rediportal-resource-manifest", default="")
    parser.add_argument(
        "--allow-missing-rediportal-resource",
        action="store_true",
        help="Allow exploratory RNA variant QC without a frozen REDIportal processed table.",
    )
    parser.add_argument(
        "--allow-legacy-duplicate-vcf",
        action="store_true",
        help="Allow exploratory RNA variant QC on legacy VCFs with duplicate exact events.",
    )
    parser.add_argument(
        "--allow-legacy-rna-variant-vcf",
        action="store_true",
        help="Allow exploratory RNA variant QC when formal rna.variant_calling.manifest.json is unavailable.",
    )
    parser.add_argument("--rna-variant-min-mapping-quality", type=float, default=RNA_VARIANT_MIN_MAPPING_QUALITY)
    parser.add_argument("--rna-variant-min-base-quality", type=float, default=20.0)
    parser.add_argument("--rna-variant-min-variant-qual", type=float, default=MIN_VARIANT_QUAL)
    parser.add_argument("--rna-variant-min-total-depth", type=int, default=RNA_VARIANT_MIN_TOTAL_DEPTH)
    parser.add_argument("--rna-variant-min-variant-allele-fraction", type=float, default=MIN_VARIANT_ALLELE_FRACTION)
    parser.add_argument("--rna-variant-primary-min-alt-reads", type=int, default=RNA_VARIANT_PRIMARY_MIN_ALT_READS)
    parser.add_argument("--rna-variant-sensitivity-alt-reads", default="2,3,5")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    manifest = build_core(build_parser().parse_args(argv))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
