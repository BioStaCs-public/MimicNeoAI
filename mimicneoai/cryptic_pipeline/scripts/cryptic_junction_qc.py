#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Production junction QC helpers for cryptic parent and peptide Core evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shlex
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from .cryptic_coordinate_utils import GenomicBlock, canonical_chromosome, parse_json_blocks
except ImportError:  # pragma: no cover
    from cryptic_coordinate_utils import GenomicBlock, canonical_chromosome, parse_json_blocks


POLICY_VERSION = "junction_qc_v1.0"
STAR_PROVENANCE_POLICY_VERSION = "star_provenance_freeze_v1.0"
STAR_CRITICAL_OPTIONS = {
    "--outSAMunmapped",
    "--outFilterType",
    "--outSAMattributes",
    "--outFilterMultimapNmax",
    "--outFilterMismatchNmax",
    "--outFilterMismatchNoverLmax",
    "--alignIntronMin",
    "--alignIntronMax",
    "--alignMatesGapMax",
    "--alignSJoverhangMin",
    "--alignSJDBoverhangMin",
    "--sjdbScore",
    "--genomeLoad",
    "--outSAMtype",
    "--quantMode",
    "--outSAMheaderHD",
    "--readFilesCommand",
}
STAR_RUNTIME_OPTIONS = {
    "--runThreadN",
    "--limitBAMsortRAM",
    "--outFileNamePrefix",
    "--readFilesIn",
    "--genomeDir",
}
PRIMARY_MIN_TUMOR_UNIQUE_READS = 2
SENSITIVITY_THRESHOLDS = (1, 2, 3, 5)
PRIMARY_PASS = "primary_core_eligible"
INTRONLESS_PASS = "intronless_not_applicable"
PROVISIONAL_LOW_SUPPORT = "provisional_junction_low_support"
PROVISIONAL_TRANSLATION = "provisional_reference_translation_mismatch"
PROVISIONAL_MAPPING = "provisional_mapping_ambiguous"
PROVISIONAL_COORDINATE = "provisional_coordinate_not_evaluable"
EXCLUDED_INVALID = "excluded_noncanonical_or_invalid_sequence"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def output_signature(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {path.name: file_identity(path) for path in paths}


def outputs_match_signature(signature: dict[str, Any]) -> bool:
    for identity in signature.values():
        if not isinstance(identity, dict):
            return False
        if file_identity(Path(str(identity.get("path", "")))) != identity:
            return False
    return True


def parse_int(value: object, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value)))
    except Exception:
        return default


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_junction_policy(
    *,
    policy_version: str = POLICY_VERSION,
    primary_min_tumor_unique_reads: int = PRIMARY_MIN_TUMOR_UNIQUE_READS,
    sensitivity_thresholds: tuple[int, ...] = SENSITIVITY_THRESHOLDS,
) -> None:
    if policy_version != POLICY_VERSION:
        raise ValueError(f"Unsupported junction QC policy version: {policy_version}")
    if int(primary_min_tumor_unique_reads) != PRIMARY_MIN_TUMOR_UNIQUE_READS:
        raise ValueError(
            f"{POLICY_VERSION} requires primary_min_tumor_unique_reads="
            f"{PRIMARY_MIN_TUMOR_UNIQUE_READS}"
        )
    if tuple(int(v) for v in sensitivity_thresholds) != SENSITIVITY_THRESHOLDS:
        raise ValueError(f"{POLICY_VERSION} requires sensitivity_thresholds={SENSITIVITY_THRESHOLDS}")


def compact(values: Iterable[object]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return ";".join(out)


def parse_cigar_junctions(chrom: str, strand: str, start0: int, cigar: str) -> list[dict[str, Any]]:
    """Return splice junctions from CIGAR N operations only."""
    pos = int(start0)
    junctions: list[dict[str, Any]] = []
    for length_text, op in re.findall(r"(\d+)([MIDNSHP=X])", str(cigar)):
        length = int(length_text)
        if op == "N":
            junctions.append(
                {
                    "chromosome": canonical_chromosome(chrom),
                    "strand": strand,
                    "junction_start0": pos,
                    "junction_end0": pos + length,
                    "cigar_op": "N",
                    "cigar": cigar,
                }
            )
            pos += length
        elif op in {"M", "D", "=", "X"}:
            pos += length
        else:
            continue
    if strand == "-":
        junctions = list(reversed(junctions))
    for idx, row in enumerate(junctions, start=1):
        row["junction_order"] = idx
        row["junction_id"] = junction_id(row["chromosome"], row["junction_start0"], row["junction_end0"], strand)
    return junctions


def parse_alignment_locus(value: object) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if not text:
        return None
    first = text.split(";", 1)[0]
    match = re.match(r"^([^:]+):(\d+)-(\d+):([+-]):MAPQ([^:]*):(.*)$", first)
    if not match:
        return None
    chrom, start, end, strand, mapq, cigar = match.groups()
    return {
        "chromosome": canonical_chromosome(chrom),
        "start0": int(start),
        "end0": int(end),
        "strand": strand,
        "mapq": parse_int(mapq, 0),
        "cigar": cigar,
    }


def junction_id(chrom: object, start0: object, end0: object, strand: object) -> str:
    return f"{canonical_chromosome(chrom)}:{int(start0)}-{int(end0)}:{strand}"


def sj_strand(code: object) -> str:
    text = str(code).strip()
    if text == "1":
        return "+"
    if text == "2":
        return "-"
    return "."


def parse_sj_out(path: Path, *, label: str) -> tuple[dict[tuple[str, int, int, str], dict[str, int]], dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"{label} SJ.out.tab is missing or empty: {path}")
    support: dict[tuple[str, int, int, str], dict[str, int]] = {}
    chromosome_styles: set[str] = set()
    duplicate_keys: list[str] = []
    rows = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                raise ValueError(f"{label} SJ.out.tab line {line_no} has fewer than 9 columns")
            raw_chrom = fields[0].strip()
            chromosome_styles.add("chr" if raw_chrom.lower().startswith("chr") else "bare")
            chrom = canonical_chromosome(raw_chrom)
            if not chrom:
                continue
            start0 = int(fields[1]) - 1
            end0 = int(fields[2])
            strand = sj_strand(fields[3])
            key = (chrom, start0, end0, strand)
            if key in support:
                duplicate_keys.append(junction_id(chrom, start0, end0, strand))
            support[key] = {
                "unique_reads": int(fields[6]),
                "multi_reads": int(fields[7]),
                "max_overhang": int(fields[8]),
                "annotated": int(fields[5]),
                "motif": int(fields[4]),
            }
            rows += 1
    if duplicate_keys:
        raise ValueError(f"{label} SJ.out.tab contains duplicate junction keys: {compact(duplicate_keys[:10])}")
    if len(chromosome_styles) > 1:
        raise ValueError(f"{label} SJ.out.tab mixes chromosome naming styles: {sorted(chromosome_styles)}")
    return support, {
        "path": str(path),
        "rows": rows,
        "sha256": sha256_file(path),
        "chromosome_style": next(iter(chromosome_styles), ""),
    }


def required_junctions_from_parent(row: dict[str, Any]) -> list[dict[str, Any]]:
    primary_count = parse_int(row.get("primary_alignment_count", 0))
    if primary_count != 1:
        return []
    locus = parse_alignment_locus(row.get("all_alignment_loci", ""))
    if not locus:
        return []
    return parse_cigar_junctions(
        str(locus["chromosome"]),
        str(locus["strand"]),
        int(locus["start0"]),
        str(locus["cigar"]),
    )


def support_for(
    support: dict[tuple[str, int, int, str], dict[str, int]],
    junction: dict[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    chrom = canonical_chromosome(junction["chromosome"])
    start0 = int(junction["junction_start0"])
    end0 = int(junction["junction_end0"])
    strand = str(junction["strand"])
    exact = support.get((chrom, start0, end0, strand), {})
    unknown = support.get((chrom, start0, end0, "."), {})
    return exact, unknown


def mapping_primary_status(parent: dict[str, Any], min_mapq: int) -> tuple[str, list[str]]:
    reasons: list[str] = []
    coord_status = str(parent.get("coordinate_mapping_status", ""))
    ref_status = str(parent.get("reference_genomic_translation_status", ""))
    primary_count = parse_int(parent.get("primary_alignment_count", 0))
    secondary_count = parse_int(parent.get("secondary_alignment_count", 0))
    supplementary_count = parse_int(parent.get("supplementary_alignment_count", 0))
    raw_mapq = parent.get("MAPQ", "")
    mapq_missing = str(raw_mapq).strip() == ""
    mapq = parse_int(raw_mapq, -1)
    if primary_count != 1:
        reasons.append("primary_alignment_count_not_one")
    if secondary_count:
        reasons.append("secondary_alignment_present")
    if supplementary_count:
        reasons.append("supplementary_alignment_present")
    if mapq_missing:
        reasons.append("mapq_missing")
    elif mapq < min_mapq:
        reasons.append(f"mapq_below_{min_mapq}")
    if (
        coord_status == "ambiguous_candidate_mapping"
        or primary_count != 1
        or secondary_count
        or supplementary_count
        or mapq_missing
        or mapq < min_mapq
    ):
        return PROVISIONAL_MAPPING, reasons
    if ref_status == "reference_translation_mismatch":
        return PROVISIONAL_TRANSLATION, ["reference_translation_mismatch"]
    if coord_status != "coordinate_evaluable":
        return PROVISIONAL_COORDINATE, reasons or [coord_status or "coordinate_not_evaluable"]
    if ref_status not in {"pass", "rna_variant_rescued"}:
        return PROVISIONAL_COORDINATE, [ref_status or "reference_translation_not_evaluable"]
    return PRIMARY_PASS, []


def evaluate_parent_junctions(
    *,
    sample: str,
    parent_coordinates: list[dict[str, Any]],
    tumor_sj_path: Path,
    normal_sj_path: Path,
    primary_min_tumor_unique_reads: int = PRIMARY_MIN_TUMOR_UNIQUE_READS,
    sensitivity_thresholds: tuple[int, ...] = SENSITIVITY_THRESHOLDS,
    min_mapq: int = 20,
) -> dict[str, Any]:
    validate_junction_policy(
        primary_min_tumor_unique_reads=primary_min_tumor_unique_reads,
        sensitivity_thresholds=sensitivity_thresholds,
    )
    tumor_support, tumor_sj_summary = parse_sj_out(tumor_sj_path, label="tumor")
    normal_support, normal_sj_summary = parse_sj_out(normal_sj_path, label="normal")
    if tumor_sj_summary["chromosome_style"] != normal_sj_summary["chromosome_style"]:
        raise ValueError(
            "Tumor and normal SJ.out.tab chromosome naming styles differ: "
            f"{tumor_sj_summary['chromosome_style']} vs {normal_sj_summary['chromosome_style']}"
        )

    junction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    sensitivity_counts = {
        threshold: {
            "threshold": threshold,
            "mapping_eligible_parents": 0,
            "intronless_parents": 0,
            "spliced_evaluable_parents": 0,
            "parents_pass": 0,
            "parents_fail": 0,
            "spliced_parents_pass": 0,
            "spliced_parents_fail": 0,
        }
        for threshold in sensitivity_thresholds
    }
    status_counts: defaultdict[str, int] = defaultdict(int)

    for parent in parent_coordinates:
        parent_id = str(parent.get("parent_record_id", ""))
        base_status, mapping_reasons = mapping_primary_status(parent, min_mapq)
        required = required_junctions_from_parent(parent) if base_status == PRIMARY_PASS else []
        required_count = len(required)
        tumor_unique_values: list[int] = []
        normal_unique_values: list[int] = []
        parent_reasons = list(mapping_reasons)

        for junction in required:
            tumor_exact, tumor_unknown = support_for(tumor_support, junction)
            normal_exact, normal_unknown = support_for(normal_support, junction)
            tumor_unique = int(tumor_exact.get("unique_reads", 0))
            normal_unique = int(normal_exact.get("unique_reads", 0))
            tumor_unique_values.append(tumor_unique)
            normal_unique_values.append(normal_unique)
            junction_rows.append({
                "sample": sample,
                "parent_record_id": parent_id,
                "junction_id": junction["junction_id"],
                "junction_order": junction["junction_order"],
                "chromosome": junction["chromosome"],
                "strand": junction["strand"],
                "junction_start0": junction["junction_start0"],
                "junction_end0": junction["junction_end0"],
                "source_cigar": junction["cigar"],
                "tumor_unique_reads": tumor_unique,
                "tumor_multi_reads": int(tumor_exact.get("multi_reads", 0)),
                "tumor_unknown_strand_unique_reads": int(tumor_unknown.get("unique_reads", 0)),
                "tumor_max_overhang": int(tumor_exact.get("max_overhang", 0)),
                "tumor_annotated": int(tumor_exact.get("annotated", 0)),
                "normal_unique_reads": normal_unique,
                "normal_multi_reads": int(normal_exact.get("multi_reads", 0)),
                "normal_unknown_strand_unique_reads": int(normal_unknown.get("unique_reads", 0)),
                "normal_max_overhang": int(normal_exact.get("max_overhang", 0)),
                "normal_annotated": int(normal_exact.get("annotated", 0)),
                "primary_support_pass": tumor_unique >= primary_min_tumor_unique_reads,
            })

        if base_status != PRIMARY_PASS:
            primary_status = base_status
            junction_status = base_status
        elif required_count == 0:
            primary_status = PRIMARY_PASS
            junction_status = INTRONLESS_PASS
        elif all(value >= primary_min_tumor_unique_reads for value in tumor_unique_values):
            primary_status = PRIMARY_PASS
            junction_status = "all_required_junctions_tumor_unique_ge2"
        else:
            primary_status = PROVISIONAL_LOW_SUPPORT
            junction_status = PROVISIONAL_LOW_SUPPORT
            parent_reasons.append(f"required_junction_tumor_unique_below_{primary_min_tumor_unique_reads}")

        for threshold, counts in sensitivity_counts.items():
            if base_status == PRIMARY_PASS:
                counts["mapping_eligible_parents"] += 1
            if required_count == 0 and base_status == PRIMARY_PASS:
                counts["intronless_parents"] += 1
                counts["parents_pass"] += 1
            elif required_count and base_status == PRIMARY_PASS and all(value >= threshold for value in tumor_unique_values):
                counts["spliced_evaluable_parents"] += 1
                counts["parents_pass"] += 1
                counts["spliced_parents_pass"] += 1
            elif required_count and base_status == PRIMARY_PASS:
                counts["spliced_evaluable_parents"] += 1
                counts["parents_fail"] += 1
                counts["spliced_parents_fail"] += 1
            else:
                counts["parents_fail"] += 1

        status_counts[primary_status] += 1
        summary_rows.append({
            "sample": sample,
            "parent_record_id": parent_id,
            "primary_core_status": primary_status,
            "junction_qc_status": junction_status,
            "junction_qc_reasons": compact(parent_reasons),
            "rna_variant_rescue_status": parent.get("rna_variant_coordinate_status", "not_evaluated"),
            "required_junction_count": required_count,
            "tumor_min_required_unique_reads": min(tumor_unique_values) if tumor_unique_values else "",
            "tumor_all_required_junctions_unique_ge1": required_count == 0 or all(v >= 1 for v in tumor_unique_values),
            "tumor_all_required_junctions_unique_ge2": required_count == 0 or all(v >= 2 for v in tumor_unique_values),
            "tumor_all_required_junctions_unique_ge3": required_count == 0 or all(v >= 3 for v in tumor_unique_values),
            "tumor_all_required_junctions_unique_ge5": required_count == 0 or all(v >= 5 for v in tumor_unique_values),
            "normal_any_required_junction_unique_ge1": any(v >= 1 for v in normal_unique_values),
            "normal_all_required_junctions_unique_ge1": bool(required_count) and all(v >= 1 for v in normal_unique_values),
            "normal_any_required_junction_unique_ge2": any(v >= 2 for v in normal_unique_values),
            "normal_all_required_junctions_unique_ge2": bool(required_count) and all(v >= 2 for v in normal_unique_values),
            "required_junction_ids": compact(row["junction_id"] for row in required),
            "normal_junction_policy": "annotate_only",
        })

    sensitivity_rows = [
        {
            "sample": sample,
            "threshold_tumor_unique_reads": threshold,
            "mapping_eligible_parents": values["mapping_eligible_parents"],
            "intronless_parents": values["intronless_parents"],
            "spliced_evaluable_parents": values["spliced_evaluable_parents"],
            "parents_pass": values["parents_pass"],
            "parents_fail": values["parents_fail"],
            "spliced_parents_pass": values["spliced_parents_pass"],
            "spliced_parents_fail": values["spliced_parents_fail"],
            "total_parent_records": len(parent_coordinates),
        }
        for threshold, values in sensitivity_counts.items()
    ]
    return {
        "policy_version": POLICY_VERSION,
        "sample": sample,
        "primary_min_tumor_unique_reads": primary_min_tumor_unique_reads,
        "sensitivity_thresholds": list(sensitivity_thresholds),
        "tumor_sj_summary": tumor_sj_summary,
        "normal_sj_summary": normal_sj_summary,
        "parent_junction_rows": junction_rows,
        "parent_summary_rows": summary_rows,
        "sensitivity_rows": sensitivity_rows,
        "stage_counts": dict(status_counts),
    }


def peptide_crossed_junction_ids(peptide_footprint: dict[str, Any]) -> list[str]:
    strand = str(peptide_footprint.get("strand", ""))
    blocks = parse_json_blocks(peptide_footprint.get("codon_blocks_transcript_order", ""))
    if len(blocks) < 2:
        return []
    ids: list[str] = []
    for prev, current in zip(blocks, blocks[1:]):
        if prev.chrom != current.chrom:
            continue
        start0 = min(prev.end0, current.end0)
        end0 = max(prev.start0, current.start0)
        if end0 > start0:
            ids.append(junction_id(prev.chrom, start0, end0, strand))
    return ids


def annotate_peptide_junctions(
    *,
    sample: str,
    peptide_parent_rows: list[dict[str, Any]],
    peptide_footprint_rows: list[dict[str, Any]],
    parent_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_parent = {str(row.get("parent_record_id", "")): row for row in parent_summary_rows}
    footprint_by_key = unique_peptide_row_index(peptide_footprint_rows, "peptide genomic footprint")
    parent_map_by_key = unique_peptide_row_index(peptide_parent_rows, "peptide parent map")
    footprint_keys = set(footprint_by_key)
    parent_map_keys = set(parent_map_by_key)
    if footprint_keys != parent_map_keys:
        missing = sorted(parent_map_keys - footprint_keys)[:5]
        extra = sorted(footprint_keys - parent_map_keys)[:5]
        raise ValueError(
            "peptide parent map and genomic footprint keys do not match: "
            f"missing_footprint={len(parent_map_keys - footprint_keys)} examples={missing}; "
            f"extra_footprint={len(footprint_keys - parent_map_keys)} examples={extra}"
        )
    rows: list[dict[str, Any]] = []
    for peptide in peptide_parent_rows:
        parent_id = str(peptide.get("parent_record_id", ""))
        peptide_id = str(peptide.get("peptide_record_id", ""))
        key = peptide_row_key(peptide)
        footprint = footprint_by_key[key]
        crossed = peptide_crossed_junction_ids(footprint)
        parent_summary = summary_by_parent.get(parent_id, {})
        rows.append({
            "sample": sample,
            "peptide_record_id": peptide_id,
            "parent_record_id": parent_id,
            "source_parent_id": peptide.get("source_parent_id", ""),
            "mhc_class": peptide.get("mhc_class", ""),
            "peptide": peptide.get("peptide", ""),
            "peptide_start": peptide.get("peptide_start", ""),
            "peptide_length": peptide.get("peptide_length", ""),
            "peptide_crosses_junction": bool(crossed),
            "crossed_required_junction_ids": compact(crossed),
            "parent_primary_core_status": parent_summary.get("primary_core_status", ""),
            "parent_junction_qc_status": parent_summary.get("junction_qc_status", ""),
            "parent_required_junction_count": parent_summary.get("required_junction_count", ""),
        })
    return rows


PEPTIDE_KEY_FIELDS = (
    "peptide_record_id",
    "parent_record_id",
    "mhc_class",
    "peptide",
    "peptide_start",
    "peptide_length",
)


def peptide_row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return tuple(str(row.get(field, "")).strip() for field in PEPTIDE_KEY_FIELDS)  # type: ignore[return-value]


def unique_peptide_row_index(
    rows: list[dict[str, Any]],
    label: str,
) -> dict[tuple[str, str, str, str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str, str, str, str], dict[str, Any]] = {}
    duplicate_examples: list[tuple[str, str, str, str, str, str]] = []
    missing_required: list[str] = []
    for idx, row in enumerate(rows, start=1):
        key = peptide_row_key(row)
        if any(part == "" for part in key):
            missing_required.append(str(idx))
            continue
        if key in out:
            duplicate_examples.append(key)
            continue
        out[key] = row
    if missing_required:
        raise ValueError(
            f"{label} rows are missing required peptide key fields {PEPTIDE_KEY_FIELDS}: "
            f"rows {', '.join(missing_required[:10])}"
        )
    if duplicate_examples:
        raise ValueError(f"{label} contains duplicate peptide keys: {duplicate_examples[:5]}")
    return out


def load_star_pair_inputs(path: Path, sample: str) -> dict[str, str]:
    rows = read_tsv(path)
    matches = [row for row in rows if row.get("tumor_sample") == sample]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one STAR pair input row for {sample}, found {len(matches)}")
    return matches[0]


def pair_row_sha256(row: dict[str, str]) -> str:
    normalized = {str(key): str(value) for key, value in sorted(row.items())}
    return sha256_text(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def manifest_output(manifest: dict[str, Any], suffix: str, label: str) -> dict[str, Any]:
    outputs = manifest.get("current_outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError(f"{label} STAR manifest lacks current_outputs")
    record = outputs.get(suffix, {})
    if not isinstance(record, dict):
        raise ValueError(f"{label} STAR manifest lacks current_outputs[{suffix!r}]")
    return record


def manifest_star_index_path(manifest: dict[str, Any]) -> str:
    if isinstance(manifest.get("star_index"), str) and manifest.get("star_index"):
        return str(manifest.get("star_index", ""))
    identity = manifest.get("star_index_identity", {})
    if isinstance(identity, dict):
        path = identity.get("path", "")
        return str(path) if isinstance(path, str) else ""
    return ""


STAR_INDEX_REQUIRED_FILES = ("Genome", "SA", "SAindex", "sjdbList.out.tab")


def star_index_identity_from_path(path: str) -> dict[str, Any]:
    if not path:
        return {}
    root = Path(path)
    return {
        "path": str(root),
        "exists": root.exists(),
        "files": {
            name: {
                "exists": (root / name).exists(),
                "size_bytes": (root / name).stat().st_size if (root / name).exists() else 0,
            }
            for name in STAR_INDEX_REQUIRED_FILES
        },
    }


def manifest_star_index_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    for key in ("star_index_identity", "star_index_record"):
        identity = manifest.get(key, {})
        if isinstance(identity, dict) and isinstance(identity.get("files"), dict):
            return identity
    return star_index_identity_from_path(manifest_star_index_path(manifest))


def star_index_fingerprint(identity: dict[str, Any]) -> str:
    files = identity.get("files", {})
    if not isinstance(files, dict):
        return ""
    records: list[tuple[str, int]] = []
    for name in STAR_INDEX_REQUIRED_FILES:
        record = files.get(name, {})
        if not isinstance(record, dict) or not record.get("exists", True):
            return ""
        try:
            size_bytes = int(record.get("size_bytes", 0))
        except (TypeError, ValueError):
            return ""
        if size_bytes <= 0:
            return ""
        records.append((name, size_bytes))
    return sha256_text(json.dumps(records, sort_keys=True, separators=(",", ":")))


def star_index_compatible(tumor: dict[str, Any], normal: dict[str, Any]) -> bool:
    tumor_path = str(tumor.get("star_index_path", "") or "")
    normal_path = str(normal.get("star_index_path", "") or "")
    if tumor_path and normal_path and tumor_path == normal_path:
        return True
    tumor_fingerprint = str(tumor.get("star_index_fingerprint", "") or "")
    normal_fingerprint = str(normal.get("star_index_fingerprint", "") or "")
    return bool(tumor_fingerprint and normal_fingerprint and tumor_fingerprint == normal_fingerprint)


def parse_star_command_line(log_out: Path) -> list[str]:
    lines = log_out.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "##### Command Line:" and idx + 1 < len(lines):
            command = lines[idx + 1].strip()
            if command.startswith("STAR"):
                return shlex.split(command)[1:]
    for line in lines:
        command = line.strip()
        if command.startswith("STAR "):
            return shlex.split(command)[1:]
    return []


def normalize_star_parameters(params: object) -> list[str]:
    values = [str(v) for v in params] if isinstance(params, list) else []
    normalized: list[str] = []
    idx = 0
    while idx < len(values):
        token = values[idx]
        if token in STAR_RUNTIME_OPTIONS:
            idx += 1
            while idx < len(values) and not values[idx].startswith("--"):
                idx += 1
            continue
        if token in STAR_CRITICAL_OPTIONS:
            normalized.append(token)
            idx += 1
            while idx < len(values) and not values[idx].startswith("--"):
                normalized.append(values[idx])
                idx += 1
            continue
        idx += 1
    return normalized


def star_parameter_digest(params: list[str]) -> str:
    return sha256_text(json.dumps(params, separators=(",", ":")))


def normalize_manifest_critical_parameters(params: object) -> dict[str, list[str]]:
    if not isinstance(params, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, value in params.items():
        option = str(key)
        if option not in STAR_CRITICAL_OPTIONS:
            continue
        if isinstance(value, list):
            out[option] = [str(v) for v in value]
        else:
            out[option] = [str(value)]
    return out


def manifest_critical_parameter_digest(params: dict[str, list[str]]) -> str:
    return sha256_text(json.dumps({key: params[key] for key in sorted(params)}, sort_keys=True))


def validate_one_star_manifest(
    *,
    role: str,
    row: dict[str, str],
    sample: str,
    normal_sample: str,
) -> dict[str, Any]:
    manifest_field = f"{role}_star_manifest"
    expected_status = "frozen_legacy_complete" if role == "tumor" else "relocated_complete"
    expected_sample = sample if role == "tumor" else normal_sample
    manifest_path = Path(str(row.get(manifest_field, "")).strip())
    manifest = read_json(manifest_path)
    if manifest.get("policy_version") != STAR_PROVENANCE_POLICY_VERSION:
        raise ValueError(f"{role} STAR manifest policy mismatch: {manifest_path}")
    if manifest.get("status") != expected_status:
        raise ValueError(f"{role} STAR manifest status mismatch: {manifest.get('status')} != {expected_status}")
    if manifest.get("tumor_sample") != sample or manifest.get("normal_sample") != normal_sample:
        raise ValueError(
            f"{role} STAR manifest sample mismatch: "
            f"{manifest.get('tumor_sample')},{manifest.get('normal_sample')} != {sample},{normal_sample}"
        )
    sj_record = manifest_output(manifest, "SJ.out.tab", role)
    log_record = manifest_output(manifest, "Log.out", role)
    expected_sj_path = str(row.get(f"{role}_sj_path", "")).strip()
    expected_sj_sha = str(row.get(f"{role}_sj_sha256", "")).strip()
    manifest_sj_path = str(sj_record.get("path", "")).strip()
    manifest_sj_sha = str(sj_record.get("sha256", "")).strip()
    if manifest_sj_path != expected_sj_path:
        raise ValueError(f"{role} STAR manifest SJ path does not match pair table")
    if manifest_sj_sha != expected_sj_sha:
        raise ValueError(f"{role} STAR manifest SJ hash does not match pair table")
    actual_sj_sha = sha256_file(Path(expected_sj_path))
    if actual_sj_sha != expected_sj_sha:
        raise ValueError(f"{role} SJ.out.tab hash mismatch for {expected_sample}: {expected_sj_path}")
    manifest_log_path = str(log_record.get("path", "")).strip()
    manifest_log_sha = str(log_record.get("sha256", "")).strip()
    if not manifest_log_path:
        raise ValueError(f"{role} STAR manifest lacks Log.out path")
    actual_log_sha = sha256_file(Path(manifest_log_path))
    if manifest_log_sha and actual_log_sha != manifest_log_sha:
        raise ValueError(f"{role} Log.out hash mismatch for {expected_sample}: {manifest_log_path}")
    expected_digest = str(row.get(f"{role}_critical_parameter_digest", "")).strip()
    manifest_compatibility = str(manifest.get("critical_parameter_compatibility", "")).strip()
    manifest_digest = str(manifest.get("critical_star_parameter_digest", "")).strip()
    critical_parameters = normalize_manifest_critical_parameters(manifest.get("critical_star_parameters", {}))
    if manifest_compatibility != "compatible":
        raise ValueError(f"{role} STAR manifest critical_parameter_compatibility is not compatible")
    if not expected_digest:
        raise ValueError(f"{role} STAR pair table lacks critical parameter digest")
    if not manifest_digest:
        raise ValueError(f"{role} STAR manifest lacks critical_star_parameter_digest")
    if not critical_parameters:
        raise ValueError(f"{role} STAR manifest lacks critical_star_parameters")
    if manifest_digest != expected_digest:
        raise ValueError(f"{role} STAR manifest critical parameter digest does not match pair table")
    if manifest_critical_parameter_digest(critical_parameters) != manifest_digest:
        raise ValueError(f"{role} STAR manifest critical parameters do not match recorded digest")
    star_parameters = normalize_star_parameters(manifest.get("star_parameters", []))
    parameter_source = "manifest"
    if not star_parameters:
        star_parameters = normalize_star_parameters(parse_star_command_line(Path(manifest_log_path)))
        parameter_source = "log_out"
    manifest_hash = sha256_file(manifest_path)
    star_index_identity = manifest_star_index_identity(manifest)
    source_manifest_summary: dict[str, Any] = {}
    if role == "normal":
        source_manifest_path = Path(str(manifest.get("source_manifest_path", "")).strip())
        source_manifest_sha = str(manifest.get("source_manifest_sha256", "")).strip()
        if not source_manifest_path.exists():
            raise FileNotFoundError(f"Normal relocated STAR source manifest is required: {source_manifest_path}")
        actual_source_sha = sha256_file(source_manifest_path)
        if source_manifest_sha and actual_source_sha != source_manifest_sha:
            raise ValueError(f"Normal relocated STAR source manifest hash mismatch: {source_manifest_path}")
        source_manifest = read_json(source_manifest_path)
        source_signature = source_manifest.get("input_signature", {})
        if source_manifest.get("status") != "completed":
            raise ValueError(f"Normal STAR source manifest status is not completed: {source_manifest_path}")
        if not isinstance(source_signature, dict) or source_signature.get("alignment_role") != "control":
            raise ValueError(f"Normal STAR source manifest alignment_role is not control: {source_manifest_path}")
        if source_signature.get("sample") != normal_sample or source_signature.get("control_sample") != normal_sample:
            raise ValueError(f"Normal STAR source manifest sample/control_sample mismatch: {source_manifest_path}")
        if source_signature.get("tumor_sample") != sample:
            raise ValueError(f"Normal STAR source manifest tumor_sample mismatch: {source_manifest_path}")
        source_manifest_summary = {
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": actual_source_sha,
            "source_manifest_status": source_manifest.get("status", ""),
            "source_alignment_role": source_signature.get("alignment_role", ""),
            "source_sample": source_signature.get("sample", ""),
            "source_tumor_sample": source_signature.get("tumor_sample", ""),
            "source_control_sample": source_signature.get("control_sample", ""),
        }
    return {
        "role": role,
        "sample": expected_sample,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_status": manifest.get("status", ""),
        "expected_sj_path": expected_sj_path,
        "actual_sj_path": expected_sj_path,
        "expected_sj_sha256": expected_sj_sha,
        "manifest_sj_sha256": manifest_sj_sha,
        "actual_sj_sha256": actual_sj_sha,
        "sj_hash_match": True,
        "sj_size_bytes": sj_record.get("size_bytes", ""),
        "star_version": str(manifest.get("star_version", "")),
        "star_index_path": manifest_star_index_path(manifest),
        "star_index_fingerprint": star_index_fingerprint(star_index_identity),
        "star_index_identity_status": str(manifest.get("star_index_identity_status", "")),
        "log_out_path": manifest_log_path,
        "expected_log_sha256": manifest_log_sha,
        "actual_log_sha256": actual_log_sha,
        "log_hash_match": True,
        "star_parameters_normalized": star_parameters,
        "star_parameter_source": parameter_source,
        "critical_star_parameters": critical_parameters,
        "critical_star_parameter_digest": manifest_digest,
        "computed_critical_star_parameter_digest": manifest_critical_parameter_digest(critical_parameters),
        **source_manifest_summary,
    }


def validate_star_pair_contract(path: Path, sample: str) -> dict[str, Any]:
    pair = load_star_pair_inputs(path, sample)
    required = [
        "tumor_sample",
        "normal_sample",
        "tumor_sj_path",
        "normal_sj_path",
        "tumor_sj_sha256",
        "normal_sj_sha256",
        "tumor_star_manifest",
        "normal_star_manifest",
        "critical_parameter_compatibility",
        "tumor_critical_parameter_digest",
        "normal_critical_parameter_digest",
    ]
    missing = [field for field in required if not str(pair.get(field, "")).strip()]
    if missing:
        raise ValueError(f"STAR pair input row for {sample} is missing required fields: {', '.join(missing)}")
    if str(pair.get("critical_parameter_compatibility", "")).strip() != "compatible":
        raise ValueError(f"STAR pair input row for {sample} does not have compatible critical STAR parameters")
    if str(pair.get("tumor_critical_parameter_digest", "")).strip() != str(pair.get("normal_critical_parameter_digest", "")).strip():
        raise ValueError(f"STAR pair input row for {sample} has mismatched critical STAR parameter digests")
    normal_sample = str(pair.get("normal_sample", "")).strip()
    tumor = validate_one_star_manifest(role="tumor", row=pair, sample=sample, normal_sample=normal_sample)
    normal = validate_one_star_manifest(role="normal", row=pair, sample=sample, normal_sample=normal_sample)
    if tumor["star_version"] and normal["star_version"] and tumor["star_version"] != normal["star_version"]:
        raise ValueError(f"Tumor and normal STAR versions differ for {sample}: {tumor['star_version']} vs {normal['star_version']}")
    if not star_index_compatible(tumor, normal):
        raise ValueError(f"Tumor and normal STAR index identity differs for {sample}")
    if not tumor["star_parameters_normalized"] or not normal["star_parameters_normalized"]:
        raise ValueError(f"Tumor and normal STAR critical parameters could not be parsed for {sample}")
    if tumor["critical_star_parameter_digest"] != normal["critical_star_parameter_digest"]:
        raise ValueError(f"Tumor and normal STAR biological parameters differ for {sample}")
    return {
        "status": "validated",
        "policy_version": STAR_PROVENANCE_POLICY_VERSION,
        "sample": sample,
        "normal_sample": normal_sample,
        "pair_table": str(path),
        "pair_table_sha256": sha256_file(path),
        "pair_row_sha256": pair_row_sha256(pair),
        "row": pair,
        "tumor": tumor,
        "normal": normal,
        "star_version_match": tumor["star_version"] == normal["star_version"],
        "star_index_compatibility": "compatible",
        "critical_parameter_compatibility": "compatible",
        "tumor_sj_path": pair["tumor_sj_path"],
        "normal_sj_path": pair["normal_sj_path"],
    }


def build_junction_qc(args: argparse.Namespace) -> dict[str, Any]:
    thresholds = tuple(int(v) for v in str(args.sensitivity_thresholds).split(",") if v)
    validate_junction_policy(
        primary_min_tumor_unique_reads=int(args.primary_min_tumor_unique_reads),
        sensitivity_thresholds=thresholds,
    )
    parent_coordinates_path = Path(args.parent_coordinates)
    star_pair_inputs_path = Path(args.star_pair_inputs)
    peptide_parent_map_path = Path(args.peptide_parent_map) if args.peptide_parent_map else None
    peptide_genomic_footprint_path = Path(args.peptide_genomic_footprint) if args.peptide_genomic_footprint else None
    if bool(peptide_parent_map_path) != bool(peptide_genomic_footprint_path):
        raise ValueError("--peptide-parent-map and --peptide-genomic-footprint must be provided together")
    star_pair_validation = validate_star_pair_contract(star_pair_inputs_path, args.sample)
    input_paths: dict[str, Any] = {
        "parent_coordinates": file_identity(parent_coordinates_path),
        "star_pair_inputs": file_identity(star_pair_inputs_path),
    }
    if peptide_parent_map_path:
        input_paths["peptide_parent_map"] = file_identity(peptide_parent_map_path)
    if peptide_genomic_footprint_path:
        input_paths["peptide_genomic_footprint"] = file_identity(peptide_genomic_footprint_path)
    input_signature = {
        "policy_version": POLICY_VERSION,
        "sample": args.sample,
        "junction_qc_script_sha256": sha256_file(Path(__file__)),
        "primary_min_tumor_unique_reads": int(args.primary_min_tumor_unique_reads),
        "sensitivity_thresholds": list(thresholds),
        "min_mapq": int(args.min_mapq),
        "star_pair_validation": {
            key: value
            for key, value in star_pair_validation.items()
            if key != "row"
        },
        "input_paths": input_paths,
    }
    outdir = Path(args.outdir)
    manifest_path = outdir / "junction_qc.manifest.json"
    required_outputs = [
        outdir / "cryptic_parent_junctions.tsv",
        outdir / "cryptic_parent_junction_summary.tsv",
        outdir / "junction_threshold_sensitivity.tsv",
        outdir / "junction_qc_stagewise.tsv",
    ]
    if peptide_parent_map_path and peptide_genomic_footprint_path:
        required_outputs.append(outdir / "cryptic_peptide_junction_evidence.tsv")
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("input_signature") != input_signature:
            raise ValueError(f"Existing junction QC manifest does not match current inputs/config: {manifest_path}")
        old_outputs = old.get("output_signature")
        if isinstance(old_outputs, dict) and outputs_match_signature(old_outputs):
            return {**old, "reused": True}

    parent_coordinates = read_tsv(parent_coordinates_path)
    pair = star_pair_validation["row"]
    result = evaluate_parent_junctions(
        sample=args.sample,
        parent_coordinates=parent_coordinates,
        tumor_sj_path=Path(pair["tumor_sj_path"]),
        normal_sj_path=Path(pair["normal_sj_path"]),
        primary_min_tumor_unique_reads=int(args.primary_min_tumor_unique_reads),
        sensitivity_thresholds=thresholds,
        min_mapq=int(args.min_mapq),
    )
    outdir.mkdir(parents=True, exist_ok=True)
    parent_junction_fields = [
        "sample", "parent_record_id", "junction_id", "junction_order", "chromosome", "strand",
        "junction_start0", "junction_end0", "source_cigar", "tumor_unique_reads", "tumor_multi_reads",
        "tumor_unknown_strand_unique_reads", "tumor_max_overhang", "tumor_annotated",
        "normal_unique_reads", "normal_multi_reads", "normal_unknown_strand_unique_reads",
        "normal_max_overhang", "normal_annotated", "primary_support_pass",
    ]
    parent_summary_fields = [
        "sample", "parent_record_id", "primary_core_status", "junction_qc_status", "junction_qc_reasons",
        "rna_variant_rescue_status", "required_junction_count", "tumor_min_required_unique_reads",
        "tumor_all_required_junctions_unique_ge1", "tumor_all_required_junctions_unique_ge2",
        "tumor_all_required_junctions_unique_ge3", "tumor_all_required_junctions_unique_ge5",
        "normal_any_required_junction_unique_ge1", "normal_all_required_junctions_unique_ge1",
        "normal_any_required_junction_unique_ge2", "normal_all_required_junctions_unique_ge2",
        "required_junction_ids", "normal_junction_policy",
    ]
    write_tsv(outdir / "cryptic_parent_junctions.tsv", result["parent_junction_rows"], parent_junction_fields)
    write_tsv(outdir / "cryptic_parent_junction_summary.tsv", result["parent_summary_rows"], parent_summary_fields)
    write_tsv(
        outdir / "junction_threshold_sensitivity.tsv",
        result["sensitivity_rows"],
        [
            "sample", "threshold_tumor_unique_reads", "mapping_eligible_parents", "intronless_parents",
            "spliced_evaluable_parents", "parents_pass", "parents_fail", "spliced_parents_pass",
            "spliced_parents_fail", "total_parent_records",
        ],
    )

    peptide_rows: list[dict[str, Any]] = []
    if peptide_parent_map_path and peptide_genomic_footprint_path:
        peptide_rows = annotate_peptide_junctions(
            sample=args.sample,
            peptide_parent_rows=read_tsv(peptide_parent_map_path),
            peptide_footprint_rows=read_tsv(peptide_genomic_footprint_path),
            parent_summary_rows=result["parent_summary_rows"],
        )
        write_tsv(
            outdir / "cryptic_peptide_junction_evidence.tsv",
            peptide_rows,
            [
                "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
                "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
                "crossed_required_junction_ids", "parent_primary_core_status",
                "parent_junction_qc_status", "parent_required_junction_count",
            ],
        )

    stage_rows = [
        {"stage": "parent_records", "count": len(parent_coordinates)},
        {"stage": "parent_junction_rows", "count": len(result["parent_junction_rows"])},
        {"stage": "peptide_junction_evidence_rows", "count": len(peptide_rows)},
    ] + [
        {"stage": f"parent_status_{status}", "count": count}
        for status, count in sorted(result["stage_counts"].items())
    ]
    write_tsv(outdir / "junction_qc_stagewise.tsv", stage_rows, ["stage", "count"])
    outputs_hash = output_signature(required_outputs)
    manifest = {
        "policy_version": POLICY_VERSION,
        "sample": args.sample,
        "run_status": "complete",
        "star_pair_inputs": str(star_pair_inputs_path),
        "star_pair_validation": {
            key: value
            for key, value in star_pair_validation.items()
            if key != "row"
        },
        "junction_qc_script_sha256": sha256_file(Path(__file__)),
        "primary_min_tumor_unique_reads": int(args.primary_min_tumor_unique_reads),
        "sensitivity_thresholds": result["sensitivity_thresholds"],
        "min_mapq": int(args.min_mapq),
        "input_signature": input_signature,
        "tumor_sj_summary": result["tumor_sj_summary"],
        "normal_sj_summary": result["normal_sj_summary"],
        "stage_counts": {row["stage"]: row["count"] for row in stage_rows},
        "outputs": {path.name: str(path) for path in required_outputs},
        "output_signature": outputs_hash,
        "reused": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("--parent-coordinates", required=True)
    parser.add_argument("--star-pair-inputs", required=True)
    parser.add_argument("--peptide-parent-map", default="")
    parser.add_argument("--peptide-genomic-footprint", default="")
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--primary-min-tumor-unique-reads", type=int, default=PRIMARY_MIN_TUMOR_UNIQUE_READS)
    parser.add_argument("--sensitivity-thresholds", default="1,2,3,5")
    parser.add_argument("--min-mapq", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    manifest = build_junction_qc(build_parser().parse_args(argv))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
