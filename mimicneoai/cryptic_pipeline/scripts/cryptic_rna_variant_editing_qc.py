#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""RNA-variant and RNA-editing QC helpers for cryptic Core v1.0.

This module evaluates RNA read support for sequence differences between the
candidate cryptic CDS and GRCh38. It does not make somatic/WES claims.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from .cryptic_coordinate_utils import (
        GenomicBlock,
        map_transcript_interval_to_genome,
        normalize_chromosome,
        reverse_complement,
        translate_cds,
    )
except ImportError:  # pragma: no cover
    from cryptic_coordinate_utils import (  # type: ignore
        GenomicBlock,
        map_transcript_interval_to_genome,
        normalize_chromosome,
        reverse_complement,
        translate_cds,
    )


POLICY_VERSION = "cryptic_rna_variant_editing_qc_v1.0"
REDIPORTAL_RESOURCE_POLICY_VERSION = "rediportal_processed_resource_v1.0"
RNA_VARIANT_CALLING_POLICY_VERSION = "cryptic_known_branch_rna_variant_calling_v1.0"
PRIMARY_MIN_ALT_READS = 3
SENSITIVITY_ALT_READS = (2, 3, 5)
MIN_VARIANT_QUAL = 30.0
MIN_TOTAL_DEPTH = 10
MIN_VARIANT_ALLELE_FRACTION = 0.05
MIN_MAPPING_QUALITY = 20.0
MIN_READ_MAPPING_QUALITY = 20
MIN_BASE_QUALITY = 20
MPILEUP_FLAG_FILTER = "0xF04"
NORMALIZATION_POLICY = "bcftools norm -f REF -m -any; bcftools norm -d exact"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _is_missing_manifest_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict, tuple, set)) and not value:
        return True
    return False


def _require_manifest_fields(manifest: dict[str, Any], required: Iterable[str], label: str) -> None:
    missing = [field for field in required if _is_missing_manifest_value(manifest.get(field))]
    if missing:
        raise ValueError(f"{label} missing required fields: {', '.join(sorted(missing))}")


def _load_json_manifest(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _identity_has_required_fields(identity: Any, label: str) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError(f"{label} identity must be a mapping")
    required = ["path", "exists", "size", "sha256"]
    missing = [field for field in required if field not in identity or _is_missing_manifest_value(identity.get(field))]
    if missing:
        raise ValueError(f"{label} identity missing required fields: {', '.join(sorted(missing))}")
    if identity.get("exists") is not True:
        raise ValueError(f"{label} identity is not marked as existing")
    return identity


def _validate_identity_matches_path(identity: Any, actual_path: Path, label: str) -> dict[str, Any]:
    identity = _identity_has_required_fields(identity, label)
    stored_path = Path(str(identity["path"]))
    if stored_path.resolve() != actual_path.resolve():
        raise ValueError(f"{label} path mismatch: expected {stored_path}, observed {actual_path}")
    observed = file_identity(actual_path)
    if observed.get("sha256") != identity.get("sha256"):
        raise ValueError(
            f"{label} SHA256 mismatch: expected {identity.get('sha256')}, observed {observed.get('sha256')}"
        )
    if int(observed.get("size", -1)) != int(identity.get("size", -2)):
        raise ValueError(
            f"{label} size mismatch: expected {identity.get('size')}, observed {observed.get('size')}"
        )
    return identity


def _digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in str(info).split(";"):
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
        else:
            out[token] = "1"
    return out


def parse_number(value: object, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        if not text or text == ".":
            return default
        return float(text)
    except Exception:
        return default


def parse_int(value: object, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text or text == ".":
            return default
        return int(float(text))
    except Exception:
        return default


def sample_name_matches(vcf_sample_name: str, expected_sample: str) -> bool:
    if vcf_sample_name == expected_sample:
        return True
    name = Path(vcf_sample_name).name
    accepted = {
        expected_sample,
        f"{expected_sample}.bam",
        f"{expected_sample}.Aligned.out.sorted.bam",
        f"{expected_sample}.sorted.bam",
    }
    return name in accepted or name.startswith(f"{expected_sample}.")


@dataclass(frozen=True)
class RnaVariantEvent:
    sample: str
    chrom: str
    pos_1based: int
    ref: str
    alt: str
    variant_type: str
    qual: str
    ref_reads: int
    alt_reads: int
    total_depth: int
    vaf: float
    mapping_quality: str
    vcf_filter: str
    vcf_record_id: str
    vcf_duplicate_status: str = "unique"

    @property
    def key(self) -> tuple[str, int, str, str]:
        return (self.chrom, self.pos_1based, self.ref, self.alt)


def _event_key_text(event: RnaVariantEvent) -> str:
    return f"{event.chrom}:{event.pos_1based}:{event.ref}>{event.alt}"


def _event_evidence_tuple(event: RnaVariantEvent) -> tuple[object, ...]:
    return (
        event.qual,
        event.ref_reads,
        event.alt_reads,
        event.total_depth,
        round(event.vaf, 12),
        event.mapping_quality,
        event.vcf_filter,
    )


def _min_numeric_text(left: str, right: str) -> str:
    left_num = parse_number(left, None)
    right_num = parse_number(right, None)
    if left_num is None or right_num is None:
        return "."
    value = min(left_num, right_num)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def _merge_duplicate_exact_event(
    existing: RnaVariantEvent,
    incoming: RnaVariantEvent,
    *,
    has_conflicting_evidence: bool,
) -> RnaVariantEvent:
    """Collapse duplicate exact events using conservative evidence values."""
    ref_reads = max(existing.ref_reads, incoming.ref_reads)
    alt_reads = min(existing.alt_reads, incoming.alt_reads)
    total_depth = ref_reads + alt_reads
    vaf = (alt_reads / total_depth) if total_depth else 0.0
    if existing.vcf_filter == incoming.vcf_filter:
        vcf_filter = existing.vcf_filter
    elif "PASS" in {existing.vcf_filter, incoming.vcf_filter}:
        vcf_filter = "mixed_filter_status"
    else:
        vcf_filter = f"{existing.vcf_filter};{incoming.vcf_filter}"
    if existing.vcf_record_id == incoming.vcf_record_id:
        record_id = existing.vcf_record_id
    else:
        record_id = "collapsed_duplicate_exact_event"
    if has_conflicting_evidence or existing.vcf_duplicate_status == "conflicting_duplicate_not_evaluable":
        duplicate_status = "conflicting_duplicate_not_evaluable"
    else:
        duplicate_status = "collapsed_duplicate_exact_event"
    return RnaVariantEvent(
        sample=existing.sample,
        chrom=existing.chrom,
        pos_1based=existing.pos_1based,
        ref=existing.ref,
        alt=existing.alt,
        variant_type=existing.variant_type,
        qual=_min_numeric_text(existing.qual, incoming.qual),
        ref_reads=ref_reads,
        alt_reads=alt_reads,
        total_depth=total_depth,
        vaf=vaf,
        mapping_quality=_min_numeric_text(existing.mapping_quality, incoming.mapping_quality),
        vcf_filter=vcf_filter,
        vcf_record_id=record_id,
        vcf_duplicate_status=duplicate_status,
    )


def variant_type(ref: str, alt: str) -> str:
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    if len(ref) == len(alt):
        return "MNV"
    return "INDEL_OR_COMPLEX"


def load_vcf_events(
    vcf_path: Path,
    *,
    expected_sample: str,
    require_mq: bool = True,
    allow_legacy_duplicate_vcf: bool = False,
) -> tuple[dict[tuple[str, int, str, str], RnaVariantEvent], dict[str, Any]]:
    if not vcf_path.exists():
        raise FileNotFoundError(vcf_path)
    events: dict[tuple[str, int, str, str], RnaVariantEvent] = {}
    format_defs: set[str] = set()
    info_defs: set[str] = set()
    sample_name = ""
    records = 0
    duplicate_exact_event_rows = 0
    duplicate_exact_event_keys: set[str] = set()
    conflicting_duplicate_exact_event_keys: set[str] = set()
    multiallelic_records = 0
    snv_records = 0
    complex_records = 0

    with open_text(vcf_path) as handle:
        for line in handle:
            if line.startswith("##FORMAT="):
                if "ID=" in line:
                    format_defs.add(line.split("ID=", 1)[1].split(",", 1)[0])
                continue
            if line.startswith("##INFO="):
                if "ID=" in line:
                    info_defs.add(line.split("ID=", 1)[1].split(",", 1)[0])
                continue
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 10:
                    raise ValueError(f"VCF has no sample column: {vcf_path}")
                sample_name = cols[9]
                if not sample_name_matches(sample_name, expected_sample):
                    raise ValueError(
                        f"VCF sample identity mismatch: expected {expected_sample}, observed {sample_name}"
                    )
                continue
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            records += 1
            chrom, pos, rec_id, ref, alt_field, qual, filt, info_text, fmt, sample_value = fields[:10]
            chrom_norm = normalize_chromosome(chrom)
            alts = alt_field.split(",")
            if len(alts) > 1:
                multiallelic_records += 1
            fmt_keys = fmt.split(":")
            sample_values = sample_value.split(":")
            sample_map = dict(zip(fmt_keys, sample_values))
            if "AD" not in sample_map or "DP" not in sample_map:
                raise ValueError(f"VCF record lacks FORMAT/AD or FORMAT/DP: {vcf_path}:{chrom}:{pos}")
            info = parse_info(info_text)
            if require_mq and "MQ" not in info:
                raise ValueError(f"VCF record lacks INFO/MQ: {vcf_path}:{chrom}:{pos}")
            ad_values = [parse_int(value, 0) for value in sample_map.get("AD", "").split(",")]
            ref_reads = ad_values[0] if ad_values else 0
            for alt_index, alt in enumerate(alts, start=1):
                vt = variant_type(ref, alt)
                if vt == "SNV":
                    snv_records += 1
                else:
                    complex_records += 1
                alt_reads = ad_values[alt_index] if alt_index < len(ad_values) else 0
                total_depth = ref_reads + alt_reads
                vaf = (alt_reads / total_depth) if total_depth else 0.0
                event = RnaVariantEvent(
                    sample=expected_sample,
                    chrom=chrom_norm,
                    pos_1based=parse_int(pos),
                    ref=ref.upper(),
                    alt=alt.upper(),
                    variant_type=vt,
                    qual=qual,
                    ref_reads=ref_reads,
                    alt_reads=alt_reads,
                    total_depth=total_depth,
                    vaf=vaf,
                    mapping_quality=str(info.get("MQ", "")),
                    vcf_filter=filt,
                    vcf_record_id=rec_id if rec_id != "." else f"{chrom_norm}:{pos}:{ref}>{alt}",
                )
                if event.key in events:
                    key_text = _event_key_text(event)
                    if not allow_legacy_duplicate_vcf:
                        raise ValueError(
                            "Duplicate exact VCF event keys are not allowed in formal RNA variant QC: "
                            f"{key_text}"
                        )
                    duplicate_exact_event_rows += 1
                    duplicate_exact_event_keys.add(key_text)
                    existing = events[event.key]
                    has_conflict = _event_evidence_tuple(existing) != _event_evidence_tuple(event)
                    if has_conflict:
                        conflicting_duplicate_exact_event_keys.add(key_text)
                    events[event.key] = _merge_duplicate_exact_event(
                        existing,
                        event,
                        has_conflicting_evidence=has_conflict,
                    )
                    continue
                events[event.key] = event
    if not sample_name:
        raise ValueError(f"VCF header sample name not found: {vcf_path}")
    summary = {
        "path": str(vcf_path),
        "sha256": sha256_file(vcf_path),
        "sample_name": sample_name,
        "sample_identity_status": "matched",
        "records": records,
        "events": len(events),
        "format_defs": sorted(format_defs),
        "info_defs": sorted(info_defs),
        "has_FORMAT_AD": "AD" in format_defs,
        "has_FORMAT_DP": "DP" in format_defs,
        "has_INFO_MQ": "MQ" in info_defs,
        "multiallelic_records": multiallelic_records,
        "duplicate_exact_event_keys": len(duplicate_exact_event_keys),
        "duplicate_exact_event_rows_collapsed": duplicate_exact_event_rows,
        "conflicting_duplicate_exact_event_keys_collapsed": len(conflicting_duplicate_exact_event_keys),
        "allow_legacy_duplicate_vcf": bool(allow_legacy_duplicate_vcf),
        "duplicate_exact_event_deduplication_policy": (
            "collapse exact genomic REF/ALT duplicates conservatively: "
            "min QUAL/MQ, min ALT reads, max REF reads; conflicting duplicate events are not evaluable"
        ),
        "snv_events": snv_records,
        "complex_events": complex_records,
    }
    return events, summary


def load_rediportal_events(path: Path, manifest_path: Path | None = None) -> tuple[set[tuple[str, int, str, str]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    manifest: dict[str, Any] = {}
    if manifest_path is not None:
        manifest = _load_json_manifest(manifest_path, "REDIportal resource manifest")
        required = [
            "policy_version",
            "reference_build",
            "processed_table_sha256",
            "processed_table_size_bytes",
            "records",
            "normalization_rules",
            "processing_script_sha256",
        ]
        _require_manifest_fields(manifest, required, "REDIportal resource manifest")
        policy = str(manifest["policy_version"])
        if policy != REDIPORTAL_RESOURCE_POLICY_VERSION:
            raise ValueError(
                f"Unsupported REDIportal resource policy_version: {policy}; "
                f"expected {REDIPORTAL_RESOURCE_POLICY_VERSION}"
            )
        build = str(manifest["reference_build"])
        if build != "GRCh38":
            raise ValueError(f"REDIportal resource manifest is not GRCh38: {build}")
        expected_sha = str(manifest["processed_table_sha256"])
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"REDIportal processed table SHA256 mismatch: expected {expected_sha}, observed {actual_sha}"
            )
        expected_size = int(manifest["processed_table_size_bytes"])
        if expected_size != path.stat().st_size:
            raise ValueError(
                f"REDIportal processed table size mismatch: expected {expected_size}, observed {path.stat().st_size}"
            )
    events: set[tuple[str, int, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chrom", "pos_1based", "ref", "alt"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"REDIportal processed table lacks required fields {sorted(required)}: {path}")
        for row in reader:
            chrom = normalize_chromosome(row.get("chrom", ""))
            pos = parse_int(row.get("pos_1based", ""))
            ref = str(row.get("ref", "")).upper()
            alt = str(row.get("alt", "")).upper()
            if not chrom or not pos or len(ref) != 1 or len(alt) != 1:
                raise ValueError(f"Invalid REDIportal processed row: {row}")
            events.add((chrom, pos, ref, alt))
    expected_records = manifest.get("records")
    if expected_records not in (None, "") and int(expected_records) != len(events):
        raise ValueError(f"REDIportal record count mismatch: expected {expected_records}, observed {len(events)}")
    return events, {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_path": str(manifest_path) if manifest_path is not None else "",
        "manifest_sha256": sha256_file(manifest_path) if manifest_path is not None else "",
        "resource_policy_version": manifest.get("policy_version", ""),
        "reference_build": manifest.get("reference_build", "GRCh38" if manifest_path is None else ""),
        "normalization_rules": manifest.get("normalization_rules", ""),
        "processing_script_sha256": manifest.get("processing_script_sha256", ""),
        "records": len(events),
        "status": "evaluated",
    }


def validate_rna_variant_calling_manifest(
    manifest_path: Path,
    *,
    vcf_path: Path,
    expected_sample: str,
) -> dict[str, Any]:
    """Validate the formal known-branch RNA variant calling manifest consumed by 08b."""
    manifest = _load_json_manifest(manifest_path, "RNA variant calling manifest")
    required = [
        "policy_version",
        "sample",
        "input_signature",
        "output_signature",
        "filter_stats",
        "run_status",
    ]
    _require_manifest_fields(manifest, required, "RNA variant calling manifest")
    policy = str(manifest["policy_version"])
    if policy != RNA_VARIANT_CALLING_POLICY_VERSION:
        raise ValueError(
            f"Unsupported RNA variant calling manifest policy_version: {policy}; "
            f"expected {RNA_VARIANT_CALLING_POLICY_VERSION}"
        )
    sample = str(manifest["sample"])
    if sample != expected_sample:
        raise ValueError(
            f"RNA variant calling manifest sample mismatch: expected {expected_sample}, observed {sample}"
        )
    if str(manifest.get("run_status")) != "complete":
        raise ValueError(f"RNA variant calling manifest run_status is not complete: {manifest.get('run_status')}")

    input_signature = manifest["input_signature"]
    if not isinstance(input_signature, dict):
        raise ValueError("RNA variant calling manifest input_signature must be a mapping")
    _require_manifest_fields(
        input_signature,
        [
            "sample",
            "sample_bam",
            "reference_fasta",
            "exon_bed",
            "parameters",
            "bcftools_version",
            "calling_script_sha256",
        ],
        "RNA variant calling input_signature",
    )
    if str(input_signature.get("sample")) != expected_sample:
        raise ValueError(
            "RNA variant calling input_signature sample mismatch: "
            f"expected {expected_sample}, observed {input_signature.get('sample')}"
        )
    _identity_has_required_fields(input_signature.get("sample_bam"), "RNA variant calling sample_bam")
    _identity_has_required_fields(input_signature.get("reference_fasta"), "RNA variant calling reference_fasta")
    _identity_has_required_fields(input_signature.get("exon_bed"), "RNA variant calling exon_bed")

    parameters = input_signature["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("RNA variant calling input_signature parameters must be a mapping")
    expected_parameters = {
        "mpileup_flag_filter": MPILEUP_FLAG_FILTER,
        "read_mapq_min": MIN_READ_MAPPING_QUALITY,
        "base_quality_min": MIN_BASE_QUALITY,
        "variant_qual_min": int(MIN_VARIANT_QUAL),
        "total_depth_min_ad_derived": MIN_TOTAL_DEPTH,
        "variant_mq_min": int(MIN_MAPPING_QUALITY),
        "vaf_min_ad_derived": MIN_VARIANT_ALLELE_FRACTION,
        "alt_reads_min": PRIMARY_MIN_ALT_READS,
        "normalization": NORMALIZATION_POLICY,
    }
    missing_params = [key for key in expected_parameters if key not in parameters]
    if missing_params:
        raise ValueError(f"RNA variant calling manifest missing parameters: {', '.join(sorted(missing_params))}")
    for key, expected in expected_parameters.items():
        observed = parameters.get(key)
        if isinstance(expected, float):
            if float(observed) != expected:
                raise ValueError(f"RNA variant calling parameter {key} mismatch: expected {expected}, observed {observed}")
        elif str(observed) != str(expected):
            raise ValueError(f"RNA variant calling parameter {key} mismatch: expected {expected}, observed {observed}")

    output_signature = manifest["output_signature"]
    if not isinstance(output_signature, dict):
        raise ValueError("RNA variant calling manifest output_signature must be a mapping")
    filtered_identity = _validate_identity_matches_path(
        output_signature.get("filtered_vcf"),
        vcf_path,
        "RNA variant calling filtered_vcf",
    )
    for label in ["raw_bcf", "call_vcf_sorted", "norm_split_dedup_vcf", "variant_evidence_tsv"]:
        _identity_has_required_fields(output_signature.get(label), f"RNA variant calling {label}")
    filter_stats = manifest["filter_stats"]
    if not isinstance(filter_stats, dict):
        raise ValueError("RNA variant calling manifest filter_stats must be a mapping")
    for field in ["normalized_records", "filtered_records"]:
        if field not in filter_stats or _is_missing_manifest_value(filter_stats.get(field)):
            raise ValueError(f"RNA variant calling manifest filter_stats missing {field}")

    return {
        "status": "validated",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "policy_version": policy,
        "sample": sample,
        "filtered_vcf": filtered_identity,
        "input_signature_sha256": _digest_json(input_signature),
        "output_signature_sha256": _digest_json(output_signature),
        "filter_stats": filter_stats,
        "parameters": parameters,
        "bcftools_version": input_signature.get("bcftools_version", ""),
        "calling_script_sha256": input_signature.get("calling_script_sha256", ""),
    }


def validate_policy(
    *,
    policy_version: str,
    primary_min_alt_reads: int,
    sensitivity_alt_reads: Iterable[int],
    min_variant_qual: float = MIN_VARIANT_QUAL,
    min_total_depth: int = MIN_TOTAL_DEPTH,
    min_variant_allele_fraction: float = MIN_VARIANT_ALLELE_FRACTION,
    min_mapping_quality: float = MIN_MAPPING_QUALITY,
) -> None:
    if policy_version != POLICY_VERSION:
        raise ValueError(f"Unsupported RNA variant/editing QC policy version: {policy_version}")
    if int(primary_min_alt_reads) != PRIMARY_MIN_ALT_READS:
        raise ValueError(f"{POLICY_VERSION} requires primary_min_alt_reads={PRIMARY_MIN_ALT_READS}")
    if tuple(int(v) for v in sensitivity_alt_reads) != SENSITIVITY_ALT_READS:
        raise ValueError(f"{POLICY_VERSION} requires sensitivity_alt_reads={SENSITIVITY_ALT_READS}")
    if float(min_variant_qual) != MIN_VARIANT_QUAL:
        raise ValueError(f"{POLICY_VERSION} requires min_variant_qual={MIN_VARIANT_QUAL:g}")
    if int(min_total_depth) != MIN_TOTAL_DEPTH:
        raise ValueError(f"{POLICY_VERSION} requires min_total_depth={MIN_TOTAL_DEPTH}")
    if float(min_variant_allele_fraction) != MIN_VARIANT_ALLELE_FRACTION:
        raise ValueError(
            f"{POLICY_VERSION} requires min_variant_allele_fraction={MIN_VARIANT_ALLELE_FRACTION:g}"
        )
    if float(min_mapping_quality) != MIN_MAPPING_QUALITY:
        raise ValueError(f"{POLICY_VERSION} requires min_mapping_quality={MIN_MAPPING_QUALITY:g}")


def genomic_snv_for_cds_offset(
    *,
    tx_blocks: list[GenomicBlock],
    strand: str,
    cds_offset0: int,
    ref_cds_base: str,
    alt_cds_base: str,
) -> tuple[str, int, str, str]:
    mapped = map_transcript_interval_to_genome(tx_blocks, strand, cds_offset0, cds_offset0 + 1)
    if len(mapped) != 1:
        raise ValueError("single CDS base mapped to multiple genomic blocks")
    block = mapped[0][2]
    if block.length != 1:
        raise ValueError("single CDS base mapping did not produce a 1-bp genomic interval")
    if strand == "-":
        ref = reverse_complement(ref_cds_base)
        alt = reverse_complement(alt_cds_base)
    else:
        ref = ref_cds_base.upper()
        alt = alt_cds_base.upper()
    return block.chrom, block.start0 + 1, ref, alt


def event_passes_thresholds(
    event: RnaVariantEvent,
    *,
    min_qual: float = MIN_VARIANT_QUAL,
    min_depth: int = MIN_TOTAL_DEPTH,
    min_vaf: float = MIN_VARIANT_ALLELE_FRACTION,
    min_alt_reads: int = PRIMARY_MIN_ALT_READS,
    min_mq: float = MIN_MAPPING_QUALITY,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    qual = parse_number(event.qual, None)
    mq = parse_number(event.mapping_quality, None)
    if event.variant_type != "SNV":
        reasons.append("complex_variant_not_evaluable")
    if event.vcf_filter != "PASS":
        reasons.append("vcf_filter_not_pass")
    if event.vcf_duplicate_status == "conflicting_duplicate_not_evaluable":
        reasons.append("conflicting_duplicate_not_evaluable")
    if qual is None or qual < min_qual:
        reasons.append("variant_qual_below_threshold")
    if event.total_depth < min_depth:
        reasons.append("total_depth_below_threshold")
    if event.vaf < min_vaf:
        reasons.append("vaf_below_threshold")
    if event.alt_reads < min_alt_reads:
        reasons.append("alt_reads_below_threshold")
    if mq is None or mq < min_mq:
        reasons.append("mapping_quality_below_threshold")
    return not reasons, reasons


def evaluate_parent_rna_variants(
    *,
    sample: str,
    parent_record_id: str,
    tx_blocks: list[GenomicBlock],
    strand: str,
    reference_cds_seq: str,
    candidate_cds_seq: str,
    candidate_parent_sequence: str,
    vcf_events: dict[tuple[str, int, str, str], RnaVariantEvent],
    rediportal_events: set[tuple[str, int, str, str]],
    rediportal_evaluation_status: str = "evaluated",
    primary_min_alt_reads: int = PRIMARY_MIN_ALT_READS,
    sensitivity_alt_reads: tuple[int, ...] = SENSITIVITY_ALT_READS,
    min_variant_qual: float = MIN_VARIANT_QUAL,
    min_total_depth: int = MIN_TOTAL_DEPTH,
    min_variant_allele_fraction: float = MIN_VARIANT_ALLELE_FRACTION,
    min_mapping_quality: float = MIN_MAPPING_QUALITY,
) -> dict[str, Any]:
    event_rows: list[dict[str, Any]] = []
    if not reference_cds_seq or not candidate_cds_seq:
        return {
            "parent_summary": {
                "sample": sample,
                "parent_record_id": parent_record_id,
                "reference_translation_status": "reference_translation_not_evaluable",
                "required_variant_count": 0,
                "supported_variant_count": 0,
                "known_editing_count": 0,
                "editing_not_evaluated_count": 0,
                "insufficient_support_count": 0,
                "complex_variant_count": 0,
                "synonymous_mismatch_count": 0,
                "rna_variant_rescue_status": "rna_variant_not_evaluable",
                "primary_core_eligibility": "provisional_coordinate_not_evaluable",
            },
            "event_rows": [],
            "variant_aa_positions": [],
            "threshold_counts": {threshold: 0 for threshold in sensitivity_alt_reads},
            "threshold_parent_pass": {threshold: False for threshold in sensitivity_alt_reads},
        }
    if len(reference_cds_seq) != len(candidate_cds_seq):
        return {
            "parent_summary": {
                "sample": sample,
                "parent_record_id": parent_record_id,
                "reference_translation_status": "reference_translation_mismatch",
                "required_variant_count": 1,
                "supported_variant_count": 0,
                "known_editing_count": 0,
                "editing_not_evaluated_count": 0,
                "insufficient_support_count": 0,
                "complex_variant_count": 1,
                "synonymous_mismatch_count": 0,
                "rna_variant_rescue_status": "rna_variant_not_evaluable",
                "primary_core_eligibility": "provisional_reference_translation_mismatch",
            },
            "event_rows": [],
            "variant_aa_positions": [],
            "threshold_counts": {threshold: 0 for threshold in sensitivity_alt_reads},
            "threshold_parent_pass": {threshold: False for threshold in sensitivity_alt_reads},
        }
    reference_translation = translate_cds(reference_cds_seq)
    if reference_translation == candidate_parent_sequence:
        return {
            "parent_summary": {
                "sample": sample,
                "parent_record_id": parent_record_id,
                "reference_translation_status": "reference_concordant",
                "required_variant_count": 0,
                "supported_variant_count": 0,
                "known_editing_count": 0,
                "editing_not_evaluated_count": 0,
                "insufficient_support_count": 0,
                "complex_variant_count": 0,
                "synonymous_mismatch_count": 0,
                "rna_variant_rescue_status": "reference_concordant",
                "primary_core_eligibility": "primary_core_eligible",
            },
            "event_rows": [],
            "variant_aa_positions": [],
            "threshold_counts": {threshold: 0 for threshold in sensitivity_alt_reads},
            "threshold_parent_pass": {threshold: False for threshold in sensitivity_alt_reads},
        }

    mismatch_offsets = [
        idx
        for idx, (ref_base, alt_base) in enumerate(zip(reference_cds_seq.upper(), candidate_cds_seq.upper()))
        if ref_base != alt_base
    ]
    required_offsets: list[int] = []
    synonymous_mismatch_count = 0
    for codon_start in range(0, len(reference_cds_seq), 3):
        ref_codon = reference_cds_seq[codon_start: codon_start + 3].upper()
        alt_codon = candidate_cds_seq[codon_start: codon_start + 3].upper()
        if len(ref_codon) != 3 or len(alt_codon) != 3:
            continue
        codon_offsets = [
            offset
            for offset in range(codon_start, codon_start + 3)
            if reference_cds_seq[offset].upper() != candidate_cds_seq[offset].upper()
        ]
        if not codon_offsets:
            continue
        if translate_cds(ref_codon) == translate_cds(alt_codon):
            synonymous_mismatch_count += len(codon_offsets)
        else:
            required_offsets.extend(codon_offsets)
    supported = 0
    editing = 0
    editing_not_evaluated = 0
    insufficient = 0
    complex_count = 0
    variant_aa_positions: set[int] = set()
    threshold_counts = {threshold: 0 for threshold in sensitivity_alt_reads}

    for offset in required_offsets:
        ref_base = reference_cds_seq[offset].upper()
        alt_base = candidate_cds_seq[offset].upper()
        aa_position = offset // 3 + 1
        codon_start = (aa_position - 1) * 3
        ref_codon = reference_cds_seq[codon_start: codon_start + 3].upper()
        alt_codon = candidate_cds_seq[codon_start: codon_start + 3].upper()
        ref_aa = translate_cds(ref_codon)
        alt_aa = translate_cds(alt_codon)
        try:
            chrom, pos_1based, genomic_ref, genomic_alt = genomic_snv_for_cds_offset(
                tx_blocks=tx_blocks,
                strand=strand,
                cds_offset0=offset,
                ref_cds_base=ref_base,
                alt_cds_base=alt_base,
            )
        except Exception as exc:
            complex_count += 1
            event_rows.append({
                "sample": sample,
                "parent_record_id": parent_record_id,
                "chrom": "",
                "pos_1based": "",
                "ref": "",
                "alt": "",
                "transcript_strand": strand,
                "cds_offset": offset,
                "aa_position": aa_position,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "variant_type": "not_evaluable",
                "qual": "",
                "ref_reads": "",
                "alt_reads": "",
                "total_depth": "",
                "vaf": "",
                "mapping_quality": "",
                "vcf_filter": "",
                "vcf_duplicate_status": "",
                "rediportal_status": "not_evaluated",
                "event_qc_status": "fail",
                "event_qc_reason": f"coordinate_projection_failed:{exc}",
            })
            continue
        key = (chrom, pos_1based, genomic_ref, genomic_alt)
        event = vcf_events.get(key)
        if rediportal_evaluation_status == "evaluated":
            rediportal_status = (
                "known_rna_editing"
                if key in rediportal_events
                else "not_detected_in_frozen_editing_resource"
            )
        else:
            rediportal_status = "editing_resource_not_evaluated"
        if rediportal_status == "known_rna_editing":
            editing += 1
        elif rediportal_status == "editing_resource_not_evaluated":
            editing_not_evaluated += 1
        if event is None:
            insufficient += 1
            event_rows.append({
                "sample": sample,
                "parent_record_id": parent_record_id,
                "chrom": chrom,
                "pos_1based": pos_1based,
                "ref": genomic_ref,
                "alt": genomic_alt,
                "transcript_strand": strand,
                "cds_offset": offset,
                "aa_position": aa_position,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "variant_type": "SNV",
                "qual": "",
                "ref_reads": "",
                "alt_reads": "",
                "total_depth": "",
                "vaf": "",
                "mapping_quality": "",
                "vcf_filter": "",
                "vcf_duplicate_status": "",
                "rediportal_status": rediportal_status,
                "event_qc_status": "fail",
                "event_qc_reason": "variant_event_not_found_in_vcf",
            })
            continue
        passed, reasons = event_passes_thresholds(
            event,
            min_qual=min_variant_qual,
            min_depth=min_total_depth,
            min_vaf=min_variant_allele_fraction,
            min_alt_reads=primary_min_alt_reads,
            min_mq=min_mapping_quality,
        )
        for threshold in sensitivity_alt_reads:
            threshold_passed, _threshold_reasons = event_passes_thresholds(
                event,
                min_qual=min_variant_qual,
                min_depth=min_total_depth,
                min_vaf=min_variant_allele_fraction,
                min_alt_reads=threshold,
                min_mq=min_mapping_quality,
            )
            if threshold_passed and rediportal_status != "known_rna_editing":
                threshold_counts[threshold] += 1
        if event.variant_type != "SNV":
            complex_count += 1
        if rediportal_status == "known_rna_editing":
            reasons = ["known_rna_editing_excluded"]
            passed = False
        if passed:
            supported += 1
            variant_aa_positions.add(aa_position)
        else:
            insufficient += 0 if rediportal_status == "known_rna_editing" else 1
        event_rows.append({
            "sample": sample,
            "parent_record_id": parent_record_id,
            "chrom": chrom,
            "pos_1based": pos_1based,
            "ref": genomic_ref,
            "alt": genomic_alt,
            "transcript_strand": strand,
            "cds_offset": offset,
            "aa_position": aa_position,
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "variant_type": event.variant_type,
            "qual": event.qual,
            "ref_reads": event.ref_reads,
            "alt_reads": event.alt_reads,
            "total_depth": event.total_depth,
            "vaf": f"{event.vaf:.6g}",
            "mapping_quality": event.mapping_quality,
            "vcf_filter": event.vcf_filter,
            "vcf_duplicate_status": event.vcf_duplicate_status,
            "rediportal_status": rediportal_status,
            "event_qc_status": "pass" if passed else "fail",
            "event_qc_reason": ";".join(reasons),
        })

    required = len(required_offsets)
    if editing:
        rescue_status = "known_rna_editing_excluded"
        eligibility = "provisional_reference_translation_mismatch"
    elif complex_count:
        rescue_status = "rna_variant_not_evaluable"
        eligibility = "provisional_reference_translation_mismatch"
    elif insufficient:
        rescue_status = "rna_variant_insufficient_support"
        eligibility = "provisional_reference_translation_mismatch"
    elif supported == required:
        rescued_translation = translate_cds(candidate_cds_seq)
        if rediportal_evaluation_status != "evaluated":
            rescue_status = "rna_variant_supported_editing_not_evaluated"
            eligibility = "exploratory_only"
        elif rescued_translation == candidate_parent_sequence:
            rescue_status = "rna_variant_rescued"
            eligibility = "primary_core_eligible"
        else:
            rescue_status = "rna_variant_not_evaluable"
            eligibility = "provisional_reference_translation_mismatch"
    else:
        rescue_status = "rna_variant_insufficient_support"
        eligibility = "provisional_reference_translation_mismatch"

    threshold_parent_pass = {
        threshold: bool(
            required
            and threshold_counts[threshold] == required
            and not editing
            and not complex_count
            and rediportal_evaluation_status == "evaluated"
        )
        for threshold in sensitivity_alt_reads
    }

    return {
        "parent_summary": {
            "sample": sample,
            "parent_record_id": parent_record_id,
            "reference_translation_status": "reference_translation_mismatch",
            "required_variant_count": required,
            "supported_variant_count": supported,
            "known_editing_count": editing,
            "editing_not_evaluated_count": editing_not_evaluated,
            "insufficient_support_count": insufficient,
            "complex_variant_count": complex_count,
            "synonymous_mismatch_count": synonymous_mismatch_count,
            "rna_variant_rescue_status": rescue_status,
            "primary_core_eligibility": eligibility,
        },
        "event_rows": event_rows,
        "variant_aa_positions": sorted(variant_aa_positions),
        "threshold_counts": threshold_counts,
        "threshold_parent_pass": threshold_parent_pass,
    }


def annotate_peptide_rna_variant_evidence(
    peptide_rows: list[dict[str, object]],
    parent_summaries: dict[str, dict[str, Any]],
    variant_aa_positions_by_parent: dict[str, list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in peptide_rows:
        parent_id = str(row.get("parent_record_id", ""))
        start = parse_int(row.get("peptide_start", ""))
        length = parse_int(row.get("peptide_length", ""))
        end = start + length - 1
        aa_positions = variant_aa_positions_by_parent.get(parent_id, [])
        overlaps = [pos for pos in aa_positions if start <= pos <= end]
        summary = parent_summaries.get(parent_id, {})
        rows.append({
            "peptide_record_id": row.get("peptide_record_id", ""),
            "parent_record_id": parent_id,
            "mhc_class": row.get("mhc_class", ""),
            "peptide": row.get("peptide", ""),
            "peptide_start": start,
            "peptide_length": length,
            "overlaps_variant_aa": "YES" if overlaps else "NO",
            "variant_dependent": "YES" if overlaps else "NO",
            "variant_aa_positions": ",".join(str(pos) for pos in overlaps),
            "rna_variant_status": summary.get("rna_variant_rescue_status", "not_evaluated"),
        })
    return rows


EVENT_FIELDS = [
    "sample", "parent_record_id", "chrom", "pos_1based", "ref", "alt",
    "transcript_strand", "cds_offset", "aa_position", "ref_aa", "alt_aa",
    "variant_type", "qual", "ref_reads", "alt_reads", "total_depth", "vaf",
    "mapping_quality", "vcf_filter", "vcf_duplicate_status", "rediportal_status",
    "event_qc_status", "event_qc_reason",
]

PARENT_SUMMARY_FIELDS = [
    "sample", "parent_record_id", "reference_translation_status",
    "required_variant_count", "supported_variant_count", "known_editing_count",
    "editing_not_evaluated_count", "insufficient_support_count", "complex_variant_count",
    "synonymous_mismatch_count",
    "rna_variant_rescue_status", "primary_core_eligibility",
]

PEPTIDE_EVIDENCE_FIELDS = [
    "peptide_record_id", "parent_record_id", "mhc_class", "peptide",
    "peptide_start", "peptide_length", "overlaps_variant_aa",
    "variant_dependent", "variant_aa_positions", "rna_variant_status",
]
