#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Freeze STAR alignment provenance for paired cryptic junction QC inputs.

This script does not move or modify STAR outputs. It records explicit tumor and
matched-normal STAR output paths, hashes lightweight files used by junction QC,
and writes deterministic manifests that downstream QC can consume instead of
guessing filenames from directory structure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


POLICY_VERSION = "star_provenance_freeze_v1.0"
FREEZE_DIR_NAME = "star-provenance-freeze"
REQUIRED_SUFFIXES = ("Aligned.out.bam", "SJ.out.tab", "Log.final.out", "Log.out")
LIGHTWEIGHT_HASH_SUFFIXES = {"SJ.out.tab", "Log.final.out", "Log.out"}
STAR_SOURCE_STATUS = "completed"
STAR_CONTROL_ROLE = "control"
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
RESUME_IGNORED_FIELDS = {
    "created_at_utc",
    "started_at_utc",
    "finished_at_utc",
    "relocated_at",
    "previous_manifest_sha256",
    "upgrade_reason",
    "upgraded_at_utc",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def stable_manifest(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in RESUME_IGNORED_FIELDS}


def star_manifest_upgrade_allowed(old: dict[str, Any], new: dict[str, Any]) -> bool:
    for key in ("policy_version", "status", "tumor_sample", "normal_sample"):
        if old.get(key) != new.get(key):
            return False
    old_outputs = old.get("current_outputs", {})
    new_outputs = new.get("current_outputs", {})
    if not isinstance(old_outputs, dict) or not isinstance(new_outputs, dict):
        return False
    for suffix in ("SJ.out.tab", "Log.out"):
        old_record = old_outputs.get(suffix, {})
        new_record = new_outputs.get(suffix, {})
        if not isinstance(old_record, dict) or not isinstance(new_record, dict):
            return False
        for field in ("path", "sha256"):
            if old_record.get(field) != new_record.get(field):
                return False
    old_bam = old_outputs.get("Aligned.out.bam", {})
    new_bam = new_outputs.get("Aligned.out.bam", {})
    if isinstance(old_bam, dict) and isinstance(new_bam, dict):
        for field in ("path", "size_bytes"):
            if old_bam.get(field) != new_bam.get(field):
                return False
    return (
        new.get("critical_parameter_compatibility") == "compatible"
        and bool(new.get("critical_star_parameter_digest"))
        and bool(new.get("critical_star_parameters"))
    )


def cohort_manifest_upgrade_allowed(old: dict[str, Any], new: dict[str, Any]) -> bool:
    for key in ("policy_version", "status", "pair_sheet", "cryptic_root", "pair_count"):
        if old.get(key) != new.get(key):
            return False
    return bool(new.get("pair_table_sha256"))


def write_json_resume(path: Path, data: dict[str, Any], *, allow_critical_contract_upgrade: bool = False) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old == data:
            return
        old_stable = stable_manifest(old)
        data_stable = stable_manifest(data)
        if old_stable == data_stable:
            return
        if allow_critical_contract_upgrade and (
            star_manifest_upgrade_allowed(old, data)
            or cohort_manifest_upgrade_allowed(old, data)
        ):
            upgraded = json.loads(json.dumps(data))
            upgraded["previous_manifest_sha256"] = sha256_file(path)
            upgraded["upgrade_reason"] = "add_star_critical_parameter_contract"
            upgraded["upgraded_at_utc"] = now_utc()
            text = json.dumps(upgraded, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            atomic_write_text(path, text)
            return
        raise RuntimeError(f"Existing STAR provenance manifest differs from current freeze: {path}")
    atomic_write_text(path, text)


def file_record(path: Path, *, sha256: bool = False, full_bam_hash: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.exists():
        record["mtime_ns"] = path.stat().st_mtime_ns
        if sha256 or full_bam_hash:
            record["sha256"] = sha256_file(path)
    return record


def manifest_hash(path: Path) -> str:
    return sha256_file(path) if path.exists() else ""


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def star_output_paths(star_dir: Path, sample: str) -> dict[str, Path]:
    return {suffix: star_dir / f"{sample}{suffix}" for suffix in REQUIRED_SUFFIXES}


def require_complete_star(star_dir: Path, sample: str) -> dict[str, Path]:
    paths = star_output_paths(star_dir, sample)
    missing = [suffix for suffix, path in paths.items() if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Incomplete STAR outputs for {sample} in {star_dir}: {', '.join(missing)}")
    return paths


def parse_pair_sheet(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").splitlines()
    rows = [line for line in text if line.strip() and not line.lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"Empty pair sheet: {path}")
    delimiter = "\t" if "\t" in rows[0] else ","
    reader = csv.DictReader(rows, delimiter=delimiter)
    if reader.fieldnames and {"tumor_sample", "normal_sample"}.issubset(set(reader.fieldnames)):
        pairs = [(str(row["tumor_sample"]).strip(), str(row["normal_sample"]).strip()) for row in reader]
    elif reader.fieldnames and "sample" in reader.fieldnames:
        pairs = []
        for row in reader:
            parts = [part.strip() for part in str(row["sample"]).split(",")]
            if len(parts) != 2:
                raise ValueError(f"Expected Tumor,Normal in sample column, got {row['sample']!r}")
            pairs.append((parts[0], parts[1]))
    else:
        pairs = []
        for line in rows:
            parts = [part.strip() for part in re.split(r"[\t,]", line) if part.strip()]
            if parts[:2] == ["tumor_sample", "normal_sample"]:
                continue
            if len(parts) != 2:
                raise ValueError(f"Expected two columns in pair sheet row: {line!r}")
            pairs.append((parts[0], parts[1]))
    validate_pairs(pairs)
    return pairs


def validate_pairs(pairs: list[tuple[str, str]]) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    tumors: set[str] = set()
    normals: set[str] = set()
    all_roles: dict[str, str] = {}
    for tumor, normal in pairs:
        if not tumor or not normal:
            raise ValueError("Tumor and normal sample IDs must be non-empty")
        if tumor == normal:
            raise ValueError(f"Tumor and normal sample IDs must differ: {tumor}")
        if (tumor, normal) in seen_pairs:
            raise ValueError(f"Duplicate tumor-normal pair: {tumor},{normal}")
        if tumor in tumors:
            raise ValueError(f"Duplicate tumor sample in pair sheet: {tumor}")
        if normal in normals:
            raise ValueError(f"Duplicate normal sample in pair sheet: {normal}")
        if tumor in all_roles and all_roles[tumor] != "tumor":
            raise ValueError(f"Sample has conflicting tumor/normal roles: {tumor}")
        if normal in all_roles and all_roles[normal] != "normal":
            raise ValueError(f"Sample has conflicting tumor/normal roles: {normal}")
        seen_pairs.add((tumor, normal))
        tumors.add(tumor)
        normals.add(normal)
        all_roles[tumor] = "tumor"
        all_roles[normal] = "normal"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_star_version(log_out: Path) -> str:
    text = log_out.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines()[:80]:
        if "STAR version" in line:
            return line.split("=", 1)[-1].strip()
        match = re.search(r"STAR[_ ]?([0-9][^\s]*)", line)
        if match:
            return match.group(0).strip()
    return ""


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


def normalize_star_parameters(params: list[Any]) -> dict[str, list[str]]:
    values = [str(value) for value in params]
    out: dict[str, list[str]] = {}
    idx = 0
    while idx < len(values):
        option = values[idx]
        if not option.startswith("--"):
            idx += 1
            continue
        idx += 1
        option_values: list[str] = []
        while idx < len(values) and not values[idx].startswith("--"):
            option_values.append(values[idx])
            idx += 1
        if option in STAR_CRITICAL_OPTIONS:
            out[option] = option_values
        elif option not in STAR_RUNTIME_OPTIONS:
            continue
    return out


def critical_parameter_digest(params: dict[str, list[str]]) -> str:
    return sha256_text(json.dumps({key: params[key] for key in sorted(params)}, sort_keys=True))


def validate_normal_source_manifest(
    manifest: dict[str, Any],
    *,
    tumor: str,
    normal: str,
    source_manifest_path: Path,
) -> None:
    if manifest.get("status") != STAR_SOURCE_STATUS:
        raise ValueError(f"Normal STAR source manifest status is not {STAR_SOURCE_STATUS}: {source_manifest_path}")
    signature = manifest.get("input_signature", {})
    if not isinstance(signature, dict):
        raise ValueError(f"Normal STAR source manifest lacks input_signature: {source_manifest_path}")
    if signature.get("alignment_role") != STAR_CONTROL_ROLE:
        raise ValueError(f"Normal STAR source manifest alignment_role is not control: {source_manifest_path}")
    if signature.get("sample") != normal or signature.get("control_sample") != normal:
        raise ValueError(f"Normal STAR source manifest sample/control_sample mismatch: {source_manifest_path}")
    if signature.get("tumor_sample") != tumor:
        raise ValueError(f"Normal STAR source manifest tumor_sample mismatch: {source_manifest_path}")


def source_output_dir_from_manifest(manifest: dict[str, Any]) -> str:
    output_dir = str(manifest.get("output_dir", "") or "")
    if output_dir:
        return output_dir
    outputs = manifest.get("outputs", {})
    if isinstance(outputs, dict):
        for value in outputs.values():
            if isinstance(value, dict) and value.get("path"):
                return str(Path(str(value["path"])).parent)
    return ""


def current_outputs(paths: dict[str, Path], *, full_bam_hash: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for suffix, path in paths.items():
        if suffix == "Aligned.out.bam":
            out[suffix] = file_record(path, full_bam_hash=full_bam_hash)
        else:
            out[suffix] = file_record(path, sha256=suffix in LIGHTWEIGHT_HASH_SUFFIXES)
    return out


def make_normal_manifest(
    *,
    tumor: str,
    normal: str,
    star_dir: Path,
    source_manifest_path: Path,
    freeze_script_sha256: str,
    relocated_at: str,
    full_bam_hash: bool,
) -> dict[str, Any]:
    paths = require_complete_star(star_dir, normal)
    source_manifest = load_json(source_manifest_path)
    validate_normal_source_manifest(source_manifest, tumor=tumor, normal=normal, source_manifest_path=source_manifest_path)
    critical_parameters = normalize_star_parameters([str(v) for v in source_manifest.get("star_parameters", [])])
    input_signature = {
        "policy_version": POLICY_VERSION,
        "manifest_type": "normal_star_relocated",
        "tumor_sample": tumor,
        "normal_sample": normal,
        "source_manifest_sha256": manifest_hash(source_manifest_path),
        "source_manifest_status": source_manifest.get("status", ""),
        "source_alignment_role": source_manifest.get("input_signature", {}).get("alignment_role", ""),
        "source_input_signature": source_manifest.get("input_signature", {}),
        "critical_star_parameters": critical_parameters,
        "critical_star_parameter_digest": critical_parameter_digest(critical_parameters),
        "current_outputs": current_outputs(paths, full_bam_hash=full_bam_hash),
        "freeze_script_sha256": freeze_script_sha256,
    }
    return {
        "policy_version": POLICY_VERSION,
        "status": "relocated_complete",
        "created_at_utc": relocated_at,
        "tumor_sample": tumor,
        "normal_sample": normal,
        "source_manifest_path": str(source_manifest_path),
        "source_manifest_sha256": manifest_hash(source_manifest_path),
        "source_output_dir": source_output_dir_from_manifest(source_manifest),
        "current_output_dir": str(star_dir),
        "source_input_signature": source_manifest.get("input_signature", {}),
        "star_version": source_manifest.get("star_version", ""),
        "star_index_identity": source_manifest.get("input_signature", {}).get("star_index", {}),
        "star_index_identity_status": "from_source_manifest",
        "star_parameters": source_manifest.get("star_parameters", []),
        "critical_star_parameters": critical_parameters,
        "critical_star_parameter_digest": critical_parameter_digest(critical_parameters),
        "critical_parameter_compatibility": "pending_pair_comparison",
        "current_outputs": current_outputs(paths, full_bam_hash=full_bam_hash),
        "sj_out_tab_rows": count_nonempty_lines(paths["SJ.out.tab"]),
        "relocated_from": source_output_dir_from_manifest(source_manifest),
        "relocated_to": str(star_dir),
        "relocated_at": relocated_at,
        "freeze_script_sha256": freeze_script_sha256,
        "input_signature": input_signature,
    }


def make_tumor_manifest(
    *,
    tumor: str,
    normal: str,
    star_dir: Path,
    star_index: str,
    freeze_script_sha256: str,
    created_at: str,
    full_bam_hash: bool,
) -> dict[str, Any]:
    paths = require_complete_star(star_dir, tumor)
    star_parameters = parse_star_command_line(paths["Log.out"])
    critical_parameters = normalize_star_parameters(star_parameters)
    input_signature = {
        "policy_version": POLICY_VERSION,
        "manifest_type": "tumor_star_frozen_legacy",
        "tumor_sample": tumor,
        "normal_sample": normal,
        "current_outputs": current_outputs(paths, full_bam_hash=full_bam_hash),
        "star_index": star_index,
        "critical_star_parameters": critical_parameters,
        "critical_star_parameter_digest": critical_parameter_digest(critical_parameters),
        "freeze_script_sha256": freeze_script_sha256,
    }
    return {
        "policy_version": POLICY_VERSION,
        "status": "frozen_legacy_complete",
        "created_at_utc": created_at,
        "tumor_sample": tumor,
        "normal_sample": normal,
        "current_output_dir": str(star_dir),
        "star_version": parse_star_version(paths["Log.out"]),
        "star_index": star_index,
        "star_index_identity_status": "reconstructed_at_freeze",
        "star_parameters": star_parameters,
        "critical_star_parameters": critical_parameters,
        "critical_star_parameter_digest": critical_parameter_digest(critical_parameters),
        "critical_parameter_compatibility": "pending_pair_comparison",
        "current_outputs": current_outputs(paths, full_bam_hash=full_bam_hash),
        "sj_out_tab_rows": count_nonempty_lines(paths["SJ.out.tab"]),
        "freeze_script_sha256": freeze_script_sha256,
        "input_signature": input_signature,
    }


def pair_row(
    *,
    tumor: str,
    normal: str,
    tumor_manifest_path: Path,
    normal_manifest_path: Path,
    tumor_manifest: dict[str, Any],
    normal_manifest: dict[str, Any],
) -> dict[str, Any]:
    tumor_outputs = tumor_manifest["current_outputs"]
    normal_outputs = normal_manifest["current_outputs"]
    return {
        "tumor_sample": tumor,
        "normal_sample": normal,
        "tumor_star_manifest": str(tumor_manifest_path),
        "normal_star_manifest": str(normal_manifest_path),
        "tumor_manifest_status": tumor_manifest["status"],
        "normal_manifest_status": normal_manifest["status"],
        "tumor_sj_path": tumor_outputs["SJ.out.tab"]["path"],
        "normal_sj_path": normal_outputs["SJ.out.tab"]["path"],
        "tumor_sj_size_bytes": tumor_outputs["SJ.out.tab"]["size_bytes"],
        "normal_sj_size_bytes": normal_outputs["SJ.out.tab"]["size_bytes"],
        "tumor_sj_sha256": tumor_outputs["SJ.out.tab"]["sha256"],
        "normal_sj_sha256": normal_outputs["SJ.out.tab"]["sha256"],
        "tumor_sj_rows": tumor_manifest["sj_out_tab_rows"],
        "normal_sj_rows": normal_manifest["sj_out_tab_rows"],
        "tumor_bam_path": tumor_outputs["Aligned.out.bam"]["path"],
        "normal_bam_path": normal_outputs["Aligned.out.bam"]["path"],
        "tumor_bam_size_bytes": tumor_outputs["Aligned.out.bam"]["size_bytes"],
        "normal_bam_size_bytes": normal_outputs["Aligned.out.bam"]["size_bytes"],
        "tumor_star_index_identity_status": tumor_manifest["star_index_identity_status"],
        "normal_star_index_identity_status": normal_manifest["star_index_identity_status"],
        "tumor_star_version": tumor_manifest.get("star_version", ""),
        "normal_star_version": normal_manifest.get("star_version", ""),
        "critical_parameter_compatibility": tumor_manifest.get("critical_parameter_compatibility", ""),
        "tumor_critical_parameter_digest": tumor_manifest.get("critical_star_parameter_digest", ""),
        "normal_critical_parameter_digest": normal_manifest.get("critical_star_parameter_digest", ""),
    }


def compare_pair_star_parameters(tumor_manifest: dict[str, Any], normal_manifest: dict[str, Any]) -> str:
    tumor_digest = str(tumor_manifest.get("critical_star_parameter_digest", ""))
    normal_digest = str(normal_manifest.get("critical_star_parameter_digest", ""))
    if not tumor_digest or not normal_digest:
        raise ValueError("Cannot compare STAR critical parameters because one digest is missing")
    if tumor_digest != normal_digest:
        tumor_params = tumor_manifest.get("critical_star_parameters", {})
        normal_params = normal_manifest.get("critical_star_parameters", {})
        raise ValueError(
            "Tumor and normal STAR critical parameters differ: "
            f"tumor={json.dumps(tumor_params, sort_keys=True)} normal={json.dumps(normal_params, sort_keys=True)}"
        )
    return "compatible"


def parse_tsv_text(text: str) -> list[dict[str, str]]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    return list(csv.DictReader(lines, delimiter="\t"))


def pair_table_upgrade_allowed(old_rows: list[dict[str, str]], new_rows: list[dict[str, Any]]) -> bool:
    if len(old_rows) != len(new_rows):
        return False
    old_by_tumor = {str(row.get("tumor_sample", "")): row for row in old_rows}
    new_by_tumor = {str(row.get("tumor_sample", "")): row for row in new_rows}
    if set(old_by_tumor) != set(new_by_tumor):
        return False
    invariant_fields = (
        "tumor_sample",
        "normal_sample",
        "tumor_sj_path",
        "normal_sj_path",
        "tumor_sj_sha256",
        "normal_sj_sha256",
    )
    for tumor, old_row in old_by_tumor.items():
        new_row = new_by_tumor[tumor]
        for field in invariant_fields:
            if str(old_row.get(field, "")) != str(new_row.get(field, "")):
                return False
        if str(new_row.get("critical_parameter_compatibility", "")) != "compatible":
            return False
        if not str(new_row.get("tumor_critical_parameter_digest", "")):
            return False
        if str(new_row.get("tumor_critical_parameter_digest", "")) != str(new_row.get("normal_critical_parameter_digest", "")):
            return False
    return True


def write_tsv_resume(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
    *,
    allow_critical_contract_upgrade: bool = False,
) -> None:
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")) for field in fields))
    text = "\n".join(lines) + "\n"
    if path.exists():
        old = path.read_text(encoding="utf-8")
        if old != text:
            if allow_critical_contract_upgrade and pair_table_upgrade_allowed(parse_tsv_text(old), rows):
                atomic_write_text(path, text)
                return
            raise RuntimeError(f"Existing STAR provenance table differs from current freeze: {path}")
        return
    atomic_write_text(path, text)


def freeze_pairs(args: argparse.Namespace) -> dict[str, Any]:
    started_at = now_utc()
    root = Path(args.cryptic_root)
    pairs = parse_pair_sheet(Path(args.pair_sheet))
    freeze_script_sha = sha256_file(Path(__file__))
    outdir = Path(args.outdir) if args.outdir else root / "bp" / "star_provenance_freeze"
    outdir.mkdir(parents=True, exist_ok=True)
    relocated_at = str(args.relocated_at or started_at)
    rows: list[dict[str, Any]] = []

    for tumor, normal in pairs:
        tumor_star_dir = root / tumor / "01-star" / tumor / f"{tumor}.star"
        normal_star_dir = root / tumor / "01-star" / normal / f"{normal}.star"
        source_manifest_path = normal_star_dir / "star_alignment.manifest.json"
        if not source_manifest_path.exists():
            raise FileNotFoundError(f"Normal STAR manifest is required for relocated freeze: {source_manifest_path}")
        freeze_dir = root / tumor / "01-star" / FREEZE_DIR_NAME
        tumor_manifest_path = freeze_dir / "tumor_star.frozen_legacy_manifest.json"
        normal_manifest_path = freeze_dir / "normal_star.relocated_manifest.json"
        tumor_manifest = make_tumor_manifest(
            tumor=tumor,
            normal=normal,
            star_dir=tumor_star_dir,
            star_index=str(args.star_index or ""),
            freeze_script_sha256=freeze_script_sha,
            created_at=relocated_at,
            full_bam_hash=bool(args.full_bam_hash),
        )
        normal_manifest = make_normal_manifest(
            tumor=tumor,
            normal=normal,
            star_dir=normal_star_dir,
            source_manifest_path=source_manifest_path,
            freeze_script_sha256=freeze_script_sha,
            relocated_at=relocated_at,
            full_bam_hash=bool(args.full_bam_hash),
        )
        compatibility = compare_pair_star_parameters(tumor_manifest, normal_manifest)
        tumor_manifest["critical_parameter_compatibility"] = compatibility
        normal_manifest["critical_parameter_compatibility"] = compatibility
        tumor_manifest["input_signature"]["critical_parameter_compatibility"] = compatibility
        normal_manifest["input_signature"]["critical_parameter_compatibility"] = compatibility
        write_json_resume(
            tumor_manifest_path,
            tumor_manifest,
            allow_critical_contract_upgrade=bool(getattr(args, "allow_critical_contract_upgrade", False)),
        )
        write_json_resume(
            normal_manifest_path,
            normal_manifest,
            allow_critical_contract_upgrade=bool(getattr(args, "allow_critical_contract_upgrade", False)),
        )
        rows.append(
            pair_row(
                tumor=tumor,
                normal=normal,
                tumor_manifest_path=tumor_manifest_path,
                normal_manifest_path=normal_manifest_path,
                tumor_manifest=tumor_manifest,
                normal_manifest=normal_manifest,
            )
        )

    fields = [
        "tumor_sample",
        "normal_sample",
        "tumor_star_manifest",
        "normal_star_manifest",
        "tumor_manifest_status",
        "normal_manifest_status",
        "tumor_sj_path",
        "normal_sj_path",
        "tumor_sj_size_bytes",
        "normal_sj_size_bytes",
        "tumor_sj_sha256",
        "normal_sj_sha256",
        "tumor_sj_rows",
        "normal_sj_rows",
        "tumor_bam_path",
        "normal_bam_path",
        "tumor_bam_size_bytes",
        "normal_bam_size_bytes",
        "tumor_star_index_identity_status",
        "normal_star_index_identity_status",
        "tumor_star_version",
        "normal_star_version",
        "critical_parameter_compatibility",
        "tumor_critical_parameter_digest",
        "normal_critical_parameter_digest",
    ]
    pair_table = outdir / "cryptic_star_pair_inputs.tsv"
    write_tsv_resume(
        pair_table,
        rows,
        fields,
        allow_critical_contract_upgrade=bool(getattr(args, "allow_critical_contract_upgrade", False)),
    )
    cohort_manifest = {
        "policy_version": POLICY_VERSION,
        "status": "complete",
        "started_at_utc": started_at,
        "finished_at_utc": now_utc(),
        "pair_sheet": str(Path(args.pair_sheet)),
        "cryptic_root": str(root),
        "pair_count": len(rows),
        "full_bam_hash": bool(args.full_bam_hash),
        "star_index": str(args.star_index or ""),
        "pair_table": str(pair_table),
        "pair_table_sha256": sha256_file(pair_table),
        "freeze_script_sha256": freeze_script_sha,
        "input_signature": {
            "policy_version": POLICY_VERSION,
            "pair_sheet_sha256": sha256_file(Path(args.pair_sheet)),
            "cryptic_root": str(root),
            "star_index": str(args.star_index or ""),
            "full_bam_hash": bool(args.full_bam_hash),
            "freeze_script_sha256": freeze_script_sha,
        },
    }
    write_json_resume(
        outdir / "star_provenance_freeze.manifest.json",
        cohort_manifest,
        allow_critical_contract_upgrade=bool(getattr(args, "allow_critical_contract_upgrade", False)),
    )
    return cohort_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cryptic-root", required=True)
    parser.add_argument("--pair-sheet", required=True)
    parser.add_argument("-o", "--outdir", default="")
    parser.add_argument("--star-index", default="")
    parser.add_argument("--relocated-at", default="")
    parser.add_argument("--full-bam-hash", action="store_true")
    parser.add_argument(
        "--allow-critical-contract-upgrade",
        action="store_true",
        help=(
            "Upgrade existing STAR freeze manifests/tables only when sample IDs, "
            "SJ paths and SJ hashes are unchanged."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    manifest = freeze_pairs(build_parser().parse_args(argv))
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
