#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
  python 01-alignment.py \
    -s Sample_001_T \
    --clean-dir ./00-clean \
    --genome-dir star_and_rsem/GRCh38_gencode.v23 \
    --out-root ./01-star \
    -p 20
"""

import argparse
import json
import shlex
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

def run(cmd, shell=False):
    """Run a command and print it; raise on non-zero exit."""
    print("[CMD]", cmd if shell else " ".join(shlex.quote(str(c)) for c in cmd), flush=True)
    if shell:
        subprocess.check_call(cmd, shell=True, executable="/bin/bash")
    else:
        subprocess.check_call(cmd)


def capture_command_output(cmd: list[str]) -> str:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except Exception as exc:
        return f"unavailable: {exc}"
    return (result.stdout or "").strip()


REQUIRED_STAR_OUTPUT_SUFFIXES = (
    "Aligned.out.bam",
    "SJ.out.tab",
    "Log.final.out",
    "Log.out",
)
EXPECTED_STAR_OUTPUT_SUFFIXES = REQUIRED_STAR_OUTPUT_SUFFIXES


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def star_output_paths(star_dir: Path, sample: str) -> dict[str, Path]:
    return {suffix: star_dir / f"{sample}{suffix}" for suffix in EXPECTED_STAR_OUTPUT_SUFFIXES}


def star_outputs_complete(star_dir: Path, sample: str) -> bool:
    paths = star_output_paths(star_dir, sample)
    return all(_nonempty(paths[suffix]) for suffix in REQUIRED_STAR_OUTPUT_SUFFIXES)


def count_sj_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def file_record(path: Path) -> dict[str, object]:
    record = {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }
    if path.exists():
        record["mtime_ns"] = path.stat().st_mtime_ns
    return record


def star_index_record(genome_dir: str) -> dict[str, object]:
    genome_path = Path(genome_dir).resolve()
    required_names = ("Genome", "SA", "SAindex", "sjdbList.out.tab")
    return {
        "path": str(genome_path),
        "exists": genome_path.exists(),
        "files": {name: file_record(genome_path / name) for name in required_names},
    }


def alignment_signature(
    *,
    sample: str,
    alignment_role: str,
    tumor_sample: str,
    control_sample: str,
    raw_r1: str,
    raw_r2: str,
    clean_r1: Path,
    clean_r2: Path,
    genome_dir: str,
    star_bin: str,
    star_version: str,
    threads: str,
    command: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "star_alignment_signature_v1",
        "sample": sample,
        "alignment_role": alignment_role,
        "tumor_sample": tumor_sample,
        "control_sample": control_sample,
        "raw_fastqs": {
            "r1": file_record(Path(raw_r1)) if raw_r1 else None,
            "r2": file_record(Path(raw_r2)) if raw_r2 else None,
        },
        "clean_fastqs": {
            "r1": file_record(clean_r1),
            "r2": file_record(clean_r2),
        },
        "star_index": star_index_record(genome_dir),
        "star_bin": str(star_bin),
        "star_version": str(star_version),
        "threads": int(threads),
        "star_parameters": command[1:],
    }


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_matching_manifest_signature(manifest_path: Path, current_signature: dict[str, object]) -> None:
    if not manifest_path.exists():
        raise RuntimeError(
            f"Complete STAR outputs exist but manifest is missing: {manifest_path}. "
            "Use a new --out-root or inspect/remove the stale STAR directory before rerun."
        )
    manifest = load_manifest(manifest_path)
    stored_signature = manifest.get("input_signature")
    if stored_signature != current_signature:
        raise RuntimeError(
            f"Complete STAR outputs exist but manifest signature does not match current inputs/parameters: {manifest_path}. "
            "Use a new --out-root or inspect/remove the stale STAR directory before rerun."
        )


def existing_star_state(star_dir: Path, sample: str, manifest_path: Path, current_signature: dict[str, object]) -> str:
    outputs = star_output_paths(star_dir, sample)
    if star_outputs_complete(star_dir, sample):
        require_matching_manifest_signature(manifest_path, current_signature)
        return "complete"
    existing = [suffix for suffix, path in outputs.items() if path.exists()]
    if existing or manifest_path.exists():
        missing = [suffix for suffix, path in outputs.items() if not _nonempty(path)]
        raise RuntimeError(
            "Incomplete STAR output exists and will not be overwritten in place. "
            f"Existing outputs: {', '.join(existing) if existing else 'none'}; "
            f"missing or empty required outputs: {', '.join(missing)}. "
            "Use a new --out-root or inspect/remove the stale STAR directory before rerun."
        )
    return "absent"


def write_manifest(
    manifest_path: Path,
    *,
    sample: str,
    alignment_role: str,
    tumor_sample: str,
    control_sample: str,
    raw_r1: str,
    raw_r2: str,
    clean_r1: Path,
    clean_r2: Path,
    genome_dir: str,
    star_bin: str,
    star_version: str = "",
    threads: str,
    star_dir: Path,
    command: list[str],
    status: str,
) -> None:
    outputs = star_output_paths(star_dir, sample)
    signature = alignment_signature(
        sample=sample,
        alignment_role=alignment_role,
        tumor_sample=tumor_sample,
        control_sample=control_sample,
        raw_r1=raw_r1,
        raw_r2=raw_r2,
        clean_r1=clean_r1,
        clean_r2=clean_r2,
        genome_dir=genome_dir,
        star_bin=star_bin,
        star_version=star_version,
        threads=threads,
        command=command,
    )
    data = {
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_signature": signature,
        "sample": sample,
        "alignment_role": alignment_role,
        "tumor_sample": tumor_sample,
        "control_sample": control_sample,
        "raw_fastqs": {
            "r1": file_record(Path(raw_r1)) if raw_r1 else None,
            "r2": file_record(Path(raw_r2)) if raw_r2 else None,
        },
        "clean_fastqs": {
            "r1": file_record(clean_r1),
            "r2": file_record(clean_r2),
        },
        "star_index": str(genome_dir),
        "star_index_record": star_index_record(genome_dir),
        "star_bin": str(star_bin),
        "star_version": star_version,
        "threads": int(threads),
        "star_parameters": command[1:],
        "command": " ".join(shlex.quote(str(c)) for c in command),
        "output_dir": str(star_dir),
        "outputs": {suffix: file_record(path) for suffix, path in outputs.items()},
        "sj_out_tab_rows": count_sj_rows(outputs["SJ.out.tab"]),
    }
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def parse_args():
    ap = argparse.ArgumentParser(
        description="STAR alignment using cleaned FASTQs produced by 00-QC.py"
    )
    ap.add_argument("-s", "--sample", required=True, help="Sample name, e.g., SAMPLE-T-RNA")
    ap.add_argument("--clean-dir", default="./00-clean", help="Directory of cleaned FASTQs (*.QC.fq.gz)")
    ap.add_argument("--star-bin", default="STAR", help="Path to STAR binary (default: STAR in PATH)")
    ap.add_argument("--genome-dir", required=True, help="STAR --genomeDir (RSEM/STAR index)")
    ap.add_argument("-p", "--threads", default="20", help="Number of threads for STAR")
    # Output root; keep layout consistent with downstream scripts: 01.star/<sample>/<sample>.star/
    ap.add_argument("--out-root", default="./01.star", help="Output root directory (default: ./01.star)")
    ap.add_argument("--alignment-role", choices=["tumor", "control"], default="tumor")
    ap.add_argument("--tumor-sample", default="", help="Tumor sample ID for paired-run manifest records")
    ap.add_argument("--control-sample", default="", help="Control sample ID for paired-run manifest records")
    ap.add_argument("--raw-fq1", default="", help="Raw R1 FASTQ path for manifest records")
    ap.add_argument("--raw-fq2", default="", help="Raw R2 FASTQ path for manifest records")
    return ap.parse_args()

def main():
    a = parse_args()

    clean_dir = Path(a.clean_dir).resolve()
    r1 = clean_dir / f"{a.sample}.R1.QC.fq.gz"
    r2 = clean_dir / f"{a.sample}.R2.QC.fq.gz"
    if not (r1.exists() and r2.exists()):
        sys.stderr.write(f"[ERR] Cleaned FASTQs not found:\n  {r1}\n  {r2}\n")
        sys.stderr.write("      (Run 00-QC.py first, or provide correct --clean-dir / --sample)\n")
        sys.exit(1)

    # STAR output directory and filename prefix (aligned with expectations in downstream pipeline)
    star_dir = Path(a.out_root, a.sample, f"{a.sample}.star").resolve()
    star_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(star_dir / a.sample)  # STAR will produce <prefix>Aligned.out.bam
    outputs = star_output_paths(star_dir, a.sample)
    manifest_path = star_dir / "star_alignment.manifest.json"

    cmd_star = [
        a.star_bin,
        "--genomeDir", a.genome_dir,
        "--outSAMunmapped", "Within",
        "--outFilterType", "BySJout",
        "--outSAMattributes", "NH", "HI", "AS", "NM", "MD",
        "--outFilterMultimapNmax", "20",
        "--outFilterMismatchNmax", "999",
        "--outFilterMismatchNoverLmax", "0.04",
        "--alignIntronMin", "20",
        "--alignIntronMax", "1000000",
        "--alignMatesGapMax", "1000000",
        "--alignSJoverhangMin", "8",
        "--alignSJDBoverhangMin", "1",
        "--sjdbScore", "1",
        "--runThreadN", str(a.threads),
        "--genomeLoad", "LoadAndKeep",
        "--limitBAMsortRAM", "40000000000",
        "--outSAMtype", "BAM", "Unsorted",
        "--quantMode", "TranscriptomeSAM",
        "--outSAMheaderHD", "@HD", "VN:1.4", "SO:unsorted",
        "--outFileNamePrefix", out_prefix,
        "--readFilesCommand", "zcat",
        "--readFilesIn", str(r1), str(r2),
    ]
    star_version = capture_command_output([a.star_bin, "--version"])

    current_signature = alignment_signature(
        sample=a.sample,
        alignment_role=a.alignment_role,
        tumor_sample=a.tumor_sample,
        control_sample=a.control_sample,
        raw_r1=a.raw_fq1,
        raw_r2=a.raw_fq2,
        clean_r1=r1,
        clean_r2=r2,
        genome_dir=a.genome_dir,
        star_bin=a.star_bin,
        star_version=star_version,
        threads=str(a.threads),
        command=cmd_star,
    )

    try:
        state = existing_star_state(star_dir, a.sample, manifest_path, current_signature)
    except RuntimeError as exc:
        sys.stderr.write(f"[ERR] {exc}\n")
        sys.exit(2)

    if state == "complete":
        print(f"[SKIP] Found complete STAR output with matching manifest: {star_dir}")
        print("[DONE] STAR finished.")
        print(" BAM   :", outputs["Aligned.out.bam"])
        print(" SJ    :", outputs["SJ.out.tab"], f"({count_sj_rows(outputs['SJ.out.tab'])} rows)")
        print(" Prefix outputs under:", star_dir)
        return

    run(cmd_star)

    required_after_run = [suffix for suffix, path in outputs.items() if not _nonempty(path)]
    if not required_after_run:
        write_manifest(
            manifest_path,
            sample=a.sample,
            alignment_role=a.alignment_role,
            tumor_sample=a.tumor_sample,
            control_sample=a.control_sample,
            raw_r1=a.raw_fq1,
            raw_r2=a.raw_fq2,
            clean_r1=r1,
            clean_r2=r2,
            genome_dir=a.genome_dir,
            star_bin=a.star_bin,
            threads=str(a.threads),
            star_dir=star_dir,
            command=cmd_star,
            status="completed",
            star_version=star_version,
        )
        print("[DONE] STAR finished.")
        print(" BAM   :", outputs["Aligned.out.bam"])
        print(" SJ    :", outputs["SJ.out.tab"], f"({count_sj_rows(outputs['SJ.out.tab'])} rows)")
        print(" Prefix outputs under:", star_dir)
    else:
        write_manifest(
            manifest_path,
            sample=a.sample,
            alignment_role=a.alignment_role,
            tumor_sample=a.tumor_sample,
            control_sample=a.control_sample,
            raw_r1=a.raw_fq1,
            raw_r2=a.raw_fq2,
            clean_r1=r1,
            clean_r2=r2,
            genome_dir=a.genome_dir,
            star_bin=a.star_bin,
            star_version=star_version,
            threads=str(a.threads),
            star_dir=star_dir,
            command=cmd_star,
            status="failed_incomplete_outputs",
        )
        sys.stderr.write(
            "[ERR] STAR did not produce required outputs: "
            + ", ".join(required_after_run)
            + "\n"
        )
        sys.exit(2)

if __name__ == "__main__":
    main()
