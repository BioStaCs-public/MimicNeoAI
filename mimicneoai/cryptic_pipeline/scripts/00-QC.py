#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
  python 00-QC.py \
    --fq1 /path/to/sample/sample.R1.fq.gz \
    --fq2 /path/to/sample/sample.R2.fq.gz \
    -o ./00-clean \
    -p 20
"""

import argparse
import sys
import subprocess
import re
from pathlib import Path

def run(cmd):
    print("[CMD]", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call(cmd)

def exists_nonempty(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def sample_from_r1(r1_name: str) -> str:
    # Support patterns like *.R1.fq.gz / *.R1.fastq.gz
    return re.sub(r"\.R1\.(fastq|fq)\.gz$", "", r1_name)

def parse_args():
    ap = argparse.ArgumentParser(
        description="fastp QC (single sample, paired-end). No --detect_adapter_for_pe."
    )
    ap.add_argument("--fq1", required=True, help="R1 FASTQ(.gz)")
    ap.add_argument("--fq2", required=True, help="R2 FASTQ(.gz)")
    ap.add_argument("-o", "--outdir", default=None, help="Output dir (default: <R1_dir>/clean)")
    ap.add_argument("-p", "--threads", default="8", help="Threads (default: 8)")
    ap.add_argument("--fastp", default="fastp", help="Path to fastp (default: fastp in PATH)")
    return ap.parse_args()

def main():
    a = parse_args()
    fq1 = Path(a.fq1).expanduser()
    fq2 = Path(a.fq2).expanduser()
    if not fq1.is_absolute():
        fq1 = Path.cwd() / fq1
    if not fq2.is_absolute():
        fq2 = Path.cwd() / fq2
    if not exists_nonempty(fq1) or not exists_nonempty(fq2):
        sys.stderr.write("[ERR] FQ1/FQ2 not found or empty\n")
        sys.exit(1)

    # Output directory: default to <R1 parent>/clean
    outdir = Path(a.outdir) if a.outdir else (fq1.parent / "clean")
    outdir.mkdir(parents=True, exist_ok=True)

    sample = sample_from_r1(fq1.name)
    out_r1 = outdir / f"{sample}.R1.QC.fq.gz"
    out_r2 = outdir / f"{sample}.R2.QC.fq.gz"
    html   = outdir / f"{sample}.fastp.html"
    json   = outdir / f"{sample}.fastp.json"

    # Fast skip if outputs already exist and are non-empty
    if exists_nonempty(out_r1) and exists_nonempty(out_r2):
        print(f"[SKIP] exists: {out_r1.name}, {out_r2.name}")
        print(f"[INFO] report: {html.name}, {json.name}")
        return

    # Version check (best-effort; do not abort on failure)
    try:
        subprocess.check_call([a.fastp, "--version"],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    except Exception:
        sys.stderr.write(f"[WARN] cannot run {a.fastp} --version; continuing...\n")

    # Core command: do not enable --detect_adapter_for_pe (keep minimal)
    run([
        a.fastp,
        "-i", str(fq1), "-I", str(fq2),
        "-o", str(out_r1), "-O", str(out_r2),
        "-w", str(a.threads),
        "-h", str(html), "-j", str(json)
    ])

    print("[DONE] QC outputs:")
    print(" ", out_r1)
    print(" ", out_r2)
    print(" ", html)
    print(" ", json)

if __name__ == "__main__":
    main()
