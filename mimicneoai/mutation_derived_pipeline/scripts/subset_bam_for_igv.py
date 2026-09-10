#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python -u ./subset_bam_for_igv.py \
  --tumor-bam ../TUMOR_SAMPLE/03.BQSR/TUMOR_SAMPLE.sorted.markdup.BQSR.bam \
  --normal-bam ../NORMAL_SAMPLE/03.BQSR/NORMAL_SAMPLE.sorted.markdup.BQSR.bam \
  --bed ./04.bed/TUMOR_SAMPLE.shared.AF_AD.pad200.bed \
  --out-dir ./05.igv_bam \
  --threads 8 \
  2>&1 | tee ./subset_igv_$(date +%Y%m%d_%H%M%S).log
'''

import argparse, os, sys, shutil, subprocess, pathlib

def req(bin_name):
    if shutil.which(bin_name) is None:
        sys.exit(f"[ERROR] required tool not found in PATH: {bin_name}")

def run_checked(cmd):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode(errors="ignore"))
        sys.exit(f"[ERROR] command failed: {' '.join(cmd)}")
    return r

def strip_bam_suffix(path):
    name = os.path.basename(path)
    return name[:-4] if name.endswith(".bam") else name

def main():
    ap = argparse.ArgumentParser(
        description="Subset BAMs by BED (for IGV) using samtools view/index"
    )
    ap.add_argument("--tumor-bam", required=True, help="Tumor BAM")
    ap.add_argument("--normal-bam", required=False, default="", help="Normal BAM (optional)")
    ap.add_argument("--bed", required=True, help="BED file of regions")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--threads", type=int, default=4, help="Threads for samtools index (default 4)")
    args = ap.parse_args()

    # deps
    req("samtools")

    # check inputs
    for f in (args.tumor_bam, args.bed):
        if not os.path.exists(f):
            sys.exit(f"[ERROR] missing file: {f}")
    if args.normal_bam and not os.path.exists(args.normal_bam):
        print(f"[WARN] normal BAM not found, skip: {args.normal_bam}", file=sys.stderr)
        args.normal_bam = ""

    outdir = pathlib.Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # tumor
    tumor_base = strip_bam_suffix(args.tumor_bam) + ".subset.bam"
    tumor_out  = str(outdir / os.path.basename(tumor_base))
    run_checked(["samtools", "view", "-b", "-L", args.bed, args.tumor_bam, "-o", tumor_out])
    run_checked(["samtools", "index", "-@", str(args.threads), tumor_out])

    # normal (optional)
    if args.normal_bam:
        normal_base = strip_bam_suffix(args.normal_bam) + ".subset.bam"
        normal_out  = str(outdir / os.path.basename(normal_base))
        run_checked(["samtools", "view", "-b", "-L", args.bed, args.normal_bam, "-o", normal_out])
        run_checked(["samtools", "index", "-@", str(args.threads), normal_out])

    print(f"[OK] Subset BAMs and indexes saved to: {os.path.abspath(outdir)}")

if __name__ == "__main__":
    main()
