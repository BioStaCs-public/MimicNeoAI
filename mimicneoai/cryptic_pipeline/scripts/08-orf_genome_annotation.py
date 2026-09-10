#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08-orf_genome_annotation.py

Generate selected aeSEP CDS FASTA and map ORF/CDS sequences back to the
reference genome for block-level genomic annotation.

This script consolidates two ORF annotation operations:
  1) select CDS records for aeSEP peptide IDs;
  2) run minimap2/samtools/bedtools/HOMER genome annotation.

Example (sanitized):
  python 08-orf_genome_annotation.py \
    -s SAMPLE_ID \
    --sample-dir /path/to/Cryptic/SAMPLE_ID \
    -o /path/to/Cryptic/SAMPLE_ID/08-annotations \
    --genome-fa /path/to/reference/GRCh38.genome.fa \
    --gtf /path/to/reference/gencode.annotation.gtf \
    --threads 36 \
    --sort-threads 8 \
    --log-file 08-orf_genome_annotation.log
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_EXTRA_TOOL_DIRS = [
    "/opt/conda/envs/mimicneoai-rna-newzlib/bin",
    "/opt/conda/envs/mimicneoai-tools/bin",
    "/opt/conda/envs/mimicneoai-homer/bin",
]


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def exists(path: str | Path) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0


def check_file(path: str | Path, label: Optional[str] = None) -> None:
    if not exists(path):
        name = f" ({label})" if label else ""
        raise FileNotFoundError(f"[ERR] missing or empty{name}: {path}")


def split_path_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item for item in raw.split(os.pathsep) if item]


def build_runtime_env(extra_tool_paths: list[str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    path_items: list[str] = []
    path_items.extend(split_path_list(env.get("MIMICNEOAI_TOOL_PATHS")))
    path_items.extend(extra_tool_paths or [])
    path_items.extend(DEFAULT_EXTRA_TOOL_DIRS)
    path_items.extend(split_path_list(env.get("PATH")))

    seen: set[str] = set()
    resolved: list[str] = []
    for item in path_items:
        if item in seen:
            continue
        if item.startswith("/opt/conda/") and not Path(item).exists():
            continue
        seen.add(item)
        resolved.append(item)
    env["PATH"] = os.pathsep.join(resolved)
    return env


def check_bin(name: str, env: dict[str, str]) -> None:
    if shutil.which(name, path=env.get("PATH")) is None:
        raise RuntimeError(f"[ERR] need `{name}` in PATH")


def quote(path: str | Path) -> str:
    return shlex.quote(str(path))


def run_shell(cmd: str, cwd: str | Path, log_fh, env: dict[str, str]) -> None:
    log("CMD: " + cmd)
    log_fh.write(f"\n[{ts()}] CMD: {cmd}\n")
    log_fh.flush()
    subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        check=True,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        executable="/bin/bash",
        env=env,
    )


def command_output(cmd: list[str], cwd: str | Path, log_fh, env: dict[str, str]) -> str:
    log("CMD: " + " ".join(map(str, cmd)))
    log_fh.write(f"\n[{ts()}] CMD: {' '.join(map(str, cmd))}\n")
    log_fh.flush()
    out = subprocess.check_output(list(map(str, cmd)), cwd=str(cwd), stderr=subprocess.STDOUT, env=env)
    text = out.decode(errors="replace").strip()
    log_fh.write(text + "\n")
    log_fh.flush()
    return text


def read_fasta_to_df(paths: Iterable[str | Path]) -> pd.DataFrame:
    """
    Read FASTA records into a DataFrame by splitting headers on whitespace.

    This intentionally mirrors the former annotation_v4.ipynb behavior:
    header fields are named h1, h2, ... and sequence is stored in seq.
    """
    records: list[tuple[str, str]] = []
    for fasta_path in paths:
        check_file(fasta_path)
        with open(fasta_path, "r") as f:
            header = None
            seq_chunks: list[str] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if header is not None:
                        records.append((header, "".join(seq_chunks)))
                    header = line[1:]
                    seq_chunks = []
                else:
                    seq_chunks.append(line)
            if header is not None:
                records.append((header, "".join(seq_chunks)))

    parts_list = [h.split() for h, _ in records]
    max_parts = max((len(p) for p in parts_list), default=0)

    rows = []
    for (header, seq), parts in zip(records, parts_list):
        row = {f"h{i + 1}": (parts[i] if i < len(parts) else None) for i in range(max_parts)}
        row["seq"] = seq
        rows.append(row)
    return pd.DataFrame(rows)


def one_line_seq(seq: str) -> str:
    return (seq or "").replace(" ", "").replace("\n", "").strip()


def default_paths(sample: str, sample_dir: str | Path) -> dict[str, Path]:
    root = Path(sample_dir)
    return {
        "pep_known": root / "02-known" / "06.filter_sORFs" / f"{sample}.lncRNA.sORFs.pep",
        "pep_novel": root / "03-novel" / "06.filter_sORFs" / f"{sample}.lncRNA.sORFs.pep",
        "cds_known": root
        / "02-known"
        / "05.transdecoder_orf"
        / f"{sample}.lncRNA.TransDecoder.LongOrfs"
        / "longest_orfs.cds",
        "cds_novel": root
        / "03-novel"
        / "05.transdecoder_orf"
        / f"{sample}.lncRNA.TransDecoder.LongOrfs"
        / "longest_orfs.cds",
    }


def write_selected_sep_cds_fasta(
    sample: str,
    pep_known: str | Path,
    pep_novel: str | Path,
    cds_known: str | Path,
    cds_novel: str | Path,
    out_fa: str | Path,
) -> tuple[int, int, int]:
    """
    Select CDS records whose h1 IDs are present in the selected aeSEP peptide FASTA.

    Returns:
        selected_pep_ids, selected_cds_records, written_records
    """
    pep_df = read_fasta_to_df([pep_known, pep_novel])
    if "h1" not in pep_df.columns:
        raise ValueError("[ERR] selected peptide FASTA headers do not contain h1")

    known_cds_df = read_fasta_to_df([cds_known])
    novel_cds_df = read_fasta_to_df([cds_novel])
    if "h1" not in known_cds_df.columns or "h1" not in novel_cds_df.columns:
        raise ValueError("[ERR] TransDecoder CDS FASTA headers do not contain h1")

    selected_ids = set(pep_df["h1"].dropna().astype(str))
    cds1 = known_cds_df[known_cds_df["h1"].isin(selected_ids)].copy()
    cds2 = novel_cds_df[novel_cds_df["h1"].isin(selected_ids)].copy()
    cds_df = pd.concat([cds1, cds2], ignore_index=True)

    out_fa = Path(out_fa)
    ensure_dir(out_fa.parent)
    with out_fa.open("w") as f:
        for _, row in cds_df.iterrows():
            header = " ".join(
                str(row[col])
                for col in ["h1", "h2", "h3", "h4"]
                if col in cds_df.columns and pd.notna(row[col])
            )
            seq = one_line_seq(row["seq"])
            f.write(f">{header}\n{seq}\n")

    return len(selected_ids), len(cds_df), sum(1 for _ in open(out_fa) if _.startswith(">"))


def run_genome_annotation(
    sample: str,
    outdir: str | Path,
    sep_cds_fa: str | Path,
    genome_fa: str | Path,
    gtf: str | Path,
    threads: int,
    sort_threads: int,
    homer_genome: str | Path | None,
    env: dict[str, str],
    log_fh,
) -> None:
    """
    Run the former run-v4.sh command sequence.

    It keeps primary + secondary alignments and filters only unmapped and
    supplementary reads with -F 0x804, with no MAPQ filter.
    """
    outdir = Path(outdir)
    sep_cds_fa = Path(sep_cds_fa)

    run_shell(
        (
            f"minimap2 -t {threads} -ax splice -k14 {quote(genome_fa)} {quote(sep_cds_fa)} "
            f"| samtools sort -@ {sort_threads} -o orf2genome.bam"
        ),
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )
    run_shell("samtools index orf2genome.bam", cwd=outdir, log_fh=log_fh, env=env)

    run_shell(
        "samtools view -bh -F 0x804 orf2genome.bam > orf2genome.noUnmap.noSup.bam",
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )
    run_shell("samtools index orf2genome.noUnmap.noSup.bam", cwd=outdir, log_fh=log_fh, env=env)

    run_shell(
        "bedtools bamtobed -bed12 -i orf2genome.noUnmap.noSup.bam > orf.noUnmap.noSup.bed12",
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )
    run_shell(
        "bedtools bed12tobed6 -i orf.noUnmap.noSup.bed12 > orf.noUnmap.noSup.blocks.bed",
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )

    run_shell(
        (
            "annotatePeaks.pl orf.noUnmap.noSup.blocks.bed "
            f"{quote(homer_genome or genome_fa)} "
            f"-gtf {quote(gtf)} -gid > orf.noUnmap.noSup.blocks.homer.tsv"
        ),
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )

    run_shell(
        (
            f"awk '$3==\"exon\"' {quote(gtf)} "
            "| bedtools intersect -s -wa -wb -a orf.noUnmap.noSup.bed12 -b - "
            "> exon.overlap.orf_gtf.noUnmap.noSup.tsv"
        ),
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )
    run_shell(
        (
            f"awk '$3==\"CDS\"' {quote(gtf)} "
            "| bedtools intersect -s -wa -wb -a orf.noUnmap.noSup.bed12 -b - "
            "> cds.overlap.orf_gtf.noUnmap.noSup.tsv"
        ),
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )

    mapped = command_output(["samtools", "view", "-c", "-F", "0x4", "orf2genome.bam"], outdir, log_fh, env)
    kept = command_output(["samtools", "view", "-c", "orf2genome.noUnmap.noSup.bam"], outdir, log_fh, env)
    secondary = command_output(
        ["samtools", "view", "-c", "-f", "0x100", "orf2genome.noUnmap.noSup.bam"],
        outdir,
        log_fh,
        env,
    )
    log_fh.write(f"mapped (primary+secondary) alignments: {mapped}\n")
    log_fh.write(f"kept (no unmapped/supplementary): {kept}\n")
    log_fh.write(f"secondary alignments kept: {secondary}\n")

    run_shell(
        "samtools view orf2genome.noUnmap.noSup.bam | awk '{print $5}' | sort -n | uniq -c | head",
        cwd=outdir,
        log_fh=log_fh,
        env=env,
    )

    for required in [
        "orf2genome.bam",
        "orf2genome.noUnmap.noSup.bam",
        "orf.noUnmap.noSup.bed12",
        "orf.noUnmap.noSup.blocks.bed",
        "orf.noUnmap.noSup.blocks.homer.tsv",
        "exon.overlap.orf_gtf.noUnmap.noSup.tsv",
        "cds.overlap.orf_gtf.noUnmap.noSup.tsv",
    ]:
        check_file(outdir / required)

    log(f"[DONE] Genome annotation finished for {sample}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate aeSEP CDS FASTA and annotate ORF genomic positions for cryptic antigen pipeline."
    )
    ap.add_argument("-s", "--sample", required=True, help="Sample ID, e.g. SAMPLE_ID")
    ap.add_argument("--sample-dir", required=True, help="Sample root directory containing 02-known and 03-novel")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory, usually SAMPLE/08-annotaions")

    ap.add_argument("--pep-known", help="Override known selected sORF peptide FASTA")
    ap.add_argument("--pep-novel", help="Override novel selected sORF peptide FASTA")
    ap.add_argument("--cds-known", help="Override known TransDecoder longest_orfs.cds")
    ap.add_argument("--cds-novel", help="Override novel TransDecoder longest_orfs.cds")
    ap.add_argument("--out-cds-fasta", help="Override output CDS FASTA path")

    ap.add_argument("--genome-fa", help="Reference genome FASTA; required unless --write-cds-only is set")
    ap.add_argument("--gtf", help="Reference GTF for HOMER/bedtools annotation; required unless --write-cds-only is set")
    ap.add_argument(
        "--homer-genome",
        help=(
            "Genome argument passed to annotatePeaks.pl. Defaults to --genome-fa. "
            "Set to a HOMER genome key such as hg38 only when that genome is installed in HOMER."
        ),
    )
    ap.add_argument(
        "--extra-tool-path",
        action="append",
        default=[],
        help=(
            "Additional directory prepended to PATH for minimap2/samtools/bedtools/HOMER. "
            "May be used multiple times. MIMICNEOAI_TOOL_PATHS is also honored."
        ),
    )
    ap.add_argument("--threads", type=int, default=36, help="minimap2 threads")
    ap.add_argument("--sort-threads", type=int, default=8, help="samtools sort threads")
    ap.add_argument(
        "--log-file",
        default="08-orf_genome_annotation.log",
        help="Log filename under --outdir, or an absolute/relative path",
    )
    ap.add_argument(
        "--write-cds-only",
        action="store_true",
        help="Only write selected SEP CDS FASTA; skip minimap2/samtools/bedtools/HOMER annotation",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    sample = args.sample
    sample_dir = Path(args.sample_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    paths = default_paths(sample, sample_dir)
    pep_known = Path(args.pep_known) if args.pep_known else paths["pep_known"]
    pep_novel = Path(args.pep_novel) if args.pep_novel else paths["pep_novel"]
    cds_known = Path(args.cds_known) if args.cds_known else paths["cds_known"]
    cds_novel = Path(args.cds_novel) if args.cds_novel else paths["cds_novel"]
    out_cds_fa = Path(args.out_cds_fasta) if args.out_cds_fasta else outdir / f"{sample}.SEPs.cds.fa"

    for path, label in [
        (pep_known, "known selected peptide FASTA"),
        (pep_novel, "novel selected peptide FASTA"),
        (cds_known, "known TransDecoder CDS"),
        (cds_novel, "novel TransDecoder CDS"),
    ]:
        check_file(path, label)

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = outdir / log_path
    ensure_dir(log_path.parent)
    with log_path.open("w") as log_fh:
        log_fh.write(f"[{ts()}] 08-orf_genome_annotation.py started\n")
        log_fh.write(f"sample: {sample}\n")
        log_fh.write(f"sample_dir: {sample_dir}\n")
        log_fh.write(f"outdir: {outdir}\n")

        n_ids, n_selected_cds, n_written = write_selected_sep_cds_fasta(
            sample=sample,
            pep_known=pep_known,
            pep_novel=pep_novel,
            cds_known=cds_known,
            cds_novel=cds_novel,
            out_fa=out_cds_fa,
        )
        msg = (
            f"[INFO] selected peptide IDs={n_ids}; selected CDS rows={n_selected_cds}; "
            f"written CDS FASTA entries={n_written} -> {out_cds_fa}"
        )
        log(msg)
        log_fh.write(msg + "\n")

        if args.write_cds_only:
            log("[DONE] CDS FASTA generation finished (--write-cds-only).")
            log_fh.write(f"[{ts()}] done: write-cds-only\n")
            return

        if not args.genome_fa or not args.gtf:
            raise ValueError("[ERR] --genome-fa and --gtf are required for full genome annotation")

        runtime_env = build_runtime_env(args.extra_tool_path)
        log_fh.write(f"runtime PATH: {runtime_env.get('PATH', '')}\n")
        for binary in ["minimap2", "samtools", "bedtools", "annotatePeaks.pl"]:
            check_bin(binary, runtime_env)
        check_file(args.genome_fa, "reference genome FASTA")
        check_file(args.gtf, "reference GTF")
        check_file(out_cds_fa, "selected SEP CDS FASTA")

        run_genome_annotation(
            sample=sample,
            outdir=outdir,
            sep_cds_fa=out_cds_fa,
            genome_fa=args.genome_fa,
            gtf=args.gtf,
            threads=args.threads,
            sort_threads=args.sort_threads,
            homer_genome=args.homer_genome,
            env=runtime_env,
            log_fh=log_fh,
        )
        log_fh.write(f"[{ts()}] 08-orf_genome_annotation.py finished\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        sys.exit(1)
