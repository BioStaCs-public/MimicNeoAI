#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08-orf_filter.py

Build ORF-level genome annotation filters before HLA binding prediction and
write an ORF-filtered aeSEP peptide FASTA.

This step reuses the same ORF filtering functions used by
09-cryptic_epitope_annotation.py, so pre-binding filtering and downstream
candidate annotation stay consistent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)


def load_postprocess_module():
    script_path = Path(__file__).with_name("09-cryptic_epitope_annotation.py")
    spec = importlib.util.spec_from_file_location("cryptic_epitope_annotation_09", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load postprocess module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fasta_record_id(header: str) -> str:
    return header.lstrip(">").split()[0]


def write_filtered_fasta(
    input_fasta: str | Path,
    output_fasta: str | Path,
    keep_ids: set[str],
) -> tuple[int, int]:
    total = 0
    kept = 0
    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    with open(input_fasta, "r") as in_fh, output_fasta.open("w") as out_fh:
        header: Optional[str] = None
        seq_lines: list[str] = []

        def flush_record() -> None:
            nonlocal total, kept
            if header is None:
                return
            total += 1
            if fasta_record_id(header) in keep_ids:
                kept += 1
                out_fh.write(header)
                if not header.endswith("\n"):
                    out_fh.write("\n")
                for seq_line in seq_lines:
                    out_fh.write(seq_line)
                    if not seq_line.endswith("\n"):
                        out_fh.write("\n")

        for line in in_fh:
            if line.startswith(">"):
                flush_record()
                header = line
                seq_lines = []
            else:
                seq_lines.append(line)
        flush_record()

    return total, kept


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate ORF annotation filter tables and an ORF-filtered aeSEP FASTA."
    )
    ap.add_argument("-s", "--sample", required=True, help="Sample ID")
    ap.add_argument("--sample-dir", required=True, help="Sample root directory")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory, usually SAMPLE/08-orf_filter")
    ap.add_argument(
        "--orf-annotation-dir",
        help="Directory containing ORF genome annotation outputs. If omitted, prefer "
        "SAMPLE/07-orf_genome_annotation and then SAMPLE/08-annotaions.",
    )
    ap.add_argument("--tx2gene", help="Override tx2gene TSV")
    ap.add_argument("--homer", help="Override HOMER block annotation TSV")
    ap.add_argument("--cds-overlap", help="Override CDS overlap TSV")
    ap.add_argument("--exon-overlap", help="Override exon overlap TSV")
    ap.add_argument("--ae-seps", help="Override input aeSEP peptide FASTA")
    ap.add_argument("--filtered-ae-seps", help="Override output ORF-filtered aeSEP FASTA")
    ap.add_argument(
        "--log-file",
        default="08-orf_filter.log",
        help="Log filename under --outdir, or an absolute/relative path",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    post = load_postprocess_module()

    sample = args.sample
    sample_dir = Path(args.sample_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.orf_annotation_dir:
        orf_annotation_dir = Path(args.orf_annotation_dir)
    else:
        new_dir = sample_dir / "07-orf_genome_annotation"
        legacy_dir = sample_dir / "08-annotaions"
        orf_annotation_dir = new_dir if new_dir.exists() else legacy_dir

    paths = post.default_paths(sample, sample_dir, orf_annotation_dir)
    tx2gene_path = Path(args.tx2gene) if args.tx2gene else paths["tx2gene"]
    homer_path = Path(args.homer) if args.homer else paths["homer"]
    cds_overlap_path = Path(args.cds_overlap) if args.cds_overlap else paths["cds_overlap"]
    exon_overlap_path = Path(args.exon_overlap) if args.exon_overlap else paths["exon_overlap"]
    ae_seps_path = Path(args.ae_seps) if args.ae_seps else paths["ae_seps"]
    filtered_fasta = (
        Path(args.filtered_ae_seps)
        if args.filtered_ae_seps
        else outdir / f"{sample}.aeSEPs.orf_filtered.pep"
    )

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = outdir / log_path

    for path, label in [
        (tx2gene_path, "tx2gene"),
        (homer_path, "HOMER block annotation"),
        (cds_overlap_path, "CDS overlap"),
        (exon_overlap_path, "exon overlap"),
        (ae_seps_path, "aeSEPs peptide FASTA"),
    ]:
        post.check_file(path, label)

    with log_path.open("w") as log_fh:
        def log_both(message: str) -> None:
            log(message)
            log_fh.write(f"[{ts()}] {message}\n")
            log_fh.flush()

        log_both("08-orf_filter.py started")
        log_both(f"sample: {sample}")
        log_both(f"sample_dir: {sample_dir}")
        log_both(f"outdir: {outdir}")
        log_both(f"orf_annotation_dir: {orf_annotation_dir}")

        tx2gene = post.load_tx2gene(tx2gene_path)
        orf_annto = post.load_homer_annotation(homer_path)
        orf_annto_kept = post.keep_orf_annotations(orf_annto)
        orf_annto_merged = post.collapse_orf_annotations(orf_annto_kept)
        orf_annto_merged = post.add_gene_names(orf_annto_merged, tx2gene)
        orf_annto_merged.to_csv(outdir / "orf_annto_merged.csv", index=False)
        log_both(f"orf_annto_merged rows: {len(orf_annto_merged)}")

        cds_overlap = post.load_intersect(cds_overlap_path)
        exon_overlap = post.load_intersect(exon_overlap_path)
        orf_final, removed = post.build_orf_final(orf_annto_merged, cds_overlap, exon_overlap)
        orf_final.to_csv(outdir / "orf_final.csv", index=False)
        removed.to_csv(outdir / "orf_removed.csv", index=False)
        log_both(f"orf_final rows: {len(orf_final)}")
        log_both(f"orf_removed rows: {len(removed)}")

        keep_ids = set(orf_final["TranscriptID"].astype(str))
        total_aeseps, kept_aeseps = write_filtered_fasta(ae_seps_path, filtered_fasta, keep_ids)
        log_both(f"aeSEPs FASTA records: {total_aeseps}")
        log_both(f"ORF-filtered aeSEPs records: {kept_aeseps}")
        log_both(f"filtered aeSEPs FASTA: {filtered_fasta}")

        summary = {
            "sample": sample,
            "sample_dir": str(sample_dir),
            "orf_annotation_dir": str(orf_annotation_dir),
            "orf_annto_merged_rows": int(len(orf_annto_merged)),
            "orf_final_rows": int(len(orf_final)),
            "orf_removed_rows": int(len(removed)),
            "aeseps_records": int(total_aeseps),
            "orf_filtered_aeseps_records": int(kept_aeseps),
            "filtered_aeseps_fasta": str(filtered_fasta),
        }
        with (outdir / "orf_filter_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)
            fh.write("\n")

        log_both("08-orf_filter.py finished")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
