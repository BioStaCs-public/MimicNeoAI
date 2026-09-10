#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
09-cryptic_epitope_annotation.py

Postprocess cryptic ORF genome annotation and merge pVACbind epitopes with
expression, ORF annotation, SEP/CDS sequence, and extended peptide sequence.

This script wraps the first three parts of the former Postprocess.ipynb:
  1) load sample-level annotation inputs,
  2) build ORF-level genome annotation filters,
  3) merge filtered cryptic epitopes and generate extended peptides.

Example (sanitized):
  python 09-cryptic_epitope_annotation.py \
    -s SAMPLE_ID \
    --sample-dir /path/to/Cryptic/SAMPLE_ID \
    -o /path/to/Cryptic/SAMPLE_ID/09-CrypticEpitopeCandidates \
    --orf-annotation-dir /path/to/Cryptic/SAMPLE_ID/08-annotations \
    --extended-length 27
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


HOMER_COLS = [
    "PeakID",
    "Chr",
    "Start",
    "End",
    "Strand",
    "Peak Score",
    "Focus Ratio/Region Size",
    "Annotation",
    "Detailed Annotation",
    "Distance to TSS",
    "Nearest PromoterID",
    "Entrez ID",
    "Nearest Unigene",
    "Nearest Refseq",
    "Nearest Ensembl",
    "Gene Name",
    "Gene Alias",
    "Gene Description",
    "Gene Type",
]

BED12_COLS = [
    "a_chr",
    "a_start",
    "a_end",
    "a_name",
    "a_score",
    "a_strand",
    "a_thickStart",
    "a_thickEnd",
    "a_itemRgb",
    "a_blockCount",
    "a_blockSizes",
    "a_blockStarts",
]

GTF_COLS = [
    "b_chr",
    "b_source",
    "b_feature",
    "b_start",
    "b_end",
    "b_score",
    "b_strand",
    "b_frame",
    "b_attr",
]

INTERSECT_COLS = BED12_COLS + GTF_COLS


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


def default_paths(sample: str, sample_dir: str | Path, orf_annotation_dir: str | Path) -> dict[str, Path]:
    root = Path(sample_dir)
    orf_dir = Path(orf_annotation_dir)
    return {
        "aberrant_annot": root / "06-aeSEPs" / f"{sample}.aberrant_noncoding.annot.csv",
        "tx2gene": root / "03-novel" / "04a.trace_to_ref" / "tx2gene.tsv",
        "homer": orf_dir / "orf.noUnmap.noSup.blocks.homer.tsv",
        "cds_overlap": orf_dir / "cds.overlap.orf_gtf.noUnmap.noSup.tsv",
        "exon_overlap": orf_dir / "exon.overlap.orf_gtf.noUnmap.noSup.tsv",
        "all_epitopes": root
        / "07-hla_binding_pred"
        / sample
        / "pvacbind"
        / "combined"
        / f"{sample}.merged.all_epitopes.tsv",
        "ae_seps": root / "06-aeSEPs" / f"{sample}.aeSEPs.pep",
        "known_cds": root
        / "02-known"
        / "05.transdecoder_orf"
        / f"{sample}.lncRNA.TransDecoder.LongOrfs"
        / "longest_orfs.cds",
        "novel_cds": root
        / "03-novel"
        / "05.transdecoder_orf"
        / f"{sample}.lncRNA.TransDecoder.LongOrfs"
        / "longest_orfs.cds",
    }


def load_aberrant_annotation(path: str | Path) -> pd.DataFrame:
    annot = pd.read_csv(path, low_memory=False)
    annot = annot.rename(columns={"Name": "Transcript"})
    annot = annot[
        [
            "Transcript",
            "TPM_tumor",
            "TPM_ctrl",
            "is_trinity",
            "tri_ref_tx",
            "enst_tag",
            "nc_class",
            "log2FC",
            "is_aberrant",
        ]
    ]
    return annot


def load_tx2gene(path: str | Path) -> pd.DataFrame:
    tx2gene = pd.read_csv(path, sep="\t", header=None)
    tx2gene.columns = ["enst_tag", "gene_id", "gene_name"]
    return tx2gene


def load_homer_annotation(path: str | Path) -> pd.DataFrame:
    orf_annto = pd.read_csv(path, sep="\t")
    orf_annto.columns = HOMER_COLS
    orf_annto = orf_annto[
        ["PeakID", "Chr", "Start", "End", "Strand", "Peak Score", "Annotation", "Detailed Annotation"]
    ].copy()

    pat = r"(ENSG\d+\.\d+)"
    orf_annto["ENSG_A"] = orf_annto["Annotation"].astype(str).str.extract(pat, expand=False)
    orf_annto["ENSG_D"] = orf_annto["Detailed Annotation"].astype(str).str.extract(pat, expand=False)

    mismatch = (
        orf_annto["ENSG_A"].notna()
        & orf_annto["ENSG_D"].notna()
        & (orf_annto["ENSG_A"] != orf_annto["ENSG_D"])
    )
    log(f"ENSG mismatch rows: {int(mismatch.sum())}")

    orf_annto["ENSG_ID"] = orf_annto["ENSG_A"].fillna(orf_annto["ENSG_D"])
    orf_annto.loc[mismatch, "ENSG_ID"] = (
        orf_annto.loc[mismatch, "ENSG_A"] + "," + orf_annto.loc[mismatch, "ENSG_D"]
    )
    orf_annto = orf_annto.drop(columns=["ENSG_A", "ENSG_D"])
    return orf_annto


def keep_orf_annotations(orf_annto: pd.DataFrame) -> pd.DataFrame:
    # 1) TranscriptID: strip trailing "-<digits>" from PeakID (HOMER multi-hit suffix)
    orf_annto = orf_annto.copy()
    orf_annto["TranscriptID"] = orf_annto["PeakID"].astype(str).str.replace(r"-\d+$", "", regex=True)

    # 2) Detailed tag (text before parentheses)
    det_tag = orf_annto["Detailed Annotation"].astype(str).str.replace(r"\s*\(.*$", "", regex=True).str.strip()

    # 3) branch-specific KEEP lists
    keep_novel = {
        "protein_coding-intron",
        "Intergenic",
        "lincRNA-intron",
        "antisense-intron",
        "lincRNA-exon",
    }
    keep_known = {
        "protein_coding-intron",
        "antisense-intron",
        "lincRNA-intron",
        "lincRNA-exon",
    }

    is_novel_tid = orf_annto["TranscriptID"].astype(str).str.startswith("TRINITY_")
    is_na = det_tag.isin(["NA", "nan", "NaN", "None", ""]) | det_tag.isna()

    allowed = np.where(is_novel_tid, det_tag.isin(keep_novel), det_tag.isin(keep_known))
    allowed = pd.Series(allowed, index=orf_annto.index)

    keep_map = {}
    for tid, idx in tqdm(orf_annto.groupby("TranscriptID", sort=False).groups.items(), desc="ORF keep map"):
        annotated_mask = ~is_na.loc[idx]

        if not annotated_mask.any():
            keep_map[tid] = False
            continue

        keep_map[tid] = bool(allowed.loc[idx][annotated_mask].all())

    keep_transcript = pd.Series(keep_map)
    orf_annto["keep_transcript"] = orf_annto["TranscriptID"].map(keep_transcript).fillna(False)

    log(f"Total TranscriptID: {orf_annto['TranscriptID'].nunique()}")
    log(f"Kept TranscriptID : {int(keep_transcript.sum())}")
    log(f"Dropped TranscriptID: {int((~keep_transcript).sum())}")

    return orf_annto[orf_annto["keep_transcript"]].copy()


def drop_ensg_and_ord(series: pd.Series) -> pd.Series:
    x = series.astype(str)
    x = x.replace({"NA": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
    x = x.str.replace(r"\s*\(ENSG\d+\.\d+[^)]*\)", "", regex=True).str.strip()
    return x


def agg_pipe_keep_na(s: pd.Series) -> str:
    parts = []
    for v in s:
        if pd.isna(v):
            parts.append("")
        else:
            parts.append(str(v).strip())
    return "|".join(parts)


def collapse_orf_annotations(orf_annto_kept: pd.DataFrame) -> pd.DataFrame:
    df = orf_annto_kept.copy()
    cols = ["Chr", "Start", "End", "Strand", "Peak Score", "Annotation", "Detailed Annotation", "ENSG_ID"]

    df["Annotation"] = drop_ensg_and_ord(df["Annotation"])
    df["Detailed Annotation"] = drop_ensg_and_ord(df["Detailed Annotation"])

    for col in ["Start", "End", "Peak Score"]:
        if col in df.columns:
            df[col] = df[col].astype("Int64").astype(str).replace("<NA>", "")

    return df.groupby("TranscriptID", sort=False)[cols].agg(agg_pipe_keep_na).reset_index()


def add_gene_names(orf_annto_merged: pd.DataFrame, tx2gene: pd.DataFrame) -> pd.DataFrame:
    tx_map = tx2gene[["gene_id", "gene_name"]].drop_duplicates().copy()
    tx_map["gene_id_nov"] = tx_map["gene_id"].astype(str).str.replace(r"\.\d+$", "", regex=True)
    gid2name = dict(zip(tx_map["gene_id_nov"], tx_map["gene_name"]))

    split_pat = re.compile(r"([|,]+)")

    def map_ensg_to_name(ensg_str):
        if pd.isna(ensg_str):
            return ""
        s = str(ensg_str).strip()
        if s == "" or s.lower() in {"na", "nan", "none"}:
            return ""
        parts = split_pat.split(s)
        out = []
        for part in parts:
            if part == "" or part is None:
                continue
            if split_pat.fullmatch(part):
                out.append(part)
            else:
                tok = part.strip()
                if tok == "" or tok.lower() in {"na", "nan", "none"}:
                    out.append("")
                    continue
                tok_nov = re.sub(r"\.\d+$", "", tok)
                out.append(gid2name.get(tok_nov, ""))
        return "".join(out)

    out = orf_annto_merged.copy()
    out["gene_name"] = out["ENSG_ID"].apply(map_ensg_to_name)
    return out


def load_intersect(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None)
    df.columns = INTERSECT_COLS
    return df


def build_orf_final(
    orf_annto_merged: pd.DataFrame,
    cds_overlap: pd.DataFrame,
    exon_overlap: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = set(orf_annto_merged["TranscriptID"].astype(str))
    cds_names = set(cds_overlap["a_name"].astype(str))
    exon_names = set(exon_overlap["a_name"].astype(str))

    tx_in_cds = tx & cds_names
    tx_in_exon = tx & exon_names
    tx_in_any = tx & (cds_names | exon_names)

    log(f"Total TranscriptID in orf_annto_merged: {len(tx)}")
    log(f"TranscriptID overlapping CDS (cds_overlap a_name): {len(tx_in_cds)}")
    log(f"TranscriptID overlapping exon (exon_overlap a_name): {len(tx_in_exon)}")
    log(f"TranscriptID overlapping CDS or exon: {len(tx_in_any)}")
    log(f"Overlap both CDS and exon: {len(tx_in_cds & tx_in_exon)}")

    sub = orf_annto_merged[orf_annto_merged["TranscriptID"].isin(tx_in_exon)].copy()
    da = sub["Detailed Annotation"].fillna("").astype(str)

    rule_empty = da.str.contains(r"^\|") | da.str.contains(r"\|$") | da.str.contains(r"\|\|")
    rule_intergenic_any = da.str.contains(r"(?<!\w)Intergenic(?!\w)")

    sub["drop_reason"] = ""
    sub.loc[rule_empty, "drop_reason"] += "empty_seg|"
    sub.loc[rule_intergenic_any, "drop_reason"] += "has_intergenic|"
    sub["drop_exon_ctx"] = sub["drop_reason"].ne("")

    tx_drop_exon_ctx = set(sub.loc[sub["drop_exon_ctx"], "TranscriptID"].astype(str))

    log(f"Exon-overlap TranscriptIDs: {len(tx_in_exon)}")
    log(f"Drop by exon-context rules: {len(tx_drop_exon_ctx)}")

    tx_drop = set(tx_in_cds) | set(tx_drop_exon_ctx)
    log(f"Final drop TranscriptIDs (CDS overlap + exon-context): {len(tx_drop)}")

    orf_final = orf_annto_merged[~orf_annto_merged["TranscriptID"].astype(str).isin(tx_drop)].copy()
    log(f"Final kept TranscriptIDs: {orf_final['TranscriptID'].nunique()}")

    removed = orf_annto_merged[orf_annto_merged["TranscriptID"].astype(str).isin(tx_drop)].copy()
    removed["removed_by"] = ""
    removed.loc[removed["TranscriptID"].astype(str).isin(tx_in_cds), "removed_by"] += "CDS_overlap|"
    removed.loc[removed["TranscriptID"].astype(str).isin(tx_drop_exon_ctx), "removed_by"] += "exon_ctx|"
    return orf_final, removed


def read_pep_to_df(path: str, remove_trailing_star: bool = True) -> pd.DataFrame:
    ids: List[str] = []
    seqs: List[str] = []

    header = None
    seq_lines: List[str] = []

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    seq = "".join(seq_lines)
                    if remove_trailing_star and seq.endswith("*"):
                        seq = seq[:-1]
                    ids.append(header.split()[0].lstrip(">"))
                    seqs.append(seq)
                header = line
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            seq = "".join(seq_lines)
            if remove_trailing_star and seq.endswith("*"):
                seq = seq[:-1]
            ids.append(header.split()[0].lstrip(">"))
            seqs.append(seq)

    return pd.DataFrame({"ID": ids, "SEP": seqs})


def extend_peptide(df: pd.DataFrame, extended_length: int) -> pd.DataFrame:
    extended_peptides = []
    for i in tqdm(range(df.shape[0]), desc="Extend peptides"):
        epitope_seq = df["Epitope Seq"].iloc[i]
        query_seq = df["SEP"].iloc[i]
        if len(query_seq) <= extended_length:
            extended_peptide = query_seq
        else:
            left_extened = right_extened = int((extended_length - len(epitope_seq)) / 2)
            left_pos = query_seq.index(epitope_seq) - left_extened
            right_pos = query_seq.index(epitope_seq) + len(epitope_seq) - 1 + right_extened
            if left_pos < 0:
                right_pos = right_pos + abs(left_pos)
                left_pos = 0
            elif right_pos > len(query_seq) - 1:
                left_pos = left_pos - (right_pos - (len(query_seq) - 1))
                right_pos = len(query_seq) - 1
            extended_peptide = query_seq[left_pos : right_pos + 1]
        extended_peptides.append(extended_peptide)

    df_extended = df.copy()
    df_extended["Extended Peptide"] = extended_peptides
    df_extended["Extended Length"] = [len(seq) for seq in df_extended["Extended Peptide"].tolist()]
    return df_extended


def merge_epitopes(
    sample: str,
    annot: pd.DataFrame,
    orf_final: pd.DataFrame,
    all_epitopes_path: str | Path,
    ae_seps_path: str | Path,
    known_cds_path: str | Path,
    novel_cds_path: str | Path,
    outdir: str | Path,
    extended_length: int,
) -> pd.DataFrame:
    outdir = Path(outdir)
    all_epitopes_raw = pd.read_csv(all_epitopes_path, sep="\t", low_memory=False)
    filtered_raw = all_epitopes_raw[all_epitopes_raw["Mutation"].isin(orf_final["TranscriptID"])].copy()
    filtered_raw.to_csv(outdir / f"{sample}.merged.all_epitopes.filtered.tsv", index=False, sep="\t")

    all_epitopes = all_epitopes_raw.rename(columns={"Mutation": "ID"})
    all_epitopes["Transcript"] = all_epitopes["ID"].astype(str).str.replace(r"\.p\d+$", "", regex=True)

    all_epitopes_annot = all_epitopes.merge(annot)
    all_epitopes_annot_filt = all_epitopes_annot[all_epitopes_annot["ID"].isin(orf_final["TranscriptID"])]
    all_epitopes_annot_filt = all_epitopes_annot_filt.merge(
        orf_final, left_on="ID", right_on="TranscriptID", how="left"
    )

    colnames = [
        "ID",
        "HLA Allele",
        "Sub-peptide Position",
        "Epitope Seq",
        "Transcript",
        "TPM_tumor",
        "TPM_ctrl",
        "log2FC",
        "Median IC50 Score",
        "Best IC50 Score",
        "Best IC50 Score Method",
        "Median Percentile",
        "Best Percentile",
        "Best Percentile Method",
        "Chr",
        "Start",
        "End",
        "Strand",
        "Peak Score",
        "Annotation",
        "Detailed Annotation",
        "ENSG_ID",
        "gene_name",
    ]
    cryptic_epitopes_annot = all_epitopes_annot_filt[colnames].copy()
    cryptic_epitopes_annot.to_csv(outdir / "cryptic_epitopes_annot.csv", index=False)

    ae_seps_df = read_pep_to_df(str(ae_seps_path))
    cds_df = read_pep_to_df(str(novel_cds_path)).rename(columns={"SEP": "CDS"})
    cds_df = pd.concat([cds_df, read_pep_to_df(str(known_cds_path)).rename(columns={"SEP": "CDS"})])

    sep_map = ae_seps_df[["ID", "SEP"]].drop_duplicates(subset=["ID"])
    cds_map = cds_df[["ID", "CDS"]].drop_duplicates(subset=["ID"])

    cryptic_epitopes_annot_merged = (
        cryptic_epitopes_annot.merge(sep_map, on="ID", how="left").merge(cds_map, on="ID", how="left")
    )
    cryptic_epitopes_annot_merged.to_csv(outdir / "cryptic_epitopes_annot_merged.csv", index=False)

    cryptic_epitopes_annot_extend = extend_peptide(cryptic_epitopes_annot_merged, extended_length)
    cryptic_epitopes_annot_extend.to_csv(outdir / "cryptic_epitopes_annot_extend.csv", index=False)

    log(f"Filtered pVACbind rows: {len(filtered_raw)} -> {sample}.merged.all_epitopes.filtered.tsv")
    log(f"cryptic_epitopes_annot rows: {len(cryptic_epitopes_annot)}")
    log(f"cryptic_epitopes_annot_extend rows: {len(cryptic_epitopes_annot_extend)}")
    return cryptic_epitopes_annot_extend


def parse_args():
    ap = argparse.ArgumentParser(
        description="Postprocess cryptic ORF annotation and prepare annotated epitope table for step 10."
    )
    ap.add_argument("-s", "--sample", required=True, help="Sample ID")
    ap.add_argument("--sample-dir", required=True, help="Sample root directory containing 06-aeSEPs and 07-hla_binding_pred")
    ap.add_argument("-o", "--outdir", required=True, help="Output directory, usually SAMPLE/09-CrypticEpitopeCandidates")
    ap.add_argument(
        "--orf-annotation-dir",
        help="Directory from step 08 containing HOMER, CDS overlap, and exon overlap outputs",
    )
    ap.add_argument("--aberrant-annot", help="Override aberrant noncoding annotation CSV")
    ap.add_argument("--tx2gene", help="Override tx2gene TSV")
    ap.add_argument("--homer", help="Override HOMER block annotation TSV")
    ap.add_argument("--cds-overlap", help="Override CDS overlap TSV")
    ap.add_argument("--exon-overlap", help="Override exon overlap TSV")
    ap.add_argument("--all-epitopes", help="Override pVACbind merged all_epitopes TSV")
    ap.add_argument("--ae-seps", help="Override aeSEPs peptide FASTA")
    ap.add_argument("--known-cds", help="Override known TransDecoder CDS FASTA")
    ap.add_argument("--novel-cds", help="Override novel TransDecoder CDS FASTA")
    ap.add_argument("--extended-length", type=int, default=27, help="Extended peptide length")
    ap.add_argument(
        "--log-file",
        default="09-cryptic_epitope_annotation.log",
        help="Log filename under --outdir, or an absolute/relative path",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    sample = args.sample
    sample_dir = Path(args.sample_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    if args.extended_length <= 0:
        raise ValueError("[ERR] --extended-length must be positive")

    orf_annotation_dir = Path(args.orf_annotation_dir) if args.orf_annotation_dir else sample_dir / "08-annotaions"
    paths = default_paths(sample, sample_dir, orf_annotation_dir)

    aberrant_annot = Path(args.aberrant_annot) if args.aberrant_annot else paths["aberrant_annot"]
    tx2gene_path = Path(args.tx2gene) if args.tx2gene else paths["tx2gene"]
    homer_path = Path(args.homer) if args.homer else paths["homer"]
    cds_overlap_path = Path(args.cds_overlap) if args.cds_overlap else paths["cds_overlap"]
    exon_overlap_path = Path(args.exon_overlap) if args.exon_overlap else paths["exon_overlap"]
    all_epitopes_path = Path(args.all_epitopes) if args.all_epitopes else paths["all_epitopes"]
    ae_seps_path = Path(args.ae_seps) if args.ae_seps else paths["ae_seps"]
    known_cds_path = Path(args.known_cds) if args.known_cds else paths["known_cds"]
    novel_cds_path = Path(args.novel_cds) if args.novel_cds else paths["novel_cds"]

    for path, label in [
        (aberrant_annot, "aberrant noncoding annotation"),
        (tx2gene_path, "tx2gene"),
        (homer_path, "HOMER block annotation"),
        (cds_overlap_path, "CDS overlap"),
        (exon_overlap_path, "exon overlap"),
        (all_epitopes_path, "pVACbind merged all_epitopes"),
        (ae_seps_path, "aeSEPs peptide FASTA"),
        (known_cds_path, "known TransDecoder CDS"),
        (novel_cds_path, "novel TransDecoder CDS"),
    ]:
        check_file(path, label)

    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = outdir / log_path
    ensure_dir(log_path.parent)

    with log_path.open("w") as log_fh:
        def log_both(message: str) -> None:
            log(message)
            log_fh.write(f"[{ts()}] {message}\n")
            log_fh.flush()

        log_both("09-cryptic_epitope_annotation.py started")
        log_both(f"sample: {sample}")
        log_both(f"sample_dir: {sample_dir}")
        log_both(f"outdir: {outdir}")
        log_both(f"orf_annotation_dir: {orf_annotation_dir}")

        annot = load_aberrant_annotation(aberrant_annot)
        tx2gene = load_tx2gene(tx2gene_path)
        log_both(f"annotation rows: {len(annot)}")
        log_both(f"tx2gene rows: {len(tx2gene)}")

        orf_annto = load_homer_annotation(homer_path)
        orf_annto_kept = keep_orf_annotations(orf_annto)
        orf_annto_merged = collapse_orf_annotations(orf_annto_kept)
        orf_annto_merged = add_gene_names(orf_annto_merged, tx2gene)
        orf_annto_merged.to_csv(outdir / "orf_annto_merged.csv", index=False)
        log_both(f"orf_annto_merged rows: {len(orf_annto_merged)}")

        cds_overlap = load_intersect(cds_overlap_path)
        exon_overlap = load_intersect(exon_overlap_path)
        orf_final, removed = build_orf_final(orf_annto_merged, cds_overlap, exon_overlap)
        orf_final.to_csv(outdir / "orf_final.csv", index=False)
        removed.to_csv(outdir / "orf_removed.csv", index=False)
        log_both(f"orf_final rows: {len(orf_final)}")
        log_both(f"orf_removed rows: {len(removed)}")

        cryptic_epitopes_annot_extend = merge_epitopes(
            sample=sample,
            annot=annot,
            orf_final=orf_final,
            all_epitopes_path=all_epitopes_path,
            ae_seps_path=ae_seps_path,
            known_cds_path=known_cds_path,
            novel_cds_path=novel_cds_path,
            outdir=outdir,
            extended_length=args.extended_length,
        )
        log_both(f"final step-09 epitope rows: {len(cryptic_epitopes_annot_extend)}")
        log_both("09-cryptic_epitope_annotation.py finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"{exc}\n")
        sys.exit(1)
