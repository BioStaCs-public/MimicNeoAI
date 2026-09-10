"""Pipeline-facing immunogenicity scoring workflow.

This module is intentionally thin: it builds peptide-HLA candidate tables from
completed binding outputs, runs the packaged source-specific immunogenicity
models on de-duplicated peptide-HLA pairs, and merges scores back to the
candidate rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from mimicneoai.functions.immunogenicity_runner import predict_default_immunogenicity_df
from mimicneoai.immunogenicity_prediction.runtime.hla import normalize_hla


IMMUNO_COLUMNS = [
    "immunogenicity_score",
    "immunogenicity_status",
    "input_qc_peptide_length",
    "input_qc_normalized_hla",
    "input_qc_flags",
    "input_qc_status",
    "input_qc_hla_reference_status",
    "immunogenicity_model_key",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument(
        "--antigen-class",
        required=True,
        choices=("mutation_derived", "mutation-derived", "cryptic", "microbial"),
    )
    parser.add_argument("--binding-dir", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-root", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_from_binding_dir(
        sample=args.sample,
        antigen_class=args.antigen_class,
        binding_dir=Path(args.binding_dir),
        outdir=Path(args.outdir),
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        model_root=args.model_root or None,
    )


def run_from_binding_dir(
    *,
    sample: str,
    antigen_class: str,
    binding_dir: Path,
    outdir: Path,
    device: str = "auto",
    batch_size: int = 512,
    workers: int = 1,
    model_root: str | None = None,
) -> int:
    outdir.mkdir(parents=True, exist_ok=True)
    source = normalize_antigen_class(antigen_class)

    if source == "mutation_derived":
        candidate = build_mutation_candidate_input(sample, binding_dir)
    else:
        candidate = build_nonmutation_candidate_input(sample, source, binding_dir)

    if candidate.skip_reason:
        write_summary(
            outdir / f"{sample}.immunogenicity.summary.json",
            sample=sample,
            antigen_class=source,
            binding_dir=binding_dir,
            outdir=outdir,
            skipped=True,
            skip_reason=candidate.skip_reason,
            input_rows=0,
            dedup_rows=0,
            status_counts={},
        )
        print(f"[immunogenicity] skipped {sample}: {candidate.skip_reason}", flush=True)
        return 0

    input_df = candidate.input_df.copy()
    input_path = outdir / f"{sample}.immunogenicity_input.tsv"
    input_df.to_csv(input_path, sep="\t", index=False)

    if input_df.empty:
        empty = input_df.copy()
        for col in IMMUNO_COLUMNS:
            if col not in empty.columns:
                empty[col] = ""
        write_standard_outputs(sample, source, outdir, empty, empty, candidate.source_df)
        write_summary(
            outdir / f"{sample}.immunogenicity.summary.json",
            sample=sample,
            antigen_class=source,
            binding_dir=binding_dir,
            outdir=outdir,
            skipped=False,
            skip_reason="",
            input_rows=0,
            dedup_rows=0,
            status_counts={},
        )
        return 0

    dedup_input = build_dedup_input(input_df, source)
    prediction_df = predict_default_immunogenicity_df(
        dedup_input,
        antigen_class=source,
        peptide_col="peptide",
        hla_col="hla",
        output_score_col="immunogenicity_score",
        batch_size=batch_size,
        device=device,
        num_processes=workers,
        include_input_qc=True,
        model_root=model_root,
    )
    prediction_df["immunogenicity_model_key"] = [
        make_model_key(source, peptide, hla)
        for peptide, hla in zip(prediction_df["peptide"], prediction_df["hla"])
    ]

    keep_prediction_cols = [
        "immunogenicity_model_key",
        "immunogenicity_score",
        "immunogenicity_status",
        "input_qc_peptide_length",
        "input_qc_normalized_hla",
        "input_qc_flags",
        "input_qc_status",
        "input_qc_hla_reference_status",
    ]
    prediction_output_cols = [
        col
        for col in ("antigen_class", "peptide", "hla", *keep_prediction_cols)
        if col in prediction_df.columns
    ]
    prediction_df[prediction_output_cols].to_csv(
        outdir / f"{sample}.immunogenicity_predictions.dedup.tsv",
        sep="\t",
        index=False,
    )

    annotated = input_df.merge(
        prediction_df[keep_prediction_cols],
        on="immunogenicity_model_key",
        how="left",
    )
    write_standard_outputs(sample, source, outdir, annotated, prediction_df, candidate.source_df)
    write_summary(
        outdir / f"{sample}.immunogenicity.summary.json",
        sample=sample,
        antigen_class=source,
        binding_dir=binding_dir,
        outdir=outdir,
        skipped=False,
        skip_reason="",
        input_rows=len(input_df),
        dedup_rows=len(prediction_df),
        status_counts=prediction_df["immunogenicity_status"].fillna("").value_counts().to_dict(),
    )
    return 0


class CandidateInput:
    def __init__(self, input_df: pd.DataFrame, source_df: pd.DataFrame | None = None, skip_reason: str = ""):
        self.input_df = input_df
        self.source_df = source_df
        self.skip_reason = skip_reason


def normalize_antigen_class(value: str) -> str:
    key = str(value).strip().lower()
    if key == "mutation-derived":
        return "mutation_derived"
    if key not in {"mutation_derived", "cryptic", "microbial"}:
        raise ValueError(f"Unsupported antigen class: {value}")
    return key


def build_mutation_candidate_input(sample: str, binding_dir: Path) -> CandidateInput:
    merged_path = find_mutation_merged(sample, binding_dir)
    if merged_path is None:
        return CandidateInput(pd.DataFrame(), skip_reason="mutation merged all_epitopes table not found")

    merged = read_table(merged_path)
    required = {"HLA Allele", "MT Epitope Seq"}
    missing = required.difference(merged.columns)
    if missing:
        return CandidateInput(pd.DataFrame(), skip_reason=f"mutation merged table missing columns: {sorted(missing)}")

    records = []
    for row_index, row in merged.iterrows():
        hla = clean_text(row.get("HLA Allele"))
        if not hla:
            continue
        for role, col in (("MT", "MT Epitope Seq"), ("WT", "WT Epitope Seq")):
            peptide = normalize_peptide(row.get(col))
            if not peptide:
                continue
            records.append(
                {
                    "sample": sample,
                    "antigen_class": "mutation_derived",
                    "source_row_index": row_index,
                    "peptide_role": role,
                    "upstream_record_id": mutation_record_id(row, row_index),
                    "gene_name": clean_text(row.get("Gene Name")),
                    "transcript": clean_text(row.get("Transcript")),
                    "variant_type": clean_text(row.get("Variant Type")),
                    "mutation": clean_text(row.get("Mutation")),
                    "peptide": peptide,
                    "hla": hla,
                    "mhc_class": infer_mhc_class(hla, row.get("Peptide Length"), peptide),
                    "immunogenicity_model_key": make_model_key("mutation_derived", peptide, hla),
                    "immunogenicity_record_key": make_record_key(
                        "mutation_derived",
                        sample,
                        mutation_record_id(row, row_index),
                        role,
                        peptide,
                        hla,
                    ),
                }
            )
    return CandidateInput(pd.DataFrame.from_records(records), source_df=merged)


def build_nonmutation_candidate_input(sample: str, antigen_class: str, binding_dir: Path) -> CandidateInput:
    root = resolve_nonmutation_binding_root(sample, binding_dir)
    summary = read_json(root / f"{sample}.mimicneoai_binding.summary.json")
    if summary.get("prediction_skipped"):
        return CandidateInput(pd.DataFrame(), skip_reason=str(summary.get("skip_reason") or "binding prediction skipped"))

    merged_path = find_nonmutation_merged(sample, root)
    prediction_path = root / "mimicneoai_binding_predictions" / "binding_predictions.long.tsv"
    if merged_path is None and not prediction_path.exists():
        return CandidateInput(pd.DataFrame(), skip_reason="binding prediction output not found")

    if merged_path is not None:
        merged = read_table(merged_path)
        peptide_col = "Epitope Seq"
        hla_col = "HLA Allele"
        if peptide_col not in merged.columns or hla_col not in merged.columns:
            return CandidateInput(pd.DataFrame(), skip_reason="merged table lacks Epitope Seq/HLA Allele columns")
        records = []
        for row_index, row in merged.iterrows():
            peptide = normalize_peptide(row.get(peptide_col))
            hla = clean_text(row.get(hla_col))
            if not peptide or not hla:
                continue
            upstream_id = clean_text(row.get("Mutation")) or str(row_index)
            records.append(
                {
                    "sample": sample,
                    "antigen_class": antigen_class,
                    "source_row_index": row_index,
                    "upstream_record_id": upstream_id,
                    "peptide": peptide,
                    "hla": hla,
                    "mhc_class": infer_mhc_class(hla, row.get("Peptide Length"), peptide),
                    "immunogenicity_model_key": make_model_key(antigen_class, peptide, hla),
                    "immunogenicity_record_key": make_record_key(
                        antigen_class,
                        sample,
                        upstream_id,
                        "candidate",
                        peptide,
                        hla,
                    ),
                }
            )
        return CandidateInput(pd.DataFrame.from_records(records), source_df=merged)

    predictions = read_table(prediction_path)
    ok = predictions[predictions.get("status", "").fillna("").astype(str).eq("ok")].copy()
    ok["peptide"] = ok["peptide"].map(normalize_peptide)
    ok["hla"] = ok["hla_allele"].map(clean_text)
    ok = ok[(ok["peptide"] != "") & (ok["hla"] != "")]
    rows = ok[["peptide", "hla", "mhc_class"]].drop_duplicates().copy()
    rows.insert(0, "sample", sample)
    rows.insert(1, "antigen_class", antigen_class)
    rows["source_row_index"] = ""
    rows["upstream_record_id"] = ""
    rows["immunogenicity_model_key"] = [
        make_model_key(antigen_class, peptide, hla)
        for peptide, hla in zip(rows["peptide"], rows["hla"])
    ]
    rows["immunogenicity_record_key"] = [
        make_record_key(antigen_class, sample, "", "candidate", peptide, hla)
        for peptide, hla in zip(rows["peptide"], rows["hla"])
    ]
    return CandidateInput(rows.reset_index(drop=True))


def build_dedup_input(input_df: pd.DataFrame, antigen_class: str) -> pd.DataFrame:
    cols = ["immunogenicity_model_key", "peptide", "hla"]
    dedup = input_df[cols].drop_duplicates("immunogenicity_model_key").copy()
    dedup.insert(1, "antigen_class", antigen_class)
    return dedup.reset_index(drop=True)


def write_standard_outputs(
    sample: str,
    antigen_class: str,
    outdir: Path,
    annotated: pd.DataFrame,
    prediction_df: pd.DataFrame,
    source_df: pd.DataFrame | None,
) -> None:
    annotated_path = outdir / f"{sample}.immunogenicity_annotated.tsv"
    annotated.to_csv(annotated_path, sep="\t", index=False)

    if antigen_class == "mutation_derived":
        mt = annotated[annotated.get("peptide_role", "").eq("MT")].copy()
        wt = annotated[annotated.get("peptide_role", "").eq("WT")].copy()
        mt.to_csv(outdir / f"{sample}.MT.immunogenicity_candidates.tsv", sep="\t", index=False)
        wt.to_csv(outdir / f"{sample}.WT.immunogenicity_sidecar.tsv", sep="\t", index=False)
        if source_df is not None and not source_df.empty:
            write_mutation_merged_with_scores(sample, outdir, source_df, annotated)


def write_mutation_merged_with_scores(
    sample: str,
    outdir: Path,
    merged: pd.DataFrame,
    annotated: pd.DataFrame,
) -> None:
    out = merged.copy()
    score_cols = [
        "immunogenicity_score",
        "immunogenicity_status",
        "input_qc_status",
        "input_qc_hla_reference_status",
        "immunogenicity_model_key",
    ]
    for role in ("MT", "WT"):
        subset = annotated[annotated["peptide_role"].eq(role)].copy()
        if subset.empty:
            for col in score_cols:
                out[f"{role} {title_col(col)}"] = ""
            continue
        subset = subset.drop_duplicates("source_row_index", keep="first").set_index("source_row_index")
        for col in score_cols:
            out[f"{role} {title_col(col)}"] = out.index.map(subset[col]).fillna("")
    out.to_csv(outdir / f"{sample}.merged.with_immunogenicity.tsv", sep="\t", index=False)


def title_col(value: str) -> str:
    return value.replace("_", " ").title().replace("Hla", "HLA")


def find_mutation_merged(sample: str, binding_dir: Path) -> Path | None:
    candidates = [
        binding_dir / "04_merged_epitopes" / f"{sample}.merged.all_epitopes.tsv",
        binding_dir / "combined" / f"{sample}.merged.all_epitopes.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(binding_dir.rglob(f"{sample}.merged.all_epitopes.tsv"), key=lambda p: len(p.parts))
    return matches[0] if matches else None


def resolve_nonmutation_binding_root(sample: str, binding_dir: Path) -> Path:
    nested = binding_dir / sample
    if (nested / f"{sample}.mimicneoai_binding.summary.json").exists() or (nested / "combined").exists():
        return nested
    return binding_dir


def find_nonmutation_merged(sample: str, root: Path) -> Path | None:
    candidates = [
        root / "combined" / f"{sample}.merged.all_epitopes.tsv",
        root / f"{sample}.merged.all_epitopes.tsv",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.rglob(f"{sample}.merged.all_epitopes.tsv"), key=lambda p: len(p.parts))
    return matches[0] if matches else None


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def read_json(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_summary(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(json_ready(values), handle, indent=2, ensure_ascii=False)


def json_ready(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def normalize_peptide(value: object) -> str:
    return clean_text(value).upper()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def make_model_key(antigen_class: str, peptide: object, hla: object) -> str:
    peptide_text = normalize_peptide(peptide)
    hla_text = normalize_hla(clean_text(hla))
    return f"{antigen_class}|{hla_text}|{peptide_text}"


def make_record_key(
    antigen_class: str,
    sample: str,
    upstream_record_id: str,
    role: str,
    peptide: object,
    hla: object,
) -> str:
    return "|".join(
        [
            antigen_class,
            sample,
            str(upstream_record_id),
            role,
            normalize_hla(clean_text(hla)),
            normalize_peptide(peptide),
        ]
    )


def mutation_record_id(row: pd.Series, row_index: int) -> str:
    values = [
        clean_text(row.get("Chromosome")),
        clean_text(row.get("Start")),
        clean_text(row.get("Reference")),
        clean_text(row.get("Variant")),
        clean_text(row.get("Transcript")),
        clean_text(row.get("HGVSc")),
        clean_text(row.get("HGVSp")),
    ]
    joined = ":".join(item for item in values if item)
    return joined or str(row_index)


def infer_mhc_class(hla: str, length_value: object, peptide: str) -> str:
    hla_text = clean_text(hla).replace("HLA-", "")
    if hla_text.startswith(("DR", "DQ", "DP", "DRA", "DRB", "DQA", "DQB", "DPA", "DPB")):
        return "MHC-II"
    try:
        length = int(float(clean_text(length_value)))
    except ValueError:
        length = len(peptide)
    return "MHC-I" if length <= 11 else "MHC-II"


if __name__ == "__main__":
    raise SystemExit(main())
