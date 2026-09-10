#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mimicneoai.immunogenicity_prediction.benchmark.metrics import write_metric_outputs


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight internal immunogenicity baselines.")
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", default=None, type=Path)
    parser.add_argument("--test-csv", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--baseline", required=True, choices=["aa_composition", "peptide_only", "hla_only", "full_phla"])
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--peptide-col", default="peptide")
    parser.add_argument("--hla-col", default="hla")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--c", default=1.0, type=float)
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--threshold", default=0.5, type=float)
    args = parser.parse_args()

    train_df = standardize(read_table(args.train_csv), args.peptide_col, args.hla_col, args.label_col)
    test_df = standardize(read_table(args.test_csv), args.peptide_col, args.hla_col, args.label_col)
    val_df = standardize(read_table(args.val_csv), args.peptide_col, args.hla_col, args.label_col) if args.val_csv else None

    estimator = build_estimator(args.baseline, c=args.c, class_weight=args.class_weight)
    x_train = make_input_table(train_df, args.baseline)
    estimator.fit(x_train, train_df["label"].astype(int))

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_predictions(
        estimator,
        test_df,
        args.outdir / "test_predictions.tsv",
        args.baseline,
    )
    metrics = write_metric_outputs(
        pd.read_csv(args.outdir / "test_predictions.tsv", sep="\t"),
        args.outdir,
        label_col="label",
        score_col="score",
        method=args.baseline,
        benchmark=args.benchmark,
        threshold=args.threshold,
        group_cols=["dataset_source", "source_type"],
    )
    if val_df is not None:
        val_outdir = args.outdir / "validation"
        val_outdir.mkdir(parents=True, exist_ok=True)
        write_predictions(estimator, val_df, val_outdir / "validation_predictions.tsv", args.baseline)
        write_metric_outputs(
            pd.read_csv(val_outdir / "validation_predictions.tsv", sep="\t"),
            val_outdir,
            label_col="label",
            score_col="score",
            method=args.baseline,
            benchmark=f"{args.benchmark}_validation",
            threshold=args.threshold,
            group_cols=["dataset_source", "source_type"],
        )

    with (args.outdir / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline": args.baseline,
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv) if args.val_csv else "",
                "test_csv": str(args.test_csv),
                "benchmark": args.benchmark,
                "c": args.c,
                "class_weight": args.class_weight,
                "test_metrics": metrics,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    print(metrics)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def standardize(df: pd.DataFrame, peptide_col: str, hla_col: str, label_col: str) -> pd.DataFrame:
    missing = [col for col in (peptide_col, hla_col, label_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")
    out = df.copy()
    out["peptide"] = out[peptide_col].astype(str).str.upper().str.strip()
    out["hla"] = out[hla_col].astype(str).str.strip()
    out["label"] = out[label_col].astype(int)
    return out


def make_input_table(df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if baseline == "aa_composition":
        rows = []
        for peptide in df["peptide"]:
            rows.append(peptide_features(peptide))
        return pd.DataFrame(rows)
    if baseline == "peptide_only":
        return pd.DataFrame({"peptide_text": df["peptide"]})
    if baseline == "hla_only":
        return pd.DataFrame({"hla_text": df["hla"]})
    if baseline == "full_phla":
        return pd.DataFrame({"peptide_text": df["peptide"], "hla_text": df["hla"]})
    raise ValueError(f"Unsupported baseline: {baseline}")


def peptide_features(peptide: str) -> dict[str, float]:
    peptide = peptide.upper()
    length = max(len(peptide), 1)
    row: dict[str, float] = {"length": float(len(peptide))}
    for aa in AMINO_ACIDS:
        row[f"aa_frac_{aa}"] = peptide.count(aa) / length
    return row


def build_estimator(baseline: str, c: float, class_weight: str) -> Pipeline:
    weight = None if class_weight == "none" else "balanced"
    clf = LogisticRegression(
        C=c,
        class_weight=weight,
        max_iter=5000,
        solver="liblinear",
        random_state=2026,
    )
    if baseline == "aa_composition":
        return Pipeline([("scale", StandardScaler()), ("clf", clf)])
    if baseline == "peptide_only":
        transform = ColumnTransformer(
            [("peptide", TfidfVectorizer(analyzer="char", ngram_range=(1, 3)), "peptide_text")]
        )
        return Pipeline([("features", transform), ("clf", clf)])
    if baseline == "hla_only":
        transform = ColumnTransformer([("hla", TfidfVectorizer(analyzer="char", ngram_range=(1, 4)), "hla_text")])
        return Pipeline([("features", transform), ("clf", clf)])
    if baseline == "full_phla":
        transform = ColumnTransformer(
            [
                ("peptide", TfidfVectorizer(analyzer="char", ngram_range=(1, 3)), "peptide_text"),
                ("hla", TfidfVectorizer(analyzer="char", ngram_range=(1, 4)), "hla_text"),
            ]
        )
        return Pipeline([("features", transform), ("clf", clf)])
    raise ValueError(f"Unsupported baseline: {baseline}")


def write_predictions(estimator: Pipeline, df: pd.DataFrame, out_path: Path, baseline: str) -> None:
    x = make_input_table(df, baseline)
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(x)[:, 1]
    else:
        raw = estimator.decision_function(x)
        scores = 1.0 / (1.0 + np.exp(-raw))
    out = df.copy()
    out["score"] = scores
    out["pred"] = (out["score"] >= 0.5).astype(int)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
