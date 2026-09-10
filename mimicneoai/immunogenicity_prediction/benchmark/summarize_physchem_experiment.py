#!/usr/bin/env python3
"""Summarize Full, Zero and Physchem-only prediction tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)


CONDITIONS = ("full", "zero", "physchem_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def prediction_path(root: Path, antigen: str, condition: str, split: str) -> Path:
    return root / antigen / condition / "predictions" / f"{split}_predictions.tsv"


def read_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    if "positive_probability" not in frame.columns:
        for alias in ("immunogenicity_score", "score", "probability"):
            if alias in frame.columns:
                frame["positive_probability"] = frame[alias]
                break
    required = {"true_label", "positive_probability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "record_id" not in frame.columns:
        frame["record_id"] = np.arange(len(frame)).astype(str)
    return frame


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    n_positive = int(labels.sum())
    top = np.argsort(-scores, kind="stable")[:n_positive]
    return {
        "n": int(len(labels)),
        "n_positive": n_positive,
        "n_negative": int(len(labels) - n_positive),
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "auprc_trapezoid": float(auc(recall, precision)),
        "accuracy_at_0_5": float(accuracy_score(labels, scores >= 0.5)),
        "ppv_at_n_positive": float(labels[top].mean()),
    }


def paired_bootstrap(
    labels: np.ndarray,
    full: np.ndarray,
    zero: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    observed = metrics(labels, full)
    reference = metrics(labels, zero)
    keys = ("auroc", "average_precision", "auprc_trapezoid", "ppv_at_n_positive")
    samples = {key: [] for key in keys}
    attempts = 0
    while len(samples["auroc"]) < n_bootstrap:
        attempts += 1
        if attempts > n_bootstrap * 20:
            raise RuntimeError("Unable to obtain enough valid bootstrap resamples")
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if sampled_labels.min() == sampled_labels.max():
            continue
        sampled_full = metrics(sampled_labels, full[index])
        sampled_zero = metrics(sampled_labels, zero[index])
        for key in keys:
            samples[key].append(sampled_full[key] - sampled_zero[key])

    result = {}
    for key in keys:
        values = np.asarray(samples[key], dtype=float)
        result[key] = {
            "full": observed[key],
            "zero": reference[key],
            "delta_full_minus_zero": observed[key] - reference[key],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "bootstrap_probability_delta_gt_zero": float(np.mean(values > 0.0)),
        }
    return result


def standalone_inference(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
) -> dict:
    observed = metrics(labels, scores)
    keys = ("auroc", "average_precision", "auprc_trapezoid", "ppv_at_n_positive")
    rng = np.random.default_rng(seed)
    bootstrap = {key: [] for key in keys}
    attempts = 0
    while len(bootstrap["auroc"]) < n_bootstrap:
        attempts += 1
        if attempts > n_bootstrap * 20:
            raise RuntimeError("Unable to obtain enough valid standalone bootstrap resamples")
        index = rng.integers(0, len(labels), len(labels))
        sampled_labels = labels[index]
        if sampled_labels.min() == sampled_labels.max():
            continue
        sampled = metrics(sampled_labels, scores[index])
        for key in keys:
            bootstrap[key].append(sampled[key])

    inference = {
        "n": observed["n"],
        "n_positive": observed["n_positive"],
        "positive_prevalence": observed["n_positive"] / observed["n"],
        "metrics": {},
    }
    for key in keys:
        values = np.asarray(bootstrap[key], dtype=float)
        inference["metrics"][key] = {
            "estimate": observed[key],
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }

    null = {key: [] for key in ("auroc", "average_precision", "auprc_trapezoid")}
    for _ in range(n_permutations):
        permuted = rng.permutation(labels)
        permuted_metrics = metrics(permuted, scores)
        for key in null:
            null[key].append(permuted_metrics[key])
    for key, values in null.items():
        values_array = np.asarray(values, dtype=float)
        inference["metrics"][key]["permutation_p_one_sided"] = float(
            (1 + np.sum(values_array >= observed[key])) / (n_permutations + 1)
        )
        inference["metrics"][key]["permutation_null_mean"] = float(values_array.mean())
    return inference


def aligned_pair(full: pd.DataFrame, zero: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = ["record_id", "true_label", "positive_probability"]
    merged = full[columns].merge(
        zero[columns],
        on="record_id",
        how="inner",
        suffixes=("_full", "_zero"),
        validate="one_to_one",
    )
    if len(merged) != len(full) or len(merged) != len(zero):
        raise ValueError("Full and Zero predictions are not one-to-one aligned")
    if not np.array_equal(merged["true_label_full"], merged["true_label_zero"]):
        raise ValueError("Full and Zero labels differ after record alignment")
    return (
        merged["true_label_full"].to_numpy(dtype=int),
        merged["positive_probability_full"].to_numpy(dtype=float),
        merged["positive_probability_zero"].to_numpy(dtype=float),
    )


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    bootstrap = {}
    standalone = {}
    antigens = sorted(path.name for path in args.run_root.iterdir() if path.is_dir() and path.name != "shared_cache" and path.name != "logs")
    for antigen in antigens:
        predictions = {}
        for condition in CONDITIONS:
            path = prediction_path(args.run_root, antigen, condition, args.split)
            if not path.exists():
                continue
            frame = read_predictions(path)
            predictions[condition] = frame
            row = {
                "antigen": antigen,
                "condition": condition,
                "split": args.split,
                **metrics(
                    frame["true_label"].to_numpy(dtype=int),
                    frame["positive_probability"].to_numpy(dtype=float),
                ),
                "prediction_path": str(path),
            }
            rows.append(row)
            if condition == "physchem_only":
                standalone[antigen] = standalone_inference(
                    frame["true_label"].to_numpy(dtype=int),
                    frame["positive_probability"].to_numpy(dtype=float),
                    args.n_bootstrap,
                    args.n_permutations,
                    args.seed,
                )
        if "full" in predictions and "zero" in predictions:
            labels, full, zero = aligned_pair(predictions["full"], predictions["zero"])
            bootstrap[antigen] = paired_bootstrap(
                labels,
                full,
                zero,
                args.n_bootstrap,
                args.seed,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / f"physchem_{args.split}_metrics.tsv", sep="\t", index=False)
    (args.outdir / f"physchem_{args.split}_paired_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.outdir / f"physchem_{args.split}_standalone_inference.json").write_text(
        json.dumps(standalone, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(bootstrap, indent=2, sort_keys=True))
    print(json.dumps(standalone, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
