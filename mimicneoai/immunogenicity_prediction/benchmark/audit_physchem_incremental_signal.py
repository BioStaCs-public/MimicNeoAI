#!/usr/bin/env python3
"""Audit whether peptide physicochemical features add signal beyond a base model.

This is a development-set diagnostic. It performs peptide-grouped out-of-fold
meta-modeling and never needs to read a held-out test set.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCALAR_COLUMNS = [
    "polarity",
    "volume",
    "net_charge",
    "hydrophobicity",
    "boman_index",
    "aliphatic_index",
    "isoelectric_point",
]
MODEL_COLUMNS = [
    "base_score",
    "physchem_logistic",
    "combined_logistic",
    "combined_logistic_shuffled",
    "physchem_mlp",
    "combined_mlp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--folds", default=5, type=int)
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--bootstrap", default=2000, type=int)
    parser.add_argument("--jobs", default=1, type=int)
    return parser.parse_args()


def parse_physchem_features(table: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    composition = table["aa_composition"].astype(str).str.split(",")
    lengths = composition.str.len()
    if lengths.nunique() != 1 or int(lengths.iloc[0]) != 18:
        raise ValueError(f"Expected 18 amino-acid composition values; found {lengths.value_counts().to_dict()}")
    composition_matrix = np.asarray(
        [[float(value.strip()) for value in row] for row in composition],
        dtype=np.float64,
    )
    scalar_matrix = table[SCALAR_COLUMNS].to_numpy(dtype=np.float64)
    features = np.column_stack((composition_matrix, scalar_matrix))
    if features.shape[1] != 25:
        raise AssertionError(f"Expected 25 features, got {features.shape[1]}")
    if not np.isfinite(features).all():
        raise ValueError("Physicochemical features contain NaN or infinite values")
    names = [f"aa_composition_{index + 1}" for index in range(18)] + SCALAR_COLUMNS
    return features, names


def score_to_logit(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score.astype(np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def metric_values(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(labels, scores)
    n_positive = int(labels.sum())
    top = np.argsort(-scores, kind="stable")[:n_positive]
    predictions = (scores >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "auprc_trapezoidal": float(auc(recall, precision)),
        "ppv_at_n_positive": float(labels[top].mean()),
        "accuracy_at_0_5": float(accuracy_score(labels, predictions)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(labels, predictions)),
    }


def logistic_search(jobs: int) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid={"model__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
        scoring="average_precision",
        refit=True,
        n_jobs=jobs,
    )


def mlp(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 8),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size=128,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    early_stopping=False,
                    random_state=seed,
                ),
            ),
        ]
    )


def fit_logistic(
    estimator: GridSearchCV,
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> GridSearchCV:
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    estimator.cv = list(inner.split(features, labels, groups))
    return estimator.fit(features, labels)


def grouped_oof_predictions(
    labels: np.ndarray,
    groups: np.ndarray,
    base_logit: np.ndarray,
    physchem: np.ndarray,
    folds: int,
    seed: int,
    jobs: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = {name: np.full(len(labels), np.nan, dtype=np.float64) for name in MODEL_COLUMNS}
    fold_ids = np.full(len(labels), -1, dtype=int)
    scores["base_score"] = 1.0 / (1.0 + np.exp(-base_logit[:, 0]))

    shuffle_rng = np.random.default_rng(seed + 100_000)
    shuffled_physchem = physchem[shuffle_rng.permutation(len(physchem))]

    for fold, (train_index, heldout_index) in enumerate(splitter.split(physchem, labels, groups)):
        fold_ids[heldout_index] = fold
        y_train = labels[train_index]
        train_groups = groups[train_index]
        fold_seed = seed * 10 + fold

        logistic_template = logistic_search(jobs)
        feature_sets = {
            "physchem_logistic": (physchem[train_index], physchem[heldout_index]),
            "combined_logistic": (
                np.column_stack((base_logit[train_index], physchem[train_index])),
                np.column_stack((base_logit[heldout_index], physchem[heldout_index])),
            ),
            "combined_logistic_shuffled": (
                np.column_stack((base_logit[train_index], shuffled_physchem[train_index])),
                np.column_stack((base_logit[heldout_index], shuffled_physchem[heldout_index])),
            ),
        }
        for name, (x_train, x_heldout) in feature_sets.items():
            fitted = fit_logistic(clone(logistic_template), x_train, y_train, train_groups, fold_seed)
            scores[name][heldout_index] = fitted.predict_proba(x_heldout)[:, 1]

        mlp_sets = {
            "physchem_mlp": (physchem[train_index], physchem[heldout_index]),
            "combined_mlp": (
                np.column_stack((base_logit[train_index], physchem[train_index])),
                np.column_stack((base_logit[heldout_index], physchem[heldout_index])),
            ),
        }
        for offset, (name, (x_train, x_heldout)) in enumerate(mlp_sets.items()):
            fitted = mlp(fold_seed + offset).fit(x_train, y_train)
            scores[name][heldout_index] = fitted.predict_proba(x_heldout)[:, 1]

    if (fold_ids < 0).any() or any(np.isnan(values).any() for values in scores.values()):
        raise RuntimeError("OOF prediction generation did not cover every row")
    return scores, fold_ids


def grouped_bootstrap_differences(
    table: pd.DataFrame,
    score_columns: list[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    group_indices = [indices.to_numpy() for _, indices in table.groupby("peptide", sort=False).groups.items()]
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | str | int]] = []
    metrics = ["auroc", "average_precision", "auprc_trapezoidal", "ppv_at_n_positive"]
    for iteration in range(n_bootstrap):
        sampled_groups = rng.integers(0, len(group_indices), len(group_indices))
        sampled_rows = np.concatenate([group_indices[index] for index in sampled_groups])
        labels = table["true_label"].to_numpy(dtype=int)[sampled_rows]
        if labels.min() == labels.max():
            continue
        base = metric_values(labels, table["base_score"].to_numpy()[sampled_rows])
        for score_column in score_columns:
            model = metric_values(labels, table[score_column].to_numpy()[sampled_rows])
            for metric in metrics:
                records.append(
                    {
                        "bootstrap": iteration,
                        "model": score_column,
                        "metric": metric,
                        "difference_vs_base": model[metric] - base[metric],
                    }
                )
    return pd.DataFrame.from_records(records)


def main() -> None:
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    table = pd.read_csv(args.predictions, sep="\t")
    required = {"record_id", "peptide", "true_label", "immunogenicity_score", "aa_composition", *SCALAR_COLUMNS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    labels = table["true_label"].to_numpy(dtype=int)
    groups = table["peptide"].astype(str).to_numpy()
    physchem, feature_names = parse_physchem_features(table)
    base_logit = score_to_logit(table["immunogenicity_score"].to_numpy(dtype=float))

    all_oof: list[pd.DataFrame] = []
    repeat_metrics: list[dict[str, float | str | int]] = []
    for repeat, seed in enumerate(seeds):
        print(f"[audit] repeat={repeat + 1}/{len(seeds)} seed={seed}", flush=True)
        scores, fold_ids = grouped_oof_predictions(
            labels=labels,
            groups=groups,
            base_logit=base_logit,
            physchem=physchem,
            folds=args.folds,
            seed=seed,
            jobs=args.jobs,
        )
        repeat_table = table[["record_id", "peptide", "hla", "true_label"]].copy()
        repeat_table.insert(0, "seed", seed)
        repeat_table.insert(1, "fold", fold_ids)
        for name, values in scores.items():
            repeat_table[name] = values
            values_metrics = metric_values(labels, values)
            for metric, metric_value in values_metrics.items():
                repeat_metrics.append(
                    {"seed": seed, "model": name, "metric": metric, "value": metric_value}
                )
        all_oof.append(repeat_table)

    oof = pd.concat(all_oof, ignore_index=True)
    oof.to_csv(args.outdir / "oof_predictions.tsv", sep="\t", index=False)
    repeat_metric_table = pd.DataFrame.from_records(repeat_metrics)
    repeat_metric_table.to_csv(args.outdir / "metrics_by_seed.tsv", sep="\t", index=False)

    ensemble = (
        oof.groupby(["record_id", "peptide", "hla", "true_label"], as_index=False)[MODEL_COLUMNS]
        .mean()
        .sort_values("record_id")
        .reset_index(drop=True)
    )
    ensemble.to_csv(args.outdir / "mean_oof_predictions.tsv", sep="\t", index=False)
    summary_records: list[dict[str, float | str | int]] = []
    ensemble_labels = ensemble["true_label"].to_numpy(dtype=int)
    for model in MODEL_COLUMNS:
        for metric, value in metric_values(ensemble_labels, ensemble[model].to_numpy()).items():
            summary_records.append(
                {"model": model, "metric": metric, "value": value, "n": len(ensemble)}
            )
    summary = pd.DataFrame.from_records(summary_records)
    summary.to_csv(args.outdir / "mean_oof_metrics.tsv", sep="\t", index=False)

    bootstrap = grouped_bootstrap_differences(
        ensemble,
        [name for name in MODEL_COLUMNS if name != "base_score"],
        args.bootstrap,
        seeds[0],
    )
    bootstrap.to_csv(args.outdir / "grouped_bootstrap_differences.tsv.gz", sep="\t", index=False)
    ci = (
        bootstrap.groupby(["model", "metric"])["difference_vs_base"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.5: "median", 0.975: "ci_high"})
    )
    point_lookup = summary.set_index(["model", "metric"])["value"]
    base_lookup = summary[summary.model == "base_score"].set_index("metric")["value"]
    ci["point_difference_vs_base"] = [
        point_lookup.loc[(model, metric)] - base_lookup.loc[metric]
        for model, metric in zip(ci.model, ci.metric)
    ]
    ci.to_csv(args.outdir / "incremental_signal_summary.tsv", sep="\t", index=False)

    metadata = {
        "input": str(args.predictions),
        "n_rows": int(len(table)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "n_unique_peptides": int(table.peptide.nunique()),
        "folds": args.folds,
        "seeds": seeds,
        "bootstrap": args.bootstrap,
        "grouping": "peptide",
        "feature_names": feature_names,
        "test_set_used": False,
    }
    (args.outdir / "audit_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.pivot(index="model", columns="metric", values="value").to_string(), flush=True)
    print("\nIncremental 95% CIs versus base score:", flush=True)
    print(ci.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
