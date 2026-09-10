#!/usr/bin/env python3
"""Exact physicochemical attribution for the legacy-concat model.

The legacy model maps standardized physicochemical features through a linear
25-to-16 projection and concatenates that result with the pHLA representation
before the final linear classifier. In evaluation mode, each feature therefore
has an exact additive contribution to the positive-vs-negative logit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


AA_GROUPS = [
    "tiny",
    "small",
    "aliphatic",
    "aromatic",
    "nonpolar",
    "polar",
    "charged",
    "basic",
    "acidic",
]
FEATURE_NAMES = (
    [f"aa_{group}_count" for group in AA_GROUPS]
    + [f"aa_{group}_mole_percent" for group in AA_GROUPS]
    + [
        "polarity",
        "volume",
        "net_charge",
        "hydrophobicity",
        "boman_index",
        "aliphatic_index",
        "isoelectric_point",
    ]
)
SCALAR_COLUMNS = FEATURE_NAMES[-7:]


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator)


def parse_feature_matrix(table: pd.DataFrame) -> np.ndarray:
    required = {"aa_composition", *SCALAR_COLUMNS}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Prediction table is missing feature columns: {missing}")

    composition = []
    for row_index, value in enumerate(table["aa_composition"]):
        parsed = [float(item.strip()) for item in str(value).split(",")]
        if len(parsed) != 18:
            raise ValueError(
                f"Expected 18 amino-acid composition values at row {row_index}; "
                f"found {len(parsed)}"
            )
        composition.append(parsed)
    scalar = table[SCALAR_COLUMNS].astype(float).to_numpy()
    matrix = np.concatenate([np.asarray(composition, dtype=float), scalar], axis=1)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(f"Expected 25 features, found {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError("Physicochemical feature matrix contains NaN or infinite values")
    return matrix


def load_checkpoint_components(checkpoint: Path) -> Tuple[np.ndarray, Dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model_state_dict", payload))
    if "fc_x2.weight" not in state or "fc.weight" not in state:
        raise KeyError("Checkpoint does not contain the legacy-concat physicochemical branch")
    fc_x2 = state["fc_x2.weight"].detach().cpu()
    classifier = state["fc.weight"].detach().cpu()
    if tuple(fc_x2.shape) != (16, 25):
        raise ValueError(f"Expected fc_x2 weight shape (16, 25), found {tuple(fc_x2.shape)}")
    sequence_dim = classifier.shape[1] - fc_x2.shape[0]
    if sequence_dim <= 0 or classifier.shape[0] != 2:
        raise ValueError(f"Unexpected final classifier shape: {tuple(classifier.shape)}")
    class_logit_difference = classifier[1, sequence_dim:] - classifier[0, sequence_dim:]
    effective_coefficients = class_logit_difference @ fc_x2
    metadata = payload.get("metadata", {})
    return effective_coefficients.numpy().astype(float), metadata


def load_scaler(path: Path, checkpoint_metadata: Dict) -> Dict:
    scaler = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_scaler = checkpoint_metadata.get("physchem_scaler")
    if checkpoint_scaler:
        for key in ("mean", "scale", "clip"):
            if not np.allclose(
                np.asarray(scaler[key], dtype=float),
                np.asarray(checkpoint_scaler[key], dtype=float),
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(f"Scaler file and checkpoint metadata differ for {key}")
    if len(scaler["mean"]) != 25 or len(scaler["scale"]) != 25:
        raise ValueError("Scaler must contain 25 means and 25 scales")
    if scaler.get("fit_source") != "train_split":
        raise ValueError(f"Expected a train-only scaler, found fit_source={scaler.get('fit_source')}")
    return scaler


def standardize(features: np.ndarray, scaler: Dict) -> np.ndarray:
    means = np.asarray(scaler["mean"], dtype=float)
    scales = np.asarray(scaler["scale"], dtype=float)
    if np.any(scales <= 0):
        raise ValueError("Scaler contains non-positive scales")
    standardized = (features - means) / scales
    clip = float(scaler.get("clip", 0.0))
    if clip > 0:
        standardized = np.clip(standardized, -clip, clip)
    return standardized


def trapezoidal_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    return float(auc(recall, precision))


def ppv_at_n_positive(labels: np.ndarray, scores: np.ndarray) -> float:
    n_positive = int(labels.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")[:n_positive]
    return float(labels[order].mean())


def metrics(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "auprc": trapezoidal_auprc(labels, scores),
        "accuracy_at_0_5": float(accuracy_score(labels, scores >= 0.5)),
        "ppv_at_n_positive": ppv_at_n_positive(labels, scores),
    }


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities.astype(float), 1e-7, 1.0 - 1e-7)
    return np.log(probabilities) - np.log1p(-probabilities)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    output = np.empty_like(logits, dtype=float)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    output[~positive] = exp_logits / (1.0 + exp_logits)
    return output


def write_markdown_summary(
    path: Path,
    baseline: Dict[str, float],
    importance: pd.DataFrame,
    n_rows: int,
    n_positive: int,
) -> None:
    lines = [
        "# Physicochemical feature attribution",
        "",
        f"Validation samples: {n_rows} ({n_positive} positive, {n_rows - n_positive} negative).",
        "",
        "The analysis uses exact additive contributions to the positive-vs-negative logit. "
        "Single-feature ablation replaces one standardized feature with zero, corresponding "
        "to the Stage 2 training-set mean.",
        "",
        "## Baseline",
        "",
        f"- AUROC: {baseline['auroc']:.6f}",
        f"- Average precision: {baseline['average_precision']:.6f}",
        f"- Trapezoidal AUPRC: {baseline['auprc']:.6f}",
        f"- PPV@Npos: {baseline['ppv_at_n_positive']:.6f}",
        "",
        "## Highest mean absolute logit contributions",
        "",
        "| Rank | Feature | Mean absolute contribution | AUPRC drop after ablation |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(importance.head(10).itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.feature} | {row.mean_absolute_logit_contribution:.6f} | "
            f"{row.auprc_performance_drop:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--score-col", default="immunogenicity_score")
    parser.add_argument("--label-col", default="true_label")
    parser.add_argument(
        "--encoded-train-cache",
        default=None,
        type=Path,
        help="Optional encoded training cache used to export a representative SHAP background.",
    )
    parser.add_argument("--shap-background-size", default=1000, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    table = read_table(args.predictions)
    if args.label_col not in table or args.score_col not in table:
        raise KeyError(f"Missing {args.label_col} or {args.score_col} in prediction table")
    labels = table[args.label_col].astype(int).to_numpy()
    scores = table[args.score_col].astype(float).to_numpy()
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Attribution requires both binary labels")

    raw_features = parse_feature_matrix(table)
    coefficients, metadata = load_checkpoint_components(args.checkpoint)
    scaler = load_scaler(args.scaler, metadata)
    standardized = standardize(raw_features, scaler)
    contributions = standardized * coefficients.reshape(1, -1)
    original_logits = safe_logit(scores)
    baseline = metrics(labels, scores)

    contribution_output = table[
        [column for column in ("record_id", "peptide", "hla", args.label_col, args.score_col) if column in table]
    ].copy()
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        contribution_output[f"contribution__{feature_name}"] = contributions[:, feature_index]
    contribution_output.to_csv(
        args.outdir / "per_sample_logit_contributions.tsv", sep="\t", index=False
    )

    if args.encoded_train_cache is not None:
        cache = torch.load(args.encoded_train_cache, map_location="cpu", weights_only=False)
        if not isinstance(cache, dict) or "x" not in cache:
            raise ValueError("Encoded training cache must be a dictionary containing x")
        training_features = cache["x"][:, : len(FEATURE_NAMES)].detach().cpu().numpy()
        training_features = standardize(training_features, scaler)
        background_size = min(int(args.shap_background_size), len(training_features))
        if background_size <= 0:
            raise ValueError("--shap-background-size must be positive")
        rng = np.random.default_rng(args.seed)
        background_indices = np.sort(
            rng.choice(len(training_features), size=background_size, replace=False)
        )
        background = training_features[background_indices]
    else:
        background_indices = np.asarray([], dtype=int)
        background = np.zeros((1, len(FEATURE_NAMES)), dtype=float)
    np.savez_compressed(
        args.outdir / "physchem_shap_inputs.npz",
        coefficients=coefficients,
        background=background,
        background_indices=background_indices,
        explained_features=standardized,
        labels=labels,
        scores=scores,
        feature_names=np.asarray(FEATURE_NAMES),
    )

    rows = []
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        ablated_scores = sigmoid(original_logits - contributions[:, feature_index])
        ablated = metrics(labels, ablated_scores)
        row = {
            "feature_index": feature_index,
            "feature": feature_name,
            "effective_logit_coefficient": coefficients[feature_index],
            "mean_signed_logit_contribution": contributions[:, feature_index].mean(),
            "mean_absolute_logit_contribution": np.abs(contributions[:, feature_index]).mean(),
            "mean_contribution_positive": contributions[labels == 1, feature_index].mean(),
            "mean_contribution_negative": contributions[labels == 0, feature_index].mean(),
        }
        for metric_name, value in ablated.items():
            row[f"ablated_{metric_name}"] = value
            row[f"{metric_name}_performance_drop"] = baseline[metric_name] - value
        rows.append(row)

    importance = pd.DataFrame(rows).sort_values(
        ["mean_absolute_logit_contribution", "feature"], ascending=[False, True]
    )
    importance.insert(0, "contribution_rank", np.arange(1, len(importance) + 1))
    importance.to_csv(args.outdir / "feature_importance_and_ablation.tsv", sep="\t", index=False)

    summary = {
        "n_validation": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": metadata.get("selected_epoch"),
        "scaler": str(args.scaler.resolve()),
        "scaler_fit_source": scaler.get("fit_source"),
        "scaler_n_fit": scaler.get("n_fit"),
        "baseline_metrics": baseline,
        "attribution_scale": "positive-vs-negative logit",
        "ablation_baseline": "Stage 2 training-set feature mean after standardization",
        "feature_names": FEATURE_NAMES,
        "shap_background_size": int(len(background)),
        "shap_background_source": (
            str(args.encoded_train_cache.resolve())
            if args.encoded_train_cache is not None
            else "training_mean_zero_reference"
        ),
    }
    (args.outdir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_markdown_summary(
        args.outdir / "README.md", baseline, importance, len(labels), int(labels.sum())
    )
    print(json.dumps(summary, indent=2))
    print("\nTop features by mean absolute logit contribution:")
    print(
        importance[
            [
                "contribution_rank",
                "feature",
                "mean_absolute_logit_contribution",
                "auprc_performance_drop",
                "auroc_performance_drop",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
