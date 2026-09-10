from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    labels: Iterable[int],
    scores: Iterable[float],
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, float | int]:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(scores), dtype=float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]
    if len(y_true) == 0:
        raise ValueError("No finite prediction scores available for metric calculation.")

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))

    out: dict[str, float | int] = {
        "n": int(len(y_true)),
        "pos": pos,
        "neg": neg,
        "positive_rate": float(pos / len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_at_threshold": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity_at_threshold": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1_at_threshold": _f1(tp, fp, fn),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "ece": float(expected_calibration_error(y_true, y_score, n_bins=n_bins)),
        "mean_ppvn": float(mean_ppvn(y_true, y_score)),
        "ppv_at_npos": float(ppv_at_npos(y_true, y_score)),
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        out["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def _f1(tp: int, fp: int, fn: int) -> float:
    denom = (2 * tp) + fp + fn
    return float((2 * tp) / denom) if denom else 0.0


def ppv_at_npos(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")[:n_pos]
    return float(np.mean(labels[order] == 1))


def mean_ppvn(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(np.sum(labels == 1))
    if n_pos == 0:
        return float("nan")
    precisions = []
    for n in range(1, n_pos + 1):
        order = np.argsort(-scores, kind="mergesort")[:n]
        precisions.append(float(np.mean(labels[order] == 1)))
    return float(np.mean(precisions))


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(scores[mask]))
        acc = float(np.mean(labels[mask]))
        ece += float(np.mean(mask)) * abs(acc - conf)
    return ece


def reliability_table(labels: Iterable[int], scores: Iterable[float], n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(scores), dtype=float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    rows = []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_score >= lo) & (y_score <= hi)
        else:
            mask = (y_score >= lo) & (y_score < hi)
        n = int(np.sum(mask))
        rows.append(
            {
                "bin": i,
                "score_min": float(lo),
                "score_max": float(hi),
                "n": n,
                "mean_score": float(np.mean(y_score[mask])) if n else float("nan"),
                "observed_positive_rate": float(np.mean(y_true[mask])) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def pr_curve_table(labels: Iterable[int], scores: Iterable[float]) -> pd.DataFrame:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(scores), dtype=float)
    valid = np.isfinite(y_score)
    precision, recall, thresholds = precision_recall_curve(y_true[valid], y_score[valid])
    threshold_values = list(thresholds) + [float("nan")]
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": threshold_values,
        }
    )


def write_metric_outputs(
    df: pd.DataFrame,
    outdir: Path,
    *,
    label_col: str,
    score_col: str,
    method: str,
    benchmark: str,
    threshold: float = 0.5,
    n_bins: int = 10,
    group_cols: list[str] | None = None,
) -> dict[str, float | int | str]:
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = compute_binary_metrics(df[label_col], df[score_col], threshold=threshold, n_bins=n_bins)
    metrics.update({"method": method, "benchmark": benchmark})

    pd.DataFrame([metrics]).to_csv(outdir / "metrics.tsv", sep="\t", index=False)
    with (outdir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    reliability_table(df[label_col], df[score_col], n_bins=n_bins).to_csv(
        outdir / "reliability_bins.tsv", sep="\t", index=False
    )
    pr_curve_table(df[label_col], df[score_col]).to_csv(outdir / "precision_recall_curve.tsv", sep="\t", index=False)

    if group_cols:
        rows = []
        for group_col in group_cols:
            if group_col not in df.columns:
                continue
            for group_value, sub in df.groupby(group_col, dropna=False):
                if len(sub) < 2:
                    continue
                row = compute_binary_metrics(sub[label_col], sub[score_col], threshold=threshold, n_bins=n_bins)
                row.update(
                    {
                        "method": method,
                        "benchmark": benchmark,
                        "group_col": group_col,
                        "group_value": group_value,
                    }
                )
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(outdir / "group_metrics.tsv", sep="\t", index=False)
    return metrics

