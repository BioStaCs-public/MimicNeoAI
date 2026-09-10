#!/usr/bin/env python3
"""Summarize paired Full/Zero validation or test learning curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mimicneoai.immunogenicity_prediction.benchmark.summarize_physchem_experiment import (
    aligned_pair,
    metrics,
    paired_bootstrap,
    read_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low-data-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def fraction_value(name: str) -> float:
    if not name.startswith("fraction_"):
        raise ValueError(f"Unexpected fraction directory: {name}")
    return int(name.removeprefix("fraction_")) / 100.0


def table_path(root: Path, antigen: str, fraction: str, condition: str, split: str) -> Path:
    return root / antigen / fraction / condition / "predictions" / f"{split}_predictions.tsv"


def pilot_path(root: Path, antigen: str, condition: str, split: str) -> Path:
    return root / antigen / condition / "predictions" / f"{split}_predictions.tsv"


def normalized_curve_auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or x[-1] <= x[0]:
        raise ValueError("Learning curve requires at least two increasing fractions")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def aligned_curve_frames(frames: dict[tuple[float, str], pd.DataFrame]):
    reference_key = sorted(frames)[0]
    reference = frames[reference_key][["record_id", "true_label"]].copy()
    reference = reference.rename(columns={"true_label": "labels"})
    for (fraction, condition), frame in sorted(frames.items()):
        score_name = f"score_{condition}_{fraction:.2f}"
        reference = reference.merge(
            frame[["record_id", "true_label", "positive_probability"]].rename(
                columns={"true_label": "candidate_labels", "positive_probability": score_name}
            ),
            on="record_id",
            how="inner",
            validate="one_to_one",
        )
        if not np.array_equal(reference["labels"], reference["candidate_labels"]):
            raise ValueError(f"Label mismatch for fraction={fraction}, condition={condition}")
        reference = reference.drop(columns="candidate_labels")
    expected = len(next(iter(frames.values())))
    if len(reference) != expected:
        raise ValueError("Learning-curve prediction tables are not one-to-one aligned")
    return reference


def curve_bootstrap(
    aligned: pd.DataFrame,
    fractions: np.ndarray,
    metric_key: str,
    n_bootstrap: int,
    seed: int,
) -> dict:
    labels = aligned["labels"].to_numpy(dtype=int)

    def curve(condition: str, index: np.ndarray | None = None) -> float:
        current_labels = labels if index is None else labels[index]
        values = []
        for fraction in fractions:
            scores = aligned[f"score_{condition}_{fraction:.2f}"].to_numpy(dtype=float)
            current_scores = scores if index is None else scores[index]
            values.append(metrics(current_labels, current_scores)[metric_key])
        return normalized_curve_auc(fractions, np.asarray(values))

    full = curve("full")
    zero = curve("zero")
    rng = np.random.default_rng(seed)
    deltas = []
    attempts = 0
    while len(deltas) < n_bootstrap:
        attempts += 1
        if attempts > n_bootstrap * 20:
            raise RuntimeError("Unable to obtain enough valid learning-curve resamples")
        index = rng.integers(0, len(labels), len(labels))
        if labels[index].min() == labels[index].max():
            continue
        deltas.append(curve("full", index) - curve("zero", index))
    values = np.asarray(deltas)
    return {
        "metric": metric_key,
        "full_normalized_area": full,
        "zero_normalized_area": zero,
        "delta_full_minus_zero": full - zero,
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "bootstrap_probability_delta_gt_zero": float(np.mean(values > 0.0)),
    }


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    point_bootstrap = {}
    area_bootstrap = {}
    antigens = sorted(path.name for path in args.low_data_root.iterdir() if path.is_dir())
    for antigen in antigens:
        frames: dict[tuple[float, str], pd.DataFrame] = {}
        fraction_dirs = sorted(
            (path for path in (args.low_data_root / antigen).iterdir() if path.is_dir()),
            key=lambda path: fraction_value(path.name),
        )
        for fraction_dir in fraction_dirs:
            fraction = fraction_value(fraction_dir.name)
            for condition in ("full", "zero"):
                path = table_path(
                    args.low_data_root,
                    antigen,
                    fraction_dir.name,
                    condition,
                    args.split,
                )
                if path.exists():
                    frames[(fraction, condition)] = read_predictions(path)
        for condition in ("full", "zero"):
            path = pilot_path(args.pilot_root, antigen, condition, args.split)
            if path.exists():
                frames[(1.0, condition)] = read_predictions(path)

        fractions = sorted(
            fraction for fraction in {key[0] for key in frames} if (fraction, "full") in frames and (fraction, "zero") in frames
        )
        point_bootstrap[antigen] = {}
        for fraction in fractions:
            full_frame = frames[(fraction, "full")]
            zero_frame = frames[(fraction, "zero")]
            labels, full_scores, zero_scores = aligned_pair(full_frame, zero_frame)
            point_bootstrap[antigen][f"{fraction:.2f}"] = paired_bootstrap(
                labels,
                full_scores,
                zero_scores,
                args.n_bootstrap,
                args.seed,
            )
            for condition, scores in (("full", full_scores), ("zero", zero_scores)):
                rows.append(
                    {
                        "antigen": antigen,
                        "fraction": fraction,
                        "condition": condition,
                        "split": args.split,
                        **metrics(labels, scores),
                    }
                )

        if len(fractions) >= 2:
            aligned = aligned_curve_frames(
                {
                    (fraction, condition): frames[(fraction, condition)]
                    for fraction in fractions
                    for condition in ("full", "zero")
                }
            )
            x = np.asarray(fractions, dtype=float)
            area_bootstrap[antigen] = {
                metric_key: curve_bootstrap(
                    aligned,
                    x,
                    metric_key,
                    args.n_bootstrap,
                    args.seed,
                )
                for metric_key in ("auroc", "average_precision", "auprc_trapezoid", "ppv_at_n_positive")
            }

    pd.DataFrame(rows).to_csv(
        args.outdir / f"physchem_learning_curve_{args.split}_metrics.tsv",
        sep="\t",
        index=False,
    )
    (args.outdir / f"physchem_learning_curve_{args.split}_point_bootstrap.json").write_text(
        json.dumps(point_bootstrap, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.outdir / f"physchem_learning_curve_{args.split}_area_bootstrap.json").write_text(
        json.dumps(area_bootstrap, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(area_bootstrap, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
