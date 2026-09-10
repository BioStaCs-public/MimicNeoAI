#!/usr/bin/env python3
"""Run and verify exact SHAP analysis for the linear physicochemical branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--top-n", default=10, type=int)
    parser.add_argument(
        "--output-prefix",
        default="microbial",
        help="Prefix used for generated SHAP plot filenames.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    payload = np.load(args.input, allow_pickle=False)
    coefficients = payload["coefficients"].astype(float)
    background = payload["background"].astype(float)
    features = payload["explained_features"].astype(float)
    labels = payload["labels"].astype(int)
    scores = payload["scores"].astype(float)
    feature_names = payload["feature_names"].astype(str).tolist()
    if coefficients.shape != (25,) or background.shape[1] != 25 or features.shape[1] != 25:
        raise ValueError("Expected a 25-dimensional linear physicochemical model")

    linear_model = (coefficients.reshape(1, -1), np.asarray([0.0]))
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer(linear_model, masker)
    explanation = explainer(features)
    shap_values = np.asarray(explanation.values, dtype=float)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != features.shape:
        raise RuntimeError(f"Unexpected SHAP shape: {shap_values.shape}")

    background_mean = background.mean(axis=0)
    analytical = (features - background_mean) * coefficients.reshape(1, -1)
    max_shap_difference = float(np.max(np.abs(shap_values - analytical)))
    branch_logits = features @ coefficients
    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if len(base_values) == 1:
        base_values = np.repeat(base_values, len(features))
    reconstructed = base_values + shap_values.sum(axis=1)
    max_additivity_error = float(np.max(np.abs(reconstructed - branch_logits)))
    if max_shap_difference > 1e-6 or max_additivity_error > 1e-6:
        raise RuntimeError(
            f"SHAP verification failed: difference={max_shap_difference}, "
            f"additivity={max_additivity_error}"
        )

    mean_absolute = np.abs(shap_values).mean(axis=0)
    relative_percent = 100.0 * mean_absolute / mean_absolute.sum()
    summary = pd.DataFrame(
        {
            "feature_index": np.arange(25),
            "feature": feature_names,
            "effective_logit_coefficient": coefficients,
            "mean_absolute_shap": mean_absolute,
            "relative_mean_absolute_shap_percent": relative_percent,
            "mean_signed_shap": shap_values.mean(axis=0),
            "mean_shap_positive": shap_values[labels == 1].mean(axis=0),
            "mean_shap_negative": shap_values[labels == 0].mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(args.outdir / "shap_feature_summary.tsv", sep="\t", index=False)

    per_sample = pd.DataFrame(shap_values, columns=[f"shap__{name}" for name in feature_names])
    per_sample.insert(0, "score", scores)
    per_sample.insert(0, "label", labels)
    per_sample.to_csv(args.outdir / "per_sample_shap_values.tsv", sep="\t", index=False)

    top = summary.head(args.top_n).sort_values("relative_mean_absolute_shap_percent")
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=180)
    bars = ax.barh(
        top["feature"],
        top["relative_mean_absolute_shap_percent"],
        color="#9d001f",
        edgecolor="none",
    )
    ax.set_xlabel("Relative mean absolute SHAP value (%)")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    for bar, value in zip(bars, top["relative_mean_absolute_shap_percent"]):
        ax.text(
            bar.get_width() + 0.12,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(top["relative_mean_absolute_shap_percent"]) * 1.2)
    fig.tight_layout()
    for extension in ("pdf", "png", "svg"):
        fig.savefig(
            args.outdir
            / f"{args.output_prefix}_physchem_shap_top{args.top_n}.{extension}",
            bbox_inches="tight",
        )
    plt.close(fig)

    display = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=features,
        feature_names=feature_names,
    )
    shap.plots.beeswarm(display, max_display=args.top_n, show=False)
    plt.gcf().set_size_inches(6.4, 4.8)
    plt.tight_layout()
    for extension in ("pdf", "png", "svg"):
        plt.savefig(
            args.outdir
            / f"{args.output_prefix}_physchem_shap_beeswarm_top{args.top_n}.{extension}",
            bbox_inches="tight",
        )
    plt.close()

    report = {
        "shap_version": shap.__version__,
        "n_background": int(len(background)),
        "n_explained": int(len(features)),
        "output_scale": "positive-vs-negative log-odds",
        "max_analytical_shap_difference": max_shap_difference,
        "max_additivity_error": max_additivity_error,
        "background_feature_mean_max_abs": float(np.max(np.abs(background_mean))),
    }
    (args.outdir / "shap_verification.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(summary.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
