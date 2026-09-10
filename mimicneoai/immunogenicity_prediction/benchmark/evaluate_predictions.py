#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimicneoai.immunogenicity_prediction.benchmark.metrics import write_metric_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate immunogenicity benchmark predictions.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--n-bins", default=10, type=int)
    parser.add_argument("--group-col", action="append", default=[])
    args = parser.parse_args()

    df = read_table(args.predictions)
    missing = [col for col in (args.label_col, args.score_col) if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")
    metrics = write_metric_outputs(
        df,
        args.outdir,
        label_col=args.label_col,
        score_col=args.score_col,
        method=args.method,
        benchmark=args.benchmark,
        threshold=args.threshold,
        n_bins=args.n_bins,
        group_cols=args.group_col,
    )
    print(metrics)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


if __name__ == "__main__":
    main()

