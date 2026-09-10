#!/usr/bin/env python3
"""Create deterministic, nested peptide-level training subsets.

Rows sharing a peptide are never split. Peptide groups are ranked by a stable
SHA-256 digest within source-label strata, so every smaller subset is contained
in every larger subset generated from the same input and seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-table", type=Path, required=True)
    parser.add_argument("--validation-table", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", required=True)
    parser.add_argument("--peptide-col", default="peptide")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--source-col", default="dataset_source")
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=separator, dtype=str, keep_default_na=False)


def stable_digest(seed: int, stratum: str, peptide: str) -> str:
    payload = f"{seed}\t{stratum}\t{peptide}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def peptide_groups(
    frame: pd.DataFrame,
    peptide_col: str,
    label_col: str,
    source_col: str,
    seed: int,
) -> pd.DataFrame:
    summaries = []
    for peptide, group in frame.groupby(peptide_col, sort=False, dropna=False):
        labels = sorted(set(group[label_col].astype(str)))
        sources = sorted(set(group[source_col].astype(str)))
        stratum = f"label={'|'.join(labels)};source={'|'.join(sources)}"
        summaries.append(
            {
                "peptide": str(peptide),
                "stratum": stratum,
                "labels": "|".join(labels),
                "sources": "|".join(sources),
                "n_rows": int(len(group)),
                "digest": stable_digest(seed, stratum, str(peptide)),
            }
        )
    groups = pd.DataFrame(summaries)
    return groups.sort_values(["stratum", "digest", "peptide"], kind="stable").reset_index(drop=True)


def selected_peptides(groups: pd.DataFrame, fraction: float) -> set[str]:
    selected: set[str] = set()
    for _, stratum in groups.groupby("stratum", sort=True):
        count = min(len(stratum), max(1, math.ceil(len(stratum) * fraction)))
        selected.update(stratum.iloc[:count]["peptide"].tolist())
    return selected


def split_summary(frame: pd.DataFrame, label_col: str, source_col: str) -> dict:
    return {
        "rows": int(len(frame)),
        "unique_peptides": int(frame["peptide"].nunique()),
        "labels": {str(key): int(value) for key, value in frame[label_col].value_counts().sort_index().items()},
        "sources": {
            str(key): int(value) for key, value in frame[source_col].value_counts().sort_index().items()
        },
    }


def main() -> None:
    args = parse_args()
    fractions = sorted(set(args.fractions))
    if not fractions or fractions[-1] > 1.0 or fractions[0] <= 0.0:
        raise ValueError("Fractions must be in the interval (0, 1]")

    train = read_table(args.train_table)
    validation = read_table(args.validation_table)
    required = {args.peptide_col, args.label_col, args.source_col}
    for name, frame in (("train", train), ("validation", validation)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
    train = train.copy()
    validation = validation.copy()
    train[args.peptide_col] = train[args.peptide_col].astype(str).str.strip().str.upper()
    validation[args.peptide_col] = validation[args.peptide_col].astype(str).str.strip().str.upper()

    overlap = set(train[args.peptide_col]).intersection(validation[args.peptide_col])
    if overlap:
        raise ValueError(f"Train/validation peptide leakage detected: {len(overlap)} peptides")

    groups = peptide_groups(
        train,
        args.peptide_col,
        args.label_col,
        args.source_col,
        args.seed,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    groups.to_csv(args.outdir / "peptide_sampling_order.tsv", sep="\t", index=False)

    report = {
        "method": "nested deterministic peptide-level source-label-stratified sampling",
        "seed": args.seed,
        "train_table": str(args.train_table),
        "validation_table": str(args.validation_table),
        "train_validation_peptide_overlap": 0,
        "full_train": split_summary(train.rename(columns={args.peptide_col: "peptide"}), args.label_col, args.source_col),
        "fixed_validation": split_summary(
            validation.rename(columns={args.peptide_col: "peptide"}), args.label_col, args.source_col
        ),
        "subsets": {},
    }
    previous: set[str] = set()
    for fraction in fractions:
        selected = selected_peptides(groups, fraction)
        if not previous.issubset(selected):
            raise AssertionError("Generated subsets are not nested")
        previous = selected
        subset = train[train[args.peptide_col].isin(selected)].copy()
        label = f"fraction_{int(round(fraction * 100)):03d}"
        subset_dir = args.outdir / label
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_path = subset_dir / "train.model_input.tsv"
        subset.to_csv(subset_path, sep="\t", index=False)
        summary_frame = subset.rename(columns={args.peptide_col: "peptide"})
        report["subsets"][label] = {
            "requested_fraction": fraction,
            "actual_row_fraction": float(len(subset) / len(train)),
            "actual_peptide_fraction": float(len(selected) / len(groups)),
            "path": str(subset_path),
            **split_summary(summary_frame, args.label_col, args.source_col),
        }

    (args.outdir / "split_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
