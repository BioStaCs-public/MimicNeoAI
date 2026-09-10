"""Re-run a selected locked Fig. 2 prediction subset without copying paper data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mimicneoai.immunogenicity_prediction.core import InferenceConfig
from mimicneoai.immunogenicity_prediction.default_models import (
    normalize_antigen_class,
    run_default_inference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify default model scores against a locked Fig. 2 TSV")
    parser.add_argument("--reference-tsv", required=True, help="Locked three-model prediction TSV")
    parser.add_argument("--antigen-class", default="microbial")
    parser.add_argument("--limit", type=int, default=32, help="Deterministic rows after locked_row_id ordering")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Maximum score difference; accommodates cross-runtime R/float feature recomputation.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-root", default=None)
    args = parser.parse_args()

    source = normalize_antigen_class(args.antigen_class)
    reference = pd.read_csv(args.reference_tsv, sep="\t")
    required = {"antigen_class", "locked_row_id", "peptide", "hla", "immunogenicity_score", "inference_status"}
    missing = required.difference(reference.columns)
    if missing:
        raise KeyError(f"Reference TSV is missing columns: {sorted(missing)}")

    accepted_classes = {source}
    if source == "cryptic":
        accepted_classes.add("cryptic_noncoding")
    subset = reference.loc[
        reference["antigen_class"].isin(accepted_classes)
        & reference["inference_status"].eq("ok")
    ].sort_values("locked_row_id", kind="stable")
    if args.limit > 0:
        subset = subset.head(args.limit)
    if subset.empty:
        raise ValueError(f"No successful locked rows found for {args.antigen_class!r}")

    cfg = InferenceConfig(
        model_path="",
        hla_source="netmhc-pseudoseq",
        batch_size=512,
        num_processes=1,
        device=args.device,
        verbose=False,
    )
    rerun = run_default_inference(
        subset[["peptide", "hla"]], source, cfg, model_root=args.model_root
    )
    if not rerun["immunogenicity_status"].eq("ok").all():
        failed = int((~rerun["immunogenicity_status"].eq("ok")).sum())
        raise RuntimeError(f"Re-run returned {failed} non-ok rows")

    delta = (rerun["immunogenicity_score"].to_numpy() - subset["immunogenicity_score"].to_numpy()).astype(float)
    max_abs_delta = float(abs(delta).max())
    mean_abs_delta = float(abs(delta).mean())
    print(
        f"source={source} records={len(subset)} max_abs_delta={max_abs_delta:.12g} "
        f"mean_abs_delta={mean_abs_delta:.12g} tolerance={args.tolerance:.12g}"
    )
    if max_abs_delta > args.tolerance:
        raise AssertionError("Locked Fig. 2 score reproduction exceeded tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
