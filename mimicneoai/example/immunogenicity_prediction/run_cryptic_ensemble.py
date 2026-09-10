"""Run the bundled cryptic immunogenicity example without exposing member IDs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from mimicneoai.immunogenicity_prediction.core import InferenceConfig
from mimicneoai.immunogenicity_prediction.default_models import run_default_inference


def main() -> int:
    parser = argparse.ArgumentParser(description="Cryptic immunogenicity prediction example")
    parser.add_argument("-c", "--configure", required=True, help="Path to example YAML")
    args = parser.parse_args()
    raw = yaml.safe_load(Path(args.configure).read_text()) or {}
    path_cfg = raw["path"]
    args_cfg = raw.get("args", {})
    io_cfg = raw.get("io", {})
    input_df = pd.read_csv(path_cfg["input_csv"])
    score_col = io_cfg.get("score_col", "immunogenicity_score")
    cfg = InferenceConfig(
        model_path="",
        hla_fasta_path=path_cfg.get("hla_fasta", ""),
        hla_source=path_cfg.get("hla_source", "fasta"),
        hla_pseudoseq_csv=tuple(path_cfg.get("hla_pseudoseq_csv", []) or ()),
        peptide_col=io_cfg.get("peptide_col", "peptide"),
        hla_col=io_cfg.get("hla_col", "hla"),
        output_score_col=score_col,
        batch_size=int(args_cfg.get("batch_size", 512)),
        device=str(args_cfg.get("device", "auto")),
        num_processes=int(args_cfg.get("num_processes", 1)),
        verbose=bool(args_cfg.get("verbose", True)),
        include_input_qc=bool(args_cfg.get("include_input_qc", False)),
    )
    result = run_default_inference(input_df, "cryptic", cfg)
    output_path = Path(path_cfg["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Done. Saved: {output_path}")
    print(f"Rows: {len(result)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
