"""YAML-configured command-line entry point for immunogenicity inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from mimicneoai.immunogenicity_prediction.core import (
    InferenceConfig,
    export_model_to_onnx,
    run_inference,
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_cfg_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    path_cfg = raw.get("path", {})
    args_cfg = raw.get("args", {})
    io_cfg = raw.get("io", {})
    return {
        "input_csv": path_cfg.get("input_csv"),
        "output_csv": path_cfg.get("output_csv"),
        "model_path": path_cfg.get("model_path"),
        "hla_fasta": path_cfg.get("hla_fasta"),
        "hla_source": path_cfg.get("hla_source", args_cfg.get("hla_source", "fasta")),
        "hla_pseudoseq_csv": path_cfg.get("hla_pseudoseq_csv", []),
        "peptide_col": io_cfg.get("peptide_col", "peptide"),
        "hla_col": io_cfg.get("hla_col", "hla"),
        "source_col": io_cfg.get("source_col", "antigen_class"),
        "score_col": io_cfg.get("score_col", "immunogenicity_score"),
        "status_col": io_cfg.get("status_col", "immunogenicity_status"),
        "padding_length": int(args_cfg.get("padding_length", 0)),
        "batch_size": int(args_cfg.get("batch_size", 512)),
        "num_processes": int(args_cfg.get("num_processes", 1)),
        "device": str(args_cfg.get("device", "auto")),
        "verbose": bool(args_cfg.get("verbose", True)),
        "include_input_qc": bool(args_cfg.get("include_input_qc", False)),
        "export_onnx": path_cfg.get("export_onnx", ""),
        "export_onnx_only": bool(args_cfg.get("export_onnx_only", False)),
        "onnx_opset": int(args_cfg.get("onnx_opset", 17)),
    }


def run_from_config(config_path: str) -> int:
    cfg = _build_cfg_dict(_load_yaml(config_path))
    if not cfg["model_path"]:
        raise ValueError("path.model_path is required in config.")

    if cfg["export_onnx"]:
        export_path = Path(cfg["export_onnx"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_model_to_onnx(
            model_path=cfg["model_path"],
            onnx_path=str(export_path),
            device_name=cfg["device"],
            opset_version=cfg["onnx_opset"],
            padding_length=cfg["padding_length"] or 419,
        )
        print(f"ONNX exported: {export_path}")
        if cfg["export_onnx_only"]:
            return 0

    for required in ("input_csv", "output_csv"):
        if not cfg[required]:
            raise ValueError(f"path.{required} is required for inference mode.")
    if cfg["hla_source"] == "fasta" and not cfg["hla_fasta"]:
        raise ValueError("path.hla_fasta is required when path.hla_source is 'fasta'.")
    if isinstance(cfg["hla_pseudoseq_csv"], str):
        cfg["hla_pseudoseq_csv"] = [cfg["hla_pseudoseq_csv"]]

    output_df = run_inference(
        pd.read_csv(cfg["input_csv"]),
        InferenceConfig(
            model_path=cfg["model_path"],
            hla_fasta_path=cfg["hla_fasta"],
            hla_source=cfg["hla_source"],
            hla_pseudoseq_csv=tuple(cfg["hla_pseudoseq_csv"] or ()),
            padding_length=cfg["padding_length"],
            peptide_col=cfg["peptide_col"],
            hla_col=cfg["hla_col"],
            source_col=cfg["source_col"],
            output_score_col=cfg["score_col"],
            output_status_col=cfg["status_col"],
            batch_size=cfg["batch_size"],
            device=cfg["device"],
            num_processes=cfg["num_processes"],
            verbose=cfg["verbose"],
            include_input_qc=cfg["include_input_qc"],
        ),
    )
    output_path = Path(cfg["output_csv"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Done. Saved: {output_path}")
    print(f"Rows: {len(output_df)}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Immunogenicity prediction pipeline")
    parser.add_argument("-c", "--configure", required=True, help="Path to configuration YAML")
    args = parser.parse_args(argv)
    return run_from_config(args.configure)


if __name__ == "__main__":
    raise SystemExit(main())
