from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def resolve_immunogenicity_num_processes(configure: Mapping[str, Any]) -> int:
    """Resolve R feature workers from a pipeline YAML configuration.

    Each pipeline inherits its existing user-facing CPU setting, accepting
    both ``args.threads`` and the legacy ``args.thread`` spelling.
    """
    args = configure.get("args", {})
    if not isinstance(args, Mapping):
        raise TypeError("configure.args must be a mapping")
    for key in ("threads", "thread"):
        value = args.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            workers = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"args.{key} must be a positive integer") from exc
        if workers < 1:
            raise ValueError(f"args.{key} must be a positive integer")
        return workers
    return 1


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _immunogenicity_paths(paths: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if paths is None:
        return {}
    path_cfg = _mapping(paths.get("path", {}), "paths.path")
    common = _mapping(path_cfg.get("common", {}), "paths.path.common")
    return _mapping(common.get("IMMUNOGENICITY", {}), "paths.path.common.IMMUNOGENICITY")


def resolve_immunogenicity_python_bin(
    configure: Mapping[str, Any],
    paths: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the Python interpreter used for runtime immunogenicity scoring.

    Immunogenicity inference depends on PyTorch. Production images may keep CPU
    and GPU PyTorch builds in dedicated virtual environments, so pipeline entry
    points should not assume that the main ``mimicneoai`` interpreter has torch.
    """

    others = _mapping(configure.get("others", {}), "configure.others")
    configured = str(others.get("immunogenicity_python_bin", "") or "").strip()
    if configured:
        return configured
    env_value = os.environ.get("MIMICNEOAI_IMMUNOGENICITY_PYTHON_BIN", "").strip()
    if env_value:
        return env_value
    path_value = str(_immunogenicity_paths(paths).get("PYTHON_BIN", "") or "").strip()
    if path_value:
        return path_value
    return sys.executable


def resolve_immunogenicity_model_root(
    configure: Mapping[str, Any],
    paths: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the default runtime model root, if the deployment config sets one."""

    others = _mapping(configure.get("others", {}), "configure.others")
    configured = str(others.get("immunogenicity_model_root", "") or "").strip()
    if configured:
        return configured
    env_value = os.environ.get("MIMICNEOAI_IMMUNOGENICITY_MODEL_ROOT", "").strip()
    if env_value:
        return env_value
    return str(_immunogenicity_paths(paths).get("MODEL_ROOT", "") or "").strip()



def predict_immunogenicity_df(
    df: pd.DataFrame,
    *,
    model_path: str,
    hla_fasta_path: str,
    peptide_col: str = "peptide",
    hla_col: str = "hla",
    output_score_col: str = "immunogenicity_score",
    batch_size: int = 512,
    device: str = "auto",
    num_processes: int = 1,
) -> pd.DataFrame:
    from mimicneoai.immunogenicity_prediction.core import InferenceConfig, run_inference

    cfg = InferenceConfig(
        model_path=model_path,
        hla_fasta_path=hla_fasta_path,
        peptide_col=peptide_col,
        hla_col=hla_col,
        output_score_col=output_score_col,
        batch_size=batch_size,
        device=device,
        num_processes=num_processes,
    )
    return run_inference(df, cfg)



def predict_immunogenicity_csv(
    input_csv: str,
    output_csv: str,
    *,
    model_path: str,
    hla_fasta_path: str,
    peptide_col: str = "peptide",
    hla_col: str = "hla",
    output_score_col: str = "immunogenicity_score",
    batch_size: int = 512,
    device: str = "auto",
    num_processes: int = 1,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    out = predict_immunogenicity_df(
        df,
        model_path=model_path,
        hla_fasta_path=hla_fasta_path,
        peptide_col=peptide_col,
        hla_col=hla_col,
        output_score_col=output_score_col,
        batch_size=batch_size,
        device=device,
        num_processes=num_processes,
    )
    out.to_csv(output_csv, index=False)
    return out


def predict_default_immunogenicity_df(
    df: pd.DataFrame,
    *,
    antigen_class: str,
    hla_source: str = "netmhc-pseudoseq",
    hla_fasta_path: str = "",
    hla_pseudoseq_csv: Sequence[str] = (),
    peptide_col: str = "peptide",
    hla_col: str = "hla",
    output_score_col: str = "immunogenicity_score",
    batch_size: int = 512,
    device: str = "auto",
    num_processes: int = 1,
    include_input_qc: bool = True,
    model_root: str | None = None,
) -> pd.DataFrame:
    """Run a recovered default model selected by antigen class.

    The cryptic class is evaluated as the packaged ensemble mean. The default
    HLA source matches the recovered runtime checkpoint metadata.
    """
    from mimicneoai.immunogenicity_prediction.core import InferenceConfig
    from mimicneoai.immunogenicity_prediction.runtime.defaults import run_default_inference

    cfg = InferenceConfig(
        model_path="",
        hla_fasta_path=hla_fasta_path,
        hla_source=hla_source,
        hla_pseudoseq_csv=tuple(hla_pseudoseq_csv),
        peptide_col=peptide_col,
        hla_col=hla_col,
        output_score_col=output_score_col,
        batch_size=batch_size,
        device=device,
        num_processes=num_processes,
        include_input_qc=include_input_qc,
    )
    return run_default_inference(df, antigen_class, cfg, model_root=model_root)


def predict_default_immunogenicity_csv(
    input_csv: str,
    output_csv: str,
    *,
    antigen_class: str,
    hla_source: str = "netmhc-pseudoseq",
    hla_fasta_path: str = "",
    hla_pseudoseq_csv: Sequence[str] = (),
    peptide_col: str = "peptide",
    hla_col: str = "hla",
    output_score_col: str = "immunogenicity_score",
    batch_size: int = 512,
    device: str = "auto",
    num_processes: int = 1,
    include_input_qc: bool = True,
    model_root: str | None = None,
) -> pd.DataFrame:
    """CSV counterpart of :func:`predict_default_immunogenicity_df`."""
    output = predict_default_immunogenicity_df(
        pd.read_csv(input_csv),
        antigen_class=antigen_class,
        hla_source=hla_source,
        hla_fasta_path=hla_fasta_path,
        hla_pseudoseq_csv=hla_pseudoseq_csv,
        peptide_col=peptide_col,
        hla_col=hla_col,
        output_score_col=output_score_col,
        batch_size=batch_size,
        device=device,
        num_processes=num_processes,
        include_input_qc=include_input_qc,
        model_root=model_root,
    )
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output
