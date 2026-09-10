"""Runtime access to the externally recovered default immunogenicity models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from mimicneoai.immunogenicity_prediction.runtime.inference import (
    InferenceConfig,
    prepare_inference_features,
    run_inference,
)


DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models" / "default"

_SOURCE_ALIASES = {
    "microbial": "microbial",
    "mutation_derived": "mutation_derived",
    "mutation-derived": "mutation_derived",
    "cryptic": "cryptic",
    "cryptic_noncoding": "cryptic",
}


def normalize_antigen_class(antigen_class: str) -> str:
    """Return the supported runtime model class for a user-facing source name."""
    key = str(antigen_class).strip().lower()
    try:
        return _SOURCE_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted({"microbial", "mutation_derived", "cryptic"}))
        raise ValueError(f"Unsupported antigen class {antigen_class!r}; use one of: {supported}.") from exc


def default_model_paths(antigen_class: str, model_root: str | Path | None = None) -> tuple[Path, ...]:
    """Resolve the externally supplied runtime weight file(s) for one antigen class."""
    source = normalize_antigen_class(antigen_class)
    root = Path(model_root) if model_root is not None else DEFAULT_MODEL_ROOT
    if source == "cryptic":
        paths = tuple(sorted((root / source).glob("member_*.pth")))
    else:
        paths = (root / source / "model.pth",)

    missing = [path for path in paths if not path.is_file()]
    if not paths or missing:
        expected = root / source / ("member_*.pth" if source == "cryptic" else "model.pth")
        raise FileNotFoundError(
            "Default immunogenicity model payload is unavailable. Expected "
            f"{expected}. Restore the separately distributed runtime weights."
        )
    return paths


def run_default_inference(
    input_df: pd.DataFrame,
    antigen_class: str,
    cfg: InferenceConfig,
    *,
    model_root: str | Path | None = None,
) -> pd.DataFrame:
    """Run a default source model; cryptic scores are the mean across all members."""
    paths = default_model_paths(antigen_class, model_root=model_root)
    prepared_feature_table = prepare_inference_features(input_df, cfg)
    results = [
        run_inference(
            input_df,
            replace(cfg, model_path=str(path)),
            prepared_feature_table=prepared_feature_table,
        )
        for path in paths
    ]

    output = results[0].copy()
    score_col = cfg.output_score_col
    status_col = cfg.output_status_col
    if len(results) == 1:
        return output

    expected_status = output[status_col].tolist()
    if any(result[status_col].tolist() != expected_status for result in results[1:]):
        raise RuntimeError("Cryptic ensemble members returned inconsistent row status values.")
    output[score_col] = pd.concat(
        [result[score_col] for result in results], axis=1
    ).mean(axis=1)
    return output
