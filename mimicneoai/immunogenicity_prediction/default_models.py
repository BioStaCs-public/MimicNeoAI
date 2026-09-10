"""Compatibility import path for the default runtime model helpers."""

from mimicneoai.immunogenicity_prediction.runtime.defaults import (
    DEFAULT_MODEL_ROOT,
    default_model_paths,
    normalize_antigen_class,
    run_default_inference,
)

__all__ = [
    "DEFAULT_MODEL_ROOT",
    "default_model_paths",
    "normalize_antigen_class",
    "run_default_inference",
]
