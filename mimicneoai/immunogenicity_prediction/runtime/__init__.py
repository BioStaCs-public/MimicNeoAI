"""Stable runtime components for immunogenicity inference.

Training and benchmark code deliberately remain outside this namespace. Public
symbols are resolved lazily so low-level modules can depend on each other
without loading model helpers during core initialization.
"""

__all__ = [
    "DEFAULT_MODEL_ROOT",
    "default_model_paths",
    "normalize_antigen_class",
    "run_default_inference",
    "InferenceConfig",
    "prepare_inference_features",
    "run_inference",
    "normalize_hla",
    "annotate_input_qc",
]


def __getattr__(name: str):
    if name in {"DEFAULT_MODEL_ROOT", "default_model_paths", "normalize_antigen_class", "run_default_inference"}:
        from mimicneoai.immunogenicity_prediction.runtime import defaults

        return getattr(defaults, name)
    if name in {"InferenceConfig", "prepare_inference_features", "run_inference"}:
        from mimicneoai.immunogenicity_prediction.runtime import inference

        return getattr(inference, name)
    if name == "normalize_hla":
        from mimicneoai.immunogenicity_prediction.runtime.hla import normalize_hla

        return normalize_hla
    if name == "annotate_input_qc":
        from mimicneoai.immunogenicity_prediction.runtime.qc import annotate_input_qc

        return annotate_input_qc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
