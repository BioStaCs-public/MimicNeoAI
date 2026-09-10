"""Stable runtime inference import surface.

The implementation remains in ``core`` during the compatibility migration so
existing downstream imports continue to work unchanged.
"""

from mimicneoai.immunogenicity_prediction.core import (
    InferenceConfig,
    export_model_to_onnx,
    prepare_inference_features,
    run_inference,
)

__all__ = [
    "InferenceConfig",
    "export_model_to_onnx",
    "prepare_inference_features",
    "run_inference",
]
