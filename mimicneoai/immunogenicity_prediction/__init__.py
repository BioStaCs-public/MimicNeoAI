__all__ = [
    "InferenceConfig",
    "prepare_inference_features",
    "run_inference",
    "export_model_to_onnx",
    "default_model_paths",
    "run_default_inference",
]


def __getattr__(name):
    if name in __all__:
        from mimicneoai.immunogenicity_prediction.api import (
            InferenceConfig,
            prepare_inference_features,
            run_inference,
            default_model_paths,
            run_default_inference,
        )
        from mimicneoai.immunogenicity_prediction.core import export_model_to_onnx

        values = {
            "InferenceConfig": InferenceConfig,
            "prepare_inference_features": prepare_inference_features,
            "run_inference": run_inference,
            "export_model_to_onnx": export_model_to_onnx,
            "default_model_paths": default_model_paths,
            "run_default_inference": run_default_inference,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
