"""NetMHCIIpan binding predictor adapter skeleton."""

from __future__ import annotations

from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.predictors.base import (
    BindingPrediction,
)
from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.task_utils import (
    BindingTask,
)


class NetMhcIiPanPredictor:
    """Adapter for NetMHCIIpan predictions."""

    name = "NetMHCIIpan"

    def predict(self, tasks: list[BindingTask]) -> list[BindingPrediction]:
        """Predict binding for NetMHCIIpan-compatible tasks."""

        raise NotImplementedError

