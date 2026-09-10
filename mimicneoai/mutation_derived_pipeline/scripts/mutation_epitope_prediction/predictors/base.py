"""Base interfaces for binding predictor adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from typing import Protocol

from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.task_utils import (
    BindingTask,
)


@dataclass(frozen=True)
class BindingPrediction:
    """Unified prediction result for one peptide-HLA-algorithm task."""

    peptide: str
    hla_allele: str
    algorithm: str
    ic50: Optional[float] = None
    percentile: Optional[float] = None
    score: Optional[float] = None


class BindingPredictor(Protocol):
    """Common interface implemented by each predictor adapter."""

    name: str

    def predict(self, tasks: list[BindingTask]) -> list[BindingPrediction]:
        """Predict binding for a list of tasks."""
