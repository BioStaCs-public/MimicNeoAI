"""Sequence-based molecular mimicry analysis.

The public API intentionally exposes only the frozen policy, pure sequence
metrics, and the high-level run entry point. Input adapters and execution
details remain internal so their schemas can evolve without expanding the
supported API surface.
"""

from .api import (
    SEQUENCE_MIMICRY_V1_2,
    MimicryRunConfig,
    SHARED_CLASSICAL_HLA_I_BINDING_V1,
    SequenceMimicryPolicy,
    SharedHlaBindingPolicy,
    central_positions,
    compute_pair_metrics,
    get_policy,
    is_sequence_mimic,
    run_sequence_mimicry,
)

__all__ = [
    "SEQUENCE_MIMICRY_V1_2",
    "MimicryRunConfig",
    "SHARED_CLASSICAL_HLA_I_BINDING_V1",
    "SequenceMimicryPolicy",
    "SharedHlaBindingPolicy",
    "central_positions",
    "compute_pair_metrics",
    "get_policy",
    "is_sequence_mimic",
    "run_sequence_mimicry",
]
