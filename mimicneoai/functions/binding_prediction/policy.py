"""Central binding prediction presets.

The presets keep user configuration compact while preserving a reproducible
record of peptide lengths, Stage 1 routing thresholds, and Stage 2 predictors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BindingPredictionPolicy:
    """Resolved binding prediction strategy."""

    name: str
    two_stage: bool
    mhc_i_lengths: tuple[int, ...]
    mhc_ii_lengths: tuple[int, ...]
    mhc_i_algorithms: tuple[str, ...]
    mhc_ii_algorithms: tuple[str, ...]
    stage1_mhc_i_algorithm: str = "NetMHCpanEL"
    stage1_mhc_ii_algorithm: str = "NetMHCIIpanEL"
    stage1_mhc_i_rank_lt: float = 10.0
    stage1_mhc_ii_rank_lt: float = 10.0

    @property
    def algorithms(self) -> tuple[str, ...]:
        return self.mhc_i_algorithms + self.mhc_ii_algorithms

    @property
    def stage1_algorithms(self) -> tuple[str, ...]:
        if not self.two_stage:
            return ()
        return (self.stage1_mhc_i_algorithm, self.stage1_mhc_ii_algorithm)


FULL_ALGORITHMS_MHC_I = (
    "MHCflurry",
    "MHCflurryEL",
    "MHCnuggetsI",
    "NetMHCpan",
    "NetMHCpanEL",
)
FULL_ALGORITHMS_MHC_II = (
    "MHCnuggetsII",
    "NNalign",
    "NetMHCIIpan",
    "NetMHCIIpanEL",
)
FAST_ALGORITHMS_MHC_II = (
    "MHCnuggetsII",
    "NetMHCIIpan",
    "NetMHCIIpanEL",
)


PRESETS: dict[str, BindingPredictionPolicy] = {
    "full": BindingPredictionPolicy(
        name="full",
        two_stage=False,
        mhc_i_lengths=(8, 9, 10, 11),
        mhc_ii_lengths=(13, 14, 15, 16, 17),
        mhc_i_algorithms=FULL_ALGORITHMS_MHC_I,
        mhc_ii_algorithms=FULL_ALGORITHMS_MHC_II,
    ),
    "fast": BindingPredictionPolicy(
        name="fast",
        two_stage=True,
        mhc_i_lengths=(8, 9, 10, 11),
        mhc_ii_lengths=(13, 14, 15, 16, 17),
        mhc_i_algorithms=FULL_ALGORITHMS_MHC_I,
        mhc_ii_algorithms=FAST_ALGORITHMS_MHC_II,
    ),
}


def resolve_binding_prediction_policy(name: str | None) -> BindingPredictionPolicy | None:
    """Return the named policy, or ``None`` when no preset was requested."""

    if name is None or not str(name).strip():
        return None
    key = str(name).strip().lower()
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported binding_prediction_preset: {name}. "
            f"Supported presets: {', '.join(sorted(PRESETS))}"
        ) from exc


def parse_int_list(value: str) -> tuple[int, ...]:
    """Parse comma- or whitespace-separated integer values."""

    values = tuple(sorted({int(item.strip()) for item in value.replace(",", " ").split() if item.strip()}))
    if not values:
        raise ValueError("At least one peptide length is required.")
    return values


def parse_algorithm_list(value: str) -> tuple[str, ...]:
    """Parse comma- or whitespace-separated algorithm names."""

    return tuple(item.strip() for item in value.replace(",", " ").split() if item.strip())
