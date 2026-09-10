"""Versioned policies for sequence-based molecular mimicry analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceMimicryPolicy:
    """Immutable eligibility and sequence-matching rules."""

    name: str
    peptide_lengths: tuple[int, ...]
    central_identity_threshold: float
    minimum_central_run: int
    minimum_positional_matches: int
    canonical_amino_acids: frozenset[str]
    primary_call: str

    def central_indices(self, peptide_length: int) -> tuple[int, ...]:
        """Return zero-based indices corresponding to P4..Pmin(8, L-1)."""

        if peptide_length not in self.peptide_lengths:
            raise ValueError(
                f"Peptide length {peptide_length} is outside {self.name}: "
                f"{self.peptide_lengths}"
            )
        return tuple(range(3, min(8, peptide_length - 1)))


@dataclass(frozen=True)
class SharedHlaBindingPolicy:
    """Post hoc evidence policy for shared classical HLA-I support."""

    name: str
    maximum_best_ic50_nm: float
    maximum_median_ic50_nm: float
    maximum_best_percentile: float
    maximum_median_percentile: float
    allowed_loci: tuple[str, ...]


SEQUENCE_MIMICRY_V1_2 = SequenceMimicryPolicy(
    name="sequence_mimicry_v1.2",
    peptide_lengths=(8, 9, 10, 11),
    central_identity_threshold=0.60,
    minimum_central_run=3,
    minimum_positional_matches=4,
    canonical_amino_acids=frozenset("ACDEFGHIKLMNPQRSTVWY"),
    primary_call="sequence_based_mimicry_v12_candidate",
)

SHARED_CLASSICAL_HLA_I_BINDING_V1 = SharedHlaBindingPolicy(
    name="shared_classical_hla_i_binding_v1.0",
    maximum_best_ic50_nm=500.0,
    maximum_median_ic50_nm=500.0,
    maximum_best_percentile=2.0,
    maximum_median_percentile=2.0,
    allowed_loci=("HLA-A*", "HLA-B*", "HLA-C*"),
)


POLICIES = {SEQUENCE_MIMICRY_V1_2.name: SEQUENCE_MIMICRY_V1_2}


def get_policy(name: str) -> SequenceMimicryPolicy:
    """Resolve a supported frozen policy by name."""

    try:
        return POLICIES[str(name).strip()]
    except KeyError as exc:
        supported = ", ".join(sorted(POLICIES))
        raise ValueError(f"Unsupported mimicry method {name!r}; supported: {supported}") from exc
