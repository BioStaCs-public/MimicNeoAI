"""Pure sequence metrics for the frozen v1.2 mimicry policy."""
from __future__ import annotations

from math import ceil

from ._models import PairMetrics
from ._policy import SEQUENCE_MIMICRY_V1_2, SequenceMimicryPolicy


_BLOSUM62_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_ROWS = (
    "4 -1 -2 -2 0 -1 -1 0 -2 -1 -1 -1 -1 -2 -1 1 0 -3 -2 0",
    "-1 5 0 -2 -3 1 0 -2 0 -3 -2 2 -1 -3 -2 -1 -1 -3 -2 -3",
    "-2 0 6 1 -3 0 0 0 1 -3 -3 0 -2 -3 -2 1 0 -4 -2 -3",
    "-2 -2 1 6 -3 0 2 -1 -1 -3 -4 -1 -3 -3 -1 0 -1 -4 -3 -3",
    "0 -3 -3 -3 9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1",
    "-1 1 0 0 -3 5 2 -2 0 -3 -2 1 0 -3 -1 0 -1 -2 -1 -2",
    "-1 0 0 2 -4 2 5 -2 0 -3 -3 1 -2 -3 -1 0 -1 -3 -2 -2",
    "0 -2 0 -1 -3 -2 -2 6 -2 -4 -4 -2 -3 -3 -2 0 -2 -2 -3 -3",
    "-2 0 1 -1 -3 0 0 -2 8 -3 -3 -1 -2 -1 -2 -1 -2 -2 2 -3",
    "-1 -3 -3 -3 -1 -3 -3 -4 -3 4 2 -3 1 0 -3 -2 -1 -3 -1 3",
    "-1 -2 -3 -4 -1 -2 -3 -4 -3 2 4 -2 2 0 -3 -2 -1 -2 -1 1",
    "-1 2 0 -1 -3 1 1 -2 -1 -3 -2 5 -1 -3 -1 0 -1 -3 -2 -2",
    "-1 -1 -2 -3 -1 0 -2 -3 -2 1 2 -1 5 0 -2 -1 -1 -1 -1 1",
    "-2 -3 -3 -3 -2 -3 -3 -3 -1 0 0 -3 0 6 -4 -2 -2 1 3 -1",
    "-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4 7 -1 -1 -4 -3 -2",
    "1 -1 1 0 -1 0 0 0 -1 -2 -2 0 -1 -2 -1 4 1 -3 -2 -2",
    "0 -1 0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1 1 5 -2 -2 0",
    "-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1 1 -4 -3 -2 11 2 -3",
    "-2 -2 -2 -3 -2 -1 -2 -3 2 -1 -1 -2 -1 3 -3 -2 -2 2 7 -1",
    "0 -3 -3 -3 -1 -2 -2 -3 -3 3 1 -2 1 -1 -2 -2 0 -3 -1 4",
)
_BLOSUM62 = {
    (row_aa, column_aa): int(value)
    for row_aa, row in zip(_BLOSUM62_ORDER, _BLOSUM62_ROWS)
    for column_aa, value in zip(_BLOSUM62_ORDER, row.split())
}


def normalize_peptide(peptide: str) -> str:
    """Normalize a peptide without altering its residues."""

    return str(peptide).strip().upper()


def is_eligible_peptide(
    peptide: str,
    policy: SequenceMimicryPolicy = SEQUENCE_MIMICRY_V1_2,
) -> bool:
    """Return whether a peptide is eligible for the selected sequence policy."""

    normalized = normalize_peptide(peptide)
    return (
        len(normalized) in policy.peptide_lengths
        and bool(normalized)
        and set(normalized) <= policy.canonical_amino_acids
    )


def central_positions(
    peptide_length: int,
    policy: SequenceMimicryPolicy = SEQUENCE_MIMICRY_V1_2,
) -> tuple[int, ...]:
    """Return zero-based central indices for the policy's P4-based window."""

    return policy.central_indices(peptide_length)


def _longest_run_at_positions(
    peptide_a: str,
    peptide_b: str,
    positions: tuple[int, ...],
) -> tuple[int, str]:
    best_length = 0
    best_end = 0
    current_length = 0
    for position in positions:
        if peptide_a[position] == peptide_b[position]:
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_end = position + 1
        else:
            current_length = 0
    return best_length, peptide_a[best_end - best_length : best_end]


def _longest_common_substring(peptide_a: str, peptide_b: str) -> tuple[int, str]:
    previous = [0] * (len(peptide_b) + 1)
    best_length = 0
    best_end = 0
    for index_a, residue_a in enumerate(peptide_a, 1):
        current = [0] * (len(peptide_b) + 1)
        for index_b, residue_b in enumerate(peptide_b, 1):
            if residue_a == residue_b:
                current[index_b] = previous[index_b - 1] + 1
                if current[index_b] > best_length:
                    best_length = current[index_b]
                    best_end = index_a
        previous = current
    return best_length, peptide_a[best_end - best_length : best_end]


def _blosum62_score(
    peptide_a: str,
    peptide_b: str,
    positions: tuple[int, ...],
) -> int:
    return sum(_BLOSUM62[(peptide_a[index], peptide_b[index])] for index in positions)


def compute_pair_metrics(
    peptide_a: str,
    peptide_b: str,
    policy: SequenceMimicryPolicy = SEQUENCE_MIMICRY_V1_2,
) -> PairMetrics:
    """Calculate primary and annotation-only metrics for an equal-length pair."""

    peptide_a = normalize_peptide(peptide_a)
    peptide_b = normalize_peptide(peptide_b)
    if len(peptide_a) != len(peptide_b):
        raise ValueError("Sequence mimicry metrics require equal-length peptides")
    if not is_eligible_peptide(peptide_a, policy) or not is_eligible_peptide(peptide_b, policy):
        raise ValueError("Sequence mimicry metrics require eligible canonical HLA-I peptides")

    positions = central_positions(len(peptide_a), policy)
    central_matches = sum(peptide_a[index] == peptide_b[index] for index in positions)
    positional_matches = sum(a == b for a, b in zip(peptide_a, peptide_b))
    central_run_length, central_run_sequence = _longest_run_at_positions(
        peptide_a, peptide_b, positions
    )
    legacy_lcs_length, legacy_lcs_sequence = _longest_common_substring(peptide_a, peptide_b)
    p2_exact = peptide_a[1] == peptide_b[1]
    pomega_exact = peptide_a[-1] == peptide_b[-1]
    return PairMetrics(
        central_positions=",".join(f"P{index + 1}" for index in positions),
        central_matches=central_matches,
        central_total=len(positions),
        central_identity=central_matches / len(positions),
        central_run_length=central_run_length,
        central_run_sequence=central_run_sequence,
        positional_match_count=positional_matches,
        full_length_identity=positional_matches / len(peptide_a),
        p2_exact_match=p2_exact,
        pomega_exact_match=pomega_exact,
        blosum62_full_score=_blosum62_score(
            peptide_a, peptide_b, tuple(range(len(peptide_a)))
        ),
        blosum62_central_score=_blosum62_score(peptide_a, peptide_b, positions),
        legacy_lcs_length=legacy_lcs_length,
        legacy_lcs_sequence=legacy_lcs_sequence,
    )


def is_sequence_mimic(
    peptide_a: str,
    peptide_b: str,
    policy: SequenceMimicryPolicy = SEQUENCE_MIMICRY_V1_2,
) -> bool:
    """Apply the frozen primary sequence-mimicry call."""

    peptide_a = normalize_peptide(peptide_a)
    peptide_b = normalize_peptide(peptide_b)
    if len(peptide_a) != len(peptide_b):
        return False
    if not is_eligible_peptide(peptide_a, policy) or not is_eligible_peptide(peptide_b, policy):
        return False
    metrics = compute_pair_metrics(peptide_a, peptide_b, policy)
    required_central_matches = ceil(
        metrics.central_total * policy.central_identity_threshold - 1e-12
    )
    return (
        metrics.central_matches >= required_central_matches
        and metrics.central_run_length >= policy.minimum_central_run
        and metrics.positional_match_count >= policy.minimum_positional_matches
    )


def central_triplet_offsets(
    peptide_length: int,
    policy: SequenceMimicryPolicy = SEQUENCE_MIMICRY_V1_2,
) -> tuple[int, ...]:
    """Return central offsets at which a required exact 3-mer can begin."""

    positions = central_positions(peptide_length, policy)
    return tuple(index for index in positions if index + 2 <= positions[-1])
