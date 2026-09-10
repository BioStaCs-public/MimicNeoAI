"""Protein FASTA and epitope window utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ProteinPair:
    """WT/MT protein sequence pair for one mutation-transcript event."""

    event_id: str
    wt_sequence: Optional[str]
    mt_sequence: str


@dataclass(frozen=True)
class EpitopeWindow:
    """Mutation-covering epitope window with optional WT counterpart."""

    event_id: str
    peptide: str
    peptide_type: str
    length: int
    start: int
    end: int
    wt_peptide: Optional[str] = None
    extended_peptide: Optional[str] = None
    extended_start: Optional[int] = None
    extended_end: Optional[int] = None


@dataclass(frozen=True)
class ExtendedPeptide:
    """A contiguous MT protein segment and its zero-based coordinates."""

    sequence: str
    start: int
    end: int


def read_pvacseq_protein_pairs(fasta_file: Path) -> list[ProteinPair]:
    """Read WT/MT protein sequence pairs from pVACtools FASTA output."""

    records: dict[str, dict[str, str]] = {}
    current_id: Optional[str] = None
    chunks: list[str] = []

    def flush() -> None:
        if current_id is None:
            return
        seq_type, event_key = _parse_pvacseq_record_id(current_id)
        records.setdefault(event_key, {})[seq_type] = "".join(chunks)

    with Path(fasta_file).open() as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush()
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        flush()

    protein_pairs: list[ProteinPair] = []
    for event_key, seqs in records.items():
        mt_sequence = seqs.get("MT")
        if not mt_sequence:
            continue
        protein_pairs.append(
            ProteinPair(
                event_id=event_key,
                wt_sequence=seqs.get("WT"),
                mt_sequence=mt_sequence,
            )
        )
    return protein_pairs


def generate_epitope_windows(
    protein_pair: ProteinPair,
    epitope_lengths: tuple[int, ...],
    extended_length: int,
    variant_type: str = "",
) -> list[EpitopeWindow]:
    """Generate mutation-covering epitope windows for one WT/MT protein pair.

    Retained windows must overlap the altered protein region and differ from
    their matched WT sequence. In-frame deletions additionally require a
    junction-aware WT comparison because MT and WT coordinates diverge.
    """

    mutation_start, mutation_end = locate_mutation_region(
        protein_pair.wt_sequence,
        protein_pair.mt_sequence,
        variant_type=variant_type,
    )
    windows: list[EpitopeWindow] = []
    mt_len = len(protein_pair.mt_sequence)
    for length in epitope_lengths:
        if length <= 0 or mt_len < length:
            continue
        if variant_type == "inframe_del" and protein_pair.wt_sequence is not None:
            windows.extend(
                _generate_inframe_deletion_windows(
                    protein_pair=protein_pair,
                    length=length,
                    extended_length=extended_length,
                    mutation_start=mutation_start,
                    mutation_end=mutation_end,
                )
            )
            continue
        for start in range(0, mt_len - length + 1):
            end = start + length
            if not window_covers_mutation(start, end, mutation_start, mutation_end):
                continue
            mt_peptide = protein_pair.mt_sequence[start:end]
            if contains_invalid_amino_acid(mt_peptide):
                continue
            wt_peptide = None
            if (
                variant_type.upper() != "FS"
                and protein_pair.wt_sequence is not None
                and len(protein_pair.wt_sequence) >= end
            ):
                wt_peptide = protein_pair.wt_sequence[start:end]
                if contains_invalid_amino_acid(wt_peptide):
                    continue
                if wt_peptide == mt_peptide:
                    continue
            extension = make_candidate_extended_peptide(
                sequence=protein_pair.mt_sequence,
                peptide_start=start,
                peptide_end=end,
                mutation_start=mutation_start,
                mutation_end=mutation_end,
                extended_length=extended_length,
                variant_type=variant_type,
            )
            windows.append(
                EpitopeWindow(
                    event_id=protein_pair.event_id,
                    peptide=mt_peptide,
                    peptide_type="MT",
                    length=length,
                    start=start,
                    end=end,
                    wt_peptide=wt_peptide,
                    extended_peptide=extension.sequence,
                    extended_start=extension.start,
                    extended_end=extension.end,
                )
            )
    return windows


def _generate_inframe_deletion_windows(
    *,
    protein_pair: ProteinPair,
    length: int,
    extended_length: int,
    mutation_start: int,
    mutation_end: int,
) -> list[EpitopeWindow]:
    """Generate in-frame deletion windows using junction-aware WT matching."""

    if protein_pair.wt_sequence is None:
        return []

    mt_windows = _subpeptides_by_position(protein_pair.mt_sequence, length)
    wt_windows = _subpeptides_by_position(protein_pair.wt_sequence, length)
    deletion_length = len(wt_windows) - len(mt_windows)
    windows: list[EpitopeWindow] = []
    previous_match_direction: Optional[str] = None
    previous_wt_position: Optional[int] = None

    for mt_position in sorted(mt_windows):
        start = mt_position - 1
        end = start + length
        if not window_covers_mutation(start, end, mutation_start, mutation_end):
            continue
        mt_peptide = mt_windows[mt_position]
        if contains_invalid_amino_acid(mt_peptide):
            continue

        baseline_wt = wt_windows.get(mt_position)
        if baseline_wt is None or contains_invalid_amino_acid(baseline_wt):
            continue

        if baseline_wt == mt_peptide:
            previous_match_direction = "left"
            previous_wt_position = mt_position
            continue

        if previous_match_direction == "right" and previous_wt_position is not None:
            best_wt_position = previous_wt_position + 1
            best_wt = wt_windows.get(best_wt_position)
            match_direction = "right"
        else:
            left_match_count = _consecutive_matches_from_left(mt_peptide, baseline_wt)
            alternate_wt_position = mt_position + deletion_length
            alternate_wt = wt_windows.get(alternate_wt_position)
            if alternate_wt is not None:
                right_match_count = _consecutive_matches_from_right(mt_peptide, alternate_wt)
            else:
                right_match_count = -1
            if alternate_wt is not None and right_match_count > left_match_count:
                best_wt_position = alternate_wt_position
                best_wt = alternate_wt
                match_direction = "right"
            else:
                best_wt_position = mt_position
                best_wt = baseline_wt
                match_direction = "left"

        previous_match_direction = match_direction
        previous_wt_position = best_wt_position
        if best_wt is None or contains_invalid_amino_acid(best_wt) or best_wt == mt_peptide:
            continue

        extension = make_candidate_extended_peptide(
            sequence=protein_pair.mt_sequence,
            peptide_start=start,
            peptide_end=end,
            mutation_start=mutation_start,
            mutation_end=mutation_end,
            extended_length=extended_length,
            variant_type="inframe_del",
        )
        windows.append(
            EpitopeWindow(
                event_id=protein_pair.event_id,
                peptide=mt_peptide,
                peptide_type="MT",
                length=length,
                start=start,
                end=end,
                wt_peptide=best_wt,
                extended_peptide=extension.sequence,
                extended_start=extension.start,
                extended_end=extension.end,
            )
        )

    return windows


def _subpeptides_by_position(sequence: str, length: int) -> dict[int, str]:
    """Return 1-based sub-peptide position to peptide sequence."""

    return {
        start + 1: sequence[start : start + length]
        for start in range(0, len(sequence) - length + 1)
    }


def _consecutive_matches_from_left(first: str, second: str) -> int:
    """Count consecutive identical amino acids from the left."""

    count = 0
    for a, b in zip(first, second):
        if a != b:
            break
        count += 1
    return count


def _consecutive_matches_from_right(first: str, second: str) -> int:
    """Count consecutive identical amino acids from the right."""

    count = 0
    for a, b in zip(reversed(first), reversed(second)):
        if a != b:
            break
        count += 1
    return count


def locate_mutation_region(
    wt_sequence: Optional[str],
    mt_sequence: str,
    variant_type: str = "",
) -> tuple[int, int]:
    """Infer the changed amino-acid interval between WT and MT sequences."""

    if wt_sequence is None:
        return (0, len(mt_sequence))

    prefix = 0
    max_prefix = min(len(wt_sequence), len(mt_sequence))
    while prefix < max_prefix and wt_sequence[prefix] == mt_sequence[prefix]:
        prefix += 1

    if variant_type.upper() == "FS":
        # Every translated residue after the frameshift breakpoint belongs to
        # the novel reading frame. Do not trim a coincidentally matching suffix.
        return (prefix, len(mt_sequence))

    wt_suffix = len(wt_sequence)
    mt_suffix = len(mt_sequence)
    while wt_suffix > prefix and mt_suffix > prefix and wt_sequence[wt_suffix - 1] == mt_sequence[mt_suffix - 1]:
        wt_suffix -= 1
        mt_suffix -= 1

    return (prefix, mt_suffix)


def window_covers_mutation(window_start: int, window_end: int, mutation_start: int, mutation_end: int) -> bool:
    """Return True if a peptide window overlaps or spans the changed region."""

    if mutation_start == mutation_end:
        # In-frame deletions are represented as a zero-length interval in the
        # mutant sequence. A valid peptide must include residues on both sides
        # of the new deletion junction.
        return window_start < mutation_start and window_end > mutation_start
    return window_start < mutation_end and window_end > mutation_start


def contains_invalid_amino_acid(sequence: str) -> bool:
    """Return True when a sequence contains unsupported amino acids."""

    return any(character in sequence for character in ("*", "X", "?", "U"))


def make_event_context_peptide(
    sequence: str,
    mutation_start: int,
    mutation_end: int,
    extended_length: int,
    variant_type: str = "",
) -> ExtendedPeptide:
    """Create one mutation-centered context sequence for an event."""

    if extended_length <= 0 or len(sequence) <= extended_length:
        return ExtendedPeptide(sequence=sequence, start=0, end=len(sequence))
    if variant_type.upper() == "FS":
        center = mutation_start
    elif mutation_end > mutation_start:
        center = (mutation_start + mutation_end - 1) // 2
    else:
        center = mutation_start
    start = max(0, center - extended_length // 2)
    end = start + extended_length
    if end > len(sequence):
        end = len(sequence)
        start = max(0, end - extended_length)
    return ExtendedPeptide(sequence=sequence[start:end], start=start, end=end)


def make_candidate_extended_peptide(
    *,
    sequence: str,
    peptide_start: int,
    peptide_end: int,
    mutation_start: int,
    mutation_end: int,
    extended_length: int,
    variant_type: str = "",
) -> ExtendedPeptide:
    """Extend one MT epitope while retaining the complete epitope sequence."""

    sequence_length = len(sequence)
    if not (0 <= peptide_start < peptide_end <= sequence_length):
        raise ValueError(
            f"Invalid peptide coordinates {peptide_start}-{peptide_end} "
            f"for MT sequence length {sequence_length}"
        )

    peptide = sequence[peptide_start:peptide_end]
    if contains_invalid_amino_acid(peptide):
        raise ValueError(f"MT epitope contains an invalid amino acid: {peptide}")

    invalid_positions = [
        index
        for index, amino_acid in enumerate(sequence)
        if amino_acid in {"*", "X", "?", "U"}
    ]
    valid_start = max(
        (index + 1 for index in invalid_positions if index < peptide_start),
        default=0,
    )
    valid_end = min(
        (index for index in invalid_positions if index >= peptide_end),
        default=sequence_length,
    )

    peptide_length = peptide_end - peptide_start
    available_length = valid_end - valid_start
    target_length = min(max(int(extended_length), peptide_length), available_length)
    if available_length <= target_length:
        extension = ExtendedPeptide(
            sequence=sequence[valid_start:valid_end],
            start=valid_start,
            end=valid_end,
        )
    else:
        latest_sequence_start = valid_end - target_length
        earliest_start = max(valid_start, peptide_end - target_length)
        latest_start = min(peptide_start, latest_sequence_start)
        if earliest_start > latest_start:
            raise ValueError(
                f"Cannot place peptide {peptide_start}-{peptide_end} inside "
                f"an extension of length {target_length}"
            )

        if variant_type.upper() == "FS":
            anchor = (peptide_start + peptide_end - 1) // 2
        elif mutation_end > mutation_start:
            anchor = (mutation_start + mutation_end - 1) // 2
        else:
            anchor = mutation_start
        ideal_start = anchor - target_length // 2
        start = min(max(ideal_start, earliest_start), latest_start)
        end = start + target_length
        extension = ExtendedPeptide(sequence=sequence[start:end], start=start, end=end)

    if peptide not in extension.sequence:
        raise ValueError(
            f"Extended peptide does not contain MT epitope: peptide={peptide} "
            f"extension={extension.sequence}"
        )
    return extension


def make_extended_peptide(sequence: str, mutation_start: int, mutation_end: int, extended_length: int) -> str:
    """Backward-compatible string wrapper for event-level context generation."""

    return make_event_context_peptide(
        sequence,
        mutation_start,
        mutation_end,
        extended_length,
    ).sequence


def _parse_pvacseq_record_id(record_id: str) -> tuple[str, str]:
    """Parse pVACtools FASTA IDs such as ``WT.1.GENE.TX.missense.1A/B``."""

    seq_type, _, event_key = record_id.partition(".")
    if seq_type not in {"WT", "MT"} or not event_key:
        raise ValueError(f"Unexpected pVACtools FASTA record ID: {record_id}")
    return seq_type, event_key


def build_protein_pair_from_converter_row(row: dict[str, str], flanking_length: int) -> ProteinPair:
    """Rebuild a pVACtools-like WT/MT protein pair from one converter TSV row.

    This is used as a fallback when ``generate_protein_fasta`` skips an entire
    long FASTA record because the long fragment contains an invalid amino acid
    such as ``X``. Short windows that do not contain the invalid residue may
    still be valid, which matches pVACseq's all-epitopes behavior.
    """

    variant_type = row["variant_type"]
    full_wt = row["wildtype_amino_acid_sequence"]
    aa_change = row["amino_acid_change"]
    protein_position = row["protein_position"]
    event_id = row["index"]

    if variant_type == "FS":
        position = int(protein_position.split("-", 1)[0]) - 1
        start = max(0, position - flanking_length)
        wt_stop = position + flanking_length
        wt_subsequence = full_wt[start:wt_stop]
        mt_sequence = row["frameshift_amino_acid_sequence"]
        mt_subsequence = mt_sequence[start:]
        return ProteinPair(event_id=event_id, wt_sequence=wt_subsequence, mt_sequence=mt_subsequence)

    if "/" not in aa_change:
        raise ValueError(f"Unsupported amino acid change for {event_id}: {aa_change}")

    wt_aa, mt_aa = aa_change.split("/", 1)
    wt_aa = _strip_stop_suffix(wt_aa)
    mt_aa = _strip_stop_suffix(mt_aa)

    if variant_type in {"missense", "inframe_ins"}:
        if wt_aa == "-":
            position = int(protein_position.split("-", 1)[0])
            wt_aa_len = 0
        else:
            position = int(protein_position.split("-", 1)[0]) - 1
            wt_aa_len = len(wt_aa)
    elif variant_type == "inframe_del":
        position = int(protein_position.split("-", 1)[0]) - 1
        wt_aa_len = len(wt_aa)
        if mt_aa == "-":
            mt_aa = ""
    else:
        raise ValueError(f"Unsupported variant type for {event_id}: {variant_type}")

    mutation_position, wt_subsequence = _get_wildtype_subsequence(
        position=position,
        full_wildtype_sequence=full_wt,
        wildtype_amino_acid_length=wt_aa_len,
        flanking_length=flanking_length,
    )
    mutation_end = mutation_position + wt_aa_len
    mt_subsequence = wt_subsequence[:mutation_position] + mt_aa + wt_subsequence[mutation_end:]
    return ProteinPair(event_id=event_id, wt_sequence=wt_subsequence, mt_sequence=mt_subsequence)


def _get_wildtype_subsequence(
    *,
    position: int,
    full_wildtype_sequence: str,
    wildtype_amino_acid_length: int,
    flanking_length: int,
) -> tuple[int, str]:
    """Extract a mutation-centered WT subsequence with protein-end adjustment."""

    peptide_sequence_length = min(2 * flanking_length + wildtype_amino_acid_length, len(full_wildtype_sequence))
    if position < flanking_length:
        return position, full_wildtype_sequence[:peptide_sequence_length]
    distance_from_end = len(full_wildtype_sequence) - 1 - position
    if distance_from_end < flanking_length:
        start = len(full_wildtype_sequence) - peptide_sequence_length
        mutation_position = peptide_sequence_length - distance_from_end - 1
        return mutation_position, full_wildtype_sequence[start:]
    start = position - flanking_length
    end = start + peptide_sequence_length
    return flanking_length, full_wildtype_sequence[start:end]


def _strip_stop_suffix(amino_acid: str) -> str:
    """Remove stop-codon suffix notation from an amino-acid change part."""

    if "*" in amino_acid:
        return amino_acid.split("*", 1)[0]
    if "X" in amino_acid:
        return amino_acid.split("X", 1)[0]
    return amino_acid
