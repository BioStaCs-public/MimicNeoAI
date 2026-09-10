"""Input-level QC annotations for peptide-HLA inference."""

from __future__ import annotations

import pandas as pd

from mimicneoai.immunogenicity_prediction.runtime.hla import normalize_hla


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
INPUT_QC_COLUMNS = (
    "input_qc_peptide_length",
    "input_qc_normalized_hla",
    "input_qc_flags",
    "input_qc_status",
)


def _clean_peptide(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def _clean_hla(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def annotate_input_qc(
    input_df: pd.DataFrame,
    *,
    peptide_col: str,
    hla_col: str,
) -> pd.DataFrame:
    """Return a copy with non-destructive peptide-HLA input QC annotations."""
    if peptide_col not in input_df.columns or hla_col not in input_df.columns:
        raise KeyError(f"Input must contain columns '{peptide_col}' and '{hla_col}'.")

    output = input_df.copy()
    peptides = [_clean_peptide(value) for value in output[peptide_col]]
    hlas = [_clean_hla(value) for value in output[hla_col]]
    normalized_hlas = [normalize_hla(value) if value else "" for value in hlas]
    pair_keys = pd.Series(
        [f"{hla}\x1f{peptide}" if hla and peptide else None for hla, peptide in zip(normalized_hlas, peptides)],
        index=output.index,
    )
    duplicate_pairs = pair_keys.notna() & pair_keys.duplicated(keep=False)

    flags: list[str] = []
    for peptide, hla, is_duplicate in zip(peptides, hlas, duplicate_pairs):
        row_flags: list[str] = []
        if not peptide:
            row_flags.append("missing_peptide")
        else:
            invalid = sorted(set(peptide).difference(CANONICAL_AMINO_ACIDS))
            if invalid:
                row_flags.append("noncanonical_amino_acid:" + "".join(invalid))
        if not hla:
            row_flags.append("missing_hla")
        if bool(is_duplicate):
            row_flags.append("duplicate_peptide_hla")
        flags.append(";".join(row_flags))

    output["input_qc_peptide_length"] = pd.array(
        [len(peptide) if peptide else pd.NA for peptide in peptides], dtype="Int64"
    )
    output["input_qc_normalized_hla"] = normalized_hlas
    output["input_qc_flags"] = flags
    output["input_qc_status"] = ["pass" if not item else "warning" for item in flags]
    return output
