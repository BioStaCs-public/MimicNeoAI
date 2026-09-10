import pandas as pd

from mimicneoai.immunogenicity_prediction.runtime.qc import annotate_input_qc


def test_input_qc_flags_noncanonical_missing_and_duplicate_rows():
    result = annotate_input_qc(
        pd.DataFrame(
            {
                "peptide": ["SIINFEKL", "siinfekl", "PEPTOIDE", None],
                "hla": ["HLA-A*02:01", "A*02:01", "HLA-B*07:02", None],
            }
        ),
        peptide_col="peptide",
        hla_col="hla",
    )

    assert result["input_qc_normalized_hla"].tolist() == ["A*02:01", "A*02:01", "B*07:02", ""]
    assert result["input_qc_peptide_length"].tolist() == [8, 8, 8, pd.NA]
    assert result.loc[0, "input_qc_flags"] == "duplicate_peptide_hla"
    assert result.loc[1, "input_qc_flags"] == "duplicate_peptide_hla"
    assert result.loc[2, "input_qc_flags"] == "noncanonical_amino_acid:O"
    assert result.loc[3, "input_qc_flags"] == "missing_peptide;missing_hla"
    assert result["input_qc_status"].tolist() == ["warning", "warning", "warning", "warning"]
