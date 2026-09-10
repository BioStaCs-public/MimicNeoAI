from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from mimicneoai.mimicry._binding import annotate_shared_hla_support
from mimicneoai.mimicry._engine import PAIR_FIELDS
from mimicneoai.mimicry._models import InputEntry


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class BindingSupportTests(unittest.TestCase):
    def test_shared_classical_allele_is_post_hoc_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_path = root / "pairs.tsv"
            pair_row = {field: "" for field in PAIR_FIELDS}
            pair_row.update(
                {
                    "candidate_id": "candidate-1",
                    "method_version": "sequence_mimicry_v1.2",
                    "patient_id": "P1",
                    "source_pair": "microbial--mutation-derived",
                    "source_a": "microbial",
                    "source_b": "mutation-derived",
                    "peptide_a": "AAAAAAAA",
                    "peptide_b": "CAAAAACC",
                    "peptide_length": 8,
                    "primary_call": "sequence_based_mimicry_v12_candidate",
                }
            )
            write_tsv(pair_path, [pair_row])

            microbial_binding = root / "microbial_binding.tsv"
            write_tsv(
                microbial_binding,
                [
                    {
                        "Epitope Seq": "AAAAAAAA",
                        "HLA Allele": "HLA-A*02:01",
                        "Best IC50 Score": 40,
                        "Median IC50 Score": 80,
                        "Best Percentile": 0.2,
                        "Median Percentile": 0.7,
                    }
                ],
            )
            mutation_binding = root / "mutation_binding.tsv"
            write_tsv(
                mutation_binding,
                [
                    {
                        "MT Epitope Seq": "CAAAAACC",
                        "HLA Allele": "HLA-A*02:01",
                        "Best MT IC50 Score": 100,
                        "Median MT IC50 Score": 200,
                        "Best MT Percentile": 0.5,
                        "Median MT Percentile": 1.5,
                    }
                ],
            )
            placeholder = root / "placeholder.tsv"
            placeholder.write_text("peptide\n", encoding="utf-8")
            entries = [
                InputEntry(
                    "P1",
                    "microbial",
                    "generic_peptide_table_v1",
                    placeholder,
                    binding_path=microbial_binding,
                ),
                InputEntry(
                    "P1",
                    "mutation-derived",
                    "generic_peptide_table_v1",
                    placeholder,
                    binding_path=mutation_binding,
                ),
            ]
            output = root / "hla.tsv.gz"
            count, statuses = annotate_shared_hla_support(pair_path, entries, output)
            with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(count, 1)
            self.assertEqual(
                rows[0]["hla_input_contract_status"],
                "classical_hla_i_universe_matched",
            )
            self.assertEqual(rows[0]["shared_supported_classical_hla_i"], "HLA-A*02:01")
            self.assertEqual(
                rows[0]["binding_evidence_status"],
                "binding_supported_sequence_mimicry_candidate",
            )
            self.assertEqual(statuses["binding_supported_sequence_mimicry_candidate"], 1)

    def test_mismatched_hla_universes_are_not_reported_as_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pair_path = root / "pairs.tsv"
            pair_row = {field: "" for field in PAIR_FIELDS}
            pair_row.update(
                {
                    "candidate_id": "candidate-1",
                    "method_version": "sequence_mimicry_v1.2",
                    "patient_id": "P1",
                    "source_pair": "microbial--cryptic",
                    "source_a": "microbial",
                    "source_b": "cryptic",
                    "peptide_a": "AAAAAAAA",
                    "peptide_b": "CAAAAACC",
                    "peptide_length": 8,
                }
            )
            write_tsv(pair_path, [pair_row])
            tables = []
            for source, peptide, allele in (
                ("microbial", "AAAAAAAA", "HLA-A*02:01"),
                ("cryptic", "CAAAAACC", "HLA-A*02:02"),
            ):
                binding = root / f"{source}.tsv"
                write_tsv(
                    binding,
                    [
                        {
                            "Epitope Seq": peptide,
                            "HLA Allele": allele,
                            "Best IC50 Score": 100,
                            "Median IC50 Score": 200,
                            "Best Percentile": 0.5,
                            "Median Percentile": 1.0,
                        }
                    ],
                )
                tables.append(
                    InputEntry(
                        "P1",
                        source,
                        "generic_peptide_table_v1",
                        binding,
                        binding_path=binding,
                    )
                )
            output = root / "hla.tsv.gz"
            _, statuses = annotate_shared_hla_support(pair_path, tables, output)
            with gzip.open(output, "rt", encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(
                row["hla_input_contract_status"],
                "classical_hla_i_universe_mismatch",
            )
            self.assertEqual(
                row["binding_evidence_status"],
                "binding_not_evaluable_hla_contract_mismatch",
            )
            self.assertEqual(
                statuses["binding_not_evaluable_hla_contract_mismatch"],
                1,
            )

    def test_thresholds_are_strict_and_missing_metrics_are_not_negative(self):
        from mimicneoai.mimicry._binding import _binding_columns, _passes_binding_policy
        from mimicneoai.mimicry._policy import SHARED_CLASSICAL_HLA_I_BINDING_V1

        columns = _binding_columns("microbial")
        at_threshold = {
            "Best IC50 Score": "500",
            "Median IC50 Score": "100",
            "Best Percentile": "1",
            "Median Percentile": "1",
        }
        missing = dict(at_threshold)
        missing["Best IC50 Score"] = ""
        self.assertFalse(
            _passes_binding_policy(at_threshold, columns, SHARED_CLASSICAL_HLA_I_BINDING_V1)
        )
        self.assertIsNone(
            _passes_binding_policy(missing, columns, SHARED_CLASSICAL_HLA_I_BINDING_V1)
        )


if __name__ == "__main__":
    unittest.main()
