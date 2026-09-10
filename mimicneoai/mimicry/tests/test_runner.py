from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from mimicneoai.mimicry._engine import run_sequence_mimicry
from mimicneoai.mimicry import MimicryRunConfig


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_gzip_tsv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class RunnerTests(unittest.TestCase):
    def build_fixture(self, root: Path) -> Path:
        tables = {
            "microbial": [
                {
                    "peptide": "AAAAAAAA",
                    "peptide_id": "micro-1",
                    "mhc_class": "HLA-I",
                    "qc_status": "core",
                    "event_id": "",
                    "wt_peptide": "",
                    "wt_control_status": "",
                }
            ],
            "cryptic": [
                {
                    "peptide": "CAAAAACC",
                    "peptide_id": "cryptic-1",
                    "mhc_class": "HLA-I",
                    "qc_status": "core",
                    "event_id": "",
                    "wt_peptide": "",
                    "wt_control_status": "",
                }
            ],
            "mutation-derived": [
                {
                    "peptide": "GAAAAATT",
                    "peptide_id": "mutation-1",
                    "mhc_class": "HLA-I",
                    "qc_status": "core",
                    "event_id": "event-1",
                    "wt_peptide": "VVVAAAVV",
                    "wt_control_status": "matched_WT",
                }
            ],
        }
        manifest_rows = []
        for source, rows in tables.items():
            path = root / f"{source}.tsv"
            write_tsv(path, rows)
            manifest_rows.append(
                {
                    "patient_id": "P1",
                    "antigen_source": source,
                    "input_format": "generic_peptide_table_v1",
                    "peptide_path": path.name,
                }
            )
        manifest = root / "input_manifest.tsv"
        write_tsv(manifest, manifest_rows)
        return manifest

    def test_all_source_pairs_and_mutation_wt_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_manifest = self.build_fixture(root)
            output = root / "output"
            config = MimicryRunConfig(
                method="sequence_mimicry_v1.2",
                input_manifest=input_manifest,
                output_dir=output,
                source_pairs=(
                    "microbial--mutation-derived",
                    "microbial--cryptic",
                    "cryptic--mutation-derived",
                ),
                workers=1,
                shard_count=2,
            )
            manifest = run_sequence_mimicry(config)
            pairs = read_gzip_tsv(output / "mimicry_pairs.tsv.gz")
            events = read_gzip_tsv(output / "mimicry_mutation_wt_evidence.tsv.gz")
            provenance = read_gzip_tsv(output / "mimicry_member_provenance.tsv.gz")
            self.assertEqual(manifest["row_counts"]["passing_unique_pairs"], 3)
            self.assertEqual({row["source_pair"] for row in pairs}, set(config.source_pairs))
            self.assertEqual(len(events), 2)
            self.assertEqual(
                {row["wt_classification"] for row in events},
                {"MT_specific_sequence_mimic"},
            )
            self.assertEqual(len(provenance), 3)
            self.assertEqual(
                {row["antigen_source"] for row in provenance},
                {"microbial", "cryptic", "mutation-derived"},
            )
            self.assertTrue(
                all(
                    row["primary_call"] == "sequence_based_mimicry_v12_candidate"
                    for row in pairs
                )
            )

            reused = run_sequence_mimicry(config)
            self.assertTrue(reused["reused"])

    def test_input_order_and_worker_count_do_not_change_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_manifest = self.build_fixture(root)
            with input_manifest.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            reversed_manifest = root / "input_manifest_reversed.tsv"
            write_tsv(reversed_manifest, list(reversed(rows)))
            source_pairs = (
                "microbial--mutation-derived",
                "microbial--cryptic",
                "cryptic--mutation-derived",
            )
            run_sequence_mimicry(
                MimicryRunConfig(
                    "sequence_mimicry_v1.2",
                    input_manifest,
                    root / "one",
                    source_pairs,
                    1,
                    2,
                )
            )
            run_sequence_mimicry(
                MimicryRunConfig(
                    "sequence_mimicry_v1.2",
                    reversed_manifest,
                    root / "two",
                    source_pairs,
                    2,
                    2,
                )
            )
            pairs_one = read_gzip_tsv(root / "one" / "mimicry_pairs.tsv.gz")
            pairs_two = read_gzip_tsv(root / "two" / "mimicry_pairs.tsv.gz")
            self.assertEqual(pairs_one, pairs_two)

    def test_changed_input_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_manifest = self.build_fixture(root)
            config = MimicryRunConfig(
                "sequence_mimicry_v1.2",
                input_manifest,
                root / "output",
                ("microbial--cryptic",),
                1,
                1,
            )
            run_sequence_mimicry(config)
            with (root / "microbial.tsv").open("a", encoding="utf-8") as handle:
                handle.write("AAAAAAAA\tmicro-2\tHLA-I\tcore\t\t\t\n")
            with self.assertRaisesRegex(ValueError, "different inputs or code"):
                run_sequence_mimicry(config)

    def test_python_api_validates_run_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_manifest = self.build_fixture(root)
            with self.assertRaisesRegex(ValueError, "workers must be at least 1"):
                run_sequence_mimicry(
                    MimicryRunConfig(
                        "sequence_mimicry_v1.2",
                        input_manifest,
                        root / "output",
                        ("microbial--cryptic",),
                        0,
                        1,
                    )
                )

    def test_binding_support_is_integrated_without_changing_sequence_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_manifest = self.build_fixture(root)
            with input_manifest.open(encoding="utf-8", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle, delimiter="\t"))
            for row in manifest_rows:
                source = row["antigen_source"]
                binding = root / f"{source}.binding.tsv"
                if source == "mutation-derived":
                    write_tsv(
                        binding,
                        [
                            {
                                "MT Epitope Seq": "GAAAAATT",
                                "HLA Allele": "HLA-A*02:01",
                                "Best MT IC50 Score": 100,
                                "Median MT IC50 Score": 200,
                                "Best MT Percentile": 0.5,
                                "Median MT Percentile": 1.0,
                            }
                        ],
                    )
                else:
                    peptide = "AAAAAAAA" if source == "microbial" else "CAAAAACC"
                    write_tsv(
                        binding,
                        [
                            {
                                "Epitope Seq": peptide,
                                "HLA Allele": "HLA-A*02:01",
                                "Best IC50 Score": 100,
                                "Median IC50 Score": 200,
                                "Best Percentile": 0.5,
                                "Median Percentile": 1.0,
                            }
                        ],
                    )
                row["binding_path"] = binding.name
            write_tsv(input_manifest, manifest_rows)

            output = root / "output"
            manifest = run_sequence_mimicry(
                MimicryRunConfig(
                    "sequence_mimicry_v1.2",
                    input_manifest,
                    output,
                    (
                        "microbial--mutation-derived",
                        "microbial--cryptic",
                        "cryptic--mutation-derived",
                    ),
                    1,
                    1,
                    binding_support_enabled=True,
                )
            )
            support = read_gzip_tsv(output / "mimicry_hla_support.tsv.gz")
            self.assertEqual(manifest["row_counts"]["passing_unique_pairs"], 3)
            self.assertEqual(len(support), 3)
            self.assertEqual(
                {row["binding_evidence_status"] for row in support},
                {"binding_supported_sequence_mimicry_candidate"},
            )


if __name__ == "__main__":
    unittest.main()
