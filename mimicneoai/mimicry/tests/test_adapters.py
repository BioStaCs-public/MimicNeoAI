from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from mimicneoai.mimicry import SEQUENCE_MIMICRY_V1_2
from mimicneoai.mimicry._adapters import load_occurrences, read_input_manifest


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


class AdapterTests(unittest.TestCase):
    def test_missing_versioned_input_column_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            peptide = root / "peptides.tsv"
            write_tsv(
                peptide,
                [{"peptide": "AAAAAAAA", "peptide_id": "p1", "mhc_class": "HLA-I"}],
            )
            manifest = root / "inputs.tsv"
            write_tsv(
                manifest,
                [
                    {
                        "patient_id": "P1",
                        "antigen_source": "microbial",
                        "input_format": "generic_peptide_table_v1",
                        "peptide_path": peptide.name,
                    }
                ],
            )
            entry = read_input_manifest(manifest)[0]
            with self.assertRaisesRegex(ValueError, "qc_status"):
                load_occurrences(entry, SEQUENCE_MIMICRY_V1_2)

    def test_native_micro_core_filters_and_expands_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = root / "core.tsv"
            provenance = root / "parent_map.tsv"
            manifest = root / "run_manifest.json"
            input_manifest = root / "inputs.tsv"
            write_tsv(
                core,
                [
                    {
                        "peptide_id": "p1",
                        "mhc_class": "HLA-I",
                        "peptide": "AAAAAAAA",
                        "peptide_qc_status": "core",
                    },
                    {
                        "peptide_id": "p2",
                        "mhc_class": "HLA-II",
                        "peptide": "AAAAAAAAAAAAA",
                        "peptide_qc_status": "core",
                    },
                ],
            )
            write_tsv(
                provenance,
                [
                    {"peptide": "AAAAAAAA", "parent_record_id": "parent-1"},
                    {"peptide": "AAAAAAAA", "parent_record_id": "parent-2"},
                ],
            )
            manifest.write_text(json.dumps({"run_status": "completed"}), encoding="utf-8")
            write_tsv(
                input_manifest,
                [
                    {
                        "patient_id": "P1",
                        "antigen_source": "microbial",
                        "input_format": "microbial_core_v1",
                        "peptide_path": core.name,
                        "provenance_path": provenance.name,
                        "run_manifest_path": manifest.name,
                    }
                ],
            )
            entry = read_input_manifest(input_manifest)[0]
            result = load_occurrences(entry, SEQUENCE_MIMICRY_V1_2)
            self.assertEqual(len(result.occurrences), 2)
            self.assertEqual(
                {row.source_record_id for row in result.occurrences},
                {"parent-1", "parent-2"},
            )
            self.assertEqual(result.counts["eligible_unique_peptides"], 1)

    def test_native_cryptic_and_mutation_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cryptic = root / "cryptic.tsv"
            cryptic_sidecar = root / "cryptic_sidecar.tsv"
            mutation = root / "mutation.tsv"
            write_tsv(
                cryptic,
                [
                    {
                        "peptide_record_id": "cp1",
                        "mhc_class": "HLA-I",
                        "peptide": "CAAAAACC",
                        "peptide_core_status": "core",
                    }
                ],
            )
            write_tsv(
                cryptic_sidecar,
                [
                    {
                        "peptide_record_id": "cp1",
                        "mhc_class": "HLA-I",
                        "peptide": "CAAAAACC",
                        "parent_record_id": "cryptic-parent-1",
                    }
                ],
            )
            write_tsv(
                mutation,
                [
                    {
                        "event_id": "event-1",
                        "mhc_class": "MHC-I",
                        "peptide_length": 8,
                        "mt_epitope_seq": "GAAAAATT",
                        "wt_epitope_seq": "VVVAAAVV",
                        "covers_mutation": "YES",
                        "wt_control_status": "matched_WT",
                    }
                ],
            )
            input_manifest = root / "inputs.tsv"
            write_tsv(
                input_manifest,
                [
                    {
                        "patient_id": "P1",
                        "antigen_source": "cryptic",
                        "input_format": "cryptic_final_core_v1_1",
                        "peptide_path": cryptic.name,
                        "provenance_path": cryptic_sidecar.name,
                    },
                    {
                        "patient_id": "P1",
                        "antigen_source": "mutation-derived",
                        "input_format": "mutation_epitope_windows_v1",
                        "peptide_path": mutation.name,
                        "provenance_path": "",
                    },
                ],
            )
            entries = read_input_manifest(input_manifest)
            cryptic_result = load_occurrences(entries[0], SEQUENCE_MIMICRY_V1_2)
            mutation_result = load_occurrences(entries[1], SEQUENCE_MIMICRY_V1_2)
            self.assertEqual(
                cryptic_result.occurrences[0].source_record_id, "cryptic-parent-1"
            )
            self.assertEqual(mutation_result.occurrences[0].event_id, "event-1")
            self.assertEqual(mutation_result.occurrences[0].wt_peptide, "VVVAAAVV")

    def test_explicit_incomplete_upstream_manifest_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            peptide = root / "peptides.tsv"
            write_tsv(
                peptide,
                [
                    {
                        "peptide": "AAAAAAAA",
                        "peptide_id": "p1",
                        "mhc_class": "HLA-I",
                        "qc_status": "core",
                    }
                ],
            )
            upstream = root / "run_manifest.json"
            upstream.write_text(json.dumps({"run_status": "failed"}), encoding="utf-8")
            input_manifest = root / "inputs.tsv"
            write_tsv(
                input_manifest,
                [
                    {
                        "patient_id": "P1",
                        "antigen_source": "microbial",
                        "input_format": "generic_peptide_table_v1",
                        "peptide_path": peptide.name,
                        "run_manifest_path": upstream.name,
                    }
                ],
            )
            entry = read_input_manifest(input_manifest)[0]
            with self.assertRaisesRegex(ValueError, "not complete"):
                load_occurrences(entry, SEQUENCE_MIMICRY_V1_2)

    def test_duplicate_patient_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            peptide = root / "peptides.tsv"
            write_tsv(
                peptide,
                [
                    {
                        "peptide": "AAAAAAAA",
                        "peptide_id": "p1",
                        "mhc_class": "HLA-I",
                        "qc_status": "core",
                    }
                ],
            )
            manifest = root / "inputs.tsv"
            rows = [
                {
                    "patient_id": "P1",
                    "antigen_source": "microbial",
                    "input_format": "generic_peptide_table_v1",
                    "peptide_path": peptide.name,
                },
                {
                    "patient_id": "P1",
                    "antigen_source": "microbial",
                    "input_format": "generic_peptide_table_v1",
                    "peptide_path": peptide.name,
                },
            ]
            write_tsv(manifest, rows)
            with self.assertRaisesRegex(ValueError, "Duplicate patient/source"):
                read_input_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
