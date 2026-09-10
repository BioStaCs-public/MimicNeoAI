from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from mimicneoai.functions.binding_prediction.qc import build_binding_qc_summary
from mimicneoai.functions.binding_prediction.schema import PREDICTION_FIELDS


class BindingQcSummaryTest(unittest.TestCase):
    def test_reports_status_support_and_primary_metric_completeness(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            task_path = root / "binding_tasks.tsv"
            prediction_path = root / "binding_predictions.long.tsv"
            self.write_tasks(task_path)
            self.write_predictions(prediction_path)

            qc = build_binding_qc_summary(task_path, prediction_path)

            self.assertTrue(qc["informational_only"])
            self.assertEqual(qc["task_rows"], 6)
            self.assertEqual(qc["prediction_rows"], 6)
            self.assertEqual(qc["result_coverage_rate"], 1.0)

            affinity = qc["algorithms"]["MHCflurry"]
            self.assertEqual(
                affinity["status_counts"],
                {"error": 1, "ok": 1, "partial_ok": 1, "skipped": 1},
            )
            self.assertEqual(affinity["execution_success_rate"], 0.666667)
            self.assertEqual(affinity["hla_support_rate"], 0.5)
            self.assertEqual(affinity["unsupported_hla_alleles"], ["HLA-F*01:01"])
            self.assertEqual(affinity["primary_metric"], "ic50")
            self.assertEqual(affinity["primary_metric_completeness_rate"], 1.0)

            el = qc["algorithms"]["MHCflurryEL"]
            self.assertEqual(el["primary_metric"], "percentile")
            self.assertEqual(el["primary_metric_completeness_rate"], 0.5)

    @staticmethod
    def write_tasks(path: Path) -> None:
        rows = [
            ("PEPTIDEA", "HLA-A*02:01", "MHCflurry", "MHC-I"),
            ("PEPTIDEB", "HLA-A*02:01", "MHCflurry", "MHC-I"),
            ("PEPTIDEC", "HLA-A*02:01", "MHCflurry", "MHC-I"),
            ("PEPTIDED", "HLA-F*01:01", "MHCflurry", "MHC-I"),
            ("PEPTIDEE", "HLA-A*02:01", "MHCflurryEL", "MHC-I"),
            ("PEPTIDEF", "HLA-A*02:01", "MHCflurryEL", "MHC-I"),
        ]
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["peptide", "hla_allele", "algorithm", "mhc_class"])
            writer.writerows(rows)

    @staticmethod
    def write_predictions(path: Path) -> None:
        rows = [
            {"peptide": "PEPTIDEA", "hla_allele": "HLA-A*02:01", "algorithm": "MHCflurry", "mhc_class": "MHC-I", "peptide_length": "8", "ic50": "25", "status": "ok"},
            {"peptide": "PEPTIDEB", "hla_allele": "HLA-A*02:01", "algorithm": "MHCflurry", "mhc_class": "MHC-I", "peptide_length": "8", "ic50": "50", "status": "partial_ok"},
            {"peptide": "PEPTIDEC", "hla_allele": "HLA-A*02:01", "algorithm": "MHCflurry", "mhc_class": "MHC-I", "peptide_length": "8", "status": "error"},
            {"peptide": "PEPTIDED", "hla_allele": "HLA-F*01:01", "algorithm": "MHCflurry", "mhc_class": "MHC-I", "peptide_length": "8", "status": "skipped"},
            {"peptide": "PEPTIDEE", "hla_allele": "HLA-A*02:01", "algorithm": "MHCflurryEL", "mhc_class": "MHC-I", "peptide_length": "8", "percentile": "0.2", "status": "ok"},
            {"peptide": "PEPTIDEF", "hla_allele": "HLA-A*02:01", "algorithm": "MHCflurryEL", "mhc_class": "MHC-I", "peptide_length": "8", "status": "partial_ok"},
        ]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
