from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from mimicneoai.functions.binding_prediction.policy import resolve_binding_prediction_policy
from mimicneoai.functions.binding_prediction.schema import PREDICTION_FIELDS
from mimicneoai.functions.binding_prediction.stage1 import (
    TASK_FIELDS,
    classify_stage1_prediction,
    route_stage2_tasks,
)


class BindingPredictionStage1PolicyTest(unittest.TestCase):
    def test_presets_are_simple_and_distinct(self) -> None:
        full = resolve_binding_prediction_policy("full")
        fast = resolve_binding_prediction_policy("fast")

        self.assertIsNotNone(full)
        self.assertIsNotNone(fast)
        assert full is not None
        assert fast is not None
        self.assertFalse(full.two_stage)
        self.assertTrue(fast.two_stage)
        self.assertIn("NNalign", full.mhc_ii_algorithms)
        self.assertNotIn("NNalign", fast.mhc_ii_algorithms)
        self.assertEqual(fast.mhc_i_lengths, (8, 9, 10, 11))
        self.assertEqual(fast.mhc_ii_lengths, (13, 14, 15, 16, 17))

    def test_stage1_threshold_and_failure_states_are_explicit(self) -> None:
        policy = resolve_binding_prediction_policy("fast")
        assert policy is not None
        task = {
            "peptide": "SYFPEITHI",
            "hla_allele": "HLA-A*02:01",
            "algorithm": "NetMHCpanEL",
            "mhc_class": "MHC-I",
        }

        self.assertEqual(
            classify_stage1_prediction(
                task,
                {"status": "ok", "percentile": "9.999", "error": ""},
                policy,
            )[0],
            "stage1_screen_pass",
        )
        self.assertEqual(
            classify_stage1_prediction(
                task,
                {"status": "ok", "percentile": "10.0", "error": ""},
                policy,
            )[0],
            "screened_out_stage1",
        )
        self.assertEqual(
            classify_stage1_prediction(
                task,
                {"status": "skipped", "percentile": "", "error": "unsupported allele"},
                policy,
            )[0],
            "unsupported_allele",
        )
        self.assertEqual(
            classify_stage1_prediction(task, None, policy)[0],
            "prediction_missing",
        )

    def test_route_stage2_tasks_keeps_only_stage1_passes(self) -> None:
        policy = resolve_binding_prediction_policy("fast")
        assert policy is not None
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            stage1_tasks = root / "stage1_tasks.tsv"
            stage1_predictions = root / "stage1_predictions.tsv"
            stage2_tasks = root / "stage2_tasks.tsv"
            routing = root / "routing.tsv"
            unsupported = root / "unsupported.tsv"
            pass_predictions = root / "stage1_pass_predictions.tsv"

            self.write_tasks(
                stage1_tasks,
                [
                    ("PASSI", "HLA-A*02:01", "NetMHCpanEL", "MHC-I"),
                    ("FAILI", "HLA-A*02:01", "NetMHCpanEL", "MHC-I"),
                    ("PASSIIPEPTIDE13", "HLA-DRB1*11:01", "NetMHCIIpanEL", "MHC-II"),
                    ("UNSUPPORTED", "HLA-F*01:01", "NetMHCpanEL", "MHC-I"),
                    ("NOPREDICT", "HLA-A*02:01", "NetMHCpanEL", "MHC-I"),
                ],
            )
            self.write_predictions(
                stage1_predictions,
                [
                    ("PASSI", "HLA-A*02:01", "NetMHCpanEL", "MHC-I", "9.9", "ok", ""),
                    ("FAILI", "HLA-A*02:01", "NetMHCpanEL", "MHC-I", "10.0", "ok", ""),
                    (
                        "PASSIIPEPTIDE13",
                        "HLA-DRB1*11:01",
                        "NetMHCIIpanEL",
                        "MHC-II",
                        "0.5",
                        "ok",
                        "",
                    ),
                    (
                        "UNSUPPORTED",
                        "HLA-F*01:01",
                        "NetMHCpanEL",
                        "MHC-I",
                        "",
                        "skipped",
                        "unsupported_allele_by_predictor",
                    ),
                ],
            )

            summary = route_stage2_tasks(
                stage1_tasks_path=stage1_tasks,
                stage1_predictions_path=stage1_predictions,
                policy=policy,
                stage2_algorithms=policy.algorithms,
                stage2_task_path=stage2_tasks,
                routing_path=routing,
                unsupported_path=unsupported,
                pass_predictions_path=pass_predictions,
            )

            rows = self.read_tsv(stage2_tasks)
            algorithms = {row["algorithm"] for row in rows}
            self.assertEqual(summary["stage1_status_counts"]["stage1_screen_pass"], 2)
            self.assertEqual(summary["stage1_status_counts"]["screened_out_stage1"], 1)
            self.assertEqual(summary["stage1_status_counts"]["unsupported_allele"], 1)
            self.assertEqual(summary["stage1_status_counts"]["prediction_missing"], 1)
            self.assertEqual(len(rows), 8)
            self.assertNotIn("NNalign", algorithms)
            self.assertEqual(
                {row["peptide"] for row in rows},
                {"PASSI", "PASSIIPEPTIDE13"},
            )
            self.assertEqual(len(self.read_tsv(pass_predictions)), 2)
            self.assertEqual(len(self.read_tsv(unsupported)), 2)

    @staticmethod
    def write_tasks(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(TASK_FIELDS)
            writer.writerows(rows)

    @staticmethod
    def write_predictions(path: Path, rows: list[tuple[str, str, str, str, str, str, str]]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            for peptide, hla, algorithm, mhc_class, percentile, status, error in rows:
                writer.writerow(
                    {
                        "peptide": peptide,
                        "hla_allele": hla,
                        "algorithm": algorithm,
                        "mhc_class": mhc_class,
                        "peptide_length": str(len(peptide)),
                        "percentile": percentile,
                        "status": status,
                        "error": error,
                    }
                )

    @staticmethod
    def read_tsv(path: Path) -> list[dict[str, str]]:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))


if __name__ == "__main__":
    unittest.main()
