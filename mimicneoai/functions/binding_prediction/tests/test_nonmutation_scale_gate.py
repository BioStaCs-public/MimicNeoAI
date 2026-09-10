from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mimicneoai.functions.binding_prediction.nonmutation_workflow import main


class NonmutationScaleGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.peptide_fasta = self.root / "peptides.fasta"
        self.peptide_fasta.write_text(">source_1\nACDEFGHIKLMNPQRSTVWYACDEF\n")
        self.hla_file = self.root / "sample_final.result.txt"
        self.hla_file.write_text("A\tA*02:01\nDRB1\tDRB1*11:01\n")

    def run_tasks_only(
        self,
        outdir: Path,
        max_task_rows: int,
        force: bool = False,
        algorithms: str = "MHCflurry,MHCnuggetsII",
    ) -> dict[str, object]:
        args = [
            "-s",
            "TEST",
            "--pep-fasta",
            str(self.peptide_fasta),
            "--hla-file",
            str(self.hla_file),
            "-o",
            str(outdir),
            "--algorithms",
            algorithms,
            "--max-task-rows",
            str(max_task_rows),
            "--skip-prediction",
        ]
        if force:
            args.append("--force-large-samples")
        self.assertEqual(main(args), 0)
        return json.loads((outdir / "TEST.mimicneoai_binding.summary.json").read_text())

    def test_scale_gate_skips_task_materialization(self) -> None:
        outdir = self.root / "scale_skip"
        summary = self.run_tasks_only(outdir, max_task_rows=1)

        task_dir = outdir / "mimicneoai_epitope_tasks"
        self.assertFalse((task_dir / "epitope_windows.tsv").exists())
        self.assertTrue((task_dir / "epitope_windows.manifest.json").exists())
        self.assertTrue((task_dir / "binding_tasks.manifest.json").exists())
        self.assertFalse((task_dir / "binding_tasks.tsv").exists())
        self.assertTrue(summary["preflight_scale_gate"])
        self.assertFalse(summary["epitope_windows_materialized"])
        self.assertTrue(summary["task_materialization_skipped_by_scale"])
        self.assertFalse(summary["task_table_materialized"])
        self.assertEqual(summary["binding_task_rows"], 0)
        self.assertGreater(summary["estimated_binding_task_rows"], 1)

    def test_task_table_is_materialized_when_allowed(self) -> None:
        for max_task_rows, force in ((1_000_000, False), (1, True)):
            with self.subTest(max_task_rows=max_task_rows, force=force):
                outdir = self.root / f"materialized_{max_task_rows}_{force}"
                summary = self.run_tasks_only(outdir, max_task_rows=max_task_rows, force=force)

                task_path = outdir / "mimicneoai_epitope_tasks" / "binding_tasks.tsv"
                self.assertTrue(task_path.exists())
                self.assertFalse(summary["task_materialization_skipped_by_scale"])
                self.assertTrue(summary["task_table_materialized"])
                self.assertEqual(summary["binding_task_rows"], summary["estimated_binding_task_rows"])

    def test_existing_intermediates_are_reused_on_resume(self) -> None:
        outdir = self.root / "resume"
        first = self.run_tasks_only(outdir, max_task_rows=1_000_000)
        second = self.run_tasks_only(outdir, max_task_rows=1_000_000)

        self.assertFalse(first["epitope_windows_reused"])
        self.assertFalse(first["binding_tasks_reused"])
        self.assertTrue(second["epitope_windows_reused"])
        self.assertTrue(second["binding_tasks_reused"])
        self.assertEqual(first["epitope_window_rows"], second["epitope_window_rows"])
        self.assertEqual(first["binding_task_rows"], second["binding_task_rows"])

    def test_changed_inputs_invalidate_only_affected_intermediates(self) -> None:
        outdir = self.root / "signature_change"
        first = self.run_tasks_only(outdir, max_task_rows=1_000_000)

        self.peptide_fasta.write_text(">source_1\nACDEFGHIKLMNPQRSTVWYACDEFGHIK\n")
        second = self.run_tasks_only(outdir, max_task_rows=1_000_000)
        self.assertFalse(second["epitope_windows_reused"])
        self.assertFalse(second["binding_tasks_reused"])
        self.assertGreater(second["epitope_window_rows"], first["epitope_window_rows"])

        third = self.run_tasks_only(
            outdir,
            max_task_rows=1_000_000,
            algorithms="MHCflurry,MHCflurryEL,MHCnuggetsII",
        )
        self.assertTrue(third["epitope_windows_reused"])
        self.assertFalse(third["binding_tasks_reused"])
        self.assertGreater(third["binding_task_rows"], second["binding_task_rows"])


if __name__ == "__main__":
    unittest.main()
