from __future__ import annotations

import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

from mimicneoai.functions.pipline_tools import raise_for_failed_samples, tools
from mimicneoai.microbial_pipeline import microbial
from mimicneoai.mutation_derived_pipeline import mutation_derived


class FakeAsyncResult:
    def __init__(self, value=None, error: Optional[Exception] = None) -> None:
        self.value = value
        self.error = error
        self.get_called = False

    def get(self):
        self.get_called = True
        if self.error is not None:
            raise self.error
        return self.value


class CommandFailurePropagationTest(unittest.TestCase):
    def make_tool(self, root: Path, sample: str = "TEST") -> tools:
        tool = tools(str(root), "FailureTest", threading.Lock())
        tool.samples = [sample, "summary"]
        tool.failed_cmds_allSamples = {sample: [], "summary": []}
        tool.done_cmds_allSamples = {sample: [], "summary": []}
        tool.run_cmds_allSamples = {sample: [], "summary": []}
        return tool

    def test_exec_cmd_records_and_raises_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            tool = self.make_tool(Path(tempdir))
            with self.assertRaises(subprocess.CalledProcessError) as ctx:
                tool.exec_cmd("sh -c 'exit 7'", "TEST")
            self.assertEqual(ctx.exception.returncode, 7)
            self.assertTrue(tool.has_failures())
            self.assertEqual(tool.run_cmds_allSamples["TEST"], [])
            self.assertEqual(len(tool.failed_cmds_allSamples["TEST"]), 1)

    def test_exec_cmd_detects_failure_before_successful_pipe_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            tool = self.make_tool(Path(tempdir))
            with self.assertRaises(subprocess.CalledProcessError):
                tool.exec_cmd("false | true", "TEST")
            self.assertTrue(tool.has_failures())

    def test_exec_cmd_with_time_records_and_reraises_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            tool = self.make_tool(Path(tempdir))
            with self.assertRaises(subprocess.TimeoutExpired):
                tool.exec_cmd_with_time("sleep 1", "TEST", timeout=0.01)
            self.assertTrue(tool.has_failures())
            self.assertEqual(tool.run_cmds_allSamples["TEST"], [])
            self.assertEqual(len(tool.failed_cmds_allSamples["TEST"]), 1)

    def test_exec_cmd_logs_user_facing_detail_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            tool = self.make_tool(Path(tempdir))
            tool.exec_cmd("echo first", "TEST", display_name="Friendly Step")
            tool.exec_cmd("echo second", "TEST", display_name="Friendly Step")

            start_log = Path(tool.start_log).read_text()
            self.assertIn("Run TEST Friendly Step!", start_log)
            self.assertIn("Detail log:", start_log)
            detail_logs = sorted(Path(tempdir).glob("log/*/detail/TEST/*FailureTest_Friendly_Step.log"))
            self.assertEqual(len(detail_logs), 2)
            self.assertIn("first", detail_logs[0].read_text())
            self.assertIn("second", detail_logs[1].read_text())

    def test_failed_exec_cmd_logs_detail_log_path(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            tool = self.make_tool(Path(tempdir))
            with self.assertRaises(subprocess.CalledProcessError):
                tool.exec_cmd("sh -c 'echo failing; exit 4'", "TEST", display_name="Failing Step")

            start_log = Path(tool.start_log).read_text()
            self.assertIn("TEST Failing Step failed!", start_log)
            self.assertIn("Detail log:", start_log)
            detail_logs = sorted(Path(tempdir).glob("log/*/detail/TEST/*FailureTest_Failing_Step.log"))
            self.assertEqual(len(detail_logs), 1)
            self.assertIn("failing", detail_logs[0].read_text())

    def test_async_results_are_all_collected_before_aggregate_failure(self) -> None:
        successful = FakeAsyncResult(value="ok")
        failed = FakeAsyncResult(error=ValueError("sample failed"))
        later_success = FakeAsyncResult(value="still ran")
        with self.assertRaisesRegex(RuntimeError, "S2: ValueError: sample failed"):
            raise_for_failed_samples(
                [("S1", successful), ("S2", failed), ("S3", later_success)]
            )
        self.assertTrue(successful.get_called)
        self.assertTrue(failed.get_called)
        self.assertTrue(later_success.get_called)

    def test_mutation_worker_reraises_binding_command_failure(self) -> None:
        config = {
            "path": {"output_dir": "/tmp/output"},
            "args": {"hla_binding_threads": 2},
            "step_name": {"annotation": "05.annotation", "hla": "06.hlatyping"},
            "others": {
                "QC": False,
                "host_variants_calling": False,
                "annotation": False,
                "species": "human",
                "seq_type": "wes",
                "hlatyping": False,
                "peptides_identification_and_binding_prediction": True,
                "tumor_with_matched_normal": True,
                "binding_prediction_backend": "mimicneoai",
            },
        }
        paths = {"path": {"common": {"PVACTOOLS": "/tools/pvactools.sif"}}}
        tool = MagicMock()
        tool.exec_cmd.side_effect = subprocess.CalledProcessError(3, "binding")
        with self.assertRaises(subprocess.CalledProcessError):
            mutation_derived._start_one_sample("TUMOR,NORMAL", config, paths, tool)

    def test_microbial_worker_stops_after_failed_step(self) -> None:
        config = {
            "others": {
                "QC": True,
                "run_host_depletion": True,
                "run_vector_decontamination": False,
                "run_pathseq": False,
                "run_microbial_peptide_identification": False,
                "run_hla_typing": False,
                "run_binding_prediction": False,
            }
        }
        tool = MagicMock()
        with (
            patch.object(microbial, "fastp", side_effect=RuntimeError("QC failed")),
            patch.object(microbial, "HostSequencesRemoving") as host_removal,
        ):
            with self.assertRaisesRegex(RuntimeError, "QC failed"):
                microbial._start_one_sample("SAMPLE", config, {}, tool)
        host_removal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
