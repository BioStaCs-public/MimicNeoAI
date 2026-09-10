from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mimicneoai.mimicry.cli import load_run_config


class ConfigurationTests(unittest.TestCase):
    def test_unknown_top_level_field_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "mimicry.yaml"
            config.write_text(
                "input_manifest: inputs.tsv\n"
                "output_dir: output\n"
                "unexpected_threshold: 0.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown mimicry configuration"):
                load_run_config(config)

    def test_binding_enabled_requires_a_boolean(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs.tsv"
            inputs.write_text("patient_id\n", encoding="utf-8")
            config = root / "mimicry.yaml"
            config.write_text(
                "input_manifest: inputs.tsv\n"
                "output_dir: output\n"
                "binding_support:\n"
                "  enabled: 'false'\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                load_run_config(config)


if __name__ == "__main__":
    unittest.main()
