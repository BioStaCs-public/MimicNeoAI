from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
HLA_TYPING_SCRIPT = PACKAGE_ROOT / "cryptic_pipeline" / "scripts" / "05-hla_typing.py"


def load_hla_typing_module():
    spec = importlib.util.spec_from_file_location("cryptic_hla_typing", HLA_TYPING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HlaHDResolutionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.module = load_hla_typing_module()

    def _make_hlahd(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)
        return path

    def test_resolves_configured_hlahd_script(self) -> None:
        configured = self._make_hlahd(self.root / "tools" / "hlahd.sh")
        freq_data = self.root / "unused" / "freq_data"
        freq_data.mkdir(parents=True)

        resolved = self.module.resolve_hlahd_script(str(configured), str(freq_data))

        self.assertEqual(Path(resolved), configured)

    def test_infers_hlahd_script_from_freq_data_parent_bin(self) -> None:
        freq_data = self.root / "hlahd.1.7.0" / "freq_data"
        freq_data.mkdir(parents=True)
        inferred = self._make_hlahd(self.root / "hlahd.1.7.0" / "bin" / "hlahd.sh")

        resolved = self.module.resolve_hlahd_script("", str(freq_data))

        self.assertEqual(Path(resolved), inferred)

    def test_missing_hlahd_script_fails_closed(self) -> None:
        freq_data = self.root / "hlahd.1.7.0" / "freq_data"
        freq_data.mkdir(parents=True)

        with self.assertRaises(FileNotFoundError) as ctx:
            self.module.resolve_hlahd_script("", str(freq_data))

        self.assertIn("HLA-HD executable hlahd.sh was not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
