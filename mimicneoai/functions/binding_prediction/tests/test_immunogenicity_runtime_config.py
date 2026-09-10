from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from mimicneoai.functions.immunogenicity_runner import (
    resolve_immunogenicity_model_root,
    resolve_immunogenicity_python_bin,
)


class ImmunogenicityRuntimeConfigTest(unittest.TestCase):
    def test_configured_python_bin_takes_precedence(self) -> None:
        self.assertEqual(
            resolve_immunogenicity_python_bin(
                {"others": {"immunogenicity_python_bin": "/opt/immuno-gpu/bin/python"}}
            ),
            "/opt/immuno-gpu/bin/python",
        )

    def test_environment_python_bin_is_fallback(self) -> None:
        with patch.dict(
            os.environ,
            {"MIMICNEOAI_IMMUNOGENICITY_PYTHON_BIN": "/opt/immuno-cpu/bin/python"},
            clear=True,
        ):
            self.assertEqual(
                resolve_immunogenicity_python_bin(
                    {"others": {}},
                    {"path": {"common": {"IMMUNOGENICITY": {"PYTHON_BIN": "/paths/bin/python"}}}},
                ),
                "/opt/immuno-cpu/bin/python",
            )

    def test_paths_yaml_python_bin_is_fallback_after_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_immunogenicity_python_bin(
                    {"others": {}},
                    {"path": {"common": {"IMMUNOGENICITY": {"PYTHON_BIN": "/paths/bin/python"}}}},
                ),
                "/paths/bin/python",
            )

    def test_current_python_is_final_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_immunogenicity_python_bin({"others": {}}), sys.executable)

    def test_model_root_resolution_order(self) -> None:
        paths = {"path": {"common": {"IMMUNOGENICITY": {"MODEL_ROOT": "/paths/models"}}}}
        self.assertEqual(
            resolve_immunogenicity_model_root(
                {"others": {"immunogenicity_model_root": "/config/models"}},
                paths,
            ),
            "/config/models",
        )
        with patch.dict(
            os.environ,
            {"MIMICNEOAI_IMMUNOGENICITY_MODEL_ROOT": "/env/models"},
            clear=True,
        ):
            self.assertEqual(resolve_immunogenicity_model_root({"others": {}}, paths), "/env/models")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_immunogenicity_model_root({"others": {}}, paths), "/paths/models")


if __name__ == "__main__":
    unittest.main()
