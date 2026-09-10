"""Shared HLA binding prediction utilities."""

from __future__ import annotations

from typing import Any


PREDICTOR_PATH_ARGUMENTS = {
    "MHCFLURRY_PREDICT_BIN": "--mhcflurry-predict-bin",
    "MHCFLURRY_DOWNLOADS_DIR": "--mhcflurry-downloads-dir",
    "MHCNUGGETS_PYTHON_BIN": "--mhcnuggets-python-bin",
    "MHCNUGGETS_SCRIPT": "--mhcnuggets-script",
    "MHCNUGGETS_CWD": "--mhcnuggets-cwd",
    "NETMHCPAN_BIN": "--netmhcpan-bin",
    "NETMHCIIPAN_BIN": "--netmhciipan-bin",
    "IEDB_MHCI_PYTHON_BIN": "--iedb-mhci-python-bin",
    "IEDB_MHCI_SCRIPT": "--iedb-mhci-script",
    "IEDB_MHCI_CWD": "--iedb-mhci-cwd",
    "IEDB_MHCII_PYTHON_BIN": "--iedb-mhcii-python-bin",
    "IEDB_MHCII_SCRIPT": "--iedb-mhcii-script",
    "IEDB_MHCII_CWD": "--iedb-mhcii-cwd",
}


def configured_predictor_cli_args(paths: dict[str, Any]) -> list[str]:
    """Translate deployment-level predictor paths into command arguments."""

    predictor_paths = (
        paths.get("path", {}).get("common", {}).get("BINDING_PREDICTORS", {})
    )
    result: list[str] = []
    for path_key, argument in PREDICTOR_PATH_ARGUMENTS.items():
        value = predictor_paths.get(path_key)
        if value:
            result.extend([argument, str(value)])
    return result
