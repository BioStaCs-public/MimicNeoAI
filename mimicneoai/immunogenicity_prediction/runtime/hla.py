"""HLA identifier normalization shared by runtime components."""

from __future__ import annotations

import re


def normalize_hla(hla: str) -> str:
    """Normalize an HLA identifier to the allele form used by local resources."""
    val = str(hla).strip()
    val = re.sub(r"^HLA-", "", val, flags=re.IGNORECASE)
    if "-" in val:
        val = re.sub(r"(?<=[0-9])-(?=[A-Z])", "/", val)
    if "/" in val:
        val = val.split("/")[-1]
    parts = val.split(":")
    if len(parts) >= 2:
        val = f"{parts[0]}:{parts[1]}"
    return val
