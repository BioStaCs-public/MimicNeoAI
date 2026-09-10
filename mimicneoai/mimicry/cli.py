"""Command-line interface for sequence-based molecular mimicry analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from ._engine import run_sequence_mimicry, validate_run_config
from ._models import MimicryRunConfig, SUPPORTED_SOURCE_PAIRS


def load_run_config(path: Path) -> MimicryRunConfig:
    """Load the compact YAML configuration used by ``mimicneoai mimicry``."""

    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    allowed_keys = {
        "method",
        "input_manifest",
        "output_dir",
        "source_pairs",
        "workers",
        "shard_count",
        "binding_support",
    }
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"Unknown mimicry configuration fields: {unknown_keys}")
    input_manifest = Path(str(raw.get("input_manifest", ""))).expanduser()
    output_dir = Path(str(raw.get("output_dir", ""))).expanduser()
    if not input_manifest.is_absolute():
        input_manifest = (path.parent / input_manifest).resolve()
    if not output_dir.is_absolute():
        output_dir = (path.parent / output_dir).resolve()
    source_pairs = raw.get("source_pairs", list(SUPPORTED_SOURCE_PAIRS))
    if not isinstance(source_pairs, list):
        raise ValueError("source_pairs must be a YAML list")
    binding_support = raw.get("binding_support", {}) or {}
    if not isinstance(binding_support, dict):
        raise ValueError("binding_support must be a YAML mapping")
    unknown_binding_keys = sorted(set(binding_support) - {"enabled"})
    if unknown_binding_keys:
        raise ValueError(
            f"Unknown binding_support configuration fields: {unknown_binding_keys}"
        )
    binding_enabled = binding_support.get("enabled", False)
    if not isinstance(binding_enabled, bool):
        raise ValueError("binding_support.enabled must be true or false")
    config = MimicryRunConfig(
        method=str(raw.get("method", "sequence_mimicry_v1.2")),
        input_manifest=input_manifest,
        output_dir=output_dir,
        source_pairs=tuple(str(value).strip() for value in source_pairs),
        workers=int(raw.get("workers", 1)),
        shard_count=int(raw.get("shard_count", 1)),
        binding_support_enabled=binding_enabled,
    )
    if not str(raw.get("input_manifest", "")).strip():
        raise ValueError("input_manifest is required")
    if not str(raw.get("output_dir", "")).strip():
        raise ValueError("output_dir is required")
    return validate_run_config(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Identify within-patient, cross-source sequence-similar peptide pairs "
            "and optionally assess shared predicted HLA-I binding."
        )
    )
    parser.add_argument("-c", "--configure", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_sequence_mimicry(load_run_config(args.configure.resolve()))
    print(
        json.dumps(
            {
                "run_status": manifest["run_status"],
                "method_version": manifest["method_version"],
                "row_counts": manifest["row_counts"],
                "reused": manifest.get("reused", False),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
