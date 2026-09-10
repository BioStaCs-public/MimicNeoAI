"""Step 00: prepare pVACtools-derived source files.

This step is the only planned boundary that calls pVACtools externally. It will
prepare:

- sorted/indexed VCF;
- pVACtools converter annotation TSV;
- WT/MT protein FASTA from ``pvacseq generate_protein_fasta``;
- command logs and output checks.

Downstream MimicNeoAI steps should not depend on pVACseq chunk-local ``Index``
values as stable identifiers.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

try:
    from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.external_pvactools import (
        PvacseqExternalConfig,
        run_generate_protein_fasta,
        run_vcf_converter,
    )
except ImportError:  # pragma: no cover - supports direct script execution before install
    from external_pvactools import (  # type: ignore
        PvacseqExternalConfig,
        run_generate_protein_fasta,
        run_vcf_converter,
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for pVACtools source preparation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("--input-vcf", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--pvactools-sif", required=True)
    parser.add_argument("--apptainer", default="apptainer")
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--tabix", default="tabix")
    parser.add_argument("--flank-length", type=int, default=25)
    parser.add_argument("--mutant-only", action="store_true")
    parser.add_argument("--no-pass-only", action="store_true")
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        help="Path to bind into the pVACtools Apptainer container. Can be used multiple times.",
    )
    parser.add_argument(
        "--converter-tsv",
        default=None,
        help="Optional converter TSV output path. Defaults to <outdir>/<sample>.pvacseq_converter.tsv.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Optional protein FASTA prefix before .wt_mt/.mutant_only.fasta.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    input_vcf = Path(args.input_vcf)
    pass_only = not args.no_pass_only
    input_vcf_dir = outdir / "00_input_vcf"
    pvactools_sources_dir = outdir / "01_pvactools_sources"
    input_vcf_dir.mkdir(parents=True, exist_ok=True)
    pvactools_sources_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = pvactools_sources_dir / f"{args.sample}.source_inputs.manifest.json"
    source_signature = {
        "sample": args.sample,
        "input_vcf": file_identity(input_vcf),
        "flank_length": args.flank_length,
        "mutant_only": args.mutant_only,
        "pass_only": pass_only,
        "pvactools_sif": optional_file_identity(Path(args.pvactools_sif)),
    }
    validate_source_manifest(source_manifest_path, source_signature)
    restore_sorted_vcf_for_resume(args.sample, input_vcf_dir, pvactools_sources_dir)

    bind_paths = [Path(p) for p in args.bind]
    for auto_bind in (input_vcf.parent, outdir, input_vcf_dir, pvactools_sources_dir):
        if auto_bind not in bind_paths:
            bind_paths.append(auto_bind)

    config = PvacseqExternalConfig(
        pvactools_sif=Path(args.pvactools_sif),
        apptainer=args.apptainer,
        bcftools=args.bcftools,
        tabix=args.tabix,
        bind_paths=tuple(bind_paths),
    )

    converter_tsv = Path(args.converter_tsv) if args.converter_tsv else pvactools_sources_dir / f"{args.sample}.pvacseq_converter.tsv"
    converter_outputs = run_vcf_converter(
        sample=args.sample,
        input_vcf=input_vcf,
        output_tsv=converter_tsv,
        config=config,
        pass_only=pass_only,
    )
    fasta_outputs = run_generate_protein_fasta(
        sample=args.sample,
        input_vcf=input_vcf,
        output_dir=pvactools_sources_dir,
        flank_length=args.flank_length,
        config=config,
        mutant_only=args.mutant_only,
        pass_only=pass_only,
        output_prefix=args.output_prefix,
    )

    # Keep sorted/indexed VCFs in a dedicated subdirectory. The pVACtools FASTA
    # command needs the sorted VCF in its working directory only while running.
    sorted_vcf_gz = move_if_needed(fasta_outputs.sorted_vcf_gz, input_vcf_dir / fasta_outputs.sorted_vcf_gz.name)
    sorted_vcf_tbi = move_if_needed(fasta_outputs.sorted_vcf_tbi, input_vcf_dir / fasta_outputs.sorted_vcf_tbi.name)
    write_source_manifest(source_manifest_path, source_signature)

    print("[DONE] pVACtools source preparation finished", flush=True)
    print(f"converter_tsv={converter_outputs.converter_tsv}", flush=True)
    print(f"protein_fasta={fasta_outputs.protein_fasta}", flush=True)
    print(f"manufacturability_tsv={fasta_outputs.manufacturability_tsv}", flush=True)
    print(f"sorted_vcf_gz={sorted_vcf_gz}", flush=True)
    print(f"sorted_vcf_tbi={sorted_vcf_tbi}", flush=True)
    print(f"source_manifest={source_manifest_path}", flush=True)
    return 0


def file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def optional_file_identity(path: Path) -> dict[str, object]:
    if path.exists():
        return file_identity(path)
    return {"path": str(path)}


def validate_source_manifest(path: Path, signature: dict[str, object]) -> None:
    """Reject stale pVACtools source reuse when a prior signature is present."""

    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        with path.open() as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid source manifest {path}: {exc}") from exc
    if manifest.get("input_signature") != signature:
        raise RuntimeError(
            "pVACtools source inputs differ from the existing output manifest. "
            "Use a new output directory rather than reusing stale converter/FASTA files."
        )


def write_source_manifest(path: Path, signature: dict[str, object]) -> None:
    with path.open("w") as handle:
        json.dump({"input_signature": signature}, handle, indent=2, ensure_ascii=False)


def move_if_needed(source: Path, target: Path) -> Path:
    """Move a file into the organized output location if needed."""

    source = Path(source)
    target = Path(target)
    if source == target:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists() and target.exists() and target.stat().st_size > 0:
        return target
    source.replace(target)
    return target


def restore_sorted_vcf_for_resume(sample: str, input_vcf_dir: Path, sources_dir: Path) -> None:
    """Make an organized sorted VCF visible to the reusable pVACtools step."""

    filename = f"{sample}.shared.VEP.rm_mismatch.sorted.vcf.gz"
    for suffix in ("", ".tbi"):
        organized = input_vcf_dir / f"{filename}{suffix}"
        source = sources_dir / f"{filename}{suffix}"
        if source.exists() or not organized.exists():
            continue
        shutil.copy2(organized, source)


if __name__ == "__main__":
    raise SystemExit(main())
