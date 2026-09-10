"""External pVACtools command wrapper.

This module is a thin boundary around the pVACtools Apptainer image. It only
assembles external commands, validates expected files, and returns output paths
to downstream MimicNeoAI code. No third-party implementation is vendored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class PvacseqExternalConfig:
    """Runtime configuration for external pVACtools calls."""

    pvactools_sif: Path
    apptainer: str = "apptainer"
    bcftools: str = "bcftools"
    tabix: str = "tabix"
    bind_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ProteinFastaOutputs:
    """Paths produced by ``pvacseq generate_protein_fasta``."""

    sorted_vcf_gz: Path
    sorted_vcf_tbi: Path
    protein_fasta: Path
    manufacturability_tsv: Path


@dataclass(frozen=True)
class VcfConverterOutputs:
    """Paths produced by pVACtools VCF conversion."""

    converter_tsv: Path


def build_bind_args(bind_paths: Iterable[Path]) -> list[str]:
    """Build Apptainer bind arguments for external pVACtools calls."""

    args: list[str] = []
    seen: set[str] = set()
    for path in bind_paths:
        path_str = str(Path(path))
        if not path_str or path_str in seen:
            continue
        seen.add(path_str)
        args.extend(["--bind", path_str])
    return args


def sort_and_index_vcf(input_vcf: Path, output_vcf_gz: Path, config: PvacseqExternalConfig) -> Path:
    """Sort and index a VCF before passing it to pVACtools."""

    input_vcf = Path(input_vcf)
    output_vcf_gz = Path(output_vcf_gz)
    output_vcf_gz.parent.mkdir(parents=True, exist_ok=True)

    output_tbi = Path(str(output_vcf_gz) + ".tbi")
    if output_vcf_gz.exists() and output_vcf_gz.stat().st_size > 0 and output_tbi.exists():
        return output_vcf_gz

    commands = [
        [
            config.bcftools,
            "sort",
            "-Oz",
            "-o",
            str(output_vcf_gz),
            str(input_vcf),
        ],
        [
            config.tabix,
            "-p",
            "vcf",
            str(output_vcf_gz),
        ],
    ]
    run_commands(commands)
    require_files([output_vcf_gz, output_tbi])
    return output_vcf_gz


def run_generate_protein_fasta(
    *,
    sample: str,
    input_vcf: Path,
    output_dir: Path,
    flank_length: int,
    config: PvacseqExternalConfig,
    mutant_only: bool = False,
    pass_only: bool = True,
    output_prefix: Optional[str] = None,
) -> ProteinFastaOutputs:
    """Run ``pvacseq generate_protein_fasta`` through Apptainer.

    The output FASTA is used only as a WT/MT protein sequence source for the
    MimicNeoAI workflow. Downstream epitope window generation is handled by this
    package, not by pVACseq.
    """

    input_vcf = Path(input_vcf)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_prefix or f"{sample}.protein.flank{flank_length}"
    if mutant_only:
        prefix = f"{prefix}.mutant_only"
    else:
        prefix = f"{prefix}.wt_mt"

    sorted_vcf_gz = output_dir / f"{sample}.shared.VEP.rm_mismatch.sorted.vcf.gz"
    sorted_vcf_gz = sort_and_index_vcf(input_vcf, sorted_vcf_gz, config)
    sorted_vcf_tbi = Path(str(sorted_vcf_gz) + ".tbi")
    protein_fasta = output_dir / f"{prefix}.fasta"
    manufacturability_tsv = Path(f"{protein_fasta}.manufacturability.tsv")

    if protein_fasta.exists() and protein_fasta.stat().st_size > 0 and manufacturability_tsv.exists():
        return ProteinFastaOutputs(
            sorted_vcf_gz=sorted_vcf_gz,
            sorted_vcf_tbi=sorted_vcf_tbi,
            protein_fasta=protein_fasta,
            manufacturability_tsv=manufacturability_tsv,
        )

    command = [
        config.apptainer,
        "exec",
        *build_bind_args(config.bind_paths),
        str(config.pvactools_sif),
        "pvacseq",
        "generate_protein_fasta",
        "-s",
        sample,
    ]
    if mutant_only:
        command.append("--mutant-only")
    if pass_only:
        command.append("--pass-only")
    command.extend([str(sorted_vcf_gz), str(flank_length), str(protein_fasta.name)])

    run_commands([command], cwd=output_dir)
    require_files([protein_fasta, manufacturability_tsv])
    return ProteinFastaOutputs(
        sorted_vcf_gz=sorted_vcf_gz,
        sorted_vcf_tbi=sorted_vcf_tbi,
        protein_fasta=protein_fasta,
        manufacturability_tsv=manufacturability_tsv,
    )


def run_vcf_converter(
    *,
    sample: str,
    input_vcf: Path,
    output_tsv: Path,
    config: PvacseqExternalConfig,
    pass_only: bool = True,
) -> VcfConverterOutputs:
    """Run the pVACtools VCF conversion step externally.

    This step obtains mutation annotation fields from the external converter.
    The returned table should be treated as an annotation source, not as the
    primary stable identifier system for the new workflow.
    """

    input_vcf = Path(input_vcf)
    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    if output_tsv.exists() and output_tsv.stat().st_size > 0:
        return VcfConverterOutputs(converter_tsv=output_tsv)

    pass_only_literal = "True" if pass_only else "False"
    python_code = (
        "from pvactools.lib.input_file_converter import VcfConverter; "
        "params={"
        f"'input_file': {str(input_vcf)!r}, "
        f"'output_file': {str(output_tsv)!r}, "
        f"'sample_name': {sample!r}, "
        f"'pass_only': {pass_only_literal}"
        "}; "
        "VcfConverter(**params).execute()"
    )
    command = [
        config.apptainer,
        "exec",
        *build_bind_args(config.bind_paths),
        str(config.pvactools_sif),
        "python",
        "-c",
        python_code,
    ]
    run_commands([command])
    require_files([output_tsv])
    return VcfConverterOutputs(converter_tsv=output_tsv)


def run_commands(commands: Sequence[Sequence[str]], cwd: Optional[Path] = None) -> None:
    """Execute shell-free command lists with logging and failure propagation."""

    for command in commands:
        printable = " ".join(shlex.quote(str(part)) for part in command)
        print(f"[CMD] {printable}", flush=True)
        subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd) if cwd is not None else None,
            check=True,
        )


def require_files(paths: Iterable[Path]) -> None:
    """Validate that all expected output files exist and are non-empty."""

    missing: list[str] = []
    for path in paths:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Expected output file(s) missing or empty: " + ", ".join(missing))
