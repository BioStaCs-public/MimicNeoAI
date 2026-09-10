"""Run the MimicNeoAI mutation-derived binding prediction workflow.

The workflow uses pVACtools only as an external source generator for mutation
annotation and WT/MT protein FASTA. MimicNeoAI then builds mutation-covering
epitope windows, runs local binding predictors, and writes a downstream-compatible
merged all_epitopes table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from mimicneoai.functions.binding_prediction.policy import resolve_binding_prediction_policy


DEFAULT_ALGORITHMS = (
    "MHCflurry",
    "MHCflurryEL",
    "MHCnuggetsI",
    "MHCnuggetsII",
    "NNalign",
    "NetMHCpan",
    "NetMHCpanEL",
    "NetMHCIIpan",
    "NetMHCIIpanEL",
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("--input-vcf", required=True)
    parser.add_argument("--hla-file", required=True)
    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="Workflow output directory, e.g. <sample>/07.binding_prediction_mimicneoai.",
    )
    parser.add_argument("--pvactools-sif", required=True)
    parser.add_argument("--apptainer", default="apptainer")
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--tabix", default="tabix")
    parser.add_argument("--bind", action="append", default=[])
    parser.add_argument(
        "--preset",
        default="",
        help="Optional compact prediction preset. Supported: full, fast. Empty preserves explicit CLI settings.",
    )
    parser.add_argument(
        "--start-from",
        choices=("source_prep", "epitope_tasks"),
        default="source_prep",
        help=(
            "Resume point. Use epitope_tasks when 01_pvactools_sources and "
            "02_epitope_tasks are already frozen and only binding/merge should run."
        ),
    )
    parser.add_argument("--mhc-i-lengths", default="8,9,10,11")
    parser.add_argument("--mhc-ii-lengths", default="15")
    parser.add_argument("--protein-flank-length", type=int, default=25)
    parser.add_argument("--extended-length", type=int, default=27)
    parser.add_argument("--algorithms", default=",".join(DEFAULT_ALGORITHMS))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mt-workers", type=int, default=None)
    parser.add_argument("--wt-workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--command-timeout", type=int, default=None)
    parser.add_argument(
        "--mhcflurry-predict-bin",
        default=os.environ.get("MIMICNEOAI_MHCFLURRY_PREDICT_BIN", "mhcflurry-predict"),
    )
    parser.add_argument(
        "--mhcflurry-downloads-dir",
        default=os.environ.get("MHCFLURRY_DOWNLOADS_DIR", ""),
    )
    parser.add_argument(
        "--mhcnuggets-python-bin",
        default=os.environ.get("MIMICNEOAI_MHCNUGGETS_PYTHON_BIN", sys.executable),
    )
    parser.add_argument(
        "--mhcnuggets-script",
        default=os.environ.get("MIMICNEOAI_MHCNUGGETS_SCRIPT", "predict.py"),
    )
    parser.add_argument(
        "--mhcnuggets-cwd",
        default=os.environ.get("MIMICNEOAI_MHCNUGGETS_CWD", "."),
    )
    parser.add_argument(
        "--netmhcpan-bin",
        default=os.environ.get("MIMICNEOAI_NETMHCPAN_BIN", "netMHCpan"),
    )
    parser.add_argument(
        "--netmhciipan-bin",
        default=os.environ.get("MIMICNEOAI_NETMHCIIPAN_BIN", "netMHCIIpan"),
    )
    parser.add_argument(
        "--iedb-mhci-python-bin",
        default=os.environ.get("MIMICNEOAI_IEDB_MHCI_PYTHON_BIN", sys.executable),
    )
    parser.add_argument(
        "--iedb-mhci-script",
        default=os.environ.get("MIMICNEOAI_IEDB_MHCI_SCRIPT", "predict_binding.py"),
    )
    parser.add_argument(
        "--iedb-mhci-cwd",
        default=os.environ.get("MIMICNEOAI_IEDB_MHCI_CWD", "."),
    )
    parser.add_argument(
        "--iedb-mhcii-python-bin",
        default=os.environ.get("MIMICNEOAI_IEDB_MHCII_PYTHON_BIN", sys.executable),
    )
    parser.add_argument(
        "--iedb-mhcii-script",
        default=os.environ.get("MIMICNEOAI_IEDB_MHCII_SCRIPT", "mhc_II_binding.py"),
    )
    parser.add_argument(
        "--iedb-mhcii-cwd",
        default=os.environ.get("MIMICNEOAI_IEDB_MHCII_CWD", "."),
    )
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--pvacseq-merged", default=None, help="Optional legacy pVACseq merged TSV for validation.")
    parser.add_argument("--no-pass-only", action="store_true")
    parser.add_argument("--no-archive-runner-workdirs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    apply_policy_overrides(args)
    outdir = Path(args.outdir)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    paths = workflow_paths(outdir, args.sample, args.protein_flank_length)
    commands = build_commands(args, paths)
    commands = commands_from_resume_point(commands, args.start_from)
    if args.dry_run:
        for label, command in commands:
            print(f"[DRY-RUN] {label}: {format_command(command)}", flush=True)
        return 0

    validate_predictor_configuration(args)
    validate_resume_inputs(args.start_from, paths)

    outdir.mkdir(parents=True, exist_ok=True)
    for key in ("epitope_tasks", "binding_predictions", "merged_epitopes", "archive"):
        paths[key].mkdir(parents=True, exist_ok=True)

    run_steps(commands, paths, python_bin=args.python_bin)
    archive_runner_workdirs(paths, args.sample, enabled=not args.no_archive_runner_workdirs)
    summary = build_workflow_summary(args, paths, started_at)
    summary_path = outdir / f"{args.sample}.binding_prediction_mimicneoai.summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[DONE] MimicNeoAI binding prediction workflow finished: {outdir}", flush=True)
    print(f"[DONE] Summary: {summary_path}", flush=True)
    return 0


def apply_policy_overrides(args: argparse.Namespace) -> None:
    """Apply a binding prediction preset while preserving explicit mode by default."""

    policy = resolve_binding_prediction_policy(args.preset)
    if policy is None:
        return
    args.mhc_i_lengths = ",".join(str(value) for value in policy.mhc_i_lengths)
    args.mhc_ii_lengths = ",".join(str(value) for value in policy.mhc_ii_lengths)
    args.algorithms = " ".join(policy.algorithms)


def workflow_paths(outdir: Path, sample: str, protein_flank_length: int) -> dict[str, Path]:
    return {
        "root": outdir,
        "pvactools_sources": outdir / "01_pvactools_sources",
        "epitope_tasks": outdir / "02_epitope_tasks",
        "binding_predictions": outdir / "03_binding_predictions",
        "merged_epitopes": outdir / "04_merged_epitopes",
        "archive": outdir / "archive",
        "converter_tsv": outdir / "01_pvactools_sources" / f"{sample}.pvacseq_converter.tsv",
        "protein_fasta": outdir / "01_pvactools_sources" / f"{sample}.protein.flank{protein_flank_length}.wt_mt.fasta",
        "binding_tasks": outdir / "02_epitope_tasks" / "binding_tasks.tsv",
        "epitope_windows": outdir / "02_epitope_tasks" / "epitope_windows.tsv",
        "prediction_peptides": outdir / "02_epitope_tasks" / "prediction_peptides.tsv",
        "variant_annotation": outdir / "02_epitope_tasks" / "variant_annotation.tsv",
        "stage1_tasks": outdir / "02_epitope_tasks" / "stage1_binding_tasks.tsv",
        "stage1_tasks_summary": outdir / "02_epitope_tasks" / "stage1_binding_tasks.summary.json",
        "stage1_routing": outdir / "02_epitope_tasks" / "stage1_routing.tsv",
        "stage1_unsupported": outdir / "02_epitope_tasks" / "stage1_unsupported_or_failed.tsv",
        "stage1_route_summary": outdir / "02_epitope_tasks" / "stage1_route.summary.json",
        "stage1_runner": outdir / "03_binding_predictions" / f"{sample}.stage1.runner",
        "stage1_predictions": outdir / "03_binding_predictions" / f"{sample}.stage1.binding_predictions.long.tsv",
        "stage1_pass_predictions": outdir / "03_binding_predictions" / f"{sample}.stage1_pass.binding_predictions.long.tsv",
        "mt_tasks": outdir / "03_binding_predictions" / f"{sample}.MT.binding_tasks.tsv",
        "wt_tasks": outdir / "03_binding_predictions" / f"{sample}.WT.binding_tasks.tsv",
        "mt_runner": outdir / "03_binding_predictions" / f"{sample}.MT.runner",
        "wt_runner": outdir / "03_binding_predictions" / f"{sample}.WT.runner",
        "mt_predictions": outdir / "03_binding_predictions" / f"{sample}.MT.binding_predictions.long.tsv",
        "wt_predictions": outdir / "03_binding_predictions" / f"{sample}.WT.binding_predictions.long.tsv",
        "merged": outdir / "04_merged_epitopes" / f"{sample}.merged.all_epitopes.tsv",
    }


def build_commands(args: argparse.Namespace, paths: dict[str, Path]) -> list[tuple[str, list[str]]]:
    python_bin = args.python_bin
    policy = resolve_binding_prediction_policy(args.preset)
    bind_args: list[str] = []
    for bind_path in args.bind:
        bind_args.extend(["--bind", bind_path])
    pass_args = ["--no-pass-only"] if args.no_pass_only else []
    chunk_args = ["--chunk-size", str(args.chunk_size)] if args.chunk_size else []
    timeout_args = ["--command-timeout", str(args.command_timeout)] if args.command_timeout else []
    pvacseq_validation_args = ["--pvacseq-merged", args.pvacseq_merged] if args.pvacseq_merged else []
    predictor_args = runner_predictor_args(args)

    commands = [
        (
            "00_prepare_pvacseq_sources",
            [
                python_bin,
                str(SCRIPT_DIR / "00_prepare_pvacseq_sources.py"),
                "-s",
                args.sample,
                "--input-vcf",
                args.input_vcf,
                "-o",
                str(paths["root"]),
                "--pvactools-sif",
                args.pvactools_sif,
                "--apptainer",
                args.apptainer,
                "--bcftools",
                args.bcftools,
                "--tabix",
                args.tabix,
                "--flank-length",
                str(args.protein_flank_length),
                *bind_args,
                *pass_args,
            ],
        ),
        (
            "01_build_epitope_tasks",
            [
                python_bin,
                str(SCRIPT_DIR / "01_build_epitope_tasks.py"),
                "-s",
                args.sample,
                "--converter-tsv",
                str(paths["converter_tsv"]),
                "--protein-fasta",
                str(paths["protein_fasta"]),
                "--hla-file",
                args.hla_file,
                "-o",
                str(paths["epitope_tasks"]),
                "--mhc-i-lengths",
                args.mhc_i_lengths,
                "--mhc-ii-lengths",
                args.mhc_ii_lengths,
                "--protein-flank-length",
                str(args.protein_flank_length),
                "--extended-length",
                str(args.extended_length),
                "--algorithms",
                args.algorithms,
                *pvacseq_validation_args,
            ],
        ),
        (
            "02_split_binding_tasks",
            [
                python_bin,
                str(SCRIPT_DIR / "02_split_binding_tasks.py"),
                "-s",
                args.sample,
                "--binding-tasks",
                str(paths["binding_tasks"]),
                "--epitope-windows",
                str(paths["epitope_windows"]),
                "-o",
                str(paths["binding_predictions"]),
                "--mt-output",
                str(paths["mt_tasks"]),
                "--wt-output",
                str(paths["wt_tasks"]),
            ],
        ),
        (
            "03_predict_MT",
            [
                python_bin,
                "-m",
                "mimicneoai.functions.binding_prediction.runner",
                "--tasks",
                str(paths["mt_tasks"]),
                "-o",
                str(paths["mt_runner"]),
                "--workers",
                str(args.mt_workers or args.workers),
                "--device",
                args.device,
                *predictor_args,
                *chunk_args,
                *timeout_args,
            ],
        ),
        (
            "03_predict_WT",
            [
                python_bin,
                "-m",
                "mimicneoai.functions.binding_prediction.runner",
                "--tasks",
                str(paths["wt_tasks"]),
                "-o",
                str(paths["wt_runner"]),
                "--workers",
                str(args.wt_workers or args.workers),
                "--device",
                args.device,
                *predictor_args,
                *chunk_args,
                *timeout_args,
            ],
        ),
        (
            "04_merge_binding_predictions",
            [
                python_bin,
                str(SCRIPT_DIR / "02_merge_binding_predictions.py"),
                "--variant-annotation",
                str(paths["variant_annotation"]),
                "--epitope-windows",
                str(paths["epitope_windows"]),
                "--binding-predictions",
                str(paths["mt_predictions"]),
                "--binding-predictions",
                str(paths["wt_predictions"]),
                "--binding-tasks",
                str(paths["mt_tasks"]),
                "--output-profile",
                "pvacseq",
                "-o",
                str(paths["merged"]),
                "--summary",
                str(paths["merged_epitopes"] / f"{args.sample}.merged.summary.json"),
                *pvacseq_validation_args,
            ],
        ),
    ]
    if policy and policy.two_stage:
        stage1_commands = [
            (
                "02_stage1_build_tasks",
                [
                    python_bin,
                    "-m",
                    "mimicneoai.functions.binding_prediction.stage1",
                    "build-mutation",
                    "--prediction-peptides",
                    str(paths["prediction_peptides"]),
                    "--hla-file",
                    args.hla_file,
                    "--preset",
                    policy.name,
                    "-o",
                    str(paths["stage1_tasks"]),
                    "--summary",
                    str(paths["stage1_tasks_summary"]),
                ],
            ),
            (
                "02_stage1_predict",
                [
                    python_bin,
                    "-m",
                    "mimicneoai.functions.binding_prediction.runner",
                    "--tasks",
                    str(paths["stage1_tasks"]),
                    "-o",
                    str(paths["stage1_runner"]),
                    "--algorithms",
                    ",".join(policy.stage1_algorithms),
                    "--workers",
                    str(args.workers),
                    "--device",
                    args.device,
                    "--netmhc-el-only",
                    *predictor_args,
                    *chunk_args,
                    *timeout_args,
                ],
            ),
            (
                "02_stage1_route",
                [
                    python_bin,
                    "-m",
                    "mimicneoai.functions.binding_prediction.stage1",
                    "route",
                    "--stage1-tasks",
                    str(paths["stage1_tasks"]),
                    "--stage1-predictions",
                    str(paths["stage1_runner"] / "binding_predictions.long.tsv"),
                    "--preset",
                    policy.name,
                    "--stage2-algorithms",
                    args.algorithms,
                    "-o",
                    str(paths["binding_tasks"]),
                    "--routing-output",
                    str(paths["stage1_routing"]),
                    "--unsupported-output",
                    str(paths["stage1_unsupported"]),
                    "--pass-predictions-output",
                    str(paths["stage1_pass_predictions"]),
                    "--summary",
                    str(paths["stage1_route_summary"]),
                ],
            ),
        ]
        commands = commands[:2] + stage1_commands + commands[2:]
    return commands


def commands_from_resume_point(
    commands: list[tuple[str, list[str]]],
    start_from: str,
) -> list[tuple[str, list[str]]]:
    """Return workflow commands beginning at the requested resume point."""

    if start_from == "source_prep":
        return commands
    if start_from == "epitope_tasks":
        start_labels = {"02_stage1_build_tasks", "02_split_binding_tasks"}
        for index, (label, _) in enumerate(commands):
            if label in start_labels:
                return commands[index:]
        raise ValueError("Cannot start from epitope_tasks: no downstream binding step was found")
    raise ValueError(f"Unsupported start_from value: {start_from}")


def validate_resume_inputs(start_from: str, paths: dict[str, Path]) -> None:
    """Validate that requested resume inputs already exist."""

    if start_from == "source_prep":
        return
    if start_from != "epitope_tasks":
        raise ValueError(f"Unsupported start_from value: {start_from}")
    required = [
        paths["converter_tsv"],
        paths["protein_fasta"],
        paths["variant_annotation"],
        paths["epitope_windows"],
        paths["prediction_peptides"],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot start from epitope_tasks because required frozen inputs are missing:\n- "
            + "\n- ".join(missing)
        )


def runner_predictor_args(args: argparse.Namespace) -> list[str]:
    values = [
        ("--mhcflurry-predict-bin", args.mhcflurry_predict_bin),
        ("--mhcflurry-downloads-dir", args.mhcflurry_downloads_dir),
        ("--mhcnuggets-python-bin", args.mhcnuggets_python_bin),
        ("--mhcnuggets-script", args.mhcnuggets_script),
        ("--mhcnuggets-cwd", args.mhcnuggets_cwd),
        ("--netmhcpan-bin", args.netmhcpan_bin),
        ("--netmhciipan-bin", args.netmhciipan_bin),
        ("--python-bin", args.iedb_mhci_python_bin),
        ("--iedb-mhci-script", args.iedb_mhci_script),
        ("--iedb-mhci-cwd", args.iedb_mhci_cwd),
        ("--iedb-mhcii-python-bin", args.iedb_mhcii_python_bin),
        ("--iedb-mhcii-script", args.iedb_mhcii_script),
        ("--iedb-mhcii-cwd", args.iedb_mhcii_cwd),
    ]
    result: list[str] = []
    for argument, value in values:
        if value:
            result.extend([argument, str(value)])
    return result


def validate_predictor_configuration(args: argparse.Namespace) -> None:
    algorithms = {
        item for item in args.algorithms.replace(",", " ").split() if item
    }
    errors: list[str] = []

    def require_executable(label: str, value: str) -> None:
        path = Path(value)
        if path.parent != Path("."):
            if not path.is_file() or not os.access(path, os.X_OK):
                errors.append(f"{label} is not executable: {value}")
        elif shutil.which(value) is None:
            errors.append(f"{label} was not found on PATH: {value}")

    def require_file(label: str, value: str) -> None:
        if not Path(value).is_file():
            errors.append(f"{label} does not exist: {value}")

    def require_dir(label: str, value: str) -> None:
        if not Path(value).is_dir():
            errors.append(f"{label} does not exist: {value}")

    if algorithms & {"MHCflurry", "MHCflurryEL"}:
        require_executable("MHCflurry executable", args.mhcflurry_predict_bin)
        require_dir("MHCflurry downloads directory", args.mhcflurry_downloads_dir)
    if algorithms & {"MHCnuggetsI", "MHCnuggetsII"}:
        require_executable("MHCnuggets Python", args.mhcnuggets_python_bin)
        require_file("MHCnuggets script", args.mhcnuggets_script)
        require_dir("MHCnuggets working directory", args.mhcnuggets_cwd)
    if algorithms & {"NetMHCpan", "NetMHCpanEL"}:
        require_executable("NetMHCpan executable", args.netmhcpan_bin)
    if algorithms & {"NetMHCIIpan", "NetMHCIIpanEL"}:
        require_executable("NetMHCIIpan executable", args.netmhciipan_bin)
    if algorithms & {"SMM", "SMMPMBEC", "PickPocket"}:
        require_executable("IEDB MHC-I Python", args.iedb_mhci_python_bin)
        require_file("IEDB MHC-I script", args.iedb_mhci_script)
        require_dir("IEDB MHC-I working directory", args.iedb_mhci_cwd)
    if "NNalign" in algorithms:
        require_executable("IEDB MHC-II Python", args.iedb_mhcii_python_bin)
        require_file("IEDB MHC-II script", args.iedb_mhcii_script)
        require_dir("IEDB MHC-II working directory", args.iedb_mhcii_cwd)

    if errors:
        raise RuntimeError(
            "Binding predictor configuration is incomplete:\n- "
            + "\n- ".join(errors)
        )


def run_steps(commands: list[tuple[str, list[str]]], paths: dict[str, Path], python_bin: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for label, command in commands:
        print(f"[RUN] {label}: {format_command(command)}", flush=True)
        subprocess.run(command, check=True, env=env)
        if label == "03_predict_MT":
            move_runner_outputs(paths["mt_runner"], paths["binding_predictions"], "MT")
        elif label == "03_predict_WT":
            move_runner_outputs(paths["wt_runner"], paths["binding_predictions"], "WT")


def move_runner_outputs(runner_dir: Path, parent: Path, source_type: str) -> None:
    sample = runner_dir.name.rsplit(f".{source_type}.runner", 1)[0]
    mapping = {
        runner_dir / "binding_predictions.long.tsv": parent / f"{sample}.{source_type}.binding_predictions.long.tsv",
        runner_dir / "binding_predictions.summary.json": parent / f"{sample}.{source_type}.binding_predictions.summary.json",
    }
    for source, target in mapping.items():
        if not source.exists():
            raise FileNotFoundError(f"Expected runner output missing: {source}")
        if target.exists():
            target.unlink()
        source.replace(target)


def archive_runner_workdirs(paths: dict[str, Path], sample: str, enabled: bool) -> None:
    if not enabled:
        return
    for source_type in ("MT", "WT"):
        runner_dir = paths[f"{source_type.lower()}_runner"]
        if not runner_dir.exists():
            continue
        target = paths["archive"] / "03_binding_predictions" / runner_dir.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            target = target.with_name(f"{target.name}.{timestamp}.old")
        shutil.move(str(runner_dir), str(target))


def build_workflow_summary(args: argparse.Namespace, paths: dict[str, Path], started_at: str) -> dict[str, object]:
    policy = resolve_binding_prediction_policy(args.preset)
    stage1_task_summary = read_json(paths["stage1_tasks_summary"])
    stage1_route_summary = read_json(paths["stage1_route_summary"])
    mt_split_summary = read_json(paths["binding_predictions"] / f"{args.sample}.split_binding_tasks.summary.json")
    mt_prediction_summary = read_json(paths["binding_predictions"] / f"{args.sample}.MT.binding_predictions.summary.json")
    wt_prediction_summary = read_json(paths["binding_predictions"] / f"{args.sample}.WT.binding_predictions.summary.json")
    merged_summary = read_json(paths["merged_epitopes"] / f"{args.sample}.merged.summary.json")
    return {
        "sample": args.sample,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workflow": "MimicNeoAI-native mutation-derived binding prediction",
        "output_dir": str(paths["root"]),
        "start_from": args.start_from,
        "binding_prediction_preset": policy.name if policy else "",
        "two_stage_binding_prediction": bool(policy and policy.two_stage),
        "pvactools_boundary": "pVACtools is used only for VCF annotation conversion and WT/MT protein FASTA source generation.",
        "pvactools_sif": args.pvactools_sif,
        "pvactools_version": capture_pvactools_version(args.apptainer, args.pvactools_sif),
        "netmhcpan_version": version_from_path("NetMHCpan", args.netmhcpan_bin),
        "netmhciipan_version": version_from_path("NetMHCIIpan", args.netmhciipan_bin),
        "algorithms": [item for item in args.algorithms.replace(",", " ").split() if item],
        "mhc_i_lengths": args.mhc_i_lengths,
        "mhc_ii_lengths": args.mhc_ii_lengths,
        "workers": args.workers,
        "mt_workers": args.mt_workers or args.workers,
        "wt_workers": args.wt_workers or args.workers,
        "stage1_task_summary": stage1_task_summary,
        "stage1_route_summary": stage1_route_summary,
        "mt_task_rows": mt_split_summary.get("mt_task_rows"),
        "wt_task_rows": mt_split_summary.get("wt_task_rows"),
        "mt_prediction_rows": mt_prediction_summary.get("prediction_rows"),
        "wt_prediction_rows": wt_prediction_summary.get("prediction_rows"),
        "binding_qc_summary": {
            "MT": mt_prediction_summary.get("qc_summary", {}),
            "WT": wt_prediction_summary.get("qc_summary", {}),
        },
        "merged_rows": merged_summary.get("rows"),
        "fs_wt_rule": "Frameshift WT Epitope Seq and corresponding WT IC50/fold-change/percentile fields are left blank.",
        "outputs": {
            "mt_binding_predictions": str(paths["mt_predictions"]),
            "wt_binding_predictions": str(paths["wt_predictions"]),
            "merged_all_epitopes": str(paths["merged"]),
        },
    }


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def capture_pvactools_version(apptainer: str, sif_path: str) -> str:
    command = [apptainer, "exec", sif_path, "pvactools", "--version"]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        queried = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        if result.returncode == 0:
            version_match = re.search(
                r"(?<!\d)(\d+(?:\.\d+){2,})(?!\d)", queried
            )
            if version_match:
                return f"pVACtools {version_match.group(1)}"
    except Exception as exc:  # pragma: no cover - best-effort metadata only
        queried = f"runtime query failed: {exc}"
    match = re.search(r"pvactools[-_](\d+(?:\.\d+)+)", Path(sif_path).name, re.IGNORECASE)
    if match:
        return f"pVACtools {match.group(1)} (from SIF filename; runtime query failed)"
    return queried[:2000]


def version_from_path(tool_name: str, executable: str) -> str:
    match = re.search(rf"{re.escape(tool_name)}[-_](\d+(?:\.\d+)+)", executable, re.IGNORECASE)
    if match:
        return f"{tool_name} {match.group(1)}"
    return str(Path(executable))


def format_command(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


if __name__ == "__main__":
    raise SystemExit(main())
