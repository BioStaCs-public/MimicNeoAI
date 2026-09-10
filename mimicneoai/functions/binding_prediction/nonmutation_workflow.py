"""Local binding workflow for non-mutation peptide antigens.

This module is used by cryptic and microbial antigen pipelines. It replaces
the pVACbind execution layer while keeping a pVACbind-like merged output table
for downstream compatibility.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable, Optional

from mimicneoai.functions.binding_prediction.policy import (
    resolve_binding_prediction_policy,
)
from mimicneoai.functions.binding_prediction.qc import build_binding_qc_summary
from mimicneoai.functions.binding_prediction.runner import main as run_binding_predictions
from mimicneoai.functions.binding_prediction.schema import PREDICTION_FIELDS, safe_float
from mimicneoai.functions.binding_prediction.stage1 import (
    route_stage2_tasks,
    write_stage1_tasks_from_nonmutation_windows,
)
from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.hla_parser import (
    parse_hlahd_result,
)


MHC_I_ALGORITHMS = {
    "BigMHC_EL",
    "BigMHC_IM",
    "DeepImmuno",
    "MHCflurry",
    "MHCflurryEL",
    "MHCnuggetsI",
    "NetMHC",
    "NetMHCpan",
    "NetMHCpanEL",
    "PickPocket",
    "SMM",
    "SMMPMBEC",
}
MHC_II_ALGORITHMS = {
    "MHCnuggetsII",
    "NNalign",
    "NetMHCIIpan",
    "NetMHCIIpanEL",
}
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
IC50_SUMMARY_ALGORITHMS = {
    "MHCflurry",
    "MHCnuggetsI",
    "NetMHC",
    "NetMHCpan",
    "PickPocket",
    "SMM",
    "SMMPMBEC",
    "MHCnuggetsII",
    "NetMHCIIpan",
    "NNalign",
    "SMMalign",
}

EPITOPE_WINDOWS_SCHEMA_VERSION = 2


PVACBIND_COMPAT_COLUMNS = [
    "Mutation",
    "HLA Allele",
    "Sub-peptide Position",
    "Epitope Seq",
    "Median IC50 Score",
    "Best IC50 Score",
    "Best IC50 Score Method",
    "Median Percentile",
    "Best Percentile",
    "Best Percentile Method",
    "MHCflurryEL Processing Score",
    "MHCflurryEL Presentation Score",
    "MHCflurryEL Presentation Percentile",
    "MHCflurry IC50 Score",
    "MHCflurry Percentile",
    "MHCnuggetsI IC50 Score",
    "MHCnuggetsI Percentile",
    "NetMHC IC50 Score",
    "NetMHC Percentile",
    "NetMHCpan IC50 Score",
    "NetMHCpan Percentile",
    "NetMHCpanEL Score",
    "NetMHCpanEL Percentile",
    "PickPocket IC50 Score",
    "PickPocket Percentile",
    "SMM IC50 Score",
    "SMM Percentile",
    "SMMPMBEC IC50 Score",
    "SMMPMBEC Percentile",
    "cterm_7mer_gravy_score",
    "max_7mer_gravy_score",
    "difficult_n_terminal_residue",
    "c_terminal_cysteine",
    "c_terminal_proline",
    "cysteine_count",
    "n_terminal_asparagine",
    "asparagine_proline_bond_count",
    "MHCnuggetsII IC50 Score",
    "MHCnuggetsII Percentile",
    "NetMHCIIpan IC50 Score",
    "NetMHCIIpan Percentile",
    "NetMHCIIpanEL Score",
    "NetMHCIIpanEL Percentile",
    "NNalign IC50 Score",
    "NNalign Percentile",
]


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("--pep-fasta", required=True, help="Input peptide or parent-sequence FASTA.")
    parser.add_argument(
        "--input-mode",
        choices=("parent-fasta", "peptide-core"),
        default="parent-fasta",
        help=(
            "parent-fasta tiles each FASTA record into candidate windows; "
            "peptide-core treats each FASTA record as an already tiled candidate peptide."
        ),
    )
    parser.add_argument("--hla-file", required=True, help="HLA-HD *_final.result.txt file.")
    parser.add_argument("-o", "--outdir", required=True, help="Output directory.")
    parser.add_argument(
        "--preset",
        default="",
        help="Optional compact prediction preset. Supported: full, fast. Empty preserves explicit CLI settings.",
    )
    parser.add_argument("--mhc-i-lengths", default="8,9,10")
    parser.add_argument("--mhc-ii-lengths", default="15")
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma- or space-separated predictor names.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument(
        "--max-runner-task-rows",
        type=int,
        default=200000,
        help="Maximum task rows loaded by one runner invocation.",
    )
    parser.add_argument(
        "--max-task-rows",
        type=int,
        default=5000000,
        help="Maximum total binding task rows before prediction is skipped unless --force-large-samples is set.",
    )
    parser.add_argument(
        "--force-large-samples",
        action="store_true",
        help="Run binding prediction even when total task rows exceed --max-task-rows.",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--command-timeout", type=int, default=None)
    parser.add_argument("--skip-prediction", action="store_true", help="Build tasks only.")
    parser.add_argument(
        "--windows-only",
        action="store_true",
        help="Write epitope windows and scale summary only. Do not write binding_tasks.tsv or run predictors.",
    )
    parser.add_argument(
        "--keep-intermediate-prefix",
        default="mimicneoai",
        help="Prefix for intermediate task and prediction directories.",
    )
    parser.add_argument("--mhcflurry-predict-bin", default=None)
    parser.add_argument("--mhcflurry-downloads-dir", default=None)
    parser.add_argument("--mhcnuggets-python-bin", default=None)
    parser.add_argument("--mhcnuggets-script", default=None)
    parser.add_argument("--mhcnuggets-cwd", default=None)
    parser.add_argument("--netmhcpan-bin", default=None)
    parser.add_argument("--netmhciipan-bin", default=None)
    parser.add_argument("--iedb-mhci-script", default=None)
    parser.add_argument("--iedb-mhci-cwd", default=None)
    parser.add_argument("--iedb-mhcii-script", default=None)
    parser.add_argument("--iedb-mhcii-cwd", default=None)
    parser.add_argument(
        "--iedb-mhci-python-bin",
        "--python-bin",
        dest="iedb_mhci_python_bin",
        default=None,
    )
    parser.add_argument("--iedb-mhcii-python-bin", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """Run the non-mutation binding workflow."""

    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy = resolve_binding_prediction_policy(args.preset)
    if policy:
        mhc_i_lengths = policy.mhc_i_lengths
        mhc_ii_lengths = policy.mhc_ii_lengths
        algorithms = list(policy.algorithms)
    else:
        mhc_i_lengths = parse_int_list(args.mhc_i_lengths)
        mhc_ii_lengths = parse_int_list(args.mhc_ii_lengths)
        algorithms = parse_str_list(args.algorithms)
    mhc_i_algorithms, mhc_ii_algorithms, unknown_algorithms = split_algorithms_by_mhc_class(algorithms)

    task_dir = outdir / f"{args.keep_intermediate_prefix}_epitope_tasks"
    pred_dir = outdir / f"{args.keep_intermediate_prefix}_binding_predictions"
    combined_dir = outdir / "combined"
    task_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    combined_dir.mkdir(parents=True, exist_ok=True)

    epitope_windows_path = task_dir / "epitope_windows.tsv"
    epitope_windows_manifest_path = task_dir / "epitope_windows.manifest.json"
    epitope_windows_signature = {
        "schema_version": EPITOPE_WINDOWS_SCHEMA_VERSION,
        "input_mode": args.input_mode,
        "peptide_fasta": file_identity(Path(args.pep_fasta)),
        "mhc_i_lengths": list(mhc_i_lengths),
        "mhc_ii_lengths": list(mhc_ii_lengths),
    }
    hla = parse_hlahd_result(Path(args.hla_file))

    if (
        epitope_windows_path.exists()
        and epitope_windows_path.stat().st_size > 0
        and manifest_signature_matches(epitope_windows_manifest_path, epitope_windows_signature)
    ):
        window_summary, mhc_i_peptides, mhc_ii_peptides = read_epitope_windows_summary(epitope_windows_path)
        epitope_windows_reused = True
    else:
        preflight_summary, preflight_mhc_i_peptides, preflight_mhc_ii_peptides = scan_epitope_windows_from_fasta(
            Path(args.pep_fasta),
            mhc_i_lengths,
            mhc_ii_lengths,
            input_mode=args.input_mode,
        )
        preflight_full_estimated_task_rows = estimate_binding_task_rows(
            len(preflight_mhc_i_peptides),
            len(hla.mhc_i),
            len(mhc_i_algorithms),
            len(preflight_mhc_ii_peptides),
            len(hla.mhc_ii),
            len(mhc_ii_algorithms),
        )
        if policy and policy.two_stage:
            preflight_estimated_task_rows = estimate_binding_task_rows(
                len(preflight_mhc_i_peptides),
                len(hla.mhc_i),
                1,
                len(preflight_mhc_ii_peptides),
                len(hla.mhc_ii),
                1,
            )
        else:
            preflight_estimated_task_rows = preflight_full_estimated_task_rows
        preflight_scale_gate = (
            preflight_estimated_task_rows > args.max_task_rows
            and not args.force_large_samples
            and not args.windows_only
        )
        if preflight_scale_gate:
            write_json(
                epitope_windows_manifest_path,
                {
                    "input_signature": epitope_windows_signature,
                    **preflight_summary,
                    "epitope_windows_materialized": False,
                    "skip_reason": (
                        f"estimated_binding_task_rows ({preflight_estimated_task_rows}) exceeds "
                        f"max_task_rows ({args.max_task_rows}); set --force-large-samples to run."
                    ),
                },
            )
            task_path = task_dir / "binding_tasks.tsv"
            task_manifest_path = task_dir / "binding_tasks.manifest.json"
            combined_dir.mkdir(parents=True, exist_ok=True)
            merged_out = combined_dir / f"{args.sample}.merged.all_epitopes.tsv"
            task_manifest = {
                "sample": args.sample,
                "binding_prediction_preset": policy.name if policy else "",
                "two_stage_binding_prediction": bool(policy and policy.two_stage),
                "estimated_binding_task_rows": preflight_estimated_task_rows,
                "full_estimated_binding_task_rows_without_stage1": preflight_full_estimated_task_rows,
                "binding_task_rows": 0,
                "max_task_rows": args.max_task_rows,
                "force_large_samples": args.force_large_samples,
                "task_table_materialized": False,
                "task_materialization_skipped_by_scale": True,
                "binding_tasks_path": str(task_path),
                "epitope_windows_materialized": False,
                "unique_mhc_i_peptides": len(preflight_mhc_i_peptides),
                "unique_mhc_ii_peptides": len(preflight_mhc_ii_peptides),
                "mhc_i_alleles": len(hla.mhc_i),
                "mhc_ii_alleles": len(hla.mhc_ii),
                "mhc_i_algorithms": mhc_i_algorithms,
                "mhc_ii_algorithms": mhc_ii_algorithms,
                "input_signature": {
                    "epitope_windows": {
                        "materialized": False,
                        "input_signature": epitope_windows_signature,
                    },
                    "hla_file": file_identity(Path(args.hla_file)),
                    "binding_prediction_preset": policy.name if policy else "",
                    "two_stage": bool(policy and policy.two_stage),
                },
            }
            write_json(task_manifest_path, task_manifest)
            skip_reason = (
                f"estimated_binding_task_rows ({preflight_estimated_task_rows}) exceeds max_task_rows "
                f"({args.max_task_rows}); set --force-large-samples to run prediction."
            )
            summary = {
                "sample": args.sample,
                "peptide_fasta": str(Path(args.pep_fasta)),
                "input_mode": args.input_mode,
                "hla_file": str(Path(args.hla_file)),
                "output_dir": str(outdir),
                "binding_prediction_preset": policy.name if policy else "",
                "two_stage_binding_prediction": bool(policy and policy.two_stage),
                "fasta_records": preflight_summary["fasta_records"],
                "epitope_window_rows": preflight_summary["epitope_window_rows"],
                "unique_mhc_i_peptides": len(preflight_mhc_i_peptides),
                "unique_mhc_ii_peptides": len(preflight_mhc_ii_peptides),
                "mhc_i_alleles": list(hla.mhc_i),
                "mhc_ii_alleles": list(hla.mhc_ii),
                "algorithms": algorithms,
                "mhc_i_algorithms": mhc_i_algorithms,
                "mhc_ii_algorithms": mhc_ii_algorithms,
                "unknown_algorithms": unknown_algorithms,
                "binding_task_rows": 0,
                "estimated_binding_task_rows": preflight_estimated_task_rows,
                "full_estimated_binding_task_rows_without_stage1": preflight_full_estimated_task_rows,
                "binding_tasks_manifest": str(task_manifest_path),
                "task_table_materialized": False,
                "task_materialization_skipped_by_scale": True,
                "algorithm_batches": [list(batch) for batch in build_algorithm_batches(mhc_i_algorithms + mhc_ii_algorithms)],
                "runner_invocations": [],
                "stage1_task_summary": {},
                "stage1_route_summary": {},
                "stage1_outputs": {},
                "max_runner_task_rows": args.max_runner_task_rows,
                "max_task_rows": args.max_task_rows,
                "force_large_samples": args.force_large_samples,
                "epitope_windows_reused": False,
                "epitope_windows_materialized": False,
                "preflight_scale_gate": True,
                "binding_tasks_reused": False,
                "windows_only": args.windows_only,
                "prediction_skipped": True,
                "prediction_skipped_by_scale": True,
                "skip_reason": skip_reason,
                "binding_qc_summary": {},
                "merged_output": str(merged_out),
                "merged_output_exists": False,
            }
            with (outdir / f"{args.sample}.mimicneoai_binding.summary.json").open("w") as handle:
                json.dump(summary, handle, indent=2, ensure_ascii=False)
            print(f"[nonmutation_binding] skip before writing epitope windows: {skip_reason}", flush=True)
            print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
            return 0
        window_summary, mhc_i_peptides, mhc_ii_peptides = write_epitope_windows_from_fasta(
            Path(args.pep_fasta),
            epitope_windows_path,
            mhc_i_lengths,
            mhc_ii_lengths,
            input_mode=args.input_mode,
        )
        write_json(
            epitope_windows_manifest_path,
            {"input_signature": epitope_windows_signature, **window_summary},
        )
        epitope_windows_reused = False

    task_path = task_dir / "binding_tasks.tsv"
    task_manifest_path = task_dir / "binding_tasks.manifest.json"
    runner_invocations: list[dict[str, object]] = []
    stage1_task_summary: dict[str, object] = {}
    stage1_route_summary: dict[str, object] = {}
    stage1_outputs: dict[str, str] = {}
    full_estimated_task_rows = estimate_binding_task_rows(
        len(mhc_i_peptides),
        len(hla.mhc_i),
        len(mhc_i_algorithms),
        len(mhc_ii_peptides),
        len(hla.mhc_ii),
        len(mhc_ii_algorithms),
    )

    if policy and policy.two_stage:
        stage1_task_path = task_dir / "stage1_binding_tasks.tsv"
        stage1_task_summary_path = task_dir / "stage1_binding_tasks.summary.json"
        stage1_prediction_path = pred_dir / "stage1_binding_predictions.long.tsv"
        stage1_pass_prediction_path = pred_dir / "stage1_pass.binding_predictions.long.tsv"
        stage1_routing_path = task_dir / "stage1_routing.tsv"
        stage1_unsupported_path = task_dir / "stage1_unsupported_or_failed.tsv"
        stage1_route_summary_path = task_dir / "stage1_route.summary.json"
        stage1_outputs = {
            "stage1_tasks": str(stage1_task_path),
            "stage1_task_summary": str(stage1_task_summary_path),
            "stage1_predictions": str(stage1_prediction_path),
            "stage1_pass_predictions": str(stage1_pass_prediction_path),
            "stage1_routing": str(stage1_routing_path),
            "stage1_unsupported_or_failed": str(stage1_unsupported_path),
            "stage1_route_summary": str(stage1_route_summary_path),
        }
        estimated_task_rows = estimate_binding_task_rows(
            len(mhc_i_peptides),
            len(hla.mhc_i),
            1,
            len(mhc_ii_peptides),
            len(hla.mhc_ii),
            1,
        )
        task_signature = {
            "epitope_windows": file_identity(epitope_windows_path),
            "hla_file": file_identity(Path(args.hla_file)),
            "binding_prediction_preset": policy.name,
            "two_stage": True,
            "stage1_algorithms": list(policy.stage1_algorithms),
            "stage2_algorithms": algorithms,
            "mhc_i_algorithms": mhc_i_algorithms,
            "mhc_ii_algorithms": mhc_ii_algorithms,
        }
        task_materialization_skipped_by_scale = (
            estimated_task_rows > args.max_task_rows
            and not args.force_large_samples
            and not args.windows_only
        )
        if args.windows_only:
            task_summary = {"binding_task_rows": 0, "estimated_binding_task_rows": estimated_task_rows}
            binding_tasks_reused = False
        elif task_materialization_skipped_by_scale:
            task_summary = {"binding_task_rows": 0, "estimated_binding_task_rows": estimated_task_rows}
            binding_tasks_reused = False
        elif (
            task_path.exists()
            and task_path.stat().st_size > 0
            and stage1_pass_prediction_path.exists()
            and stage1_pass_prediction_path.stat().st_size > 0
            and manifest_signature_matches(task_manifest_path, task_signature)
            and read_json(task_manifest_path).get("task_table_materialized") is True
        ):
            task_summary = read_binding_task_summary(task_path)
            task_summary["estimated_binding_task_rows"] = estimated_task_rows
            stage1_task_summary = read_json(stage1_task_summary_path)
            stage1_route_summary = read_json(stage1_route_summary_path)
            binding_tasks_reused = True
        elif args.skip_prediction:
            stage1_task_summary = write_stage1_tasks_from_nonmutation_windows(
                epitope_windows_path,
                Path(args.hla_file),
                policy,
                stage1_task_path,
            )
            write_json(stage1_task_summary_path, stage1_task_summary)
            task_summary = {"binding_task_rows": 0, "estimated_binding_task_rows": estimated_task_rows}
            binding_tasks_reused = False
        else:
            stage1_task_summary = write_stage1_tasks_from_nonmutation_windows(
                epitope_windows_path,
                Path(args.hla_file),
                policy,
                stage1_task_path,
            )
            write_json(stage1_task_summary_path, stage1_task_summary)
            stage1_prediction_paths = run_prediction_batches(
                stage1_task_path,
                list(policy.stage1_algorithms),
                pred_dir / "stage1",
                args,
                runner_invocations,
                netmhc_el_only=True,
            )
            concatenate_prediction_tables(stage1_prediction_paths, stage1_prediction_path)
            stage1_route_summary = route_stage2_tasks(
                stage1_tasks_path=stage1_task_path,
                stage1_predictions_path=stage1_prediction_path,
                policy=policy,
                stage2_algorithms=policy.algorithms,
                stage2_task_path=task_path,
                routing_path=stage1_routing_path,
                unsupported_path=stage1_unsupported_path,
                pass_predictions_path=stage1_pass_prediction_path,
            )
            write_json(stage1_route_summary_path, stage1_route_summary)
            task_summary = read_binding_task_summary(task_path)
            task_summary["estimated_binding_task_rows"] = estimated_task_rows
            binding_tasks_reused = False
    else:
        task_signature = {
            "epitope_windows": file_identity(epitope_windows_path),
            "hla_file": file_identity(Path(args.hla_file)),
            "binding_prediction_preset": policy.name if policy else "",
            "two_stage": False,
            "mhc_i_algorithms": mhc_i_algorithms,
            "mhc_ii_algorithms": mhc_ii_algorithms,
        }
        estimated_task_rows = full_estimated_task_rows
        task_materialization_skipped_by_scale = (
            estimated_task_rows > args.max_task_rows
            and not args.force_large_samples
            and not args.windows_only
        )
        if args.windows_only:
            task_summary = {"binding_task_rows": 0, "estimated_binding_task_rows": estimated_task_rows}
            binding_tasks_reused = False
        elif task_materialization_skipped_by_scale:
            task_summary = {"binding_task_rows": 0, "estimated_binding_task_rows": estimated_task_rows}
            binding_tasks_reused = False
        elif (
            task_path.exists()
            and task_path.stat().st_size > 0
            and manifest_signature_matches(task_manifest_path, task_signature)
            and read_json(task_manifest_path).get("task_table_materialized") is True
        ):
            task_summary = read_binding_task_summary(task_path)
            task_summary["estimated_binding_task_rows"] = estimated_task_rows
            binding_tasks_reused = True
        else:
            task_summary = write_binding_task_table(
                task_path,
                sorted(mhc_i_peptides),
                list(hla.mhc_i),
                mhc_i_algorithms,
                sorted(mhc_ii_peptides),
                list(hla.mhc_ii),
                mhc_ii_algorithms,
            )
            task_summary["estimated_binding_task_rows"] = estimated_task_rows
            binding_tasks_reused = False

    task_table_materialized = (
        not args.windows_only
        and not task_materialization_skipped_by_scale
        and not (policy and policy.two_stage and args.skip_prediction)
    )
    task_manifest = {
        "sample": args.sample,
        "binding_prediction_preset": policy.name if policy else "",
        "two_stage_binding_prediction": bool(policy and policy.two_stage),
        "estimated_binding_task_rows": estimated_task_rows,
        "full_estimated_binding_task_rows_without_stage1": full_estimated_task_rows,
        "binding_task_rows": int(task_summary["binding_task_rows"]),
        "max_task_rows": args.max_task_rows,
        "force_large_samples": args.force_large_samples,
        "task_table_materialized": task_table_materialized,
        "task_materialization_skipped_by_scale": task_materialization_skipped_by_scale,
        "binding_tasks_path": str(task_path),
        "epitope_windows_materialized": epitope_windows_path.exists(),
        "preexisting_binding_tasks_ignored": task_materialization_skipped_by_scale and task_path.exists(),
        "unique_mhc_i_peptides": len(mhc_i_peptides),
        "unique_mhc_ii_peptides": len(mhc_ii_peptides),
        "mhc_i_alleles": len(hla.mhc_i),
        "mhc_ii_alleles": len(hla.mhc_ii),
        "mhc_i_algorithms": mhc_i_algorithms,
        "mhc_ii_algorithms": mhc_ii_algorithms,
        "stage1": stage1_outputs,
        "input_signature": task_signature,
    }
    write_json(task_manifest_path, task_manifest)

    prediction_path = pred_dir / "binding_predictions.long.tsv"
    prediction_paths: list[Path] = []
    binding_qc_summary: dict[str, object] = {}
    binding_task_rows = int(task_summary["binding_task_rows"])
    prediction_skipped_by_scale = task_materialization_skipped_by_scale
    skip_reason = ""
    if (
        not args.windows_only
        and not args.skip_prediction
        and not prediction_skipped_by_scale
        and binding_task_rows > args.max_task_rows
        and not args.force_large_samples
    ):
        prediction_skipped_by_scale = True
        skip_reason = (
            f"stage2_binding_task_rows ({binding_task_rows}) exceeds max_task_rows "
            f"({args.max_task_rows}); set --force-large-samples to run prediction."
        )
    if args.windows_only:
        merged_out = combined_dir / f"{args.sample}.merged.all_epitopes.tsv"
    elif prediction_skipped_by_scale:
        merged_out = combined_dir / f"{args.sample}.merged.all_epitopes.tsv"
        if not skip_reason:
            skip_reason = (
                f"estimated_binding_task_rows ({estimated_task_rows}) exceeds max_task_rows "
                f"({args.max_task_rows}); set --force-large-samples to run prediction."
            )
        print(f"[nonmutation_binding] skip prediction: {skip_reason}", flush=True)
    elif not args.skip_prediction:
        if policy and policy.two_stage:
            stage1_pass_prediction_path = Path(stage1_outputs["stage1_pass_predictions"])
            if stage1_pass_prediction_path.exists():
                prediction_paths.append(stage1_pass_prediction_path)
            stage2_algorithms = [
                algorithm for algorithm in algorithms if algorithm not in set(policy.stage1_algorithms)
            ]
            prediction_paths.extend(
                run_prediction_batches(
                    task_path,
                    stage2_algorithms,
                    pred_dir / "stage2",
                    args,
                    runner_invocations,
                )
            )
        else:
            prediction_paths.extend(
                run_prediction_batches(
                    task_path,
                    mhc_i_algorithms + mhc_ii_algorithms,
                    pred_dir,
                    args,
                    runner_invocations,
                )
            )
        concatenate_prediction_tables(prediction_paths, prediction_path)
        merged_out = combined_dir / f"{args.sample}.merged.all_epitopes.tsv"
        merge_pvacbind_compatible(epitope_windows_path, prediction_path, task_path, merged_out)
        binding_qc_summary = build_binding_qc_summary(task_path, prediction_path)
    else:
        merged_out = combined_dir / f"{args.sample}.merged.all_epitopes.tsv"

    summary = {
        "sample": args.sample,
        "peptide_fasta": str(Path(args.pep_fasta)),
        "input_mode": args.input_mode,
        "hla_file": str(Path(args.hla_file)),
        "output_dir": str(outdir),
        "binding_prediction_preset": policy.name if policy else "",
        "two_stage_binding_prediction": bool(policy and policy.two_stage),
        "fasta_records": window_summary["fasta_records"],
        "epitope_window_rows": window_summary["epitope_window_rows"],
        "unique_mhc_i_peptides": len(mhc_i_peptides),
        "unique_mhc_ii_peptides": len(mhc_ii_peptides),
        "mhc_i_alleles": list(hla.mhc_i),
        "mhc_ii_alleles": list(hla.mhc_ii),
        "algorithms": algorithms,
        "mhc_i_algorithms": mhc_i_algorithms,
        "mhc_ii_algorithms": mhc_ii_algorithms,
        "unknown_algorithms": unknown_algorithms,
        "binding_task_rows": binding_task_rows,
        "estimated_binding_task_rows": task_summary["estimated_binding_task_rows"],
        "full_estimated_binding_task_rows_without_stage1": full_estimated_task_rows,
        "binding_tasks_manifest": str(task_manifest_path),
        "task_table_materialized": task_manifest["task_table_materialized"],
        "task_materialization_skipped_by_scale": task_materialization_skipped_by_scale,
        "algorithm_batches": [list(batch) for batch in build_algorithm_batches(mhc_i_algorithms + mhc_ii_algorithms)],
        "runner_invocations": runner_invocations,
        "stage1_task_summary": stage1_task_summary,
        "stage1_route_summary": stage1_route_summary,
        "stage1_outputs": stage1_outputs,
        "max_runner_task_rows": args.max_runner_task_rows,
        "max_task_rows": args.max_task_rows,
        "force_large_samples": args.force_large_samples,
        "epitope_windows_reused": epitope_windows_reused,
        "epitope_windows_materialized": epitope_windows_path.exists(),
        "preflight_scale_gate": False,
        "binding_tasks_reused": binding_tasks_reused,
        "windows_only": args.windows_only,
        "prediction_skipped": args.skip_prediction or args.windows_only or prediction_skipped_by_scale,
        "prediction_skipped_by_scale": prediction_skipped_by_scale,
        "skip_reason": skip_reason,
        "binding_qc_summary": binding_qc_summary,
        "merged_output": str(merged_out),
        "merged_output_exists": merged_out.exists(),
    }
    with (outdir / f"{args.sample}.mimicneoai_binding.summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item.strip()) for item in value.replace(",", " ").split() if item.strip()}))
    if not values:
        raise ValueError("At least one peptide length is required.")
    return values


def file_identity(path: Path) -> dict[str, object]:
    """Return a fast identity used to validate reusable intermediates."""

    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_json(path: Path) -> dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def manifest_signature_matches(path: Path, signature: dict[str, object]) -> bool:
    return read_json(path).get("input_signature") == signature


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def split_algorithms_by_mhc_class(algorithms: list[str]) -> tuple[list[str], list[str], list[str]]:
    mhc_i = [algorithm for algorithm in algorithms if algorithm in MHC_I_ALGORITHMS]
    mhc_ii = [algorithm for algorithm in algorithms if algorithm in MHC_II_ALGORITHMS]
    unknown = [algorithm for algorithm in algorithms if algorithm not in MHC_I_ALGORITHMS and algorithm not in MHC_II_ALGORITHMS]
    return mhc_i, mhc_ii, unknown


def build_algorithm_batches(algorithms: list[str]) -> list[tuple[str, ...]]:
    """Group algorithms to keep runner memory bounded while preserving shared commands."""

    requested = set(algorithms)
    preferred_batches = [
        ("MHCflurry", "MHCflurryEL"),
        ("MHCnuggetsI",),
        ("NetMHCpan", "NetMHCpanEL"),
        ("MHCnuggetsII",),
        ("NNalign",),
        ("NetMHCIIpan", "NetMHCIIpanEL"),
        ("PickPocket",),
        ("SMM",),
        ("SMMPMBEC",),
    ]
    batches: list[tuple[str, ...]] = []
    consumed: set[str] = set()
    for batch in preferred_batches:
        selected = tuple(algorithm for algorithm in batch if algorithm in requested)
        if selected:
            batches.append(selected)
            consumed.update(selected)
    for algorithm in sorted(requested.difference(consumed)):
        batches.append((algorithm,))
    return batches


def run_prediction_batches(
    task_path: Path,
    algorithms: list[str],
    prediction_root: Path,
    args: argparse.Namespace,
    runner_invocations: list[dict[str, object]],
    *,
    netmhc_el_only: bool = False,
) -> list[Path]:
    """Run runner shards for the selected algorithms and return prediction tables."""

    prediction_paths: list[Path] = []
    for batch in build_algorithm_batches(algorithms):
        batch_label = sanitize_batch_label(batch)
        task_shards = split_task_file_for_algorithm_batch(
            task_path,
            batch,
            prediction_root / "task_shards" / batch_label,
            args.max_runner_task_rows,
        )
        for shard_index, shard_task_path in enumerate(task_shards, start=1):
            shard_dir = prediction_root / batch_label / f"shard_{shard_index:04d}"
            runner_args = [
                "--tasks",
                str(shard_task_path),
                "-o",
                str(shard_dir),
                "--algorithms",
                ",".join(batch),
                "--workers",
                str(args.workers),
                "--device",
                args.device,
                "--gpu-id",
                args.gpu_id,
            ]
            if netmhc_el_only:
                runner_args.append("--netmhc-el-only")
            if args.chunk_size:
                runner_args.extend(["--chunk-size", str(args.chunk_size)])
            if args.command_timeout:
                runner_args.extend(["--command-timeout", str(args.command_timeout)])
            append_optional_runner_args(args, runner_args)
            runner_invocations.append(
                {
                    "batch": list(batch),
                    "task_file": str(shard_task_path),
                    "outdir": str(shard_dir),
                    "netmhc_el_only": netmhc_el_only,
                }
            )
            run_binding_predictions(runner_args)
            batch_prediction_path = shard_dir / "binding_predictions.long.tsv"
            if batch_prediction_path.exists():
                prediction_paths.append(batch_prediction_path)
    return prediction_paths


def sanitize_batch_label(batch: tuple[str, ...]) -> str:
    return "__".join(batch).replace("/", "_")


def read_fasta(path: Path) -> Iterable[tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    header: Optional[str] = None
    chunks: list[str] = []
    with opener(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield (
                        fasta_record_id(header),
                        normalize_peptide_sequence("".join(chunks)),
                    )
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield (
                fasta_record_id(header),
                normalize_peptide_sequence("".join(chunks)),
            )


def fasta_record_id(header: str) -> str:
    """Return the stable record identifier from a FASTA description line."""

    fields = header.strip().split()
    if not fields:
        raise ValueError("FASTA header does not contain a record identifier")
    return fields[0]


def normalize_peptide_sequence(sequence: str) -> str:
    return sequence.replace("*", "").replace(" ", "").strip().upper()


def scan_epitope_windows_from_fasta(
    fasta_path: Path,
    mhc_i_lengths: tuple[int, ...],
    mhc_ii_lengths: tuple[int, ...],
    input_mode: str = "parent-fasta",
) -> tuple[dict[str, int], set[str], set[str]]:
    """Count candidate windows and unique peptides without writing the window table."""

    if input_mode == "peptide-core":
        return scan_epitope_windows_from_peptide_core(fasta_path, mhc_i_lengths, mhc_ii_lengths)
    if input_mode != "parent-fasta":
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    mhc_i_peptides: set[str] = set()
    mhc_ii_peptides: set[str] = set()
    fasta_records = 0
    epitope_window_rows = 0
    for _header, sequence in read_fasta(fasta_path):
        fasta_records += 1
        if not sequence:
            continue
        for length in mhc_i_lengths:
            if len(sequence) < length:
                continue
            for start in range(0, len(sequence) - length + 1):
                mhc_i_peptides.add(sequence[start : start + length])
                epitope_window_rows += 1
        for length in mhc_ii_lengths:
            if len(sequence) < length:
                continue
            for start in range(0, len(sequence) - length + 1):
                mhc_ii_peptides.add(sequence[start : start + length])
                epitope_window_rows += 1
    return (
        {"fasta_records": fasta_records, "epitope_window_rows": epitope_window_rows},
        mhc_i_peptides,
        mhc_ii_peptides,
    )


def write_epitope_windows_from_fasta(
    fasta_path: Path,
    output_path: Path,
    mhc_i_lengths: tuple[int, ...],
    mhc_ii_lengths: tuple[int, ...],
    input_mode: str = "parent-fasta",
) -> tuple[dict[str, int], set[str], set[str]]:
    if input_mode == "peptide-core":
        return write_epitope_windows_from_peptide_core(
            fasta_path,
            output_path,
            mhc_i_lengths,
            mhc_ii_lengths,
        )
    if input_mode != "parent-fasta":
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mhc_i_peptides: set[str] = set()
    mhc_ii_peptides: set[str] = set()
    fasta_records = 0
    epitope_window_rows = 0

    fieldnames = [
        "Mutation",
        "HLA Allele",
        "Sub-peptide Position",
        "Epitope Seq",
        "Peptide Length",
        "mhc_class",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for header, sequence in read_fasta(fasta_path):
            fasta_records += 1
            if not sequence:
                continue
            for length in mhc_i_lengths:
                for row in iter_window_rows(header, sequence, length, "MHC-I"):
                    writer.writerow(row)
                    mhc_i_peptides.add(row["Epitope Seq"])
                    epitope_window_rows += 1
            for length in mhc_ii_lengths:
                for row in iter_window_rows(header, sequence, length, "MHC-II"):
                    writer.writerow(row)
                    mhc_ii_peptides.add(row["Epitope Seq"])
                    epitope_window_rows += 1
    return (
        {"fasta_records": fasta_records, "epitope_window_rows": epitope_window_rows},
        mhc_i_peptides,
        mhc_ii_peptides,
    )


def infer_peptide_core_mhc_class(
    record_id: str,
    sequence: str,
    mhc_i_lengths: tuple[int, ...],
    mhc_ii_lengths: tuple[int, ...],
) -> str:
    """Infer MHC class for a pre-tiled peptide record from its length."""

    length = len(sequence)
    is_mhc_i = length in mhc_i_lengths
    is_mhc_ii = length in mhc_ii_lengths
    if is_mhc_i and is_mhc_ii:
        raise ValueError(
            f"Peptide-core FASTA record {record_id!r} has ambiguous length {length}; "
            "MHC-I and MHC-II length sets must not overlap."
        )
    if is_mhc_i:
        return "MHC-I"
    if is_mhc_ii:
        return "MHC-II"
    raise ValueError(
        f"Peptide-core FASTA record {record_id!r} has length {length}, which is not in "
        f"MHC-I lengths {list(mhc_i_lengths)} or MHC-II lengths {list(mhc_ii_lengths)}."
    )


def iter_peptide_core_rows(
    fasta_path: Path,
    mhc_i_lengths: tuple[int, ...],
    mhc_ii_lengths: tuple[int, ...],
) -> Iterable[dict[str, str]]:
    """Yield one epitope-window row per already tiled peptide FASTA record."""

    for record_id, sequence in read_fasta(fasta_path):
        if not sequence:
            continue
        mhc_class = infer_peptide_core_mhc_class(record_id, sequence, mhc_i_lengths, mhc_ii_lengths)
        yield {
            "Mutation": record_id,
            "HLA Allele": "",
            "Sub-peptide Position": "1",
            "Epitope Seq": sequence,
            "Peptide Length": str(len(sequence)),
            "mhc_class": mhc_class,
        }


def scan_epitope_windows_from_peptide_core(
    fasta_path: Path,
    mhc_i_lengths: tuple[int, ...],
    mhc_ii_lengths: tuple[int, ...],
) -> tuple[dict[str, int], set[str], set[str]]:
    mhc_i_peptides: set[str] = set()
    mhc_ii_peptides: set[str] = set()
    fasta_records = 0
    epitope_window_rows = 0
    for row in iter_peptide_core_rows(fasta_path, mhc_i_lengths, mhc_ii_lengths):
        fasta_records += 1
        peptide = row["Epitope Seq"]
        if row["mhc_class"] == "MHC-I":
            mhc_i_peptides.add(peptide)
        else:
            mhc_ii_peptides.add(peptide)
        epitope_window_rows += 1
    return (
        {"fasta_records": fasta_records, "epitope_window_rows": epitope_window_rows},
        mhc_i_peptides,
        mhc_ii_peptides,
    )


def write_epitope_windows_from_peptide_core(
    fasta_path: Path,
    output_path: Path,
    mhc_i_lengths: tuple[int, ...],
    mhc_ii_lengths: tuple[int, ...],
) -> tuple[dict[str, int], set[str], set[str]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mhc_i_peptides: set[str] = set()
    mhc_ii_peptides: set[str] = set()
    fasta_records = 0
    epitope_window_rows = 0
    fieldnames = [
        "Mutation",
        "HLA Allele",
        "Sub-peptide Position",
        "Epitope Seq",
        "Peptide Length",
        "mhc_class",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in iter_peptide_core_rows(fasta_path, mhc_i_lengths, mhc_ii_lengths):
            writer.writerow(row)
            fasta_records += 1
            peptide = row["Epitope Seq"]
            if row["mhc_class"] == "MHC-I":
                mhc_i_peptides.add(peptide)
            else:
                mhc_ii_peptides.add(peptide)
            epitope_window_rows += 1
    return (
        {"fasta_records": fasta_records, "epitope_window_rows": epitope_window_rows},
        mhc_i_peptides,
        mhc_ii_peptides,
    )


def read_epitope_windows_summary(path: Path) -> tuple[dict[str, int], set[str], set[str]]:
    """Read a previously generated epitope window table and recover task inputs."""

    mhc_i_peptides: set[str] = set()
    mhc_ii_peptides: set[str] = set()
    mutation_ids: set[str] = set()
    rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Mutation", "Epitope Seq", "mhc_class"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            mutation = row.get("Mutation") or ""
            if mutation:
                mutation_ids.add(mutation)
            peptide = (row.get("Epitope Seq") or "").strip().upper()
            if not peptide:
                continue
            mhc_class = row.get("mhc_class") or ""
            if mhc_class == "MHC-I":
                mhc_i_peptides.add(peptide)
            elif mhc_class == "MHC-II":
                mhc_ii_peptides.add(peptide)
    return (
        {"fasta_records": len(mutation_ids), "epitope_window_rows": rows},
        mhc_i_peptides,
        mhc_ii_peptides,
    )


def build_epitope_windows(
    fasta_records: list[tuple[str, str]],
    mhc_i_lengths: tuple[int, ...],
    mhc_ii_lengths: tuple[int, ...],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for header, sequence in fasta_records:
        if not sequence:
            continue
        for length in mhc_i_lengths:
            rows.extend(iter_window_rows(header, sequence, length, "MHC-I"))
        for length in mhc_ii_lengths:
            rows.extend(iter_window_rows(header, sequence, length, "MHC-II"))
    return rows


def iter_window_rows(header: str, sequence: str, length: int, mhc_class: str) -> Iterable[dict[str, str]]:
    if length <= 0 or len(sequence) < length:
        return
    for start in range(0, len(sequence) - length + 1):
        peptide = sequence[start : start + length]
        yield {
            "Mutation": header,
            "HLA Allele": "",
            "Sub-peptide Position": str(start + 1),
            "Epitope Seq": peptide,
            "Peptide Length": str(length),
            "mhc_class": mhc_class,
        }


def write_binding_task_table(
    path: Path,
    mhc_i_peptides: list[str],
    mhc_i_hlas: list[str],
    mhc_i_algorithms: list[str],
    mhc_ii_peptides: list[str],
    mhc_ii_hlas: list[str],
    mhc_ii_algorithms: list[str],
) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=["peptide", "hla_allele", "algorithm", "mhc_class"])
        writer.writeheader()
        rows += write_task_product(writer, mhc_i_peptides, mhc_i_hlas, mhc_i_algorithms, "MHC-I")
        rows += write_task_product(writer, mhc_ii_peptides, mhc_ii_hlas, mhc_ii_algorithms, "MHC-II")
    return {"binding_task_rows": rows}


def read_binding_task_summary(path: Path) -> dict[str, int]:
    """Count rows in a previously generated binding task table."""

    rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"peptide", "hla_allele", "algorithm", "mhc_class"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        for _ in reader:
            rows += 1
    return {"binding_task_rows": rows}


def estimate_binding_task_rows(
    mhc_i_peptides: int,
    mhc_i_hlas: int,
    mhc_i_algorithms: int,
    mhc_ii_peptides: int,
    mhc_ii_hlas: int,
    mhc_ii_algorithms: int,
) -> int:
    return (mhc_i_peptides * mhc_i_hlas * mhc_i_algorithms) + (
        mhc_ii_peptides * mhc_ii_hlas * mhc_ii_algorithms
    )


def split_task_file_for_algorithm_batch(
    task_path: Path,
    algorithms: tuple[str, ...],
    outdir: Path,
    max_rows: int,
) -> list[Path]:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    outdir.mkdir(parents=True, exist_ok=True)
    algorithm_set = set(algorithms)
    output_paths: list[Path] = []
    fieldnames = ["peptide", "hla_allele", "algorithm", "mhc_class"]
    shard_handle = None
    writer = None
    rows_in_shard = 0
    shard_index = 0

    def close_current() -> None:
        nonlocal shard_handle
        if shard_handle is not None:
            shard_handle.close()
            shard_handle = None

    try:
        with task_path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("algorithm") not in algorithm_set:
                    continue
                if shard_handle is None or rows_in_shard >= max_rows:
                    close_current()
                    shard_index += 1
                    shard_path = outdir / f"binding_tasks.shard_{shard_index:04d}.tsv"
                    output_paths.append(shard_path)
                    shard_handle = shard_path.open("w", newline="")
                    writer = csv.DictWriter(shard_handle, delimiter="\t", fieldnames=fieldnames)
                    writer.writeheader()
                    rows_in_shard = 0
                assert writer is not None
                writer.writerow({key: row.get(key, "") for key in fieldnames})
                rows_in_shard += 1
    finally:
        close_current()
    return output_paths


def write_task_product(
    writer: csv.DictWriter,
    peptides: list[str],
    hla_alleles: list[str],
    algorithms: list[str],
    mhc_class: str,
) -> int:
    rows = 0
    for hla_allele in sorted(set(hla_alleles)):
        for peptide in peptides:
            for algorithm in sorted(set(algorithms)):
                if not peptide or not hla_allele or not algorithm:
                    continue
                writer.writerow(
                    {
                        "peptide": peptide,
                        "hla_allele": hla_allele,
                        "algorithm": algorithm,
                        "mhc_class": mhc_class,
                    }
                )
                rows += 1
    return rows


def merge_pvacbind_compatible(
    epitope_windows_path: Path,
    prediction_path: Path,
    task_path: Path,
    output_path: Path,
) -> None:
    predictions, prediction_summary = read_prediction_index(prediction_path)
    hla_by_class_length, algorithms_raw, task_keys, task_rows = collect_task_metadata(task_path)
    algorithms = ordered_algorithms(algorithms_raw)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    metric_counts: Counter[str] = Counter()
    with epitope_windows_path.open(newline="") as ep_handle, output_path.open("w", newline="") as handle:
        reader = csv.DictReader(ep_handle, delimiter="\t")
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=PVACBIND_COMPAT_COLUMNS)
        writer.writeheader()
        for epitope in reader:
            mhc_class = epitope["mhc_class"]
            peptide_length = int(epitope["Peptide Length"])
            peptide = (epitope["Epitope Seq"] or "").strip().upper()
            for hla_allele in hla_by_class_length.get((mhc_class, peptide_length), []):
                if (peptide, hla_allele, mhc_class, peptide_length) not in task_keys:
                    continue
                row, metrics = build_pvacbind_row(epitope, hla_allele, algorithms, predictions)
                writer.writerow(row)
                row_count += 1
                metric_counts.update(metrics)

    summary = {
        "output": str(output_path),
        "rows": row_count,
        "algorithms": algorithms,
        "task_rows": task_rows,
        "task_key_count": len(task_keys),
        "prediction_summary": prediction_summary,
        "metric_counts": dict(metric_counts),
    }
    with output_path.with_suffix(output_path.suffix + ".summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[nonmutation_binding] wrote {output_path}", flush=True)


def concatenate_prediction_tables(paths: list[Path], output_path: Path) -> None:
    """Concatenate batch-level normalized prediction tables."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as out_handle:
        out_handle.write("\t".join(PREDICTION_FIELDS) + "\n")
        for path in paths:
            if not path.exists() or path.stat().st_size == 0:
                continue
            with path.open(newline="") as in_handle:
                header = in_handle.readline()
                if not header:
                    continue
                fields = header.rstrip("\r\n").split("\t")
                missing = set(PREDICTION_FIELDS).difference(fields)
                if missing:
                    raise ValueError(
                        f"{path} missing required prediction columns: {sorted(missing)}"
                    )
                for line in in_handle:
                    out_handle.write(line)


def read_prediction_index(path: Path) -> tuple[dict[tuple[str, str, str, str, int], dict[str, str]], dict[str, object]]:
    index: dict[tuple[str, str, str, str, int], dict[str, str]] = {}
    status_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    rows = 0
    if not path.exists():
        return index, {"rows": 0, "status_counts": {}, "algorithm_counts": {}}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows += 1
            status = row.get("status", "")
            algorithm = row.get("algorithm", "")
            status_counts[status] += 1
            algorithm_counts[algorithm] += 1
            if status not in {"ok", "partial_ok"}:
                continue
            peptide = (row.get("peptide") or "").strip().upper()
            hla_allele = (row.get("hla_allele") or "").strip()
            mhc_class = (row.get("mhc_class") or "").strip()
            peptide_length = safe_int(row.get("peptide_length"))
            if not peptide or not hla_allele or peptide_length is None:
                continue
            key = (peptide, hla_allele, algorithm, mhc_class, peptide_length)
            existing = index.get(key)
            if existing is None or prediction_quality(row) < prediction_quality(existing):
                index[key] = row
    return index, {
        "rows": rows,
        "usable_prediction_keys": len(index),
        "status_counts": dict(status_counts),
        "algorithm_counts": dict(algorithm_counts),
    }


def prediction_quality(row: dict[str, str]) -> tuple[int, int, int]:
    status_score = 0 if row.get("status") == "ok" else 1
    missing_ic50 = 0 if safe_float(row.get("ic50")) is not None else 1
    missing_percentile = 0 if safe_float(row.get("percentile")) is not None else 1
    return (status_score, missing_ic50, missing_percentile)


def collect_task_metadata(task_path: Path) -> tuple[dict[tuple[str, int], list[str]], set[str], set[tuple[str, str, str, int]], int]:
    values: dict[tuple[str, int], set[str]] = defaultdict(set)
    algorithms: set[str] = set()
    task_keys: set[tuple[str, str, str, int]] = set()
    rows = 0
    with task_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            rows += 1
            peptide = (row.get("peptide") or "").strip().upper()
            if not peptide:
                continue
            mhc_class = (row.get("mhc_class") or "").strip()
            hla_allele = (row.get("hla_allele") or "").strip()
            algorithm = (row.get("algorithm") or "").strip()
            values[(mhc_class, len(peptide))].add(hla_allele)
            algorithms.add(algorithm)
            task_keys.add((peptide, hla_allele, mhc_class, len(peptide)))
    return {key: sorted(value) for key, value in values.items()}, algorithms, task_keys, rows


def ordered_algorithms(algorithms: set[str]) -> list[str]:
    preferred = [
        "MHCflurry",
        "MHCflurryEL",
        "MHCnuggetsI",
        "NetMHC",
        "NetMHCpan",
        "NetMHCpanEL",
        "PickPocket",
        "SMM",
        "SMMPMBEC",
        "MHCnuggetsII",
        "NNalign",
        "NetMHCIIpan",
        "NetMHCIIpanEL",
    ]
    known = [algorithm for algorithm in preferred if algorithm in algorithms]
    return known + sorted(algorithms.difference(preferred))


def build_pvacbind_row(
    epitope: dict[str, str],
    hla_allele: str,
    algorithms: list[str],
    predictions: dict[tuple[str, str, str, str, int], dict[str, str]],
) -> tuple[dict[str, str], Counter[str]]:
    peptide = epitope["Epitope Seq"]
    mhc_class = epitope["mhc_class"]
    peptide_length = int(epitope["Peptide Length"])
    row = {field: "" for field in PVACBIND_COMPAT_COLUMNS}
    row.update(
        {
            "Mutation": epitope["Mutation"],
            "HLA Allele": hla_allele,
            "Sub-peptide Position": epitope["Sub-peptide Position"],
            "Epitope Seq": peptide,
        }
    )

    ic50_by_algorithm: dict[str, float] = {}
    percentile_by_algorithm: dict[str, float] = {}
    metrics: Counter[str] = Counter()
    for algorithm in algorithms:
        prediction = predictions.get((peptide, hla_allele, algorithm, mhc_class, peptide_length))
        if not prediction:
            continue
        fill_algorithm_columns(row, algorithm, prediction)
        ic50 = safe_float(prediction.get("ic50"))
        percentile = safe_float(prediction.get("percentile"))
        if ic50 is not None and algorithm in IC50_SUMMARY_ALGORITHMS:
            ic50_by_algorithm[algorithm] = ic50
        if percentile is not None:
            percentile_by_algorithm[algorithm] = percentile
        metrics[f"{algorithm}_present"] += 1

    if ic50_by_algorithm:
        best_method = min(ic50_by_algorithm, key=ic50_by_algorithm.get)
        row["Best IC50 Score Method"] = best_method
        row["Best IC50 Score"] = fmt_float(ic50_by_algorithm[best_method])
        row["Median IC50 Score"] = fmt_float(median(ic50_by_algorithm.values()))
        metrics["rows_with_ic50"] += 1
    if percentile_by_algorithm:
        best_method = min(percentile_by_algorithm, key=percentile_by_algorithm.get)
        row["Best Percentile Method"] = best_method
        row["Best Percentile"] = fmt_float(percentile_by_algorithm[best_method])
        row["Median Percentile"] = fmt_float(median(percentile_by_algorithm.values()))
        metrics["rows_with_percentile"] += 1
    return row, metrics


def fill_algorithm_columns(row: dict[str, str], algorithm: str, prediction: dict[str, str]) -> None:
    if algorithm == "MHCflurryEL":
        row["MHCflurryEL Presentation Score"] = prediction.get("score", "")
        row["MHCflurryEL Presentation Percentile"] = prediction.get("percentile", "")
        return
    if algorithm in {"NetMHCpanEL", "NetMHCIIpanEL"}:
        row[f"{algorithm} Score"] = prediction.get("score", "")
        row[f"{algorithm} Percentile"] = prediction.get("percentile", "")
        return
    ic50_key = f"{algorithm} IC50 Score"
    pct_key = f"{algorithm} Percentile"
    if ic50_key in row:
        row[ic50_key] = prediction.get("ic50", "")
    if pct_key in row:
        row[pct_key] = prediction.get("percentile", "")


def append_optional_runner_args(args: argparse.Namespace, runner_args: list[str]) -> None:
    mapping = {
        "mhcflurry_predict_bin": "--mhcflurry-predict-bin",
        "mhcflurry_downloads_dir": "--mhcflurry-downloads-dir",
        "mhcnuggets_python_bin": "--mhcnuggets-python-bin",
        "mhcnuggets_script": "--mhcnuggets-script",
        "mhcnuggets_cwd": "--mhcnuggets-cwd",
        "netmhcpan_bin": "--netmhcpan-bin",
        "netmhciipan_bin": "--netmhciipan-bin",
        "iedb_mhci_script": "--iedb-mhci-script",
        "iedb_mhci_cwd": "--iedb-mhci-cwd",
        "iedb_mhcii_script": "--iedb-mhcii-script",
        "iedb_mhcii_cwd": "--iedb-mhcii-cwd",
        "iedb_mhci_python_bin": "--python-bin",
        "iedb_mhcii_python_bin": "--iedb-mhcii-python-bin",
    }
    for attr, cli_name in mapping.items():
        value = getattr(args, attr)
        if value:
            runner_args.extend([cli_name, value])


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_int(value: object) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except ValueError:
        return None


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
