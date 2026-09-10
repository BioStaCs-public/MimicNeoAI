"""Run local HLA binding predictors on a de-duplicated task table."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from mimicneoai.functions.binding_prediction.adapters import (
    SUPPORTED_ALGORITHMS,
    AdapterConfig,
    adapter_for_algorithm,
)
from mimicneoai.functions.binding_prediction.allele_support import AlleleSupportMatrix
from mimicneoai.functions.binding_prediction.qc import build_binding_qc_summary
from mimicneoai.functions.binding_prediction.schema import (
    PREDICTION_FIELDS,
    BindingTask,
    PredictionJob,
    chunked,
    read_binding_tasks,
    read_prediction_rows,
)
from mimicneoai.functions.nodemon_pool import NoDaemonPool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="TSV with peptide, hla_allele, algorithm, mhc_class.")
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--algorithms", default="", help="Optional comma-separated algorithm subset.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--tf-intra-op-threads", type=int, default=1)
    parser.add_argument("--tf-inter-op-threads", type=int, default=1)
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=None,
        help="Optional per-predictor subprocess timeout in seconds. Timed-out chunks are written as error rows.",
    )
    parser.add_argument("--mhcflurry-predict-bin", default=AdapterConfig.mhcflurry_predict_bin)
    parser.add_argument(
        "--mhcflurry-downloads-dir",
        default=AdapterConfig.mhcflurry_downloads_dir,
    )
    parser.add_argument("--mhcnuggets-python-bin", default=AdapterConfig.mhcnuggets_python_bin)
    parser.add_argument("--mhcnuggets-script", default=AdapterConfig.mhcnuggets_script)
    parser.add_argument("--mhcnuggets-cwd", default=AdapterConfig.mhcnuggets_cwd)
    parser.add_argument(
        "--mhcnuggets-rank-output",
        action="store_true",
        default=True,
        help="Ask MHCnuggets to additionally compute human proteome percentile ranks.",
    )
    parser.add_argument(
        "--no-mhcnuggets-rank-output",
        action="store_false",
        dest="mhcnuggets_rank_output",
        help="Disable optional MHCnuggets human proteome percentile ranks.",
    )
    parser.add_argument("--netmhcpan-bin", default=AdapterConfig.netmhcpan_bin)
    parser.add_argument("--netmhciipan-bin", default=AdapterConfig.netmhciipan_bin)
    parser.add_argument(
        "--netmhc-el-only",
        action="store_true",
        help="Run NetMHCpan/NetMHCIIpan without BA output when only EL algorithms are requested.",
    )
    parser.add_argument("--iedb-mhci-script", default=AdapterConfig.iedb_mhci_script)
    parser.add_argument("--iedb-mhci-cwd", default=AdapterConfig.iedb_mhci_cwd)
    parser.add_argument("--iedb-mhcii-script", default=AdapterConfig.iedb_mhcii_script)
    parser.add_argument("--iedb-mhcii-cwd", default=AdapterConfig.iedb_mhcii_cwd)
    parser.add_argument(
        "--iedb-mhci-python-bin",
        "--python-bin",
        dest="python_bin",
        default=AdapterConfig.python_bin,
    )
    parser.add_argument("--iedb-mhcii-python-bin", default=AdapterConfig.iedb_mhcii_python_bin)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    algorithm_filter = parse_algorithm_filter(args.algorithms)
    tasks = read_binding_tasks(Path(args.tasks), algorithms=algorithm_filter)
    config = AdapterConfig(
        mhcflurry_predict_bin=args.mhcflurry_predict_bin,
        mhcflurry_downloads_dir=args.mhcflurry_downloads_dir,
        mhcnuggets_python_bin=args.mhcnuggets_python_bin,
        mhcnuggets_script=args.mhcnuggets_script,
        mhcnuggets_cwd=args.mhcnuggets_cwd,
        mhcnuggets_rank_output=args.mhcnuggets_rank_output,
        netmhcpan_bin=args.netmhcpan_bin,
        netmhciipan_bin=args.netmhciipan_bin,
        netmhc_el_only=args.netmhc_el_only,
        iedb_mhci_script=args.iedb_mhci_script,
        iedb_mhci_cwd=args.iedb_mhci_cwd,
        iedb_mhcii_script=args.iedb_mhcii_script,
        iedb_mhcii_cwd=args.iedb_mhcii_cwd,
        python_bin=args.python_bin,
        iedb_mhcii_python_bin=args.iedb_mhcii_python_bin,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
        device=args.device,
        gpu_id=args.gpu_id,
        tf_intra_op_threads=args.tf_intra_op_threads,
        tf_inter_op_threads=args.tf_inter_op_threads,
        command_timeout=args.command_timeout,
    )

    allele_support = AlleleSupportMatrix(config)
    runnable_tasks, skipped_tasks = filter_unsupported_tasks(tasks, allele_support)
    validate_predictor_runtime(runnable_tasks, config)
    jobs = build_jobs(runnable_tasks, outdir / "chunks", config)
    print(
        f"[binding_prediction] tasks={len(tasks)} runnable={len(runnable_tasks)} "
        f"skipped={len(skipped_tasks)} jobs={len(jobs)} workers={args.workers}",
        flush=True,
    )

    job_payloads = [(job, config) for job in jobs]
    if args.workers <= 1:
        normalized_paths = [run_job_payload(payload) for payload in job_payloads]
    else:
        with NoDaemonPool(processes=args.workers) as pool:
            normalized_paths = pool.map(run_job_payload, job_payloads)

    final_path = outdir / "binding_predictions.long.tsv"
    summary_path = outdir / "binding_predictions.summary.json"
    summary = merge_predictions(normalized_paths, final_path, skipped_tasks)
    summary["qc_summary"] = build_binding_qc_summary(Path(args.tasks), final_path)
    summary.update(
        {
            "task_rows": len(tasks),
            "runnable_task_rows": len(runnable_tasks),
            "skipped_task_rows": len(skipped_tasks),
            "skipped_task_counts": summarize_skipped_tasks(skipped_tasks),
            "job_count": len(jobs),
            "workers": args.workers,
            "device": args.device,
            "gpu_id": args.gpu_id if args.device == "gpu" else "",
            "mhcnuggets_rank_output": args.mhcnuggets_rank_output,
            "netmhc_el_only": args.netmhc_el_only,
            "command_timeout": args.command_timeout,
            "algorithms": sorted({task.algorithm for task in tasks}),
            "supported_algorithms": SUPPORTED_ALGORITHMS,
            "allele_support_matrix": allele_support.summary(),
            "predictor_runtime": predictor_runtime_summary(config),
        }
    )
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[binding_prediction] wrote {final_path}", flush=True)
    print(f"[binding_prediction] wrote {summary_path}", flush=True)
    qc_summary = summary.get("qc_summary", {})
    if prediction_run_failed(len(runnable_tasks), qc_summary):
        print(
            "[binding_prediction] no usable predictions were produced for runnable tasks",
            flush=True,
        )
        return 2
    return 0


def prediction_run_failed(
    runnable_task_rows: int, qc_summary: dict[str, object]
) -> bool:
    """Return whether execution produced no usable result for runnable tasks."""

    return runnable_task_rows > 0 and int(qc_summary.get("usable_result_rows", 0)) == 0


def validate_predictor_runtime(
    tasks: list[BindingTask], config: AdapterConfig
) -> None:
    """Fail early when a requested predictor runtime cannot start."""

    algorithms = {task.algorithm for task in tasks}
    if "NNalign" not in algorithms:
        return

    command = [
        config.iedb_mhcii_python_bin,
        config.iedb_mhcii_script,
        "method",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=config.iedb_mhcii_cwd,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "IEDB MHC-II runtime validation failed for NNalign: "
            f"{exc}. Configure --iedb-mhcii-python-bin with the dedicated "
            "IEDB Python environment."
        ) from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "no diagnostic output").strip()
        raise RuntimeError(
            "IEDB MHC-II runtime validation failed for NNalign "
            f"(exit code {result.returncode}): {detail[:2000]}"
        )


def parse_algorithm_filter(value: str) -> Optional[set[str]]:
    if not value.strip():
        return None
    return {item.strip() for item in value.replace(",", " ").split() if item.strip()}


def predictor_runtime_summary(config: AdapterConfig) -> dict[str, object]:
    """Record the configured predictor executables used by this runner."""

    return {
        "mhcflurry_predict_bin": config.mhcflurry_predict_bin,
        "mhcflurry_downloads_dir": config.mhcflurry_downloads_dir,
        "mhcnuggets_python_bin": config.mhcnuggets_python_bin,
        "mhcnuggets_script": config.mhcnuggets_script,
        "mhcnuggets_cwd": config.mhcnuggets_cwd,
        "netmhcpan_bin": config.netmhcpan_bin,
        "netmhciipan_bin": config.netmhciipan_bin,
        "netmhc_el_only": config.netmhc_el_only,
        "iedb_mhci_script": config.iedb_mhci_script,
        "iedb_mhci_cwd": config.iedb_mhci_cwd,
        "iedb_mhcii_script": config.iedb_mhcii_script,
        "iedb_mhcii_cwd": config.iedb_mhcii_cwd,
        "python_bin": config.python_bin,
        "iedb_mhcii_python_bin": config.iedb_mhcii_python_bin,
    }


def filter_unsupported_tasks(
    tasks: list[BindingTask],
    allele_support: Optional[AlleleSupportMatrix] = None,
) -> tuple[list[BindingTask], list[tuple[BindingTask, str]]]:
    allele_support = allele_support or AlleleSupportMatrix(AdapterConfig())
    runnable: list[BindingTask] = []
    skipped: list[tuple[BindingTask, str]] = []
    for task in tasks:
        reason = unsupported_reason(task, allele_support)
        if reason:
            skipped.append((task, reason))
        else:
            runnable.append(task)
    return runnable, skipped


def unsupported_reason(
    task: BindingTask,
    allele_support: Optional[AlleleSupportMatrix] = None,
) -> str:
    allele_support = allele_support or AlleleSupportMatrix(AdapterConfig())
    if allele_support.supports(task) is False:
        return "unsupported_allele_by_predictor"
    return ""


def summarize_skipped_tasks(skipped_tasks: list[tuple[BindingTask, str]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for task, reason in skipped_tasks:
        counts[f"{task.algorithm}|{task.hla_allele}|{reason}"] += 1
    return dict(counts)


def build_jobs(tasks: list[BindingTask], outdir: Path, config: AdapterConfig) -> list[PredictionJob]:
    peptide_algorithm_map: dict[tuple[str, str, str, int, str], set[str]] = defaultdict(set)
    for task in tasks:
        adapter_for_algorithm(task.algorithm, config)
        family = algorithm_family(task.algorithm)
        peptide_algorithm_map[
            (family, task.mhc_class, task.hla_allele, task.peptide_length, task.peptide)
        ].add(task.algorithm)

    grouped: dict[tuple[str, str, str, int, tuple[str, ...]], set[str]] = defaultdict(set)
    for (family, mhc_class, hla_allele, peptide_length, peptide), algorithms in peptide_algorithm_map.items():
        run_algorithm = select_run_algorithm(family, algorithms)
        grouped[(run_algorithm, mhc_class, hla_allele, peptide_length, tuple(sorted(algorithms)))].add(peptide)

    jobs: list[PredictionJob] = []
    for (algorithm, mhc_class, hla_allele, peptide_length, output_algorithms), peptides in sorted(grouped.items()):
        adapter = adapter_for_algorithm(algorithm, config)
        for chunk_index, chunk in chunked(sorted(peptides), adapter.chunk_size()):
            jobs.append(
                PredictionJob(
                    algorithm=algorithm,
                    mhc_class=mhc_class,
                    hla_allele=hla_allele,
                    peptide_length=peptide_length,
                    chunk_index=chunk_index,
                    peptides=chunk,
                    outdir=str(outdir),
                    output_algorithms=output_algorithms,
                )
            )
    return jobs


def algorithm_family(algorithm: str) -> str:
    if algorithm in {"NetMHCpan", "NetMHCpanEL"}:
        return "NetMHCpan"
    if algorithm in {"NetMHCIIpan", "NetMHCIIpanEL"}:
        return "NetMHCIIpan"
    if algorithm in {"MHCflurry", "MHCflurryEL"}:
        return "MHCflurry"
    return algorithm


def select_run_algorithm(family: str, algorithms: set[str]) -> str:
    """Choose the minimal local command needed to produce requested algorithm output."""

    if family == "NetMHCpan":
        return "NetMHCpan"
    if family == "NetMHCIIpan":
        return "NetMHCIIpan"
    if family == "MHCflurry":
        return "MHCflurryEL" if "MHCflurryEL" in algorithms else "MHCflurry"
    return family


def run_job_payload(payload: tuple[PredictionJob, AdapterConfig]) -> str:
    job, config = payload
    adapter = adapter_for_algorithm(job.algorithm, config)
    return str(adapter.run_job(job))


def merge_predictions(
    paths: list[str],
    output_path: Path,
    skipped_tasks: Optional[list[tuple[BindingTask, str]]] = None,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    row_count = 0
    with output_path.open("w", newline="") as handle:
        handle.write("\t".join(PREDICTION_FIELDS) + "\n")
        for path_string in sorted(set(paths)):
            path = Path(path_string)
            if not path.exists():
                continue
            for row in read_prediction_rows(path):
                row_count += 1
                status_counts[row.get("status", "")] += 1
                algorithm_counts[row.get("algorithm", "")] += 1
                handle.write("\t".join(row.get(field, "") for field in PREDICTION_FIELDS) + "\n")
        for task, reason in skipped_tasks or []:
            row_count += 1
            status_counts["skipped"] += 1
            algorithm_counts[task.algorithm] += 1
            row = {
                "peptide": task.peptide,
                "hla_allele": task.hla_allele,
                "algorithm": task.algorithm,
                "mhc_class": task.mhc_class,
                "peptide_length": str(task.peptide_length),
                "status": "skipped",
                "error": reason,
            }
            handle.write("\t".join(row.get(field, "") for field in PREDICTION_FIELDS) + "\n")
    return {
        "prediction_rows": row_count,
        "status_counts": dict(status_counts),
        "algorithm_counts": dict(algorithm_counts),
    }


if __name__ == "__main__":
    raise SystemExit(main())
