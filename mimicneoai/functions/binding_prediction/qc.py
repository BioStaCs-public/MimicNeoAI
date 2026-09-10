"""Informational quality summary for normalized binding predictions."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from mimicneoai.functions.binding_prediction.schema import safe_float


PRIMARY_METRIC = {
    "MHCflurry": "ic50",
    "MHCflurryEL": "percentile",
    "MHCnuggetsI": "ic50",
    "MHCnuggetsII": "ic50",
    "NNalign": "ic50",
    "NetMHCpan": "ic50",
    "NetMHCpanEL": "percentile",
    "NetMHCIIpan": "ic50",
    "NetMHCIIpanEL": "percentile",
    "PickPocket": "ic50",
    "SMM": "ic50",
    "SMMPMBEC": "ic50",
}

USABLE_STATUSES = frozenset({"ok", "partial_ok"})


def build_binding_qc_summary(task_path: Path, prediction_path: Path) -> dict[str, object]:
    """Summarize coverage and output completeness without enforcing thresholds."""

    task_counts: Counter[str] = Counter()
    requested_hla: dict[str, set[str]] = defaultdict(set)
    with task_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_columns(task_path, reader.fieldnames, {"algorithm", "hla_allele"})
        for row in reader:
            algorithm = (row.get("algorithm") or "").strip()
            hla_allele = (row.get("hla_allele") or "").strip()
            if not algorithm:
                continue
            task_counts[algorithm] += 1
            if hla_allele:
                requested_hla[algorithm].add(hla_allele)

    prediction_counts: Counter[str] = Counter()
    status_by_algorithm: dict[str, Counter[str]] = defaultdict(Counter)
    supported_hla: dict[str, set[str]] = defaultdict(set)
    metric_counts: dict[str, Counter[str]] = defaultdict(Counter)
    with prediction_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        _require_columns(
            prediction_path,
            reader.fieldnames,
            {"algorithm", "hla_allele", "status", "ic50", "percentile"},
        )
        for row in reader:
            algorithm = (row.get("algorithm") or "").strip()
            status = (row.get("status") or "").strip()
            hla_allele = (row.get("hla_allele") or "").strip()
            if not algorithm:
                continue
            prediction_counts[algorithm] += 1
            status_by_algorithm[algorithm][status] += 1
            if status != "skipped" and hla_allele:
                supported_hla[algorithm].add(hla_allele)
            if status in USABLE_STATUSES:
                metric_counts[algorithm]["usable"] += 1
                if safe_float(row.get("ic50")) is not None:
                    metric_counts[algorithm]["ic50"] += 1
                if safe_float(row.get("percentile")) is not None:
                    metric_counts[algorithm]["percentile"] += 1

    algorithms = sorted(set(task_counts) | set(prediction_counts))
    algorithm_qc = {
        algorithm: _algorithm_summary(
            algorithm,
            task_counts,
            prediction_counts,
            status_by_algorithm,
            requested_hla,
            supported_hla,
            metric_counts,
        )
        for algorithm in algorithms
    }

    overall_status = Counter()
    for counts in status_by_algorithm.values():
        overall_status.update(counts)
    usable_rows = sum(overall_status[status] for status in USABLE_STATUSES)
    error_rows = overall_status["error"]
    runnable_result_rows = usable_rows + error_rows
    requested_algorithm_hla = {
        (algorithm, allele)
        for algorithm, alleles in requested_hla.items()
        for allele in alleles
    }
    supported_algorithm_hla = {
        (algorithm, allele)
        for algorithm, alleles in supported_hla.items()
        for allele in alleles
    }

    task_rows = sum(task_counts.values())
    prediction_rows = sum(prediction_counts.values())
    return {
        "informational_only": True,
        "task_rows": task_rows,
        "prediction_rows": prediction_rows,
        "result_coverage_rate": _ratio(prediction_rows, task_rows),
        "missing_result_rows": max(task_rows - prediction_rows, 0),
        "extra_result_rows": max(prediction_rows - task_rows, 0),
        "status_counts": dict(sorted(overall_status.items())),
        "runnable_result_rows": runnable_result_rows,
        "usable_result_rows": usable_rows,
        "execution_success_rate": _ratio(usable_rows, runnable_result_rows),
        "requested_algorithm_hla_pairs": len(requested_algorithm_hla),
        "supported_algorithm_hla_pairs": len(supported_algorithm_hla),
        "algorithm_hla_support_rate": _ratio(
            len(supported_algorithm_hla), len(requested_algorithm_hla)
        ),
        "algorithms": algorithm_qc,
        "definitions": {
            "usable_statuses": sorted(USABLE_STATUSES),
            "execution_success_rate": "(ok + partial_ok) / (ok + partial_ok + error)",
            "hla_support_rate": "supported requested HLA alleles / requested HLA alleles, calculated per algorithm",
            "primary_metric_completeness_rate": "usable rows containing the algorithm's primary metric / usable rows",
        },
    }


def _algorithm_summary(
    algorithm: str,
    task_counts: Counter[str],
    prediction_counts: Counter[str],
    status_by_algorithm: dict[str, Counter[str]],
    requested_hla: dict[str, set[str]],
    supported_hla: dict[str, set[str]],
    metric_counts: dict[str, Counter[str]],
) -> dict[str, object]:
    statuses = status_by_algorithm[algorithm]
    usable_rows = sum(statuses[status] for status in USABLE_STATUSES)
    runnable_rows = usable_rows + statuses["error"]
    requested = requested_hla[algorithm]
    supported = supported_hla[algorithm]
    primary_metric = PRIMARY_METRIC.get(algorithm, "unspecified")
    primary_complete = (
        metric_counts[algorithm][primary_metric]
        if primary_metric in {"ic50", "percentile"}
        else None
    )
    return {
        "requested_task_rows": task_counts[algorithm],
        "prediction_rows": prediction_counts[algorithm],
        "result_coverage_rate": _ratio(
            prediction_counts[algorithm], task_counts[algorithm]
        ),
        "status_counts": dict(sorted(statuses.items())),
        "runnable_result_rows": runnable_rows,
        "usable_result_rows": usable_rows,
        "execution_success_rate": _ratio(usable_rows, runnable_rows),
        "requested_hla_count": len(requested),
        "supported_hla_count": len(supported),
        "hla_support_rate": _ratio(len(supported), len(requested)),
        "unsupported_hla_alleles": sorted(requested - supported),
        "ic50_complete_rows": metric_counts[algorithm]["ic50"],
        "percentile_complete_rows": metric_counts[algorithm]["percentile"],
        "primary_metric": primary_metric,
        "primary_metric_complete_rows": primary_complete,
        "primary_metric_completeness_rate": (
            _ratio(primary_complete, usable_rows)
            if primary_complete is not None
            else None
        ),
    }


def _ratio(numerator: Optional[int], denominator: int) -> Optional[float]:
    if numerator is None or denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _require_columns(path: Path, fieldnames, required: set[str]) -> None:
    missing = required.difference(fieldnames or [])
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
