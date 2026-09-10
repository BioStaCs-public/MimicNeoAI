"""Stage 1 EL-rank routing for local binding prediction."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from mimicneoai.functions.binding_prediction.policy import (
    BindingPredictionPolicy,
    parse_algorithm_list,
    resolve_binding_prediction_policy,
)
from mimicneoai.functions.binding_prediction.schema import PREDICTION_FIELDS, safe_float
from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.hla_parser import (
    parse_hlahd_result,
)


TASK_FIELDS = ["peptide", "hla_allele", "algorithm", "mhc_class"]
STAGE1_ROUTING_FIELDS = [
    "peptide",
    "hla_allele",
    "mhc_class",
    "peptide_length",
    "stage1_algorithm",
    "stage1_percentile",
    "stage1_status",
    "prediction_status",
    "error",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_nonmutation = subparsers.add_parser("build-nonmutation")
    build_nonmutation.add_argument("--epitope-windows", required=True)
    build_nonmutation.add_argument("--hla-file", required=True)
    build_nonmutation.add_argument("--preset", default="fast")
    build_nonmutation.add_argument("-o", "--output", required=True)
    build_nonmutation.add_argument("--summary", default=None)

    build_mutation = subparsers.add_parser("build-mutation")
    build_mutation.add_argument("--prediction-peptides", required=True)
    build_mutation.add_argument("--hla-file", required=True)
    build_mutation.add_argument("--preset", default="fast")
    build_mutation.add_argument("-o", "--output", required=True)
    build_mutation.add_argument("--summary", default=None)

    route = subparsers.add_parser("route")
    route.add_argument("--stage1-tasks", required=True)
    route.add_argument("--stage1-predictions", required=True)
    route.add_argument("--preset", default="fast")
    route.add_argument("--stage2-algorithms", default="")
    route.add_argument("-o", "--output", required=True, help="Stage 2 task table.")
    route.add_argument("--routing-output", required=True)
    route.add_argument("--unsupported-output", required=True)
    route.add_argument("--pass-predictions-output", required=True)
    route.add_argument("--summary", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    policy = resolve_binding_prediction_policy(args.preset)
    if policy is None or not policy.two_stage:
        raise ValueError("Stage 1 routing requires a two-stage preset such as 'fast'.")

    if args.command == "build-nonmutation":
        summary = write_stage1_tasks_from_nonmutation_windows(
            Path(args.epitope_windows),
            Path(args.hla_file),
            policy,
            Path(args.output),
        )
        write_json(Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json"), summary)
    elif args.command == "build-mutation":
        summary = write_stage1_tasks_from_mutation_prediction_peptides(
            Path(args.prediction_peptides),
            Path(args.hla_file),
            policy,
            Path(args.output),
        )
        write_json(Path(args.summary) if args.summary else Path(args.output).with_suffix(".summary.json"), summary)
    elif args.command == "route":
        stage2_algorithms = (
            parse_algorithm_list(args.stage2_algorithms)
            if args.stage2_algorithms.strip()
            else policy.algorithms
        )
        summary = route_stage2_tasks(
            stage1_tasks_path=Path(args.stage1_tasks),
            stage1_predictions_path=Path(args.stage1_predictions),
            policy=policy,
            stage2_algorithms=stage2_algorithms,
            stage2_task_path=Path(args.output),
            routing_path=Path(args.routing_output),
            unsupported_path=Path(args.unsupported_output),
            pass_predictions_path=Path(args.pass_predictions_output),
        )
        write_json(Path(args.summary), summary)
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"Unsupported command: {args.command}")
    return 0


def write_stage1_tasks_from_nonmutation_windows(
    epitope_windows_path: Path,
    hla_file: Path,
    policy: BindingPredictionPolicy,
    output_path: Path,
) -> dict[str, object]:
    peptides_by_class = collect_nonmutation_peptides(epitope_windows_path)
    return write_stage1_task_products(peptides_by_class, hla_file, policy, output_path)


def write_stage1_tasks_from_mutation_prediction_peptides(
    prediction_peptides_path: Path,
    hla_file: Path,
    policy: BindingPredictionPolicy,
    output_path: Path,
) -> dict[str, object]:
    peptides_by_class = collect_mutation_mt_peptides(prediction_peptides_path)
    return write_stage1_task_products(peptides_by_class, hla_file, policy, output_path)


def collect_nonmutation_peptides(epitope_windows_path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {"MHC-I": set(), "MHC-II": set()}
    with epitope_windows_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Epitope Seq", "mhc_class"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{epitope_windows_path} missing required columns: {sorted(missing)}")
        for row in reader:
            peptide = (row.get("Epitope Seq") or "").strip().upper()
            mhc_class = (row.get("mhc_class") or "").strip()
            if peptide and mhc_class in result:
                result[mhc_class].add(peptide)
    return result


def collect_mutation_mt_peptides(prediction_peptides_path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {"MHC-I": set(), "MHC-II": set()}
    with prediction_peptides_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"peptide", "mhc_class", "source_types"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{prediction_peptides_path} missing required columns: {sorted(missing)}")
        for row in reader:
            source_types = {item.strip() for item in (row.get("source_types") or "").split(",") if item.strip()}
            if "MT" not in source_types:
                continue
            peptide = (row.get("peptide") or "").strip().upper()
            mhc_class = (row.get("mhc_class") or "").strip()
            if peptide and mhc_class in result:
                result[mhc_class].add(peptide)
    return result


def write_stage1_task_products(
    peptides_by_class: dict[str, set[str]],
    hla_file: Path,
    policy: BindingPredictionPolicy,
    output_path: Path,
) -> dict[str, object]:
    hla = parse_hlahd_result(hla_file)
    rows = []
    rows.extend(
        build_task_rows(
            sorted(peptides_by_class.get("MHC-I", set())),
            list(hla.mhc_i),
            [policy.stage1_mhc_i_algorithm],
            "MHC-I",
        )
    )
    rows.extend(
        build_task_rows(
            sorted(peptides_by_class.get("MHC-II", set())),
            list(hla.mhc_ii),
            [policy.stage1_mhc_ii_algorithm],
            "MHC-II",
        )
    )
    write_task_tsv(output_path, rows)
    return {
        "preset": policy.name,
        "stage1_mhc_i_algorithm": policy.stage1_mhc_i_algorithm,
        "stage1_mhc_ii_algorithm": policy.stage1_mhc_ii_algorithm,
        "stage1_mhc_i_rank_lt": policy.stage1_mhc_i_rank_lt,
        "stage1_mhc_ii_rank_lt": policy.stage1_mhc_ii_rank_lt,
        "unique_mhc_i_peptides": len(peptides_by_class.get("MHC-I", set())),
        "unique_mhc_ii_peptides": len(peptides_by_class.get("MHC-II", set())),
        "mhc_i_alleles": list(hla.mhc_i),
        "mhc_ii_alleles": list(hla.mhc_ii),
        "stage1_task_rows": len(rows),
        "stage1_tasks": str(output_path),
    }


def build_task_rows(
    peptides: list[str],
    hla_alleles: list[str],
    algorithms: list[str],
    mhc_class: str,
) -> list[dict[str, str]]:
    rows = []
    for hla_allele in sorted(set(hla_alleles)):
        for peptide in peptides:
            for algorithm in algorithms:
                rows.append(
                    {
                        "peptide": peptide,
                        "hla_allele": hla_allele,
                        "algorithm": algorithm,
                        "mhc_class": mhc_class,
                    }
                )
    return rows


def route_stage2_tasks(
    stage1_tasks_path: Path,
    stage1_predictions_path: Path,
    policy: BindingPredictionPolicy,
    stage2_algorithms: tuple[str, ...],
    stage2_task_path: Path,
    routing_path: Path,
    unsupported_path: Path,
    pass_predictions_path: Path,
) -> dict[str, object]:
    stage1_tasks = read_task_rows(stage1_tasks_path)
    predictions = read_prediction_index(stage1_predictions_path)
    stage2_algorithms_by_class = split_stage2_algorithms(stage2_algorithms)

    routing_rows: list[dict[str, str]] = []
    unsupported_rows: list[dict[str, str]] = []
    pass_prediction_keys: set[tuple[str, str, str, str, int]] = set()
    stage2_rows: list[dict[str, str]] = []
    status_counts: Counter[str] = Counter()

    for task in stage1_tasks:
        key = task_key(task)
        prediction = predictions.get(key)
        stage1_status, percentile, prediction_status, error = classify_stage1_prediction(task, prediction, policy)
        route_row = {
            "peptide": task["peptide"],
            "hla_allele": task["hla_allele"],
            "mhc_class": task["mhc_class"],
            "peptide_length": str(len(task["peptide"])),
            "stage1_algorithm": task["algorithm"],
            "stage1_percentile": "" if percentile is None else f"{percentile:.6g}",
            "stage1_status": stage1_status,
            "prediction_status": prediction_status,
            "error": error,
        }
        routing_rows.append(route_row)
        status_counts[stage1_status] += 1
        if stage1_status not in {"stage1_screen_pass", "screened_out_stage1"}:
            unsupported_rows.append(route_row)
        if stage1_status != "stage1_screen_pass":
            continue
        if prediction is not None:
            pass_prediction_keys.add(key)
        for algorithm in stage2_algorithms_by_class.get(task["mhc_class"], []):
            stage2_rows.append(
                {
                    "peptide": task["peptide"],
                    "hla_allele": task["hla_allele"],
                    "algorithm": algorithm,
                    "mhc_class": task["mhc_class"],
                }
            )

    stage2_rows = unique_task_rows(stage2_rows)
    write_task_tsv(stage2_task_path, stage2_rows)
    write_dict_tsv(routing_path, STAGE1_ROUTING_FIELDS, routing_rows)
    write_dict_tsv(unsupported_path, STAGE1_ROUTING_FIELDS, unsupported_rows)
    write_prediction_subset(stage1_predictions_path, pass_predictions_path, pass_prediction_keys)
    return {
        "preset": policy.name,
        "stage1_tasks": str(stage1_tasks_path),
        "stage1_predictions": str(stage1_predictions_path),
        "stage1_task_rows": len(stage1_tasks),
        "stage1_status_counts": dict(status_counts),
        "stage2_task_rows": len(stage2_rows),
        "stage2_algorithms": list(stage2_algorithms),
        "stage1_pass_predictions": str(pass_predictions_path),
        "routing_output": str(routing_path),
        "unsupported_output": str(unsupported_path),
    }


def split_stage2_algorithms(algorithms: tuple[str, ...]) -> dict[str, list[str]]:
    mhc_i = {
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
    mhc_ii = {"MHCnuggetsII", "NNalign", "NetMHCIIpan", "NetMHCIIpanEL"}
    return {
        "MHC-I": [algorithm for algorithm in algorithms if algorithm in mhc_i],
        "MHC-II": [algorithm for algorithm in algorithms if algorithm in mhc_ii],
    }


def classify_stage1_prediction(
    task: dict[str, str],
    prediction: Optional[dict[str, str]],
    policy: BindingPredictionPolicy,
) -> tuple[str, Optional[float], str, str]:
    if prediction is None:
        return "prediction_missing", None, "", ""
    prediction_status = (prediction.get("status") or "").strip()
    error = prediction.get("error") or ""
    if prediction_status == "skipped":
        return "unsupported_allele", None, prediction_status, error
    if prediction_status not in {"ok", "partial_ok"}:
        status = "parse_failed" if "parse" in error.lower() else "task_failed"
        return status, None, prediction_status, error
    percentile = safe_float(prediction.get("percentile"))
    if percentile is None:
        return "prediction_missing", None, prediction_status, error
    threshold = (
        policy.stage1_mhc_i_rank_lt
        if task["mhc_class"] == "MHC-I"
        else policy.stage1_mhc_ii_rank_lt
    )
    if percentile < threshold:
        return "stage1_screen_pass", percentile, prediction_status, error
    return "screened_out_stage1", percentile, prediction_status, error


def read_task_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = set(TASK_FIELDS)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        return [
            {field: (row.get(field) or "").strip() for field in TASK_FIELDS}
            for row in reader
            if (row.get("peptide") or "").strip()
        ]


def read_prediction_index(path: Path) -> dict[tuple[str, str, str, str, int], dict[str, str]]:
    index: dict[tuple[str, str, str, str, int], dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            peptide = (row.get("peptide") or "").strip().upper()
            algorithm = (row.get("algorithm") or "").strip()
            hla_allele = (row.get("hla_allele") or "").strip()
            mhc_class = (row.get("mhc_class") or "").strip()
            if not peptide or not algorithm or not hla_allele or not mhc_class:
                continue
            index[(peptide, hla_allele, algorithm, mhc_class, len(peptide))] = row
    return index


def task_key(task: dict[str, str]) -> tuple[str, str, str, str, int]:
    peptide = task["peptide"].strip().upper()
    return (peptide, task["hla_allele"], task["algorithm"], task["mhc_class"], len(peptide))


def unique_task_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique = []
    for row in rows:
        key = tuple(row[field] for field in TASK_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return sorted(unique, key=lambda row: tuple(row[field] for field in TASK_FIELDS))


def write_task_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    write_dict_tsv(path, TASK_FIELDS, unique_task_rows(rows))


def write_dict_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def write_prediction_subset(
    prediction_path: Path,
    output_path: Path,
    keep_keys: set[tuple[str, str, str, str, int]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open(newline="") as in_handle, output_path.open("w", newline="") as out_handle:
        reader = csv.DictReader(in_handle, delimiter="\t")
        writer = csv.DictWriter(out_handle, delimiter="\t", fieldnames=PREDICTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            peptide = (row.get("peptide") or "").strip().upper()
            key = (
                peptide,
                (row.get("hla_allele") or "").strip(),
                (row.get("algorithm") or "").strip(),
                (row.get("mhc_class") or "").strip(),
                len(peptide),
            )
            if key in keep_keys:
                writer.writerow({field: row.get(field, "") for field in PREDICTION_FIELDS})


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
