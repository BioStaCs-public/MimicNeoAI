"""Step 02: merge local binding predictions into a downstream-compatible wide table.

This script joins:

- variant_annotation.tsv
- epitope_windows.tsv
- one or more binding_predictions.long.tsv files
- the binding task table used to define HLA/algorithm coverage

It recomputes MT/WT algorithm columns, WT/MT fold change, and Best/Median
summary metrics from local prediction results. Unsupported or failed predictor
rows remain traceable in the summary but are excluded from Best/Median metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Optional


PREFERRED_ALGORITHMS = [
    "MHCflurry",
    "MHCflurryEL",
    "MHCnuggetsI",
    "NetMHCpan",
    "NetMHCpanEL",
    "SMM",
    "SMMPMBEC",
    "MHCnuggetsII",
    "NNalign",
    "NetMHCIIpan",
    "NetMHCIIpanEL",
]

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

PVACSEQ_COLUMNS = [
    "Chromosome",
    "Start",
    "Stop",
    "Reference",
    "Variant",
    "Transcript",
    "Transcript Support Level",
    "Transcript Length",
    "Biotype",
    "Ensembl Gene ID",
    "Variant Type",
    "Mutation",
    "Protein Position",
    "Gene Name",
    "HGVSc",
    "HGVSp",
    "HLA Allele",
    "Peptide Length",
    "Sub-peptide Position",
    "Mutation Position",
    "MT Epitope Seq",
    "WT Epitope Seq",
    "Best MT IC50 Score Method",
    "Best MT IC50 Score",
    "Corresponding WT IC50 Score",
    "Corresponding Fold Change",
    "Best MT Percentile Method",
    "Best MT Percentile",
    "Corresponding WT Percentile",
    "Tumor DNA Depth",
    "Tumor DNA VAF",
    "Tumor RNA Depth",
    "Tumor RNA VAF",
    "Normal Depth",
    "Normal VAF",
    "Gene Expression",
    "Transcript Expression",
    "Median MT IC50 Score",
    "Median WT IC50 Score",
    "Median Fold Change",
    "Median MT Percentile",
    "Median WT Percentile",
    "BigMHC_EL WT Score",
    "BigMHC_EL MT Score",
    "BigMHC_IM WT Score",
    "BigMHC_IM MT Score",
    "MHCflurryEL Processing WT Score",
    "MHCflurryEL Processing MT Score",
    "MHCflurryEL Presentation WT Score",
    "MHCflurryEL Presentation MT Score",
    "MHCflurryEL Presentation WT Percentile",
    "MHCflurryEL Presentation MT Percentile",
    "MHCflurry WT IC50 Score",
    "MHCflurry MT IC50 Score",
    "MHCflurry WT Percentile",
    "MHCflurry MT Percentile",
    "MHCnuggetsI WT IC50 Score",
    "MHCnuggetsI MT IC50 Score",
    "MHCnuggetsI WT Percentile",
    "MHCnuggetsI MT Percentile",
    "NetMHC WT IC50 Score",
    "NetMHC MT IC50 Score",
    "NetMHC WT Percentile",
    "NetMHC MT Percentile",
    "NetMHCpan WT IC50 Score",
    "NetMHCpan MT IC50 Score",
    "NetMHCpan WT Percentile",
    "NetMHCpan MT Percentile",
    "NetMHCpanEL WT Score",
    "NetMHCpanEL MT Score",
    "NetMHCpanEL WT Percentile",
    "NetMHCpanEL MT Percentile",
    "PickPocket WT IC50 Score",
    "PickPocket MT IC50 Score",
    "PickPocket WT Percentile",
    "PickPocket MT Percentile",
    "SMM WT IC50 Score",
    "SMM MT IC50 Score",
    "SMM WT Percentile",
    "SMM MT Percentile",
    "SMMPMBEC WT IC50 Score",
    "SMMPMBEC MT IC50 Score",
    "SMMPMBEC WT Percentile",
    "SMMPMBEC MT Percentile",
    "Index",
    "DeepImmuno WT Score",
    "DeepImmuno MT Score",
    "cterm_7mer_gravy_score",
    "max_7mer_gravy_score",
    "difficult_n_terminal_residue",
    "c_terminal_cysteine",
    "c_terminal_proline",
    "cysteine_count",
    "n_terminal_asparagine",
    "asparagine_proline_bond_count",
    "MHCnuggetsII WT IC50 Score",
    "MHCnuggetsII MT IC50 Score",
    "MHCnuggetsII WT Percentile",
    "MHCnuggetsII MT Percentile",
    "NetMHCIIpan WT IC50 Score",
    "NetMHCIIpan MT IC50 Score",
    "NetMHCIIpan WT Percentile",
    "NetMHCIIpan MT Percentile",
    "NetMHCIIpanEL WT Score",
    "NetMHCIIpanEL MT Score",
    "NetMHCIIpanEL WT Percentile",
    "NetMHCIIpanEL MT Percentile",
    "NNalign WT IC50 Score",
    "NNalign MT IC50 Score",
    "NNalign WT Percentile",
    "NNalign MT Percentile",
]

EXTENDED_CORE_COLUMNS = PVACSEQ_COLUMNS[:42] + [
    "Extended MT Epitope Seq",
    "Extended Length",
    "WT Control Status",
    "WT Control Reason",
    "Event ID",
    "pVACseq Index",
    "Index",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-annotation", required=True)
    parser.add_argument("--epitope-windows", required=True)
    parser.add_argument(
        "--binding-predictions",
        required=True,
        action="append",
        help="One or more binding_predictions.long.tsv files. Repeat this argument for MT and WT outputs.",
    )
    parser.add_argument("--binding-tasks", required=True, help="Task table defining HLA and algorithm coverage.")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument(
        "--output-profile",
        choices=["pvacseq", "extended"],
        default="pvacseq",
        help="pvacseq writes the legacy pVACseq-compatible 111-column schema; extended keeps local debug/status columns.",
    )
    parser.add_argument("--pvacseq-merged", default=None, help="Optional original pVACseq merged TSV for validation.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    summary_path = Path(args.summary) if args.summary else output.with_suffix(output.suffix + ".summary.json")

    tasks = read_tasks(Path(args.binding_tasks))
    algorithms = ordered_algorithms(tasks["algorithms"])
    prediction_index, prediction_summary = read_predictions([Path(p) for p in args.binding_predictions])
    annotations = read_annotation(Path(args.variant_annotation))
    min_window_start = collect_min_window_starts(Path(args.epitope_windows))

    output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = output_columns(args.output_profile, algorithms)

    row_count = 0
    metric_counts: Counter[str] = Counter()
    with Path(args.epitope_windows).open(newline="") as ep_handle, output.open("w", newline="") as out_handle:
        reader = csv.DictReader(ep_handle, delimiter="\t")
        writer = csv.DictWriter(out_handle, delimiter="\t", fieldnames=output_fields)
        writer.writeheader()
        for epitope in reader:
            mhc_class = epitope["mhc_class"]
            peptide_length = int(epitope["peptide_length"])
            hla_alleles = tasks["hla_by_class_length"].get((mhc_class, peptide_length), [])
            if not hla_alleles:
                continue
            mt_peptide = (epitope["mt_epitope_seq"] or "").strip()
            annotation = annotations.get(epitope["pvacseq_index"], {})
            for hla_allele in hla_alleles:
                if (mt_peptide, hla_allele, mhc_class, peptide_length) not in tasks["task_keys"]:
                    continue
                row, row_metrics = build_output_row(
                    epitope=epitope,
                    annotation=annotation,
                    hla_allele=hla_allele,
                    algorithms=algorithms,
                    prediction_index=prediction_index,
                    min_window_start=min_window_start,
                    output_fields=output_fields,
                )
                writer.writerow(row)
                row_count += 1
                metric_counts.update(row_metrics)

    summary: dict[str, object] = {
        "output": str(output),
        "rows": row_count,
        "algorithms": algorithms,
        "output_profile": args.output_profile,
        "task_summary": summarize_tasks(tasks),
        "prediction_summary": prediction_summary,
        "metric_counts": dict(metric_counts),
    }
    if args.pvacseq_merged:
        summary["pvacseq_validation"] = validate_against_pvacseq(
            Path(args.pvacseq_merged),
            output,
            tasks["hla_by_class_length"],
        )
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[merge_binding_predictions] wrote {output}")
    print(f"[merge_binding_predictions] wrote {summary_path}")
    return 0


def read_tasks(path: Path) -> dict[str, object]:
    hla_by_class_length: dict[tuple[str, int], set[str]] = defaultdict(set)
    algorithms: set[str] = set()
    task_keys: set[tuple[str, str, str, int]] = set()
    rows = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            peptide = row["peptide"].strip()
            if not peptide:
                continue
            mhc_class = row["mhc_class"].strip()
            algorithm = row["algorithm"].strip()
            hla_allele = row["hla_allele"].strip()
            hla_by_class_length[(mhc_class, len(peptide))].add(hla_allele)
            algorithms.add(algorithm)
            task_keys.add((peptide, hla_allele, mhc_class, len(peptide)))
            rows += 1
    return {
        "rows": rows,
        "algorithms": algorithms,
        "hla_by_class_length": {key: sorted(value) for key, value in hla_by_class_length.items()},
        "task_keys": task_keys,
    }


def summarize_tasks(tasks: dict[str, object]) -> dict[str, object]:
    hla_by_class_length = tasks["hla_by_class_length"]
    return {
        "rows": tasks["rows"],
        "algorithms": ordered_algorithms(tasks["algorithms"]),
        "task_key_count": len(tasks["task_keys"]),
        "hla_by_class_length": {
            f"{key[0]}:{key[1]}": value
            for key, value in sorted(hla_by_class_length.items(), key=lambda item: (item[0][0], item[0][1]))
        },
    }


def ordered_algorithms(algorithms: set[str]) -> list[str]:
    known = [algorithm for algorithm in PREFERRED_ALGORITHMS if algorithm in algorithms]
    extra = sorted(algorithms.difference(PREFERRED_ALGORITHMS))
    return known + extra


def read_predictions(paths: list[Path]) -> tuple[dict[tuple[str, str, str, str, int], dict[str, str]], dict[str, object]]:
    index: dict[tuple[str, str, str, str, int], dict[str, str]] = {}
    status_counts: Counter[str] = Counter()
    algorithm_counts: Counter[str] = Counter()
    duplicate_count = 0
    rows = 0
    for path in paths:
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
                peptide = row.get("peptide", "").strip()
                hla_allele = row.get("hla_allele", "").strip()
                mhc_class = row.get("mhc_class", "").strip()
                peptide_length = safe_int(row.get("peptide_length"))
                if not peptide or not hla_allele or peptide_length is None:
                    continue
                key = (peptide, hla_allele, algorithm, mhc_class, peptide_length)
                existing = index.get(key)
                if existing is not None:
                    duplicate_count += 1
                    if prediction_quality(row) <= prediction_quality(existing):
                        continue
                index[key] = row
    summary = {
        "input_files": [str(path) for path in paths],
        "rows": rows,
        "usable_prediction_keys": len(index),
        "duplicate_keys": duplicate_count,
        "status_counts": dict(status_counts),
        "algorithm_counts": dict(algorithm_counts),
    }
    return index, summary


def prediction_quality(row: dict[str, str]) -> tuple[int, int, int]:
    status_score = 0 if row.get("status") == "ok" else 1
    missing_ic50 = 0 if safe_float(row.get("ic50")) is not None else 1
    missing_percentile = 0 if safe_float(row.get("percentile")) is not None else 1
    return (status_score, missing_ic50, missing_percentile)


def read_annotation(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            result[row["pvacseq_index"]] = row
    return result


def collect_min_window_starts(path: Path) -> dict[tuple[str, str, int], int]:
    result: dict[tuple[str, str, int], int] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (row["pvacseq_index"], row["mhc_class"], int(row["peptide_length"]))
            start = int(row["window_start_0based"])
            result[key] = min(result.get(key, start), start)
    return result


def build_output_row(
    epitope: dict[str, str],
    annotation: dict[str, str],
    hla_allele: str,
    algorithms: list[str],
    prediction_index: dict[tuple[str, str, str, str, int], dict[str, str]],
    min_window_start: dict[tuple[str, str, int], int],
    output_fields: list[str],
) -> tuple[dict[str, str], Counter[str]]:
    mhc_class = epitope["mhc_class"]
    peptide_length = int(epitope["peptide_length"])
    mt_peptide = epitope["mt_epitope_seq"]
    has_reliable_wt = has_matched_wt_control(epitope)
    wt_peptide = epitope.get("wt_epitope_seq", "") if has_reliable_wt else ""
    key_min = (epitope["pvacseq_index"], mhc_class, peptide_length)
    window_start = int(epitope["window_start_0based"])
    mutation_start = int(epitope["mutation_start_0based"])
    sub_peptide_position = window_start - min_window_start.get(key_min, window_start) + 1
    mutation_position = format_mutation_position(epitope, window_start)

    row = {field: "" for field in output_fields}
    base_values = {
        "Chromosome": annotation.get("chromosome_name", ""),
        "Start": annotation.get("start", ""),
        "Stop": annotation.get("stop", ""),
        "Reference": annotation.get("reference", ""),
        "Variant": annotation.get("variant", ""),
        "Transcript": annotation.get("transcript_name", epitope.get("transcript_name", "")),
        "Transcript Support Level": annotation.get("transcript_support_level", ""),
        "Transcript Length": annotation.get("transcript_length", ""),
        "Biotype": annotation.get("biotype", ""),
        "Ensembl Gene ID": annotation.get("ensembl_gene_id", ""),
        "Variant Type": annotation.get("variant_type", epitope.get("variant_type", "")),
        "Mutation": annotation.get("amino_acid_change", epitope.get("amino_acid_change", "")),
        "Protein Position": annotation.get("protein_position", epitope.get("protein_position", "")),
        "Gene Name": annotation.get("gene_name", epitope.get("gene_name", "")),
        "HGVSc": annotation.get("hgvsc", ""),
        "HGVSp": annotation.get("hgvsp", ""),
        "HLA Allele": hla_allele,
        "Peptide Length": str(peptide_length),
        "Sub-peptide Position": str(sub_peptide_position),
        "Mutation Position": str(mutation_position),
        "MT Epitope Seq": mt_peptide,
        "WT Epitope Seq": wt_peptide if has_reliable_wt else "",
        "Tumor DNA Depth": annotation.get("tdna_depth", ""),
        "Tumor DNA VAF": annotation.get("tdna_vaf", ""),
        "Tumor RNA Depth": annotation.get("trna_depth", ""),
        "Tumor RNA VAF": annotation.get("trna_vaf", ""),
        "Normal Depth": annotation.get("normal_depth", ""),
        "Normal VAF": annotation.get("normal_vaf", ""),
        "Gene Expression": annotation.get("gene_expression", ""),
        "Transcript Expression": annotation.get("transcript_expression", ""),
        "Extended MT Epitope Seq": epitope.get("extended_mt_epitope_seq", ""),
        "Extended Length": epitope.get("extended_length", ""),
        "WT Control Status": epitope.get("wt_control_status", ""),
        "WT Control Reason": epitope.get("wt_control_reason", ""),
        "Event ID": epitope.get("event_id", ""),
        "pVACseq Index": epitope.get("pvacseq_index", ""),
        "Index": epitope.get("pvacseq_index", ""),
    }
    for key, value in base_values.items():
        set_if_present(row, key, value)

    mt_ic50_by_algorithm: dict[str, float] = {}
    wt_ic50_by_algorithm: dict[str, float] = {}
    mt_percentile_by_algorithm: dict[str, float] = {}
    wt_percentile_by_algorithm: dict[str, float] = {}
    fold_by_algorithm: dict[str, float] = {}
    metrics: Counter[str] = Counter()

    for algorithm in algorithms:
        mt_pred = prediction_index.get((mt_peptide, hla_allele, algorithm, mhc_class, peptide_length))
        wt_pred = (
            prediction_index.get((wt_peptide, hla_allele, algorithm, mhc_class, peptide_length))
            if wt_peptide
            else None
        )
        fill_algorithm_columns(row, algorithm, "MT", mt_pred)
        fill_algorithm_columns(row, algorithm, "WT", wt_pred)
        mt_ic50 = safe_float(mt_pred.get("ic50") if mt_pred else None)
        wt_ic50 = safe_float(wt_pred.get("ic50") if wt_pred else None)
        mt_percentile = safe_float(mt_pred.get("percentile") if mt_pred else None)
        wt_percentile = safe_float(wt_pred.get("percentile") if wt_pred else None)
        if mt_ic50 is not None and algorithm in IC50_SUMMARY_ALGORITHMS:
            mt_ic50_by_algorithm[algorithm] = mt_ic50
        if wt_ic50 is not None and algorithm in IC50_SUMMARY_ALGORITHMS:
            wt_ic50_by_algorithm[algorithm] = wt_ic50
        if mt_percentile is not None:
            mt_percentile_by_algorithm[algorithm] = mt_percentile
        if wt_percentile is not None:
            wt_percentile_by_algorithm[algorithm] = wt_percentile
        if algorithm in IC50_SUMMARY_ALGORITHMS and mt_ic50 is not None and wt_ic50 is not None and mt_ic50 > 0:
            fold = wt_ic50 / mt_ic50
            fold_by_algorithm[algorithm] = fold
            set_if_present(row, f"{algorithm} Fold Change", fmt_float(fold))
        if mt_pred:
            metrics[f"{algorithm}_mt_present"] += 1
        if wt_pred:
            metrics[f"{algorithm}_wt_present"] += 1

    add_summary_metrics(row, mt_ic50_by_algorithm, wt_ic50_by_algorithm, fold_by_algorithm, mt_percentile_by_algorithm, wt_percentile_by_algorithm)
    if mt_ic50_by_algorithm:
        metrics["rows_with_mt_ic50"] += 1
    if wt_ic50_by_algorithm:
        metrics["rows_with_wt_ic50"] += 1
    if fold_by_algorithm:
        metrics["rows_with_fold_change"] += 1
    return row, metrics


def has_matched_wt_control(epitope: dict[str, str]) -> bool:
    """Return True when this MT peptide has a conventional matched WT epitope."""

    status = (epitope.get("wt_control_status") or "").strip()
    if status:
        return status == "matched_WT"
    return epitope.get("variant_type", "") != "FS" and bool(epitope.get("wt_epitope_seq", ""))


def fill_algorithm_columns(row: dict[str, str], algorithm: str, source: str, prediction: Optional[dict[str, str]]) -> None:
    if not prediction:
        return
    prefix = f"{algorithm} {source}"
    set_if_present(row, f"{prefix} IC50 Score", prediction.get("ic50", ""))
    set_if_present(row, f"{prefix} Percentile", prediction.get("percentile", ""))
    set_if_present(row, f"{prefix} Score", prediction.get("score", ""))
    set_if_present(row, f"{prefix} Status", prediction.get("status", ""))

    if algorithm == "MHCflurryEL":
        set_if_present(row, f"MHCflurryEL Presentation {source} Score", prediction.get("score", ""))
        set_if_present(row, f"MHCflurryEL Presentation {source} Percentile", prediction.get("percentile", ""))
    elif algorithm in {"NetMHCpanEL", "NetMHCIIpanEL"}:
        set_if_present(row, f"{algorithm} {source} Score", prediction.get("score", ""))
        set_if_present(row, f"{algorithm} {source} Percentile", prediction.get("percentile", ""))


def set_if_present(row: dict[str, str], key: str, value: object) -> None:
    if key in row:
        row[key] = "" if value is None else str(value)


def format_mutation_position(epitope: dict[str, str], window_start: int) -> str:
    """Format the mutation position inside the epitope window."""

    window_end = int(epitope["window_end_0based"])
    mutation_start = int(epitope["mutation_start_0based"])
    mutation_end = int(epitope["mutation_end_0based"])
    variant_type = epitope.get("variant_type", "")

    if variant_type == "inframe_del" and mutation_start == mutation_end:
        deleted_length = deleted_amino_acid_count(epitope.get("amino_acid_change", ""))
        junction_position = mutation_start - window_start + 1
        start_position = max(1, junction_position - deleted_length)
        end_position = max(start_position, junction_position)
        return format_position_range(start_position, end_position)

    overlap_start = max(window_start, mutation_start)
    overlap_end = min(window_end, mutation_end)
    start_position = overlap_start - window_start + 1
    end_position = overlap_end - window_start
    if end_position < start_position:
        end_position = start_position
    return format_position_range(start_position, end_position)


def deleted_amino_acid_count(amino_acid_change: str) -> int:
    """Return the deleted residue count from an amino-acid change string."""

    if "/" not in amino_acid_change:
        return 1
    wt_aa, mt_aa = amino_acid_change.split("/", 1)
    wt_aa = strip_stop_markers(wt_aa)
    mt_aa = strip_stop_markers(mt_aa)
    if mt_aa == "-":
        mt_aa = ""
    return max(1, len(wt_aa) - len(mt_aa))


def strip_stop_markers(amino_acid: str) -> str:
    for marker in ("*", "X"):
        if marker in amino_acid:
            return amino_acid.split(marker, 1)[0]
    return amino_acid


def format_position_range(start_position: int, end_position: int) -> str:
    if start_position == end_position:
        return str(start_position)
    return f"{start_position}-{end_position}"


def add_summary_metrics(
    row: dict[str, str],
    mt_ic50_by_algorithm: dict[str, float],
    wt_ic50_by_algorithm: dict[str, float],
    fold_by_algorithm: dict[str, float],
    mt_percentile_by_algorithm: dict[str, float],
    wt_percentile_by_algorithm: dict[str, float],
) -> None:
    if mt_ic50_by_algorithm:
        best_method = min(mt_ic50_by_algorithm, key=mt_ic50_by_algorithm.get)
        row["Best MT IC50 Score Method"] = best_method
        row["Best MT IC50 Score"] = fmt_float(mt_ic50_by_algorithm[best_method])
        row["Corresponding WT IC50 Score"] = fmt_float(wt_ic50_by_algorithm.get(best_method))
        row["Corresponding Fold Change"] = fmt_float(fold_by_algorithm.get(best_method))
        row["Median MT IC50 Score"] = fmt_float(median(mt_ic50_by_algorithm.values()))
    if wt_ic50_by_algorithm:
        row["Median WT IC50 Score"] = fmt_float(median(wt_ic50_by_algorithm.values()))
    if fold_by_algorithm:
        row["Median Fold Change"] = fmt_float(median(fold_by_algorithm.values()))
    if mt_percentile_by_algorithm:
        best_method = min(mt_percentile_by_algorithm, key=mt_percentile_by_algorithm.get)
        row["Best MT Percentile Method"] = best_method
        row["Best MT Percentile"] = fmt_float(mt_percentile_by_algorithm[best_method])
        row["Corresponding WT Percentile"] = fmt_float(wt_percentile_by_algorithm.get(best_method))
        row["Median MT Percentile"] = fmt_float(median(mt_percentile_by_algorithm.values()))
    if wt_percentile_by_algorithm:
        row["Median WT Percentile"] = fmt_float(median(wt_percentile_by_algorithm.values()))


def algorithm_columns(algorithms: list[str]) -> list[str]:
    columns: list[str] = []
    for algorithm in algorithms:
        columns.extend(
            [
                f"{algorithm} WT IC50 Score",
                f"{algorithm} MT IC50 Score",
                f"{algorithm} Fold Change",
                f"{algorithm} WT Percentile",
                f"{algorithm} MT Percentile",
                f"{algorithm} WT Score",
                f"{algorithm} MT Score",
                f"{algorithm} WT Status",
                f"{algorithm} MT Status",
            ]
        )
    return columns


def output_columns(profile: str, algorithms: list[str]) -> list[str]:
    if profile == "pvacseq":
        return PVACSEQ_COLUMNS
    if profile == "extended":
        return EXTENDED_CORE_COLUMNS + algorithm_columns(algorithms)
    raise ValueError(f"Unsupported output profile: {profile}")


def validate_against_pvacseq(pvacseq_path: Path, generated_path: Path, hla_by_class_length: dict[tuple[str, int], list[str]]) -> dict[str, object]:
    generated_keys = collect_validation_keys(generated_path)
    generated_hla_by_length = {
        length: set(hlas)
        for (mhc_class, length), hlas in hla_by_class_length.items()
        if mhc_class in {"MHC-I", "MHC-II"}
    }
    original_keys: set[tuple[str, str, str, str, str, str, str]] = set()
    original_rows_total = 0
    original_rows_comparable = 0
    original_hla_by_length: dict[int, set[str]] = defaultdict(set)
    with pvacseq_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            original_rows_total += 1
            length = int(row["Peptide Length"])
            original_hla_by_length[length].add(row["HLA Allele"])
            if row["HLA Allele"] not in generated_hla_by_length.get(length, set()):
                continue
            key = validation_key(row)
            original_keys.add(key)
            original_rows_comparable += 1
    shared = generated_keys & original_keys
    missing = original_keys - generated_keys
    extra = generated_keys - original_keys
    return {
        "pvacseq_rows_total": original_rows_total,
        "pvacseq_rows_comparable_hla": original_rows_comparable,
        "generated_rows": len(generated_keys),
        "shared_keys": len(shared),
        "missing_from_generated": len(missing),
        "extra_in_generated": len(extra),
        "missing_examples": [list(item) for item in sorted(missing)[:10]],
        "extra_examples": [list(item) for item in sorted(extra)[:10]],
        "pvacseq_hla_count_by_length": {str(k): len(v) for k, v in sorted(original_hla_by_length.items())},
        "generated_hla_count_by_length": {str(k): len(v) for k, v in sorted(generated_hla_by_length.items())},
    }


def collect_validation_keys(path: Path) -> set[tuple[str, str, str, str, str, str, str]]:
    keys: set[tuple[str, str, str, str, str, str, str]] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            keys.add(validation_key(row))
    return keys


def validation_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str, str]:
    wt_epitope = row.get("WT Epitope Seq", "")
    if row.get("Variant Type", "") == "FS":
        wt_epitope = "<FS_WT_IGNORED>"
    return (
        row.get("Transcript", ""),
        row.get("HGVSc", ""),
        row.get("HGVSp", ""),
        row.get("HLA Allele", ""),
        row.get("Peptide Length", ""),
        row.get("MT Epitope Seq", ""),
        wt_epitope,
    )


def safe_int(value: object) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except ValueError:
        return None


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except ValueError:
        return None


def fmt_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
