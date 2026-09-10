"""Step 02: split binding tasks into MT and WT task tables.

The upstream task table is de-duplicated by peptide-HLA-algorithm. This script
uses epitope_windows.tsv to decide whether each peptide came from mutant
epitopes, wildtype counterparts, or both. Frameshift WT counterparts are not
emitted because downstream reports treat frameshift antigens as mutant
neo-sequence evidence without a conventional WT paired epitope.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


TASK_FIELDS = ["peptide", "hla_allele", "algorithm", "mhc_class"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("--binding-tasks", required=True)
    parser.add_argument("--epitope-windows", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--mt-output", default=None)
    parser.add_argument("--wt-output", default=None)
    parser.add_argument("--summary", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mt_output = Path(args.mt_output) if args.mt_output else outdir / f"{args.sample}.MT.binding_tasks.tsv"
    wt_output = Path(args.wt_output) if args.wt_output else outdir / f"{args.sample}.WT.binding_tasks.tsv"
    summary_path = Path(args.summary) if args.summary else outdir / f"{args.sample}.split_binding_tasks.summary.json"

    peptide_sources, source_summary = collect_peptide_sources(Path(args.epitope_windows))
    mt_rows: list[dict[str, str]] = []
    wt_rows: list[dict[str, str]] = []
    mt_task_templates: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    all_rows = 0
    skipped_rows = 0

    with Path(args.binding_tasks).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        validate_fields(reader.fieldnames, Path(args.binding_tasks))
        for row in reader:
            all_rows += 1
            task = {field: (row.get(field) or "").strip() for field in TASK_FIELDS}
            key = (task["mhc_class"], task["peptide"], len(task["peptide"]))
            sources = peptide_sources.get(key, set())
            if "MT" in sources:
                mt_rows.append(task)
                mt_task_templates[key].append(task)
            if "WT" in sources:
                wt_rows.append(task)
            if not sources:
                skipped_rows += 1

    inferred_wt_rows, inferred_summary = build_wt_tasks_from_epitope_windows(Path(args.epitope_windows), mt_task_templates)
    wt_rows.extend(inferred_wt_rows)
    source_summary.update(inferred_summary)

    mt_rows = unique_rows(mt_rows)
    wt_rows = unique_rows(wt_rows)
    write_tsv(mt_output, mt_rows)
    write_tsv(wt_output, wt_rows)

    summary = {
        "sample": args.sample,
        "input_binding_tasks": str(Path(args.binding_tasks)),
        "input_epitope_windows": str(Path(args.epitope_windows)),
        "mt_output": str(mt_output),
        "wt_output": str(wt_output),
        "input_task_rows": all_rows,
        "mt_task_rows": len(mt_rows),
        "wt_task_rows": len(wt_rows),
        "skipped_task_rows_no_source": skipped_rows,
        "peptide_source_summary": source_summary,
        "fs_wt_rule": "Frameshift WT counterparts are excluded from WT binding tasks and merged WT fields.",
    }
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"[split_binding_tasks] wrote {mt_output}", flush=True)
    print(f"[split_binding_tasks] wrote {wt_output}", flush=True)
    print(f"[split_binding_tasks] wrote {summary_path}", flush=True)
    return 0


def collect_peptide_sources(epitope_windows: Path) -> tuple[dict[tuple[str, str, int], set[str]], dict[str, object]]:
    peptide_sources: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    source_counts: Counter[str] = Counter()
    row_count = 0
    with epitope_windows.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"mhc_class", "peptide_length", "mt_epitope_seq", "wt_epitope_seq", "variant_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{epitope_windows} missing required columns: {sorted(missing)}")
        for row in reader:
            row_count += 1
            mhc_class = (row.get("mhc_class") or "").strip()
            peptide_length = int(row.get("peptide_length") or 0)
            variant_type = (row.get("variant_type") or "").strip()

            mt_peptide = (row.get("mt_epitope_seq") or "").strip()
            if mt_peptide:
                peptide_sources[(mhc_class, mt_peptide, peptide_length)].add("MT")
                source_counts["MT"] += 1

            wt_peptide = (row.get("wt_epitope_seq") or "").strip()
            if wt_peptide and variant_type != "FS":
                peptide_sources[(mhc_class, wt_peptide, peptide_length)].add("WT")
                source_counts["WT"] += 1
            elif wt_peptide and variant_type == "FS":
                source_counts["FS_WT_SKIPPED"] += 1

    both_source_count = sum(1 for sources in peptide_sources.values() if sources == {"MT", "WT"})
    return peptide_sources, {
        "epitope_window_rows": row_count,
        "source_occurrences": dict(source_counts),
        "unique_prediction_peptides": len(peptide_sources),
        "unique_peptides_with_both_mt_and_wt_source": both_source_count,
    }


def build_wt_tasks_from_epitope_windows(
    epitope_windows: Path,
    mt_task_templates: dict[tuple[str, str, int], list[dict[str, str]]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    wt_rows: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    missing_template_count = 0
    row_count = 0
    with epitope_windows.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"mhc_class", "peptide_length", "mt_epitope_seq", "wt_epitope_seq", "variant_type"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{epitope_windows} missing required columns: {sorted(missing)}")
        for row in reader:
            row_count += 1
            mhc_class = (row.get("mhc_class") or "").strip()
            peptide_length = int(row.get("peptide_length") or 0)
            variant_type = (row.get("variant_type") or "").strip()

            mt_peptide = (row.get("mt_epitope_seq") or "").strip()
            if mt_peptide:
                source_counts["MT"] += 1

            wt_peptide = (row.get("wt_epitope_seq") or "").strip()
            if wt_peptide and variant_type != "FS":
                source_counts["WT"] += 1
                templates = mt_task_templates.get((mhc_class, mt_peptide, peptide_length), [])
                if not templates:
                    missing_template_count += 1
                    continue
                for template in templates:
                    wt_rows.append(
                        {
                            "peptide": wt_peptide,
                            "hla_allele": template["hla_allele"],
                            "algorithm": template["algorithm"],
                            "mhc_class": template["mhc_class"],
                        }
                    )
            elif wt_peptide and variant_type == "FS":
                source_counts["FS_WT_SKIPPED"] += 1

    return wt_rows, {
        "epitope_window_rows": row_count,
        "source_occurrences": dict(source_counts),
        "mt_template_keys": len(mt_task_templates),
        "wt_window_rows_without_mt_task_template": missing_template_count,
    }


def validate_fields(fieldnames: Optional[list[str]], path: Path) -> None:
    missing = set(TASK_FIELDS).difference(fieldnames or [])
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")


def unique_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for row in rows:
        key = tuple(row[field] for field in TASK_FIELDS)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return sorted(unique, key=lambda row: tuple(row[field] for field in TASK_FIELDS))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=TASK_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
