"""Step 01: build mutation epitope windows and prediction tasks.

This step merges the former annotation, epitope-window, HLA parsing, and task
construction stages. It should produce:

- stable variant/transcript annotation table;
- mutation-covering MT/WT epitope windows;
- event-level MT context and candidate-specific extended peptides;
- de-duplicated peptide-HLA-algorithm prediction task table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

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

# Peptide generation is restricted to coding changes with an explicit mutant
# protein sequence. Other consequences are excluded from downstream tasks.
SUPPORTED_VARIANT_TYPES = frozenset({"missense", "FS", "inframe_ins", "inframe_del"})

EVENT_QC_FIELDNAMES = [
    "event_status",
    "qc_reason",
    "protein_pair_source",
    "event_id",
    "pvacseq_index",
    "gene_name",
    "transcript_name",
    "variant_type",
    "amino_acid_change",
    "protein_position",
    "chromosome_name",
    "start",
    "stop",
    "reference",
    "variant",
    "hgvsc",
    "hgvsp",
    "mutation_start_0based",
    "mutation_end_0based",
    "mt_protein_fragment_length",
    "wt_protein_fragment_length",
    "epitope_window_count",
    "mhc_i_window_count",
    "mhc_ii_window_count",
    "extended_peptide_generated",
]

WT_CONTEXT_CONTROL_FIELDNAMES = [
    "event_id",
    "pvacseq_index",
    "gene_name",
    "transcript_name",
    "variant_type",
    "amino_acid_change",
    "protein_position",
    "mutation_start_0based",
    "mutation_end_0based",
    "sequence_role",
    "wt_context_control_sequence",
    "wt_context_control_length",
    "wt_context_control_start_0based",
    "wt_context_control_end_0based",
    "wt_protein_fragment_length",
    "mt_protein_fragment_length",
    "control_reason",
]

try:
    from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.hla_parser import (
        parse_hlahd_result,
    )
    from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.sequence_utils import (
        build_protein_pair_from_converter_row,
        generate_epitope_windows,
        locate_mutation_region,
        make_event_context_peptide,
        read_pvacseq_protein_pairs,
        window_covers_mutation,
    )
    from mimicneoai.mutation_derived_pipeline.scripts.mutation_epitope_prediction.task_utils import (
        build_binding_tasks,
    )
except ImportError:  # pragma: no cover - supports direct script execution before install
    from hla_parser import parse_hlahd_result  # type: ignore
    from sequence_utils import (  # type: ignore
        build_protein_pair_from_converter_row,
        generate_epitope_windows,
        locate_mutation_region,
        make_event_context_peptide,
        read_pvacseq_protein_pairs,
        window_covers_mutation,
    )
    from task_utils import build_binding_tasks  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for task construction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--sample", required=True)
    parser.add_argument("--converter-tsv", required=True)
    parser.add_argument("--protein-fasta", required=True)
    parser.add_argument("--hla-file", required=True)
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--mhc-i-lengths", default="8,9,10,11")
    parser.add_argument("--mhc-ii-lengths", default="15")
    parser.add_argument(
        "--protein-flank-length",
        type=int,
        default=25,
        help="Flanking length used for pVACtools protein FASTA generation.",
    )
    parser.add_argument("--extended-length", type=int, default=27)
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help=(
            "Comma- or space-separated predictor names. Defaults exclude "
            "SMM, SMMPMBEC, and PickPocket because they are retained mainly "
            "for legacy pVACseq compatibility."
        ),
    )
    parser.add_argument(
        "--pvacseq-merged",
        default=None,
        help="Optional original pVACseq merged all_epitopes TSV for peptide-space comparison.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""

    args = build_parser().parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mhc_i_lengths = parse_int_list(args.mhc_i_lengths)
    mhc_ii_lengths = parse_int_list(args.mhc_ii_lengths)
    algorithms = parse_str_list(args.algorithms)
    mhc_i_algorithms, mhc_ii_algorithms, unknown_algorithms = split_algorithms_by_mhc_class(algorithms)

    converter_annotation_rows, _ = build_variant_annotation(Path(args.converter_tsv))
    annotation_rows = [
        row
        for row in converter_annotation_rows
        if str(row.get("variant_type", "")) in SUPPORTED_VARIANT_TYPES
    ]
    unsupported_annotation_rows = [
        row
        for row in converter_annotation_rows
        if str(row.get("variant_type", "")) not in SUPPORTED_VARIANT_TYPES
    ]
    index_to_event = {
        str(row["pvacseq_index"]): str(row["event_id"])
        for row in annotation_rows
    }
    annotation_by_index = {str(row["pvacseq_index"]): row for row in annotation_rows}
    write_tsv(outdir / "variant_annotation.tsv", annotation_rows)
    if unsupported_annotation_rows:
        write_tsv(
            outdir / "unsupported_variant_annotations.tsv",
            unsupported_annotation_rows,
        )
    write_tsv(
        outdir / "index_event_map.tsv",
        [{"pvacseq_index": k, "event_id": v} for k, v in sorted(index_to_event.items())],
    )

    protein_pairs = read_pvacseq_protein_pairs(Path(args.protein_fasta))
    pair_map = {pair.event_id: pair for pair in protein_pairs}

    epitope_rows: list[dict[str, object]] = []
    extended_rows: list[dict[str, object]] = []
    wt_context_control_rows: list[dict[str, object]] = []
    missing_annotation: list[str] = []
    event_qc_rows: list[dict[str, object]] = []
    pair_status = Counter()

    for row in unsupported_annotation_rows:
        event_qc_rows.append(
            event_qc_row(
                row,
                event_status="unsupported_variant_type",
                qc_reason=f"variant_type={row.get('variant_type', '')} is not supported for peptide generation",
            )
        )

    for pvacseq_index, event_id in index_to_event.items():
        annotation = annotation_by_index[pvacseq_index]
        protein_pair_source = "fasta_pair_found"
        pair = pair_map.get(pvacseq_index)
        if pair is None:
            try:
                pair = build_protein_pair_from_converter_row(
                    annotation,
                    flanking_length=args.protein_flank_length,
                )
            except Exception as exc:
                pair_status["missing_protein_context"] += 1
                missing_annotation.append(pvacseq_index)
                event_qc_rows.append(
                    event_qc_row(
                        annotation,
                        event_status="missing_protein_context",
                        qc_reason=str(exc),
                    )
                )
                continue
            protein_pair_source = "rebuilt_from_converter"
            pair_status["rebuilt_from_converter"] += 1
        else:
            pair_status["fasta_pair_found"] += 1
        variant_type = str(annotation.get("variant_type", ""))
        try:
            mutation_start, mutation_end = locate_mutation_region(
                pair.wt_sequence,
                pair.mt_sequence,
                variant_type=variant_type,
            )
        except Exception as exc:
            pair_status["mutation_region_error"] += 1
            event_qc_rows.append(
                event_qc_row(
                    annotation,
                    event_status="mutation_region_error",
                    qc_reason=str(exc),
                    protein_pair_source=protein_pair_source,
                    mt_protein_fragment_length=len(pair.mt_sequence),
                    wt_protein_fragment_length=len(pair.wt_sequence) if pair.wt_sequence else "",
                )
            )
            continue
        all_lengths = tuple(sorted(set(mhc_i_lengths + mhc_ii_lengths)))
        try:
            windows = generate_epitope_windows(
                pair,
                all_lengths,
                args.extended_length,
                variant_type=variant_type,
            )
        except Exception as exc:
            pair_status["epitope_window_error"] += 1
            event_qc_rows.append(
                event_qc_row(
                    annotation,
                    event_status="epitope_window_error",
                    qc_reason=str(exc),
                    protein_pair_source=protein_pair_source,
                    mutation_start=mutation_start,
                    mutation_end=mutation_end,
                    mt_protein_fragment_length=len(pair.mt_sequence),
                    wt_protein_fragment_length=len(pair.wt_sequence) if pair.wt_sequence else "",
                )
            )
            continue
        if not windows:
            pair_status["no_window"] += 1
            event_qc_rows.append(
                event_qc_row(
                    annotation,
                    event_status="no_window",
                    qc_reason="no mutation-distinguishable peptide window for requested lengths",
                    protein_pair_source=protein_pair_source,
                    mutation_start=mutation_start,
                    mutation_end=mutation_end,
                    mt_protein_fragment_length=len(pair.mt_sequence),
                    wt_protein_fragment_length=len(pair.wt_sequence) if pair.wt_sequence else "",
                )
            )
            continue

        try:
            event_context = make_event_context_peptide(
                pair.mt_sequence,
                mutation_start,
                mutation_end,
                args.extended_length,
                variant_type=variant_type,
            )
        except Exception as exc:
            pair_status["event_context_error"] += 1
            event_qc_rows.append(
                event_qc_row(
                    annotation,
                    event_status="event_context_error",
                    qc_reason=str(exc),
                    protein_pair_source=protein_pair_source,
                    mutation_start=mutation_start,
                    mutation_end=mutation_end,
                    mt_protein_fragment_length=len(pair.mt_sequence),
                    wt_protein_fragment_length=len(pair.wt_sequence) if pair.wt_sequence else "",
                    epitope_window_count=len(windows),
                    mhc_i_window_count=sum(1 for window in windows if window.length in mhc_i_lengths),
                    mhc_ii_window_count=sum(1 for window in windows if window.length in mhc_ii_lengths),
                )
            )
            continue
        extended_rows.append(
            {
                "event_id": event_id,
                "pvacseq_index": pvacseq_index,
                "gene_name": annotation.get("gene_name", ""),
                "transcript_name": annotation.get("transcript_name", ""),
                "variant_type": variant_type,
                "amino_acid_change": annotation.get("amino_acid_change", ""),
                "protein_position": annotation.get("protein_position", ""),
                "mutation_start_0based": mutation_start,
                "mutation_end_0based": mutation_end,
                "sequence_role": "event_context",
                "event_context_mt_sequence": event_context.sequence,
                "event_context_length": len(event_context.sequence),
                "event_context_start_0based": event_context.start,
                "event_context_end_0based": event_context.end,
                "mt_protein_fragment_length": len(pair.mt_sequence),
                "wt_protein_fragment_length": len(pair.wt_sequence) if pair.wt_sequence else "",
            }
        )
        if variant_type.upper() == "FS":
            wt_context_control_rows.append(
                wt_context_control_row(
                    event_id=event_id,
                    pvacseq_index=pvacseq_index,
                    annotation=annotation,
                    pair=pair,
                    mutation_start=mutation_start,
                    mutation_end=mutation_end,
                    extended_length=args.extended_length,
                )
            )
        event_qc_rows.append(
            event_qc_row(
                annotation,
                event_status="rebuilt_from_converter" if protein_pair_source == "rebuilt_from_converter" else "ok",
                qc_reason="",
                protein_pair_source=protein_pair_source,
                mutation_start=mutation_start,
                mutation_end=mutation_end,
                mt_protein_fragment_length=len(pair.mt_sequence),
                wt_protein_fragment_length=len(pair.wt_sequence) if pair.wt_sequence else "",
                epitope_window_count=len(windows),
                mhc_i_window_count=sum(1 for window in windows if window.length in mhc_i_lengths),
                mhc_ii_window_count=sum(1 for window in windows if window.length in mhc_ii_lengths),
                extended_peptide_generated=True,
            )
        )

        for window in windows:
            if window.length in mhc_i_lengths:
                epitope_rows.append(epitope_row(event_id, pvacseq_index, annotation, window, "MHC-I", mutation_start, mutation_end))
            if window.length in mhc_ii_lengths:
                epitope_rows.append(epitope_row(event_id, pvacseq_index, annotation, window, "MHC-II", mutation_start, mutation_end))

    write_tsv_with_fieldnames(outdir / "event_qc.tsv", event_qc_rows, EVENT_QC_FIELDNAMES)
    write_tsv_with_fieldnames(
        outdir / "excluded_or_failed_events.tsv",
        [
            row
            for row in event_qc_rows
            if row.get("event_status") not in {"ok", "rebuilt_from_converter"}
        ],
        EVENT_QC_FIELDNAMES,
    )
    write_tsv(outdir / "extended_peptides.tsv", extended_rows)
    write_tsv_with_fieldnames(
        outdir / "wt_context_controls.tsv",
        wt_context_control_rows,
        WT_CONTEXT_CONTROL_FIELDNAMES,
    )
    write_tsv(outdir / "epitope_windows.tsv", epitope_rows)

    extension_mapping_failures = sum(
        not extension_contains_window(row)
        for row in epitope_rows
    )
    fs_wt_epitope_nonempty_rows = sum(
        str(row.get("variant_type", "")).upper() == "FS"
        and bool(str(row.get("wt_epitope_seq", "")))
        for row in epitope_rows
    )
    wt_control_status_counts = Counter(str(row.get("wt_control_status", "")) for row in epitope_rows)
    nonfs_wt_not_evaluable_rows = sum(
        str(row.get("variant_type", "")).upper() != "FS"
        and str(row.get("wt_control_status", "")) == "WT_not_evaluable"
        for row in epitope_rows
    )
    fs_event_ids_with_windows = {
        str(row.get("event_id", ""))
        for row in epitope_rows
        if str(row.get("variant_type", "")).upper() == "FS"
    }
    fs_context_control_event_ids = {
        str(row.get("event_id", ""))
        for row in wt_context_control_rows
    }
    missing_fs_wt_context_control_events = sorted(
        fs_event_ids_with_windows - fs_context_control_event_ids
    )
    if extension_mapping_failures or fs_wt_epitope_nonempty_rows or missing_fs_wt_context_control_events:
        raise RuntimeError(
            "Epitope construction invariant failed: "
            f"extension_mapping_failures={extension_mapping_failures}, "
            f"fs_wt_epitope_nonempty_rows={fs_wt_epitope_nonempty_rows}, "
            f"missing_fs_wt_context_control_events={len(missing_fs_wt_context_control_events)}"
        )

    hlas = parse_hlahd_result(Path(args.hla_file))
    prediction_peptide_rows = build_prediction_peptide_rows(epitope_rows)
    write_tsv(outdir / "prediction_peptides.tsv", prediction_peptide_rows)

    mhc_i_peptides = sorted(
        {
            str(row["peptide"])
            for row in prediction_peptide_rows
            if row["mhc_class"] == "MHC-I" and row["peptide"]
        }
    )
    mhc_ii_peptides = sorted(
        {
            str(row["peptide"])
            for row in prediction_peptide_rows
            if row["mhc_class"] == "MHC-II" and row["peptide"]
        }
    )
    tasks = build_binding_tasks(mhc_i_peptides, list(hlas.mhc_i), mhc_i_algorithms, "MHC-I")
    tasks.extend(build_binding_tasks(mhc_ii_peptides, list(hlas.mhc_ii), mhc_ii_algorithms, "MHC-II"))
    task_rows = [
        {
            "peptide": task.peptide,
            "hla_allele": task.hla_allele,
            "algorithm": task.algorithm,
            "mhc_class": task.mhc_class,
        }
        for task in tasks
    ]
    write_tsv(outdir / "binding_tasks.tsv", task_rows)

    summary: dict[str, object] = {
        "sample": args.sample,
        "converter_annotation_rows": len(converter_annotation_rows),
        "variant_annotation_rows": len(annotation_rows),
        "supported_variant_types": sorted(SUPPORTED_VARIANT_TYPES),
        "unsupported_variant_annotation_rows": len(unsupported_annotation_rows),
        "unsupported_variant_type_counts": dict(
            Counter(str(row.get("variant_type", "")) for row in unsupported_annotation_rows)
        ),
        "protein_pairs": len(protein_pairs),
        "fasta_pair_status": dict(pair_status),
        "missing_fasta_pair_count": len(missing_annotation),
        "event_qc_status_counts": dict(Counter(str(row.get("event_status", "")) for row in event_qc_rows)),
        "epitope_window_rows": len(epitope_rows),
        "extended_peptide_rows": len(extended_rows),
        "wt_context_control_rows": len(wt_context_control_rows),
        "extension_mapping_failures": extension_mapping_failures,
        "fs_wt_epitope_nonempty_rows": fs_wt_epitope_nonempty_rows,
        "wt_control_status_counts": dict(wt_control_status_counts),
        "nonfs_wt_not_evaluable_rows": nonfs_wt_not_evaluable_rows,
        "missing_fs_wt_context_control_events": len(missing_fs_wt_context_control_events),
        "prediction_peptide_rows": len(prediction_peptide_rows),
        "unique_mhc_i_prediction_peptides": len(mhc_i_peptides),
        "unique_mhc_ii_prediction_peptides": len(mhc_ii_peptides),
        "mhc_i_alleles": list(hlas.mhc_i),
        "mhc_ii_alleles": list(hlas.mhc_ii),
        "algorithms": algorithms,
        "mhc_i_algorithms": mhc_i_algorithms,
        "mhc_ii_algorithms": mhc_ii_algorithms,
        "unknown_algorithms": unknown_algorithms,
        "binding_task_rows": len(task_rows),
    }

    if args.pvacseq_merged:
        comparison_rows, comparison_summary = compare_with_pvacseq_merged(
            Path(args.pvacseq_merged),
            epitope_rows,
        )
        write_tsv(outdir / "pvacseq_peptide_space_comparison.tsv", comparison_rows)
        summary["pvacseq_comparison"] = comparison_summary

    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("[DONE] epitope task construction finished", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


def parse_int_list(value: str) -> tuple[int, ...]:
    """Parse comma-separated integers."""

    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_str_list(value: str) -> list[str]:
    """Parse comma- or whitespace-separated strings."""

    normalized = value.replace(",", " ")
    return [item.strip() for item in normalized.split() if item.strip()]


def split_algorithms_by_mhc_class(algorithms: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split mixed predictor names into MHC-I and MHC-II compatible groups."""

    mhc_i = [algorithm for algorithm in algorithms if algorithm in MHC_I_ALGORITHMS]
    mhc_ii = [algorithm for algorithm in algorithms if algorithm in MHC_II_ALGORITHMS]
    unknown = [
        algorithm
        for algorithm in algorithms
        if algorithm not in MHC_I_ALGORITHMS and algorithm not in MHC_II_ALGORITHMS
    ]
    return mhc_i, mhc_ii, unknown


def build_variant_annotation(converter_tsv: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
    """Build stable annotation rows and pVACtools-index mapping."""

    rows: list[dict[str, object]] = []
    index_to_event: dict[str, str] = {}
    seen_event_ids: Counter[str] = Counter()
    with Path(converter_tsv).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            base_event_id = make_event_id(row)
            seen_event_ids[base_event_id] += 1
            event_id = base_event_id
            if seen_event_ids[base_event_id] > 1:
                event_id = f"{base_event_id}|dup{seen_event_ids[base_event_id]}"

            pvacseq_index = row["index"]
            out_row: dict[str, object] = {"event_id": event_id, "pvacseq_index": pvacseq_index}
            for key, value in row.items():
                out_row[key] = value
            rows.append(out_row)
            index_to_event[pvacseq_index] = event_id
    return rows, index_to_event


def make_event_id(row: dict[str, str]) -> str:
    """Create a stable mutation-transcript event ID independent of pVACseq Index."""

    fields = [
        f"{row.get('chromosome_name', '')}:{row.get('start', '')}-{row.get('stop', '')}:{row.get('reference', '')}>{row.get('variant', '')}",
        row.get("gene_name", ""),
        row.get("transcript_name", ""),
        row.get("hgvsc", ""),
        row.get("hgvsp", ""),
        row.get("variant_type", ""),
        row.get("protein_position", ""),
        row.get("amino_acid_change", ""),
    ]
    return "|".join(fields)


def epitope_row(event_id: str, pvacseq_index: str, annotation: dict[str, object], window, mhc_class: str, mutation_start: int, mutation_end: int) -> dict[str, object]:
    """Convert an EpitopeWindow object into a TSV row."""

    covers_mutation = window_covers_mutation(
        window.start,
        window.end,
        mutation_start,
        mutation_end,
    )
    if not covers_mutation:
        raise ValueError(
            f"Epitope window does not cover the altered region: "
            f"event={event_id} peptide={window.peptide} "
            f"window={window.start}-{window.end} mutation={mutation_start}-{mutation_end}"
        )

    wt_control_status, wt_control_reason = classify_wt_control(
        str(annotation.get("variant_type", "")),
        window.wt_peptide,
    )
    return {
        "event_id": event_id,
        "pvacseq_index": pvacseq_index,
        "gene_name": annotation.get("gene_name", ""),
        "transcript_name": annotation.get("transcript_name", ""),
        "variant_type": annotation.get("variant_type", ""),
        "amino_acid_change": annotation.get("amino_acid_change", ""),
        "protein_position": annotation.get("protein_position", ""),
        "mhc_class": mhc_class,
        "peptide_length": window.length,
        "mt_epitope_seq": window.peptide,
        "wt_epitope_seq": window.wt_peptide or "",
        "window_start_0based": window.start,
        "window_end_0based": window.end,
        "mutation_start_0based": mutation_start,
        "mutation_end_0based": mutation_end,
        "covers_mutation": "YES" if covers_mutation else "NO",
        "wt_control_status": wt_control_status,
        "wt_control_reason": wt_control_reason,
        "extended_mt_epitope_seq": window.extended_peptide or "",
        "extended_length": len(window.extended_peptide or ""),
        "extended_start_0based": window.extended_start if window.extended_start is not None else "",
        "extended_end_0based": window.extended_end if window.extended_end is not None else "",
    }


def classify_wt_control(variant_type: str, wt_peptide: object) -> tuple[str, str]:
    """Classify whether a peptide row has a usable same-window WT control."""

    normalized_variant_type = variant_type.upper()
    if normalized_variant_type == "FS":
        return ("WT_context_control", "frameshift_has_no_conventional_matched_WT_epitope")
    if wt_peptide:
        return ("matched_WT", "")
    return ("WT_not_evaluable", "no_same_position_same_length_WT_epitope")


def wt_context_control_row(
    *,
    event_id: str,
    pvacseq_index: str,
    annotation: dict[str, object],
    pair,
    mutation_start: int,
    mutation_end: int,
    extended_length: int,
) -> dict[str, object]:
    """Build an event-level WT context control row for frameshift events."""

    if not pair.wt_sequence:
        raise ValueError(f"Frameshift event lacks WT protein context: {event_id}")
    wt_context = make_event_context_peptide(
        pair.wt_sequence,
        mutation_start,
        mutation_start,
        extended_length,
        variant_type="",
    )
    return {
        "event_id": event_id,
        "pvacseq_index": pvacseq_index,
        "gene_name": annotation.get("gene_name", ""),
        "transcript_name": annotation.get("transcript_name", ""),
        "variant_type": annotation.get("variant_type", ""),
        "amino_acid_change": annotation.get("amino_acid_change", ""),
        "protein_position": annotation.get("protein_position", ""),
        "mutation_start_0based": mutation_start,
        "mutation_end_0based": mutation_end,
        "sequence_role": "WT_context_control",
        "wt_context_control_sequence": wt_context.sequence,
        "wt_context_control_length": len(wt_context.sequence),
        "wt_context_control_start_0based": wt_context.start,
        "wt_context_control_end_0based": wt_context.end,
        "wt_protein_fragment_length": len(pair.wt_sequence),
        "mt_protein_fragment_length": len(pair.mt_sequence),
        "control_reason": "frameshift_breakpoint_context",
    }


def event_qc_row(
    annotation: dict[str, object],
    *,
    event_status: str,
    qc_reason: str,
    protein_pair_source: str = "",
    mutation_start: object = "",
    mutation_end: object = "",
    mt_protein_fragment_length: object = "",
    wt_protein_fragment_length: object = "",
    epitope_window_count: int = 0,
    mhc_i_window_count: int = 0,
    mhc_ii_window_count: int = 0,
    extended_peptide_generated: bool = False,
) -> dict[str, object]:
    """Create one row of event-level peptide-generation QC."""

    return {
        "event_status": event_status,
        "qc_reason": qc_reason,
        "protein_pair_source": protein_pair_source,
        "event_id": annotation.get("event_id", ""),
        "pvacseq_index": annotation.get("pvacseq_index", ""),
        "gene_name": annotation.get("gene_name", ""),
        "transcript_name": annotation.get("transcript_name", ""),
        "variant_type": annotation.get("variant_type", ""),
        "amino_acid_change": annotation.get("amino_acid_change", ""),
        "protein_position": annotation.get("protein_position", ""),
        "chromosome_name": annotation.get("chromosome_name", ""),
        "start": annotation.get("start", ""),
        "stop": annotation.get("stop", ""),
        "reference": annotation.get("reference", ""),
        "variant": annotation.get("variant", ""),
        "hgvsc": annotation.get("hgvsc", ""),
        "hgvsp": annotation.get("hgvsp", ""),
        "mutation_start_0based": mutation_start,
        "mutation_end_0based": mutation_end,
        "mt_protein_fragment_length": mt_protein_fragment_length,
        "wt_protein_fragment_length": wt_protein_fragment_length,
        "epitope_window_count": epitope_window_count,
        "mhc_i_window_count": mhc_i_window_count,
        "mhc_ii_window_count": mhc_ii_window_count,
        "extended_peptide_generated": "YES" if extended_peptide_generated else "NO",
    }


def extension_contains_window(row: dict[str, object]) -> bool:
    """Validate candidate extension containment using coordinates and sequence."""

    try:
        window_start = int(row["window_start_0based"])
        window_end = int(row["window_end_0based"])
        extended_start = int(row["extended_start_0based"])
        extended_end = int(row["extended_end_0based"])
    except (KeyError, TypeError, ValueError):
        return False

    peptide = str(row.get("mt_epitope_seq", ""))
    extension = str(row.get("extended_mt_epitope_seq", ""))
    if not peptide or not extension:
        return False
    if extended_start > window_start or extended_end < window_end:
        return False
    if extended_end - extended_start != len(extension):
        return False

    relative_start = window_start - extended_start
    relative_end = relative_start + len(peptide)
    return extension[relative_start:relative_end] == peptide


def build_prediction_peptide_rows(epitope_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Build unique MT/WT peptide rows that need binding prediction.

    The task table itself is intentionally de-duplicated by peptide-HLA-algorithm.
    This companion table records whether a peptide came from MT, WT, or both, so
    later merge code can reconstruct MT/WT evidence and fold-change values.
    """

    peptide_meta: dict[tuple[str, str, int], dict[str, object]] = {}
    for row in epitope_rows:
        mhc_class = str(row["mhc_class"])
        peptide_length = int(row["peptide_length"])
        event_id = str(row["event_id"])
        for source_type, column in (("MT", "mt_epitope_seq"), ("WT", "wt_epitope_seq")):
            if source_type == "WT" and str(row.get("variant_type", "")) == "FS":
                continue
            peptide = str(row.get(column, "") or "")
            if not peptide:
                continue
            key = (mhc_class, peptide, peptide_length)
            item = peptide_meta.setdefault(
                key,
                {
                    "mhc_class": mhc_class,
                    "peptide_length": peptide_length,
                    "peptide": peptide,
                    "source_types": set(),
                    "event_ids": set(),
                },
            )
            item["source_types"].add(source_type)
            item["event_ids"].add(event_id)

    rows: list[dict[str, object]] = []
    for mhc_class, peptide, peptide_length in sorted(peptide_meta):
        item = peptide_meta[(mhc_class, peptide, peptide_length)]
        source_types = sorted(item["source_types"])
        event_ids = sorted(item["event_ids"])
        rows.append(
            {
                "mhc_class": mhc_class,
                "peptide_length": peptide_length,
                "peptide": peptide,
                "source_types": ",".join(source_types),
                "event_count": len(event_ids),
                "event_ids": ";".join(event_ids),
            }
        )
    return rows


def compare_with_pvacseq_merged(pvacseq_merged: Path, epitope_rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Compare generated MT peptide space with an original pVACseq merged table."""

    generated: dict[tuple[str, int], set[str]] = {}
    for row in epitope_rows:
        key = (str(row["mhc_class"]), int(row["peptide_length"]))
        generated.setdefault(key, set()).add(str(row["mt_epitope_seq"]))

    original: dict[tuple[str, int], set[str]] = {}
    with Path(pvacseq_merged).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            length = int(row["Peptide Length"])
            mhc_class = "MHC-I" if length <= 11 else "MHC-II"
            original.setdefault((mhc_class, length), set()).add(row["MT Epitope Seq"])

    keys = sorted(set(generated) | set(original))
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for key in keys:
        gen = generated.get(key, set())
        orig = original.get(key, set())
        missing = orig - gen
        extra = gen - orig
        row = {
            "mhc_class": key[0],
            "peptide_length": key[1],
            "generated_unique_mt_peptides": len(gen),
            "pvacseq_unique_mt_peptides": len(orig),
            "shared_unique_mt_peptides": len(gen & orig),
            "missing_from_generated": len(missing),
            "extra_in_generated": len(extra),
            "missing_examples": ",".join(sorted(missing)[:10]),
            "extra_examples": ",".join(sorted(extra)[:10]),
        }
        rows.append(row)
        summary[f"{key[0]}_{key[1]}"] = row
    return rows, summary


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to a TSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_tsv_with_fieldnames(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write rows to a TSV file with a stable header even when empty."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})


if __name__ == "__main__":
    raise SystemExit(main())
