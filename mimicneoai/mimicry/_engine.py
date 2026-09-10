"""Deterministic indexed execution for sequence-based molecular mimicry."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ._adapters import load_occurrences, load_unique_peptides, read_input_manifest
from ._binding import annotate_shared_hla_support
from ._io import file_identity, open_text, read_json, read_tsv, sha256_file, write_json, write_tsv
from ._metrics import (
    central_triplet_offsets,
    compute_pair_metrics,
    is_eligible_peptide,
    is_sequence_mimic,
)
from ._models import (
    InputEntry,
    MimicryRunConfig,
    PeptideOccurrence,
    SUPPORTED_SOURCE_PAIRS,
)
from ._policy import SequenceMimicryPolicy, get_policy


PAIR_FIELDS = [
    "candidate_id",
    "method_version",
    "patient_id",
    "source_pair",
    "source_a",
    "source_b",
    "peptide_a",
    "peptide_b",
    "peptide_length",
    "central_positions",
    "central_matches",
    "central_total",
    "central_identity",
    "central_run_length",
    "central_run_sequence",
    "positional_match_count",
    "full_length_identity",
    "p2_exact_match",
    "pomega_exact_match",
    "exact_p2_pomega_high_confidence",
    "blosum62_full_score",
    "blosum62_central_score",
    "legacy_lcs_length",
    "legacy_lcs_sequence",
    "legacy_lcs_ratio",
    "primary_call",
    "analysis_scope",
    "mutation_event_count",
    "mt_specific_event_count",
    "wt_compatible_event_count",
    "wt_not_evaluable_event_count",
]

EVENT_FIELDS = [
    "candidate_id",
    "method_version",
    "patient_id",
    "source_pair",
    "nonmutation_source",
    "nonmutation_peptide",
    "mt_peptide",
    "wt_peptide",
    "peptide_length",
    "event_id",
    "mutation_occurrence_id",
    "wt_control_status",
    "wt_classification",
    "wt_passes_v12",
    "wt_central_identity",
    "wt_central_run_length",
    "wt_positional_match_count",
    "mt_minus_wt_central_identity",
]

PROVENANCE_FIELDS = [
    "patient_id",
    "antigen_source",
    "peptide",
    "peptide_length",
    "peptide_id",
    "occurrence_id",
    "source_record_id",
    "event_id",
    "wt_peptide",
    "wt_control_status",
    "passing_pair_count",
    "source_path",
    "source_row_number",
    "source_metadata_json",
]

STAGE_FIELDS = [
    "row_type",
    "patient_id",
    "source_pair",
    "peptide_length",
    "shard_id",
    "shard_count",
    "source_a_unique_peptides",
    "source_b_unique_peptides",
    "query_source",
    "query_unique_peptides_in_shard",
    "indexed_source",
    "indexed_unique_peptides",
    "evaluable_same_length_pairs",
    "index_recalled_pairs",
    "passing_unique_pairs",
    "mutation_event_rows",
    "elapsed_seconds",
]

INPUT_QC_FIELDS = [
    "patient_id",
    "antigen_source",
    "input_format",
    "input_rows",
    "eligible_core_rows",
    "eligible_occurrence_rows",
    "eligible_unique_peptides",
    "excluded_non_core_or_non_hla_i_rows",
    "excluded_invalid_sequence_or_length_rows",
    "peptide_path",
    "provenance_path",
    "run_manifest_path",
    "status",
]


def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _candidate_id(
    method: str,
    patient_id: str,
    source_pair: str,
    peptide_a: str,
    peptide_b: str,
) -> str:
    key = "|".join((method, patient_id, source_pair, peptide_a, peptide_b, str(len(peptide_a))))
    return "mimic-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _stable_shard(
    patient_id: str,
    source_pair: str,
    peptide_length: int,
    query_source: str,
    peptide: str,
    shard_count: int,
) -> int:
    key = f"{patient_id}|{source_pair}|{peptide_length}|{query_source}|{peptide}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % shard_count


def _group_by_peptide(
    occurrences: Iterable[PeptideOccurrence],
    peptide_length: int,
) -> dict[str, list[PeptideOccurrence]]:
    grouped: dict[str, list[PeptideOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        if len(occurrence.peptide) == peptide_length:
            grouped[occurrence.peptide].append(occurrence)
    return dict(grouped)


def _build_triplet_index(
    peptides: Iterable[str],
    peptide_length: int,
    policy: SequenceMimicryPolicy,
) -> dict[tuple[int, str], set[str]]:
    index: dict[tuple[int, str], set[str]] = defaultdict(set)
    for peptide in peptides:
        for offset in central_triplet_offsets(peptide_length, policy):
            index[(offset, peptide[offset : offset + 3])].add(peptide)
    return dict(index)


def _pair_metric_row(metrics, peptide_length: int) -> dict[str, object]:
    return {
        "central_positions": metrics.central_positions,
        "central_matches": metrics.central_matches,
        "central_total": metrics.central_total,
        "central_identity": f"{metrics.central_identity:.6f}",
        "central_run_length": metrics.central_run_length,
        "central_run_sequence": metrics.central_run_sequence,
        "positional_match_count": metrics.positional_match_count,
        "full_length_identity": f"{metrics.full_length_identity:.6f}",
        "p2_exact_match": int(metrics.p2_exact_match),
        "pomega_exact_match": int(metrics.pomega_exact_match),
        "exact_p2_pomega_high_confidence": int(metrics.exact_p2_pomega),
        "blosum62_full_score": metrics.blosum62_full_score,
        "blosum62_central_score": metrics.blosum62_central_score,
        "legacy_lcs_length": metrics.legacy_lcs_length,
        "legacy_lcs_sequence": metrics.legacy_lcs_sequence,
        "legacy_lcs_ratio": f"{metrics.legacy_lcs_length / peptide_length:.6f}",
    }


def _classify_wt(
    nonmutation_peptide: str,
    mutation_occurrence: PeptideOccurrence,
    policy: SequenceMimicryPolicy,
) -> dict[str, object]:
    status = mutation_occurrence.wt_control_status.strip().lower()
    wt_peptide = mutation_occurrence.wt_peptide
    base = {
        "wt_passes_v12": "",
        "wt_central_identity": "",
        "wt_central_run_length": "",
        "wt_positional_match_count": "",
        "mt_minus_wt_central_identity": "",
    }
    if status == "wt_context_control":
        return {**base, "wt_classification": "WT_not_evaluable_for_neo_specificity"}
    if status != "matched_wt" or len(wt_peptide) != len(nonmutation_peptide):
        return {**base, "wt_classification": "WT_not_evaluable"}
    if not is_eligible_peptide(wt_peptide, policy):
        return {**base, "wt_classification": "WT_not_evaluable"}

    wt_metrics = compute_pair_metrics(nonmutation_peptide, wt_peptide, policy)
    mt_metrics = compute_pair_metrics(nonmutation_peptide, mutation_occurrence.peptide, policy)
    wt_passes = is_sequence_mimic(nonmutation_peptide, wt_peptide, policy)
    return {
        "wt_classification": (
            "WT_compatible_sequence_mimic" if wt_passes else "MT_specific_sequence_mimic"
        ),
        "wt_passes_v12": int(wt_passes),
        "wt_central_identity": f"{wt_metrics.central_identity:.6f}",
        "wt_central_run_length": wt_metrics.central_run_length,
        "wt_positional_match_count": wt_metrics.positional_match_count,
        "mt_minus_wt_central_identity": (
            f"{mt_metrics.central_identity - wt_metrics.central_identity:.6f}"
        ),
    }


def _run_search_shard(payload: dict[str, object]) -> dict[str, object]:
    started = time.monotonic()
    patient_id = str(payload["patient_id"])
    source_pair = str(payload["source_pair"])
    source_a, source_b = source_pair.split("--", 1)
    peptide_length = int(payload["peptide_length"])
    shard_id = int(payload["shard_id"])
    shard_count = int(payload["shard_count"])
    policy = get_policy(str(payload["method"]))
    entry_a = payload["entry_a"]
    entry_b = payload["entry_b"]
    if not isinstance(entry_a, InputEntry) or not isinstance(entry_b, InputEntry):
        raise TypeError("Internal mimicry task contains invalid input entries")

    peptides_a = load_unique_peptides(entry_a, policy, peptide_length)
    peptides_b = load_unique_peptides(entry_b, policy, peptide_length)
    mutation_rows: dict[str, list[PeptideOccurrence]] = {}
    if source_a == "mutation-derived":
        mutation_rows = _group_by_peptide(
            load_occurrences(entry_a, policy).occurrences, peptide_length
        )
    elif source_b == "mutation-derived":
        mutation_rows = _group_by_peptide(
            load_occurrences(entry_b, policy).occurrences, peptide_length
        )
    if len(peptides_a) <= len(peptides_b):
        indexed_source, indexed_peptides = source_a, peptides_a
        query_source, query_peptides_all = source_b, peptides_b
    else:
        indexed_source, indexed_peptides = source_b, peptides_b
        query_source, query_peptides_all = source_a, peptides_a

    query_peptides = [
        peptide
        for peptide in sorted(query_peptides_all)
        if _stable_shard(
            patient_id,
            source_pair,
            peptide_length,
            query_source,
            peptide,
            shard_count,
        )
        == shard_id
    ]
    index = _build_triplet_index(indexed_peptides, peptide_length, policy)
    part_dir = Path(str(payload["part_dir"]))
    part_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{_safe_component(patient_id)}.{_safe_component(source_pair)}."
        f"len{peptide_length}.shard{shard_id:04d}"
    )
    pair_path = part_dir / f"{stem}.pairs.tsv.gz"
    event_path = part_dir / f"{stem}.mutation_wt.tsv.gz"
    pair_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    recalled_count = 0

    for query_peptide in query_peptides:
        recalled: set[str] = set()
        for offset in central_triplet_offsets(peptide_length, policy):
            recalled.update(index.get((offset, query_peptide[offset : offset + 3]), set()))
        recalled_count += len(recalled)
        for indexed_peptide in sorted(recalled):
            if query_source == source_a:
                peptide_a, peptide_b = query_peptide, indexed_peptide
            else:
                peptide_a, peptide_b = indexed_peptide, query_peptide
            if not is_sequence_mimic(peptide_a, peptide_b, policy):
                continue
            candidate_id = _candidate_id(
                policy.name, patient_id, source_pair, peptide_a, peptide_b
            )
            metrics = compute_pair_metrics(peptide_a, peptide_b, policy)
            mutation_occurrences: list[PeptideOccurrence] = []
            nonmutation_source = ""
            nonmutation_peptide = ""
            if source_a == "mutation-derived":
                mutation_occurrences = mutation_rows[peptide_a]
                nonmutation_source, nonmutation_peptide = source_b, peptide_b
            elif source_b == "mutation-derived":
                mutation_occurrences = mutation_rows[peptide_b]
                nonmutation_source, nonmutation_peptide = source_a, peptide_a

            classifications = Counter()
            for occurrence in mutation_occurrences:
                wt_result = _classify_wt(nonmutation_peptide, occurrence, policy)
                classifications[str(wt_result["wt_classification"])] += 1
                event_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "method_version": policy.name,
                        "patient_id": patient_id,
                        "source_pair": source_pair,
                        "nonmutation_source": nonmutation_source,
                        "nonmutation_peptide": nonmutation_peptide,
                        "mt_peptide": occurrence.peptide,
                        "wt_peptide": occurrence.wt_peptide,
                        "peptide_length": peptide_length,
                        "event_id": occurrence.event_id,
                        "mutation_occurrence_id": occurrence.occurrence_id,
                        "wt_control_status": occurrence.wt_control_status,
                        **wt_result,
                    }
                )
            pair_rows.append(
                {
                    "candidate_id": candidate_id,
                    "method_version": policy.name,
                    "patient_id": patient_id,
                    "source_pair": source_pair,
                    "source_a": source_a,
                    "source_b": source_b,
                    "peptide_a": peptide_a,
                    "peptide_b": peptide_b,
                    "peptide_length": peptide_length,
                    **_pair_metric_row(metrics, peptide_length),
                    "primary_call": policy.primary_call,
                    "analysis_scope": "within_patient_cross_source_sequence_only",
                    "mutation_event_count": len(mutation_occurrences),
                    "mt_specific_event_count": classifications["MT_specific_sequence_mimic"],
                    "wt_compatible_event_count": classifications[
                        "WT_compatible_sequence_mimic"
                    ],
                    "wt_not_evaluable_event_count": sum(
                        count
                        for label, count in classifications.items()
                        if label.startswith("WT_not_evaluable")
                    ),
                }
            )

    write_tsv(pair_path, PAIR_FIELDS, pair_rows)
    write_tsv(event_path, EVENT_FIELDS, event_rows)
    return {
        "row_type": "shard",
        "patient_id": patient_id,
        "source_pair": source_pair,
        "peptide_length": peptide_length,
        "shard_id": shard_id,
        "shard_count": shard_count,
        "source_a_unique_peptides": len(peptides_a),
        "source_b_unique_peptides": len(peptides_b),
        "query_source": query_source,
        "query_unique_peptides_in_shard": len(query_peptides),
        "indexed_source": indexed_source,
        "indexed_unique_peptides": len(indexed_peptides),
        "evaluable_same_length_pairs": len(query_peptides) * len(indexed_peptides),
        "index_recalled_pairs": recalled_count,
        "passing_unique_pairs": len(pair_rows),
        "mutation_event_rows": len(event_rows),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pair_path": str(pair_path),
        "event_path": str(event_path),
    }


def _merge_shard_outputs(paths: list[Path], output: Path, fieldnames: list[str]) -> int:
    def rows():
        for path in sorted(paths):
            with open_text(path) as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                if reader.fieldnames != fieldnames:
                    raise ValueError(f"Unexpected part schema: {path}")
                yield from reader

    return write_tsv(output, fieldnames, rows())


def _build_stagewise_qc_rows(
    task_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Return shard records followed by auditable patient/length summaries."""

    rows = list(task_results)
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in task_results:
        grouped[
            (
                str(row["patient_id"]),
                str(row["source_pair"]),
                int(row["peptide_length"]),
            )
        ].append(row)
    for (patient_id, source_pair, peptide_length), group in sorted(grouped.items()):
        first = group[0]
        rows.append(
            {
                "row_type": "patient_length_summary",
                "patient_id": patient_id,
                "source_pair": source_pair,
                "peptide_length": peptide_length,
                "shard_id": "all",
                "shard_count": first["shard_count"],
                "source_a_unique_peptides": first["source_a_unique_peptides"],
                "source_b_unique_peptides": first["source_b_unique_peptides"],
                "query_source": first["query_source"],
                "query_unique_peptides_in_shard": sum(
                    int(row["query_unique_peptides_in_shard"]) for row in group
                ),
                "indexed_source": first["indexed_source"],
                "indexed_unique_peptides": first["indexed_unique_peptides"],
                "evaluable_same_length_pairs": sum(
                    int(row["evaluable_same_length_pairs"]) for row in group
                ),
                "index_recalled_pairs": sum(int(row["index_recalled_pairs"]) for row in group),
                "passing_unique_pairs": sum(int(row["passing_unique_pairs"]) for row in group),
                "mutation_event_rows": sum(int(row["mutation_event_rows"]) for row in group),
                "elapsed_seconds": round(
                    sum(float(row["elapsed_seconds"]) for row in group), 3
                ),
            }
        )
    return rows


def _build_member_provenance(
    pair_path: Path,
    entries: list[InputEntry],
    policy: SequenceMimicryPolicy,
    output_path: Path,
) -> int:
    pair_counts: Counter[tuple[str, str, str]] = Counter()
    for _, row in read_tsv(pair_path):
        pair_counts[(row["patient_id"], row["source_a"], row["peptide_a"])] += 1
        pair_counts[(row["patient_id"], row["source_b"], row["peptide_b"])] += 1

    def rows():
        for entry in sorted(entries, key=lambda item: (item.patient_id, item.antigen_source)):
            result = load_occurrences(entry, policy)
            for occurrence in sorted(
                result.occurrences,
                key=lambda item: (item.peptide, item.occurrence_id),
            ):
                count = pair_counts[
                    (occurrence.patient_id, occurrence.antigen_source, occurrence.peptide)
                ]
                if not count:
                    continue
                yield {
                    "patient_id": occurrence.patient_id,
                    "antigen_source": occurrence.antigen_source,
                    "peptide": occurrence.peptide,
                    "peptide_length": len(occurrence.peptide),
                    "peptide_id": occurrence.peptide_id,
                    "occurrence_id": occurrence.occurrence_id,
                    "source_record_id": occurrence.source_record_id,
                    "event_id": occurrence.event_id,
                    "wt_peptide": occurrence.wt_peptide,
                    "wt_control_status": occurrence.wt_control_status,
                    "passing_pair_count": count,
                    "source_path": occurrence.source_path,
                    "source_row_number": occurrence.source_row_number,
                    "source_metadata_json": json.dumps(
                        occurrence.metadata, sort_keys=True, separators=(",", ":")
                    ),
                }

    return write_tsv(output_path, PROVENANCE_FIELDS, rows())


def _validate_source_coverage(
    entries: list[InputEntry], source_pairs: tuple[str, ...]
) -> dict[tuple[str, str], InputEntry]:
    indexed = {(entry.patient_id, entry.antigen_source): entry for entry in entries}
    patients = sorted({entry.patient_id for entry in entries})
    for source_pair in source_pairs:
        source_a, source_b = source_pair.split("--", 1)
        eligible_patients = [
            patient
            for patient in patients
            if (patient, source_a) in indexed or (patient, source_b) in indexed
        ]
        if not eligible_patients:
            raise ValueError(f"No manifest inputs found for requested source pair {source_pair}")
        for patient in eligible_patients:
            missing = [
                source
                for source in (source_a, source_b)
                if (patient, source) not in indexed
            ]
            if missing:
                raise ValueError(
                    f"Patient {patient} is missing {missing} for requested pair {source_pair}"
                )
    return indexed


def _build_input_signature(
    config: MimicryRunConfig,
    entries: list[InputEntry],
    policy: SequenceMimicryPolicy,
) -> dict[str, object]:
    module_paths = [
        Path(__file__),
        Path(__file__).with_name("_metrics.py"),
        Path(__file__).with_name("_adapters.py"),
        Path(__file__).with_name("_binding.py"),
        Path(__file__).with_name("_io.py"),
        Path(__file__).with_name("_models.py"),
        Path(__file__).with_name("_policy.py"),
    ]
    return {
        "method": policy.name,
        "source_pairs": list(config.source_pairs),
        "shard_count": config.shard_count,
        "input_manifest": file_identity(config.input_manifest),
        "inputs": [
            {
                "patient_id": entry.patient_id,
                "antigen_source": entry.antigen_source,
                "input_format": entry.input_format,
                "peptide": file_identity(entry.peptide_path),
                "provenance": file_identity(entry.provenance_path)
                if entry.provenance_path
                else None,
                "run_manifest": file_identity(entry.run_manifest_path)
                if entry.run_manifest_path
                else None,
                "binding": file_identity(entry.binding_path)
                if config.binding_support_enabled and entry.binding_path
                else None,
            }
            for entry in sorted(entries, key=lambda item: (item.patient_id, item.antigen_source))
        ],
        "code": {path.name: sha256_file(path) for path in module_paths},
        "binding_support_enabled": config.binding_support_enabled,
    }


def _validate_resume_contract(
    manifest_path: Path,
    input_signature: dict[str, object],
) -> bool:
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    if manifest.get("run_status") != "complete":
        raise ValueError(f"Existing mimicry manifest is not complete: {manifest_path}")
    if manifest.get("input_signature") != input_signature:
        raise ValueError("Existing mimicry output was generated from different inputs or code")
    outputs = manifest.get("output_signature")
    if not isinstance(outputs, dict):
        raise ValueError("Existing mimicry manifest has no output signature")
    for name, expected in outputs.items():
        if not isinstance(expected, dict):
            raise ValueError(f"Invalid output identity in manifest: {name}")
        path = Path(str(expected.get("path", "")))
        if file_identity(path) != expected:
            raise ValueError(f"Existing mimicry output identity mismatch: {path}")
    return True


def run_sequence_mimicry(config: MimicryRunConfig) -> dict[str, object]:
    """Run sequence mimicry from a versioned input manifest."""

    config = validate_run_config(config)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    policy = get_policy(config.method)
    entries = read_input_manifest(config.input_manifest)
    if config.binding_support_enabled:
        missing_binding = [
            f"{entry.patient_id}/{entry.antigen_source}"
            for entry in entries
            if entry.binding_path is None
        ]
        if missing_binding:
            raise ValueError(
                "binding_support.enabled requires binding_path for every input: "
                + ", ".join(missing_binding)
            )
    indexed_entries = _validate_source_coverage(entries, config.source_pairs)
    input_signature = _build_input_signature(config, entries, policy)
    manifest_path = config.output_dir / "run_manifest.json"
    if _validate_resume_contract(manifest_path, input_signature):
        manifest = read_json(manifest_path)
        manifest["reused"] = True
        return manifest

    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ValueError(
            f"Output directory is not empty and has no reusable manifest: {config.output_dir}"
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    part_dir = config.output_dir / ".parts"

    input_qc_rows = []
    for entry in sorted(entries, key=lambda item: (item.patient_id, item.antigen_source)):
        result = load_occurrences(entry, policy)
        input_qc_rows.append(
            {
                "patient_id": entry.patient_id,
                "antigen_source": entry.antigen_source,
                "input_format": entry.input_format,
                **result.counts,
                "peptide_path": str(entry.peptide_path),
                "provenance_path": str(entry.provenance_path or ""),
                "run_manifest_path": str(entry.run_manifest_path or ""),
                "status": "eligible_input_validated",
            }
        )
    input_qc_path = config.output_dir / "mimicry_input_qc.tsv"
    write_tsv(input_qc_path, INPUT_QC_FIELDS, input_qc_rows)

    tasks: list[dict[str, object]] = []
    patients = sorted({entry.patient_id for entry in entries})
    for patient_id in patients:
        for source_pair in config.source_pairs:
            source_a, source_b = source_pair.split("--", 1)
            if (patient_id, source_a) not in indexed_entries:
                continue
            for peptide_length in policy.peptide_lengths:
                for shard_id in range(config.shard_count):
                    tasks.append(
                        {
                            "method": policy.name,
                            "patient_id": patient_id,
                            "source_pair": source_pair,
                            "peptide_length": peptide_length,
                            "shard_id": shard_id,
                            "shard_count": config.shard_count,
                            "entry_a": indexed_entries[(patient_id, source_a)],
                            "entry_b": indexed_entries[(patient_id, source_b)],
                            "part_dir": str(part_dir),
                        }
                    )

    worker_count = max(1, min(config.workers, len(tasks), os.cpu_count() or 1))
    if worker_count == 1:
        task_results = [_run_search_shard(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            task_results = list(executor.map(_run_search_shard, tasks, chunksize=1))
    task_results.sort(
        key=lambda row: (
            str(row["patient_id"]),
            str(row["source_pair"]),
            int(row["peptide_length"]),
            int(row["shard_id"]),
        )
    )

    pairs_path = config.output_dir / "mimicry_pairs.tsv.gz"
    events_path = config.output_dir / "mimicry_mutation_wt_evidence.tsv.gz"
    provenance_path = config.output_dir / "mimicry_member_provenance.tsv.gz"
    stagewise_path = config.output_dir / "mimicry_stagewise_qc.tsv"
    pair_count = _merge_shard_outputs(
        [Path(str(row["pair_path"])) for row in task_results], pairs_path, PAIR_FIELDS
    )
    event_count = _merge_shard_outputs(
        [Path(str(row["event_path"])) for row in task_results], events_path, EVENT_FIELDS
    )
    provenance_count = _build_member_provenance(
        pairs_path, entries, policy, provenance_path
    )
    stage_rows = _build_stagewise_qc_rows(task_results)
    write_tsv(stagewise_path, STAGE_FIELDS, stage_rows)

    output_paths = {
        "mimicry_pairs.tsv.gz": pairs_path,
        "mimicry_mutation_wt_evidence.tsv.gz": events_path,
        "mimicry_member_provenance.tsv.gz": provenance_path,
        "mimicry_input_qc.tsv": input_qc_path,
        "mimicry_stagewise_qc.tsv": stagewise_path,
    }
    hla_support_count = 0
    hla_support_status_counts: dict[str, int] = {}
    if config.binding_support_enabled:
        hla_support_path = config.output_dir / "mimicry_hla_support.tsv.gz"
        hla_support_count, hla_support_status_counts = annotate_shared_hla_support(
            pairs_path, entries, hla_support_path
        )
        output_paths["mimicry_hla_support.tsv.gz"] = hla_support_path
    finished_at = datetime.now(timezone.utc)
    manifest: dict[str, object] = {
        "run_status": "complete",
        "method_version": policy.name,
        "primary_call": policy.primary_call,
        "analysis_scope": "within_patient_cross_source_sequence_only",
        "binding_role": "not_used_for_primary_call",
        "hla_role": "not_used_for_primary_call",
        "binding_support": {
            "enabled": config.binding_support_enabled,
            "policy": "shared_classical_hla_i_binding_v1.0"
            if config.binding_support_enabled
            else "not_evaluated",
            "status_counts": hla_support_status_counts,
        },
        "input_signature": input_signature,
        "resolved_policy": {
            "peptide_lengths": list(policy.peptide_lengths),
            "central_identity_threshold": policy.central_identity_threshold,
            "minimum_central_run": policy.minimum_central_run,
            "minimum_positional_matches": policy.minimum_positional_matches,
            "canonical_amino_acids_only": True,
            "legacy_lcs_role": "annotation_only",
        },
        "runtime": {
            "workers": worker_count,
            "shard_count": config.shard_count,
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "row_counts": {
            "passing_unique_pairs": pair_count,
            "mutation_wt_evidence": event_count,
            "member_provenance": provenance_count,
            "input_qc": len(input_qc_rows),
            "stagewise_qc": len(stage_rows),
            "hla_support": hla_support_count,
        },
        "output_signature": {
            name: file_identity(path) for name, path in output_paths.items()
        },
        "reused": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def validate_run_config(config: MimicryRunConfig) -> MimicryRunConfig:
    """Validate and normalize a high-level run configuration."""

    get_policy(config.method)
    if not config.input_manifest.is_file():
        raise FileNotFoundError(config.input_manifest)
    if config.workers < 1:
        raise ValueError("workers must be at least 1")
    if config.shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not config.source_pairs:
        raise ValueError("At least one source pair is required")
    invalid = sorted(set(config.source_pairs) - set(SUPPORTED_SOURCE_PAIRS))
    if invalid:
        raise ValueError(
            f"Unsupported source pairs {invalid}; supported: {list(SUPPORTED_SOURCE_PAIRS)}"
        )
    if len(config.source_pairs) != len(set(config.source_pairs)):
        raise ValueError("source_pairs contains duplicates")
    return replace(config, source_pairs=tuple(sorted(config.source_pairs)))
