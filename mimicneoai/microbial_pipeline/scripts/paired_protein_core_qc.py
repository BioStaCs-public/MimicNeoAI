#!/usr/bin/env python3
"""Build pair-level microbial peptide Core after matched-normal subtraction.

This step consumes per-sample protein-hit tables from
06.MicrobialPeptidesIdentification and writes a tumor-only peptide Core for
binding prediction. Matched-normal subtraction uses all technically valid normal
protein hits; contaminant blacklist filtering is applied afterward to tumor
residual peptides. It does not run HLA typing or binding prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from mimicneoai.microbial_pipeline.scripts.get_data_for_binding_pred import (
    extract_qseqid_tag,
    normalize_parent_sequence,
)


POLICY_VERSION = "microbial_pair_core_qc_v1.0"
CANDIDATE_SELECTION_POLICY_VERSION = "ranked_prebinding_selection_v1.0"
MICROBIAL_RANKING_POLICY = "microbial_source_abundance_v1.0"
RANKED_NORMAL_SCAN_BATCH_SIZE = 50_000
RANKED_PARENT_SCAN_MAX_WORKERS = 20
STANDARD_COLUMNS = {
    "qseqid",
    "sseqid",
    "sseq",
    "pident",
    "evalue",
    "qcovs",
}
MHC_CLASS_I = "MHC-I"
MHC_CLASS_II = "MHC-II"
ALL_MODE = "all"
RANKED_CAP_MODE = "ranked_cap"


def parse_int_list(value: str) -> list[int]:
    lengths = []
    for token in str(value).replace(",", " ").split():
        token = token.strip()
        if token:
            lengths.append(int(token))
    if not lengths:
        raise ValueError("At least one peptide length is required")
    return sorted(set(lengths))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def output_signature(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    return {path.name: file_identity(path) for path in paths}


def outputs_match_signature(signature: dict[str, object]) -> bool:
    for value in signature.values():
        if not isinstance(value, dict):
            return False
        path = Path(str(value.get("path", "")))
        if file_identity(path) != value:
            return False
    return True


def normalize_sseqid(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    if text.startswith("ref|"):
        text = text[4:]
    return text.rstrip("|")


def normalize_accession(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return normalize_sseqid(text)


def normalize_taxid_set(value: object) -> str:
    return ",".join(sorted(set(split_taxids(value))))


def source_group_id(protein_accession: object, ctaxa_ids: object) -> str:
    protein = normalize_accession(protein_accession)
    taxids = normalize_taxid_set(ctaxa_ids)
    return sha256_text(f"{protein}|{taxids}")[:20]


def parse_fragment_id(qseqid: object) -> tuple[str, str]:
    """Parse a stable original fragment identity from a qseqid-like field."""
    if pd.isna(qseqid):
        return "", "missing_qseqid"
    text = str(qseqid).strip()
    if not text:
        return "", "empty_qseqid"
    if text.startswith("protIndex:"):
        parts = text.split("|")
        # Current normalized qseqid format:
        # protIndex:<n>|<sample>|<QNAME>|<alignment_position>|<CIGAR>|YP:Z:...|CTAXA:...|PROT:...
        # Fragment support must be counted by original read name, not by alignment
        # position or protein hit.
        if len(parts) >= 3:
            sample = parts[1].strip()
            qname = parts[2].strip()
            if sample and qname:
                return f"{sample}|{qname}", "parsed"
        text = "|".join(parts[1:]) if len(parts) > 1 else ""
    elif "|YP:Z:" in text:
        parts = text.split("|")
        if len(parts) >= 2:
            sample = parts[0].strip()
            qname = parts[1].strip()
            if sample and qname:
                return f"{sample}|{qname}", "parsed"
    for marker in ("|CTAXA:", "|PROT:"):
        if marker in text:
            text = text.split(marker, 1)[0]
    if " YP:Z:" in text:
        text = text.split(" YP:Z:", 1)[0]
    if "|YP:Z:" in text:
        text = text.split("|YP:Z:", 1)[0]
    text = text.split()[0] if text.split() else ""
    if text.endswith("/1") or text.endswith("/2"):
        text = text[:-2]
    if text.endswith(".1") or text.endswith(".2"):
        # Some paired-read encoders use read-name.1/read-name.2.
        text = text[:-2]
    if not text:
        return "", "unparseable_qseqid"
    return text, "parsed"


def validate_candidate_selection(
    mode: str,
    max_hla_i_peptides: Optional[int],
    max_hla_ii_peptides: Optional[int],
) -> tuple[str, Optional[int], Optional[int]]:
    mode = str(mode or ALL_MODE).strip().lower()
    if mode not in {ALL_MODE, RANKED_CAP_MODE}:
        raise ValueError(f"candidate_selection.mode must be '{ALL_MODE}' or '{RANKED_CAP_MODE}', got {mode!r}")
    if mode == ALL_MODE:
        return mode, None, None
    if max_hla_i_peptides is None or int(max_hla_i_peptides) <= 0:
        raise ValueError("ranked_cap requires positive max_hla_i_peptides")
    if max_hla_ii_peptides is None or int(max_hla_ii_peptides) <= 0:
        raise ValueError("ranked_cap requires positive max_hla_ii_peptides")
    return mode, int(max_hla_i_peptides), int(max_hla_ii_peptides)


def split_taxids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [token.strip() for token in text.split(",") if token.strip()]


def validate_pair_id(pair_id: str, tumor_sample: str, normal_sample: str) -> None:
    if not tumor_sample or not normal_sample:
        raise ValueError("tumor_sample and normal_sample must be non-empty")
    if tumor_sample == normal_sample:
        raise ValueError("tumor_sample and normal_sample must be different")
    if pair_id:
        parts = [part.strip() for part in pair_id.split(",")]
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError("pair_id must be formatted as Tumor,Normal")
        if parts != [tumor_sample, normal_sample]:
            raise ValueError("pair_id does not match tumor_sample and normal_sample")


def load_contaminant_taxids(
    path: Optional[Path],
    *,
    allow_missing: bool = False,
    expected_sha256: str = "",
) -> tuple[set[str], dict[str, str], str]:
    if path is None:
        if allow_missing:
            return set(), {}, ""
        raise FileNotFoundError(
            "Contaminant blacklist is required for paired microbial Core QC. "
            "Use --allow-missing-blacklist only for exploratory runs."
        )
    if not path.exists():
        raise FileNotFoundError(f"Contaminant blacklist not found: {path}")
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            f"Contaminant blacklist SHA256 mismatch for {path}: "
            f"expected {expected_sha256}, observed {actual_sha256}"
        )
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"tax_id", "verdict"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Contaminant blacklist missing required columns: {missing}")
    verdict_by_taxid = {
        str(row.tax_id).strip(): str(row.verdict).strip()
        for row in df.itertuples(index=False)
        if str(row.tax_id).strip()
    }
    contaminants = {
        taxid
        for taxid, verdict in verdict_by_taxid.items()
        if verdict.lower() == "contaminant"
    }
    return contaminants, verdict_by_taxid, actual_sha256


def classify_blacklist_status(taxids: Iterable[str], contaminant_taxids: set[str]) -> str:
    taxid_set = {str(t).strip() for t in taxids if str(t).strip()}
    if not contaminant_taxids:
        return "not_evaluated"
    if not taxid_set:
        return "unresolved"
    contaminant_overlap = taxid_set.intersection(contaminant_taxids)
    if not contaminant_overlap:
        return "not_contaminant"
    if contaminant_overlap == taxid_set:
        return "exclusive_contaminant"
    return "mixed"


def read_protein_hits(
    path: Path,
    sample: str,
    role: str,
    min_pident: float,
    max_evalue: float,
    min_qcovs: float,
    contaminant_taxids: set[str],
    exclude_blacklist: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read new or legacy protein-hit table and split pass/excluded rows."""
    if not path.exists():
        raise FileNotFoundError(f"{role} protein-hit table not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    if df.empty:
        columns = [
            "parent_record_id",
            "sample",
            "role",
            "source_row_number",
            "qseqid",
            "protein_accession",
            "ctaxa_ids",
            "canonical_parent_sequence",
            "parent_sequence_sha256",
            "pident",
            "evalue",
            "qcovs",
            "bitscore",
            "fragment_id",
            "fragment_parse_status",
            "source_group_id",
            "blacklist_status",
            "parent_qc_status",
            "parent_qc_reasons",
            "parent_qc_flags",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    missing_standard = sorted(STANDARD_COLUMNS.difference(df.columns))
    if missing_standard and "canonical_parent_sequence" not in df.columns:
        raise ValueError(
            f"{role} protein-hit table is neither standard nor legacy-compatible; "
            f"missing columns: {missing_standard}"
        )
    if "qcovs" not in df.columns:
        raise ValueError(f"{role} protein-hit table missing required column: qcovs")

    source_numbers = (
        pd.to_numeric(df.get("source_row_number", pd.Series(range(1, len(df) + 1))), errors="coerce")
        .fillna(pd.Series(range(1, len(df) + 1)))
        .astype(int)
    )

    if "canonical_parent_sequence" in df.columns:
        raw_sequence = df["canonical_parent_sequence"]
    else:
        raw_sequence = df["sseq"]

    normalized = raw_sequence.map(normalize_parent_sequence)
    canonical_sequence = [item[0] for item in normalized]
    sequence_reasons = [item[1] for item in normalized]

    pident = pd.to_numeric(df.get("pident"), errors="coerce")
    evalue = pd.to_numeric(df.get("evalue"), errors="coerce")
    qcovs = pd.to_numeric(df.get("qcovs"), errors="coerce")
    bitscore = pd.to_numeric(df.get("bitscore", pd.Series([""] * len(df))), errors="coerce")

    protein_accession = df.get("protein_accession", pd.Series([""] * len(df))).fillna("").astype(str)
    if protein_accession.eq("").all() and "qseqid" in df.columns:
        protein_accession = df["qseqid"].map(lambda value: extract_qseqid_tag(value, "PROT"))
    if protein_accession.eq("").all() and "sseqid" in df.columns:
        protein_accession = df["sseqid"].map(normalize_sseqid)

    ctaxa_ids = df.get("ctaxa_ids", pd.Series([""] * len(df))).fillna("").astype(str)
    if ctaxa_ids.eq("").all() and "qseqid" in df.columns:
        ctaxa_ids = df["qseqid"].map(lambda value: extract_qseqid_tag(value, "CTAXA"))

    qseqid_series = df.get("qseqid", pd.Series([""] * len(df))).fillna("").astype(str)
    fragment_qseqid_series = (
        df["original_qseqid"].fillna("").astype(str)
        if "original_qseqid" in df.columns
        else qseqid_series
    )
    source_number_values = source_numbers.tolist()
    qseqid_values = qseqid_series.tolist()
    fragment_qseqid_values = fragment_qseqid_series.tolist()
    protein_values = protein_accession.tolist()
    taxid_values = ctaxa_ids.tolist()
    pident_values = pident.tolist()
    evalue_values = evalue.tolist()
    qcovs_values = qcovs.tolist()
    bitscore_values = bitscore.tolist()
    parent_hash_cache: dict[str, str] = {}
    source_group_cache: dict[tuple[str, str], str] = {}
    rows = []
    for (
        source_row_number,
        qseqid,
        fragment_qseqid,
        protein,
        taxids_value,
        sequence,
        sequence_reason,
        pident_value,
        evalue_value,
        qcovs_value,
        bitscore_value,
    ) in zip(
        source_number_values,
        qseqid_values,
        fragment_qseqid_values,
        protein_values,
        taxid_values,
        canonical_sequence,
        sequence_reasons,
        pident_values,
        evalue_values,
        qcovs_values,
        bitscore_values,
    ):
        reasons = list(sequence_reason)
        if pd.isna(pident_value) or float(pident_value) != float(min_pident):
            reasons.append(f"pident_not_equal_{min_pident:g}")
        if pd.isna(evalue_value) or float(evalue_value) > float(max_evalue):
            reasons.append(f"evalue_gt_{max_evalue:g}")
        if pd.isna(qcovs_value) or float(qcovs_value) < float(min_qcovs):
            reasons.append(f"qcovs_lt_{min_qcovs:g}")

        taxids = split_taxids(taxids_value)
        blacklist_status = classify_blacklist_status(taxids, contaminant_taxids)
        fragment_id, fragment_parse_status = parse_fragment_id(fragment_qseqid)
        flags = []
        if blacklist_status == "exclusive_contaminant":
            if exclude_blacklist:
                reasons.append("exclusive_contaminant")
            else:
                flags.append("blacklist_exclusive_contaminant_retained_for_subtraction")
        elif blacklist_status in {"mixed", "unresolved"}:
            flags.append(f"blacklist_{blacklist_status}_retained")

        source_row_number = int(source_row_number)
        if sequence:
            parent_hash = parent_hash_cache.get(sequence)
            if parent_hash is None:
                parent_hash = sha256_text(sequence)
                parent_hash_cache[sequence] = parent_hash
        else:
            parent_hash = ""
        source_group_key = (protein, taxids_value)
        source_id = source_group_cache.get(source_group_key)
        if source_id is None:
            source_id = source_group_id(protein, taxids_value)
            source_group_cache[source_group_key] = source_id
        parent_record_id = f"{role}:{sample}:{source_row_number}"
        row = {
            "parent_record_id": parent_record_id,
            "sample": sample,
            "role": role,
            "source_row_number": source_row_number,
            "qseqid": qseqid,
            "protein_accession": protein,
            "ctaxa_ids": taxids_value,
            "canonical_parent_sequence": sequence,
            "parent_sequence_sha256": parent_hash,
            "pident": "" if pd.isna(pident_value) else float(pident_value),
            "evalue": "" if pd.isna(evalue_value) else float(evalue_value),
            "qcovs": "" if pd.isna(qcovs_value) else float(qcovs_value),
            "bitscore": "" if pd.isna(bitscore_value) else float(bitscore_value),
            "fragment_id": fragment_id,
            "fragment_parse_status": fragment_parse_status,
            "source_group_id": source_id,
            "blacklist_status": blacklist_status,
            "parent_qc_status": "excluded" if reasons else "pass",
            "parent_qc_reasons": ";".join(reasons),
            "parent_qc_flags": ";".join(flags),
        }
        rows.append(row)

    all_rows = pd.DataFrame(rows)
    passing = all_rows[all_rows["parent_qc_status"] == "pass"].copy()
    excluded = all_rows[all_rows["parent_qc_status"] != "pass"].copy()
    return passing.reset_index(drop=True), excluded.reset_index(drop=True)


def iter_peptides(sequence: str, lengths: Iterable[int]) -> Iterable[tuple[str, int]]:
    for length in lengths:
        if len(sequence) < length:
            continue
        for start in range(0, len(sequence) - length + 1):
            yield sequence[start : start + length], start + 1


def count_peptide_windows(sequence: str, lengths: Iterable[int]) -> int:
    return sum(max(0, len(sequence) - length + 1) for length in lengths)


def estimate_parent_peptide_windows(
    parents: pd.DataFrame,
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
) -> int:
    if parents.empty:
        return 0
    lengths = sorted(set(mhc_i_lengths + mhc_ii_lengths))
    return int(
        sum(
            count_peptide_windows(str(sequence), lengths)
            for sequence in parents["canonical_parent_sequence"].fillna("")
        )
    )


def build_peptide_maps(
    parents: pd.DataFrame,
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str, str], list[int]]]:
    """Build unique peptide map and peptide-parent position map."""
    peptide_map: dict[tuple[str, str], dict] = {}
    parent_positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for row in parents.itertuples(index=False):
        parent_id = str(row.parent_record_id)
        sequence = str(row.canonical_parent_sequence)
        for mhc_class, lengths in ((MHC_CLASS_I, mhc_i_lengths), (MHC_CLASS_II, mhc_ii_lengths)):
            for peptide, start in iter_peptides(sequence, lengths):
                key = (mhc_class, peptide)
                entry = peptide_map.setdefault(
                    key,
                    {
                        "mhc_class": mhc_class,
                        "peptide": peptide,
                        "peptide_length": len(peptide),
                        "parent_record_ids": set(),
                        "protein_accessions": set(),
                        "ctaxa_ids": set(),
                        "blacklist_statuses": set(),
                    },
                )
                entry["parent_record_ids"].add(parent_id)
                if str(row.protein_accession):
                    entry["protein_accessions"].add(str(row.protein_accession))
                for taxid in split_taxids(row.ctaxa_ids):
                    entry["ctaxa_ids"].add(taxid)
                entry["blacklist_statuses"].add(str(row.blacklist_status))
                parent_positions[(mhc_class, peptide, parent_id)].append(start)

    return peptide_map, parent_positions


def build_candidate_tumor_peptide_maps(
    support_by_parent_hash: dict[str, list[tuple[str, str, str, str, str]]],
    sequence_by_parent_hash: dict[str, str],
    candidate_keys: set[tuple[str, str]],
    source_rank_by_group: dict[str, int],
    tumor_sample: str,
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
    scan_workers: int,
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str, str], list[int]]]:
    """Aggregate all tumor support records for a compact peptide key set."""
    peptide_map: dict[tuple[str, str], dict] = {}
    parent_positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    if not candidate_keys or not support_by_parent_hash:
        return peptide_map, parent_positions

    parent_items = [
        (parent_hash, sequence_by_parent_hash[parent_hash], support_rows)
        for parent_hash, support_rows in support_by_parent_hash.items()
    ]
    worker_count = ranked_parent_scan_workers(len(parent_items), len(candidate_keys), scan_workers)
    if worker_count > 1:
        chunks = split_even_chunks(parent_items, worker_count)
        args = [
            (
                chunk,
                candidate_keys,
                source_rank_by_group,
                tumor_sample,
                mhc_i_lengths,
                mhc_ii_lengths,
            )
            for chunk in chunks
            if chunk
        ]
        with ProcessPoolExecutor(max_workers=len(args)) as pool:
            for chunk_peptide_map, chunk_parent_positions in pool.map(scan_tumor_parent_chunk, args):
                merge_peptide_maps(peptide_map, chunk_peptide_map)
                merge_parent_position_maps(parent_positions, chunk_parent_positions)
    else:
        chunk_peptide_map, chunk_parent_positions = scan_tumor_parent_chunk(
            (
                parent_items,
                candidate_keys,
                source_rank_by_group,
                tumor_sample,
                mhc_i_lengths,
                mhc_ii_lengths,
            )
        )
        merge_peptide_maps(peptide_map, chunk_peptide_map)
        merge_parent_position_maps(parent_positions, chunk_parent_positions)
    return peptide_map, parent_positions


def ranked_parent_scan_workers(parent_count: int, candidate_count: int, scan_workers: int) -> int:
    """Return bounded worker count for ranked parent candidate scans."""
    if parent_count < 20_000 or candidate_count > 5_000:
        return 1
    raw = os.environ.get("MIMICNEOAI_MICROBIAL_CORE_SCAN_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 1
    else:
        requested = scan_workers
    return max(1, min(requested, RANKED_PARENT_SCAN_MAX_WORKERS, parent_count))


def split_even_chunks(items: list, n_chunks: int) -> list[list]:
    if n_chunks <= 1:
        return [items]
    chunk_size = max(1, (len(items) + n_chunks - 1) // n_chunks)
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def candidate_sequence_hits(
    sequence: str,
    candidate_keys: set[tuple[str, str]],
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
) -> dict[tuple[str, str], list[int]]:
    hits: dict[tuple[str, str], list[int]] = defaultdict(list)
    if len(candidate_keys) <= 512:
        for mhc_class, peptide in candidate_keys:
            start0 = sequence.find(peptide)
            while start0 >= 0:
                hits[(mhc_class, peptide)].append(start0 + 1)
                start0 = sequence.find(peptide, start0 + 1)
        return hits
    for mhc_class, lengths in ((MHC_CLASS_I, mhc_i_lengths), (MHC_CLASS_II, mhc_ii_lengths)):
        for peptide, start in iter_peptides(sequence, lengths):
            key = (mhc_class, peptide)
            if key in candidate_keys:
                hits[key].append(start)
    return hits


def ranked_candidate_scan_workers(candidate_count: int, scan_workers: int) -> int:
    """Return bounded worker count for tumor peptide chunk scans against normal."""
    if candidate_count < 256:
        return 1
    raw = os.environ.get("MIMICNEOAI_MICROBIAL_CORE_SCAN_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 1
    else:
        requested = scan_workers
    return max(1, min(requested, RANKED_PARENT_SCAN_MAX_WORKERS, candidate_count))


def scan_tumor_parent_chunk(args):
    (
        parent_items,
        candidate_keys,
        source_rank_by_group,
        tumor_sample,
        mhc_i_lengths,
        mhc_ii_lengths,
    ) = args
    peptide_map: dict[tuple[str, str], dict] = {}
    parent_positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for _parent_hash, sequence, support_rows in parent_items:
        hits = candidate_sequence_hits(sequence, candidate_keys, mhc_i_lengths, mhc_ii_lengths)
        if not hits:
            continue
        for parent_id, source_id, protein, taxids_value, blacklist_status in support_rows:
            source_rank = int(source_rank_by_group.get(source_id, 0) or 0)
            for key, positions in hits.items():
                mhc_class, peptide = key
                entry = peptide_map.setdefault(
                    key,
                    {
                        "mhc_class": mhc_class,
                        "peptide": peptide,
                        "peptide_length": len(peptide),
                        "parent_record_ids": set(),
                        "protein_accessions": set(),
                        "ctaxa_ids": set(),
                        "blacklist_statuses": set(),
                        "source_group_ids": set(),
                        "best_source_rank": source_rank,
                        "selection_digest": stable_selection_digest(tumor_sample, mhc_class, peptide),
                    },
                )
                if source_rank and (
                    not entry.get("best_source_rank")
                    or source_rank < int(entry.get("best_source_rank", source_rank))
                ):
                    entry["best_source_rank"] = source_rank
                entry["parent_record_ids"].add(parent_id)
                entry["source_group_ids"].add(source_id)
                if protein:
                    entry["protein_accessions"].add(protein)
                for taxid in split_taxids(taxids_value):
                    entry["ctaxa_ids"].add(taxid)
                entry["blacklist_statuses"].add(blacklist_status)
                parent_positions[(mhc_class, peptide, parent_id)].extend(positions)
    return peptide_map, dict(parent_positions)


def scan_normal_for_candidate_chunk(args):
    (
        candidate_keys,
        normal_items,
        mhc_i_lengths,
        mhc_ii_lengths,
    ) = args
    candidate_key_set = set(candidate_keys)
    matches: dict[tuple[str, str], dict] = {}
    for _parent_hash, sequence, support_rows in normal_items:
        hits = candidate_sequence_hits(sequence, candidate_key_set, mhc_i_lengths, mhc_ii_lengths)
        if not hits:
            continue
        for parent_id, _source_id, protein, taxids_value, _blacklist_status in support_rows:
            for key in hits:
                entry = matches.setdefault(
                    key,
                    {
                        "parent_record_ids": set(),
                        "protein_accessions": set(),
                        "ctaxa_ids": set(),
                    },
                )
                entry["parent_record_ids"].add(parent_id)
                if protein:
                    entry["protein_accessions"].add(protein)
                for taxid in split_taxids(taxids_value):
                    entry["ctaxa_ids"].add(taxid)
    return matches


def merge_peptide_maps(target: dict[tuple[str, str], dict], source: dict[tuple[str, str], dict]) -> None:
    for key, entry in source.items():
        existing = target.setdefault(key, entry)
        if existing is entry:
            continue
        for field in ("parent_record_ids", "protein_accessions", "ctaxa_ids", "blacklist_statuses", "source_group_ids"):
            existing[field].update(entry.get(field, set()))
        incoming_rank = int(entry.get("best_source_rank", 0) or 0)
        current_rank = int(existing.get("best_source_rank", 0) or 0)
        if incoming_rank and (not current_rank or incoming_rank < current_rank):
            existing["best_source_rank"] = incoming_rank


def merge_parent_position_maps(
    target: dict[tuple[str, str, str], list[int]],
    source: dict[tuple[str, str, str], list[int]],
) -> None:
    for key, values in source.items():
        target[key].extend(values)


def build_tumor_parent_support_index(
    parents: pd.DataFrame,
) -> tuple[dict[str, list[tuple[str, str, str, str, str]]], dict[str, str]]:
    """Build reusable tumor parent support grouped by parent sequence hash."""
    support_by_parent_hash: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    sequence_by_parent_hash: dict[str, str] = {}
    if parents.empty:
        return support_by_parent_hash, sequence_by_parent_hash
    required = [
        "parent_sequence_sha256",
        "canonical_parent_sequence",
        "parent_record_id",
        "source_group_id",
        "protein_accession",
        "ctaxa_ids",
        "blacklist_status",
    ]
    work = parents[required].fillna("").astype(str)
    for (
        parent_hash,
        sequence,
        parent_id,
        source_id,
        protein,
        taxids_value,
        blacklist_status,
    ) in zip(
        work["parent_sequence_sha256"].tolist(),
        work["canonical_parent_sequence"].tolist(),
        work["parent_record_id"].tolist(),
        work["source_group_id"].tolist(),
        work["protein_accession"].tolist(),
        work["ctaxa_ids"].tolist(),
        work["blacklist_status"].tolist(),
    ):
        support_by_parent_hash[parent_hash].append(
            (parent_id, source_id, protein, taxids_value, blacklist_status)
        )
        sequence_by_parent_hash.setdefault(parent_hash, sequence)
    return support_by_parent_hash, sequence_by_parent_hash


def peptide_entries_to_df(
    entries: list[dict],
    pair_id: str,
    tumor_sample: str,
    normal_sample: str,
    status: str,
    reason: str = "",
) -> pd.DataFrame:
    columns = [
        "peptide_id",
        "pair_id",
        "tumor_sample",
        "normal_sample",
        "mhc_class",
        "peptide",
        "peptide_length",
        "peptide_qc_status",
        "peptide_qc_reasons",
        "peptide_qc_flags",
        "parent_record_count",
        "protein_count",
        "taxon_count",
    ]
    rows = []
    for idx, entry in enumerate(entries, start=1):
        rows.append(
            {
                "peptide_id": f"microbial_core_{idx:07d}" if status == "core" else "",
                "pair_id": pair_id,
                "tumor_sample": tumor_sample,
                "normal_sample": normal_sample,
                "mhc_class": entry["mhc_class"],
                "peptide": entry["peptide"],
                "peptide_length": entry["peptide_length"],
                "peptide_qc_status": status,
                "peptide_qc_reasons": entry.get("peptide_qc_reasons", reason),
                "peptide_qc_flags": entry.get("peptide_qc_flags", ""),
                "parent_record_count": len(entry["parent_record_ids"]),
                "protein_count": len(entry["protein_accessions"]),
                "taxon_count": len(entry["ctaxa_ids"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def peptide_blacklist_status(entry: dict) -> str:
    statuses = {str(s) for s in entry.get("blacklist_statuses", set()) if str(s)}
    if statuses and statuses == {"exclusive_contaminant"}:
        return "exclusive_contaminant"
    flags = []
    if "exclusive_contaminant" in statuses:
        flags.append("blacklist_partial_contaminant_retained")
    if "mixed" in statuses:
        flags.append("blacklist_mixed_retained")
    if "unresolved" in statuses:
        flags.append("blacklist_unresolved_retained")
    entry["peptide_qc_flags"] = ";".join(flags)
    return "retained"


def build_parent_map_df(
    core_entries: list[dict],
    excluded_entries: list[dict],
    parent_positions: dict[tuple[str, str, str], list[int]],
) -> pd.DataFrame:
    columns = [
        "mhc_class",
        "peptide",
        "peptide_length",
        "peptide_qc_status",
        "parent_record_id",
        "positions_1based",
    ]
    rows = []
    for status, entries in (("core", core_entries), ("excluded", excluded_entries)):
        for entry in entries:
            for parent_id in sorted(entry["parent_record_ids"]):
                positions = parent_positions.get((entry["mhc_class"], entry["peptide"], parent_id), [])
                rows.append(
                    {
                        "mhc_class": entry["mhc_class"],
                        "peptide": entry["peptide"],
                        "peptide_length": entry["peptide_length"],
                        "peptide_qc_status": status,
                        "parent_record_id": parent_id,
                        "positions_1based": ",".join(str(pos) for pos in sorted(set(positions))),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def build_matched_normal_df(
    excluded_entries: list[dict],
    normal_peptide_map: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    columns = [
        "mhc_class",
        "peptide",
        "peptide_length",
        "normal_parent_record_count",
        "normal_protein_accessions",
        "normal_ctaxa_ids",
        "normal_parent_record_ids",
    ]
    rows = []
    for entry in excluded_entries:
        normal_entry = normal_peptide_map.get((entry["mhc_class"], entry["peptide"]), {})
        rows.append(
            {
                "mhc_class": entry["mhc_class"],
                "peptide": entry["peptide"],
                "peptide_length": entry["peptide_length"],
                "normal_parent_record_count": len(normal_entry.get("parent_record_ids", set())),
                "normal_protein_accessions": ",".join(sorted(normal_entry.get("protein_accessions", set()))),
                "normal_ctaxa_ids": ",".join(sorted(normal_entry.get("ctaxa_ids", set()))),
                "normal_parent_record_ids": ",".join(sorted(normal_entry.get("parent_record_ids", set()))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_fasta(core_df: pd.DataFrame, mhc_class: str, path: Path) -> None:
    subset = core_df[core_df["mhc_class"] == mhc_class].copy()
    lines = []
    for row in subset.itertuples(index=False):
        lines.append(f">{row.peptide_id}|{row.mhc_class}|len={row.peptide_length}")
        lines.append(str(row.peptide))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_combined_fasta(core_df: pd.DataFrame, path: Path) -> None:
    lines = []
    for row in core_df.itertuples(index=False):
        lines.append(f">{row.peptide_id}|{row.mhc_class}|len={row.peptide_length}")
        lines.append(str(row.peptide))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)


def write_stagewise_qc(path: Path, rows: list[tuple[str, int]]) -> None:
    pd.DataFrame([{"step": step, "n": int(count)} for step, count in rows]).to_csv(
        path,
        sep="\t",
        index=False,
    )


def stable_selection_digest(sample: str, mhc_class: str, peptide: str) -> str:
    return sha256_text(f"{sample}|microbial|{mhc_class}|{len(peptide)}|{peptide}")


def ensure_ranked_fragment_parseable(parents: pd.DataFrame, role: str) -> None:
    bad = parents[parents["fragment_parse_status"].astype(str).ne("parsed")]
    if not bad.empty:
        examples = ",".join(bad["qseqid"].astype(str).head(3))
        raise ValueError(
            f"{role} qseqid could not be parsed for ranked_cap fragment counting; "
            f"examples: {examples}"
        )


def build_source_rank_table(
    tumor_core: pd.DataFrame,
    normal_pass: pd.DataFrame,
    *,
    abundance_pseudocount: float,
) -> pd.DataFrame:
    """Build deterministic microbial source ranking table."""
    columns = [
        "source_group_id",
        "protein_accession",
        "ctaxa_ids",
        "tumor_unique_fragment_count",
        "normal_unique_fragment_count",
        "tumor_abundance_per_million",
        "tumor_normal_enrichment",
        "best_qcovs",
        "best_bitscore",
        "best_evalue",
        "source_digest",
        "source_rank",
        "source_selection_status",
    ]
    if tumor_core.empty:
        return pd.DataFrame(columns=columns)

    total_tumor_fragments = max(1, tumor_core["fragment_id"].astype(str).nunique())
    normal_counts = (
        normal_pass.groupby("source_group_id")["fragment_id"].nunique().to_dict()
        if not normal_pass.empty
        else {}
    )
    work = tumor_core[
        [
            "source_group_id",
            "protein_accession",
            "ctaxa_ids",
            "fragment_id",
            "bitscore",
            "qcovs",
            "evalue",
        ]
    ].copy()
    work["_protein_norm"] = work["protein_accession"].fillna("").map(normalize_accession)
    work["_taxids_norm"] = work["ctaxa_ids"].fillna("").map(normalize_taxid_set)
    work["_bitscore_num"] = pd.to_numeric(work["bitscore"], errors="coerce")
    work["_qcovs_num"] = pd.to_numeric(work["qcovs"], errors="coerce")
    work["_evalue_num"] = pd.to_numeric(work["evalue"], errors="coerce")
    ranked = (
        work.groupby("source_group_id", sort=False)
        .agg(
            protein_accession=("_protein_norm", "first"),
            ctaxa_ids=("_taxids_norm", "first"),
            tumor_unique_fragment_count=("fragment_id", "nunique"),
            best_bitscore=("_bitscore_num", "max"),
            best_qcovs=("_qcovs_num", "max"),
            best_evalue=("_evalue_num", "min"),
        )
        .reset_index()
    )
    ranked["normal_unique_fragment_count"] = ranked["source_group_id"].map(normal_counts).fillna(0).astype(int)
    ranked["tumor_unique_fragment_count"] = ranked["tumor_unique_fragment_count"].astype(int)
    ranked["tumor_abundance_per_million"] = (
        ranked["tumor_unique_fragment_count"] / total_tumor_fragments * 1_000_000
    )
    ranked["tumor_normal_enrichment"] = (
        (ranked["tumor_unique_fragment_count"] + abundance_pseudocount)
        / (ranked["normal_unique_fragment_count"] + abundance_pseudocount)
    )
    ranked["source_digest"] = [
        sha256_text(f"{protein}|{taxids}|{source_id}")
        for protein, taxids, source_id in zip(
            ranked["protein_accession"].astype(str),
            ranked["ctaxa_ids"].astype(str),
            ranked["source_group_id"].astype(str),
        )
    ]
    ranked["source_rank"] = 0
    ranked["source_selection_status"] = "not_evaluated"
    ranked = ranked[columns]
    ranked["_sort_best_bitscore"] = pd.to_numeric(ranked["best_bitscore"], errors="coerce").fillna(-1)
    ranked["_sort_best_qcovs"] = pd.to_numeric(ranked["best_qcovs"], errors="coerce").fillna(-1)
    ranked["_sort_best_evalue"] = pd.to_numeric(ranked["best_evalue"], errors="coerce").fillna(float("inf"))
    ranked = ranked.sort_values(
        [
            "tumor_unique_fragment_count",
            "tumor_abundance_per_million",
            "tumor_normal_enrichment",
            "normal_unique_fragment_count",
            "_sort_best_qcovs",
            "_sort_best_bitscore",
            "_sort_best_evalue",
            "protein_accession",
            "source_digest",
        ],
        ascending=[False, False, False, True, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["source_rank"] = range(1, len(ranked) + 1)
    return ranked.drop(columns=["_sort_best_bitscore", "_sort_best_qcovs", "_sort_best_evalue"])


def entries_for_source_group(
    group: pd.DataFrame,
    source_rank: int,
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
) -> dict[tuple[str, str], dict]:
    """Generate unique peptide entries for one ranked source group."""
    entries: dict[tuple[str, str], dict] = {}
    seen_parent_sequences: set[str] = set()
    for row in group.itertuples(index=False):
        parent_hash = str(row.parent_sequence_sha256)
        if parent_hash in seen_parent_sequences:
            continue
        seen_parent_sequences.add(parent_hash)
        sequence = str(row.canonical_parent_sequence)
        for mhc_class, lengths in ((MHC_CLASS_I, mhc_i_lengths), (MHC_CLASS_II, mhc_ii_lengths)):
            for peptide, start in iter_peptides(sequence, lengths):
                key = (mhc_class, peptide)
                entry = entries.setdefault(
                    key,
                    {
                        "mhc_class": mhc_class,
                        "peptide": peptide,
                        "peptide_length": len(peptide),
                        "parent_record_ids": set(),
                        "protein_accessions": set(),
                        "ctaxa_ids": set(),
                        "blacklist_statuses": set(),
                        "source_group_ids": set(),
                        "best_source_rank": source_rank,
                        "selection_digest": stable_selection_digest(str(row.sample), mhc_class, peptide),
                    },
                )
                entry["parent_record_ids"].add(str(row.parent_record_id))
                entry["source_group_ids"].add(str(row.source_group_id))
                if str(row.protein_accession):
                    entry["protein_accessions"].add(str(row.protein_accession))
                for taxid in split_taxids(row.ctaxa_ids):
                    entry["ctaxa_ids"].add(taxid)
                entry["blacklist_statuses"].add(str(row.blacklist_status))
    return entries


def normal_exact_matches_for_candidates(
    normal_support_by_parent_hash: dict[str, list[tuple[str, str, str, str, str]]],
    normal_sequence_by_parent_hash: dict[str, str],
    candidate_keys: set[tuple[str, str]],
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
    scan_workers: int,
) -> dict[tuple[str, str], dict]:
    """Scan tumor peptide chunks against the full technically eligible normal space."""
    matches: dict[tuple[str, str], dict] = {}
    if not candidate_keys or not normal_support_by_parent_hash:
        return matches
    normal_items = [
        (parent_hash, normal_sequence_by_parent_hash[parent_hash], support_rows)
        for parent_hash, support_rows in normal_support_by_parent_hash.items()
    ]
    candidate_key_list = sorted(candidate_keys, key=lambda item: (item[0], len(item[1]), item[1]))
    worker_count = ranked_candidate_scan_workers(len(candidate_key_list), scan_workers)
    if worker_count > 1:
        candidate_chunks = split_even_chunks(candidate_key_list, worker_count)
        args = [
            (
                chunk,
                normal_items,
                mhc_i_lengths,
                mhc_ii_lengths,
            )
            for chunk in candidate_chunks
            if chunk
        ]
        with ProcessPoolExecutor(max_workers=len(args)) as pool:
            for chunk_matches in pool.map(scan_normal_for_candidate_chunk, args):
                merge_normal_match_maps(matches, chunk_matches)
    else:
        chunk_matches = scan_normal_for_candidate_chunk(
            (
                candidate_key_list,
                normal_items,
                mhc_i_lengths,
                mhc_ii_lengths,
            )
        )
        merge_normal_match_maps(matches, chunk_matches)
    return matches


def merge_normal_match_maps(target: dict[tuple[str, str], dict], source: dict[tuple[str, str], dict]) -> None:
    for key, entry in source.items():
        merged = target.setdefault(
            key,
            {
                "parent_record_ids": set(),
                "protein_accessions": set(),
                "ctaxa_ids": set(),
            },
        )
        for field in ("parent_record_ids", "protein_accessions", "ctaxa_ids"):
            merged[field].update(entry.get(field, set()))


def build_ranked_parent_map_df(
    tumor_core: pd.DataFrame,
    selected_keys: set[tuple[str, str]],
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
) -> pd.DataFrame:
    columns = [
        "mhc_class",
        "peptide",
        "peptide_length",
        "peptide_qc_status",
        "parent_record_id",
        "positions_1based",
    ]
    rows: list[dict[str, object]] = []
    parent_map_cols = ["parent_record_id", "canonical_parent_sequence"]
    for row in tumor_core[parent_map_cols].itertuples(index=False):
        seen_positions: dict[tuple[str, str], list[int]] = defaultdict(list)
        sequence = str(row.canonical_parent_sequence)
        for mhc_class, lengths in ((MHC_CLASS_I, mhc_i_lengths), (MHC_CLASS_II, mhc_ii_lengths)):
            for peptide, start in iter_peptides(sequence, lengths):
                key = (mhc_class, peptide)
                if key in selected_keys:
                    seen_positions[key].append(start)
        for (mhc_class, peptide), positions in seen_positions.items():
            rows.append(
                {
                    "mhc_class": mhc_class,
                    "peptide": peptide,
                    "peptide_length": len(peptide),
                    "peptide_qc_status": "core",
                    "parent_record_id": row.parent_record_id,
                    "positions_1based": ",".join(str(pos) for pos in sorted(set(positions))),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_ranked_pair_core_outputs(
    tumor_core: pd.DataFrame,
    normal_pass: pd.DataFrame,
    outdir: Path,
    pair_id: str,
    tumor_sample: str,
    normal_sample: str,
    mhc_i_lengths: list[int],
    mhc_ii_lengths: list[int],
    max_hla_i_peptides: int,
    max_hla_ii_peptides: int,
    abundance_pseudocount: float,
    scan_workers: int,
) -> dict[str, object]:
    ensure_ranked_fragment_parseable(tumor_core, "tumor")
    ensure_ranked_fragment_parseable(normal_pass, "normal")
    source_rank = build_source_rank_table(
        tumor_core,
        normal_pass,
        abundance_pseudocount=abundance_pseudocount,
    )
    caps = {MHC_CLASS_I: max_hla_i_peptides, MHC_CLASS_II: max_hla_ii_peptides}
    selected_by_class: dict[str, dict[tuple[str, str], dict]] = {
        MHC_CLASS_I: {},
        MHC_CLASS_II: {},
    }
    selected_entries_by_key: dict[tuple[str, str], dict] = {}
    matched_normal_excluded_by_key: dict[tuple[str, str], dict] = {}
    blacklist_excluded_by_key: dict[tuple[str, str], dict] = {}
    peptide_evidence_rows: list[dict[str, object]] = []
    examined_keys: set[tuple[str, str]] = set()
    pending_order: list[tuple[tuple[str, str], str, int]] = []
    pending_keys: set[tuple[str, str]] = set()
    source_status_counts: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_rank_by_group = {
        str(row.source_group_id): int(row.source_rank)
        for row in source_rank.itertuples(index=False)
    }
    tumor_core_by_source = {
        str(source_id): group
        for source_id, group in tumor_core.groupby("source_group_id", sort=False)
    }
    source_selection_status_by_group: dict[str, str] = {}
    normal_support_by_parent_hash, normal_sequence_by_parent_hash = build_tumor_parent_support_index(normal_pass)
    tumor_support_by_parent_hash, tumor_sequence_by_parent_hash = build_tumor_parent_support_index(tumor_core)
    ranked_normal_scan_batch_size = min(
        RANKED_NORMAL_SCAN_BATCH_SIZE,
        max(1_000, (max_hla_i_peptides + max_hla_ii_peptides) * 5),
    )
    parent_positions_accum: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    normal_match_map_accum: dict[tuple[str, str], dict] = {}
    boundary_source = ""
    normal_scan_batches = 0

    def class_has_capacity(mhc_class: str) -> bool:
        return len(selected_by_class[mhc_class]) < caps[mhc_class]

    def caps_full() -> bool:
        return all(len(selected_by_class[mhc]) >= caps[mhc] for mhc in caps)

    def merge_parent_positions(source: dict[tuple[str, str, str], list[int]]) -> None:
        for key, values in source.items():
            parent_positions_accum[key].extend(values)

    def merge_normal_matches(source: dict[tuple[str, str], dict]) -> None:
        for key, entry in source.items():
            target = normal_match_map_accum.setdefault(
                key,
                {
                    "parent_record_ids": set(),
                    "protein_accessions": set(),
                    "ctaxa_ids": set(),
                },
            )
            for field in ("parent_record_ids", "protein_accessions", "ctaxa_ids"):
                target[field].update(entry.get(field, set()))

    def add_evidence(
        key: tuple[str, str],
        source_group_id: str,
        source_rank_value: int,
        status: str,
        reason: str,
    ) -> None:
        source_status_counts[source_group_id][status] += 1
        peptide_evidence_rows.append(
            {
                "mhc_class": key[0],
                "peptide": key[1],
                "peptide_length": len(key[1]),
                "source_group_id": source_group_id,
                "source_rank": source_rank_value,
                "selection_status": status,
                "selection_reason": reason,
                "selection_digest": stable_selection_digest(tumor_sample, key[0], key[1]),
            }
        )

    def flush_pending() -> None:
        nonlocal boundary_source, normal_scan_batches
        if not pending_order:
            return
        batch_keys = set(pending_keys)
        normal_matches = normal_exact_matches_for_candidates(
            normal_support_by_parent_hash,
            normal_sequence_by_parent_hash,
            batch_keys,
            mhc_i_lengths,
            mhc_ii_lengths,
            scan_workers,
        )
        normal_scan_batches += 1
        merge_normal_matches(normal_matches)
        tumor_entries, parent_positions = build_candidate_tumor_peptide_maps(
            tumor_support_by_parent_hash,
            tumor_sequence_by_parent_hash,
            batch_keys,
            source_rank_by_group,
            tumor_sample,
            mhc_i_lengths,
            mhc_ii_lengths,
            scan_workers,
        )
        merge_parent_positions(parent_positions)

        for key, source_group_id, source_rank_value in pending_order:
            entry = tumor_entries.get(key)
            if entry is None:
                add_evidence(
                    key,
                    source_group_id,
                    source_rank_value,
                    "excluded_by_upstream_qc",
                    "candidate_support_not_recoverable",
                )
                continue
            reason = ""
            status = "selected_for_binding"
            if key in normal_matches:
                reason = "excluded_exact_peptide_in_matched_normal"
                status = "excluded_exact_matched_normal"
                entry["peptide_qc_reasons"] = reason
                matched_normal_excluded_by_key[key] = entry
            elif peptide_blacklist_status(entry) == "exclusive_contaminant":
                reason = "excluded_exclusive_contaminant"
                status = "excluded_by_upstream_qc"
                entry["peptide_qc_reasons"] = reason
                blacklist_excluded_by_key[key] = entry
            elif not class_has_capacity(key[0]):
                reason = "not_selected_due_to_analysis_cap"
                status = "not_selected_due_to_analysis_cap"
                if not boundary_source:
                    boundary_source = source_group_id
            else:
                selected_by_class[key[0]][key] = entry
                selected_entries_by_key[key] = entry
            add_evidence(key, source_group_id, source_rank_value, status, reason)

        pending_order.clear()
        pending_keys.clear()

    def ordered_source_keys(group: pd.DataFrame) -> list[tuple[str, str]]:
        entries = entries_for_source_group(group, 0, mhc_i_lengths, mhc_ii_lengths)
        return sorted(
            entries,
            key=lambda item: (item[0], len(item[1]), stable_selection_digest(tumor_sample, item[0], item[1])),
        )

    for rank_row in source_rank.itertuples(index=False):
        source_group = str(rank_row.source_group_id)
        if caps_full():
            source_selection_status_by_group[source_group] = "not_selected_due_to_analysis_cap"
            continue
        group = tumor_core_by_source.get(source_group)
        if group is None:
            source_selection_status_by_group[source_group] = "no_source_rows"
            continue
        candidate_keys = [
            key
            for key in ordered_source_keys(group)
            if key not in examined_keys and key[0] in caps and class_has_capacity(key[0])
        ]
        if not candidate_keys:
            source_selection_status_by_group[source_group] = "no_new_candidate_before_cap"
            continue
        source_rank_value = int(rank_row.source_rank)
        for key in candidate_keys:
            pending_order.append((key, source_group, source_rank_value))
            pending_keys.add(key)
            examined_keys.add(key)
            if len(pending_order) >= ranked_normal_scan_batch_size:
                flush_pending()
                if caps_full():
                    break
        source_selection_status_by_group[source_group] = "examined_for_selection"
        if caps_full():
            continue

    flush_pending()
    if source_selection_status_by_group:
        mapped_status = source_rank["source_group_id"].astype(str).map(source_selection_status_by_group)
        source_rank["source_selection_status"] = mapped_status.fillna(source_rank["source_selection_status"])

    for idx, row in source_rank.iterrows():
        source_id = str(row["source_group_id"])
        counts = source_status_counts.get(source_id, {})
        if not counts:
            continue
        if counts.get("selected_for_binding", 0) and counts.get("not_selected_due_to_analysis_cap", 0):
            source_rank.at[idx, "source_selection_status"] = "boundary_partially_selected"
        elif counts.get("selected_for_binding", 0):
            source_rank.at[idx, "source_selection_status"] = "selected_for_binding"
        elif counts.get("not_selected_due_to_analysis_cap", 0):
            source_rank.at[idx, "source_selection_status"] = "not_selected_due_to_analysis_cap"
        elif counts.get("excluded_exact_matched_normal", 0) or counts.get("excluded_by_upstream_qc", 0):
            source_rank.at[idx, "source_selection_status"] = "examined_all_excluded"
        else:
            source_rank.at[idx, "source_selection_status"] = "examined_no_selected_peptides"

    core_entries = [
        selected_entries_by_key[key]
        for key in sorted(
            selected_entries_by_key,
            key=lambda item: (item[0], len(item[1]), stable_selection_digest(tumor_sample, item[0], item[1])),
        )
    ]
    matched_normal_excluded_entries = [
        matched_normal_excluded_by_key[key]
        for key in sorted(
            matched_normal_excluded_by_key,
            key=lambda item: (item[0], len(item[1]), stable_selection_digest(tumor_sample, item[0], item[1])),
        )
    ]
    blacklist_excluded_entries = [
        blacklist_excluded_by_key[key]
        for key in sorted(
            blacklist_excluded_by_key,
            key=lambda item: (item[0], len(item[1]), stable_selection_digest(tumor_sample, item[0], item[1])),
        )
    ]
    excluded_entries = matched_normal_excluded_entries + blacklist_excluded_entries
    core_df = peptide_entries_to_df(core_entries, pair_id, tumor_sample, normal_sample, "core")
    excluded_peptide_df = peptide_entries_to_df(excluded_entries, pair_id, tumor_sample, normal_sample, "excluded")
    selected_keys = set(selected_entries_by_key)
    parent_map_df = build_parent_map_df(core_entries, excluded_entries, parent_positions_accum)
    matched_normal_df = build_matched_normal_df(matched_normal_excluded_entries, normal_match_map_accum)
    source_rank.to_csv(outdir / "microbial_ranked_source.tsv", sep="\t", index=False)
    pd.DataFrame(peptide_evidence_rows).to_csv(
        outdir / "microbial_peptide_selection_evidence.tsv",
        sep="\t",
        index=False,
    )
    deferred = source_rank[source_rank["source_selection_status"].eq("not_selected_due_to_analysis_cap")]
    deferred.to_csv(outdir / "microbial_deferred_source.tsv", sep="\t", index=False)
    return {
        "core_df": core_df,
        "excluded_peptide_df": excluded_peptide_df,
        "parent_map_df": parent_map_df,
        "matched_normal_df": matched_normal_df,
        "source_rank_df": source_rank,
        "peptide_evidence_rows": peptide_evidence_rows,
        "boundary_source_group_id": boundary_source,
        "examined_candidate_peptides": len(examined_keys),
        "selected_hla_i": len(selected_by_class[MHC_CLASS_I]),
        "selected_hla_ii": len(selected_by_class[MHC_CLASS_II]),
        "matched_normal_excluded": len(matched_normal_excluded_entries),
        "blacklist_excluded": len(blacklist_excluded_entries),
        "normal_scan_batches": normal_scan_batches,
        "ranked_normal_scan_batch_size": ranked_normal_scan_batch_size,
        "cap_reached_hla_i": len(selected_by_class[MHC_CLASS_I]) >= caps[MHC_CLASS_I],
        "cap_reached_hla_ii": len(selected_by_class[MHC_CLASS_II]) >= caps[MHC_CLASS_II],
        "deferred_source_groups": int((source_rank["source_selection_status"] == "not_selected_due_to_analysis_cap").sum()),
    }


def build_pair_core(
    tumor_protein_hits: Path,
    normal_protein_hits: Path,
    outdir: Path,
    tumor_sample: str,
    normal_sample: str,
    pair_id: Optional[str] = None,
    blacklist: Optional[Path] = None,
    mhc_i_lengths: Optional[list[int]] = None,
    mhc_ii_lengths: Optional[list[int]] = None,
    min_pident: float = 100.0,
    max_evalue: float = 1e-5,
    min_qcovs: float = 90.0,
    allow_missing_blacklist: bool = False,
    blacklist_sha256: str = "",
    max_estimated_peptide_windows: int = 20_000_000,
    candidate_selection_mode: str = ALL_MODE,
    max_hla_i_peptides: Optional[int] = None,
    max_hla_ii_peptides: Optional[int] = None,
    ranking_abundance_pseudocount: float = 0.5,
    scan_workers: int = RANKED_PARENT_SCAN_MAX_WORKERS,
) -> dict:
    mhc_i_lengths = mhc_i_lengths or [8, 9, 10, 11]
    mhc_ii_lengths = mhc_ii_lengths or [13, 14, 15, 16, 17]
    pair_id = pair_id or f"{tumor_sample},{normal_sample}"
    validate_pair_id(pair_id, tumor_sample, normal_sample)
    candidate_selection_mode, max_hla_i_peptides, max_hla_ii_peptides = validate_candidate_selection(
        candidate_selection_mode,
        max_hla_i_peptides,
        max_hla_ii_peptides,
    )
    scan_workers = max(1, min(int(scan_workers), RANKED_PARENT_SCAN_MAX_WORKERS))
    outdir.mkdir(parents=True, exist_ok=True)

    contaminant_taxids, verdict_by_taxid, actual_blacklist_sha256 = load_contaminant_taxids(
        blacklist,
        allow_missing=allow_missing_blacklist,
        expected_sha256=blacklist_sha256,
    )
    input_signature = {
        "policy_version": POLICY_VERSION,
        "script_sha256": sha256_file(Path(__file__)),
        "pair_id": pair_id,
        "tumor_sample": tumor_sample,
        "normal_sample": normal_sample,
        "tumor_protein_hits": file_identity(tumor_protein_hits),
        "normal_protein_hits": file_identity(normal_protein_hits),
        "blacklist": file_identity(blacklist) if blacklist else {},
        "blacklist_sha256": actual_blacklist_sha256,
        "blacklist_sha256_expected": blacklist_sha256,
        "allow_missing_blacklist": allow_missing_blacklist,
        "mhc_i_lengths": mhc_i_lengths,
        "mhc_ii_lengths": mhc_ii_lengths,
        "min_pident": float(min_pident),
        "max_evalue": float(max_evalue),
        "min_qcovs": float(min_qcovs),
        "paired_core_max_estimated_peptide_windows": int(max_estimated_peptide_windows),
        "candidate_selection_policy_version": CANDIDATE_SELECTION_POLICY_VERSION,
        "candidate_selection": {
            "mode": candidate_selection_mode,
            "max_hla_i_peptides": max_hla_i_peptides,
            "max_hla_ii_peptides": max_hla_ii_peptides,
            "ranking_policy": MICROBIAL_RANKING_POLICY if candidate_selection_mode == RANKED_CAP_MODE else "",
            "ranking_abundance_pseudocount": float(ranking_abundance_pseudocount),
            "ranked_normal_scan_batch_size": RANKED_NORMAL_SCAN_BATCH_SIZE,
            "scan_workers": int(scan_workers),
        },
    }
    manifest_path = outdir / "run_manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old.get("input_signature") != input_signature:
            raise ValueError(
                f"Existing paired microbial Core manifest does not match current inputs/config: {manifest_path}"
            )
        old_output_signature = old.get("output_signature")
        if isinstance(old_output_signature, dict) and outputs_match_signature(old_output_signature):
            return {**old, "reused": True}
        raise ValueError(
            f"Existing paired microbial Core manifest matches inputs but output files are incomplete or changed: {manifest_path}"
        )

    tumor_pass, tumor_excluded = read_protein_hits(
        tumor_protein_hits,
        tumor_sample,
        "tumor",
        min_pident,
        max_evalue,
        min_qcovs,
        contaminant_taxids,
        exclude_blacklist=False,
    )
    normal_pass, normal_excluded = read_protein_hits(
        normal_protein_hits,
        normal_sample,
        "normal",
        min_pident,
        max_evalue,
        min_qcovs,
        contaminant_taxids,
        exclude_blacklist=False,
    )

    normal_parent_hashes = set(normal_pass["parent_sequence_sha256"].dropna().astype(str))
    exact_parent_mask = tumor_pass["parent_sequence_sha256"].astype(str).isin(normal_parent_hashes)
    tumor_parent_exact_excluded = tumor_pass[exact_parent_mask].copy()
    if not tumor_parent_exact_excluded.empty:
        tumor_parent_exact_excluded["parent_qc_status"] = "excluded"
        tumor_parent_exact_excluded["parent_qc_reasons"] = "excluded_exact_parent_in_matched_normal"
    tumor_core = tumor_pass[~exact_parent_mask].copy().reset_index(drop=True)

    parent_excluded = pd.concat(
        [tumor_excluded, normal_excluded, tumor_parent_exact_excluded],
        ignore_index=True,
    )

    estimated_tumor_parent_peptide_windows = estimate_parent_peptide_windows(
        tumor_core,
        mhc_i_lengths,
        mhc_ii_lengths,
    )
    estimated_normal_parent_peptide_windows = estimate_parent_peptide_windows(
        normal_pass,
        mhc_i_lengths,
        mhc_ii_lengths,
    )
    estimated_total_parent_peptide_windows = (
        estimated_tumor_parent_peptide_windows + estimated_normal_parent_peptide_windows
    )
    preflight_stagewise_rows = [
        ("tumor_parent_input_rows", len(tumor_pass) + len(tumor_excluded)),
        ("normal_parent_input_rows", len(normal_pass) + len(normal_excluded)),
        ("tumor_parent_qc_pass", len(tumor_pass)),
        ("normal_parent_qc_pass", len(normal_pass)),
        ("tumor_parent_qc_excluded", len(tumor_excluded)),
        ("normal_parent_qc_excluded", len(normal_excluded)),
        ("tumor_parent_exact_normal_excluded", len(tumor_parent_exact_excluded)),
        ("tumor_parent_core", len(tumor_core)),
        ("estimated_tumor_parent_peptide_windows", estimated_tumor_parent_peptide_windows),
        ("estimated_normal_parent_peptide_windows", estimated_normal_parent_peptide_windows),
        ("estimated_total_parent_peptide_windows", estimated_total_parent_peptide_windows),
        ("paired_core_max_estimated_peptide_windows", int(max_estimated_peptide_windows)),
        ("candidate_selection_mode_ranked_cap", 1 if candidate_selection_mode == RANKED_CAP_MODE else 0),
        ("candidate_selection_max_hla_i_peptides", int(max_hla_i_peptides or 0)),
        ("candidate_selection_max_hla_ii_peptides", int(max_hla_ii_peptides or 0)),
    ]
    if (
        candidate_selection_mode == ALL_MODE
        and max_estimated_peptide_windows
        and estimated_total_parent_peptide_windows > max_estimated_peptide_windows
    ):
        skip_reason = (
            "estimated_total_parent_peptide_windows "
            f"({estimated_total_parent_peptide_windows}) exceeds "
            f"paired_core_max_estimated_peptide_windows ({max_estimated_peptide_windows})"
        )
        stagewise_rows = preflight_stagewise_rows + [("scale_gate_skipped", 1)]
        stagewise_path = outdir / "stagewise_qc.tsv"
        write_stagewise_qc(stagewise_path, stagewise_rows)
        scale_outputs = [stagewise_path]
        manifest = {
            "policy_version": POLICY_VERSION,
            "run_status": "scale_gate_skipped",
            "scale_gate_skipped": True,
            "input_signature": input_signature,
            "skip_reason": skip_reason,
            "pair_id": pair_id,
            "tumor_sample": tumor_sample,
            "normal_sample": normal_sample,
            "tumor_protein_hits": str(tumor_protein_hits),
            "normal_protein_hits": str(normal_protein_hits),
            "tumor_protein_hits_sha256": sha256_file(tumor_protein_hits),
            "normal_protein_hits_sha256": sha256_file(normal_protein_hits),
            "blacklist": str(blacklist) if blacklist else "",
            "blacklist_sha256": actual_blacklist_sha256,
            "blacklist_sha256_expected": blacklist_sha256,
            "allow_missing_blacklist": allow_missing_blacklist,
            "blacklist_evaluation": "enabled" if blacklist else "not_evaluated",
            "blacklist_taxids": len(verdict_by_taxid),
            "contaminant_taxids": len(contaminant_taxids),
            "mhc_i_lengths": mhc_i_lengths,
            "mhc_ii_lengths": mhc_ii_lengths,
            "min_pident": min_pident,
            "max_evalue": max_evalue,
            "min_qcovs": min_qcovs,
            "paired_core_max_estimated_peptide_windows": int(max_estimated_peptide_windows),
            "candidate_selection_policy_version": CANDIDATE_SELECTION_POLICY_VERSION,
            "candidate_selection": {
                "mode": candidate_selection_mode,
                "max_hla_i_peptides": max_hla_i_peptides,
                "max_hla_ii_peptides": max_hla_ii_peptides,
                "ranking_policy": MICROBIAL_RANKING_POLICY,
                "ranking_abundance_pseudocount": ranking_abundance_pseudocount,
                "scan_workers": int(scan_workers),
            },
            "outputs": {
                "stagewise_qc": str(stagewise_path),
            },
            "output_signature": output_signature(scale_outputs),
            "stagewise_counts": {step: int(count) for step, count in stagewise_rows},
        }
        write_json(outdir / "run_manifest.json", manifest)
        return manifest

    selection_manifest: dict[str, object] = {
        "mode": candidate_selection_mode,
        "max_hla_i_peptides": max_hla_i_peptides,
        "max_hla_ii_peptides": max_hla_ii_peptides,
        "ranking_policy": MICROBIAL_RANKING_POLICY if candidate_selection_mode == RANKED_CAP_MODE else "",
        "ranking_abundance_pseudocount": ranking_abundance_pseudocount,
        "scan_workers": int(scan_workers),
    }
    ranked_counts: dict[str, int] = {}

    if candidate_selection_mode == RANKED_CAP_MODE:
        ranked_result = build_ranked_pair_core_outputs(
            tumor_core=tumor_core,
            normal_pass=normal_pass,
            outdir=outdir,
            pair_id=pair_id,
            tumor_sample=tumor_sample,
            normal_sample=normal_sample,
            mhc_i_lengths=mhc_i_lengths,
            mhc_ii_lengths=mhc_ii_lengths,
            max_hla_i_peptides=int(max_hla_i_peptides),
            max_hla_ii_peptides=int(max_hla_ii_peptides),
            abundance_pseudocount=ranking_abundance_pseudocount,
            scan_workers=scan_workers,
        )
        core_df = ranked_result["core_df"]
        excluded_peptide_df = ranked_result["excluded_peptide_df"]
        parent_map_df = ranked_result["parent_map_df"]
        matched_normal_df = ranked_result["matched_normal_df"]
        tumor_unique_peptides_before_normal_subtraction = int(ranked_result["examined_candidate_peptides"])
        normal_unique_peptides_for_subtraction = 0
        matched_normal_excluded_count = int(ranked_result["matched_normal_excluded"])
        blacklist_excluded_count = int(ranked_result["blacklist_excluded"])
        ranked_counts = {
            "ranked_selected_hla_i": int(ranked_result["selected_hla_i"]),
            "ranked_selected_hla_ii": int(ranked_result["selected_hla_ii"]),
            "ranked_examined_candidate_peptides": int(ranked_result["examined_candidate_peptides"]),
            "ranked_deferred_source_groups": int(ranked_result["deferred_source_groups"]),
            "ranked_normal_scan_batches": int(ranked_result["normal_scan_batches"]),
            "ranked_normal_scan_batch_size": int(ranked_result["ranked_normal_scan_batch_size"]),
            "ranked_scan_workers": int(scan_workers),
            "ranked_cap_reached_hla_i": 1 if ranked_result["cap_reached_hla_i"] else 0,
            "ranked_cap_reached_hla_ii": 1 if ranked_result["cap_reached_hla_ii"] else 0,
        }
        selection_manifest.update(
            {
                "boundary_source_group_id": ranked_result["boundary_source_group_id"],
                "cap_reached_hla_i": bool(ranked_result["cap_reached_hla_i"]),
                "cap_reached_hla_ii": bool(ranked_result["cap_reached_hla_ii"]),
                "examined_candidate_peptides": int(ranked_result["examined_candidate_peptides"]),
                "deferred_source_groups": int(ranked_result["deferred_source_groups"]),
                "normal_scan_batches": int(ranked_result["normal_scan_batches"]),
                "ranked_normal_scan_batch_size": int(ranked_result["ranked_normal_scan_batch_size"]),
                "scan_workers": int(scan_workers),
            }
        )
    else:
        tumor_peptide_map, tumor_parent_positions = build_peptide_maps(
            tumor_core,
            mhc_i_lengths,
            mhc_ii_lengths,
        )
        normal_peptide_map, _ = build_peptide_maps(
            normal_pass,
            mhc_i_lengths,
            mhc_ii_lengths,
        )
        normal_peptide_keys = set(normal_peptide_map)

        core_entries = []
        matched_normal_excluded_entries = []
        blacklist_excluded_entries = []
        for key in sorted(tumor_peptide_map, key=lambda x: (x[0], len(x[1]), x[1])):
            entry = tumor_peptide_map[key]
            if key in normal_peptide_keys:
                entry["peptide_qc_reasons"] = "excluded_exact_peptide_in_matched_normal"
                matched_normal_excluded_entries.append(entry)
            elif peptide_blacklist_status(entry) == "exclusive_contaminant":
                entry["peptide_qc_reasons"] = "excluded_exclusive_contaminant"
                blacklist_excluded_entries.append(entry)
            else:
                core_entries.append(entry)
        excluded_entries = matched_normal_excluded_entries + blacklist_excluded_entries

        core_df = peptide_entries_to_df(
            core_entries,
            pair_id,
            tumor_sample,
            normal_sample,
            "core",
        )
        excluded_peptide_df = peptide_entries_to_df(
            excluded_entries,
            pair_id,
            tumor_sample,
            normal_sample,
            "excluded",
        )
        parent_map_df = build_parent_map_df(core_entries, excluded_entries, tumor_parent_positions)
        matched_normal_df = build_matched_normal_df(matched_normal_excluded_entries, normal_peptide_map)
        tumor_unique_peptides_before_normal_subtraction = len(tumor_peptide_map)
        normal_unique_peptides_for_subtraction = len(normal_peptide_map)
        matched_normal_excluded_count = len(matched_normal_excluded_entries)
        blacklist_excluded_count = len(blacklist_excluded_entries)

    write_tsv(tumor_core, outdir / "microbial_parent_core.tsv")
    write_tsv(parent_excluded, outdir / "microbial_parent_excluded.tsv")
    write_tsv(core_df, outdir / "microbial_peptide_core.tsv")
    write_tsv(core_df[core_df["mhc_class"] == MHC_CLASS_I].copy(), outdir / "microbial_peptide_core_hla_i.tsv")
    write_tsv(core_df[core_df["mhc_class"] == MHC_CLASS_II].copy(), outdir / "microbial_peptide_core_hla_ii.tsv")
    write_tsv(excluded_peptide_df, outdir / "microbial_peptide_excluded.tsv")
    write_tsv(parent_map_df, outdir / "microbial_peptide_parent_map.tsv")
    write_tsv(matched_normal_df, outdir / "matched_normal_peptide.tsv")
    write_combined_fasta(core_df, outdir / "microbial_peptide_core.fasta")
    write_fasta(core_df, MHC_CLASS_I, outdir / "microbial_peptide_core_hla_i.fasta")
    write_fasta(core_df, MHC_CLASS_II, outdir / "microbial_peptide_core_hla_ii.fasta")

    stagewise_rows = preflight_stagewise_rows + [
        ("scale_gate_skipped", 0),
        ("tumor_unique_peptides_before_normal_subtraction", tumor_unique_peptides_before_normal_subtraction),
        ("normal_unique_peptides_for_subtraction", normal_unique_peptides_for_subtraction),
        ("normal_unique_peptides_for_subtraction_not_materialized", 1 if candidate_selection_mode == RANKED_CAP_MODE else 0),
        ("tumor_unique_peptides_excluded_exact_normal", matched_normal_excluded_count),
        ("tumor_unique_peptides_excluded_blacklist", blacklist_excluded_count),
        ("tumor_unique_peptides_core", len(core_df)),
        ("tumor_unique_peptides_core_hla_i", int((core_df["mhc_class"] == MHC_CLASS_I).sum()) if not core_df.empty else 0),
        ("tumor_unique_peptides_core_hla_ii", int((core_df["mhc_class"] == MHC_CLASS_II).sum()) if not core_df.empty else 0),
    ] + list(ranked_counts.items())
    write_stagewise_qc(outdir / "stagewise_qc.tsv", stagewise_rows)
    complete_output_paths = [
        outdir / "microbial_parent_core.tsv",
        outdir / "microbial_parent_excluded.tsv",
        outdir / "microbial_peptide_core.tsv",
        outdir / "microbial_peptide_core_hla_i.tsv",
        outdir / "microbial_peptide_core_hla_ii.tsv",
        outdir / "microbial_peptide_core.fasta",
        outdir / "microbial_peptide_core_hla_i.fasta",
        outdir / "microbial_peptide_core_hla_ii.fasta",
        outdir / "microbial_peptide_excluded.tsv",
        outdir / "microbial_peptide_parent_map.tsv",
        outdir / "matched_normal_peptide.tsv",
        outdir / "stagewise_qc.tsv",
    ]
    if candidate_selection_mode == RANKED_CAP_MODE:
        complete_output_paths.extend([
            outdir / "microbial_ranked_source.tsv",
            outdir / "microbial_peptide_selection_evidence.tsv",
            outdir / "microbial_deferred_source.tsv",
        ])

    manifest = {
        "policy_version": POLICY_VERSION,
        "run_status": "completed",
        "scale_gate_skipped": False,
        "input_signature": input_signature,
        "skip_reason": "",
        "pair_id": pair_id,
        "tumor_sample": tumor_sample,
        "normal_sample": normal_sample,
        "tumor_protein_hits": str(tumor_protein_hits),
        "normal_protein_hits": str(normal_protein_hits),
        "tumor_protein_hits_sha256": sha256_file(tumor_protein_hits),
        "normal_protein_hits_sha256": sha256_file(normal_protein_hits),
        "blacklist": str(blacklist) if blacklist else "",
        "blacklist_sha256": actual_blacklist_sha256,
        "blacklist_sha256_expected": blacklist_sha256,
        "allow_missing_blacklist": allow_missing_blacklist,
        "blacklist_evaluation": "enabled" if blacklist else "not_evaluated",
        "blacklist_taxids": len(verdict_by_taxid),
        "contaminant_taxids": len(contaminant_taxids),
        "mhc_i_lengths": mhc_i_lengths,
        "mhc_ii_lengths": mhc_ii_lengths,
        "min_pident": min_pident,
        "max_evalue": max_evalue,
        "min_qcovs": min_qcovs,
        "paired_core_max_estimated_peptide_windows": int(max_estimated_peptide_windows),
        "candidate_selection_policy_version": CANDIDATE_SELECTION_POLICY_VERSION,
        "candidate_selection": selection_manifest,
        "outputs": {
            "microbial_parent_core": str(outdir / "microbial_parent_core.tsv"),
            "microbial_parent_excluded": str(outdir / "microbial_parent_excluded.tsv"),
            "microbial_peptide_core": str(outdir / "microbial_peptide_core.tsv"),
            "microbial_peptide_core_hla_i": str(outdir / "microbial_peptide_core_hla_i.tsv"),
            "microbial_peptide_core_hla_ii": str(outdir / "microbial_peptide_core_hla_ii.tsv"),
            "microbial_peptide_core_fasta": str(outdir / "microbial_peptide_core.fasta"),
            "microbial_peptide_core_hla_i_fasta": str(outdir / "microbial_peptide_core_hla_i.fasta"),
            "microbial_peptide_core_hla_ii_fasta": str(outdir / "microbial_peptide_core_hla_ii.fasta"),
            "microbial_peptide_excluded": str(outdir / "microbial_peptide_excluded.tsv"),
            "microbial_peptide_parent_map": str(outdir / "microbial_peptide_parent_map.tsv"),
            "matched_normal_peptide": str(outdir / "matched_normal_peptide.tsv"),
            "stagewise_qc": str(outdir / "stagewise_qc.tsv"),
            "microbial_ranked_source": str(outdir / "microbial_ranked_source.tsv") if candidate_selection_mode == RANKED_CAP_MODE else "",
            "microbial_peptide_selection_evidence": str(outdir / "microbial_peptide_selection_evidence.tsv") if candidate_selection_mode == RANKED_CAP_MODE else "",
            "microbial_deferred_source": str(outdir / "microbial_deferred_source.tsv") if candidate_selection_mode == RANKED_CAP_MODE else "",
        },
        "output_signature": output_signature(complete_output_paths),
        "stagewise_counts": {step: int(count) for step, count in stagewise_rows},
    }
    write_json(outdir / "run_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-id", default="")
    parser.add_argument("--tumor-sample", required=True)
    parser.add_argument("--normal-sample", required=True)
    parser.add_argument("--tumor-protein-hits", required=True)
    parser.add_argument("--normal-protein-hits", required=True)
    parser.add_argument("--blacklist", default="")
    parser.add_argument("--blacklist-sha256", default="")
    parser.add_argument(
        "--allow-missing-blacklist",
        action="store_true",
        help="Exploratory mode only: continue when no contaminant blacklist is configured.",
    )
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument("--mhc-i-lengths", default="8,9,10,11")
    parser.add_argument("--mhc-ii-lengths", default="13,14,15,16,17")
    parser.add_argument("--min-pident", type=float, default=100.0)
    parser.add_argument("--max-evalue", type=float, default=1e-5)
    parser.add_argument("--min-qcovs", type=float, default=90.0)
    parser.add_argument(
        "--max-estimated-peptide-windows",
        type=int,
        default=20_000_000,
        help="Skip Core materialization when estimated tumor+normal parent peptide windows exceed this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--candidate-selection-mode",
        choices=[ALL_MODE, RANKED_CAP_MODE],
        default=ALL_MODE,
        help="Use all candidates or rank source groups and cap unique HLA-I/HLA-II peptides before binding.",
    )
    parser.add_argument("--max-hla-i-peptides", type=int, default=None)
    parser.add_argument("--max-hla-ii-peptides", type=int, default=None)
    parser.add_argument(
        "--ranking-abundance-pseudocount",
        type=float,
        default=0.5,
        help="Pseudocount used when ranking microbial T/N enrichment in ranked_cap mode.",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=RANKED_PARENT_SCAN_MAX_WORKERS,
        help=f"Workers for ranked parent support scans; clamped to {RANKED_PARENT_SCAN_MAX_WORKERS}.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_pair_core(
        tumor_protein_hits=Path(args.tumor_protein_hits),
        normal_protein_hits=Path(args.normal_protein_hits),
        outdir=Path(args.outdir),
        tumor_sample=args.tumor_sample,
        normal_sample=args.normal_sample,
        pair_id=args.pair_id or None,
        blacklist=Path(args.blacklist) if args.blacklist else None,
        mhc_i_lengths=parse_int_list(args.mhc_i_lengths),
        mhc_ii_lengths=parse_int_list(args.mhc_ii_lengths),
        min_pident=args.min_pident,
        max_evalue=args.max_evalue,
        min_qcovs=args.min_qcovs,
        allow_missing_blacklist=args.allow_missing_blacklist,
        blacklist_sha256=args.blacklist_sha256,
        max_estimated_peptide_windows=args.max_estimated_peptide_windows,
        candidate_selection_mode=args.candidate_selection_mode,
        max_hla_i_peptides=args.max_hla_i_peptides,
        max_hla_ii_peptides=args.max_hla_ii_peptides,
        ranking_abundance_pseudocount=args.ranking_abundance_pseudocount,
        scan_workers=args.scan_workers,
    )
    print(json.dumps(manifest["stagewise_counts"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
