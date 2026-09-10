#!/usr/bin/env python3
"""Coordinate helpers for Cryptic Core and external-normal QC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


VALID_CONTIGS = {str(i) for i in range(1, 23)} | {"X", "Y"}


@dataclass(frozen=True)
class GenomicBlock:
    chrom: str
    strand: str
    start0: int
    end0: int

    @property
    def length(self) -> int:
        return self.end0 - self.start0

    def text(self) -> str:
        return f"{self.chrom}:{self.start0}-{self.end0}"


def normalize_chromosome(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    text = text.upper()
    return text if text in VALID_CONTIGS else text


def canonical_chromosome(value: object) -> str:
    chrom = normalize_chromosome(value)
    if chrom not in VALID_CONTIGS:
        return ""
    return chrom


def parse_blocks_text(value: object, strand: str) -> list[GenomicBlock]:
    blocks: list[GenomicBlock] = []
    text = str(value or "").strip()
    if not text:
        return blocks
    for token in text.split(";"):
        token = token.strip()
        if not token:
            continue
        chrom, coords = token.split(":", 1)
        start, end = coords.split("-", 1)
        blocks.append(GenomicBlock(canonical_chromosome(chrom), strand, int(start), int(end)))
    return blocks


def blocks_to_text(blocks: Iterable[GenomicBlock]) -> str:
    return ";".join(block.text() for block in blocks)


def genomic_order(blocks: Iterable[GenomicBlock]) -> list[GenomicBlock]:
    return sorted(blocks, key=lambda block: (block.chrom, block.start0, block.end0))


def transcript_order(blocks: Iterable[GenomicBlock], strand: str) -> list[GenomicBlock]:
    ordered = sorted(blocks, key=lambda block: (block.start0, block.end0))
    if strand == "-":
        ordered = list(reversed(ordered))
    return ordered


def map_transcript_interval_to_genome(
    blocks_in_transcript_order: list[GenomicBlock],
    strand: str,
    start0: int,
    end0: int,
) -> list[tuple[int, int, GenomicBlock]]:
    """Map a transcript interval to genomic block intervals.

    Returns tuples of (segment_transcript_start0, segment_transcript_end0,
    genomic_block_segment) in transcript order.
    """

    if end0 < start0:
        raise ValueError("transcript interval end must be >= start")
    out: list[tuple[int, int, GenomicBlock]] = []
    cursor = 0
    for block in blocks_in_transcript_order:
        block_tx_start = cursor
        block_tx_end = cursor + block.length
        cursor = block_tx_end
        overlap_start = max(start0, block_tx_start)
        overlap_end = min(end0, block_tx_end)
        if overlap_start >= overlap_end:
            continue
        local_start = overlap_start - block_tx_start
        local_end = overlap_end - block_tx_start
        if strand == "-":
            g_start = block.end0 - local_end
            g_end = block.end0 - local_start
        else:
            g_start = block.start0 + local_start
            g_end = block.start0 + local_end
        out.append(
            (
                overlap_start,
                overlap_end,
                GenomicBlock(block.chrom, strand, g_start, g_end),
            )
        )
    if sum(end - start for start, end, _block in out) != end0 - start0:
        raise ValueError("transcript interval is outside mapped blocks")
    return out


def reverse_complement(seq: str) -> str:
    return str(seq).translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1].upper()


def open_reference_fasta(path: Path):
    try:
        import pysam
    except Exception as exc:  # pragma: no cover - exercised only when runtime lacks pysam
        raise RuntimeError("pysam is required for indexed reference FASTA access") from exc

    return pysam.FastaFile(str(path))


def reference_contig_lookup(reference_fasta) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in reference_fasta.references:
        chrom = canonical_chromosome(name)
        if chrom and chrom not in lookup:
            lookup[chrom] = name
    return lookup


def extract_transcript_cds_from_reference(
    reference_fasta,
    contig_lookup: dict[str, str],
    blocks_in_transcript_order: list[GenomicBlock],
) -> tuple[str, list[str]]:
    """Extract strand-aware transcript/CDS sequence from genomic blocks."""

    if not blocks_in_transcript_order:
        return "", ["missing_transcript_blocks"]
    pieces: list[str] = []
    reasons: list[str] = []
    strands = {block.strand for block in blocks_in_transcript_order}
    if len(strands) != 1 or not strands.issubset({"+", "-"}):
        return "", ["invalid_or_mixed_strand_blocks"]
    for block in blocks_in_transcript_order:
        ref_name = contig_lookup.get(block.chrom)
        if not ref_name:
            reasons.append(f"reference_contig_missing:{block.chrom}")
            continue
        try:
            piece = reference_fasta.fetch(ref_name, block.start0, block.end0).upper()
        except Exception as exc:
            reasons.append(f"reference_fetch_failed:{block.chrom}:{block.start0}-{block.end0}:{exc}")
            continue
        if len(piece) != block.length:
            reasons.append(f"reference_fetch_length_mismatch:{block.chrom}:{block.start0}-{block.end0}")
            continue
        pieces.append(reverse_complement(piece) if block.strand == "-" else piece)
    if reasons:
        return "", reasons
    return "".join(pieces), []


GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def translate_cds(seq: str) -> str:
    clean = str(seq).replace(" ", "").replace("\n", "").strip().upper()
    aas = []
    for idx in range(0, len(clean) - 2, 3):
        aas.append(GENETIC_CODE.get(clean[idx : idx + 3], "X"))
    peptide = "".join(aas)
    return peptide[:-1] if peptide.endswith("*") else peptide


def parse_bed12(path: Path) -> dict[str, list[dict[str, object]]]:
    entries: dict[str, list[dict[str, object]]] = {}
    if not path or not path.exists():
        return entries
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            chrom = canonical_chromosome(fields[0])
            chrom_start = int(fields[1])
            strand = fields[5]
            block_sizes = [int(v) for v in fields[10].rstrip(",").split(",") if v]
            block_starts = [int(v) for v in fields[11].rstrip(",").split(",") if v]
            blocks = [
                GenomicBlock(chrom, strand, chrom_start + rel_start, chrom_start + rel_start + size)
                for rel_start, size in zip(block_starts, block_sizes)
            ]
            row = {
                "chromosome": chrom,
                "strand": strand,
                "start0": chrom_start,
                "end0": int(fields[2]),
                "name": fields[3],
                "mapq_from_bed_score": fields[4],
                "blocks": blocks,
            }
            entries.setdefault(fields[3], []).append(row)
    return entries


def read_fasta_records(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    if not path or not path.exists():
        return records
    header: Optional[str] = None
    chunks: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks).upper()
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            records[header] = "".join(chunks).upper()
    return records


def load_bam_alignment_stats(path: Path) -> dict[str, dict[str, object]]:
    try:
        import pysam
    except Exception as exc:  # pragma: no cover - exercised only when runtime lacks pysam
        raise RuntimeError("pysam is required for Cryptic coordinate QC") from exc

    stats: dict[str, dict[str, object]] = {}
    with pysam.AlignmentFile(str(path), "rb") as bam:
        for rec in bam.fetch(until_eof=True):
            if rec.is_unmapped:
                continue
            item = stats.setdefault(
                rec.query_name,
                {
                    "primary": [],
                    "secondary": [],
                    "supplementary": [],
                },
            )
            aln = {
                "query_name": rec.query_name,
                "chromosome": canonical_chromosome(rec.reference_name),
                "strand": "-" if rec.is_reverse else "+",
                "start0": int(rec.reference_start),
                "end0": int(rec.reference_end),
                "mapq": int(rec.mapping_quality),
                "cigar": rec.cigarstring or "",
                "blocks": [GenomicBlock(canonical_chromosome(rec.reference_name), "-" if rec.is_reverse else "+", a, b) for a, b in rec.get_blocks()],
            }
            if rec.is_supplementary:
                item["supplementary"].append(aln)
            elif rec.is_secondary:
                item["secondary"].append(aln)
            else:
                item["primary"].append(aln)
    return stats


def parse_json_blocks(value: object) -> list[GenomicBlock]:
    text = str(value or "").strip()
    if not text:
        return []
    rows = json.loads(text)
    return [
        GenomicBlock(
            canonical_chromosome(row["chromosome"]),
            str(row["strand"]),
            int(row["start0"]),
            int(row["end0"]),
        )
        for row in rows
    ]


def blocks_to_json(blocks: Iterable[GenomicBlock]) -> str:
    rows = [
        {
            "chromosome": block.chrom,
            "strand": block.strand,
            "start0": block.start0,
            "end0": block.end0,
        }
        for block in blocks
    ]
    return json.dumps(rows, separators=(",", ":"))


def interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def map_segment_to_reference_offset(segment: GenomicBlock, reference_blocks: list[GenomicBlock]) -> Optional[tuple[int, int]]:
    cursor = 0
    for block in reference_blocks:
        block_len = block.length
        if segment.chrom == block.chrom and segment.start0 >= block.start0 and segment.end0 <= block.end0:
            if block.strand == "-":
                start = cursor + (block.end0 - segment.end0)
                end = cursor + (block.end0 - segment.start0)
            else:
                start = cursor + (segment.start0 - block.start0)
                end = cursor + (segment.end0 - block.start0)
            return start, end
        cursor += block_len
    return None


def coordinate_overlap_exists(candidate_blocks: list[GenomicBlock], reference_blocks: list[GenomicBlock]) -> bool:
    for cand in candidate_blocks:
        for ref in reference_blocks:
            if cand.chrom == ref.chrom and interval_overlap(cand.start0, cand.end0, ref.start0, ref.end0):
                return True
    return False


def classify_coordinate_match(
    candidate_blocks: list[GenomicBlock],
    candidate_cds_start0: int,
    reference_blocks: list[GenomicBlock],
) -> tuple[str, str]:
    """Classify candidate peptide footprint against one normal smORF ORFCDS."""

    if not candidate_blocks or not reference_blocks:
        return "not_evaluable_reference_or_coordinate_mismatch", "missing_candidate_or_reference_blocks"
    if candidate_blocks[0].strand != reference_blocks[0].strand:
        return "not_evaluable_reference_or_coordinate_mismatch", "strand_mismatch"
    if not coordinate_overlap_exists(candidate_blocks, reference_blocks):
        return "no_coordinate_overlap", "no_coordinate_overlap"

    mapped_offsets: list[tuple[int, int]] = []
    segment_candidate_start = candidate_cds_start0
    for segment in candidate_blocks:
        mapped = map_segment_to_reference_offset(segment, reference_blocks)
        if mapped is None:
            return "partial_coordinate_overlap", "candidate_footprint_not_fully_covered_by_reference_orfcds"
        mapped_offsets.append(mapped)
        if (mapped[0] - segment_candidate_start) % 3 != 0:
            return "coordinate_overlap_frame_discordant", "reading_frame_discordant"
        segment_candidate_start += segment.length

    for idx in range(1, len(mapped_offsets)):
        if mapped_offsets[idx][0] != mapped_offsets[idx - 1][1]:
            return "junction_chain_incompatible", "reference_transcript_offsets_not_contiguous"

    if mapped_offsets[0][0] % 3 != 0:
        return "coordinate_overlap_frame_discordant", "reading_frame_discordant"
    return "coordinate_frame_concordant", "complete_footprint_covered_in_same_reading_frame"
