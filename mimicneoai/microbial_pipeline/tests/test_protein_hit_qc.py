from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mimicneoai.microbial_pipeline.scripts.get_data_for_binding_pred import (
    get_data_for_binding_pred,
    normalize_parent_sequence,
)
from mimicneoai.microbial_pipeline.scripts.microbial_peptides import (
    _normalize_diamond_blastx_output,
)


BLASTX_COLUMNS = [
    "qseqid",
    "qlen",
    "sseqid",
    "qseq",
    "sseq",
    "stitle",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
    "qcovhsp",
    "qcovs",
]

DIAMOND_COLUMNS = [
    "qseqid",
    "qlen",
    "sseqid",
    "qseq",
    "sseq",
    "stitle",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
    "qcovhsp",
]


class DummyTool:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def write_log(self, msg: str, level: str = "info") -> None:
        self.messages.append((level, msg))


def make_row(
    read: str,
    accession: str,
    parent: str,
    *,
    taxid: str = "111",
    pident: float = 100.0,
    evalue: float = 1e-20,
    qcovs: float = 95.0,
) -> list[object]:
    return [
        f"{read}|YP:Z:{taxid}",
        len(parent),
        f"ref|{accession}|",
        parent,
        parent,
        "microbial protein",
        pident,
        len(parent),
        0,
        0,
        1,
        len(parent),
        1,
        len(parent),
        evalue,
        50.0,
        qcovs,
        qcovs,
    ]


class MicrobialProteinHitQCTest(unittest.TestCase):
    def write_input(self, root: Path, rows: list[list[object]], columns=None, filename: str = "hits.tsv") -> None:
        pd.DataFrame(rows).to_csv(
            root / filename,
            sep="\t",
            header=False,
            index=False,
        )

    def catalog(self) -> pd.DataFrame:
        return pd.DataFrame({"prot_id": ["WP_1.1", "WP_2.1"], "tax_id": ["111", "222"]})

    def test_canonical_outputs_and_exclusions_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            rows = [
                make_row("read_pass_terminal_stop", "WP_1.1", "ACDEFGHIK*"),
                make_row("read_internal_stop", "WP_1.1", "ACD*EFGHIK"),
                make_row("read_x", "WP_1.1", "ACDEFGHIX"),
                make_row("read_gap", "WP_1.1", "ACDE-FGHIK"),
                make_row("read_pident", "WP_1.1", "ACDEFGHIK", pident=99.9),
                make_row("read_qcovs", "WP_1.1", "ACDEFGHIK", qcovs=89.0),
                make_row("read_catalog", "WP_2.1", "ACDEFGHIK", taxid="111"),
            ]
            self.write_input(root, rows)

            summary = get_data_for_binding_pred(
                blast_file="hits.tsv",
                colnames=BLASTX_COLUMNS,
                pvacbind_file="SAMPLE.peptide.fasta",
                output_blastx=str(root),
                min_pident=100,
                catalog_df=self.catalog(),
                max_evalue=1e-5,
                min_qcovs=90,
                sample="SAMPLE",
                tool=DummyTool(),
            )

            self.assertEqual(summary["after_pident_eq_100"], 5)
            self.assertEqual(summary["after_qcovs_90"], 4)
            self.assertEqual(summary["after_canonical_parent_sequence"], 1)

            protein_hits = pd.read_csv(root / "SAMPLE.protein_hits.filtered.tsv", sep="\t")
            self.assertEqual(len(protein_hits), 1)
            self.assertEqual(protein_hits.loc[0, "canonical_parent_sequence"], "ACDEFGHIK")
            self.assertTrue(bool(protein_hits.loc[0, "terminal_stop_removed"]))
            self.assertEqual(protein_hits.loc[0, "protein_accession"], "WP_1.1")
            self.assertEqual(str(protein_hits.loc[0, "ctaxa_ids"]), "111")
            self.assertEqual(int(protein_hits.loc[0, "source_row_number"]), 1)

            legacy = pd.read_csv(root / "SAMPLE.blastx.filtered", sep="\t")
            self.assertEqual(len(legacy), 1)
            self.assertIn("protIndex", legacy.columns)

            fasta_text = (root / "SAMPLE.peptide.fasta").read_text()
            self.assertIn("ACDEFGHIK", fasta_text)
            self.assertNotIn("*", fasta_text)

            excluded = pd.read_csv(root / "SAMPLE.protein_hits.excluded.tsv", sep="\t")
            reasons = ";".join(excluded["row_qc_reasons"].astype(str))
            self.assertIn("internal_stop", reasons)
            self.assertIn("noncanonical_amino_acid:X", reasons)
            self.assertIn("gap_in_parent_sequence", reasons)
            self.assertIn("pident_not_equal_100", reasons)
            self.assertIn("qcovs_lt_90", reasons)
            self.assertIn("protein_taxon_catalog_mismatch", reasons)

    def test_missing_qcovs_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            row = make_row("read", "WP_1.1", "ACDEFGHIK")[:-1]
            self.write_input(root, [row])

            with self.assertRaisesRegex(ValueError, "qcovs"):
                get_data_for_binding_pred(
                    blast_file="hits.tsv",
                    colnames=BLASTX_COLUMNS[:-1],
                    pvacbind_file="SAMPLE.peptide.fasta",
                    output_blastx=str(root),
                    min_pident=100,
                    catalog_df=self.catalog(),
                    sample="SAMPLE",
                    tool=DummyTool(),
                )

    def test_diamond_normalization_produces_blastx_compatible_input(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            diamond_in = root / "SAMPLE.diamond.blastx"
            normalized = root / "SAMPLE.normalized.diamond.blastx"
            row = make_row("read", "WP_2.1", "ACDEFGHIK", taxid="222")[:-1]
            pd.DataFrame([row]).to_csv(diamond_in, sep="\t", header=False, index=False)

            _normalize_diamond_blastx_output(
                input_path=str(diamond_in),
                output_path=str(normalized),
                colnames=DIAMOND_COLUMNS,
                tool=DummyTool(),
                sample="SAMPLE",
            )

            df = pd.read_csv(normalized, sep="\t", header=None, names=BLASTX_COLUMNS)
            self.assertEqual(df.loc[0, "qcovs"], df.loc[0, "qcovhsp"])
            self.assertEqual(df.loc[0, "qseq"], df.loc[0, "sseq"])
            self.assertEqual(df.loc[0, "sseqid"], "ref|WP_2.1|")

    def test_parent_sequence_normalization_policy(self) -> None:
        self.assertEqual(normalize_parent_sequence("acdefghik*"), ("ACDEFGHIK", [], True))
        parent, reasons, terminal_removed = normalize_parent_sequence("ACD*EFGHIK")
        self.assertEqual(parent, "ACD*EFGHIK")
        self.assertFalse(terminal_removed)
        self.assertIn("internal_stop", reasons)


if __name__ == "__main__":
    unittest.main()
