from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mimicneoai.microbial_pipeline.scripts.paired_protein_core_qc import (
    build_pair_core,
    validate_pair_id,
)


HIT_COLUMNS = [
    "qseqid",
    "sseqid",
    "sseq",
    "pident",
    "evalue",
    "qcovs",
    "protein_accession",
    "ctaxa_ids",
    "canonical_parent_sequence",
    "source_row_number",
]


def hit_row(
    source_row_number: int,
    protein: str,
    taxids: str,
    sequence: str,
    *,
    pident: float = 100.0,
    evalue: float = 1e-20,
    qcovs: float = 95.0,
) -> dict:
    return {
        "qseqid": f"read{source_row_number}|CTAXA:{taxids}|PROT:{protein}",
        "sseqid": f"ref|{protein}|",
        "sseq": sequence,
        "pident": pident,
        "evalue": evalue,
        "qcovs": qcovs,
        "protein_accession": protein,
        "ctaxa_ids": taxids,
        "canonical_parent_sequence": sequence,
        "source_row_number": source_row_number,
    }


class PairedProteinCoreQCTest(unittest.TestCase):
    def write_hits(self, path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
        columns = columns or HIT_COLUMNS
        pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)

    def write_blacklist(self, path: Path) -> None:
        pd.DataFrame(
            [
                {"tax_id": "999", "name": "Known contaminant", "verdict": "Contaminant"},
                {"tax_id": "111", "name": "Allowed microbe", "verdict": "Not a contaminant"},
            ]
        ).to_csv(path, sep="\t", index=False)

    def test_pair_core_subtracts_normal_peptides_and_keeps_traceability(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tumor_hits = root / "tumor.protein_hits.filtered.tsv"
            normal_hits = root / "normal.protein_hits.filtered.tsv"
            blacklist = root / "blacklist.tsv"
            outdir = root / "06b.MicrobialProteinCoreQC_v1.0"
            self.write_blacklist(blacklist)

            self.write_hits(
                tumor_hits,
                [
                    hit_row(1, "WP_T_SHARED.1", "111", "ACDEFGHIKLMN"),
                    hit_row(2, "WP_T_EXACT.1", "111", "LMNPQRSTVWY"),
                    hit_row(3, "WP_T_CONTAM.1", "999", "QRSTVWYACDE"),
                    hit_row(4, "WP_T_MIXED.1", "999,111", "TVWYACDEFGH"),
                    hit_row(5, "WP_T_INVALID.1", "111", "ACD*EFGHIK"),
                    hit_row(6, "WP_T_LONG.1", "111", "ACDEFGHIKLMNPQRST"),
                ],
            )
            self.write_hits(
                normal_hits,
                [
                    hit_row(1, "WP_N_SHARED.1", "111", "MNPACDEFGHIKQRST"),
                    hit_row(2, "WP_N_EXACT.1", "111", "LMNPQRSTVWY"),
                ],
            )

            manifest = build_pair_core(
                tumor_protein_hits=tumor_hits,
                normal_protein_hits=normal_hits,
                outdir=outdir,
                tumor_sample="T",
                normal_sample="N",
                pair_id="T,N",
                blacklist=blacklist,
                mhc_i_lengths=[8, 9],
                mhc_ii_lengths=[13],
            )

            self.assertTrue((outdir / "microbial_peptide_core.tsv").exists())
            self.assertTrue((outdir / "microbial_peptide_core.fasta").exists())
            self.assertTrue((outdir / "microbial_peptide_core_hla_i.fasta").exists())
            self.assertTrue((outdir / "microbial_peptide_core_hla_ii.fasta").exists())
            self.assertEqual(manifest["stagewise_counts"]["tumor_parent_qc_pass"], 5)
            self.assertEqual(manifest["stagewise_counts"]["tumor_parent_qc_excluded"], 1)
            self.assertEqual(manifest["stagewise_counts"]["tumor_parent_exact_normal_excluded"], 1)
            self.assertGreater(manifest["stagewise_counts"]["tumor_unique_peptides_excluded_blacklist"], 0)
            self.assertEqual(manifest["blacklist_evaluation"], "enabled")

            parent_core = pd.read_csv(outdir / "microbial_parent_core.tsv", sep="\t")
            self.assertIn("mixed", set(parent_core["blacklist_status"]))
            mixed_flags = ";".join(parent_core["parent_qc_flags"].fillna("").astype(str))
            self.assertIn("blacklist_mixed_retained", mixed_flags)
            self.assertNotIn("WP_T_EXACT.1", set(parent_core["protein_accession"]))

            parent_excluded = pd.read_csv(outdir / "microbial_parent_excluded.tsv", sep="\t")
            reasons = ";".join(parent_excluded["parent_qc_reasons"].fillna("").astype(str))
            self.assertIn("internal_stop", reasons)
            self.assertIn("excluded_exact_parent_in_matched_normal", reasons)

            peptide_core = pd.read_csv(outdir / "microbial_peptide_core.tsv", sep="\t")
            peptide_excluded = pd.read_csv(outdir / "microbial_peptide_excluded.tsv", sep="\t")
            matched_normal = pd.read_csv(outdir / "matched_normal_peptide.tsv", sep="\t")
            self.assertNotIn("ACDEFGHIK", set(peptide_core["peptide"]))
            self.assertIn("ACDEFGHIK", set(peptide_excluded["peptide"]))
            self.assertIn("ACDEFGHIK", set(matched_normal["peptide"]))
            self.assertIn("excluded_exclusive_contaminant", set(peptide_excluded["peptide_qc_reasons"]))
            self.assertTrue((peptide_core["mhc_class"] == "MHC-II").any())

            parent_map = pd.read_csv(outdir / "microbial_peptide_parent_map.tsv", sep="\t")
            self.assertIn("parent_record_id", parent_map.columns)
            self.assertIn("positions_1based", parent_map.columns)

            hla_i_fasta = (outdir / "microbial_peptide_core_hla_i.fasta").read_text()
            hla_ii_fasta = (outdir / "microbial_peptide_core_hla_ii.fasta").read_text()
            combined_fasta = (outdir / "microbial_peptide_core.fasta").read_text()
            self.assertIn(">microbial_core_", hla_i_fasta)
            self.assertIn("MHC-II", hla_ii_fasta)
            self.assertEqual(combined_fasta.count(">"), len(peptide_core))

    def test_missing_qcovs_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tumor_hits = root / "tumor.tsv"
            normal_hits = root / "normal.tsv"
            rows = [hit_row(1, "WP_1.1", "111", "ACDEFGHIKLMN")]
            self.write_hits(tumor_hits, rows, columns=[c for c in HIT_COLUMNS if c != "qcovs"])
            self.write_hits(normal_hits, rows)

            with self.assertRaisesRegex(ValueError, "qcovs"):
                build_pair_core(
                    tumor_protein_hits=tumor_hits,
                    normal_protein_hits=normal_hits,
                    outdir=root / "out",
                    tumor_sample="T",
                    normal_sample="N",
                    allow_missing_blacklist=True,
                )

    def test_normal_blacklisted_hits_still_subtract_shared_peptides(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tumor_hits = root / "tumor.tsv"
            normal_hits = root / "normal.tsv"
            blacklist = root / "blacklist.tsv"
            outdir = root / "out"
            self.write_blacklist(blacklist)
            self.write_hits(
                tumor_hits,
                [hit_row(1, "T_ALLOWED.1", "111", "ACDEFGHIKLM")],
            )
            self.write_hits(
                normal_hits,
                [hit_row(1, "N_CONTAM.1", "999", "VVACDEFGHIKWW")],
            )

            manifest = build_pair_core(
                tumor_protein_hits=tumor_hits,
                normal_protein_hits=normal_hits,
                outdir=outdir,
                tumor_sample="T",
                normal_sample="N",
                blacklist=blacklist,
                mhc_i_lengths=[9],
                mhc_ii_lengths=[13],
            )

            peptide_core = pd.read_csv(outdir / "microbial_peptide_core.tsv", sep="\t")
            peptide_excluded = pd.read_csv(outdir / "microbial_peptide_excluded.tsv", sep="\t")
            matched_normal = pd.read_csv(outdir / "matched_normal_peptide.tsv", sep="\t")
            self.assertNotIn("ACDEFGHIK", set(peptide_core["peptide"]))
            self.assertGreater(len(peptide_core), 0)
            self.assertEqual(set(peptide_excluded["peptide_qc_reasons"]), {"excluded_exact_peptide_in_matched_normal"})
            self.assertIn("ACDEFGHIK", set(matched_normal["peptide"]))
            self.assertEqual(manifest["stagewise_counts"]["normal_parent_qc_pass"], 1)
            self.assertEqual(manifest["stagewise_counts"]["tumor_unique_peptides_excluded_exact_normal"], 1)

    def test_blacklist_sha256_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tumor_hits = root / "tumor.tsv"
            normal_hits = root / "normal.tsv"
            blacklist = root / "blacklist.tsv"
            self.write_blacklist(blacklist)
            rows = [hit_row(1, "WP_1.1", "111", "ACDEFGHIKLMN")]
            self.write_hits(tumor_hits, rows)
            self.write_hits(normal_hits, rows)

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                build_pair_core(
                    tumor_protein_hits=tumor_hits,
                    normal_protein_hits=normal_hits,
                    outdir=root / "out",
                    tumor_sample="T",
                    normal_sample="N",
                    blacklist=blacklist,
                    blacklist_sha256="not_the_real_hash",
                )

    def test_scale_gate_writes_summary_without_materializing_core(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            root = Path(tempdir)
            tumor_hits = root / "tumor.tsv"
            normal_hits = root / "normal.tsv"
            blacklist = root / "blacklist.tsv"
            outdir = root / "out"
            self.write_blacklist(blacklist)
            self.write_hits(
                tumor_hits,
                [hit_row(1, "T_BIG.1", "111", "ACDEFGHIKLMNPQRST")],
            )
            self.write_hits(
                normal_hits,
                [hit_row(1, "N_BIG.1", "111", "TVWYACDEFGHIKLMNP")],
            )

            manifest = build_pair_core(
                tumor_protein_hits=tumor_hits,
                normal_protein_hits=normal_hits,
                outdir=outdir,
                tumor_sample="T",
                normal_sample="N",
                blacklist=blacklist,
                mhc_i_lengths=[8, 9],
                mhc_ii_lengths=[13],
                max_estimated_peptide_windows=1,
            )

            self.assertEqual(manifest["run_status"], "scale_gate_skipped")
            self.assertTrue(manifest["scale_gate_skipped"])
            self.assertIn("estimated_total_parent_peptide_windows", manifest["stagewise_counts"])
            self.assertGreater(manifest["stagewise_counts"]["estimated_total_parent_peptide_windows"], 1)
            self.assertTrue((outdir / "stagewise_qc.tsv").exists())
            self.assertTrue((outdir / "run_manifest.json").exists())
            self.assertFalse((outdir / "microbial_peptide_core.fasta").exists())
            self.assertFalse((outdir / "microbial_peptide_core.tsv").exists())

    def test_pair_id_validation_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "different"):
            validate_pair_id("S,S", "S", "S")
        with self.assertRaisesRegex(ValueError, "formatted"):
            validate_pair_id("T,N,extra", "T", "N")
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_pair_id("T,Other", "T", "N")


if __name__ == "__main__":
    unittest.main()
