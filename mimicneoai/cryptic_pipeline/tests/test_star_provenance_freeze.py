from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mimicneoai.cryptic_pipeline.scripts.freeze_star_provenance import freeze_pairs


class StarProvenanceFreezeTest(unittest.TestCase):
    def _write_star_outputs(self, star_dir: Path, sample: str) -> None:
        star_dir.mkdir(parents=True, exist_ok=True)
        (star_dir / f"{sample}Aligned.out.bam").write_bytes(b"BAM\x01header\n")
        (star_dir / f"{sample}SJ.out.tab").write_text(
            "chr1\t101\t200\t1\t1\t1\t3\t0\t20\n"
            "chr1\t301\t400\t1\t1\t0\t1\t2\t12\n"
        )
        (star_dir / f"{sample}Log.final.out").write_text("Number of input reads | 10\n")
        (star_dir / f"{sample}Log.out").write_text(
            "STAR version=STAR_2.5.3a_modified\n"
            "##### Command Line:\n"
            "STAR --genomeDir /idx --outSAMunmapped Within --outFilterType BySJout "
            "--outSAMattributes NH HI AS NM MD --outFilterMultimapNmax 20 "
            "--outFilterMismatchNmax 999 --outFilterMismatchNoverLmax 0.04 "
            "--alignIntronMin 20 --alignIntronMax 1000000 --alignMatesGapMax 1000000 "
            "--alignSJoverhangMin 8 --alignSJDBoverhangMin 1 --sjdbScore 1 "
            "--runThreadN 30 --genomeLoad LoadAndKeep --limitBAMsortRAM 40000000000 "
            "--outSAMtype BAM Unsorted --quantMode TranscriptomeSAM "
            "--outSAMheaderHD @HD VN:1.4 SO:unsorted --outFileNamePrefix /out/prefix "
            "--readFilesCommand zcat --readFilesIn r1.fq.gz r2.fq.gz\n"
        )

    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        cryptic = root / "Cryptic"
        tumor = "T-RNA"
        normal = "N-RNA"
        tumor_star = cryptic / tumor / "01-star" / tumor / f"{tumor}.star"
        normal_star = cryptic / tumor / "01-star" / normal / f"{normal}.star"
        self._write_star_outputs(tumor_star, tumor)
        self._write_star_outputs(normal_star, normal)
        source_manifest = {
            "status": "completed",
            "output_dir": "/old/batch/Cryptic/T-RNA/01-star/N-RNA/N-RNA.star",
            "star_version": "STAR_2.5.3a_modified",
            "star_parameters": [
                "--genomeDir", "/idx",
                "--outSAMunmapped", "Within",
                "--outFilterType", "BySJout",
                "--outSAMattributes", "NH", "HI", "AS", "NM", "MD",
                "--outFilterMultimapNmax", "20",
                "--outFilterMismatchNmax", "999",
                "--outFilterMismatchNoverLmax", "0.04",
                "--alignIntronMin", "20",
                "--alignIntronMax", "1000000",
                "--alignMatesGapMax", "1000000",
                "--alignSJoverhangMin", "8",
                "--alignSJDBoverhangMin", "1",
                "--sjdbScore", "1",
                "--runThreadN", "20",
                "--genomeLoad", "LoadAndKeep",
                "--limitBAMsortRAM", "40000000000",
                "--outSAMtype", "BAM", "Unsorted",
                "--quantMode", "TranscriptomeSAM",
                "--outSAMheaderHD", "@HD", "VN:1.4", "SO:unsorted",
                "--outFileNamePrefix", "/old/prefix",
                "--readFilesCommand", "zcat",
                "--readFilesIn", "n1.fq.gz", "n2.fq.gz",
            ],
            "input_signature": {
                "alignment_role": "control",
                "sample": normal,
                "tumor_sample": tumor,
                "control_sample": normal,
                "star_index": {"path": "/idx", "files": {"Genome": {"size_bytes": 1}}},
            },
            "outputs": {
                "SJ.out.tab": {"path": "/old/batch/Cryptic/T-RNA/01-star/N-RNA/N-RNA.star/N-RNASJ.out.tab"}
            },
        }
        (normal_star / "star_alignment.manifest.json").write_text(json.dumps(source_manifest, indent=2))
        pair_sheet = root / "pairs.tsv"
        pair_sheet.write_text("tumor_sample\tnormal_sample\nT-RNA\tN-RNA\n")
        return cryptic, pair_sheet

    def _args(self, cryptic: Path, pair_sheet: Path, outdir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            cryptic_root=str(cryptic),
            pair_sheet=str(pair_sheet),
            outdir=str(outdir),
            star_index="/idx",
            relocated_at="2026-07-28T00:00:00+00:00",
            full_bam_hash=False,
            allow_critical_contract_upgrade=False,
        )

    def _args_without_fixed_relocated_at(self, cryptic: Path, pair_sheet: Path, outdir: Path) -> argparse.Namespace:
        args = self._args(cryptic, pair_sheet, outdir)
        args.relocated_at = ""
        return args

    def test_freezes_relocated_normal_and_legacy_tumor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cryptic, pair_sheet = self._write_fixture(root)
            outdir = root / "freeze"

            manifest = freeze_pairs(self._args(cryptic, pair_sheet, outdir))

            self.assertEqual(manifest["status"], "complete")
            table = pd.read_csv(outdir / "cryptic_star_pair_inputs.tsv", sep="\t")
            self.assertEqual(table.shape[0], 1)
            row = table.iloc[0]
            self.assertTrue(str(row["tumor_sj_path"]).endswith("T-RNASJ.out.tab"))
            self.assertTrue(str(row["normal_sj_path"]).endswith("N-RNASJ.out.tab"))
            self.assertEqual(row["tumor_manifest_status"], "frozen_legacy_complete")
            self.assertEqual(row["normal_manifest_status"], "relocated_complete")
            self.assertEqual(int(row["tumor_sj_rows"]), 2)
            self.assertEqual(int(row["normal_sj_rows"]), 2)

            freeze_dir = cryptic / "T-RNA" / "01-star" / "star-provenance-freeze"
            tumor_manifest = json.loads((freeze_dir / "tumor_star.frozen_legacy_manifest.json").read_text())
            normal_manifest = json.loads((freeze_dir / "normal_star.relocated_manifest.json").read_text())
            self.assertEqual(tumor_manifest["star_index_identity_status"], "reconstructed_at_freeze")
            self.assertEqual(tumor_manifest["critical_parameter_compatibility"], "compatible")
            self.assertEqual(normal_manifest["critical_parameter_compatibility"], "compatible")
            self.assertEqual(row["critical_parameter_compatibility"], "compatible")
            self.assertEqual(row["tumor_critical_parameter_digest"], row["normal_critical_parameter_digest"])
            self.assertEqual(normal_manifest["relocated_from"], "/old/batch/Cryptic/T-RNA/01-star/N-RNA/N-RNA.star")
            self.assertIn("sha256", normal_manifest["current_outputs"]["SJ.out.tab"])
            self.assertNotIn("sha256", normal_manifest["current_outputs"]["Aligned.out.bam"])

            reused = freeze_pairs(self._args(cryptic, pair_sheet, outdir))
            self.assertEqual(reused["pair_count"], 1)

    def test_resume_ignores_volatile_relocated_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cryptic, pair_sheet = self._write_fixture(root)
            outdir = root / "freeze"
            first = freeze_pairs(self._args_without_fixed_relocated_at(cryptic, pair_sheet, outdir))
            second = freeze_pairs(self._args_without_fixed_relocated_at(cryptic, pair_sheet, outdir))
            self.assertEqual(first["pair_count"], second["pair_count"])

    def test_pair_sheet_duplicate_tumor_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cryptic, pair_sheet = self._write_fixture(root)
            pair_sheet.write_text("tumor_sample\tnormal_sample\nT-RNA\tN1-RNA\nT-RNA\tN2-RNA\n")
            with self.assertRaisesRegex(ValueError, "Duplicate tumor"):
                freeze_pairs(self._args(cryptic, pair_sheet, root / "freeze"))

    def test_existing_mismatched_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cryptic, pair_sheet = self._write_fixture(root)
            outdir = root / "freeze"
            freeze_pairs(self._args(cryptic, pair_sheet, outdir))
            target = cryptic / "T-RNA" / "01-star" / "star-provenance-freeze" / "tumor_star.frozen_legacy_manifest.json"
            saved = json.loads(target.read_text())
            saved["status"] = "tampered"
            target.write_text(json.dumps(saved, indent=2))
            with self.assertRaisesRegex(RuntimeError, "differs"):
                freeze_pairs(self._args(cryptic, pair_sheet, outdir))

    def test_allows_controlled_upgrade_from_old_critical_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cryptic, pair_sheet = self._write_fixture(root)
            outdir = root / "freeze"
            freeze_pairs(self._args(cryptic, pair_sheet, outdir))

            freeze_dir = cryptic / "T-RNA" / "01-star" / "star-provenance-freeze"
            for manifest_path in [
                freeze_dir / "tumor_star.frozen_legacy_manifest.json",
                freeze_dir / "normal_star.relocated_manifest.json",
            ]:
                manifest = json.loads(manifest_path.read_text())
                for key in [
                    "critical_parameter_compatibility",
                    "critical_star_parameters",
                    "critical_star_parameter_digest",
                ]:
                    manifest.pop(key, None)
                manifest.get("input_signature", {}).pop("critical_parameter_compatibility", None)
                manifest_path.write_text(json.dumps(manifest, indent=2))

            table = pd.read_csv(outdir / "cryptic_star_pair_inputs.tsv", sep="\t")
            table = table.drop(
                columns=[
                    "critical_parameter_compatibility",
                    "tumor_critical_parameter_digest",
                    "normal_critical_parameter_digest",
                ]
            )
            table.to_csv(outdir / "cryptic_star_pair_inputs.tsv", sep="\t", index=False)

            with self.assertRaisesRegex(RuntimeError, "differs"):
                freeze_pairs(self._args(cryptic, pair_sheet, outdir))

            args = self._args(cryptic, pair_sheet, outdir)
            args.allow_critical_contract_upgrade = True
            freeze_pairs(args)

            upgraded_table = pd.read_csv(outdir / "cryptic_star_pair_inputs.tsv", sep="\t")
            self.assertIn("critical_parameter_compatibility", upgraded_table.columns)
            self.assertEqual(upgraded_table.loc[0, "critical_parameter_compatibility"], "compatible")
            upgraded_manifest = json.loads((freeze_dir / "tumor_star.frozen_legacy_manifest.json").read_text())
            self.assertIn("previous_manifest_sha256", upgraded_manifest)
            self.assertEqual(upgraded_manifest["critical_parameter_compatibility"], "compatible")

    def test_normal_source_role_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cryptic, pair_sheet = self._write_fixture(root)
            source_manifest_path = cryptic / "T-RNA" / "01-star" / "N-RNA" / "N-RNA.star" / "star_alignment.manifest.json"
            source = json.loads(source_manifest_path.read_text())
            source["input_signature"]["alignment_role"] = "tumor"
            source_manifest_path.write_text(json.dumps(source, indent=2))
            with self.assertRaisesRegex(ValueError, "alignment_role"):
                freeze_pairs(self._args(cryptic, pair_sheet, root / "freeze"))

    def test_critical_star_parameter_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cryptic, pair_sheet = self._write_fixture(root)
            source_manifest_path = cryptic / "T-RNA" / "01-star" / "N-RNA" / "N-RNA.star" / "star_alignment.manifest.json"
            source = json.loads(source_manifest_path.read_text())
            params = source["star_parameters"]
            params[params.index("--alignSJoverhangMin") + 1] = "12"
            source_manifest_path.write_text(json.dumps(source, indent=2))
            with self.assertRaisesRegex(ValueError, "critical parameters differ"):
                freeze_pairs(self._args(cryptic, pair_sheet, root / "freeze"))


if __name__ == "__main__":
    unittest.main()
