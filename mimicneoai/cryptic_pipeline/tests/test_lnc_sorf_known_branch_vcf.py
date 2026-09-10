from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "02-lnc_sORF_pipeline.py"
SPEC = importlib.util.spec_from_file_location("lnc_sorf_pipeline", SCRIPT_PATH)
lnc_sorf_pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(lnc_sorf_pipeline)


class LncSorfKnownBranchVcfTest(unittest.TestCase):
    def test_lnc_bed_uses_merged_exons_not_transcript_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gtf = root / "lnc.gtf"
            gtf.write_text(
                'chr1\tsrc\ttranscript\t1\t1000\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
                'chr1\tsrc\texon\t1\t10\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
                'chr1\tsrc\texon\t8\t20\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
                'chr1\tsrc\texon\t101\t120\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
            )
            bed = root / "lncRNA.exons.merged.bed"
            lnc_sorf_pipeline.step_gtf_to_bed_transcripts(str(gtf), str(bed))
            lines = bed.read_text().splitlines()
            self.assertEqual(lines[0].split("\t")[:3], ["chr1", "0", "20"])
            self.assertEqual(lines[1].split("\t")[:3], ["chr1", "100", "120"])
            self.assertEqual(len(lines), 2)

    def test_filter_normalized_vcf_uses_ad_vaf_mq_alt_and_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            norm_vcf = root / "norm.vcf"
            flt_vcf = root / "rna.flt.vcf.gz"
            evidence = root / "rna.variant_evidence.tsv"
            norm_vcf.write_text(
                "##fileformat=VCFv4.2\n"
                "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths\">\n"
                "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n"
                "##INFO=<ID=MQ,Number=1,Type=Float,Description=\"Mapping quality\">\n"
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n"
                "chr1\t101\t.\tA\tG\t35\t.\tMQ=60\tGT:AD:DP\t0/1:7,3:10\n"
                "chr1\t102\t.\tA\tG\t35\t.\tMQ=60\tGT:AD:DP\t0/1:9,1:10\n"
                "chr1\t103\t.\tA\tG\t35\tLowQual\tMQ=60\tGT:AD:DP\t0/1:7,3:10\n"
            )

            def fake_run(cmd):
                if cmd[:3] == ["bcftools", "view", "-Oz"]:
                    out_path = Path(cmd[4])
                    in_path = Path(cmd[5])
                    shutil.copyfile(in_path, out_path)
                elif cmd[:3] == ["bcftools", "index", "-f"]:
                    Path(str(cmd[3]) + ".csi").write_text("")
                else:
                    raise AssertionError(cmd)

            with patch.object(lnc_sorf_pipeline, "run", side_effect=fake_run):
                stats = lnc_sorf_pipeline.filter_normalized_vcf_by_ad(
                    str(norm_vcf),
                    str(flt_vcf),
                    str(evidence),
                    qual_min=30,
                    dp_min=10,
                    mq_min=20,
                    af_min=0.05,
                    alt_min=3,
                )

            self.assertEqual(stats, {"normalized_records": 3, "filtered_records": 1})
            out_lines = [line for line in flt_vcf.read_text().splitlines() if not line.startswith("#")]
            self.assertEqual(len(out_lines), 1)
            self.assertEqual(out_lines[0].split("\t")[6], "PASS")
            evidence_text = evidence.read_text()
            self.assertIn("alt_reads_below_threshold", evidence_text)
            self.assertIn("vcf_filter_not_pass", evidence_text)

    def _write_resume_fixture(self, root: Path) -> tuple[Path, Path, Path, Path, Path]:
        bam = root / "sample.Aligned.out.sorted.bam"
        ref = root / "GRCh38.fa"
        bed = root / "lncRNA.exons.merged.bed"
        out_dir = root / "04k.bcf_consensus"
        out_dir.mkdir()
        for path in [bam, ref, bed]:
            path.write_text(path.name + "\n")
        (root / "sample.Aligned.out.sorted.bam.bai").write_text("index\n")
        outputs = {
            "raw_bcf": out_dir / "rna.raw.bcf",
            "call_vcf_sorted": out_dir / "rna.call.sorted.vcf.gz",
            "norm_split_vcf": out_dir / "rna.call.norm.split.vcf.gz",
            "norm_split_dedup_vcf": out_dir / "rna.call.norm.split.dedup.vcf.gz",
            "filtered_vcf": out_dir / "rna.flt.vcf.gz",
            "variant_evidence_tsv": out_dir / "rna.variant_evidence.tsv",
        }
        for label, path in outputs.items():
            path.write_text(label + "\n")
        input_signature = {
            "policy_version": lnc_sorf_pipeline.RNA_VARIANT_CALLING_POLICY_VERSION,
            "sample": "TEST",
            "sample_bam": lnc_sorf_pipeline.file_identity(bam),
            "reference_fasta": lnc_sorf_pipeline.file_identity(ref),
            "exon_bed": lnc_sorf_pipeline.file_identity(bed),
            "parameters": {
                "mpileup_flag_filter": lnc_sorf_pipeline.RNA_VARIANT_CALLING_FLAG_FILTER,
                "read_mapq_min": 20,
                "base_quality_min": 20,
                "depth_cap": 100000,
                "variant_qual_min": 30,
                "total_depth_min_ad_derived": 10,
                "variant_mq_min": 20,
                "vaf_min_ad_derived": 0.05,
                "alt_reads_min": 3,
                "normalization": lnc_sorf_pipeline.RNA_VARIANT_CALLING_NORMALIZATION_POLICY,
            },
            "bcftools_version": "bcftools 1.20",
            "calling_script_sha256": lnc_sorf_pipeline.sha256_file(lnc_sorf_pipeline.SCRIPT_PATH if hasattr(lnc_sorf_pipeline, "SCRIPT_PATH") else SCRIPT_PATH),
        }
        manifest = {
            "policy_version": lnc_sorf_pipeline.RNA_VARIANT_CALLING_POLICY_VERSION,
            "sample": "TEST",
            "run_status": "complete",
            "input_signature": input_signature,
            "output_signature": {label: lnc_sorf_pipeline.file_identity(path) for label, path in outputs.items()},
            "filter_stats": {"normalized_records": 7, "filtered_records": 3},
        }
        (out_dir / "rna.variant_calling.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return bam, ref, bed, out_dir, out_dir / "rna.flt.vcf.gz"

    def test_step_call_rnaseq_variants_resume_requires_matching_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bam, ref, bed, out_dir, flt = self._write_resume_fixture(root)
            manifest_path = out_dir / "rna.variant_calling.manifest.json"
            before = manifest_path.read_text()
            with patch.object(lnc_sorf_pipeline, "command_version", return_value="bcftools 1.20"), \
                    patch.object(lnc_sorf_pipeline, "run", side_effect=AssertionError("resume should not run commands")):
                observed = lnc_sorf_pipeline.step_call_rnaseq_variants(
                    str(bam),
                    str(ref),
                    str(bed),
                    str(out_dir),
                    sample="TEST",
                )
            self.assertEqual(Path(observed), flt)
            self.assertEqual(manifest_path.read_text(), before)

            with patch.object(lnc_sorf_pipeline, "command_version", return_value="bcftools 1.20"):
                with self.assertRaisesRegex(RuntimeError, "does not match current inputs"):
                    lnc_sorf_pipeline.step_call_rnaseq_variants(
                        str(bam),
                        str(ref),
                        str(bed),
                        str(out_dir),
                        mq_min=30,
                        sample="TEST",
                    )
            self.assertEqual(manifest_path.read_text(), before)

    def test_step_call_rnaseq_variants_resume_rejects_output_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bam, ref, bed, out_dir, flt = self._write_resume_fixture(root)
            flt.write_text(flt.read_text() + "changed\n")
            with patch.object(lnc_sorf_pipeline, "command_version", return_value="bcftools 1.20"):
                with self.assertRaisesRegex(RuntimeError, "output_signature no longer matches"):
                    lnc_sorf_pipeline.step_call_rnaseq_variants(
                        str(bam),
                        str(ref),
                        str(bed),
                        str(out_dir),
                        sample="TEST",
                    )

    def test_step_call_rnaseq_variants_splits_and_deduplicates_in_two_norm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bam = root / "sample.Aligned.out.sorted.bam"
            ref = root / "GRCh38.fa"
            bed = root / "lncRNA.exons.merged.bed"
            out_dir = root / "04k.bcf_consensus"
            out_dir.mkdir()
            for path in [bam, ref, bed]:
                path.write_text(path.name + "\n")
            (root / "sample.Aligned.out.sorted.bam.bai").write_text("index\n")

            class FakeStdout:
                def close(self) -> None:
                    return None

            class FakeProc:
                def __init__(self, cmd, stdout=None, stdin=None):
                    self.cmd = list(cmd)
                    self.stdout = FakeStdout()
                    if self.cmd[:2] == ["bcftools", "call"]:
                        Path(self.cmd[self.cmd.index("-o") + 1]).write_text("raw\n")

                def wait(self) -> int:
                    return 0

            commands: list[list[str]] = []

            def fake_run(cmd):
                cmd = list(map(str, cmd))
                commands.append(cmd)
                if "-o" in cmd:
                    Path(cmd[cmd.index("-o") + 1]).write_text("out\n")
                if cmd[:3] == ["bcftools", "index", "-f"]:
                    Path(cmd[3] + ".csi").write_text("index\n")

            def fake_filter(norm_vcf, flt_vcf, evidence_tsv, **_kwargs):
                self.assertTrue(Path(norm_vcf).name.endswith(".dedup.vcf.gz"))
                Path(flt_vcf).write_text("##fileformat=VCFv4.2\n")
                Path(evidence_tsv).write_text("chrom\tpos\n")
                return {"normalized_records": 0, "filtered_records": 0}

            with patch.object(lnc_sorf_pipeline, "command_version", return_value="bcftools 1.20"), \
                    patch.object(lnc_sorf_pipeline.subprocess, "Popen", side_effect=FakeProc), \
                    patch.object(lnc_sorf_pipeline, "run", side_effect=fake_run), \
                    patch.object(lnc_sorf_pipeline, "filter_normalized_vcf_by_ad", side_effect=fake_filter):
                observed = lnc_sorf_pipeline.step_call_rnaseq_variants(
                    str(bam),
                    str(ref),
                    str(bed),
                    str(out_dir),
                    sample="TEST",
                )

            self.assertEqual(Path(observed), out_dir / "rna.flt.vcf.gz")
            norm_commands = [cmd for cmd in commands if cmd[:2] == ["bcftools", "norm"]]
            self.assertEqual(len(norm_commands), 2)
            self.assertIn("-m", norm_commands[0])
            self.assertNotIn("-d", norm_commands[0])
            self.assertIn("-d", norm_commands[1])
            self.assertNotIn("-m", norm_commands[1])
            manifest = json.loads((out_dir / "rna.variant_calling.manifest.json").read_text())
            self.assertIn("norm_split_vcf", manifest["output_signature"])


if __name__ == "__main__":
    unittest.main()
