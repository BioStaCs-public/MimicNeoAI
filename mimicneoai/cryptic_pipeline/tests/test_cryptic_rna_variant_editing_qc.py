from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

from mimicneoai.cryptic_pipeline.scripts.cryptic_coordinate_utils import GenomicBlock
from mimicneoai.cryptic_pipeline.scripts.cryptic_rna_variant_editing_qc import (
    POLICY_VERSION,
    REDIPORTAL_RESOURCE_POLICY_VERSION,
    RNA_VARIANT_CALLING_POLICY_VERSION,
    sha256_file,
    evaluate_parent_rna_variants,
    load_rediportal_events,
    load_vcf_events,
    sample_name_matches,
    validate_rna_variant_calling_manifest,
    validate_policy,
)


class CrypticRnaVariantEditingQCTest(unittest.TestCase):
    def _write_vcf(self, root: Path, rows: list[str], sample: str = "TEST") -> Path:
        path = root / "rna.call.sorted.tags.vcf"
        path.write_text(
            "\n".join([
                "##fileformat=VCFv4.2",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths\">",
                "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read depth\">",
                "##INFO=<ID=MQ,Number=1,Type=Float,Description=\"Mapping quality\">",
                f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}",
                *rows,
            ]) + "\n"
        )
        return path

    def _event_row(
        self,
        *,
        chrom: str = "chr1",
        pos: int = 101,
        ref: str = "A",
        alt: str = "G",
        qual: int = 35,
        mq: int = 60,
        ad: str = "7,3",
        dp: int = 10,
        vcf_filter: str = "PASS",
    ) -> str:
        return f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\t{vcf_filter}\tMQ={mq}\tGT:AD:DP\t0/1:{ad}:{dp}"

    def test_sample_identity_accepts_bam_sample_column(self) -> None:
        self.assertTrue(sample_name_matches("/some/path/WP-T-RNA.Aligned.out.sorted.bam", "WP-T-RNA"))
        self.assertFalse(sample_name_matches("/some/path/WP-N-RNA.Aligned.out.sorted.bam", "WP-T-RNA"))

    def test_vcf_loader_uses_format_ad_and_requires_mq(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vcf = self._write_vcf(root, [self._event_row(ad="20,5", dp=99)])
            events, summary = load_vcf_events(vcf, expected_sample="TEST")
            event = events[("1", 101, "A", "G")]
            self.assertEqual(event.ref_reads, 20)
            self.assertEqual(event.alt_reads, 5)
            self.assertEqual(event.total_depth, 25)
            self.assertAlmostEqual(event.vaf, 0.2)
            self.assertEqual(summary["has_INFO_MQ"], True)

            no_mq = self._write_vcf(
                root,
                ["chr1\t101\t.\tA\tG\t35\tPASS\t.\tGT:AD:DP\t0/1:7,3:10"],
            )
            with self.assertRaisesRegex(ValueError, "lacks INFO/MQ"):
                load_vcf_events(no_mq, expected_sample="TEST")

    def test_vcf_loader_collapses_duplicate_exact_events_conservatively(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vcf = self._write_vcf(root, [
                self._event_row(qual=40, mq=60, ad="7,4", dp=11),
                self._event_row(qual=35, mq=25, ad="10,3", dp=13),
            ])
            with self.assertRaisesRegex(ValueError, "Duplicate exact VCF event keys"):
                load_vcf_events(vcf, expected_sample="TEST")

            events, summary = load_vcf_events(vcf, expected_sample="TEST", allow_legacy_duplicate_vcf=True)
            event = events[("1", 101, "A", "G")]
            self.assertEqual(len(events), 1)
            self.assertEqual(event.qual, "35")
            self.assertEqual(event.mapping_quality, "25")
            self.assertEqual(event.ref_reads, 10)
            self.assertEqual(event.alt_reads, 3)
            self.assertEqual(event.total_depth, 13)
            self.assertAlmostEqual(event.vaf, 3 / 13)
            self.assertEqual(event.vcf_duplicate_status, "conflicting_duplicate_not_evaluable")
            self.assertEqual(summary["duplicate_exact_event_keys"], 1)
            self.assertEqual(summary["duplicate_exact_event_rows_collapsed"], 1)
            self.assertEqual(summary["conflicting_duplicate_exact_event_keys_collapsed"], 1)

    def test_redigo_exact_allele_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            redi = root / "rediportal.tsv"
            redi.write_text(
                "chrom\tpos_1based\tref\talt\tediting_type\tresource_record_id\tsource_release\n"
                "chr1\t101\tA\tT\tA-to-I\tR1\tfixture\n"
            )
            events, _summary = load_rediportal_events(redi)
            self.assertIn(("1", 101, "A", "T"), events)
            self.assertNotIn(("1", 101, "A", "G"), events)

            manifest = root / "rediportal.manifest.json"
            manifest.write_text(
                "{"
                f"\"policy_version\":\"{REDIPORTAL_RESOURCE_POLICY_VERSION}\","
                f"\"reference_build\":\"GRCh38\","
                f"\"processed_table_sha256\":\"{sha256_file(redi)}\","
                f"\"processed_table_size_bytes\":{redi.stat().st_size},"
                "\"records\":1,"
                "\"normalization_rules\":\"GRCh38 exact REF/ALT forward-strand allele\","
                "\"processing_script_sha256\":\"scriptsha\""
                "}"
            )
            events_with_manifest, summary = load_rediportal_events(redi, manifest)
            self.assertEqual(events_with_manifest, events)
            self.assertEqual(summary["manifest_sha256"], sha256_file(manifest))

            bad_manifest = root / "bad.manifest.json"
            bad_manifest.write_text(
                "{"
                f"\"policy_version\":\"{REDIPORTAL_RESOURCE_POLICY_VERSION}\","
                "\"reference_build\":\"GRCh38\","
                "\"processed_table_sha256\":\"bad\","
                f"\"processed_table_size_bytes\":{redi.stat().st_size},"
                "\"records\":1,"
                "\"normalization_rules\":\"GRCh38 exact REF/ALT forward-strand allele\","
                "\"processing_script_sha256\":\"scriptsha\""
                "}"
            )
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_rediportal_events(redi, bad_manifest)

            empty_manifest = root / "empty.manifest.json"
            empty_manifest.write_text("{}")
            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_rediportal_events(redi, empty_manifest)

            missing_build = root / "missing_build.manifest.json"
            missing_build.write_text(
                "{"
                f"\"policy_version\":\"{REDIPORTAL_RESOURCE_POLICY_VERSION}\","
                f"\"processed_table_sha256\":\"{sha256_file(redi)}\","
                f"\"processed_table_size_bytes\":{redi.stat().st_size},"
                "\"records\":1,"
                "\"normalization_rules\":\"GRCh38 exact REF/ALT forward-strand allele\","
                "\"processing_script_sha256\":\"scriptsha\""
                "}"
            )
            with self.assertRaisesRegex(ValueError, "reference_build"):
                load_rediportal_events(redi, missing_build)

            wrong_policy = root / "wrong_policy.manifest.json"
            wrong_policy.write_text(
                "{"
                "\"policy_version\":\"old_policy\","
                "\"reference_build\":\"GRCh38\","
                f"\"processed_table_sha256\":\"{sha256_file(redi)}\","
                f"\"processed_table_size_bytes\":{redi.stat().st_size},"
                "\"records\":1,"
                "\"normalization_rules\":\"GRCh38 exact REF/ALT forward-strand allele\","
                "\"processing_script_sha256\":\"scriptsha\""
                "}"
            )
            with self.assertRaisesRegex(ValueError, "policy_version"):
                load_rediportal_events(redi, wrong_policy)

            wrong_records = root / "wrong_records.manifest.json"
            wrong_records.write_text(
                "{"
                f"\"policy_version\":\"{REDIPORTAL_RESOURCE_POLICY_VERSION}\","
                "\"reference_build\":\"GRCh38\","
                f"\"processed_table_sha256\":\"{sha256_file(redi)}\","
                f"\"processed_table_size_bytes\":{redi.stat().st_size},"
                "\"records\":2,"
                "\"normalization_rules\":\"GRCh38 exact REF/ALT forward-strand allele\","
                "\"processing_script_sha256\":\"scriptsha\""
                "}"
            )
            with self.assertRaisesRegex(ValueError, "record count mismatch"):
                load_rediportal_events(redi, wrong_records)

    def test_positive_and_negative_strand_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plus_vcf = self._write_vcf(root, [self._event_row(pos=101, ref="A", alt="G")])
            plus_events, _summary = load_vcf_events(plus_vcf, expected_sample="TEST")
            plus = evaluate_parent_rna_variants(
                sample="TEST",
                parent_record_id="P_PLUS",
                tx_blocks=[GenomicBlock("1", "+", 100, 103)],
                strand="+",
                reference_cds_seq="ACT",
                candidate_cds_seq="GCT",
                candidate_parent_sequence="A",
                vcf_events=plus_events,
                rediportal_events=set(),
            )
            self.assertEqual(plus["parent_summary"]["rna_variant_rescue_status"], "rna_variant_rescued")
            self.assertEqual(plus["event_rows"][0]["aa_position"], 1)

            neg_vcf = self._write_vcf(root, [self._event_row(pos=103, ref="T", alt="C")])
            neg_events, _summary = load_vcf_events(neg_vcf, expected_sample="TEST")
            neg = evaluate_parent_rna_variants(
                sample="TEST",
                parent_record_id="P_NEG",
                tx_blocks=[GenomicBlock("1", "-", 100, 103)],
                strand="-",
                reference_cds_seq="ACT",
                candidate_cds_seq="GCT",
                candidate_parent_sequence="A",
                vcf_events=neg_events,
                rediportal_events=set(),
            )
            self.assertEqual(neg["parent_summary"]["rna_variant_rescue_status"], "rna_variant_rescued")
            self.assertEqual((neg["event_rows"][0]["chrom"], neg["event_rows"][0]["pos_1based"]), ("1", 103))
            self.assertEqual((neg["event_rows"][0]["ref"], neg["event_rows"][0]["alt"]), ("T", "C"))

    def test_rna_variant_calling_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vcf = self._write_vcf(root, [self._event_row()])
            bam = root / "sample.bam"
            ref = root / "ref.fa"
            bed = root / "exons.bed"
            evidence = root / "rna.variant_evidence.tsv"
            for path in [bam, ref, bed, evidence]:
                path.write_text(path.name + "\n")

            def ident(path: Path) -> dict[str, object]:
                return {
                    "path": str(path),
                    "exists": True,
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }

            manifest = {
                "policy_version": RNA_VARIANT_CALLING_POLICY_VERSION,
                "sample": "TEST",
                "run_status": "complete",
                "input_signature": {
                    "policy_version": RNA_VARIANT_CALLING_POLICY_VERSION,
                    "sample": "TEST",
                    "sample_bam": ident(bam),
                    "reference_fasta": ident(ref),
                    "exon_bed": ident(bed),
                    "parameters": {
                        "mpileup_flag_filter": "0xF04",
                        "read_mapq_min": 20,
                        "base_quality_min": 20,
                        "depth_cap": 100000,
                        "variant_qual_min": 30,
                        "total_depth_min_ad_derived": 10,
                        "variant_mq_min": 20,
                        "vaf_min_ad_derived": 0.05,
                        "alt_reads_min": 3,
                        "normalization": "bcftools norm -f REF -m -any; bcftools norm -d exact",
                    },
                    "bcftools_version": "bcftools 1.20",
                    "calling_script_sha256": "scriptsha",
                },
                "output_signature": {
                    "raw_bcf": ident(root / "sample.bam"),
                    "call_vcf_sorted": ident(vcf),
                    "norm_split_dedup_vcf": ident(vcf),
                    "filtered_vcf": ident(vcf),
                    "variant_evidence_tsv": ident(evidence),
                },
                "filter_stats": {"normalized_records": 1, "filtered_records": 1},
            }
            manifest_path = root / "rna.variant_calling.manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            summary = validate_rna_variant_calling_manifest(
                manifest_path,
                vcf_path=vcf,
                expected_sample="TEST",
            )
            self.assertEqual(summary["status"], "validated")

            bad_sample = root / "bad_sample.json"
            altered = json.loads(json.dumps(manifest))
            altered["sample"] = "OTHER"
            bad_sample.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "sample mismatch"):
                validate_rna_variant_calling_manifest(bad_sample, vcf_path=vcf, expected_sample="TEST")

            bad_param = root / "bad_param.json"
            altered = json.loads(json.dumps(manifest))
            altered["input_signature"]["parameters"]["variant_mq_min"] = 30
            bad_param.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "variant_mq_min mismatch"):
                validate_rna_variant_calling_manifest(bad_param, vcf_path=vcf, expected_sample="TEST")

            vcf.write_text(vcf.read_text() + "chr1\t999\t.\tC\tT\t35\tPASS\tMQ=60\tGT:AD:DP\t0/1:7,3:10\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_rna_variant_calling_manifest(manifest_path, vcf_path=vcf, expected_sample="TEST")

    def test_thresholds_and_editing_block_primary_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            low_alt_vcf = self._write_vcf(root, [self._event_row(ad="9,1")])
            low_events, _summary = load_vcf_events(low_alt_vcf, expected_sample="TEST")
            low = evaluate_parent_rna_variants(
                sample="TEST",
                parent_record_id="P_LOW",
                tx_blocks=[GenomicBlock("1", "+", 100, 103)],
                strand="+",
                reference_cds_seq="ACT",
                candidate_cds_seq="GCT",
                candidate_parent_sequence="A",
                vcf_events=low_events,
                rediportal_events=set(),
            )
            self.assertEqual(low["parent_summary"]["rna_variant_rescue_status"], "rna_variant_insufficient_support")
            self.assertEqual(low["parent_summary"]["primary_core_eligibility"], "provisional_reference_translation_mismatch")

            edit_vcf = self._write_vcf(root, [self._event_row(pos=101, ref="A", alt="G")])
            edit_events, _summary = load_vcf_events(edit_vcf, expected_sample="TEST")
            editing = evaluate_parent_rna_variants(
                sample="TEST",
                parent_record_id="P_EDIT",
                tx_blocks=[GenomicBlock("1", "+", 100, 103)],
                strand="+",
                reference_cds_seq="ACT",
                candidate_cds_seq="GCT",
                candidate_parent_sequence="A",
                vcf_events=edit_events,
                rediportal_events={("1", 101, "A", "G")},
            )
            self.assertEqual(editing["parent_summary"]["rna_variant_rescue_status"], "known_rna_editing_excluded")
            self.assertEqual(editing["event_rows"][0]["rediportal_status"], "known_rna_editing")

            exploratory = evaluate_parent_rna_variants(
                sample="TEST",
                parent_record_id="P_EXPLORATORY",
                tx_blocks=[GenomicBlock("1", "+", 100, 103)],
                strand="+",
                reference_cds_seq="ACT",
                candidate_cds_seq="GCT",
                candidate_parent_sequence="A",
                vcf_events=edit_events,
                rediportal_events=set(),
                rediportal_evaluation_status="not_evaluated_missing_resource_allowed",
            )
            self.assertEqual(exploratory["event_rows"][0]["rediportal_status"], "editing_resource_not_evaluated")
            self.assertEqual(
                exploratory["parent_summary"]["rna_variant_rescue_status"],
                "rna_variant_supported_editing_not_evaluated",
            )
            self.assertEqual(exploratory["parent_summary"]["primary_core_eligibility"], "exploratory_only")

            filtered_vcf = self._write_vcf(root, [self._event_row(vcf_filter="LowQual")])
            filtered_events, _summary = load_vcf_events(filtered_vcf, expected_sample="TEST")
            filtered = evaluate_parent_rna_variants(
                sample="TEST",
                parent_record_id="P_FILTER",
                tx_blocks=[GenomicBlock("1", "+", 100, 103)],
                strand="+",
                reference_cds_seq="ACT",
                candidate_cds_seq="GCT",
                candidate_parent_sequence="A",
                vcf_events=filtered_events,
                rediportal_events=set(),
            )
            self.assertEqual(filtered["parent_summary"]["rna_variant_rescue_status"], "rna_variant_insufficient_support")
            self.assertIn("vcf_filter_not_pass", filtered["event_rows"][0]["event_qc_reason"])

    def test_synonymous_cds_mismatch_does_not_block_protein_changing_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vcf = self._write_vcf(root, [self._event_row(pos=104, ref="G", alt="A")])
            events, _summary = load_vcf_events(vcf, expected_sample="TEST")
            result = evaluate_parent_rna_variants(
                sample="TEST",
                parent_record_id="P_SYN",
                tx_blocks=[GenomicBlock("1", "+", 100, 106)],
                strand="+",
                reference_cds_seq="GCTGCT",
                candidate_cds_seq="GCCACT",
                candidate_parent_sequence="AT",
                vcf_events=events,
                rediportal_events=set(),
            )
            summary = result["parent_summary"]
            self.assertEqual(summary["rna_variant_rescue_status"], "rna_variant_rescued")
            self.assertEqual(summary["required_variant_count"], 1)
            self.assertEqual(summary["synonymous_mismatch_count"], 1)

    def test_policy_v1_rejects_threshold_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary_min_alt_reads=3"):
            validate_policy(policy_version=POLICY_VERSION, primary_min_alt_reads=2, sensitivity_alt_reads=(2, 3, 5))


if __name__ == "__main__":
    unittest.main()
