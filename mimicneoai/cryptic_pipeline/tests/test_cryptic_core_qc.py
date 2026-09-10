from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mimicneoai.cryptic_pipeline.scripts.cryptic_core_qc import build_core, load_human_reference_matches
from mimicneoai.cryptic_pipeline.scripts.cryptic_junction_qc import manifest_critical_parameter_digest


class CrypticCoreQCTest(unittest.TestCase):
    codons = {
        "A": "GCT", "C": "TGT", "D": "GAT", "E": "GAA", "F": "TTT",
        "G": "GGT", "H": "CAT", "I": "ATT", "K": "AAA", "L": "TTA",
        "M": "ATG", "N": "AAT", "P": "CCT", "Q": "CAA", "R": "CGT",
        "S": "TCT", "T": "ACT", "V": "GTT", "W": "TGG", "Y": "TAT",
    }

    def _nt_for_peptide(self, peptide: str) -> str:
        return "".join(self.codons[aa] for aa in peptide)

    def _reverse_complement(self, seq: str) -> str:
        return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]

    def _sha256_file(self, path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _write_star_pair(self, root: Path, tumor_sj: Path, normal_sj: Path) -> Path:
        tumor_sha = self._sha256_file(tumor_sj)
        normal_sha = self._sha256_file(normal_sj)
        star_cmd = (
            "STAR --runThreadN 8 --genomeDir {index} --readFilesIn R1.fq.gz R2.fq.gz "
            "--readFilesCommand zcat --outSAMtype BAM Unsorted "
            "--outFilterMultimapNmax 20 --alignSJoverhangMin 8 "
            "--alignSJDBoverhangMin 1 --alignIntronMin 20 --alignIntronMax 1000000 "
            "--alignMatesGapMax 1000000 --outFileNamePrefix out/"
        ).format(index=root / "STAR-index")
        tumor_log = root / "tumor.Log.out"
        normal_log = root / "normal.Log.out"
        tumor_log.write_text(f"##### Command Line:\n{star_cmd}\n")
        normal_log.write_text(f"##### Command Line:\n{star_cmd}\n")
        tumor_log_sha = self._sha256_file(tumor_log)
        normal_log_sha = self._sha256_file(normal_log)
        critical_parameters = {
            "--readFilesCommand": ["zcat"],
            "--outSAMtype": ["BAM", "Unsorted"],
            "--outFilterMultimapNmax": ["20"],
            "--alignSJoverhangMin": ["8"],
            "--alignSJDBoverhangMin": ["1"],
            "--alignIntronMin": ["20"],
            "--alignIntronMax": ["1000000"],
            "--alignMatesGapMax": ["1000000"],
        }
        critical_digest = manifest_critical_parameter_digest(critical_parameters)
        tumor_manifest = root / "tumor_star.json"
        normal_manifest = root / "normal_star.json"
        source_manifest = root / "normal_source_star_alignment.manifest.json"
        source_manifest.write_text(json.dumps({
            "status": "completed",
            "input_signature": {
                "alignment_role": "control",
                "sample": "CTRL",
                "tumor_sample": "TEST",
                "control_sample": "CTRL",
            },
        }))
        source_manifest_sha = self._sha256_file(source_manifest)
        common = {
            "policy_version": "star_provenance_freeze_v1.0",
            "tumor_sample": "TEST",
            "normal_sample": "CTRL",
            "star_version": "STAR_2.5.3a_modified",
            "star_index": str(root / "STAR-index"),
        }
        tumor_manifest.write_text(json.dumps({
            **common,
            "status": "frozen_legacy_complete",
            "critical_parameter_compatibility": "compatible",
            "critical_star_parameters": critical_parameters,
            "critical_star_parameter_digest": critical_digest,
            "current_outputs": {
                "SJ.out.tab": {
                    "path": str(tumor_sj),
                    "sha256": tumor_sha,
                    "size_bytes": tumor_sj.stat().st_size,
                },
                "Log.out": {
                    "path": str(tumor_log),
                    "sha256": tumor_log_sha,
                    "size_bytes": tumor_log.stat().st_size,
                },
            },
        }))
        normal_manifest.write_text(json.dumps({
            **common,
            "status": "relocated_complete",
            "critical_parameter_compatibility": "compatible",
            "critical_star_parameters": critical_parameters,
            "critical_star_parameter_digest": critical_digest,
            "source_manifest_path": str(source_manifest),
            "source_manifest_sha256": source_manifest_sha,
            "current_outputs": {
                "SJ.out.tab": {
                    "path": str(normal_sj),
                    "sha256": normal_sha,
                    "size_bytes": normal_sj.stat().st_size,
                },
                "Log.out": {
                    "path": str(normal_log),
                    "sha256": normal_log_sha,
                    "size_bytes": normal_log.stat().st_size,
                },
            },
        }))
        star_pairs = root / "star_pairs.tsv"
        star_pairs.write_text(
            "tumor_sample\tnormal_sample\ttumor_sj_path\tnormal_sj_path\t"
            "tumor_sj_sha256\tnormal_sj_sha256\ttumor_star_manifest\tnormal_star_manifest\t"
            "critical_parameter_compatibility\ttumor_critical_parameter_digest\tnormal_critical_parameter_digest\n"
            f"TEST\tCTRL\t{tumor_sj}\t{normal_sj}\t{tumor_sha}\t{normal_sha}\t"
            f"{tumor_manifest}\t{normal_manifest}\tcompatible\t{critical_digest}\t{critical_digest}\n"
        )
        return star_pairs

    def _write_rna_vcf(self, root: Path, rows: list[str], sample: str = "TEST") -> Path:
        path = root / "rna.call.sorted.tags.vcf"
        path.write_text(
            "\n".join([
                "##fileformat=VCFv4.2",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
                "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths\">",
                "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read depth\">",
                "##INFO=<ID=MQ,Number=1,Type=Float,Description=\"Mapping quality\">",
                f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}.Aligned.out.sorted.bam",
                *rows,
            ]) + "\n"
        )
        return path

    def _write_rna_calling_manifest(self, vcf: Path, sample: str = "TEST") -> Path:
        root = vcf.parent
        bam = root / f"{sample}.Aligned.out.sorted.bam"
        ref = root / "GRCh38.fa"
        bed = root / "lncRNA.exons.merged.bed"
        evidence = root / "rna.variant_evidence.tsv"
        for path in [bam, ref, bed, evidence]:
            if not path.exists():
                path.write_text(path.name + "\n")

        def ident(path: Path) -> dict[str, object]:
            return {
                "path": str(path),
                "exists": True,
                "size": path.stat().st_size,
                "sha256": self._sha256_file(path),
            }

        manifest = root / "rna.variant_calling.manifest.json"
        manifest.write_text(json.dumps({
            "policy_version": "cryptic_known_branch_rna_variant_calling_v1.0",
            "sample": sample,
            "run_status": "complete",
            "input_signature": {
                "policy_version": "cryptic_known_branch_rna_variant_calling_v1.0",
                "sample": sample,
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
                "calling_script_sha256": "fixture_script",
            },
            "output_signature": {
                "raw_bcf": ident(bam),
                "call_vcf_sorted": ident(vcf),
                "norm_split_dedup_vcf": ident(vcf),
                "filtered_vcf": ident(vcf),
                "variant_evidence_tsv": ident(evidence),
            },
            "filter_stats": {"normalized_records": 1, "filtered_records": 1},
        }))
        return manifest

    def _write_rediportal(self, root: Path, rows: list[str] | None = None) -> Path:
        path = root / "rediportal.processed.tsv"
        path.write_text(
            "chrom\tpos_1based\tref\talt\tediting_type\tresource_record_id\tsource_release\n"
            + "\n".join(rows or [])
            + ("\n" if rows else "")
        )
        return path

    def _write_rediportal_manifest(self, path: Path) -> Path:
        manifest = path.with_suffix(".manifest.json")
        manifest.write_text(json.dumps({
            "policy_version": "rediportal_processed_resource_v1.0",
            "reference_build": "GRCh38",
            "processed_table_sha256": self._sha256_file(path),
            "processed_table_size_bytes": path.stat().st_size,
            "records": len(path.read_text().splitlines()) - 1,
            "normalization_rules": "fixture_exact_forward_allele",
            "processing_script_sha256": "fixture",
        }))
        return manifest

    def _write_inputs(self, root: Path) -> dict[str, Path]:
        annot = pd.DataFrame(
            [
                {
                    "Name": "ENST_NC.1",
                    "nc_class": "noncoding",
                    "is_aberrant": True,
                    "TPM_tumor": 10.0,
                    "TPM_ctrl": 0.0,
                    "log2FC": 6.0,
                },
                {
                    "Name": "ENST_COD.1",
                    "nc_class": "coding_or_other",
                    "is_aberrant": True,
                    "TPM_tumor": 10.0,
                    "TPM_ctrl": 0.0,
                    "log2FC": 6.0,
                },
                {
                    "Name": "TRINITY_X|NOVEL|.|.",
                    "nc_class": "novel",
                    "is_aberrant": True,
                    "TPM_tumor": 8.0,
                    "TPM_ctrl": 0.0,
                    "log2FC": 5.0,
                },
                {
                    "Name": "ENST_BAD.1",
                    "nc_class": "noncoding",
                    "is_aberrant": True,
                    "TPM_tumor": 9.0,
                    "TPM_ctrl": 0.0,
                    "log2FC": 5.5,
                },
                {
                    "Name": "ENST_RELAXED.1",
                    "nc_class": "noncoding",
                    "is_aberrant": True,
                    "TPM_tumor": 1.0,
                    "TPM_ctrl": 0.0,
                    "log2FC": 5.0,
                },
                {
                    "Name": "ENST_DUP.1",
                    "nc_class": "noncoding",
                    "is_aberrant": True,
                    "TPM_tumor": 11.0,
                    "TPM_ctrl": 0.0,
                    "log2FC": 6.5,
                },
            ]
        )
        annot_path = root / "sample.aberrant_noncoding.annot.csv"
        annot.to_csv(annot_path, index=False)

        orf_ids = [
            "ENST_NC.1.p1",
            "ENST_COD.1.p1",
            "TRINITY_X|NOVEL|.|..p1",
            "ENST_BAD.1.p1",
            "ENST_RELAXED.1.p1",
            "ENST_DUP.1.p1",
        ]
        orf_final = pd.DataFrame({"TranscriptID": orf_ids})
        orf_final_path = root / "orf_final.csv"
        orf_final.to_csv(orf_final_path, index=False)

        fasta_path = root / "orf_filtered.fa"
        fasta_path.write_text(
            ">ENST_NC.1.p1\nACDEFGHIKLMNP\n"
            ">ENST_COD.1.p1\nACDEFGHIKLMNP\n"
            ">TRINITY_X|NOVEL|.|..p1\nKLMNPQRSTVWY*\n"
            ">ENST_BAD.1.p1\nACDEF*GHIKLMNP\n"
            ">ENST_RELAXED.1.p1\nTVWYACDEFGHIK\n"
            ">ENST_DUP.1.p1\nACDEFGHIKLMNP\n"
        )

        human_path = root / "human.fa"
        human_path.write_text(">HUMAN1\nTTTACDEFGHITTT\n")

        return {
            "annot": annot_path,
            "orf_final": orf_final_path,
            "fasta": fasta_path,
            "human": human_path,
        }

    def _write_coordinate_inputs(self, root: Path, parent_sequences: dict[str, str]) -> dict[str, Path]:
        import pysam

        bed12 = root / "orf.noUnmap.noSup.bed12"
        bam = root / "orf2genome.noUnmap.noSup.bam"
        cds = root / "sample.SEPs.cds.fa"
        ref = root / "GRCh38.test.fa"
        contig = ["N"] * 20000

        def place(start0: int, seq: str) -> None:
            contig[start0 : start0 + len(seq)] = list(seq)

        with bed12.open("w") as handle:
            for idx, (parent_id, peptide) in enumerate(parent_sequences.items()):
                start = 100 + idx * 1000
                cds_seq = self._nt_for_peptide(peptide)
                size = len(cds_seq)
                strand = "-"
                block_sizes = [size]
                block_starts = [0]
                if parent_id == "ENST_NC.1.p1":
                    strand = "+"
                    block_sizes = [18, size - 18]
                    block_starts = [0, 100]
                    place(start, cds_seq[:18])
                    place(start + 100, cds_seq[18:])
                    chrom_end = start + 100 + block_sizes[-1]
                elif parent_id.startswith("TRINITY"):
                    place(start, self._reverse_complement(cds_seq))
                    chrom_end = start + size
                else:
                    strand = "+"
                    place(start, cds_seq)
                    chrom_end = start + size
                handle.write(
                    f"chr1\t{start}\t{chrom_end}\t{parent_id}\t30\t{strand}\t"
                    f"{start}\t{chrom_end}\t0\t{len(block_sizes)}\t"
                    f"{','.join(str(v) for v in block_sizes)},\t"
                    f"{','.join(str(v) for v in block_starts)},\n"
                )

        ref.write_text(">chr1\n" + "".join(contig) + "\n")
        pysam.faidx(str(ref))

        with cds.open("w") as handle:
            for parent_id, peptide in parent_sequences.items():
                handle.write(f">{parent_id}\n{self._nt_for_peptide(peptide)}\n")

        header = {"HD": {"VN": "1.6"}, "SQ": [{"SN": "chr1", "LN": 1000000}]}
        with pysam.AlignmentFile(bam, "wb", header=header) as out:
            for idx, (parent_id, peptide) in enumerate(parent_sequences.items()):
                start = 100 + idx * 1000
                size = len(peptide) * 3
                rec = pysam.AlignedSegment()
                rec.query_name = parent_id
                rec.query_sequence = "A" * size
                rec.flag = 16 if parent_id.startswith("TRINITY") else 0
                rec.reference_id = 0
                rec.reference_start = start
                rec.mapping_quality = 30
                if parent_id == "ENST_NC.1.p1":
                    rec.cigar = [(0, 18), (3, 82), (0, size - 18)]
                else:
                    rec.cigar = [(0, size)]
                out.write(rec)
                if parent_id.endswith("DUP.1.p1"):
                    sec = pysam.AlignedSegment()
                    sec.query_name = parent_id
                    sec.query_sequence = "A" * size
                    sec.flag = 256
                    sec.reference_id = 0
                    sec.reference_start = start + 200
                    sec.mapping_quality = 30
                    sec.cigar = [(0, size)]
                    out.write(sec)
                    sup = pysam.AlignedSegment()
                    sup.query_name = parent_id
                    sup.query_sequence = "A" * size
                    sup.flag = 2048
                    sup.reference_id = 0
                    sup.reference_start = start + 400
                    sup.mapping_quality = 30
                    sup.cigar = [(0, size)]
                    out.write(sup)
        return {"bed12": bed12, "bam": bam, "cds": cds, "ref": ref}

    def _args(self, paths: dict[str, Path], outdir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            sample="TEST",
            policy_version="cryptic_core_qc_v1.0",
            ae_seps_fasta=str(paths["fasta"]),
            aeseps_annotation=str(paths["annot"]),
            orf_filtered_fasta=str(paths["fasta"]),
            orf_final=str(paths["orf_final"]),
            orf_bed12="",
            orf_bam="",
            orf_cds_fasta="",
            outdir=str(outdir),
            human_proteome_fasta=str(paths["human"]),
            allow_missing_human_reference=False,
            matched_control_sample="CTRL",
            reference_genome_fasta="",
            reference_gtf="",
            reference_lnc_gtf="",
            reference_build="GRCh38",
            strandedness="reverse",
            min_tpm_tumor=5.0,
            max_tpm_ctrl=0.5,
            min_log2fc=4.0,
            mhc_i_lengths="8",
            mhc_ii_lengths="13",
            candidate_selection_mode="all",
            max_hla_i_peptides=None,
            max_hla_ii_peptides=None,
            coordinate_min_mapq=20,
        )

    def test_core_filters_and_resume_signature(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            outdir = root / "core"

            manifest = build_core(self._args(paths, outdir))
            self.assertEqual(manifest["run_status"], "complete")

            parents = pd.read_csv(outdir / "cryptic_parent_records.tsv", sep="\t")
            core_parents = pd.read_csv(outdir / "cryptic_parent_core.tsv", sep="\t")
            excluded = pd.read_csv(outdir / "cryptic_parent_excluded.tsv", sep="\t")
            peptide_map = pd.read_csv(outdir / "cryptic_peptide_parent_map.tsv", sep="\t")

            self.assertEqual(
                set(core_parents["parent_record_id"]),
                {"ENST_NC.1.p1", "TRINITY_X|NOVEL|.|..p1", "ENST_DUP.1.p1"},
            )
            self.assertIn("ENST_COD.1.p1", set(excluded["parent_record_id"]))
            self.assertIn("ENST_BAD.1.p1", set(excluded["parent_record_id"]))
            self.assertIn("ENST_RELAXED.1.p1", set(excluded["parent_record_id"]))
            bad_reason = parents.set_index("parent_record_id").loc["ENST_BAD.1.p1", "parent_qc_reasons"]
            self.assertIn("internal_stop", bad_reason)
            relaxed_reason = parents.set_index("parent_record_id").loc["ENST_RELAXED.1.p1", "parent_qc_reasons"]
            self.assertIn("tumor_tpm_below_threshold", relaxed_reason)
            relaxed = parents.set_index("parent_record_id").loc["ENST_RELAXED.1.p1"]
            self.assertEqual(relaxed["expression_qc_status"], "fail")
            self.assertFalse(bool(relaxed["upstream_is_aberrant_consistent"]))

            matched = peptide_map[peptide_map["human_reference_peptide_status"].eq("human_reference_peptide_match")]
            self.assertEqual(set(matched["peptide"]), {"ACDEFGHI"})
            hla_i_fasta = (outdir / "cryptic_peptide_core_hla_i.fasta").read_text()
            self.assertNotIn("ACDEFGHI", hla_i_fasta)
            self.assertIn("KLMNPQRS", hla_i_fasta)
            hla_i_headers = [line for line in hla_i_fasta.splitlines() if line.startswith(">")]
            self.assertEqual(len(hla_i_headers), len(set(hla_i_headers)))
            self.assertTrue(all(header.startswith(">cryptic_core_MHC-I_") for header in hla_i_headers))

            reused = build_core(self._args(paths, outdir))
            self.assertTrue(reused["reused"])

            mutated_args = self._args(paths, outdir)
            mutated_args.min_tpm_tumor = 1.0
            with self.assertRaises(ValueError):
                build_core(mutated_args)

    def test_code_signature_change_invalidates_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            outdir = root / "code_signature"
            args = self._args(paths, outdir)

            manifest = build_core(args)
            self.assertIn("script_sha256", manifest["input_signature"])

            manifest_path = outdir / "run_manifest.json"
            saved = json.loads(manifest_path.read_text())
            saved["input_signature"]["script_sha256"] = "different-script"
            manifest_path.write_text(json.dumps(saved, indent=2))

            with self.assertRaises(ValueError):
                build_core(args)

    def test_zero_candidate_core_can_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            outdir = root / "zero"
            args = self._args(paths, outdir)
            args.min_tpm_tumor = 1000.0

            manifest = build_core(args)
            self.assertEqual(manifest["stage_counts"]["parent_core_records"], 0)
            self.assertEqual(manifest["stage_counts"]["unique_peptide_core_records"], 0)
            self.assertEqual((outdir / "cryptic_peptide_core.fasta").read_text(), "")
            reused = build_core(args)
            self.assertTrue(reused["reused"])

    def test_ranked_cap_limits_unique_peptides_and_records_deferred_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            outdir = root / "ranked"
            args = self._args(paths, outdir)
            args.candidate_selection_mode = "ranked_cap"
            args.max_hla_i_peptides = 1
            args.max_hla_ii_peptides = 1

            manifest = build_core(args)

            self.assertEqual(manifest["candidate_selection"]["mode"], "ranked_cap")
            self.assertEqual(manifest["stage_counts"]["hla_i_unique_peptide_core_records"], 1)
            self.assertEqual(manifest["stage_counts"]["hla_ii_unique_peptide_core_records"], 1)
            self.assertTrue((outdir / "cryptic_parent_ranked.tsv").exists())
            self.assertTrue((outdir / "cryptic_peptide_selection_evidence.tsv").exists())
            self.assertTrue((outdir / "cryptic_deferred_parent.tsv").exists())
            self.assertTrue((outdir / "cryptic_peptide_deferred.tsv").exists())
            evidence = pd.read_csv(outdir / "cryptic_peptide_selection_evidence.tsv", sep="\t")
            self.assertIn("human_reference_peptide_match", set(evidence["selection_reason"].dropna().astype(str)))
            self.assertIn("selected_for_binding", set(evidence["selection_status"]))
            parent_map = pd.read_csv(outdir / "cryptic_peptide_parent_map.tsv", sep="\t")
            selected = parent_map[parent_map["peptide_record_id"].fillna("").astype(str).ne("")]
            self.assertEqual(selected[["mhc_class", "peptide"]].drop_duplicates().shape[0], 2)
            self.assertTrue(
                selected["human_reference_peptide_status"].eq("human_reference_peptide_not_detected").all()
            )
            stage = pd.read_csv(outdir / "stagewise_qc.tsv", sep="\t").set_index("stage")["count"].to_dict()
            self.assertEqual(manifest["candidate_selection"]["materialization_policy"], "initial_cap_plus_ranked_parent_stream")
            self.assertFalse(manifest["candidate_selection"]["evaluated_deferred_peptide_universe_complete"])
            self.assertIn("unmaterialized_deferred_parent_records", manifest["candidate_selection"])
            self.assertEqual(
                int(stage["peptide_excluded_window_rows"]),
                int(parent_map["human_reference_peptide_status"].eq("human_reference_peptide_match").sum()),
            )
            deferred = pd.read_csv(outdir / "cryptic_peptide_deferred.tsv", sep="\t")
            if not deferred.empty:
                self.assertTrue(deferred["peptide_core_status"].eq("deferred").all())
            self.assertNotIn("not_selected_due_to_analysis_cap", set(parent_map["peptide_qc_reasons"].dropna().astype(str)))

    def test_ranked_cap_repeated_parent_map_is_bound_to_materialized_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent_count = 2000
            annot_rows = []
            orf_ids = []
            fasta_path = root / "orf_filtered.fa"
            with fasta_path.open("w") as fasta:
                for idx in range(parent_count):
                    transcript = f"TX_REPEAT_{idx}"
                    parent_id = f"{transcript}.p1"
                    annot_rows.append({
                        "Name": transcript,
                        "nc_class": "noncoding",
                        "is_aberrant": True,
                        "TPM_tumor": 20.0 - (idx * 0.0001),
                        "TPM_ctrl": 0.0,
                        "log2FC": 6.0,
                    })
                    orf_ids.append(parent_id)
                    fasta.write(f">{parent_id}\nACDEFGHIKLMNPQRSTVWY\n")
            annot_path = root / "sample.aberrant_noncoding.annot.csv"
            pd.DataFrame(annot_rows).to_csv(annot_path, index=False)
            orf_final_path = root / "orf_final.csv"
            pd.DataFrame({"TranscriptID": orf_ids}).to_csv(orf_final_path, index=False)
            human_path = root / "human.fa"
            human_path.write_text(">HUMAN\nYYYYYYYYYYYYYYYYYYYY\n")
            paths = {"annot": annot_path, "orf_final": orf_final_path, "fasta": fasta_path, "human": human_path}
            outdir = root / "ranked_repeat"
            args = self._args(paths, outdir)
            args.candidate_selection_mode = "ranked_cap"
            args.max_hla_i_peptides = 1
            args.max_hla_ii_peptides = 1

            manifest = build_core(args)

            parent_map = pd.read_csv(outdir / "cryptic_peptide_parent_map.tsv", sep="\t")
            self.assertLessEqual(len(parent_map), 4)
            self.assertEqual(manifest["candidate_selection"]["selected_hla_i"], 1)
            self.assertEqual(manifest["candidate_selection"]["selected_hla_ii"], 1)
            self.assertLessEqual(manifest["candidate_selection"]["materialized_parent_records"], 2)
            self.assertGreater(manifest["candidate_selection"]["unmaterialized_deferred_parent_records"], 1900)

    def test_human_reference_keeps_standard_windows_around_u(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            human = root / "human_with_u.fa"
            human.write_text(">HUMAN_U\nUUUACDEFGHIKUUUKLMNPQRS\n")

            matches, summary = load_human_reference_matches(
                human,
                {"ACDEFGHI", "KLMNPQRS", "FGHIKKLM"},
            )

            self.assertEqual(matches, {"ACDEFGHI", "KLMNPQRS"})
            self.assertEqual(summary["records_encountered"], 1)
            self.assertEqual(summary["records_with_noncanonical_residues"], 1)
            self.assertEqual(summary["standard_windows_evaluated"], 3)

    def test_v11_writes_coordinate_sidecars_and_marks_secondary_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            coordinate_inputs = self._write_coordinate_inputs(
                root,
                {
                    "ENST_NC.1.p1": "ACDEFGHIKLMNP",
                    "TRINITY_X|NOVEL|.|..p1": "KLMNPQRSTVWY",
                    "ENST_DUP.1.p1": "ACDEFGHIKLMNP",
                },
            )
            outdir = root / "core_v11"
            args = self._args(paths, outdir)
            args.policy_version = "cryptic_core_qc_v1.1"
            args.orf_bed12 = str(coordinate_inputs["bed12"])
            args.orf_bam = str(coordinate_inputs["bam"])
            args.orf_cds_fasta = str(coordinate_inputs["cds"])
            args.reference_genome_fasta = str(coordinate_inputs["ref"])

            manifest = build_core(args)

            self.assertEqual(manifest["policy_version"], "cryptic_core_qc_v1.1")
            self.assertIn("cryptic_parent_coordinates.tsv", manifest["output_signature"])
            self.assertIn("cryptic_parent_orfcds.tsv", manifest["output_signature"])
            self.assertIn("cryptic_peptide_genomic_footprint.tsv", manifest["output_signature"])
            parent_coords = pd.read_csv(outdir / "cryptic_parent_coordinates.tsv", sep="\t")
            status_by_parent = parent_coords.set_index("parent_record_id")["coordinate_mapping_status"].to_dict()
            self.assertEqual(status_by_parent["ENST_NC.1.p1"], "coordinate_evaluable")
            self.assertEqual(status_by_parent["TRINITY_X|NOVEL|.|..p1"], "coordinate_evaluable")
            self.assertEqual(status_by_parent["ENST_DUP.1.p1"], "ambiguous_candidate_mapping")
            self.assertEqual(
                parent_coords.set_index("parent_record_id").loc[
                    "ENST_NC.1.p1", "reference_genomic_translation_status"
                ],
                "pass",
            )
            self.assertEqual(
                parent_coords.set_index("parent_record_id").loc[
                    "TRINITY_X|NOVEL|.|..p1", "reference_genomic_translation_status"
                ],
                "pass",
            )
            self.assertEqual(
                int(parent_coords.set_index("parent_record_id").loc[
                    "ENST_DUP.1.p1", "supplementary_alignment_count"
                ]),
                1,
            )
            footprints = pd.read_csv(outdir / "cryptic_peptide_genomic_footprint.tsv", sep="\t")
            self.assertIn("coordinate_evaluable", set(footprints["candidate_coordinate_status"]))
            self.assertIn("ambiguous_candidate_mapping", set(footprints["candidate_coordinate_status"]))
            self.assertEqual(manifest["stage_counts"]["candidate_parents_coordinate_evaluable"], 2)
            self.assertEqual(manifest["stage_counts"]["candidate_parents_coordinate_not_evaluable"], 1)
            self.assertEqual(manifest["stage_counts"]["candidate_parents_reference_translation_pass"], 3)

    def test_v11_junction_qc_filters_before_peptide_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            coordinate_inputs = self._write_coordinate_inputs(
                root,
                {
                    "ENST_NC.1.p1": "ACDEFGHIKLMNP",
                    "TRINITY_X|NOVEL|.|..p1": "KLMNPQRSTVWY",
                    "ENST_DUP.1.p1": "ACDEFGHIKLMNP",
                },
            )
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            tumor_sj.write_text("chr1\t119\t200\t1\t1\t1\t2\t0\t20\n")
            normal_sj.write_text("chr1\t119\t200\t1\t1\t1\t1\t0\t20\n")
            star_pairs = self._write_star_pair(root, tumor_sj, normal_sj)
            outdir = root / "core_v11_junction"
            args = self._args(paths, outdir)
            args.policy_version = "cryptic_core_qc_v1.1"
            args.orf_bed12 = str(coordinate_inputs["bed12"])
            args.orf_bam = str(coordinate_inputs["bam"])
            args.orf_cds_fasta = str(coordinate_inputs["cds"])
            args.reference_genome_fasta = str(coordinate_inputs["ref"])
            args.junction_qc_enabled = True
            args.junction_policy_version = "junction_qc_v1.0"
            args.star_pair_inputs = str(star_pairs)
            args.primary_min_tumor_unique_reads = 2
            args.junction_sensitivity_thresholds = "1,2,3,5"

            manifest = build_core(args)

            core_parents = pd.read_csv(outdir / "cryptic_parent_core.tsv", sep="\t")
            excluded = pd.read_csv(outdir / "cryptic_parent_excluded.tsv", sep="\t")
            junction_summary = pd.read_csv(outdir / "cryptic_parent_junction_summary.tsv", sep="\t")
            self.assertEqual(set(core_parents["parent_record_id"]), {"ENST_NC.1.p1", "TRINITY_X|NOVEL|.|..p1"})
            self.assertIn("ENST_DUP.1.p1", set(excluded["parent_record_id"]))
            self.assertTrue((outdir / "cryptic_peptide_junction_evidence.tsv").exists())
            self.assertEqual(
                junction_summary.set_index("parent_record_id").loc["ENST_NC.1.p1", "junction_qc_status"],
                "all_required_junctions_tumor_unique_ge2",
            )
            self.assertEqual(manifest["junction_qc"]["enabled"], True)
            self.assertEqual(manifest["stage_counts"]["junction_parent_primary_core_eligible"], 2)
            self.assertIn("junction_qc_sha256", manifest["input_signature"]["junction_qc"])
            self.assertIn("pair_row_sha256", manifest["input_signature"]["junction_qc"]["star_pair_validation"])

            saved_path = outdir / "run_manifest.json"
            saved = json.loads(saved_path.read_text())
            saved["input_signature"]["junction_qc"]["junction_qc_sha256"] = "changed"
            saved_path.write_text(json.dumps(saved, indent=2))
            with self.assertRaises(ValueError):
                build_core(args)

    def test_v11_junction_qc_allows_ranked_cap_with_ranked_parent_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            coordinate_inputs = self._write_coordinate_inputs(root, {"ENST_NC.1.p1": "ACDEFGHIKLMNP"})
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            tumor_sj.write_text("chr1\t119\t200\t1\t1\t1\t2\t0\t20\n")
            normal_sj.write_text("chr1\t119\t200\t1\t1\t1\t0\t0\t20\n")
            args = self._args(paths, root / "core_v11_ranked_junction")
            args.policy_version = "cryptic_core_qc_v1.1"
            args.orf_bed12 = str(coordinate_inputs["bed12"])
            args.orf_bam = str(coordinate_inputs["bam"])
            args.orf_cds_fasta = str(coordinate_inputs["cds"])
            args.reference_genome_fasta = str(coordinate_inputs["ref"])
            args.junction_qc_enabled = True
            args.junction_policy_version = "junction_qc_v1.0"
            args.star_pair_inputs = str(self._write_star_pair(root, tumor_sj, normal_sj))
            args.primary_min_tumor_unique_reads = 2
            args.junction_sensitivity_thresholds = "1,2,3,5"
            args.candidate_selection_mode = "ranked_cap"
            args.max_hla_i_peptides = 1
            args.max_hla_ii_peptides = 1
            manifest = build_core(args)

            self.assertEqual(manifest["run_status"], "complete")
            self.assertFalse(manifest["candidate_selection"]["evaluated_deferred_peptide_universe_complete"])
            self.assertEqual(manifest["candidate_selection"]["materialization_coverage"], "selected_and_boundary_candidates")
            self.assertEqual(manifest["stage_counts"]["junction_qc_enabled"], 1)
            parent_ranked = pd.read_csv(root / "core_v11_ranked_junction" / "cryptic_parent_ranked.tsv", sep="\t")
            self.assertIn("ranking_digest", parent_ranked.columns)
            sensitivity_header = (
                root / "core_v11_ranked_junction" / "junction_threshold_sensitivity.tsv"
            ).read_text().splitlines()[0].split("\t")
            self.assertEqual(len(sensitivity_header), len(set(sensitivity_header)))

    def test_v11_reference_translation_mismatch_is_not_coordinate_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            coordinate_inputs = self._write_coordinate_inputs(
                root,
                {
                    "ENST_NC.1.p1": "ACDEFGHIKLMNP",
                    "TRINITY_X|NOVEL|.|..p1": "KLMNPQRSTVWY",
                    "ENST_DUP.1.p1": "ACDEFGHIKLMNP",
                },
            )
            ref = Path(coordinate_inputs["ref"])
            text = ref.read_text()
            ref.write_text(text.replace("GCT", "AAA", 1))
            import pysam
            pysam.faidx(str(ref))

            outdir = root / "core_v11_mismatch"
            args = self._args(paths, outdir)
            args.policy_version = "cryptic_core_qc_v1.1"
            args.orf_bed12 = str(coordinate_inputs["bed12"])
            args.orf_bam = str(coordinate_inputs["bam"])
            args.orf_cds_fasta = str(coordinate_inputs["cds"])
            args.reference_genome_fasta = str(ref)

            manifest = build_core(args)
            parent_coords = pd.read_csv(outdir / "cryptic_parent_coordinates.tsv", sep="\t")
            row = parent_coords.set_index("parent_record_id").loc["ENST_NC.1.p1"]
            self.assertEqual(row["reference_genomic_translation_status"], "reference_translation_mismatch")
            self.assertIn("reference_translation_mismatch", row["coordinate_mapping_reasons"])
            self.assertEqual(row["rna_variant_coordinate_status"], "RNA_variant_aware_not_evaluated")
            self.assertEqual(manifest["stage_counts"]["candidate_parents_reference_translation_mismatch"], 1)

    def test_v11_rna_variant_rescue_restores_reference_mismatch_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self._write_inputs(root)
            coordinate_inputs = self._write_coordinate_inputs(
                root,
                {
                    "ENST_NC.1.p1": "ACDEFGHIKLMNP",
                    "TRINITY_X|NOVEL|.|..p1": "KLMNPQRSTVWY",
                    "ENST_DUP.1.p1": "ACDEFGHIKLMNP",
                },
            )
            ref = Path(coordinate_inputs["ref"])
            text = ref.read_text()
            ref.write_text(text.replace("GCT", "ACT", 1))
            import pysam
            pysam.faidx(str(ref))
            rna_vcf = self._write_rna_vcf(
                root,
                ["chr1\t101\t.\tA\tG\t35\tPASS\tMQ=60\tGT:AD:DP\t0/1:7,3:10"],
            )
            rna_calling_manifest = self._write_rna_calling_manifest(rna_vcf)
            rediportal = self._write_rediportal(root)
            rediportal_manifest = self._write_rediportal_manifest(rediportal)

            outdir = root / "core_v11_rna_rescue"
            args = self._args(paths, outdir)
            args.policy_version = "cryptic_core_qc_v1.1"
            args.orf_bed12 = str(coordinate_inputs["bed12"])
            args.orf_bam = str(coordinate_inputs["bam"])
            args.orf_cds_fasta = str(coordinate_inputs["cds"])
            args.reference_genome_fasta = str(ref)
            args.rna_variant_editing_qc_enabled = True
            args.rna_variant_qc_policy_version = "cryptic_rna_variant_editing_qc_v1.0"
            args.rna_variant_vcf = str(rna_vcf)
            args.rna_variant_calling_manifest = str(rna_calling_manifest)
            args.rediportal_processed_table = str(rediportal)
            args.rediportal_resource_manifest = str(rediportal_manifest)
            args.allow_missing_rediportal_resource = False
            args.allow_legacy_rna_variant_vcf = False
            args.allow_legacy_duplicate_vcf = False
            args.rna_variant_min_mapping_quality = 20.0
            args.rna_variant_min_base_quality = 20.0
            args.rna_variant_min_variant_qual = 30.0
            args.rna_variant_min_total_depth = 10
            args.rna_variant_min_variant_allele_fraction = 0.05
            args.rna_variant_primary_min_alt_reads = 3
            args.rna_variant_sensitivity_alt_reads = "2,3,5"

            manifest = build_core(args)

            parent_coords = pd.read_csv(outdir / "cryptic_parent_coordinates.tsv", sep="\t")
            row = parent_coords.set_index("parent_record_id").loc["ENST_NC.1.p1"]
            self.assertEqual(row["reference_genomic_translation_status"], "rna_variant_rescued")
            self.assertEqual(row["rna_variant_coordinate_status"], "rna_variant_rescued")
            core_parents = pd.read_csv(outdir / "cryptic_parent_core.tsv", sep="\t")
            self.assertIn("ENST_NC.1.p1", set(core_parents["parent_record_id"]))
            parent_summary = pd.read_csv(outdir / "cryptic_parent_rna_variant_summary.tsv", sep="\t")
            self.assertEqual(
                parent_summary.set_index("parent_record_id").loc["ENST_NC.1.p1", "rna_variant_rescue_status"],
                "rna_variant_rescued",
            )
            peptide_evidence = pd.read_csv(outdir / "cryptic_peptide_rna_variant_evidence.tsv", sep="\t")
            self.assertIn("YES", set(peptide_evidence["overlaps_variant_aa"].astype(str)))
            self.assertEqual(manifest["stage_counts"]["candidate_parents_reference_translation_rna_variant_rescued"], 1)
            self.assertIn("rna_variant_qc.manifest.json", manifest["output_signature"])

            exploratory_outdir = root / "core_v11_rna_rescue_exploratory"
            exploratory_args = self._args(paths, exploratory_outdir)
            exploratory_args.policy_version = "cryptic_core_qc_v1.1"
            exploratory_args.orf_bed12 = str(coordinate_inputs["bed12"])
            exploratory_args.orf_bam = str(coordinate_inputs["bam"])
            exploratory_args.orf_cds_fasta = str(coordinate_inputs["cds"])
            exploratory_args.reference_genome_fasta = str(ref)
            exploratory_args.rna_variant_editing_qc_enabled = True
            exploratory_args.rna_variant_qc_policy_version = "cryptic_rna_variant_editing_qc_v1.0"
            exploratory_args.rna_variant_vcf = str(rna_vcf)
            exploratory_args.rna_variant_calling_manifest = str(root / "missing.variant_calling.manifest.json")
            exploratory_args.rediportal_processed_table = ""
            exploratory_args.rediportal_resource_manifest = ""
            exploratory_args.allow_missing_rediportal_resource = True
            exploratory_args.allow_legacy_rna_variant_vcf = True
            exploratory_args.allow_legacy_duplicate_vcf = False
            exploratory_args.rna_variant_min_mapping_quality = 20.0
            exploratory_args.rna_variant_min_base_quality = 20.0
            exploratory_args.rna_variant_min_variant_qual = 30.0
            exploratory_args.rna_variant_min_total_depth = 10
            exploratory_args.rna_variant_min_variant_allele_fraction = 0.05
            exploratory_args.rna_variant_primary_min_alt_reads = 3
            exploratory_args.rna_variant_sensitivity_alt_reads = "2,3,5"
            exploratory_manifest = build_core(exploratory_args)
            self.assertEqual(exploratory_manifest["run_status"], "complete_exploratory")
            self.assertFalse(exploratory_manifest["binding_eligible"])
            exploratory_summary = pd.read_csv(
                exploratory_outdir / "cryptic_parent_rna_variant_summary.tsv",
                sep="\t",
            )
            self.assertEqual(
                exploratory_summary.set_index("parent_record_id").loc[
                    "ENST_NC.1.p1", "rna_variant_rescue_status"
                ],
                "rna_variant_supported_editing_not_evaluated",
            )
            exploratory_core = pd.read_csv(exploratory_outdir / "cryptic_parent_core.tsv", sep="\t")
            self.assertNotIn("ENST_NC.1.p1", set(exploratory_core["parent_record_id"]))

            saved_path = outdir / "run_manifest.json"
            saved = json.loads(saved_path.read_text())
            saved["input_signature"]["rna_variant_editing_qc"]["rna_variant_qc_sha256"] = "changed"
            saved_path.write_text(json.dumps(saved, indent=2))
            with self.assertRaises(ValueError):
                build_core(args)


if __name__ == "__main__":
    unittest.main()
