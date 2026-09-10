from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from mimicneoai.cryptic_pipeline.scripts.cryptic_junction_qc import (
    PRIMARY_PASS,
    PROVISIONAL_LOW_SUPPORT,
    PROVISIONAL_MAPPING,
    PROVISIONAL_TRANSLATION,
    annotate_peptide_junctions,
    build_junction_qc,
    evaluate_parent_junctions,
    parse_cigar_junctions,
    parse_sj_out,
    manifest_critical_parameter_digest,
    sha256_file,
    validate_star_pair_contract,
)


class CrypticJunctionQCTest(unittest.TestCase):
    def _write_sj(self, path: Path, rows: list[tuple[str, int, int, int, int, int, int, int, int]]) -> None:
        path.write_text("\n".join("\t".join(map(str, row)) for row in rows) + "\n")

    def _write_star_pair(self, root: Path, tumor_sj: Path, normal_sj: Path) -> Path:
        tumor_sha = sha256_file(tumor_sj)
        normal_sha = sha256_file(normal_sj)
        star_index = root / "STAR-index"
        star_index.mkdir()
        for name in ("Genome", "SA", "SAindex", "sjdbList.out.tab"):
            (star_index / name).write_text(f"{name}\n")
        star_cmd = (
            "STAR --runThreadN 8 --genomeDir {index} --readFilesIn R1.fq.gz R2.fq.gz "
            "--readFilesCommand zcat --outSAMtype BAM Unsorted "
            "--outFilterMultimapNmax 20 --alignSJoverhangMin 8 "
            "--alignSJDBoverhangMin 1 --alignIntronMin 20 --alignIntronMax 1000000 "
            "--alignMatesGapMax 1000000 --outFileNamePrefix out/"
        ).format(index=star_index)
        tumor_log = root / "tumor.Log.out"
        normal_log = root / "normal.Log.out"
        tumor_log.write_text(f"##### Command Line:\n{star_cmd}\n")
        normal_log.write_text(f"##### Command Line:\n{star_cmd}\n")
        tumor_log_sha = sha256_file(tumor_log)
        normal_log_sha = sha256_file(normal_log)
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
        source_manifest_sha = sha256_file(source_manifest)
        common = {
            "policy_version": "star_provenance_freeze_v1.0",
            "tumor_sample": "TEST",
            "normal_sample": "CTRL",
            "star_version": "STAR_2.5.3a_modified",
            "star_index": str(root / "STAR-index"),
            "current_outputs": {},
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
        pair = root / "star_pairs.tsv"
        pair.write_text(
            "tumor_sample\tnormal_sample\ttumor_sj_path\tnormal_sj_path\t"
            "tumor_sj_sha256\tnormal_sj_sha256\ttumor_star_manifest\tnormal_star_manifest\t"
            "critical_parameter_compatibility\ttumor_critical_parameter_digest\tnormal_critical_parameter_digest\n"
            f"TEST\tCTRL\t{tumor_sj}\t{normal_sj}\t{tumor_sha}\t{normal_sha}\t"
            f"{tumor_manifest}\t{normal_manifest}\tcompatible\t{critical_digest}\t{critical_digest}\n"
        )
        return pair

    def _parent(self, parent_id: str, cigar: str, *, mapq: int = 30, secondary: int = 0, supplementary: int = 0) -> dict:
        return {
            "sample": "TEST",
            "parent_record_id": parent_id,
            "chromosome": "1",
            "strand": "+",
            "primary_alignment_count": 1,
            "secondary_alignment_count": secondary,
            "supplementary_alignment_count": supplementary,
            "MAPQ": mapq,
            "reference_genomic_translation_status": "pass",
            "coordinate_mapping_status": "coordinate_evaluable",
            "all_alignment_loci": f"1:100-221:+:MAPQ{mapq}:{cigar}",
        }

    def test_cigar_n_creates_junction_but_deletion_does_not(self) -> None:
        n_junctions = parse_cigar_junctions("1", "+", 100, "18M82N21M")
        d_junctions = parse_cigar_junctions("1", "+", 100, "18M82D21M")
        self.assertEqual(len(n_junctions), 1)
        self.assertEqual(n_junctions[0]["junction_start0"], 118)
        self.assertEqual(n_junctions[0]["junction_end0"], 200)
        self.assertEqual(d_junctions, [])

    def test_parent_primary_threshold_and_normal_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 2, 5, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 1, 0, 16)])

            result = evaluate_parent_junctions(
                sample="TEST",
                parent_coordinates=[
                    self._parent("P1", "18M82N21M"),
                    self._parent("P2", "39M"),
                    self._parent("P3", "18M82N21M", secondary=1),
                ],
                tumor_sj_path=tumor_sj,
                normal_sj_path=normal_sj,
            )

            summaries = {row["parent_record_id"]: row for row in result["parent_summary_rows"]}
            self.assertEqual(summaries["P1"]["primary_core_status"], PRIMARY_PASS)
            self.assertEqual(summaries["P1"]["normal_any_required_junction_unique_ge1"], True)
            self.assertEqual(summaries["P1"]["normal_all_required_junctions_unique_ge2"], False)
            self.assertEqual(summaries["P2"]["junction_qc_status"], "intronless_not_applicable")
            self.assertEqual(summaries["P3"]["primary_core_status"], PROVISIONAL_MAPPING)
            self.assertEqual(result["sensitivity_rows"][0]["threshold_tumor_unique_reads"], 1)

    def test_policy_v1_rejects_mutable_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 3, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 0, 0, 16)])
            with self.assertRaisesRegex(ValueError, "primary_min_tumor_unique_reads=2"):
                evaluate_parent_junctions(
                    sample="TEST",
                    parent_coordinates=[self._parent("P1", "18M82N21M")],
                    tumor_sj_path=tumor_sj,
                    normal_sj_path=normal_sj,
                    primary_min_tumor_unique_reads=3,
                )

    def test_low_unique_read_is_sensitivity_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 1, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 0, 0, 16)])

            result = evaluate_parent_junctions(
                sample="TEST",
                parent_coordinates=[self._parent("P1", "18M82N21M")],
                tumor_sj_path=tumor_sj,
                normal_sj_path=normal_sj,
            )
            summary = result["parent_summary_rows"][0]
            self.assertEqual(summary["primary_core_status"], PROVISIONAL_LOW_SUPPORT)
            sensitivity = {row["threshold_tumor_unique_reads"]: row for row in result["sensitivity_rows"]}
            self.assertEqual(sensitivity[1]["parents_pass"], 1)
            self.assertEqual(sensitivity[2]["parents_pass"], 0)

    def test_unknown_strand_does_not_count_as_primary_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 0, 1, 1, 5, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 0, 1, 1, 5, 0, 16)])

            result = evaluate_parent_junctions(
                sample="TEST",
                parent_coordinates=[self._parent("P1", "18M82N21M")],
                tumor_sj_path=tumor_sj,
                normal_sj_path=normal_sj,
            )
            self.assertEqual(result["parent_summary_rows"][0]["primary_core_status"], PROVISIONAL_LOW_SUPPORT)
            self.assertEqual(result["parent_junction_rows"][0]["tumor_unknown_strand_unique_reads"], 5)
            self.assertEqual(result["parent_junction_rows"][0]["tumor_unique_reads"], 0)

    def test_mapping_priority_keeps_specific_provisional_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 2, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 0, 0, 16)])
            low_mapq = self._parent("LOWMAPQ", "18M82N21M", mapq=5)
            mismatch = self._parent("MISMATCH", "18M82N21M")
            mismatch["coordinate_mapping_status"] = "not_evaluable_candidate_coordinate_qc"
            mismatch["reference_genomic_translation_status"] = "reference_translation_mismatch"
            result = evaluate_parent_junctions(
                sample="TEST",
                parent_coordinates=[low_mapq, mismatch],
                tumor_sj_path=tumor_sj,
                normal_sj_path=normal_sj,
            )
            summaries = {row["parent_record_id"]: row for row in result["parent_summary_rows"]}
            self.assertEqual(summaries["LOWMAPQ"]["primary_core_status"], PROVISIONAL_MAPPING)
            self.assertEqual(summaries["MISMATCH"]["primary_core_status"], PROVISIONAL_TRANSLATION)

    def test_duplicate_sj_key_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sj = Path(tmp) / "SJ.out.tab"
            self._write_sj(
                sj,
                [
                    ("chr1", 119, 200, 1, 1, 1, 1, 0, 20),
                    ("chr1", 119, 200, 1, 1, 1, 2, 0, 20),
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_sj_out(sj, label="tumor")

    def test_peptide_junction_annotation(self) -> None:
        peptide_rows = [
            {
                "sample": "TEST",
                "peptide_record_id": "pep1",
                "parent_record_id": "P1",
                "source_parent_id": "SRC1",
                "mhc_class": "MHC-I",
                "peptide": "ACDEFGHI",
                "peptide_start": 1,
                "peptide_length": 8,
            }
        ]
        footprint_rows = [
            {
                "sample": "TEST",
                "peptide_record_id": "pep1",
                "parent_record_id": "P1",
                "mhc_class": "MHC-I",
                "peptide": "ACDEFGHI",
                "peptide_start": 1,
                "peptide_length": 8,
                "strand": "+",
                "codon_blocks_transcript_order": (
                    '[{"chromosome":"1","strand":"+","start0":100,"end0":118},'
                    '{"chromosome":"1","strand":"+","start0":200,"end0":206}]'
                ),
            }
        ]
        parent_summary = [
            {
                "parent_record_id": "P1",
                "primary_core_status": PRIMARY_PASS,
                "junction_qc_status": "all_required_junctions_tumor_unique_ge2",
                "required_junction_count": 1,
            }
        ]
        rows = annotate_peptide_junctions(
            sample="TEST",
            peptide_parent_rows=peptide_rows,
            peptide_footprint_rows=footprint_rows,
            parent_summary_rows=parent_summary,
        )
        self.assertTrue(rows[0]["peptide_crosses_junction"])
        self.assertEqual(rows[0]["crossed_required_junction_ids"], "1:118-200:+")

    def test_same_parent_same_peptide_at_two_positions_uses_strict_key(self) -> None:
        peptide_rows = [
            {
                "sample": "TEST",
                "peptide_record_id": "pepX",
                "parent_record_id": "P1",
                "source_parent_id": "SRC1",
                "mhc_class": "MHC-I",
                "peptide": "ACDEFGHI",
                "peptide_start": start,
                "peptide_length": 8,
            }
            for start in (1, 10)
        ]
        footprint_rows = [
            {
                "sample": "TEST",
                "peptide_record_id": "pepX",
                "parent_record_id": "P1",
                "mhc_class": "MHC-I",
                "peptide": "ACDEFGHI",
                "peptide_start": 1,
                "peptide_length": 8,
                "strand": "+",
                "codon_blocks_transcript_order": (
                    '[{"chromosome":"1","strand":"+","start0":100,"end0":118},'
                    '{"chromosome":"1","strand":"+","start0":200,"end0":206}]'
                ),
            },
            {
                "sample": "TEST",
                "peptide_record_id": "pepX",
                "parent_record_id": "P1",
                "mhc_class": "MHC-I",
                "peptide": "ACDEFGHI",
                "peptide_start": 10,
                "peptide_length": 8,
                "strand": "+",
                "codon_blocks_transcript_order": '[{"chromosome":"1","strand":"+","start0":300,"end0":324}]',
            },
        ]
        parent_summary = [
            {
                "parent_record_id": "P1",
                "primary_core_status": PRIMARY_PASS,
                "junction_qc_status": "all_required_junctions_tumor_unique_ge2",
                "required_junction_count": 1,
            }
        ]
        rows = annotate_peptide_junctions(
            sample="TEST",
            peptide_parent_rows=peptide_rows,
            peptide_footprint_rows=footprint_rows,
            parent_summary_rows=parent_summary,
        )
        by_start = {int(row["peptide_start"]): row for row in rows}
        self.assertTrue(by_start[1]["peptide_crosses_junction"])
        self.assertFalse(by_start[10]["peptide_crosses_junction"])

    def test_peptide_parent_map_and_footprint_mismatch_fails_closed(self) -> None:
        peptide_rows = [{
            "peptide_record_id": "pep1", "parent_record_id": "P1", "mhc_class": "MHC-I",
            "peptide": "ACDEFGHI", "peptide_start": 1, "peptide_length": 8,
        }]
        footprint_rows = [{
            "peptide_record_id": "pep1", "parent_record_id": "P1", "mhc_class": "MHC-I",
            "peptide": "ACDEFGHI", "peptide_start": 2, "peptide_length": 8,
        }]
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            annotate_peptide_junctions(
                sample="TEST",
                peptide_parent_rows=peptide_rows,
                peptide_footprint_rows=footprint_rows,
                parent_summary_rows=[],
            )

    def test_star_pair_contract_validates_manifest_and_actual_sj_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 2, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 0, 0, 16)])
            pair = self._write_star_pair(root, tumor_sj, normal_sj)
            validation = validate_star_pair_contract(pair, "TEST")
            self.assertEqual(validation["status"], "validated")
            tumor_sj.write_text(tumor_sj.read_text() + "chr1\t300\t350\t1\t1\t1\t1\t0\t10\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_star_pair_contract(pair, "TEST")

    def test_star_pair_contract_accepts_relocated_index_identity_without_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 2, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 0, 0, 16)])
            pair = self._write_star_pair(root, tumor_sj, normal_sj)
            normal_manifest_path = root / "normal_star.json"
            normal_manifest = json.loads(normal_manifest_path.read_text())
            normal_manifest.pop("star_index", None)
            normal_manifest["star_index_identity"] = {
                "files": {
                    name: {"exists": True, "size_bytes": (root / "STAR-index" / name).stat().st_size}
                    for name in ("Genome", "SA", "SAindex", "sjdbList.out.tab")
                }
            }
            normal_manifest["star_index_identity_status"] = "from_source_manifest"
            normal_manifest_path.write_text(json.dumps(normal_manifest, sort_keys=True))
            validation = validate_star_pair_contract(pair, "TEST")
            self.assertEqual(validation["status"], "validated")
            self.assertEqual(validation["star_index_compatibility"], "compatible")

    def test_standalone_build_fails_when_manifest_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            parent_coordinates = root / "parents.tsv"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 2, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 0, 0, 16)])
            pair = self._write_star_pair(root, tumor_sj, normal_sj)
            text = pair.read_text().replace(str(root / "tumor_star.json"), str(root / "missing.json"))
            pair.write_text(text)
            parent_coordinates.write_text(
                "sample\tparent_record_id\tchromosome\tstrand\tprimary_alignment_count\tsecondary_alignment_count\t"
                "supplementary_alignment_count\tMAPQ\treference_genomic_translation_status\tcoordinate_mapping_status\t"
                "all_alignment_loci\n"
                "TEST\tP1\t1\t+\t1\t0\t0\t30\tpass\tcoordinate_evaluable\t1:100-221:+:MAPQ30:18M82N21M\n"
            )
            with self.assertRaises(FileNotFoundError):
                build_junction_qc(argparse.Namespace(
                    sample="TEST",
                    parent_coordinates=str(parent_coordinates),
                    star_pair_inputs=str(pair),
                    peptide_parent_map="",
                    peptide_genomic_footprint="",
                    outdir=str(root / "out"),
                    primary_min_tumor_unique_reads=2,
                    sensitivity_thresholds="1,2,3,5",
                    min_mapq=20,
                ))

    def test_standalone_manifest_records_input_output_hash_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tumor_sj = root / "tumor.SJ.out.tab"
            normal_sj = root / "normal.SJ.out.tab"
            parent_coordinates = root / "parents.tsv"
            self._write_sj(tumor_sj, [("chr1", 119, 200, 1, 1, 1, 2, 0, 20)])
            self._write_sj(normal_sj, [("chr1", 119, 200, 1, 1, 1, 0, 0, 16)])
            pair = self._write_star_pair(root, tumor_sj, normal_sj)
            parent_coordinates.write_text(
                "sample\tparent_record_id\tchromosome\tstrand\tprimary_alignment_count\tsecondary_alignment_count\t"
                "supplementary_alignment_count\tMAPQ\treference_genomic_translation_status\tcoordinate_mapping_status\t"
                "all_alignment_loci\n"
                "TEST\tP1\t1\t+\t1\t0\t0\t30\tpass\tcoordinate_evaluable\t1:100-221:+:MAPQ30:18M82N21M\n"
            )
            args = argparse.Namespace(
                sample="TEST",
                parent_coordinates=str(parent_coordinates),
                star_pair_inputs=str(pair),
                peptide_parent_map="",
                peptide_genomic_footprint="",
                outdir=str(root / "out"),
                primary_min_tumor_unique_reads=2,
                sensitivity_thresholds="1,2,3,5",
                min_mapq=20,
            )
            manifest = build_junction_qc(args)
            self.assertEqual(manifest["run_status"], "complete")
            self.assertIn("parent_coordinates", manifest["input_signature"]["input_paths"])
            self.assertIn("cryptic_parent_junctions.tsv", manifest["output_signature"])
            reused = build_junction_qc(args)
            self.assertTrue(reused["reused"])
            parent_coordinates.write_text(parent_coordinates.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_junction_qc(args)


if __name__ == "__main__":
    unittest.main()
