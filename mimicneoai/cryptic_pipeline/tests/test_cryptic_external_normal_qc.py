from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mimicneoai.cryptic_pipeline.scripts.cryptic_external_normal_qc import (
    build_external_normal_qc,
    load_human_reference_matches as load_external_human_reference_matches,
    sha256_file,
    validate_final_core_contract,
)
from mimicneoai.cryptic_pipeline.scripts.cryptic_core_qc import (
    load_human_reference_matches as load_core_human_reference_matches,
)
from mimicneoai.cryptic_pipeline.scripts.cryptic_coordinate_utils import (
    GenomicBlock,
    classify_coordinate_match,
    blocks_to_json,
)


def write_tsv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)


def write_tsv_gz(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        pd.DataFrame(rows, columns=columns).to_csv(handle, sep="\t", index=False)


class CrypticExternalNormalQCTest(unittest.TestCase):
    core_columns = [
        "sample",
        "peptide_record_id",
        "mhc_class",
        "peptide_length",
        "peptide",
        "peptide_sequence_sha256",
        "peptide_core_status",
        "peptide_qc_reasons",
        "human_reference_peptide_status",
        "supporting_window_count",
    ]
    parent_columns = [
        "sample",
        "peptide_record_id",
        "parent_record_id",
        "source_parent_id",
        "mhc_class",
        "peptide_start",
        "peptide_length",
        "peptide",
        "peptide_sequence_sha256",
        "peptide_core_status",
        "peptide_qc_reasons",
        "human_reference_peptide_status",
        "supporting_window_count",
    ]

    def _with_default_core_metadata(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        normalized = []
        support_counts = {}
        for row in rows:
            key = (str(row.get("mhc_class", "")), str(row.get("peptide", "")))
            support_counts[key] = support_counts.get(key, 0) + 1
        for row in rows:
            out = dict(row)
            peptide = str(out.get("peptide", ""))
            key = (str(out.get("mhc_class", "")), peptide)
            out.setdefault("peptide_sequence_sha256", hashlib.sha256(peptide.encode()).hexdigest())
            out.setdefault("peptide_core_status", "core")
            out.setdefault("peptide_qc_reasons", "")
            out.setdefault("human_reference_peptide_status", "human_reference_peptide_not_detected")
            out.setdefault("supporting_window_count", support_counts[key])
            normalized.append(out)
        return normalized

    def default_core_rows(self) -> list[dict[str, object]]:
        return [
            {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep1",
                "mhc_class": "MHC-I",
                "peptide_length": 8,
                "peptide": "ACDEFGHI",
            },
            {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep2",
                "mhc_class": "MHC-I",
                "peptide_length": 8,
                "peptide": "CDEFGHIK",
            },
            {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep3",
                "mhc_class": "MHC-II",
                "peptide_length": 13,
                "peptide": "ACDEFGHIKLMNP",
            },
            {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep4",
                "mhc_class": "MHC-I",
                "peptide_length": 8,
                "peptide": "KLMNPQRS",
            },
            {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep5",
                "mhc_class": "MHC-I",
                "peptide_length": 8,
                "peptide": "QRSTVWYA",
            },
        ]

    def default_parent_rows(self, core_rows: list[dict[str, object]]) -> list[dict[str, object]]:
        rows = [
            {
                "sample": row["sample"],
                "peptide_record_id": row["peptide_record_id"],
                "parent_record_id": f"parent_{row['peptide_record_id']}",
                "source_parent_id": f"source_{row['peptide_record_id']}",
                "mhc_class": row["mhc_class"],
                "peptide_start": 1,
                "peptide_length": row["peptide_length"],
                "peptide": row["peptide"],
            }
            for row in core_rows
        ]
        rows.append(
            {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep5",
                "parent_record_id": "parent_pep5_b",
                "source_parent_id": "source_pep5_b",
                "mhc_class": "MHC-I",
                "peptide_start": 1,
                "peptide_length": 8,
                "peptide": "QRSTVWYA",
            }
        )
        return rows

    def write_08b_manifest(
        self,
        manifest: Path,
        core: Path,
        parent_map: Path,
        *,
        run_status: str = "complete",
        policy_version: str = "cryptic_core_qc_v1.0",
        output_signature_override: dict[str, object] | None = None,
    ) -> None:
        output_signature = {
            "cryptic_peptide_core.tsv": {
                "path": str(core),
                "exists": core.exists(),
                "size": core.stat().st_size if core.exists() else 0,
                "sha256": sha256_file(core) if core.exists() else "",
            },
            "cryptic_peptide_parent_map.tsv": {
                "path": str(parent_map),
                "exists": parent_map.exists(),
                "size": parent_map.stat().st_size if parent_map.exists() else 0,
                "sha256": sha256_file(parent_map) if parent_map.exists() else "",
            },
        }
        if output_signature_override:
            output_signature.update(output_signature_override)
        manifest.write_text(
            json.dumps(
                {
                    "policy_version": policy_version,
                    "run_status": run_status,
                    "binding_eligible": run_status == "complete",
                    "candidate_selection_policy_version": "ranked_prebinding_selection_v1.0",
                    "candidate_selection": {
                        "mode": "all",
                        "ranking_policy": "",
                    },
                    "input_signature": {
                        "mhc_i_lengths": [8, 9, 10, 11],
                        "mhc_ii_lengths": [13, 14, 15, 16, 17],
                    },
                    "output_signature": output_signature,
                },
                indent=2,
            )
        )

    def _write_08b_core(
        self,
        root: Path,
        *,
        core_rows: list[dict[str, object]] | None = None,
        parent_rows: list[dict[str, object]] | None = None,
        run_status: str = "complete",
        policy_version: str = "cryptic_core_qc_v1.0",
    ) -> dict[str, Path]:
        core_rows = self.default_core_rows() if core_rows is None else core_rows
        parent_rows = self.default_parent_rows(core_rows) if parent_rows is None else parent_rows
        core_rows = self._with_default_core_metadata(core_rows)
        parent_rows = self._with_default_core_metadata(parent_rows)
        core = root / "08b" / "cryptic_peptide_core.tsv"
        parent_map = root / "08b" / "cryptic_peptide_parent_map.tsv"
        manifest = root / "08b" / "run_manifest.json"
        write_tsv(core, core_rows, self.core_columns)
        write_tsv(parent_map, parent_rows, self.parent_columns)
        self.write_08b_manifest(
            manifest,
            core,
            parent_map,
            run_status=run_status,
            policy_version=policy_version,
        )
        return {"core": core, "parent_map": parent_map, "manifest": manifest}

    def _write_resources(
        self,
        root: Path,
        *,
        freeze_version: str = "external_normal_resources_v1.0_20260727",
        hla_i_lengths: list[int] | None = None,
        hla_ii_lengths: list[int] | None = None,
    ) -> dict[str, Path]:
        resource_root = root / "resources"
        processed = resource_root / "processed"
        smorf_index = processed / "normal_smorf_peptide_match_index.tsv.gz"
        smorf_parent_map = processed / "normal_smorf_peptide_parent_map.tsv.gz"
        hla_index = processed / "normal_hla_peptide_match_index.tsv.gz"
        hla_evidence = processed / "normal_hla_ligand_evidence.tsv.gz"

        write_tsv_gz(
            smorf_index,
            [
                {"hla_class": "HLA-I", "peptide_sequence": "ACDEFGHI", "peptide_length": 8},
                {"hla_class": "HLA-II", "peptide_sequence": "CDEFGHIK", "peptide_length": 8},
                {"hla_class": "HLA-I", "peptide_sequence": "KLMNPQRS", "peptide_length": 8},
            ],
            ["hla_class", "peptide_sequence", "peptide_length"],
        )
        write_tsv_gz(
            smorf_parent_map,
            [
                {
                    "hla_class": "HLA-I",
                    "peptide_sequence": "ACDEFGHI",
                    "iorf_id": "iORF1",
                    "orf_id": "ORF1",
                    "source_record_id": "S1",
                    "peptide_start_1based": 1,
                },
                {
                    "hla_class": "HLA-II",
                    "peptide_sequence": "CDEFGHIK",
                    "iorf_id": "iORF_CLASS_MISMATCH",
                    "orf_id": "ORF_CLASS_MISMATCH",
                    "source_record_id": "S2",
                    "peptide_start_1based": 1,
                },
                {
                    "hla_class": "HLA-I",
                    "peptide_sequence": "KLMNPQRS",
                    "iorf_id": "iORF4",
                    "orf_id": "ORF4",
                    "source_record_id": "S4",
                    "peptide_start_1based": 2,
                },
            ],
            ["hla_class", "peptide_sequence", "iorf_id", "orf_id", "source_record_id", "peptide_start_1based"],
        )
        write_tsv_gz(
            hla_index,
            [
                {"hla_class": "HLA-II", "peptide_sequence": "ACDEFGHIKLMNP", "peptide_length": 13},
                {"hla_class": "HLA-I", "peptide_sequence": "KLMNPQRS", "peptide_length": 8},
            ],
            ["hla_class", "peptide_sequence", "peptide_length"],
        )
        write_tsv_gz(
            hla_evidence,
            [
                {
                    "hla_class": "HLA-II",
                    "peptide_sequence": "ACDEFGHIKLMNP",
                    "peptide_sequence_id": "HLA_SEQ_3",
                    "donor": "D1",
                    "donor_hla_genotype": "HLA-DRB1*01:01",
                    "tissue": "colon",
                    "source_resource": "HLA_Ligand_Atlas",
                },
                {
                    "hla_class": "HLA-I",
                    "peptide_sequence": "KLMNPQRS",
                    "peptide_sequence_id": "HLA_SEQ_4",
                    "donor": "D2",
                    "donor_hla_genotype": "HLA-A*02:01",
                    "tissue": "ileum",
                    "source_resource": "HLA_Ligand_Atlas",
                },
            ],
            [
                "hla_class",
                "peptide_sequence",
                "peptide_sequence_id",
                "donor",
                "donor_hla_genotype",
                "tissue",
                "source_resource",
            ],
        )

        manifest = resource_root / "resource_manifest.json"
        processed_files = {}
        for path in [smorf_index, smorf_parent_map, hla_index, hla_evidence]:
            rel = path.relative_to(resource_root).as_posix()
            processed_files[rel] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        manifest.write_text(
            json.dumps(
                {
                    "freeze_version": freeze_version,
                    "matching_policy": {
                        "smorf": "exact peptide sequence against windows generated from published smORFs",
                        "hla_ligand": "exact peptide sequence in the same HLA class",
                        "hla_i_lengths": hla_i_lengths or [8, 9, 10, 11],
                        "hla_ii_lengths": hla_ii_lengths or [13, 14, 15, 16, 17],
                    },
                    "processed_files": processed_files,
                },
                indent=2,
            )
        )
        return {
            "manifest": manifest,
            "smorf_index": smorf_index,
            "smorf_parent_map": smorf_parent_map,
            "hla_index": hla_index,
            "hla_evidence": hla_evidence,
        }

    def _write_coordinate_resources(
        self,
        root: Path,
        *,
        freeze_version: str = "smorf_coordinate_resources_v1.1_20260727",
    ) -> dict[str, Path]:
        resource_root = root / "coordinate_resources"
        processed = resource_root / "processed"
        coords = processed / "normal_smorfs_coordinates.tsv.gz"
        orfcds = processed / "normal_smorf_orfcds.tsv.gz"
        coord_rows = []
        block_rows = []
        for idx in range(1, 7768):
            iorf = "iORF_MATCH" if idx == 1 else f"iORF_FAKE_{idx}"
            orf = "ORF_MATCH" if idx == 1 else f"ORF_FAKE_{idx}"
            chrom = "1" if idx == 1 else "2"
            start = 100 if idx == 1 else 1000000 + idx * 30
            end = start + 24
            coord_rows.append({
                "iorf_id": iorf,
                "orf_id": orf,
                "source_record_id": f"SMORF_{idx}",
                "reference_build": "GRCh38",
                "coordinate_qc_status": "pass",
            })
            block_rows.append({
                "iorf_id": iorf,
                "orf_id": orf,
                "chromosome": chrom,
                "strand": "+",
                "start0": start,
                "end0": end,
                "transcript_block_order": 1,
            })
        write_tsv_gz(
            coords,
            coord_rows,
            ["iorf_id", "orf_id", "source_record_id", "reference_build", "coordinate_qc_status"],
        )
        write_tsv_gz(
            orfcds,
            block_rows,
            ["iorf_id", "orf_id", "chromosome", "strand", "start0", "end0", "transcript_block_order"],
        )
        manifest = resource_root / "resource_manifest.json"
        processed_files = {}
        for path in [coords, orfcds]:
            rel = path.relative_to(resource_root).as_posix()
            processed_files[rel] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest.write_text(
            json.dumps(
                {
                    "freeze_version": freeze_version,
                    "source": {"reference_build": "GRCh38"},
                    "coordinate_policy": {"normalized_internal_coordinates": "0_based_half_open"},
                    "qc": {"formal_coordinate_qc_passes": 7767},
                    "processed_files": processed_files,
                },
                indent=2,
            )
        )
        return {"coordinate_manifest": manifest, "normal_smorf_coordinates": coords, "normal_smorf_orfcds": orfcds}

    def _write_v11_coordinate_sidecars(
        self,
        root: Path,
        inputs: dict[str, Path],
        core_rows: list[dict[str, object]],
        parent_rows: list[dict[str, object]],
        *,
        bad_parent_status: str | None = None,
    ) -> dict[str, Path]:
        sidecar_root = inputs["core"].parent
        parent_coordinates = sidecar_root / "cryptic_parent_coordinates.tsv"
        parent_orfcds = sidecar_root / "cryptic_parent_orfcds.tsv"
        footprint = sidecar_root / "cryptic_peptide_genomic_footprint.tsv"
        parent_ids = sorted({str(row["parent_record_id"]) for row in parent_rows})
        block_length_by_parent: dict[str, int] = {}
        for row in parent_rows:
            parent_id = str(row["parent_record_id"])
            block_length_by_parent[parent_id] = max(
                block_length_by_parent.get(parent_id, 0),
                int(row["peptide_length"]) * 3,
            )
        parent_coordinate_rows = []
        parent_orfcds_rows = []
        for idx, parent_id in enumerate(parent_ids, start=1):
            status = bad_parent_status or "coordinate_evaluable"
            block_length = block_length_by_parent[parent_id]
            blocks = [GenomicBlock("1", "+", 100, 100 + block_length)]
            parent_coordinate_rows.append({
                "sample": "CRYPTIC-T",
                "parent_record_id": parent_id,
                "block_count": 1,
                "blocks_transcript_order": blocks_to_json(blocks),
                "block_total_length": block_length,
                "coordinate_mapping_status": status,
            })
            parent_orfcds_rows.append({
                "sample": "CRYPTIC-T",
                "parent_record_id": parent_id,
                "transcript_block_order": 1,
                "chromosome": "1",
                "strand": "+",
                "start0": 100,
                "end0": 100 + block_length,
                "block_length": block_length,
                "derived_phase": 0,
                "phase_provenance": "synthetic",
                "coordinate_mapping_status": status,
            })
        footprint_rows = []
        for row in parent_rows:
            peptide_id = str(row["peptide_record_id"])
            if not peptide_id:
                continue
            blocks = [GenomicBlock("1", "+", 100, 100 + int(row["peptide_length"]) * 3)]
            footprint_rows.append({
                "sample": row["sample"],
                "peptide_record_id": peptide_id,
                "parent_record_id": row["parent_record_id"],
                "source_parent_id": row["source_parent_id"],
                "mhc_class": row["mhc_class"],
                "peptide": row["peptide"],
                "peptide_start": 1,
                "peptide_length": row["peptide_length"],
                "codon_blocks_transcript_order": blocks_to_json(blocks),
                "peptide_cds_start0": 0,
                "peptide_cds_end0": int(row["peptide_length"]) * 3,
                "candidate_coordinate_status": status,
                "reference_build": "GRCh38",
            })
        write_tsv(
            parent_coordinates,
            parent_coordinate_rows,
            [
                "sample", "parent_record_id", "block_count", "blocks_transcript_order",
                "block_total_length", "coordinate_mapping_status",
            ],
        )
        write_tsv(
            parent_orfcds,
            parent_orfcds_rows,
            [
                "sample", "parent_record_id", "transcript_block_order", "chromosome", "strand",
                "start0", "end0", "block_length", "derived_phase", "phase_provenance",
                "coordinate_mapping_status",
            ],
        )
        write_tsv(
            footprint,
            footprint_rows,
            [
                "sample", "peptide_record_id", "parent_record_id", "source_parent_id",
                "mhc_class", "peptide", "peptide_start", "peptide_length",
                "codon_blocks_transcript_order", "peptide_cds_start0", "peptide_cds_end0",
                "candidate_coordinate_status", "reference_build",
            ],
        )
        self.write_08b_manifest(
            inputs["manifest"],
            inputs["core"],
            inputs["parent_map"],
            policy_version="cryptic_core_qc_v1.1",
            output_signature_override={
                "cryptic_parent_coordinates.tsv": {
                    "path": str(parent_coordinates),
                    "exists": True,
                    "size": parent_coordinates.stat().st_size,
                    "sha256": sha256_file(parent_coordinates),
                },
                "cryptic_parent_orfcds.tsv": {
                    "path": str(parent_orfcds),
                    "exists": True,
                    "size": parent_orfcds.stat().st_size,
                    "sha256": sha256_file(parent_orfcds),
                },
                "cryptic_peptide_genomic_footprint.tsv": {
                    "path": str(footprint),
                    "exists": True,
                    "size": footprint.stat().st_size,
                    "sha256": sha256_file(footprint),
                },
            },
        )
        return {
            "cryptic_parent_coordinates": parent_coordinates,
            "cryptic_parent_orfcds": parent_orfcds,
            "cryptic_peptide_genomic_footprint": footprint,
        }

    def _refresh_v11_manifest_signature(self, inputs: dict[str, Path], sidecars: dict[str, Path]) -> None:
        self.write_08b_manifest(
            inputs["manifest"],
            inputs["core"],
            inputs["parent_map"],
            policy_version="cryptic_core_qc_v1.1",
            output_signature_override={
                "cryptic_parent_coordinates.tsv": {
                    "path": str(sidecars["cryptic_parent_coordinates"]),
                    "exists": True,
                    "size": sidecars["cryptic_parent_coordinates"].stat().st_size,
                    "sha256": sha256_file(sidecars["cryptic_parent_coordinates"]),
                },
                "cryptic_parent_orfcds.tsv": {
                    "path": str(sidecars["cryptic_parent_orfcds"]),
                    "exists": True,
                    "size": sidecars["cryptic_parent_orfcds"].stat().st_size,
                    "sha256": sha256_file(sidecars["cryptic_parent_orfcds"]),
                },
                "cryptic_peptide_genomic_footprint.tsv": {
                    "path": str(sidecars["cryptic_peptide_genomic_footprint"]),
                    "exists": True,
                    "size": sidecars["cryptic_peptide_genomic_footprint"].stat().st_size,
                    "sha256": sha256_file(sidecars["cryptic_peptide_genomic_footprint"]),
                },
            },
        )

    def _args(self, inputs: dict[str, Path], resources: dict[str, Path], outdir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            sample="CRYPTIC-T",
            policy_version="cryptic_external_normal_qc_v1.0",
            cryptic_peptide_core=str(inputs["core"]),
            cryptic_peptide_parent_map=str(inputs["parent_map"]),
            cryptic_peptide_deferred=str(inputs.get("deferred", "")),
            cryptic_peptide_deferred_parent_map=str(inputs.get("deferred_parent_map", "")),
            cryptic_parent_core=str(inputs.get("parent_core", "")),
            cryptic_parent_ranked=str(inputs.get("parent_ranked", "")),
            cryptic_parent_junction_summary=str(inputs.get("parent_junction_summary", "")),
            cryptic_peptide_junction_evidence=str(inputs.get("peptide_junction_evidence", "")),
            upstream_manifest=str(inputs["manifest"]),
            human_proteome_fasta=str(inputs.get("human_proteome", "")),
            cryptic_parent_coordinates=str(inputs.get("cryptic_parent_coordinates", "")),
            cryptic_parent_orfcds=str(inputs.get("cryptic_parent_orfcds", "")),
            cryptic_peptide_genomic_footprint=str(inputs.get("cryptic_peptide_genomic_footprint", "")),
            outdir=str(outdir),
            resource_manifest=str(resources.get("manifest", "")),
            smorf_match_index=str(resources.get("smorf_index", "")),
            smorf_parent_map=str(resources.get("smorf_parent_map", "")),
            hla_ligand_match_index=str(resources.get("hla_index", "")),
            hla_ligand_evidence=str(resources.get("hla_evidence", "")),
            coordinate_resource_manifest=str(resources.get("coordinate_manifest", "")),
            normal_smorf_coordinates=str(resources.get("normal_smorf_coordinates", "")),
            normal_smorf_orfcds=str(resources.get("normal_smorf_orfcds", "")),
            allow_missing_external_normal_resources=False,
            coordinate_matching_enabled=False,
            max_hla_i_peptides=0,
            max_hla_ii_peptides=0,
        )

    def test_exact_sequence_and_class_matching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root)
            outdir = root / "08c"

            manifest = build_external_normal_qc(self._args(inputs, resources, outdir))

            self.assertEqual(manifest["external_normal_evaluation"], "evaluated")
            self.assertEqual(manifest["stage_counts"]["source_core_unique_peptides"], 5)
            self.assertEqual(manifest["stage_counts"]["annotated_core_unique_peptides"], 5)
            self.assertEqual(manifest["stage_counts"]["normal_smorf_exact_match_peptides"], 2)
            self.assertEqual(manifest["stage_counts"]["normal_hla_ligand_exact_match_peptides"], 2)
            self.assertEqual(manifest["stage_counts"]["both_external_normal_evidence_peptides"], 1)
            self.assertEqual(manifest["stage_counts"]["external_normal_excluded_unique_peptides"], 3)
            self.assertEqual(manifest["stage_counts"]["external_normal_excluded_unique_peptides_legacy_combined"], 3)
            self.assertEqual(manifest["stage_counts"]["external_normal_exact_excluded_unique_peptides"], 3)
            self.assertEqual(manifest["stage_counts"]["external_normal_exact_and_coordinate_frame_excluded_unique_peptides"], 0)
            self.assertEqual(manifest["stage_counts"]["total_qc_excluded_unique_peptides"], 3)
            self.assertEqual(manifest["stage_counts"]["tumor_restricted_primary_core_unique_peptides"], 2)

            annotated = pd.read_csv(outdir / "cryptic_external_normal_annotated_core.tsv", sep="\t")
            retained = pd.read_csv(outdir / "cryptic_tumor_restricted_primary_core.tsv", sep="\t")
            excluded = pd.read_csv(outdir / "cryptic_external_normal_excluded.tsv", sep="\t")
            self.assertEqual(len(annotated), 5)
            self.assertEqual(len(retained) + len(excluded), len(annotated))
            self.assertEqual(set(retained["peptide_record_id"]), {"pep2", "pep5"})
            self.assertEqual(
                retained.set_index("peptide_record_id").loc["pep2", "normal_translation_status"],
                "normal_translation_not_detected_in_evaluated_smorf_atlas",
            )
            self.assertEqual(set(excluded["peptide_record_id"]), {"pep1", "pep3", "pep4"})
            both = excluded.set_index("peptide_record_id").loc["pep4", "external_normal_qc_reasons"]
            self.assertIn("normal_smorf_exact_peptide_match", both)
            self.assertIn("normal_hla_ligand_exact_peptide_match", both)

            fasta = (outdir / "cryptic_tumor_restricted_primary_core.fasta").read_text()
            self.assertIn(">pep2\nCDEFGHIK\n", fasta)
            self.assertIn(">pep5\nQRSTVWYA\n", fasta)
            self.assertNotIn("ACDEFGHI", fasta)

            evidence = pd.read_csv(outdir / "cryptic_external_normal_evidence.tsv", sep="\t")
            self.assertIn("normal_smorf_translation", set(evidence["evidence_type"]))
            self.assertIn("normal_hla_ligand", set(evidence["evidence_type"]))
            self.assertIn("source_core_parent_map", set(evidence["evidence_type"]))
            hla_evidence = evidence[evidence["evidence_type"].eq("normal_hla_ligand")]
            self.assertIn("HLA-A*02:01", set(hla_evidence["hla_ligand_donor_hla_genotype"]))
            self.assertTrue(
                evidence["normal_smorf_coordinate_status"]
                .eq("not_evaluable_coordinates_not_materialized")
                .all()
            )
            sidecar = pd.read_csv(outdir / "cryptic_final_peptide_parent_sidecar.tsv", sep="\t")
            self.assertEqual(set(sidecar["peptide_record_id"]), {"pep2", "pep5"})
            self.assertEqual(len(sidecar[sidecar["peptide_record_id"].eq("pep5")]), 2)
            self.assertFalse(sidecar[["parent_record_id", "external_normal_status"]].isna().any().any())
            self.assertTrue(sidecar["human_reference_peptide_status"].eq("human_reference_peptide_not_detected").all())
            self.assertEqual(manifest["binding_eligible"], True)
            self.assertEqual(manifest["binding_input_mode"], "peptide-core")
            self.assertEqual(manifest["final_binding_fasta_identity"]["sha256"], sha256_file(outdir / "cryptic_tumor_restricted_primary_core.fasta"))

            reused = build_external_normal_qc(self._args(inputs, resources, outdir))
            self.assertTrue(reused["reused"])

    def test_external_normal_refills_from_deferred_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = [
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep1",
                    "mhc_class": "MHC-I",
                    "peptide_length": 8,
                    "peptide": "ACDEFGHI",
                },
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep2",
                    "mhc_class": "MHC-I",
                    "peptide_length": 8,
                    "peptide": "CDEFGHIK",
                },
            ]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=self.default_parent_rows(core_rows)[: len(core_rows)],
            )
            deferred = root / "08b" / "cryptic_peptide_deferred.tsv"
            write_tsv(
                deferred,
                [
                    {
                        "sample": "CRYPTIC-T",
                        "peptide_record_id": "",
                        "parent_record_id": "parent_deferred_1",
                        "source_parent_id": "source_deferred_1",
                        "mhc_class": "MHC-I",
                        "peptide_start": 1,
                        "peptide_length": 8,
                        "peptide": "QRSTVWYA",
                    },
                    {
                        "sample": "CRYPTIC-T",
                        "peptide_record_id": "",
                        "parent_record_id": "parent_deferred_2",
                        "source_parent_id": "source_deferred_2",
                        "mhc_class": "MHC-I",
                        "peptide_start": 1,
                        "peptide_length": 8,
                        "peptide": "KLMNPQRS",
                    },
                ],
                self.parent_columns,
            )
            inputs["deferred"] = deferred
            resources = self._write_resources(root)
            outdir = root / "08c_refill"
            args = self._args(inputs, resources, outdir)
            args.max_hla_i_peptides = 2
            args.max_hla_ii_peptides = 1

            manifest = build_external_normal_qc(args)

            self.assertEqual(manifest["stage_counts"]["source_core_unique_peptides"], 2)
            self.assertEqual(manifest["stage_counts"]["source_deferred_unique_peptides"], 2)
            self.assertEqual(manifest["stage_counts"]["external_normal_refill_selected_unique_peptides"], 1)
            retained = pd.read_csv(outdir / "cryptic_tumor_restricted_primary_core.tsv", sep="\t")
            self.assertEqual(set(retained["peptide"]), {"CDEFGHIK", "QRSTVWYA"})
            self.assertIn("deferred_refill", set(retained["prebinding_selection_origin"]))
            refill_deferred = pd.read_csv(outdir / "cryptic_external_normal_refill_deferred.tsv", sep="\t")
            self.assertTrue(refill_deferred.empty)

    def test_v11_ranked_junction_refill_materializes_full_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = [
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep1",
                    "mhc_class": "MHC-I",
                    "peptide_length": 8,
                    "peptide": "ACDEFGHI",
                }
            ]
            parent_rows = [
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep1",
                    "parent_record_id": "parent_initial",
                    "source_parent_id": "source_initial",
                    "mhc_class": "MHC-I",
                    "peptide_start": 1,
                    "peptide_length": 8,
                    "peptide": "ACDEFGHI",
                },
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "",
                    "parent_record_id": "parent_refill",
                    "source_parent_id": "source_refill",
                    "mhc_class": "MHC-I",
                    "peptide_start": 1,
                    "peptide_length": 8,
                    "peptide": "QRSTVWYA",
                },
            ]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=[parent_rows[0]],
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            coords = pd.read_csv(sidecars["cryptic_parent_coordinates"], sep="\t", dtype=str)
            coords["chromosome"] = "1"
            coords["strand"] = "+"
            coords["reference_genomic_translation_status"] = "pass"
            coords.loc[
                coords["parent_record_id"].eq("parent_refill"),
                ["chromosome", "genomic_start0", "genomic_end0", "block_total_length"],
            ] = ["2", "500", "536", "36"]
            coords.loc[coords["parent_record_id"].eq("parent_refill"), "blocks_transcript_order"] = blocks_to_json(
                [GenomicBlock("2", "+", 500, 536)]
            )
            coords.to_csv(sidecars["cryptic_parent_coordinates"], sep="\t", index=False)
            orfcds = pd.read_csv(sidecars["cryptic_parent_orfcds"], sep="\t", dtype=str)
            orfcds.loc[
                orfcds["parent_record_id"].eq("parent_refill"),
                ["chromosome", "start0", "end0", "block_length"],
            ] = ["2", "500", "536", "36"]
            orfcds.to_csv(sidecars["cryptic_parent_orfcds"], sep="\t", index=False)

            parent_core = root / "08b" / "cryptic_parent_core.tsv"
            parent_ranked = root / "08b" / "cryptic_parent_ranked.tsv"
            deferred_parent_map = root / "08b" / "cryptic_peptide_deferred_parent_map.tsv"
            parent_junction = root / "08b" / "cryptic_parent_junction_summary.tsv"
            peptide_junction = root / "08b" / "cryptic_peptide_junction_evidence.tsv"
            write_tsv(
                parent_core,
                [
                    {
                        "sample": "CRYPTIC-T",
                        "parent_record_id": "parent_initial",
                        "source_parent_id": "source_initial",
                        "nc_class": "novel",
                        "TPM_tumor": 20,
                        "TPM_ctrl": 0,
                        "log2FC": 8,
                        "expression_qc_status": "pass",
                        "parent_sequence": "ACDEFGHIKLMN",
                        "parent_sequence_sha256": "sha_initial",
                        "cryptic_discovery_fdr_status": "not_estimable_single_pair_rule_based_qc",
                    },
                    {
                        "sample": "CRYPTIC-T",
                        "parent_record_id": "parent_refill",
                        "source_parent_id": "source_refill",
                        "nc_class": "novel",
                        "TPM_tumor": 10,
                        "TPM_ctrl": 0,
                        "log2FC": 7,
                        "expression_qc_status": "pass",
                        "parent_sequence": "QRSTVWYACDEF",
                        "parent_sequence_sha256": "sha_refill",
                        "cryptic_discovery_fdr_status": "not_estimable_single_pair_rule_based_qc",
                    },
                ],
                [
                    "sample", "parent_record_id", "source_parent_id", "nc_class", "TPM_tumor", "TPM_ctrl",
                    "log2FC", "expression_qc_status", "parent_sequence", "parent_sequence_sha256",
                    "cryptic_discovery_fdr_status",
                ],
            )
            write_tsv(
                parent_ranked,
                [
                    {"sample": "CRYPTIC-T", "parent_record_id": "parent_initial", "source_parent_id": "source_initial", "parent_rank": 1},
                    {"sample": "CRYPTIC-T", "parent_record_id": "parent_refill", "source_parent_id": "source_refill", "parent_rank": 2},
                ],
                ["sample", "parent_record_id", "source_parent_id", "parent_rank"],
            )
            write_tsv(deferred_parent_map, [], self.parent_columns)
            write_tsv(
                parent_junction,
                [
                    {
                        "sample": "CRYPTIC-T",
                        "parent_record_id": "parent_initial",
                        "primary_core_status": "primary_core_eligible",
                        "junction_qc_status": "intronless_not_applicable",
                        "junction_qc_reasons": "",
                        "rna_variant_rescue_status": "not_evaluated",
                        "required_junction_count": 0,
                        "tumor_min_required_unique_reads": "",
                        "normal_any_required_junction_unique_ge1": False,
                        "normal_all_required_junctions_unique_ge1": False,
                        "normal_any_required_junction_unique_ge2": False,
                        "normal_all_required_junctions_unique_ge2": False,
                    },
                    {
                        "sample": "CRYPTIC-T",
                        "parent_record_id": "parent_refill",
                        "primary_core_status": "primary_core_eligible",
                        "junction_qc_status": "intronless_not_applicable",
                        "junction_qc_reasons": "",
                        "rna_variant_rescue_status": "not_evaluated",
                        "required_junction_count": 0,
                        "tumor_min_required_unique_reads": "",
                        "normal_any_required_junction_unique_ge1": False,
                        "normal_all_required_junctions_unique_ge1": False,
                        "normal_any_required_junction_unique_ge2": False,
                        "normal_all_required_junctions_unique_ge2": False,
                    },
                ],
                [
                    "sample", "parent_record_id", "primary_core_status", "junction_qc_status",
                    "junction_qc_reasons", "rna_variant_rescue_status", "required_junction_count",
                    "tumor_min_required_unique_reads", "normal_any_required_junction_unique_ge1",
                    "normal_all_required_junctions_unique_ge1", "normal_any_required_junction_unique_ge2",
                    "normal_all_required_junctions_unique_ge2",
                ],
            )
            write_tsv(
                peptide_junction,
                [
                    {
                        "sample": "CRYPTIC-T",
                        "peptide_record_id": "pep1",
                        "parent_record_id": "parent_initial",
                        "source_parent_id": "source_initial",
                        "mhc_class": "MHC-I",
                        "peptide": "ACDEFGHI",
                        "peptide_start": 1,
                        "peptide_length": 8,
                        "peptide_crosses_junction": False,
                        "crossed_required_junction_ids": "",
                        "parent_primary_core_status": "primary_core_eligible",
                        "parent_junction_qc_status": "intronless_not_applicable",
                        "parent_required_junction_count": 0,
                    }
                ],
                [
                    "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
                    "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
                    "crossed_required_junction_ids", "parent_primary_core_status",
                    "parent_junction_qc_status", "parent_required_junction_count",
                ],
            )
            human = root / "human.fa"
            human.write_text(">HUMAN\nACDEFGHIKLMN\n")
            inputs.update({
                **sidecars,
                "parent_core": parent_core,
                "parent_ranked": parent_ranked,
                "deferred_parent_map": deferred_parent_map,
                "parent_junction_summary": parent_junction,
                "peptide_junction_evidence": peptide_junction,
                "human_proteome": human,
            })
            output_signature = {
                path.name: {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in [
                    inputs["core"],
                    inputs["parent_map"],
                    sidecars["cryptic_parent_coordinates"],
                    sidecars["cryptic_parent_orfcds"],
                    sidecars["cryptic_peptide_genomic_footprint"],
                    parent_core,
                    parent_ranked,
                    deferred_parent_map,
                    parent_junction,
                    peptide_junction,
                ]
            }
            inputs["manifest"].write_text(
                json.dumps(
                    {
                        "policy_version": "cryptic_core_qc_v1.1",
                        "run_status": "complete",
                        "binding_eligible": True,
                        "candidate_selection_policy_version": "ranked_prebinding_selection_v1.0",
                        "candidate_selection": {
                            "mode": "ranked_cap",
                            "max_hla_i_peptides": 1,
                            "max_hla_ii_peptides": 1,
                        },
                        "human_reference_summary": {"status": "evaluated", "path": str(human)},
                        "input_signature": {
                            "mhc_i_lengths": [8],
                            "mhc_ii_lengths": [13],
                        },
                        "output_signature": output_signature,
                    },
                    indent=2,
                )
            )
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            outdir = root / "08c_v11_refill"
            args = self._args(inputs, resources, outdir)
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            args.max_hla_i_peptides = 1

            manifest = build_external_normal_qc(args)

            retained = pd.read_csv(outdir / "cryptic_tumor_restricted_primary_core.tsv", sep="\t")
            retained = retained.fillna("")
            self.assertEqual(set(retained["peptide"]), {"QRSTVWYA"})
            self.assertEqual(manifest["stage_counts"]["external_normal_refill_selected_unique_peptides"], 1)
            retained_row = retained.iloc[0].to_dict()
            self.assertEqual(retained_row["peptide_core_status"], "core")
            self.assertEqual(str(retained_row.get("peptide_qc_reasons", "")), "")
            self.assertEqual(retained_row["peptide_sequence_sha256"], hashlib.sha256(b"QRSTVWYA").hexdigest())
            self.assertEqual(retained_row["sample"], "CRYPTIC-T")
            self.assertEqual(retained_row["parent_record_id"], "parent_refill")
            self.assertEqual(retained_row["source_parent_id"], "source_refill")
            self.assertEqual(int(retained_row["peptide_start"]), 1)
            self.assertEqual(int(retained_row["peptide_length"]), 8)
            self.assertEqual(retained_row["prebinding_selection_origin"], "deferred_refill")
            self.assertEqual(retained_row["human_reference_peptide_status"], "human_reference_peptide_not_detected")
            self.assertEqual(int(retained_row["supporting_window_count"]), 1)
            self.assertTrue(manifest["final_cap_reached_hla_i"])
            self.assertFalse(manifest["final_exhausted_before_cap_hla_i"])
            refill_footprint = pd.read_csv(outdir / "cryptic_external_normal_refill_genomic_footprint.tsv", sep="\t")
            refill_junction = pd.read_csv(outdir / "cryptic_external_normal_refill_junction_evidence.tsv", sep="\t")
            self.assertFalse(refill_footprint.empty)
            self.assertFalse(refill_junction.empty)
            self.assertTrue(refill_footprint["candidate_coordinate_status"].eq("coordinate_evaluable").all())
            sidecar = pd.read_csv(outdir / "cryptic_final_peptide_parent_sidecar.tsv", sep="\t")
            refill_sidecar = sidecar[sidecar["selection_origin"].eq("deferred_refill")]
            self.assertEqual(set(refill_sidecar["parent_record_id"]), {"parent_refill"})
            self.assertEqual(set(refill_sidecar["human_reference_peptide_status"]), {"human_reference_peptide_not_detected"})
            self.assertEqual(set(refill_sidecar["peptide_start"].astype(int)), {1})
            self.assertFalse(
                refill_sidecar[
                    ["coordinate_mapping_status", "junction_qc_status", "peptide_crosses_junction"]
                ].isna().any().any()
            )
            self.assertTrue(set(refill_footprint["peptide_length"].astype(int)).issubset({8, 13}))

    def test_v11_ranked_initial_coordinate_match_from_second_parent_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            peptide = "QRSTVWYA"
            core_rows = [{
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep_shared",
                "mhc_class": "MHC-I",
                "peptide_length": 8,
                "peptide": peptide,
            }]
            initial_parent = {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "pep_shared",
                "parent_record_id": "parent_a",
                "source_parent_id": "source_a",
                "mhc_class": "MHC-I",
                "peptide_start": 1,
                "peptide_length": 8,
                "peptide": peptide,
            }
            second_parent = dict(initial_parent)
            second_parent.update({
                "peptide_record_id": "",
                "parent_record_id": "parent_b",
                "source_parent_id": "source_b",
            })
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=[initial_parent],
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, [initial_parent, second_parent])
            coords = pd.read_csv(sidecars["cryptic_parent_coordinates"], sep="\t", dtype=str)
            coords["chromosome"] = "1"
            coords["strand"] = "+"
            coords["reference_genomic_translation_status"] = "pass"
            coords.loc[coords["parent_record_id"].eq("parent_a"), ["chromosome", "genomic_start0", "genomic_end0", "block_total_length"]] = [
                "2", "500", "536", "36",
            ]
            coords.loc[coords["parent_record_id"].eq("parent_a"), "blocks_transcript_order"] = blocks_to_json(
                [GenomicBlock("2", "+", 500, 536)]
            )
            coords.loc[coords["parent_record_id"].eq("parent_b"), ["chromosome", "genomic_start0", "genomic_end0", "block_total_length"]] = [
                "1", "100", "136", "36",
            ]
            coords.loc[coords["parent_record_id"].eq("parent_b"), "blocks_transcript_order"] = blocks_to_json(
                [GenomicBlock("1", "+", 100, 136)]
            )
            coords.to_csv(sidecars["cryptic_parent_coordinates"], sep="\t", index=False)
            orfcds = pd.read_csv(sidecars["cryptic_parent_orfcds"], sep="\t", dtype=str)
            orfcds.loc[orfcds["parent_record_id"].eq("parent_a"), ["chromosome", "start0", "end0", "block_length"]] = [
                "2", "500", "536", "36",
            ]
            orfcds.loc[orfcds["parent_record_id"].eq("parent_b"), ["chromosome", "start0", "end0", "block_length"]] = [
                "1", "100", "136", "36",
            ]
            orfcds.to_csv(sidecars["cryptic_parent_orfcds"], sep="\t", index=False)

            parent_core = root / "08b" / "cryptic_parent_core.tsv"
            parent_ranked = root / "08b" / "cryptic_parent_ranked.tsv"
            deferred_parent_map = root / "08b" / "cryptic_peptide_deferred_parent_map.tsv"
            parent_junction = root / "08b" / "cryptic_parent_junction_summary.tsv"
            peptide_junction = root / "08b" / "cryptic_peptide_junction_evidence.tsv"
            parent_core_rows = []
            for parent_id, source_id, rank in [("parent_a", "source_a", 1), ("parent_b", "source_b", 2)]:
                parent_core_rows.append({
                    "sample": "CRYPTIC-T",
                    "parent_record_id": parent_id,
                    "source_parent_id": source_id,
                    "nc_class": "novel",
                    "TPM_tumor": 20,
                    "TPM_ctrl": 0,
                    "log2FC": 8,
                    "expression_qc_status": "pass",
                    "parent_sequence": peptide + "CDEF",
                    "parent_sequence_sha256": f"sha_{parent_id}",
                    "cryptic_discovery_fdr_status": "not_estimable_single_pair_rule_based_qc",
                })
            write_tsv(parent_core, parent_core_rows, list(parent_core_rows[0]))
            write_tsv(
                parent_ranked,
                [
                    {"sample": "CRYPTIC-T", "parent_record_id": "parent_a", "source_parent_id": "source_a", "parent_rank": 1},
                    {"sample": "CRYPTIC-T", "parent_record_id": "parent_b", "source_parent_id": "source_b", "parent_rank": 2},
                ],
                ["sample", "parent_record_id", "source_parent_id", "parent_rank"],
            )
            write_tsv(deferred_parent_map, [], self.parent_columns)
            junction_rows = []
            for parent_id in ["parent_a", "parent_b"]:
                junction_rows.append({
                    "sample": "CRYPTIC-T",
                    "parent_record_id": parent_id,
                    "primary_core_status": "primary_core_eligible",
                    "junction_qc_status": "intronless_not_applicable",
                    "junction_qc_reasons": "",
                    "rna_variant_rescue_status": "reference_concordant",
                    "required_junction_count": 0,
                    "tumor_min_required_unique_reads": "",
                    "normal_any_required_junction_unique_ge1": False,
                    "normal_all_required_junctions_unique_ge1": False,
                    "normal_any_required_junction_unique_ge2": False,
                    "normal_all_required_junctions_unique_ge2": False,
                })
            write_tsv(
                parent_junction,
                junction_rows,
                [
                    "sample", "parent_record_id", "primary_core_status", "junction_qc_status",
                    "junction_qc_reasons", "rna_variant_rescue_status", "required_junction_count",
                    "tumor_min_required_unique_reads", "normal_any_required_junction_unique_ge1",
                    "normal_all_required_junctions_unique_ge1", "normal_any_required_junction_unique_ge2",
                    "normal_all_required_junctions_unique_ge2",
                ],
            )
            write_tsv(
                peptide_junction,
                [{
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep_shared",
                    "parent_record_id": "parent_a",
                    "source_parent_id": "source_a",
                    "mhc_class": "MHC-I",
                    "peptide": peptide,
                    "peptide_start": 1,
                    "peptide_length": 8,
                    "peptide_crosses_junction": False,
                    "crossed_required_junction_ids": "",
                    "parent_primary_core_status": "primary_core_eligible",
                    "parent_junction_qc_status": "intronless_not_applicable",
                    "parent_required_junction_count": 0,
                }],
                [
                    "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
                    "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
                    "crossed_required_junction_ids", "parent_primary_core_status",
                    "parent_junction_qc_status", "parent_required_junction_count",
                ],
            )
            human = root / "human.fa"
            human.write_text(">HUMAN\nACDEFGHIKLMN\n")
            inputs.update({
                **sidecars,
                "parent_core": parent_core,
                "parent_ranked": parent_ranked,
                "deferred_parent_map": deferred_parent_map,
                "parent_junction_summary": parent_junction,
                "peptide_junction_evidence": peptide_junction,
                "human_proteome": human,
            })
            output_signature = {
                path.name: {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in [
                    inputs["core"], inputs["parent_map"], sidecars["cryptic_parent_coordinates"],
                    sidecars["cryptic_parent_orfcds"], sidecars["cryptic_peptide_genomic_footprint"],
                    parent_core, parent_ranked, deferred_parent_map, parent_junction, peptide_junction,
                ]
            }
            inputs["manifest"].write_text(
                json.dumps(
                    {
                        "policy_version": "cryptic_core_qc_v1.1",
                        "run_status": "complete",
                        "binding_eligible": True,
                        "candidate_selection_policy_version": "ranked_prebinding_selection_v1.0",
                        "candidate_selection": {"mode": "ranked_cap", "max_hla_i_peptides": 1, "max_hla_ii_peptides": 1},
                        "human_reference_summary": {"status": "evaluated", "path": str(human)},
                        "input_signature": {"mhc_i_lengths": [8], "mhc_ii_lengths": [13]},
                        "output_signature": output_signature,
                    },
                    indent=2,
                )
            )
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            args = self._args(inputs, resources, root / "08c_second_parent_initial")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            args.max_hla_i_peptides = 1
            args.max_hla_ii_peptides = 1

            build_external_normal_qc(args)

            completed = pd.read_csv(root / "08c_second_parent_initial" / "cryptic_external_normal_completed_parent_map.tsv", sep="\t")
            excluded = pd.read_csv(root / "08c_second_parent_initial" / "cryptic_external_normal_excluded.tsv", sep="\t")
            self.assertEqual(set(completed["parent_record_id"]), {"parent_a", "parent_b"})
            self.assertIn("external_normal_coordinate_frame_excluded", set(excluded["external_normal_status"]))
            self.assertIn(peptide, set(excluded["peptide"]))

    def test_v11_ranked_refill_coordinate_match_from_second_parent_excludes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            peptide = "QRSTVWYA"
            inputs = self._write_08b_core(root, core_rows=[], parent_rows=[], policy_version="cryptic_core_qc_v1.1")
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, [], [])
            parent_core = root / "08b" / "cryptic_parent_core.tsv"
            parent_ranked = root / "08b" / "cryptic_parent_ranked.tsv"
            deferred_parent_map = root / "08b" / "cryptic_peptide_deferred_parent_map.tsv"
            parent_junction = root / "08b" / "cryptic_parent_junction_summary.tsv"
            peptide_junction = root / "08b" / "cryptic_peptide_junction_evidence.tsv"
            parent_core_rows = []
            for parent_id, source_id in [("parent_refill_a", "source_refill_a"), ("parent_refill_b", "source_refill_b")]:
                parent_core_rows.append({
                    "sample": "CRYPTIC-T",
                    "parent_record_id": parent_id,
                    "source_parent_id": source_id,
                    "nc_class": "novel",
                    "TPM_tumor": 20,
                    "TPM_ctrl": 0,
                    "log2FC": 8,
                    "expression_qc_status": "pass",
                    "parent_sequence": peptide + "CDEF",
                    "parent_sequence_sha256": f"sha_{parent_id}",
                    "cryptic_discovery_fdr_status": "not_estimable_single_pair_rule_based_qc",
                })
            write_tsv(parent_core, parent_core_rows, list(parent_core_rows[0]))
            write_tsv(
                parent_ranked,
                [
                    {"sample": "CRYPTIC-T", "parent_record_id": "parent_refill_a", "source_parent_id": "source_refill_a", "parent_rank": 1},
                    {"sample": "CRYPTIC-T", "parent_record_id": "parent_refill_b", "source_parent_id": "source_refill_b", "parent_rank": 2},
                ],
                ["sample", "parent_record_id", "source_parent_id", "parent_rank"],
            )
            write_tsv(deferred_parent_map, [], self.parent_columns)
            coord_rows = [
                {
                    "sample": "CRYPTIC-T",
                    "parent_record_id": "parent_refill_a",
                    "block_count": 1,
                    "blocks_transcript_order": blocks_to_json([GenomicBlock("2", "+", 500, 536)]),
                    "block_total_length": 36,
                    "coordinate_mapping_status": "coordinate_evaluable",
                    "reference_genomic_translation_status": "pass",
                    "chromosome": "2",
                    "strand": "+",
                },
                {
                    "sample": "CRYPTIC-T",
                    "parent_record_id": "parent_refill_b",
                    "block_count": 1,
                    "blocks_transcript_order": blocks_to_json([GenomicBlock("1", "+", 100, 136)]),
                    "block_total_length": 36,
                    "coordinate_mapping_status": "coordinate_evaluable",
                    "reference_genomic_translation_status": "pass",
                    "chromosome": "1",
                    "strand": "+",
                },
            ]
            pd.DataFrame(coord_rows).to_csv(sidecars["cryptic_parent_coordinates"], sep="\t", index=False)
            pd.DataFrame([
                {
                    "sample": "CRYPTIC-T",
                    "parent_record_id": "parent_refill_a",
                    "transcript_block_order": 1,
                    "chromosome": "2",
                    "strand": "+",
                    "start0": 500,
                    "end0": 536,
                    "block_length": 36,
                    "derived_phase": 0,
                    "phase_provenance": "synthetic",
                    "coordinate_mapping_status": "coordinate_evaluable",
                },
                {
                    "sample": "CRYPTIC-T",
                    "parent_record_id": "parent_refill_b",
                    "transcript_block_order": 1,
                    "chromosome": "1",
                    "strand": "+",
                    "start0": 100,
                    "end0": 136,
                    "block_length": 36,
                    "derived_phase": 0,
                    "phase_provenance": "synthetic",
                    "coordinate_mapping_status": "coordinate_evaluable",
                },
            ]).to_csv(sidecars["cryptic_parent_orfcds"], sep="\t", index=False)
            write_tsv(sidecars["cryptic_peptide_genomic_footprint"], [], [
                "sample", "peptide_record_id", "parent_record_id", "source_parent_id",
                "mhc_class", "peptide", "peptide_start", "peptide_length",
                "codon_blocks_transcript_order", "peptide_cds_start0", "peptide_cds_end0",
                "candidate_coordinate_status", "reference_build",
            ])
            junction_rows = []
            for parent_id in ["parent_refill_a", "parent_refill_b"]:
                junction_rows.append({
                    "sample": "CRYPTIC-T",
                    "parent_record_id": parent_id,
                    "primary_core_status": "primary_core_eligible",
                    "junction_qc_status": "intronless_not_applicable",
                    "junction_qc_reasons": "",
                    "rna_variant_rescue_status": "reference_concordant",
                    "required_junction_count": 0,
                    "tumor_min_required_unique_reads": "",
                    "normal_any_required_junction_unique_ge1": False,
                    "normal_all_required_junctions_unique_ge1": False,
                    "normal_any_required_junction_unique_ge2": False,
                    "normal_all_required_junctions_unique_ge2": False,
                })
            write_tsv(
                parent_junction,
                junction_rows,
                [
                    "sample", "parent_record_id", "primary_core_status", "junction_qc_status",
                    "junction_qc_reasons", "rna_variant_rescue_status", "required_junction_count",
                    "tumor_min_required_unique_reads", "normal_any_required_junction_unique_ge1",
                    "normal_all_required_junctions_unique_ge1", "normal_any_required_junction_unique_ge2",
                    "normal_all_required_junctions_unique_ge2",
                ],
            )
            write_tsv(peptide_junction, [], [
                "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
                "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
                "crossed_required_junction_ids", "parent_primary_core_status",
                "parent_junction_qc_status", "parent_required_junction_count",
            ])
            human = root / "human.fa"
            human.write_text(">HUMAN\nACDEFGHIKLMN\n")
            output_signature = {
                path.name: {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in [
                    inputs["core"], inputs["parent_map"], sidecars["cryptic_parent_coordinates"],
                    sidecars["cryptic_parent_orfcds"], sidecars["cryptic_peptide_genomic_footprint"],
                    parent_core, parent_ranked, deferred_parent_map, parent_junction, peptide_junction,
                ]
            }
            inputs["manifest"].write_text(
                json.dumps(
                    {
                        "policy_version": "cryptic_core_qc_v1.1",
                        "run_status": "complete",
                        "binding_eligible": True,
                        "candidate_selection_policy_version": "ranked_prebinding_selection_v1.0",
                        "candidate_selection": {"mode": "ranked_cap", "max_hla_i_peptides": 1, "max_hla_ii_peptides": 1},
                        "human_reference_summary": {"status": "evaluated", "path": str(human)},
                        "input_signature": {"mhc_i_lengths": [8], "mhc_ii_lengths": [13]},
                        "output_signature": output_signature,
                    },
                    indent=2,
                )
            )
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            inputs.update({
                **sidecars,
                "parent_core": parent_core,
                "parent_ranked": parent_ranked,
                "deferred_parent_map": deferred_parent_map,
                "parent_junction_summary": parent_junction,
                "peptide_junction_evidence": peptide_junction,
                "human_proteome": human,
            })
            args = self._args(inputs, resources, root / "08c_second_parent_refill")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            args.max_hla_i_peptides = 1
            args.max_hla_ii_peptides = 1

            build_external_normal_qc(args)

            refill_map = pd.read_csv(root / "08c_second_parent_refill" / "cryptic_external_normal_refill_parent_map.tsv", sep="\t")
            excluded = pd.read_csv(root / "08c_second_parent_refill" / "cryptic_external_normal_excluded.tsv", sep="\t")
            self.assertEqual(set(refill_map.loc[refill_map["peptide"].eq(peptide), "parent_record_id"]), {"parent_refill_a", "parent_refill_b"})
            self.assertIn("external_normal_coordinate_frame_excluded", set(excluded["external_normal_status"]))
            self.assertIn(peptide, set(excluded["peptide"]))

    def test_human_reference_matching_does_not_cross_noncanonical_residues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            human = root / "human.fa"
            human.write_text(">HUMAN\nACDUFGHIKLMNXQRSTVWYA\n")
            candidates = {"ACDFGHIK", "LMNXQRST", "QRSTVWYA"}

            external_matches, external_summary = load_external_human_reference_matches(human, candidates)
            core_matches, core_summary = load_core_human_reference_matches(human, candidates)

            self.assertEqual(external_matches, core_matches)
            self.assertNotIn("ACDFGHIK", external_matches)
            self.assertNotIn("LMNXQRST", external_matches)
            self.assertIn("QRSTVWYA", external_matches)
            self.assertEqual(external_summary["matched_peptides"], core_summary["matched_peptides"])

    def test_v11_ranked_zero_candidate_complete_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root, core_rows=[], parent_rows=[], policy_version="cryptic_core_qc_v1.1")
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, [], [])
            parent_core = root / "08b" / "cryptic_parent_core.tsv"
            parent_ranked = root / "08b" / "cryptic_parent_ranked.tsv"
            deferred_parent_map = root / "08b" / "cryptic_peptide_deferred_parent_map.tsv"
            parent_junction = root / "08b" / "cryptic_parent_junction_summary.tsv"
            peptide_junction = root / "08b" / "cryptic_peptide_junction_evidence.tsv"
            write_tsv(parent_core, [], ["sample", "parent_record_id", "source_parent_id", "parent_sequence"])
            write_tsv(parent_ranked, [], ["sample", "parent_record_id", "source_parent_id", "parent_rank"])
            write_tsv(deferred_parent_map, [], self.parent_columns)
            write_tsv(
                parent_junction,
                [],
                [
                    "sample", "parent_record_id", "primary_core_status", "junction_qc_status",
                    "junction_qc_reasons", "rna_variant_rescue_status", "required_junction_count",
                    "tumor_min_required_unique_reads", "normal_any_required_junction_unique_ge1",
                    "normal_all_required_junctions_unique_ge1", "normal_any_required_junction_unique_ge2",
                    "normal_all_required_junctions_unique_ge2",
                ],
            )
            write_tsv(
                peptide_junction,
                [],
                [
                    "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
                    "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
                    "crossed_required_junction_ids", "parent_primary_core_status",
                    "parent_junction_qc_status", "parent_required_junction_count",
                ],
            )
            human = root / "human.fa"
            human.write_text(">HUMAN\nACDEFGHIKLMN\n")
            output_signature = {
                path.name: {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in [
                    inputs["core"],
                    inputs["parent_map"],
                    sidecars["cryptic_parent_coordinates"],
                    sidecars["cryptic_parent_orfcds"],
                    sidecars["cryptic_peptide_genomic_footprint"],
                    parent_core,
                    parent_ranked,
                    deferred_parent_map,
                    parent_junction,
                    peptide_junction,
                ]
            }
            inputs["manifest"].write_text(
                json.dumps(
                    {
                        "policy_version": "cryptic_core_qc_v1.1",
                        "run_status": "complete",
                        "binding_eligible": True,
                        "candidate_selection_policy_version": "ranked_prebinding_selection_v1.0",
                        "candidate_selection": {
                            "mode": "ranked_cap",
                            "max_hla_i_peptides": 1,
                            "max_hla_ii_peptides": 1,
                        },
                        "human_reference_summary": {"status": "evaluated", "path": str(human)},
                        "input_signature": {
                            "mhc_i_lengths": [8],
                            "mhc_ii_lengths": [13],
                        },
                        "output_signature": output_signature,
                    },
                    indent=2,
                )
            )
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            inputs.update({
                **sidecars,
                "parent_core": parent_core,
                "parent_ranked": parent_ranked,
                "deferred_parent_map": deferred_parent_map,
                "parent_junction_summary": parent_junction,
                "peptide_junction_evidence": peptide_junction,
                "human_proteome": human,
            })
            args = self._args(inputs, resources, root / "08c_zero_ranked")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            args.allow_missing_external_normal_resources = False
            args.external_normal_resource_manifest = str(resources["manifest"])
            args.coordinate_resource_manifest = str(resources["coordinate_manifest"])
            args.max_hla_i_peptides = 1
            args.max_hla_ii_peptides = 1

            manifest = build_external_normal_qc(args)
            reused = build_external_normal_qc(args)

            self.assertEqual(manifest["run_status"], "complete")
            self.assertTrue(manifest["binding_eligible"])
            self.assertEqual(manifest["final_unique_peptide_counts"]["total"], 0)
            self.assertTrue(reused["reused"])

    def test_v11_ranked_human_reference_refill_exclusion_is_not_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root, core_rows=[], parent_rows=[], policy_version="cryptic_core_qc_v1.1")
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, [], [])
            parent_core = root / "08b" / "cryptic_parent_core.tsv"
            parent_ranked = root / "08b" / "cryptic_parent_ranked.tsv"
            deferred_parent_map = root / "08b" / "cryptic_peptide_deferred_parent_map.tsv"
            parent_junction = root / "08b" / "cryptic_parent_junction_summary.tsv"
            peptide_junction = root / "08b" / "cryptic_peptide_junction_evidence.tsv"
            parent_sequence = "QRSTVWYA"
            parent_core_row = {
                "sample": "CRYPTIC-T",
                "parent_record_id": "parent_refill_human",
                "source_parent_id": "source_refill_human",
                "parent_sequence": parent_sequence,
                "parent_sequence_sha256": "sha_parent",
                "nc_class": "noncoding",
                "TPM_tumor": 10,
                "TPM_ctrl": 0,
                "log2FC": 5,
                "expression_qc_status": "pass",
                "cryptic_discovery_fdr_status": "not_estimable_single_pair_rule_based_qc",
            }
            write_tsv(parent_core, [parent_core_row], list(parent_core_row))
            write_tsv(
                parent_ranked,
                [{"sample": "CRYPTIC-T", "parent_record_id": "parent_refill_human", "source_parent_id": "source_refill_human", "parent_rank": 1}],
                ["sample", "parent_record_id", "source_parent_id", "parent_rank"],
            )
            write_tsv(deferred_parent_map, [], self.parent_columns)
            coord_row = {
                "sample": "CRYPTIC-T",
                "parent_record_id": "parent_refill_human",
                "block_count": 1,
                "blocks_transcript_order": blocks_to_json([GenomicBlock("1", "+", 100, 124)]),
                "block_total_length": 24,
                "coordinate_mapping_status": "coordinate_evaluable",
                "reference_genomic_translation_status": "pass",
                "chromosome": "1",
                "strand": "+",
            }
            pd.DataFrame([coord_row]).to_csv(sidecars["cryptic_parent_coordinates"], sep="\t", index=False)
            pd.DataFrame([{
                "sample": "CRYPTIC-T",
                "parent_record_id": "parent_refill_human",
                "transcript_block_order": 1,
                "chromosome": "1",
                "strand": "+",
                "start0": 100,
                "end0": 124,
                "block_length": 24,
                "derived_phase": 0,
                "phase_provenance": "synthetic",
                "coordinate_mapping_status": "coordinate_evaluable",
            }]).to_csv(sidecars["cryptic_parent_orfcds"], sep="\t", index=False)
            write_tsv(sidecars["cryptic_peptide_genomic_footprint"], [], [
                "sample", "peptide_record_id", "parent_record_id", "source_parent_id",
                "mhc_class", "peptide", "peptide_start", "peptide_length",
                "codon_blocks_transcript_order", "peptide_cds_start0", "peptide_cds_end0",
                "candidate_coordinate_status", "reference_build",
            ])
            write_tsv(
                parent_junction,
                [{
                    "sample": "CRYPTIC-T",
                    "parent_record_id": "parent_refill_human",
                    "primary_core_status": "primary_core_eligible",
                    "junction_qc_status": "intronless_not_applicable",
                    "junction_qc_reasons": "",
                    "rna_variant_rescue_status": "reference_concordant",
                    "required_junction_count": 0,
                    "tumor_min_required_unique_reads": "",
                    "normal_any_required_junction_unique_ge1": False,
                    "normal_all_required_junctions_unique_ge1": False,
                    "normal_any_required_junction_unique_ge2": False,
                    "normal_all_required_junctions_unique_ge2": False,
                }],
                [
                    "sample", "parent_record_id", "primary_core_status", "junction_qc_status",
                    "junction_qc_reasons", "rna_variant_rescue_status", "required_junction_count",
                    "tumor_min_required_unique_reads", "normal_any_required_junction_unique_ge1",
                    "normal_all_required_junctions_unique_ge1", "normal_any_required_junction_unique_ge2",
                    "normal_all_required_junctions_unique_ge2",
                ],
            )
            write_tsv(peptide_junction, [], [
                "sample", "peptide_record_id", "parent_record_id", "source_parent_id", "mhc_class",
                "peptide", "peptide_start", "peptide_length", "peptide_crosses_junction",
                "crossed_required_junction_ids", "parent_primary_core_status",
                "parent_junction_qc_status", "parent_required_junction_count",
            ])
            human = root / "human.fa"
            human.write_text(">HUMAN\nQRSTVWYA\n")
            output_signature = {
                path.name: {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256_file(path)}
                for path in [
                    inputs["core"],
                    inputs["parent_map"],
                    sidecars["cryptic_parent_coordinates"],
                    sidecars["cryptic_parent_orfcds"],
                    sidecars["cryptic_peptide_genomic_footprint"],
                    parent_core,
                    parent_ranked,
                    deferred_parent_map,
                    parent_junction,
                    peptide_junction,
                ]
            }
            inputs["manifest"].write_text(
                json.dumps(
                    {
                        "policy_version": "cryptic_core_qc_v1.1",
                        "run_status": "complete",
                        "binding_eligible": True,
                        "candidate_selection_policy_version": "ranked_prebinding_selection_v1.0",
                        "candidate_selection": {"mode": "ranked_cap", "max_hla_i_peptides": 1, "max_hla_ii_peptides": 1},
                        "human_reference_summary": {"status": "evaluated", "path": str(human)},
                        "input_signature": {"mhc_i_lengths": [8], "mhc_ii_lengths": [13]},
                        "output_signature": output_signature,
                    },
                    indent=2,
                )
            )
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            inputs.update({
                **sidecars,
                "parent_core": parent_core,
                "parent_ranked": parent_ranked,
                "deferred_parent_map": deferred_parent_map,
                "parent_junction_summary": parent_junction,
                "peptide_junction_evidence": peptide_junction,
                "human_proteome": human,
            })
            args = self._args(inputs, resources, root / "08c_human_refill")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            args.allow_missing_external_normal_resources = False
            args.external_normal_resource_manifest = str(resources["manifest"])
            args.coordinate_resource_manifest = str(resources["coordinate_manifest"])
            args.max_hla_i_peptides = 1
            args.max_hla_ii_peptides = 1

            manifest = build_external_normal_qc(args)
            excluded = pd.read_csv(root / "08c_human_refill" / "cryptic_external_normal_excluded.tsv", sep="\t")
            deferred = pd.read_csv(root / "08c_human_refill" / "cryptic_external_normal_refill_deferred.tsv", sep="\t")

            self.assertEqual(manifest["final_unique_peptide_counts"]["total"], 0)
            self.assertTrue(deferred.empty)
            self.assertIn("excluded_human_reference_match_before_external_normal_qc", set(excluded["external_normal_status"]))
            self.assertEqual(manifest["stage_counts"]["human_reference_refill_excluded_unique_peptides"], 1)
            self.assertEqual(manifest["stage_counts"]["total_qc_excluded_unique_peptides"], 1)
            self.assertEqual(manifest["stage_counts"]["external_normal_deferred_after_qc_unique_peptides"], 0)

    def test_coordinate_classifier_synthetic_cases(self) -> None:
        ref_plus = [GenomicBlock("1", "+", 100, 124)]
        self.assertEqual(
            classify_coordinate_match([GenomicBlock("1", "+", 100, 124)], 0, ref_plus)[0],
            "coordinate_frame_concordant",
        )
        ref_minus = [GenomicBlock("1", "-", 100, 124)]
        self.assertEqual(
            classify_coordinate_match([GenomicBlock("1", "-", 100, 124)], 0, ref_minus)[0],
            "coordinate_frame_concordant",
        )
        self.assertEqual(
            classify_coordinate_match([GenomicBlock("1", "+", 101, 125)], 0, [GenomicBlock("1", "+", 100, 130)])[0],
            "coordinate_overlap_frame_discordant",
        )
        self.assertEqual(
            classify_coordinate_match([GenomicBlock("1", "+", 95, 119)], 0, ref_plus)[0],
            "partial_coordinate_overlap",
        )
        self.assertEqual(
            classify_coordinate_match(
                [GenomicBlock("1", "+", 100, 106), GenomicBlock("1", "+", 202, 220)],
                0,
                [GenomicBlock("1", "+", 100, 110), GenomicBlock("1", "+", 200, 230)],
            )[0],
            "junction_chain_incompatible",
        )

    def test_v11_coordinate_matching_excludes_coordinate_only_without_double_counting_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = [
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep_coord",
                    "mhc_class": "MHC-I",
                    "peptide_length": 8,
                    "peptide": "AAAAAAAA",
                },
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep_both",
                    "mhc_class": "MHC-I",
                    "peptide_length": 8,
                    "peptide": "ACDEFGHI",
                },
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep_retained",
                    "mhc_class": "MHC-I",
                    "peptide_length": 8,
                    "peptide": "CDEFGHIK",
                },
            ]
            parent_rows = self.default_parent_rows(core_rows)[:3]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=parent_rows,
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            footprint = pd.read_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", dtype=str)
            retained_idx = footprint["peptide_record_id"].eq("pep_retained")
            footprint.loc[retained_idx, "codon_blocks_transcript_order"] = blocks_to_json(
                [GenomicBlock("1", "+", 1000, 1024)]
            )
            footprint.to_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", index=False)
            self._refresh_v11_manifest_signature(inputs, sidecars)
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            args = self._args({**inputs, **sidecars}, resources, root / "08c_v11")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True

            manifest = build_external_normal_qc(args)

            self.assertEqual(manifest["stage_counts"]["v1.0_exact_excluded_peptides"], 1)
            self.assertEqual(manifest["stage_counts"]["coordinate_frame_concordant_peptides"], 2)
            self.assertEqual(manifest["stage_counts"]["v1.1_coordinate_additional_excluded_peptides"], 1)
            self.assertEqual(manifest["stage_counts"]["final_primary_core_unique_peptides"], 1)
            annotated = pd.read_csv(root / "08c_v11" / "cryptic_external_normal_annotated_core.tsv", sep="\t")
            self.assertEqual(len(annotated), 3)
            retained = pd.read_csv(root / "08c_v11" / "cryptic_tumor_restricted_primary_core.tsv", sep="\t")
            self.assertEqual(set(retained["peptide_record_id"]), {"pep_retained"})
            excluded = pd.read_csv(root / "08c_v11" / "cryptic_external_normal_excluded.tsv", sep="\t")
            reasons = excluded.set_index("peptide_record_id")["external_normal_qc_reasons"].to_dict()
            statuses = excluded.set_index("peptide_record_id")["external_normal_status"].to_dict()
            self.assertIn("normal_smorf_coordinate_frame_concordant", reasons["pep_coord"])
            self.assertIn("normal_smorf_exact_peptide_match", reasons["pep_both"])
            self.assertIn("normal_smorf_coordinate_frame_concordant", reasons["pep_both"])
            self.assertEqual(statuses["pep_coord"], "external_normal_coordinate_frame_excluded")
            self.assertEqual(statuses["pep_both"], "external_normal_exact_and_coordinate_frame_excluded")

    def test_v11_coordinate_hash_mismatches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = self.default_core_rows()[:1]
            parent_rows = [
                {
                    "sample": row["sample"],
                    "peptide_record_id": row["peptide_record_id"],
                    "parent_record_id": f"parent_{row['peptide_record_id']}",
                    "source_parent_id": f"source_{row['peptide_record_id']}",
                    "mhc_class": row["mhc_class"],
                    "peptide_start": 1,
                    "peptide_length": row["peptide_length"],
                    "peptide": row["peptide"],
                }
                for row in core_rows
            ]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=parent_rows,
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            coords = pd.read_csv(sidecars["cryptic_parent_coordinates"], sep="\t", dtype=str)
            coords["chromosome"] = "2"
            coords["blocks_transcript_order"] = blocks_to_json([GenomicBlock("2", "+", 500, 524)])
            coords.to_csv(sidecars["cryptic_parent_coordinates"], sep="\t", index=False)
            orfcds = pd.read_csv(sidecars["cryptic_parent_orfcds"], sep="\t", dtype=str)
            orfcds["chromosome"] = "2"
            orfcds["start0"] = "500"
            orfcds["end0"] = "524"
            orfcds.to_csv(sidecars["cryptic_parent_orfcds"], sep="\t", index=False)
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            sidecars["cryptic_peptide_genomic_footprint"].write_text(
                sidecars["cryptic_peptide_genomic_footprint"].read_text() + "\n"
            )
            args = self._args({**inputs, **sidecars}, resources, root / "08c_v11_bad_sidecar")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            with self.assertRaisesRegex(ValueError, "output_signature"):
                build_external_normal_qc(args)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = self.default_core_rows()[:1]
            parent_rows = [
                {
                    "sample": row["sample"],
                    "peptide_record_id": row["peptide_record_id"],
                    "parent_record_id": f"parent_{row['peptide_record_id']}",
                    "source_parent_id": f"source_{row['peptide_record_id']}",
                    "mhc_class": row["mhc_class"],
                    "peptide_start": 1,
                    "peptide_length": row["peptide_length"],
                    "peptide": row["peptide"],
                }
                for row in core_rows
            ]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=parent_rows,
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            with gzip.open(resources["normal_smorf_orfcds"], "at") as handle:
                handle.write("bad\n")
            args = self._args({**inputs, **sidecars}, resources, root / "08c_v11_bad_resource")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_external_normal_qc(args)

    def test_v11_coordinate_sidecar_content_mismatch_fails_after_hash_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = self.default_core_rows()[:1]
            parent_rows = self.default_parent_rows(core_rows)[:1]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=parent_rows,
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            footprint = pd.read_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", dtype=str)
            footprint.loc[0, "peptide"] = "TTTTTTTT"
            footprint.to_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", index=False)
            self._refresh_v11_manifest_signature(inputs, sidecars)
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            args = self._args({**inputs, **sidecars}, resources, root / "08c_v11_bad_content")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            with self.assertRaisesRegex(ValueError, "disagrees with Core"):
                build_external_normal_qc(args)

    def test_v11_coordinate_footprint_keys_must_match_parent_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = self.default_core_rows()[:1]
            parent_rows = self.default_parent_rows(core_rows)[:1]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=parent_rows,
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            footprint = pd.read_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", dtype=str)
            footprint.iloc[0:0].to_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", index=False)
            self._refresh_v11_manifest_signature(inputs, sidecars)
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            args = self._args({**inputs, **sidecars}, resources, root / "08c_v11_missing_footprint")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            with self.assertRaisesRegex(ValueError, "keys must exactly match"):
                build_external_normal_qc(args)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = self.default_core_rows()[:1]
            parent_rows = self.default_parent_rows(core_rows)[:1]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=parent_rows,
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            footprint = pd.read_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", dtype=str)
            pd.concat([footprint, footprint.iloc[[0]]], ignore_index=True).to_csv(
                sidecars["cryptic_peptide_genomic_footprint"],
                sep="\t",
                index=False,
            )
            self._refresh_v11_manifest_signature(inputs, sidecars)
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            args = self._args({**inputs, **sidecars}, resources, root / "08c_v11_duplicate_footprint")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True
            with self.assertRaisesRegex(ValueError, "duplicate coordinate footprint keys"):
                build_external_normal_qc(args)

    def test_v11_empty_core_can_complete_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(
                root,
                core_rows=[],
                parent_rows=[],
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, [], [])
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            outdir = root / "empty_v11"
            args = self._args({**inputs, **sidecars}, resources, outdir)
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True

            manifest = build_external_normal_qc(args)

            self.assertEqual(manifest["stage_counts"]["source_core_unique_peptides"], 0)
            self.assertEqual(manifest["stage_counts"]["final_primary_core_unique_peptides"], 0)
            reused = build_external_normal_qc(args)
            self.assertTrue(reused["reused"])

            saved = json.loads((outdir / "run_manifest.json").read_text())
            saved["input_signature"]["coordinate_utils_sha256"] = "different-helper"
            (outdir / "run_manifest.json").write_text(json.dumps(saved, indent=2))
            with self.assertRaisesRegex(ValueError, "manifest does not match"):
                build_external_normal_qc(args)

    def test_coordinate_matching_enabled_fails_in_v1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._args(self._write_08b_core(root), self._write_resources(root), root / "08c")
            args.coordinate_matching_enabled = True
            with self.assertRaisesRegex(ValueError, "coordinate matching is not implemented in policy v1.0"):
                build_external_normal_qc(args)

    def test_upstream_manifest_missing_or_incomplete_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root)
            inputs["manifest"].unlink()
            with self.assertRaisesRegex(FileNotFoundError, "upstream"):
                build_external_normal_qc(self._args(inputs, resources, root / "missing"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root, run_status="running")
            resources = self._write_resources(root)
            with self.assertRaisesRegex(ValueError, "run_status=complete"):
                build_external_normal_qc(self._args(inputs, resources, root / "incomplete"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            saved = json.loads(inputs["manifest"].read_text())
            saved["binding_eligible"] = False
            inputs["manifest"].write_text(json.dumps(saved, indent=2))
            resources = self._write_resources(root)
            with self.assertRaisesRegex(ValueError, "binding_eligible=true"):
                build_external_normal_qc(self._args(inputs, resources, root / "not_binding_eligible"))

    def test_upstream_output_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root)
            inputs["core"].write_text(inputs["core"].read_text() + "CRYPTIC-T\tpepX\tMHC-I\t8\tACDEFGHI\n")

            with self.assertRaisesRegex(ValueError, "output_signature"):
                build_external_normal_qc(self._args(inputs, resources, root / "08c"))

    def test_v11_final_core_bad_peptide_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = [
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep_bad_sha",
                    "mhc_class": "MHC-I",
                    "peptide_length": 8,
                    "peptide": "QRSTVWYA",
                    "peptide_sequence_sha256": "wrong",
                }
            ]
            parent_rows = [
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "pep_bad_sha",
                    "parent_record_id": "parent_bad_sha",
                    "source_parent_id": "source_bad_sha",
                    "mhc_class": "MHC-I",
                    "peptide_start": 1,
                    "peptide_length": 8,
                    "peptide": "QRSTVWYA",
                }
            ]
            inputs = self._write_08b_core(
                root,
                core_rows=core_rows,
                parent_rows=parent_rows,
                policy_version="cryptic_core_qc_v1.1",
            )
            sidecars = self._write_v11_coordinate_sidecars(root, inputs, core_rows, parent_rows)
            coords = pd.read_csv(sidecars["cryptic_parent_coordinates"], sep="\t", dtype=str).fillna("")
            coords["chromosome"] = "2"
            coords["blocks_transcript_order"] = blocks_to_json([GenomicBlock("2", "+", 500, 524)])
            coords.to_csv(sidecars["cryptic_parent_coordinates"], sep="\t", index=False)
            orfcds = pd.read_csv(sidecars["cryptic_parent_orfcds"], sep="\t", dtype=str).fillna("")
            orfcds["chromosome"] = "2"
            orfcds["start0"] = "500"
            orfcds["end0"] = "524"
            orfcds.to_csv(sidecars["cryptic_parent_orfcds"], sep="\t", index=False)
            footprint = pd.read_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", dtype=str).fillna("")
            footprint["codon_blocks_transcript_order"] = blocks_to_json([GenomicBlock("2", "+", 500, 524)])
            footprint.to_csv(sidecars["cryptic_peptide_genomic_footprint"], sep="\t", index=False)
            self._refresh_v11_manifest_signature(inputs, sidecars)
            resources = {**self._write_resources(root), **self._write_coordinate_resources(root)}
            args = self._args({**inputs, **sidecars}, resources, root / "08c_bad_sha")
            args.policy_version = "cryptic_external_normal_qc_v1.1"
            args.coordinate_matching_enabled = True

            with self.assertRaisesRegex(AssertionError, "peptide_sequence_sha256"):
                build_external_normal_qc(args)

    def test_final_sidecar_not_evaluated_human_reference_fails_closed(self) -> None:
        retained = [{
            "sample": "CRYPTIC-T",
            "peptide_record_id": "pep1",
            "parent_record_id": "parent1",
            "mhc_class": "MHC-I",
            "peptide": "QRSTVWYA",
            "peptide_start": 1,
            "peptide_length": 8,
            "peptide_sequence_sha256": hashlib.sha256(b"QRSTVWYA").hexdigest(),
            "peptide_core_status": "core",
            "peptide_qc_reasons": "",
            "human_reference_peptide_status": "human_reference_peptide_not_detected",
        }]
        sidecar = [{
            **retained[0],
            "human_reference_peptide_status": "not_evaluated",
        }]
        footprint = [{
            **retained[0],
        }]
        junction = [{
            **retained[0],
        }]
        with self.assertRaisesRegex(AssertionError, "human-reference"):
            validate_final_core_contract(
                retained,
                sidecar,
                footprint,
                junction,
                [8],
                [13],
                human_reference_evaluated=True,
                formal_v11_ranked=True,
            )

    def test_parent_map_orphan_and_core_missing_parent_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = self.default_core_rows()
            parent_rows = self.default_parent_rows(core_rows)
            parent_rows.append(
                {
                    "sample": "CRYPTIC-T",
                    "peptide_record_id": "orphan",
                    "parent_record_id": "parent_orphan",
                    "source_parent_id": "source_orphan",
                    "mhc_class": "MHC-I",
                    "peptide_start": 1,
                    "peptide_length": 8,
                    "peptide": "ACDEFGHI",
                }
            )
            inputs = self._write_08b_core(root, core_rows=core_rows, parent_rows=parent_rows)
            resources = self._write_resources(root)
            with self.assertRaisesRegex(ValueError, "orphan peptide_record_id"):
                build_external_normal_qc(self._args(inputs, resources, root / "orphan"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core_rows = self.default_core_rows()
            parent_rows = [row for row in self.default_parent_rows(core_rows) if row["peptide_record_id"] != "pep5"]
            inputs = self._write_08b_core(root, core_rows=core_rows, parent_rows=parent_rows)
            resources = self._write_resources(root)
            with self.assertRaisesRegex(ValueError, "missing parent-map"):
                build_external_normal_qc(self._args(inputs, resources, root / "missing_parent"))

    def test_noncore_parent_map_rows_with_blank_ids_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root)
            parent_map = pd.read_csv(inputs["parent_map"], sep="\t", dtype=str, keep_default_na=False)
            parent_map["peptide_core_status"] = "core"
            parent_map.loc[len(parent_map)] = {
                "sample": "CRYPTIC-T",
                "peptide_record_id": "",
                "parent_record_id": "excluded_parent",
                "source_parent_id": "excluded_source",
                "mhc_class": "MHC-I",
                "peptide_length": "8",
                "peptide": "ACDEFGHI",
                "peptide_core_status": "excluded",
            }
            parent_map.to_csv(inputs["parent_map"], sep="\t", index=False)
            self.write_08b_manifest(inputs["manifest"], inputs["core"], inputs["parent_map"])

            manifest = build_external_normal_qc(self._args(inputs, resources, root / "08c"))

            evidence = pd.read_csv(root / "08c" / "cryptic_external_normal_evidence.tsv", sep="\t")
            self.assertEqual(manifest["stage_counts"]["source_core_unique_peptides"], 5)
            self.assertNotIn("excluded_parent", set(evidence["resource_record_id"]))

    def test_parent_map_sequence_class_length_and_sample_mismatches_fail(self) -> None:
        mutations = [
            ("sample", "OTHER", "sample mismatch"),
            ("peptide", "ACDEFGHK", "disagree"),
            ("peptide_length", 9, "peptide_length does not match"),
            ("mhc_class", "MHC-II", "HLA-II peptide length"),
        ]
        for column, value, pattern in mutations:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                core_rows = self.default_core_rows()
                parent_rows = self.default_parent_rows(core_rows)
                parent_rows[0][column] = value
                inputs = self._write_08b_core(root, core_rows=core_rows, parent_rows=parent_rows)
                resources = self._write_resources(root)
                with self.assertRaisesRegex(ValueError, pattern):
                    build_external_normal_qc(self._args(inputs, resources, root / f"bad_{column}"))

    def test_invalid_sequence_class_length_and_sample_fail(self) -> None:
        mutations = [
            ("sample", "OTHER", "sample mismatch"),
            ("peptide", "ACDEFGHX", "non-standard"),
            ("peptide_length", 9, "peptide_length does not match"),
            ("mhc_class", "MHC-III", "unsupported mhc_class"),
        ]
        for column, value, pattern in mutations:
            with self.subTest(column=column), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                core_rows = self.default_core_rows()
                core_rows[0][column] = value
                parent_rows = self.default_parent_rows(core_rows)
                inputs = self._write_08b_core(root, core_rows=core_rows, parent_rows=parent_rows)
                resources = self._write_resources(root)
                with self.assertRaisesRegex(ValueError, pattern):
                    build_external_normal_qc(self._args(inputs, resources, root / f"bad_core_{column}"))

    def test_wrong_resource_contract_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root, freeze_version="wrong")
            with self.assertRaisesRegex(ValueError, "freeze_version"):
                build_external_normal_qc(self._args(inputs, resources, root / "wrong_freeze"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root, hla_i_lengths=[8, 9, 10])
            with self.assertRaisesRegex(ValueError, "HLA-I length policy"):
                build_external_normal_qc(self._args(inputs, resources, root / "wrong_hla_policy"))

    def test_resource_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root)
            with gzip.open(resources["smorf_index"], "at") as handle:
                handle.write("HLA-I\tMISMATCH\t8\n")

            with self.assertRaises(ValueError):
                build_external_normal_qc(self._args(inputs, resources, root / "08c"))

    def test_missing_resources_exploratory_output_is_not_primary_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = {"manifest": root / "missing_manifest.json"}
            args = self._args(inputs, resources, root / "08c_exploratory")
            args.allow_missing_external_normal_resources = True

            manifest = build_external_normal_qc(args)

            self.assertEqual(manifest["external_normal_evaluation"], "not_evaluated")
            self.assertEqual(manifest["binding_input_fasta"], "")
            self.assertEqual(manifest["binding_input_mode"], "not_available")
            self.assertEqual(manifest["stage_counts"]["tumor_restricted_primary_core_unique_peptides"], 0)
            annotated = pd.read_csv(args.outdir + "/cryptic_external_normal_annotated_core.tsv", sep="\t")
            primary = pd.read_csv(args.outdir + "/cryptic_tumor_restricted_primary_core.tsv", sep="\t")
            excluded = pd.read_csv(args.outdir + "/cryptic_external_normal_excluded.tsv", sep="\t")
            self.assertEqual(len(annotated), 5)
            self.assertEqual(len(primary), 0)
            self.assertEqual(len(excluded), len(annotated))
            self.assertTrue(excluded["external_normal_status"].eq("external_normal_resources_not_evaluated").all())

    def test_empty_core_can_complete_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root, core_rows=[], parent_rows=[])
            resources = self._write_resources(root)
            outdir = root / "empty"

            manifest = build_external_normal_qc(self._args(inputs, resources, outdir))

            self.assertEqual(manifest["stage_counts"]["source_core_unique_peptides"], 0)
            self.assertEqual(manifest["stage_counts"]["tumor_restricted_primary_core_unique_peptides"], 0)
            self.assertEqual((outdir / "cryptic_tumor_restricted_primary_core.fasta").read_text(), "")
            reused = build_external_normal_qc(self._args(inputs, resources, outdir))
            self.assertTrue(reused["reused"])

    def test_code_signature_change_invalidates_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self._write_08b_core(root)
            resources = self._write_resources(root)
            outdir = root / "08c"
            args = self._args(inputs, resources, outdir)

            manifest = build_external_normal_qc(args)
            self.assertIn("script_sha256", manifest["input_signature"])

            manifest_path = outdir / "run_manifest.json"
            saved = json.loads(manifest_path.read_text())
            saved["input_signature"]["script_sha256"] = "different-script"
            manifest_path.write_text(json.dumps(saved, indent=2))

            with self.assertRaises(ValueError):
                build_external_normal_qc(args)


if __name__ == "__main__":
    unittest.main()
