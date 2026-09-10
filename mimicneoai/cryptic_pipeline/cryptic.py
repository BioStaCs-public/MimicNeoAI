# coding=utf-8
"""
Launcher for the cryptic (lncRNA/sORF-derived) antigen pipeline.
It mirrors the original run.sh flow:

00  QC (tumor + optional control)
01  STAR alignment
02  lncRNA/sORF discovery (known + novel/trinity)
03  Salmon quant (tumor)
04  Salmon quant (control)
05  HLA typing (hlahd)
06  Extract aberrantly expressed sORF peptides (aeSEPs)
07  ORF genome annotation
08  ORF-level filtering
08b Cryptic Core QC
08c External-normal exact sequence QC
09  HLA binding prediction (pvacbind/IEDB or MimicNeoAI backend)

This module:
- Uses absolute imports within the 'mimicneoai' package
- Avoids side effects at import time
- Reads YAML 'configure' and 'paths' like other pipelines
- Spawns a process pool across samples
"""

from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
import yaml
import argparse
import traceback
import shlex
import sys
from typing import Dict, Any, List, Tuple
from multiprocessing import Manager
import multiprocessing.pool
from importlib.resources import files
from mimicneoai.functions.binding_prediction import configured_predictor_cli_args
from mimicneoai.functions.immunogenicity_runner import (
    resolve_immunogenicity_model_root,
    resolve_immunogenicity_python_bin,
)
from mimicneoai.functions.pipline_tools import raise_for_failed_samples, tools


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_peptide_core_binding_manifest(manifest_path: str, fasta_path: str) -> None:
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"peptide-core binding requires 08c manifest: {manifest_path}")
    if not os.path.isfile(fasta_path):
        raise FileNotFoundError(f"peptide-core binding FASTA does not exist: {fasta_path}")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("run_status") != "complete":
        raise ValueError(f"08c manifest must have run_status=complete: {manifest_path}")
    if manifest.get("binding_eligible") is not True:
        raise ValueError(f"08c manifest must have binding_eligible=true before binding: {manifest_path}")
    if manifest.get("binding_input_mode") != "peptide-core":
        raise ValueError(f"08c manifest must have binding_input_mode=peptide-core: {manifest_path}")
    identity = manifest.get("final_binding_fasta_identity", {})
    if not isinstance(identity, dict):
        raise ValueError(f"08c manifest missing final_binding_fasta_identity: {manifest_path}")
    if os.path.abspath(str(identity.get("path", ""))) != os.path.abspath(fasta_path):
        raise ValueError("08c final FASTA path does not match binding input")
    if int(identity.get("size", -1)) != os.path.getsize(fasta_path):
        raise ValueError("08c final FASTA size does not match manifest")
    if str(identity.get("sha256", "")) != _sha256_file(fasta_path):
        raise ValueError("08c final FASTA SHA256 does not match manifest")


# ---------------------- Constants ----------------------
FLAG = "CrypticAntigen"
STEP_NAME = {
    "QC": "00-clean",
    "align": "01-star",
    "known": "02-known",
    "novel": "03-novel",
    "salmon": "04-salmon_quant",
    "hla": "05-hla_typing",
    "aeseps": "06-aeSEPs",
    "orf_genome_annotation": "07-orf_genome_annotation",
    "orf_filter": "08-orf_filter",
    "cryptic_core": "08b-cryptic_core_qc",
    "external_normal": "08c-external_normal_qc",
    "pvacbind": "09-hla_binding_pred",
    "mimicneoai_binding": "09-hla_binding_pred_mimicneoai",
    "immunogenicity": "10-immunogenicity_prediction_mimicneoai",
}

STAR_FREEZE_REQUIRED_SUFFIXES = ("Aligned.out.bam", "SJ.out.tab", "Log.final.out", "Log.out")


# ---------------------- Helpers ----------------------
def _script_path(rel_name: str) -> str:
    """
    Resolve the absolute path of a script inside cryptic_pipeline/scripts/.
    Example: _script_path("00-QC.py")
    """
    pkg_path = files("mimicneoai.cryptic_pipeline.scripts")
    return str(pkg_path / rel_name)


def _run_cmd(
    tool: tools,
    run_sample_id: str,
    cmd: List[str],
    cwd: str | None = None,
    display_name: str | None = None,
) -> None:
    """
    Execute a shell command via tool.exec_cmd with logging; raise on failure.
    All commands are string-joined safely with shlex.quote.
    """
    cmd = ' '.join(shlex.quote(c) for c in cmd)

    prev_cwd = os.getcwd()
    try:
        if cwd:
            os.chdir(cwd)
        # Single entry point for execution: handles logging, sample ID tagging,
        # stdout/stderr capturing, and returning codes.
        tool.exec_cmd(cmd, run_sample_id, pipline='cryptic', display_name=display_name)
    except Exception:
        tool.write_log(f"[CMD FAILED]\n{traceback.format_exc()}", "error")
        raise
    finally:
        if cwd:
            os.chdir(prev_cwd)


def _resolve_tumor_control(sample: str, ctrl_from_cfg: str | None) -> Tuple[str, str | None]:
    """
    Resolve tumor and control sample names.
    Supports two formats:
      1) samples list item is "TUMOR,CTRL"
      2) samples list item is "TUMOR", and control comes from configure['others']['control_sample']
    """
    if "," in sample:
        tumor, ctrl = sample.split(",", 1)
        return tumor.strip(), ctrl.strip()
    return sample.strip(), (ctrl_from_cfg.strip() if ctrl_from_cfg else None)


def _external_normal_resource_value(
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    config_key: str,
    paths_key: str,
) -> str:
    config_resources = configure.get("external_normal_resources", {}) or {}
    if not isinstance(config_resources, dict):
        raise ValueError("external_normal_resources must be a mapping")
    configured = str(config_resources.get(config_key, "") or "").strip()
    if configured:
        return configured
    return str(
        paths.get("database", {})
        .get("cryptic", {})
        .get("EXTERNAL_NORMAL_RESOURCES", {})
        .get(paths_key, "")
        or ""
    ).strip()


def _write_auto_star_pair_sheet(pair_sheet_path: str, tumor_sample: str, ctrl_sample: str | None) -> None:
    if not ctrl_sample:
        raise ValueError(
            "junction_qc.auto_freeze_star_provenance requires paired sample format: Tumor,Normal"
        )
    if tumor_sample == ctrl_sample:
        raise ValueError("STAR provenance freeze requires distinct tumor and control sample IDs")
    path = Path(pair_sheet_path)
    text = f"tumor_sample\tnormal_sample\n{tumor_sample}\t{ctrl_sample}\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise RuntimeError(f"Existing auto STAR pair sheet differs from current sample pair: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _star_dir_for_pair_freeze(cryptic_root: str, tumor_sample: str, star_sample: str) -> Path:
    return (
        Path(cryptic_root)
        / tumor_sample
        / STEP_NAME["align"]
        / star_sample
        / f"{star_sample}.star"
    )


def _require_star_provenance_inputs(cryptic_root: str, tumor_sample: str, ctrl_sample: str | None) -> None:
    if not ctrl_sample:
        raise ValueError("junction_qc.auto_freeze_star_provenance requires matched-control STAR outputs")
    missing: list[str] = []
    for star_sample in (tumor_sample, ctrl_sample):
        star_dir = _star_dir_for_pair_freeze(cryptic_root, tumor_sample, star_sample)
        for suffix in STAR_FREEZE_REQUIRED_SUFFIXES:
            path = star_dir / f"{star_sample}{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(str(path))
    normal_manifest = _star_dir_for_pair_freeze(
        cryptic_root, tumor_sample, ctrl_sample
    ) / "star_alignment.manifest.json"
    if not normal_manifest.exists() or normal_manifest.stat().st_size == 0:
        missing.append(str(normal_manifest))
    if missing:
        raise FileNotFoundError(
            "junction_qc.auto_freeze_star_provenance requires complete tumor/control STAR outputs. "
            "Enable others.alignment_control or provide junction_qc.star_pair_inputs. Missing: "
            + ", ".join(missing)
        )


def _resolve_star_pair_inputs_for_junction_qc(
    tool: tools,
    run_sample_id: str,
    junction_qc: Dict[str, Any],
    cryptic_root: str,
    dir01: str,
    tumor_sample: str,
    ctrl_sample: str | None,
    star_index: str,
) -> str:
    explicit_pair_inputs = str(junction_qc.get("star_pair_inputs", "") or "").strip()
    if explicit_pair_inputs:
        return explicit_pair_inputs
    if not bool(junction_qc.get("auto_freeze_star_provenance", False)):
        raise ValueError(
            "junction_qc.enabled requires junction_qc.star_pair_inputs, or "
            "junction_qc.auto_freeze_star_provenance: true"
        )

    _require_star_provenance_inputs(cryptic_root, tumor_sample, ctrl_sample)

    freeze_outdir = str(junction_qc.get("star_provenance_outdir", "") or "").strip()
    if not freeze_outdir:
        freeze_outdir = os.path.join(dir01, "star-provenance-freeze")
    elif not os.path.isabs(freeze_outdir):
        freeze_outdir = os.path.join(dir01, freeze_outdir)

    pair_sheet = os.path.join(freeze_outdir, "auto_star_pair_sheet.tsv")
    _write_auto_star_pair_sheet(pair_sheet, tumor_sample, ctrl_sample)

    cmd = [
        sys.executable, _script_path("freeze_star_provenance.py"),
        "--cryptic-root", cryptic_root,
        "--pair-sheet", pair_sheet,
        "-o", freeze_outdir,
    ]
    if star_index:
        cmd.extend(["--star-index", star_index])
    if bool(junction_qc.get("full_bam_hash", False)):
        cmd.append("--full-bam-hash")
    if bool(junction_qc.get("allow_critical_contract_upgrade", False)):
        cmd.append("--allow-critical-contract-upgrade")
    _run_cmd(tool, run_sample_id, cmd, display_name="STAR provenance freeze")
    return os.path.join(freeze_outdir, "cryptic_star_pair_inputs.tsv")


# ---------------------- One-sample pipeline ----------------------
def _run_one_sample(
    sample: str,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
) -> None:
    """
    Orchestrate all pipeline steps for a single sample, honoring toggles in the YAML.
    """
    try:
        # ---------- Toggles and core params ----------
        args_cfg = configure["args"]
        others = configure["others"]
        ipt_root = configure["path"]["input_dir"]
        out_root = configure["path"]["output_dir"]

        # Threads (kept simple: using a shared 'threads' entry unless split later)
        n_qc = int(args_cfg.get("threads", 30))
        n_align = int(args_cfg.get("threads", 30))
        n_lncsorf = int(args_cfg.get("threads", 30))
        n_trinity_cpu = int(args_cfg.get("threads", 30))
        n_salmon = int(args_cfg.get("threads", 30))
        n_hla = int(args_cfg.get("threads", 30))
        n_pvacbind = int(args_cfg.get("hla_binding_threads", 5))

        # Feature switches
        do_qc = bool(others.get("QC", True))
        do_align = bool(others.get("alignment", True))
        do_align_ctrl = bool(others.get("alignment_control", False))
        do_known = bool(others.get("known", True))
        do_novel = bool(others.get("novel", True))
        do_quant = bool(others.get("salmon_quant", True))
        do_quant_ctrl = bool(others.get("salmon_quant_control", True))
        do_hla = bool(others.get("hlatyping", True))
        do_aeseps = bool(others.get("extract_aeseps", True))
        do_orf_annotation = bool(others.get("orf_genome_annotation", True))
        do_orf_filter = bool(others.get("orf_filter", True))
        do_cryptic_core = bool(others.get("cryptic_core_qc", False))
        do_external_normal = bool(others.get("cryptic_external_normal_qc", False))
        do_pvacbind = bool(others.get("hla_binding_pred", True))
        allow_missing_external_normal = bool(others.get("allow_missing_external_normal_resources", False))

        # Tumor/control resolution
        tumor_sample, ctrl_sample = _resolve_tumor_control(sample, None)

        # External tools (paths)
        TRINITY_SIF = paths["path"]["cryptic"]["TRINITY_SIF"]
        PVACTOOLS_SIF = paths["path"]["common"]["PVACTOOLS"]

        # References (cryptic branch)
        STAR_GENOME_DIR = paths["database"]["cryptic"]["STAR_GENOME_DIR"]

        REF_DIR = paths["database"]["cryptic"]["REF"]["REF_DIR"]
        REF_FA = paths["database"]["cryptic"]["REF"]["REF_FA"]
        REF_GTF = paths["database"]["cryptic"]["REF"]["REF_GTF"]
        REF_LNC_GTF = paths["database"]["cryptic"]["REF"]["REF_LNC_GTF"]
        # Backward compatibility: prefer REF_GENOME if provided, else REF_FA
        REF_GENOME = paths["database"]["cryptic"]["REF"].get("REF_GENOME", REF_FA)

        # HLAHD resources (common)
        HLA_FREQ_DIR = paths["database"]["common"]["HLA"]["FREQ_DATA_DIR"]
        HLA_GENE = paths["database"]["common"]["HLA"]["HLA_GENE"]
        HLA_DICT = paths["database"]["common"]["HLA"]["DICTIONARY"]
        HLA_BOWTIE2_INDEX = paths["database"]["common"]["HLA"]["BOWTIE2_INDEX"]
        HLAHD_SCRIPT = paths["database"]["common"]["HLA"].get("HLAHD_SCRIPT", "")

        # Output layout
        OPT = os.path.join(out_root, tumor_sample)
        DIR00 = os.path.join(OPT, STEP_NAME["QC"])
        DIR01 = os.path.join(OPT, STEP_NAME["align"])
        DIR02_KNOWN = os.path.join(OPT, STEP_NAME["known"])
        DIR03_NOVEL = os.path.join(OPT, STEP_NAME["novel"])
        DIR04 = os.path.join(OPT, STEP_NAME["salmon"])
        DIR05 = os.path.join(OPT, STEP_NAME["hla"])
        DIR06 = os.path.join(OPT, STEP_NAME["aeseps"])
        DIR07_ORF = os.path.join(OPT, STEP_NAME["orf_genome_annotation"])
        DIR08_ORF = os.path.join(OPT, STEP_NAME["orf_filter"])
        DIR08B_CORE = os.path.join(OPT, STEP_NAME["cryptic_core"])
        DIR08C_EXTERNAL = os.path.join(OPT, STEP_NAME["external_normal"])
        DIR07 = os.path.join(OPT, STEP_NAME["pvacbind"])
        SHARED = os.path.join(OPT, "023-shared")

        os.makedirs(OPT, exist_ok=True)

        # Raw FASTQs (provided by the caller/configure)
        RAW_R1 = f"{ipt_root}/{tumor_sample}/{tumor_sample}.R1.fq.gz"
        RAW_R2 = f"{ipt_root}/{tumor_sample}/{tumor_sample}.R2.fq.gz"

        RAW_CTRL_R1 = None
        RAW_CTRL_R2 = None
        if ctrl_sample:
            RAW_CTRL_R1 = f"{ipt_root}/{ctrl_sample}/{ctrl_sample}.R1.fq.gz"
            RAW_CTRL_R2 = f"{ipt_root}/{ctrl_sample}/{ctrl_sample}.R2.fq.gz"

        if do_align_ctrl:
            if not ctrl_sample:
                raise ValueError("alignment_control requires paired sample format: Tumor,Normal")
            if ctrl_sample == tumor_sample:
                raise ValueError("alignment_control requires distinct tumor and control sample IDs")
            missing_ctrl_fastqs = [
                path for path in (RAW_CTRL_R1, RAW_CTRL_R2)
                if not path or not os.path.isfile(path)
            ]
            if missing_ctrl_fastqs:
                raise FileNotFoundError(
                    "alignment_control requires existing control FASTQ input(s): "
                    + ", ".join(missing_ctrl_fastqs)
                )

        # Derived files
        QC_R1 = os.path.join(DIR00, f"{tumor_sample}.R1.QC.fq.gz")
        QC_R2 = os.path.join(DIR00, f"{tumor_sample}.R2.QC.fq.gz")

        QC_CTRL_R1 = os.path.join(DIR00, f"{ctrl_sample}.R1.QC.fq.gz") if ctrl_sample else None
        QC_CTRL_R2 = os.path.join(DIR00, f"{ctrl_sample}.R2.QC.fq.gz") if ctrl_sample else None

        REF_TX_FA = os.path.join(DIR03_NOVEL, "04a.trace_to_ref", "ref.transcripts.fa")
        CONTIGS_FA = os.path.join(DIR03_NOVEL, "04a.trace_to_ref", "contigs.annot.fa")

        QUANT_TUMOR = os.path.join(DIR04, "salmon_quant", "quant.sf")
        QUANT_CTRL = os.path.join(DIR04, "salmon_quant_control", "quant.sf")
        DEDUP_MAP = os.path.join(DIR04, f"{tumor_sample}.merged_tx.dedup.map.tsv")

        AESEPs_PEP = os.path.join(DIR06, f"{tumor_sample}.aeSEPs.pep")
        ABERRANT_TABLE = os.path.join(DIR06, f"{tumor_sample}.aberrant_noncoding.annot.csv")
        HLA_FINAL_TXT = os.path.join(DIR05, tumor_sample, "result", f"{tumor_sample}_final.result.txt")
        ORF_FILTERED_AESEPs_PEP = os.path.join(DIR08_ORF, f"{tumor_sample}.aeSEPs.orf_filtered.pep")
        ORF_FINAL_TABLE = os.path.join(DIR08_ORF, "orf_final.csv")
        CRYPTIC_CORE_TSV = os.path.join(DIR08B_CORE, "cryptic_peptide_core.tsv")
        CRYPTIC_CORE_PARENT_MAP = os.path.join(DIR08B_CORE, "cryptic_peptide_parent_map.tsv")
        CRYPTIC_DEFERRED_PEPTIDE = os.path.join(DIR08B_CORE, "cryptic_peptide_deferred.tsv")
        CRYPTIC_DEFERRED_PARENT_MAP = os.path.join(DIR08B_CORE, "cryptic_peptide_deferred_parent_map.tsv")
        CRYPTIC_PARENT_CORE_TSV = os.path.join(DIR08B_CORE, "cryptic_parent_core.tsv")
        CRYPTIC_PARENT_RANKED = os.path.join(DIR08B_CORE, "cryptic_parent_ranked.tsv")
        CRYPTIC_CORE_MANIFEST = os.path.join(DIR08B_CORE, "run_manifest.json")
        CRYPTIC_CORE_FASTA = os.path.join(DIR08B_CORE, "cryptic_peptide_core.fasta")
        CRYPTIC_PARENT_COORDINATES = os.path.join(DIR08B_CORE, "cryptic_parent_coordinates.tsv")
        CRYPTIC_PARENT_ORFCDS = os.path.join(DIR08B_CORE, "cryptic_parent_orfcds.tsv")
        CRYPTIC_PEPTIDE_FOOTPRINT = os.path.join(DIR08B_CORE, "cryptic_peptide_genomic_footprint.tsv")
        CRYPTIC_PARENT_JUNCTION_SUMMARY = os.path.join(DIR08B_CORE, "cryptic_parent_junction_summary.tsv")
        CRYPTIC_PEPTIDE_JUNCTION_EVIDENCE = os.path.join(DIR08B_CORE, "cryptic_peptide_junction_evidence.tsv")
        CRYPTIC_PRIMARY_CORE_FASTA = os.path.join(DIR08C_EXTERNAL, "cryptic_tumor_restricted_primary_core.fasta")
        CRYPTIC_EXTERNAL_MANIFEST = os.path.join(DIR08C_EXTERNAL, "run_manifest.json")
        ORF_BED12 = os.path.join(DIR07_ORF, "orf.noUnmap.noSup.bed12")
        ORF_BAM = os.path.join(DIR07_ORF, "orf2genome.bam")
        ORF_CDS_FASTA = os.path.join(DIR07_ORF, f"{tumor_sample}.SEPs.cds.fa")

        # Minimum requirements for aeSEPs (can be overridden in YAML)
        min_tpm_tumor = float(others.get("min_tpm_tumor", 5.0))
        max_tpm_ctrl = float(others.get("max_tpm_ctrl", 0.5))
        min_log2fc = float(others.get("min_log2fc", 4.0))
        strandedness = str(others.get("strandedness", "reverse")).strip().lower()

        # ---------- 00 QC (tumor) ----------
        if do_qc:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("00-QC.py"),
                "--fq1", RAW_R1,
                "--fq2", RAW_R2,
                "-o", DIR00,
                "-p", str(n_qc),
            ], display_name="Tumor RNA QC")

        # ---------- 00 QC (control) ----------
        if do_qc and ctrl_sample:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("00-QC.py"),
                "--fq1", RAW_CTRL_R1,
                "--fq2", RAW_CTRL_R2,
                "-o", DIR00,
                "-p", str(n_qc),
            ], display_name="Control RNA QC")

        # ---------- 01 STAR alignment ----------
        if do_align:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("01-alignment.py"),
                "-s", tumor_sample,
                "--clean-dir", DIR00,
                "--genome-dir", STAR_GENOME_DIR,
                "--out-root", DIR01,
                "-p", str(n_align),
                "--alignment-role", "tumor",
                "--tumor-sample", tumor_sample,
                "--raw-fq1", RAW_R1,
                "--raw-fq2", RAW_R2,
            ], display_name="STAR alignment")

        if do_align_ctrl:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("01-alignment.py"),
                "-s", ctrl_sample,
                "--clean-dir", DIR00,
                "--genome-dir", STAR_GENOME_DIR,
                "--out-root", DIR01,
                "-p", str(n_align),
                "--alignment-role", "control",
                "--tumor-sample", tumor_sample,
                "--control-sample", ctrl_sample,
                "--raw-fq1", RAW_CTRL_R1,
                "--raw-fq2", RAW_CTRL_R2,
            ], display_name="Control STAR alignment")

        # Use produced BAM if available (to pass into step 02)
        IN_BAM = os.path.join(DIR01, tumor_sample, f"{tumor_sample}.star", f"{tumor_sample}Aligned.out.bam")
        extra_in_bam = ["--in-bam", IN_BAM] if os.path.isfile(IN_BAM) else []

        # ---------- 02 known lnc/sORF ----------
        if do_known:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("02-lnc_sORF_pipeline.py"),
                "-s", tumor_sample,
                "-m", "known",
                "-o", DIR02_KNOWN,
                "--shared-dir", SHARED,
                "--threads", str(n_lncsorf),
                "--ref-dir", REF_DIR,
                "--ref-fa", REF_FA,
                "--ref-gtf", REF_GTF,
                "--ref-lnc-gtf", REF_LNC_GTF,
                *extra_in_bam,
            ], display_name="Known noncoding sORF detection")

        # ---------- 02 novel lnc/sORF (with Trinity) ----------
        if do_novel:
            cmd_novel = [
                sys.executable, _script_path("02-lnc_sORF_pipeline.py"),
                "-s", tumor_sample,
                "-m", "novel",
                "-o", DIR03_NOVEL,
                "--shared-dir", SHARED,
                "--threads", str(n_lncsorf),
                "--strandedness", strandedness,
                "--trinity-mode", str(others.get("trinity_mode", "apptainer")),
                "--trinity-sif", TRINITY_SIF,
                "--trinity-cpu", str(n_trinity_cpu),
                "--trinity-mem", str(args_cfg.get("trinity_mem", "100G")),
                "--ref-dir", REF_DIR,
                "--ref-fa", REF_FA,
                "--ref-gtf", REF_GTF,
                "--ref-lnc-gtf", REF_LNC_GTF,
                *extra_in_bam,
            ]
            _run_cmd(tool, sample, cmd_novel, display_name="Novel noncoding sORF discovery")

        # ---------- 03 Salmon quant (tumor) ----------
        if do_quant:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("03-salmon_quant.py"),
                "-s", tumor_sample,
                "-o", DIR04,
                "--ref-tx-fa", REF_TX_FA,
                "--contigs-fa", CONTIGS_FA,
                "--genome-fa", REF_GENOME,
                "--fq1", QC_R1,
                "--fq2", QC_R2,
                "--threads", str(n_salmon),
                "--kmer", str(int(others.get("salmon_kmer", 31))),
            ], display_name="Tumor transcript quantification")

        # ---------- 04 Salmon quant (control) ----------
        if do_quant_ctrl and ctrl_sample:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("04-salmon_quant_control.py"),
                "--fq1", QC_CTRL_R1,
                "--fq2", QC_CTRL_R2,
                "-i", os.path.join(DIR04, "salmon_index"),
                "-o", os.path.join(DIR04, "salmon_quant_control"),
                "-p", str(n_salmon),
            ], display_name="Control transcript quantification")

        # ---------- 05 HLA typing (hlahd) ----------
        if do_hla:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("05-hla_typing.py"),
                "-s", tumor_sample,
                "--r1", QC_R1,
                "--r2", QC_R2,
                "--output-dir", OPT,
                "--step-name-hla", os.path.basename(DIR05),
                "-t", str(n_hla),
                "--freq-data-dir", HLA_FREQ_DIR,
                "--HLA-gene", HLA_GENE,
                "--dictionary", HLA_DICT,
                "--hla-gen", HLA_BOWTIE2_INDEX,
                *(["--hlahd-bin", HLAHD_SCRIPT] if HLAHD_SCRIPT else []),
            ], display_name="HLA-HD typing")

        # ---------- 06 Extract aeSEPs ----------
        if do_aeseps:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("06-sORF-encoded_peptides.py"),
                "--quant-tumor", QUANT_TUMOR,
                "--quant-ctrl", QUANT_CTRL if ctrl_sample else "",
                "--dedup-map", DEDUP_MAP,
                "--known-tx-fa", os.path.join(DIR02_KNOWN, "04k.bcf_consensus", "lncRNA.consensus.fa"),
                "--pep-known", os.path.join(DIR02_KNOWN, "06.filter_sORFs", f"{tumor_sample}.lncRNA.sORFs.pep"),
                "--pep-novel", os.path.join(DIR03_NOVEL, "06.filter_sORFs", f"{tumor_sample}.lncRNA.sORFs.pep"),
                "--out-dir", DIR06,
                "--out-name", os.path.basename(AESEPs_PEP),
                "--save-table", os.path.basename(ABERRANT_TABLE),
                "--min-tpm-tumor", str(min_tpm_tumor),
                "--max-tpm-ctrl", str(max_tpm_ctrl),
                "--min-log2fc", str(min_log2fc),
            ], display_name="aeSEP extraction")

        # ---------- 07 ORF genome annotation ----------
        if do_orf_annotation:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("08-orf_genome_annotation.py"),
                "-s", tumor_sample,
                "--sample-dir", OPT,
                "-o", DIR07_ORF,
                "--genome-fa", REF_GENOME,
                "--gtf", REF_GTF,
                "--threads", str(int(others.get("orf_annotation_threads", n_lncsorf))),
                "--sort-threads", str(int(others.get("orf_annotation_sort_threads", 8))),
            ], display_name="ORF genome annotation")

        binding_pep_fasta = AESEPs_PEP
        binding_input_mode = "parent-fasta"
        human_proteome_fasta = ""
        if do_orf_filter:
            _run_cmd(tool, sample, [
                sys.executable, _script_path("08-orf_filter.py"),
                "-s", tumor_sample,
                "--sample-dir", OPT,
                "-o", DIR08_ORF,
                "--orf-annotation-dir", DIR07_ORF,
            ], display_name="ORF annotation filtering")
            binding_pep_fasta = ORF_FILTERED_AESEPs_PEP

        if do_cryptic_core:
            if not do_orf_filter:
                missing_orf_filter = [
                    path for path in (ORF_FILTERED_AESEPs_PEP, ORF_FINAL_TABLE)
                    if not os.path.exists(path) or os.path.getsize(path) == 0
                ]
                if missing_orf_filter:
                    raise ValueError(
                        "cryptic_core_qc requires orf_filter to be enabled, or existing "
                        "08-orf_filter outputs to be present: "
                        + ", ".join(missing_orf_filter)
                    )
                binding_pep_fasta = ORF_FILTERED_AESEPs_PEP
            candidate_selection = configure.get("candidate_selection", {}) or {}
            cryptic_core_policy_version = str(
                others.get("cryptic_core_qc_policy_version", "cryptic_core_qc_v1.0")
            ).strip()
            human_proteome_fasta = str(others.get("human_reference_proteome_fasta", "") or "").strip()
            if not human_proteome_fasta:
                human_proteome_fasta = str(
                    paths.get("database", {})
                    .get("common", {})
                    .get("HUMAN_PROTEOME", {})
                    .get("CANONICAL_FASTA", "")
                    or ""
                ).strip()
            cmd = [
                sys.executable, _script_path("cryptic_core_qc.py"),
                "-s", tumor_sample,
                "--policy-version", cryptic_core_policy_version,
                "--matched-control-sample", ctrl_sample or "",
                "--ae-seps-fasta", AESEPs_PEP,
                "--aeseps-annotation", ABERRANT_TABLE,
                "--orf-filtered-fasta", ORF_FILTERED_AESEPs_PEP,
                "--orf-final", ORF_FINAL_TABLE,
                "-o", DIR08B_CORE,
                "--reference-genome-fasta", REF_GENOME,
                "--reference-gtf", REF_GTF,
                "--reference-lnc-gtf", REF_LNC_GTF,
                "--reference-build", str(others.get("reference_build", "GRCh38")),
                "--strandedness", strandedness,
                "--min-tpm-tumor", str(min_tpm_tumor),
                "--max-tpm-ctrl", str(max_tpm_ctrl),
                "--min-log2fc", str(min_log2fc),
                "--mhc-i-lengths", str(others.get("mhcI_lengths", "8,9,10,11")),
                "--mhc-ii-lengths", str(others.get("mhcII_lengths", "13,14,15,16,17")),
                "--candidate-selection-mode", str(candidate_selection.get("mode", "all")),
            ]
            if candidate_selection.get("max_hla_i_peptides") is not None:
                cmd.extend(["--max-hla-i-peptides", str(int(candidate_selection.get("max_hla_i_peptides")))])
            if candidate_selection.get("max_hla_ii_peptides") is not None:
                cmd.extend(["--max-hla-ii-peptides", str(int(candidate_selection.get("max_hla_ii_peptides")))])
            if human_proteome_fasta:
                cmd.extend(["--human-proteome-fasta", human_proteome_fasta])
            if bool(others.get("allow_missing_human_reference", False)):
                cmd.append("--allow-missing-human-reference")
            if cryptic_core_policy_version == "cryptic_core_qc_v1.1":
                cmd.extend([
                    "--orf-bed12", ORF_BED12,
                    "--orf-bam", ORF_BAM,
                    "--orf-cds-fasta", ORF_CDS_FASTA,
                    "--coordinate-min-mapq", str(int(others.get("coordinate_min_mapq", 20))),
                ])
            junction_qc = configure.get("junction_qc", {}) or {}
            if not isinstance(junction_qc, dict):
                raise ValueError("junction_qc must be a mapping")
            if bool(junction_qc.get("enabled", False)):
                star_pair_inputs = _resolve_star_pair_inputs_for_junction_qc(
                    tool=tool,
                    run_sample_id=sample,
                    junction_qc=junction_qc,
                    cryptic_root=out_root,
                    dir01=DIR01,
                    tumor_sample=tumor_sample,
                    ctrl_sample=ctrl_sample,
                    star_index=STAR_GENOME_DIR,
                )
                cmd.extend([
                    "--junction-qc-enabled",
                    "--junction-policy-version", str(junction_qc.get("policy_version", "junction_qc_v1.0")),
                    "--star-pair-inputs", star_pair_inputs,
                    "--primary-min-tumor-unique-reads",
                    str(int(junction_qc.get("primary_min_tumor_unique_reads", 2))),
                    "--junction-sensitivity-thresholds",
                    str(junction_qc.get("sensitivity_thresholds", "1,2,3,5")),
                ])
            rna_variant_qc = configure.get("rna_variant_editing_qc", {}) or {}
            if not isinstance(rna_variant_qc, dict):
                raise ValueError("rna_variant_editing_qc must be a mapping")
            if bool(rna_variant_qc.get("enabled", False)):
                rna_variant_vcf = str(rna_variant_qc.get("rna_variant_vcf", "") or "").strip()
                if not rna_variant_vcf:
                    rna_variant_vcf = os.path.join(DIR02_KNOWN, "04k.bcf_consensus", "rna.flt.vcf.gz")
                rna_variant_calling_manifest = str(rna_variant_qc.get("rna_variant_calling_manifest", "") or "").strip()
                if not rna_variant_calling_manifest:
                    rna_variant_calling_manifest = os.path.join(
                        os.path.dirname(rna_variant_vcf),
                        "rna.variant_calling.manifest.json",
                    )
                rediportal_table = str(rna_variant_qc.get("rediportal_processed_table", "") or "").strip()
                if not rediportal_table:
                    rediportal_table = str(
                        paths.get("database", {})
                        .get("cryptic", {})
                        .get("RNA_VARIANT_EDITING_QC", {})
                        .get("REDIPORTAL_PROCESSED_TABLE", "")
                        or ""
                    ).strip()
                rediportal_manifest = str(rna_variant_qc.get("rediportal_resource_manifest", "") or "").strip()
                if not rediportal_manifest:
                    rediportal_manifest = str(
                        paths.get("database", {})
                        .get("cryptic", {})
                        .get("RNA_VARIANT_EDITING_QC", {})
                        .get("REDIPORTAL_RESOURCE_MANIFEST", "")
                        or ""
                    ).strip()
                allow_missing_rediportal = bool(rna_variant_qc.get("allow_missing_rediportal_resource", False))
                allow_legacy_rna_variant_vcf = bool(rna_variant_qc.get("allow_legacy_rna_variant_vcf", False))
                allow_legacy_duplicate_vcf = bool(rna_variant_qc.get("allow_legacy_duplicate_vcf", False))
                if do_pvacbind and allow_missing_rediportal:
                    raise ValueError(
                        "allow_missing_rediportal_resource is exploratory only; "
                        "disable hla_binding_pred or provide a formal REDIportal resource"
                    )
                if do_pvacbind and allow_legacy_rna_variant_vcf:
                    raise ValueError(
                        "allow_legacy_rna_variant_vcf is exploratory only; "
                        "disable hla_binding_pred or provide a formal RNA variant calling manifest"
                    )
                if do_pvacbind and allow_legacy_duplicate_vcf:
                    raise ValueError(
                        "allow_legacy_duplicate_vcf is exploratory only; "
                        "disable hla_binding_pred or provide a normalized duplicate-free RNA VCF"
                    )
                cmd.extend([
                    "--rna-variant-editing-qc-enabled",
                    "--rna-variant-qc-policy-version",
                    str(rna_variant_qc.get("policy_version", "cryptic_rna_variant_editing_qc_v1.0")),
                    "--rna-variant-vcf",
                    rna_variant_vcf,
                    "--rna-variant-calling-manifest",
                    rna_variant_calling_manifest,
                    "--rna-variant-min-mapping-quality",
                    str(float(rna_variant_qc.get("min_read_mapping_quality", 20))),
                    "--rna-variant-min-base-quality",
                    str(float(rna_variant_qc.get("min_base_quality", 20))),
                    "--rna-variant-min-variant-qual",
                    str(float(rna_variant_qc.get("min_variant_qual", 30))),
                    "--rna-variant-min-total-depth",
                    str(int(rna_variant_qc.get("min_total_depth", 10))),
                    "--rna-variant-min-variant-allele-fraction",
                    str(float(rna_variant_qc.get("min_variant_allele_fraction", 0.05))),
                    "--rna-variant-primary-min-alt-reads",
                    str(int(rna_variant_qc.get("primary_min_alt_reads", 3))),
                    "--rna-variant-sensitivity-alt-reads",
                    str(rna_variant_qc.get("sensitivity_alt_reads", "2,3,5")),
                ])
                if rediportal_table:
                    cmd.extend(["--rediportal-processed-table", rediportal_table])
                if rediportal_manifest:
                    cmd.extend(["--rediportal-resource-manifest", rediportal_manifest])
                if allow_missing_rediportal:
                    cmd.append("--allow-missing-rediportal-resource")
                if allow_legacy_rna_variant_vcf:
                    cmd.append("--allow-legacy-rna-variant-vcf")
                if allow_legacy_duplicate_vcf:
                    cmd.append("--allow-legacy-duplicate-vcf")
            _run_cmd(tool, sample, cmd, display_name="Cryptic Core QC")
            binding_pep_fasta = CRYPTIC_CORE_FASTA
            binding_input_mode = "peptide-core"

        if do_external_normal:
            if not do_cryptic_core:
                raise ValueError("cryptic_external_normal_qc requires cryptic_core_qc to be enabled")
            if allow_missing_external_normal and do_pvacbind:
                raise ValueError(
                    "allow_missing_external_normal_resources is exploratory only; "
                    "disable hla_binding_pred or provide formal external-normal resources"
                )
            external_resources = configure.get("external_normal_resources", {}) or {}
            if not isinstance(external_resources, dict):
                raise ValueError("external_normal_resources must be a mapping")
            external_policy_version = str(
                external_resources.get("policy_version", "cryptic_external_normal_qc_v1.0")
            ).strip()
            cmd = [
                sys.executable, _script_path("cryptic_external_normal_qc.py"),
                "-s", tumor_sample,
                "--policy-version", external_policy_version,
                "--cryptic-peptide-core", CRYPTIC_CORE_TSV,
                "--cryptic-peptide-parent-map", CRYPTIC_CORE_PARENT_MAP,
                "--cryptic-parent-core", CRYPTIC_PARENT_CORE_TSV,
                "--upstream-manifest", CRYPTIC_CORE_MANIFEST,
                "--human-proteome-fasta", human_proteome_fasta,
                "-o", DIR08C_EXTERNAL,
                "--resource-manifest", _external_normal_resource_value(configure, paths, "manifest", "MANIFEST"),
                "--smorf-match-index", _external_normal_resource_value(
                    configure, paths, "smorf_match_index", "SMORF_MATCH_INDEX"
                ),
                "--smorf-parent-map", _external_normal_resource_value(
                    configure, paths, "smorf_parent_map", "SMORF_PARENT_MAP"
                ),
                "--hla-ligand-match-index", _external_normal_resource_value(
                    configure, paths, "hla_ligand_match_index", "HLA_LIGAND_MATCH_INDEX"
                ),
                "--hla-ligand-evidence", _external_normal_resource_value(
                    configure, paths, "hla_ligand_evidence", "HLA_LIGAND_EVIDENCE"
                ),
            ]
            candidate_selection = configure.get("candidate_selection", {}) or {}
            if str(candidate_selection.get("mode", "all")).strip().lower() == "ranked_cap":
                cmd.extend([
                    "--cryptic-peptide-deferred", CRYPTIC_DEFERRED_PEPTIDE,
                    "--cryptic-peptide-deferred-parent-map", CRYPTIC_DEFERRED_PARENT_MAP,
                    "--cryptic-parent-ranked", CRYPTIC_PARENT_RANKED,
                    "--max-hla-i-peptides", str(int(candidate_selection.get("max_hla_i_peptides"))),
                    "--max-hla-ii-peptides", str(int(candidate_selection.get("max_hla_ii_peptides"))),
                ])
            if allow_missing_external_normal:
                cmd.append("--allow-missing-external-normal-resources")
            if bool(external_resources.get("coordinate_matching_enabled", False)):
                cmd.append("--coordinate-matching-enabled")
            if external_policy_version == "cryptic_external_normal_qc_v1.1":
                cmd.extend([
                    "--cryptic-parent-coordinates", CRYPTIC_PARENT_COORDINATES,
                    "--cryptic-parent-orfcds", CRYPTIC_PARENT_ORFCDS,
                    "--cryptic-peptide-genomic-footprint", CRYPTIC_PEPTIDE_FOOTPRINT,
                    "--cryptic-parent-junction-summary", CRYPTIC_PARENT_JUNCTION_SUMMARY,
                    "--cryptic-peptide-junction-evidence", CRYPTIC_PEPTIDE_JUNCTION_EVIDENCE,
                    "--coordinate-resource-manifest", _external_normal_resource_value(
                        configure, paths, "coordinate_manifest", "COORDINATE_MANIFEST"
                    ),
                    "--normal-smorf-coordinates", _external_normal_resource_value(
                        configure, paths, "smorf_coordinates", "SMORF_COORDINATES"
                    ),
                    "--normal-smorf-orfcds", _external_normal_resource_value(
                        configure, paths, "smorf_orfcds", "SMORF_ORFCDS"
                    ),
                ])
            _run_cmd(tool, sample, cmd, display_name="External normal resource QC")
            binding_pep_fasta = CRYPTIC_PRIMARY_CORE_FASTA
            binding_input_mode = "peptide-core"
            if do_pvacbind:
                _validate_peptide_core_binding_manifest(CRYPTIC_EXTERNAL_MANIFEST, binding_pep_fasta)

        # ---------- 09 HLA binding prediction ----------
        binding_output_dir = ""
        if do_pvacbind:
            backend = str(others.get("binding_prediction_backend", "pvactools")).strip().lower()
            e1_lengths = others.get("mhcI_lengths", "8,9,10")
            e2_lengths = others.get("mhcII_lengths", "15")
            if backend == "pvactools":
                if binding_input_mode == "peptide-core":
                    raise ValueError(
                        "cryptic_core_qc peptide-core input requires binding_prediction_backend: mimicneoai"
                    )
                algos = others.get(
                    "algo",
                    "BigMHC_EL BigMHC_IM DeepImmuno MHCflurry MHCflurryEL MHCnuggetsI "
                    "MHCnuggetsII NNalign NetMHC NetMHCIIpan NetMHCIIpanEL NetMHCpan "
                    "NetMHCpanEL PickPocket SMM SMMPMBEC",
                )
                _run_cmd(tool, sample, [
                    sys.executable, _script_path("07-hla_binding_pred.py"),
                    "-s", tumor_sample,
                    "--pep-fasta", binding_pep_fasta,
                    "--hla-file", HLA_FINAL_TXT,
                    "-o", DIR07,
                    "--pvactools", PVACTOOLS_SIF,
                    "-t", str(n_pvacbind),
                    "--algos", algos,
                    "--e1-lengths", e1_lengths,
                    "--e2-lengths", e2_lengths,
                ], display_name="pVACtools cryptic binding prediction")
                binding_output_dir = DIR07
            elif backend == "mimicneoai":
                algos = others.get(
                    "binding_prediction_algorithms",
                    "MHCflurry MHCflurryEL MHCnuggetsI MHCnuggetsII NNalign "
                    "NetMHCpan NetMHCpanEL NetMHCIIpan NetMHCIIpanEL",
                )
                binding_step_value = others.get(
                    "binding_prediction_step_name",
                    configure.get("step_name", {}).get("mimicneoai_binding", STEP_NAME["mimicneoai_binding"]),
                )
                binding_step = str(binding_step_value).strip()
                if not binding_step or Path(binding_step).name != binding_step:
                    raise ValueError("binding_prediction_step_name must be a single directory name")
                outdir_mimicneoai = os.path.join(OPT, binding_step)
                binding_output_dir = outdir_mimicneoai
                cmd = [
                    sys.executable, _script_path("07-hla_binding_pred_mimicneoai.py"),
                    "-s", tumor_sample,
                    "--pep-fasta", binding_pep_fasta,
                    "--input-mode", binding_input_mode,
                    "--hla-file", HLA_FINAL_TXT,
                    "-o", outdir_mimicneoai,
                    "-t", str(int(others.get("binding_prediction_workers", n_pvacbind))),
                    "--algorithms", algos,
                    "--mhc-i-lengths", e1_lengths,
                    "--mhc-ii-lengths", e2_lengths,
                    "--max-task-rows", str(int(others.get("binding_prediction_max_task_rows", 5000000))),
                ]
                if bool(others.get("binding_prediction_force_large_samples", False)):
                    cmd.append("--force-large-samples")
                preset = str(others.get("binding_prediction_preset", "")).strip()
                if preset:
                    cmd.extend(["--preset", preset])
                cmd.extend(configured_predictor_cli_args(paths))
                _run_cmd(tool, sample, cmd, display_name="MimicNeoAI cryptic binding prediction")
            else:
                raise ValueError(f"Unsupported binding_prediction_backend: {backend}")

        if do_pvacbind and bool(others.get("run_immunogenicity_prediction", False)):
            immunogenicity_step = str(
                others.get(
                    "immunogenicity_step_name",
                    configure.get("step_name", {}).get("immunogenicity", STEP_NAME["immunogenicity"]),
                )
            ).strip()
            if not immunogenicity_step or Path(immunogenicity_step).name != immunogenicity_step:
                raise ValueError("immunogenicity_step_name must be a single directory name")
            immunogenicity_outdir = os.path.join(OPT, immunogenicity_step)
            model_root = resolve_immunogenicity_model_root(configure, paths)
            cmd = [
                resolve_immunogenicity_python_bin(configure, paths),
                "-m",
                "mimicneoai.functions.immunogenicity_workflow",
                "-s",
                tumor_sample,
                "--antigen-class",
                "cryptic",
                "--binding-dir",
                binding_output_dir,
                "-o",
                immunogenicity_outdir,
                "--device",
                str(others.get("immunogenicity_device", "auto")),
                "--batch-size",
                str(int(others.get("immunogenicity_batch_size", 512))),
                "--workers",
                str(int(others.get("immunogenicity_workers", configure.get("args", {}).get("threads", n_pvacbind)))),
            ]
            if model_root:
                cmd.extend(["--model-root", model_root])
            _run_cmd(tool, sample, cmd, display_name="MimicNeoAI cryptic immunogenicity prediction")

        tool.write_log(f"[DONE] Completed cryptic pipeline: {OPT}", "info")

    except Exception:
        tool.write_log(f"Worker error for sample '{sample}':\n{traceback.format_exc()}", "error")
        raise


# -------- Non-daemon Pool (optional: needed only if you spawn pools inside workers) --------
class _NoDaemonProcess(multiprocessing.Process):
    """A Process whose daemon attribute is always False (allows child processes)."""
    # For Python < 3.8
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


if hasattr(multiprocessing, "get_start_method"):
    # Python ≥ 3.8: override Process constructor on Pool to use _NoDaemonProcess
    class NoDaemonPool(multiprocessing.pool.Pool):
        @staticmethod
        def Process(_, *args, **kwargs):
            return _NoDaemonProcess(*args, **kwargs)
else:
    # Python < 3.8 fallback
    class NoDaemonPool(multiprocessing.pool.Pool):
        Process = _NoDaemonProcess


# ---------------------- Multiprocess runner ----------------------
def _run_pool(samples: List[str], pool_size: int, configure, paths, tool: tools) -> None:
    """
    Submit one asynchronous task per sample and wait for completion.
    Errors inside workers are reported via tool.print_pool_error.
    """
    async_results = []
    with NoDaemonPool(processes=pool_size) as pool:
        for s in samples:
            async_results.append(
                (
                    s,
                    pool.apply_async(
                        _run_one_sample,
                        (s, configure, paths, tool),
                        error_callback=tool.print_pool_error,
                    ),
                )
            )
        pool.close()
        pool.join()
    raise_for_failed_samples(async_results)


def _peek_output_dir(cfg_path: str) -> str | None:
    """
    Quickly peek 'configure' YAML to obtain path.output_dir for choosing a working directory.
    Return None on any failure; do not raise.
    """
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        out_dir = (data.get("path") or {}).get("output_dir")
        if not out_dir:
            return None
        return str(Path(out_dir).expanduser())
    except Exception:
        return None


# ---------------------- CLI entry ----------------------
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MimicNeoAI (Cryptic pipeline)")
    parser.add_argument("-c", "--configure", required=True, help="Path to configuration YAML")
    parser.add_argument("-p", "--paths", dest="paths", required=False, help="Path to paths YAML")
    parser.add_argument("--workdir", default=None, help="Working directory (default: config.path.output_dir or CWD)")
    args = parser.parse_args(argv)

    # 1) Decide working/log root: --workdir > configure.path.output_dir > CWD
    cfg_output_dir = _peek_output_dir(args.configure)
    base_out = Path(cfg_output_dir)
    cfg_output_dir = str(base_out / "Cryptic")
    workdir = args.workdir or cfg_output_dir or os.getcwd()
    workdir = str(Path(workdir).resolve())

    # Manager + tool/logger (log root = workdir)
    mgr = Manager()
    lock = mgr.Lock()
    tool_obj = tools(workdir, FLAG, lock)

    tool_obj.write_log(f"work_path: {tool_obj.sys_path}", "info")
    tool_obj.write_log(f"start_log: {tool_obj.start_log}", "info")
    tool_obj.write_log(f"cmd_log: {tool_obj.cmd_log_dir}", "info")
    if args.workdir is None and cfg_output_dir:
        tool_obj.write_log(f"workdir inferred from configure.path.output_dir: {cfg_output_dir}", "info")

    # Load YAMLs
    configure = tool_obj.get_configure(args.configure)
    paths = tool_obj.get_paths(args.paths)
    tool_obj.write_log(f"configures: {configure}", "info")
    tool_obj.write_log(f"paths: {paths}", "info")

    # Samples & pool
    samples = [str(s) for s in configure["samples"]]
    pool_size = int(configure.get("args", {}).get("pool_size", 1))

    # Inject step names (non-destructive)
    configure = dict(configure)
    configure.setdefault("step_name", {}).update(STEP_NAME)
    base_out = Path(configure["path"]["output_dir"])
    configure["path"]["output_dir"] = str(base_out / "Cryptic")

    # Share variables across processes
    tool_obj.sharing_variable(mgr, samples)

    # Run
    exit_code = 0
    try:
        _run_pool(samples, pool_size, configure, paths, tool_obj)
    except Exception:
        exit_code = 1
        tool_obj.write_log(
            f"Pipeline completed with failed sample(s):\n{traceback.format_exc()}",
            "error",
        )
    finally:
        tool_obj.summary()

    if tool_obj.has_failures():
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
