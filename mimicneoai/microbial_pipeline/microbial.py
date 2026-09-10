# coding=utf-8
from __future__ import annotations
import argparse
import os
import traceback
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from mimicneoai.functions.fastp import fastp
from mimicneoai.functions.pipline_tools import raise_for_failed_samples, tools
from mimicneoai.functions.hlatyping import hlahd
from mimicneoai.microbial_pipeline.scripts.microbial_peptides import (
    HostSequencesRemoving,
    VectorContaminationRemoving,
    MicrobialTaxasQuantification,
    MicrobialPeptidesIdentification,
    MicrobialPairedProteinCoreQC,
    MicrobialPeptidesBindingPrediction,
    MicrobialImmunogenicityPrediction,
)
# -------- Non-daemon Pool (only needed if workers spawn child processes) --------
from mimicneoai.functions.nodemon_pool import NoDaemonPool


FLAG = "MicrobialAntigen"
STEP_NAME = {
    "QC": "00.QC",
    "hg38": "01.HostSequencesRemovingStep1",
    "t2t": "02.HostSequencesRemovingStep2",
    "mm10": "02b.HostSequencesRemovingStep3",
    "vector": "03.VectorContaminationRemoving",
    "pathseq": "04.MicrobialTaxaQuantificationStep1",
    "nucleic": "05.MicrobialTaxaQuantificationStep2",
    "blastx": "06.MicrobialPeptidesIdentification",
    "paired_core": "06b.MicrobialProteinCoreQC_v1.0",
    "hla": "07.HlaTyping",
    "pvacbind": "08.MicrobialPeptidesBindingPrediction",
}


@dataclass(frozen=True)
class MicrobialSampleUnit:
    label: str
    tumor_sample: str
    normal_sample: Optional[str] = None


def _parse_sample_units(samples: List[str], tumor_with_matched_normal: bool) -> List[MicrobialSampleUnit]:
    """Parse config samples once into execution units."""

    units: List[MicrobialSampleUnit] = []
    if not tumor_with_matched_normal:
        for raw in samples:
            sample = str(raw).strip()
            if not sample:
                raise ValueError("Empty sample entry is not allowed")
            if "," in sample:
                raise ValueError(
                    "Sample entries with commas require others.tumor_with_matched_normal: true"
                )
            units.append(MicrobialSampleUnit(label=sample, tumor_sample=sample))
        return units

    seen_pairs: set[tuple[str, str]] = set()
    sample_roles: Dict[str, str] = {}
    seen_tumors: set[str] = set()
    seen_normals: set[str] = set()
    for raw in samples:
        parts = [part.strip() for part in str(raw).split(",")]
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError(
                f"Paired microbial sample must be formatted exactly as Tumor,Normal: {raw!r}"
            )
        tumor_sample, normal_sample = parts
        if tumor_sample == normal_sample:
            raise ValueError(f"Tumor and normal sample must be different: {raw!r}")
        pair = (tumor_sample, normal_sample)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate microbial Tumor,Normal pair: {raw!r}")
        if tumor_sample in seen_tumors:
            raise ValueError(f"Duplicate microbial tumor sample in paired mode: {tumor_sample}")
        if normal_sample in seen_normals:
            raise ValueError(f"Duplicate microbial normal sample in paired mode: {normal_sample}")
        for sample, role in ((tumor_sample, "tumor"), (normal_sample, "normal")):
            previous_role = sample_roles.get(sample)
            if previous_role and previous_role != role:
                raise ValueError(
                    f"Microbial sample role conflict for {sample}: {previous_role} and {role}"
                )
            sample_roles[sample] = role
        seen_pairs.add(pair)
        seen_tumors.add(tumor_sample)
        seen_normals.add(normal_sample)
        units.append(
            MicrobialSampleUnit(
                label=f"{tumor_sample},{normal_sample}",
                tumor_sample=tumor_sample,
                normal_sample=normal_sample,
            )
        )
    return units

def _start_one_sample(
    sample: str,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
) -> None:
    """Run the microbial pipeline for a single sample with per-step guards."""
    o = configure["others"]

    run_qc                   = o["QC"]
    run_host_depletion       = o["run_host_depletion"]
    run_vector_decontam      = o["run_vector_decontamination"]
    run_pathseq              = o["run_pathseq"]
    run_microbial_peptide_identification    = o["run_microbial_peptide_identification"]
    run_hla_typing           = o["run_hla_typing"]
    run_binding_prediction   = o["run_binding_prediction"]

    if run_qc:
        try:
            fastp(sample, sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"QC error:\n{traceback.format_exc()}", "error")
            raise

    if run_host_depletion:
        try:
            HostSequencesRemoving(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"HostSequencesRemoving error:\n{traceback.format_exc()}", "error")
            raise

    if run_vector_decontam:
        try:
            VectorContaminationRemoving(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"VectorContaminationRemoving error:\n{traceback.format_exc()}", "error")
            raise

    if run_pathseq:
        try:
            MicrobialTaxasQuantification(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"MicrobialTaxasQuantification error:\n{traceback.format_exc()}", "error")
            raise

    if run_microbial_peptide_identification:
        try:
            MicrobialPeptidesIdentification(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"MicrobialPeptidesIdentification error:\n{traceback.format_exc()}", "error")
            raise

    if run_hla_typing:
        try:
            hlahd(sample, sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"HLA typing error:\n{traceback.format_exc()}", "error")
            raise

    if run_binding_prediction:
        try:
            MicrobialPeptidesBindingPrediction(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"Binding prediction error:\n{traceback.format_exc()}", "error")
            raise
    elif bool(o.get("run_immunogenicity_prediction", False)):
        try:
            MicrobialImmunogenicityPrediction(sample, configure, tool, paths=paths)
        except Exception:
            tool.write_log(f"Immunogenicity prediction error:\n{traceback.format_exc()}", "error")
            raise


def _run_pre_binding_steps_for_sample(
    sample: str,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
) -> None:
    """Run microbial upstream steps through protein-hit identification for one sample."""

    o = configure["others"]
    if o["QC"]:
        try:
            fastp(sample, sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"QC error:\n{traceback.format_exc()}", "error")
            raise
    if o["run_host_depletion"]:
        try:
            HostSequencesRemoving(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"HostSequencesRemoving error:\n{traceback.format_exc()}", "error")
            raise
    if o["run_vector_decontamination"]:
        try:
            VectorContaminationRemoving(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"VectorContaminationRemoving error:\n{traceback.format_exc()}", "error")
            raise
    if o["run_pathseq"]:
        try:
            MicrobialTaxasQuantification(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"MicrobialTaxasQuantification error:\n{traceback.format_exc()}", "error")
            raise
    if o["run_microbial_peptide_identification"]:
        try:
            MicrobialPeptidesIdentification(sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"MicrobialPeptidesIdentification error:\n{traceback.format_exc()}", "error")
            raise


def _start_one_pair(
    unit: MicrobialSampleUnit,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
) -> None:
    """Run a paired Tumor,Normal microbial task unit."""

    if not unit.normal_sample:
        raise ValueError(f"Paired microbial task is missing normal sample: {unit}")

    o = configure["others"]
    tumor_sample = unit.tumor_sample
    normal_sample = unit.normal_sample

    _run_pre_binding_steps_for_sample(tumor_sample, configure, paths, tool)
    _run_pre_binding_steps_for_sample(normal_sample, configure, paths, tool)

    if o["run_hla_typing"]:
        try:
            hlahd(tumor_sample, tumor_sample, configure, paths, tool)
        except Exception:
            tool.write_log(f"HLA typing error:\n{traceback.format_exc()}", "error")
            raise

    run_core_qc = (
        bool(o.get("run_paired_core_qc", False))
        or bool(o["run_binding_prediction"])
        or bool(o.get("run_immunogenicity_prediction", False))
    )
    paired_core = None
    if run_core_qc:
        try:
            paired_core = MicrobialPairedProteinCoreQC(
                tumor_sample,
                normal_sample,
                configure,
                paths,
                tool,
            )
        except Exception:
            tool.write_log(f"Paired microbial Core QC error:\n{traceback.format_exc()}", "error")
            raise

    if o["run_binding_prediction"]:
        backend = str(o.get("binding_prediction_backend", "mimicneoai")).strip().lower()
        if backend != "mimicneoai":
            raise ValueError("Paired microbial mode requires binding_prediction_backend: mimicneoai")
        try:
            if paired_core is None:
                raise RuntimeError("Paired microbial Core QC did not return a Core FASTA for binding.")
            if paired_core.get("scale_gate_skipped"):
                raise RuntimeError(
                    "Paired microbial Core QC was skipped by the scale gate; "
                    "increase paired_core_max_estimated_peptide_windows or set it to 0 before binding."
                )
            MicrobialPeptidesBindingPrediction(
                tumor_sample,
                configure,
                paths,
                tool,
                peptide_fa=paired_core["core_fasta"],
                input_mode="peptide-core",
                run_sample_id=unit.label,
            )
        except Exception:
            tool.write_log(f"Paired binding prediction error:\n{traceback.format_exc()}", "error")
            raise
    elif bool(o.get("run_immunogenicity_prediction", False)):
        try:
            MicrobialImmunogenicityPrediction(tumor_sample, configure, tool, run_sample_id=unit.label, paths=paths)
        except Exception:
            tool.write_log(f"Immunogenicity prediction error:\n{traceback.format_exc()}", "error")
            raise


def _run_pipeline(sample_units: List[MicrobialSampleUnit], pool_size: int, configure, paths, tool: tools) -> None:
    """Run the pipeline across samples using a (non-daemon) process pool."""
    async_results = []
    with NoDaemonPool(pool_size) as pool:
        for unit in sample_units:
            target = _start_one_pair if unit.normal_sample else _start_one_sample
            args = (unit, configure, paths, tool) if unit.normal_sample else (unit.tumor_sample, configure, paths, tool)
            async_results.append(
                (
                    unit.label,
                    pool.apply_async(
                        target,
                        args,
                        error_callback=tool.print_pool_error,
                    ),
                )
            )
        pool.close()
        pool.join()
    raise_for_failed_samples(async_results)


def _peek_output_dir(cfg_path: str) -> Optional[str]:
    """Quickly read YAML to fetch path.output_dir; return None if missing or on error."""
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("path", {}).get("output_dir")
    except Exception:
        return None


def _prepare_runtime_directories(configure: Dict[str, Any]) -> None:
    """Create and validate writable runtime directories from the user config."""

    tmp_value = str(configure.get("path", {}).get("tmp_dir", "")).strip()
    if not tmp_value:
        raise ValueError("configure.path.tmp_dir must be set")

    tmp_dir = Path(tmp_value).expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if not tmp_dir.is_dir():
        raise NotADirectoryError(f"Configured tmp_dir is not a directory: {tmp_dir}")
    if not os.access(tmp_dir, os.R_OK | os.W_OK | os.X_OK):
        raise PermissionError(f"Configured tmp_dir is not readable and writable: {tmp_dir}")
    configure["path"]["tmp_dir"] = str(tmp_dir)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the microbial antigen pipeline."""
    parser = argparse.ArgumentParser(description="Microbial Antigen pipeline")
    parser.add_argument("-c", "--configure", required=True, help="Path to configuration YAML")
    parser.add_argument("-p", "--paths", dest="paths", required=False, help="Path to paths YAML")
    parser.add_argument(
        "--workdir", default=None,
        help="Working directory (default: config.path.output_dir if set, else CWD)"
    )
    args = parser.parse_args(argv)

    # 1) Decide working/log root directory with precedence:
    #    --workdir (explicit) > configure.path.output_dir > current working directory
    cfg_output_dir = _peek_output_dir(args.configure)
    base_out = Path(cfg_output_dir)
    cfg_output_dir = str(base_out / "Microbial")
    workdir = args.workdir or cfg_output_dir or os.getcwd()
    workdir = str(Path(workdir).resolve())

    # 2) Initialize logger/tools (log root = workdir)
    mgr = Manager()
    log_lock = mgr.Lock()
    tool = tools(workdir, FLAG, log_lock)

    tool.write_log(f"work_path: {tool.sys_path}", "info")
    tool.write_log(f"start_log: {tool.start_log}", "info")
    tool.write_log(f"cmd_log: {tool.cmd_log_dir}", "info")
    if args.workdir is None and cfg_output_dir:
        tool.write_log(f"workdir inferred from configure.path.output_dir: {cfg_output_dir}", "info")

    # 3) Load configuration and paths (tools.get_configure also persists a copy for reproducibility)
    configure = tool.get_configure(args.configure)
    paths = tool.get_paths(args.paths)
    _prepare_runtime_directories(configure)
    tool.write_log(f"configures: {configure}", "info")
    tool.write_log(f"paths: {paths}", "info")

    # 4) Fill in step names and collect sample list
    samples = list(configure.get("samples", []))
    configure = dict(configure)  # avoid mutating the original object
    configure.setdefault("step_name", {}).update(STEP_NAME)
    base_out = Path(configure["path"]["output_dir"])
    configure["path"]["output_dir"] = str(base_out / "Microbial")
    sample_units = _parse_sample_units(
        [str(sample) for sample in samples],
        bool(configure.get("others", {}).get("tumor_with_matched_normal", False)),
    )

    # 5) Initialize shared state & execute in parallel
    tool.sharing_variable(mgr, [unit.label for unit in sample_units])
    pool_size = int(configure["args"]["pool_size"])
    exit_code = 0
    try:
        _run_pipeline(sample_units, pool_size, configure, paths, tool)
    except Exception:
        exit_code = 1
        tool.write_log(
            f"Pipeline completed with failed sample(s):\n{traceback.format_exc()}",
            "error",
        )
    finally:
        tool.summary()

    if tool.has_failures():
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
