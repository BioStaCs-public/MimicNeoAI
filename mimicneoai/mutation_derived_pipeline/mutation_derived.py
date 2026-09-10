# coding=utf-8
"""
Launcher for the mutation-derived (neoantigen) pipeline:
- QC (fastp)
- Variant calling (WES/WGS/RNA where supported)
- Annotation (VEP)
- HLA typing
- Neoantigen identification & binding prediction
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import traceback
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional

from mimicneoai.functions.binding_prediction import configured_predictor_cli_args
from mimicneoai.functions.fastp import fastp
from mimicneoai.functions.hlatyping import hlahd
from mimicneoai.functions.immunogenicity_runner import (
    resolve_immunogenicity_model_root,
    resolve_immunogenicity_python_bin,
)
from mimicneoai.functions.pipline_tools import raise_for_failed_samples, tools
from mimicneoai.mutation_derived_pipeline.scripts.annotation import annotation_vcf
from mimicneoai.mutation_derived_pipeline.scripts.hla_binding_pred import Pvacseq
from mimicneoai.mutation_derived_pipeline.scripts.variants_calling import (
    variants_calling_start,
)
from mimicneoai.functions.nodemon_pool import NoDaemonPool


MIMICNEOAI_BINDING_SCRIPT = (
    Path(__file__).resolve().parent
    / "scripts"
    / "mutation_epitope_prediction"
    / "run_mimicneoai_binding_prediction.py"
)

# -------- Constants --------
FLAG = "Neoantigen"
STEP_NAME = {
    "QC": "00.QC",
    "alignment": "01.alignment",
    "markdup": "02.markdup",
    "bqsr": "03.bqsr",
    "variants_calling": "04.variants_calling",
    "annotation": "05.annotation",
    "hla": "06.hlatyping",
    "pvacseq": "07.binding_prediction",
    "mimicneoai_binding": "07.binding_prediction_mimicneoai",
    "immunogenicity": "08.immunogenicity_prediction_mimicneoai",
}


def _common_tool_path(paths: Dict[str, Any], key: str, default: str) -> str:
    """Return a common runtime tool path from paths.yaml with a command-name fallback."""

    common = (paths.get("path", {}) or {}).get("common", {}) or {}
    return str(common.get(key, default))


# -------- Worker helpers --------
def _variants_calling_and_annotation(
    sample: str,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
) -> None:
    """Run variant calling and optional annotation for a single sample."""
    host_variants_calling = configure["others"]["host_variants_calling"]
    do_annotation = configure["others"]["annotation"]
    species = configure["others"]["species"]  # e.g., "human"
    seq_type = configure["others"]["seq_type"]  # e.g., "wes"
    tumor_with_matched_normal = configure["others"]["tumor_with_matched_normal"]

    # Dispatch to the appropriate variant-calling entry point
    if host_variants_calling:
        funcs = {
            "wes_human": variants_calling_start,
            # Extend here as more modalities/species are supported:
            # "wgs_human": wgs_human_mutation_calling_start,
            # "rna_human": rna_mutation_calling_start,
        }
        key = f"{seq_type}_{species}"
        func = funcs.get(key)
        if func is None:
            raise ValueError(f"No variant-calling function for key='{key}'")
        func(sample, tool, configure, paths)

    # VEP-based annotation
    if do_annotation:
        if tumor_with_matched_normal:
            tumor_sample = sample.split(",")[0]
            _ = annotation_vcf(sample, tumor_sample, tool, paths, configure)
        else:
            tool.write_log(
                f"tumor_with_matched_normal must be True. Got False for sample='{sample}'. "
                f"Skip annotation.",
                "error",
            )


def _run_mutation_immunogenicity(
    sample: str,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
    binding_outdir: str,
) -> None:
    """Run source-specific immunogenicity scoring from a completed binding directory."""
    others = configure.get("others", {})
    output_dir = configure["path"]["output_dir"]
    tumor_sample = sample.split(",")[0]
    immunogenicity_step = str(
        others.get(
            "immunogenicity_step_name",
            configure.get("step_name", {}).get("immunogenicity", STEP_NAME["immunogenicity"]),
        )
    ).strip()
    if not immunogenicity_step or Path(immunogenicity_step).name != immunogenicity_step:
        raise ValueError("immunogenicity_step_name must be a single directory name")
    immunogenicity_outdir = f"{output_dir}/{tumor_sample}/{immunogenicity_step}"
    model_root = resolve_immunogenicity_model_root(configure, paths)
    cmd = [
        resolve_immunogenicity_python_bin(configure, paths),
        "-m",
        "mimicneoai.functions.immunogenicity_workflow",
        "-s",
        tumor_sample,
        "--antigen-class",
        "mutation_derived",
        "--binding-dir",
        binding_outdir,
        "-o",
        immunogenicity_outdir,
        "--device",
        str(others.get("immunogenicity_device", "auto")),
        "--batch-size",
        str(int(others.get("immunogenicity_batch_size", 512))),
        "--workers",
        str(int(others.get(
            "immunogenicity_workers",
            configure.get("args", {}).get("threads", configure.get("args", {}).get("thread", 1)),
        ))),
    ]
    if model_root:
        cmd.extend(["--model-root", model_root])
    tool.exec_cmd(
        " ".join(shlex.quote(item) for item in cmd),
        sample,
        pipline="mutation",
        display_name="MimicNeoAI mutation immunogenicity prediction",
    )


def _start_one_sample(
    sample: str,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
) -> None:
    """End-to-end per-sample workflow: QC → variants/annotation → HLA → peptide identification & binding prediction."""
    try:
        sample = str(sample)

        # Runtime toggles
        do_qc = configure["others"]["QC"]
        do_hlatyping = configure["others"]["hlatyping"]
        do_binding_pred = configure["others"]["peptides_identification_and_binding_prediction"]
        tumor_with_matched_normal = configure["others"]["tumor_with_matched_normal"]

        # HARD REQUIREMENT
        if not tumor_with_matched_normal:
            tool.write_log(
                f"tumor_with_matched_normal must be True (tumor,normal paired). "
                f"Got False for sample='{sample}'. Worker will exit.",
                "error",
            )
            raise ValueError(
                f"tumor_with_matched_normal must be True for sample '{sample}'"
            )

        # 1) Read QC
        if do_qc:
            tumor_sample, normal_sample = sample.split(",")[0], sample.split(",")[1]
            fastp(sample, tumor_sample, configure, paths, tool)
            fastp(sample, normal_sample, configure, paths, tool)

        # 2) Variant calling + annotation
        _variants_calling_and_annotation(sample, configure, paths, tool)

        # 3) HLA typing
        if do_hlatyping:
            tumor_sample = sample.split(",")[0]
            hlahd(sample, tumor_sample, configure, paths, tool)

        # 4) Peptide identification & binding prediction
        binding_outdir = ""
        if do_binding_pred:
            output_dir = configure["path"]["output_dir"]
            step_name_vep = configure["step_name"]["annotation"]
            step_name_hla = configure["step_name"]["hla"]

            tumor_sample = sample.split(",")[0]
            output_vep = f"{output_dir}/{tumor_sample}/{step_name_vep}/"
            output_hla = f"{output_dir}/{tumor_sample}/{step_name_hla}/"
            backend = str(configure.get("others", {}).get("binding_prediction_backend", "mimicneoai")).strip().lower()
            binding_outdir = ""
            if backend == "pvactools":
                pvacseq_step = configure.get("step_name", {}).get(
                    "pvacseq",
                    STEP_NAME["pvacseq"],
                )
                binding_outdir = f"{output_dir}/{tumor_sample}/{pvacseq_step}"
                binding_pred_runner = Pvacseq(tool)
                binding_pred_runner.run_pvacseq_parallel(
                    sample, tumor_sample, output_vep, output_hla, configure, paths
                )
            elif backend == "mimicneoai":
                others = configure.get("others", {})
                input_vcf = f"{output_vep}/{tumor_sample}.shared.VEP.rm_mismatch.vcf"
                hla_file = f"{output_hla}/{tumor_sample}/result/{tumor_sample}_final.result.txt"
                apptainer_bin = str(
                    others.get("apptainer", _common_tool_path(paths, "APPTAINER_BIN", "apptainer"))
                )
                bcftools_bin = str(
                    others.get("bcftools", _common_tool_path(paths, "BCFTOOLS_BIN", "bcftools"))
                )
                tabix_bin = str(
                    others.get("tabix", _common_tool_path(paths, "TABIX_BIN", "tabix"))
                )
                binding_step_value = others.get("binding_prediction_step_name")
                if binding_step_value is None:
                    binding_step_value = configure.get("step_name", {}).get(
                        "mimicneoai_binding",
                        STEP_NAME["mimicneoai_binding"],
                    )
                binding_step = str(binding_step_value).strip()
                if not binding_step or Path(binding_step).name != binding_step:
                    raise ValueError(
                        "binding_prediction_step_name must be a single directory name"
                    )
                outdir = f"{output_dir}/{tumor_sample}/{binding_step}"
                binding_outdir = outdir
                cmd = [
                    sys.executable,
                    str(MIMICNEOAI_BINDING_SCRIPT),
                    "-s",
                    tumor_sample,
                    "--input-vcf",
                    input_vcf,
                    "--hla-file",
                    hla_file,
                    "--pvactools-sif",
                    str(paths["path"]["common"]["PVACTOOLS"]),
                    "-o",
                    outdir,
                    "--apptainer",
                    apptainer_bin,
                    "--bcftools",
                    bcftools_bin,
                    "--tabix",
                    tabix_bin,
                    "--mhc-i-lengths",
                    str(others.get("mhc_i_epitope_lengths", "8,9,10,11")),
                    "--mhc-ii-lengths",
                    str(others.get("mhc_ii_epitope_lengths", "13,14,15,16,17")),
                    "--algorithms",
                    str(others.get(
                        "binding_prediction_algorithms",
                        "MHCflurry MHCflurryEL MHCnuggetsI MHCnuggetsII "
                        "NetMHCpan NetMHCpanEL NetMHCIIpan NetMHCIIpanEL",
                    )),
                    "--workers",
                    str(int(others.get("binding_prediction_workers", configure.get("args", {}).get("hla_binding_threads", 5)))),
                ]
                cmd.extend(configured_predictor_cli_args(paths))
                preset = str(others.get("binding_prediction_preset", "fast")).strip()
                if preset:
                    cmd.extend(["--preset", preset])
                start_from = str(others.get("binding_prediction_start_from", "")).strip()
                if start_from:
                    cmd.extend(["--start-from", start_from])
                tool.exec_cmd(
                    " ".join(shlex.quote(item) for item in cmd),
                    sample,
                    pipline="mutation",
                    display_name="MimicNeoAI mutation binding prediction",
                )
            else:
                raise ValueError(f"Unsupported binding_prediction_backend: {backend}")

            if bool(configure.get("others", {}).get("run_immunogenicity_prediction", False)):
                _run_mutation_immunogenicity(sample, configure, paths, tool, binding_outdir)
        elif bool(configure.get("others", {}).get("run_immunogenicity_prediction", False)):
            output_dir = configure["path"]["output_dir"]
            tumor_sample = sample.split(",")[0]
            binding_step = str(
                configure.get("others", {}).get(
                    "binding_prediction_step_name",
                    configure.get("step_name", {}).get("mimicneoai_binding", STEP_NAME["mimicneoai_binding"]),
                )
            ).strip()
            if not binding_step or Path(binding_step).name != binding_step:
                raise ValueError("binding_prediction_step_name must be a single directory name")
            binding_outdir = f"{output_dir}/{tumor_sample}/{binding_step}"
            _run_mutation_immunogenicity(sample, configure, paths, tool, binding_outdir)

    except Exception:
        tool.write_log(f"Worker crashed:\n{traceback.format_exc()}", "error")
        raise


def _run_pipeline(
    samples: List[str],
    pool_size: int,
    configure: Dict[str, Any],
    paths: Dict[str, Any],
    tool: tools,
) -> None:
    """Execute the pipeline across samples using a non-daemon process pool."""
    async_results = []
    with NoDaemonPool(processes=pool_size) as pool:
        for sample in samples:
            async_results.append(
                (
                    sample,
                    pool.apply_async(
                        _start_one_sample,
                        (sample, configure, paths, tool),
                        error_callback=tool.print_pool_error,
                    ),
                )
            )
        pool.close()
        pool.join()
    raise_for_failed_samples(async_results)


def _peek_output_dir(cfg_path: str) -> Optional[str]:
    """Return configure.path.output_dir from YAML without fully parsing user config."""
    try:
        import yaml
    except Exception:
        return None
    try:
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
        return (data.get("path", {}) or {}).get("output_dir", None)
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
    """CLI entry point for the mutation-derived (neoantigen) pipeline."""
    parser = argparse.ArgumentParser(description="MimicNeoAI (Mutation-derived pipeline)")
    parser.add_argument(
        "-c", "--configure", required=True, help="Path to configuration YAML"
    )
    parser.add_argument(
        "-p", "--paths", dest="paths", required=False, help="Path to paths YAML"
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory root (default: config.path.output_dir or current directory)",
    )
    args = parser.parse_args(argv)

    # 1) Determine working directory precedence: --workdir > configure.path.output_dir > CWD
    cfg_output_dir = _peek_output_dir(args.configure)
    base_out = Path(cfg_output_dir)
    cfg_output_dir = str(base_out / "Mutation-derived")

    workdir = args.workdir or cfg_output_dir or os.getcwd()
    workdir = str(Path(workdir).resolve())

    # Initialize shared manager and logger/tool (log root = workdir)
    mgr = Manager()
    log_lock = mgr.Lock()
    tool_obj = tools(workdir, FLAG, log_lock)

    tool_obj.write_log(f"work_path: {tool_obj.sys_path}", "info")
    tool_obj.write_log(f"start_log: {tool_obj.start_log}", "info")
    tool_obj.write_log(f"cmd_log: {tool_obj.cmd_log_dir}", "info")
    if args.workdir is None and cfg_output_dir:
        tool_obj.write_log(
            f"workdir inferred from configure.path.output_dir: {cfg_output_dir}", "info"
        )

    # Load runtime configurations
    configure = tool_obj.get_configure(args.configure)
    paths = tool_obj.get_paths(args.paths)
    _prepare_runtime_directories(configure)
    tool_obj.write_log(f"configures: {configure}", "info")
    tool_obj.write_log(f"paths: {paths}", "info")

    # Sample list and step-name mapping
    samples = [str(s) for s in configure["samples"]]
    configure = dict(configure)  # shallow copy to avoid mutating the original
    configure.setdefault("step_name", {}).update(STEP_NAME)
    base_out = Path(configure["path"]["output_dir"])
    configure["path"]["output_dir"] = str(base_out / "Mutation-derived")

    # Share variables across processes
    tool_obj.sharing_variable(mgr, samples)

    pool_size = int(configure["args"]["pool_size"])
    exit_code = 0
    try:
        _run_pipeline(samples, pool_size, configure, paths, tool_obj)
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
    # Optional Windows support:
    # from multiprocessing import freeze_support; freeze_support()
    raise SystemExit(main())
