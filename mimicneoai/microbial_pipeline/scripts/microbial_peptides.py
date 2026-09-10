# coding=utf-8
import os
import json
import shlex
import sys
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from mimicneoai.functions.binding_prediction import configured_predictor_cli_args
from mimicneoai.functions.immunogenicity_runner import (
    resolve_immunogenicity_model_root,
    resolve_immunogenicity_python_bin,
)
from mimicneoai.functions.utils import format_java_heap
import pandas as pd
from mimicneoai.microbial_pipeline.scripts.get_data_for_blastx import get_data
from mimicneoai.microbial_pipeline.scripts.get_data_for_binding_pred import (
    PROTEIN_HIT_QC_POLICY_VERSION,
    get_data_for_binding_pred,
)
from mimicneoai.microbial_pipeline.scripts.hla_binding_pred import pvacbind


def _script_path(rel_name: str) -> str:
    pkg_path = files("mimicneoai.microbial_pipeline.scripts")
    return str(pkg_path / rel_name)


DEFAULT_FAST_BINDING_ALGORITHMS = (
    "MHCflurry MHCflurryEL MHCnuggetsI MHCnuggetsII "
    "NetMHCpan NetMHCpanEL NetMHCIIpan NetMHCIIpanEL"
)


def _microbial_binding_dir(sample, configure):
    output_path = configure['path']['output_dir'] + "/"
    others = configure.get("others", {})
    binding_step = str(others.get(
        "binding_prediction_step_name",
        "08.MicrobialPeptidesBindingPrediction_mimicneoai",
    )).strip()
    if not binding_step or Path(binding_step).name != binding_step:
        raise ValueError("binding_prediction_step_name must be a single directory name")
    return output_path + f"{sample}/{binding_step}/"


def _paired_core_step_name(configure):
    step_name = str(configure.get("others", {}).get(
        "paired_core_step_name",
        "06b.MicrobialProteinCoreQC_v1.0",
    )).strip()
    if not step_name or Path(step_name).name != step_name:
        raise ValueError("paired_core_step_name must be a single directory name")
    return step_name


def _microbial_protein_hits_path(sample, configure):
    output_path = configure['path']['output_dir'].rstrip("/") + "/"
    step_name_blastx = configure['step_name']['blastx']
    output_blastx = output_path + f"{sample}/{step_name_blastx}/"
    preferred = f"{output_blastx}{sample}.protein_hits.filtered.tsv"
    legacy = f"{output_blastx}{sample}.blastx.filtered"
    if os.path.exists(preferred):
        return preferred
    if os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(
        f"No microbial protein-hit table found for {sample}: expected {preferred} or {legacy}"
    )


def _resolve_contaminant_blacklist(paths):
    try:
        value = paths["database"]["microbial"]["BLACKLISTS"]["CONTAMINANT_TAXIDS"]
    except KeyError:
        return ""
    return str(value).strip()


def MicrobialPairedProteinCoreQC(tumor_sample, normal_sample, configure, paths, tool):
    """Build tumor-only peptide Core after matched-normal peptide subtraction."""

    output_path = configure['path']['output_dir'].rstrip("/") + "/"
    others = configure.get("others", {})
    candidate_selection = configure.get("candidate_selection", {}) or {}
    scan_workers = int(configure.get("args", {}).get("thread", 1))
    outdir = output_path + f"{tumor_sample}/{_paired_core_step_name(configure)}/"
    tumor_hits = _microbial_protein_hits_path(tumor_sample, configure)
    normal_hits = _microbial_protein_hits_path(normal_sample, configure)

    cmd = [
        sys.executable,
        _script_path("paired_protein_core_qc.py"),
        "--pair-id",
        f"{tumor_sample},{normal_sample}",
        "--tumor-sample",
        str(tumor_sample),
        "--normal-sample",
        str(normal_sample),
        "--tumor-protein-hits",
        tumor_hits,
        "--normal-protein-hits",
        normal_hits,
        "-o",
        outdir,
        "--mhc-i-lengths",
        str(others.get("mhcI_lengths", "8,9,10,11")),
        "--mhc-ii-lengths",
        str(others.get("mhcII_lengths", "13,14,15,16,17")),
        "--min-pident",
        str(float(others.get("blastx_min_percent_identity", 100))),
        "--max-evalue",
        str(float(others.get("blastx_max_evalue", 1e-5))),
        "--min-qcovs",
        str(float(others.get("blastx_min_query_coverage", 90))),
        "--max-estimated-peptide-windows",
        str(int(others.get("paired_core_max_estimated_peptide_windows", 20_000_000))),
        "--candidate-selection-mode",
        str(candidate_selection.get("mode", "all")),
        "--scan-workers",
        str(scan_workers),
    ]
    if candidate_selection.get("max_hla_i_peptides") is not None:
        cmd.extend(["--max-hla-i-peptides", str(int(candidate_selection.get("max_hla_i_peptides")))])
    if candidate_selection.get("max_hla_ii_peptides") is not None:
        cmd.extend(["--max-hla-ii-peptides", str(int(candidate_selection.get("max_hla_ii_peptides")))])
    if candidate_selection.get("ranking_abundance_pseudocount") is not None:
        cmd.extend([
            "--ranking-abundance-pseudocount",
            str(float(candidate_selection.get("ranking_abundance_pseudocount"))),
        ])
    blacklist = _resolve_contaminant_blacklist(paths)
    allow_missing_blacklist = bool(others.get("allow_missing_blacklist", False))
    if blacklist:
        cmd.extend(["--blacklist", blacklist])
    elif not allow_missing_blacklist:
        raise ValueError(
            "Paired microbial Core QC requires database.microbial.BLACKLISTS.CONTAMINANT_TAXIDS. "
            "Set others.allow_missing_blacklist: true only for exploratory runs."
        )
    if allow_missing_blacklist:
        cmd.append("--allow-missing-blacklist")
    expected_blacklist_sha256 = str(others.get("microbial_contaminant_blacklist_sha256", "")).strip()
    if expected_blacklist_sha256:
        cmd.extend(["--blacklist-sha256", expected_blacklist_sha256])
    manifest_path = f"{outdir}run_manifest.json"
    tool.exec_cmd(
        " ".join(shlex.quote(item) for item in cmd),
        f"{tumor_sample},{normal_sample}",
        display_name="Paired microbial Core QC",
    )

    core_fasta = f"{outdir}microbial_peptide_core.fasta"
    if not os.path.exists(core_fasta):
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as handle:
                manifest = json.load(handle)
        if manifest.get("scale_gate_skipped") or manifest.get("run_status") == "scale_gate_skipped":
            return {
                "outdir": outdir,
                "core_fasta": "",
                "scale_gate_skipped": True,
                "manifest": manifest_path,
            }
        raise FileNotFoundError(f"Paired microbial Core FASTA was not created: {core_fasta}")
    return {"outdir": outdir, "core_fasta": core_fasta, "scale_gate_skipped": False, "manifest": manifest_path}


def MicrobialImmunogenicityPrediction(
    sample,
    configure,
    tool,
    binding_output_dir=None,
    run_sample_id=None,
    paths=None,
):
    """Run microbial immunogenicity scoring from an existing binding directory."""
    output_path = configure['path']['output_dir'] + "/"
    others = configure.get("others", {})
    binding_output_dir = binding_output_dir or _microbial_binding_dir(sample, configure)
    immunogenicity_step = str(
        others.get(
            "immunogenicity_step_name",
            "09.ImmunogenicityPrediction_mimicneoai",
        )
    ).strip()
    if not immunogenicity_step or Path(immunogenicity_step).name != immunogenicity_step:
        raise ValueError("immunogenicity_step_name must be a single directory name")
    immunogenicity_outdir = output_path + f"{sample}/{immunogenicity_step}/"
    model_root = resolve_immunogenicity_model_root(configure, paths)
    cmd = [
        resolve_immunogenicity_python_bin(configure, paths),
        "-m",
        "mimicneoai.functions.immunogenicity_workflow",
        "-s",
        sample,
        "--antigen-class",
        "microbial",
        "--binding-dir",
        binding_output_dir,
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
        run_sample_id or sample,
        pipline="microbial",
        display_name="MimicNeoAI microbial immunogenicity prediction",
    )


def HostSequencesRemoving(sample, configure, paths, tool):
    """
    Remove host sequences by aligning reads to hg38 then T2T, and collecting unmapped reads.
    Optionally run an extra mm10 depletion stage after T2T.

    Pipeline (paired-end):
      FASTQ -> bwa mem hg38 -> name-sorted BAM -> extract read-unmapped (-f 4)
            -> FASTQ (paired R1/R2 forwarded; singleton/anomalous saved for audit)
            -> bwa mem T2T  -> name-sorted BAM -> extract read-unmapped (-f 4)
            -> final unmapped BAM
      If configured:
            -> bwa mem mm10 -> name-sorted BAM -> extract read-unmapped (-f 4)
            -> final unmapped BAM
      Always: samtools flagstat on final unmapped BAM.
    """
    sample = str(sample)

    # ------------------------------------------------------------------
    # 0) Read configs / references
    # ------------------------------------------------------------------
    input_root  = configure["path"]["input_dir"].rstrip("/")   # raw/QC input root
    output_root = configure["path"]["output_dir"].rstrip("/")  # pipeline output root

    thread       = int(configure["args"]["thread"])
    mem_perthread = configure["args"]["mem_perthread"]

    pair     = bool(configure["others"]["pair"])
    seq_type = str(configure["others"]["seq_type"])
    QC       = bool(configure["others"]["QC"])
    remove_mouse_host = bool(configure["others"].get("remove_mouse_host", False))

    host_fa_hg38 = paths["database"]["microbial"]["HOST"]["HG38"]["FA"]
    host_fa_t2t  = paths["database"]["microbial"]["HOST"]["T2T"]["FA"]
    host_fa_mm10 = paths["database"]["microbial"]["HOST"]["MM10"]["FA"] if remove_mouse_host else None

    step_qc   = configure["step_name"]["QC"]
    step_hg38 = configure["step_name"]["hg38"]
    step_t2t  = configure["step_name"]["t2t"]
    step_mm10 = configure["step_name"].get("mm10", "02b.HostSequencesRemovingStep3")

    # ------------------------------------------------------------------
    # 1) Utilities (centralize command construction)
    # ------------------------------------------------------------------
    def _mkdir(p: str, display_name: str = "Prepare output directory"):
        tool.judge_then_exec(sample, f"mkdir -p {p}", p, display_name=display_name)

    def _rg() -> str:
        # Note: keep exact escaping required by bwa
        return f"@RG\\tID:{sample}\\tSM:{sample}"

    # def _unmapped_flag(is_paired: bool) -> int:
    #     # paired: 0x4 (read unmapped) + 0x8 (mate unmapped) = 12, strict both ends unmapped
    #     return 12 if is_paired else 4

    def _unmapped_flag() -> int:
        # Always keep reads where the read itself is unmapped (0x4)
        return 4

    def _flagstat_cmd(bam: str) -> str:
        return f"samtools flagstat -@ {thread} {bam} > {bam}.flagstat.txt"

    def _fastq_inputs():
        """
        Return FASTQ input paths depending on QC/pair setting.
        Keeps your existing directory convention (note the extra {sample}/).
        """
        if QC:
            qc_dir = f"{output_root}/{sample}/{step_qc}".rstrip("/")
            # Your original code uses: {output_qc}/{sample}/{sample}.QC.R1.fq.gz
            base = f"{qc_dir}/{sample}/{sample}.QC"
        else:
            base = f"{input_root}/{sample}/{sample}"

        if pair:
            return (f"{base}.R1.fq.gz", f"{base}.R2.fq.gz")
        else:
            return (f"{base}.fq.gz",)

    # ------------------------------------------------------------------
    # 2) Paths (grouped, predictable)
    # ------------------------------------------------------------------
    out_hg38_dir = f"{output_root}/{sample}/{step_hg38}/"
    out_t2t_dir  = f"{output_root}/{sample}/{step_t2t}/"
    out_mm10_dir = f"{output_root}/{sample}/{step_mm10}/"

    # hg38 intermediate
    hg38_name_bam   = f"{out_hg38_dir}{sample}_{seq_type}.hg38.name_sorted.bam"
    hg38_unmap_bam  = f"{out_hg38_dir}{sample}_{seq_type}.hg38.unmapped.bam"

    # hg38 unmapped fastqs
    hg38_unmap_r1   = f"{out_hg38_dir}{sample}_{seq_type}.hg38.unmapped.R1.fq"
    hg38_unmap_r2   = f"{out_hg38_dir}{sample}_{seq_type}.hg38.unmapped.R2.fq"
    hg38_unmap_anom    = f"{out_hg38_dir}{sample}_{seq_type}.hg38.unmapped.anomalous.fq"   # -0
    hg38_unmap_single  = f"{out_hg38_dir}{sample}_{seq_type}.hg38.unmapped.singleton.fq"   # -s (NEW)
    hg38_unmap_se   = f"{out_hg38_dir}{sample}_{seq_type}.hg38.unmapped.fq"

    # T2T outputs
    t2t_name_bam    = f"{out_t2t_dir}{sample}_{seq_type}.hg38unmap.t2t.name_sorted.bam"
    t2t_unmap_bam   = f"{out_t2t_dir}{sample}_{seq_type}.hg38unmap.t2t.unmapped.bam"

    # mm10 outputs (written under dedicated step directory)
    t2t_unmap_for_mm10_r1   = f"{out_mm10_dir}{sample}_{seq_type}.hg38unmap.t2tunmap.mm10.input.R1.fq"
    t2t_unmap_for_mm10_r2   = f"{out_mm10_dir}{sample}_{seq_type}.hg38unmap.t2tunmap.mm10.input.R2.fq"
    t2t_unmap_for_mm10_anom = f"{out_mm10_dir}{sample}_{seq_type}.hg38unmap.t2tunmap.mm10.input.anomalous.fq"
    t2t_unmap_for_mm10_single = f"{out_mm10_dir}{sample}_{seq_type}.hg38unmap.t2tunmap.mm10.input.singleton.fq"
    t2t_unmap_for_mm10_se   = f"{out_mm10_dir}{sample}_{seq_type}.hg38unmap.t2tunmap.mm10.input.fq"
    mm10_name_bam   = f"{out_mm10_dir}{sample}_{seq_type}.hg38unmap.t2tunmap.mm10.name_sorted.bam"
    mm10_final_unmap_bam = (
        f"{out_mm10_dir}{sample}_{seq_type}.hg38unmap.t2tunmap.mm10unmap.unmapped.bam"
    )

    final_unmap_bam = mm10_final_unmap_bam if remove_mouse_host else t2t_unmap_bam

    _mkdir(out_hg38_dir, display_name="Prepare hg38 host depletion directory")
    _mkdir(out_t2t_dir, display_name="Prepare T2T host depletion directory")
    if remove_mouse_host:
        _mkdir(out_mm10_dir, display_name="Prepare mouse host depletion directory")

    # ------------------------------------------------------------------
    # 3) Decide whether to run full pipeline
    # ------------------------------------------------------------------
    need_run = (not os.path.exists(final_unmap_bam)) or (os.path.getsize(final_unmap_bam) == 0)

    # ------------------------------------------------------------------
    # 4) hg38 stage: FASTQ -> hg38 align -> name-sort -> extract unmapped -> FASTQ
    # ------------------------------------------------------------------
    if need_run:
        fastqs_in = _fastq_inputs()

        # 4.1–4.3: name-sorted BAM
        fq_part = " ".join(fastqs_in)
        cmd_align_sort_hg38 = (
            f"bwa mem -q -t {thread} "
            f"-R '{_rg()}' "
            f"{host_fa_hg38} {fq_part} | "
            f"samtools view -b -@ {thread} - | "
            f"samtools sort -n -@ {thread} -m {mem_perthread} "
            f"-o {hg38_name_bam} -"
        )
        tool.judge_then_exec(sample, cmd_align_sort_hg38, hg38_name_bam, display_name="Host depletion against hg38")

        # 4.4 stats on name-sorted BAM
        tool.judge_then_exec(sample, _flagstat_cmd(hg38_name_bam), f"{hg38_name_bam}.flagstat.txt", display_name="hg38 host alignment QC")


        # 4.5 extract unmapped
        flag = _unmapped_flag()
        cmd_extract_hg38_unmap = (
            f"samtools view -b -@ {thread} -f {flag} "
            f"-o {hg38_unmap_bam} {hg38_name_bam}"
        )
        tool.judge_then_exec(sample, cmd_extract_hg38_unmap, hg38_unmap_bam, display_name="Extract hg38-unmapped reads")

        # 4.6 BAM -> FASTQ(s) for T2T
        if pair:
            # -1/-2: proper pairs
            # -0: anomalous/other reads (saved for audit; not forwarded to T2T)
            # -s: singleton reads (saved for audit; not forwarded to T2T)
            cmd_bam2fq = (
                f"samtools fastq -@ {thread} {hg38_unmap_bam} "
                f"-1 {hg38_unmap_r1} "
                f"-2 {hg38_unmap_r2} "
                f"-0 {hg38_unmap_anom} "
                f"-s {hg38_unmap_single}"
            )
            tool.judge_then_exec(sample, cmd_bam2fq, hg38_unmap_r1, display_name="Convert hg38-unmapped BAM to FASTQ")
        else:
            cmd_bam2fq = f"samtools fastq -@ {thread} {hg38_unmap_bam} > {hg38_unmap_se}"
            tool.judge_then_exec(sample, cmd_bam2fq, hg38_unmap_se, display_name="Convert hg38-unmapped BAM to FASTQ")

        # ------------------------------------------------------------------
        # 5) T2T stage: hg38-unmapped FASTQ -> T2T align+name-sort -> extract unmapped
        # ------------------------------------------------------------------
        if pair:
            t2t_fastqs = (hg38_unmap_r1, hg38_unmap_r2)
        else:
            t2t_fastqs = (hg38_unmap_se,)

        fq_t2t_part = " ".join(t2t_fastqs)
        cmd_align_sort_t2t = (
            f"bwa mem -q -t {thread} "
            f"-R '{_rg()}' "
            f"{host_fa_t2t} {fq_t2t_part} | "
            f"samtools view -b -@ {thread} - | "
            f"samtools sort -n -@ {thread} -m {mem_perthread} "
            f"-o {t2t_name_bam} -"
        )
        tool.judge_then_exec(sample, cmd_align_sort_t2t, t2t_name_bam, display_name="Host depletion against T2T")


        t2t_flag = _unmapped_flag()
        cmd_extract_t2t_unmap = (
            f"samtools view -b -@ {thread} -f {t2t_flag} "
            f"-o {t2t_unmap_bam} {t2t_name_bam}"
        )
        tool.judge_then_exec(sample, cmd_extract_t2t_unmap, t2t_unmap_bam, display_name="Extract T2T-unmapped reads")

        if remove_mouse_host:
            if pair:
                cmd_t2t_bam2fq = (
                    f"samtools fastq -@ {thread} {t2t_unmap_bam} "
                    f"-1 {t2t_unmap_for_mm10_r1} "
                    f"-2 {t2t_unmap_for_mm10_r2} "
                    f"-0 {t2t_unmap_for_mm10_anom} "
                    f"-s {t2t_unmap_for_mm10_single}"
                )
                tool.judge_then_exec(sample, cmd_t2t_bam2fq, t2t_unmap_for_mm10_r1, display_name="Convert T2T-unmapped BAM to FASTQ")
                mm10_fastqs = (t2t_unmap_for_mm10_r1, t2t_unmap_for_mm10_r2)
            else:
                cmd_t2t_bam2fq = f"samtools fastq -@ {thread} {t2t_unmap_bam} > {t2t_unmap_for_mm10_se}"
                tool.judge_then_exec(sample, cmd_t2t_bam2fq, t2t_unmap_for_mm10_se, display_name="Convert T2T-unmapped BAM to FASTQ")
                mm10_fastqs = (t2t_unmap_for_mm10_se,)

            fq_mm10_part = " ".join(mm10_fastqs)
            cmd_align_sort_mm10 = (
                f"bwa mem -q -t {thread} "
                f"-R '{_rg()}' "
                f"{host_fa_mm10} {fq_mm10_part} | "
                f"samtools view -b -@ {thread} - | "
                f"samtools sort -n -@ {thread} -m {mem_perthread} "
                f"-o {mm10_name_bam} -"
            )
            tool.judge_then_exec(sample, cmd_align_sort_mm10, mm10_name_bam, display_name="Host depletion against mouse")

            mm10_flag = _unmapped_flag()
            cmd_extract_mm10_unmap = (
                f"samtools view -b -@ {thread} -f {mm10_flag} "
                f"-o {final_unmap_bam} {mm10_name_bam}"
            )
            tool.judge_then_exec(sample, cmd_extract_mm10_unmap, final_unmap_bam, display_name="Extract mouse-unmapped reads")

    # ------------------------------------------------------------------
    # 6) Always: stats for final output
    # ------------------------------------------------------------------
    tool.judge_then_exec(sample, _flagstat_cmd(final_unmap_bam), f"{final_unmap_bam}.flagstat.txt", display_name="Host-depleted read QC")

def VectorContaminationRemoving(
    sample: str,
    configure: dict,
    paths: dict,
    tool,
):
    """
    Remove vector contamination (UniVec) from host-removed BAM.

    Input BAM (dynamic):
      - remove_mouse_host=False:
        <output_dir>/<sample>/<step_t2t>/<sample>_<seq_type>.hg38unmap.t2t.unmapped.bam
      - remove_mouse_host=True:
        <output_dir>/<sample>/<step_mm10>/<sample>_<seq_type>.hg38unmap.t2tunmap.mm10unmap.unmapped.bam

    Output:
      <output_dir>/<sample>/<step_vector>/
        ├── 01.fastq/
        │     ├── *.R1.fq.gz
        │     ├── *.R2.fq.gz
        │     ├── *.singleton.fq.gz
        │     └── *.cat0.fq.gz
        └── 02.bam/
              ├── *.paired.vector_unmapped.bam
              ├── *.single.vector_unmapped.bam
              └── *.vector_unmapped.merged.bam   (FINAL)
    """
    sample = str(sample)

    # ------------------------------------------------------------------
    # 0) Read configs
    # ------------------------------------------------------------------
    output_root = configure["path"]["output_dir"].rstrip("/")
    thread = int(configure["args"]["thread"])
    mem_perthread = configure["args"].get("mem_perthread", "1G")

    pair = bool(configure["others"]["pair"])
    seq_type = str(configure["others"]["seq_type"])
    remove_mouse_host = bool(configure["others"].get("remove_mouse_host", False))

    step_t2t    = configure["step_name"]["t2t"]
    step_mm10   = configure["step_name"].get("mm10", "02b.HostSequencesRemovingStep3")
    step_vector = configure["step_name"]["vector"]

    # ------------------------------------------------------------------
    # 1) Input BAM (from HostSequencesRemoving, based on remove_mouse_host)
    # ------------------------------------------------------------------
    if remove_mouse_host:
        host_step = step_mm10
        host_chain = "hg38unmap.t2tunmap.mm10unmap"
    else:
        host_step = step_t2t
        host_chain = "hg38unmap.t2t"

    in_bam = (
        f"{output_root}/{sample}/{host_step}/"
        f"{sample}_{seq_type}.{host_chain}.unmapped.bam"
    )

    # ------------------------------------------------------------------
    # 2) UniVec reference
    # ------------------------------------------------------------------
    univec_ref = paths["database"]["microbial"]["MICROBES"]["UniVec"]
    if os.path.isdir(univec_ref):
        vector_fa = os.path.join(univec_ref, "UniVec.fa")
    else:
        vector_fa = univec_ref

    # ------------------------------------------------------------------
    # 3) Output directories
    # ------------------------------------------------------------------
    out_dir = f"{output_root}/{sample}/{step_vector}/"
    fq_dir  = f"{out_dir}01.fastq/"
    bam_dir = f"{out_dir}02.bam/"

    tool.judge_then_exec(sample, f"mkdir -p {fq_dir}", fq_dir, display_name="Prepare vector FASTQ directory")
    tool.judge_then_exec(sample, f"mkdir -p {bam_dir}", bam_dir, display_name="Prepare vector BAM directory")

    # ------------------------------------------------------------------
    # 4) Filenames
    # ------------------------------------------------------------------
    r1_fq = f"{fq_dir}{sample}_{seq_type}.host_removed.R1.fq.gz"
    r2_fq = f"{fq_dir}{sample}_{seq_type}.host_removed.R2.fq.gz"
    se_fq = f"{fq_dir}{sample}_{seq_type}.host_removed.singleton.fq.gz"
    cat0_fq = f"{fq_dir}{sample}_{seq_type}.host_removed.cat0.fq.gz"

    pe_unmap_bam = f"{bam_dir}{sample}_{seq_type}.paired.vector_unmapped.bam"
    se_unmap_bam = f"{bam_dir}{sample}_{seq_type}.single.vector_unmapped.bam"
    merged_bam   = f"{bam_dir}{sample}_{seq_type}.vector_unmapped.merged.bam"

    # ------------------------------------------------------------------
    # 5) BAM -> FASTQ
    # ------------------------------------------------------------------
    if pair:
        cmd_bam2fq = (
            f"samtools sort -n -@ {thread} -m {mem_perthread} {in_bam} | "
            f"samtools fastq -@ {thread} - "
            f"-1 {r1_fq} -2 {r2_fq} "
            f"-s {se_fq} -0 {cat0_fq}"
        )
        tool.judge_then_exec(sample, cmd_bam2fq, r1_fq, display_name="Convert host-depleted BAM to FASTQ")
    else:
        cmd_bam2fq = (
            f"samtools sort -n -@ {thread} -m {mem_perthread} {in_bam} | "
            f"samtools fastq -@ {thread} - > {se_fq}"
        )
        tool.judge_then_exec(sample, cmd_bam2fq, se_fq, display_name="Convert host-depleted BAM to FASTQ")

    # ------------------------------------------------------------------
    # 6) BWA -> UniVec (PE)
    # ------------------------------------------------------------------
    pe_vec_bam = f"{bam_dir}{sample}_{seq_type}.paired.vector.bam"
    if pair:
        cmd_bwa_pe = (
            f"bwa mem -q -t {thread} "
            f"-R '@RG\\tID:{sample}\\tSM:{sample}' "
            f"{vector_fa} {r1_fq} {r2_fq} | "
            f"samtools view -b -@ {thread} -o {pe_vec_bam} -"
        )
        tool.judge_then_exec(sample, cmd_bwa_pe, pe_vec_bam, display_name="Vector contamination screening")

    # ------------------------------------------------------------------
    # 7) BWA -> UniVec (SE)
    # ------------------------------------------------------------------
    se_vec_bam = f"{bam_dir}{sample}_{seq_type}.single.vector.bam"
    cmd_bwa_se = (
        f"bwa mem -q -t {thread} "
        f"-R '@RG\\tID:{sample}_SE\\tSM:{sample}' "
        f"{vector_fa} {se_fq} | "
        f"samtools view -b -@ {thread} -o {se_vec_bam} -"
    )
    tool.judge_then_exec(sample, cmd_bwa_se, se_vec_bam, display_name="Vector contamination screening for singletons")

    # ------------------------------------------------------------------
    # 8) Extract vector-unmapped reads (统一 -f 4)
    # ------------------------------------------------------------------
    unmapped_flag = 4

    cmd_unmap_pe = (
        f"samtools view -b -@ {thread} -f {unmapped_flag} "
        f"-o {pe_unmap_bam} {pe_vec_bam}"
    )
    cmd_unmap_se = (
        f"samtools view -b -@ {thread} -f {unmapped_flag} "
        f"-o {se_unmap_bam} {se_vec_bam}"
    )

    if pair:
        tool.judge_then_exec(sample, cmd_unmap_pe, pe_unmap_bam, display_name="Extract paired vector-unmapped reads")
    tool.judge_then_exec(sample, cmd_unmap_se, se_unmap_bam, display_name="Extract singleton vector-unmapped reads")

    # ------------------------------------------------------------------
    # 9) Merge PE / SE unmapped BAMs
    # ------------------------------------------------------------------
    have_pe = os.path.exists(pe_unmap_bam) and os.path.getsize(pe_unmap_bam) > 0
    have_se = os.path.exists(se_unmap_bam) and os.path.getsize(se_unmap_bam) > 0

    if have_pe and have_se:
        cmd_merge = f"samtools merge -@ {thread} {merged_bam} {pe_unmap_bam} {se_unmap_bam}"
        tool.judge_then_exec(sample, cmd_merge, merged_bam, display_name="Merge vector-clean reads")
    elif have_pe:
        cmd_copy = f"samtools view -b {pe_unmap_bam} -o {merged_bam}"
        tool.judge_then_exec(sample, cmd_copy, merged_bam, display_name="Merge vector-clean reads")
    elif have_se:
        cmd_copy = f"samtools view -b {se_unmap_bam} -o {merged_bam}"
        tool.judge_then_exec(sample, cmd_copy, merged_bam, display_name="Merge vector-clean reads")
    else:
        tool.write_log(sample, f"[WARN] {sample}: no vector-unmapped reads.\n")

    # ------------------------------------------------------------------
    # 10) QC stats (optional but recommended)
    # ------------------------------------------------------------------
    if os.path.exists(merged_bam) and os.path.getsize(merged_bam) > 0:
        cmd_flagstat = (
            f"samtools flagstat -@ {thread} {merged_bam} "
            f"> {merged_bam}.flagstat.txt"
        )
        tool.judge_then_exec(sample, cmd_flagstat, f"{merged_bam}.flagstat.txt", display_name="Vector-clean read QC")



def MicrobialTaxasQuantification(sample, configure, paths, tool):
    """Quantify microbial taxa using GATK PathSeq and prepare inputs for downstream steps.

    Notes (updated):
      - Host removal (hg38 + T2T) has already been done upstream.
      - Vector contamination removal (UniVec) has already been done upstream.
      - Therefore PathSeq is run WITHOUT host filtering (--filter-bwa-image / --kmer-file removed),
        and takes the vector-clean merged BAM as input.
    """
    sample = str(sample)

    # ---- Configuration ----
    tmp_dir     = configure['path']['tmp_dir'].rstrip("/") + "/"
    output_path = configure['path']['output_dir'].rstrip("/") + "/"
    thread      = int(configure['args']['thread'])
    # PathSeqPipelineSpark 建议限制本地线程数，避免资源过度竞争
    pathseq_threads = min(thread, 5)

    mem = configure['args']['mem']
    # ---- Example usage ----
    _, xmx, xms = format_java_heap(configure["args"]["mem"], xms_ratio=0.5)

    pair     = configure['others']['pair']
    seq_type = configure['others']['seq_type']

    # ---- Reference databases ----
    microbe_dict  = paths['database']['microbial']['MICROBES']['DICT']
    microbe_img   = paths['database']['microbial']['MICROBES']['IMG']
    taxonomy_file = paths['database']['microbial']['MICROBES']['TAXONOMY_DB']

    # ---- Tools ----
    gatk_jar = paths['path']['common']['GATK_JAR']

    # ---- Step directories ----
    step_name_pathseq = configure['step_name']['pathseq']
    step_name_nucleic = configure['step_name']['nucleic']
    step_name_vector  = configure['step_name']['vector']  # NEW: vector step

    output_pathseq = output_path + f'{sample}/{step_name_pathseq}/'
    output_nucleic = output_path + f'{sample}/{step_name_nucleic}/'

    # Vector step BAM dir (matches your VectorContaminationRemoving layout)
    output_vector = output_path + f'{sample}/{step_name_vector}/'
    bam_dir = output_vector + "02.bam/"

    # ---- Input BAM: vector-unmapped merged BAM ----
    merged_bam = f"{bam_dir}{sample}_{seq_type}.vector_unmapped.merged.bam"

    # Ensure output directories exist
    tool.judge_then_exec(sample, f"mkdir -p {output_pathseq}", output_pathseq, display_name="Prepare microbial taxa directory")
    tool.judge_then_exec(sample, f"mkdir -p {output_nucleic}", output_nucleic, display_name="Prepare microbial read extraction directory")

    # Validate input BAM exists
    if not os.path.exists(merged_bam) or os.path.getsize(merged_bam) == 0:
        raise FileNotFoundError(
            f"[{sample}] Vector-clean merged BAM is missing/empty: {merged_bam}"
        )

    pathseq_bam     = f"{output_pathseq}{sample}_{seq_type}_output.pathseq.bam"
    scores_txt      = f"{output_pathseq}{sample}_{seq_type}_output.pathseq.txt"
    score_metrics   = f"{output_pathseq}{sample}_{seq_type}_score.metrics.txt"
    filter_metrics  = f"{output_pathseq}{sample}_{seq_type}_filter.metrics.txt"

    # ---- Run PathSeq (Spark local mode) ----
    # If PathSeq scores already exist, skip rerun to save time.
    if not os.path.exists(scores_txt) or os.path.getsize(scores_txt) == 0:
        cmd_pathseq = (
            f"java -Xms{xms} -Xmx{xmx} -jar {gatk_jar} PathSeqPipelineSpark "
            f"--spark-master local[{pathseq_threads}] "
            f"--tmp-dir {tmp_dir} "
            f"--input {merged_bam} "
            f"--min-clipped-read-length 50 "
            # REMOVED (host filtering already done upstream):
            # f"--filter-bwa-image {host_img} "
            # f"--kmer-file {host_kmer} "
            f"--microbe-dict {microbe_dict} "
            f"--microbe-bwa-image {microbe_img} "
            f"--taxonomy-file {taxonomy_file} "
            f"--output {pathseq_bam} "
            f"--scores-output {scores_txt} "
            f"--score-metrics {score_metrics} "
            f"--filter-metrics {filter_metrics} "
            f"--divide-by-genome-length true"
        )
        tool.judge_then_exec(sample, cmd_pathseq, scores_txt, display_name="Microbial taxa quantification")

    # === New behavior (unchanged) ===
    # Directly read PathSeq BAM and extract YP-tagged reads into a fasta.gz file,
    # plus a small summary TSV describing read counts.
    summary_tsv = os.path.join(output_nucleic, f"{sample}.pathseq_selected.summary.tsv")

    # Skip re-extraction if summary already exists (idempotent behavior).
    if not os.path.exists(summary_tsv) or os.path.getsize(summary_tsv) == 0:
        tool.write_log(f"[{sample}] Extracting YP-tagged reads from PathSeq BAM", "info")
        start = datetime.now()
        get_data(
            sample,
            output_pathseq=output_pathseq,
            output_nucleic=output_nucleic,
            pair=pair,
            seq_type=seq_type,
        )
        end = datetime.now()
        using_time = tool.print_time(end - start)
        tool.write_log(f"[{sample}] YP-tagged read extraction completed in {using_time}", "info")



def _parse_outfmt_fields(outfmt: str):
    outfmt_clean = str(outfmt).strip().strip("'\"")
    tokens = outfmt_clean.split()
    if not tokens or tokens[0] != "6":
        raise ValueError(f"OUTFMT must start with 6, got: {outfmt_clean}")
    return tokens[1:]


def _normalize_diamond_blastx_output(input_path: str, output_path: str, colnames, tool, sample: str):
    df = pd.read_csv(input_path, sep="\t", header=None, names=colnames)

    if "qcovhsp" not in df.columns:
        raise ValueError("DIAMOND output must contain qcovhsp for coverage normalization.")

    df["sseqid"] = (
        df["sseqid"]
        .astype(str)
        .str.replace(r"^ref\|", "", regex=True)
        .str.rstrip("|")
        .map(lambda x: f"ref|{x}|")
    )
    df["qcovs"] = df["qcovhsp"]

    pident = pd.to_numeric(df["pident"], errors="coerce")
    pident100 = pident == 100.0
    df.loc[pident100, "qseq"] = df.loc[pident100, "sseq"]

    blastx_colnames = [
        "qseqid", "qlen", "sseqid", "qseq", "sseq", "stitle", "pident",
        "length", "mismatch", "gapopen", "qstart", "qend", "sstart",
        "send", "evalue", "bitscore", "qcovhsp", "qcovs",
    ]
    missing = [c for c in blastx_colnames if c not in df.columns]
    if missing:
        raise ValueError(f"DIAMOND output missing required columns after normalization: {missing}")

    df[blastx_colnames].to_csv(output_path, sep="\t", header=False, index=False)
    tool.write_log(
        f"[{sample}] Normalized DIAMOND output for BLASTX-compatible filtering: {output_path}",
        "info",
    )


def MicrobialPeptidesIdentification(sample, configure, paths, tool):
    """Identify microbial peptides via BLASTX against a protein database.

    Args:
        sample (str): Sample ID.
        configure (dict): Pipeline configuration dictionary.
        paths (dict): Paths dictionary (references, tools).
        tool: Execution tool object.
    """
    output_path = configure['path']['output_dir'] + "/"
    step_name_nucleic = configure['step_name']['nucleic']
    step_name_blastx  = configure['step_name']['blastx']

    output_nucleic = output_path + f'{sample}/{step_name_nucleic}/'
    output_blastx  = output_path + f'{sample}/{step_name_blastx}/'

    thread = configure['args']['thread']


    # Protein search config
    search_engine = str(
        configure.get("others", {}).get("microbial_peptide_search_engine", "blastx")
    ).strip().lower()
    if search_engine not in {"blastx", "diamond"}:
        raise ValueError(
            f"Unsupported microbial_peptide_search_engine: {search_engine}. "
            "Use 'blastx' or 'diamond'."
        )

    blast_cfg = paths['database']['microbial']['BLAST']
    diamond_cfg = paths['database']['microbial'].get('DIAMOND', {})

    # Ensure output directory exists
    tool.judge_then_exec(sample, f"mkdir -p {output_blastx}", output_blastx, display_name="Prepare microbial peptide directory")

    # Run BLASTX on nucleic sequences produced earlier
    fa_gz = f"{output_nucleic}{sample}.pathseq_selected.fa.gz"
    fa_fa = f"{output_nucleic}{sample}.pathseq_selected.fa"
    blastx_out = f"{output_blastx}{sample}.blastx"
    diamond_out = f"{output_blastx}{sample}.diamond.blastx"
    normalized_diamond_out = f"{output_blastx}{sample}.normalized.diamond.blastx"

    # 1) Decompress the gzipped FASTA.
    #    This step uses exec_cmd instead of judge_then_exec because it is considered
    #    an internal preprocessing step and should not appear as the main “Running:” command.
    cmd_unzip = f"zcat {fa_gz} > {fa_fa}"
    tool.exec_cmd(cmd_unzip, sample, display_name="Prepare microbial peptide FASTA")

    # 2) Run protein search using the uncompressed FASTA.
    if search_engine == "diamond":
        diamond_db = diamond_cfg["DB"]
        diamond_outfmt = str(diamond_cfg["OUTFMT"]).strip().strip("'\"")
        cmd_search = (
            f"diamond blastx "
            f"-d {diamond_db} "
            f"-q {fa_fa} "
            f"-o {diamond_out} "
            f"-p {thread} "
            f"--outfmt {diamond_outfmt} "
        )
        search_out = diamond_out
        filter_input_file = f"{sample}.normalized.diamond.blastx"
    else:
        db_dir = blast_cfg['DB_DIR']
        outfmt = blast_cfg['OUTFMT']
        cmd_search = (
            f"blastx "
            f"-query {fa_fa} "
            f"-out {blastx_out} "
            f"-db {db_dir} "
            f"-outfmt {outfmt} "
            f"-num_threads {thread} "
        )
        search_out = blastx_out
        filter_input_file = f"{sample}.blastx"

    protein_search_label = "DIAMOND microbial protein search" if search_engine == "diamond" else "BLASTX microbial protein search"
    tool.judge_then_exec(sample, cmd_search, search_out, display_name=protein_search_label)

    if search_engine == "diamond" and (
        not os.path.exists(normalized_diamond_out)
        or os.path.getsize(normalized_diamond_out) == 0
    ):
        diamond_colnames = _parse_outfmt_fields(diamond_cfg["OUTFMT"])
        _normalize_diamond_blastx_output(
            input_path=diamond_out,
            output_path=normalized_diamond_out,
            colnames=diamond_colnames,
            tool=tool,
            sample=sample,
        )

    # 3) Optionally remove the temporary uncompressed FASTA to save disk space.
    #    The removal is performed safely via Python rather than `rm -f`,
    #    with additional checks to avoid accidental deletion of unexpected files.
    if os.path.exists(search_out) and os.path.getsize(search_out) > 0:
        if os.path.isfile(fa_fa):
            # Sanity check to ensure this is the expected temporary FASTA file.
            if fa_fa.endswith(".fa"):
                try:
                    os.remove(fa_fa)
                    tool.write_log(f"[{sample}] Removed temporary FASTA: {fa_fa}", "info")
                except OSError as e:
                    tool.write_log(
                        f"[{sample}] Failed to remove temporary FASTA {fa_fa}: {e}",
                        "warning",
                    )
            else:
                tool.write_log(
                    f"[{sample}] Skipped deletion of {fa_fa}: unexpected extension (not .fa)",
                    "warning",
                )

    # Load catalog table for validating protein search hits
    catalog_path = blast_cfg['CATALOG_PROT']
    catalog_df = pd.read_csv(catalog_path, sep="\t", header=None)
    catalog_df.columns = ['prot_id', 'tax_id']
    catalog_df["tax_id"] = catalog_df["tax_id"].astype(str)

    # Parse BLASTX-compatible column names from the configured outfmt string
    # The outfmt string has the form: '6 qseqid qlen sseqid ... qcovhsp qcovs'
    blast_colnames = _parse_outfmt_fields(blast_cfg['OUTFMT'])

    # Load BLASTX filtering thresholds from the configuration file
    blastx_min_pident = float(configure['others']['blastx_min_percent_identity'])
    blastx_max_evalue = float(configure['others']['blastx_max_evalue'])
    # use BLASTX-reported qcovs as the coverage filter
    blastx_min_qcovs = float(configure['others']['blastx_min_query_coverage'])

    # Convert protein-search hits into filtered protein-hit tables and a legacy FASTA.
    tool.write_log("[INFO] Running microbial protein-hit QC", "info")
    start = datetime.now()

    get_data_for_binding_pred(
        blast_file=filter_input_file,
        colnames=blast_colnames,
        pvacbind_file=f"{sample}.peptide.fasta",
        output_blastx=output_blastx,
        min_pident=blastx_min_pident,
        catalog_df=catalog_df,
        max_evalue=blastx_max_evalue,
        min_qcovs=blastx_min_qcovs,   # ← 新参数名
        policy_version=str(
            configure.get("others", {}).get(
                "protein_hit_qc_policy_version",
                PROTEIN_HIT_QC_POLICY_VERSION,
            )
        ),
        sample=sample,
        tool=tool,
    )

    end = datetime.now()
    tool.print_time(end - start)



def MicrobialPeptidesBindingPrediction(
    sample,
    configure,
    paths,
    tool,
    peptide_fa=None,
    input_mode="parent-fasta",
    run_sample_id=None,
):
    """Run peptide-MHC binding prediction on BLASTX-derived peptides.

    Args:
        sample (str): Sample ID.
        configure (dict): Pipeline configuration dictionary.
        paths (dict): Paths dictionary (references, tools).
        tool: Execution tool object.
    """
    output_path = configure['path']['output_dir'] + "/"
    step_name_blastx   = configure['step_name']['blastx']
    step_name_pvacbind = configure['step_name']['pvacbind']
    step_name_hla      = configure['step_name']['hla']

    output_blastx   = output_path + f'{sample}/{step_name_blastx}/'
    output_pvacbind = output_path + f'{sample}/{step_name_pvacbind}/'

    # Only run if peptide FASTA exists. Paired mode passes a Core FASTA explicitly.
    explicit_peptide_fa = peptide_fa is not None
    peptide_fa = peptide_fa or f"{output_blastx}/{sample}.peptide.fasta"
    if os.path.exists(peptide_fa):
        backend = str(configure.get("others", {}).get("binding_prediction_backend", "mimicneoai")).strip().lower()
        binding_output_dir = ""
        if backend == "pvactools":
            if input_mode != "parent-fasta":
                raise ValueError("pVACtools backend only supports parent-fasta microbial binding input.")
            tool.judge_then_exec(sample, f"mkdir -p {output_pvacbind}", output_pvacbind, display_name="Prepare pVACtools microbial binding directory")
            pvacbind(sample, configure, paths, tool)
            binding_output_dir = output_pvacbind
        elif backend == "mimicneoai":
            others = configure.get("others", {})
            binding_step = str(others.get(
                "binding_prediction_step_name",
                "08.MicrobialPeptidesBindingPrediction_mimicneoai",
            )).strip()
            if not binding_step or Path(binding_step).name != binding_step:
                raise ValueError("binding_prediction_step_name must be a single directory name")
            output_mimicneoai = output_path + f"{sample}/{binding_step}/"
            binding_output_dir = output_mimicneoai
            output_hla = output_path + f"{sample}/{step_name_hla}/"
            hla_file = f"{output_hla}{sample}/result/{sample}_final.result.txt"
            cmd = [
                sys.executable,
                _script_path("hla_binding_pred_mimicneoai.py"),
                "-s",
                sample,
                "--pep-fasta",
                peptide_fa,
                "--input-mode",
                input_mode,
                "--hla-file",
                hla_file,
                "-o",
                output_mimicneoai,
                "-t",
                str(int(others.get("binding_prediction_workers", configure.get("args", {}).get("hla_binding_threads", 5)))),
                "--algorithms",
                str(others.get(
                    "binding_prediction_algorithms",
                    DEFAULT_FAST_BINDING_ALGORITHMS,
                )),
                "--mhc-i-lengths",
                str(others.get("mhcI_lengths", "8,9,10,11")),
                "--mhc-ii-lengths",
                str(others.get("mhcII_lengths", "13,14,15,16,17")),
                "--max-task-rows",
                str(int(others.get("binding_prediction_max_task_rows", 5000000))),
            ]
            if bool(others.get("binding_prediction_force_large_samples", False)):
                cmd.append("--force-large-samples")
            preset = str(others.get("binding_prediction_preset", "fast")).strip()
            if preset:
                cmd.extend(["--preset", preset])
            cmd.extend(configured_predictor_cli_args(paths))
            tool.exec_cmd(
                " ".join(shlex.quote(item) for item in cmd),
                run_sample_id or sample,
                pipline="microbial",
                display_name="MimicNeoAI microbial binding prediction",
            )
        else:
            raise ValueError(f"Unsupported binding_prediction_backend: {backend}")

        if bool(configure.get("others", {}).get("run_immunogenicity_prediction", False)):
            MicrobialImmunogenicityPrediction(
                sample,
                configure,
                tool,
                binding_output_dir,
                run_sample_id=run_sample_id,
                paths=paths,
            )
    elif explicit_peptide_fa:
        raise FileNotFoundError(f"Explicit microbial binding FASTA not found: {peptide_fa}")
