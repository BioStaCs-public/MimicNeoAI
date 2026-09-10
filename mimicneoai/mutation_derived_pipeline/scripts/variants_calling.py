# coding=utf-8
# Human WES processing pipeline

"""
Pipeline wrapper for alignment and variant calling.

Required software versions (for reproducibility/documentation):
- bwa >= 0.7.17
- samtools >= 1.5
- GATK >= 4.6.0.0
- Java 17
"""
import os
import shlex
import sys
import traceback
from typing import Dict, Any
from mimicneoai.mutation_derived_pipeline.scripts.run_mutect2 import Mutect2
from mimicneoai.mutation_derived_pipeline.scripts.run_manta_strelka2 import strelka2_main
from mimicneoai.mutation_derived_pipeline.scripts.run_vardict import vardict_main
from mimicneoai.mutation_derived_pipeline.scripts.ensure_calling_bed import _ensure_calling_bed
from mimicneoai.mutation_derived_pipeline.scripts.merge_mutect_strelka_vardict import merge_mutect_strelka_vardict
from mimicneoai.functions.nodemon_pool import NoDaemonPool
from mimicneoai.functions.utils import format_java_heap



def mutation_calling(run_sample_id: str, sample_name: str, tool, configure: dict, paths: dict):
    """
    Align paired-end FASTQs to the reference genome and perform preprocessing:
      1) BWA-MEM alignment + samtools view + samtools sort (piped; no intermediate BAM)
      2) Mark duplicates (GATK MarkDuplicatesSpark)
      3) Base Quality Score Recalibration (BQSR) with interval padding

    Returns:
        (dir_variants_calling_root, bqsr_bam_path)
    """
    # --- References & tools ---
    ref = paths["database"]["neoantigen"]["HG38"]["REF_FASTA"]
    bundle_path = paths["database"]["neoantigen"]["HG38"]["BUNDLE"].rstrip("/") + "/"
    gatk_jar = paths["path"]["common"]["GATK_JAR"]

    # --- I/O & runtime config ---
    tmp_dir = configure["path"]["tmp_dir"].rstrip("/") + "/"
    input_dir = configure["path"]["input_dir"].rstrip("/") + "/"
    output_dir = configure["path"]["output_dir"].rstrip("/") + "/"
    bed_file = configure["others"]["bed_file"]
    QC = bool(configure["others"]["QC"])

    # --- Knobs ---
    pad_bp = int(configure["others"].get("pad_bp", 100))  # --interval-padding
    thread = int(configure["args"]["thread"])
    _, xmx, xms = format_java_heap(configure["args"]["mem"], xms_ratio=0.5)
    gatk = f"java -Xms{xms} -Xmx{xmx} -jar {gatk_jar}"

    # --- Step names ---
    step = configure["step_name"]
    step_qc = step["QC"]
    step_alignment = step["alignment"]
    step_markdup = step["markdup"]
    step_bqsr = step["bqsr"]
    step_varcall = step["variants_calling"]

    # --- Per-sample root ---
    sample_root = os.path.join(output_dir, sample_name) + "/"
    tool.judge_then_exec(run_sample_id, f"mkdir -p {sample_root}", sample_root)

    dir_alignment = os.path.join(sample_root, step_alignment) + "/"
    dir_markdup = os.path.join(sample_root, step_markdup) + "/"
    dir_bqsr = os.path.join(sample_root, step_bqsr) + "/"
    dir_varcall_root = os.path.join(sample_root, step_varcall) + "/"

    tool.judge_then_exec(run_sample_id, f"mkdir -p {dir_alignment}", dir_alignment)
    tool.judge_then_exec(run_sample_id, f"mkdir -p {dir_markdup}", dir_markdup)
    tool.judge_then_exec(run_sample_id, f"mkdir -p {dir_bqsr}", dir_bqsr)
    tool.judge_then_exec(run_sample_id, f"mkdir -p {dir_varcall_root}", dir_varcall_root)

    # --- Input FASTQs (QC or raw) ---
    if QC:
        # expected QC outputs: <sample_root>/00.QC/<sample>/<sample>.QC.R1.fq.gz
        qc_root = os.path.join(sample_root, step_qc) + "/"
        input_file1 = os.path.join(qc_root, sample_name, f"{sample_name}.QC.R1.fq.gz")
        input_file2 = os.path.join(qc_root, sample_name, f"{sample_name}.QC.R2.fq.gz")
    else:
        input_file1 = os.path.join(input_dir, sample_name, f"{sample_name}.R1.fq.gz")
        input_file2 = os.path.join(input_dir, sample_name, f"{sample_name}.R2.fq.gz")

    # --- 1) Alignment + Sort (pipe; produce sorted BAM directly) ---
    sorted_bam = os.path.join(dir_alignment, f"{sample_name}.sorted.bam")
    bwa_log = os.path.join(dir_alignment, f"{sample_name}.bwa.log")

    cmd_align_sort = (
        "bash -c \"set -euo pipefail; "
        "bwa mem -t {thread} -R '@RG\\tID:foo_lane\\tPL:UNKNOWN\\tLB:library\\tSM:{sm}' {ref} {fq1} {fq2} "
        "2> {bwa_log} "
        "| samtools view -S -b - "
        "| samtools sort -@ {thread} -m 1G -O bam -o {sorted_bam} -\""
    ).format(
        thread=thread,
        sm=sample_name,
        ref=ref,
        fq1=input_file1,
        fq2=input_file2,
        bwa_log=bwa_log,
        sorted_bam=sorted_bam,
    )
    tool.judge_then_exec(run_sample_id, cmd_align_sort, sorted_bam)

    sorted_bai = sorted_bam + ".bai"
    cmd_index_sorted = f"samtools index -@ {thread} {sorted_bam}"
    tool.judge_then_exec(run_sample_id, cmd_index_sorted, sorted_bai)

    # --- 2) Mark duplicates (MarkDuplicatesSpark) ---
    markdup_bam = os.path.join(dir_markdup, f"{sample_name}.sorted.markdup.bam")
    markdup_metrics = os.path.join(dir_markdup, f"{sample_name}.markdup_metrics.txt")

    cmd_markdup = (
        f"{gatk} MarkDuplicatesSpark --tmp-dir {tmp_dir} --spark-master local[{thread}] "
        f"-I {sorted_bam} -O {markdup_bam} -M {markdup_metrics}"
    )
    tool.judge_then_exec(run_sample_id, cmd_markdup, markdup_bam)

    markdup_bai = markdup_bam + ".bai"
    cmd_index_markdup = f"samtools index -@ {thread} {markdup_bam}"
    tool.judge_then_exec(run_sample_id, cmd_index_markdup, markdup_bai)

    # --- 3) BQSR (with --interval-padding) ---
    recal_table = os.path.join(dir_bqsr, f"{sample_name}.sorted.markdup.recal_data.table")

    bundle_file1 = bundle_path + "1000G_phase1.snps.high_confidence.hg38.vcf.gz"
    bundle_file2 = bundle_path + "Mills_and_1000G_gold_standard.indels.hg38.vcf.gz"
    bundle_file3 = bundle_path + "dbsnp_138.hg38.vcf.gz"

    cmd_recal_table = (
        f"{gatk} BaseRecalibratorSpark --tmp-dir {tmp_dir} --spark-master local[{thread}] "
        f"-R {ref} -I {markdup_bam} "
        f"--known-sites {bundle_file1} --known-sites {bundle_file2} --known-sites {bundle_file3} "
        f"-O {recal_table} -L {bed_file} --interval-padding {pad_bp}"
    )
    tool.judge_then_exec(run_sample_id, cmd_recal_table, recal_table)

    bqsr_bam = os.path.join(dir_bqsr, f"{sample_name}.sorted.markdup.BQSR.bam")
    cmd_apply_bqsr = (
        f"{gatk} ApplyBQSRSpark --tmp-dir {tmp_dir} --spark-master local[{thread}] "
        f"-R {ref} -I {markdup_bam} --bqsr-recal-file {recal_table} "
        f"-O {bqsr_bam} -L {bed_file} --interval-padding {pad_bp}"
    )
    tool.judge_then_exec(run_sample_id, cmd_apply_bqsr, bqsr_bam)


    # --- 4) bamdst coverage (after markdup) ---
    bamdst_bed = configure["others"].get(
        "bed_file",
        "ERROR!",  # fallback
    )
    bamdst_dir = os.path.join(dir_markdup, "bamdst") + "/"
    tool.judge_then_exec(run_sample_id, f"mkdir -p {bamdst_dir}", bamdst_dir)

    # bamdst typical creates coverage.report / depth.freq or similar; choose a stable sentinel
    bamdst_done = os.path.join(bamdst_dir, "coverage.report")
    cmd_bamdst = f"bamdst -p {bamdst_bed} {markdup_bam} -o {bamdst_dir}"
    tool.judge_then_exec(run_sample_id, cmd_bamdst, bamdst_done)

    prefixed_coverage_report = os.path.join(bamdst_dir, f"{sample_name}.coverage.report")
    cmd_rename_coverage_report = f"mv {bamdst_done} {prefixed_coverage_report}"
    tool.judge_then_exec(run_sample_id, cmd_rename_coverage_report, prefixed_coverage_report)

    return dir_varcall_root, bqsr_bam


def _require(cfg: dict, key: str, ctx: str) -> str:
    """Small helper: require a key in dict, else raise ValueError."""
    v = cfg.get(key, None)
    if v is None or str(v).strip() == "":
        raise ValueError(f"Missing required {ctx}: '{key}'")
    return v

def _caller_mutect2(job: Dict[str, Any]) -> str:
    """Mutect2 worker."""
    tool = job["tool"]
    try:
        tool.write_log("Worker(Mutect2) started.", "info")
        Mutect2(
            run_sample_id=job["run_sample_id"],
            sample_name=job["sample_name"],
            tool=tool,
            configure=job["configure"],
            paths=job["paths"],
            tumor_sample=job["tumor_sample"],
            normal_sample=job["normal_sample"],
            dir_varcall_root_tumor=job["dir_varcall_root_tumor"],
            tumor_bam=job["tumor_bam"],
            normal_bam=job["normal_bam"],
        )
        tool.write_log("Worker(Mutect2) finished.", "info")
        return "Mutect2:OK"
    except Exception:
        tool.write_log(f"Worker(Mutect2) crashed:\n{traceback.format_exc()}", "error")
        raise

def _caller_strelka2(job: Dict[str, Any]) -> str:
    """Strelka2 (Manta+Strelka2) worker."""
    tool = job["tool"]
    try:
        out_dir_strelka2 = job["out_dir_strelka2"]
        os.makedirs(out_dir_strelka2, exist_ok=True)

        argv_strelka2 = [
            "--sample-t", job["tumor_sample"],
            "--sample-n", job["normal_sample"],
            "--ref", job["ref"],
            "--call-regions", job["call_regions_bed_gz"],
            "--tumor-bam", job["tumor_bam"],
            "--normal-bam", job["normal_bam"],
            "--threads", str(job["threads"]),
            "--work-root", out_dir_strelka2,
            "--bcftools", job["bcftools"],
        ]
        tool.write_log(f"Worker(Strelka2) CMD: {' '.join(argv_strelka2)}", "info")

        strelka2_main(argv_strelka2, tool=tool, run_sample_id=job["run_sample_id"])

        tool.write_log("Worker(Strelka2) finished.", "info")
        return "Strelka2:OK"
    except Exception:
        tool.write_log(f"Worker(Strelka2) crashed:\n{traceback.format_exc()}", "error")
        raise


def _caller_vardict(job: Dict[str, Any]) -> str:
    """VarDict worker."""
    tool = job["tool"]
    try:
        out_dir_vardict = job["out_dir_vardict"]
        os.makedirs(out_dir_vardict, exist_ok=True)

        argv_vardict = [
            "--af-thr", str(job["af_thr"]),
            "--sample-t", job["tumor_sample"],
            "--sample-n", job["normal_sample"],
            "--threads", str(job["vd_threads"]),
            "--ref", job["ref"],
            "--bed", job["calling_bed"],
            "--tumor-bam", job["tumor_bam"],
            "--normal-bam", job["normal_bam"],
            "--out-dir", out_dir_vardict,
            "--prefix", job["prefix"],
            "--bcftools", job["bcftools"],
        ]
        tool.write_log(f"Worker(VarDict) CMD: {' '.join(argv_vardict)}", "info")

        vardict_main(argv_vardict, tool=tool, run_sample_id=job["run_sample_id"])

        tool.write_log("Worker(VarDict) finished.", "info")
        return "VarDict:OK"
    except Exception:
        tool.write_log(f"Worker(VarDict) crashed:\n{traceback.format_exc()}", "error")
        raise


def variants_calling_start(sample_name: str, tool, configure: dict, paths: dict):
    """
    Entry point:
      1) preprocess (T/N) once: alignment+markdup+BQSR
      2) run 3 callers in parallel: Mutect2 + Strelka2 + VarDict

    Output directory layout (tumor-root):
      <output>/<tumor>/04.variants_calling/Mutect2/
      <output>/<tumor>/04.variants_calling/Strelka2/
      <output>/<tumor>/04.variants_calling/VarDict/
    """
    try:
        # ---- matched-normal validation ----
        if not bool(configure["others"]["tumor_with_matched_normal"]):
            raise ValueError(
                "Only matched-normal mode is supported "
                "(tumor_with_matched_normal=True)."
            )
        if "," not in sample_name:
            raise ValueError("Matched-normal mode requires sample_name='TUMOR,NORMAL'")

        tumor_sample, normal_sample = [x.strip() for x in sample_name.split(",")[:2]]

        # ---- preprocessing once (T/N) ----
        dir_varcall_root_tumor, tumor_bam = mutation_calling(
            run_sample_id=sample_name,
            sample_name=tumor_sample,
            tool=tool,
            configure=configure,
            paths=paths,
        )
        _, normal_bam = mutation_calling(
            run_sample_id=sample_name,
            sample_name=normal_sample,
            tool=tool,
            configure=configure,
            paths=paths,
        )

        if bool(configure["others"].get("pre_mutation_calling_only", False)):
            tool.write_log("pre_mutation_calling_only=True; stop after preprocessing.", "info")
            return

        # ---- common paths/knobs ----
        ref = paths["database"]["neoantigen"]["HG38"]["REF_FASTA"]
        bcftools = str(configure["others"].get("bcftools", "bcftools")).strip()

        # ---- build calling BEDs from raw bed_file ----
        bed_file = configure["others"]["bed_file"]
        pad_bp = int(configure["others"].get("pad_bp", 100))
        calling_bed, call_regions_bed_gz = _ensure_calling_bed(
            run_sample_id=sample_name,
            tool=tool,
            dir_varcall_root_tumor=dir_varcall_root_tumor,
            bed_file=bed_file,
            pad_bp=pad_bp,
            bcftools=bcftools,
        )

        # ---- threads / thresholds ----
        threads = int(configure["args"].get("thread", 20))  # Strelka2
        vd_threads = int(configure["args"].get("thread", 20))
        af_thr = str(configure["others"].get("min_allele_fraction", 0.02))

        # ---- output dirs ----
        out_dir_strelka2 = os.path.join(dir_varcall_root_tumor, "Strelka2")
        out_dir_vardict = os.path.join(dir_varcall_root_tumor, "VarDict")
        os.makedirs(out_dir_strelka2, exist_ok=True)
        os.makedirs(out_dir_vardict, exist_ok=True)

        # ---- prepare jobs (carry tool into all callers) ----
        job_mutect2 = dict(
            run_sample_id=sample_name,
            sample_name=sample_name,
            tool=tool,
            configure=configure,
            paths=paths,
            tumor_sample=tumor_sample,
            normal_sample=normal_sample,
            dir_varcall_root_tumor=dir_varcall_root_tumor,
            tumor_bam=tumor_bam,
            normal_bam=normal_bam,
        )

        job_strelka2 = dict(
            run_sample_id=sample_name,
            tool=tool,
            tumor_sample=tumor_sample,
            normal_sample=normal_sample,
            ref=ref,
            call_regions_bed_gz=call_regions_bed_gz,
            tumor_bam=tumor_bam,
            normal_bam=normal_bam,
            threads=threads,
            out_dir_strelka2=out_dir_strelka2,
            bcftools=bcftools,
        )

        job_vardict = dict(
            run_sample_id=sample_name,
            tool=tool,
            af_thr=af_thr,
            tumor_sample=tumor_sample,
            normal_sample=normal_sample,
            vd_threads=vd_threads,
            ref=ref,
            calling_bed=calling_bed,
            tumor_bam=tumor_bam,
            normal_bam=normal_bam,
            out_dir_vardict=out_dir_vardict,
            prefix=tumor_sample,
            bcftools=bcftools,
        )

        tool.write_log(
            f"Run callers in parallel: Mutect2 + Strelka2(Manta+Strelka2) + VarDict. Tumor={tumor_sample}",
            "info",
        )

        # ---- parallel execution ----
        results = []
        with NoDaemonPool(processes=3) as pool:
            results.append(
                pool.apply_async(
                    _caller_mutect2,
                    (job_mutect2,),
                    error_callback=tool.print_pool_error,
                )
            )
            results.append(
                pool.apply_async(
                    _caller_strelka2,
                    (job_strelka2,),
                    error_callback=tool.print_pool_error,
                )
            )
            results.append(
                pool.apply_async(
                    _caller_vardict,
                    (job_vardict,),
                    error_callback=tool.print_pool_error,
                )
            )
            pool.close()
            pool.join()

        for r in results:
            _ = r.get()


        # ---- merge 3 callers (PASS only) ----
        tool.write_log("Merge callers: Mutect2 + Strelka2 + VarDict (PASS only)", "info")

        # Mutect2 PASS
        mutect_pass = os.path.join(
            dir_varcall_root_tumor,
            "Mutect2",
            f"{tumor_sample}.mutect.filtered.PASS.vcf.gz",
        )

        # Strelka2 PASS
        strelka_pass = os.path.join(
            out_dir_strelka2,
            "strelka2",
            tumor_sample,
            "merged",
            f"{tumor_sample}.strelka.somatic.merged.PASS.vcf.gz",
        )

        # VarDict PASS
        vardict_pass = os.path.join(
            out_dir_vardict,
            f"{tumor_sample}.vardict.paired.PASS.vcf.gz",
        )

        # Ensure PASS files exist (hard check before merge)
        for p, tag in (
            (mutect_pass, "Mutect2 PASS"),
            (strelka_pass, "Strelka2 PASS"),
            (vardict_pass, "VarDict PASS"),
        ):
            if not os.path.exists(p):
                tool.write_log(f"[ERROR] {tag} not found: {p}", "error")
                raise RuntimeError(f"{tag} not found: {p}")
            if ("PASS" not in os.path.basename(p)) and (".pass" not in os.path.basename(p).lower()):
                tool.write_log(f"[ERROR] {tag} does not look PASS-filtered: {p}", "error")
                raise RuntimeError(f"{tag} not PASS-filtered: {p}")

        merge_dir = os.path.join(dir_varcall_root_tumor, "Merge3Callers")
        tool.judge_then_exec(sample_name, f"mkdir -p {merge_dir}", merge_dir)

        # mutect_pass / strelka_pass / vardict_pass
        merge_outputs = merge_mutect_strelka_vardict(
            tool=tool,
            run_sample_id=sample_name,
            ref=ref,
            mutect=mutect_pass,
            strelka=strelka_pass,
            vardict=vardict_pass,
            tumor=tumor_sample,
            normal=normal_sample,
            out_dir=merge_dir,
            prefix=tumor_sample,
        )
        tool.write_log(f"Merge done. Outputs in: {merge_dir}", "info")

        # ---- subset BAMs for IGV using Merge3Callers 04.bed pad200 BED ----
        igv_bam_dir = os.path.join(merge_dir, "05.igv_bam")
        subset_script = os.path.join(os.path.dirname(__file__), "subset_bam_for_igv.py")
        tumor_subset_bam = os.path.join(
            igv_bam_dir,
            f"{os.path.basename(tumor_bam)[:-4]}.subset.bam",
        )
        cmd_subset_igv_bam = (
            f"{shlex.quote(sys.executable)} -u {shlex.quote(subset_script)} "
            f"--tumor-bam {shlex.quote(tumor_bam)} "
            f"--normal-bam {shlex.quote(normal_bam)} "
            f"--bed {shlex.quote(merge_outputs['bed_pad200'])} "
            f"--out-dir {shlex.quote(igv_bam_dir)} "
            f"--threads {threads}"
        )
        tool.judge_then_exec(sample_name, cmd_subset_igv_bam, tumor_subset_bam)
        tool.write_log(f"IGV subset BAMs done. Outputs in: {igv_bam_dir}", "info")

        tool.write_log(
            f"Variant calling finished (Mutect2/Strelka2/VarDict). Tumor={tumor_sample}",
            "info",
        )

    except Exception:
        tool.write_log(f"variants_calling_start crashed:\n{traceback.format_exc()}", "error")
        raise
