# coding=utf-8

from pathlib import Path
import gzip
import shlex
from typing import List


def _resolve_vep_plugin_mount(vep_data_dir: str, plugins_value: str):
    """
    Resolve the host bind and in-container plugin directory for VEP plugins.

    `plugins_value` may be either:
      - a simple directory name such as ``VEP_plugins-release-110``
      - a relative path under ``vep_data_dir``
      - an absolute host path
    """
    vep_data_path = Path(vep_data_dir).expanduser().resolve()
    plugins_raw = (plugins_value or "VEP_plugins-release-110").strip()
    plugins_path = Path(plugins_raw).expanduser()

    if plugins_path.is_absolute():
        host_plugins_dir = plugins_path.resolve()
        fallback_under_vep_data = (vep_data_path / host_plugins_dir.name).resolve()
        # Be forgiving with stale absolute paths: if the configured absolute plugin
        # path points outside vep_data, but vep_data contains a same-named plugin
        # directory, prefer the colocated copy under vep_data.
        if (
            host_plugins_dir != fallback_under_vep_data
            and fallback_under_vep_data.exists()
        ):
            host_plugins_dir = fallback_under_vep_data
    else:
        host_plugins_dir = (vep_data_path / plugins_path).resolve()

    try:
        rel_plugins_dir = host_plugins_dir.relative_to(vep_data_path)
    except ValueError:
        return str(host_plugins_dir), "/vep_plugins/"

    rel_plugins_dir_str = rel_plugins_dir.as_posix().strip("/")
    if not rel_plugins_dir_str:
        return str(host_plugins_dir), "/vep_data/"
    return str(host_plugins_dir), f"/vep_data/{rel_plugins_dir_str}/"


def _open_text_auto(path: str, mode: str):
    """
    Open a text file transparently for .gz or plain files.

    Args:
        path: File path; if it ends with '.gz' a gzip handle is used.
        mode: 'rt' for read-text, 'wt' for write-text.
    """
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def split_vcf(input_vcf: str, output_dir: str, file_suffix: str, chunks: int = 10) -> None:
    """
    Split a VCF file into multiple chunks with an even distribution of variants.

    Parameters:
        input_vcf (str): Path to the input VCF (.vcf or .vcf.gz).
        output_dir (str): Directory to write chunked files.
        file_suffix (str): Prefix for output chunk file names.
        chunks (int): Number of chunks to create (default: 10). Must be >= 1.

    Returns:
        None. Writes files to `output_dir`.

    Output Files:
        {output_dir}/{file_suffix}.chunk_{0..N-1}.vcf  or  .vcf.gz (matches input extension)
    """
    if chunks < 1:
        raise ValueError("`chunks` must be >= 1")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header: List[str] = []
    variants: List[str] = []

    # Read header and variants
    with _open_text_auto(input_vcf, "rt") as f:
        for line in f:
            if line.startswith("#"):
                header.append(line)
            else:
                variants.append(line)

    total_variants = len(variants)
    if total_variants == 0:
        # Still emit empty chunks with header for reproducibility
        for i in range(chunks):
            ext = ".vcf.gz" if input_vcf.endswith(".gz") else ".vcf"
            output_file = out_dir / f"{file_suffix}.chunk_{i}{ext}"
            with _open_text_auto(str(output_file), "wt") as w:
                w.writelines(header)
        return

    base = total_variants // chunks
    remainder = total_variants % chunks

    start = 0
    for i in range(chunks):
        size = base + (1 if i < remainder else 0)
        end = start + size
        chunk_variants = variants[start:end]

        ext = ".vcf.gz" if input_vcf.endswith(".gz") else ".vcf"
        output_file = out_dir / f"{file_suffix}.chunk_{i}{ext}"

        with _open_text_auto(str(output_file), "wt") as w:
            w.writelines(header)
            w.writelines(chunk_variants)

        start = end


def annotation_vcf(run_sample_id, sample, tool, paths, configure):
    """
    Run VEP annotation inside an Apptainer/Singularity container, remove
    ref-transcript mismatches (hard filter), keep PASS only using bcftools,
    index, then split the PASS VCF into chunks.

    The function expects `tool` to provide:
        - tool.judge_then_exec(run_id, cmd, expect_path)
        - tool.write_log(msg, level)

    Notes:
      - Input VCF is from Merge3Callers shared_mutect:
        {output_dir}/{sample}/04.variants_calling/Merge3Callers/02.shared_mutect/
        {sample}.mutect.shared.bypos.vcf.gz
      - Output naming (recommended semantic):
        {sample}.mutect.shared.bypos.VEP.vcf
        {sample}.mutect.shared.bypos.VEP.rm_mismatch.vcf
        {sample}.mutect.shared.bypos.VEP.rm_mismatch.PASS.vcf.gz (+ index)
    """
    # -------- resolve paths / args --------
    human_vep_sif = paths["path"]["neoantigen"]["VEP_SIF"]
    vep_data_dir = paths["database"]["neoantigen"]["VEP_DATA_DIR"]

    ref_fasta_path = Path(paths["database"]["neoantigen"]["HG38"]["REF_FASTA"]).expanduser()
    hg38_ref_dir = str(ref_fasta_path.parent)
    ref_name = ref_fasta_path.name

    plugins_value = (
        paths.get("database", {})
             .get("neoantigen", {})
             .get("VEP_PLUGINS_VERSION", "VEP_plugins-release-110")
    )
    host_plugins_dir, container_plugins_dir = _resolve_vep_plugin_mount(
        vep_data_dir,
        plugins_value,
    )

    output_dir = configure["path"]["output_dir"].rstrip("/")
    thread = int(configure["args"]["thread"])
    hla_binding_threads = int(configure["args"]["hla_binding_threads"])

    bcftools = (
        configure.get("args", {}).get("bcftools")
        or paths.get("path", {}).get("neoantigen", {}).get("BCFTOOLS")
        or "bcftools"
    )

    # Output dir
    vep_dir = f"{output_dir}/{sample}/{configure['step_name']['annotation']}/"
    tool.judge_then_exec(run_sample_id, f"mkdir -p {shlex.quote(vep_dir)}", vep_dir)

    # Input shared VCF
    variants_step = configure["step_name"]["variants_calling"]
    data_dir = f"{output_dir}/{sample}/{variants_step}/Merge3Callers/02.shared_mutect/"
    suffix = "shared"

    abs_input_vcfgz = f"{data_dir}{sample}.{suffix}.vcf.gz"
    tool.write_log(f"Annotation input VCF: {abs_input_vcfgz}", "info")
    tool.write_log(f"VEP plugin setting: {plugins_value}", "info")
    tool.write_log(
        f"VEP plugin bind: {host_plugins_dir} -> {container_plugins_dir}",
        "info",
    )
    tool.judge_then_exec(run_sample_id, f"test -s {shlex.quote(abs_input_vcfgz)}", abs_input_vcfgz)

    input_file = f"/data_dir/{sample}.{suffix}.vcf.gz"

    # Outputs
    abs_vep_vcf = f"{vep_dir}{sample}.{suffix}.VEP.vcf"
    abs_rm_mismatch_vcf = f"{vep_dir}{sample}.{suffix}.VEP.rm_mismatch.vcf"
    abs_pass_vcfgz = f"{vep_dir}{sample}.{suffix}.VEP.rm_mismatch.PASS.vcf.gz"

    output_file_vep = f"/output_dir/{Path(abs_vep_vcf).name}"

    # VEP options (match bash)
    vep_options = (
        "--offline --cache --format vcf --vcf "
        "--symbol --terms SO --tsl --biotype --hgvs "
        "--plugin Frameshift --plugin Wildtype "
        "--af --af_1kg "
        "--sift b "
        "--transcript_version "
        "--mane_select --canonical"
    )

    cmd_vep_args = [
        "apptainer", "exec",
        "-B", f"{vep_dir}:/output_dir/",
        "-B", f"{data_dir}:/data_dir/",
        "-B", f"{vep_data_dir}:/vep_data/",
        "-B", f"{host_plugins_dir}:{container_plugins_dir}",
        "-B", f"{hg38_ref_dir}:/fasta_data/",
        human_vep_sif,
        "vep",
        "--fork", str(max(1, min(thread, 5))),
        "--input_file", input_file,
        "--output_file", output_file_vep,
        "--dir_cache", "/vep_data/",
        f"--fasta=/fasta_data/{ref_name}",
        f"--dir_plugins={container_plugins_dir}",
    ] + vep_options.split()

    cmd_vep = " ".join(shlex.quote(x) for x in cmd_vep_args)
    tool.write_log("Run VEP annotation.", "info")
    tool.judge_then_exec(run_sample_id, cmd_vep, abs_vep_vcf)

    # Remove ref-transcript mismatches (keep original style)
    cmd_rm = f"ref-transcript-mismatch-reporter {shlex.quote(abs_vep_vcf)} -f hard"
    tool.write_log("Remove ref-transcript mismatches (hard filter).", "info")
    abs_tool_filtered_vcf = f"{vep_dir}{sample}.{suffix}.VEP.filtered.vcf"
    tool.judge_then_exec(run_sample_id, cmd_rm, abs_tool_filtered_vcf)

    cmd_rename = (
        f"mv -f {shlex.quote(abs_tool_filtered_vcf)} {shlex.quote(abs_rm_mismatch_vcf)}"
    )
    tool.judge_then_exec(run_sample_id, cmd_rename, abs_rm_mismatch_vcf)

    # PASS only + index (bcftools)
    tool.write_log("VEP: bcftools PASS + index", "info")
    cmd_pass = (
        f"{shlex.quote(bcftools)} view -f PASS -Oz -o {shlex.quote(abs_pass_vcfgz)} {shlex.quote(abs_rm_mismatch_vcf)} "
        f"&& {shlex.quote(bcftools)} index -t {shlex.quote(abs_pass_vcfgz)}"
    )
    tool.judge_then_exec(run_sample_id, cmd_pass, abs_pass_vcfgz)

    # Split into chunks
    chunk_dir = f"{vep_dir}chunks"
    tool.judge_then_exec(run_sample_id, f"mkdir -p {shlex.quote(chunk_dir)}", chunk_dir)

    tool.write_log(f"Split PASS-only VCF into {hla_binding_threads} chunks.", "info")
    split_vcf(
        input_vcf=abs_pass_vcfgz,
        output_dir=chunk_dir,
        file_suffix=f"{sample}.{suffix}.VEP.rm_mismatch.PASS",
        chunks=hla_binding_threads
    )
    tool.write_log("VCF splitting completed.", "info")
    return vep_dir
