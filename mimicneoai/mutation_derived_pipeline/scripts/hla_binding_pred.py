# coding=utf-8
import os
import shlex
from typing import Tuple, Dict, Any, Optional, List
from mimicneoai.functions.nodemon_pool import NoDaemonPool
from mimicneoai.functions.pipline_tools import raise_for_failed_samples

class Pvacseq:
    """
    Wrapper for running pVACseq in chunked mode (per-VCF chunk),
    with utilities to parse HLA-HD results and merge outputs.
    """

    def __init__(self, tool):
        """
        Args:
            tool: An object exposing judge_then_exec(...) and write_log(...), etc.
        """
        self.tool = tool

    # ------------------------------
    # HLA-HD parsing
    # ------------------------------
    def get_hlahd_results(self, sample: str, output_hla: str) -> Tuple[str, str]:
        """
        Parse HLA-HD results and return comma-separated HLA-I and HLA-II allele lists.
        HLA fields are normalized to 2-digit resolution (e.g., A*02:01).

        Args:
            sample: Sample identifier.
            output_hla: Base directory containing HLA-HD results.

        Returns:
            (hlaI, hlaII): Two comma-separated strings (no trailing commas).
        """
        hlaI_genes = ['A', 'B', 'C', 'E', 'F', 'G']
        hlaII_genes = [
            # HLA-DR
            'DRA', 'DRB1', 'DRB3', 'DRB4', 'DRB5',
            # HLA-DQ
            'DQA1', 'DQB1',
            # HLA-DP
            'DPA1', 'DPB1'
        ]

        hla_file = os.path.join(output_hla, sample, "result", f"{sample}_final.result.txt")
        if not os.path.exists(hla_file) or os.path.getsize(hla_file) == 0:
            self.tool.write_log(f"[{sample}] HLA-HD result file missing/empty: {hla_file}", "error")
            return "", ""

        hlaI_list: List[str] = []
        hlaII_list: List[str] = []

        with open(hla_file, "r") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if not fields:
                    continue
                gene, alleles = fields[0], fields[1:]

                # HLA-I
                if gene in hlaI_genes:
                    for h in alleles:
                        if h and h != '-' and h != 'Not typed':
                            parts = h.split(":")
                            if len(parts) >= 2:
                                h2 = ":".join(parts[:2])  # e.g., A*02:01
                                hlaI_list.append(h2)

                # HLA-II
                if gene in hlaII_genes:
                    for h in alleles:
                        if h and h != '-' and h != 'Not typed':
                            parts = h.split(":")
                            if len(parts) >= 2:
                                h2 = ":".join(parts[:2])  # e.g., DQA1*03:02 or DRB1*04:05
                                # Some outputs may include combined strings like 'DQA1*XX:YY-DQB1*AA:BB'
                                if "-" in h2:
                                    rhs = h2.split("-", 1)[1]  # take the right-hand side allele
                                    hlaII_list.append(rhs)
                                else:
                                    hlaII_list.append(h2)

        return ",".join(hlaI_list), ",".join(hlaII_list)

    # ------------------------------
    # pVACseq execution
    # ------------------------------
    def generate_protein_fasta(
        self,
        run_sample_id: str,
        sample: str,
        output_vcf: str,
        configure: Dict[str, Any],
        pathes: Dict[str, Any],
    ) -> str:
        """
        Reproduce the validated manual workflow under 07.binding_prediction:
          1) bcftools sort the rm_mismatch VCF
          2) tabix index
          3) pvacseq generate_protein_fasta inside Apptainer
        """
        output_dir = os.path.join(configure["path"]["output_dir"], "")
        step_name_pvacseq = configure["step_name"]["pvacseq"]
        output_pvacseq = os.path.join(output_dir, sample, step_name_pvacseq, "")
        os.makedirs(output_pvacseq, exist_ok=True)
        self.tool.judge_then_exec(run_sample_id, f"mkdir -p {output_pvacseq}", output_pvacseq)

        pvactools_sif = pathes["path"]["common"]["PVACTOOLS"]
        bcftools = str(configure.get("others", {}).get("bcftools", "bcftools")).strip()

        input_rm_mismatch_vcf = os.path.join(
            output_vcf,
            f"{sample}.shared.VEP.rm_mismatch.vcf",
        )
        sorted_vcfgz = os.path.join(
            output_pvacseq,
            f"{sample}.shared.VEP.rm_mismatch.sorted.vcf.gz",
        )
        protein_fasta = os.path.join(output_pvacseq, f"{sample}.protein.fasta")
        manufacturability_tsv = f"{protein_fasta}.manufacturability.tsv"

        cmd_sort_and_index = (
            f"{shlex.quote(bcftools)} sort -Oz -o {shlex.quote(sorted_vcfgz)} "
            f"{shlex.quote(input_rm_mismatch_vcf)} "
            f"&& tabix -p vcf {shlex.quote(sorted_vcfgz)}"
        )
        self.tool.judge_then_exec(run_sample_id, cmd_sort_and_index, sorted_vcfgz)

        bind_arg = f"--bind {output_vcf},{output_dir} "
        cmd_generate_protein = (
            f"cd {shlex.quote(output_pvacseq)} && "
            f"apptainer exec {bind_arg}"
            f"{pvactools_sif} pvacseq generate_protein_fasta "
            f"-s {shlex.quote(sample)} --mutant-only --pass-only "
            f"{shlex.quote(sorted_vcfgz)} 13 {shlex.quote(os.path.basename(protein_fasta))}"
        )
        self.tool.judge_then_exec(run_sample_id, cmd_generate_protein, manufacturability_tsv)
        return protein_fasta

    def pvacseq(
        self,
        run_sample_id: str,
        sample: str,
        output_vcf: str,
        output_hla: str,
        configure: Dict[str, Any],
        pathes: Dict[str, Any],
        vcf_chunk: Optional[str] = None,
    ) -> str:
        """
        Run pVACseq for a full VCF or a single VCF chunk.

        Args:
            run_sample_id: Logical run id (used by the tool logger).
            sample: Sample identifier.
            output_vcf: Root directory where VCFs are located (bound into container).
            output_hla: Root directory for HLA results (bound into container).
            configure: Configuration dictionary.
            pathes: Paths dictionary (should include pathes['path']['common']['PVACTOOLS']).
            vcf_chunk: Chunk index (as string) if running a specific chunk; None for whole VCF.

        Returns:
            The output directory path for this pVACseq run.
        """
        # Algorithm set (overridable via configure['others']['algo'])
        _default_algo = (
            "BigMHC_EL BigMHC_IM DeepImmuno MHCflurry MHCflurryEL "
            "MHCnuggetsI MHCnuggetsII NNalign NetMHC NetMHCIIpan "
            "NetMHCIIpanEL NetMHCpan NetMHCpanEL PickPocket SMM SMMPMBEC"
        )
        algoIandII = configure.get("others", {}).get("algo", _default_algo)

        thread = int(configure["args"]["thread"])
        output_dir = os.path.join(configure["path"]["output_dir"], "")  # ensure trailing sep
        step_name_annotation = configure["step_name"]["annotation"]
        step_name_pvacseq = configure["step_name"]["pvacseq"]

        pvactools_sif = pathes["path"]["common"]["PVACTOOLS"]

        # Output directory (chunk-aware)
        if vcf_chunk is not None:
            output_pvacseq = os.path.join(output_dir, sample, step_name_pvacseq, "chunks", f"chunk_{vcf_chunk}")
        else:
            output_pvacseq = os.path.join(output_dir, sample, step_name_pvacseq, "")

        os.makedirs(output_pvacseq, exist_ok=True)
        # Also allow upstream tool to log the mkdir action (keeps legacy behavior)
        self.tool.judge_then_exec(run_sample_id, f"mkdir -p {output_pvacseq}", output_pvacseq)

        # Input VCF naming (must match annotation_vcf outputs)
        base_prefix = "shared.VEP.rm_mismatch.PASS"

        # HLA parsing
        hlaI, hlaII = self.get_hlahd_results(sample, output_hla)
        if not hlaI and not hlaII:
            self.tool.write_log("HLA-HD results are all Not_typed or missing!", "error")
            raise RuntimeError("HLA-HD results are all Not_typed or missing")

        # Build combined HLA string without trailing commas
        HLA = ",".join([x for x in [hlaI, hlaII] if x]).strip(",")

        # Decide input VCF and threads (chunked runs use single thread inside the container)
        if vcf_chunk is not None:
            # e.g., <...>/05.annotation/chunks/{sample}.shared.VEP.rm_mismatch.PASS.chunk_{i}.vcf.gz
            input_vcf = os.path.join(
                output_dir, sample, step_name_annotation, "chunks",
                f"{sample}.{base_prefix}.chunk_{vcf_chunk}.vcf.gz"
            )
            running_thread = 1
        else:
            # e.g., <...>/05.annotation/{sample}.shared.VEP.rm_mismatch.PASS.vcf.gz
            input_vcf = os.path.join(
                output_dir, sample, step_name_annotation,
                f"{sample}.{base_prefix}.vcf.gz"
            )
            running_thread = thread

        # Container binding: bind VCF root, HLA root, and output root
        bind_arg = f"--bind {output_vcf},{output_hla},{output_dir} "
        # Epitope lengths (tool-agnostic naming)
        e1 = str(configure.get("others", {}).get("mhc_i_epitope_lengths", "8,9,10"))
        e2 = str(configure.get("others", {}).get("mhc_ii_epitope_lengths", "15"))
        cmd = (
            f"apptainer exec {bind_arg}"
            f"{pvactools_sif} pvacseq run "
            f"{input_vcf} {sample} {HLA} {algoIandII} "
            f"{output_pvacseq} -e1 {e1} -e2 {e2} "
            f"--iedb-install-directory /opt/iedb "
            f"-t {running_thread} --fasta-size 100000 "
            f"< /dev/null"
        )

        # Target file to watch (kept consistent with upstream)
        target = os.path.join(output_pvacseq, "combined", f"{sample}.all_epitopes.tsv")

        self.tool.judge_then_exec(run_sample_id, cmd, target, vcf_chunk)
        return output_pvacseq

    def run_pvacseq_parallel(
        self,
        run_sample_id: str,
        sample: str,
        output_vcf: str,
        output_hla: str,
        configure: Dict[str, Any],
        pathes: Dict[str, Any],
    ) -> None:
        """
        Execute pVACseq in parallel across N chunks (N == threads in config).

        Args:
            run_sample_id: Logical run id (used by the tool logger).
            sample: Sample identifier.
            output_vcf: Root directory where VCFs are located.
            output_hla: Root directory for HLA results.
            configure: Configuration dictionary.
            pathes: Paths dictionary.
        """
        thread = int(configure["args"]["hla_binding_threads"])
        output_dir = os.path.join(configure["path"]["output_dir"], "")
        step_name_pvacseq = configure["step_name"]["pvacseq"]
        output_pvacseq = os.path.join(output_dir, sample, step_name_pvacseq, "")

        self.generate_protein_fasta(run_sample_id, sample, output_vcf, configure, pathes)

        pool = NoDaemonPool(thread)
        async_results = []
        for i in range(thread):
            chunk = str(i)
            async_results.append(
                (
                    f"{sample}:chunk_{chunk}",
                    pool.apply_async(
                        self.pvacseq,
                        (run_sample_id, sample, output_vcf, output_hla, configure, pathes, chunk),
                        error_callback=self.tool.print_pool_error,
                    ),
                )
            )
        pool.close()
        pool.join()
        raise_for_failed_samples(async_results)

        # Merge chunk outputs
        self.merge_chunk_results(sample, output_pvacseq, thread)

    # ------------------------------
    # Merge utilities
    # ------------------------------
    def merge_chunk_results(self, sample: str, output_dir: str, num_chunks: int) -> None:
        """
        Merge per-chunk TSVs into a single file:
          {output_dir}/combined/{sample}.merged.all_epitopes.tsv

        Only merges files that exist and have data rows (beyond header).

        Args:
            sample: Sample identifier.
            output_dir: Base pVACseq output directory.
            num_chunks: Number of chunks to scan.
        """
        merged_dir = os.path.join(output_dir, "combined")
        os.makedirs(merged_dir, exist_ok=True)

        merged_epitopes = os.path.join(merged_dir, f"{sample}.merged.all_epitopes.tsv")

        # Candidate chunk paths
        candidates = [
            os.path.join(output_dir, "chunks", f"chunk_{i}", "combined", f"{sample}.all_epitopes.tsv")
            for i in range(num_chunks)
        ]

        # Filter existing, non-empty files
        existing = [p for p in candidates if os.path.exists(p) and os.path.getsize(p) > 0]
        if not existing:
            self.tool.write_log("No chunk result files found to merge.", "error")
            return

        wrote_header = False
        total_rows = 0
        used_files: List[str] = []

        try:
            with open(merged_epitopes, "w") as out_f:
                for path in existing:
                    with open(path, "r") as in_f:
                        header = in_f.readline()
                        body_lines = in_f.readlines()

                        # Skip purely-header files
                        if not body_lines:
                            continue

                        if not wrote_header:
                            out_f.write(header)
                            wrote_header = True

                        out_f.writelines(body_lines)
                        total_rows += len(body_lines)
                        used_files.append(path)

            if total_rows == 0:
                self.tool.write_log("All chunk files contain only headers (no data).", "error")
                try:
                    os.remove(merged_epitopes)
                except OSError:
                    pass
                return

            self.tool.write_log(
                f"Merged {len(used_files)} files with {total_rows} data rows -> {merged_epitopes}",
                "info"
            )
        except IOError as e:
            raise RuntimeError(f"Error while merging files: {e}")
