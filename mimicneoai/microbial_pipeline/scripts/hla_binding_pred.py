# coding=utf-8
import os

from mimicneoai.functions.nodemon_pool import NoDaemonPool
from mimicneoai.functions.pipline_tools import raise_for_failed_samples


def get_hlahd_results(sample: str, output_hla: str):
    """
    Extract HLA typing results from HLA-HD outputs.

    Args:
        sample: Sample identifier.
        output_hla: Directory containing HLA-HD results.

    Returns:
        (hlaI, hlaII): Two comma-separated strings of normalized HLA alleles.
                       For each allele, keep only the first two fields (e.g., A*02:01).
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
        # Gracefully return empty results if the expected file is missing/empty
        return "", ""

    hlaI, hlaII = [], []

    with open(hla_file, "r") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if not cols:
                continue
            gene = cols[0]
            alleles = cols[1:]

            # HLA-I genes
            if gene in hlaI_genes:
                for h in alleles:
                    if h and h != '-' and h != 'Not typed':
                        parts = h.split(":")
                        if len(parts) >= 2:
                            h2 = ":".join(parts[:2])  # e.g., A*02:01
                            hlaI.append(h2)

            # HLA-II genes
            if gene in hlaII_genes:
                for h in alleles:
                    if h and h != '-' and h != 'Not typed':
                        parts = h.split(":")
                        if len(parts) >= 2:
                            h2 = ":".join(parts[:2])  # e.g., DQA1*03:02 or DRB1*04:05
                            # Some tools output DQA1*XX:YY-DQB1*AA:BB in one field. If so, keep the RHS.
                            if "-" in h2:
                                rhs = h2.split("-", 1)[1]
                                hlaII.append(rhs)
                            else:
                                hlaII.append(h2)

    # Join and strip trailing separators
    return ",".join(hlaI), ",".join(hlaII)


def pvacbind(sample, configure, pathes, tool):
    """
    Run pVACbind in chunked mode with a non-daemon Pool.
    Each chunk is submitted via tool.judge_then_exec(sample, cmd, target, chunk_tag).

    Notes:
      - Splits an input FASTA into N chunks (N = threads).
      - Launches pVACbind for each chunk.
      - Merges per-chunk TSVs if present.
      - Writes a final 'done' flag when merged output is available.
    """
    import gzip

    # ---- Small utilities (local scope, minimal changes) ----
    def _ensure_dir(p): os.makedirs(p, exist_ok=True)
    def _is_gz(p): return str(p).endswith(".gz")

    def _fasta_iter(fp):
        _open = gzip.open if _is_gz(fp) else open
        h, seq = None, []
        with _open(fp, "rt") as fh:
            for line in fh:
                if line.startswith(">"):
                    if h is not None:
                        yield h, "".join(seq)
                    h = line[1:].strip()
                    seq = []
                else:
                    seq.append(line.strip())
            if h is not None:
                yield h, "".join(seq)

    def _write_fasta(entries, out_path):
        _ensure_dir(os.path.dirname(out_path))
        if not entries:
            open(out_path, "w").close()
            return
        with open(out_path, "w") as fo:
            for h, s in entries:
                fo.write(">" + h + "\n")
                fo.write(s + "\n")

    def _split_fasta_roundrobin(in_fa, out_root, n_chunks, sample_name):
        buffers, idx, total = [[] for _ in range(n_chunks)], 0, 0
        for h, s in _fasta_iter(in_fa):
            buffers[idx].append((h, s))
            idx = (idx + 1) % n_chunks
            total += 1
        chunk_paths = []
        for i in range(n_chunks):
            out_fa = os.path.join(out_root, "chunks", f"chunk_{i}", f"{sample_name}.chunk_{i}.pep")
            _write_fasta(buffers[i], out_fa)
            chunk_paths.append(out_fa)
        return chunk_paths, total

    def _merge_chunk_results(sample_name, out_root, n_chunks):
        merged_dir = os.path.join(out_root, "combined")
        _ensure_dir(merged_dir)
        merged_out = os.path.join(merged_dir, f"{sample_name}.merged.all_epitopes.tsv")

        candidates = []
        for i in range(n_chunks):
            p = os.path.join(out_root, "chunks", f"chunk_{i}", "combined", f"{sample_name}.all_epitopes.tsv")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                candidates.append(p)
        if not candidates:
            return None

        wrote_header, total_rows = False, 0
        with open(merged_out, "w") as out_f:
            for path in candidates:
                with open(path, "r") as in_f:
                    header = in_f.readline()
                    body = in_f.readlines()
                    if not body:
                        continue
                    if not wrote_header:
                        out_f.write(header)
                        wrote_header = True
                    out_f.writelines(body)
                    total_rows += len(body)
        return merged_out if total_rows > 0 else None

    # ---- Algorithm selection (configurable via others.algo; otherwise defaults) ----
    _default_algo = (
        "BigMHC_EL BigMHC_IM DeepImmuno MHCflurry MHCflurryEL "
        "MHCnuggetsI MHCnuggetsII NNalign NetMHC NetMHCIIpan "
        "NetMHCIIpanEL NetMHCpan NetMHCpanEL PickPocket SMM SMMPMBEC"
    )
    algoIandII = configure.get('others', {}).get('algo', _default_algo)

    # ---- Extract paths/parameters from config ----
    thread = int(configure['args']['hla_binding_threads'])
    opt_dir = os.path.join(configure['path']['output_dir'], "")
    pvactools_sif = pathes['path']['common']['PVACTOOLS']

    step_name_blastx   = configure['step_name']['blastx']
    step_name_pvacbind = configure['step_name']['pvacbind']
    step_name_hla      = configure['step_name']['hla']

    output_blastx   = os.path.join(opt_dir, sample, step_name_blastx, "")
    output_pvacbind = os.path.join(opt_dir, sample, step_name_pvacbind, "")
    output_hla      = os.path.join(opt_dir, sample, step_name_hla, "")

    # ---- Parse HLA typing results ----
    hlaI, hlaII = get_hlahd_results(sample, output_hla)
    if hlaI == '' and hlaII == '':
        tool.write_log(f"[{sample}] HLA-HD results are all 'Not typed' or missing.", "error")
        return
    HLA = (hlaI + ("," if hlaI and hlaII else "") + hlaII)

    # ---- Input FASTA ----
    in_fa = os.path.join(output_blastx, f"{sample}.peptide.fasta")
    if not os.path.exists(in_fa) or os.path.getsize(in_fa) == 0:
        tool.write_log(f"[{sample}] Peptide FASTA not found or empty: {in_fa}", "error")
        return

    # ---- Chunking (N = threads) ----
    n_chunks = max(1, thread)
    _ensure_dir(output_pvacbind)
    chunk_paths, total_entries = _split_fasta_roundrobin(in_fa, output_pvacbind, n_chunks, sample)
    if total_entries == 0:
        tool.write_log(f"[{sample}] No peptide entries found in {in_fa}", "error")
        return

    # ---- Common environment/bindings for container ----
    # env_prefix = 'export TF_CPP_MIN_LOG_LEVEL=2; export CUDA_VISIBLE_DEVICES=""; '  # optional
    bind_arg = f"--bind {output_blastx},{output_hla},{opt_dir} "

    # ---- Build per-chunk commands + targets ----
    tasks = []  # (cmd, target, chunk_tag)
    for i, chunk_fa in enumerate(chunk_paths):
        if (not os.path.exists(chunk_fa)) or os.path.getsize(chunk_fa) == 0:
            continue
        out_dir_i = os.path.join(output_pvacbind, "chunks", f"chunk_{i}")
        _ensure_dir(out_dir_i)
        target_i = os.path.join(out_dir_i, "combined", f"{sample}.all_epitopes.tsv")  # file to watch

        cmd_i = (
            # f"{env_prefix}"
            f"apptainer exec {bind_arg}"
            f"{pvactools_sif} pvacbind run "
            f"{chunk_fa} {sample} {HLA} {algoIandII} "
            f"{out_dir_i} -e1 8,9,10 -e2 15 "
            f"--iedb-install-directory /opt/iedb "
            f"-t 1 --fasta-size 100000 "
            f"< /dev/null"
        )
        tasks.append((cmd_i, target_i, f"chunk_{i}"))

    if not tasks:
        tool.write_log(f"[{sample}] All chunks are empty; nothing to run.", "error")
        return

    # ---- Dispatch in parallel via NoDaemonPool ----
    pool = NoDaemonPool(min(len(tasks), n_chunks))
    async_results = []
    for cmd_i, target_i, chunk_tag in tasks:
        async_results.append(
            (
                f"{sample}:{chunk_tag}",
                pool.apply_async(
                    tool.judge_then_exec,
                    (sample, cmd_i, target_i, chunk_tag),
                    error_callback=getattr(tool, "print_pool_error", None),
                ),
            )
        )
    pool.close()
    pool.join()
    raise_for_failed_samples(async_results)

    # ---- Merge chunk results ----
    merged_path = _merge_chunk_results(sample, output_pvacbind, n_chunks)

    # ---- Final completion flag (kept for downstream convention) ----
    final_done_flag_dir = os.path.join(output_pvacbind, "combined")
    _ensure_dir(final_done_flag_dir)
    final_done_flag = os.path.join(final_done_flag_dir, f"{sample}.done.txt")

    if merged_path and os.path.exists(merged_path) and os.path.getsize(merged_path) > 0:
        with open(final_done_flag, "w") as f:
            f.write("pvacbind chunked run done\n")
    else:
        tool.write_log(
            f"[{sample}] No merged output generated under {os.path.join(output_pvacbind, 'combined')}/",
            "error",
        )
