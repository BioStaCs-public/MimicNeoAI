#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example (sanitized):
    conda activate pvactools4.2.1
    export TF_CPP_MIN_LOG_LEVEL=2
    export CUDA_VISIBLE_DEVICES=""
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
    /path/to/conda/envs/pvactools4.2.1/bin/python 07-hla_binding_pred.py \
      -s SAMPLE_ID \
      --pep-fasta /path/to/SAMPLE_ID/06-aeSEPs/SAMPLE_ID.aeSEPs.pep \
      --hla-file  /path/to/SAMPLE_ID/05-hla_typing/SAMPLE_ID_final.result.txt \
      -o /path/to/SAMPLE_ID/07-hla_binding_pred/ \
      --iedb-install-directory /path/to/IEDB \
      --pvactools /path/to/pvactools.sif \
      -t 30 \
      --algos "NNalign NetMHC NetMHCIIpan SMM SMMPMBEC SMMalign NetMHCpanEL NetMHCIIpanEL" \
      --e1-lengths 8,9,10 \
      --e2-lengths 15

Notes:
- This script runs pVACbind in parallel on peptide FASTA by splitting into N chunks.
- It uses an Apptainer/Singularity container (pvactools.sif) and relies on /opt/iedb inside the container.
- The --iedb-install-directory argument is kept for CLI parity but is not bound (container provides /opt/iedb).
"""

import os
import sys
import argparse
import gzip
import multiprocessing
import multiprocessing.pool

try:
    from mimicneoai.functions.nodemon_pool import NoDaemonPool
except ModuleNotFoundError as exc:
    # Allow direct script execution from repo checkout without editable install.
    if getattr(exc, "name", "") != "mimicneoai":
        raise

    class _NoDaemonProcess(multiprocessing.Process):
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass

        daemon = property(_get_daemon, _set_daemon)

    class NoDaemonPool(multiprocessing.pool.Pool):
        @staticmethod
        def Process(_, *args, **kwargs):
            return _NoDaemonProcess(*args, **kwargs)

# -------- I/O helpers --------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def is_gz(p):
    return str(p).endswith(".gz")

def fasta_iter(fp):
    """Yield tuples (header_without_>, sequence_no_newlines). Supports .gz/.fa/.fasta."""
    _open = gzip.open if is_gz(fp) else open
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

def write_fasta(entries, out_path):
    # write single-line sequences for robustness
    with open(out_path, "w") as fo:
        for h, s in entries:
            fo.write(">" + h + "\n")
            fo.write(s + "\n")

# -------- HLA parsing (HLA-HD style tab-delimited result) --------
def parse_hlahd_result(hla_file):
    """
    Parse an HLA-HD final.result.txt-like table.

    Returns:
        (hlaI_csv, hlaII_csv): strings of comma-separated allele two-field resolutions, e.g. "A*02:01,B*07:02"
                               Note: For class II, alleles with '-' (e.g., "DRB1-DRB3") are post-processed to keep the right part.
    """
    hlaI_set = ['A', 'B', 'C', 'E', 'F', 'G']
    hlaII_set = [
        # HLA-DR
        'DRA', 'DRB1', 'DRB3', 'DRB4', 'DRB5',
        # HLA-DQ
        'DQA1', 'DQB1',
        # HLA-DP
        'DPA1', 'DPB1'
    ]

    hlaI = ''
    hlaII = ''
    with open(hla_file, "r") as f:
        for line in f:
            cols = line.rstrip("\n").split('\t')
            if not cols:
                continue
            head = cols[0]
            if head in hlaI_set:
                for h in cols[1:]:
                    h_split = h.split(":")
                    if h not in ('-', 'Not typed') and len(h_split) >= 2:
                        h_2 = ':'.join(h_split[:2])
                        hlaI += h_2 + ','
            if head in hlaII_set:
                for h in cols[1:]:
                    h_split = h.split(":")
                    if h not in ('-', 'Not typed') and len(h_split) >= 2:
                        h_2 = ':'.join(h_split[:2])
                        # strip left part if like "DRB1-DRB3:xx:xx"
                        if '-' in h_2:
                            h_2 = h_2.split('-')[1]
                        hlaII += h_2 + ','
    return hlaI, hlaII

# -------- FASTA sharding (round-robin) --------
def split_fasta_roundrobin(in_fa, out_root, n_chunks, sample):
    """
    Distribute entries to n_chunks in round-robin order.
    Output:
      {out_root}/chunks/chunk_{i}/{sample}.chunk_{i}.pep
    """
    ensure_dir(os.path.join(out_root, "chunks"))
    buffers = [[] for _ in range(n_chunks)]
    idx = 0
    total = 0
    for h, s in fasta_iter(in_fa):
        buffers[idx].append((h, s))
        idx = (idx + 1) % n_chunks
        total += 1
    chunk_paths = []
    for i in range(n_chunks):
        chunk_dir = os.path.join(out_root, "chunks", f"chunk_{i}")
        ensure_dir(chunk_dir)
        out_fa = os.path.join(chunk_dir, f"{sample}.chunk_{i}.pep")
        write_fasta(buffers[i], out_fa)
        chunk_paths.append(out_fa)
    print(f"[INFO] Split {total} peptide entries into {n_chunks} chunks.")
    return chunk_paths

# -------- Command runner --------
def run_cmd(cmd):
    print("[CMD]", cmd, flush=True)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed (exit {rc}): {cmd}")

def _build_bind_arg(out_root: str, chunk_fa: str) -> str:
    """
    Bind only the host output root and the chunk fasta directory to the same paths inside the container.
    IEDB is NOT bound; the container provides /opt/iedb internally.
    """
    binds = {
        os.path.abspath(out_root),
        os.path.abspath(os.path.dirname(chunk_fa)),
    }
    parts = [f"-B {b}:{b}" for b in sorted(binds)]
    return (" ".join(parts) + " ") if parts else ""

def pvacbind_one_chunk(sample, chunk_idx, chunk_fa, hla_csv,
                       algos_all, e1_lengths, e2_lengths,
                       pvactools_sif, iedb_dir_unused, out_root):
    """
    Run pVACbind via `apptainer exec`. IEDB path uses container's /opt/iedb.
    """
    out_dir = os.path.join(out_root, "chunks", f"chunk_{chunk_idx}")
    ensure_dir(out_dir)

    bind_arg = _build_bind_arg(out_root=out_root, chunk_fa=chunk_fa)

    cmd = (
        f"apptainer exec {bind_arg}"
        f"{pvactools_sif} pvacbind run "
        f"{chunk_fa} {sample} {hla_csv} {algos_all} {out_dir} "
        f"-e1 {e1_lengths} -e2 {e2_lengths} "
        f"--iedb-install-directory /opt/iedb "
        f"-t 1 --fasta-size 100000 "
        f"< /dev/null"
    )
    run_cmd(cmd)

def merge_chunk_results(sample, out_root, n_chunks):
    """
    Merge all chunk combined/{sample}.all_epitopes.tsv files into:
        {out_root}/combined/{sample}.merged.all_epitopes.tsv
    """
    merged_dir = os.path.join(out_root, "combined")
    ensure_dir(merged_dir)
    merged_out = os.path.join(merged_dir, f"{sample}.merged.all_epitopes.tsv")

    candidates = []
    for i in range(n_chunks):
        p = os.path.join(out_root, "chunks", f"chunk_{i}",
                         "combined", f"{sample}.all_epitopes.tsv")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            candidates.append(p)

    if not candidates:
        print("[ERROR] No chunk result files found to merge.")
        return

    wrote_header = False
    total_rows = 0
    used = []
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
                used.append(path)

    if total_rows == 0:
        print("[ERROR] All files contained headers only (no data rows).")
        try:
            os.remove(merged_out)
        except OSError:
            pass
        return

    print(f"[INFO] Merge done: {len(used)} files, {total_rows} rows -> {merged_out}")

def main():
    ap = argparse.ArgumentParser(
        description="Run pVACbind (single command with -e1/-e2) on aeSEP peptides using chunked multiprocessing."
    )
    ap.add_argument("-s", "--sample", required=True, help="Sample ID")
    ap.add_argument("--pep-fasta", required=True, help="Input peptide FASTA (*.pep)")
    ap.add_argument("--hla-file", required=True, help="HLA typing result file (*_final.result.txt)")
    ap.add_argument("-o", "--outdir", required=True, help="Output root (will create {sample}/pvacbind/)")

    # Tool paths
    ap.add_argument("--pvactools", default="pvactools",
                    help="Path to pvactools.sif (Apptainer image)")
    ap.add_argument("--iedb-install-directory", dest="iedb_install_directory", required=False,
                    help="Path to IEDB (kept for CLI parity; container uses /opt/iedb)")

    # Parallelization / sharding
    ap.add_argument("-t", "--threads", type=int, default=10,
                    help="Number of chunks / parallel workers (default: 10)")

    # Algorithms & peptide lengths
    ap.add_argument(
        "--algos",
        default=("BigMHC_EL BigMHC_IM DeepImmuno MHCflurry MHCflurryEL "
                 "MHCnuggetsI MHCnuggetsII NNalign NetMHC NetMHCIIpan "
                 "NetMHCIIpanEL NetMHCpan NetMHCpanEL PickPocket SMM SMMPMBEC"),
        help="pVACbind algorithm list (space-separated; combined Class I/II)"
    )
    ap.add_argument("--e1-lengths", default="8,9,10",
                    help="-e1 (Class I peptide lengths, comma-separated)")
    ap.add_argument("--e2-lengths", default="15",
                    help="-e2 (Class II peptide lengths, comma-separated)")

    args = ap.parse_args()
    sample = args.sample
    pep_fa = args.pep_fasta
    hla_file = args.hla_file
    out_root = os.path.join(args.outdir, sample, "pvacbind")
    ensure_dir(out_root)

    # Parse HLA
    hlaI, hlaII = parse_hlahd_result(hla_file)
    if hlaI == '' and hlaII == '':
        print("[ERROR] No usable HLA alleles found (all were 'Not typed').")
        sys.exit(1)
    HLA = (hlaI + hlaII).strip(',')
    if HLA == '':
        print("[ERROR] Failed to parse any HLA alleles.")
        sys.exit(1)

    # Shard FASTA
    n_chunks = max(1, int(args.threads))
    chunk_paths = split_fasta_roundrobin(pep_fa, out_root, n_chunks, sample)

    # Parallel execution: one pvacbind command per chunk (with -e1/-e2)
    pool = NoDaemonPool(n_chunks)
    tasks = []
    for i, chunk_fa in enumerate(chunk_paths):
        tasks.append(pool.apply_async(
            pvacbind_one_chunk,
            (sample, i, chunk_fa, HLA,
             args.algos, args.e1_lengths, args.e2_lengths,
             args.pvactools, args.iedb_install_directory, out_root)
        ))
    pool.close()
    pool.join()
    for t in tasks:
        t.get()

    # Merge outputs
    merge_chunk_results(sample, out_root, n_chunks)
    print("[DONE] pVACbind finished.")

if __name__ == "__main__":
    main()
