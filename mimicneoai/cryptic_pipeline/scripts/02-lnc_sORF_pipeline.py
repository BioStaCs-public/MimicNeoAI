#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02-lnc_sORF_pipeline.py

- Supports both "novel" and "known" branches.
- Adds --shared-dir: the known/novel branches can share [0]/[1] (mutually exclusive extraction + sort/index) intermediates.
- known branch: call variants on lncRNA regions via bcftools, write consensus sequences, then run TransDecoder.

Example:
  # Shared intermediate directory
  SHARED="/path/to/results/<SAMPLE>/02-shared"

  python 02-lnc_sORF_pipeline.py -s <SAMPLE> -m known -o ./known \
    --shared-dir "$SHARED" --threads 20 --in-bam /path/to/Aligned.out.bam \
    --ref-dir /path/to/ref --ref-fa /path/to/GRCh38.p3.genome.fa \
    --ref-gtf /path/to/gencode.v23.annotation.gtf \
    --ref-lnc-gtf /path/to/gencode.v23.long_noncoding_RNAs.gtf \
    --bcf-mapq 20 --bcf-baseq 20 --bcf-depth-cap 100000 \
    --bcf-qual-min 30 --bcf-dp-min 10 --bcf-mq-min 20 --bcf-af-min 0.05 \
    --consensus-hap A

  python 02-lnc_sORF_pipeline.py -s <SAMPLE> -m novel -o ./novel \
    --shared-dir "$SHARED" --threads 20 --trinity-cpu 20 --trinity-mem 100G \
    --pid 0.95 --cov 0.70 --ambig-delta 0.05 \
    --in-bam /path/to/Aligned.out.bam
"""

import argparse, os, sys, subprocess, shutil, re, time, gzip, json, hashlib
from pathlib import Path
from collections import Counter


# ------------------ Utility basics ------------------

def ts(): return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg): print(f"[{ts()}] {msg}", flush=True)


def run(cmd, cwd=None, stdout=None):
    log("CMD: " + " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=cwd, stdout=stdout)


def exists(p): return p and Path(p).exists() and Path(p).stat().st_size > 0


def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)


def check_bin(name):
    if shutil.which(name) is None:
        sys.stderr.write(f"[ERR] need `{name}` in PATH\n");
        sys.exit(1)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


QC = []
RNA_VARIANT_CALLING_POLICY_VERSION = "cryptic_known_branch_rna_variant_calling_v1.0"
RNA_VARIANT_CALLING_NORMALIZATION_POLICY = "bcftools norm -f REF -m -any; bcftools norm -d exact"
RNA_VARIANT_CALLING_FLAG_FILTER = "0xF04"  # unmapped + secondary + QC-failed + duplicate + supplementary


def qc_add(section, kvs=None, path=None):
    QC.append(f"## {section}")
    if path: QC.append(f"Path: {path}")
    if kvs:
        if isinstance(kvs, dict):
            for k, v in kvs.items(): QC.append(f"{k}: {v}")
        else:
            for k, v in kvs: QC.append(f"{k}: {v}")
    QC.append("")


def command_version(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", errors="replace")
        return out.splitlines()[0] if out.splitlines() else ""
    except Exception as exc:
        return f"unavailable:{exc.__class__.__name__}"


def output_signature(paths):
    return {label: file_identity(path) for label, path in paths.items()}


def outputs_match_signature(signature):
    if not isinstance(signature, dict):
        return False
    for label, expected in signature.items():
        if not isinstance(expected, dict):
            return False
        path = expected.get("path")
        if not path:
            return False
        observed = file_identity(path)
        for key in ("exists", "size", "sha256"):
            if observed.get(key) != expected.get(key):
                return False
    return True


def write_json_atomic(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


# ------------------ File/text helpers ------------------

def read_gtf_tx2gene(gtf):
    tx2g = {}
    with open(gtf, "r") as fh:
        for line in fh:
            if line.startswith("#"): continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "transcript": continue
            attr = f[8]
            tid = re.search(r'transcript_id "([^"]+)"', attr)
            gid = re.search(r'gene_id "([^"]+)"', attr)
            gn = re.search(r'gene_name "([^"]+)"', attr)
            tid = tid.group(1) if tid else "."
            gid = gid.group(1) if gid else "."
            gn = gn.group(1) if gn else "."
            tx2g[tid] = (gid, gn)
    return tx2g


def gtf_attr_value(attr, key, default="."):
    m = re.search(rf'{re.escape(key)} "([^"]+)"', attr)
    return m.group(1) if m else default


def iter_gtf_rows(gtf, feature=None):
    with open(gtf, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t")
            if len(row) < 9:
                continue
            if feature is not None and row[2] != feature:
                continue
            yield row, line.rstrip("\n")


def write_sorted_bed6(records, out_bed):
    records = sorted(records, key=lambda x: (x[0], int(x[1]), int(x[2])))
    with open(out_bed, "w") as fo:
        for row in records:
            fo.write("\t".join(map(str, row)) + "\n")


def write_fa_with_ann(in_fa, ann_map, out_fa):
    with open(in_fa, "r") as fi, open(out_fa, "w") as fo:
        for line in fi:
            if line.startswith(">"):
                rid = line[1:].strip().split()[0]
                tx, gid, gn = ann_map.get(rid, ("NOVEL", ".", "."))
                if tx == "NOVEL":
                    fo.write(f">{rid}|NOVEL|.|.\n")
                else:
                    fo.write(f">{rid}|{tx}|{gid}|{gn}\n")
            else:
                fo.write(line)


def parse_paf_best_hits(paf, pid_thr, cov_thr, ambig_delta):
    """Return: dict contig -> (best_tx, pid, cov, score, flag in {OK,AMBIG,NOVEL})"""
    best, second, bp, bc, bt = {}, {}, {}, {}, {}
    seen = set()
    with open(paf, "r") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 12: continue
            q, qlen, qstart, qend, t = f[0], int(f[1]), int(f[2]), int(f[3]), f[5]
            nmatch = int(f[9]);
            alnlen = int(f[10])
            pid = (nmatch / float(alnlen)) if alnlen > 0 else 0.0
            cov = ((qend - qstart) / float(qlen)) if qlen > 0 else 0.0
            seen.add(q)
            if pid >= pid_thr and cov >= cov_thr:
                sc = pid * cov
                if sc > best.get(q, -1):
                    second[q] = best.get(q, -1)
                    best[q] = sc;
                    bt[q] = t;
                    bp[q] = pid;
                    bc[q] = cov
                elif sc > second.get(q, -1):
                    second[q] = sc
    out = {}
    for q in seen:
        if q in bt:
            sc1 = best.get(q, 0.0);
            sc2 = second.get(q, 0.0)
            flag = "OK" if (sc1 - sc2) >= ambig_delta else "AMBIG"
            out[q] = (bt[q], bp[q], bc[q], sc1, flag)
        else:
            out[q] = ("NOVEL", None, None, None, "NOVEL")
    return out


def fa_ids(fa):
    ids = []
    with open(fa, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                ids.append(line[1:].split()[0].strip())
    return ids


def qc_bam_counts(bam):
    if not shutil.which("samtools"): return None

    def count(args):
        out = subprocess.check_output(args).decode().strip()
        try:
            return int(out)
        except:
            return None

    total = count(["samtools", "view", "-c", bam])
    mapped = count(["samtools", "view", "-c", "-F", "4", bam])
    proper = count(["samtools", "view", "-c", "-f", "2", bam])
    return total, mapped, proper


def qc_fastq_reads(fq):
    n = 0;
    op = gzip.open if str(fq).endswith(".gz") else open
    with op(fq, "rt", errors="ignore") as fh:
        for _ in fh: n += 1
    return n // 4


def qc_fasta(fa):
    nseq = 0;
    bases = 0
    with open(fa, "r") as fh:
        for line in fh:
            if line.startswith(">"):
                nseq += 1
            else:
                bases += len(line.strip())
    return nseq, bases


def qc_gtf_counts(gtf):
    nt = nx = 0
    with open(gtf, "r") as fh:
        for line in fh:
            if line.startswith("#"): continue
            f = line.split("\t")
            if len(f) < 3: continue
            if f[2] == "transcript":
                nt += 1
            elif f[2] == "exon":
                nx += 1
    return nt, nx


def qc_bed_lines(bed):
    with open(bed, "r") as handle:
        return sum(1 for _ in handle)


def qc_paf_lines(paf): return sum(1 for _ in open(paf, "r"))


def qc_contig2tx_flags(c2tx):
    c = Counter()
    with open(c2tx, "r") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 6: c[f[5]] += 1
    return dict(c)


def qc_fasta_seqs_only(fa): return sum(1 for _ in open(fa, "r") if _.startswith(">"))


# ------------------ Alignment/assembly/traceback ------------------

def step_extract_mapped_primary(in_bam, out_bam, threads, min_mapq=1, exclude_flags=2308):
    """
    Keep only: mapped & primary (exclude unmapped/secondary/supplementary).
    -F 2308 = 4(unmapped) + 256(secondary) + 2048(supplementary)
    """
    log(">> [0] Extract mutually exclusive set: mapped primary (exclude unmapped/secondary/supplementary)")
    if not exists(out_bam):
        cmd = ["samtools", "view", "-@", str(threads), "-b", "-F", str(exclude_flags)]
        if min_mapq and min_mapq > 0:
            cmd += ["-q", str(min_mapq)]
        cmd += [in_bam, "-o", out_bam]
        run(cmd)

    # Do not index here; next step will sort + index
    try:
        subprocess.check_call(["samtools", "quickcheck", out_bam])
    except subprocess.CalledProcessError:
        log("  [WARN] samtools quickcheck failed (file may be truncated).")

    return out_bam


def step_sort_index(in_bam, out_bam, threads):
    log(">> [1] Sort and index")
    if not exists(out_bam):
        run(["samtools", "sort", "-@", threads, in_bam, "-o", out_bam])
    bai = out_bam + ".bai"
    if not Path(bai).exists():
        run(["samtools", "index", out_bam])
    qc = qc_bam_counts(out_bam)
    if qc:
        total, mapped, pp = qc
        log(f"   Sorted BAM: total={total}, mapped={mapped}, proper_pairs={pp}")
        qc_add("Step [1] Sorted BAM", {"total": total, "mapped": mapped, "proper_pairs": pp}, out_bam)
    return out_bam


def step_stringtie_gffcompare(sorted_bam, ref_gtf, threads, out_dir, strandedness="reverse"):
    log(">> [1_1] StringTie + gffcompare")
    st_gtf = str(Path(out_dir) / "stringtie.gtf")
    comp_prefix = str(Path(out_dir) / "comp")
    comp_annot = comp_prefix + ".annotated.gtf"

    strand = str(strandedness).strip().lower()
    # StringTie strandedness:
    # reverse -> --rf
    # forward -> --fr
    # none    -> no extra option
    strand_args = []
    if strand == "reverse":
        strand_args = ["--rf"]
    elif strand == "forward":
        strand_args = ["--fr"]
    elif strand == "none":
        strand_args = []
    else:
        raise ValueError(f"Invalid strandedness: {strandedness}. Use reverse/forward/none")

    if not exists(st_gtf):
        run(["stringtie", sorted_bam, "-G", ref_gtf, "-o", st_gtf, "-p", threads, *strand_args])
    if not exists(comp_annot):
        run(["gffcompare", "-r", ref_gtf, "-o", comp_prefix, st_gtf])
    nt, nx = qc_gtf_counts(st_gtf)
    nt2, nx2 = qc_gtf_counts(comp_annot)
    log(f"   StringTie.gtf: transcripts={nt}, exons={nx}")
    log(f"   comp.annotated.gtf: transcripts={nt2}, exons={nx2}")
    qc_add("Step [1_1] StringTie.gtf", {"transcripts": nt, "exons": nx}, st_gtf)
    qc_add("Step [1_1] gffcompare.annot", {"transcripts": nt2, "exons": nx2}, comp_annot)
    return st_gtf, comp_annot


def step_pick_novel(comp_annot, out_dir, class_codes="u|i|x"):
    log(f">> [1_2] Select novel ({class_codes}) and export BED")
    ensure_dir(out_dir)
    novel_ids = str(Path(out_dir) / "novel.tx.ids")
    novel_gtf = str(Path(out_dir) / "novel_lncRNA.gtf")
    novel_bed = str(Path(out_dir) / "novel_lncRNA.transcripts.bed")
    if not exists(novel_ids):
        keep_codes = set(class_codes.split("|"))
        ids = {
            gtf_attr_value(row[8], "transcript_id")
            for row, _ in iter_gtf_rows(comp_annot, feature="transcript")
            if gtf_attr_value(row[8], "class_code") in keep_codes
            and gtf_attr_value(row[8], "transcript_id") != "."
        }
        with open(novel_ids, "w") as fo:
            for tid in sorted(ids):
                fo.write(tid + "\n")
    n_ids = sum(1 for _ in open(novel_ids)) if exists(novel_ids) else 0
    log(f"   novel.ids: {n_ids}")
    qc_add("Step [1_2] novel.tx.ids", {"novel_transcripts": n_ids}, novel_ids)

    if exists(novel_ids) and not exists(novel_gtf):
        keep = {line.strip() for line in open(novel_ids) if line.strip()}
        with open(novel_gtf, "w") as fo:
            for row, raw in iter_gtf_rows(comp_annot, feature="transcript"):
                if gtf_attr_value(row[8], "transcript_id") in keep:
                    fo.write(raw + "\n")
    if exists(novel_gtf):
        nt, nx = qc_gtf_counts(novel_gtf)
        log(f"   novel.gtf: transcripts={nt}, exons={nx}")
        qc_add("Step [1_2] novel_lncRNA.gtf", {"transcripts": nt, "exons": nx}, novel_gtf)

    if exists(novel_gtf) and not exists(novel_bed):
        records = []
        for row, _ in iter_gtf_rows(novel_gtf, feature="transcript"):
            attr = row[8]
            tid = gtf_attr_value(attr, "transcript_id")
            cc = gtf_attr_value(attr, "class_code")
            rg = gtf_attr_value(attr, "ref_gene_id")
            cr = gtf_attr_value(attr, "cmp_ref")
            name = f"{tid}|{cc}|{rg}|{cr}"
            records.append((row[0], int(row[3]) - 1, row[4], name, ".", row[6]))
        write_sorted_bed6(records, novel_bed)
    if exists(novel_bed):
        lines = qc_bed_lines(novel_bed)
        log(f"   novel.bed lines: {lines}")
        qc_add("Step [1_2] novel_lncRNA.transcripts.bed", {"lines": lines}, novel_bed)
    return novel_ids, novel_gtf, novel_bed


def step_filter_bam(sorted_bam, ref_fa, bed, threads, out_bam):
    log(">> [2] Filter BAM by target regions")
    if not exists(out_bam):
        run(["samtools", "view", "-@", threads, "-b", "-T", ref_fa, "-L", bed, sorted_bam, "-o", out_bam])
        run(["samtools", "index", out_bam])
    qc = qc_bam_counts(out_bam)
    if qc:
        total, mapped, pp = qc
        log(f"   lncRNA.bam: total={total}, mapped={mapped}, proper_pairs={pp}")
        qc_add("Step [2] lncRNA.filtered.bam", {"total": total, "mapped": mapped, "proper_pairs": pp}, out_bam)
    return out_bam


def step_bam2fastq(lnc_bam, threads, out_dir, sample):
    log(">> [3] BAM → FASTQ (paired)")
    name_bam = str(Path(out_dir) / f"{sample}.lncRNA.nameSorted.bam")
    fq1 = str(Path(out_dir) / f"{sample}.lncRNA.R1.fq.gz")
    fq2 = str(Path(out_dir) / f"{sample}.lncRNA.R2.fq.gz")
    if not (exists(fq1) and exists(fq2)):
        run(["samtools", "sort", "-@", threads, "-n", lnc_bam, "-o", name_bam])
        run(["bash", "-lc", f"samtools fastq -@ {threads} -N -1 >(gzip -c > {fq1}) -2 >(gzip -c > {fq2}) {name_bam}"])
    r1 = qc_fastq_reads(fq1);
    r2 = qc_fastq_reads(fq2)
    log(f"   R1 reads: {r1}; R2 reads: {r2}")
    qc_add("Step [3] BAM→FASTQ", {"R1_reads": r1, "R2_reads": r2}, f"{fq1} | {fq2}")
    return fq1, fq2


# ==== 2) Trinity wrapper ====

from pathlib import Path
import os


def step_trinity(fq1, fq2, cpu, mem, out_dir,
                 trinity_mode="native", trinity_sif=None, apptainer_bin="apptainer", sample=""):
    """
    Unified entry: call native or apptainer-based Trinity according to trinity_mode.
    TMP directory is fixed at out_dir/tmp.
    """
    log(">> [4] Trinity assembly")

    out_dir = Path(out_dir).resolve()
    # out_dir = _ensure_trinity_safe_outdir(out_dir)  # Trinity sometimes expects 'trinity' in the path
    trinity_out = f"{out_dir}/{sample}.lncRNA.trinity"
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = f"{trinity_out}.Trinity.fasta"
    if not exists(out_path):
        if trinity_mode == "native":
            run_trinity_native(fq1, fq2, cpu, mem, trinity_out)
        elif trinity_mode == "apptainer":
            if not trinity_sif:
                raise RuntimeError("--trinity-mode apptainer requires --trinity-sif to be provided")
            run_trinity_apptainer(fq1, fq2, cpu, mem, out_dir, tmp_dir, trinity_sif, apptainer_bin, sample)
        else:
            raise ValueError(f"Unknown trinity_mode: {trinity_mode}")

    nseq, nbases = qc_fasta(out_path)
    log(f"   Trinity.fasta QC: contigs={nseq}, bases={nbases}")
    qc_add("Step [4] Trinity.fasta", {"contigs": nseq, "bases": nbases}, str(out_path))
    return str(out_path)


def _ensure_trinity_safe_outdir(out_dir: Path) -> Path:
    """Ensure output path contains 'trinity'; if not, create a child dir accordingly."""
    if "trinity" not in out_dir.name.lower():
        out_dir = out_dir / (out_dir.name + ".trinity_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_trinity_native(fq1, fq2, cpu, mem, out_dir: Path):
    """Call Trinity directly (native)."""
    run([
        "Trinity",
        "--seqType", "fq",
        "--left", fq1,
        "--right", fq2,
        "--CPU", str(cpu),
        "--max_memory", str(mem),
        "--output", str(out_dir)
    ])


def run_trinity_apptainer(fq1, fq2, cpu, mem, out_dir: Path, tmp_dir: Path,
                          trinity_sif: str, apptainer_bin: str = "apptainer", sample: str = ""):
    """Run Trinity within Apptainer; inputs read-only, outputs/tmp writable."""
    fq1 = Path(fq1).resolve()
    fq2 = Path(fq2).resolve()
    in_par1 = fq1.parent
    in_par2 = fq2.parent

    binds = []
    if in_par1 == in_par2:
        binds += [f"{in_par1}:/in:ro"]
        fq1_in = f"/in/{fq1.name}"
        fq2_in = f"/in/{fq2.name}"
    else:
        binds += [f"{in_par1}:/in1:ro", f"{in_par2}:/in2:ro"]
        fq1_in = f"/in1/{fq1.name}"
        fq2_in = f"/in2/{fq2.name}"

    binds += [f"{out_dir}:/out", f"{tmp_dir}:/tmp"]

    cmd = [
        apptainer_bin, "exec",
        "--no-home", "--cleanenv", "--contain",
        "--pwd", "/out",
        "--env", "TMPDIR=/tmp"
    ]
    for b in binds:
        cmd += ["--bind", b]

    cmd += [
        str(trinity_sif),
        "Trinity",
        "--seqType", "fq",
        "--left", fq1_in,
        "--right", fq2_in,
        "--CPU", str(cpu),
        "--max_memory", str(mem),
        "--output", f"/out/{sample}.lncRNA.trinity/"
    ]
    run(cmd)


# ==== 2) Trinity wrapper end ====

def step_trace_to_ref(trinity_fa, ref_fa, ref_gtf, out_dir, threads, pid_thr, cov_thr, ambig_delta):
    log(">> [4a] Align Trinity contigs to reference transcripts (minimap2) and annotate back")
    ensure_dir(out_dir)
    ref_tx_fa = str(Path(out_dir) / "ref.transcripts.fa")
    if not exists(ref_tx_fa):
        run(["gffread", "-w", ref_tx_fa, "-g", ref_fa, ref_gtf])
    nref, nrefb = qc_fasta(ref_tx_fa)
    log(f"   ref.transcripts.fa QC: transcripts={nref}, bases={nrefb}")
    qc_add("Step [4a] ref.transcripts.fa", {"transcripts": nref, "bases": nrefb}, ref_tx_fa)

    tx2gene_tsv = str(Path(out_dir) / "tx2gene.tsv")
    if not exists(tx2gene_tsv):
        tx2g = read_gtf_tx2gene(ref_gtf)
        with open(tx2gene_tsv, "w") as fo:
            for tid, (gid, gn) in sorted(tx2g.items()):
                fo.write(f"{tid}\t{gid}\t{gn}\n")
    else:
        tx2g = read_gtf_tx2gene(ref_gtf)

    paf_out = str(Path(out_dir) / "trinity_vs_tx.paf")
    if not exists(paf_out):
        with open(paf_out, "w") as fo:
            run(["minimap2", "-cx", "asm5", "-N", "10", "-t", threads, ref_tx_fa, trinity_fa], stdout=fo)
    plines = qc_paf_lines(paf_out)
    log(f"   PAF lines: {plines}")
    qc_add("Step [4a] trinity_vs_tx.paf", {"lines": plines}, paf_out)

    hits = parse_paf_best_hits(paf_out, float(pid_thr), float(cov_thr), float(ambig_delta))

    c2tx = str(Path(out_dir) / "contig2tx.tsv")
    c2gene = str(Path(out_dir) / "contig2gene.tsv")
    annot_fa = str(Path(out_dir) / "contigs.annot.fa")
    summary = str(Path(out_dir) / "traceback.summary.txt")

    with open(c2tx, "w") as fo:
        for q, (t, pid, cov, sc, flag) in sorted(hits.items()):
            pid_s = f"{pid:.6f}" if pid is not None else "."
            cov_s = f"{cov:.6f}" if cov is not None else "."
            sc_s = f"{sc:.6f}" if sc is not None else "."
            fo.write(f"{q}\t{t}\t{pid_s}\t{cov_s}\t{sc_s}\t{flag}\n")

    with open(c2gene, "w") as fh2:
        for q, (t, pid, cov, sc, flag) in sorted(hits.items()):
            if t == "NOVEL":
                fh2.write(f"{q}\tNOVEL\t.\t.\t{pid or '.'}\t{cov or '.'}\t{sc or '.'}\t{flag}\n")
            else:
                gid, gn = tx2g.get(t, (".", "."))
                fh2.write(f"{q}\t{t}\t{gid}\t{gn}\t{pid:.6f}\t{cov:.6f}\t{sc:.6f}\t{flag}\n")

    ann_map = {}
    with open(c2gene, "r") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[1] == "NOVEL": continue
            ann_map[f[0]] = (f[1], f[2], f[3])
    write_fa_with_ann(trinity_fa, ann_map, annot_fa)

    total = qc_fasta_seqs_only(trinity_fa)
    flags = qc_contig2tx_flags(c2tx)
    with open(summary, "w") as fo:
        fo.write(f"Total contigs : {total}\n")
        fo.write(f"Mapped (OK)   : {flags.get('OK', 0)}\n")
        fo.write(f"AMBIG         : {flags.get('AMBIG', 0)}\n")
        fo.write(f"NOVEL         : {flags.get('NOVEL', 0)}\n")
        fo.write(f"PID thr       : {pid_thr}\n")
        fo.write(f"COV thr       : {cov_thr}\n")
        fo.write(f"AMBIG delta   : {ambig_delta}\n")

    log(f"   contigs.annot.fa QC: seqs={qc_fasta_seqs_only(annot_fa)}")
    log(f"   contig2tx flags: {flags}")
    qc_add("Step [4a] contig2tx flags", flags, c2tx)
    qc_add("Step [4a] contigs.annot.fa", {"seqs": qc_fasta_seqs_only(annot_fa)}, annot_fa)
    return {
        "ref_tx_fa": ref_tx_fa,
        "paf": paf_out,
        "c2tx": c2tx,
        "c2gene": c2gene,
        "annot_fa": annot_fa,
        "summary": summary,
    }


# ------------------ known branch: bcftools + consensus ------------------

def step_gtf_to_bed_transcripts(gtf, out_bed):
    """Export merged lncRNA exon intervals as BED (0-based, half-open)."""
    log(">> [4k] Generate merged lncRNA exon BED from ref_lnc_gtf")
    if not exists(out_bed):
        intervals = []
        for row, _ in iter_gtf_rows(gtf, feature="exon"):
            intervals.append((row[0], int(row[3]) - 1, int(row[4])))
        intervals.sort(key=lambda x: (x[0], x[1], x[2]))
        merged = []
        for chrom, start, end in intervals:
            if not merged or merged[-1][0] != chrom or start > merged[-1][2]:
                merged.append([chrom, start, end])
            else:
                merged[-1][2] = max(merged[-1][2], end)
        records = [
            (chrom, start, end, f"lncRNA_exon_{idx:06d}", ".", ".")
            for idx, (chrom, start, end) in enumerate(merged, start=1)
        ]
        write_sorted_bed6(records, out_bed)
    lines = qc_bed_lines(out_bed) if Path(out_bed).exists() else 0
    log(f"   lncRNA.exons.merged.bed lines: {lines}")
    qc_add("Step [4k] lncRNA.exons.merged.bed", {"lines": lines}, out_bed)
    return out_bed


def step_known_gffread(ref_fa, ref_lnc_gtf, out_dir, sample):
    log(">> [4k] gffread export (reference) lncRNA transcripts")
    ensure_dir(out_dir)
    fa = str(Path(out_dir) / "lncRNA.transcripts.fa")
    if not exists(fa):
        run(["gffread", "-w", fa, "-g", ref_fa, ref_lnc_gtf])
    nseq, nb = qc_fasta(fa)
    log(f"   lncRNA.transcripts.fa: transcripts={nseq}, bases={nb}")
    qc_add("Step [4k] lncRNA.transcripts.fa", {"transcripts": nseq, "bases": nb}, fa)
    return fa


def parse_info_field(info):
    out = {}
    for token in str(info).split(";"):
        if not token:
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            out[key] = value
        else:
            out[token] = "1"
    return out


def parse_number(value, default=None):
    try:
        text = str(value).strip()
        if not text or text == ".":
            return default
        return float(text)
    except Exception:
        return default


def parse_int_value(value, default=0):
    try:
        text = str(value).strip()
        if not text or text == ".":
            return default
        return int(float(text))
    except Exception:
        return default


def filter_normalized_vcf_by_ad(norm_vcf, flt_vcf, evidence_tsv,
                                qual_min=30, dp_min=10, mq_min=20,
                                af_min=0.05, alt_min=3):
    opener = gzip.open if str(norm_vcf).endswith(".gz") else open
    tmp_vcf = str(Path(flt_vcf).with_suffix("")) + ".tmp.vcf"
    total = kept = 0
    duplicate_keys = set()
    seen_keys = set()
    with opener(norm_vcf, "rt", encoding="utf-8", errors="replace") as inp, \
            open(tmp_vcf, "w") as out_vcf, \
            open(evidence_tsv, "w") as evidence:
        evidence.write(
            "chrom\tpos_1based\tref\talt\tqual\tfilter\tmq\tref_reads\talt_reads\t"
            "total_depth\tvaf\tevent_qc_status\tevent_qc_reason\n"
        )
        for line in inp:
            if line.startswith("#"):
                out_vcf.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            total += 1
            chrom, pos, _rec_id, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
            if "," in alt:
                raise ValueError(f"Normalized VCF still has multiallelic ALT at {chrom}:{pos}:{ref}>{alt}")
            key = (chrom, pos, ref, alt)
            if key in seen_keys:
                duplicate_keys.add(f"{chrom}:{pos}:{ref}>{alt}")
                continue
            seen_keys.add(key)

            qual = parse_number(fields[5], None)
            info = parse_info_field(fields[7])
            mq = parse_number(info.get("MQ", ""), None)
            fmt_keys = fields[8].split(":")
            sample_values = fields[9].split(":")
            sample_map = dict(zip(fmt_keys, sample_values))
            ad_values = [parse_int_value(value, 0) for value in sample_map.get("AD", "").split(",") if value != ""]
            if len(ad_values) < 2:
                raise ValueError(f"VCF record lacks usable FORMAT/AD after normalization: {chrom}:{pos}:{ref}>{alt}")
            ref_reads = ad_values[0]
            alt_reads = ad_values[1]
            total_depth = ref_reads + alt_reads
            vaf = (alt_reads / total_depth) if total_depth else 0.0

            reasons = []
            if fields[6] not in {"PASS", "."}:
                reasons.append("vcf_filter_not_pass")
            if qual is None or qual < qual_min:
                reasons.append("variant_qual_below_threshold")
            if total_depth < dp_min:
                reasons.append("total_depth_below_threshold")
            if vaf < af_min:
                reasons.append("vaf_below_threshold")
            if alt_reads < alt_min:
                reasons.append("alt_reads_below_threshold")
            if mq is None or mq < mq_min:
                reasons.append("mapping_quality_below_threshold")

            status = "pass" if not reasons else "fail"
            evidence.write(
                f"{chrom}\t{pos}\t{ref}\t{alt}\t{fields[5]}\t{fields[6]}\t{info.get('MQ', '')}\t"
                f"{ref_reads}\t{alt_reads}\t{total_depth}\t{vaf:.6g}\t{status}\t{';'.join(reasons)}\n"
            )
            if status == "pass":
                fields[6] = "PASS"
                out_vcf.write("\t".join(fields) + "\n")
                kept += 1
    if duplicate_keys:
        raise ValueError(f"Duplicate exact event keys remain after normalization: {sorted(duplicate_keys)[:5]}")
    run(["bcftools", "view", "-Oz", "-o", flt_vcf, tmp_vcf])
    run(["bcftools", "index", "-f", flt_vcf])
    try:
        os.remove(tmp_vcf)
    except OSError:
        pass
    return {"normalized_records": total, "filtered_records": kept}


def step_call_rnaseq_variants(sorted_bam, ref_fa, bed, out_dir,
                              mapq=20, baseq=20, depth_cap=100000,
                              qual_min=30, dp_min=10, mq_min=20, af_min=0.05,
                              alt_min=3, sample=""):
    """
    Call RNA-seq variants within BED regions + hard-filter.
    - mpileup collects AD/DP with explicit MAPQ/base-quality thresholds.
    - VCF is normalized, split, exact-deduplicated, then filtered by AD-derived
      depth/VAF/ALT reads plus QUAL and INFO/MQ.
    """
    log(">> [4k+] bcftools variant calling (restricted to lncRNA BED)")
    ensure_dir(out_dir)

    # Ensure .bai exists
    bai = sorted_bam + ".bai"
    if not Path(bai).exists():
        run(["samtools", "index", sorted_bam])

    raw_bcf = str(Path(out_dir) / "rna.raw.bcf")
    call_vcf = str(Path(out_dir) / "rna.call.vcf.gz")
    call_vcf_sorted = str(Path(out_dir) / "rna.call.sorted.vcf.gz")
    split_vcf = str(Path(out_dir) / "rna.call.norm.split.vcf.gz")
    norm_vcf = str(Path(out_dir) / "rna.call.norm.split.dedup.vcf.gz")
    flt_vcf = str(Path(out_dir) / "rna.flt.vcf.gz")
    evidence_tsv = str(Path(out_dir) / "rna.variant_evidence.tsv")
    manifest_json = str(Path(out_dir) / "rna.variant_calling.manifest.json")
    legacy_outputs = [raw_bcf, call_vcf, call_vcf_sorted, split_vcf, norm_vcf, flt_vcf, evidence_tsv]

    flag_filter = RNA_VARIANT_CALLING_FLAG_FILTER
    parameters = {
        "mpileup_flag_filter": flag_filter,
        "read_mapq_min": mapq,
        "base_quality_min": baseq,
        "depth_cap": depth_cap,
        "variant_qual_min": qual_min,
        "total_depth_min_ad_derived": dp_min,
        "variant_mq_min": mq_min,
        "vaf_min_ad_derived": af_min,
        "alt_reads_min": alt_min,
        "normalization": RNA_VARIANT_CALLING_NORMALIZATION_POLICY,
    }
    input_signature = {
        "policy_version": RNA_VARIANT_CALLING_POLICY_VERSION,
        "sample": str(sample or ""),
        "sample_bam": file_identity(sorted_bam),
        "reference_fasta": file_identity(ref_fa),
        "exon_bed": file_identity(bed),
        "parameters": parameters,
        "bcftools_version": command_version(["bcftools", "--version"]),
        "calling_script_sha256": sha256_file(Path(__file__)),
    }

    if exists(manifest_json):
        with open(manifest_json, "r", encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        if old_manifest.get("policy_version") != RNA_VARIANT_CALLING_POLICY_VERSION:
            raise RuntimeError(f"Existing RNA variant calling manifest has unsupported policy: {manifest_json}")
        if old_manifest.get("input_signature") != input_signature:
            raise RuntimeError(
                "Existing RNA variant calling manifest does not match current inputs/parameters; "
                f"use a new output directory before rerun: {manifest_json}"
            )
        old_output_signature = old_manifest.get("output_signature")
        if not outputs_match_signature(old_output_signature):
            raise RuntimeError(f"Existing RNA variant calling output_signature no longer matches files: {manifest_json}")
        filter_stats = old_manifest.get("filter_stats", {})
        qc_add("Step [4k+] bcftools (mpileup/call/filter)", {
            "mapq": mapq, "baseq": baseq, "depth_cap": depth_cap,
            "qual_min": qual_min, "dp_min": dp_min, "mq_min": mq_min,
            "af_min": af_min, "alt_min": alt_min,
            "normalized_records": filter_stats.get("normalized_records"),
            "filtered_records": filter_stats.get("filtered_records"),
            "resume": True,
        }, flt_vcf)
        return flt_vcf

    if any(exists(path) for path in legacy_outputs):
        raise RuntimeError(
            "Existing RNA VCF intermediates lack rna.variant_calling.manifest.json; "
            "use a new output directory or remove the old 04k.bcf_consensus outputs before formal rerun."
        )

    # 1) mpileup | call
    if not exists(raw_bcf):
        mp_cmd = ["bcftools", "mpileup", "-Ou", "-f", ref_fa,
                  "-R", bed, "--ff", flag_filter,
                  "-q", str(mapq), "-Q", str(baseq),
                  "-d", str(depth_cap),
                  "-a", "AD,DP",
                  sorted_bam]
        log("CMD(pipeline): " + " ".join(mp_cmd) + " | bcftools call -mv -Ob -o " + raw_bcf)
        p1 = subprocess.Popen(mp_cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(["bcftools", "call", "-mv", "-Ob", "-o", raw_bcf], stdin=p1.stdout)
        p1.stdout.close()
        rc2 = p2.wait();
        rc1 = p1.wait()
        if rc1 != 0 or rc2 != 0:
            raise subprocess.CalledProcessError(rc2, "bcftools call pipeline")

    # 2) BCF -> VCF.GZ
    if not exists(call_vcf):
        run(["bcftools", "view", "-Oz", "-o", call_vcf, raw_bcf])

    # 3) sort + index (avoid "Unsorted positions")
    if not exists(call_vcf_sorted):
        run(["bcftools", "sort", "-Oz", "-o", call_vcf_sorted, call_vcf])
    run(["bcftools", "index", "-f", call_vcf_sorted])

    # 4) normalize/split multiallelics, then exact-deduplicate.
    # bcftools rejects combining "-m -any" and "-d exact" in current releases.
    if not exists(split_vcf):
        run(["bcftools", "norm", "-f", ref_fa, "-m", "-any", "-Oz", "-o", split_vcf, call_vcf_sorted])
    run(["bcftools", "index", "-f", split_vcf])
    if not exists(norm_vcf):
        run(["bcftools", "norm", "-d", "exact", "-Oz", "-o", norm_vcf, split_vcf])
    run(["bcftools", "index", "-f", norm_vcf])

    # 5) AD-derived evidence table and final pass-only VCF
    filter_stats = {"normalized_records": 0, "filtered_records": 0}
    if not exists(flt_vcf):
        filter_stats = filter_normalized_vcf_by_ad(
            norm_vcf, flt_vcf, evidence_tsv,
            qual_min=qual_min, dp_min=dp_min, mq_min=mq_min,
            af_min=af_min, alt_min=alt_min,
        )
    elif not exists(evidence_tsv):
        raise RuntimeError(f"Filtered VCF exists but evidence table is missing: {evidence_tsv}")

    outputs = {
            "raw_bcf": file_identity(raw_bcf),
            "call_vcf_sorted": file_identity(call_vcf_sorted),
            "norm_split_vcf": file_identity(split_vcf),
            "norm_split_dedup_vcf": file_identity(norm_vcf),
            "filtered_vcf": file_identity(flt_vcf),
            "variant_evidence_tsv": file_identity(evidence_tsv),
    }
    manifest = {
        "policy_version": RNA_VARIANT_CALLING_POLICY_VERSION,
        "sample": str(sample or ""),
        "run_status": "complete",
        "input_signature": input_signature,
        "output_signature": outputs,
        "filter_stats": filter_stats,
    }
    write_json_atomic(manifest_json, manifest)

    qc_add("Step [4k+] bcftools (mpileup/call/filter)", {
        "mapq": mapq, "baseq": baseq, "depth_cap": depth_cap,
        "qual_min": qual_min, "dp_min": dp_min, "mq_min": mq_min,
        "af_min": af_min, "alt_min": alt_min,
        "normalized_records": filter_stats.get("normalized_records"),
        "filtered_records": filter_stats.get("filtered_records"),
    }, flt_vcf)
    return flt_vcf


def step_consensus_genome(ref_genome_fa, flt_vcf, hap, out_fa):
    """
    Write VCF to reference genome at whole-genome level to obtain individualized FASTA.
    Note: VCF CHROM names must match those in ref_genome_fa (e.g., chr1 vs 1).
    """
    log(">> [4k++] bcftools consensus (genome)")
    if not exists(out_fa):
        run(["bcftools", "consensus", "-H", hap, "-f", ref_genome_fa, flt_vcf, "-o", out_fa])
    nseq, nb = qc_fasta(out_fa)
    log(f"   genome.consensus.fa: seqs={nseq}, bases={nb}")
    qc_add("Step [4k++] genome.consensus.fa", {"seqs": nseq, "bases": nb}, out_fa)
    return out_fa


def step_extract_lnc_from_genome(genome_consensus_fa, ref_lnc_gtf, out_fa):
    """
    Use the genome-level consensus (as -g) to extract 'sample-specific' lncRNA transcripts from lncRNA GTF.
    """
    log(">> [4k+++] gffread extract consensus lncRNA transcripts")
    if not exists(out_fa):
        run(["gffread", "-w", out_fa, "-g", genome_consensus_fa, ref_lnc_gtf])
    nseq, nb = qc_fasta(out_fa)
    log(f"   lncRNA.consensus.fa: transcripts={nseq}, bases={nb}")
    qc_add("Step [4k+++] lncRNA.consensus.fa", {"transcripts": nseq, "bases": nb}, out_fa)
    return out_fa


# ------------------ sORF steps ------------------

def step_transdecoder(td_input, out_dir, sample, min_aa):
    log(f">> [5] TransDecoder.LongOrfs (m>={min_aa}aa)")
    ensure_dir(out_dir)
    td_out = str(Path(out_dir) / f"{sample}.lncRNA.TransDecoder.LongOrfs")
    longest_pep = str(Path(td_out) / "longest_orfs.pep")
    if not exists(longest_pep):
        run(["TransDecoder.LongOrfs", "-t", td_input, "-m", str(min_aa), "-G", "universal", "--output_dir", td_out])
    npep = qc_fasta_seqs_only(longest_pep)
    log(f"   longest_orfs.pep: entries={npep}")
    qc_add("Step [5] TransDecoder.LongOrfs", {"entries": npep, "min_aa": min_aa}, longest_pep)
    return td_out, longest_pep


def step_filter_sorfs(longest_pep, out_path):
    log(">> [6] Filter sORFs (<100 aa)")
    if not exists(out_path):
        with open(longest_pep, "r") as fi, open(out_path, "w") as fo:
            h = None;
            s = []

            def flush():
                if h:
                    aa_len = sum(len(x) for x in s)
                    if 0 < aa_len < 100:
                        fo.write(h);
                        fo.write("".join(s) + "\n")

            for line in fi:
                if line.startswith(">"):
                    flush();
                    h = line;
                    s = []
                else:
                    s.append(line.strip())
            flush()
    ns = qc_fasta_seqs_only(out_path) if exists(out_path) else 0
    log(f"   sORFs.pep: entries(<100aa)={ns}")
    qc_add("Step [6] sORFs.pep", {"entries_lt_100aa": ns}, out_path)
    return out_path


def print_summary(mode, paths):
    print("\n== Done ==")

    if mode == "novel":
        labels = {
            "sorted_bam": "Sorted BAM",
            "stringtie_gtf": "StringTie GTF",
            "comp_annot": "gffcompare annotated GTF",
            "novel_bed": "novel transcript BED",
            "lnc_bam": "Filtered BAM",
            "fq1": "R1 FASTQ",
            "fq2": "R2 FASTQ",
            "trinity_fa": "Trinity assembly FASTA",
            "ref_tx_fa": "Reference transcripts FASTA",
            "paf": "PAF",
            "c2tx": "contig→tx",
            "c2gene": "contig→tx→gene",
            "annot_fa": "Annotated FASTA",
            "summary": "Traceback summary",
            "longest_pep": "Longest ORFs",
            "sorfs_pep": "sORFs pep",
        }
        order = [
            "sorted_bam", "stringtie_gtf", "comp_annot", "novel_bed", "lnc_bam",
            "fq1", "fq2", "trinity_fa", "ref_tx_fa", "paf", "c2tx", "c2gene",
            "annot_fa", "summary", "longest_pep", "sorfs_pep"
        ]
    else:
        labels = {
            "sorted_bam": "Sorted BAM",
            "lnc_bed": "lncRNA BED",
            "ref_lnc_fa": "Reference lncRNA FASTA",
            "flt_vcf": "Filtered VCF",
            "consensus_fa": "Sample-specific lncRNA FASTA",
            "longest_pep": "Longest ORFs",
            "sorfs_pep": "sORFs pep",
        }
        order = ["sorted_bam", "lnc_bed", "ref_lnc_fa", "flt_vcf", "consensus_fa", "longest_pep", "sorfs_pep"]

    for k in order:
        p = paths.get(k)
        if p and exists(p):
            print(f" - {labels[k]} : {p}")


# ------------------ Args ------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--sample", default="Sample_001_T")
    ap.add_argument("-m", "--mode", default="novel", choices=["novel", "known"])
    ap.add_argument("-o", "--outdir", default=os.getcwd())
    ap.add_argument("--shared-dir", default=None,
                    help="Sample-level shared intermediate dir (used by both known/novel for 01.sort_index)")

    ap.add_argument("--threads", default="20")
    ap.add_argument(
        "--strandedness",
        choices=["reverse", "forward", "none"],
        default="reverse",
        help="StringTie strandedness: reverse->--rf, forward->--fr, none->no extra option",
    )
    ap.add_argument("--trinity-cpu", default="20")
    ap.add_argument("--trinity-mem", default="100G")
    ap.add_argument("--pid", default="0.95")
    ap.add_argument("--cov", default="0.70")
    ap.add_argument("--ambig-delta", default="0.05")

    # refs
    ap.add_argument("--ref-dir", default="/path/to/ref")
    ap.add_argument("--ref-fa", default=None)
    ap.add_argument("--ref-gtf", default=None)
    ap.add_argument("--ref-lnc-gtf", default=None)

    # BAM
    ap.add_argument("--in-bam", required=True, help="Input BAM (STAR output)")

    # TransDecoder
    ap.add_argument("--td-min-aa", type=int, default=10)

    # Mutually exclusive extraction / sorting
    ap.add_argument("--min-mapq", type=int, default=1)
    ap.add_argument("--exclude-flags", type=int, default=2308)

    # known-branch bcftools filters
    ap.add_argument("--bcf-mapq", type=int, default=20)
    ap.add_argument("--bcf-baseq", type=int, default=20)
    ap.add_argument("--bcf-depth-cap", type=int, default=100000)
    ap.add_argument("--bcf-qual-min", type=int, default=30)
    ap.add_argument("--bcf-dp-min", type=int, default=10)
    ap.add_argument("--bcf-mq-min", type=int, default=20)
    ap.add_argument("--bcf-af-min", type=float, default=0.05)
    ap.add_argument("--bcf-alt-min", type=int, default=3)
    ap.add_argument("--consensus-hap", default="A", help="bcftools consensus -H (A/1/2). Default A")
    # === Execution mode & SIF image path ===
    ap.add_argument("--trinity-mode", choices=["native", "apptainer"], default="native",
                    help="How to execute Trinity: native=call Trinity directly; apptainer=run via Apptainer (requires --trinity-sif)")
    ap.add_argument("--trinity-sif", default=None,
                    help="Path to Trinity SIF image (required when --trinity-mode apptainer)")
    ap.add_argument("--apptainer", default="apptainer",
                    help="Apptainer executable name/path (default: apptainer)")
    return ap.parse_args()


# ------------------ Main ------------------

def main():
    a = parse_args()

    # refs
    ref_fa = a.ref_fa or os.path.join(a.ref_dir, "GRCh38.p3.genome.fa")
    ref_gtf = a.ref_gtf or os.path.join(a.ref_dir, "gencode.v23.annotation.gtf")
    ref_lnc = a.ref_lnc_gtf or os.path.join(a.ref_dir, "gencode.v23.long_noncoding_RNAs.gtf")

    # bins
    check_bin("gffread")
    check_bin("samtools")
    if a.mode == "novel":
        for b in ("stringtie", "gffcompare", "minimap2"): check_bin(b)
    else:
        check_bin("bcftools")

    # directories (d01 can be shared)
    shared_root = a.shared_dir if a.shared_dir else a.outdir
    d01 = Path(shared_root, "01.sort_index")  # shared by novel/known
    d01_1 = Path(a.outdir, "01a.novel_lncRNA")  # novel
    d02 = Path(a.outdir, "02.filter_lncRNA_bam")  # novel
    d03 = Path(a.outdir, "03.bam2fastq")  # novel
    d04 = Path(a.outdir, "04.trinity_asm")  # novel
    d04a = Path(a.outdir, "04a.trace_to_ref")  # novel
    d04k = Path(a.outdir, "04.known_tx_fa")  # known (reference export)
    d04k2 = Path(a.outdir, "04k.bcf_consensus")  # known (bcftools outputs)
    d05 = Path(a.outdir, "05.transdecoder_orf")
    d06 = Path(a.outdir, "06.filter_sORFs")
    for d in (d05, d06, d01): ensure_dir(d)
    if a.mode == "novel":
        for d in (d01_1, d02, d03, d04, d04a): ensure_dir(d)
    else:
        for d in (d04k, d04k2): ensure_dir(d)

    in_bam = a.in_bam
    out_paths = {}
    t0 = time.time()

    # Always perform [0]/[1] (skip if already present); shared by known/novel
    assert exists(in_bam), f"BAM not found: {in_bam}"
    mapped_primary_bam = str(Path(d01) / f"{a.sample}.host.mapped.primary.bam")
    step_extract_mapped_primary(in_bam, mapped_primary_bam, a.threads,
                                min_mapq=a.min_mapq, exclude_flags=a.exclude_flags)
    sorted_bam = str(Path(d01) / f"{a.sample}.Aligned.out.sorted.bam")
    step_sort_index(mapped_primary_bam, sorted_bam, a.threads)
    out_paths["sorted_bam"] = sorted_bam

    if a.mode == "novel":
        assert exists(ref_gtf), f"ref gtf missing: {ref_gtf}"

        # [1_1] ST + gffcompare
        st_gtf, comp_annot = step_stringtie_gffcompare(
            sorted_bam,
            ref_gtf,
            a.threads,
            d01_1,
            strandedness=a.strandedness,
        )
        out_paths["stringtie_gtf"] = st_gtf;
        out_paths["comp_annot"] = comp_annot

        # [1_2] novel bed
        _, _, novel_bed = step_pick_novel(comp_annot, d01_1, class_codes="u|i|x")
        out_paths["novel_bed"] = novel_bed

        # [2] filter BAM
        lnc_bam = str(Path(d02) / f"{a.sample}.lncRNA.sorted.bam")
        step_filter_bam(sorted_bam, ref_fa, novel_bed, a.threads, lnc_bam)
        out_paths["lnc_bam"] = lnc_bam

        # [3] BAM→FASTQ
        fq1, fq2 = step_bam2fastq(lnc_bam, a.threads, d03, a.sample)
        out_paths["fq1"], out_paths["fq2"] = fq1, fq2

        # [4] Trinity
        trinity_fa = step_trinity(
            fq1, fq2,
            cpu=a.trinity_cpu,
            mem=a.trinity_mem,
            out_dir=str(Path(d04)),
            trinity_mode=a.trinity_mode,
            trinity_sif=a.trinity_sif,
            apptainer_bin=a.apptainer,
            sample=a.sample
        )
        out_paths["trinity_fa"] = trinity_fa

        # [4a] traceback/annotation
        trace = step_trace_to_ref(trinity_fa, ref_fa, ref_gtf, d04a, a.threads, a.pid, a.cov, a.ambig_delta)
        out_paths.update(trace)
        td_input_for_td = trace["annot_fa"]

    else:
        # ===== known branch: bcftools + consensus on lncRNA regions =====
        assert exists(ref_lnc), f"lncRNA gtf missing: {ref_lnc}"

        # Export reference lncRNA transcripts (for consensus -f)
        ref_lnc_fa = step_known_gffread(ref_fa, ref_lnc, d04k, a.sample)
        out_paths["ref_lnc_fa"] = ref_lnc_fa

        # Generate merged exon BED from ref_lnc_gtf (restrict calling region)
        lnc_bed = str(Path(d04k) / "lncRNA.exons.merged.bed")
        step_gtf_to_bed_transcripts(ref_lnc, lnc_bed)
        out_paths["lnc_bed"] = lnc_bed

        # bcftools call + filter on sorted_bam (with .bai ensured)
        flt_vcf = step_call_rnaseq_variants(
            sorted_bam, ref_fa, lnc_bed, d04k2,
            mapq=a.bcf_mapq, baseq=a.bcf_baseq, depth_cap=a.bcf_depth_cap,
            qual_min=a.bcf_qual_min, dp_min=a.bcf_dp_min, mq_min=a.bcf_mq_min,
            af_min=a.bcf_af_min, alt_min=a.bcf_alt_min, sample=a.sample
        )
        out_paths["flt_vcf"] = flt_vcf

        # 1) genome-level consensus
        genome_consensus_fa = str(Path(d04k2) / "genome.consensus.fa")
        step_consensus_genome(ref_fa, flt_vcf, a.consensus_hap, genome_consensus_fa)
        out_paths["genome_consensus_fa"] = genome_consensus_fa

        # 2) extract sample-specific lncRNA transcripts using consensus genome + lncRNA GTF
        consensus_fa = str(Path(d04k2) / "lncRNA.consensus.fa")
        step_extract_lnc_from_genome(genome_consensus_fa, ref_lnc, consensus_fa)
        out_paths["consensus_fa"] = consensus_fa

        # 3) input for TransDecoder
        td_input_for_td = consensus_fa

    # [5] TransDecoder.LongOrfs
    td_out, longest_pep = step_transdecoder(td_input_for_td, d05, a.sample, a.td_min_aa)
    out_paths["longest_pep"] = longest_pep

    # [6] sORF <100aa
    sorfs = str(Path(d06) / f"{a.sample}.lncRNA.sORFs.pep")
    step_filter_sorfs(longest_pep, sorfs)
    out_paths["sorfs_pep"] = sorfs

    # Summary
    elapsed = time.time() - t0
    log(f"Pipeline done in {elapsed:.1f}s")
    qc_add("Pipeline", {"elapsed_seconds": f"{elapsed:.1f}"})
    qc_file = Path(a.outdir) / "QC.summary.txt"
    with open(qc_file, "w") as fo:
        fo.write("# QC Summary\n\n");
        fo.write("\n".join(QC))
    log(f"QC summary written: {qc_file}")

    print_summary(a.mode, out_paths)


if __name__ == "__main__":
    main()
