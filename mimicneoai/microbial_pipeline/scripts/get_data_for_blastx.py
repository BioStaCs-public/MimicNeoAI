import os
import gzip
import pandas as pd


def get_data(
    sample,
    output_pathseq,
    output_nucleic,
    pair,
    seq_type,
    **kwargs,
):
    """
    Extract YP-tagged reads from the PathSeq BAM and write them to a gzipped FASTA file.
    Also generate a small summary TSV describing total and YP-tagged read counts.

    Parameters
    ----------
    sample : str
        Sample ID.
    output_pathseq : str
        Directory containing PathSeq outputs (BAM + score files).
    output_nucleic : str
        Directory where nucleic-acid-level outputs will be written.
    pair : bool
        Whether the original sequencing data are paired-end.
    seq_type : str
        Sequencing type label (e.g. 'rna'), used in BAM filename.
    **kwargs :
        Placeholder for unused legacy parameters
        (catalog_genome_rank_file, tax_id_hierarchy_file, etc.).
    """
    sample = str(sample)
    try:
        import pysam
    except ImportError as exc:
        raise ImportError("pysam is required for PathSeq BAM extraction") from exc

    # PathSeq BAM produced by GATK PathSeqPipelineSpark
    bam_path = os.path.join(
        output_pathseq,
        f"{sample}_{seq_type}_output.pathseq.bam"
    )

    if not os.path.exists(bam_path) or os.path.getsize(bam_path) == 0:
        raise FileNotFoundError(f"PathSeq BAM not found or empty: {bam_path}")

    # Output FASTA (YP-selected reads) and summary TSV
    fa_path = os.path.join(output_nucleic, f"{sample}.pathseq_selected.fa.gz")
    summary_tsv = os.path.join(output_nucleic, f"{sample}.pathseq_selected.summary.tsv")

    # Counters
    total_reads = 0
    total_paired_reads = 0
    total_single_reads = 0

    yp_reads = 0
    yp_paired_reads = 0
    yp_single_reads = 0

    fa_reads = 0

    with pysam.AlignmentFile(bam_path, "rb") as bam, gzip.open(fa_path, "wt") as fa_out:
        for read in bam.fetch(until_eof=True):
            total_reads += 1

            # Paired vs single based on SAM flag (0x1)
            if read.is_paired:
                total_paired_reads += 1
            else:
                total_single_reads += 1

            # Only keep reads that have the YP tag (PathSeq-selected microbial reads)
            if not read.has_tag("YP"):
                continue

            yp_reads += 1
            if read.is_paired:
                yp_paired_reads += 1
            else:
                yp_single_reads += 1

            # Build FASTA header: sample|QNAME|FLAG|CIGAR|YP:<value>
            try:
                yp_value = read.get_tag("YP")
            except KeyError:
                # Should not happen if has_tag("YP") is True, but guard just in case
                yp_value = ""

            header_fields = [
                sample,
                read.query_name,
                str(read.flag),
                (read.cigarstring or "*"),
                f"YP:Z:{yp_value}",
            ]
            header = ">" + "|".join(header_fields)
            seq = read.query_sequence or ""

            fa_out.write(header + "\n")
            fa_out.write(seq + "\n")
            fa_reads += 1

    # Compute single-read counts as a sanity check
    # (for paired-end data most reads should be counted as "paired")
    if total_single_reads < 0:
        total_single_reads = 0

    # Collect summary statistics and write to TSV
    summary = {
        "sample": sample,
        "total_reads": total_reads,
        "total_paired_reads": total_paired_reads,
        "total_single_reads": total_single_reads,
        "yp_reads": yp_reads,
        "yp_paired_reads": yp_paired_reads,
        "yp_single_reads": yp_single_reads,
        "fa_reads": fa_reads,
        "bam_path": bam_path,
        "fa_path": fa_path,
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(summary_tsv, sep="\t", index=False)
