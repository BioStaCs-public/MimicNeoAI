# coding=utf-8
import os
import traceback

def fastp(run_sample_id,sample, configure, pathes, tool):
    """
    Process FASTQ files using fastp for quality control
    
    Args:
        sample (str): Sample identifier/name
        configure (dict): Configuration dictionary containing path and parameter settings
        pathes (dict): Dictionary containing tool paths
        tool (object): Helper object with execution and logging methods
    """
    try:
        # Input directory containing raw FASTQ files
        input_dir = configure['path']['input_dir'] + "/"
        # Output directory for processed files
        output_dir = configure['path']['output_dir'] + "/"
        # Path to fastp executable
        # Number of threads to use
        thread = configure['args']['thread']
        # Sequencing type: True for paired-end, False for single-end
        pair = configure['others']['pair']
        # Quality control output directory for this sample
        qc_dir = f"{output_dir}/{sample}/00.QC/{sample}/"
        
        # Create output directory if not exists
        tool.judge_then_exec(
            run_sample_id,
            f"mkdir -p {qc_dir}",
            qc_dir,
            display_name="Prepare QC directory",
        )

        if pair:
            # Paired-end processing command
            cmd = f"fastp -w {thread} -i {input_dir}/{sample}/{sample}.R1.fq.gz -I {input_dir}/{sample}/{sample}.R2.fq.gz -o {qc_dir}/{sample}.QC.R1.fq.gz -O {qc_dir}/{sample}.QC.R2.fq.gz -j {qc_dir}/{sample}.fastp.json -h {qc_dir}/{sample}.fastp.html"
            tool.judge_then_exec(
                run_sample_id,
                cmd,
                f"{qc_dir}/{sample}.QC.R1.fq.gz",
                display_name="fastp QC",
            )
        else:
            # Single-end processing command
            cmd = f"fastp -w {thread} -i {input_dir}/{sample}/{sample}.fq.gz -o {qc_dir}/{sample}.QC.fq.gz -j {qc_dir}/{sample}.fastp.json -h {qc_dir}/{sample}.fastp.html"
            tool.judge_then_exec(
                run_sample_id,
                cmd,
                f"{qc_dir}/{sample}.QC.fq.gz",
                display_name="fastp QC",
            )
        
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"fastp error occurred: {error_message}")
        tool.write_log(f"fastp error occurred: {error_message}","error")
        raise
