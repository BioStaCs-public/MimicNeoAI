# coding=utf-8
import os

def hlahd(sample, configure, pathes, tool):
    """
    Perform HLA typing analysis using HLaHD pipeline
    
    Args:
        sample (str): Sample identifier for processing
        configure (dict): Configuration dictionary containing:
            - path: Input/Output directory paths
            - args: Processing parameters like thread count
            - others: Quality control status
            - step_name: Pipeline step names
        pathes (dict): Paths dictionary containing:
            - freq_data_dir: Frequency data directory
            - HLA_gene: HLA gene reference path
            - dictionary: HLA dictionary file path
        tool (object): Helper object with command execution and logging methods
    """
    # Input directory containing raw data files
    input_path = configure['path']['input_dir'] + "/"
    # Output directory for analysis results
    output_path = configure['path']['output_dir'] + "/"
    thread = configure['args']['thread']
    freq_data_dir = pathes['path']['freq_data_dir']
    HLA_gene = pathes['path']['HLA_gene']
    dictionary = pathes['path']['dictionary']
    QC = configure['others']['QC']
    
    # Output directory configuration
    step_name_hla = configure['step_name']['hla']
    output_hla = output_path + f'/{sample}/{step_name_hla}/'
    cmd_mkdir = f"mkdir -p {output_hla}"
    tool.judge_then_exec(sample, cmd_mkdir, output_hla)
    
    # Determine input files based on QC status
    if QC:  
        in1 = f'{output_path}/{sample}/00.QC/{sample}/{sample}.QC.R1.fq.gz'
        in2 = f'{output_path}/{sample}/00.QC/{sample}/{sample}.QC.R2.fq.gz'
    else:    
        in1 = f'{input_path}/{sample}/{sample}.R1.fq.gz'
        in2 = f'{input_path}/{sample}/{sample}.R2.fq.gz'
    
    # Decompressed file paths
    in1_compress = f'{output_hla}/{sample}/fastq/{sample}.R1.fq'
    in2_compress = f'{output_hla}/{sample}/fastq/{sample}.R2.fq'
    
    # Prepare input files if final result doesn't exist
    if not os.path.exists(f"{output_hla}/{sample}/result/{sample}_final.result.txt"): 
        cmd_mkdir = f"mkdir -p {output_hla}/{sample}/fastq/"
        tool.judge_then_exec(sample, cmd_mkdir, f"{output_hla}/{sample}/fastq/")
        # Decompress input files
        cmd0 = f"gunzip -c {in1} > {in1_compress}"
        tool.judge_then_exec(sample, cmd0, in1_compress)
        cmd1 = f"gunzip -c {in2} > {in2_compress}"
        tool.judge_then_exec(sample, cmd1, in2_compress)
    
    # HLAHD analysis command
    cmd2 = f'hlahd.sh -t {thread} -m 100 -f {freq_data_dir} {in1} {in2} {HLA_gene} {dictionary} {sample} {output_hla}'  
    
    # Cleanup commands
    cmd3 = f'rm {in1_compress} {in2_compress}'
    cmd4 = f"rm {output_hla}/{sample}/mapfile/*"
    cmd5 = f"rm {output_hla}/{sample}/intron/*"
    cmd6 = f"rm {output_hla}/{sample}/exon/*"
    
    # Execute main analysis command
    tool.judge_then_exec(sample, cmd2, f"{output_hla}/{sample}/result/{sample}_final.result.txt")
    
    # Conditional cleanup of intermediate files
    if os.path.exists(in1_compress):
        tool.exec_cmd(cmd3, sample)
    if os.path.exists(f"{output_hla}/{sample}/mapfile/"):
        tool.exec_cmd(cmd4, sample)
    if os.path.exists(f"{output_hla}/{sample}/intron/"):
        tool.exec_cmd(cmd5, sample)
    if os.path.exists(f"{output_hla}/{sample}/exon/"):
        tool.exec_cmd(cmd6, sample)