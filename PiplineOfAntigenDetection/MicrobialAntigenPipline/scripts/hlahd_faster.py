# coding=utf-8
import os

def is_directory_empty(dir_path):
    """Check if a directory is empty.
    
    Args:
        dir_path (str): Path to directory to check
        
    Returns:
        bool: True if directory is empty, False otherwise
    """
    return not os.listdir(dir_path)

def hlahd(sample, configure, pathes, tool):
    """
    Perform HLA typing analysis using HLaHD pipeline
    
    Args:
        sample (str): Sample identifier
        configure (dict): Configuration dictionary with path/parameter settings
        pathes (dict): Dictionary containing tool/file paths
        tool (object): Helper object with execution and logging methods
    """
    # Input directory containing raw files
    input_path = configure['path']['input_dir'] + "/"
    # Output directory for analysis results
    output_path = configure['path']['output_dir'] + "/"
    # Processing parameters
    thread = configure['args']['thread']
    time_out = configure['others']['time_out']
    
    # Reference data paths
    freq_data_dir = pathes['path']['freq_data_dir']
    HLA_gene = pathes['path']['HLA_gene']
    dictionary = pathes['path']['dictionary']
    hla_gen = pathes['path']['hla_gen']
    
    # Output directory configuration
    step_name_hla = configure['step_name']['hla']
    output_hla = output_path + f'/{sample}/{step_name_hla}/'
    fastq_dir = f"{output_hla}/{sample}/fastq/"
    
    # Determine input files based on QC status
    QC = configure['others']['QC']
    if QC:  
        in1 = f'{output_path}/{sample}/00.QC/{sample}/{sample}.QC.R1.fq.gz'
        in2 = f'{output_path}/{sample}/00.QC/{sample}/{sample}.QC.R2.fq.gz'
    else:    
        in1 = f'{input_path}/{sample}/{sample}.R1.fq.gz'
        in2 = f'{input_path}/{sample}/{sample}.R2.fq.gz'

    # Pipeline command definitions
    # Create output directory
    cmd_0 = f"mkdir -p {fastq_dir}"
    # Bowtie2 alignment command
    cmd_1 = f"bowtie2 -p {thread} -x {hla_gen} -1 {in1} -2 {in2} -S {fastq_dir}/{sample}.hlamap.sam"
    # SAMtools view command for mapped reads
    cmd_2 = f"samtools view -@ {thread} -h -F 4 {fastq_dir}/{sample}.hlamap.sam > {fastq_dir}/{sample}.mapped.sam"
    # Convert SAM to FASTQ
    cmd_3 = f"samtools fastq -@ {thread} -1 {fastq_dir}/{sample}.hlatmp.1.fastq -2 {fastq_dir}/{sample}.hlatmp.2.fastq -0 /dev/null -s /dev/null -n {fastq_dir}/{sample}.mapped.sam"
    
    # FASTQ processing commands
    awk_cmd_1 = "'{if(NR%4 == 1){O=$0;gsub(\"/1\",\" 1\",O);print O}else{print $0}}'"
    awk_cmd_2 = "'{if(NR%4 == 1){O=$0;gsub(\"/2\",\" 2\",O);print O}else{print $0}}'"
    cmd_4 = f"cat {fastq_dir}/{sample}.hlatmp.1.fastq |awk {awk_cmd_1} > {fastq_dir}/{sample}.hla.1.fastq"
    cmd_5 = f"cat {fastq_dir}/{sample}.hlatmp.2.fastq |awk {awk_cmd_2} > {fastq_dir}/{sample}.hla.2.fastq"
    
    # HLaHD main analysis command
    cmd_6 = f'hlahd.sh -t {thread} -m 30 -f {freq_data_dir} {fastq_dir}/{sample}.hla.1.fastq {fastq_dir}/{sample}.hla.2.fastq {HLA_gene} {dictionary} {sample} {output_hla}'  
    
    # Cleanup commands
    cmd_7 = f'rm {fastq_dir}/{sample}.hlamap.sam'
    cmd_8 = f"rm {output_hla}/{sample}/mapfile/*"
    cmd_9 = f"rm {output_hla}/{sample}/intron/*"
    cmd_10 = f"rm {output_hla}/{sample}/exon/*"

    # Execute pipeline steps conditionally
    if not os.path.exists(f"{output_hla}/{sample}/result/{sample}_final.result.txt"): 
        tool.judge_then_exec(sample, cmd_0, fastq_dir)
        tool.judge_then_exec(sample, cmd_1, f"{fastq_dir}/{sample}.hla.1.fastq")
        tool.judge_then_exec(sample, cmd_2, f"{fastq_dir}/{sample}.mapped.sam")
        tool.judge_then_exec(sample, cmd_3, f"{fastq_dir}/{sample}.hlatmp.1.fastq")
        tool.judge_then_exec(sample, cmd_4, f"{fastq_dir}/{sample}.hla.1.fastq")
        tool.judge_then_exec(sample, cmd_5, f"{fastq_dir}/{sample}.hla.2.fastq")
        tool.judge_then_exec_with_time(sample, cmd_6, f"{output_hla}/{sample}/result/{sample}_final.result.txt", time_out)

    # Conditional cleanup of intermediate files
    mapfile_dir = f"{output_hla}/{sample}/mapfile/"
    intron_dir = f"{output_hla}/{sample}/intron/"
    exon_dir = f"{output_hla}/{sample}/exon/"
    
    if os.path.exists(f'{fastq_dir}/{sample}.hlamap.sam'):
        tool.exec_cmd(cmd_7, sample)
    if os.path.exists(mapfile_dir) and not is_directory_empty(mapfile_dir):
        tool.exec_cmd(cmd_8, sample)
    if os.path.exists(intron_dir) and not is_directory_empty(intron_dir):
        tool.exec_cmd(cmd_9, sample)
    if os.path.exists(exon_dir) and not is_directory_empty(exon_dir):
        tool.exec_cmd(cmd_10, sample)