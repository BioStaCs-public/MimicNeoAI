# coding=utf-8
import os
import shlex
import shutil

def is_directory_empty(dir_path):
    """Check if a directory is empty.
    
    Args:
        dir_path (str): Path to directory to check
        
    Returns:
        bool: True if directory is empty, False otherwise
    """
    return not os.listdir(dir_path)

def resolve_hlahd_script(configured_path, freq_data_dir):
    candidates = []
    if configured_path:
        candidates.append(configured_path)
    which_path = shutil.which("hlahd.sh")
    if which_path:
        candidates.append(which_path)

    hla_root = os.path.dirname(os.path.abspath(str(freq_data_dir).rstrip("/")))
    candidates.extend([
        os.path.join(hla_root, "bin", "hlahd.sh"),
        os.path.join(hla_root, "hlahd.sh"),
    ])

    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        candidate = os.path.abspath(os.path.expanduser(candidate))
        checked.append(candidate)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "HLA-HD executable hlahd.sh was not found or is not executable. "
        "Configure database.common.HLA.HLAHD_SCRIPT. "
        f"Checked: {', '.join(checked)}"
    )

def hlahd(run_sample_id, sample, configure, paths, tool):
    """
    Perform HLA typing analysis using HLaHD pipeline
    
    Args:
        sample (str): Sample identifier
        configure (dict): Configuration dictionary with path/parameter settings
        paths (dict): Dictionary containing tool/file paths
        tool (object): Helper object with execution and logging methods
    """
    # Input directory containing raw files
    input_path = configure['path']['input_dir'] + "/"
    # Output directory for analysis results
    output_path = configure['path']['output_dir'] + "/"
    # Processing parameters
    thread = configure['args']['thread']
    #time_out = configure['others']['time_out']
    
    # Reference data paths
    hla_config = paths['database']['common']['HLA']
    freq_data_dir = hla_config['FREQ_DATA_DIR']
    HLA_gene = hla_config['HLA_GENE']
    dictionary = hla_config['DICTIONARY']
    hla_gen = hla_config['BOWTIE2_INDEX']
    hlahd_bin = resolve_hlahd_script(hla_config.get('HLAHD_SCRIPT'), freq_data_dir)
    hlahd_bin_dir = os.path.dirname(hlahd_bin)
    
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
    cmd_6 = (
        f'PATH={shlex.quote(hlahd_bin_dir)}:$PATH {shlex.quote(hlahd_bin)} '
        f'-t {thread} -f {freq_data_dir} '
        f'{fastq_dir}/{sample}.hla.1.fastq {fastq_dir}/{sample}.hla.2.fastq '
        f'{HLA_gene} {dictionary} {sample} {output_hla}'
    )
    
    # Cleanup commands
    cmd_7 = f'rm {fastq_dir}/*'
    cmd_8 = f"rm {output_hla}/{sample}/mapfile/*"
    cmd_9 = f"rm {output_hla}/{sample}/intron/*"
    cmd_10 = f"rm {output_hla}/{sample}/exon/*"

    # Execute pipeline steps conditionally
    final_result = f"{output_hla}/{sample}/result/{sample}_final.result.txt"
    if not os.path.exists(final_result):
        tool.judge_then_exec(run_sample_id, cmd_0, fastq_dir, display_name="Prepare HLA typing FASTQ directory")
        tool.judge_then_exec(run_sample_id, cmd_1, f"{fastq_dir}/{sample}.hla.1.fastq", display_name="HLA read alignment")
        tool.judge_then_exec(run_sample_id, cmd_2, f"{fastq_dir}/{sample}.mapped.sam", display_name="Extract mapped HLA reads")
        tool.judge_then_exec(run_sample_id, cmd_3, f"{fastq_dir}/{sample}.hlatmp.1.fastq", display_name="Convert HLA reads to FASTQ")
        tool.judge_then_exec(run_sample_id, cmd_4, f"{fastq_dir}/{sample}.hla.1.fastq", display_name="Format HLA read 1 FASTQ")
        tool.judge_then_exec(run_sample_id, cmd_5, f"{fastq_dir}/{sample}.hla.2.fastq", display_name="Format HLA read 2 FASTQ")
        tool.judge_then_exec(run_sample_id, cmd_6, final_result, display_name="HLA-HD typing")
    else:
        tool.write_log(f"[{sample}] HLA-HD final result already exists, skip: {final_result}", "warn")

    # Conditional cleanup of intermediate files
    mapfile_dir = f"{output_hla}/{sample}/mapfile/"
    intron_dir = f"{output_hla}/{sample}/intron/"
    exon_dir = f"{output_hla}/{sample}/exon/"
    
    if os.path.exists(f'{fastq_dir}/{sample}.hlamap.sam'):
        tool.exec_cmd(cmd_7, run_sample_id, display_name="Clean HLA typing FASTQ intermediates")
    if os.path.exists(mapfile_dir) and not is_directory_empty(mapfile_dir):
        tool.exec_cmd(cmd_8, run_sample_id, display_name="Clean HLA typing mapfile intermediates")
    if os.path.exists(intron_dir) and not is_directory_empty(intron_dir):
        tool.exec_cmd(cmd_9, run_sample_id, display_name="Clean HLA typing intron intermediates")
    if os.path.exists(exon_dir) and not is_directory_empty(exon_dir):
        tool.exec_cmd(cmd_10, run_sample_id, display_name="Clean HLA typing exon intermediates")
