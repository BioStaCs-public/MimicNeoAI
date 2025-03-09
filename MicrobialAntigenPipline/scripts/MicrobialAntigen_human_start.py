# coding=utf-8
import sys 
import os
import traceback
sys.path.append("./scripts/")
from get_data_for_blastx import get_data
from get_data_for_pvacbind import get_data_for_pvacbind
from pvacbind import pvacbind, get_hlahd_results
from datetime import datetime
sys.path.append("../functions/")
from fastp import fastp

def HostSequencesRemoving(sample, configure, pathes, tool):
    """Remove host sequences using hg38 and T2T references
    Args:
        sample: Sample ID string
        configure: Configuration dictionary
        pathes: Paths dictionary
        tool: Execution tool object
    """
    sample = str(sample)
    # Input directory configuration
    input_path = configure['path']['input_dir'] + "/"  # Input files directory
    output_path = configure['path']['output_dir'] + "/"  # Output directory
    thread = configure['args']['thread']  # CPU threads per sample
    mem_perthread = configure['args']['mem_perthread']  # Memory per thread
    pair = configure['others']['pair']  # Paired-end sequencing flag
    seq_type = configure['others']['seq_type']  # Pathseq parameter
    QC = configure['others']['QC']  # Quality control status
    
    # Reference genome paths
    host_fa_hg38 = pathes['path']['host_fa_hg38']
    host_fa_t2t = pathes['path']['host_fa_t2t']
        
    # Output directories
    step_name_qc = configure['step_name']['QC']
    step_name_hg38 = configure['step_name']['hg38']
    step_name_t2t = configure['step_name']['t2t']
    output_qc = output_path + f'{sample}/{step_name_qc}/'
    output_hg38 = output_path + f'{sample}/{step_name_hg38}/'
    output_t2t = output_path+ f'{sample}/{step_name_t2t}/'

    # Alignment files
    sam_file = f"{output_hg38}{sample}_{seq_type}_hg38_align.sam"
    bam_file = f"{output_hg38}{sample}_{seq_type}_hg38_align.bam"
    sorted_bam_file = f"{output_hg38}{sample}_{seq_type}_hg38_align_sort.bam"
    
    # Create output directory
    cmd_mkdir = f"mkdir -p {output_hg38}"
    tool.judge_then_exec(sample,cmd_mkdir,output_hg38)
    
    if not os.path.exists(f"{output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam"):
        # Alignment command construction
        if pair:
            if QC:
                input_path = output_qc                
                cmd1_1 = f"bwa mem -q -t {thread} -R \'@RG\\tID:{sample}\\tSM:{sample}\' {host_fa_hg38} {input_path}/{sample}/{sample}.QC.R1.fq.gz {input_path}/{sample}/{sample}.QC.R2.fq.gz > {sam_file}"
            else:
                cmd1_1 = f'bwa mem -q -t {thread} -R \'@RG\\tID:{sample}\\tSM:{sample}\' {host_fa_hg38} {input_path}/{sample}/{sample}.R1.fq.gz {input_path}/{sample}/{sample}.R2.fq.gz > {sam_file}'
        else:
            if QC:
                input_path = output_qc
                cmd1_1 = f'bwa mem -q -t {thread} -R \'@RG\\tID:{sample}\\tSM:{sample}\' {host_fa_hg38} {input_path}/{sample}/{sample}.QC.fq.gz > {sam_file}'
            else:
                cmd1_1 = f'bwa mem -q -t {thread} -R \'@RG\\tID:{sample}\\tSM:{sample}\' {host_fa_hg38} {input_path}/{sample}/{sample}.fq.gz > {sam_file}'

        # SAM processing pipeline
        cmd1_2 = f'samtools view -bS -@ {thread} {sam_file} > {bam_file}'
        cmd1_3 = f"rm {sam_file}"
        cmd1_4 = f'samtools sort -@ {thread} -m {mem_perthread} -o {sorted_bam_file} {bam_file}'
        cmd1_5 = f"rm {bam_file}"
        tool.judge_then_exec(sample,cmd1_1,sam_file)
        tool.judge_then_exec(sample,cmd1_2,bam_file)
        
        # Preserved commented command
        # if os.path.exists(bam_file):
        #      tool.exec_cmd(cmd1_3,sample)
        # tool.judge_then_exec(sample,cmd1_4,sorted_bam_file)
        # if os.path.exists(sorted_bam_file):
        #      tool.exec_cmd(cmd1_5,sample)
        
        # Generate alignment statistics
        cmd_samFlag = f'samtools flagstat -@ {thread} {sorted_bam_file} > {sorted_bam_file}.flagstat.txt'
        tool.judge_then_exec(sample,cmd_samFlag,f'{sorted_bam_file}.flagstat.txt')
        
        # Extract unmapped reads
        cmd2 = f'samtools view -b -@ {thread} -f 4 -o {output_hg38}{sample}_{seq_type}_hg38_unmap.bam {sorted_bam_file}'
        tool.judge_then_exec(sample,cmd2,f'{output_hg38}{sample}_{seq_type}_hg38_unmap.bam')
        
        # File cleanup commands (preserved as comments)
        cmd2_1 = f"rm {sorted_bam_file}"
        if pair:
            cmd2_2 = f"rm {input_path}/{sample}/{sample}.QC.R1.fq.gz {input_path}/{sample}/{sample}.QC.R2.fq.gz"
        else:
            cmd2_2 = f"rm {input_path}/{sample}/{sample}.QC.fq.gz"
            
        # Preserved commented command    
        # if os.path.exists(f'{output_hg38}{sample}_{seq_type}_hg38_unmap.bam'):
        #     tool.exec_cmd(cmd2_1,sample)
        # if QC:
        #     tool.exec_cmd(cmd2_2,sample)
        
        # Convert BAM to FASTQ
        if pair: 
            cmd3 = f'samtools fastq -@ {thread} {output_hg38}{sample}_{seq_type}_hg38_unmap.bam -1 {output_hg38}{sample}_{seq_type}_hg38_unmap.R1.fq -2 {output_hg38}{sample}_{seq_type}_hg38_unmap.R2.fq -s /dev/null'
            tool.judge_then_exec(sample,cmd3,f'{output_hg38}{sample}_{seq_type}_hg38_unmap.R1.fq')
        else:
            cmd3 = f'samtools fastq -@ {thread} {output_hg38}{sample}_{seq_type}_hg38_unmap.bam > {output_hg38}{sample}_{seq_type}_hg38_unmap.fq'
            tool.judge_then_exec(sample,cmd3,f'{output_hg38}{sample}_{seq_type}_hg38_unmap.fq')
        
        # T2T alignment processing
        cmd_mkdir = f"mkdir -p {output_t2t}"
        tool.judge_then_exec(sample,cmd_mkdir,output_t2t)
        
        if pair:
            cmd4 = f'bwa mem -q -t {thread} -R \'@RG\\tID:{sample}\\tSM:{sample}\' {host_fa_t2t} {output_hg38}{sample}_{seq_type}_hg38_unmap.R1.fq {output_hg38}{sample}_{seq_type}_hg38_unmap.R2.fq | samtools view -bS -@ {thread} | samtools sort -@ {thread} -m {mem_perthread} -o {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_align_sort.bam'
        else:
            cmd4 = f'bwa mem -q -t {thread} -R \'@RG\\tID:{sample}\\tSM:{sample}\' {host_fa_t2t} {output_hg38}{sample}_{seq_type}_hg38_unmap.fq | samtools view -bS -@ {thread} | samtools sort -@ {thread} -m {mem_perthread} -o {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_align_sort.bam'
        tool.judge_then_exec(sample,cmd4,f'{output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_align_sort.bam')
        
        # Final unmapped reads extraction
        cmd6 = f'samtools view -b -@ {thread} -f 4 -o {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_align_sort.bam'
        tool.judge_then_exec(sample,cmd6,f'{output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam')
    
    # Final statistics generation
    cmd_samFlag = f'samtools flagstat -@ {thread} {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam > {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam.flagstat.txt'
    tool.judge_then_exec(sample,cmd_samFlag,f'{output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam.flagstat.txt')

def MicrobialTaxasQuantification(sample,configure,pathes,tool):
    """Perform microbial taxonomy quantification
    Args:
        sample: Sample ID string
        configure: Configuration dictionary
        pathes: Paths dictionary
        tool: Execution tool object
    """
    sample = str(sample)
    # Configuration parameters
    tmp_dir = configure['path']['tmp_dir'] + "/"  # Temporary directory
    output_path = configure['path']['output_dir'] + "/"  # Output directory
    thread = configure['args']['thread']  # CPU threads
    mem = configure['args']['mem']  # Memory allocation
    pair = configure['others']['pair']  # Sequencing type
    seq_type = configure['others']['seq_type']  # Analysis type
    match_length_threshold = configure['others']['match_length_threshold']  # Alignment threshold
    score_threshold = configure['others']['score_threshold']  # Score cutoff
    
    # Reference data paths
    host_img_hg38 = pathes['path']['host_img_hg38']
    host_kmer_hg38 = pathes['path']['host_kmer_hg38']
    host_img_t2t = pathes['path']['host_img_t2t']
    host_kmer_t2t = pathes['path']['host_kmer_t2t']
    microbe_dict = pathes['path']['microbe_dict']
    microbe_img = pathes['path']['microbe_img']
    taxonomy_file = pathes['path']['taxonomy_file']
    microbial_genome_length = pathes['path']['microbial_genome_length']
    catalog_genome_rank_file = pathes['path']['catalog_genome']
    tax_id_hierarchy_file = pathes['path']['tax_id_hierarchy']
    java17 = pathes['path']['java17']
    gatk_jar = pathes['path']['gatk_4.6_jar']
    
    # Output directories
    step_name_pathseq = configure['step_name']['pathseq']
    step_name_nucleic = configure['step_name']['nucleic']
    output_pathseq = output_path + f'{sample}/{step_name_pathseq}/'
    output_nucleic = output_path + f'{sample}/{step_name_nucleic}/'
    
    # Create output directory
    cmd_mkdir = f"mkdir -p {output_pathseq}"
    tool.judge_then_exec(sample,cmd_mkdir,output_pathseq)
    
    # Preserved commented command
    # cmd7 = f'{java17} -Xms{mem} -Xmx{mem} -jar {gatk_jar} PathSeqPipelineSpark --spark-master local --tmp-dir {tmp_dir} --input {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam --filter-bwa-image {host_img_hg38} --kmer-file {host_kmer_hg38} --min-clipped-read-length 50 --microbe-dict {microbe_dict} --microbe-bwa-image {microbe_img} --taxonomy-file {taxonomy_file} --output {output_pathseq}{sample}_{seq_type}_output.pathseq.bam --scores-output {output_pathseq}{sample}_{seq_type}_output.pathseq.txt --score-metrics {output_pathseq}{sample}_{seq_type}_score.metrics.txt --filter-metrics {output_pathseq}{sample}_{seq_type}_filter.metrics.txt --divide-by-genome-length true'

    # Active PathSeq command
    cmd7 = f'{java17} -Xms{mem} -Xmx{mem} -jar {gatk_jar} PathSeqPipelineSpark --spark-master local[{thread}] --tmp-dir {tmp_dir} --input {output_t2t}{sample}_{seq_type}_hg38_unmap_t2t_unmap.bam --min-clipped-read-length 50 --microbe-dict {microbe_dict} --microbe-bwa-image {microbe_img} --taxonomy-file {taxonomy_file} --output {output_pathseq}{sample}_{seq_type}_output.pathseq.bam --scores-output {output_pathseq}{sample}_{seq_type}_output.pathseq.txt --score-metrics {output_pathseq}{sample}_{seq_type}_score.metrics.txt --filter-metrics {output_pathseq}{sample}_{seq_type}_filter.metrics.txt --divide-by-genome-length true'
    tool.judge_then_exec(sample,cmd7,f'{output_pathseq}{sample}_{seq_type}_output.pathseq.txt') 
    
    # Create nucleic output directory
    cmd_mkdir = f"mkdir -p {output_nucleic}"
    tool.judge_then_exec(sample,cmd_mkdir,output_nucleic)
    
    # Convert BAM to text
    cmd8 = f"samtools view -@ {thread} {output_pathseq}{sample}_{seq_type}_output.pathseq.bam > {output_nucleic}{sample}.txt"
    tool.judge_then_exec(sample,cmd8,f'{output_nucleic}{sample}.txt')

    # Microbial sequence extraction
    if not os.path.exists(f'{output_nucleic}/{sample}.microbe_abundance_genus.csv'):
        tool.write_log(f"[{sample}] Extracting microbial sequences for BLASTX","info")
        start = datetime.now()
        get_data(sample,catalog_genome_rank_file,tax_id_hierarchy_file,output_hg38,output_nucleic,microbial_genome_length,match_length_threshold,score_threshold,pair,seq_type)
        end = datetime.now()
        using_time = tool.print_time(end-start)
        tool.write_log(f"[{sample}] Sequence extraction completed in {using_time}","info")

def MicrobialPeptidesIdentification(sample,configure,pathes,tool):    
    """Identify microbial peptides using BLASTX
    Args:
        sample: Sample ID string
        configure: Configuration dictionary
        pathes: Paths dictionary
        tool: Execution tool object
    """
    output_path = configure['path']['output_dir'] + "/"  # Output directory
    step_name_nucleic = configure['step_name']['nucleic']
    step_name_blastx = configure['step_name']['blastx']
    output_nucleic = output_path + f'{sample}/{step_name_nucleic}/'
    output_blastx = output_path + f'{sample}/{step_name_blastx}/'
    thread = configure['args']['thread']  # CPU threads allocation
    db_dir = pathes['database']['db_dir']  # BLAST database path
    outfmt = pathes['database']['outfmt']  # Output format
    
    # Create output directory
    cmd_mkdir = f"mkdir -p {output_blastx}"
    tool.judge_then_exec(sample,cmd_mkdir,output_blastx)
    
    # BLASTX command
    cmd9 = f"blastx -query {output_nucleic}{sample}.fasta -out {output_blastx}{sample}.taxidlist.blast -db {db_dir} -outfmt {outfmt} -evalue 5e-2 -num_threads {thread} -max_target_seqs 5 "
    tool.judge_then_exec(sample,cmd9,f'{output_blastx}/{sample}.peptide.fasta')

    # Process BLAST results
    if not os.path.exists(f'{output_blastx}/{sample}.peptide.fasta'):
        tool.write_log("Processing BLAST results for pVACbind","info")         
        start = datetime.now()
        get_data_for_pvacbind(f'{sample}.taxidlist.blast',f'{sample}.peptide.fasta',output_blastx)
        end = datetime.now()
        tool.print_time(end-start)

def MicrobialPeptidesBindingPrediction(sample,configure,pathes,tool):
    """Predict peptide-MHC binding using pVACbind
    Args:
        sample: Sample ID string
        configure: Configuration dictionary
        pathes: Paths dictionary
        tool: Execution tool object
    """
    output_path = configure['path']['output_dir'] + "/"  # Output directory
    step_name_blastx = configure['step_name']['blastx']
    step_name_hla = configure['step_name']['hla']
    step_name_pvacbind = configure['step_name']['pvacbind']
    output_blastx = output_path + f'{sample}/{step_name_blastx}/'
    output_hla = output_path + f'{sample}/{step_name_hla}/'
    output_pvacbind = output_path + f'{sample}/{step_name_pvacbind}/'

    if os.path.exists(f'{output_blastx}/{sample}.peptide.fasta'):
        # Create output directory
        cmd_mkdir = f"mkdir -p {output_pvacbind}"
        tool.judge_then_exec(sample,cmd_mkdir,output_pvacbind)
        
        # Execute pVACbind prediction
        pvacbind(sample,output_blastx,output_hla,output_pvacbind,configure,pathes,tool)