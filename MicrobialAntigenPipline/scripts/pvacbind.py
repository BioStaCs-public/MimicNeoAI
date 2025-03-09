# coding=utf-8
import os
import pandas as pd

def get_hlahd_results(sample, output_hla):
    """
    Extracts HLA typing results from HLA-HD output files.
    
    Parameters:
    - sample (str): Sample identifier
    - output_hla (str): Directory path containing HLA-HD results
    
    Returns:
    - tuple: (hlaI, hlaII) strings containing comma-separated HLA alleles
    """
    hlaI_set = ['A','B','C','E','F','G','H','J','K','L','V']
    hlaII_set = ['DRA','DRB1','DRB2','DRB3','DRB4','DRB5','DRB6','DRB7','DRB8','DRB9',
                'DQA1','DQB1','DPA1','DPB1','DMA','DMB','DOA','DOB']
    hla_file = f"{output_hla}/{sample}/result/{sample}_final.result.txt"
    hlaI = ''
    hlaII = ''
    with open(hla_file, "r") as f:
        for line in f.readlines():
            new_line = line.replace("\n", "").split('\t')
            # Process HLA-I alleles
            if new_line[0] in hlaI_set:
                for h in new_line[1:]:
                    h_split = h.split(":")
                    if h != '-' and h != 'Not typed' and len(h_split) >= 2:
                        h_2 = ':'.join(h_split[:2])  # Keep first two fields for compatibility
                        hlaI += h_2 + ','
            # Process HLA-II alleles
            if new_line[0] in hlaII_set:
                for h in new_line[1:]:
                    h_split = h.split(":")
                    if h != '-' and h != 'Not typed' and len(h_split) >= 2:
                        h_2 = ':'.join(h_split[:2])
                        hlaII += h_2.split('-')[1] + ','
    return hlaI, hlaII

def pvacbind(sample, fa_dir, output_hla, output_dir, configure, pathes, tool):
    """
    Executes pVACbind pipeline for neoantigen prediction.
    
    Parameters:
    - sample (str): Sample identifier
    - fa_dir (str): Directory containing input FASTA files
    - output_hla (str): Path to HLA typing results
    - output_dir (str): Output directory for results
    - configure (dict): Configuration parameters including:
        - args: Dictionary with 'thread' for thread count
        - path: Dictionary with 'output_dir' and 'tmp_dir'
    - pathes (dict): Path configurations including:
        - path: Dictionary with 'sif_file' and 'pvacbind' executable
        - iedb-install-directory: Path to IEDB installation
    - tool (object): Tool object with execution methods:
        - judge_then_exec: Method to conditionally execute commands
        - write_log: Method for logging messages
    """
    # Algorithm configurations
    algoIandII = 'NNalign NetMHC NetMHCIIpan SMM SMMPMBEC SMMalign NetMHCpanEL NetMHCIIpanEL'
    algoI = 'NNalign NetMHC SMM SMMPMBEC SMMalign NetMHCpanEL'

    # Extract configuration parameters
    thread = configure['args']['thread']
    opt_dir = configure['path']['output_dir'] + '/'
    tmp = configure['path']['tmp_dir']
    sif_file = pathes['path']['sif_file']
    pvacbind = pathes['path']['pvacbind']
    iedb_install_directory = pathes['path']['iedb-install-directory']
    
    # Define pipeline step names
    step_name_blastx = configure['step_name']['blastx']
    step_name_pvacbind = configure['step_name']['pvacbind']
    
    # Construct output paths
    output_blastx = opt_dir + f'{sample}/{step_name_blastx}/'
    output_pvacbind = opt_dir + f'{sample}/{step_name_pvacbind}/'

    # Get HLA typing results
    hlaI, hlaII = get_hlahd_results(sample, output_hla) 
    
    if hlaI != '' or hlaII != '':
        HLA = hlaI + hlaII[:-1]  # Remove trailing comma
        cmd = (f"{pvacbind} run {output_blastx}/{sample}.peptide.fasta {sample} {HLA} {algoIandII} "
               f"{output_pvacbind} -e1 8,9,10 -e2 15 --iedb-install-directory {iedb_install_directory} "
               f"-t {thread} --fasta-size 100000")
        
        # Execute command conditionally
        tool.judge_then_exec(sample, cmd, f"{output_dir}/combined/{sample}.done.txt")
        
        # Create completion marker
        if os.path.exists(f"{output_dir}/combined/{sample}.all_epitopes.tsv"):
            with open(f"{output_dir}/combined/{sample}.done.txt", "w") as f:
                f.write(f"{cmd} done!")        
    else:
        tool.write_log(f"[{sample}] hlahd results is all Not_typed!", "error")

def pvacbind_for_mouse(sample, fa_dir, output_dir, configure, pathes, tool):
    """
    Placeholder function for mouse-specific pVACbind implementation.
    
    Parameters:
    - sample (str): Sample identifier
    - fa_dir (str): Directory containing input FASTA files
    - output_dir (str): Output directory for results
    - configure (dict): Configuration parameters
    - pathes (dict): Path configurations
    - tool (object): Tool execution object
    """
    pass