# coding=utf-8
import os
import yaml
import sys
import argparse
import traceback
import multiprocessing
import multiprocessing.pool
sys.path.append("../functions/")
from fastp import fastp
from pipline_tools import tools
from multiprocessing import Pool,Manager
sys.path.append("./scripts/")
from hlahd_faster import hlahd
#from hlahd import hlahd
#from MicrobialAntigen_mouse_start import Microbial_Taxas_Annotation
from MicrobialAntigen_human_start import HostSequencesRemoving,MicrobialTaxasQuantification,MicrobialPeptidesIdentification,MicrobialPeptidesBindingPrediction

# Pipeline identifier
flag = 'MicrobialAntigen'
# Initialize tool class
log_lock = multiprocessing.Manager().Lock()  # Create a Lock object
tool = tools(os.getcwd(),flag,log_lock)
tool.write_log(f"work_path: {tool.sys_path}","info")
tool.write_log(f"start_log: {tool.start_log}","info")
tool.write_log(f"cmd_log: {tool.cmd_log}","info")

# Argument parser setup
parser = argparse.ArgumentParser(description='MicNeo')
parser.add_argument('-c','--configure',help="Configuration file path")
args = parser.parse_args()

# Load configuration
configure = tool.get_configure(args.configure)
tool.write_log(f"configures: {configure}","info")

# Load path configurations
with open(f"./configures/{flag}_pathes.yaml","r") as file:
    pathes = yaml.safe_load(file)
tool.write_log(f"pathes: {pathes}","info")

# Read configuration parameters
samples = configure['samples']
species = configure['others']['species']  # Species selection
pool_size = configure['args']['pool_size']  # Concurrent sample processing count
QC = configure['others']['QC']
host_sequences_removing = configure['others']['host_sequences_removing']
microbial_taxas_quantification = configure['others']['microbial_taxas_quantification']
microbial_peptides_identification = configure['others']['microbial_peptides_identification']
hlatyping = configure['others']['hlatyping']
microbial_peptides_bindingPrediction = configure['others']['microbial_peptides_bindingPrediction']

# Assign samples to tool class
tool.samples = samples

# Configure pipeline step names
configure['step_name'] = {}
configure['step_name']['QC'] = '00.QC'
configure['step_name']['hg38'] = '01.HostSequencesRemovingStep1'
configure['step_name']['t2t'] = '02.HostSequencesRemovingStep2'
configure['step_name']['pathseq'] = '03.MicrobialTaxasQuantificationStep1'
configure['step_name']['nucleic'] = '04.MicrobialTaxasQuantificationStep2'  
configure['step_name']['blastx'] = '05.MicrobialPeptidesIdentification'
configure['step_name']['hla'] = '06.HlaTyping'
configure['step_name']['pvacbind'] = '07.MicrobialPeptidesBindingPrediction' 

def start(sample):
    """
    Main processing pipeline for a sample
    
    Args:
        sample (str): Sample ID to process
    """
    sample = str(sample)

    if QC:
        try:
            fastp(sample,configure,pathes,tool)
        except Exception as e:
            error_message = traceback.format_exc()
            tool.write_log(f"QC Error occurred: {error_message}","error")
            
    if host_sequences_removing:        
        try:
            HostSequencesRemoving(sample,configure,pathes,tool)
        except Exception as e:
            error_message = traceback.format_exc()
            tool.write_log(f"HostSequencesRemoving Error occurred: {error_message}","error")
            
    if microbial_taxas_quantification:
        try:
            MicrobialTaxasQuantification(sample,configure,pathes,tool)
        except Exception as e:
            error_message = traceback.format_exc()
            tool.write_log(f"MicrobialTaxasQuantification Error occurred: {error_message}","error")
            
    if microbial_peptides_identification:
        try:
            MicrobialPeptidesIdentification(sample,configure,pathes,tool)
        except Exception as e:
            error_message = traceback.format_exc()
            tool.write_log(f"MicrobialPeptidesIdentification Error occurred: {error_message}","error")

    # HLA typing process
    if hlatyping:
        try:
           hlahd(sample,configure,pathes,tool)
        except Exception as e:
            error_message = traceback.format_exc()
            tool.write_log(f"Hlatyping Error occurred: {error_message}","error")
            
    if microbial_peptides_bindingPrediction:
        try:
            MicrobialPeptidesBindingPrediction(sample,configure,pathes,tool)
        except Exception as e:
            error_message = traceback.format_exc()
            tool.write_log(f"MicrobialPeptidesBindingPrediction Error occurred: {error_message}","error")

## MicrobialAntigen_v1.0
def MicrobialAntigen():
    """
    Main execution function for parallel processing of samples
    """
    pool = Pool(pool_size)
    for count,sample in enumerate(samples):
        pool.apply_async(start,(sample,),error_callback=tool.print_pool_error)
    pool.close()
    pool.join()
  
if __name__ == '__main__':
  # Initialize shared variables
  tool.sharing_variable(Manager(),samples)
  # Start multiprocessing
  MicrobialAntigen()
  tool.summary()