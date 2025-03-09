def get_data_for_pvacbind(blast_file,pvacbind_file,output_blastx):
    """Process BLAST results for pVACbind input
    
    Args:
        blast_file (str): Path to input BLAST results file containing alignment data
        pvacbind_file (str): Name of output file to write filtered peptide sequences
        output_blastx (str): Output directory path where both input and output files reside
        
    Processes BLAST output to extract perfect-match peptides and formats them for pVACbind input"""
    context = ""
    with open(f"{output_blastx}/{blast_file}",'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            id = f">protIndex:{i}|" + line.split("\t")[0]  # Create FASTA header with protein index
            peptide = line.split("\t")[2]  # Extract peptide sequence from BLAST result
            pident = line.split("\t")[5]  # Get percentage identity value
            # Keep only perfect matches (100% identity) with non-empty peptides
            if peptide != '' and int(float(pident)) == 100:
                context += id + '\n' + peptide + '\n'
    context = context[:-1]  # Remove trailing newline
    with open(f"{output_blastx}/{pvacbind_file}","w") as f:
        f.write(context)