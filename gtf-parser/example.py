def parse_gencode(self):
    """Parse GENCODE GTF to extract splice sites."""
    from gtfparse import read_gtf
    from pyfaidx import Fasta
    import numpy as np
    
    fasta = Fasta(self.fasta_file)
    acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
    donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
    self.transcript_positions = {chrom: set() for chrom in self.target_chromosomes}
    
    # Read GTF file using gtfparse
    df = read_gtf(
        self.gtf_file,
        features={'exon', 'transcript'},  # Only load exon and transcript features
        result_type='pandas'  # Get pandas DataFrame
    )
    
    # Filter for protein coding genes on target chromosomes
    df = df[
        (df['feature'].isin(['exon', 'transcript'])) &
        (df['gene_type'] == 'protein_coding') &
        (df['strand'] == '+') &
        (df['seqname'].isin(self.target_chromosomes))
    ]
    
    # Filter for primary transcripts
    df = df[df['tag'].str.contains('GENCODE_Primary', na=False)]
    
    # Get all transcript positions
    transcript_df = df[df['feature'] == 'transcript']
    for _, transcript in transcript_df.iterrows():
        chrom = transcript['seqname']
        self.transcript_positions[chrom].update(
            range(transcript['start']-1, transcript['end'])
        )
    
    # Process exons to get splice sites
    exon_df = df[df['feature'] == 'exon']
    for _, exon in exon_df.iterrows():
        chrom = exon['seqname']
        donor_sites[chrom][exon['end']-1] = 1     # End of exon is donor site
        acceptor_sites[chrom][exon['start']-1] = 1 # Start of exon is acceptor site
    
    return acceptor_sites, donor_sites