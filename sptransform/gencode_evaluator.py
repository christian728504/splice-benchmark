import os
import pickle
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Tuple

class GencodeSpliceSiteEvaluator:
    def __init__(self, gencode_gtf: str, fasta_file: str):
        """
        Initialize evaluator with GENCODE GTF and genome FASTA.
        
        Args:
            gencode_gtf: Path to GENCODE GTF annotation file
            fasta_file: Path to reference genome FASTA
        """
        self.gtf_file = gencode_gtf
        self.fasta_file = fasta_file
        self.target_chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        
    def _parse_gene_type(self, attributes: str) -> str:
        """
        Parse gene_type from GTF attributes string.
        
        Args:
            attributes: Semi-colon separated string of attributes
            
        Returns:
            Gene type string or empty string if not found
        """
        attrs = attributes.split('; ')
        for attr in attrs:
            if attr.startswith('gene_type'):
                return attr.split('"')[1]
        return ''
        
    def parse_gencode(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Parse GENCODE GTF to extract splice sites.
        
        Returns:
            Tuple of (acceptor_sites, donor_sites) dictionaries mapping 
            chromosome to binary arrays
        """
        # Initialize dictionaries to store site positions
        fasta = Fasta(self.fasta_file)
        acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        
        # Read GTF and filter for exons on target chromosomes
        exons = pd.read_csv(
            self.gtf_file, 
            sep='\t',
            comment='#',
            names=['chrom', 'source', 'feature', 'start', 'end', 'score', 
                   'strand', 'frame', 'attributes']
        )
        
        # Apply gene type parsing to filter for protein coding genes
        exons['gene_type'] = exons['attributes'].apply(self._parse_gene_type)

        for info in exons['gene_type']:
            print(info)
        
        # Filter for relevant exons
        exons = exons[
            (exons['source'] == 'HAVANA') &
            (exons['feature'] == 'exon') & 
            (exons['strand'] == '+') & 
            (exons['gene_type'] == 'protein_coding') &
            (exons['chrom'].isin(self.target_chromosomes))
        ]
        
        for _, exon in exons.iterrows():
            chrom = exon['chrom']
            donor_sites[chrom][exon['end']-1] = 1    # End of exon is donor site
            acceptor_sites[chrom][exon['start']-1] = 1  # Start of exon is acceptor site
        
        return acceptor_sites, donor_sites
    
    def load_predictions(self, acceptor_pkl: str, donor_pkl: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load serialized prediction arrays.
        
        Args:
            acceptor_pkl: Path to acceptor predictions pickle
            donor_pkl: Path to donor predictions pickle
            
        Returns:
            Tuple of (acceptor_predictions, donor_predictions) dictionaries
        """
        with open(acceptor_pkl, 'rb') as f:
            acceptor_pred = pickle.load(f)
        with open(donor_pkl, 'rb') as f:
            donor_pred = pickle.load(f)
        return acceptor_pred, donor_pred
    
    def calculate_metrics(self, ground_truth: Dict[str, np.ndarray], 
                         predictions: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """
        Calculate AUPRC and other metrics for predictions.
        
        Args:
            ground_truth: Dict mapping chromosomes to ground truth arrays
            predictions: Dict mapping chromosomes to prediction arrays
            
        Returns:
            Tuple of (precision, recall, auprc)
        """
        # Concatenate all chromosomes
        all_truth = np.concatenate([ground_truth[chrom] for chrom in self.target_chromosomes])
        all_preds = np.concatenate([predictions[chrom] for chrom in self.target_chromosomes])
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        return precision, recall, auprc