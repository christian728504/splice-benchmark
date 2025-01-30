dimport os
import pickle
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Tuple, Set
from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
from tqdm import tqdm

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
        self.transcript_positions = {chrom: set() for chrom in self.target_chromosomes}
        
    def _parse_attribute(self, attributes: str, key: str) -> str:
        """Parse specific attribute from GTF attributes string."""
        attrs = attributes.split('; ')
        for attr in attrs:
            if attr.startswith(f'{key} "'):
                return attr.split('"')[1]
        return ''
        
    def parse_gencode(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Parse GENCODE GTF to extract splice sites."""
        fasta = Fasta(self.fasta_file)
        acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        
        df = pd.read_csv(
            self.gtf_file, 
            sep='\t',
            comment='#',
            names=['chrom', 'source', 'feature', 'start', 'end', 'score', 
                   'strand', 'frame', 'attributes']
        )
        
        df['gene_type'] = df['attributes'].apply(lambda x: self._parse_attribute(x, 'gene_type'))
        df['transcript_id'] = df['attributes'].apply(lambda x: self._parse_attribute(x, 'transcript_id'))
        
        exons = df[
            (df['source'] == 'HAVANA') &
            (df['feature'] == 'exon') & 
            (df['strand'] == '+') & 
            (df['gene_type'] == 'protein_coding') &
            (df['attributes'].str.contains('tag "GENCODE_Primary"')) &
            (df['chrom'].isin(self.target_chromosomes))
        ]
        
        relevant_transcript_ids = set(exons['transcript_id'].unique())
        
        transcripts = df[
            (df['feature'] == 'transcript') &
            (df['transcript_id'].isin(relevant_transcript_ids)) &
            (df['chrom'].isin(self.target_chromosomes))
        ]
        
        for _, transcript in transcripts.iterrows():
            chrom = transcript['chrom']
            self.transcript_positions[chrom].update(range(transcript['start']-1, transcript['end']))
        
        for _, exon in exons.iterrows():
            chrom = exon['chrom']
            donor_sites[chrom][exon['end']-1] = 1    # End of exon is donor site
            acceptor_sites[chrom][exon['start']-1] = 1  # Start of exon is acceptor site
        
        return acceptor_sites, donor_sites
    
    def generate_predictions(self, ground_truth):
        """
        Generate splice site predictions for each chromosome with sequence padding.
        
        Args:
            ground_truth: Dict of ground truth (needed for length tracking)
        
        Returns:
            Tuple of dicts: (acceptor_predictions, donor_predictions)
        """
        acceptor_predictions = {}
        donor_predictions = {}
        fasta = Fasta(self.fasta_file)
        step_size = 1000  # Step size for sliding window
        
        for chrom in tqdm(self.target_chromosomes, desc="Processing chromosomes"):
            # Get chromosome length
            chrom_length = len(ground_truth[chrom])
            
            # Create padded sequence
            # Use a 4000 bp sequence of [0,0,0,0] at start and end
            pad_seq = ''.join(['N'] * 4000)
            full_chrom_seq = pad_seq + str(fasta[chrom]) + pad_seq
            
            # Initialize prediction arrays with extra padding length
            padded_length = chrom_length + 8000
            acceptor_predictions[chrom] = np.zeros(padded_length)
            donor_predictions[chrom] = np.zeros(padded_length)
            
            step_size = 1000  # Step size for sliding window
            
            # Slide window across padded chromosome sequence
            for pos in tqdm(range(0, padded_length - self.window_size, step_size), 
                        desc=f"{chrom} sliding"):
                
                # Extract sequence
                sequence = full_chrom_seq[pos:pos + self.window_size]
                if len(sequence) != self.window_size:
                    continue
                
                # Get predictions
                encoded_seq = self.annotator.model.one_hot_encode(sequence)
                encoded_seq = torch.tensor(encoded_seq).to(self.annotator.model.device)
                encoded_seq = encoded_seq.unsqueeze(0).float().transpose(1, 2)
                
                with torch.no_grad():
                    output = self.annotator.model.step(encoded_seq)
                    output = output.cpu().numpy()[0]  # Shape: (18, 1000)
                
                # Separate acceptor and donor predictions
                acceptor_probs = output[1]  # Splice Acceptor (SA) scores
                donor_probs = output[2]     # Splice Donor (SD) scores
                
                # Update predictions for the central region
                center_start = pos + 4000
                center_end = center_start + 1000
                
                # Update acceptor prediction
                acceptor_predictions[chrom][center_start:center_end] = np.maximum(
                    acceptor_predictions[chrom][center_start:center_end],
                    acceptor_probs
                )
                
                # Update donor prediction
                donor_predictions[chrom][center_start+1:center_end+1] = np.maximum(
                    donor_predictions[chrom][center_start+1:center_end+1],
                    donor_probs
                )
            
            # Trim predictions back to original chromosome length
            acceptor_predictions[chrom] = acceptor_predictions[chrom][4000:-4000]
            donor_predictions[chrom] = donor_predictions[chrom][4000:-4000]
        
        return acceptor_predictions, donor_predictions
    
    def load_predictions(self, acceptor_pkl: str, donor_pkl: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load and trim prediction arrays to only include positions within genes."""
        with open(acceptor_pkl, 'rb') as f:
            acceptor_pred = pickle.load(f)
        with open(donor_pkl, 'rb') as f:
            donor_pred = pickle.load(f)
            
        trimmed_acceptor = {}
        trimmed_donor = {}
        
        for chrom in self.target_chromosomes:
            trimmed_acceptor[chrom] = np.zeros_like(acceptor_pred[chrom])
            trimmed_donor[chrom] = np.zeros_like(donor_pred[chrom])
            
            transcript_pos = np.array(list(self.transcript_positions[chrom]))
            if len(transcript_pos) > 0:
                trimmed_acceptor[chrom][transcript_pos] = acceptor_pred[chrom][transcript_pos]
                trimmed_donor[chrom][transcript_pos] = donor_pred[chrom][transcript_pos]
        
        return trimmed_acceptor, trimmed_donor
    
    def calculate_metrics(self, ground_truth: Dict[str, np.ndarray], 
                         predictions: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        """Calculate AUPRC and other metrics including top-k accuracy for predictions."""
        # Concatenate all chromosomes
        all_truth = np.concatenate([ground_truth[chrom] for chrom in self.target_chromosomes])
        all_preds = np.concatenate([predictions[chrom] for chrom in self.target_chromosomes])
        
        # Calculate precision-recall curve and AUPRC
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        # Calculate top-k accuracy
        k = int(np.sum(all_truth))  # k = number of actual positive sites
        top_k_indices = np.argsort(all_preds)[-k:]  # Get indices of k highest predictions
        top_k_accuracy = np.sum(all_truth[top_k_indices]) / k
        
        return precision, recall, auprc, top_k_accuracy