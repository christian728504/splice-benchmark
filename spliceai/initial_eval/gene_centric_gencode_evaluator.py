import os
import pickle
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Tuple, Set
from tensorflow.keras.models import load_model
from importlib.resources import files
from spliceai.utils import one_hot_encode
from tqdm.notebook import tqdm

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

    def generate_spliceai_predictions(self, ground_truth):
        """Generate splice site predictions using SpliceAI model with optimized GPU usage."""
        
        acceptor_predictions = {}
        donor_predictions = {}
        fasta = Fasta(self.fasta_file)
        context = 15000
        step_size = 5000
        batch_size = 32
        
        # Load models once
        print("Loading models...")
        models = []
        for x in range(1, 6):
            model = load_model(files('spliceai').joinpath(f'models/spliceai{x}.h5'))
            models.append(model)
    
        def one_hot_encode_batch(sequences):
            """Optimized one-hot encoding for a batch of sequences."""
            sequence_length = len(sequences[0])
            batch_size = len(sequences)
            encoding = np.zeros((batch_size, sequence_length, 4), dtype=np.float32)
            
            base_to_index = np.zeros(128, dtype=np.int8)
            base_to_index[ord('A')] = 0
            base_to_index[ord('C')] = 1
            base_to_index[ord('G')] = 2
            base_to_index[ord('T')] = 3
            
            for i, seq in enumerate(sequences):
                indices = base_to_index[np.frombuffer(seq.encode(), dtype=np.int8)]
                valid_bases = indices >= 0
                encoding[i, np.arange(len(seq))[valid_bases], indices[valid_bases]] = 1
                
            return encoding
    
        for chrom in tqdm(self.target_chromosomes, desc="Processing chromosomes"):
            chrom_length = len(ground_truth[chrom])
            pad_seq = 'N' * (context//3)
            full_chrom_seq = pad_seq + str(fasta[chrom]) + pad_seq
            
            # Initialize prediction arrays
            padded_length = chrom_length + 2*(context//3)
            acceptor_predictions[chrom] = np.zeros(padded_length)
            donor_predictions[chrom] = np.zeros(padded_length)
            
            # Pre-calculate all positions
            positions = list(range(0, padded_length - context, step_size))
            
            # Process in batches
            for batch_idx in tqdm(range(0, len(positions), batch_size), 
                                desc=f"{chrom} processing"):
                # Extract batch positions
                batch_end = min(batch_idx + batch_size, len(positions))
                batch_positions = positions[batch_idx:batch_end]
                
                # Prepare sequences
                sequences = [full_chrom_seq[pos:pos + context] 
                           for pos in batch_positions]
                
                # One-hot encode
                x = one_hot_encode_batch(sequences)
                
                # Get predictions
                predictions = []
                for model in models:
                    # Using numpy array directly without converting to tensor
                    pred = model(x, training=False)
                    if isinstance(pred, np.ndarray):
                        predictions.append(pred)
                    else:
                        predictions.append(pred.numpy())
                
                # Average predictions
                y = np.mean(predictions, axis=0)
                
                # Update predictions
                for i, pos in enumerate(batch_positions):
                    center_start = pos + context//3
                    center_end = center_start + step_size
                    
                    acceptor_predictions[chrom][center_start:center_end] = np.maximum(
                        acceptor_predictions[chrom][center_start:center_end],
                        y[i, :step_size, 1]
                    )
                    donor_predictions[chrom][center_start:center_end] = np.maximum(
                        donor_predictions[chrom][center_start:center_end],
                        y[i, :step_size, 2]
                    )
            
            # Trim padding
            acceptor_predictions[chrom] = acceptor_predictions[chrom][context//3:-context//3]
            donor_predictions[chrom] = donor_predictions[chrom][context//3:-context//3]
        
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
