import os
import pickle
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Tuple, Set
from pkg_resources import resource_filename
from pangolin.model import Pangolin, L, W, AR
import torch
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
        # self.target_chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        self.target_chromosomes = ['chr1']  
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

    def generate_pangolin_predictions(self, ground_truth):
        """Generate splice site predictions using Pangolin model with optimized GPU usage."""
        
        acceptor_predictions = {}
        donor_predictions = {}
        context = 15000  # Total context size (5000 before + 5000 prediction + 5000 after)
        step_size = 5000  # Size of prediction window
        batch_size = 32
        
        # Initialize Pangolin models
        print("Loading models...")
        models = []
        for i in range(1, 6):  # Load 5 models for ensemble
            model = Pangolin(L, W, AR)  # These should be imported from pangolin.model
            if torch.cuda.is_available():
                model.cuda()
                weights = torch.load(resource_filename("pangolin", f"models/final.{i}.7.3"))
            else:
                weights = torch.load(resource_filename("pangolin", f"models/final.{i}.7.3"),
                                map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)

        IN_MAP = np.asarray([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        
        def one_hot_encode(seq, strand):
            """Optimized one-hot encoding following Pangolin's implementation."""
            seq = seq.upper().replace('A', '1').replace('C', '2')
            seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
            if strand == '+':
                seq = np.asarray(list(map(int, list(seq))))
            elif strand == '-':
                seq = np.asarray(list(map(int, list(seq[::-1]))))
                seq = (5 - seq) % 5  # Reverse complement
            return IN_MAP[seq.astype('int8')]
        
        fasta = Fasta(self.fasta_file)
        
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
                batch_end = min(batch_idx + batch_size, len(positions))
                batch_positions = positions[batch_idx:batch_end]
                current_batch_size = len(batch_positions)
                
                # Prepare sequences
                sequences = [full_chrom_seq[pos:pos + context] 
                        for pos in batch_positions]
                
                # Convert to one-hot encoded tensors
                batch_sequences = []
                for seq in sequences:
                    # Process both strands
                    forward = one_hot_encode(seq, '+').T  # Add transpose here
                    reverse = one_hot_encode(seq, '-').T  # Add transpose here
                    batch_sequences.extend([forward, reverse])

                x = torch.from_numpy(np.stack(batch_sequences)).float()
                if torch.cuda.is_available():
                    x = x.cuda()
                
                # Get predictions from all models
                acceptor_scores = []
                donor_scores = []
                
                for model in models:
                    with torch.no_grad():
                        preds = model(x)
                        if isinstance(preds, torch.Tensor):
                            preds = preds.cpu().numpy()
                        
                        # Average predictions across tissues (indices 1,2,4,5,7,8,10,11)
                        # Acceptor indices: 1,4,7,10
                        # Donor indices: 2,5,8,11
                        acceptor_idxs = [1,4,7,10]
                        donor_idxs = [2,5,8,11]
                        
                        # Average across strands and tissues
                        acc_pred = np.mean([preds[:current_batch_size, idx, :] for idx in acceptor_idxs], axis=0)
                        don_pred = np.mean([preds[:current_batch_size, idx, :] for idx in donor_idxs], axis=0)
                        
                        acceptor_scores.append(acc_pred)
                        donor_scores.append(don_pred)
                
                # Average predictions across models
                acceptor_batch = np.mean(acceptor_scores, axis=0)
                donor_batch = np.mean(donor_scores, axis=0)
                
                # Update predictions
                for i, pos in enumerate(batch_positions):
                    center_start = pos + context//3
                    center_end = center_start + step_size
                    
                    acceptor_predictions[chrom][center_start:center_end] = np.maximum(
                        acceptor_predictions[chrom][center_start:center_end],
                        acceptor_batch[i, :step_size]
                    )
                    donor_predictions[chrom][center_start:center_end] = np.maximum(
                        donor_predictions[chrom][center_start:center_end],
                        donor_batch[i, :step_size]
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
