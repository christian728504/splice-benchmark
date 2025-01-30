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
from gtfparse import read_gtf

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
        # self.target_chromosomes = ['chr1']
        self.transcript_positions = {chrom: set() for chrom in self.target_chromosomes}

    def get_gene_names_from_tsv(self, tsv_file: str) -> set:
        """Extract set of gene names from TSV file."""
        df = pd.read_csv(tsv_file, sep='\t')
        # Extract gene names from name column
        df = df[df['strand'] == '+']
        gene_names = set(df['name'].values)
        
        return gene_names

    def parse_gencode(self, gene_names: Set[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Parse GENCODE GTF to extract splice sites.
        
        Args:
            gene_names: Optional set of gene names to filter for. If None, uses all genes.
        Returns:
            Tuple of dictionaries mapping chromosome names to numpy arrays of acceptor and donor sites
        """
        fasta = Fasta(self.fasta_file)
        acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}

        # Read GTF file using gtfparse
        df = read_gtf(
            self.gtf_file,
            features={'exon', 'transcript'},
            result_type='pandas'
        )

        # Apply basic filters
        df = df[
            (df['feature'].isin(['exon', 'transcript'])) &
            (df['gene_type'] == 'protein_coding') &
            (df['strand'] == '+') &
            (df['seqname'].isin(self.target_chromosomes))
        ]

        # Filter by gene names if provided
        if gene_names is not None:
            # Convert gene names to lowercase for comparison
            gene_names_lower = {name.lower() for name in gene_names}
            df = df[df['gene_name'].str.lower().isin(gene_names_lower)]

        # Process transcripts
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
            donor_sites[chrom][exon['end']-1] = 1
            acceptor_sites[chrom][exon['start']-1] = 1

        return acceptor_sites, donor_sites

    def generate_spliceai_predictions(self, ground_truth):
        """Generate splice site predictions using SpliceAI model with optimized GPU usage."""
        
        acceptor_predictions = {}
        donor_predictions = {}
        fasta = Fasta(self.fasta_file)
        context = 15000
        step_size = 5000
        batch_size = 128
        
        # Load models once
        print("Loading models...")
        models = []
        for x in range(1, 6):
            model = load_model(files('spliceai').joinpath(f'/pi/zhiping.weng-umw/data/ramirezc/splice-benchmark/spliceai/models/spliceai{x}.h5'))
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
