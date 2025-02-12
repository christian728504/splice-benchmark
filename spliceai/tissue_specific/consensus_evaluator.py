import os
import pickle
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import tensorflow as tf
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Tuple, Set, Optional, List
from tensorflow.keras.models import load_model
from importlib.resources import files
from spliceai.utils import one_hot_encode
from tqdm.notebook import tqdm
from gtfparse import read_gtf

class ConsensusSpliceSiteEvaluator:
    def __init__(self, gencode_gtf: str, consensus_fasta: str):
        """
        Initialize evaluator with GENCODE GTF and consensus FASTA.
        
        Args:
            gencode_gtf: Path to GENCODE GTF annotation file
            consensus_fasta: Path to consensus sequence FASTA
        """
        self.gtf_file = gencode_gtf
        self.consensus_fasta = consensus_fasta
        self.target_chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        self.transcript_positions = {chrom: set() for chrom in self.target_chromosomes}
        
    def filter_expressed_genes(self, quant_tsv: str, min_tpm: float = 2.0) -> Set[str]:
        """
        Filter genes based on expression level from quantification TSV.
        
        Args:
            quant_tsv: Path to gene quantification TSV file
            min_tpm: Minimum TPM threshold for considering a gene as expressed
            
        Returns:
            Set of expressed gene IDs
        """
        df = pd.read_csv(quant_tsv, sep='\t')
        expressed_genes = set(df[df['TPM'] >= min_tpm]['gene_id'].values)
        return expressed_genes

    def parse_gencode(self, 
                    expressed_genes: Optional[Set[str]] = None,
                    ground_truth_file: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Parse GENCODE GTF to extract splice sites from APPRIS principal transcripts,
        optionally filtering for expressed genes.
        
        Args:
            expressed_genes: Set of gene IDs that meet expression threshold
            ground_truth_file: Optional path to load serialized ground truth
            
        Returns:
            Tuple of dictionaries mapping chromosome names to numpy arrays of acceptor and donor sites
        """
        # Check if we should load from serialized file
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'rb') as f:
                return pickle.load(f)
        
        fasta = Fasta(self.consensus_fasta)
        acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}

        # Read GTF file
        df = read_gtf(
            self.gtf_file,
            features={'exon', 'transcript'},
            result_type='pandas'
        )

        # Filter for APPRIS principal transcripts
        df['is_appris_principal'] = df['tag'].str.contains('appris_principal', na=False)
        
        # Apply all filters
        df = df[
            (df['feature'].isin(['exon', 'transcript'])) &
            (df['gene_type'] == 'protein_coding') &
            (df['strand'] == '+') &
            (df['seqname'].isin(self.target_chromosomes)) &
            (df['is_appris_principal'])
        ]

        # Filter by expressed genes if provided
        if expressed_genes is not None:
            df = df[df['gene_id'].isin(expressed_genes)]

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

        # Serialize ground truth if path provided
        if ground_truth_file:
            with open(ground_truth_file, 'wb') as f:
                pickle.dump((acceptor_sites, donor_sites), f)

        return acceptor_sites, donor_sites

    def generate_spliceai_predictions(self, 
                                    predictions_file: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate splice site predictions using SpliceAI model for consensus sequences.
        Only returns predictions within transcript regions.
        
        Args:
            predictions_file: Optional path to save serialized predictions
            
        Returns:
            Tuple of dictionaries containing filtered acceptor and donor predictions
        """
        acceptor_predictions = {}
        donor_predictions = {}
        fasta = Fasta(self.consensus_fasta)
        context = 15000
        step_size = 5000
        batch_size = 128
        
        # Load models
        print("Loading models...")
        models = []
        for x in range(1, 6):
            model = load_model(files('spliceai').joinpath(f'/pi/zhiping.weng-umw/data/ramirezc/splice-benchmark/spliceai/models/spliceai{x}.h5'))
            models.append(model)

        def one_hot_encode_batch(sequences):
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
            chrom_length = len(fasta[chrom])
            pad_seq = 'N' * (context//3)
            full_chrom_seq = pad_seq + str(fasta[chrom]) + pad_seq
            
            # Initialize prediction arrays
            padded_length = chrom_length + 2*(context//3)
            acceptor_predictions[chrom] = np.zeros(padded_length)
            donor_predictions[chrom] = np.zeros(padded_length)
            
            positions = list(range(0, padded_length - context, step_size))
            
            for batch_idx in tqdm(range(0, len(positions), batch_size), 
                                desc=f"{chrom} processing"):
                batch_end = min(batch_idx + batch_size, len(positions))
                batch_positions = positions[batch_idx:batch_end]
                
                sequences = [full_chrom_seq[pos:pos + context] 
                        for pos in batch_positions]
                
                x = one_hot_encode_batch(sequences)
                
                predictions = []
                for model in models:
                    pred = model(x, training=False)
                    if isinstance(pred, np.ndarray):
                        predictions.append(pred)
                    else:
                        predictions.append(pred.numpy())
                
                y = np.mean(predictions, axis=0)
                
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
            
            # Zero out predictions outside transcript regions
            transcript_mask = np.zeros(len(acceptor_predictions[chrom]), dtype=bool)
            transcript_positions = np.array(list(self.transcript_positions[chrom]))
            if len(transcript_positions) > 0:
                transcript_mask[transcript_positions] = True
                
            acceptor_predictions[chrom] = acceptor_predictions[chrom] * transcript_mask
            donor_predictions[chrom] = donor_predictions[chrom] * transcript_mask
        
        # Serialize predictions if path provided
        if predictions_file:
            with open(predictions_file, 'wb') as f:
                pickle.dump((acceptor_predictions, donor_predictions), f)
        
        return acceptor_predictions, donor_predictions
    
    
    def calculate_metrics(self, 
                         ground_truth: Dict[str, np.ndarray], 
                         predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Calculate AUPRC and other metrics including top-k accuracy for predictions."""
        all_truth = np.concatenate([ground_truth[chrom] for chrom in self.target_chromosomes])
        all_preds = np.concatenate([predictions[chrom] for chrom in self.target_chromosomes])
        
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        k = int(np.sum(all_truth))
        top_k_indices = np.argsort(all_preds)[-k:]
        top_k_accuracy = np.sum(all_truth[top_k_indices]) / k
        
        return precision, recall, auprc, top_k_accuracy