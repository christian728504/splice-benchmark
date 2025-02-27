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
from tqdm import tqdm
from gtfparse import read_gtf

class ConsensusSpliceSiteEvaluator:
    def __init__(self, gencode_gtf: str, consensus_fasta: str):
        self.gtf_file = gencode_gtf
        self.consensus_fasta = consensus_fasta
        self.target_chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        self.exon_sites = {chrom: {'donor': [], 'acceptor': []} for chrom in self.target_chromosomes}

    def filter_expressed_transcripts(self, quant_tsv: str, min_tpm: float = 2.0) -> Set[str]:
        """
        Filter transcripts based on TPM from transcript quantification TSV.
        
        Args:
            quant_tsv: Path to transcript quantification TSV file
            min_tpm: Minimum TPM threshold for considering a transcript as expressed
                
        Returns:
            Set of expressed transcript IDs
        """
        df = pd.read_csv(quant_tsv, sep='\t')
        expressed_transcripts = set(df[df['TPM'] >= min_tpm]['transcript_id'].values)
        
        print(f"\nTranscript filtering summary:")
        print(f"Total transcripts: {len(df)}")
        print(f"Expressed transcripts (TPM >= {min_tpm}): {len(expressed_transcripts)}")

        return expressed_transcripts

    def _filter_gencode(self, expressed_transcripts: Optional[Set[str]] = None) -> pd.DataFrame:
        """Helper method to filter GENCODE GTF data."""
        df = read_gtf(self.gtf_file, features={'exon'}, result_type='pandas')
        df = df[
            (df['feature'] == 'exon') &
            (df['gene_type'] == 'protein_coding') &
            (df['seqname'].isin(self.target_chromosomes))
        ]
        
        if expressed_transcripts is not None:
            df = df[df['transcript_id'].isin(expressed_transcripts)]
            
        return df
    
    # def _populate_splice_sites(self, df: pd.DataFrame):
    #     """Helper method to populate self.exon_sites from filtered GTF data."""
    #     self.exon_sites = {chrom: {'donor': [], 'acceptor': []} for chrom in self.target_chromosomes}

    #     for _, transcript_exons in df.groupby('transcript_id'):
    #         if len(transcript_exons) == 1:  # Skip single-exon transcripts
    #             continue
                
    #         chrom = transcript_exons.iloc[0]['seqname']
    #         strand = transcript_exons.iloc[0]['strand']
    #         exons = transcript_exons.sort_values('start')
            
    #         for _, exon in exons.iterrows():
    #             start = exon['start'] - 1
    #             end = exon['end'] - 1
                
    #             if strand == '+':
    #                 self.exon_sites[chrom]['donor'].append({'position': end, 'strand': '+'})
    #                 self.exon_sites[chrom]['acceptor'].append({'position': start, 'strand': '+'})
    #             else:
    #                 self.exon_sites[chrom]['donor'].append({'position': start, 'strand': '-'})
    #                 self.exon_sites[chrom]['acceptor'].append({'position': end, 'strand': '-'})

    def _populate_splice_sites(self, df: pd.DataFrame):
        """
        Helper method to populate self.exon_sites from filtered GTF data.
        Excludes the start of the first exon and the end of the last exon for each transcript.
        """
        self.exon_sites = {chrom: {'donor': [], 'acceptor': []} for chrom in self.target_chromosomes}
        
        for _, transcript_exons in df.groupby('transcript_id'):
            if len(transcript_exons) <= 1:
                continue
                
            chrom = transcript_exons.iloc[0]['seqname']
            strand = transcript_exons.iloc[0]['strand']

            exons = transcript_exons.sort_values('start')
            
            first_exon_idx = 0
            last_exon_idx = len(exons) - 1
            
            for idx, (_, exon) in enumerate(exons.iterrows()):
                start = exon['start'] - 1
                end = exon['end'] - 1
                
                if strand == '+':
                    if idx != last_exon_idx:
                        self.exon_sites[chrom]['donor'].append({'position': end, 'strand': '+'})
                    if idx != first_exon_idx:
                        self.exon_sites[chrom]['acceptor'].append({'position': start, 'strand': '+'})
                else:
                    if idx != first_exon_idx:
                        self.exon_sites[chrom]['donor'].append({'position': start, 'strand': '-'})
                    if idx != last_exon_idx:
                        self.exon_sites[chrom]['acceptor'].append({'position': end, 'strand': '-'})

    def get_ground_truth(self, 
                        expressed_transcripts: Optional[Set[str]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate ground truth binary arrays from GTF.
        
        Args:
            expressed_transcripts: Optional set of transcript IDs to filter by
            ground_truth_file: Optional path to save/load ground truth data
            
        Returns:
            Tuple of dictionaries containing binary arrays for acceptor and donor sites
        """
        fasta = Fasta(self.consensus_fasta)
        acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        
        # Always process GTF to populate self.exon_sites
        df = self._filter_gencode(expressed_transcripts)
        self._populate_splice_sites(df)
        
        # Create binary arrays from populated sites
        for chrom in self.target_chromosomes:
            for site in self.exon_sites[chrom]['donor']:
                donor_sites[chrom][site['position']] = 1
            for site in self.exon_sites[chrom]['acceptor']:
                acceptor_sites[chrom][site['position']] = 1
                    
        return acceptor_sites, donor_sites

    def generate_spliceai_predictions(self, 
                                    predictions_file: Optional[str] = None,
                                    sequence_length: int = 5000,
                                    context_length: int = 10000,
                                    batch_size: int = 128) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Generate SpliceAI predictions centered on each splice site.
        For each splice site, gets predictions for the entire 5000bp window around it.
        """
        acceptor_predictions = {}
        donor_predictions = {}
        fasta = Fasta(self.consensus_fasta)
        
        print("Loading models...")
        models = []
        for x in range(1, 6):
            model = load_model(f'/data/zusers/ramirezc/splice-benchmark/spliceai/models/spliceai{x}.h5', compile=False)
            models.append(model)

        def reverse_complement(seq):
            comp = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N', '*': 'N'}
            return ''.join(comp.get(base, 'N') for base in reversed(seq.upper()))

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

        def process_batch(batch, metadata, models):
            """Process a batch of sequences and update predictions for entire windows."""
            if not batch:
                return
            
            x = one_hot_encode_batch(batch)
            
            # Get predictions from ensemble
            predictions = []
            for model in models:
                pred = model(x, training=False)
                if isinstance(pred, np.ndarray):
                    predictions.append(pred)
                else:
                    predictions.append(pred.numpy())
            
            # Average predictions
            avg_preds = np.mean(predictions, axis=0)  # Shape: [batch_size, seq_length, 2]
            
            # Update prediction arrays for each sequence in batch
            for i, meta in enumerate(metadata):
                center_position = meta['position']
                chrom = meta['chrom']
                strand = meta['strand']
                site_type = meta['type']
                
                # Get the entire window of predictions (all 5000 positions)
                acceptor_window = avg_preds[i, :, 1]  # Shape: [seq_length]
                donor_window = avg_preds[i, :, 2]     # Shape: [seq_length]
                
                # If on negative strand, reverse the predictions
                if strand == '-':
                    acceptor_window = acceptor_window[::-1]
                    donor_window = donor_window[::-1]
                
                # Calculate the genomic range for this window
                half_window = sequence_length // 2
                window_start = center_position - half_window
                window_end = center_position + half_window
                
                # Map predictions back to genomic coordinates
                for j in range(sequence_length):
                    genome_pos = window_start + j
                    
                    # Only update if position is within chromosome bounds
                    if 0 <= genome_pos < len(acceptor_predictions[chrom]):
                        # Update predictions at this genomic position
                        acceptor_predictions[chrom][genome_pos] = max(
                            acceptor_predictions[chrom][genome_pos],
                            float(acceptor_window[j])
                        )
                        donor_predictions[chrom][genome_pos] = max(
                            donor_predictions[chrom][genome_pos],
                            float(donor_window[j])
                        )

        for chrom in tqdm(self.target_chromosomes, desc="Processing chromosomes"):
            print(f"\nProcessing chromosome {chrom}")
            chrom_length = len(fasta[chrom])
            acceptor_predictions[chrom] = np.zeros(chrom_length)
            donor_predictions[chrom] = np.zeros(chrom_length)
            
            for site_type in ['donor', 'acceptor']:
                current_batch = []
                batch_metadata = []
                
                for site in tqdm(self.exon_sites[chrom][site_type], 
                            desc=f"Processing {site_type} sites"):
                    position = site['position']
                    strand = site['strand']
                    
                    # Calculate window around splice site (sequence_length + context_length)
                    total_length = sequence_length + context_length
                    half_total = total_length // 2
                    
                    window_start = max(0, position - (half_total - 1))
                    window_end = min(chrom_length, position + half_total)
                    
                    seq = str(fasta[chrom][window_start:window_end])
                    
                    # Add padding if necessary
                    pad_left = max(0, half_total - (position - window_start))
                    pad_right = max(0, half_total - (window_end - position))
                    seq = 'N' * pad_left + seq + 'N' * pad_right
                    
                    if strand == '-':
                        seq = reverse_complement(seq)
                    
                    current_batch.append(seq)
                    batch_metadata.append({
                        'position': position,
                        'strand': strand,
                        'type': site_type,
                        'chrom': chrom
                    })
                    
                    if len(current_batch) == batch_size:
                        process_batch(current_batch, batch_metadata, models)
                        current_batch = []
                        batch_metadata = []
                
                # Process remaining sequences in the last batch
                if current_batch:
                    process_batch(current_batch, batch_metadata, models)

        if predictions_file:
            print(f"\nSerializing predictions to {predictions_file}")
            with open(predictions_file, 'wb') as f:
                pickle.dump((acceptor_predictions, donor_predictions), f)
        
        return acceptor_predictions, donor_predictions

    def load_serialized_data(self, 
                           file_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Load serialized predictions or ground truth data.
        
        Args:
            file_path: Path to serialized data file
            
        Returns:
            Tuple of dictionaries containing acceptor and donor data
        """
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def calculate_metrics(self, 
                        ground_truth: Dict[str, np.ndarray], 
                        predictions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Calculate AUPRC and other metrics using only regions where predictions were made.
        """
        print(f"Calculating AUPRC using non-zero prediction regions...")
        
        filtered_truth = []
        filtered_preds = []
        
        for chrom in self.target_chromosomes:
            # Get mask where predictions were made (non-zero)
            pred_mask = predictions[chrom] != 0
            print(f"Chromosome {chrom}: {np.sum(pred_mask)} positions evaluated")
            
            # Add filtered regions
            filtered_truth.append(ground_truth[chrom][pred_mask])
            filtered_preds.append(predictions[chrom][pred_mask])
        
        # Concatenate all filtered regions
        all_truth = np.concatenate(filtered_truth)
        all_preds = np.concatenate(filtered_preds)
        
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        k = int(np.sum(all_truth))
        top_k_indices = np.argsort(all_preds)[-k:]
        top_k_accuracy = np.sum(all_truth[top_k_indices]) / k
        
        return precision, recall, auprc, top_k_accuracy