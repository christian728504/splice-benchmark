import subprocess
import os
import numpy as np
import pandas as pd
import bioframe
from sklearn.metrics import precision_recall_curve, auc
from pyfaidx import Fasta
from sptransformer import Annotator
import torch
from tqdm.notebook import tqdm
from scipy import stats

class SpliceSiteEvaluator:
    def __init__(self, bam_file: str, fasta_file: str):
        """
        Initialize the splice site evaluator with separate processing for acceptor and donor sites.
        
        Args:
            bam_file: Path to BAM alignment file
            fasta_file: Path to hg38 FASTA file
        """
        self.bam_file = bam_file
        self.fasta_file = fasta_file
        self.target_chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        self.window_size = 9000  # Size of sliding window for predictions
        
        # Initialize model
        self.annotator = Annotator()
        
        # Ensure BAM index exists
        self._ensure_bam_index()

    def _ensure_bam_index(self):
        """Create BAM index if it doesn't exist."""
        if not os.path.exists(f"{self.bam_file}.bai"):
            subprocess.run(['samtools', 'index', self.bam_file], check=True)

    def generate_ground_truth(self):
        """
        Generate ground truth labels from BAM file using a median read count threshold.
        
        Returns:
            Tuple of dicts: (acceptor_ground_truth, donor_ground_truth)
        """
        # Extract junctions using regtools
        bed_output = "junctions.bed"
        if os.path.exists(bed_output):
            print(f"Found existing {bed_output}, using existing file...")
        else:
            print("Extracting junctions using regtools...")
            subprocess.run([
                'regtools', 'junctions', 'extract',
                '-s', 'RF',
                self.bam_file,
                '-o', bed_output
            ], check=True)
        
        # Load junctions
        junctions_df = bioframe.read_table(bed_output, schema='bed12')
        
        # Calculate median read count threshold
        median_read_count = junctions_df['score'].median()
        print(f"Median junction read count: {median_read_count}")
        
        # Create ground truth arrays for acceptor and donor sites
        fasta = Fasta(self.fasta_file)
        acceptor_ground_truth = {}
        donor_ground_truth = {}
        
        # Print per-chromosome statistics
        print("\nPer-chromosome junction counts:")
        for chrom in self.target_chromosomes:
            # Initialize arrays for chromosome
            chrom_length = len(fasta[chrom])
            acceptor_ground_truth[chrom] = np.zeros(chrom_length, dtype=np.int8)
            donor_ground_truth[chrom] = np.zeros(chrom_length, dtype=np.int8)
            
            # Filter junctions for this chromosome and positive strand
            chrom_junctions = junctions_df[
                (junctions_df['chrom'] == chrom) & 
                (junctions_df['strand'] == '+')
            ]
            
            # Mark sites above median read count
            for _, junction in chrom_junctions.iterrows():
                # If junction score is above median, mark as positive
                if junction['score'] > median_read_count:
                    # Mark acceptor site (start position)
                    acceptor_ground_truth[chrom][junction['end']] = 1
                    # Mark donor site (end position)
                    donor_ground_truth[chrom][junction['start']] = 1
            
            print(f"{chrom}: Total junctions {len(chrom_junctions)}, " +
                  f"Positive acceptor sites: {np.sum(acceptor_ground_truth[chrom])}, " +
                  f"Positive donor sites: {np.sum(donor_ground_truth[chrom])}")
        
        return acceptor_ground_truth, donor_ground_truth

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

    def calculate_auprc(self, ground_truth, predictions):
        """
        Calculate AUPRC for a specific site type (acceptor or donor).
        
        Args:
            ground_truth: Dict of ground truth arrays
            predictions: Dict of prediction arrays
        
        Returns:
            AUPRC score
        """
        # Flatten arrays for all chromosomes
        all_truth = np.concatenate([ground_truth[chrom] for chrom in self.target_chromosomes])
        all_preds = np.concatenate([predictions[chrom] for chrom in self.target_chromosomes])
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        return auprc

    def evaluate(self):
        """
        Run complete evaluation pipeline with separate processing for acceptor and donor sites.
        
        Returns:
            Tuple containing:
            - Acceptor ground truth
            - Donor ground truth
            - Acceptor predictions
            - Donor predictions
            - Acceptor AUPRC
            - Donor AUPRC
            - Mean AUPRC
        """
        print("Generating ground truth labels...")
        acceptor_ground_truth, donor_ground_truth = self.generate_ground_truth()
        
        print("\nGenerating predictions...")
        acceptor_predictions, donor_predictions = self.generate_predictions(acceptor_ground_truth)
        
        print("\nCalculating AUPRC...")
        acceptor_auprc = self.calculate_auprc(acceptor_ground_truth, acceptor_predictions)
        donor_auprc = self.calculate_auprc(donor_ground_truth, donor_predictions)
        
        # Calculate mean AUPRC
        mean_auprc = (acceptor_auprc + donor_auprc) / 2
        
        print(f"\nAcceptor Site AUPRC: {acceptor_auprc:.4f}")
        print(f"Donor Site AUPRC: {donor_auprc:.4f}")
        print(f"Mean AUPRC: {mean_auprc:.4f}")
        
        return (
            acceptor_ground_truth, 
            donor_ground_truth, 
            acceptor_predictions, 
            donor_predictions, 
            acceptor_auprc, 
            donor_auprc,
            mean_auprc
        )