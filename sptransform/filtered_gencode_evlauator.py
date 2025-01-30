import os
import pickle
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Tuple

class GencodeSpliceSiteEvaluator:
    def __init__(self, gencode_gtf: str, fasta_file: str, window_size: int = 10000):
        self.gtf_file = gencode_gtf
        self.fasta_file = fasta_file
        self.target_chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
        self.window_size = window_size
        
    def parse_gencode(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        fasta = Fasta(self.fasta_file)
        acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        
        exons = pd.read_csv(
            self.gtf_file, 
            sep='\t',
            comment='#',
            names=['chrom', 'source', 'feature', 'start', 'end', 'score', 
                   'strand', 'frame', 'attributes']
        )
        exons = exons[
            (exons['source'] == 'HAVANA') &
            (exons['feature'] == 'exon') & 
            (exons['strand'] == '+') & 
            (exons['chrom'].isin(self.target_chromosomes))
        ]
        
        for _, exon in exons.iterrows():
            chrom = exon['chrom']
            donor_sites[chrom][exon['end']-1] = 1
            acceptor_sites[chrom][exon['start']-1] = 1
        
        return acceptor_sites, donor_sites
    
    # def _apply_windows(self, predictions: Dict[str, np.ndarray], ground_truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    #     """Filter predictions to only include windows around ground truth sites."""
    #     filtered = {}
    #     for chrom in self.target_chromosomes:
    #         # Initialize filtered array with zeros
    #         filtered[chrom] = np.zeros_like(predictions[chrom])
            
    #         # Get ground truth positions
    #         annotation_positions = np.where(ground_truth[chrom] == 1)[0]
            
    #         # For each annotation, copy window of predictions
    #         half_window = self.window_size // 2
    #         for pos in annotation_positions:
    #             start = max(0, pos - half_window)
    #             end = min(len(predictions[chrom]), pos + half_window)
    #             filtered[chrom][start:end] = predictions[chrom][start:end]
                
    #     return filtered

    def _apply_windows(self, predictions: Dict[str, np.ndarray], ground_truth: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        filtered = {}
        for chrom in self.target_chromosomes:
            filtered[chrom] = np.zeros_like(predictions[chrom])
            annotation_positions = np.where(ground_truth[chrom] == 1)[0]
            
            half_window = self.window_size // 2
            for pos in annotation_positions:
                start = max(0, pos - half_window)
                end = min(len(predictions[chrom]), pos + half_window + 1)  # Added +1
                filtered[chrom][start:end] = predictions[chrom][start:end]
                
        return filtered
    
    def load_predictions(self, acceptor_pkl: str, donor_pkl: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Load and filter predictions using annotation windows."""
        with open(acceptor_pkl, 'rb') as f:
            acceptor_pred = pickle.load(f)
        with open(donor_pkl, 'rb') as f:
            donor_pred = pickle.load(f)
        
        # Get ground truth first
        acceptor_truth, donor_truth = self.parse_gencode()
            
        # Filter predictions using windows around annotations
        acceptor_filtered = self._apply_windows(acceptor_pred, acceptor_truth)
        donor_filtered = self._apply_windows(donor_pred, donor_truth)
        
        return acceptor_filtered, donor_filtered
    
    def calculate_metrics(self, ground_truth: Dict[str, np.ndarray], 
                         predictions: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
        all_truth = np.concatenate([ground_truth[chrom] for chrom in self.target_chromosomes])
        all_preds = np.concatenate([predictions[chrom] for chrom in self.target_chromosomes])
        
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        return precision, recall, auprc