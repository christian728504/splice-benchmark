from pyfaidx import Fasta
import torch
import numpy as np
from gtfparse import read_gtf
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm
import pickle
from pkg_resources import resource_filename
from pangolin.model import Pangolin, L, W, AR

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
        
    def parse_gencode(self):
        """Parse GENCODE GTF to extract splice sites."""
        fasta = Fasta(self.fasta_file)
        splice_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        
        # Read GTF file using gtfparse
        df = read_gtf(
            self.gtf_file,
            features={'exon', 'transcript'},
            result_type='pandas'
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
            
        print(transcript_df.shape)
        print(transcript_df.head())

        for _, transcript in transcript_df.iterrows():
            # Find median length of all transcripts
            transcript_lengths = []
            transcript_lengths.append(transcript['end'] - transcript['start'])
        
        median_transcript_length = np.median(transcript_lengths)
        print(f"Median transcript length: {median_transcript_length}")
        
        # Process exons to get splice sites
        exon_df = df[df['feature'] == 'exon']
        for _, exon in exon_df.iterrows():
            chrom = exon['seqname']
            # Mark both start and end as splice sites without distinguishing type
            splice_sites[chrom][exon['end']-1] = 1
            splice_sites[chrom][exon['start']-1] = 1
        
        return splice_sites
        
    def generate_predictions(self, ground_truth):
        """Generate splice site predictions using Pangolin model."""
        
        predictions = {}
        context = 15000  # Total context size (5000 before + 5000 prediction + 5000 after)
        step_size = 5000  # Size of prediction window
        batch_size = 128
        
        # Initialize Pangolin models for all tissue types
        print("Loading models...")
        model_nums = [0, 2, 4, 6]  # P(splice) models for Heart, Liver, Brain, Testis
        models = []
        for i in model_nums:
            for j in range(1, 6):
                model = Pangolin(L, W, AR)
                if torch.cuda.is_available():
                    model.cuda()
                    weights = torch.load(resource_filename("pangolin", f"models/final.{j}.{i}.3"))
                else:
                    weights = torch.load(resource_filename("pangolin", f"models/final.{j}.{i}.3"),
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
            """One-hot encoding following Pangolin's implementation."""
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
            predictions[chrom] = np.zeros(padded_length)
            
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
                    forward = one_hot_encode(seq, '+').T
                    batch_sequences.append(forward)
                
                x = torch.from_numpy(np.stack(batch_sequences)).float()
                if torch.cuda.is_available():
                    x = x.cuda()
                
                # Get predictions from all models
                splice_scores = []
                
                for model in models:
                    with torch.no_grad():
                        preds = model(x)
                        if isinstance(preds, torch.Tensor):
                            preds = preds.cpu().numpy()
                        
                        # For each model, get the probability OF being a splice site (index 1)
                        splice_scores.append(preds[:current_batch_size, 1, :])
                
                # Average predictions across all models
                batch_predictions = np.mean(splice_scores, axis=0)
                
                # Update predictions
                for i, pos in enumerate(batch_positions):
                    center_start = pos + context//3
                    center_end = center_start + step_size
                    
                    predictions[chrom][center_start:center_end] = np.maximum(
                        predictions[chrom][center_start:center_end],
                        batch_predictions[i, :step_size]
                    )
            
            # Trim padding
            predictions[chrom] = predictions[chrom][context//3:-context//3]
        
        return predictions
    
    def load_predictions(self, predictions_pkl):
        """Load and trim prediction arrays to only include positions within genes."""
        with open(predictions_pkl, 'rb') as f:
            pred = pickle.load(f)
            
        trimmed_pred = {}
        
        for chrom in self.target_chromosomes:
            trimmed_pred[chrom] = np.zeros_like(pred[chrom])
            transcript_pos = np.array(list(self.transcript_positions[chrom]))
            if len(transcript_pos) > 0:
                trimmed_pred[chrom][transcript_pos] = pred[chrom][transcript_pos]
        
        return trimmed_pred
    
    def calculate_metrics(self, ground_truth, predictions):
        """Calculate AUPRC and other metrics including top-k accuracy."""
        # Concatenate all chromosomes
        all_truth = np.concatenate([ground_truth[chrom] for chrom in self.target_chromosomes])
        all_preds = np.concatenate([predictions[chrom] for chrom in self.target_chromosomes])
        
        # Calculate precision-recall curve and AUPRC
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        # Calculate top-k accuracy
        k = int(np.sum(all_truth))  # k = number of actual splice sites
        top_k_indices = np.argsort(all_preds)[-k:]
        top_k_accuracy = np.sum(all_truth[top_k_indices]) / k
        
        return precision, recall, auprc, top_k_accuracy