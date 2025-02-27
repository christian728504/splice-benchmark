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

    def parse_gencode(self, 
                    expressed_transcripts: Optional[Set[str]] = None,
                    ground_truth_file: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Parse GENCODE GTF to extract splice sites.
        Handles both positive and negative strands.
        """
        if ground_truth_file and os.path.exists(ground_truth_file):
            with open(ground_truth_file, 'rb') as f:
                return pickle.load(f)
        
        fasta = Fasta(self.consensus_fasta)
        acceptor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        donor_sites = {chrom: np.zeros(len(fasta[chrom])) for chrom in self.target_chromosomes}
        self.transcript_positions = {chrom: [] for chrom in self.target_chromosomes}

        # Read GTF file
        df = read_gtf(
            self.gtf_file,
            features={'exon', 'transcript'},
            result_type='pandas'
        )
        
        # Apply filters (without strand filter)
        df = df[
            (df['feature'].isin(['exon', 'transcript'])) &
            (df['gene_type'] == 'protein_coding') &
            (df['seqname'].isin(self.target_chromosomes))
        ]

        # Filter by expressed transcripts if provided
        if expressed_transcripts is not None:
            df = df[df['transcript_id'].isin(expressed_transcripts)]

        # Process transcripts to store positions and strand
        transcript_df = df[df['feature'] == 'transcript']
        for _, transcript in transcript_df.iterrows():
            chrom = transcript['seqname']
            self.transcript_positions[chrom].append({
                'start': transcript['start']-1,
                'end': transcript['end']-1,
                'strand': transcript['strand']
            })

        # Process exons to get splice sites
        exon_df = df[df['feature'] == 'exon']
        for _, exon in exon_df.iterrows():
            chrom = exon['seqname']
            if exon['strand'] == '+':
                # Positive strand: donor at end, acceptor at start
                donor_sites[chrom][exon['end']-1] = 1
                acceptor_sites[chrom][exon['start']-1] = 1
            else:
                # Negative strand: donor at start, acceptor at end
                donor_sites[chrom][exon['start']-1] = 1
                acceptor_sites[chrom][exon['end']-1] = 1

        if ground_truth_file:
            with open(ground_truth_file, 'wb') as f:
                pickle.dump((acceptor_sites, donor_sites), f)

        return acceptor_sites, donor_sites

    def generate_spliceai_predictions(self, predictions_file: Optional[str] = None,
                                    sequence_length: int = 5000,
                                    context_length: int = 10000,
                                    batch_size: int = 128) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Modified version with debug statements and fixed block handling
        """
        acceptor_predictions = {}
        donor_predictions = {}
        fasta = Fasta(self.consensus_fasta)
        
        # Load models
        print("Loading models...")
        models = []
        for x in range(1, 6):
            model = load_model(f'/data/zusers/ramirezc/splice-benchmark/spliceai/models/spliceai{x}.h5', compile=False)
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
        
        def reverse_complement(seq):
            """
            Return reverse complement of sequence.
            Unmapped regions (*) are treated as N.
            """
            comp = {
                'A': 'T', 
                'C': 'G', 
                'G': 'C', 
                'T': 'A', 
                'N': 'N'
            }
            return ''.join(comp[base] for base in reversed(seq.upper()))

        for chrom in tqdm(self.target_chromosomes, desc="Processing chromosomes"):
            print(f"\nProcessing chromosome {chrom}")
            chrom_length = len(fasta[chrom])
            acceptor_predictions[chrom] = np.zeros(chrom_length)
            donor_predictions[chrom] = np.zeros(chrom_length)
            
            for transcript in self.transcript_positions[chrom]:
                start = transcript['start']
                end = transcript['end']
                strand = transcript['strand']
                
                print(f"\nProcessing transcript {start}-{end} ({strand} strand)")
                
                # Add context
                region_start = max(0, start - context_length//2)
                region_end = min(chrom_length, end + context_length//2)
                print(f"Region with context: {region_start}-{region_end}")
                
                # Get sequence with context
                seq = str(fasta[chrom][region_start:region_end])
                if strand == '-':
                    seq = reverse_complement(seq)
                print(f"Sequence length: {len(seq)}")
                
                # Add padding if needed
                pad_left = max(0, context_length//2 - (start - region_start))
                pad_right = max(0, context_length//2 - (region_end - end))
                seq = 'N' * pad_left + seq + 'N' * pad_right
                print(f"Padding: left={pad_left}, right={pad_right}")
                
                # Split into blocks
                # Modified blocking to ensure proper context handling
                effective_length = len(seq) - context_length
                num_blocks = (effective_length + sequence_length - 1) // sequence_length
                blocks = []
                block_coords = []  # Store coordinates for debugging
                
                print(f"Number of blocks: {num_blocks}")
                
                for i in range(num_blocks):
                    block_start = i * sequence_length
                    block_end = min(block_start + sequence_length + context_length, len(seq))
                    if block_end - block_start >= sequence_length + context_length:
                        blocks.append(seq[block_start:block_end])
                        block_coords.append((block_start, block_end))
                        print(f"Block {i}: {block_start}-{block_end} (len={block_end-block_start})")
                
                # Process blocks in batches
                for batch_start in range(0, len(blocks), batch_size):
                    batch_end = min(batch_start + batch_size, len(blocks))
                    batch_blocks = blocks[batch_start:batch_end]
                    
                    if not batch_blocks:
                        continue
                        
                    x = one_hot_encode_batch(batch_blocks)
                    
                    # Get predictions from ensemble
                    predictions = []
                    for model in models:
                        pred = model(x, training=False)
                        if isinstance(pred, np.ndarray):
                            predictions.append(pred)
                        else:
                            predictions.append(pred.numpy())
                    y = np.mean(predictions, axis=0)
                    
                    # Place predictions back in genome coordinates
                    for i, block_preds in enumerate(y):
                        block_idx = batch_start + i
                        block_start_coord = block_coords[block_idx][0]
                        
                        # Calculate genomic coordinates
                        # Adjust for context window
                        pred_start = region_start + block_start_coord + context_length//2
                        pred_end = min(pred_start + sequence_length, region_end)
                        
                        print(f"Placing predictions at genomic coordinates: {pred_start}-{pred_end}")
                        print(f"Raw prediction stats - Acceptor: min={np.min(block_preds[:sequence_length, 1]):.4f}, "
                            f"max={np.max(block_preds[:sequence_length, 1]):.4f}")
                        print(f"Raw prediction stats - Donor: min={np.min(block_preds[:sequence_length, 2]):.4f}, "
                            f"max={np.max(block_preds[:sequence_length, 2]):.4f}")
                        
                        if strand == '+':
                            acceptor_pred = block_preds[:sequence_length, 1]
                            donor_pred = block_preds[:sequence_length, 2]
                        else:
                            acceptor_pred = block_preds[:sequence_length, 2][::-1]
                            donor_pred = block_preds[:sequence_length, 1][::-1]
                        
                        # Update predictions using maximum values
                        pred_length = pred_end - pred_start
                        acceptor_predictions[chrom][pred_start:pred_end] = np.maximum(
                            acceptor_predictions[chrom][pred_start:pred_end],
                            acceptor_pred[:pred_length]
                        )
                        donor_predictions[chrom][pred_start:pred_end] = np.maximum(
                            donor_predictions[chrom][pred_start:pred_end],
                            donor_pred[:pred_length]
                        )
            
            # Print summary stats for chromosome
            print(f"\nChromosome {chrom} summary:")
            print(f"Acceptor predictions - min: {np.min(acceptor_predictions[chrom]):.4f}, "
                f"max: {np.max(acceptor_predictions[chrom]):.4f}")
            print(f"Donor predictions - min: {np.min(donor_predictions[chrom]):.4f}, "
                f"max: {np.max(donor_predictions[chrom]):.4f}")
        
        # Serialize predictions if path provided
        if predictions_file:
            with open(predictions_file, 'wb') as f:
                pickle.dump((acceptor_predictions, donor_predictions), f)
        
        return acceptor_predictions, donor_predictions

    def debug_sequence_blocking(self, 
                            sequence_length: int = 5000,
                            context_length: int = 10000):
        """Debug version that only shows sequence construction and blocking"""
        fasta = Fasta(self.consensus_fasta)
        
        for chrom in self.target_chromosomes[:1]:  # Just look at first chromosome
            print(f"\nProcessing chromosome {chrom}")
            chrom_length = len(fasta[chrom])
            
            # Just look at first few transcripts
            for transcript in self.transcript_positions[chrom][:2]:
                start = transcript['start']
                end = transcript['end']
                strand = transcript['strand']
                
                print(f"\n{'='*80}")
                print(f"Transcript {start}-{end} ({strand} strand)")
                print(f"Transcript length: {end-start}")
                
                # Add context
                region_start = max(0, start - context_length//2)
                region_end = min(chrom_length, end + context_length//2)
                print(f"Region with context: {region_start}-{region_end}")
                print(f"Context sizes: left={start-region_start}, right={region_end-end}")
                
                # Get sequence with context
                seq = str(fasta[chrom][region_start:region_end])
                print(f"Original sequence length: {len(seq)}")
                print(f"First 50bp: {seq[:50]}")
                print(f"Last 50bp: {seq[-50:]}")
                
                if strand == '-':
                    seq = reverse_complement(seq)
                    print(f"Reverse complemented:")
                    print(f"First 50bp: {seq[:50]}")
                    print(f"Last 50bp: {seq[-50:]}")
                
                # Add padding if needed
                pad_left = max(0, context_length//2 - (start - region_start))
                pad_right = max(0, context_length//2 - (region_end - end))
                seq = 'N' * pad_left + seq + 'N' * pad_right
                if pad_left > 0 or pad_right > 0:
                    print(f"Added padding: left={pad_left}, right={pad_right}")
                    print(f"New sequence length: {len(seq)}")
                
                # Show blocking
                effective_length = len(seq) - context_length
                num_blocks = (effective_length + sequence_length - 1) // sequence_length
                print(f"\nBlocking information:")
                print(f"Effective length (without context): {effective_length}")
                print(f"Number of blocks needed: {num_blocks}")
                
                # Show first few blocks
                for i in range(min(3, num_blocks)):
                    block_start = i * sequence_length
                    block_end = min(block_start + sequence_length + context_length, len(seq))
                    
                    print(f"\nBlock {i}:")
                    print(f"Block coordinates in sequence: {block_start}-{block_end}")
                    print(f"Block size: {block_end-block_start}")
                    if block_end - block_start >= sequence_length + context_length:
                        print("Block has full context")
                        # Show context regions
                        context_left = seq[block_start:block_start+context_length//2]
                        center = seq[block_start+context_length//2:block_end-context_length//2]
                        context_right = seq[block_end-context_length//2:block_end]
                        print(f"Left context (first 50bp): {context_left[:50]}...")
                        print(f"Center region (first 50bp): {center[:50]}...")
                        print(f"Right context (last 50bp): ...{context_right[-50:]}")
                    else:
                        print("Block is truncated")
                    
                    # Show where predictions would be placed
                    pred_start = region_start + block_start + context_length//2
                    pred_end = min(pred_start + sequence_length, region_end)
                    print(f"Would place predictions at: {pred_start}-{pred_end}")

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
        """Calculate AUPRC and other metrics including top-k accuracy for predictions."""
        all_truth = np.concatenate([ground_truth[chrom] for chrom in self.target_chromosomes])
        all_preds = np.concatenate([predictions[chrom] for chrom in self.target_chromosomes])
        
        precision, recall, _ = precision_recall_curve(all_truth, all_preds)
        auprc = auc(recall, precision)
        
        k = int(np.sum(all_truth))
        top_k_indices = np.argsort(all_preds)[-k:]
        top_k_accuracy = np.sum(all_truth[top_k_indices]) / k
        
        return precision, recall, auprc, top_k_accuracy