import os
import pickle
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, Tuple, Set
from tqdm.notebook import tqdm
import jax
import jax.numpy as jnp
import haiku as hk
from nucleotide_transformer.pretrained import get_pretrained_segment_nt_model
from subprocess import check_output

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
        self.target_chromosomes = ['chr20', 'chr21']
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

    def get_gpu_info(self):
        try:
            output = check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,nounits,noheader'])
            used, total, util = map(int, output.decode('utf-8').strip().split(','))
            return f"GPU Memory: {used}/{total} MB | GPU Util: {util}%"
        except:
            return "GPU monitoring failed"

    def generate_segmentnt_predictions(self, ground_truth):
        """Generate splice site predictions using SegmentNT model with JAX."""
        import jax
        import jax.numpy as jnp
        import haiku as hk
        from nucleotide_transformer.pretrained import get_pretrained_segment_nt_model
        
        acceptor_predictions = {}
        donor_predictions = {}
        fasta = Fasta(self.fasta_file)
        
        # Initialize model
        print("Loading SegmentNT model...")
        max_tokens = 8336
        parameters, forward_fn, tokenizer, config = get_pretrained_segment_nt_model(
            model_name="segment_nt",
            max_positions=max_tokens + 1
        )
        
        # Transform model
        forward_fn = hk.transform(forward_fn)
        
        # Set up JAX function
        devices = jax.devices("gpu")  # or "gpu" if available
        apply_fn = jax.pmap(forward_fn.apply, devices=devices, donate_argnums=(0,))
        
        # Prepare model parameters
        random_key = jax.random.PRNGKey(seed=0)
        keys = jax.device_put_replicated(random_key, devices=devices)
        parameters = jax.device_put_replicated(parameters, devices=devices)
        
        # Get feature indices
        donor_idx = config.features.index('splice_donor')
        acceptor_idx = config.features.index('splice_acceptor')
        
        sequence_length = max_tokens * 6
        
        for chrom in tqdm(self.target_chromosomes, desc="Processing chromosomes"):
            chrom_length = len(ground_truth[chrom])
            full_sequence = str(fasta[chrom])

            print(len(ground_truth[chrom]))
            print(len(fasta[chrom]))
            
            acceptor_predictions[chrom] = np.zeros(chrom_length)
            donor_predictions[chrom] = np.zeros(chrom_length)
            
            pbar = tqdm(range(0, chrom_length, sequence_length))
            for start_pos in pbar:
                end_pos = min(start_pos + sequence_length, chrom_length)
                sequence = [full_sequence[start_pos:end_pos].upper()]
                
                pbar.set_postfix_str(self.get_gpu_info())

                # Skip if sequence has any N
                if 'N' in sequence[0]:
                    continue

                # Tokenize
                tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequence)]
                tokens = jnp.stack([jnp.asarray(tokens_ids, dtype=jnp.int32)], axis=0)
                
                # Generate predictions
                outputs = apply_fn(parameters, keys, tokens)
                probabilities = jax.nn.softmax(outputs["logits"], axis=-1)[..., -1]
                
                # Extract splice site probabilities
                donor_probs = np.array(probabilities[0, 0, :, donor_idx])
                acceptor_probs = np.array(probabilities[0, 0, :, acceptor_idx])
                
                # Map back to genome coordinates (using 3 predictions per token)
                window_size = (end_pos - start_pos) / len(donor_probs)
                positions = np.arange(len(donor_probs)) * window_size
                positions = positions.astype(int) + start_pos
                
                # Update predictions
                for i, pos in enumerate(positions):
                    if pos < chrom_length:
                        acceptor_predictions[chrom][pos] = acceptor_probs[i]
                        donor_predictions[chrom][pos] = donor_probs[i]
        
        return acceptor_predictions, donor_predictions

    def trim_predictions(self, acceptor_pred, donor_pred):
        """Load and trim prediction arrays to only include positions within genes."""
            
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
