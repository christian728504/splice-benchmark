import pickle
import numpy as np
import tensorflow as tf
import h5py
from keras.models import load_model
from tqdm import tqdm

def read_fasta(fasta_file):
    """Parse FASTA file without using pyfaidx"""
    sequences = {}
    current_chrom = None
    current_seq = []
    
    print("Reading FASTA file...")
    with open(fasta_file, 'r') as f:
        for line in tqdm(f):
            line = line.strip()
            if line.startswith('>'):
                if current_chrom:
                    sequences[current_chrom] = ''.join(current_seq)
                current_chrom = line[1:].split()[0]  # Get chromosome name
                current_seq = []
            else:
                current_seq.append(line.upper())
                
    if current_chrom:
        sequences[current_chrom] = ''.join(current_seq)
        
    return sequences

def generate_splice_predictions(fasta_file, model_dir, output_dir):
    """
    Generate SpliceAI predictions and save to pickle files.
    
    Args:
        fasta_file: Path to reference genome FASTA
        model_dir: Directory containing SpliceAI model files (spliceai1.h5 through spliceai5.h5)
        output_dir: Directory to save output pickle files
    """

    # Configure TensorFlow to use all cores
    num_cores = 16
    config = tf.ConfigProto(
        intra_op_parallelism_threads=num_cores,
        inter_op_parallelism_threads=num_cores,
        allow_soft_placement=True
    )

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    
    # Parameters
    chromosomes = ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
    context = 15000
    step_size = 5000
    batch_size = 128
    
    # Load genome
    genome = read_fasta(fasta_file)
    
    # Load models
    print("Loading models...")
    models = []
    for x in tqdm(range(1, 6)):
        model = load_model(f'{model_dir}/spliceai{x}.h5')
        models.append(model)

    def one_hot_encode_batch(sequences):
        sequence_length = len(sequences[0])
        batch_size = len(sequences)
        encoding = np.zeros((batch_size, sequence_length, 4), dtype=np.float32)
        
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq):
                if base in base_to_index:
                    encoding[i, j, base_to_index[base]] = 1
        return encoding

    acceptor_predictions = {}
    donor_predictions = {}
    
    print("Processing chromosomes...")
    for chrom in tqdm(chromosomes, desc="Chromosomes"):
        if chrom not in genome:
            print(f"Warning: {chrom} not found in genome, skipping...")
            continue
            
        print(f"\nProcessing {chrom}...")
        chrom_seq = genome[chrom]
        chrom_length = len(chrom_seq)
        pad_seq = 'N' * (context//3)
        full_chrom_seq = pad_seq + chrom_seq + pad_seq
        
        padded_length = chrom_length + 2*(context//3)
        acceptor_predictions[chrom] = np.zeros(padded_length)
        donor_predictions[chrom] = np.zeros(padded_length)
        
        positions = range(0, padded_length - context, step_size)
        num_batches = (len(positions) + batch_size - 1) // batch_size
        
        batch_pbar = tqdm(total=num_batches, desc=f"{chrom} batches")
        
        for batch_start in range(0, len(positions), batch_size):
            batch_end = min(batch_start + batch_size, len(positions))
            batch_positions = positions[batch_start:batch_end]
            
            sequences = [full_chrom_seq[pos:pos + context] 
                       for pos in batch_positions]
            
            x = one_hot_encode_batch(sequences)
            
            predictions = []
            for model in models:
                pred = model.predict(x, batch_size=batch_size)
                predictions.append(pred)
            
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
            
            batch_pbar.update(1)
        
        batch_pbar.close()
        
        # Trim padding
        acceptor_predictions[chrom] = acceptor_predictions[chrom][context//3:-context//3]
        donor_predictions[chrom] = donor_predictions[chrom][context//3:-context//3]
        
        # Free memory
        genome[chrom] = None
    
    print("\nSaving predictions...")
    with open(f'{output_dir}/acceptor_pred.pkl', 'wb') as f:
        pickle.dump(acceptor_predictions, f)
    with open(f'{output_dir}/donor_pred.pkl', 'wb') as f:
        pickle.dump(donor_predictions, f)
    print("Done!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate SpliceAI predictions')
    parser.add_argument('--fasta', required=True, help='Path to reference genome FASTA')
    parser.add_argument('--model-dir', required=True, help='Directory containing SpliceAI models')
    parser.add_argument('--output-dir', required=True, help='Directory to save predictions')
    
    args = parser.parse_args()
    generate_splice_predictions(args.fasta, args.model_dir, args.output_dir)