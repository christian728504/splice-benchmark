# %%
import os

os.chdir('/data/zusers/ramirezc/splice-benchmark/spliceai/tissue_specific')

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyfaidx import Fasta
from typing import Dict, Tuple, Set, Optional
from consensus_evaluator import ConsensusSpliceSiteEvaluator

# %%
# Initialize evaluator
evaluator = ConsensusSpliceSiteEvaluator(
    gencode_gtf="gencode.v29.primary_assembly.annotation_UCSC_names.gtf",
    consensus_fasta="GM12878/GM12878.fasta"
)

# %%
# Filter for expressed genes
expressed_transcripts = evaluator.filter_expressed_transcripts(
    quant_tsv="GM12878/ENCFF190NFH.tsv",
    min_tpm=2.0
)

print(list(expressed_transcripts)[:50])

# %%
# Load or generate and save ground truth
ground_truth_acceptor, ground_truth_donor = evaluator.get_ground_truth(
    expressed_transcripts=expressed_transcripts,
    ground_truth_file="GM12878/ground_truth.pkl"
)

# %%
# Generate predictions with serialization
pred_acceptor, pred_donor = evaluator.generate_spliceai_predictions(
    predictions_file="GM12878/predictions.pkl"
)

# %%
ground_truth_acceptor, ground_truth_donor = evaluator.load_serialized_data('GM12878/ground_truth.pkl')
pred_acceptor, pred_donor = evaluator.load_serialized_data('GM12878/predictions.pkl')

# %%
# Add this code to check predictions
print("\nPrediction verification:")
for chrom in evaluator.target_chromosomes:
    nonzero_count = np.count_nonzero(pred_acceptor[chrom])
    high_value_count = np.sum(pred_acceptor[chrom] >= 0.9)
    print(f"Chromosome {chrom}: {nonzero_count} non-zero predictions, {high_value_count} values ≥ 0.9")
    if nonzero_count > 0:
        # Check distribution of values
        values, counts = np.unique(pred_acceptor[chrom][pred_acceptor[chrom] > 0], return_counts=True)
        print(f"Value distribution: {list(zip(values[:5], counts[:5]))}")

# %%
# Calculate metrics
acceptor_precision, acceptor_recall, acceptor_auprc, acceptor_top_k = evaluator.calculate_metrics(
    ground_truth_acceptor, 
    pred_acceptor
)
donor_precision, donor_recall, donor_auprc, donor_top_k = evaluator.calculate_metrics(
    ground_truth_donor,
    pred_donor
)

# %%
# Calculate mean metrics
mean_auprc = (acceptor_auprc + donor_auprc) / 2
mean_top_k = (acceptor_top_k + donor_top_k) / 2

# Plot precision-recall curves
plt.figure(figsize=(10, 6))
plt.plot(acceptor_recall, acceptor_precision, label=f'Acceptor (AUPRC={acceptor_auprc:.3f})')
plt.plot(donor_recall, donor_precision, label=f'Donor (AUPRC={donor_auprc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curves\nMean AUPRC: {mean_auprc:.3f}, Mean Top-k: {mean_top_k:.3f}')
plt.legend()
plt.grid(True)
plt.savefig("GM12878/auprc_topk_spliceai.png", dpi=300)

# Print results
print(f"Acceptor AUPRC: {acceptor_auprc:.4f}, Top-k: {acceptor_top_k:.4f}")
print(f"Donor AUPRC: {donor_auprc:.4f}, Top-k: {donor_top_k:.4f}")
print(f"Mean AUPRC: {mean_auprc:.4f}, Mean Top-k: {mean_top_k:.4f}")

# %%
# Get the indices for the region of interest 
gene_gt = ground_truth_ac}ceptor['chr1'][944203:958458]
gene_pred = pred_acceptor['chr1'][944203:958458]

# Get absolute indices where values are positive
gt_positive_indices = {idx + 944203 for idx in np.where(gene_gt == 1)[0]}
pred_positive_indices = {idx + 944203 for idx in np.where(gene_pred >= 0.9)[0]}

# Write to file
with open('GM12878/splice_indices.txt', 'w') as f:
    f.write("Ground_Truth_Position\tPrediction_Position\n")  # Header
    
    # Write all positions, marking matches
    all_positions = sorted(gt_positive_indices | pred_positive_indices)
    for pos in all_positions:
        gt_mark = str(pos) if pos in gt_positive_indices else "-"
        pred_mark = str(pos) if pos in pred_positive_indices else "-"
        f.write(f"{gt_mark}\t{pred_mark}\n")

# Print summary statistics
print(f"Number of ground truth splice sites: {len(gt_positive_indices)}")
print(f"Number of predicted splice sites: {len(pred_positive_indices)}")
print(f"Number of matching positions: {len(gt_positive_indices & pred_positive_indices)}")

# %%



