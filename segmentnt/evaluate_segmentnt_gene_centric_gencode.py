#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir('/pi/zhiping.weng-umw/data/ramirezc/splice-benchmark')


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from pickle_serialize import save
from pickle_serialize import load
from gene_centric_gencode_evaluator import GencodeSpliceSiteEvaluator


# In[3]:


# Initialize evaluator
evaluator = GencodeSpliceSiteEvaluator(
    gencode_gtf="/data/Splice/data/gencode.v47.basic.annotation.gtf",
    fasta_file="/data/genomes/hg38/hg38.fa"
)


# In[4]:


# Generate ground truth from GENCODE
acceptor_truth, donor_truth = evaluator.parse_gencode()

# save(acceptor_truth, "acceptor_truth")
# save(donor_truth, "donor_truth")

# # Load ground truth
# acceptor_truth = load("acceptor_truth")
# donor_truth = load("donor_truth")


# In[5]:


# Generate predictions using SegmentNT model
acceptor_pred, donor_pred = evaluator.generate_segmentnt_predictions(acceptor_truth)

save(acceptor_pred, "segmentnt_acceptor_pred")
save(donor_pred, "segmentnt_donor_pred")


# In[6]:


# Trim predictions to only include positions within genes
acceptor_pred = load('segmentnt_acceptor_pred')
donor_pred = load('segmentnt_donor_pred')

trimmed_acceptor, trimmed_donor = evaluator.trim_predictions(acceptor_pred, donor_pred)


# In[7]:


# Get metrics including top-k accuracy
acc_precision, acc_recall, acc_auprc, acc_topk = evaluator.calculate_metrics(
    acceptor_truth, trimmed_acceptor
)
don_precision, don_recall, don_auprc, don_topk = evaluator.calculate_metrics(
    donor_truth, trimmed_donor  
)


# In[8]:


# Calculate mean metrics
mean_auprc = (acc_auprc + don_auprc) / 2
mean_topk = (acc_topk + don_topk) / 2

# Plot precision-recall curves
plt.figure(figsize=(10, 6))
plt.plot(acc_recall, acc_precision, label=f'Acceptor (AUPRC={acc_auprc:.3f})')
plt.plot(don_recall, don_precision, label=f'Donor (AUPRC={don_auprc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curves\nMean AUPRC: {mean_auprc:.3f}, Mean Top-k: {mean_topk:.3f}')
plt.legend()
plt.grid(True)
plt.savefig("auprc_topk_spliceai.png", dpi=300)
plt.show()

# Print results
print(f"Acceptor AUPRC: {acc_auprc:.4f}, Top-k: {acc_topk:.4f}")
print(f"Donor AUPRC: {don_auprc:.4f}, Top-k: {don_topk:.4f}")
print(f"Mean AUPRC: {mean_auprc:.4f}, Mean Top-k: {mean_topk:.4f}")


# In[11]:


get_ipython().system('jupyter nbconvert --to python evaluate_segmentnt_gene_centric_gencode.ipynb')


# In[10]:


for chrom in acceptor_truth.keys():
   truth_sites = np.where(acceptor_truth[chrom] == 1)[0][:200]
   pred_sites = np.where(trimmed_acceptor[chrom] > 0.9)[0][:200]
   
   print(f"\n{chrom}")
   print("Truth sites:", truth_sites)
   print("Predicted sites (>0.9):", pred_sites)


# In[ ]:




