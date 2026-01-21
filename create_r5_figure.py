#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Recall@5 figure similar to fixed_size_128_hardness_stratified.pdf
but for Recall@5 and without the baseline line.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paper style
def set_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linestyle": ":", "grid.alpha": 0.30,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

palette = sns.color_palette("Set2", n_colors=8)

# Load the R@5 results
print("Loading Recall@5 results...")
df = pd.read_csv("figs/fixed_size_128_hardness_stratified_r5.csv")

print(f"Loaded {len(df)} bins")

# Set paper style
set_paper_style()

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)

bin_labels = df['max_sim_bin'].tolist()
recalls = df['recall_at_5'].tolist()
n_queries = df['n_queries'].tolist()

# Create bar plot
bars = ax.bar(range(len(bin_labels)), recalls, color=palette[0], alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (recall, n) in enumerate(zip(recalls, n_queries)):
    ax.text(i, recall + 0.02, f'{recall:.3f}\n(n={n})', 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel('Maximum Tanimoto Similarity of Hardest Decoy', fontsize=16)
ax.set_ylabel('Recall@5', fontsize=16)
ax.set_title(f'Retrieval Performance vs. Chemical Hardness\n(Fixed Pool Size: N=128)', 
             fontsize=18, pad=10)
ax.set_xticks(range(len(bin_labels)))
ax.set_xticklabels(bin_labels, fontsize=14, rotation=45, ha='right')
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, linestyle=':', axis='y')

# No baseline line (removed as requested)

plt.tight_layout()

# Save figure
fig_path = "figs/fixed_size_128_hardness_stratified_r5.pdf"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {fig_path}")
plt.close()

print("\nFigure creation complete!")
