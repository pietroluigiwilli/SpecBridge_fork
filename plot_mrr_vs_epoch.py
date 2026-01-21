#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot MRR vs epoch for alignment vs contrastive training
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paper style (from specbridge_analysis.py)
def set_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linestyle": ":", "grid.alpha": 0.30,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

# Colors from seaborn set2 palette
palette = sns.color_palette("Set2", n_colors=8)
COLOR_ALIGN = palette[2]      # First color for alignment
COLOR_CONTRASTIVE = palette[1]  # Second color for contrastive

def main():
    # Paths
    align_csv = "/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/eval_summary_val_all.csv"
    contrastive_csv = "/cluster/tufts/liulab/yiwan01/SpecBridge/runs/ablation_study_spectraverse/3_pretrained_contrastive/eval_summary_val_all.csv"
    output_dir = "/cluster/tufts/liulab/yiwan01/SpecBridge/figs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read data
    df_align = pd.read_csv(align_csv)
    # Contrastive CSV doesn't have header, so we need to specify column names
    df_contrastive = pd.read_csv(contrastive_csv, header=None, 
                                  names=['ckpt', 'step', 'R@1', 'R@5', 'R@20', 'MRR', 'median_rank', 'total_queries', 'evaluated'])
    
    # Filter out 'last' checkpoint entries (they're not at a specific step)
    # Check for both 'last' string and entries where step contains 'last'
    df_align = df_align[df_align['step'].astype(str).str.lower() != 'last'].copy()
    df_contrastive = df_contrastive[df_contrastive['step'].astype(str).str.lower() != 'last'].copy()
    
    # Convert step to numeric (errors='coerce' will convert invalid entries to NaN, then we drop them)
    df_align['step'] = pd.to_numeric(df_align['step'], errors='coerce')
    df_contrastive['step'] = pd.to_numeric(df_contrastive['step'], errors='coerce')
    
    # Drop any rows with NaN steps
    df_align = df_align.dropna(subset=['step']).copy()
    df_contrastive = df_contrastive.dropna(subset=['step']).copy()
    df_align = df_align.sort_values('step')
    df_contrastive = df_contrastive.sort_values('step')
    
    # Use steps directly (not epochs)
    df_align['step_plot'] = df_align['step']
    df_contrastive['step_plot'] = df_contrastive['step']
    
    # Set style
    set_paper_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
    
    # Plot alignment
    ax.plot(df_align['step_plot'], df_align['MRR'], 
            color=COLOR_ALIGN, linewidth=1.5, label='Alignment', alpha=0.9)
    
    # Plot contrastive
    ax.plot(df_contrastive['step_plot'], df_contrastive['MRR'], 
            color=COLOR_CONTRASTIVE, linewidth=1.5, label='Contrastive', alpha=0.9)
    
    # Labels and title
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Validation MRR')
    ax.set_title('Validation MRR vs. training steps')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Legend
    ax.legend(frameon=False, loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, "mrr_vs_epoch.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()

