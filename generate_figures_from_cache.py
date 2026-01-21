#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load cache and generate only the two requested figures:
- hist_cos_query_target_vs_candidates.png
- hist_cos_true.png
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, asdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Define QueryRecord locally so pickle can load it
@dataclass
class QueryRecord:
    title: str
    true_smiles: str
    n_cands: int
    rank_true: Optional[int]
    cos_true: Optional[float]
    margin: Optional[float]
    angle_deg: Optional[float]
    pearson: Optional[float]
    spearman: Optional[float]
    sa5: Optional[float]
    sa10: Optional[float]
    sa20: Optional[float]
    auroc_tani05: Optional[float]
    ap_tani05: Optional[float]
    auroc_tani06: Optional[float]
    ap_tani06: Optional[float]
    auroc_tani07: Optional[float]
    ap_tani07: Optional[float]

# Use seaborn set2 color palette
sns.set_palette("Set2")
palette = sns.color_palette("Set2")
BLUE = palette[2]
ORANGE = palette[1]

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

def savefig(outdir: str, name: str, transparent: bool=False):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    plt.savefig(base + ".pdf", transparent=transparent)
    plt.savefig(base + ".png", dpi=300, transparent=transparent)
    plt.close()

# Load cache file
cache_file = "runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/analysis_ckpt_020600/cache/cache_a15b3cc93f2f4476.pkl"
outdir = "runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/analysis_ckpt_020600"
outfig = os.path.join(outdir, "figs")

print(f"Loading cache from: {cache_file}")
with open(cache_file, "rb") as f:
    X = pickle.load(f)

# Extract data
recs = X["recs"]
cos_target = X.get("cos_target", None)
cos_candidates_mean = X.get("cos_candidates_mean", None)

# Convert recs to DataFrame
df = pd.DataFrame([asdict(r) for r in recs])

set_paper_style()

# Figure 1: hist_cos_true.png
print("Generating hist_cos_true.png...")
vals = df["cos_true"].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
if vals.size:
    plt.figure(figsize=(3.35, 2.6), dpi=300)
    plt.hist(vals, bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.3, density=False)
    plt.xlabel("cos_true")
    plt.ylabel("count")
    plt.grid(axis="y")
    plt.title("Cosine to true")
    savefig(outfig, "hist_cos_true")
    print(f"Saved: {os.path.join(outfig, 'hist_cos_true.png')}")
else:
    print("Warning: No cos_true values found in cache")

# Figure 2: hist_cos_query_target_vs_candidates.png
print("Generating hist_cos_query_target_vs_candidates.png...")
if cos_target and cos_candidates_mean:
    # Convert to lists if numpy arrays
    if isinstance(cos_target, np.ndarray):
        cos_target = cos_target.tolist()
    if isinstance(cos_candidates_mean, np.ndarray):
        cos_candidates_mean = cos_candidates_mean.tolist()
    
    min_len = min(len(cos_target), len(cos_candidates_mean))
    cos_target_aligned = cos_target[:min_len]
    cos_candidates_mean_aligned = cos_candidates_mean[:min_len]
    
    if len(cos_target_aligned) > 0:
        plt.figure(figsize=(3.35, 2.6), dpi=300)
        bins = np.linspace(min(min(cos_target_aligned), min(cos_candidates_mean_aligned)) - 0.1, 
                          max(max(cos_target_aligned), max(cos_candidates_mean_aligned)) + 0.1, 100)
        plt.hist(cos_target_aligned, bins=bins, alpha=0.7, color=BLUE, 
                edgecolor="white", linewidth=0.03, label="Spectrum and target molecules", density=False)
        plt.hist(cos_candidates_mean_aligned, bins=bins, alpha=0.7, color=ORANGE, 
                edgecolor="white", linewidth=0.03, label="Spectrum and candidates", density=False)
        plt.xlabel("Cosine similarity")
        plt.ylabel("Counts")
        plt.xlim(0, 1)
        # No xlim restriction - show full range
        plt.legend(frameon=False, loc="best", fontsize=7)
        plt.grid(axis="y", alpha=0.3)
        plt.title("Distribution of cosine similarities")
        savefig(outfig, "hist_cos_query_target_vs_candidates")
        print(f"Saved: {os.path.join(outfig, 'hist_cos_query_target_vs_candidates.png')}")
    else:
        print("Warning: No aligned cosine values found")
else:
    print(f"Warning: Missing data - cos_target: {cos_target is not None}, cos_candidates_mean: {cos_candidates_mean is not None}")

print(f"\nDone! Figures saved to: {outfig}")
