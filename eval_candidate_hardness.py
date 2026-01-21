#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate candidate hardness by computing Tanimoto similarity between ground truth
and candidates. Compares MassSpecGym vs Spectraverse/PubChem candidate pools to
demonstrate that PubChem pools are harder (higher similarity = more isomers).
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    print("Warning: RDKit not available. Tanimoto similarity cannot be computed.")
    _HAS_RDKIT = False

# Paper style with larger text
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

# Colors from seaborn set2 palette
palette = sns.color_palette("Set2", n_colors=8)

def ecfp4(smiles: str, radius: int = 2, nbits: int = 4096):
    """Compute ECFP4 fingerprint for a SMILES string."""
    if not _HAS_RDKIT or not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

def tanimoto_similarity(fp1, fp2, default: float = np.nan) -> float:
    """Compute Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return default
    try:
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return default

def compute_similarity_metrics(gt_smiles: str, candidate_smiles_list: list, 
                                radius: int = 2, nbits: int = 4096, 
                                hard_threshold: float = 0.8, top_k: int = 10) -> dict:
    """
    Compute comprehensive similarity metrics between ground truth and all candidates.
    
    Args:
        gt_smiles: Ground truth SMILES string
        candidate_smiles_list: List of candidate SMILES strings
        radius: Morgan fingerprint radius (default 2 for ECFP4)
        nbits: Number of bits in fingerprint (default 4096)
        hard_threshold: Tanimoto threshold for "hard" candidates (default 0.8)
        top_k: Number of top candidates to average for mean top-k similarity (default 10)
    
    Returns:
        Dictionary with:
        - max_similarity: Maximum Tanimoto similarity
        - mean_topk_similarity: Mean of top-k similarities
        - n_hard: Number of candidates with similarity > hard_threshold
        - pool_size: Total number of valid candidates
        - all_similarities: List of all similarities (for debugging)
    """
    if not _HAS_RDKIT:
        return {
            'max_similarity': np.nan,
            'mean_topk_similarity': np.nan,
            'n_hard': 0,
            'pool_size': 0,
            'all_similarities': []
        }
    
    gt_fp = ecfp4(gt_smiles, radius=radius, nbits=nbits)
    if gt_fp is None:
        return {
            'max_similarity': np.nan,
            'mean_topk_similarity': np.nan,
            'n_hard': 0,
            'pool_size': 0,
            'all_similarities': []
        }
    
    similarities = []
    valid_candidates = 0
    
    for cand_smiles in candidate_smiles_list:
        if not cand_smiles or cand_smiles == gt_smiles:
            # Skip empty or identical candidates (self-similarity is always 1.0)
            continue
        
        cand_fp = ecfp4(cand_smiles, radius=radius, nbits=nbits)
        if cand_fp is None:
            continue
        
        sim = tanimoto_similarity(gt_fp, cand_fp)
        if not np.isnan(sim) and sim < 1.0:
            # Skip similarities of 1.0 (identical molecules, even if SMILES differ)
            similarities.append(sim)
            valid_candidates += 1
    
    # If no valid candidates found, return NaN values
    if valid_candidates == 0:
        return {
            'max_similarity': np.nan,
            'mean_topk_similarity': np.nan,
            'n_hard': 0,
            'pool_size': 0,
            'all_similarities': []
        }
    
    similarities = np.array(similarities)
    
    # Compute metrics
    max_sim = float(np.max(similarities))
    
    # Mean of top-k similarities
    k = min(top_k, len(similarities))
    topk_sims = np.partition(similarities, -k)[-k:]  # Get top k
    mean_topk = float(np.mean(topk_sims))
    
    # Count hard candidates
    n_hard = int(np.sum(similarities > hard_threshold))
    
    return {
        'max_similarity': max_sim,
        'mean_topk_similarity': mean_topk,
        'n_hard': n_hard,
        'pool_size': valid_candidates,
        'all_similarities': similarities.tolist()
    }

def load_candidate_map(candidates_path: str):
    """Load candidate map from JSON or PKL file."""
    if candidates_path.lower().endswith(".pkl"):
        import pickle
        with open(candidates_path, "rb") as f:
            return pickle.load(f)
    else:
        with open(candidates_path, "r") as f:
            return json.load(f)

def analyze_candidate_hardness(candidates_path: str, dataset_name: str, 
                                sample_size: int = None, seed: int = 1234,
                                hard_threshold: float = 0.8, top_k: int = 10):
    """
    Analyze candidate hardness by computing comprehensive similarity metrics.
    
    Args:
        candidates_path: Path to candidate map file (JSON or PKL)
        dataset_name: Name of dataset (for labeling)
        sample_size: Optional sample size to limit computation (for speed)
        seed: Random seed for sampling
        hard_threshold: Tanimoto threshold for "hard" candidates (default 0.8)
        top_k: Number of top candidates to average (default 10)
    
    Returns:
        Dictionary with statistics and lists of all metrics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing candidate hardness for: {dataset_name}")
    print(f"Loading candidates from: {candidates_path}")
    print(f"{'='*60}")
    
    # Load candidate map
    cand_map = load_candidate_map(candidates_path)
    
    # Convert to list of (gt_smiles, candidate_list) tuples
    items = list(cand_map.items())
    
    # Sample if requested
    if sample_size is not None and sample_size < len(items):
        np.random.seed(seed)
        indices = np.random.choice(len(items), size=sample_size, replace=False)
        items = [items[i] for i in indices]
        print(f"Sampling {sample_size} queries from {len(cand_map)} total queries")
    
    print(f"Computing similarity metrics for {len(items)} queries...")
    print(f"  Hard threshold: Tanimoto > {hard_threshold}")
    print(f"  Top-K for mean: {top_k}")
    
    max_similarities = []
    mean_topk_similarities = []
    n_hard_list = []
    pool_sizes = []
    failed_queries = 0
    skipped_queries = 0
    
    for gt_smiles, candidate_list in tqdm(items, desc="Computing similarities"):
        # Handle different candidate list formats
        if isinstance(candidate_list, (list, tuple)):
            candidates = list(candidate_list)
        elif isinstance(candidate_list, dict):
            # If it's a dict, extract the list (could be {'candidates': [...]})
            candidates = candidate_list.get('candidates', list(candidate_list.values()))
        else:
            skipped_queries += 1
            continue
        
        if not candidates:
            skipped_queries += 1
            continue
        
        metrics = compute_similarity_metrics(gt_smiles, candidates, 
                                             hard_threshold=hard_threshold, 
                                             top_k=top_k)
        
        if np.isnan(metrics['max_similarity']):
            failed_queries += 1
        else:
            max_similarities.append(metrics['max_similarity'])
            mean_topk_similarities.append(metrics['mean_topk_similarity'])
            n_hard_list.append(metrics['n_hard'])
            pool_sizes.append(metrics['pool_size'])
    
    print(f"\nCompleted analysis:")
    print(f"  Successful queries: {len(max_similarities)}")
    print(f"  Failed queries: {failed_queries}")
    print(f"  Skipped queries: {skipped_queries}")
    
    if len(max_similarities) == 0:
        print("ERROR: No valid similarities computed!")
        return None
    
    max_similarities = np.array(max_similarities)
    mean_topk_similarities = np.array(mean_topk_similarities)
    n_hard_list = np.array(n_hard_list)
    pool_sizes = np.array(pool_sizes)
    
    # Compute statistics for max similarity
    def compute_stats(arr, name):
        return {
            f'{name}_mean': float(np.mean(arr)),
            f'{name}_median': float(np.median(arr)),
            f'{name}_std': float(np.std(arr)),
            f'{name}_min': float(np.min(arr)),
            f'{name}_max': float(np.max(arr)),
            f'{name}_q25': float(np.percentile(arr, 25)),
            f'{name}_q75': float(np.percentile(arr, 75)),
        }
    
    stats = {
        'dataset': dataset_name,
        'n_queries': len(max_similarities),
        'hard_threshold': hard_threshold,
        'top_k': top_k,
        **compute_stats(max_similarities, 'max_sim'),
        **compute_stats(mean_topk_similarities, 'mean_topk_sim'),
        **compute_stats(n_hard_list, 'n_hard'),
        **compute_stats(pool_sizes, 'pool_size'),
        'max_similarities': max_similarities.tolist(),
        'mean_topk_similarities': mean_topk_similarities.tolist(),
        'n_hard_list': n_hard_list.tolist(),
        'pool_sizes': pool_sizes.tolist()
    }
    
    print(f"\nStatistics:")
    print(f"  Max Similarity:")
    print(f"    Mean: {stats['max_sim_mean']:.4f}, Median: {stats['max_sim_median']:.4f}")
    print(f"  Mean Top-{top_k} Similarity:")
    print(f"    Mean: {stats['mean_topk_sim_mean']:.4f}, Median: {stats['mean_topk_sim_median']:.4f}")
    print(f"  Number of Hard Candidates (Tanimoto > {hard_threshold}):")
    print(f"    Mean: {stats['n_hard_mean']:.2f}, Median: {stats['n_hard_median']:.2f}")
    print(f"  Pool Size:")
    print(f"    Mean: {stats['pool_size_mean']:.2f}, Median: {stats['pool_size_median']:.2f}")
    
    return stats

def main():
    ap = argparse.ArgumentParser(
        description="Evaluate candidate hardness via Tanimoto similarity analysis"
    )
    ap.add_argument(
        "--msgym-candidates",
        type=str,
        required=True,
        help="Path to MassSpecGym candidate map (JSON or PKL)"
    )
    ap.add_argument(
        "--spectraverse-candidates",
        type=str,
        required=True,
        help="Path to Spectraverse candidate map (JSON or PKL)"
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="figs",
        help="Output directory for results and figures"
    )
    ap.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional: sample this many queries per dataset (for faster computation)"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for sampling"
    )
    ap.add_argument(
        "--fp-radius",
        type=int,
        default=2,
        help="Morgan fingerprint radius (default 2 for ECFP4)"
    )
    ap.add_argument(
        "--fp-nbits",
        type=int,
        default=4096,
        help="Number of bits in fingerprint (default 4096)"
    )
    ap.add_argument(
        "--hard-threshold",
        type=float,
        default=0.8,
        help="Tanimoto threshold for 'hard' candidates (default 0.8)"
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top candidates to average for mean top-k similarity (default 10)"
    )
    args = ap.parse_args()
    
    if not _HAS_RDKIT:
        print("ERROR: RDKit is required for this analysis. Please install RDKit.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze both datasets
    msgym_stats = analyze_candidate_hardness(
        args.msgym_candidates,
        "MassSpecGym",
        sample_size=args.sample_size,
        seed=args.seed,
        hard_threshold=args.hard_threshold,
        top_k=args.top_k
    )
    
    spectraverse_stats = analyze_candidate_hardness(
        args.spectraverse_candidates,
        "Spectraverse",
        sample_size=args.sample_size,
        seed=args.seed,
        hard_threshold=args.hard_threshold,
        top_k=args.top_k
    )
    
    if msgym_stats is None or spectraverse_stats is None:
        print("ERROR: Failed to compute statistics for one or both datasets.")
        return
    
    # Save statistics to JSON (exclude list fields)
    exclude_fields = ['max_similarities', 'mean_topk_similarities', 'n_hard_list', 'pool_sizes']
    stats_output = {
        'massspecgym': {k: v for k, v in msgym_stats.items() if k not in exclude_fields},
        'spectraverse': {k: v for k, v in spectraverse_stats.items() if k not in exclude_fields}
    }
    
    stats_json_path = os.path.join(args.output_dir, "candidate_hardness_stats.json")
    with open(stats_json_path, "w") as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nStatistics saved to: {stats_json_path}")
    
    # Create comparison DataFrame with all metrics
    n_msgym = len(msgym_stats['max_similarities'])
    n_spec = len(spectraverse_stats['max_similarities'])
    
    df = pd.DataFrame({
        'Dataset': ['MassSpecGym'] * n_msgym + ['Spectraverse'] * n_spec,
        'Max_Tanimoto_Similarity': msgym_stats['max_similarities'] + spectraverse_stats['max_similarities'],
        'Mean_TopK_Tanimoto_Similarity': msgym_stats['mean_topk_similarities'] + spectraverse_stats['mean_topk_similarities'],
        'N_Hard_Candidates': msgym_stats['n_hard_list'] + spectraverse_stats['n_hard_list'],
        'Pool_Size': msgym_stats['pool_sizes'] + spectraverse_stats['pool_sizes']
    })
    
    csv_path = os.path.join(args.output_dir, "candidate_hardness_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")
    
    # Create visualizations
    set_paper_style()
    
    msgym_sims = np.array(msgym_stats['max_similarities'])
    spectraverse_sims = np.array(spectraverse_stats['max_similarities'])
    msgym_topk = np.array(msgym_stats['mean_topk_similarities'])
    spectraverse_topk = np.array(spectraverse_stats['mean_topk_similarities'])
    msgym_nhard = np.array(msgym_stats['n_hard_list'])
    spectraverse_nhard = np.array(spectraverse_stats['n_hard_list'])
    
    # Figure 1: Histogram comparison - Max Similarity
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1
    
    ax.hist(msgym_sims, bins=bins, alpha=0.6, label='MassSpecGym', 
            color=palette[2], edgecolor='black', linewidth=0.5, density=True)
    ax.hist(spectraverse_sims, bins=bins, alpha=0.6, label='Spectraverse', 
            color=palette[1], edgecolor='black', linewidth=0.5, density=True)
    
    ax.set_xlabel('Maximum Tanimoto Similarity (ECFP4)', fontsize=16)
    ax.set_ylabel('Density', fontsize=16)
    ax.set_title('Distribution of Maximum Tanimoto Similarity\n(Ground Truth vs. Candidates)', 
                 fontsize=18, pad=10)
    ax.legend(frameon=True, fontsize=14, loc='best')
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # Add text box with statistics
    textstr = (
        f"MassSpecGym:\n"
        f"  Mean: {msgym_stats['max_sim_mean']:.3f}\n"
        f"  Median: {msgym_stats['max_sim_median']:.3f}\n\n"
        f"Spectraverse:\n"
        f"  Mean: {spectraverse_stats['max_sim_mean']:.3f}\n"
        f"  Median: {spectraverse_stats['max_sim_median']:.3f}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    hist_path = os.path.join(args.output_dir, "candidate_hardness_histogram.pdf")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to: {hist_path}")
    plt.close()
    
    # Figure 2: Box plot comparison - Max Similarity
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    box_data = [msgym_sims, spectraverse_sims]
    box_labels = ['MassSpecGym', 'Spectraverse']
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], [palette[0], palette[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Maximum Tanimoto Similarity (ECFP4)', fontsize=16)
    ax.set_title('Candidate Hardness Comparison\n(Max Tanimoto Similarity)', 
                 fontsize=18, pad=10)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    plt.tight_layout()
    
    boxplot_path = os.path.join(args.output_dir, "candidate_hardness_boxplot.pdf")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {boxplot_path}")
    plt.close()
    
    # Figure 3: Mean Top-K Similarity comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    box_data = [msgym_topk, spectraverse_topk]
    box_labels = ['MassSpecGym', 'Spectraverse']
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], [palette[0], palette[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel(f'Mean Top-{args.top_k} Tanimoto Similarity (ECFP4)', fontsize=16)
    ax.set_title('Hard Neighborhood Density\n(Mean Top-K Similarity)', 
                 fontsize=18, pad=10)
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    plt.tight_layout()
    
    topk_path = os.path.join(args.output_dir, "candidate_hardness_topk_boxplot.pdf")
    plt.savefig(topk_path, dpi=300, bbox_inches='tight')
    print(f"Top-K box plot saved to: {topk_path}")
    plt.close()
    
    # Figure 4: Number of Hard Candidates
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    
    # Use log scale for y-axis since numbers can vary widely
    box_data = [msgym_nhard, spectraverse_nhard]
    box_labels = ['MassSpecGym', 'Spectraverse']
    
    bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], [palette[0], palette[1]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel(f'Number of Hard Candidates\n(Tanimoto > {args.hard_threshold})', fontsize=16)
    ax.set_title('Effective Hard Candidate Count', 
                 fontsize=18, pad=10)
    ax.set_yscale('log')  # Log scale for better visualization
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    plt.tight_layout()
    
    nhard_path = os.path.join(args.output_dir, "candidate_hardness_nhard_boxplot.pdf")
    plt.savefig(nhard_path, dpi=300, bbox_inches='tight')
    print(f"Hard candidates count plot saved to: {nhard_path}")
    plt.close()
    
    # Print summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"Max Tanimoto Similarity:")
    print(f"  MassSpecGym:   Mean={msgym_stats['max_sim_mean']:.4f}, Median={msgym_stats['max_sim_median']:.4f}")
    print(f"  Spectraverse:  Mean={spectraverse_stats['max_sim_mean']:.4f}, Median={spectraverse_stats['max_sim_median']:.4f}")
    print(f"  Difference:    {spectraverse_stats['max_sim_mean'] - msgym_stats['max_sim_mean']:.4f}")
    print(f"\nMean Top-{args.top_k} Tanimoto Similarity:")
    print(f"  MassSpecGym:   Mean={msgym_stats['mean_topk_sim_mean']:.4f}, Median={msgym_stats['mean_topk_sim_median']:.4f}")
    print(f"  Spectraverse:  Mean={spectraverse_stats['mean_topk_sim_mean']:.4f}, Median={spectraverse_stats['mean_topk_sim_median']:.4f}")
    print(f"  Difference:    {spectraverse_stats['mean_topk_sim_mean'] - msgym_stats['mean_topk_sim_mean']:.4f}")
    print(f"\nNumber of Hard Candidates (Tanimoto > {args.hard_threshold}):")
    print(f"  MassSpecGym:   Mean={msgym_stats['n_hard_mean']:.2f}, Median={msgym_stats['n_hard_median']:.2f}")
    print(f"  Spectraverse:  Mean={spectraverse_stats['n_hard_mean']:.2f}, Median={spectraverse_stats['n_hard_median']:.2f}")
    print(f"  Ratio:         {spectraverse_stats['n_hard_mean'] / max(msgym_stats['n_hard_mean'], 0.01):.2f}x")
    print(f"\nPool Size:")
    print(f"  MassSpecGym:   Mean={msgym_stats['pool_size_mean']:.2f}, Median={msgym_stats['pool_size_median']:.2f}")
    print(f"  Spectraverse:  Mean={spectraverse_stats['pool_size_mean']:.2f}, Median={spectraverse_stats['pool_size_median']:.2f}")
    print(f"\nNOTE: Larger pool sizes are statistically more likely to contain")
    print(f"high-similarity decoys by chance. The higher similarity in Spectraverse")
    print(f"reflects both pool size effects AND intrinsic hardness (isomerism).")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
