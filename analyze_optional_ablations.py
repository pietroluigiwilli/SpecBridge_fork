#!/usr/bin/env python3
"""
Analysis script for optional ablations.
Generates summary tables and plots.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

def load_results(base_dir, fold="val"):
    """Load all evaluation results."""
    base_path = Path(base_dir)
    master_csv = base_path / f"optional_ablations_summary_{fold}.csv"
    
    if not master_csv.exists():
        print(f"Master CSV not found: {master_csv}")
        return None
    
    df = pd.read_csv(master_csv)
    # Filter out NA rows
    df = df[df['R@5'] != 'NA']
    df['R@5'] = pd.to_numeric(df['R@5'], errors='coerce')
    df['R@1'] = pd.to_numeric(df['R@1'], errors='coerce')
    df['R@20'] = pd.to_numeric(df['R@20'], errors='coerce')
    df['MRR'] = pd.to_numeric(df['MRR'], errors='coerce')
    
    return df

def analyze_mapper_capacity(df):
    """Analyze mapper capacity ablation results."""
    mapper_df = df[df['config'].str.startswith('mapper_')].copy()
    
    if mapper_df.empty:
        print("No mapper capacity results found.")
        return None
    
    # Parse n_blocks and mapper_hidden from config name
    mapper_df['n_blocks'] = mapper_df['config'].str.extract(r'mapper_n(\d+)_h(\d+)')[0].astype(int)
    mapper_df['mapper_hidden'] = mapper_df['config'].str.extract(r'mapper_n(\d+)_h(\d+)')[1].astype(int)
    
    # Get best result per configuration
    best_per_config = mapper_df.loc[mapper_df.groupby('config')['R@5'].idxmax()]
    
    # Create pivot table
    pivot = best_per_config.pivot_table(
        values='R@5',
        index='n_blocks',
        columns='mapper_hidden',
        aggfunc='mean'
    )
    
    return pivot, best_per_config

def analyze_unfreezing_depth(df):
    """Analyze spectrum unfreezing depth results."""
    unfreeze_df = df[df['config'].str.startswith('unfreeze_')].copy()
    
    if unfreeze_df.empty:
        print("No unfreezing depth results found.")
        return None
    
    # Parse unfreeze depth
    unfreeze_df['unfreeze_depth'] = unfreeze_df['config'].str.extract(r'unfreeze_(\d+)')[0].astype(int)
    
    # Get best result per configuration
    best_per_config = unfreeze_df.loc[unfreeze_df.groupby('config')['R@5'].idxmax()]
    
    return best_per_config.sort_values('unfreeze_depth')

def plot_mapper_capacity(pivot, output_dir):
    """Plot mapper capacity heatmap."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'R@5'})
    plt.title('Mapper Capacity Ablation: R@5 by n_blocks and mapper_hidden')
    plt.xlabel('Mapper Hidden Width')
    plt.ylabel('Number of Residual Blocks')
    plt.tight_layout()
    plt.savefig(output_dir / 'mapper_capacity_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved mapper capacity plot to {output_dir / 'mapper_capacity_ablation.pdf'}")

def plot_unfreezing_depth(unfreeze_df, output_dir):
    """Plot unfreezing depth results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R@5 vs unfreeze depth
    axes[0].plot(unfreeze_df['unfreeze_depth'], unfreeze_df['R@5'], 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Unfreeze Depth (number of layers)')
    axes[0].set_ylabel('R@5')
    axes[0].set_title('R@5 vs Spectrum Unfreezing Depth')
    axes[0].grid(True, alpha=0.3)
    
    # MRR vs unfreeze depth
    axes[1].plot(unfreeze_df['unfreeze_depth'], unfreeze_df['MRR'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Unfreeze Depth (number of layers)')
    axes[1].set_ylabel('MRR')
    axes[1].set_title('MRR vs Spectrum Unfreezing Depth')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'unfreezing_depth_ablation.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved unfreezing depth plot to {output_dir / 'unfreezing_depth_ablation.pdf'}")

def main():
    parser = argparse.ArgumentParser(description='Analyze optional ablation results')
    parser.add_argument('--base-dir', type=str, default='runs/optional_ablations_spectraverse',
                      help='Base directory with ablation results')
    parser.add_argument('--fold', type=str, default='val', help='Fold to analyze (val/test)')
    parser.add_argument('--output-dir', type=str, default='figs',
                      help='Output directory for plots')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    df = load_results(args.base_dir, args.fold)
    if df is None or df.empty:
        print("No results found.")
        return
    
    print(f"Loaded {len(df)} evaluation results")
    print()
    
    # Analyze mapper capacity
    print("=" * 60)
    print("Mapper Capacity Analysis")
    print("=" * 60)
    mapper_result = analyze_mapper_capacity(df)
    if mapper_result:
        pivot, best = mapper_result
        print("\nBest R@5 per configuration:")
        print(pivot)
        print()
        plot_mapper_capacity(pivot, output_dir)
    
    # Analyze unfreezing depth
    print("\n" + "=" * 60)
    print("Spectrum Unfreezing Depth Analysis")
    print("=" * 60)
    unfreeze_result = analyze_unfreezing_depth(df)
    if unfreeze_result is not None:
        print("\nBest results by unfreezing depth:")
        print(unfreeze_result[['unfreeze_depth', 'R@1', 'R@5', 'R@20', 'MRR']])
        print()
        plot_unfreezing_depth(unfreeze_result, output_dir)
    
    # Procrustes warm start (if available)
    print("\n" + "=" * 60)
    print("Procrustes Warm Start Analysis")
    print("=" * 60)
    procrustes_df = df[df['config'].str.startswith('procrustes_')]
    if not procrustes_df.empty:
        best_procrustes = procrustes_df.loc[procrustes_df['R@5'].idxmax()]
        print("\nBest Procrustes warm start result:")
        print(best_procrustes[['config', 'R@1', 'R@5', 'R@20', 'MRR']])
    else:
        print("No Procrustes warm start results found.")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()


