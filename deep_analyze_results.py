#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep analysis of results to understand counterintuitive patterns.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("DEEP ANALYSIS OF RESULTS")
print("="*80)

# Load results
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")
df_valid = df[df['max_similarity'] < 1.0].copy()

# Compute metrics
df_valid['correct_at_1'] = df_valid['rank'] == 0
df_valid['correct_at_5'] = df_valid['rank'] <= 4
df_valid['correct_at_20'] = df_valid['rank'] <= 19
df_valid['reciprocal_rank'] = 1.0 / (df_valid['rank'] + 1)

# Bin queries
bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
bin_labels = ['<0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.99']

print("\n" + "="*80)
print("ISSUE 1: Counterintuitive Performance Patterns")
print("="*80)
print("\nObserved anomalies:")
print("- 0.7-0.8 bin has R@5 = 0.4791 > 0.6-0.7 bin R@5 = 0.4007")
print("- 0.8-0.9 bin has R@20 = 0.6844 > 0.6-0.7 bin R@20 = 0.5939")
print("- 0.5-0.6 bin has highest R@20 = 0.7379")
print("\nThese don't follow expected trend: higher similarity → lower performance")

# Analyze each bin in detail
print("\n" + "="*80)
print("DETAILED BIN ANALYSIS")
print("="*80)

for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
    bin_df = df_valid[(df_valid['max_similarity'] >= low) & (df_valid['max_similarity'] < high)]
    if len(bin_df) > 0:
        print(f"\n{label} (max_sim: {low:.1f}-{high:.1f}):")
        print(f"  N queries: {len(bin_df)}")
        print(f"  R@1: {bin_df['correct_at_1'].mean():.4f}")
        print(f"  R@5: {bin_df['correct_at_5'].mean():.4f}")
        print(f"  R@20: {bin_df['correct_at_20'].mean():.4f}")
        print(f"  MRR: {bin_df['reciprocal_rank'].mean():.4f}")
        print(f"  Median rank: {bin_df['rank'].median():.1f}")
        print(f"  Mean rank: {bin_df['rank'].mean():.1f}")
        print(f"  Max similarity stats:")
        print(f"    Mean: {bin_df['max_similarity'].mean():.4f}")
        print(f"    Median: {bin_df['max_similarity'].median():.4f}")
        print(f"    Min: {bin_df['max_similarity'].min():.4f}")
        print(f"    Max: {bin_df['max_similarity'].max():.4f}")
        print(f"  Pool size stats:")
        print(f"    Mean: {bin_df['pool_size_capped'].mean():.1f}")
        print(f"    Median: {bin_df['pool_size_capped'].median():.1f}")
        print(f"    Min: {bin_df['pool_size_capped'].min():.0f}")
        print(f"    Max: {bin_df['pool_size_capped'].max():.0f}")
        
        # Rank distribution
        rank_dist = bin_df['rank'].value_counts().sort_index()
        print(f"  Rank distribution (top 10):")
        for rank, count in rank_dist.head(10).items():
            pct = 100 * count / len(bin_df)
            print(f"    Rank {int(rank):3d}: {count:5d} ({pct:5.2f}%)")

# Check for correlation between max_similarity and performance within bins
print("\n" + "="*80)
print("CORRELATION ANALYSIS WITHIN BINS")
print("="*80)

for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
    bin_df = df_valid[(df_valid['max_similarity'] >= low) & (df_valid['max_similarity'] < high)]
    if len(bin_df) > 0:
        corr_maxsim_rank = bin_df['max_similarity'].corr(bin_df['rank'])
        corr_maxsim_r5 = bin_df['max_similarity'].corr(bin_df['correct_at_5'].astype(int))
        print(f"\n{label}:")
        print(f"  Correlation (max_sim, rank): {corr_maxsim_rank:.4f}")
        print(f"  Correlation (max_sim, R@5): {corr_maxsim_r5:.4f}")

# Analyze pool size effect
print("\n" + "="*80)
print("POOL SIZE EFFECT ANALYSIS")
print("="*80)

# Check if pool size explains the anomalies
for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
    bin_df = df_valid[(df_valid['max_similarity'] >= low) & (df_valid['max_similarity'] < high)]
    if len(bin_df) > 0:
        small_pools = bin_df[bin_df['pool_size_capped'] < 100]
        large_pools = bin_df[bin_df['pool_size_capped'] >= 100]
        
        print(f"\n{label}:")
        if len(small_pools) > 0:
            print(f"  Small pools (<100): n={len(small_pools)}, R@5={small_pools['correct_at_5'].mean():.4f}, mean_rank={small_pools['rank'].mean():.1f}")
        if len(large_pools) > 0:
            print(f"  Large pools (>=100): n={len(large_pools)}, R@5={large_pools['correct_at_5'].mean():.4f}, mean_rank={large_pools['rank'].mean():.1f}")

# Check if there are outliers or data quality issues
print("\n" + "="*80)
print("DATA QUALITY CHECKS")
print("="*80)

# Check for queries with rank = 0 but max_similarity > 0.9
high_sim_perfect = df_valid[(df_valid['max_similarity'] >= 0.9) & (df_valid['rank'] == 0)]
print(f"\nQueries with max_sim >= 0.9 and rank = 0 (perfect): {len(high_sim_perfect)}")
if len(high_sim_perfect) > 0:
    print("  This is suspicious - very high similarity but perfect rank")
    print(f"  Mean max_sim: {high_sim_perfect['max_similarity'].mean():.4f}")
    print(f"  Sample max_sim values: {high_sim_perfect['max_similarity'].head(5).tolist()}")

# Check for queries with very low max_similarity but high rank
low_sim_poor = df_valid[(df_valid['max_similarity'] < 0.4) & (df_valid['rank'] > 50)]
print(f"\nQueries with max_sim < 0.4 and rank > 50 (poor): {len(low_sim_poor)}")
if len(low_sim_poor) > 0:
    print(f"  Mean max_sim: {low_sim_poor['max_similarity'].mean():.4f}")
    print(f"  Mean rank: {low_sim_poor['rank'].mean():.1f}")

# Analyze the specific problematic bins
print("\n" + "="*80)
print("ANALYZING PROBLEMATIC BINS")
print("="*80)

# 0.7-0.8 vs 0.6-0.7
bin_67 = df_valid[(df_valid['max_similarity'] >= 0.6) & (df_valid['max_similarity'] < 0.7)]
bin_78 = df_valid[(df_valid['max_similarity'] >= 0.7) & (df_valid['max_similarity'] < 0.8)]

print(f"\n0.6-0.7 bin (R@5 = {bin_67['correct_at_5'].mean():.4f}):")
print(f"  Mean max_sim: {bin_67['max_similarity'].mean():.4f}")
print(f"  Mean pool size: {bin_67['pool_size_capped'].mean():.1f}")
print(f"  Rank distribution: median={bin_67['rank'].median():.1f}, mean={bin_67['rank'].mean():.1f}")

print(f"\n0.7-0.8 bin (R@5 = {bin_78['correct_at_5'].mean():.4f}):")
print(f"  Mean max_sim: {bin_78['max_similarity'].mean():.4f}")
print(f"  Mean pool size: {bin_78['pool_size_capped'].mean():.1f}")
print(f"  Rank distribution: median={bin_78['rank'].median():.1f}, mean={bin_78['rank'].mean():.1f}")

# Check if there's a bimodal distribution
print("\n" + "="*80)
print("BIMODAL DISTRIBUTION CHECK")
print("="*80)

for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
    bin_df = df_valid[(df_valid['max_similarity'] >= low) & (df_valid['max_similarity'] < high)]
    if len(bin_df) > 0:
        perfect = bin_df[bin_df['rank'] == 0]
        poor = bin_df[bin_df['rank'] > 20]
        print(f"\n{label}:")
        print(f"  Perfect (rank=0): {len(perfect)} ({100*len(perfect)/len(bin_df):.1f}%)")
        print(f"  Poor (rank>20): {len(poor)} ({100*len(poor)/len(bin_df):.1f}%)")
        if len(perfect) > 0 and len(poor) > 0:
            print(f"  Perfect mean max_sim: {perfect['max_similarity'].mean():.4f}")
            print(f"  Poor mean max_sim: {poor['max_similarity'].mean():.4f}")

# Hypothesis: Maybe max_similarity is not the right metric
print("\n" + "="*80)
print("HYPOTHESIS: Is max_similarity the right metric?")
print("="*80)
print("\nMax similarity only considers the SINGLE hardest decoy.")
print("For Recall@5 and Recall@20, we should consider top-5 or top-20 similarities.")
print("A query might have:")
print("  - One very similar decoy (high max_sim) but others are easy")
print("  - Many moderately similar decoys (moderate max_sim) but all are hard")
print("\nThis could explain why 0.7-0.8 performs better than 0.6-0.7:")
print("  - 0.6-0.7 might have many moderately similar decoys")
print("  - 0.7-0.8 might have one very similar decoy but others are easier")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("1. Use top-5 mean similarity instead of max similarity for Recall@5 analysis")
print("2. Use top-20 mean similarity instead of max similarity for Recall@20 analysis")
print("3. Check if the top-5 similarity computation is complete")
print("4. Investigate why some bins have counterintuitive performance")
print("="*80)
