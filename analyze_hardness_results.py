#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze the hardness stratification results to understand why 0.9-0.99 bin has high accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")

print("="*80)
print("ANALYSIS: Why does 0.9-0.99 similarity bin have high accuracy?")
print("="*80)

# Filter to 0.9-0.99 bin
high_sim_df = df[(df['max_similarity'] >= 0.9) & (df['max_similarity'] < 0.99)].copy()

print(f"\nTotal queries in 0.9-0.99 bin: {len(high_sim_df)}")
print(f"Recall@1: {high_sim_df['correct'].mean():.4f}")
print(f"Median rank: {high_sim_df['rank'].median():.1f}")

# Analyze similarity distribution
print(f"\n{'='*80}")
print("Max Similarity Distribution in 0.9-0.99 Bin:")
print(f"{'='*80}")
print(f"  Min: {high_sim_df['max_similarity'].min():.4f}")
print(f"  Max: {high_sim_df['max_similarity'].max():.4f}")
print(f"  Mean: {high_sim_df['max_similarity'].mean():.4f}")
print(f"  Median: {high_sim_df['max_similarity'].median():.4f}")
print(f"  95th percentile: {high_sim_df['max_similarity'].quantile(0.95):.4f}")
print(f"  99th percentile: {high_sim_df['max_similarity'].quantile(0.99):.4f}")

# Check how many have very high similarity (>= 0.99)
very_high = high_sim_df[high_sim_df['max_similarity'] >= 0.99]
print(f"\nQueries with max_similarity >= 0.99: {len(very_high)} ({100*len(very_high)/len(high_sim_df):.1f}%)")
if len(very_high) > 0:
    print(f"  Recall@1 for >=0.99: {very_high['correct'].mean():.4f}")
    print(f"  Median rank for >=0.99: {very_high['rank'].median():.1f}")

# Check queries with max_similarity = 1.0 (shouldn't happen after deduplication)
exact_1 = df[df['max_similarity'] >= 1.0]
print(f"\n{'='*80}")
print(f"WARNING: Queries with max_similarity >= 1.0: {len(exact_1)}")
if len(exact_1) > 0:
    print(f"  These should have been deduplicated!")
    print(f"  Recall@1: {exact_1['correct'].mean():.4f}")
    print(f"  Median rank: {exact_1['rank'].median():.1f}")
    print(f"\n  Sample queries with max_sim >= 1.0:")
    for idx, row in exact_1.head(5).iterrows():
        print(f"    Query {row['query_idx']}: max_sim={row['max_similarity']:.4f}, rank={row['rank']}, correct={row['correct']}")

# Analyze by similarity sub-bins
print(f"\n{'='*80}")
print("Breakdown by Similarity Sub-bins (0.9-0.99):")
print(f"{'='*80}")
sub_bins = [
    (0.9, 0.92, "0.90-0.92"),
    (0.92, 0.94, "0.92-0.94"),
    (0.94, 0.96, "0.94-0.96"),
    (0.96, 0.98, "0.96-0.98"),
    (0.98, 0.99, "0.98-0.99"),
]

for low, high, label in sub_bins:
    sub_df = high_sim_df[(high_sim_df['max_similarity'] >= low) & (high_sim_df['max_similarity'] < high)]
    if len(sub_df) > 0:
        print(f"{label:12s}: n={len(sub_df):5d}, recall={sub_df['correct'].mean():.4f}, median_rank={sub_df['rank'].median():.1f}")

# Compare with other bins
print(f"\n{'='*80}")
print("Comparison with Other Bins:")
print(f"{'='*80}")
bins = [
    (0.0, 0.4, "<0.4"),
    (0.4, 0.5, "0.4-0.5"),
    (0.5, 0.6, "0.5-0.6"),
    (0.6, 0.7, "0.6-0.7"),
    (0.7, 0.8, "0.7-0.8"),
    (0.8, 0.9, "0.8-0.9"),
    (0.9, 0.99, "0.9-0.99"),
]

for low, high, label in bins:
    bin_df = df[(df['max_similarity'] >= low) & (df['max_similarity'] < high)]
    if len(bin_df) > 0:
        print(f"{label:12s}: n={len(bin_df):5d}, recall={bin_df['correct'].mean():.4f}, median_rank={bin_df['rank'].median():.1f}")

# Analyze rank distribution for 0.9-0.99 bin
print(f"\n{'='*80}")
print("Rank Distribution for 0.9-0.99 Bin:")
print(f"{'='*80}")
rank_counts = high_sim_df['rank'].value_counts().sort_index()
print("Rank distribution:")
for rank, count in rank_counts.head(10).items():
    pct = 100 * count / len(high_sim_df)
    print(f"  Rank {int(rank):3d}: {count:5d} queries ({pct:5.2f}%)")

# Check if there's a correlation between max_similarity and accuracy within 0.9-0.99
print(f"\n{'='*80}")
print("Correlation Analysis:")
print(f"{'='*80}")
correlation = high_sim_df['max_similarity'].corr(high_sim_df['correct'].astype(int))
print(f"Correlation between max_similarity and correctness (0.9-0.99 bin): {correlation:.4f}")

# Check pool sizes
print(f"\n{'='*80}")
print("Pool Size Analysis for 0.9-0.99 Bin:")
print(f"{'='*80}")
print(f"  Mean capped pool size: {high_sim_df['pool_size_capped'].mean():.1f}")
print(f"  Mean full pool size: {high_sim_df['pool_size_full'].mean():.1f}")
print(f"  Median capped pool size: {high_sim_df['pool_size_capped'].median():.1f}")

# Hypothesis: Maybe queries with very high similarity (0.99+) are actually easier
# because the "hardest decoy" is so similar it might be the same molecule
print(f"\n{'='*80}")
print("HYPOTHESIS TESTING:")
print(f"{'='*80}")
print("\nHypothesis 1: Very high similarity (>=0.99) might indicate duplicate molecules")
very_high_sim = high_sim_df[high_sim_df['max_similarity'] >= 0.99]
lower_high_sim = high_sim_df[(high_sim_df['max_similarity'] >= 0.9) & (high_sim_df['max_similarity'] < 0.99)]
if len(very_high_sim) > 0 and len(lower_high_sim) > 0:
    print(f"  >=0.99: recall={very_high_sim['correct'].mean():.4f}, n={len(very_high_sim)}")
    print(f"  0.9-0.99: recall={lower_high_sim['correct'].mean():.4f}, n={len(lower_high_sim)}")

print("\nHypothesis 2: Maybe the model is just very good at distinguishing near-duplicates")
print(f"  But this contradicts the expectation that higher similarity = harder task")

print("\nHypothesis 3: Maybe queries with high similarity have fewer total candidates")
corr_pool_correct = high_sim_df['pool_size_capped'].corr(high_sim_df['correct'].astype(int))
print(f"  Correlation between pool size and correctness: {corr_pool_correct:.4f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("If max_similarity >= 0.99 queries have very high accuracy, they might be")
print("cases where the 'hardest decoy' is actually the same molecule (different SMILES).")
print("These should have been deduplicated but weren't caught by fingerprint sim=1.0 check.")
print("="*80)
