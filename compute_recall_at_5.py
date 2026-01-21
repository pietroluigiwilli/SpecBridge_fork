#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute Recall@5 from existing per-query results.
"""

import pandas as pd
import numpy as np

# Load the results
print("Loading results from figs/fixed_size_128_per_query_results.csv...")
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")

print(f"Loaded {len(df)} queries")

# Compute correct@5 (rank <= 4, since rank is 0-indexed)
df['correct_at_5'] = df['rank'] <= 4

# Exclude queries with max_similarity >= 1.0 (should have been deduplicated)
df_valid = df[df['max_similarity'] < 1.0].copy()
invalid_count = len(df) - len(df_valid)
if invalid_count > 0:
    print(f"\nExcluding {invalid_count} queries with max_similarity >= 1.0 (should have been deduplicated)")

# Bin queries by max similarity
bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
bin_labels = ['<0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.99']

bin_stats = {label: {'correct_at_1': 0, 'correct_at_5': 0, 'total': 0, 'ranks': []} for label in bin_labels}

for idx, row in df_valid.iterrows():
    max_sim = row['max_similarity']
    bin_idx = np.digitize([max_sim], bins)[0] - 1
    if bin_idx < 0:
        bin_idx = 0
    elif bin_idx >= len(bin_labels):
        bin_idx = len(bin_labels) - 1
    label = bin_labels[bin_idx]
    
    bin_stats[label]['total'] += 1
    if row['correct']:  # Recall@1
        bin_stats[label]['correct_at_1'] += 1
    if row['correct_at_5']:  # Recall@5
        bin_stats[label]['correct_at_5'] += 1
    bin_stats[label]['ranks'].append(row['rank'])

# Compute statistics per bin
results = []
for label in bin_labels:
    stats = bin_stats[label]
    if stats['total'] > 0:
        recall_at_1 = stats['correct_at_1'] / stats['total']
        recall_at_5 = stats['correct_at_5'] / stats['total']
        median_rank = np.median(stats['ranks']) if stats['ranks'] else np.nan
        results.append({
            'max_sim_bin': label,
            'n_queries': stats['total'],
            'recall_at_1': recall_at_1,
            'recall_at_5': recall_at_5,
            'median_rank': median_rank
        })

# Print results
print(f"\n{'='*80}")
print(f"RECALL@1 AND RECALL@5 FOR FIXED POOL SIZE (N=128) STRATIFIED BY MAX SIMILARITY")
print(f"{'='*80}")
print(f"{'MaxSim Bin':<12s} {'N':>8s} {'R@1':>8s} {'R@5':>8s} {'Med Rank':>10s}")
print(f"{'-'*80}")
for row in results:
    print(f"{row['max_sim_bin']:<12s} {row['n_queries']:>8d} {row['recall_at_1']:>8.4f} {row['recall_at_5']:>8.4f} {row['median_rank']:>10.1f}")
print(f"{'='*80}\n")

# Save results
output_df = pd.DataFrame(results)
output_path = "figs/fixed_size_128_hardness_stratified_r5.csv"
output_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Also print just Recall@5 for easy copy-paste
print(f"\n{'='*80}")
print(f"RECALL@5 FOR FIXED POOL SIZE (N=128) STRATIFIED BY MAX SIMILARITY")
print(f"{'='*80}")
for row in results:
    print(f"MaxSim {row['max_sim_bin']:12s}: {row['recall_at_5']:.4f} (n={row['n_queries']:5d}, median_rank={row['median_rank']:.1f})")
print(f"{'='*80}\n")
