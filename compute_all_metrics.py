#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute comprehensive metrics: R@1, R@5, R@20, MRR, and Median Rank
for fixed-size hardness analysis.
"""

import pandas as pd
import numpy as np

print("="*80)
print("Computing Comprehensive Metrics")
print("="*80)

# Load results
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")
print(f"\nTotal queries: {len(df)}")

# Exclude queries with max_similarity >= 1.0
df_valid = df[df['max_similarity'] < 1.0].copy()
print(f"Valid queries (max_sim < 1.0): {len(df_valid)}")
print(f"Excluded queries (max_sim >= 1.0): {len(df) - len(df_valid)}")

# Compute all metrics
df_valid['correct_at_1'] = df_valid['rank'] == 0
df_valid['correct_at_5'] = df_valid['rank'] <= 4
df_valid['correct_at_20'] = df_valid['rank'] <= 19
df_valid['reciprocal_rank'] = 1.0 / (df_valid['rank'] + 1)  # MRR: 1/(rank+1)

# Bin queries by max similarity
bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
bin_labels = ['<0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.99']

bin_stats = {label: {
    'correct_at_1': 0,
    'correct_at_5': 0,
    'correct_at_20': 0,
    'reciprocal_ranks': [],
    'ranks': [],
    'total': 0
} for label in bin_labels}

for idx, row in df_valid.iterrows():
    max_sim = row['max_similarity']
    bin_idx = np.digitize([max_sim], bins)[0] - 1
    if bin_idx < 0:
        bin_idx = 0
    elif bin_idx >= len(bin_labels):
        bin_idx = len(bin_labels) - 1
    label = bin_labels[bin_idx]
    
    bin_stats[label]['total'] += 1
    if row['correct_at_1']:
        bin_stats[label]['correct_at_1'] += 1
    if row['correct_at_5']:
        bin_stats[label]['correct_at_5'] += 1
    if row['correct_at_20']:
        bin_stats[label]['correct_at_20'] += 1
    bin_stats[label]['reciprocal_ranks'].append(row['reciprocal_rank'])
    bin_stats[label]['ranks'].append(row['rank'])

# Compute statistics per bin
results = []
for label in bin_labels:
    stats = bin_stats[label]
    if stats['total'] > 0:
        recall_at_1 = stats['correct_at_1'] / stats['total']
        recall_at_5 = stats['correct_at_5'] / stats['total']
        recall_at_20 = stats['correct_at_20'] / stats['total']
        mrr = np.mean(stats['reciprocal_ranks']) if stats['reciprocal_ranks'] else np.nan
        median_rank = np.median(stats['ranks']) if stats['ranks'] else np.nan
        results.append({
            'max_sim_bin': label,
            'n_queries': stats['total'],
            'recall_at_1': recall_at_1,
            'recall_at_5': recall_at_5,
            'recall_at_20': recall_at_20,
            'mrr': mrr,
            'median_rank': median_rank
        })

# Compute overall metrics
overall_r1 = df_valid['correct_at_1'].mean()
overall_r5 = df_valid['correct_at_5'].mean()
overall_r20 = df_valid['correct_at_20'].mean()
overall_mrr = df_valid['reciprocal_rank'].mean()
overall_median_rank = df_valid['rank'].median()

# Print results
print(f"\n{'='*80}")
print("COMPREHENSIVE METRICS BY MAX SIMILARITY BIN")
print(f"{'='*80}")
print(f"{'Bin':<12} {'N':>8} {'R@1':>8} {'R@5':>8} {'R@20':>8} {'MRR':>8} {'Med Rank':>10}")
print("-" * 80)

for row in results:
    print(f"{row['max_sim_bin']:<12} {row['n_queries']:>8} "
          f"{row['recall_at_1']:>8.4f} {row['recall_at_5']:>8.4f} {row['recall_at_20']:>8.4f} "
          f"{row['mrr']:>8.4f} {row['median_rank']:>10.1f}")

print("-" * 80)
print(f"{'OVERALL':<12} {len(df_valid):>8} "
      f"{overall_r1:>8.4f} {overall_r5:>8.4f} {overall_r20:>8.4f} "
      f"{overall_mrr:>8.4f} {overall_median_rank:>10.1f}")
print(f"{'='*80}\n")

# Save results
output_df = pd.DataFrame(results)
output_path = "figs/fixed_size_128_hardness_stratified_all_metrics.csv"
output_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Also save overall metrics
overall_results = pd.DataFrame([{
    'metric': 'overall',
    'n_queries': len(df_valid),
    'recall_at_1': overall_r1,
    'recall_at_5': overall_r5,
    'recall_at_20': overall_r20,
    'mrr': overall_mrr,
    'median_rank': overall_median_rank
}])
overall_path = "figs/fixed_size_128_hardness_overall_metrics.csv"
overall_results.to_csv(overall_path, index=False)
print(f"Overall metrics saved to: {overall_path}")

# Print formatted summary for easy copy-paste
print(f"\n{'='*80}")
print("FORMATTED SUMMARY (for easy copy-paste)")
print(f"{'='*80}")
print("\nBy Bin:")
for row in results:
    print(f"MaxSim {row['max_sim_bin']:12s}: R@1={row['recall_at_1']:.4f}, R@5={row['recall_at_5']:.4f}, "
          f"R@20={row['recall_at_20']:.4f}, MRR={row['mrr']:.4f}, MedRank={row['median_rank']:.1f} "
          f"(n={row['n_queries']})")

print(f"\nOverall: R@1={overall_r1:.4f}, R@5={overall_r5:.4f}, R@20={overall_r20:.4f}, "
      f"MRR={overall_mrr:.4f}, MedRank={overall_median_rank:.1f} (n={len(df_valid)})")
print(f"{'='*80}\n")
