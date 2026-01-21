#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation of the fixed-size hardness results.
"""

import pandas as pd
import numpy as np

print("="*80)
print("COMPREHENSIVE EVALUATION OF RESULTS")
print("="*80)

# Load results
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")
print(f"\nTotal queries: {len(df)}")

# Exclude queries with max_similarity >= 1.0
df_valid = df[df['max_similarity'] < 1.0].copy()
print(f"Valid queries (max_sim < 1.0): {len(df_valid)}")
print(f"Excluded queries (max_sim >= 1.0): {len(df) - len(df_valid)}")

# Compute metrics
df_valid['correct_at_5'] = df_valid['rank'] <= 4
df_valid['correct_at_10'] = df_valid['rank'] <= 9

print(f"\n{'='*80}")
print("OVERALL PERFORMANCE")
print(f"{'='*80}")
print(f"Recall@1: {df_valid['correct'].mean():.4f}")
print(f"Recall@5: {df_valid['correct_at_5'].mean():.4f}")
print(f"Recall@10: {df_valid['correct_at_10'].mean():.4f}")
print(f"Median rank: {df_valid['rank'].median():.1f}")
print(f"Mean rank: {df_valid['rank'].mean():.1f}")

# Analyze by similarity bins
print(f"\n{'='*80}")
print("PERFORMANCE BY MAX SIMILARITY BIN")
print(f"{'='*80}")

bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
bin_labels = ['<0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.99']

print(f"\n{'Bin':<12} {'N':>8} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'Med Rank':>10} {'Mean Rank':>10}")
print("-" * 80)

for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
    bin_df = df_valid[(df_valid['max_similarity'] >= low) & (df_valid['max_similarity'] < high)]
    if len(bin_df) > 0:
        r1 = bin_df['correct'].mean()
        r5 = bin_df['correct_at_5'].mean()
        r10 = bin_df['correct_at_10'].mean()
        med_rank = bin_df['rank'].median()
        mean_rank = bin_df['rank'].mean()
        print(f"{label:<12} {len(bin_df):>8} {r1:>8.4f} {r5:>8.4f} {r10:>8.4f} {med_rank:>10.1f} {mean_rank:>10.1f}")

# Deep dive into <0.4 bin
print(f"\n{'='*80}")
print("DEEP DIVE: <0.4 SIMILARITY BIN")
print(f"{'='*80}")

low_sim = df_valid[df_valid['max_similarity'] < 0.4].copy()
print(f"\nTotal queries: {len(low_sim)}")
print(f"Recall@1: {low_sim['correct'].mean():.4f}")
print(f"Recall@5: {low_sim['correct_at_5'].mean():.4f}")
print(f"Recall@10: {low_sim['correct_at_10'].mean():.4f}")
print(f"Median rank: {low_sim['rank'].median():.1f}")
print(f"Mean rank: {low_sim['rank'].mean():.1f}")

# Rank distribution
print(f"\nRank distribution:")
rank_dist = low_sim['rank'].value_counts().sort_index()
for rank, count in rank_dist.head(15).items():
    pct = 100 * count / len(low_sim)
    print(f"  Rank {int(rank):3d}: {count:5d} ({pct:5.2f}%)")

# Analyze poor vs good performers
print(f"\n{'='*80}")
print("POOR vs GOOD PERFORMERS IN <0.4 BIN")
print(f"{'='*80}")

poor = low_sim[low_sim['rank'] > 4]
good = low_sim[low_sim['rank'] <= 4]

print(f"\nPoor performers (rank > 4): {len(poor)} ({100*len(poor)/len(low_sim):.1f}%)")
print(f"  Mean rank: {poor['rank'].mean():.1f}")
print(f"  Median rank: {poor['rank'].median():.1f}")
print(f"  Mean max_sim: {poor['max_similarity'].mean():.4f}")
print(f"  Mean pool size: {poor['pool_size_capped'].mean():.1f}")

print(f"\nGood performers (rank <= 4): {len(good)} ({100*len(good)/len(low_sim):.1f}%)")
print(f"  Mean rank: {good['rank'].mean():.1f}")
print(f"  Median rank: {good['rank'].median():.1f}")
print(f"  Mean max_sim: {good['max_similarity'].mean():.4f}")
print(f"  Mean pool size: {good['pool_size_capped'].mean():.1f}")

# Check if there's a pattern with pool size
print(f"\n{'='*80}")
print("POOL SIZE ANALYSIS")
print(f"{'='*80}")

# Check queries with very small pools
small_pools = low_sim[low_sim['pool_size_capped'] < 100]
large_pools = low_sim[low_sim['pool_size_capped'] >= 100]

if len(small_pools) > 0:
    print(f"\nSmall pools (< 100): {len(small_pools)} queries")
    print(f"  Recall@5: {small_pools['correct_at_5'].mean():.4f}")
    print(f"  Mean rank: {small_pools['rank'].mean():.1f}")

if len(large_pools) > 0:
    print(f"\nLarge pools (>= 100): {len(large_pools)} queries")
    print(f"  Recall@5: {large_pools['correct_at_5'].mean():.4f}")
    print(f"  Mean rank: {large_pools['rank'].mean():.1f}")

# Check for queries with rank = 0 (perfect)
perfect = low_sim[low_sim['rank'] == 0]
print(f"\nPerfect predictions (rank = 0): {len(perfect)} ({100*len(perfect)/len(low_sim):.1f}%)")

# Check for queries with very high ranks
very_high_rank = low_sim[low_sim['rank'] > 50]
print(f"Very poor predictions (rank > 50): {len(very_high_rank)} ({100*len(very_high_rank)/len(low_sim):.1f}%)")
if len(very_high_rank) > 0:
    print(f"  Mean rank: {very_high_rank['rank'].mean():.1f}")
    print(f"  Max rank: {very_high_rank['rank'].max():.0f}")

print(f"\n{'='*80}")
print("CONCLUSIONS")
print(f"{'='*80}")
print(f"1. Overall Recall@5: {df_valid['correct_at_5'].mean():.4f}")
print(f"2. For <0.4 similarity (easy cases): Recall@5 = {low_sim['correct_at_5'].mean():.4f}")
print(f"3. {100*len(poor)/len(low_sim):.1f}% of <0.4 queries have rank > 4")
print(f"4. This suggests the model struggles even when candidates are very different")
print(f"5. Possible reasons:")
print(f"   - Model limitations in embedding space")
print(f"   - Embedding quality issues")
print(f"   - Query spectrum quality issues")
print(f"   - Or genuine difficulty in some cases")
print("="*80)
