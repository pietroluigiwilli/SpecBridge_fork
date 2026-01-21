#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Investigate why queries with max_similarity < 0.4 only get 56% R@5
when the ground truth should be easily distinguishable.
"""

import pandas as pd
import numpy as np
import json
import pickle
import os

# Load results
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")
df_valid = df[df['max_similarity'] < 1.0].copy()
df_valid['correct_at_5'] = df_valid['rank'] <= 4

# Focus on <0.4 bin with poor performance
low_sim = df_valid[df_valid['max_similarity'] < 0.4].copy()
low_sim_poor = low_sim[low_sim['rank'] > 4].copy()  # Queries with rank > 4

print("="*80)
print("INVESTIGATION: Why do <0.4 similarity queries only get 56% R@5?")
print("="*80)
print(f"\nTotal queries with max_similarity < 0.4: {len(low_sim)}")
print(f"Queries with rank > 4 (poor performance): {len(low_sim_poor)} ({100*len(low_sim_poor)/len(low_sim):.1f}%)")
print(f"Recall@5: {low_sim['correct_at_5'].mean():.4f}")
print(f"Recall@1: {low_sim['correct'].mean():.4f}")

# Check rank distribution for poor performers
print(f"\n{'='*80}")
print("Rank Distribution for Poor Performers (rank > 4):")
print(f"{'='*80}")
print(low_sim_poor['rank'].value_counts().sort_index().head(20))

# Load candidate map to verify ground truth presence
candidate_file = "/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
if os.path.exists(candidate_file):
    print(f"\n{'='*80}")
    print("Checking if ground truth is in candidate sets...")
    print(f"{'='*80}")
    
    with open(candidate_file, 'rb') as f:
        cand_map_full = pickle.load(f)
    
    # Sample some poor performers
    sample_poor = low_sim_poor.head(10)
    missing_gt = 0
    
    for idx, row in sample_poor.iterrows():
        gt_smiles = row['gt_smiles']
        cand_list = cand_map_full.get(gt_smiles, [])
        
        if not cand_list:
            print(f"\nQuery {row['query_idx']}: NO CANDIDATES FOUND!")
            print(f"  GT: {gt_smiles[:60]}...")
            missing_gt += 1
        elif gt_smiles not in cand_list:
            # Check with stripped version
            gt_stripped = gt_smiles.strip()
            if gt_stripped not in cand_list:
                print(f"\nQuery {row['query_idx']}: Ground truth NOT in candidate list!")
                print(f"  GT: {gt_smiles[:60]}...")
                print(f"  Candidate list size: {len(cand_list)}")
                print(f"  Rank: {row['rank']}, Max sim: {row['max_similarity']:.4f}")
                missing_gt += 1
            else:
                print(f"\nQuery {row['query_idx']}: Ground truth found (stripped)")
        else:
            print(f"\nQuery {row['query_idx']}: Ground truth IS in candidate list")
            print(f"  Rank: {row['rank']}, Max sim: {row['max_similarity']:.4f}")
            print(f"  Candidate list size: {len(cand_list)}")
    
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"Sampled {len(sample_poor)} poor performers")
    print(f"Ground truth missing/not found: {missing_gt}")

# Analyze by pool size
print(f"\n{'='*80}")
print("Analysis by Pool Size:")
print(f"{'='*80}")
print("Poor performers (rank > 4):")
print(f"  Mean pool size: {low_sim_poor['pool_size_capped'].mean():.1f}")
print(f"  Median pool size: {low_sim_poor['pool_size_capped'].median():.1f}")
print("\nGood performers (rank <= 4):")
good_performers = low_sim[low_sim['rank'] <= 4]
print(f"  Mean pool size: {good_performers['pool_size_capped'].mean():.1f}")
print(f"  Median pool size: {good_performers['pool_size_capped'].median():.1f}")

# Check if there's a correlation
print(f"\n{'='*80}")
print("Correlation Analysis:")
print(f"{'='*80}")
corr_rank_pool = low_sim['rank'].corr(low_sim['pool_size_capped'])
print(f"Correlation between rank and pool size: {corr_rank_pool:.4f}")

# Check max_similarity distribution
print(f"\n{'='*80}")
print("Max Similarity Distribution:")
print(f"{'='*80}")
print(f"Poor performers:")
print(f"  Mean max_sim: {low_sim_poor['max_similarity'].mean():.4f}")
print(f"  Median max_sim: {low_sim_poor['max_similarity'].median():.4f}")
print(f"\nGood performers:")
print(f"  Mean max_sim: {good_performers['max_similarity'].mean():.4f}")
print(f"  Median max_sim: {good_performers['max_similarity'].median():.4f}")

print(f"\n{'='*80}")
print("HYPOTHESIS:")
print(f"{'='*80}")
print("If all candidates have similarity < 0.4, the ground truth (sim=1.0)")
print("should be easily distinguishable. The fact that we only get 56% R@5")
print("suggests:")
print("1. The model may not be working well for some queries")
print("2. There might be embedding issues")
print("3. The ground truth might not always be in the candidate set")
print("4. There could be canonicalization/matching issues")
print("="*80)
