#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if ground truth is in candidate sets and investigate matching issues.
"""

import pandas as pd
import pickle
import os

# Load results
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")
df_valid = df[df['max_similarity'] < 1.0].copy()

# Focus on <0.4 bin
low_sim = df_valid[df_valid['max_similarity'] < 0.4].copy()

# Load candidate map
candidate_file = "/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
with open(candidate_file, 'rb') as f:
    cand_map_full = pickle.load(f)

print("="*80)
print("Checking Ground Truth Presence in Candidate Sets")
print("="*80)

missing_count = 0
found_count = 0
found_stripped = 0
total_checked = 0

for idx, row in low_sim.iterrows():
    gt_smiles = row['gt_smiles']
    total_checked += 1
    
    # Check if GT is in candidate map
    if gt_smiles in cand_map_full:
        cand_list = cand_map_full[gt_smiles]
        if gt_smiles in cand_list:
            found_count += 1
        elif gt_smiles.strip() in cand_list:
            found_stripped += 1
        else:
            missing_count += 1
            if missing_count <= 5:  # Show first 5 examples
                print(f"\nMissing example {missing_count}:")
                print(f"  GT: {gt_smiles[:80]}")
                print(f"  In map as key: Yes")
                print(f"  In candidate list: No")
                print(f"  Candidate list size: {len(cand_list)}")
                print(f"  Rank: {row['rank']}")
    else:
        # Check with stripped version
        gt_stripped = gt_smiles.strip()
        if gt_stripped in cand_map_full:
            cand_list = cand_map_full[gt_stripped]
            if gt_stripped in cand_list or gt_smiles in cand_list:
                found_stripped += 1
            else:
                missing_count += 1
                if missing_count <= 5:
                    print(f"\nMissing example {missing_count}:")
                    print(f"  GT: {gt_smiles[:80]}")
                    print(f"  GT stripped: {gt_stripped[:80]}")
                    print(f"  In map as key (stripped): Yes")
                    print(f"  In candidate list: No")
                    print(f"  Candidate list size: {len(cand_list)}")
                    print(f"  Rank: {row['rank']}")
        else:
            missing_count += 1
            if missing_count <= 5:
                print(f"\nMissing example {missing_count}:")
                print(f"  GT: {gt_smiles[:80]}")
                print(f"  NOT in candidate map at all!")
                print(f"  Rank: {row['rank']}")

print(f"\n{'='*80}")
print("SUMMARY:")
print(f"{'='*80}")
print(f"Total queries checked: {total_checked}")
print(f"Ground truth found (exact match): {found_count} ({100*found_count/total_checked:.1f}%)")
print(f"Ground truth found (stripped match): {found_stripped} ({100*found_stripped/total_checked:.1f}%)")
print(f"Ground truth MISSING: {missing_count} ({100*missing_count/total_checked:.1f}%)")
print(f"\nThis explains why R@5 is only 56% - {100*missing_count/total_checked:.1f}% of queries")
print(f"don't have the ground truth in their candidate sets!")
