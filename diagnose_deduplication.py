#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose why queries have max_similarity >= 1.0 after deduplication.
"""

import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def ecfp4(smiles: str, radius: int = 2, nbits: int = 4096):
    """Compute ECFP4 fingerprint for a SMILES string."""
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

def tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    if fp1 is None or fp2 is None:
        return None
    try:
        return float(DataStructs.TanimotoSimilarity(fp1, fp2))
    except Exception:
        return None

# Load results
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")

# Find queries with max_similarity >= 1.0
invalid = df[df['max_similarity'] >= 1.0]

print(f"Found {len(invalid)} queries with max_similarity >= 1.0")
print(f"\nSample queries:")
for idx, row in invalid.head(10).iterrows():
    print(f"\nQuery {row['query_idx']}:")
    print(f"  GT SMILES: {row['gt_smiles']}")
    print(f"  Max similarity: {row['max_similarity']:.4f}")
    print(f"  Rank: {row['rank']}, Correct: {row['correct']}")
    print(f"  Pool size (capped): {row['pool_size_capped']}")

# Try to load candidate map to check deduplication
print("\n" + "="*80)
print("Checking if we can verify deduplication...")
print("="*80)

# Check a few sample queries manually
sample_queries = invalid.head(3)
for idx, row in sample_queries.iterrows():
    gt_smiles = row['gt_smiles']
    gt_fp = ecfp4(gt_smiles)
    if gt_fp:
        print(f"\nQuery {row['query_idx']} (GT: {gt_smiles[:50]}...):")
        print(f"  GT fingerprint on bits: {len(gt_fp.GetOnBits())} bits set")
        print(f"  This query has max_similarity = {row['max_similarity']:.4f}")
        print(f"  This suggests there's a candidate with identical fingerprint that wasn't deduplicated")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("Queries with max_similarity >= 1.0 indicate that:")
print("1. There are candidates with identical fingerprints (sim=1.0) to the ground truth")
print("2. These should have been deduplicated but weren't")
print("3. Possible causes:")
print("   - Deduplication is not being applied correctly")
print("   - Candidates are being added after deduplication")
print("   - Ground truth is being compared to itself")
print("="*80)
