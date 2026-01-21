#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute top-5 similarities for each query and re-analyze Recall@5 results.
This is more appropriate for Recall@5 evaluation.
"""

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except ImportError:
    print("ERROR: RDKit is required for this analysis. Please install RDKit.")
    _HAS_RDKIT = False
    exit(1)

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

def compute_top5_similarities(gt_smiles: str, candidate_smiles_list: list, fp_cache: dict = None) -> dict:
    """
    Compute top-5 Tanimoto similarities between ground truth and all candidates.
    Returns dict with:
    - max_sim: maximum similarity (1st highest)
    - top5_mean: mean of top-5 similarities
    - top5_median: median of top-5 similarities
    - top5_min: minimum of top-5 (5th highest)
    - top5_sims: list of top-5 similarities
    """
    # Use cache if provided
    if fp_cache is not None:
        if gt_smiles not in fp_cache:
            fp_cache[gt_smiles] = ecfp4(gt_smiles)
        gt_fp = fp_cache[gt_smiles]
    else:
        gt_fp = ecfp4(gt_smiles)
    
    if gt_fp is None:
        return {
            'max_sim': np.nan,
            'top5_mean': np.nan,
            'top5_median': np.nan,
            'top5_min': np.nan,
            'top5_sims': []
        }
    
    # Get ground truth fingerprint bits for efficient comparison
    gt_on_bits = set(gt_fp.GetOnBits()) if gt_fp else None
    
    similarities = []
    
    for cand_smiles in candidate_smiles_list:
        if not cand_smiles or cand_smiles == gt_smiles:
            continue
        
        # Use cache if provided
        if fp_cache is not None:
            if cand_smiles not in fp_cache:
                fp_cache[cand_smiles] = ecfp4(cand_smiles)
            cand_fp = fp_cache[cand_smiles]
        else:
            cand_fp = ecfp4(cand_smiles)
        
        if cand_fp is None:
            continue
        
        # Fast check: if fingerprints have identical on bits, they're the same molecule
        if gt_on_bits is not None:
            cand_on_bits = set(cand_fp.GetOnBits())
            if cand_on_bits == gt_on_bits:
                # Identical fingerprints, skip
                continue
        
        sim = tanimoto_similarity(gt_fp, cand_fp)
        if not np.isnan(sim) and sim < 1.0:  # Exclude exact matches
            similarities.append(sim)
    
    if len(similarities) == 0:
        return {
            'max_sim': np.nan,
            'top5_mean': np.nan,
            'top5_median': np.nan,
            'top5_min': np.nan,
            'top5_sims': []
        }
    
    # Get top-5 similarities
    similarities.sort(reverse=True)
    top5 = similarities[:5]
    
    return {
        'max_sim': top5[0] if top5 else np.nan,
        'top5_mean': np.mean(top5) if top5 else np.nan,
        'top5_median': np.median(top5) if top5 else np.nan,
        'top5_min': top5[-1] if len(top5) >= 5 else (top5[-1] if top5 else np.nan),
        'top5_sims': top5
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

def deduplicate_by_fingerprint(smiles_list: list, threshold: float = 1.0, prefer_smiles: str = None) -> list:
    """
    Deduplicate a list of SMILES strings based on fingerprint similarity.
    For threshold=1.0, uses optimized fingerprint-based grouping.
    """
    if not smiles_list:
        return []
    
    if threshold >= 1.0:
        # Compute fingerprints and group by identical bit vectors
        fp_to_smiles = {}
        
        for smi in smiles_list:
            if not smi:
                continue
            fp = ecfp4(smi)
            if fp is not None:
                on_bits = tuple(fp.GetOnBits())
                if on_bits not in fp_to_smiles:
                    fp_to_smiles[on_bits] = []
                fp_to_smiles[on_bits].append(smi)
        
        # For each fingerprint group, keep only one SMILES (prefer prefer_smiles if provided)
        kept = []
        for on_bits, smi_group in fp_to_smiles.items():
            if prefer_smiles and prefer_smiles in smi_group:
                kept.append(prefer_smiles)
            else:
                kept.append(smi_group[0])
        
        return kept
    
    # For threshold < 1.0, would need pairwise comparison (not implemented here)
    return smiles_list

def cap_candidate_map(cand_map_raw, pool_cap, seed=1234):
    """Cap candidate pools and deduplicate (same logic as eval_fixed_size_hardness.py)."""
    rng = np.random.default_rng(seed)
    cand_map = {}
    
    for k, vs in tqdm(cand_map_raw.items(), desc="Capping candidate pools"):
        ck = k.strip() if k else k
        if not isinstance(vs, list):
            try:
                vs = list(vs)
            except:
                vs = []
        
        # String-based deduplication
        vals = []
        seen = set()
        for v in vs:
            if not v:
                continue
            cv = v.strip()
            if cv not in seen:
                vals.append(cv)
                seen.add(cv)
        
        # Ensure ground truth is in the list
        if ck and ck not in seen:
            vals.append(ck)
        
        # Deduplicate by fingerprint
        vals = deduplicate_by_fingerprint(vals, threshold=1.0, prefer_smiles=ck)
        
        # Ensure ground truth is in the list after deduplication
        if ck and ck not in vals:
            gt_fp = ecfp4(ck)
            if gt_fp:
                gt_on_bits = tuple(gt_fp.GetOnBits())
                found_duplicate = False
                for i, cand in enumerate(vals):
                    cand_fp = ecfp4(cand)
                    if cand_fp:
                        cand_on_bits = tuple(cand_fp.GetOnBits())
                        if cand_on_bits == gt_on_bits:
                            vals[i] = ck
                            found_duplicate = True
                            break
                if not found_duplicate:
                    vals.append(ck)
            else:
                vals.append(ck)
        
        # Cap the pool
        if pool_cap is not None and len(vals) > pool_cap:
            if ck and ck in vals:
                vals_without_gt = [v for v in vals if v != ck]
                n_to_sample = pool_cap - 1
                if len(vals_without_gt) > n_to_sample:
                    sampled = rng.choice(vals_without_gt, size=n_to_sample, replace=False).tolist()
                    vals = sampled + [ck]
                else:
                    vals = vals_without_gt + [ck]
            else:
                vals = rng.choice(vals, size=pool_cap, replace=False).tolist()
        
        if vals:
            cand_map[ck] = vals
    
    return cand_map

# Main analysis
print("="*80)
print("Computing Top-5 Similarities for Recall@5 Analysis")
print("="*80)

# Load per-query results
print("\nLoading per-query results...")
df = pd.read_csv("figs/fixed_size_128_per_query_results.csv")
print(f"Loaded {len(df)} queries")

# Exclude queries with max_similarity >= 1.0
df_valid = df[df['max_similarity'] < 1.0].copy()
print(f"Valid queries (after excluding max_sim >= 1.0): {len(df_valid)}")

# Load candidate map
candidate_file = "/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
if not os.path.exists(candidate_file):
    # Try alternative locations
    candidate_files = [
        "data/candidates_test_val.pkl",
        "data/candidates.json",
        "data/candidates.pkl",
    ]
    candidate_file = None
    for cf in candidate_files:
        if os.path.exists(cf):
            candidate_file = cf
            break

if candidate_file is None or not os.path.exists(candidate_file):
    print("\nWARNING: Could not find candidate map file.")
    print("Please update the candidate_file path in the script.")
    exit(1)

print(f"\nLoading candidate map from {candidate_file}...")
cand_map_full = load_candidate_map(candidate_file)
print(f"Loaded {len(cand_map_full)} entries")

# Cap the candidate map (same as in evaluation, pool_cap=128, seed=1234)
print("\nCapping candidate pools to size 128 (matching evaluation)...")
cand_map = cap_candidate_map(cand_map_full, pool_cap=128, seed=1234)
print(f"Capped to {len(cand_map)} entries")

# Compute top-5 similarities for each query
print("\nComputing top-5 similarities for each query...")
print("This may take a while - computing fingerprints for all candidates...")

# Check for checkpoint
checkpoint_file = "figs/top5_similarity_checkpoint.pkl"
import pickle

top5_stats = []
processed_indices = set()

if os.path.exists(checkpoint_file):
    print(f"Loading checkpoint from {checkpoint_file}...")
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            top5_stats = checkpoint_data.get('top5_stats', [])
            processed_indices = set(checkpoint_data.get('processed_indices', []))
            print(f"Resuming from checkpoint: {len(processed_indices)}/{len(df_valid)} queries processed")
    except:
        print("Checkpoint file corrupted, starting fresh")
        top5_stats = []
        processed_indices = set()

# Create a fingerprint cache to avoid recomputing
fp_cache = {}

def get_fp_cached(smiles):
    """Get fingerprint with caching."""
    if smiles in fp_cache:
        return fp_cache[smiles]
    fp = ecfp4(smiles)
    fp_cache[smiles] = fp
    return fp

# Process queries with progress bar
for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="Computing top-5 sims"):
    query_idx = row['query_idx']
    
    # Skip if already processed
    if query_idx in processed_indices:
        continue
    
    gt_smiles = row['gt_smiles']
    cand_list = cand_map.get(gt_smiles, [])
    
    if not cand_list:
        top5_stats.append({
            'query_idx': query_idx,
            'max_sim': row['max_similarity'],
            'top5_mean': np.nan,
            'top5_median': np.nan,
            'top5_min': np.nan,
        })
        processed_indices.add(query_idx)
        continue
    
    # Use cached fingerprints
    stats = compute_top5_similarities(gt_smiles, cand_list, fp_cache=fp_cache)
    top5_stats.append({
        'query_idx': query_idx,
        'max_sim': stats['max_sim'],
        'top5_mean': stats['top5_mean'],
        'top5_median': stats['top5_median'],
        'top5_min': stats['top5_min'],
    })
    processed_indices.add(query_idx)
    
    # Save checkpoint every 1000 queries
    if len(processed_indices) % 1000 == 0:
        checkpoint_data = {
            'top5_stats': top5_stats,
            'processed_indices': list(processed_indices)
        }
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"\nCheckpoint saved: {len(processed_indices)}/{len(df_valid)} queries processed")

# Final checkpoint save
if top5_stats:
    checkpoint_data = {
        'top5_stats': top5_stats,
        'processed_indices': list(processed_indices)
    }
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"\nFinal checkpoint saved: {len(processed_indices)}/{len(df_valid)} queries processed")

# Add to dataframe
top5_df = pd.DataFrame(top5_stats)
df_valid = df_valid.merge(top5_df, on='query_idx', how='left', suffixes=('', '_computed'))

# Compute correct@5
df_valid['correct_at_5'] = df_valid['rank'] <= 4

print("\n" + "="*80)
print("Analysis Results")
print("="*80)

# Function to bin and compute stats
def analyze_by_metric(df, metric_col, metric_name):
    """Bin by a similarity metric and compute Recall@5."""
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    bin_labels = ['<0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.99']
    
    # Filter out NaN values
    df_clean = df[df[metric_col].notna()].copy()
    
    bin_stats = {label: {'correct_at_5': 0, 'total': 0, 'ranks': []} for label in bin_labels}
    
    for idx, row in df_clean.iterrows():
        metric_val = row[metric_col]
        bin_idx = np.digitize([metric_val], bins)[0] - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= len(bin_labels):
            bin_idx = len(bin_labels) - 1
        label = bin_labels[bin_idx]
        
        bin_stats[label]['total'] += 1
        if row['correct_at_5']:
            bin_stats[label]['correct_at_5'] += 1
        bin_stats[label]['ranks'].append(row['rank'])
    
    results = []
    for label in bin_labels:
        stats = bin_stats[label]
        if stats['total'] > 0:
            recall_at_5 = stats['correct_at_5'] / stats['total']
            median_rank = np.median(stats['ranks']) if stats['ranks'] else np.nan
            results.append({
                'max_sim_bin': label,
                'n_queries': stats['total'],
                'recall_at_5': recall_at_5,
                'median_rank': median_rank
            })
    
    return results

# Analyze by different metrics
print("\n1. By Maximum Similarity (current method):")
results_max = analyze_by_metric(df_valid, 'max_similarity', 'max_sim')
for r in results_max:
    print(f"  {r['max_sim_bin']:12s}: R@5={r['recall_at_5']:.4f} (n={r['n_queries']:5d})")

print("\n2. By Top-5 Mean Similarity:")
results_top5_mean = analyze_by_metric(df_valid, 'top5_mean', 'top5_mean')
for r in results_top5_mean:
    print(f"  {r['max_sim_bin']:12s}: R@5={r['recall_at_5']:.4f} (n={r['n_queries']:5d})")

print("\n3. By Top-5 Median Similarity:")
results_top5_median = analyze_by_metric(df_valid, 'top5_median', 'top5_median')
for r in results_top5_median:
    print(f"  {r['max_sim_bin']:12s}: R@5={r['recall_at_5']:.4f} (n={r['n_queries']:5d})")

print("\n4. By Top-5 Minimum (5th highest) Similarity:")
results_top5_min = analyze_by_metric(df_valid, 'top5_min', 'top5_min')
for r in results_top5_min:
    print(f"  {r['max_sim_bin']:12s}: R@5={r['recall_at_5']:.4f} (n={r['n_queries']:5d})")

# Save results
print("\n" + "="*80)
print("Saving results...")

# Save by top5_mean (seems most informative)
output_df = pd.DataFrame(results_top5_mean)
output_path = "figs/fixed_size_128_hardness_stratified_r5_top5mean.csv"
output_df.to_csv(output_path, index=False)
print(f"Saved top-5 mean results to: {output_path}")

# Also save the enhanced per-query results
enhanced_path = "figs/fixed_size_128_per_query_results_top5.csv"
df_valid.to_csv(enhanced_path, index=False)
print(f"Saved enhanced per-query results to: {enhanced_path}")

print("\n" + "="*80)
print("RECOMMENDATION:")
print("="*80)
print("For Recall@5, using Top-5 Mean Similarity provides a better measure")
print("of task difficulty than just the maximum similarity, as it considers")
print("all top competitors that could affect the rank-5 result.")
print("="*80)
