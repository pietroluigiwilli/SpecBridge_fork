#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Disentangle Pool Size from Chemical Hardness.

This script analyzes retrieval performance at fixed pool size (N=128) stratified
by the maximum Tanimoto similarity of the hardest decoy. This proves that
chemical hardness (isomerism), not pool size, is the primary driver of difficulty.

The key insight: Even with a small pool size matching MassSpecGym, performance
drops dramatically when the pool contains high-similarity isomers.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
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

# Paper style
def set_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linestyle": ":", "grid.alpha": 0.30,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

palette = sns.color_palette("Set2", n_colors=8)

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

def deduplicate_by_fingerprint(smiles_list: list, threshold: float = 1.0, prefer_smiles: str = None) -> list:
    """
    Deduplicate a list of SMILES strings based on fingerprint similarity.
    If two molecules have fingerprint similarity >= threshold (default 1.0), 
    they are considered the same molecule and only one is kept.
    
    Args:
        smiles_list: List of SMILES strings to deduplicate
        threshold: Similarity threshold for considering molecules identical (default 1.0)
        prefer_smiles: If provided, prefer keeping this SMILES when duplicates are found
    
    Returns:
        Deduplicated list preserving order (preferred SMILES kept if provided).
    """
    if not smiles_list:
        return []
    
    # For threshold=1.0, we can optimize by using fingerprint bit vectors as keys
    # If two fingerprints have Tanimoto similarity = 1.0, they are identical bit vectors
    if threshold >= 1.0:
        # Compute fingerprints and group by identical bit vectors
        fp_to_smiles = {}  # fingerprint key -> list of SMILES with that fingerprint
        
        for smi in smiles_list:
            if not smi:
                continue
            fp = ecfp4(smi)
            if fp is not None:
                # Convert bit vector to a hashable key using the on bits
                # Get the list of on bits and convert to tuple for hashing
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
                kept.append(smi_group[0])  # Keep first occurrence
        
        return kept
    
    # For threshold < 1.0, use the slower pairwise comparison method
    # Compute fingerprints for all molecules
    fps = {}
    valid_smiles = []
    for smi in smiles_list:
        if not smi:
            continue
        fp = ecfp4(smi)
        if fp is not None:
            fps[smi] = fp
            valid_smiles.append(smi)
    
    if not valid_smiles:
        return []
    
    # If prefer_smiles is provided and in the list, process it first
    if prefer_smiles and prefer_smiles in valid_smiles:
        # Move preferred SMILES to front
        valid_smiles = [prefer_smiles] + [s for s in valid_smiles if s != prefer_smiles]
    
    # Group molecules by fingerprint similarity
    # For each molecule, check if it's identical (sim=1) to any already kept molecule
    kept = []
    kept_fps = []
    
    for smi in valid_smiles:
        fp = fps[smi]
        is_duplicate = False
        
        # Check against all kept molecules
        for kept_fp in kept_fps:
            sim = tanimoto_similarity(fp, kept_fp)
            if not np.isnan(sim) and sim >= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            kept.append(smi)
            kept_fps.append(fp)
    
    return kept

def compute_max_similarity(gt_smiles: str, candidate_smiles_list: list) -> float:
    """
    Compute maximum Tanimoto similarity between ground truth and all candidates.
    Excludes the ground truth itself if present, and any candidates with similarity = 1.0
    (which are identical molecules).
    
    Note: Candidates with similarity >= 0.99 might still be the same molecule (different SMILES
    representations), but we only exclude exact matches (sim = 1.0) to avoid being too aggressive.
    """
    gt_fp = ecfp4(gt_smiles)
    if gt_fp is None:
        return np.nan
    
    # Get ground truth fingerprint bits for efficient comparison
    gt_on_bits = set(gt_fp.GetOnBits()) if gt_fp else None
    
    max_sim = 0.0
    valid_candidates = 0
    
    for cand_smiles in candidate_smiles_list:
        if not cand_smiles or cand_smiles == gt_smiles:
            continue
        
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
        if not np.isnan(sim):
            # Skip candidates that are identical to ground truth (sim = 1.0)
            if sim >= 1.0:
                continue
            max_sim = max(max_sim, sim)
            valid_candidates += 1
    
    if valid_candidates == 0:
        return np.nan
    
    return max_sim

def load_candidate_map(candidates_path: str):
    """Load candidate map from JSON or PKL file."""
    if candidates_path.lower().endswith(".pkl"):
        import pickle
        with open(candidates_path, "rb") as f:
            return pickle.load(f)
    else:
        with open(candidates_path, "r") as f:
            return json.load(f)

def cap_candidate_map(cand_map_raw, pool_cap, seed=1234):
    """
    Cap candidate pools to a maximum size (same logic as eval_candidate_pool_size.py).
    Also deduplicates candidates based on fingerprint similarity = 1.0.
    """
    rng = np.random.default_rng(seed)
    cand_map = {}
    
    total = len(cand_map_raw)
    for idx, (k, vs) in enumerate(tqdm(cand_map_raw.items(), desc="Capping and deduplicating candidate pools", total=total)):
        ck = k.strip() if k else k
        if not isinstance(vs, list):
            try:
                vs = list(vs)
            except:
                vs = []
        
        # First, do string-based deduplication
        vals = []
        seen = set()
        for v in vs:
            if not v:
                continue
            cv = v.strip()
            if cv not in seen:
                vals.append(cv)
                seen.add(cv)
        
        # Ensure true SMILES is in the candidate list (for evaluation)
        # This is necessary to compute ranks, but similarity computation will exclude it
        if ck and ck not in seen:
            vals.append(ck)
        
        # Deduplicate by fingerprint similarity (removes molecules with sim=1.0)
        # This handles cases where different SMILES represent the same molecule
        # Prefer keeping the ground truth SMILES (ck) if it's in the list
        vals = deduplicate_by_fingerprint(vals, threshold=1.0, prefer_smiles=ck)
        
        # Ensure ground truth is in the list (it should be, but double-check)
        # If a candidate with the same fingerprint exists, replace it with ground truth
        if ck and ck not in vals:
            # Check if any candidate has the same fingerprint as ground truth
            gt_fp = ecfp4(ck)
            if gt_fp:
                gt_on_bits = tuple(gt_fp.GetOnBits())
                found_duplicate = False
                for i, cand in enumerate(vals):
                    cand_fp = ecfp4(cand)
                    if cand_fp:
                        cand_on_bits = tuple(cand_fp.GetOnBits())
                        if cand_on_bits == gt_on_bits:
                            # Replace this candidate with ground truth
                            vals[i] = ck
                            found_duplicate = True
                            break
                if not found_duplicate:
                    vals.append(ck)
            else:
                vals.append(ck)
        
        if pool_cap is not None and len(vals) > pool_cap:
            # Ensure ground truth is always in the final pool
            if ck and ck in vals:
                # Remove ground truth temporarily, sample, then add it back
                vals_without_gt = [v for v in vals if v != ck]
                n_to_sample = pool_cap - 1  # Reserve one slot for ground truth
                if len(vals_without_gt) > n_to_sample:
                    sampled = rng.choice(vals_without_gt, size=n_to_sample, replace=False).tolist()
                    vals = sampled + [ck]
                else:
                    vals = vals_without_gt + [ck]
            else:
                # Ground truth not in list, just sample normally
                vals = rng.choice(vals, size=pool_cap, replace=False).tolist()
        
        if vals:
            cand_map[ck] = vals
    
    return cand_map

def analyze_fixed_size_hardness(args, pool_cap=128):
    """
    Main analysis: Stratify queries by max similarity and compute Recall@1 per bin.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Fixed Pool Size (N={pool_cap}) Stratified by Hardness")
    print(f"{'='*60}\n")
    
    # Load full candidate map (for computing max similarity)
    print("Loading candidate map...")
    cand_map_full = load_candidate_map(args.candidates)
    
    # Load MGF to get query SMILES
    from specbridge.data.massspecgym import MassSpecGymDataset
    dataset = MassSpecGymDataset(args.mgf, args.meta_json, folds={args.fold_query})
    
    print(f"Loaded {len(dataset)} queries from {args.fold_query} fold")
    
    # CRITICAL FIX: We need to compute max similarity from the CAPPED pool, not the full pool!
    # Otherwise, if the hardest decoy is removed during random capping, the task becomes easier
    # but we still think it's hard based on the full pool similarity.
    
    # Cap the candidate map FIRST (this represents the fixed pool size)
    print(f"\nCapping candidate pools to size {pool_cap}...")
    cand_map_capped = cap_candidate_map(cand_map_full, pool_cap, seed=args.seed)
    
    # Verify ground truth is in candidate sets
    print("\nVerifying ground truth presence in candidate sets...")
    missing_gt = 0
    total_checked = 0
    for item in dataset:
        gt_smiles = item.get('smiles') or item.get('smiles_true')
        if not gt_smiles:
            continue
        total_checked += 1
        cand_list = cand_map_capped.get(gt_smiles, [])
        if not cand_list or gt_smiles not in cand_list:
            # Try with stripped version
            gt_stripped = gt_smiles.strip()
            cand_list = cand_map_capped.get(gt_stripped, [])
            if not cand_list or gt_stripped not in cand_list:
                missing_gt += 1
    print(f"Checked {total_checked} queries: {missing_gt} missing ground truth in candidate set")
    if missing_gt > 0:
        print(f"WARNING: {missing_gt} queries have missing ground truth in candidate sets!")
    
    # Compute max similarity for each query using CAPPED candidate pool
    # Check for existing checkpoint to avoid recomputing
    checkpoint_file = os.path.join(args.output_dir, f"max_similarity_checkpoint_{args.pool_cap}.json")
    total_items = len(dataset)
    
    # Check checkpoint version - if it was computed from full pool, we need to recompute
    checkpoint_version = "v2_capped_pool"  # Version marker
    
    if os.path.exists(checkpoint_file):
        print(f"\nLoading max similarity checkpoint from {checkpoint_file}...")
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            checkpoint_ver = checkpoint_data.get('version', 'v1_full_pool')  # Old checkpoints don't have version
            
            if checkpoint_ver != checkpoint_version:
                print(f"WARNING: Checkpoint version mismatch!")
                print(f"  Checkpoint version: {checkpoint_ver}")
                print(f"  Required version: {checkpoint_version}")
                print(f"  Old checkpoint was computed from FULL pool, but we need CAPPED pool.")
                print(f"  Deleting old checkpoint and recomputing...")
                os.remove(checkpoint_file)
                query_results = []
                processed_indices = set()
            else:
                query_results = checkpoint_data.get('query_results', [])
                processed_indices = set(checkpoint_data.get('processed_indices', []))
                print(f"Loaded {len(query_results)} queries from checkpoint ({len(processed_indices)}/{total_items} processed)")
                
                # If all items are already processed, skip computation
                if len(processed_indices) >= total_items:
                    print("All queries already processed. Skipping max similarity computation.")
                else:
                    print(f"Resuming computation from query {len(processed_indices)}/{total_items}")
    else:
        query_results = []
        processed_indices = set()
        print("\nNo checkpoint found. Starting fresh computation.")
    
    # Only compute if not all items are processed
    if len(processed_indices) < total_items:
        print(f"\nComputing max similarity for each query (using CAPPED pool, N={pool_cap})...")
        
        # Save checkpoint every 1000 queries
        checkpoint_interval = 1000
        
        for i, item in enumerate(tqdm(dataset, desc="Computing max similarity", total=total_items)):
            # Skip if already processed
            if i in processed_indices:
                continue
            
            gt_smiles = item.get('smiles') or item.get('smiles_true')
            if not gt_smiles:
                processed_indices.add(i)
                continue
            
            # Get CAPPED candidate list for this query (same as used in evaluation)
            cand_list = cand_map_capped.get(gt_smiles, [])
            if not cand_list:
                processed_indices.add(i)
                continue
            
            # Compute max similarity from CAPPED pool (this represents the actual hardness)
            max_sim = compute_max_similarity(gt_smiles, cand_list)
            
            if np.isnan(max_sim):
                processed_indices.add(i)
                continue
            
            query_results.append({
                'query_idx': i,
                'gt_smiles': gt_smiles,
                'max_similarity': max_sim,
                'pool_size_capped': len(cand_list),
                'pool_size_full': len(cand_map_full.get(gt_smiles, []))
            })
            processed_indices.add(i)
            
            # Save checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'version': checkpoint_version,
                    'query_results': query_results,
                    'processed_indices': list(processed_indices)
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f)
        
        # Final checkpoint save
        checkpoint_data = {
            'version': checkpoint_version,
            'query_results': query_results,
            'processed_indices': list(processed_indices)
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        print(f"Computed max similarity for {len(query_results)} queries")
    else:
        print(f"Using existing max similarity results for {len(query_results)} queries")
    
    # Now we need per-query success/failure data from N=128 evaluation
    
    # Since we can't easily get per-query results from the existing eval script,
    # we'll create a custom evaluation that outputs per-query ranks
    print("\nRunning custom evaluation to get per-query results...")
    per_query_ranks = run_custom_evaluation(args, cand_map_capped, dataset)
    
    # Match query results with ranks
    for qr in query_results:
        gt_smiles = qr['gt_smiles']
        if gt_smiles in per_query_ranks:
            qr['rank'] = per_query_ranks[gt_smiles]
            qr['correct'] = (per_query_ranks[gt_smiles] == 0)
        else:
            qr['rank'] = None
            qr['correct'] = None
    
    # Filter out queries without rank data
    query_results = [qr for qr in query_results if qr['rank'] is not None]
    print(f"Matched {len(query_results)} queries with evaluation results")
    
    # Bin queries by max similarity
    bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    bin_labels = ['<0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-0.99']
    
    bin_stats = {label: {'correct': 0, 'total': 0, 'ranks': []} for label in bin_labels}
    
    # Track queries with max_similarity >= 1.0 (should have been deduplicated)
    invalid_queries = []
    
    for qr in query_results:
        max_sim = qr['max_similarity']
        
        # Exclude queries with max_similarity >= 1.0 (these should have been deduplicated)
        if max_sim >= 1.0:
            invalid_queries.append(qr)
            continue
        
        bin_idx = np.digitize([max_sim], bins)[0] - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx >= len(bin_labels):
            bin_idx = len(bin_labels) - 1
        label = bin_labels[bin_idx]
        
        bin_stats[label]['total'] += 1
        if qr['correct']:
            bin_stats[label]['correct'] += 1
        bin_stats[label]['ranks'].append(qr['rank'])
    
    # Report invalid queries
    if invalid_queries:
        print(f"\nWARNING: {len(invalid_queries)} queries have max_similarity >= 1.0")
        print(f"  These should have been deduplicated but weren't!")
        print(f"  They are being excluded from the analysis.")
        print(f"  Recall@1 for these queries: {sum(1 for qr in invalid_queries if qr['correct']) / len(invalid_queries):.4f}")
        print(f"  This suggests deduplication is not working correctly for some queries.\n")
    
    # Compute statistics per bin
    results = []
    for label in bin_labels:
        stats = bin_stats[label]
        if stats['total'] > 0:
            recall = stats['correct'] / stats['total']
            median_rank = np.median(stats['ranks']) if stats['ranks'] else np.nan
            results.append({
                'max_sim_bin': label,
                'n_queries': stats['total'],
                'recall_at_1': recall,
                'median_rank': median_rank
            })
    
    # Diagnostic analysis for 0.9-0.99 bin
    print("\n" + "="*60)
    print("DIAGNOSTIC ANALYSIS: 0.9-0.99 Similarity Bin")
    print("="*60)
    high_sim_queries = [qr for qr in query_results if 0.9 <= qr['max_similarity'] < 0.99]
    if high_sim_queries:
        print(f"Total queries in 0.9-0.99 bin: {len(high_sim_queries)}")
        print(f"Recall@1: {sum(1 for qr in high_sim_queries if qr['correct']) / len(high_sim_queries):.4f}")
        print(f"Median rank: {np.median([qr['rank'] for qr in high_sim_queries]):.1f}")
        
        # Analyze similarity distribution
        sims = [qr['max_similarity'] for qr in high_sim_queries]
        print(f"\nMax similarity distribution:")
        print(f"  Min: {min(sims):.4f}")
        print(f"  Max: {max(sims):.4f}")
        print(f"  Mean: {np.mean(sims):.4f}")
        print(f"  Median: {np.median(sims):.4f}")
        print(f"  95th percentile: {np.percentile(sims, 95):.4f}")
        print(f"  99th percentile: {np.percentile(sims, 99):.4f}")
        
        # Check how many have similarity >= 0.99 (might be duplicates)
        very_high_sim = [s for s in sims if s >= 0.99]
        print(f"\nQueries with max_similarity >= 0.99: {len(very_high_sim)} ({100*len(very_high_sim)/len(high_sim_queries):.1f}%)")
        if very_high_sim:
            very_high_queries = [qr for qr in high_sim_queries if qr['max_similarity'] >= 0.99]
            very_high_recall = sum(1 for qr in very_high_queries if qr['correct']) / len(very_high_queries)
            print(f"  Recall@1 for >=0.99: {very_high_recall:.4f}")
        
        # Sample a few queries to check actual candidates
        print(f"\nSampling 5 queries to inspect candidates:")
        sample_queries = np.random.choice(high_sim_queries, size=min(5, len(high_sim_queries)), replace=False)
        for qr in sample_queries:
            gt_smiles = qr['gt_smiles']
            cand_list = cand_map_capped.get(gt_smiles, [])
            if cand_list:
                # Compute similarities to ground truth
                gt_fp = ecfp4(gt_smiles)
                if gt_fp:
                    cand_sims = []
                    for cand in cand_list[:10]:  # Check first 10 candidates
                        if cand != gt_smiles:
                            cand_fp = ecfp4(cand)
                            if cand_fp:
                                sim = tanimoto_similarity(gt_fp, cand_fp)
                                if not np.isnan(sim) and sim < 1.0:
                                    cand_sims.append((cand, sim))
                    cand_sims.sort(key=lambda x: x[1], reverse=True)
                    print(f"\n  GT: {gt_smiles[:50]}...")
                    print(f"  Max similarity: {qr['max_similarity']:.4f}, Rank: {qr['rank']}, Correct: {qr['correct']}")
                    if cand_sims:
                        print(f"    Top candidate: {cand_sims[0][1]:.4f} similarity")
                        if len(cand_sims) > 1:
                            print(f"    2nd candidate: {cand_sims[1][1]:.4f} similarity")
    print("="*60 + "\n")
    
    return results, query_results

def run_custom_evaluation(args, cand_map_capped, dataset):
    """
    Run a custom evaluation that outputs per-query ranks.
    Uses the same evaluation logic as specbridge.eval.candidates but saves per-query results.
    """
    import torch
    from torch.utils.data import DataLoader
    from specbridge.adapters.dreams_adapter import load_dreams_encoder
    from specbridge.models.mapper import DreamsToMolCondition
    from specbridge.data.massspecgym import collate_massspecgym
    import torch.nn.functional as F
    
    device = torch.device("cpu" if args.cpu else "cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model on {device}...")
    
    # Load model using the same approach as candidates.py
    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)
    
    model = DreamsToMolCondition(
        dreams_backbone,
        d_out=args.cond_dim,
        mapper_hidden=args.mapper_hidden,
        gaussian=not getattr(args, 'no_gaussian', True),
        mol_space=args.mol_space,
        chemberta_model=getattr(args, "chemberta_model", None),
        args=args
    ).to(device).eval()
    
    # Load checkpoint state
    state = torch.load(args.adapter_ckpt, map_location="cpu")
    print(model.load_state_dict(state.get("model", {}), strict=False))
    for p in model.parameters():
        p.requires_grad = False
    
    # Build mol embed function (same as candidates.py)
    from specbridge.eval.candidates import build_mol_embed_fn, _collate_with_smiles
    embed_fn = build_mol_embed_fn(args, model, device)
    
    # Get vocab sizes from dataset (same as candidates.py)
    formula_vocab = max(2, getattr(dataset, "_formula_vocab", 0) or 32)
    adduct_vocab  = max(2, getattr(dataset, "_adduct_vocab", 0) or 16)
    charge_vocab  = max(2, getattr(dataset, "_charge_vocab", 0) or 8)
    fp_bits = getattr(args, 'fp_bits', 4096)
    
    # Create collate function (same as candidates.py)
    collate = lambda b: _collate_with_smiles(
        b, args.spec_bins, formula_vocab, adduct_vocab, charge_vocab, args.fp_bits, seed=args.seed
    )
    
    # Create data loader
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    
    # Canonicalize function
    def _canon_smi(s):
        if s is None:
            return None
        return s.strip()
    
    # Precompute candidate embeddings (with optional cache)
    all_candidates = set()
    for cand_list in cand_map_capped.values():
        all_candidates.update(cand_list)
    all_candidates = sorted(list(all_candidates))
    
    cand_embeddings = {}
    if args.cache_cand_emb and os.path.exists(args.cache_cand_emb):
        try:
            cached = torch.load(args.cache_cand_emb, map_location="cpu")
            print(f"Loaded {len(cached)} cached candidate embeddings")
            cand_embeddings = {(_canon_smi(k) or k): v for k, v in cached.items()}
        except:
            pass
    
    # Compute missing embeddings
    missing = [c for c in all_candidates if (_canon_smi(c) or c) not in cand_embeddings]
    if missing:
        print(f"Computing embeddings for {len(missing)} candidates...")
        with torch.no_grad():
            Z = embed_fn(missing, device)  # [N, D]
            for smi, z in zip(missing, Z):
                cand_embeddings[_canon_smi(smi) or smi] = z.cpu()
        
        # Save cache
        if args.cache_cand_emb:
            torch.save(cand_embeddings, args.cache_cand_emb)
            print(f"Saved candidate embeddings cache to {args.cache_cand_emb}")
    
    # Evaluate
    per_query_ranks = {}
    
    print("Evaluating queries...")
    with torch.no_grad():
        for batch in tqdm(dl, desc="Computing ranks"):
            s = batch["spectra"].to(device)
            meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["meta"].items()}
            smiles_true = batch["smiles_true"]
            
            # Get query embeddings using the model (same as candidates.py)
            z_s, z_m, z_hat, mu, lv = model(s, meta, None, inference=False)
            # Use mapped embeddings (deterministic)
            z_query = mu  # [B, D]
            
            B = s.size(0)
            for i in range(B):
                gt_smiles = smiles_true[i]
                gt_smi = _canon_smi(gt_smiles) or gt_smiles.strip()
                
                # Try multiple lookup keys to find the candidate list
                cand_list = None
                lookup_keys = [gt_smi, gt_smiles, gt_smiles.strip()]
                for key in lookup_keys:
                    if key in cand_map_capped:
                        cand_list = cand_map_capped[key]
                        break
                
                # If still not found, try to find by matching any key
                if cand_list is None:
                    for map_key, map_cands in cand_map_capped.items():
                        # Check if ground truth matches the key (with canonicalization)
                        if (_canon_smi(map_key) == gt_smi or 
                            map_key == gt_smi or 
                            map_key == gt_smiles or
                            map_key.strip() == gt_smiles.strip()):
                            cand_list = map_cands
                            break
                
                if not cand_list:
                    # No candidate list found - skip this query
                    continue
                
                # Verify ground truth is in candidate list
                gt_found = False
                for sm in cand_list:
                    csm = _canon_smi(sm) or sm.strip()
                    if csm == gt_smi or sm == gt_smiles or sm == gt_smi:
                        gt_found = True
                        break
                
                if not gt_found:
                    # Add ground truth if not found
                    cand_list.append(gt_smi)
                
                # Get candidate embeddings
                Z = []
                kept = []
                for sm in cand_list:
                    csm = _canon_smi(sm) or sm.strip()
                    emb_key = csm
                    if emb_key in cand_embeddings:
                        Z.append(cand_embeddings[emb_key].to(device))
                        kept.append(csm)
                    else:
                        # Compute embedding on the fly if missing
                        with torch.no_grad():
                            z_missing = embed_fn([sm], device)
                            cand_embeddings[emb_key] = z_missing[0].cpu()
                            Z.append(z_missing[0])
                            kept.append(csm)
                
                if not Z:
                    continue
                
                Z = torch.stack(Z, dim=0)  # [C, D]
                zqi = z_query[i]  # [D]
                
                # Compute similarities
                sims = zqi @ Z.T  # [C]
                order = torch.argsort(sims, descending=True)
                
                # Find rank of ground truth (must be in kept since we verified it's in cand_list)
                gt_canon = _canon_smi(gt_smiles) or gt_smiles.strip()
                gt_idx = None
                
                # Try multiple matching strategies
                for idx, kept_smi in enumerate(kept):
                    if (kept_smi == gt_canon or 
                        kept_smi == gt_smi or 
                        kept_smi == gt_smiles or
                        _canon_smi(kept_smi) == gt_canon):
                        gt_idx = idx
                        break
                
                if gt_idx is not None:
                    pos = (order == gt_idx).nonzero(as_tuple=False)
                    rank = int(pos.item()) if pos.numel() > 0 else len(kept) - 1
                    per_query_ranks[gt_smi] = rank
                else:
                    # This shouldn't happen, but log it if it does
                    print(f"WARNING: Ground truth {gt_smiles} not found in kept list for query {i}")
                    print(f"  gt_smi: {gt_smi}, gt_canon: {gt_canon}")
                    print(f"  kept list (first 5): {kept[:5]}")
                    # Assign worst rank
                    per_query_ranks[gt_smi] = len(kept)
    
    return per_query_ranks

def main():
    ap = argparse.ArgumentParser(
        description="Analyze fixed pool size performance stratified by chemical hardness"
    )
    ap.add_argument("--mgf", required=True, type=str)
    ap.add_argument("--meta-json", type=str, default=None)
    ap.add_argument("--candidates", required=True, type=str)
    ap.add_argument("--adapter-ckpt", required=True, type=str)
    ap.add_argument("--dreams-ckpt", type=str, default=None)
    ap.add_argument("--fold-query", type=str, default="test")
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=2048)
    ap.add_argument("--mapper-hidden", type=int, default=2048)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--mol-space", choices=["adapter", "ecfp", "chemberta"], default="chemberta")
    ap.add_argument("--chemberta-model", type=str, default="Derify/ChemBERTa_augmented_pubchem_13m")
    ap.add_argument("--cache-cand-emb", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--output-dir", type=str, default="figs")
    ap.add_argument("--pool-cap", type=int, default=128, help="Fixed pool size to analyze")
    ap.add_argument("--fp-bits", type=int, default=4096, help="Number of fingerprint bits (for ECFP)")
    args = ap.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis
    bin_results, query_results = analyze_fixed_size_hardness(args, pool_cap=args.pool_cap)
    
    # Save results
    df_bins = pd.DataFrame(bin_results)
    csv_path = os.path.join(args.output_dir, f"fixed_size_{args.pool_cap}_hardness_stratified.csv")
    df_bins.to_csv(csv_path, index=False)
    print(f"\nBinned results saved to: {csv_path}")
    
    df_queries = pd.DataFrame(query_results)
    csv_queries_path = os.path.join(args.output_dir, f"fixed_size_{args.pool_cap}_per_query_results.csv")
    df_queries.to_csv(csv_queries_path, index=False)
    print(f"Per-query results saved to: {csv_queries_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RECALL@1 FOR FIXED POOL SIZE (N={args.pool_cap}) STRATIFIED BY MAX SIMILARITY")
    print(f"{'='*60}")
    for row in bin_results:
        print(f"MaxSim {row['max_sim_bin']:12s}: {row['recall_at_1']:.4f} (n={row['n_queries']:5d}, median_rank={row['median_rank']:.1f})")
    print(f"{'='*60}\n")
    
    # Create visualization
    set_paper_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=300)
    
    bin_labels = [r['max_sim_bin'] for r in bin_results]
    recalls = [r['recall_at_1'] for r in bin_results]
    n_queries = [r['n_queries'] for r in bin_results]
    
    # Create bar plot
    bars = ax.bar(range(len(bin_labels)), recalls, color=palette[0], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (recall, n) in enumerate(zip(recalls, n_queries)):
        ax.text(i, recall + 0.02, f'{recall:.3f}\n(n={n})', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Maximum Tanimoto Similarity of Hardest Decoy', fontsize=16)
    ax.set_ylabel('Recall@1', fontsize=16)
    ax.set_title(f'Retrieval Performance vs. Chemical Hardness\n(Fixed Pool Size: N={args.pool_cap})', 
                 fontsize=18, pad=10)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=14, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')
    
    # Add horizontal line at MassSpecGym baseline (84.7%)
    ax.axhline(y=0.847, color='red', linestyle='--', linewidth=2, alpha=0.7, label='MassSpecGym baseline (84.7%)')
    ax.legend(fontsize=12, loc='upper right')
    
    plt.tight_layout()
    
    fig_path = os.path.join(args.output_dir, f"fixed_size_{args.pool_cap}_hardness_stratified.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
