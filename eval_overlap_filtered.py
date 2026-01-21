#!/usr/bin/env python3
"""
Evaluate a checkpoint on MassSpecGym test set with different overlap filtering thresholds.

This script:
1. Checks overlap between test set and pretrain data at multiple thresholds
2. Filters test set to remove overlapping spectra at each threshold
3. Evaluates checkpoint performance on each filtered subset
4. Saves results to CSV for analysis

Overlap results are cached to avoid recomputing.
"""

import argparse
import sys
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Set, Dict, Any
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import overlap checking functions
import importlib.util
overlap_module_path = project_root / "check_pretrain_overlap.py"
spec = importlib.util.spec_from_file_location("check_pretrain_overlap", overlap_module_path)
check_overlap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_overlap)

from specbridge.data.massspecgym import MassSpecGymDataset
import h5py


def get_overlap_cache_path(mgf_path: str, gems_path: str, thresholds: List[float], 
                          num_bins: int, max_mz: float) -> str:
    """Generate a cache file path based on input parameters."""
    # Create a hash of the parameters to create a unique cache file
    cache_key = f"{mgf_path}_{gems_path}_{sorted(thresholds)}_{num_bins}_{max_mz}"
    cache_hash = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    cache_dir = project_root / "cache"
    cache_dir.mkdir(exist_ok=True)
    return str(cache_dir / f"overlap_cache_{cache_hash}.pkl")


def load_overlap_cache(cache_path: str) -> Dict[float, Set[int]]:
    """Load overlap results from cache."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load overlap cache: {e}")
    return None


def save_overlap_cache(cache_path: str, threshold_overlaps: Dict[float, Set[int]]):
    """Save overlap results to cache."""
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(threshold_overlaps, f)
        print(f"Saved overlap results to cache: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not save overlap cache: {e}")


def get_overlapping_indices_at_thresholds(
    test_spectra: List[Dict[str, Any]],
    gems_path: str,
    thresholds: List[float],
    num_bins: int = 20000,
    max_mz: float = 2000.0,
    pretrain_batch_size: int = 10000,
    max_pretrain: int = None
) -> Dict[float, Set[int]]:
    """
    Find overlapping indices at multiple similarity thresholds.
    
    Returns:
        Dict mapping threshold -> set of overlapping test indices
    """
    print(f"Finding overlaps at thresholds: {thresholds}")
    
    # Check dataset size to decide on streaming vs in-memory
    with h5py.File(gems_path, 'r') as f:
        total_pretrain = f['spectrum'].shape[0]
        if max_pretrain is not None:
            total_pretrain = min(total_pretrain, max_pretrain)
    
    use_streaming = total_pretrain > 100000 or max_pretrain is None
    
    # For each threshold, find overlapping indices
    threshold_overlaps: Dict[float, Set[int]] = {}
    
    if use_streaming:
        print("Using streaming approach...")
        from tqdm import tqdm
        import torch.nn.functional as F
        
        print("Binning test spectra...")
        test_binned = []
        for spec in tqdm(test_spectra, desc="Binning test"):
            binned = check_overlap.bin_spectrum(spec['mz'], spec['intensity'], num_bins=num_bins, max_mz=max_mz)
            test_binned.append(binned)
        
        test_binned = torch.stack(test_binned)  # Shape: [N_test, num_bins]
        test_norm = test_binned / (test_binned.norm(dim=1, keepdim=True) + 1e-8)
        
        # Track maximum similarity for each test spectrum
        max_similarities = torch.zeros(len(test_spectra), dtype=torch.float32)
        
        print("Processing pretrain data in batches...")
        with h5py.File(gems_path, 'r') as f:
            spec_data = f['spectrum']
            total_pretrain = spec_data.shape[0]
            if max_pretrain is not None:
                total_pretrain = min(total_pretrain, max_pretrain)
            
            num_batches = (total_pretrain + pretrain_batch_size - 1) // pretrain_batch_size
            
            for batch_idx in tqdm(range(num_batches), desc="Processing pretrain batches"):
                start_idx = batch_idx * pretrain_batch_size
                end_idx = min(start_idx + pretrain_batch_size, total_pretrain)
                batch_size_actual = end_idx - start_idx
                
                # Load batch from HDF5
                batch_data = spec_data[start_idx:end_idx, :, :]  # (batch_size, 2, num_peaks)
                
                # Extract and bin pretrain batch
                pretrain_batch_binned = []
                for i in range(batch_size_actual):
                    mz_array = batch_data[i, 0, :]
                    intensity_array = batch_data[i, 1, :]
                    mask = intensity_array > 1e-10
                    
                    if mask.sum() > 0:
                        mz = torch.tensor(mz_array[mask], dtype=torch.float32)
                        intensity = torch.tensor(intensity_array[mask], dtype=torch.float32)
                        binned = check_overlap.bin_spectrum(mz, intensity, num_bins=num_bins, max_mz=max_mz)
                        pretrain_batch_binned.append(binned)
                
                if len(pretrain_batch_binned) == 0:
                    continue
                
                # Stack and normalize pretrain batch
                pretrain_batch = torch.stack(pretrain_batch_binned)  # (batch_size, num_bins)
                pretrain_batch_norm = pretrain_batch / (pretrain_batch.norm(dim=1, keepdim=True) + 1e-8)
                
                # Compute similarities: (N_test, num_bins) @ (num_bins, batch_size) = (N_test, batch_size)
                cos_sims = test_norm @ pretrain_batch_norm.T  # (N_test, batch_size)
                
                # Update maximum similarities
                batch_max = cos_sims.max(dim=1)[0]  # (N_test,)
                max_similarities = torch.maximum(max_similarities, batch_max)
        
        # Now filter by each threshold
        max_sims_np = max_similarities.numpy()
        for threshold in thresholds:
            overlapping_mask = max_sims_np >= threshold
            overlapping_indices = set(torch.where(torch.tensor(overlapping_mask))[0].tolist())
            threshold_overlaps[threshold] = overlapping_indices
            print(f"Threshold {threshold}: {len(overlapping_indices)} overlapping spectra")
    
    else:
        print("Using in-memory approach...")
        pretrain_spectra = check_overlap.load_gems_spectra(gems_path, max_spectra=max_pretrain)
        
        if len(pretrain_spectra) == 0:
            raise ValueError("No spectra loaded from GeMS")
        
        from tqdm import tqdm
        
        print("Binning pretrain spectra...")
        pretrain_binned = []
        for spec in tqdm(pretrain_spectra, desc="Binning pretrain"):
            binned = check_overlap.bin_spectrum(spec['mz'], spec['intensity'], num_bins=num_bins, max_mz=max_mz)
            pretrain_binned.append(binned)
        
        pretrain_binned = torch.stack(pretrain_binned)  # Shape: [N_pretrain, num_bins]
        pretrain_norm = pretrain_binned / (pretrain_binned.norm(dim=1, keepdim=True) + 1e-8)
        
        print("Binning test spectra...")
        test_binned = []
        for spec in tqdm(test_spectra, desc="Binning test"):
            binned = check_overlap.bin_spectrum(spec['mz'], spec['intensity'], num_bins=num_bins, max_mz=max_mz)
            test_binned.append(binned)
        
        test_binned = torch.stack(test_binned)  # Shape: [N_test, num_bins]
        test_norm = test_binned / (test_binned.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute all similarities
        print("Computing similarities...")
        cos_sims = test_norm @ pretrain_norm.T  # (N_test, N_pretrain)
        max_similarities = cos_sims.max(dim=1)[0]  # (N_test,)
        
        # Filter by each threshold
        max_sims_np = max_similarities.numpy()
        for threshold in thresholds:
            overlapping_mask = max_sims_np >= threshold
            overlapping_indices = set(torch.where(torch.tensor(overlapping_mask))[0].tolist())
            threshold_overlaps[threshold] = overlapping_indices
            print(f"Threshold {threshold}: {len(overlapping_indices)} overlapping spectra")
    
    return threshold_overlaps


def create_filtered_dataset_indices(
    total_size: int,
    overlapping_indices: Set[int]
) -> List[int]:
    """Create list of indices for filtered dataset (excluding overlapping ones)."""
    all_indices = set(range(total_size))
    filtered_indices = sorted(list(all_indices - overlapping_indices))
    return filtered_indices


def run_evaluation(
    ckpt_path: str,
    mgf_path: str,
    candidates_path: str,
    dreams_ckpt: str,
    filtered_indices: List[int],
    output_log: str,
    args_dict: Dict[str, Any]
) -> Dict[str, float]:
    """
    Run evaluation on filtered dataset and parse metrics from log.
    
    Returns:
        Dict with metrics: R@1, R@5, R@20, MRR, median_rank, total_queries, evaluated
    """
    # Import evaluation components
    from specbridge.eval.candidates import (
        build_model, build_mol_embed_fn, _collate_with_smiles,
        _metrics_from_ranks, _canon_smi
    )
    from specbridge.data.massspecgym import MassSpecGymDataset
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    import torch
    import random
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create a wrapper for filtered dataset
    class FilteredDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = sorted(indices)
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.base_dataset[self.indices[idx]]
    
    # Load full dataset
    full_ds = MassSpecGymDataset(mgf_path, folds={'test'})
    
    # Create filtered dataset
    filtered_ds = FilteredDataset(full_ds, filtered_indices)
    
    # Setup dataset parameters
    formula_vocab = max(2, getattr(full_ds, "_formula_vocab", 0) or 32)
    adduct_vocab = max(2, getattr(full_ds, "_adduct_vocab", 0) or 16)
    charge_vocab = max(2, getattr(full_ds, "_charge_vocab", 0) or 8)
    
    # Create data loader
    collate = lambda b: _collate_with_smiles(
        b, args_dict.get("spec_bins", 2048), formula_vocab, 
        adduct_vocab, charge_vocab, args_dict.get("fp_bits", 2048), 
        seed=args_dict.get("seed", 1234)
    )
    dl = DataLoader(filtered_ds, batch_size=args_dict.get("batch_size", 32), 
                    shuffle=False, num_workers=0, collate_fn=collate)
    
    # Load candidates
    if os.path.splitext(candidates_path)[1].lower() == ".pkl":
        with open(candidates_path, "rb") as f:
            cand_map_raw = pickle.load(f)
    else:
        with open(candidates_path, "r") as f:
            cand_map_raw = json.load(f)
    
    cand_map = {}
    for k, vs in cand_map_raw.items():
        ck = _canon_smi(k) or k.strip()
        vals = []
        for v in vs:
            if not v:
                continue
            cv = _canon_smi(v) or v.strip()
            vals.append(cv)
        if ck not in vals:
            vals.append(ck)
        random.shuffle(vals)
        cand_map[ck] = list(dict.fromkeys(vals))
    
    # Build model
    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    eval_args = Args({
        "adapter_ckpt": ckpt_path,
        "dreams_ckpt": dreams_ckpt,
        "spec_bins": args_dict.get("spec_bins", 2048),
        "cond_dim": args_dict.get("cond_dim", 2048),
        "mapper_hidden": args_dict.get("mapper_hidden", 2048),
        "no_gaussian": True,
        "n_blocks": args_dict.get("n_blocks", 8),
        "mol_space": args_dict.get("mol_space", "chemberta"),
        "chemberta_model": args_dict.get("chemberta_model", "Derify/ChemBERTa_augmented_pubchem_13m"),
        "fp_bits": args_dict.get("fp_bits", 2048),
        "use_mapped": True,
        "deterministic_map": True,
    })
    
    model = build_model(eval_args, device)
    embed_fn = build_mol_embed_fn(eval_args, model, device)
    
    # Collect candidate universe
    all_query_true = []
    for batch in dl:
        all_query_true.extend(batch["smiles_true"])
    
    need_smiles = set()
    covered_titles = 0
    for smi in all_query_true:
        csmi = _canon_smi(smi) or smi.strip()
        if csmi in cand_map:
            covered_titles += 1
            need_smiles.update(cand_map[csmi])
    need_smiles = sorted(list(need_smiles))
    diag_msg = f"[candidates] queries={len(all_query_true)} | queries_with_candidates={covered_titles} | unique_candidates={len(need_smiles)}"
    print(diag_msg)
    
    # Precompute candidate embeddings
    cand_z = {}
    cache_path = args_dict.get("cache_cand_emb")
    loaded_cache = False
    if cache_path and os.path.isfile(cache_path):
        try:
            cand_z = torch.load(cache_path, map_location="cpu")
            loaded_cache = True
            print(f"[cache] loaded candidate embeddings from {cache_path} (|Z|={len(cand_z)})")
        except Exception:
            cand_z = {}
    
    if not loaded_cache and len(need_smiles) > 0:
        Z = embed_fn(need_smiles, device)
        cand_z = {(_canon_smi(s) or s.strip()): z for s, z in zip(need_smiles, Z)}
        if cache_path:
            torch.save(cand_z, cache_path)
            print(f"[cache] saved candidate embeddings to {cache_path}")
    
    uniq_true = sorted({(_canon_smi(s) or s.strip()) for s in all_query_true if s})
    Zt = embed_fn(uniq_true, device)
    true_z = {s: z for s, z in zip(uniq_true, Zt)}
    
    # Evaluate
    ranks = []
    used = 0
    total = 0
    missing_true = 0
    missing_cand = 0
    cand_sizes = []
    
    with open(output_log, 'w') as log_file:
        import sys
        original_stdout = sys.stdout
        log_file.write(diag_msg + "\n")
        sys.stdout = log_file
        
        try:
            with torch.no_grad():
                for batch in dl:
                    s = batch["spectra"].to(device)
                    meta = {k: (v.to(device) if torch.is_tensor(v) else v) 
                           for k, v in batch["meta"].items()}
                    smiles_true = batch["smiles_true"]
                    
                    z_s, z_m, z_hat, mu, lv = model(s, meta, None, inference=False)
                    
                    if lv is None:
                        z_query = mu
                    else:
                        z_query = mu
                    
                    B = s.size(0)
                    for i in range(B):
                        total += 1
                        true_raw = smiles_true[i]
                        true_smi = _canon_smi(true_raw) or true_raw.strip()
                        cand_list = cand_map.get(true_smi, None)
                        if cand_list is None or len(cand_list) == 0:
                            missing_cand += 1
                            continue
                        used += 1
                        
                        Z = []
                        kept = []
                        for sm in cand_list:
                            csm = _canon_smi(sm) or sm.strip()
                            zi = cand_z.get(csm, None)
                            if zi is not None:
                                Z.append(zi.to(device))
                                kept.append(csm)
                        if not Z:
                            missing_cand += 1
                            continue
                        
                        Z = torch.stack(Z, dim=0)
                        cand_sizes.append(Z.size(0))
                        
                        zqi = z_query[i]
                        Z = Z.to(zqi.device, non_blocking=True)
                        sims = zqi @ Z.T
                        
                        order = torch.argsort(sims, descending=True)
                        
                        if true_smi in kept:
                            idx = kept.index(true_smi)
                            pos = (order == idx).nonzero(as_tuple=False)
                            r = int(pos.item()) if pos.numel() > 0 else Z.size(0) - 1
                            ranks.append(r)
                        else:
                            missing_true += 1
            
            # Compute metrics
            m = _metrics_from_ranks(ranks)
            n_c = len(cand_sizes)
            avg_c = (sum(cand_sizes) / n_c) if n_c else 0.0
            med_c = float(sorted(cand_sizes)[n_c // 2]) if n_c else float("nan")
            
            print(f"[eval] total_queries={total} | evaluated={used} | true_found_in_list={len(ranks)} | "
                  f"missing_true={missing_true} | missing_candlist={missing_cand}")
            print(f"[candidates] avg_size={avg_c:.1f} | median_size={med_c:.0f}")
            for k in ("R@1", "R@5", "R@10", "R@20"):
                print(f"{k:>10s}: {m.get(k, 0.0):.5f}")
            print(f"{'MRR':>10s}: {m['MRR']:.5f}")
            print(f"{'median_rank':>10s}: {m['median_rank']:.1f}")
            
        finally:
            sys.stdout = original_stdout
    
    # Parse metrics from log
    metrics = parse_metrics_from_log(output_log)
    return metrics


def parse_metrics_from_log(log_path: str) -> Dict[str, float]:
    """Parse evaluation metrics from log file."""
    metrics = {
        "R@1": float('nan'),
        "R@5": float('nan'),
        "R@20": float('nan'),
        "MRR": float('nan'),
        "median_rank": float('nan'),
        "total_queries": 0,
        "evaluated": 0
    }
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        import re
        r1_match = re.search(r'R@1\s*:\s*([\d.]+)', content)
        r5_match = re.search(r'R@5\s*:\s*([\d.]+)', content)
        r20_match = re.search(r'R@20\s*:\s*([\d.]+)', content)
        mrr_match = re.search(r'MRR\s*:\s*([\d.]+)', content)
        med_match = re.search(r'median_rank\s*:\s*([\d.]+)', content)
        eval_match = re.search(r'\[eval\]\s+total_queries=(\d+).*?evaluated=(\d+)', content)
        
        if r1_match:
            metrics["R@1"] = float(r1_match.group(1))
        if r5_match:
            metrics["R@5"] = float(r5_match.group(1))
        if r20_match:
            metrics["R@20"] = float(r20_match.group(1))
        if mrr_match:
            metrics["MRR"] = float(mrr_match.group(1))
        if med_match:
            metrics["median_rank"] = float(med_match.group(1))
        if eval_match:
            metrics["total_queries"] = int(eval_match.group(1))
            metrics["evaluated"] = int(eval_match.group(2))
            
    except Exception as e:
        print(f"Error parsing log {log_path}: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint on filtered test sets based on overlap thresholds"
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--msgym-mgf", type=str,
                       default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf")
    parser.add_argument("--gems-url", type=str,
                       default="https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/GeMS_A/GeMS_A10.hdf5")
    parser.add_argument("--gems-path", type=str,
                       default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/GeMS_A10.hdf5")
    parser.add_argument("--candidates", type=str, required=True)
    parser.add_argument("--dreams-ckpt", type=str, required=True)
    parser.add_argument("--thresholds", type=str, default="1.0,0.99,0.97,0.95,0.9,0.8,0.7")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--overlap-cache", type=str, default=None,
                       help="Path to save/load overlap cache (auto-generated if not provided)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable overlap caching")
    
    # Evaluation args
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--spec-bins", type=int, default=2048)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--cond-dim", type=int, default=2048)
    parser.add_argument("--mapper-hidden", type=int, default=2048)
    parser.add_argument("--mol-space", type=str, default="chemberta",
                       choices=["chemberta", "ecfp", "adapter"])
    parser.add_argument("--cache-cand-emb", type=str, default=None)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--chemberta-model", type=str,
                       default="Derify/ChemBERTa_augmented_pubchem_13m")
    parser.add_argument("--seed", type=int, default=1234)
    
    # Overlap check args
    parser.add_argument("--num-bins", type=int, default=20000)
    parser.add_argument("--max-mz", type=float, default=2000.0)
    parser.add_argument("--pretrain-batch-size", type=int, default=10000)
    parser.add_argument("--max-pretrain", type=int, default=None)
    
    args = parser.parse_args()
    
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    thresholds = sorted(thresholds, reverse=True)
    
    print("=" * 80)
    print("Overlap-Filtered Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Thresholds: {thresholds}")
    print(f"Output CSV: {args.output_csv}")
    
    # Setup log directory
    if args.log_dir is None:
        args.log_dir = os.path.join(os.path.dirname(args.output_csv), "eval_logs_overlap")
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load test dataset
    print("\n" + "=" * 80)
    print("Loading MassSpecGym test set...")
    print("=" * 80)
    test_dataset = MassSpecGymDataset(args.msgym_mgf, folds={'test'})
    print(f"Loaded {len(test_dataset)} test spectra")
    
    # Extract test spectra for overlap checking
    test_spectra = []
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        test_spectra.append({
            'mz': item['mz'],
            'intensity': item['intensity'],
            'title': item['title'],
            'smiles': item.get('smiles', ''),
        })
    
    # Download/load GeMS data
    print("\n" + "=" * 80)
    print("Checking GeMS pretrain data (self-supervised training dataset)...")
    print("=" * 80)
    gems_path = check_overlap.download_gems_data(args.gems_url, args.gems_path)
    
    # Determine cache path
    if args.no_cache:
        overlap_cache_path = None
    else:
        if args.overlap_cache:
            overlap_cache_path = args.overlap_cache
        else:
            overlap_cache_path = get_overlap_cache_path(
                args.msgym_mgf, gems_path, thresholds,
                args.num_bins, args.max_mz
            )
    
    # Find overlapping indices at all thresholds (with caching)
    print("\n" + "=" * 80)
    print("Finding overlapping spectra at all thresholds...")
    print("=" * 80)
    filtering_thresholds = [t for t in thresholds if t < 1.0]
    
    if filtering_thresholds:
        # Try to load from cache
        threshold_overlaps = None
        if overlap_cache_path and not args.no_cache:
            print(f"Checking for overlap cache at: {overlap_cache_path}")
            threshold_overlaps = load_overlap_cache(overlap_cache_path)
            if threshold_overlaps is not None:
                print(f"Loaded overlap results from cache!")
                # Verify all thresholds are present
                missing_thresholds = [t for t in filtering_thresholds if t not in threshold_overlaps]
                if missing_thresholds:
                    print(f"Warning: Cache missing thresholds {missing_thresholds}, will recompute")
                    threshold_overlaps = None
        
        # Compute if not cached
        if threshold_overlaps is None:
            print("Computing overlap (this may take a while)...")
            threshold_overlaps = get_overlapping_indices_at_thresholds(
                test_spectra,
                gems_path,
                filtering_thresholds,
                num_bins=args.num_bins,
                max_mz=args.max_mz,
                pretrain_batch_size=args.pretrain_batch_size,
                max_pretrain=args.max_pretrain
            )
            
            # Save to cache
            if overlap_cache_path and not args.no_cache:
                save_overlap_cache(overlap_cache_path, threshold_overlaps)
        else:
            print(f"Using cached overlap results for thresholds: {list(threshold_overlaps.keys())}")
    else:
        threshold_overlaps = {}
    
    # Prepare results
    results = []
    eval_args = {
        "batch_size": args.batch_size,
        "spec_bins": args.spec_bins,
        "fp_bits": args.fp_bits,
        "cond_dim": args.cond_dim,
        "mapper_hidden": args.mapper_hidden,
        "mol_space": args.mol_space,
        "cache_cand_emb": args.cache_cand_emb or "",
        "n_blocks": args.n_blocks,
        "chemberta_model": args.chemberta_model,
        "seed": args.seed,
    }
    
    # Evaluate at each threshold
    print("\n" + "=" * 80)
    print("Running evaluations...")
    print("=" * 80)
    
    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")
        
        if threshold >= 1.0:
            filtered_indices = list(range(len(test_dataset)))
            overlap_count = 0
        else:
            overlapping_indices = threshold_overlaps[threshold]
            filtered_indices = create_filtered_dataset_indices(
                len(test_dataset),
                overlapping_indices
            )
            overlap_count = len(overlapping_indices)
        
        print(f"Total test spectra: {len(test_dataset)}")
        print(f"Overlapping (>= {threshold}): {overlap_count}")
        print(f"Filtered (remaining): {len(filtered_indices)}")
        
        log_path = os.path.join(args.log_dir, f"eval_threshold_{threshold:.2f}.log")
        print(f"Running evaluation... (log: {log_path})")
        
        metrics = run_evaluation(
            args.ckpt,
            args.msgym_mgf,
            args.candidates,
            args.dreams_ckpt,
            filtered_indices,
            log_path,
            eval_args
        )
        
        result_row = {
            "threshold": threshold,
            "total_spectra": len(test_dataset),
            "overlapping": overlap_count,
            "filtered_remaining": len(filtered_indices),
            "R@1": metrics["R@1"],
            "R@5": metrics["R@5"],
            "R@20": metrics["R@20"],
            "MRR": metrics["MRR"],
            "median_rank": metrics["median_rank"],
            "total_queries": metrics["total_queries"],
            "evaluated": metrics["evaluated"],
        }
        results.append(result_row)
        
        print(f"Results: R@1={metrics['R@1']:.4f}, R@5={metrics['R@5']:.4f}, R@20={metrics['R@20']:.4f}, MRR={metrics['MRR']:.4f}")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {args.output_csv}")
    if overlap_cache_path and not args.no_cache:
        print(f"Overlap cache saved to: {overlap_cache_path}")
    print(f"{'=' * 80}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

