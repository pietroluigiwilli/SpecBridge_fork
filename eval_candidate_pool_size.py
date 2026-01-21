#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate sensitivity to candidate pool size by downsampling PubChem candidates
at inference time (no retraining).
"""

import os
import json
import argparse
import re
import numpy as np
import torch
import subprocess
import tempfile
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Paper style with larger text
def set_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 16,  # Increased further
        "axes.titlesize": 18,  # Increased further
        "axes.labelsize": 16,  # Increased further
        "legend.fontsize": 14,  # Increased further
        "xtick.labelsize": 14,  # Increased further
        "ytick.labelsize": 14,  # Increased further
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linestyle": ":", "grid.alpha": 0.30,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

# Colors from seaborn set2 palette
palette = sns.color_palette("Set2", n_colors=8)

def _canon_smi(s):
    """Simple canonicalization - just strip for now."""
    if s is None:
        return None
    return s.strip()

def cap_candidate_map(cand_map_raw, pool_cap, seed=1234):
    """Cap candidate pools to a maximum size."""
    rng = np.random.default_rng(seed)
    cand_map = {}
    
    for k, vs in cand_map_raw.items():
        ck = _canon_smi(k) or (k.strip() if k else k)
        # Handle both list and other iterable types
        if not isinstance(vs, list):
            try:
                vs = list(vs)
            except:
                vs = []
        
        vals = []
        seen = set()
        for v in vs:
            if not v:
                continue
            cv = _canon_smi(v) or v.strip()
            if cv not in seen:
                vals.append(cv)
                seen.add(cv)
        if ck and ck not in seen:
            vals.append(ck)
        
        # Cap the candidate pool size
        if pool_cap is not None and len(vals) > pool_cap:
            vals = rng.choice(vals, size=pool_cap, replace=False).tolist()
        
        if vals:
            cand_map[ck] = vals
    
    return cand_map

def run_evaluation(args, pool_cap, temp_cand_file):
    """Run evaluation with capped candidates."""
    # Build command
    cmd = [
        "python", "-m", "specbridge.eval.candidates",
        "--mgf", args.mgf,
        "--adapter-ckpt", args.adapter_ckpt,
        "--candidates", temp_cand_file,
        "--fold-query", args.fold_query,
        "--use-mapped",
        "--deterministic-map",
        "--batch-size", str(args.batch_size),
        "--cond-dim", str(args.cond_dim),
        "--mapper-hidden", str(args.mapper_hidden),
        "--n-blocks", str(args.n_blocks),
        "--no-gaussian",
        "--mol-space", args.mol_space,
        "--chemberta-model", args.chemberta_model,
        "--spec-bins", str(args.spec_bins),
        "--seed", str(args.seed),
    ]
    
    if args.dreams_ckpt:
        cmd.extend(["--dreams-ckpt", args.dreams_ckpt])
    
    if args.meta_json:
        cmd.extend(["--meta-json", args.meta_json])
    
    if args.cache_cand_emb:
        cmd.extend(["--cache-cand-emb", args.cache_cand_emb])
    
    if args.cpu:
        cmd.append("--cpu")
    
    # Run evaluation
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running evaluation for cap={pool_cap}:")
        print(result.stderr)
        return None
    
    # Parse metrics from output
    output = result.stdout
    metrics = {}
    
    # Parse R@K
    for k in [1, 5, 10, 20]:
        pattern = rf'R@{k}:\s+([\d.]+)'
        match = re.search(pattern, output)
        if match:
            metrics[f"R@{k}"] = float(match.group(1))
        else:
            metrics[f"R@{k}"] = 0.0
    
    # Parse MRR
    pattern = r'MRR:\s+([\d.]+)'
    match = re.search(pattern, output)
    if match:
        metrics["MRR"] = float(match.group(1))
    else:
        metrics["MRR"] = 0.0
    
    # Parse median rank
    pattern = r'median_rank:\s+([\d.]+)'
    match = re.search(pattern, output)
    if match:
        metrics["median_rank"] = float(match.group(1))
    else:
        metrics["median_rank"] = float("nan")
    
    # Parse total queries and evaluated
    pattern = r'total_queries=(\d+)'
    match = re.search(pattern, output)
    if match:
        metrics["total_queries"] = int(match.group(1))
    else:
        metrics["total_queries"] = 0
    
    pattern = r'evaluated=(\d+)'
    match = re.search(pattern, output)
    if match:
        metrics["evaluated"] = int(match.group(1))
    else:
        metrics["evaluated"] = 0
    
    metrics["pool_cap"] = pool_cap if pool_cap is not None else "all"
    
    return metrics

def main():
    ap = argparse.ArgumentParser("Evaluate sensitivity to candidate pool size")
    ap.add_argument("--mgf", required=True, type=str)
    ap.add_argument("--meta-json", type=str, default=None)
    ap.add_argument("--candidates", required=True, type=str, help="JSON or PKL with candidate map")
    ap.add_argument("--adapter-ckpt", required=True, type=str)
    ap.add_argument("--dreams-ckpt", type=str, default=None)
    ap.add_argument("--fold-query", type=str, default="test", help="test or val")
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
    args = ap.parse_args()
    
    # Check if results CSV already exists
    results_csv = os.path.join(args.output_dir, "candidate_pool_size_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    
    if os.path.exists(results_csv):
        print(f"Results CSV already exists at: {results_csv}")
        print("Skipping evaluation and plotting from existing results...")
        df = pd.read_csv(results_csv)
    else:
        # Load candidate map
        if args.candidates.lower().endswith(".pkl"):
            import pickle
            with open(args.candidates, "rb") as f:
                cand_map_raw = pickle.load(f)
        else:
            with open(args.candidates, "r") as f:
                cand_map_raw = json.load(f)
        
        # Pool size caps to test
        pool_caps = [128, 256, 512, 1024, None]  # None means all
        
        # Evaluate for each pool cap
        results = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for cap in pool_caps:
                print(f"\n{'='*60}")
                print(f"Evaluating with pool cap: {cap if cap is not None else 'all'}")
                print(f"{'='*60}")
                
                # Cap candidate map
                cand_map_capped = cap_candidate_map(cand_map_raw, cap, seed=args.seed)
                
                # Save to temporary file (use same format as input)
                if args.candidates.lower().endswith(".pkl"):
                    import pickle
                    temp_cand_file = os.path.join(tmpdir, f"candidates_cap_{cap if cap else 'all'}.pkl")
                    with open(temp_cand_file, "wb") as f:
                        pickle.dump(cand_map_capped, f)
                else:
                    temp_cand_file = os.path.join(tmpdir, f"candidates_cap_{cap if cap else 'all'}.json")
                    with open(temp_cand_file, "w") as f:
                        json.dump(cand_map_capped, f)
                
                # Run evaluation
                metrics = run_evaluation(args, cap, temp_cand_file)
                if metrics:
                    results.append(metrics)
                    print(f"R@1: {metrics['R@1']:.4f}, R@5: {metrics['R@5']:.4f}, "
                          f"R@20: {metrics['R@20']:.4f}, MRR: {metrics['MRR']:.4f}")
                else:
                    print(f"Failed to get metrics for cap={cap}")
        
        if not results:
            print("No results collected. Exiting.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(results_csv, index=False)
        print(f"\nResults saved to: {results_csv}")
    
    # Create figure
    set_paper_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
    
    # Prepare data for plotting
    pool_caps_plot = [128, 256, 512, 1024]
    pool_caps_labels = ["128", "256", "512", "1024"]
    
    # Get "all" row
    all_rows = df[df["pool_cap"] == "all"]
    has_all = len(all_rows) > 0
    if has_all:
        all_row = all_rows.iloc[0]
        pool_caps_plot.append(2000)  # Use 2000 as placeholder for "all"
        pool_caps_labels.append("all")
    
    # Collect all values
    r1_vals = []
    r5_vals = []
    r20_vals = []
    mrr_vals = []
    
    for cap in [128, 256, 512, 1024]:
        # Convert cap to string for comparison since CSV stores as string
        row = df[df["pool_cap"].astype(str) == str(cap)]
        if len(row) > 0:
            r1_vals.append(row.iloc[0]["R@1"])
            r5_vals.append(row.iloc[0]["R@5"])
            r20_vals.append(row.iloc[0]["R@20"])
            mrr_vals.append(row.iloc[0]["MRR"])
        else:
            r1_vals.append(0.0)
            r5_vals.append(0.0)
            r20_vals.append(0.0)
            mrr_vals.append(0.0)
    
    if has_all:
        r1_vals.append(all_row["R@1"])
        r5_vals.append(all_row["R@5"])
        r20_vals.append(all_row["R@20"])
        mrr_vals.append(all_row["MRR"])
    
    # Use better spacing to avoid overlap - increase gaps between points
    # Map to better spaced positions with larger gaps
    pool_caps_numeric = []
    for c in pool_caps_plot:
        if isinstance(c, (int, float)):
            if c == 128:
                pool_caps_numeric.append(0)  # Start at 0
            elif c == 256:
                pool_caps_numeric.append(300)  # Larger gap from 128
            elif c == 512:
                pool_caps_numeric.append(700)  # Larger gap from 256
            elif c == 1024:
                pool_caps_numeric.append(1500)  # Larger gap from 512
            else:
                pool_caps_numeric.append(c)
        else:
            pool_caps_numeric.append(2500)  # Larger gap from 1024
    
    # Plot all metrics on the same axes
    ax.plot(pool_caps_numeric, r1_vals, marker='o', linewidth=2.5, markersize=10, label='R@1', color=palette[0], alpha=0.9)
    ax.plot(pool_caps_numeric, r5_vals, marker='s', linewidth=2.5, markersize=10, label='R@5', color=palette[1], alpha=0.9)
    ax.plot(pool_caps_numeric, r20_vals, marker='^', linewidth=2.5, markersize=10, label='R@20', color=palette[2], alpha=0.9)
    ax.plot(pool_caps_numeric, mrr_vals, marker='d', linewidth=2.5, markersize=10, label='MRR', color=palette[3], alpha=0.9)
    ax.set_xlabel('Candidate pool size cap', fontsize=16)
    ax.set_ylabel('Metric value', fontsize=16)
    ax.set_title('Recall@K and MRR vs. candidate pool size', fontsize=18, pad=10)
    ax.set_xticks(pool_caps_numeric)
    ax.set_xticklabels(pool_caps_labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(frameon=False, loc='best', fontsize=14)
    
    plt.tight_layout()
    
    output_path = os.path.join(args.output_dir, "candidate_pool_size_sensitivity.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
