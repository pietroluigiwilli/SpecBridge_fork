#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute UMAP visualizations by computing embeddings directly.
Computes molecular embeddings and mapped embeddings, then generates UMAP plots.
"""

from __future__ import annotations
import os
import argparse
import warnings
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional deps
_HAS_UMAP = False
try:
    import umap
    _HAS_UMAP = True
except Exception:
    pass

_HAS_TSNE = False
try:
    from sklearn.manifold import TSNE
    _HAS_TSNE = True
except Exception:
    pass

_HAS_PCA = False
try:
    from sklearn.decomposition import PCA
    _HAS_PCA = True
except Exception:
    pass

# ==== SpecBridge imports ====
from specbridge.utils.common import set_seed
from specbridge.eval.candidates import (
    build_model, build_mol_embed_fn, _canon_smi
)
from specbridge.data.massspecgym import MassSpecGymDataset, collate_massspecgym as base_collate

# ====================== Paper style & palette ======================
BLUE   = "#2D6CDF"  # mapped/correct
ORANGE = "#FF7F0E"  # molecules/incorrect

def set_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "grid.linestyle": ":", "grid.alpha": 0.30,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

def savefig(outdir: str, name: str, transparent: bool=False):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    plt.savefig(base + ".pdf", transparent=transparent)
    plt.savefig(base + ".png", dpi=300, transparent=transparent)
    plt.close()

def _l2norm(t: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return t / (t.norm(dim=dim, keepdim=True).clamp_min(eps))

def _l2norm_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def _plot_reduction_simple(X, title, fname, color=None, outdir="figs", method="umap"):
    """Simple dimensionality reduction visualization (UMAP, t-SNE, or PCA)."""
    if len(X) < 5:
        print(f"[Warning] Not enough samples ({len(X)}) for {method.upper()}")
        return False
    
    # Filter out None values if present
    valid_mask = np.array([x is not None and (not isinstance(x, np.ndarray) or x.size > 0) for x in X])
    if not np.any(valid_mask):
        print(f"[Warning] No valid embeddings found")
        return False
    
    X_valid = [X[i] for i in range(len(X)) if valid_mask[i]]
    
    # Convert to numpy array and normalize
    X_arr = np.asarray(X_valid)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    Xn = _l2norm_np(X_arr)
    
    # Compute reduction
    if method.lower() == "umap":
        if not _HAS_UMAP:
            print(f"[Error] UMAP library not available. Install with: pip install umap-learn")
            return False
        print(f"[{method.upper()}] Computing {method.upper()} for {len(Xn)} embeddings...")
        reducer = umap.UMAP(n_neighbors=60, min_dist=0.10, metric="cosine", random_state=1234)
        Y = reducer.fit_transform(Xn)
    elif method.lower() == "tsne":
        if not _HAS_TSNE:
            print(f"[Error] t-SNE library not available. Install with: pip install scikit-learn")
            return False
        print(f"[{method.upper()}] Computing {method.upper()} for {len(Xn)} embeddings...")
        reducer = TSNE(n_components=2, random_state=1234, perplexity=30, metric="cosine", init="pca")
        Y = reducer.fit_transform(Xn)
    elif method.lower() == "pca":
        if not _HAS_PCA:
            print(f"[Error] PCA library not available. Install with: pip install scikit-learn")
            return False
        print(f"[{method.upper()}] Computing {method.upper()} for {len(Xn)} embeddings...")
        reducer = PCA(n_components=2, random_state=1234)
        Y = reducer.fit_transform(Xn)
        # Add variance explained to title
        var_explained = reducer.explained_variance_ratio_
        title = f"{title}\n(Variance explained: {var_explained[0]:.1%}, {var_explained[1]:.1%})"
    else:
        print(f"[Error] Unknown method: {method}")
        return False

    # Simple scatter plot with single color
    plt.figure(figsize=(4, 4), dpi=300)
    plot_color = color if color else BLUE
    plt.scatter(Y[:,0], Y[:,1], s=1.5, c=plot_color, alpha=0.6,
                edgecolors="none", linewidths=0, rasterized=False)
    plt.title(title, fontsize=9)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout(pad=0.2)
    savefig(outdir, fname, transparent=True)
    print(f"[{method.upper()}] Saved: {os.path.join(outdir, fname)}.{{png,pdf}}")
    return True

def _plot_reduction_combined(mapped_embeds, mol_embeds, outdir="figs", method="umap"):
    """Create combined reduction plot to compare mapped vs molecular in same space."""
    # Filter valid embeddings
    mapped_valid = [x for x in mapped_embeds if x is not None and (not isinstance(x, np.ndarray) or x.size > 0)]
    mol_valid = [x for x in mol_embeds if x is not None and (not isinstance(x, np.ndarray) or x.size > 0)]
    
    if len(mapped_valid) < 5 or len(mol_valid) < 5:
        print(f"[Warning] Not enough valid embeddings for combined {method.upper()} (mapped={len(mapped_valid)}, mol={len(mol_valid)})")
        return False
    
    # Combine and normalize
    combined_embeds = mapped_valid + mol_valid
    combined_Xn = _l2norm_np(np.asarray(combined_embeds))
    
    # Compute reduction
    if method.lower() == "umap":
        if not _HAS_UMAP:
            return False
        print(f"[{method.upper()}] Computing combined {method.upper()} for {len(combined_Xn)} embeddings...")
        combined_reducer = umap.UMAP(n_neighbors=60, min_dist=0.10, metric="cosine", random_state=1234)
        combined_Y = combined_reducer.fit_transform(combined_Xn)
        title = "Mapped vs Molecular embeddings (same UMAP space)"
    elif method.lower() == "tsne":
        if not _HAS_TSNE:
            return False
        print(f"[{method.upper()}] Computing combined {method.upper()} for {len(combined_Xn)} embeddings...")
        combined_reducer = TSNE(n_components=2, random_state=1234, perplexity=30, metric="cosine", init="pca")
        combined_Y = combined_reducer.fit_transform(combined_Xn)
        title = "Mapped vs Molecular embeddings (same t-SNE space)"
    elif method.lower() == "pca":
        if not _HAS_PCA:
            return False
        print(f"[{method.upper()}] Computing combined {method.upper()} for {len(combined_Xn)} embeddings...")
        combined_reducer = PCA(n_components=2, random_state=1234)
        combined_Y = combined_reducer.fit_transform(combined_Xn)
        var_explained = combined_reducer.explained_variance_ratio_
        title = f"Mapped vs Molecular embeddings (same PCA space)\n(Variance: {var_explained[0]:.1%}, {var_explained[1]:.1%})"
    else:
        return False
    
    # Split back into mapped and molecular
    n = len(mapped_valid)
    mapped_Y = combined_Y[:n]
    mol_Y = combined_Y[n:]
    
    # Plot
    plt.figure(figsize=(4, 4), dpi=300)
    plt.scatter(mapped_Y[:,0], mapped_Y[:,1], s=1.5, c=BLUE, alpha=0.6,
                edgecolors="none", linewidths=0, label="Mapped ($\\hat z$)", rasterized=False, marker="o")
    plt.scatter(mol_Y[:,0], mol_Y[:,1], s=1.5, c=ORANGE, alpha=0.6,
                edgecolors="none", linewidths=0, label="Molecular ($z_m$)", rasterized=False, marker="^")
    plt.legend(frameon=False, loc="best", markerscale=2.0, fontsize=7)
    plt.title(title, fontsize=9)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout(pad=0.2)
    savefig(outdir, f"{method}_mapped_vs_mol_combined", transparent=True)
    print(f"[{method.upper()}] Saved: {os.path.join(outdir, method + '_mapped_vs_mol_combined')}.{{png,pdf}}")
    
    # Compute cosine similarity statistics
    mapped_np = _l2norm_np(np.asarray(mapped_valid))
    mol_np = _l2norm_np(np.asarray(mol_valid))
    min_len = min(len(mapped_np), len(mol_np))
    cosines = np.array([np.dot(mapped_np[i], mol_np[i]) for i in range(min_len)])
    print(f"\n=== Mapped vs Molecular Embedding Alignment ===")
    print(f"Mean cosine similarity: {np.mean(cosines):.4f}")
    print(f"Median cosine similarity: {np.median(cosines):.4f}")
    print(f"Min/Max: {np.min(cosines):.4f} / {np.max(cosines):.4f}")
    print(f"Std: {np.std(cosines):.4f}")
    return True

def _collate_with_smiles(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed):
    out = base_collate(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed=seed)
    out["titles"] = [ex["title"] for ex in base_batch]
    out["smiles_true"] = [ex["smiles"] for ex in base_batch]
    return out

def main():
    ap = argparse.ArgumentParser("Compute UMAP visualizations by computing embeddings directly")
    # Data & model (same as run_umap_analysis.sh)
    ap.add_argument("--mgf", required=True)
    ap.add_argument("--meta-json", type=str, default=None)
    ap.add_argument("--adapter-ckpt", type=str, default=None)
    ap.add_argument("--dreams-ckpt", type=str, default=None)
    ap.add_argument("--fold-query", type=str, default=None)
    # Inference/space
    ap.add_argument("--use-mapped", action="store_true")
    ap.add_argument("--deterministic-map", action="store_true")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=2048)
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--mapper-hidden", type=int, default=2048)
    ap.add_argument("--no-gaussian", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mol-space", choices=["adapter", "ecfp", "chemberta"], default="chemberta")
    ap.add_argument("--ecfp-radius", type=int, default=2)
    ap.add_argument("--chemberta-model", type=str, default="Derify/ChemBERTa_augmented_pubchem_13m")
    # Output
    ap.add_argument("--outdir", type=str, default="figs", help="Output directory for figures")
    ap.add_argument("--plot-mapped", action="store_true", help="Plot mapped embeddings")
    ap.add_argument("--plot-mol", action="store_true", help="Plot molecular embeddings")
    ap.add_argument("--plot-combined", action="store_true", help="Plot combined mapped vs molecular")
    ap.add_argument("--plot-all", action="store_true", help="Plot all available visualizations")
    ap.add_argument("--methods", type=str, default="umap,tsne,pca", help="Comma-separated list of methods: umap,tsne,pca")
    args = ap.parse_args()
    
    if not _HAS_UMAP:
        print("[Error] UMAP library not available. Install with: pip install umap-learn")
        return
    
    set_paper_style()
    outfig = os.path.join(args.outdir, "figs")
    os.makedirs(outfig, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    
    # Dataset / folds
    folds = None
    if args.fold_query:
        folds = {f.strip().lower() for f in args.fold_query.split(",") if f.strip()}
    ds = MassSpecGymDataset(args.mgf, args.meta_json, folds=folds)
    
    def _collate(b):
        return _collate_with_smiles(
            b, args.spec_bins,
            max(2, getattr(ds, "_formula_vocab", 0) or 32),
            max(2, getattr(ds, "_adduct_vocab", 0) or 16),
            max(2, getattr(ds, "_charge_vocab", 0) or 8),
            args.fp_bits, seed=args.seed
        )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=_collate)
    
    # Build model + embed fn
    print("[Model] Building model and molecular encoder...")
    model = build_model(args, device)
    embed_fn = build_mol_embed_fn(args, model, device)
    
    # Accumulators for embeddings
    dreams_embeds: List[np.ndarray] = []
    mapped_embeds: List[np.ndarray] = []
    mol_embeds: List[np.ndarray] = []
    
    print("[Compute] Computing embeddings...")
    progress = tqdm(dl, desc="[Computing embeddings]")
    for batch in progress:
        s = batch["spectra"].to(device)
        meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["meta"].items()}
        smiles_true = batch["smiles_true"]
        
        with torch.no_grad():
            z_s, z_m, z_hat, mu, lv = model(s, meta, None, inference=False)
        
        # Choose query embedding (mapped)
        if args.use_mapped:
            if args.deterministic_map or (lv is None):
                z_query = mu
            elif args.samples > 1:
                zq = [model.mapB.sample(mu, lv, deterministic=False) for _ in range(args.samples)]
                z_query = torch.stack(zq, dim=0)  # [K,B,D]
            else:
                z_query = model.mapB.sample(mu, lv, deterministic=False)
        else:
            z_query = F.normalize(z_s, dim=-1)
        
        B = s.size(0)
        for i in range(B):
            # Store embeddings
            z_s_i = _l2norm(z_s[i].detach().cpu(), dim=-1).numpy()
            if z_query.dim() == 3:
                zqi = _l2norm(z_query[:, i, :], dim=-1).mean(0)  # average K samples
            else:
                zqi = _l2norm(z_query[i], dim=-1)
            
            dreams_embeds.append(z_s_i)
            mapped_embeds.append(zqi.detach().cpu().numpy())
            
            # Store molecular embedding from model forward pass (z_m)
            z_m_i = _l2norm(z_m[i].detach().cpu(), dim=-1).numpy()
            mol_embeds.append(z_m_i)
    
    print(f"[Compute] Computed {len(dreams_embeds)} embeddings")
    
    # Determine what to plot
    plot_mapped = args.plot_mapped or args.plot_all
    plot_mol = args.plot_mol or args.plot_all
    plot_combined = args.plot_combined or args.plot_all
    
    # If no specific plot requested, plot all available
    if not (plot_mapped or plot_mol or plot_combined):
        plot_mapped = True
        plot_mol = True
        plot_combined = True
    
    # Parse methods
    methods = [m.strip().lower() for m in args.methods.split(",")]
    valid_methods = []
    for method in methods:
        if method == "umap" and _HAS_UMAP:
            valid_methods.append("umap")
        elif method == "tsne" and _HAS_TSNE:
            valid_methods.append("tsne")
        elif method == "pca" and _HAS_PCA:
            valid_methods.append("pca")
        elif method not in ["umap", "tsne", "pca"]:
            print(f"[Warning] Unknown method: {method}, skipping")
    
    if not valid_methods:
        print("[Error] No valid reduction methods available. Install required packages:")
        if "umap" in methods and not _HAS_UMAP:
            print("  - pip install umap-learn")
        if "tsne" in methods and not _HAS_TSNE:
            print("  - pip install scikit-learn")
        if "pca" in methods and not _HAS_PCA:
            print("  - pip install scikit-learn")
        return
    
    # Filter to common valid indices
    if mapped_embeds and mol_embeds:
        mapped_valid = np.array([x is not None and (not isinstance(x, np.ndarray) or x.size > 0) for x in mapped_embeds])
        mol_valid = np.array([x is not None and (not isinstance(x, np.ndarray) or x.size > 0) for x in mol_embeds])
        min_len = min(len(mapped_valid), len(mol_valid))
        mapped_valid = mapped_valid[:min_len]
        mol_valid = mol_valid[:min_len]
        common_valid = mapped_valid & mol_valid
        
        if np.sum(common_valid) >= 5:
            mapped_embeds = [mapped_embeds[i] for i in range(min_len) if common_valid[i]]
            mol_embeds = [mol_embeds[i] for i in range(min_len) if common_valid[i]]
            print(f"[Filter] Using {len(mapped_embeds)} common valid embeddings")
    
    # Generate plots for each method
    print(f"\n[Plot] Generating visualizations using methods: {', '.join(valid_methods)}")
    for method in valid_methods:
        if plot_mapped and mapped_embeds:
            _plot_reduction_simple(mapped_embeds, f"Mapped spectra embeddings ($\\hat z$)", 
                                 f"{method}_mapped", color=BLUE, outdir=outfig, method=method)
        
        if plot_mol and mol_embeds:
            _plot_reduction_simple(mol_embeds, f"Molecular embeddings from model ($z_m$)", 
                                 f"{method}_mol", color=ORANGE, outdir=outfig, method=method)
        
        if plot_combined and mapped_embeds and mol_embeds:
            _plot_reduction_combined(mapped_embeds, mol_embeds, outdir=outfig, method=method)
    
    print(f"\nDone! Visualizations saved to {outfig}/")
    for method in valid_methods:
        print(f"  [{method.upper()}]")
        if plot_mapped:
            print(f"    - {method}_mapped.{{png,pdf}}")
        if plot_mol:
            print(f"    - {method}_mol.{{png,pdf}}")
        if plot_combined:
            print(f"    - {method}_mapped_vs_mol_combined.{{png,pdf}}")

if __name__ == "__main__":
    main()
