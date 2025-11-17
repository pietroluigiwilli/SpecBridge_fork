#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpecBridge analysis (blue/orange paper theme)

New in this version
-------------------
A) Separate UMAPs:
   - figs/umap_dreams_classes.{png,pdf}
   - figs/umap_mapped_classes.{png,pdf}
   (colored by ClassyFire class; legend shows top K classes, others = "Other")

B) Molecule inset images are OFF by default and, if enabled, render with
   transparent backgrounds (RDKit). Control with --umap-insets N.

C) Pairwise distance evaluation (same-class vs different-class) in both spaces:
   - figs/dist_pairs_same_vs_diff_dreams.{png,pdf}
   - figs/dist_pairs_same_vs_diff_mapped.{png,pdf}
   - stats to dist_eval.json (means/medians, KS/MWU, Cohen d, Cliff delta, AUC)

What else is included (unchanged from your prior version)
---------------------------------------------------------
• Per-query structure–geometry analysis: Pearson/Spearman, SA@k, Analog AUROC/AP
• Histograms of cosine-to-true / margin / angle
• Caching support
"""

from __future__ import annotations
import os, json, pickle, math, argparse, warnings, hashlib, itertools
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Optional deps ----------
_HAS_RDKIT = False
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Draw
    _HAS_RDKIT = True
except Exception:
    pass

_HAS_UMAP = False
try:
    import umap
    _HAS_UMAP = True
except Exception:
    pass

_HAS_SCIPY = False
try:
    from scipy.stats import ks_2samp, mannwhitneyu
    _HAS_SCIPY = True
except Exception:
    pass

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA

# ==== SpecBridge bits (adjust to your repo layout if needed) ====
from specbridge.utils.common import set_seed
from specbridge.eval.candidates import (
    build_model, build_mol_embed_fn, _canon_smi, _metrics_from_ranks
)
from specbridge.data.massspecgym import MassSpecGymDataset, collate_massspecgym as base_collate


# ====================== Paper style & palette ======================
BLUE   = "#2D6CDF"  # mapped/correct
ORANGE = "#FF7F0E"  # molecules/incorrect
GREEN  = "#2CA02C"
RED    = "#D62728"
PURPLE = "#9467BD"
BROWN  = "#8C564B"
PINK   = "#E377C2"
GREY   = "#7f7f7f"

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


# ====================== Utilities ======================
def _l2norm(t: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return t / (t.norm(dim=dim, keepdim=True).clamp_min(eps))

def _l2norm_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def ecfp4(smiles: str, radius: int = 2, nbits: int = 4096):
    if not _HAS_RDKIT or not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)

def tani(fp1, fp2, default: float = np.nan) -> float:
    if fp1 is None or fp2 is None:
        return default
    return float(DataStructs.TanimotoSimilarity(fp1, fp2))

def angle_from_cos(c: float) -> float:
    c = max(-1.0, min(1.0, float(c)))
    return float(math.degrees(math.acos(c)))

def pearson_spearman(x: np.ndarray, y: np.ndarray):
    from scipy.stats import pearsonr, spearmanr
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return (np.nan, np.nan)
    return (float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0]))

def auroc_ap(scores: np.ndarray, labels: np.ndarray):
    if len(np.unique(labels)) < 2:
        return (np.nan, np.nan)
    try:
        return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))
    except Exception:
        return (np.nan, np.nan)

def bootstrap_ci(arr: List[float], iters: int = 1000, seed: int = 1234, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    clean = np.array([a for a in arr if not np.isnan(a)], dtype=float)
    if clean.size == 0:
        return (np.nan, np.nan, np.nan)
    stats = []
    for _ in range(iters):
        s = rng.choice(clean, size=clean.size, replace=True)
        stats.append(np.nanmean(s))
    lo = np.quantile(stats, alpha/2)
    hi = np.quantile(stats, 1 - alpha/2)
    return (float(np.mean(clean)), float(lo), float(hi))


# ====================== Cache helpers ======================
def _compute_cache_key(args) -> str:
    key_parts = {
        "mgf": args.mgf, "meta_json": args.meta_json or "", "candidates": args.candidates,
        "dreams_ckpt": args.dreams_ckpt or "", "adapter_ckpt": args.adapter_ckpt or "",
        "fold_query": args.fold_query or "", "use_mapped": args.use_mapped,
        "deterministic_map": args.deterministic_map, "samples": args.samples,
        "aggregate": args.aggregate, "spec_bins": args.spec_bins, "fp_bits": args.fp_bits,
        "cond_dim": args.cond_dim, "mapper_hidden": args.mapper_hidden,
        "no_gaussian": args.no_gaussian, "seed": args.seed,
        "mol_space": args.mol_space, "ecfp_radius": args.ecfp_radius,
        "chemberta_model": args.chemberta_model,
        "classyfire": args.classyfire or "", "class_field": args.class_field,
    }
    key_str = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]

def _save_cache(cache_dir: str, cache_key: str, cand_z, true_z, recs,
                umap_pairs, umap_ok, umap_adducts, umap_charges, umap_margins,
                dreams_embeds, mapped_embeds, mol_embeds, labels):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"cache_{cache_key}.pkl")
    payload = dict(
        cand_z={k: v.cpu().numpy() for k,v in cand_z.items()},
        true_z={k: v.cpu().numpy() for k,v in true_z.items()},
        recs=recs, umap_pairs=umap_pairs, umap_ok=umap_ok,
        umap_adducts=umap_adducts, umap_charges=umap_charges, umap_margins=umap_margins,
        dreams_embeds=np.asarray(dreams_embeds), mapped_embeds=np.asarray(mapped_embeds),
        mol_embeds=np.asarray(mol_embeds) if mol_embeds and len(mol_embeds) > 0 else None,
        labels=labels,
    )
    with open(cache_file, "wb") as f:
        pickle.dump(payload, f)
    print(f"[Cache] Saved {cache_file}")

def _load_cache(cache_dir: str, cache_key: str):
    path = os.path.join(cache_dir, f"cache_{cache_key}.pkl")
    if not os.path.exists(path): return None
    try:
        with open(path, "rb") as f:
            X = pickle.load(f)
        cand_z = {k: torch.from_numpy(v) for k,v in X["cand_z"].items()}
        true_z = {k: torch.from_numpy(v) for k,v in X["true_z"].items()}
        dreams_embeds = X.get("dreams_embeds", None)
        mapped_embeds = X.get("mapped_embeds", None)
        mol_embeds = X.get("mol_embeds", None)
        labels = X.get("labels", None)
        
        # Convert numpy arrays to lists for consistency
        if dreams_embeds is not None and isinstance(dreams_embeds, np.ndarray):
            dreams_embeds = dreams_embeds.tolist()
        if mapped_embeds is not None and isinstance(mapped_embeds, np.ndarray):
            mapped_embeds = mapped_embeds.tolist()
        if mol_embeds is not None and isinstance(mol_embeds, np.ndarray):
            mol_embeds = mol_embeds.tolist()
        
        return (cand_z, true_z, X["recs"], X["umap_pairs"], X["umap_ok"],
                X.get("umap_adducts", []), X.get("umap_charges", []), X.get("umap_margins", []),
                dreams_embeds, mapped_embeds, mol_embeds, labels)
    except Exception as e:
        warnings.warn(f"[Cache] Failed to load: {e}")
        return None


# ====================== Dataclass ======================
@dataclass
class QueryRecord:
    title: str
    true_smiles: str
    n_cands: int
    rank_true: Optional[int]
    cos_true: Optional[float]
    margin: Optional[float]
    angle_deg: Optional[float]
    pearson: Optional[float]
    spearman: Optional[float]
    sa5: Optional[float]
    sa10: Optional[float]
    sa20: Optional[float]
    auroc_tani05: Optional[float]
    ap_tani05: Optional[float]
    auroc_tani06: Optional[float]
    ap_tani06: Optional[float]
    auroc_tani07: Optional[float]
    ap_tani07: Optional[float]


# ====================== Pair sampling & stats ======================
def _sample_pairs(labels: List[str], max_pairs_same=200000, max_pairs_diff=200000, rng=None):
    """
    Return two lists of index pairs: same-class and different-class.
    Balanced across classes (best-effort).
    """
    rng = rng or np.random.default_rng(1234)
    by_cls: Dict[str, List[int]] = {}
    for i, c in enumerate(labels):
        by_cls.setdefault(c, []).append(i)

    # same-class
    same = []
    each_target = max(1, max_pairs_same // max(1, len(by_cls)))
    for c, idxs in by_cls.items():
        if len(idxs) < 2: continue
        m = min(each_target, len(idxs)*(len(idxs)-1)//2)
        for _ in range(m):
            i, j = rng.choice(idxs, size=2, replace=False)
            same.append((i,j))
    if len(same) > max_pairs_same:
        same = [same[k] for k in rng.choice(len(same), size=max_pairs_same, replace=False)]

    # different-class
    all_idx = np.arange(len(labels))
    diff = []
    tries = 0
    need = max_pairs_diff
    while len(diff) < need and tries < 5*need:
        i, j = rng.choice(all_idx, size=2, replace=False)
        if labels[i] != labels[j]:
            diff.append((i,j))
        tries += 1
    return same, diff

def _cosine_distance(X: np.ndarray, pairs: List[Tuple[int,int]]):
    # X should be L2-normalized rows
    dists = []
    for i, j in pairs:
        c = float(np.dot(X[i], X[j]))
        dists.append(1.0 - c)
    return np.array(dists, dtype=float)

def _effect_sizes(a: np.ndarray, b: np.ndarray):
    """Return dict of effect size metrics comparing a (same) vs b (diff)."""
    out = {}
    # means/medians
    out["mean_same"] = float(np.nanmean(a))
    out["mean_diff"] = float(np.nanmean(b))
    out["median_same"] = float(np.nanmedian(a))
    out["median_diff"] = float(np.nanmedian(b))
    # cohen d
    va, vb = np.nanvar(a, ddof=1), np.nanvar(b, ddof=1)
    sp = np.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(1,(len(a)+len(b)-2)))
    out["cohen_d"] = float((np.nanmean(b) - np.nanmean(a)) / (sp + 1e-12))
    # Cliff's delta
    # δ ≈ 2*AUC - 1 when scores define ordering. We will compute AUC below and map.
    # KS / MWU / AUC (if SciPy/Sklearn available)
    if _HAS_SCIPY:
        ks = ks_2samp(a, b, alternative="two-sided", mode="auto")
        out["ks_stat"] = float(ks.statistic); out["ks_p"] = float(ks.pvalue)
        mwu = mannwhitneyu(a, b, alternative="two-sided")
        out["mwu_U"] = float(mwu.statistic); out["mwu_p"] = float(mwu.pvalue)
    try:
        # smaller distance => "same". Convert to a score where higher => "same".
        y = np.concatenate([np.ones_like(a), np.zeros_like(b)])
        s = np.concatenate([1.0 - a, 1.0 - b])  # larger = more likely same-class
        auc = roc_auc_score(y, s)
        out["auc_same_vs_diff"] = float(auc)
        out["cliffs_delta"] = float(2*auc - 1.0)
    except Exception:
        out["auc_same_vs_diff"] = float("nan")
        out["cliffs_delta"] = float("nan")
    return out


# ====================== Main ======================
def main():
    ap = argparse.ArgumentParser("SpecBridge analysis (blue/orange)")
    # Data & model
    ap.add_argument("--mgf", required=True)
    ap.add_argument("--meta-json", type=str, default=None)
    ap.add_argument("--candidates", required=True)  # .pkl or .json {true: [cand,...]}
    ap.add_argument("--dreams-ckpt", type=str, default=None)
    ap.add_argument("--adapter-ckpt", type=str, default=None)
    ap.add_argument("--fold-query", type=str, default=None)
    # Inference/space
    ap.add_argument("--use-mapped", action="store_true")
    ap.add_argument("--deterministic-map", action="store_true")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"])
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=2048)
    ap.add_argument("--mapper-hidden", type=int, default=2048)
    ap.add_argument("--no-gaussian", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--mol-space", choices=["adapter", "ecfp", "chemberta"], default="chemberta")
    ap.add_argument("--ecfp-radius", type=int, default=2)
    ap.add_argument("--chemberta-model", type=str, default="Derify/ChemBERTa_augmented_pubchem_13m")
    # Output / viz
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--umap", action="store_true")
    ap.add_argument("--max-arrows", type=int, default=2000)
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--force-recompute", action="store_true")
    # ClassyFire labels for class-colored plots & distance eval
    ap.add_argument("--classyfire", type=str, default=None, help="JSON mapping SMILES -> ClassyFire dict")
    ap.add_argument("--class-field", type=str, default="Class", help="Which key to use as label (e.g., Class, Superclass, Direct_parent)")
    ap.add_argument("--umap-topk-classes", type=int, default=6, help="only show legend for top-K frequent classes")
    ap.add_argument("--umap-insets", type=int, default=0, help="number of molecule insets to draw (0 = off). Insets have transparent background.")
    ap.add_argument("--pair-max-same", type=int, default=200000)
    ap.add_argument("--pair-max-diff", type=int, default=200000)
    args = ap.parse_args()

    set_paper_style()
    outfig = os.path.join(args.outdir, "figs")
    os.makedirs(outfig, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # Cache
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(args.outdir, "cache")
    cache_key = _compute_cache_key(args) if not args.no_cache else None

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

    # Candidate map
    if args.candidates.lower().endswith(".pkl"):
        with open(args.candidates, "rb") as f:
            cand_map_raw = pickle.load(f)
    else:
        with open(args.candidates, "r") as f:
            cand_map_raw = json.load(f)
    cand_map: Dict[str, List[str]] = {}
    for k, vs in cand_map_raw.items():
        ck = _canon_smi(k) or (k.strip() if k else k)
        vals, seen = [], set()
        for v in vs:
            if not v: continue
            cv = _canon_smi(v) or v.strip()
            if cv not in seen:
                vals.append(cv); seen.add(cv)
        if ck and ck not in seen:
            vals.append(ck)
        cand_map[ck] = vals

    # ClassyFire mapping (optional but recommended)
    class_map: Dict[str, str] = {}
    if args.classyfire and os.path.exists(args.classyfire):
        with open(args.classyfire, "r") as f:
            cf = json.load(f)
        for smi, dd in cf.items():
            lab = dd.get(args.class_field, None)
            if lab is None or str(lab).strip() == "":
                lab = dd.get("Direct_parent", None) or dd.get("Superclass", None) or "Other"
            class_map[_canon_smi(smi) or smi.strip()] = str(lab)
        print(f"[ClassyFire] Loaded {len(class_map)} labels (field: {args.class_field})")

    # Try cache
    cached = None
    if cache_key and not args.force_recompute and not args.no_cache:
        cached = _load_cache(cache_dir, cache_key)

    # Initialize cosine similarity accumulators (used for histogram)
    cos_target: List[float] = []
    cos_candidates_mean: List[float] = []
    
    if cached is not None:
        (cand_z, true_z, recs, umap_pairs, umap_ok,
         umap_adducts, umap_charges, umap_margins,
         dreams_embeds, mapped_embeds, mol_embeds, labels) = cached
        print("[Cache] Using cached results")
        # Note: cos_target and cos_candidates_mean not available from cache, will be empty
    else:
        # Build model + embed fn
        model = build_model(args, device)
        embed_fn = build_mol_embed_fn(args, model, device)

        # Collect all true smiles to know what we need
        all_true = []
        for batch in dl:
            all_true.extend(batch["smiles_true"])

        needed = set()
        for smi in all_true:
            csmi = _canon_smi(smi) or (smi.strip() if smi else smi)
            if csmi in cand_map:
                needed.update(cand_map[csmi])
        need_smiles = sorted(list(needed))

        # Precompute candidate embeddings
        cand_z: Dict[str, torch.Tensor] = {}
        if need_smiles:
            with torch.no_grad():
                Z = embed_fn(need_smiles, device)
                for s, z in zip(need_smiles, Z):
                    cand_z[_canon_smi(s) or s.strip()] = z.cpu()

        # Precompute true embeddings
        uniq_true = sorted({(_canon_smi(s) or s.strip()) for s in all_true if s})
        true_z = {}
        with torch.no_grad():
            if len(uniq_true):
                Zt = embed_fn(uniq_true, device)
                true_z = {s: z.cpu() for s, z in zip(uniq_true, Zt)}

        # Accumulators
        recs: List[QueryRecord] = []
        umap_pairs: List[np.ndarray] = []
        umap_ok: List[int] = []
        umap_adducts: List[str] = []
        umap_charges: List[int] = []
        umap_margins: List[float] = []
        
        # Initialize cosine similarity accumulators (will be populated in loop)
        cos_target = []
        cos_candidates_mean = []

        # For distance evaluation
        dreams_embeds: List[np.ndarray] = []
        mapped_embeds: List[np.ndarray] = []
        mol_embeds: List[np.ndarray] = []
        labels: List[str] = []

        progress = tqdm(dl, desc="[SpecBridge analysis]")
        for batch in progress:
            s = batch["spectra"].to(device)
            meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["meta"].items()}
            smiles_true = batch["smiles_true"]
            titles = batch["titles"]

            with torch.no_grad():
                z_s, z_m, z_hat, mu, lv = model(s, meta, None, inference=False)

            # choose query embedding
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
                true_raw = smiles_true[i]
                true_smi = _canon_smi(true_raw) or (true_raw.strip() if true_raw else true_raw)
                title = titles[i] if isinstance(titles, list) else str(titles[i])

                # store embeddings + labels for pairwise analysis
                z_s_i = _l2norm(z_s[i].detach().cpu(), dim=-1).numpy()
                if z_query.dim() == 3:
                    zqi = _l2norm(z_query[:, i, :], dim=-1).mean(0)  # average K samples for global stats
                else:
                    zqi = _l2norm(z_query[i], dim=-1)
                dreams_embeds.append(z_s_i)
                mapped_embeds.append(zqi.detach().cpu().numpy())
                # Store molecular embedding from model forward pass (z_m) - this is what mapped should align with
                z_m_i = _l2norm(z_m[i].detach().cpu(), dim=-1).numpy()
                mol_embeds.append(z_m_i)
                labels.append(class_map.get(true_smi, "Other"))

                # candidate list
                clist = cand_map.get(true_smi, None)
                if not clist:  # still record global stuff
                    continue

                kept, Zc = [], []
                for sm in clist:
                    csm = _canon_smi(sm) or sm.strip()
                    zi = cand_z.get(csm, None)
                    if zi is None: continue
                    Zc.append(zi.unsqueeze(0))
                    kept.append(csm)
                if not Zc:
                    continue
                Zc = torch.cat(Zc, dim=0)
                Zc = _l2norm(Zc, dim=-1).to(device)
                C  = Zc.size(0)

                # similarity for ranking
                if z_query.dim() == 3:
                    zqi_rank = _l2norm(z_query[:, i, :].to(device), dim=-1)  # [K,D]
                    sims = zqi_rank @ Zc.T
                    sims = sims.max(dim=0).values if args.aggregate == "max" else sims.mean(dim=0)
                else:
                    zqi_rank = _l2norm(z_query[i].to(device), dim=-1)        # [D]
                    sims = zqi_rank @ Zc.T                                   # [C]
                sims_np = sims.detach().float().cpu().numpy()
                order = torch.argsort(sims, descending=True)
                
                # Collect mean cosine similarity with all candidates
                cos_candidates_mean.append(float(np.mean(sims_np)))

                rank_true = None
                cos_true  = None
                margin    = None
                angle_deg = None
                correct   = 0

                # Compute cosine similarity with target molecule (for histogram)
                if true_smi in kept:
                    idx_true = kept.index(true_smi)
                    pos = (order == idx_true).nonzero(as_tuple=False)
                    if pos.numel() > 0:
                        rank_true = int(pos.item())
                        correct = int(rank_true == 0)
                    cos_true = float(sims[idx_true].item())
                    # Collect cosine similarity with target molecule
                    cos_target.append(cos_true)
                    if C >= 2:
                        mask = torch.ones(C, dtype=torch.bool, device=sims.device)
                        mask[idx_true] = False
                        best_non_true = float(sims[mask].max().item())
                        margin = float(cos_true - best_non_true)
                    angle_deg = angle_from_cos(cos_true)
                else:
                    # If target not in candidates, compute similarity directly with target embedding
                    if true_smi in true_z:
                        z_target = _l2norm(true_z[true_smi], dim=-1).to(device)
                        if z_query.dim() == 3:
                            zqi_target = _l2norm(z_query[:, i, :].to(device), dim=-1)
                            if args.aggregate == "max":
                                cos_target_val = float((zqi_target @ z_target).max().item())
                            else:
                                cos_target_val = float((zqi_target @ z_target).mean().item())
                        else:
                            zqi_target = _l2norm(z_query[i].to(device), dim=-1)
                            cos_target_val = float((zqi_target @ z_target).item())
                        cos_target.append(cos_target_val)
                    else:
                        # If target embedding not available, skip this query for histogram
                        # (but still process for other metrics)
                        pass

                    # UMAP stash (mapped vs true)
                    if args.umap and (true_smi in true_z):
                        # zqi_rank is [K,D] when z_query.dim()==3, else [D]
                        zhat = _l2norm(zqi_rank.mean(0) if z_query.dim() == 3 else zqi_rank, dim=-1)
                        zhat = zhat.detach().cpu().numpy()
                        ytru = _l2norm(true_z[true_smi], dim=-1).detach().cpu().numpy()
                        # Ensure both are 1D arrays with same shape
                        if zhat.ndim == 0:
                            zhat = zhat.reshape(1)
                        if ytru.ndim == 0:
                            ytru = ytru.reshape(1)
                        if zhat.shape != ytru.shape:
                            # If shapes still don't match, ensure both are flattened to 1D
                            zhat = zhat.flatten()
                            ytru = ytru.flatten()
                        umap_pairs.append(np.stack([zhat, ytru], axis=0))
                        umap_ok.append(correct)
                        umap_margins.append(margin if margin is not None else np.nan)
                        # meta
                        adduct = str(meta.get("adduct", "unknown"))
                        charge = int(meta.get("charge", 0)) if isinstance(meta.get("charge", 0), (int, np.integer)) else 0
                        umap_adducts.append(adduct); umap_charges.append(charge)

                # ECFP/Tanimoto in-candidate diagnostics
                pearson = spearman = sa5 = sa10 = sa20 = np.nan
                auroc_tani = {0.5: np.nan, 0.6: np.nan, 0.7: np.nan}
                ap_tani    = {0.5: np.nan, 0.6: np.nan, 0.7: np.nan}
                if _HAS_RDKIT and kept:
                    fp_true = ecfp4(true_smi)
                    fps = [ecfp4(sm) for sm in kept]
                    tanis = np.array([tani(fp_true, f) for f in fps], dtype=float)

                    pearson, spearman = pearson_spearman(sims_np, tanis)
                    order_np = np.argsort(-sims_np)
                    for k in (5, 10, 20):
                        topk = order_np[:min(k, C)]
                        val = float(np.nanmean(tanis[topk])) if len(topk) else np.nan
                        if k == 5:  sa5  = val
                        if k == 10: sa10 = val
                        if k == 20: sa20 = val
                    for tau in (0.5, 0.6, 0.7):
                        labels_bin = (tanis >= tau).astype(int)
                        au, ap = auroc_ap(sims_np, labels_bin)
                        auroc_tani[tau], ap_tani[tau] = au, ap

                recs.append(QueryRecord(
                    title=title, true_smiles=true_smi, n_cands=C,
                    rank_true=rank_true, cos_true=cos_true, margin=margin, angle_deg=angle_deg,
                    pearson=pearson, spearman=spearman,
                    sa5=sa5, sa10=sa10, sa20=sa20,
                    auroc_tani05=auroc_tani[0.5], ap_tani05=ap_tani[0.5],
                    auroc_tani06=auroc_tani[0.6], ap_tani06=ap_tani[0.6],
                    auroc_tani07=auroc_tani[0.7], ap_tani07=ap_tani[0.7],
                ))

        if cache_key and not args.no_cache:
            _save_cache(cache_dir, cache_key, cand_z, true_z, recs, umap_pairs, umap_ok,
                        umap_adducts, umap_charges, umap_margins,
                        dreams_embeds, mapped_embeds, mol_embeds, labels)

    # Save per-query CSV
    df = pd.DataFrame([asdict(r) for r in recs])
    os.makedirs(args.outdir, exist_ok=True)
    df.to_csv(os.path.join(args.outdir, "per_query_metrics.csv"), index=False)

    # Aggregates with bootstrap CIs
    def agg(col): return bootstrap_ci(df[col].astype(float).tolist())
    aggs = {
        "pearson": agg("pearson"), "spearman": agg("spearman"),
        "sa5": agg("sa5"), "sa10": agg("sa10"), "sa20": agg("sa20"),
        "auroc_tani05": agg("auroc_tani05"), "ap_tani05": agg("ap_tani05"),
        "auroc_tani06": agg("auroc_tani06"), "ap_tani06": agg("ap_tani06"),
        "auroc_tani07": agg("auroc_tani07"), "ap_tani07": agg("ap_tani07"),
        "cos_true": agg("cos_true"), "margin": agg("margin"), "angle_deg": agg("angle_deg"),
    }
    with open(os.path.join(args.outdir, "aggregates.json"), "w") as f:
        json.dump({k: {"mean": v[0], "ci_lo": v[1], "ci_hi": v[2]} for k, v in aggs.items()}, f, indent=2)

    # === Histograms (cos/angle/margin) ===
    for col, title in [("cos_true","Cosine to true"),
                       ("margin","Cosine margin (true - best non-true)"),
                       ("angle_deg","Angle to true (degrees)")]:
        vals = df[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna().values
        if vals.size:
            plt.figure(figsize=(3.35, 2.6), dpi=300)
            plt.hist(vals, bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.3, density=False)
            plt.xlabel(col); plt.ylabel("count"); plt.grid(axis="y")
            plt.title(title)
            savefig(outfig, f"hist_{col}")
    
    # === Cosine similarity distribution: query spectra vs target/candidate molecules ===
    # Ensure both lists have the same length (only use queries where we have both values)
    if cos_target and cos_candidates_mean:
        min_len = min(len(cos_target), len(cos_candidates_mean))
        cos_target_aligned = cos_target[:min_len]
        cos_candidates_mean_aligned = cos_candidates_mean[:min_len]
        
        if len(cos_target_aligned) > 0:
            plt.figure(figsize=(3.35, 2.6), dpi=300)
            bins = np.linspace(min(min(cos_target_aligned), min(cos_candidates_mean_aligned)) - 0.1, 
                              max(max(cos_target_aligned), max(cos_candidates_mean_aligned)) + 0.1, 50)
            plt.hist(cos_target_aligned, bins=bins, alpha=0.7, color=BLUE, 
                    edgecolor="white", linewidth=0.3, label="Spectrum and target molecules", density=False)
            plt.hist(cos_candidates_mean_aligned, bins=bins, alpha=0.7, color=ORANGE, 
                    edgecolor="white", linewidth=0.3, label="Mean cosine sim - spectrum and candidates", density=False)
            plt.xlabel("Cosine similarity")
            plt.ylabel("Counts")
            plt.legend(frameon=False, loc="best", fontsize=7)
            plt.grid(axis="y", alpha=0.3)
            plt.title("Distribution of cosine similarities")
            savefig(outfig, "hist_cos_query_target_vs_candidates")

    # === Class-colored UMAPS (separate files) ===
    def _find_best_legend_pos(Y, margin=0.1):
        """Find the corner with least data density for legend placement."""
        x_min, x_max = Y[:, 0].min(), Y[:, 0].max()
        y_min, y_max = Y[:, 1].min(), Y[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Define corner regions
        corners = {
            'upper right': (x_max - margin * x_range, y_max - margin * y_range),
            'upper left': (x_min + margin * x_range, y_max - margin * y_range),
            'lower right': (x_max - margin * x_range, y_min + margin * y_range),
            'lower left': (x_min + margin * x_range, y_min + margin * y_range),
        }
        
        # Count points in each corner region
        corner_size = 0.15  # 15% of range
        counts = {}
        for name, (cx, cy) in corners.items():
            mask = ((Y[:, 0] >= cx - corner_size * x_range) & (Y[:, 0] <= cx + corner_size * x_range) &
                    (Y[:, 1] >= cy - corner_size * y_range) & (Y[:, 1] <= cy + corner_size * y_range))
            counts[name] = np.sum(mask)
        
        # Return corner with least points
        best_corner = min(counts, key=counts.get)
        loc_map = {
            'upper right': 'upper right',
            'upper left': 'upper left',
            'lower right': 'lower right',
            'lower left': 'lower left',
        }
        return loc_map[best_corner]
    
    def _plot_umap_class(X, labels, title, fname):
        if not _HAS_UMAP or len(X) < 5:
            return
        # Filter out None values if present (for mol_embeds)
        valid_mask = np.array([x is not None for x in X])
        if not np.any(valid_mask):
            return
        X_valid = [X[i] for i in range(len(X)) if valid_mask[i]]
        labels_valid = [labels[i] for i in range(len(labels)) if valid_mask[i]]
        
        Xn = _l2norm_np(np.asarray(X_valid))
        reducer = umap.UMAP(n_neighbors=60, min_dist=0.10, metric="cosine", random_state=args.seed)
        Y = reducer.fit_transform(Xn)

        # top-K classes in legend
        lab_arr = np.asarray(labels_valid, dtype=object)
        uniq, counts = np.unique(lab_arr, return_counts=True)
        order = np.argsort(-counts)
        uniq = uniq[order]; counts = counts[order]
        topK = set(uniq[:args.umap_topk_classes].tolist())
        mapped = np.array([l if l in topK else "Other" for l in lab_arr], dtype=object)

        # More distinguishable color palette (using distinct colors from matplotlib tab10 + additional)
        palette = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # grey
            "#bcbd22",  # olive
            "#17becf",  # cyan
            "#ff9896",  # light red
            "#c5b0d5",  # light purple
        ]
        names = sorted(set(mapped.tolist()), key=lambda x: (x!="Other", x))
        color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}

        # Larger figure to accommodate legend
        plt.figure(figsize=(5.5, 4.5), dpi=300)
        for n in names:
            m = (mapped == n)
            if np.any(m):
                # Smaller, sharper points - no edge for crisp rendering
                plt.scatter(Y[m,0], Y[m,1], s=2, c=color_map[n], alpha=0.6,
                            edgecolors="none", linewidths=0, label=n, rasterized=False)
        
        # Find best legend position (corner with least data)
        best_loc = _find_best_legend_pos(Y)
        plt.legend(frameon=False, loc=best_loc, markerscale=2.0, ncol=2, 
                  fontsize=5, columnspacing=0.8, handletextpad=0.3)
        plt.title(title, fontsize=9)
        plt.xticks([]); plt.yticks([])
        plt.tight_layout(pad=0.2)
        # transparent background for overlay-friendly exports
        savefig(outfig, fname, transparent=True)

    # Build inputs for class UMAPs if we have labels
    # Ensure all three UMAPs use the same data (intersection of valid indices)
    if args.umap and (labels is not None) and len(labels) >= 5:
        # Convert to lists if they're numpy arrays (from cache)
        if isinstance(dreams_embeds, np.ndarray):
            dreams_embeds = dreams_embeds.tolist()
        if isinstance(mapped_embeds, np.ndarray):
            mapped_embeds = mapped_embeds.tolist()
        if mol_embeds is not None and isinstance(mol_embeds, np.ndarray):
            mol_embeds = mol_embeds.tolist()
        
        # Find intersection of valid indices across all embeddings
        dreams_valid = np.array([x is not None and (not isinstance(x, np.ndarray) or x.size > 0) for x in dreams_embeds])
        mapped_valid = np.array([x is not None and (not isinstance(x, np.ndarray) or x.size > 0) for x in mapped_embeds])
        
        print(f"[UMAP] Total samples: dreams={len(dreams_embeds)}, mapped={len(mapped_embeds)}, mol={len(mol_embeds) if mol_embeds is not None else 0}")
        print(f"[UMAP] Valid samples: dreams={np.sum(dreams_valid)}, mapped={np.sum(mapped_valid)}, mol={np.sum([x is not None and (not isinstance(x, np.ndarray) or x.size > 0) for x in mol_embeds]) if mol_embeds is not None else 0}")
        
        if mol_embeds is not None and len(mol_embeds) > 0:
            mol_valid = np.array([x is not None and (not isinstance(x, np.ndarray) or x.size > 0) for x in mol_embeds])
            # Ensure all arrays have same length
            min_len = min(len(dreams_valid), len(mapped_valid), len(mol_valid))
            dreams_valid = dreams_valid[:min_len]
            mapped_valid = mapped_valid[:min_len]
            mol_valid = mol_valid[:min_len]
            # Use intersection: all must be valid
            common_valid = dreams_valid & mapped_valid & mol_valid
        else:
            # If no mol_embeds, use intersection of dreams and mapped
            min_len = min(len(dreams_valid), len(mapped_valid))
            dreams_valid = dreams_valid[:min_len]
            mapped_valid = mapped_valid[:min_len]
            common_valid = dreams_valid & mapped_valid
        
        print(f"[UMAP] Common valid samples: {np.sum(common_valid)}")
        
        if np.sum(common_valid) < 5:
            print(f"[Warning] Not enough common valid embeddings ({np.sum(common_valid)}) for UMAP plots")
        else:
            # Filter all embeddings and labels to use same indices
            dreams_common = [dreams_embeds[i] for i in range(min(len(dreams_embeds), len(common_valid))) if i < len(common_valid) and common_valid[i]]
            mapped_common = [mapped_embeds[i] for i in range(min(len(mapped_embeds), len(common_valid))) if i < len(common_valid) and common_valid[i]]
            labels_common = [labels[i] for i in range(min(len(labels), len(common_valid))) if i < len(common_valid) and common_valid[i]]
            
            print(f"[UMAP] Filtered samples for plotting: {len(dreams_common)}")
            
            _plot_umap_class(dreams_common, labels_common, "DreaMS embeddings of mass spectra ($z_s$)", "umap_dreams_classes")
            _plot_umap_class(mapped_common, labels_common, "Mapped spectra embeddings ($\\hat z$)", "umap_mapped_classes")
            
            # Add molecular embedding UMAP if available
            if mol_embeds is not None and len(mol_embeds) > 0:
                mol_common = [mol_embeds[i] for i in range(min(len(mol_embeds), len(common_valid))) if i < len(common_valid) and common_valid[i]]
                print(f"[UMAP] Molecular samples for plotting: {len(mol_common)}")
                _plot_umap_class(mol_common, labels_common, "Molecular embeddings from model ($z_m$)", "umap_mol_classes")
            
            # Create combined UMAP to compare mapped vs molecular in same space
            # This shows if mapped embeddings actually align with molecular embeddings
            # Use the same common_valid indices as the separate UMAPs
            if mol_embeds is not None and np.sum(common_valid) >= 5:
                # Use the same filtered embeddings as the separate UMAPs
                mapped_embeds_valid = mapped_common
                mol_embeds_valid = mol_common
                labels_valid = labels_common
                
                combined_embeds = mapped_embeds_valid + mol_embeds_valid
                combined_Xn = _l2norm_np(np.asarray(combined_embeds))
                combined_reducer = umap.UMAP(n_neighbors=60, min_dist=0.10, metric="cosine", random_state=args.seed)
                combined_Y = combined_reducer.fit_transform(combined_Xn)
                
                # Split back into mapped and molecular
                n = len(mapped_embeds_valid)
                mapped_Y = combined_Y[:n]
                mol_Y = combined_Y[n:]
                
                # Plot comparison using same settings as _plot_umap_class
                lab_arr = np.asarray(labels_valid, dtype=object)
                uniq, counts = np.unique(lab_arr, return_counts=True)
                order = np.argsort(-counts)
                uniq = uniq[order]
                topK = set(uniq[:args.umap_topk_classes].tolist())
                mapped_labs = np.array([l if l in topK else "Other" for l in labels_valid], dtype=object)
                
                palette = [
                    "#1f77b4",  # blue
                    "#ff7f0e",  # orange
                    "#2ca02c",  # green
                    "#d62728",  # red
                    "#9467bd",  # purple
                    "#8c564b",  # brown
                    "#e377c2",  # pink
                    "#7f7f7f",  # grey
                    "#bcbd22",  # olive
                    "#17becf",  # cyan
                    "#ff9896",  # light red
                    "#c5b0d5",  # light purple
                ]
                names = sorted(set(mapped_labs.tolist()), key=lambda x: (x!="Other", x))
                color_map = {n: palette[i % len(palette)] for i, n in enumerate(names)}
                
                plt.figure(figsize=(5.5, 4.5), dpi=300)
                # Plot molecular embeddings (circles)
                for n in names:
                    m = (mapped_labs == n)
                    if np.any(m):
                        plt.scatter(mol_Y[m,0], mol_Y[m,1], s=1.2, c=color_map[n], alpha=0.6,
                                   edgecolors="none", linewidths=0, label=f"{n} (mol)", marker="o", rasterized=False)
                # Plot mapped embeddings (triangles) - same settings
                for n in names:
                    m = (mapped_labs == n)
                    if np.any(m):
                        plt.scatter(mapped_Y[m,0], mapped_Y[m,1], s=1.2, c=color_map[n], alpha=0.6,
                                   edgecolors="none", linewidths=0, label=f"{n} (mapped)", marker="^", rasterized=False)
                # Find best legend position for combined plot
                combined_Y_all = np.vstack([mapped_Y, mol_Y])
                best_loc_combined = _find_best_legend_pos(combined_Y_all)
                plt.legend(frameon=False, loc=best_loc_combined, markerscale=2.0, ncol=2, 
                          fontsize=5, columnspacing=0.8, handletextpad=0.3)
                plt.title("Mapped vs Molecular embeddings (same UMAP space)", fontsize=9)
                plt.xticks([]); plt.yticks([])
                plt.tight_layout(pad=0.2)
                savefig(outfig, "umap_mapped_vs_mol_combined", transparent=True)
                
                # Also compute cosine similarity statistics
                mapped_np = _l2norm_np(np.asarray(mapped_embeds_valid))
                mol_np = _l2norm_np(np.asarray(mol_embeds_valid))
                cosines = np.array([np.dot(mapped_np[i], mol_np[i]) for i in range(len(mapped_embeds_valid))])
                print(f"\n=== Mapped vs Molecular Embedding Alignment ===")
                print(f"Mean cosine similarity: {np.mean(cosines):.4f}")
                print(f"Median cosine similarity: {np.median(cosines):.4f}")
                print(f"Min/Max: {np.min(cosines):.4f} / {np.max(cosines):.4f}")
                print(f"Std: {np.std(cosines):.4f}")

    # === Pairwise distance analysis (dreams vs mapped) ===
    dist_summary = {}
    if (labels is not None) and len(labels) >= 5:
        rng = np.random.default_rng(args.seed)
        same_pairs, diff_pairs = _sample_pairs(labels, args.pair_max_same, args.pair_max_diff, rng)

        def _one_space(name, X):
            Xn = _l2norm_np(np.asarray(X))
            ds = _cosine_distance(Xn, same_pairs)
            dd = _cosine_distance(Xn, diff_pairs)
            stats = _effect_sizes(ds, dd)
            # figure
            plt.figure(figsize=(3.35, 2.6), dpi=300)
            bins = 50
            plt.hist(ds, bins=bins, density=True, alpha=0.60, color=BLUE, label="same class")
            plt.hist(dd, bins=bins, density=True, alpha=0.60, color=ORANGE, label="different class")
            plt.xlabel("cosine distance")
            plt.ylabel("density")
            plt.title(f"Pairwise distances in {name} space")
            plt.grid(axis="y")
            plt.legend(frameon=False)
            savefig(outfig, f"dist_pairs_same_vs_diff_{name}")
            return stats

        dist_summary["dreams"] = _one_space("dreams", dreams_embeds)
        dist_summary["mapped"] = _one_space("mapped", mapped_embeds)

        with open(os.path.join(args.outdir, "dist_eval.json"), "w") as f:
            json.dump(dist_summary, f, indent=2)

    # Console summary
    print("\n=== Aggregates (mean [95% CI]) ===")
    for k, (m, lo, hi) in aggs.items():
        print(f"{k:>14s}: {m:.4f} [{lo:.4f}, {hi:.4f}]")
    if dist_summary:
        print("\n=== Distance separation (same vs different) ===")
        for space, s in dist_summary.items():
            msg = (f"{space:>7s}: mean_same={s['mean_same']:.4f}, mean_diff={s['mean_diff']:.4f}, "
                   f"AUC={s.get('auc_same_vs_diff', float('nan')):.3f}, "
                   f"Cohen d={s['cohen_d']:.3f}, Cliff δ={s.get('cliffs_delta', float('nan')):.3f}")
            if "ks_stat" in s:
                msg += f", KS={s['ks_stat']:.3f} (p={s['ks_p']:.1e})"
            print(msg)
    print(f"\nSaved CSV/JSON and figures to: {args.outdir}")


# --- collate wrapper (unchanged) ---------------------------------------------
def _collate_with_smiles(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed):
    out = base_collate(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed=seed)
    out["titles"] = [ex["title"] for ex in base_batch]
    out["smiles_true"] = [ex["smiles"] for ex in base_batch]
    return out


if __name__ == "__main__":
    main()
