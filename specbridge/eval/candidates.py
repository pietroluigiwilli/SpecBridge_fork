# specbridge/eval/candidates.py
from __future__ import annotations
import argparse, os, pickle, hashlib
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json
# Optional canonicalization (safer keys). Falls back to raw if RDKit is missing.
try:
    from rdkit import Chem
    from rdkit import DataStructs
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

# Optional MCES support
import pulp
from myopic_mces.myopic_mces import MCES


import random
from specbridge.utils.common import set_seed
from specbridge.adapters.dreams_adapter import load_dreams_encoder, DreamsAdapter

from specbridge.models.mol import MolFeaturizer, MolEncoder, MolAdapter
from specbridge.models.mapper import MapperB, DreamsToMolCondition
from specbridge.data.massspecgym import MassSpecGymDataset

# -------------------------
# Helpers
# -------------------------
def inchikey14(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        ik = Chem.MolToInchiKey(m)
        return ik.split("-")[0] if ik else None
    except Exception:
        return None

def _hash_smiles(smiles_list: List[str]) -> str:
    s = "\n".join(sorted(smiles_list))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _canon_smi(s: Optional[str]) -> Optional[str]:
    """Canonicalize SMILES if RDKit is available; else strip whitespace."""
    if s is None:
        return None
    return s
    s = s.strip()
    if not s or not _HAS_RDKIT:
        return s
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol) if mol is not None else s

def _metrics_from_ranks(ranks: List[int], k_vals=(1, 5, 10, 20)) -> Dict[str, float]:
    n = len(ranks)
    if n == 0:
        return {f"R@{k}": 0.0 for k in k_vals} | {"MRR": 0.0, "median_rank": float("nan")}
    mrr = sum(1.0 / (r + 1) for r in ranks) / n
    med = float(sorted(ranks)[n // 2])
    out = {"MRR": mrr, "median_rank": med}
    for k in k_vals:
        out[f"R@{k}"] = sum(1 for r in ranks if r < k) / n
    return out

# MCES helper
class MyopicMCES:
    """Wrapper class for MCES computation with silent solver."""
    def __init__(
        self,
        ind: int = 0,  # dummy index
        solver: str = None,  # Use the first available solver
        threshold: int = 15,  # MCES threshold
        always_stronger_bound: bool = True,  # "False" makes computations a lot faster, but leads to overall higher MCES values
        solver_options: dict = None
    ):
        self.ind = ind
        if solver is None:
            solver = pulp.listSolvers(onlyAvailable=True)[0] if pulp else None
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)  # make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1: str, smiles_2: str) -> float:
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            ind=self.ind,
            threshold=self.threshold,
            always_stronger_bound=self.always_stronger_bound,
            solver=self.solver,
            solver_options=self.solver_options
        )
        dist = retval[1]
        return dist

def mces_distance(smi_a: str, smi_b: str) -> float:
    """Compute MCES distance between two SMILES strings using MyopicMCES."""
    mces = MyopicMCES()
    return float(mces(smi_a, smi_b))

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors (1D or 2D)."""
    # Ensure both are 2D for consistent handling
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    # Normalize along last dimension
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    # Compute dot product (assuming same batch size or broadcasting)
    return float((a_norm * b_norm).sum(dim=-1).item())

def build_mol_embed_fn(args, model, device, mol_adapter=None, state=None):
    """
    Returns a function: embed_fn(List[str]) -> torch.FloatTensor [N, cond_dim] (L2-normalized, on CPU)
    - adapter  : uses your trained MolAdapter forward (fp -> cond_dim)
    - ecfp     : fixed Morgan fingerprint -> pad/crop to cond_dim -> normalize
    - chemberta: CLS embedding -> pad/crop to cond_dim -> normalize
    """
    if args.mol_space == "chemberta":
        # from transformers import AutoTokenizer, AutoModel
        # tok = AutoTokenizer.from_pretrained(args.chemberta_model)
        # mdl = AutoModel.from_pretrained(args.chemberta_model).to(device).eval()
        
        for p in model.parameters(): p.requires_grad = False

        # Try to reconstruct trained projection
        # W = state.get("chem_proj.weight") if isinstance(state, dict) else None
        # b = state.get("chem_proj.bias")   if isinstance(state, dict) else None
        # proj = None
        # if isinstance(W, torch.Tensor):
        #     proj = torch.nn.Linear(W.size(1), W.size(0), bias=(b is not None)).to(device)
        #     with torch.no_grad():
        #         proj.weight.copy_(W.to(device))
        #         if b is not None: proj.bias.copy_(b.to(device))
        #     proj.eval()
        #     for p in proj.parameters(): p.requires_grad = False

        def embed_fn(smiles_list: list[str], device, bs: int = 512) -> torch.Tensor:
            out = []
            with torch.no_grad():
                for i in range(0, len(smiles_list), bs):
                    chunk = smiles_list[i:i+bs]
                    h = model._chemberta_embed(chunk, device)
                    out.append(h.cpu())
            Z = torch.cat(out, dim=0)
            return Z
        return embed_fn

    else:
        raise ValueError(f"Unknown --mol-space: {args.mol_space}")


def _collate_with_smiles(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed):
    from specbridge.data.massspecgym import collate_massspecgym as base_collate
    out = base_collate(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed=seed)
    out["titles"] = [ex["title"] for ex in base_batch]
    out["smiles_true"] = [ex["smiles"] for ex in base_batch]
    return out

def build_model(args, device):
    import torch.nn as nn
   # --- Load backbone first (kept frozen for eval)
    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)

    model = DreamsToMolCondition(
            dreams_backbone,
            d_out=args.cond_dim,
            mapper_hidden=args.mapper_hidden,
            gaussian=not args.no_gaussian,
            mol_space=args.mol_space,
            chemberta_model=getattr(args, "chemberta_model", None),
            args=args
        ).to(device).eval()
    state = torch.load(args.adapter_ckpt, map_location="cpu")
    print(model.load_state_dict(state.get("model", {}), strict=False))
    for p in model.parameters():
        p.requires_grad = False
    return model
    # --- Peek checkpoint to recover shapes used during training
    state = None
    saved_args = {}
    if args.adapter_ckpt:
        sd_raw = torch.load(args.adapter_ckpt, map_location="cpu")
        if isinstance(sd_raw, dict) and "model" in sd_raw and isinstance(sd_raw["model"], dict):
            state = sd_raw["model"]
            if isinstance(sd_raw.get("args", None), dict):
                saved_args = sd_raw["args"]
        elif isinstance(sd_raw, dict) and "state_dict" in sd_raw and isinstance(sd_raw["state_dict"], dict):
            state = sd_raw["state_dict"]
            if isinstance(sd_raw.get("args", None), dict):
                saved_args = sd_raw["args"]
        else:
            state = sd_raw

    # Helper to infer a hidden size from first projection layer
    def _infer_hidden(prefix: str, fallback: int = 0) -> int:
        if state is None:
            return fallback
        # try both flat and nested proj layouts
        for k in (f"{prefix}.proj.0.weight",
                f"{prefix}.proj.0.0.weight",  # first Linear in a nested block
                f"{prefix}.proj.0.2.weight"): # second Linear in a nested block
            w = state.get(k, None)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                return int(w.shape[0])  # out_features of first Linear we find
        return fallback


    # Helper to check if module uses a nonzero hidden
    spec_hidden = _infer_hidden("spec", fallback=0)
    mol_hidden  = _infer_hidden("mol",  fallback=0)
    print(spec_hidden, mol_hidden)
    # Mapper hidden: look at mu.0.weight if present
    def _infer_mapper_hidden(fallback: int = args.mapper_hidden):
        if state is None: return fallback
        w0 = state.get("mapB.mu.0.weight", None)
        if isinstance(w0, torch.Tensor) and w0.ndim == 2:
            return int(w0.shape[0])
        return fallback
    mapper_hidden = _infer_mapper_hidden()

    # If checkpoint saved cond_dim etc., adopt them (only for adapter-space eval)
    cond_dim_ckpt = int(saved_args.get("cond_dim", args.cond_dim))
    cond_dim = cond_dim_ckpt if args.mol_space == "adapter" else args.cond_dim

    # --- Build modules with inferred shapes
    spec = DreamsAdapter(dreams_backbone, d_out=cond_dim, hidden=0, freeze_backbone=True).to(device).eval()

    # mol_encoder = MolEncoder(d_in=args.fp_bits, embed_dim=512).to(device).eval()
    # mol_adapter = MolAdapter(mol_encoder, d_out=cond_dim, hidden=mol_hidden).to(device).eval()

    mapper = MapperB(d=cond_dim, hidden=mapper_hidden, gaussian=(not args.no_gaussian)).to(device).eval()

    # --- Shape-safe load (but should now mostly match)
    if state is not None:
        def _shape_filter_load(module, st, prefix):
            msd = module.state_dict()
            pick = {}
            for k, v in st.items():
                if not k.startswith(prefix + "."):
                    continue
                name = k[len(prefix) + 1:]
                if name in msd and msd[name].shape == v.shape:
                    pick[name] = v
            missing, unexpected = module.load_state_dict(pick,strict=False)
            dropped = [k for k in st.keys() if k.startswith(prefix + ".") and (k[len(prefix)+1:] not in pick)]
            if dropped:
                print(f"[ckpt] dropped (shape mismatch) under {prefix}:")
                for dk in dropped:
                    print("   -", dk)
            return missing, unexpected, dropped

        miss, unexp, drop = [], [], []
        m, u, d = _shape_filter_load(spec,       state, "spec");  miss += m; unexp += u; drop += d
        m, u, d = _shape_filter_load(mapper,     state, "mapB");  miss += m; unexp += u; drop += d
        print(miss, drop)
        print(f"[ckpt] loaded with missing={len(miss)} unexpected={len(unexp)} dropped_mismatch={len(drop)}")

    return spec, mapper, state



# -------------------------
# Main
# -------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Per-spectrum candidate ranking (PKL: true_smi -> [candidate_smi,...])")
    ap.add_argument("--mgf", required=True, type=str)
    ap.add_argument("--dreams-ckpt", type=str, default=None)
    ap.add_argument("--adapter-ckpt", type=str, default=None, help="Trained adapter/mapper checkpoint")
    ap.add_argument("--candidates", required=True, type=str, help="*.pkl with {true_smi: [cand_smi,...]}")
    ap.add_argument("--meta-json", type=str, default=None)
    ap.add_argument("--fold-query", type=str, default=None, help="e.g. val or test (comma-separated ok)")
    ap.add_argument("--use-mapped", action="store_true", help="use spectra→mol mapped embedding for query")
    ap.add_argument("--samples", type=int, default=1, help="samples per query if Gaussian mapper")
    ap.add_argument("--aggregate", type=str, default="max", choices=["max", "mean"], help="aggregate over samples")
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=1024)   # single source of truth
    ap.add_argument("--mapper-hidden", type=int, default=0)
    ap.add_argument("--no-gaussian", action="store_true")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cache-cand-emb", type=str, default=None,
                    help="optional path to save/load candidate embedding cache (.pt)")
    ap.add_argument("--use-true", action="store_true",
                    help="Use true molecule embedding as the query (diagnostic upper bound).")
    ap.add_argument("--deterministic-map", action="store_true",
                    help="Use mu (no sampling) for mapped queries.")
    ap.add_argument("--mol-space", choices=["adapter", "ecfp", "chemberta"], default="adapter",
                help="Embedding space for molecule candidates/true SMILES.")
    ap.add_argument("--ecfp-radius", type=int, default=2,
                    help="ECFP radius when --mol-space=ecfp.")
    ap.add_argument("--chemberta-model", type=str, default="seyonec/ChemBERTa-zinc-base-v1",
                    help="HF model id when --mol-space=chemberta.")
    ap.add_argument("--compute-mces", action="store_true",
                    help="Compute MCES distance for top-1 predictions.")
    ap.add_argument("--compute-similarity", action="store_true",
                    help="Compute cosine similarity between query and true molecule embeddings.")

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # Dataset (fold-aware)
    folds = None
    if args.fold_query:
        folds = {f.strip().lower() for f in args.fold_query.split(",") if f.strip()}
    ds = MassSpecGymDataset(args.mgf, args.meta_json, folds=folds)

    formula_vocab = max(2, getattr(ds, "_formula_vocab", 0) or 32)
    adduct_vocab  = max(2, getattr(ds, "_adduct_vocab", 0) or 16)
    charge_vocab  = max(2, getattr(ds, "_charge_vocab", 0) or 8)

    # Optional quick slice
    if args.limit:
        class _Slice(torch.utils.data.Dataset):
            def __init__(self, base, n): self.base, self.n = base, n
            def __len__(self): return min(self.n, len(self.base))
            def __getitem__(self, i): return self.base[i]
        ds = _Slice(ds, args.limit)

    collate = lambda b: _collate_with_smiles(
        b, args.spec_bins, formula_vocab, adduct_vocab, charge_vocab, args.fp_bits, seed=args.seed
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # --- Load PKL candidates: {true_smi: [cand_smi,...]} and canonicalize both keys & values ---
    if os.path.splitext(args.candidates)[1].lower() == ".pkl":
        assert os.path.splitext(args.candidates)[1].lower() == ".pkl", "Expected .pkl for --candidates"
        with open(args.candidates, "rb") as f:
            cand_map_raw: Dict[str, List[str]] = pickle.load(f)
    else:
        with open(args.candidates, "r") as f:
            cand_map_raw: Dict[str, List[str]] = json.load(f)

    cand_map: Dict[str, List[str]] = {}
    for k, vs in cand_map_raw.items():
        ck = _canon_smi(k) or k.strip()
        vals = []
        
        random.shuffle(vs)
        for v in vs:
            if not v:
                continue
            cv = _canon_smi(v) or v.strip()
            vals.append(cv)
        vals.append(ck)
        cand_map[ck] = list(dict.fromkeys(vals))  # unique, preserve order
        
        
    # Build models
    model= build_model(args, device)
    # spec, mapper, state = build_model(args, device)
    
    # from types import SimpleNamespace
    # adapter_wrapper = SimpleNamespace(mol=mol_adapter)  # gives .mol attribute
    # embed_fn, embed_meta = build_mol_embed_fn(args, device, adapter_model=adapter_wrapper)
    # embed_fn = model._chemberta_embed
    embed_fn = build_mol_embed_fn(args, model, device
                                )  # <<< pass state

    # Collect candidate universe actually needed
    all_query_true: List[str] = []
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
    print(f"[candidates] queries={len(all_query_true)} | queries_with_candidates={covered_titles} | unique_candidates={len(need_smiles)}")

    # Precompute candidate embeddings (with optional cache)
    # --- Precompute candidate embeddings (with optional cache)
    cand_z: Dict[str, torch.Tensor] = {}
    cache_path = args.cache_cand_emb
    loaded_cache = False
    if cache_path and os.path.isfile(cache_path):
        try:
            cand_z = torch.load(cache_path, map_location="cpu")
            loaded_cache = True
            print(f"[cache] loaded candidate embeddings from {cache_path} (|Z|={len(cand_z)})")
        except Exception:
            cand_z = {}

    if not loaded_cache:
        Z = embed_fn(need_smiles, device)  # [N, D] (ECFP->padded/cropped->L2 norm)
        cand_z = {(_canon_smi(s) or s.strip()): z for s, z in zip(need_smiles, Z)}
        if cache_path:
            torch.save(cand_z, cache_path)
            print(f"[cache] saved candidate embeddings to {cache_path}")

    uniq_true = sorted({(_canon_smi(s) or s.strip()) for s in all_query_true if s})
    Zt = embed_fn(uniq_true, device)
    true_z = {s: z for s, z in zip(uniq_true, Zt)}


    # Evaluate
    ranks: List[int] = []
    used = 0
    total = 0
    missing_true = 0
    missing_cand = 0
    cand_sizes: List[int] = []
    # MCES and similarity metrics
    mces_distances: List[float] = []
    cosine_similarities: List[float] = []
    
    for batch in dl:
        s = batch["spectra"].to(device)
        meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch["meta"].items()}
        # print(meta['peaks'].shape)
        # del meta['peaks']
        
        # continue
        smiles_true = batch["smiles_true"]
        z_s, z_m, z_hat, mu, lv = model(s, meta, None, inference=False)
        # Build query embeddings per mode
        
        if args.use_true:
            z_query = None  # per-item lookup
        else:
            # z_s = spec(s, meta)  # [B,D]
            if args.use_mapped:
                # mu, lv = mapper(z_s)
                if args.deterministic_map or (lv is None):
                    # z_query = F.normalize(mu, dim=-1)  # [B,D]
                    z_query = mu
                elif args.samples > 1:
                    zq = [mapper.sample(mu, lv, deterministic=False) for _ in range(args.samples)]
                    z_query = torch.stack(zq, dim=0)  # [K,B,D]
                else:
                    z_query = mapper.sample(mu, lv, deterministic=False)  # [B,D]
            else:
                z_query = F.normalize(z_s, dim=-1)  # [B,D]

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

            # Stack candidate Z (canonicalized lookup)
            Z = []
            kept = []
            for sm in cand_list:
                csm = _canon_smi(sm) or sm.strip()
                zi = cand_z.get(csm, None).to(device)
                # with torch.no_grad():
                #     zi = F.normalize(model.chem_proj(zi), dim=-1)
                if zi is not None:
                    Z.append(zi)
                    kept.append(csm)
            if not Z:
                missing_cand += 1
                continue

            # Z = F.normalize(torch.stack(Z, dim=0), dim=-1)  # [C,D]
            Z = torch.stack(Z, dim=0)
            cand_sizes.append(Z.size(0))

            if args.use_true:
                zqi = true_z.get(true_smi, None)
                Z = Z.to(zqi.device, non_blocking=True)

                if zqi is None:
                    missing_true += 1
                    continue
                # zqi = F.normalize(zqi, dim=-1)  # [D]
                sims = zqi @ Z.T  # [C]
            else:
                if z_query.dim() == 3:
                    # [K,B,D] -> aggregate over K
                    # zqi = F.normalize(z_query[:, i, :], dim=-1)  # [K,D]
                    zqi = z_query[:, i, :]
                    Z = Z.to(zqi.device, non_blocking=True)

                    sims = zqi @ Z.T  # [K,C]
                    sims = sims.max(dim=0).values if args.aggregate == "max" else sims.mean(dim=0)
                else:
                    # zqi = F.normalize(z_query[i], dim=-1)  # [D]
                    zqi = z_query[i]
                    Z = Z.to(zqi.device, non_blocking=True)

                    sims = zqi @ Z.T  # [C]
            # print(true_smi)
            # print(sims)
            # exit()
            order = torch.argsort(sims, descending=True)

            # Rank of the true SMILES within its candidate list (if present)
            # true_inchikey14 = inchikey14(true_smi)
            # ik_kept = list(set([inchikey14(k) for k in kept]))
            # if true_inchikey14 in ik_kept:
            #     idx = ik_kept.index(true_inchikey14)
            #     pos = (order == idx).nonzero(as_tuple=False)
            #     r = int(pos.item()) if pos.numel() > 0 else Z.size(0) - 1
            #     ranks.append(r)
            # else:
            #     missing_true += 1
            if true_smi in kept:
                idx = kept.index(true_smi)
                pos = (order == idx).nonzero(as_tuple=False)
                r = int(pos.item()) if pos.numel() > 0 else Z.size(0) - 1
                ranks.append(r)
                
                # Compute cosine similarity between query and true molecule
                if args.compute_similarity:
                    true_emb = true_z.get(true_smi, None)
                    if true_emb is not None:
                        # Get query embedding (handle different dimensions)
                        if args.use_true:
                            query_emb = true_emb.to(device)
                        else:
                            if z_query.dim() == 3:
                                query_emb = z_query[:, i, :].mean(dim=0) if args.aggregate == "mean" else z_query[:, i, :].max(dim=0).values
                            else:
                                query_emb = z_query[i]
                        
                        # Move true_emb to same device as query_emb
                        true_emb_device = true_emb.to(query_emb.device)
                        
                        # Normalize and compute cosine similarity
                        cos_sim = cosine_similarity(query_emb, true_emb_device)
                        cosine_similarities.append(cos_sim)
                
                # Compute MCES distance between true SMILES and top-1 predicted SMILES
                # (regardless of whether top-1 is correct or not)
                if args.compute_mces:
                    top1_idx = int(order[0].item())
                    top1_smi = kept[top1_idx]
                    mces_dist = mces_distance(true_smi, top1_smi)
                    # print(f"MCES distance between {true_smi} and {top1_smi}: {mces_dist}")
                    if mces_dist != float('inf'):
                        mces_distances.append(mces_dist)
            else:
                missing_true += 1

    # Metrics
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
    
    # Report cosine similarity metrics
    if args.compute_similarity and len(cosine_similarities) > 0:
        cos_mean = sum(cosine_similarities) / len(cosine_similarities)
        cos_median = sorted(cosine_similarities)[len(cosine_similarities) // 2]
        print(f"{'cos_sim_mean':>10s}: {cos_mean:.4f}")
        print(f"{'cos_sim_median':>10s}: {cos_median:.4f}")
    
    # Report MCES metrics
    if args.compute_mces and len(mces_distances) > 0:
        mces_mean = sum(mces_distances) / len(mces_distances)
        mces_median = sorted(mces_distances)[len(mces_distances) // 2]
        print(f"{'mces@1_mean':>10s}: {mces_mean:.2f}")
        print(f"{'mces@1_median':>10s}: {mces_median:.2f}")
        print(f"[mces] computed for {len(mces_distances)} entries (true vs top-1 predicted)")

if __name__ == "__main__":
    main()
