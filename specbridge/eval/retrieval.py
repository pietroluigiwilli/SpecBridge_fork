# specbridge/eval/retrieval.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Optional
import argparse, math, json, os, sys
import torch
import torch.nn.functional as F
import numpy as np

from specbridge.adapters.dreams_adapter import load_dreams_encoder, DreamsAdapter
from specbridge.models.mol import MolFeaturizer, MolEncoder, MolAdapter
from specbridge.models.mapper import MapperB
from specbridge.utils.common import unit_normalize, set_seed
from specbridge.data.massspecgym import MassSpecGymDataset, bin_peaks

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

def _canon_smiles(smiles: str) -> str:
    s = (smiles or "").strip()
    if not s:
        return ""
    if _HAS_RDKIT:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            return Chem.MolToSmiles(m, canonical=True)
    return s  # fallback

@torch.no_grad()
def _encode_batch(
    batch: List[dict],
    dreams: DreamsAdapter,
    mol_adapter: MolAdapter,
    featurizer: MolFeaturizer,
    spec_bins: int,
    device: torch.device,
):
    # Build peaks tensor and binned spectra
    peaks_list, spectra, smiles = [], [], []
    for ex in batch:
        mz = ex["mz"].to(device)
        inten = ex["intensity"].to(device)
        peaks = torch.stack([mz, inten], dim=-1)
        peaks_list.append(peaks)
        spectra.append(bin_peaks(mz, inten, num_bins=spec_bins))
        smiles.append(_canon_smiles(ex.get("smiles", "")))
    spectra = torch.stack(spectra, dim=0)
    # Pad peaks
    max_len = max(p.size(0) for p in peaks_list)
    pad = torch.zeros(len(batch), max_len, 2, device=device)
    for i,p in enumerate(peaks_list):
        L = p.size(0)
        pad[i,:L,:] = p
    meta = {"peaks": pad}  # DreamsAdapter will pool this
    # Molecule features
    mol_feats = featurizer.featurize(smiles).to(device)
    # Encodings
    z_s = dreams(spectra, meta)                 # spec embeddings
    z_m = mol_adapter(mol_feats)                # mol embeddings (ground truth)
    return z_s, z_m, smiles

def _proto_by_mol(z_m: torch.Tensor, smiles: List[str]) -> Tuple[torch.Tensor, List[str], Dict[str,int]]:
    # Average embeddings per unique SMILES (prototype gallery)
    by_id: Dict[str, List[int]] = {}
    for i,s in enumerate(smiles):
        if not s:  # skip empties
            continue
        by_id.setdefault(s, []).append(i)
    ids = sorted(by_id.keys())
    if len(ids) == 0:
        return torch.empty(0, z_m.size(1), device=z_m.device), [], {}
    protos = []
    for s in ids:
        idx = torch.tensor(by_id[s], device=z_m.device, dtype=torch.long)
        proto = unit_normalize(z_m.index_select(0, idx).mean(dim=0, keepdim=True))
        protos.append(proto)
    Z = torch.cat(protos, dim=0)
    lookup = {s:i for i,s in enumerate(ids)}
    return Z, ids, lookup

def _metrics_from_ranks(ranks: np.ndarray, K_list=(1,5,10)) -> Dict[str,float]:
    out = {}
    n = ranks.size
    for k in K_list:
        out[f"R@{k}"] = float((ranks < k).mean())
    out["MRR"] = float((1.0 / (ranks + 1)).mean())
    out["median_rank"] = float(np.median(ranks + 1))
    return out

def _least_squares_R2(X: torch.Tensor, Y: torch.Tensor) -> float:
    # Fit W (dxd) to minimize ||XW - Y||^2 on train half, evaluate R^2 on test half
    n = X.size(0)
    perm = torch.randperm(n, device=X.device)
    ntr = int(0.5*n)
    tr, te = perm[:ntr], perm[ntr:]
    Xtr, Ytr = X[tr], Y[tr]
    Xte, Yte = X[te], Y[te]
    # Closed form W = (X^T X)^-1 X^T Y
    XtX = Xtr.T @ Xtr + 1e-6*torch.eye(Xtr.size(1), device=X.device)
    W = torch.linalg.solve(XtX, Xtr.T @ Ytr)
    Yhat = Xte @ W
    ss_res = torch.sum((Yte - Yhat)**2).item()
    ss_tot = torch.sum((Yte - Yte.mean(dim=0))**2).item()
    R2 = 1.0 - (ss_res / max(ss_tot, 1e-9))
    return float(max(min(R2, 1.0), -1.0))

def _replicate_stats(z_s: torch.Tensor, smiles: List[str]) -> Dict[str,float]:
    # Within-molecule variance vs between-molecule variance (lower ratio is better)
    by_id: Dict[str, List[int]] = {}
    for i,s in enumerate(smiles):
        if s: by_id.setdefault(s, []).append(i)
    groups = [torch.tensor(ix, device=z_s.device, dtype=torch.long) for s,ix in by_id.items() if len(ix)>=2]
    if len(groups) < 2:  # not enough replicates
        return {"replicate_ratio": float("nan")}
    means = []
    within = []
    for g in groups:
        Zg = unit_normalize(z_s.index_select(0, g))
        mu = Zg.mean(dim=0, keepdim=True)
        means.append(mu)
        within.append(((Zg - mu)**2).sum(dim=1).mean())
    within_var = torch.stack(within).mean().item()
    means = unit_normalize(torch.cat(means, dim=0))
    between_var = ((means - means.mean(dim=0, keepdim=True))**2).sum(dim=1).mean().item()
    return {"replicate_ratio": float(within_var / max(between_var, 1e-9))}

def main():
    ap = argparse.ArgumentParser(description="SpecBridge retrieval / alignment evaluation")
    ap.add_argument("--mgf", type=str, required=True)
    ap.add_argument("--dreams-ckpt", type=str, default=None)
    ap.add_argument("--adapter-ckpt", type=str, default=None, help="optional: load trained mapper/proj weights")
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=512)
    ap.add_argument("--mapper-hidden", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=50000, help="max spectra to evaluate (for speed)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--use-mapped", action="store_true", help="use mapped z_hat vs. raw spec z_s for retrieval")
    ap.add_argument("--fold-query", type=str, default='test',
                help="Query fold(s), e.g. 'val' or 'test' or 'val,test'")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    ds = MassSpecGymDataset(args.mgf, meta_json=None)
    N = min(len(ds), args.limit)

    # Build models (same dims as training)
    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec-bins if hasattr(args, 'spec-bins') else args.spec_bins, d_out=1024)
    dreams = DreamsAdapter(dreams_backbone, d_out=args.cond_dim, hidden=0, freeze_backbone=True).to(device).eval()
    mol_encoder = MolEncoder(d_in=args.fp_bits, embed_dim=512)
    mol = MolAdapter(mol_encoder, d_out=args.cond_dim, hidden=0).to(device).eval()
    mapper = MapperB(d=args.cond_dim, hidden=args.mapper_hidden, gaussian=True).to(device).eval()

    if args.adapter_ckpt:
        ck = torch.load(args.adapter_ckpt, map_location="cpu")
        # be permissive with keys: look for 'model' or individual modules
        sd = ck.get("model", ck)
        dreams.load_state_dict({k.replace("spec.", ""):v for k,v in sd.items() if k.startswith("spec.")}, strict=False)
        mol.load_state_dict({k.replace("mol.", ""):v for k,v in sd.items() if k.startswith("mol.")}, strict=False)
        mapper.load_state_dict({k.replace("mapB.", ""):v for k,v in sd.items() if k.startswith("mapB.")}, strict=False)

    featurizer = MolFeaturizer(fp_bits=args.fp_bits)

    # Pass 1: encode everything (streamed)
    B = args.batch_size
    zs_all, zm_all, smi_all = [], [], []
    with torch.no_grad():
        for i in range(0, N, B):
            batch = [ds[j] for j in range(i, min(i+B, N))]
            z_s, z_m, smiles = _encode_batch(batch, dreams, mol, featurizer, args.spec_bins, device)
            zs_all.append(unit_normalize(z_s))
            zm_all.append(unit_normalize(z_m))
            smi_all.extend(smiles)
    Zs = unit_normalize(torch.cat(zs_all, dim=0))
    Zm = unit_normalize(torch.cat(zm_all, dim=0))

    # Optional: map spec→mol (this is what your adapter is supposed to learn)
    if args.use_mapped:
        mu, lv = mapper(Zs)
        Zq = unit_normalize(mapper.sample(mu, lv, deterministic=True))
    else:
        Zq = Zs  # direct contrastive space sanity

    # Build molecule prototype gallery (one vector per unique SMILES)
    Zm_proto, ids, id_lookup = _proto_by_mol(Zm, smi_all)
    if Zm_proto.size(0) == 0:
        print("[warn] no valid SMILES in dataset slice; cannot evaluate retrieval")
        sys.exit(0)

    # For each query, get the correct mol id (by its SMILES)
    gt = []
    for s in smi_all:
        if s in id_lookup:
            gt.append(id_lookup[s])
        else:
            gt.append(-1)
    gt = np.array(gt, dtype=np.int64)

    # Similarity to gallery and ranks
    sims = (Zq @ Zm_proto.T).float().cpu()  # [N, M]
    ranks = np.full((sims.size(0),), fill_value=np.inf)
    for i in range(sims.size(0)):
        if gt[i] < 0:
            continue
        row = sims[i].detach().cpu().numpy()
        # higher is better; argsort descending
        order = np.argsort(-row)
        ranks[i] = int(np.where(order == gt[i])[0][0])
    mask = np.isfinite(ranks)
    ranks = ranks[mask].astype(np.int64)
    metrics = _metrics_from_ranks(ranks, K_list=(1,5,10))
    print(f"[retrieval] evaluated {int(mask.sum())} / {len(gt)} queries against {Zm_proto.size(0)} molecules")
    for k,v in metrics.items():
        print(f"  {k:>12s}: {v:.4f}")

    # Alignment R^2 (held-out)
    R2 = _least_squares_R2(Zs, Zm)
    print(f"[alignment] linear R^2 (spec→mol): {R2:.4f}")

    # Replicate consistency (variance ratio)
    rep = _replicate_stats(Zs, smi_all)
    print(f"[replicates] within/between variance ratio: {rep['replicate_ratio']:.4f}")

if __name__ == "__main__":
    main()
