"""
SpecBridge entry script
----------------------
Thin shim that preserves the legacy script entry-point. For usage, prefer the
installed CLI:

    specbridge --help

This script forwards to the same training functions and remains runnable.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Iterable, Dict, Any

import argparse
import math
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.multiprocessing.set_start_method('spawn')

from specbridge.utils.common import set_seed, unit_normalize, ln_baseline


# ================================================================
# Projection heads
# ================================================================

class MLPHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, d_out)
            )
        else:
            self.net = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
    def forward(self, x):
        return self.norm(self.net(x))


from specbridge.adapters.dreams_adapter import DummyDreams, load_dreams_encoder, DreamsAdapter


from specbridge.models.mol import MolFeaturizer, MolEncoder, MolAdapter
from specbridge.data.massspecgym import MassSpecGymDataset, collate_massspecgym, validate_dataset


# ================================================================
# Mapper B (spectra -> molecule embedding space)
# ================================================================

from specbridge.models.mapper import MapperB


# ================================================================
# Contrastive loss (InfoNCE)
# ================================================================

from specbridge.losses.contrastive import InfoNCELoss


# ================================================================
# Forward spectral loss & toy predictor
# ================================================================

from specbridge.losses.forward import ForwardSpecLossConfig, ForwardSpectralLoss

from specbridge.predictors.toy import ToySpecPredictor


# ================================================================
# Conditioning mix helper
# ================================================================

def mix_condition(z_m_true: torch.Tensor, z_m_hat: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    if not training:
        return z_m_hat
    if z_m_true.dim() == 1:
        z_m_true = z_m_true.unsqueeze(0)
    if z_m_hat.dim() == 1:
        z_m_hat = z_m_hat.unsqueeze(0)
    B_true, D_true = z_m_true.size(0), z_m_true.size(1)
    B_hat, D_hat = z_m_hat.size(0), z_m_hat.size(1)
    if D_true != D_hat:
        raise ValueError(f"cond dim mismatch: {D_true} vs {D_hat}")
    if B_true != B_hat:
        if B_true == 1:
            z_m_true = z_m_true.expand(B_hat, -1)
        elif B_hat == 1:
            z_m_hat = z_m_hat.expand(B_true, -1)
        else:
            raise ValueError(f"batch mismatch: {B_true} vs {B_hat}")
    B = z_m_true.size(0)
    device = z_m_true.device
    mask = (torch.rand(B, device=device) < p).float().unsqueeze(-1)
    cond = mask * z_m_true + (1 - mask) * z_m_hat
    return unit_normalize(cond)


# ================================================================
# Small discriminator (optional; off by default in demo)
# ================================================================

class SmallDisc(nn.Module):
    def __init__(self, d: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def disc_loss(disc: SmallDisc, z_real: torch.Tensor, z_fake: torch.Tensor):
    real = disc(z_real)
    fake = disc(z_fake.detach())
    loss_D = (F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean())
    loss_G = -disc(z_fake).mean()
    return loss_D, loss_G


# ================================================================
# Toy decoder (API-compatible surface)
# ================================================================

from specbridge.decoders.toy import ToyDecoder


# ================================================================
# End-to-end adapter wrapper
# ================================================================

class DreamsToMolCondition(nn.Module):
    def __init__(self, dreams_encoder: nn.Module, mol_encoder: nn.Module, d_out: int = 512, mapper_hidden: int = 0, gaussian: bool = True):
        super().__init__()
        self.spec = DreamsAdapter(dreams_encoder, d_out=d_out, hidden=0, freeze_backbone=True)
        self.mol  = MolAdapter(mol_encoder, d_out=d_out, hidden=0)
        self.mapB = MapperB(d=d_out, hidden=mapper_hidden, gaussian=gaussian)
        self.contrast = InfoNCELoss(temperature=0.07, learnable_temp=True)
    def forward(self, spectra_binned, meta, mol_feats, inference: bool = False):
        z_s = self.spec(spectra_binned, meta)
        z_m = self.mol(mol_feats)
        mu_s, lv_s = self.mapB(z_s)
        z_hat = self.mapB.sample(mu_s, lv_s, deterministic=inference)
        return z_s, z_m, z_hat, mu_s, lv_s
    def align_losses(self, z_s, z_m, mu_s, lv_s, w_con=1.0, w_map=1.0, w_ortho=1e-3):
        L_con = self.contrast(z_s, z_m)
        L_map = ((mu_s - z_m.detach())**2).mean()
        L_ortho = self.mapB.orthogonality_penalty() * (w_ortho / max(1e-8, self.mapB.ortho_weight))
        return w_con*L_con + w_map*L_map + L_ortho, {"L_con": L_con.detach(), "L_map": L_map.detach(), "L_ortho": L_ortho.detach()}


def _inbatch_diag_metrics(z_s: torch.Tensor, z_m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        z_s = unit_normalize(z_s)
        z_m = unit_normalize(z_m)
        logits = z_s @ z_m.T
        B = logits.size(0)
        preds = torch.argmax(logits, dim=1)
        labels = torch.arange(B, device=logits.device)
        acc1 = (preds == labels).float().mean()
        diag = torch.diagonal(logits)
        if B > 1:
            mask = ~torch.eye(B, dtype=torch.bool, device=logits.device)
            cos_neg_mean = logits[mask].mean()
        else:
            cos_neg_mean = torch.tensor(0.0, device=logits.device)
        return acc1, diag.mean(), cos_neg_mean


# ================================================================
# Training demo (synthetic data)
# ================================================================

def make_synthetic_batch(B: int, spec_bins: int, fp_bits: int, device: torch.device):
    spectra = torch.rand(B, spec_bins, device=device)
    formula = F.one_hot(torch.randint(0, 32, (B,), device=device), num_classes=32).float()
    adduct  = F.one_hot(torch.randint(0, 16, (B,), device=device), num_classes=16).float()
    charge  = F.one_hot(torch.randint(0, 16, (B,), device=device), num_classes=16).float()
    meta = {"formula": formula, "adduct": adduct, "charge": charge}

    base_smiles = [
        "C", "CC", "CCC", "CCCC", "CCCCC",
        "CO", "CCO", "CCCO", "CCCCO",
        "C=O", "CC=O", "CCC=O",
        "O=C=O", "OC=O", "CC(=O)O", "CCC(=O)O",
        "c1ccccc1", "c1ccncc1",
        "CCN", "CCCN", "CCOC", "CCS"
    ]
    smiles = [random.choice(base_smiles) for _ in range(B)]

    featurizer = MolFeaturizer(fp_bits=fp_bits)
    mol_feats = featurizer.featurize(smiles).to(device)

    with torch.no_grad():
        proj = torch.randn(fp_bits, 512, device=device)
        graph_target = F.normalize(mol_feats @ proj, dim=-1)

    return {"spectra": spectra, "mol_feats": mol_feats, "graph_target": graph_target, "meta": meta}


def train_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)
    mol_encoder = MolEncoder(d_in=args.fp_bits, embed_dim=512)

    model = DreamsToMolCondition(dreams_backbone, mol_encoder, d_out=args.cond_dim, mapper_hidden=args.mapper_hidden, gaussian=not args.no_gaussian).to(device)

    decoder = ToyDecoder(cond_dim=args.cond_dim, graph_dim=512).to(device)
    spec_pred = ToySpecPredictor(graph_dim=512, spec_bins=args.spec_bins).to(device)

    fwd_loss = ForwardSpectralLoss()

    params = list(p for p in model.parameters() if p.requires_grad) + list(decoder.parameters()) + list(spec_pred.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    base_ln = ln_baseline(args.batch_size)

    for step in range(1, args.steps + 1):
        batch = make_synthetic_batch(args.batch_size, args.spec_bins, args.fp_bits, device)
        s = batch["spectra"]; m = batch["mol_feats"]; gt_graph = batch["graph_target"]; meta = batch["meta"]

        z_s, z_m, z_hat, mu_s, lv_s = model(s, meta, m, inference=False)
        L_align, logs = model.align_losses(z_s, z_m, mu_s, lv_s, w_con=1.0, w_map=1.0, w_ortho=1e-3)

        cond = mix_condition(z_m, z_hat, p=0.5, training=True)
        dec_out = decoder(gt_graph, cond=cond, formula=meta["formula"], adduct=meta["adduct"], charge=meta["charge"])
        L_dec = dec_out["loss"]

        s_pred = spec_pred(dec_out["graph"], meta)
        L_fwd = fwd_loss(s_pred, s)

        loss = L_align + args.w_dec * L_dec + args.w_fwd * L_fwd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            acc1, cos_pos, cos_neg = _inbatch_diag_metrics(z_s, z_m)
            print(
                f"step {step:05d} | total {loss.item():.4f} | L_con {logs['L_con'].item():.4f} | "
                f"L_map {logs['L_map'].item():.4f} | L_dec {L_dec.item():.4f} | L_fwd {L_fwd.item():.4f} | "
                f"acc@1 {acc1.item():.3f} | cos(+) {cos_pos.item():.3f} | cos(-) {cos_neg.item():.3f} | ln(B) {base_ln:.3f}"
            )

    print("[done] demo training complete.")


# ================================================================
# MassSpecGym-style dataset (MGF + optional JSON meta)
# ================================================================

# Cache for stable graph projection across batches
_PROJ_CACHE: Dict[int, torch.Tensor] = {}

def _get_stable_proj(fp_bits: int, device: torch.device, seed: int = 1337) -> torch.Tensor:
    key = (fp_bits, device.type)
    if key not in _PROJ_CACHE:
        g = torch.Generator(device=device).manual_seed(seed)
        _PROJ_CACHE[key] = torch.randn(fp_bits, 512, generator=g, device=device)
    return _PROJ_CACHE[key]


def _norm_key(x: Any) -> str:
    return str(x).strip().lower()


def _build_meta_map(meta_obj: Any) -> Dict[str, dict]:
    """Accepts dict (title->meta) or list of records. Returns {normalized_title: meta}."""
    out: Dict[str, dict] = {}
    if isinstance(meta_obj, dict):
        for k, v in meta_obj.items():
            out[_norm_key(k)] = v
        return out
    if isinstance(meta_obj, list):
        # Try common keys
        candidate_keys = ["title", "spectrum_id", "scan", "id", "name"]
        for rec in meta_obj:
            if not isinstance(rec, dict):
                continue
            key_val = None
            for ck in candidate_keys:
                if ck in rec:
                    key_val = rec[ck]
                    break
            if key_val is None:
                continue
            out[_norm_key(key_val)] = rec
        return out
    return out


class MassSpecGymDataset(torch.utils.data.Dataset):
    """Minimal MGF+JSON dataset that yields (binned spectra or peaks, smiles, meta)."""
    def __init__(self, mgf_path: str, meta_json: Optional[str] = None):
        super().__init__()
        try:
            from pyteomics import mgf  # type: ignore
        except Exception as e:
            raise ImportError(f"pyteomics is required to read MGF: {e}")
        self.meta_map: Dict[str, dict] = {}
        if meta_json is not None:
            with open(meta_json, 'r') as f:
                meta_obj = json.load(f)
            self.meta_map = _build_meta_map(meta_obj)
        self._records = []
        self._adduct_set = set()
        self._charge_set = set()
        with mgf.MGF(mgf_path) as reader:
            for spec in reader:
                params = spec.get('params', {})
                # title: prefer NAME/Title/TITLE/scans, then fallback
                title = (params.get('NAME') or params.get('name') or
                        params.get('Title') or params.get('title') or
                        params.get('TITLE') or params.get('scans') or
                        f"idx_{len(self._records)}")
                title = str(title)
                adduct = (params.get('ADDUCT') or "").strip()
                self._adduct_set.add(adduct)
                charge = 1
                if isinstance(params.get('CHARGE'), (int, float, str)):
                    try:
                        charge = int(str(params.get('CHARGE')).replace('+','').replace('-','-1'))
                    except Exception:
                        pass
                self._charge_set.add(charge)
                mz = torch.tensor(spec.get('m/z array'), dtype=torch.float32)
                inten = torch.tensor(spec.get('intensity array'), dtype=torch.float32)
                self._records.append({"title": title, "mz": mz, "intensity": inten, "params": params})
        adducts, charges = set(), set()
        for rec in self._records:
            a = str(rec["params"].get("ADDUCT") or "").strip()
            if a:
                adducts.add(a)
            ch_raw = rec["params"].get("CHARGE")
            ch_val = None
            if ch_raw is not None:
                try:
                    ch_val = int(str(ch_raw).replace('+','').replace('−','-').replace('–','-').strip())
                except Exception:
                    ch_val = None
            if ch_val is None:
                # simple inference from adduct suffix like [M+H]+
                if a.endswith('+'): ch_val = 1
                elif a.endswith('-'): ch_val = -1
            if ch_val is not None:
                charges.add(ch_val)
        self._formula_vocab = 0
        self._adduct_vocab = 0
        self._charge_vocab = 0
        for k, v in self.meta_map.items():
             self._formula_vocab = max(self._formula_vocab, int(v.get('formula_idx', -1)) + 1)
             self._adduct_vocab = max(self._adduct_vocab, int(v.get('adduct_idx', -1)) + 1)
             self._charge_vocab = max(self._charge_vocab, int(v.get('charge_idx', -1)) + 1)
        self._adduct_to_idx = {a:i for i,a in enumerate(sorted(adducts))}
        self._charge_to_idx = {c:i for i,c in enumerate(sorted(charges))}
        self._adduct_vocab = max(self._adduct_vocab, len(self._adduct_to_idx))
        self._charge_vocab = max(self._charge_vocab, len(self._charge_to_idx))

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx: int):
        rec = self._records[idx]
        title = rec["title"]
        meta_src = self.meta_map.get(_norm_key(title), {})
        # Try to extract SMILES from multiple possible fields
        p = rec['params']
        smiles = (p.get('SMILES') or p.get('smiles') or p.get('Smiles') or 'C')
        a = str(p.get("ADDUCT") or "").strip()
        adduct_idx = (self._adduct_to_idx.get(a) if getattr(self, "_adduct_to_idx", None) else None)

        ch_raw = p.get("CHARGE")
        ch_val = None
        if ch_raw is not None:
            try:
                ch_val = int(str(ch_raw).replace('+','').replace('−','-').replace('–','-').strip())
            except Exception:
                pass
        if ch_val is None:
            if a.endswith('+'): ch_val = 1
            elif a.endswith('-'): ch_val = -1
        charge_idx = (self._charge_to_idx.get(ch_val) if (ch_val is not None and getattr(self, "_charge_to_idx", None)) else None)

        meta = {
            "formula_idx": meta_src.get('formula_idx', None),  # optional: add a formula vocab later
            "adduct_idx": adduct_idx,
            "charge_idx": charge_idx,
            "nce": p.get('COLLISION_ENERGY'),
            "instrument": p.get('INSTRUMENT_TYPE'),
        }
        print(meta)


        return {"mz": rec["mz"], "intensity": rec["intensity"], "title": title, "smiles": smiles, "meta": meta}


def bin_peaks(mz: torch.Tensor, intensity: torch.Tensor, num_bins: int, max_mz: float = 2000.0) -> torch.Tensor:
    device = mz.device
    bins = torch.zeros(num_bins, device=device)
    if mz.numel() == 0:
        return bins
    idx = torch.clamp((mz / max_mz) * num_bins, min=0, max=num_bins - 1e-6).long()
    idx = torch.min(idx, torch.tensor(num_bins - 1, device=device))
    bins.index_add_(0, idx, intensity)
    return bins


def collate_massspecgym(batch: List[dict], spec_bins: int, formula_vocab: int, adduct_vocab: int, charge_vocab: int, fp_bits: int, seed: int = 1337) -> dict:
    device = torch.device('cpu') #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spectra = []
    peaks_list = []
    for ex in batch:
        mz = ex['mz'].to(device)
        inten = ex['intensity'].to(device)
        peaks = torch.stack([mz, inten], dim=-1)
        peaks_list.append(peaks)
        spectra.append(bin_peaks(mz, inten, num_bins=spec_bins))
    spectra = torch.stack(spectra, dim=0)

    B = len(batch)
    def idx_to_onehot(idx_list, K):
        if K <= 0:
            return F.one_hot(torch.randint(0, max(2, B), (B,), device=device), num_classes=max(2, B)).float()
        idx_t = torch.tensor([(-1 if x is None else int(x)) for x in idx_list], device=device)
        idx_t = torch.clamp(idx_t, min=0)
        return F.one_hot(idx_t, num_classes=K).float()

    formula = idx_to_onehot([ex['meta'].get('formula_idx') for ex in batch], formula_vocab)
    adduct  = idx_to_onehot([ex['meta'].get('adduct_idx') for ex in batch], adduct_vocab)
    charge  = idx_to_onehot([ex['meta'].get('charge_idx') for ex in batch], charge_vocab)
    meta = {"formula": formula, "adduct": adduct, "charge": charge, "peaks": torch.nn.utils.rnn.pad_sequence(peaks_list, batch_first=True, padding_value=0.0)}

    smiles = [ex['smiles'] for ex in batch]
    mol_feats = MolFeaturizer(fp_bits=fp_bits).featurize(smiles)

    with torch.no_grad():
        proj = _get_stable_proj(fp_bits, device=device, seed=seed)
        graph_target = F.normalize(mol_feats @ proj, dim=-1)

    return {"spectra": spectra, "mol_feats": mol_feats, "graph_target": graph_target, "meta": meta}


def import_from_path(path: str):
    mod_name, cls_name = path.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def validate_dataset(ds: MassSpecGymDataset) -> None:
    titles = [rec['title'] for rec in ds._records]
    keys = set(ds.meta_map.keys())
    hit = sum(1 for t in titles if _norm_key(t) in keys)
    coverage = 100.0 * hit / max(1, len(titles))
    # Sample smiles distribution
    smiles = []
    for i in range(min(1000, len(ds))):
        ex = ds[i]
        smiles.append(ex['smiles'])
    uniq = len(set(smiles))
    cnt_c = sum(1 for s in smiles if s == 'C')
    print(f"[dataset] MGF records: {len(ds)} | META coverage: {coverage:.1f}% ({hit}/{len(titles)}) | unique SMILES (first 1k): {uniq} | '#C'={cnt_c}")
    if coverage < 50:
        print("[warn] Low META coverage: titles in JSON likely don't match MGF TITLEs. Check normalization or key fields.")
    if uniq <= 2 or cnt_c > 0.5 * len(smiles):
        print("[warn] SMILES look degenerate (many 'C'). Contrastive will sit at ln(B). Ensure meta JSON has per-spectrum SMILES.")


def train_real(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    ds = MassSpecGymDataset(args.mgf, args.meta_json)
    validate_dataset(ds)

    formula_vocab = args.formula_vocab if args.formula_vocab is not None else max(2, getattr(ds, '_formula_vocab', 0) or 32)
    adduct_vocab  = args.adduct_vocab if args.adduct_vocab is not None else max(2, getattr(ds, '_adduct_vocab', 0) or 16)
    charge_vocab  = args.charge_vocab if args.charge_vocab is not None else max(2, getattr(ds, '_charge_vocab', 0) or 8)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_massspecgym(
            b, args.spec_bins, formula_vocab, adduct_vocab, charge_vocab, args.fp_bits, seed=args.seed
        ),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0 and args.persistent_workers),
    )

    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)
    mol_encoder = MolEncoder(d_in=args.fp_bits, embed_dim=512)
    model = DreamsToMolCondition(dreams_backbone, mol_encoder, d_out=args.cond_dim, mapper_hidden=args.mapper_hidden, gaussian=not args.no_gaussian).to(device)
    print("[device]",
      "cuda_available=", torch.cuda.is_available(),
      "device=", device,
      "model=", next(model.parameters()).device,
      "dreams=", next(model.spec.dreams.parameters()).device)

    if args.unfreeze_last > 0:
        model.spec.unfreeze_last(n_layers=args.unfreeze_last)

    if args.decoder_import:
        DecoderClass = import_from_path(args.decoder_import)
        decoder = DecoderClass(cond_dim=args.cond_dim).to(device)
    else:
        decoder = ToyDecoder(cond_dim=args.cond_dim, graph_dim=512).to(device)
    if args.predictor_import:
        PredictorClass = import_from_path(args.predictor_import)
        spec_pred = PredictorClass(graph_dim=512, spec_bins=args.spec_bins).to(device)
    else:
        spec_pred = ToySpecPredictor(graph_dim=512, spec_bins=args.spec_bins).to(device)

    fwd_loss = ForwardSpectralLoss()
    params = list(p for p in model.parameters() if p.requires_grad) + list(decoder.parameters()) + list(spec_pred.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    use_amp = bool(args.amp and torch.cuda.is_available() and not args.cpu)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    base_ln = ln_baseline(args.batch_size)
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            step += 1
            s = batch['spectra'].to(device, non_blocking=True)
            m = batch['mol_feats'].to(device, non_blocking=True)
            gt_graph = batch['graph_target'].to(device, non_blocking=True)
            meta_cpu = batch['meta']
            meta = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in meta_cpu.items()}

            z_s, z_m, z_hat, mu_s, lv_s = model(s, meta, m, inference=False)
            L_align, logs = model.align_losses(z_s, z_m, mu_s, lv_s, w_con=1.0, w_map=1.0, w_ortho=1e-3)

            cond = mix_condition(z_m, z_hat, p=0.5, training=True)
            dec_out = decoder(gt_graph, cond=cond, formula=meta["formula"], adduct=meta["adduct"], charge=meta["charge"])
            L_dec = dec_out["loss"]

            s_pred = spec_pred(dec_out["graph"], meta)
            L_fwd = fwd_loss(s_pred, s)

            with torch.cuda.amp.autocast(enabled=use_amp):
            # forward + loss as-is
                loss = L_align + args.w_dec * L_dec + args.w_fwd * L_fwd
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()

            if step % args.log_every == 0:
                acc1, cos_pos, cos_neg = _inbatch_diag_metrics(z_s, z_m)
                print(
                    f"step {step:05d} | total {loss.item():.4f} | L_con {logs['L_con'].item():.4f} | "
                    f"L_map {logs['L_map'].item():.4f} | L_dec {L_dec.item():.4f} | L_fwd {L_fwd.item():.4f} | "
                    f"acc@1 {acc1.item():.3f} | cos(+) {cos_pos.item():.3f} | cos(-) {cos_neg.item():.3f} | ln(B) {base_ln:.3f}"
                )
            if args.max_steps and step >= args.max_steps:
                print("[done] real training reached max steps.")
                return
    print("[done] real training complete.")


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="run synthetic training demo")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    p.add_argument("--dreams-ckpt", type=str, default=None, help="path to real DreaMS checkpoint (optional)")
    p.add_argument("--spec-bins", type=int, default=2048, help="spectral bins for binned spectra representation")
    p.add_argument("--fp-bits", type=int, default=2048, help="fingerprint bits (Morgan)")
    p.add_argument("--cond-dim", type=int, default=512, help="conditioning embedding dim")
    p.add_argument("--mapper-hidden", type=int, default=0, help="hidden dim for MapperB (0 = linear)")
    p.add_argument("--no-gaussian", action="store_true", help="disable Gaussian uncertainty in MapperB")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--w-dec", type=float, default=1.0)
    p.add_argument("--w-fwd", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    # Real data flags
    p.add_argument("--mgf", type=str, default=None, help="MGF file path for real training")
    p.add_argument("--meta-json", type=str, default=None, help="Optional JSON with per-spectrum metadata and SMILES")
    p.add_argument("--formula-vocab", type=int, default=None, help="Override formula one-hot size")
    p.add_argument("--adduct-vocab", type=int, default=None, help="Override adduct one-hot size")
    p.add_argument("--charge-vocab", type=int, default=None, help="Override charge one-hot size")
    p.add_argument("--epochs", type=int, default=1, help="Epochs for real training")
    p.add_argument("--max-steps", type=int, default=None, help="Optional max steps cutoff for real training")
    p.add_argument("--unfreeze-last", type=int, default=0, help="Unfreeze last N layers in DreaMS backbone")
    p.add_argument("--decoder-import", type=str, default=None, help="Optional 'module:Class' for your diffusion decoder")
    p.add_argument("--predictor-import", type=str, default=None, help="Optional 'module:Class' for your spectrum predictor")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; 0 avoids CUDA/fork issues")
    p.add_argument("--pin-memory", action="store_true", help="Pin host memory for faster H2D copies")
    p.add_argument("--persistent-workers", action="store_true", help="Keep workers alive between epochs (requires num-workers>0)")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (autocast)")

    args = p.parse_args()

    if args.demo and args.mgf is None:
        train_demo(args)
    elif args.mgf is not None:
        train_real(args)
    else:
        # Forward to package CLI for consistency
        try:
            from specbridge.cli import main as cli_main
            cli_main()
        except Exception:
            print("Provide --demo for synthetic or --mgf / --meta-json for real training.")
