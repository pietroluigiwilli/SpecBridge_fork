"""
SpecBridge entry script
----------------------
Thin shim that preserves the legacy script entry-point. For usage, prefer the
installed CLI:

    specbridge --help

This script forwards to the same training functions and remains runnable.
"""
from __future__ import annotations
from typing import Optional, Tuple, List, Iterable, Dict, Any

import argparse
import math
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel

# Use spawn so CUDA works with DataLoader workers on most clusters
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    # It's OK if it's already set by another import in the process
    pass

from specbridge.utils.common import set_seed, unit_normalize, ln_baseline,_topk_hard_neg_indices, hard_inbatch_nce, _load_cand_map, _build_formula_bank,_formula_of, EarlyStopper

from specbridge.adapters.dreams_adapter import DummyDreams, load_dreams_encoder, DreamsAdapter
from specbridge.models.mol import MolFeaturizer, MolEncoder, MolAdapter
from specbridge.data.massspecgym import MassSpecGymDataset, collate_massspecgym, validate_dataset, bin_peaks
from specbridge.models.mapper import MapperB
from specbridge.losses.contrastive import InfoNCELoss, _embed_smiles_list, isomer_ce
from specbridge.losses.supcon import supcon_loss
from specbridge.losses.forward import ForwardSpectralLoss
from specbridge.predictors.toy import ToySpecPredictor
from specbridge.data.sampler import BalancedBatchSampler, ReplicateBatchSampler

# ================================================================
# Conditioning mix helper
# ================================================================

def mix_condition(z_m_true: torch.Tensor, z_m_hat: torch.Tensor, p: float = 0.5, training: bool = True) -> torch.Tensor:
    """Stochastically mix ground-truth and predicted mol embeddings.
    Keeps the adapter honest while enabling teacher-forced conditioning.
    """
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
# Toy decoder (API-compatible surface)
# ================================================================

from specbridge.decoders.toy import ToyDecoder


@torch.no_grad()
def run_val(model, dl_val, args) -> dict:
    model.eval()
    device = next(model.parameters()).device  # <<< was missing

    totals = {'total': 0.0, 'L_con': 0.0, 'L_con_m': 0.0, 'L_sup': 0.0, 'L_map': 0.0}
    count = 0

    for bi, batch in enumerate(dl_val):
        if bi >= args.val_batches:
            break

        # mirror train: move to device
        s = batch['spectra'].to(device, non_blocking=True)
        m = batch['mol_feats'].to(device, non_blocking=True)
        meta = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch['meta'].items()}

        # forward -> embeddings
        z_s, z_m, z_hat, mu_s, lv_s = model(s, meta, m, inference=True)
        acc1, cos_pos, cos_neg = _inbatch_diag_metrics(z_s, z_m)
        # keys for SupCon (if present)
        keys = meta.get("smi_key", None)

        # alignment losses (no raw 'smiles' arg)
        L_align, logs = model.align_losses(
            z_s, z_m, mu_s, lv_s,
            w_con=args.w_con,
            w_con_mapped=args.w_con_mapped,
            w_map=args.w_map,
            w_ortho=args.w_ortho,
            supcon_keys=keys,
            w_sup=args.w_supcon,
            sup_temp=args.supcon_temp,
            stop_mol_in_con=not args.allow_mol_update_in_con,
        )

        totals['total'] += float(L_align.item())
        for k in ('L_con','L_con_m','L_sup','L_map'):
            if k in logs:
                totals[k] += float(logs[k])
        count += 1

    for k in totals:
        totals[k] /= max(count, 1)
    return {f'val_{k}': v for k, v in totals.items()}, acc1


# ================================================================
# End-to-end adapter wrapper
# ================================================================

class DreamsToMolCondition(nn.Module):
    def __init__(
        self,
        dreams_encoder: nn.Module,
        mol_encoder: nn.Module,
        d_out: int = 512,
        mapper_hidden: int = 0,
        gaussian: bool = True,
        mol_space: str = "adapter",
        chemberta_model: str | None = None,
        fp_bits: int = 2048,
    ):
        super().__init__()
        self.mol_space = mol_space
        self.fp_bits = fp_bits

        # spec branch (frozen backbone, learnable proj already inside DreamsAdapter)
        self.spec = DreamsAdapter(dreams_encoder, d_out=d_out, hidden=0, freeze_backbone=True)

        # mol branch: either learnable adapter (legacy) or frozen pretrained
        if mol_space == "adapter":
            self.mol = MolAdapter(mol_encoder, d_out=d_out, hidden=0)  # trainable
            self.chem_tok = None
            self.chem_mdl = None
            self.chem_proj = None
        elif mol_space == "ecfp":
            # No learnable parameters needed for fixed ECFP space.
            # We may still want a registered buffer if you later add a learned projection,
            # but for now we normalize the raw 0/1 vector and expect cond_dim == fp_bits.
            self.mol = None
            self.chem_tok = None
            self.chem_mdl = None
            self.chem_proj = None
        elif mol_space == "chemberta":
            assert chemberta_model is not None, "Provide --chemberta-model for --mol-space chemberta"
            from transformers import AutoTokenizer, AutoModel
            self.chem_tok = AutoTokenizer.from_pretrained(chemberta_model)
            self.chem_mdl = AutoModel.from_pretrained(chemberta_model)
            for p in self.chem_mdl.parameters():
                p.requires_grad = False  # freeze ChemBERTa
            hid = int(self.chem_mdl.config.hidden_size)
            # Trainable projection into your conditioning space
            self.chem_proj = nn.Linear(hid, d_out)
            self.mol = None
        else:
            raise ValueError(f"Unknown mol_space: {mol_space}")

        # mapper and contrastive loss
        self.mapB = MapperB(d=d_out, hidden=mapper_hidden, gaussian=gaussian)
        self.contrast = InfoNCELoss(temperature=0.07, learnable_temp=True)

    def _chemberta_embed(self, smiles: list[str], device: torch.device) -> torch.Tensor:
        assert self.chem_tok is not None and self.chem_mdl is not None and self.chem_proj is not None
        toks = self.chem_tok(smiles, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():  # model is frozen
            h = self.chem_mdl(**toks).last_hidden_state[:, 0]  # CLS [B, hidden]
        z = self.chem_proj(h)  # trainable projection
        return F.normalize(z, dim=-1)

    def forward(self, spectra_binned, meta, mol_feats, inference: bool = False):
        """
        meta must include a list[str] SMILES under key 'smi_key' (or 'smiles').
        mol_feats is only used when mol_space == 'ecfp' (0/1 fingerprint tensor).
        """
        z_s = self.spec(spectra_binned, meta)

        if self.mol_space == "adapter":
            z_m = self.mol(mol_feats)  # trainable fp->cond
        elif self.mol_space == "ecfp":
            # Expect shape [B, fp_bits]; set cond_dim == fp_bits, or add an explicit learnable proj here if you want.
            z_m = F.normalize(mol_feats.float(), dim=-1)
        elif self.mol_space == "chemberta":
            smiles = meta.get("smi_key", None) or meta.get("smiles", None)
            if smiles is None:
                raise RuntimeError("Need meta['smi_key'] or meta['smiles'] (list[str]) for ChemBERTa embedding.")
            z_m = self._chemberta_embed(smiles, z_s.device)
        else:
            raise RuntimeError("unreachable")

        mu_s, lv_s = self.mapB(z_s)
        z_hat = self.mapB.sample(mu_s, lv_s, deterministic=inference)
        return z_s, z_m, z_hat, mu_s, lv_s

    def align_losses(
        self,
        z_s, z_m, mu_s, lv_s,
        w_con=1.0, w_map=1.0, w_ortho=1e-3, w_con_mapped=1.0,
        stop_mol_in_con=True,
        supcon_keys=None, w_sup=0.0, sup_temp=0.07,
    ):
        z_s_n = F.normalize(z_s, dim=-1)
        z_m_n = F.normalize(z_m, dim=-1)
        mu_n  = F.normalize(mu_s, dim=-1)

        z_m_for_con = z_m_n.detach() if stop_mol_in_con else z_m_n
        L_con   = self.contrast(z_s_n, z_m_for_con)
        L_con_m = self.contrast(mu_n, z_m_n.detach())
        L_map   = F.mse_loss(mu_n, z_m_n.detach())
        L_ortho = self.mapB.orthogonality_penalty() * w_ortho

        L_sup = torch.tensor(0.0, device=z_s.device)
        if (w_sup > 0.0) and (supcon_keys is not None):
            L_sup = supcon_loss(z_s, z_m, supcon_keys, temperature=sup_temp)

        L = (w_con * L_con) + (w_con_mapped * L_con_m) + (w_map * L_map) + L_ortho + (w_sup * L_sup)
        logs = {"L_con": L_con.detach(), "L_con_m": L_con_m.detach(), "L_map": L_map.detach(),
                "L_ortho": L_ortho.detach(), "L_sup": L_sup.detach()}
        return L, logs








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

    # Optional resume
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        try:
            decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
            spec_pred.load_state_dict(ckpt.get("spec_pred", {}), strict=False)
        except Exception:
            pass
        try:
            opt.load_state_dict(ckpt.get("opt", {}))
        except Exception as e:
            print(f"[resume] optimizer state not loaded: {e}")
        start_step = int(ckpt.get("step", 0))

    base_ln = ln_baseline(args.batch_size)

    for step in range(start_step + 1, start_step + args.steps + 1):
        batch = make_synthetic_batch(args.batch_size, args.spec_bins, args.fp_bits, device)
        s = batch["spectra"]; m = batch["mol_feats"]; gt_graph = batch["graph_target"]; meta = batch["meta"]

        z_s, z_m, z_hat, mu_s, lv_s = model(s, meta, m, inference=False)
        L_align, logs = model.align_losses(
            z_s, z_m, mu_s, lv_s,
            w_con=args.w_con, w_map=args.w_map, w_ortho=args.w_ortho,
            w_con_mapped=args.w_con_mapped, stop_mol_in_con=not args.allow_mol_update_in_con
        )

        cond = mix_condition(z_m, z_hat, p=0.5, training=True)
        dec_out = decoder(gt_graph, cond=cond, formula=meta["formula"], adduct=meta["adduct"], charge=meta["charge"])
        L_dec = dec_out["loss"]

        s_pred = spec_pred(dec_out["graph"], meta)
        L_fwd = fwd_loss(s_pred, s)

        loss = L_align + args.w_dec * L_dec + args.w_fwd * L_fwd
        if not torch.isfinite(loss):
            print(
                f"[warn] non-finite loss at step {step}: total={loss.item()} "
                f"L_con={logs['L_con'].item():.4f} L_con_m={logs['L_con_m'].item():.4f} "
                f"L_map={logs['L_map'].item():.4f} L_sup={logs['L_sup'].item():.4f} L_ortho={logs['L_ortho'].item():.6f}"
            )
            opt.zero_grad(set_to_none=True)
            continue

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


def import_from_path(path: str):
    mod_name, cls_name = path.split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)



# ================================================================
# Real training loop (MGF + optional JSON meta)
# ================================================================

def train_real(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)

    # Dataset (fold-aware)
    folds = None
    if args.fold:
        folds = {f.strip().lower() for f in args.fold.split(",") if f.strip()}
    ds = MassSpecGymDataset(args.mgf, args.meta_json, folds=folds)
    print("[dataset]", ds.stats_str())
    validate_dataset(ds)

    # Vocab sizes for one-hot meta
    formula_vocab = args.formula_vocab if args.formula_vocab is not None else max(2, getattr(ds, "_formula_vocab", 0) or 32)
    adduct_vocab  = args.adduct_vocab  if args.adduct_vocab  is not None else max(2, getattr(ds, "_adduct_vocab", 0) or 16)
    charge_vocab  = args.charge_vocab  if args.charge_vocab  is not None else max(2, getattr(ds, "_charge_vocab", 0) or 8)

    # Val split (same vocabs as train; no shuffling)
    ds_val = MassSpecGymDataset(args.mgf, args.meta_json, folds={'val'})
    collate_val = lambda b: collate_massspecgym(
        b, args.spec_bins, formula_vocab, adduct_vocab, charge_vocab, args.fp_bits, seed=args.seed
    )
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory,
                        persistent_workers=False, collate_fn=collate_val)

    # Batching with K replicates per identity
    # sampler = BalancedBatchSampler(ds._records, batch_size=args.batch_size, K=args.K, shuffle=True)
    

    sampler = ReplicateBatchSampler(ds._records, args.batch_size, K=args.supcon_k, seed=args.seed)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=lambda b: collate_massspecgym(
            b, args.spec_bins, formula_vocab, adduct_vocab, charge_vocab, args.fp_bits, seed=args.seed
        ),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0 and args.persistent_workers),
    )

        # Models
    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)
    mol_encoder = MolEncoder(d_in=args.fp_bits, embed_dim=512)

    model = DreamsToMolCondition(
        dreams_backbone,
        mol_encoder,
        d_out=args.cond_dim,
        mapper_hidden=args.mapper_hidden,
        gaussian=not args.no_gaussian,
        mol_space=args.mol_space,
        chemberta_model=getattr(args, "chemberta_model", None),
        fp_bits=args.fp_bits,
    ).to(device)
    larger = args.early_metric in {"val_acc", "val_acc1", "val_acc1_mapped"}
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, larger_is_better=larger)
    best_ckpt_path = os.path.join(args.outdir, "best_val.pt")

    print(
        "[device]",
        "cuda_available=", torch.cuda.is_available(),
        "device=", device,
        "model=", next(model.parameters()).device,
        "dreams=", next(model.spec.dreams.parameters()).device,
    )
    if args.freeze_mol_adapter:
        if getattr(model, "mol", None) is not None:
            for p in model.mol.parameters():
                p.requires_grad = False
            print("[freeze] froze MolAdapter parameters")
        else:
            print(f"[freeze] --freeze-mol-adapter ignored (mol_space='{args.mol_space}' has no MolAdapter)")
            
    if args.unfreeze_last > 0 and args.unfreeze_after == 0:
        model.spec.unfreeze_last(n_layers=args.unfreeze_last)

    # Decoder/predictor (optional, default to Toy*)
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
    cand_map = formula_bank = None
    cand_featurizer = None
    if args.train_candidates:
        cand_map = _load_cand_map(args.train_candidates)
        formula_bank = _build_formula_bank(cand_map, use_rdkit=args.iso_use_rdkit)
        cand_featurizer = MolFeaturizer(fp_bits=args.fp_bits)

    os.makedirs(args.outdir, exist_ok=True)

    # Optimizer / scaler
    fwd_loss = ForwardSpectralLoss()
    params = list(p for p in model.parameters() if p.requires_grad) + list(decoder.parameters()) + list(spec_pred.parameters())
    use_amp = bool(args.amp and torch.cuda.is_available() and not args.cpu)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Optional resume
    start_step = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt.get("model", {}), strict=False)
        try:
            decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
            spec_pred.load_state_dict(ckpt.get("spec_pred", {}), strict=False)
        except Exception:
            pass
        try:
            opt.load_state_dict(ckpt.get("opt", {}))
        except Exception as e:
            print(f"[resume] optimizer state not loaded: {e}")
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] resumed from step {start_step}")

    base_ln = ln_baseline(args.batch_size)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # warmup + cosine
    total_steps = args.epochs * len(loader)
    warmup = int(0.05 * total_steps)  # ~5%
    def lr_lambda(it):
        if it < warmup:
            return (it + 1) / max(1, warmup)
        t = (it - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    step = start_step
    for epoch in range(args.epochs):
        for batch in loader:
            if args.unfreeze_last > 0 and args.unfreeze_after and step == args.unfreeze_after:
                model.spec.unfreeze_last(n_layers=args.unfreeze_last)
                print(f"[adapter] unfroze last {args.unfreeze_last} layer(s) of DreaMS at step {step}")

            step += 1
            s = batch['spectra'].to(device, non_blocking=True)
            m = batch['mol_feats'].to(device, non_blocking=True)
            gt_graph = batch['graph_target'].to(device, non_blocking=True)
            meta_cpu = batch['meta']
            meta = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in meta_cpu.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                # Encodings
                z_s, z_m, z_hat, mu_s, lv_s = model(s, meta, m, inference=False)
                # # Alignment pieces
                # L_con  = model.contrast(z_s, z_m)                     # InfoNCE
                # L_map  = ((mu_s - z_m.detach())**2).mean()            # spec→mol mean regression
                # L_ortho = args.w_ortho * model.mapB.orthogonality_penalty()

                # # Multi-positive SupCon keyed by canonical SMILES (provided by collate)
                keys = meta.get("smi_key", None)                      # list[str], len=B
                # if keys is None:
                #     # Fall back to InfoNCE-only if keys are unavailable
                #     L_sup = torch.tensor(0.0, device=s.device)
                #     w_sup = 0.0
                # else:
                #     L_sup = supcon_loss(z_s, z_m, keys, temperature=args.supcon_temp)
                #     w_sup = args.w_supcon

                # Final alignment loss
                keys = meta.get("smi_key", None)
                # ramp = min(1.0, step / (0.2 * total_steps))  # 20% ramp
                # w_map_eff = args.w_map * ramp
                L_align, logs = model.align_losses(
                    z_s, z_m, mu_s, lv_s,
                    w_con=args.w_con,
                    w_con_mapped=args.w_con_mapped,
                    w_map=args.w_map,
                    w_ortho=args.w_ortho,
                    supcon_keys=keys,
                    w_sup=args.w_supcon,
                    sup_temp=args.supcon_temp,
                    stop_mol_in_con=not args.allow_mol_update_in_con,
                )

                # Optional decoding/predictive losses
                cond = mix_condition(z_m, z_hat, p=0.5, training=True)
                dec_out = decoder(gt_graph, cond=cond, formula=meta["formula"], adduct=meta["adduct"], charge=meta["charge"])
                L_dec = dec_out["loss"]
                s_pred = spec_pred(dec_out["graph"], meta)
                L_fwd = fwd_loss(s_pred, s)
                keys_in_batch = torch.arange(z_s.size(0), device=z_s.device)
                hard_lists = _topk_hard_neg_indices(batch["mol_feats"].to(z_s.device), topk=getattr(args, "hard_topk", 8))

                L_hard = hard_inbatch_nce(mu_s, z_m, keys_in_batch, hard_lists, temperature=getattr(args, "hard_temp", 0.07))

                # 3) Add it with a weight
                logs["L_hard"] = L_hard.detach()
                loss = L_align + args.w_dec * L_dec + args.w_fwd * L_fwd + args.w_hard * L_hard
                # if not args.no_gaussian:
                #     z_samp = model.mapB.sample(mu_s, lv_s, deterministic=False)
                #     L_sample = 1.0 - F.cosine_similarity(F.normalize(z_samp, -1), F.normalize(z_m, -1)).mean()
                #     loss += 0.2 * L_sample
                # loss = L_align + args.w_dec * L_dec + args.w_fwd * L_fwd

            # Guard against NaNs/Infs to keep training alive
            if not torch.isfinite(loss):
                print(
                    f"step {step:05d} | total {loss.item():.4f} | "
                    f"L_con {logs['L_con'].item():.4f} | L_con_m {logs['L_con_m'].item():.4f} | "
                    f"L_map {logs['L_map'].item():.4f} | L_sup {logs['L_sup'].item():.4f} | L_ortho {logs['L_ortho'].item():.6f} | L_hard {logs['L_hard'].item():.6f} |"
                    f"L_dec {L_dec.item():.4f} | L_fwd {L_fwd.item():.4f} | "
                    f"acc@1 {acc1.item():.3f} | cos(+) {cos_pos.item():.3f} | cos(-) {cos_neg.item():.3f} | ln(B) {base_ln:.3f}"
                )
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
            sched.step()


            if step % args.log_every == 0:
                acc1, cos_pos, cos_neg = _inbatch_diag_metrics(z_s, z_m)
                print(
                    f"step {step:05d} | total {loss.item():.4f} | "
                    f"L_con {logs['L_con'].item():.4f} | L_con_m {logs['L_con_m'].item():.4f} | "
                    f"L_map {logs['L_map'].item():.4f} | L_sup {logs['L_sup'].item():.4f} | L_ortho {logs['L_ortho'].item():.6f} | L_hard {logs['L_hard'].item():.6f} | "
                    f"L_dec {L_dec.item():.4f} | L_fwd {L_fwd.item():.4f} | "
                    f"acc@1 {acc1.item():.3f} | cos(+) {cos_pos.item():.3f} | cos(-) {cos_neg.item():.3f} | ln(B) {base_ln:.3f}"
                )
            if args.early_stop and (step % args.val_every == 0):
                val_metrics, acc_val = run_val(model, dl_val, args)
                metric_key = args.early_metric       # e.g., 'val_total'
                if metric_key == 'val_acc':
                    current = acc_val
                else:
                    current = val_metrics[metric_key]
                improved = stopper.step(current)
                print(f"[val] step {step} | {metric_key}={current:.4f} | "
                    f"{'IMPROVED' if improved else f'no-improve ({stopper.bad}/{args.patience})'}")

                if improved:
                    torch.save({'model': model.state_dict(), 'args': vars(args), 'step': step},
                            best_ckpt_path)
                    exits = False

                if stopper.should_stop():
                    print(f"[early-stop] patience exhausted at step {step}. "
                        f"Best {metric_key} so far={(-stopper.best if args.early_metric.startswith('val_') else stopper.best):.4f}")
                    exits = True
                    break

                model.train()  # re-enable dropout/bn



            if args.save_every and step % args.save_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "decoder": decoder.state_dict(),
                    "spec_pred": spec_pred.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "args": vars(args),
                }, os.path.join(args.outdir, f"ckpt_{step:06d}.pt"))

            if args.max_steps and step >= args.max_steps:
                print("[done] real training reached max steps.")
                return
        if exits:
            break
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
    p.add_argument("--w-dec", type=float, default=0.0)
    p.add_argument("--w-fwd", type=float, default=0.0)
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
    p.add_argument("--unfreeze-after", type=int, default=0, help="Unfreeze dreams after N steps (0=immediate/no-op)")
    p.add_argument("--outdir", type=str, default="runs/specbridge")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--fold", type=str, default='train', help="Filter by MGF FOLD (e.g., 'train', 'val', 'test' or comma-separated)")
    p.add_argument("--w-supcon", type=float, default=1.0, help="weight for multi-positive SupCon(z_s,z_m)")
    p.add_argument("--supcon-temp", type=float, default=0.07, help="temperature for SupCon")
    p.add_argument("--w-con", type=float, default=1.0, help="weight for InfoNCE alignment")
    p.add_argument("--w-map", type=float, default=5.0, help="weight for spec→mol mean regression")
    p.add_argument("--w-ortho", type=float, default=1e-3, help="weight for mapper orthogonality penalty")
    p.add_argument("--K", type=int, default=4, help="#replicates per identity in each batch (BalancedBatchSampler)")
    p.add_argument("--w-con-mapped", type=float, default=1.0,
                help="weight for InfoNCE(mu_s, z_m)")
    p.add_argument("--allow-mol-update-in-con", action="store_true",
                help="if set, do NOT detach z_m in the z_s vs z_m InfoNCE")
    p.add_argument("--supcon-k", type=int, default=4, help="replicates per SMILES per batch")

    p.add_argument("--freeze-mol-adapter", action="store_true")
    p.add_argument("--w-hard", type=float, default=1.0, help="weight for hard-negative in-batch CE on mu_s vs z_m")
    p.add_argument("--hard-topk", type=int, default=8, help="#hard negatives per anchor from fingerprint similarity")
    p.add_argument("--hard-temp", type=float, default=0.07, help="temperature for hard-negative CE")
    p.add_argument("--train-candidates", type=str, default=None,
                help="PKL {true_smi: [cand_smiles,...]} to mine same-formula negatives")
    p.add_argument("--w-iso", type=float, default=1.0, help="weight for same-formula isomer CE")
    p.add_argument("--iso-k", type=int, default=8, help="# same-formula negatives per anchor")
    p.add_argument("--iso-temp", type=float, default=0.07, help="temperature for isomer CE")
    p.add_argument("--iso-use-rdkit", action="store_true",
                help="derive formula from SMILES with RDKit; else fall back to dataset string formula")
    p.add_argument("--mol-space", choices=["ecfp", "chemberta", "adapter"], default="ecfp")
    p.add_argument("--chemberta-model", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument('--val-every', type=int, default=1000, help='steps between val checks')
    p.add_argument('--val-batches', type=int, default=64, help='number of mini-batches to sample from val each check')
    p.add_argument('--early-stop', action='store_true', help='enable early stopping on validation')
    p.add_argument('--patience', type=int, default=10, help='number of val checks with no improvement before stop')
    p.add_argument('--min-delta', type=float, default=0.0, help='required improvement (absolute) to reset patience')
    p.add_argument('--early-metric', choices=['val_total','val_L_con_m','val_L_con','val_sup', 'val_acc'],
                    default='val_acc', help='which val metric to monitor (smaller is better)')


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