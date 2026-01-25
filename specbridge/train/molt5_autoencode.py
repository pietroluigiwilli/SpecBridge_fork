# specbridge/train/molt5_autoencode.py
"""
Train MolT5 to auto-encode SMILES from a conditioning embedding (ChemBERTa or projected)
using MassSpecGym.mgf folds, same style as your other scripts.

Examples
--------
# Train on train fold (ChemBERTa embedding as condition), save checkpoint
python -m specbridge.train.molt5_autoencode \
  --mgf path/to/your/data.mgf \
  --fold-query train \
  --epochs 3 --batch-size 64 \
  --mode chem --chemberta-model seyonec/ChemBERTa-zinc-base-v1 \
  --t5 laituan245/molt5-small --prompt-len 10 --freeze-lm \
  --ckpt-out runs/molt5_chem_auto.pt

# Eval on test fold using the trained checkpoint
python -m specbridge.train.molt5_autoencode \
  --mgf path/to/your/data.mgf \
  --fold-query test \
  --mode chem --chemberta-model seyonec/ChemBERTa-zinc-base-v1 \
  --t5 laituan245/molt5-small --prompt-len 10 --freeze-lm \
  --ckpt-in runs/molt5_chem_auto.pt --eval-only
"""
from __future__ import annotations
import os, argparse, math, random
from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel
from specbridge.data.massspecgym import MassSpecGymDataset
from specbridge.models.generatorT5 import SoftPromptT5

# ---------------- Utilities ----------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------- Collate: we only need SMILES ----------------

@dataclass
class CollateSmiles:
    def __call__(self, batch):
        # Each item from MassSpecGymDataset is a record dict. We only need SMILES strings.
        out = []
        for ex in batch:
            s = ex.get("smiles", None)
            if s:
                s = s.strip()
                if s:
                    out.append(s)
        return out

# ---------------- Embedders ----------------

class ChemEmbedder(nn.Module):
    """
    Frozen text encoder for molecules:
      - BERT-like (ChemBERTa): CLS or mean-pool (we default to mean-pool if pooler is absent)
      - T5-like (MolT5 encoder): encoder-only mean-pool
    """
    def __init__(self, model_name: str, pool: str = "mean"):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # Ensure pad token exists for batching
        if getattr(self.tok, "pad_token", None) is None and getattr(self.tok, "eos_token", None) is not None:
            self.tok.pad_token = self.tok.eos_token
        self.mdl = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        for p in self.mdl.parameters():
            p.requires_grad = False
        self.mdl.eval()
        self.pool = pool

    @torch.no_grad()
    def forward(self, smiles: List[str], device: torch.device) -> torch.Tensor:
        toks = self.tok(smiles, padding=True, truncation=True, return_tensors="pt").to(device)
        # is_encdec = getattr(self.mdl.config, "is_encoder_decoder", False)
        self.mdl.to(device)
        with torch.no_grad(): 
            out = self.mdl(**toks)
        h = out.last_hidden_state[:, 0]  # [B, T, H]
        # prefer mean-pool with mask
        return  F.normalize(h, dim=-1)

class ProjectedEmbedder(nn.Module):
    """
    Wraps ChemEmbedder and applies a FROZEN projection taken from your SpecBridge checkpoint
    (keys: 'chem_proj.weight' / 'chem_proj.bias'). If not found, uses identity (pad/crop disabled here).
    """
    def __init__(self, chem_model: str, ckpt_path: Optional[str]):
        super().__init__()
        self.chem = ChemEmbedder(chem_model)
        self.proj: Optional[nn.Linear] = None

        W = b = None
        if ckpt_path and os.path.isfile(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            st = sd.get("gen", None)  # allow loading from gen-only ckpt
            # If this is a SpecBridge training ckpt, projection will be under 'model'
            if st is None:
                st = sd.get("model", sd)
            W = st.get("chem_proj.weight", None) if isinstance(st, dict) else None
            b = st.get("chem_proj.bias", None) if isinstance(st, dict) else None

        if isinstance(W, torch.Tensor):
            out_dim, in_dim = W.size(0), W.size(1)
            self.proj = nn.Linear(in_dim, out_dim, bias=(b is not None))
            with torch.no_grad():
                self.proj.weight.copy_(W)
                if b is not None:
                    self.proj.bias.copy_(b)
            for p in self.proj.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def forward(self, smiles: List[str], device: torch.device) -> torch.Tensor:
        z = self.chem(smiles, device)
        if self.proj is not None:
            z = self.proj(z)
        return z

# ---------------- Metrics ----------------

def exact_match(pred: List[str], gold: List[str]) -> float:
    good = 0
    for p, g in zip(pred, gold):
        good += (p.strip() == g.strip())
    return good / max(1, len(gold))

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser("MolT5 auto-encoding of SMILES conditioned on ChemBERTa (or projected) embeddings, using MassSpecGym folds.")
    # Data
    ap.add_argument("--mgf", type=str, required=True, help="MassSpecGym .mgf path")
    ap.add_argument("--meta-json", type=str, default=None, help="Optional JSON with per-spectrum metadata (including SMILES)")
    ap.add_argument("--fold-query", type=str, default=None, help="Which fold(s) to load as the dataset for this run, e.g. 'train' or 'test' (comma-separated ok)")
    # Conditioning mode
    ap.add_argument("--mode", choices=["chem", "proj"], default="chem", help="chem=ChemBERTa/MolT5 encoder pooled; proj=frozen projection from SpecBridge ckpt")
    ap.add_argument("--chemberta-model", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    ap.add_argument("--ckpt-proj", type=str, default=None, help="SpecBridge ckpt providing chem_proj.* (used when --mode=proj)")
    # Decoder (MolT5)
    ap.add_argument("--t5", type=str, default="laituan245/molt5-small")
    ap.add_argument("--prompt-len", type=int, default=10)
    ap.add_argument("--freeze-lm", action="store_true", help="freeze the T5 weights; train only soft prompt + mapper")
    # Training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pin-memory", action="store_true")
    ap.add_argument("--max-len", type=int, default=128, help="generation length / filter long SMILES (dataset)")
    # Checkpoints
    ap.add_argument("--ckpt-out", type=str, default=None, help="Where to save trained MolT5 (soft prompt + mapper)")
    ap.add_argument("--ckpt-in", type=str, default=None, help="Load a trained MolT5 checkpoint (for eval or resume)")
    # Eval
    ap.add_argument("--eval-only", action="store_true", help="Do not train; just evaluate on the provided fold")
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--sample", action="store_true", help="sampling instead of beam search during eval")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Dataset (fold-aware; consistent with your other scripts)
    folds = None
    if args.fold_query:
        folds = {f.strip().lower() for f in args.fold_query.split(",") if f.strip()}
    ds = MassSpecGymDataset(args.mgf, args.meta_json, folds=folds)

    # A simple DataLoader that only returns lists[str] of SMILES
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(not args.eval_only),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=CollateSmiles(),
    )

    # -------- Condition providers
    if args.mode == "chem":
        embedder: nn.Module = ChemEmbedder(args.chemberta_model)
    else:
        embedder = ProjectedEmbedder(args.chemberta_model, args.ckpt-proj if hasattr(args, "ckpt-proj") else args.ckpt_proj)

    # Infer cond_dim by a probe forward
    with torch.no_grad():
        probe_batch = ["CCO", "CCN"]
        cond_dim = int(embedder(probe_batch, device).size(1))

    # -------- MolT5 decoder (soft prompt + mapper)
    gen = SoftPromptT5(
        lm_name=args.t5,
        cond_dim=cond_dim,
        prompt_len=args.prompt_len,
        hidden=0,
        freeze_lm=args.freeze_lm,
        use_selfies=False,
    ).to(device)

    # Optional: resume/eval load
    if args.ckpt_in and os.path.isfile(args.ckpt_in):
        state = torch.load(args.ckpt_in, map_location="cpu")
        if "gen" in state:
            gen.load_state_dict(state["gen"], strict=False)
            print(f"[load] Loaded gen from {args.ckpt_in}")
        else:
            # gracefully handle loading a raw state_dict
            gen.load_state_dict(state, strict=False)
            print(f"[load] Loaded state_dict from {args.ckpt_in}")

    # -------- Eval function
    @torch.no_grad()
    def run_eval(split_name="eval") -> float:
        gen.eval()
        preds, golds = [], []
        for smiles_batch in dl:
            if len(smiles_batch) == 0:
                continue
            z = embedder(smiles_batch, device)
            out = gen.generate(
                z,
                max_new_tokens=args.max_len,
                num_beams=args.num_beams if not args.sample else 1,
                do_sample=args.sample,
                num_return_sequences=1,
            )
            preds.extend(out)
            golds.extend(smiles_batch)
        em = exact_match(preds, golds)
        print(f"[{split_name}] EM={em:.4f}  n={len(golds)}")
        return em

    # -------- If eval-only, just evaluate this fold and exit
    if args.eval_only:
        run_eval(split_name=("test" if (folds == {"test"}) else "eval"))
        return

    # -------- Train
    trainable = [p for p in gen.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps = args.epochs * max(1, len(dl))
    warmup = int(0.05 * steps)
    def lr_lambda(it):
        if it < warmup:
            return (it + 1) / max(1, warmup)
        t = (it - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    it = 0
    gen.train()
    for epoch in range(1, args.epochs + 1):
        for smiles_batch in dl:
            if len(smiles_batch) == 0:
                continue
            it += 1
            z = embedder(smiles_batch, device)
            out = gen(z, tgt_text=smiles_batch, max_src_len=32, max_tgt_len=args.max_len)
            loss = out["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()

            if it % 50 == 0:
                print(f"epoch {epoch:03d} it {it:06d} | loss {loss.item():.4f}")

    # Save final (optional)
    if args.ckpt_out:
        to_save = {
            "gen": gen.state_dict(),
            "args": vars(args),
        }
        torch.save(to_save, args.ckpt_out)
        print(f"[save] wrote {args.ckpt_out}")

    # (Optional) quick eval on the same fold you trained on (useful sanity check)
    run_eval(split_name=("train" if (folds == {"train"}) else "eval"))

if __name__ == "__main__":
    main()
