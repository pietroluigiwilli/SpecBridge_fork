from __future__ import annotations
import os, re, json, argparse, csv
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# --- RDKit validity check (optional but helpful) ---
try:
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

# --- SpecBridge pieces (your code) ---
from specbridge.adapters.dreams_adapter import load_dreams_encoder
from specbridge.models.mapper import DreamsToMolCondition
from specbridge.data.massspecgym import MassSpecGymDataset
from torch.utils.data import DataLoader

# ---------------- MolGPT bits (repo must be on PYTHONPATH) ----------------
# We keep imports loose because repo structures vary slightly.
def _import_molgpt():
    # Typical layout: model defines GPT/GPTConfig, a sample() function in a utils or trainer
    from molgpt.train.model import GPT, GPTConfig
    # sample() function is usually in model or a utils; we implement a minimal sampler if missing.
    from molgpt.train.utils import sample
    return GPT, GPTConfig, sample
    # except Exception:
    #     # Fallback nanoGPT-style top-k sampler that calls model(idx, prop=...)
    #     def _sample(model, idx, max_new_tokens, temperature=1.0, top_k=None, prop=None):
    #         model.eval()
    #         device = idx.device
    #         for _ in range(max_new_tokens):
    #             idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
    #             with torch.no_grad():
    #                 logits = model(idx_cond, prop=prop)  # [B,T,V]
    #                 logits = logits[:, -1, :] / max(1e-8, temperature)
    #                 if top_k is not None:
    #                     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    #                     logits[logits < v[:, [-1]]] = -float('Inf')
    #                 probs = torch.softmax(logits, dim=-1)
    #                 next_token = torch.multinomial(probs, num_samples=1)  # [B,1]
    #             idx = torch.cat((idx, next_token), dim=1)
    #         return idx
    #     return GPT, GPTConfig, _sample

GPT, GPTConfig, sample = _import_molgpt()

# ---------------- Tokenizer / vocab ----------------
_SMILES_PATTERN = r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
_REGEX = re.compile(_SMILES_PATTERN)

def load_vocab(vocab_json: Optional[str]) -> Tuple[Dict[str,int], Dict[int,str]]:
    if vocab_json:
        with open(vocab_json, "r") as f:
            mapping = json.load(f)
        # allow either {"stoi":{...},"itos":[...]} or a flat {"token":id,...}
        if "stoi" in mapping and "itos" in mapping:
            stoi = {k:int(v) for k,v in mapping["stoi"].items()}
            itos = {int(i):s for i,s in enumerate(mapping["itos"])}
        else:
            stoi = {k:int(v) for k,v in mapping.items()}
            itos = {v:k for k,v in stoi.items()}
        return stoi, itos
    raise RuntimeError("No vocab provided in ckpt and --vocab-json not set.")

def encode_smiles(smiles: str, stoi: Dict[str,int]) -> List[int]:
    toks = _REGEX.findall(smiles)
    return [stoi[t] for t in toks if t in stoi]

def decode_tokens(tok_ids: List[int], itos: Dict[int,str]) -> str:
    s = ''.join(itos.get(int(i), '') for i in tok_ids)
    # Many MolGPT setups use '<' as BOS/SEP marker. Strip it like you did.
    return s.replace('<', '')

# ---------------- Utility ----------------
def unit_norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(1e-9))

# ---------------- Build Spec→mapped embedding ----------------
def build_specbridge_mapper(args, device) -> DreamsToMolCondition:
    dreams = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)
    model = DreamsToMolCondition(
        dreams, d_out=args.cond_dim, mapper_hidden=args.mapper_hidden,
        gaussian=True if not args.no_gaussian else False,
        mol_space="chemberta", chemberta_model=args.chemberta_model, args=args
    ).to(device).eval()
    if args.adapter_ckpt and os.path.isfile(args.adapter_ckpt):
        sd = torch.load(args.adapter_ckpt, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)
    return model

# ---------------- Load MolGPT model ----------------
def load_molgpt(m_ckpt: str, m_config_json: Optional[str], vocab_json: Optional[str], device: torch.device):
    sd = torch.load(m_ckpt, map_location="cpu")
    # find state_dict and config
    if isinstance(sd, dict) and any(k in sd for k in ("state_dict","model","model_state","model_state_dict")):
        state = sd.get("state_dict", None) or sd.get("model", None) or sd.get("model_state", None) or sd.get("model_state_dict", None)
    elif isinstance(sd, dict) and all(k.endswith(".weight") or k.endswith(".bias") for k in sd.keys()):
        state = sd
    else:
        state = sd

    # try config inside ckpt
    cfg = None
    for key in ("config","model_args","hparams"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            cfg = sd[key]; break
    if cfg is None and m_config_json:
        with open(m_config_json, "r") as f:
            cfg = json.load(f)
    if cfg is None:
        raise RuntimeError("MolGPT config not found in ckpt; pass --molgpt-config pointing to JSON with GPTConfig args.")

    model = GPT(GPTConfig(**cfg))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[molgpt] loaded with missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()

    # vocab
    if "stoi" in sd and "itos" in sd:
        stoi = {k:int(v) for k,v in sd["stoi"].items()}
        itos = {int(i):s for i,s in enumerate(sd["itos"])}
    else:
        stoi, itos = load_vocab(vocab_json)

    return model, stoi, itos, cfg


# ---------------- Data loader ----------------
def _collate_with_smiles(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed):
    from specbridge.data.massspecgym import collate_massspecgym as base
    out = base(base_batch, spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits, seed=seed)
    out["titles"] = [ex["title"] for ex in base_batch]
    out["smiles_true"] = [ex["smiles"] for ex in base_batch]
    return out

# ---------------- Main generation loop ----------------
def main():
    ap = argparse.ArgumentParser("Generate SMILES from mass spectra using MolGPT conditioned on mapped embeddings")
    # SpecBridge / dataset
    ap.add_argument("--mgf", required=True, type=str)
    ap.add_argument("--meta-json", type=str, default=None)
    ap.add_argument("--fold-query", type=str, default="test")
    ap.add_argument("--dreams-ckpt", type=str, required=True)
    ap.add_argument("--adapter-ckpt", type=str, required=True)
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=2048)
    ap.add_argument("--mapper-hidden", type=int, default=2048)
    ap.add_argument("--no-gaussian", action="store_true")
    ap.add_argument("--chemberta-model", type=str, default="Derify/ChemBERTa_augmented_pubchem_13m")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)

    # MolGPT
    ap.add_argument("--molgpt-ckpt", type=str, required=True)
    ap.add_argument("--molgpt-config", type=str, default=None, help="JSON with GPTConfig kwargs if ckpt lacks them")
    ap.add_argument("--vocab-json", type=str, default=None, help="JSON with stoi/itos if ckpt lacks them")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--samples-per-spec", type=int, default=1)
    ap.add_argument("--start-token", type=str, default="C", help="seed context (e.g., 'C')")

    # Output
    ap.add_argument("--out-csv", type=str, required=True)

    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    folds = None
    if args.fold_query:
        folds = {f.strip().lower() for f in args.fold_query.split(",") if f.strip()}
    ds = MassSpecGymDataset(args.mgf, args.meta_json, folds=folds)
    formula_vocab = max(2, getattr(ds, "_formula_vocab", 0) or 32)
    adduct_vocab  = max(2, getattr(ds, "_adduct_vocab", 0) or 16)
    charge_vocab  = max(2, getattr(ds, "_charge_vocab", 0) or 8)
    collate = lambda b: _collate_with_smiles(b, args.spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits=2048, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    # SpecBridge mapper (to produce mapped embeddings)
    mapper = build_specbridge_mapper(args, device)

    # MolGPT + vocab
    molgpt, stoi, itos, cfg = load_molgpt(args.molgpt_ckpt, args.molgpt_config, args.vocab_json, device)

    # Infer prop dim expected by MolGPT
    prop_dim = getattr(molgpt.config, "prop_size", None) or getattr(molgpt.config, "prop_dim", None)
    if prop_dim is None:
        # Try to sniff from an attribute used inside model
        prop_dim = getattr(molgpt, "prop_dim", None)
    if prop_dim is None:
        raise RuntimeError("Could not infer MolGPT property dimension. Put it in the config JSON as 'prop_size' or 'prop_dim'.")

    # Prepare BOS from regex-seeded context
    def make_seed_batch(bsz: int) -> torch.Tensor:
        toks = _REGEX.findall(args.start_token)
        idx = torch.tensor([[stoi[t] for t in toks if t in stoi]], dtype=torch.long)
        return idx.repeat(bsz, 1)

    # Generate
    out_rows = []
    for batch in tqdm(dl):
        s = batch["spectra"].to(device)
        meta = {k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch["meta"].items()}
        # Forward mapper (no mol feats at inference)
        with torch.no_grad():
            z_s, z_m, z_hat, mu, lv = mapper(s, meta, None, inference=True)
            prop = z_m # [B, cond_dim]
            prop = prop.to(device)

        B = s.size(0)
        # For each spectrum, sample N sequences independently (batched for speed).
        # We chunk into groups to keep VRAM in check.
        per = args.samples_per_spec
        groups = 1
        for i in tqdm(range(B)):
            all_smiles = []
            for g in range(groups):
                bsz = 16 if (g < groups - 1) else (per - 16 * (groups - 1))
                if bsz <= 0: break
                x0 = make_seed_batch(bsz).to(device)
                # Repeat the prop row bsz times
                p = prop[i:i+1].repeat(bsz, 1)
                y = sample(
                    molgpt, x0, molgpt.config.block_size,
                    temperature=args.temperature, top_k=args.top_k, prop=p
                )  # [bsz, T]
                # decode + clean
                for r in y:
                    smi = decode_tokens(r.tolist(), itos)
                    if _HAS_RDKIT:
                        m = Chem.MolFromSmiles(smi)
                        if m is None: continue
                        smi = Chem.MolToSmiles(m)  # canon
                    all_smiles.append(smi)

            # Deduplicate & keep a handful
            uniq = list(dict.fromkeys(all_smiles))
            out_rows.append({
                "title": batch["titles"][i],
                "smiles_true": Chem.MolToSmiles(Chem.MolFromSmiles(batch["smiles_true"][i])),
                "n_gen": len(all_smiles),
                "n_valid_uniq": len(uniq),
                "samples": ";".join(uniq[:50]),
            })
        # break

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["title","smiles_true","n_gen","n_valid_uniq","samples"])
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[done] wrote {len(out_rows)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
