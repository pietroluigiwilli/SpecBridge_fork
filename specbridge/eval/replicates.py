# specbridge/eval/replicates.py
from __future__ import annotations
import argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from specbridge.adapters.dreams_adapter import load_dreams_encoder, DreamsAdapter
from specbridge.data.massspecgym import MassSpecGymDataset, collate_massspecgym

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser("Spec-to-spec replicate retrieval (same SMILES)")
    ap.add_argument("--mgf", required=True)
    ap.add_argument("--dreams-ckpt", type=str, default=None)
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--fold", type=str, default="test")
    ap.add_argument("--cpu", action="store_true")
    
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ds = MassSpecGymDataset(args.mgf, meta_json=None, folds={args.fold})
    if args.limit:
        limit_n = min(args.limit, len(ds))
        ds = torch.utils.data.Subset(ds, range(limit_n))

    # quick collate
    formula_vocab = max(2, getattr(ds, "_formula_vocab", 0) or 32)
    adduct_vocab  = max(2, getattr(ds, "_adduct_vocab", 0) or 16)
    charge_vocab  = max(2, getattr(ds, "_charge_vocab", 0) or 8)
    def coll(b): 
        out = collate_massspecgym(b, args.spec_bins, formula_vocab, adduct_vocab, charge_vocab, fp_bits=2048, seed=1337)
        out["smiles"] = [ex["smiles"] for ex in b]
        return out
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=coll)

    # spec encoder
    dreams = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)
    spec = DreamsAdapter(dreams, d_out=512, hidden=0, freeze_backbone=True).to(device).eval()

    Z, SMI = [], []
    for batch in dl:
        s = batch["spectra"].to(device)
        meta = {k: (v.to(device) if torch.is_tensor(v) else v) for k,v in batch["meta"].items()}
        z = spec(s, meta)      # [B,D]
        Z.append(F.normalize(z, dim=-1).cpu())
        SMI.extend(batch["smiles"])
    Z = torch.cat(Z, dim=0)     # [N,D]

    # brute-force similarity & ranks among spectra
    sims = Z @ Z.T              # [N,N]
    N = sims.size(0)
    ranks = []
    for i in range(N):
        sim_i = sims[i].clone()
        sim_i[i] = -1e9  # remove self
        order = torch.argsort(sim_i, descending=True)
        # find first neighbor with same SMILES
        r = N
        for rnk, j in enumerate(order):
            if SMI[i] == SMI[int(j)]:
                r = rnk
                break
        ranks.append(r)

    # metrics
    n = len(ranks)
    r1 = sum(1 for r in ranks if r < 1)/n
    r5 = sum(1 for r in ranks if r < 5)/n
    r10 = sum(1 for r in ranks if r < 10)/n
    mrr = sum(1/(r+1) for r in ranks)/n
    med = float(sorted(ranks)[n//2])
    print(f"[replicate] N={n}  R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f}  MRR={mrr:.3f}  median_rank={med:.1f}")

if __name__ == "__main__":
    main()
