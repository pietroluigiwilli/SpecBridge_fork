import math
import random
import torch
import torch.nn.functional as F
from typing import List

def set_seed(seed: int = 1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unit_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def ln_baseline(batch_size: int) -> float:
    return float(math.log(max(1, batch_size)))


def inbatch_diag_metrics(z_s: torch.Tensor, z_m: torch.Tensor):
    with torch.no_grad():
        from .common import unit_normalize as _norm  # self-import safe
        z_s = _norm(z_s)
        z_m = _norm(z_m)
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

def pget(p: dict, *names, default=None):
    """Robust param getter: tries given names + their lower/upper variants."""
    for n in names:
        if n in p: return p[n]
        nl, nu = n.lower(), n.upper()
        if nl in p: return p[nl]
        if nu in p: return p[nu]
    return default

@torch.no_grad()
def _topk_hard_neg_indices(mol_feats: torch.Tensor, topk: int = 8) -> List[List[int]]:
    """
    For each row i in mol_feats [B, F], return a list of indices for the top-k
    most similar *other* molecules (hard negatives) using cosine similarity in fingerprint space.
    """
    B = mol_feats.size(0)
    # [B,B] cosine similarity (fingerprint proxy for "chemical closeness")
    S = F.normalize(mol_feats, dim=-1) @ F.normalize(mol_feats, dim=-1).T
    S.fill_diagonal_(-1.0)  # exclude self
    hard_idx = torch.topk(S, k=min(topk, max(1, B - 1)), dim=1).indices  # [B, K]
    return [hard_idx[i].tolist() for i in range(B)]


def hard_inbatch_nce(z_query: torch.Tensor, z_keys: torch.Tensor,
                     pos_indices: torch.Tensor,
                     hard_lists: List[List[int]],
                     temperature: float = 0.07) -> torch.Tensor:
    """
    For each anchor i, build logits against its positive (pos_indices[i]) and
    a *subset* of hard negatives defined by hard_lists[i]. Compute CE.
    z_query, z_keys are [B, D] (e.g., z_query = z_s or mu_s; z_keys = z_m).
    pos_indices maps i -> i (if pairs are aligned 1:1); pass a tensor just for clarity.
    """
    zq = F.normalize(z_query, dim=-1)
    zk = F.normalize(z_keys,  dim=-1)
    B, D = zq.shape
    losses = []
    for i in range(B):
        pos_j = int(pos_indices[i].item())
        cand_js = [pos_j] + [j for j in hard_lists[i] if j != pos_j]
        Z = zk[cand_js]                           # [1+K, D]
        sims = (zq[i:i+1] @ Z.T) / temperature    # [1, 1+K]
        target = torch.tensor([0], device=zq.device)  # pos is at index 0
        losses.append(F.cross_entropy(sims, target))
    return torch.stack(losses).mean()

# Canonicalize SMILES (safe fallback if RDKit missing)
try:
    from rdkit import Chem
    def _canon_smi(s):
        if not s: return s
        return s
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s.strip()
    def _formula_of(s):
        m = Chem.MolFromSmiles(s)
        if not m: return None
        from rdkit.Chem import rdMolDescriptors as rdmd
        return rdmd.CalcMolFormula(m)
except Exception:
    def _canon_smi(s): return s.strip() if s else s
    def _formula_of(s): return None  # will fall back to meta formula

def _load_cand_map(path: str):
    import pickle
    with open(path, "rb") as f:
        raw = pickle.load(f)  # {true_smi: [cand_smiles,...]}
    out = {}
    for k, vs in raw.items():
        ck = _canon_smi(k)
        out[ck] = [ _canon_smi(v) for v in vs if v ]
    return out

def _build_formula_bank(cand_map, use_rdkit=True):
    # Map formula -> set of candidate SMILES (for sampling isomer negatives)
    bank = {}
    for t_smi, lst in cand_map.items():
        if use_rdkit and _formula_of(t_smi):
            f = _formula_of(t_smi)
        else:
            # fallback: key by string formula from dataset; we’ll pass it at call time
            f = None
        for s in [t_smi] + lst:
            fs = _formula_of(s) if use_rdkit else None
            if fs is None:  # leave open; we’ll index by fallback formula later
                continue
            bank.setdefault(fs, set()).add(s)
    return {k: sorted(v) for k, v in bank.items()}

class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0, larger_is_better: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.sign = 1.0 if larger_is_better else -1.0
        self.best = None
        self.bad = 0

    def step(self, value: float) -> bool:
        score = self.sign * value
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.bad = 0
            return True  # improved
        self.bad += 1
        return False

    def should_stop(self) -> bool:
        return self.bad >= self.patience

