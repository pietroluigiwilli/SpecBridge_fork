from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import torch
import torch.nn.functional as F

from specbridge.utils.common import pget

# Optional SMILES canonicalization
try:
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

# Cache for stable graph projection across batches
_PROJ_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}

def _get_stable_proj(fp_bits: int, device: torch.device, seed: int = 1337) -> torch.Tensor:
    key = (fp_bits, device.type)
    if key not in _PROJ_CACHE:
        g = torch.Generator(device=device).manual_seed(seed)
        _PROJ_CACHE[key] = torch.randn(fp_bits, 512, generator=g, device=device)
    return _PROJ_CACHE[key]

def _norm_key(x: Any) -> str:
    return str(x).strip().lower()

_FOLD_ALIASES = {
    "valid": "val",
    "validation": "val",
}

def _norm_fold(x: str | None) -> str | None:
    if not x:
        return None
    f = str(x).strip().lower()
    return _FOLD_ALIASES.get(f, f)

def _canon_smiles(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s
    s = s.strip()
    if not s or not _HAS_RDKIT:
        return s
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m is not None else s

_CHARGE_RE = re.compile(r"([+-])(\d*)$")  # matches "+", "-", "2+", "3-", etc.

def _parse_charge(params: Dict[str, Any], adduct: Optional[str]) -> Optional[int]:
    """
    Return signed integer charge if available.
    Priority: CHARGE field -> sign in ADDUCT -> None
    """
    ch = params.get('CHARGE') or params.get('Charge') or params.get('charge')
    if ch is not None:
        # CHARGE can be like "2+" or "-1" or "1" or 1
        try:
            if isinstance(ch, (int, float)):
                return int(ch)
            chs = str(ch).strip()
            m = _CHARGE_RE.search(chs)
            if m:
                sign, mag = m.group(1), m.group(2)
                mag = int(mag) if mag else 1
                return mag if sign == '+' else -mag
            # plain int string
            return int(float(chs))
        except Exception:
            pass
    if adduct:
        m = _CHARGE_RE.search(adduct.strip())
        if m:
            sign, mag = m.group(1), m.group(2)
            mag = int(mag) if mag else 1
            return mag if sign == '+' else -mag
    return None

class MassSpecGymDataset(torch.utils.data.Dataset):
    """
    MGF-backed dataset that reads per-spectrum metadata directly from the file:
      - SMILES, FORMULA, ADDUCT, CHARGE (or derived from ADDUCT), COLLISION_ENERGY, INSTRUMENT_TYPE, FOLD
    If a meta_json is provided, it can override by TITLE (kept for compatibility),
    but it's not required.
    """
    def __init__(self, mgf_path: str, meta_json: Optional[str] = None, folds: Optional[set[str]] = None):
        super().__init__()
        try:
            from pyteomics import mgf  # type: ignore
        except Exception as e:
            raise ImportError(f"pyteomics is required to read MGF: {e}")

        # Optional override map (by title) if provided
        self.meta_map: Dict[str, dict] = {}
        if meta_json is not None:
            with open(meta_json, 'r') as f:
                meta_obj = json.load(f)
            # tolerate list or dict
            if isinstance(meta_obj, dict):
                self.meta_map = { _norm_key(k): v for k, v in meta_obj.items() }
            elif isinstance(meta_obj, list):
                candidate_keys = ["title", "spectrum_id", "scan", "id", "name"]
                for rec in meta_obj:
                    if isinstance(rec, dict):
                        key_val = None
                        for ck in candidate_keys:
                            if ck in rec:
                                key_val = rec[ck]; break
                        if key_val is not None:
                            self.meta_map[_norm_key(key_val)] = rec

        # Read MGF and harvest metadata
        self._records: List[dict] = []
        self.fold_counts: Dict[str, int] = {}
        formula_set: set[str] = set()
        adduct_set: set[str] = set()
        charge_set: set[int] = set()

        with mgf.MGF(mgf_path) as reader:
            for spec in reader:
                params = spec.get('params', {})
                title = params.get('title') or params.get('TITLE') or params.get('scans') or f"idx_{len(self._records)}"

                smiles = params.get('SMILES') or params.get('smiles')
                # smiles = _canon_smiles(smiles) or 'C'  # minimal fallback

                formula = params.get('FORMULA') or params.get('formula')
                adduct  = params.get('ADDUCT') or params.get('adduct')
                instr   = params.get('INSTRUMENT_TYPE') or params.get('instrument') or params.get('Instrument')
                try:
                    nce = params.get('COLLISION_ENERGY') or params.get('collision_energy')
                    nce = float(nce) if nce is not None else None
                except Exception:
                    nce = None

                chg = _parse_charge(params, adduct)

                fold = _norm_fold(params.get('FOLD') or params.get('fold') or params.get('Fold'))

                rec = {
                    "title": title,
                    "mz": torch.tensor(spec.get('m/z array'), dtype=torch.float32),
                    "intensity": torch.tensor(spec.get('intensity array'), dtype=torch.float32),
                    "params": params,
                    "fold": fold,
                    "smiles": smiles,
                    "formula": formula,
                    "adduct": adduct,
                    "charge": chg,
                    "nce": nce,
                    "instrument": instr,
                    "smi_key": smiles,
                }
                # print(rec)
                self._records.append(rec)

                if formula: formula_set.add(str(formula))
                if adduct:  adduct_set.add(str(adduct))
                if chg is not None: charge_set.add(int(chg))
                # break
        # Optional fold filtering
        if folds:
            folds = { _norm_fold(f) for f in folds }
            self._records = [r for r in self._records if r["fold"] in folds]

        # Stats & vocab
        self.fold_counts = {}
        for r in self._records:
            f = r["fold"] or "unknown"
            self.fold_counts[f] = self.fold_counts.get(f, 0) + 1

        self._formula_list = sorted(formula_set) if formula_set else []
        self._adduct_list  = sorted(adduct_set) if adduct_set else []
        self._charge_list  = sorted(charge_set) if charge_set else []

        self._formula_to_idx = {f: i for i, f in enumerate(self._formula_list)}
        self._adduct_to_idx  = {a: i for i, a in enumerate(self._adduct_list)}
        self._charge_to_idx  = {c: i for i, c in enumerate(self._charge_list)}

        self._formula_vocab = max(0, len(self._formula_list))
        self._adduct_vocab  = max(0, len(self._adduct_list))
        self._charge_vocab  = max(0, len(self._charge_list))

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx: int):
        rec = self._records[idx]
        title = rec["title"]
        meta_src = self.meta_map.get(title, {})
        # NEW: fallback to MGF params if meta_json doesn't provide SMILES
        smiles = meta_src.get("smiles", None)
        if not smiles:
            smiles = rec["params"].get("SMILES") or rec["params"].get("smiles") or "C"
        smiles = _canon_smiles(smiles) or smiles  # canonicalize if RDKit present
        meta = {
            "formula_idx": meta_src.get('formula_idx', None),
            "adduct_idx": meta_src.get('adduct_idx', None),
            "charge_idx": meta_src.get('charge_idx', None),
            "nce": meta_src.get('nce', rec["params"].get("COLLISION_ENERGY") or rec["params"].get("collision_energy")),
            "instrument": meta_src.get('instrument', rec["params"].get("INSTRUMENT_TYPE") or rec["params"].get("instrument_type")),
            "fold": rec["fold"],
            "smi_key": smiles,
        }
        return {"mz": rec["mz"], "intensity": rec["intensity"], "title": title, "smiles": smiles, "meta": meta}


    def stats_str(self) -> str:
        parts = [f"{k}={v}" for k, v in sorted(self.fold_counts.items())]
        return (
            f"records={len(self)} | " + " ".join(parts) +
            f" | formulas={self._formula_vocab} | adducts={self._adduct_vocab} | charges={self._charge_vocab}"
        )

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
    device = torch.device('cpu')
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
    adduct  = idx_to_onehot([ex['meta'].get('adduct_idx') for ex in batch],  adduct_vocab)
    charge  = idx_to_onehot([ex['meta'].get('charge_idx') for ex in batch],  charge_vocab)

    meta = {
        "formula": formula,
        "adduct": adduct,
        "charge": charge,
        "peaks": torch.nn.utils.rnn.pad_sequence(peaks_list, batch_first=True, padding_value=0.0),
    }
    meta["smi_key"] = [ex["meta"]["smi_key"] for ex in batch]   # list[str], length B
    # Molecule features
    from specbridge.models.mol import MolFeaturizer
    smiles = [ex['smiles'] for ex in batch]
    mol_feats = MolFeaturizer(fp_bits=fp_bits).featurize(smiles)

    # Proxy graph target (stable across batches for the same SMILES)
    with torch.no_grad():
        proj = _get_stable_proj(fp_bits, device=device, seed=seed)
        graph_target = torch.nn.functional.normalize(mol_feats @ proj, dim=-1)

    return {"spectra": spectra, "mol_feats": mol_feats, "graph_target": graph_target, "meta": meta}

def validate_dataset(ds: MassSpecGymDataset) -> None:
    # Purely descriptive; no reliance on meta_json
    n = len(ds)
    n_preview = min(2000, n)
    smiles = [ds[i]['smiles'] for i in range(n_preview)]
    uniq = len(set(smiles))
    cnt_c = sum(1 for s in smiles if s == 'C')
    print(f"[dataset] {ds.stats_str()} | preview_unique_smiles={uniq} | '#C'={cnt_c}")
    if uniq <= 2 or cnt_c > 0.5 * n_preview:
        print("[warn] SMILES look degenerate (many 'C'). Ensure your MGF has SMILES headers.")

