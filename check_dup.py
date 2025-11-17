#!/usr/bin/env python
from __future__ import annotations
import argparse, os, sys
from typing import Iterable, Set, Optional, List
from datasets import load_dataset
# --- Try RDKit canonicalization (safer set keys); else whitespace-strip only
try:
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

def canon_smi(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return s
    if not _HAS_RDKIT:
        return s
    mol = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol) if mol is not None else s

def canon_set(smiles_iter: Iterable[str]) -> Set[str]:
    out = set()
    for s in smiles_iter:
        if not s: 
            continue
        cs = canon_smi(s) or s.strip()
        out.add(cs)
    return out

def load_massspecgym_test(mgf_path: str, meta_json: Optional[str]) -> List[str]:
    # Uses your repo loader
    from specbridge.data.massspecgym import MassSpecGymDataset
    ds = MassSpecGymDataset(mgf_path, meta_json, folds={'train'})
    smi = []
    for rec in ds._records:  # each rec has rec['smiles']
        s = rec.get('smiles', None)
        if s:
            smi.append(s)
    return smi

def load_derify(name: str, split: str = "train") -> List[str]:
    """
    name: "Derify/augmented_canonical_druglike_QED_43M" or "Derify/druglike"
    tries common smile(s) column names.
    """
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    # guess the SMILES column
    for col in ("canonical_smiles", "SMILES", "canonical_smiles", "canonical", "smi"):
        if col in ds.column_names:
            return [s for s in ds[col] if isinstance(s, str) and s]
    # last resort: try to find a string column with typical SMILES chars
    guess = None
    for col in ds.column_names:
        if isinstance(ds[0][col], str):
            guess = col; break
    if guess is None:
        raise RuntimeError(f"Could not find a SMILES-like column in {name}. Columns: {ds.column_names}")
    return [s for s in ds[guess] if isinstance(s, str) and s]

def main():
    ap = argparse.ArgumentParser("Check overlap of MassSpecGym TEST SMILES with Derify datasets")
    ap.add_argument("--mgf", required=True, type=str)
    ap.add_argument("--meta-json", type=str, default=None)
    ap.add_argument("--derify-split", type=str, default="train",
                    help="HF split to read (most Derify sets only have 'train')")
    ap.add_argument("--sample-derify", type=int, default=None,
                    help="Optional, subsample first N Derify molecules for quicker testing")
    args = ap.parse_args()

    # Load MassSpecGym test
    ms_test = load_massspecgym_test(args.mgf, args.meta_json)
    ms_test_set = set(ms_test)
    print(f"[MassSpecGym] test SMILES: {len(ms_test_set)} unique after canon")

    # Load Derify big set
    d1 = load_derify("Derify/augmented_canonical_pubchem_13m", split=args.derify_split)
    if args.sample_derify:
        d1 = d1[:args.sample_derify]
    d1_set = set(d1)
    # Load Derify/druglike
    d2 = load_derify("Derify/druglike", split=args.derify_split)
    if args.sample_derify:
        d2 = d2[:args.sample_derify]
    d2_set = set(d2)

    # Overlaps
    inter_1 = ms_test_set & d1_set
    inter_2 = ms_test_set & d2_set

    print("\n=== Overlap Report ===")
    print(f"MassSpecGym test unique:         {len(ms_test_set):,}")
    print(f"Derify/QED_43M unique:           {len(d1_set):,} (sampled: {args.sample_derify or 'no'})")
    print(f"Derify/druglike unique:          {len(d2_set):,} (sampled: {args.sample_derify or 'no'})")
    print(f"Overlap with QED_43M:            {len(inter_1):,}  ({len(inter_1)/max(1,len(ms_test_set)):.2%} of test)")
    print(f"Overlap with druglike:           {len(inter_2):,}  ({len(inter_2)/max(1,len(ms_test_set)):.2%} of test)")
    print(f"Overlap with either (union):     {len((inter_1 | inter_2)):,}  ({len((inter_1|inter_2))/max(1,len(ms_test_set)):.2%} of test)")

if __name__ == "__main__":
    main()
