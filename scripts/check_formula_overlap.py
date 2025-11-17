#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check for formula overlap between massspecgym candidate dictionary and spectraverse dataset.
If formulas overlap, we can reuse candidates from massspecgym for spectraverse entries.
"""

import pickle
import argparse
from collections import defaultdict
from typing import Dict, List, Set

# Try to import RDKit for formula calculation
try:
    import os
    import sys
    # Set RDBASE if not already set (needed for some RDKit installations)
    if 'RDBASE' not in os.environ:
        # Try common conda environment paths
        possible_paths = [
            '/cluster/tufts/liulab/yiwan01/miniconda3/envs/specbridge',
            os.path.expanduser('~/miniconda3/envs/specbridge'),
            os.path.expanduser('~/anaconda3/envs/specbridge'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['RDBASE'] = path
                break
    
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as rdmd
    RDKIT_AVAILABLE = True
    
    def get_formula_from_smiles(smi: str) -> str:
        """Get molecular formula from SMILES."""
        try:
            if not smi or not smi.strip():
                return None
            mol = Chem.MolFromSmiles(smi.strip())
            if mol is None:
                return None
            return rdmd.CalcMolFormula(mol)
        except Exception as e:
            return None
except ImportError as e:
    print(f"WARNING: RDKit not available: {e}")
    print("Will try to extract formulas from MGF headers instead.")
    RDKIT_AVAILABLE = False
    def get_formula_from_smiles(smi: str) -> str:
        return None

def load_massspecgym_candidates(pkl_path: str) -> Dict[str, List[str]]:
    """Load massspecgym candidate dictionary."""
    print(f"Loading massspecgym candidates from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        cand_dict = pickle.load(f)
    print(f"Loaded {len(cand_dict)} entries")
    return cand_dict

def build_formula_to_candidates(cand_dict: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """Build a mapping from formula to set of candidate SMILES."""
    formula_to_cands = defaultdict(set)
    total_processed = 0
    total_failed = 0
    
    print("Extracting formulas from massspecgym candidates...")
    for true_smi, cand_smiles in cand_dict.items():
        formula = get_formula_from_smiles(true_smi)
        if formula:
            # Add all candidate SMILES for this formula
            for cand_smi in cand_smiles:
                if cand_smi:
                    formula_to_cands[formula].add(cand_smi)
            total_processed += 1
        else:
            total_failed += 1
    
    print(f"Processed {total_processed} entries, {total_failed} failed to get formula")
    print(f"Found {len(formula_to_cands)} unique formulas")
    return formula_to_cands

def extract_formulas_from_mgf(mgf_path: str) -> Set[str]:
    """Extract formulas from MGF file headers."""
    formulas = set()
    print(f"Extracting formulas from {mgf_path}...")
    
    with open(mgf_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line_upper = line.strip().upper()
            if line_upper.startswith('FORMULA='):
                formula = line.split('=', 1)[1].strip()
                if formula:
                    formulas.add(formula)
    
    print(f"Found {len(formulas)} unique formulas in MGF file")
    return formulas

def extract_formulas_from_candidates(pkl_path: str) -> Set[str]:
    """Extract formulas from spectraverse candidate pickle (if it has formula info)."""
    # This would require loading the MGF to get formulas, so we'll use MGF directly
    return set()

def main():
    ap = argparse.ArgumentParser(description="Check formula overlap between massspecgym and spectraverse")
    ap.add_argument("massspecgym_pkl", type=str, help="Path to massspecgym cand_dict_large_form.pkl")
    ap.add_argument("spectraverse_mgf", type=str, help="Path to spectraverse MGF file")
    ap.add_argument("--output", type=str, default=None, help="Output path for formula-to-candidates mapping (pickle)")
    args = ap.parse_args()
    
    # Load massspecgym candidates
    msgym_cand_dict = load_massspecgym_candidates(args.massspecgym_pkl)
    
    # Build formula-to-candidates mapping from massspecgym
    msgym_formula_to_cands = build_formula_to_candidates(msgym_cand_dict)
    
    # Extract formulas from spectraverse
    spectraverse_formulas = extract_formulas_from_mgf(args.spectraverse_mgf)
    
    # Find overlaps
    overlapping_formulas = spectraverse_formulas.intersection(set(msgym_formula_to_cands.keys()))
    msgym_only = set(msgym_formula_to_cands.keys()) - spectraverse_formulas
    spectraverse_only = spectraverse_formulas - set(msgym_formula_to_cands.keys())
    
    print("\n" + "="*60)
    print("OVERLAP ANALYSIS")
    print("="*60)
    print(f"MassSpecGym formulas: {len(msgym_formula_to_cands)}")
    print(f"Spectraverse formulas: {len(spectraverse_formulas)}")
    print(f"Overlapping formulas: {len(overlapping_formulas)}")
    print(f"MassSpecGym only: {len(msgym_only)}")
    print(f"Spectraverse only: {len(spectraverse_only)}")
    
    if overlapping_formulas:
        print(f"\nOverlap percentage: {len(overlapping_formulas) / len(spectraverse_formulas) * 100:.1f}%")
        print(f"\nSample overlapping formulas (first 10):")
        for i, formula in enumerate(sorted(overlapping_formulas)[:10]):
            num_cands = len(msgym_formula_to_cands[formula])
            print(f"  {formula}: {num_cands} candidates available")
        
        # Count total candidate SMILES available for reuse
        total_reusable_cands = sum(len(msgym_formula_to_cands[f]) for f in overlapping_formulas)
        print(f"\nTotal candidate SMILES available for reuse: {total_reusable_cands}")
        
        # Show distribution of candidate counts
        cand_counts = [len(msgym_formula_to_cands[f]) for f in overlapping_formulas]
        if cand_counts:
            print(f"\nCandidate count statistics for overlapping formulas:")
            print(f"  Min: {min(cand_counts)}")
            print(f"  Max: {max(cand_counts)}")
            print(f"  Mean: {sum(cand_counts) / len(cand_counts):.1f}")
            print(f"  Median: {sorted(cand_counts)[len(cand_counts)//2]}")
    else:
        print("\nNo overlapping formulas found!")
    
    # Save formula-to-candidates mapping if requested
    if args.output:
        print(f"\nSaving formula-to-candidates mapping to {args.output}...")
        # Convert sets to lists for pickle compatibility
        output_dict = {f: list(cands) for f, cands in msgym_formula_to_cands.items()}
        with open(args.output, 'wb') as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(output_dict)} formula mappings")
    
    # Also save just the overlapping formulas mapping
    if overlapping_formulas and args.output:
        overlap_output = args.output.replace('.pkl', '_overlap.pkl')
        overlap_dict = {f: list(msgym_formula_to_cands[f]) for f in overlapping_formulas}
        with open(overlap_output, 'wb') as f:
            pickle.dump(overlap_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(overlap_dict)} overlapping formula mappings to {overlap_output}")

if __name__ == "__main__":
    main()

