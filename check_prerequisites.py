#!/usr/bin/env python3
"""
Quick check script to verify prerequisites for running SpecBridge on all_spectra.mgf
"""
import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file exists and report its size."""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ {description}: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"✗ {description}: {path} (NOT FOUND)")
        return False

def check_mgf_format(mgf_path):
    """Check if MGF file has required fields."""
    print(f"\nChecking MGF file format: {mgf_path}")
    try:
        from pyteomics import mgf
        with mgf.MGF(mgf_path) as reader:
            count = 0
            has_smiles = 0
            has_formula = 0
            has_adduct = 0
            for spec in reader:
                count += 1
                if count > 100:  # Sample first 100 spectra
                    break
                params = spec.get('params', {})
                if params.get('SMILES') or params.get('smiles'):
                    has_smiles += 1
                if params.get('FORMULA') or params.get('formula'):
                    has_formula += 1
                if params.get('ADDUCT') or params.get('adduct'):
                    has_adduct += 1
            
            print(f"  Sampled {count} spectra")
            print(f"  Spectra with SMILES: {has_smiles}/{count} ({100*has_smiles/count:.1f}%)")
            print(f"  Spectra with FORMULA: {has_formula}/{count} ({100*has_formula/count:.1f}%)")
            print(f"  Spectra with ADDUCT: {has_adduct}/{count} ({100*has_adduct/count:.1f}%)")
            
            if has_smiles == 0:
                print("  ⚠ WARNING: No SMILES found in MGF headers!")
                print("     For TRAINING: You'll need a metadata JSON file with SMILES")
                print("     For EVALUATION: You'll need candidate files with SMILES")
            return has_smiles > 0, has_formula > 0
    except Exception as e:
        print(f"  ✗ Error reading MGF: {e}")
        return False, False

def main():
    base_dir = Path("/cluster/tufts/liulab/yiwan01/SpecBridge")
    data_dir = base_dir / "data"
    
    print("=" * 70)
    print("SpecBridge Prerequisites Check for all_spectra.mgf")
    print("=" * 70)
    
    # Check MGF file
    mgf_path = data_dir / "all_spectra.mgf"
    has_mgf = check_file(mgf_path, "MGF file")
    
    if has_mgf:
        has_smiles, has_formula = check_mgf_format(mgf_path)
    
    # Check DreaMS checkpoint
    dreams_ckpt = data_dir / "ssl_model.ckpt"
    has_dreams = check_file(dreams_ckpt, "DreaMS checkpoint")
    
    # Check for candidate files
    print("\nChecking for candidate files:")
    candidate_files = [
        "cand_dict_merged.pkl",
        "cand_dict_large_smiles.pkl",
        "candidates_test_val.pkl",
    ]
    found_candidates = []
    for cand_file in candidate_files:
        cand_path = data_dir / cand_file
        if check_file(cand_path, f"Candidate file ({cand_file})"):
            found_candidates.append(cand_path)
    
    # Check for adapter checkpoints (for evaluation)
    print("\nChecking for adapter checkpoints (for evaluation):")
    runs_dir = base_dir / "runs"
    if runs_dir.exists():
        ckpt_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if ckpt_dirs:
            print(f"  Found {len(ckpt_dirs)} run directories")
            for ckpt_dir in ckpt_dirs[:5]:  # Show first 5
                ckpts = list(ckpt_dir.glob("ckpt_*.pt"))
                if ckpts:
                    print(f"    {ckpt_dir.name}: {len(ckpts)} checkpoints")
        else:
            print("  No run directories found")
    else:
        print("  ✗ runs/ directory not found")
    
    # Summary and recommendations
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    if not has_mgf:
        print("✗ MGF file not found. Cannot proceed.")
        return
    
    if not has_dreams:
        print("✗ DreaMS checkpoint not found. Download from:")
        print("  https://zenodo.org/records/10997887")
        return
    
    print("\nFor EVALUATION:")
    if found_candidates:
        print("✓ You have candidate files. You can run evaluation with:")
        print(f"  python -m specbridge.eval.candidates \\")
        print(f"    --mgf {mgf_path} \\")
        print(f"    --dreams-ckpt {dreams_ckpt} \\")
        print(f"    --adapter-ckpt <path_to_adapter_checkpoint> \\")
        print(f"    --candidates {found_candidates[0]} \\")
        print(f"    --batch-size 32 --cond-dim 2048 --mapper-hidden 2048 \\")
        print(f"    --mol-space chemberta --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m")
    else:
        print("⚠ No candidate files found for all_spectra.mgf")
        print("  Generate candidates first:")
        print(f"  python {data_dir / 'build_pubchem_candidates.py'} \\")
        print(f"    {mgf_path} all_spectra_candidates.pkl \\")
        print(f"    --max-per 200 --ppm 10 --workers 16")
    
    print("\nFor TRAINING:")
    if not has_smiles:
        print("⚠ MGF file lacks SMILES in headers")
        print("  Option 1: Add SMILES to MGF headers (SMILES=... in each spectrum)")
        print("  Option 2: Create a metadata JSON file mapping spectrum titles to SMILES")
        print("  Option 3: Use a different MGF file that includes SMILES")
    else:
        print("✓ MGF file has SMILES. You can run training with:")
        print(f"  python dreams_condition_adapter.py \\")
        print(f"    --mgf {mgf_path} \\")
        print(f"    --dreams-ckpt {dreams_ckpt} \\")
        print(f"    --fold train --batch-size 128 --epochs 2 \\")
        print(f"    --cond-dim 2048 --mapper-hidden 2048 \\")
        print(f"    --outdir runs/specbridge_all_spectra")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()





