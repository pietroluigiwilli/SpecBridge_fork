#!/usr/bin/env python3
"""
Benchmark script to measure ChemBERTa embedding time per molecule.

Measures:
1. Time to embed a single molecule (single inference)
2. Time to embed molecules in batches (batched inference)
3. Throughput (molecules per second)
"""

import argparse
import time
import statistics
import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def benchmark_single_molecule_embedding(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    smiles: str,
    num_warmup: int = 10,
    num_trials: int = 100,
) -> dict:
    """
    Benchmark time to embed a single molecule.
    
    Args:
        model: ChemBERTa model
        tokenizer: ChemBERTa tokenizer
        device: torch device
        smiles: SMILES string to embed
        num_warmup: number of warmup runs
        num_trials: number of trial runs
    
    Returns:
        Dictionary with timing statistics
    """
    print(f"\n[Benchmark] Single molecule embedding")
    print(f"  SMILES: {smiles}")
    print(f"  Warmup runs: {num_warmup}, Trial runs: {num_trials}")
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            toks = tokenizer([smiles], padding=True, truncation=True, return_tensors="pt").to(device)
            _ = model(**toks).last_hidden_state[:, 0]
    
    # Synchronize GPU if available
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark trials
    times = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_trials), desc="Single molecule trials"):
            # Measure embedding time
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            toks = tokenizer([smiles], padding=True, truncation=True, return_tensors="pt").to(device)
            h = model(**toks).last_hidden_state[:, 0]
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
    
    stats = {
        "num_trials": len(times),
        "time_ms": {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "std": statistics.stdev(times) if len(times) > 1 else 0.0,
            "min": min(times),
            "max": max(times),
        },
        "throughput_molecules_per_sec": 1000 / statistics.mean(times),
    }
    
    print(f"  Mean time: {stats['time_ms']['mean']:.3f} ± {stats['time_ms']['std']:.3f} ms per molecule")
    print(f"  Median time: {stats['time_ms']['median']:.3f} ms per molecule")
    print(f"  Throughput: {stats['throughput_molecules_per_sec']:.1f} molecules/second")
    
    return stats


def benchmark_batched_embedding(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: torch.device,
    smiles_list: List[str],
    batch_size: int = 32,
    num_warmup: int = 3,
    num_trials: int = 10,
) -> dict:
    """
    Benchmark time to embed molecules in batches.
    
    Args:
        model: ChemBERTa model
        tokenizer: ChemBERTa tokenizer
        device: torch device
        smiles_list: List of SMILES strings to embed
        batch_size: Batch size for embedding
        num_warmup: number of warmup batches
        num_trials: number of trial batches
    
    Returns:
        Dictionary with timing statistics
    """
    print(f"\n[Benchmark] Batched molecule embedding")
    print(f"  Number of molecules: {len(smiles_list)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Warmup batches: {num_warmup}, Trial batches: {num_trials}")
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for i in range(num_warmup):
            batch = smiles_list[i * batch_size:(i + 1) * batch_size]
            if not batch:
                break
            toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            _ = model(**toks).last_hidden_state[:, 0]
    
    # Synchronize GPU if available
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark trials
    batch_times = []
    per_molecule_times = []
    
    with torch.no_grad():
        for trial in tqdm(range(num_trials), desc="Batched trials"):
            start_idx = (trial * batch_size) % len(smiles_list)
            end_idx = min(start_idx + batch_size, len(smiles_list))
            batch = smiles_list[start_idx:end_idx]
            
            if not batch:
                break
            
            # Measure batch embedding time
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            toks = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            h = model(**toks).last_hidden_state[:, 0]
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            batch_time_ms = (end - start) * 1000
            batch_times.append(batch_time_ms)
            per_molecule_times.append(batch_time_ms / len(batch))
    
    stats = {
        "batch_size": batch_size,
        "num_trials": len(batch_times),
        "batch_time_ms": {
            "mean": statistics.mean(batch_times),
            "median": statistics.median(batch_times),
            "std": statistics.stdev(batch_times) if len(batch_times) > 1 else 0.0,
            "min": min(batch_times),
            "max": max(batch_times),
        },
        "per_molecule_time_ms": {
            "mean": statistics.mean(per_molecule_times),
            "median": statistics.median(per_molecule_times),
            "std": statistics.stdev(per_molecule_times) if len(per_molecule_times) > 1 else 0.0,
            "min": min(per_molecule_times),
            "max": max(per_molecule_times),
        },
        "throughput_molecules_per_sec": 1000 / statistics.mean(per_molecule_times),
    }
    
    print(f"  Batch time: {stats['batch_time_ms']['mean']:.2f} ± {stats['batch_time_ms']['std']:.2f} ms")
    print(f"  Per-molecule time: {stats['per_molecule_time_ms']['mean']:.3f} ± {stats['per_molecule_time_ms']['std']:.3f} ms")
    print(f"  Throughput: {stats['throughput_molecules_per_sec']:.1f} molecules/second")
    
    return stats


def main():
    ap = argparse.ArgumentParser(description="Benchmark ChemBERTa embedding time")
    ap.add_argument(
        "--chemberta-model",
        type=str,
        default="Derify/ChemBERTa_augmented_pubchem_13m",
        help="ChemBERTa model name or path"
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Auto-detects if not specified"
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for batched benchmark"
    )
    ap.add_argument(
        "--num-molecules",
        type=int,
        default=1000,
        help="Number of molecules to use for batched benchmark"
    )
    ap.add_argument(
        "--num-trials-single",
        type=int,
        default=100,
        help="Number of trials for single molecule benchmark"
    )
    ap.add_argument(
        "--num-trials-batch",
        type=int,
        default=10,
        help="Number of trials for batched benchmark"
    )
    ap.add_argument(
        "--skip-single",
        action="store_true",
        help="Skip single molecule benchmark"
    )
    ap.add_argument(
        "--skip-batch",
        action="store_true",
        help="Skip batched benchmark"
    )
    
    args = ap.parse_args()
    
    # Determine device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("ChemBERTa Embedding Time Benchmark")
    print("=" * 80)
    print(f"Model: {args.chemberta_model}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\n[Loading] ChemBERTa model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.chemberta_model)
    model = AutoModel.from_pretrained(args.chemberta_model).to(device)
    model.eval()
    print(f"  Model loaded. Hidden size: {model.config.hidden_size}")
    
    results = {}
    
    # Benchmark 1: Single molecule embedding
    if not args.skip_single:
        test_smiles = "CCO"  # Ethanol
        results["single"] = benchmark_single_molecule_embedding(
            model, tokenizer, device, test_smiles,
            num_trials=args.num_trials_single
        )
    
    # Benchmark 2: Batched embedding
    if not args.skip_batch:
        # Generate a list of test SMILES (simple molecules)
        test_smiles_list = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "CCCCCCCCCC",  # Decane
            "c1ccccc1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=CC=C(C=C1)C(C)CC(=O)O",  # Ibuprofen variant
            "C1=CC=CC=C1",  # Benzene (different representation)
            "CCCC",  # Butane
            "CC(C)O",  # Isopropanol
        ] * (args.num_molecules // 10 + 1)  # Repeat to get enough molecules
        test_smiles_list = test_smiles_list[:args.num_molecules]
        
        results["batched"] = benchmark_batched_embedding(
            model, tokenizer, device, test_smiles_list,
            batch_size=args.batch_size,
            num_trials=args.num_trials_batch
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if "single" in results:
        single = results["single"]
        print(f"\nSingle Molecule Embedding:")
        print(f"  Time: {single['time_ms']['mean']:.3f} ± {single['time_ms']['std']:.3f} ms per molecule")
        print(f"  Throughput: {single['throughput_molecules_per_sec']:.1f} molecules/second")
    
    if "batched" in results:
        batch = results["batched"]
        print(f"\nBatched Embedding (batch_size={batch['batch_size']}):")
        print(f"  Time: {batch['per_molecule_time_ms']['mean']:.3f} ± {batch['per_molecule_time_ms']['std']:.3f} ms per molecule")
        print(f"  Throughput: {batch['throughput_molecules_per_sec']:.1f} molecules/second")
        if "single" in results:
            speedup = results["single"]["time_ms"]["mean"] / batch["per_molecule_time_ms"]["mean"]
            print(f"  Speedup vs single: {speedup:.2f}x")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
