#!/usr/bin/env python3
"""
Benchmark script to validate inference throughput numbers for SpecBridge.

Measures:
1. Encoding time per spectrum (batched)
2. Retrieval time per query (dot product against pre-computed index)
3. Total throughput (queries per second)

Expected results (from paper):
- Encoding time: 3.2 ms per spectrum (batched)
- Retrieval time: <0.1 ms per query
- Total throughput: ~300 queries per second
"""

from __future__ import annotations
import argparse
import time
import statistics
from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from specbridge.adapters.dreams_adapter import load_dreams_encoder, DreamsAdapter
from specbridge.models.mol import MolFeaturizer, MolEncoder, MolAdapter
from specbridge.models.mapper import MapperB
from specbridge.utils.common import unit_normalize, set_seed
from specbridge.data.massspecgym import MassSpecGymDataset, bin_peaks

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False


def _canon_smiles(smiles: str) -> str:
    s = (smiles or "").strip()
    if not s:
        return ""
    if _HAS_RDKIT:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            return Chem.MolToSmiles(m, canonical=True)
    return s


@torch.no_grad()
def encode_batch(
    batch: List[dict],
    dreams: DreamsAdapter,
    mol_adapter: MolAdapter,
    featurizer: MolFeaturizer,
    spec_bins: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Encode a batch of spectra and molecules."""
    peaks_list, spectra, smiles = [], [], []
    for ex in batch:
        mz = ex["mz"].to(device)
        inten = ex["intensity"].to(device)
        peaks = torch.stack([mz, inten], dim=-1)
        peaks_list.append(peaks)
        spectra.append(bin_peaks(mz, inten, num_bins=spec_bins))
        smiles.append(_canon_smiles(ex.get("smiles", "")))
    spectra = torch.stack(spectra, dim=0)
    # Pad peaks
    max_len = max(p.size(0) for p in peaks_list)
    pad = torch.zeros(len(batch), max_len, 2, device=device)
    for i, p in enumerate(peaks_list):
        L = p.size(0)
        pad[i, :L, :] = p
    meta = {"peaks": pad}
    # Molecule features
    mol_feats = featurizer.featurize(smiles).to(device)
    # Encodings
    z_s = dreams(spectra, meta)  # spec embeddings
    z_m = mol_adapter(mol_feats)  # mol embeddings
    return z_s, z_m, smiles


def benchmark_encoding(
    dataset: MassSpecGymDataset,
    dreams: DreamsAdapter,
    mol_adapter: MolAdapter,
    featurizer: MolFeaturizer,
    spec_bins: int,
    device: torch.device,
    batch_size: int,
    num_warmup: int = 3,
    num_trials: int = 10,
) -> dict:
    """
    Benchmark encoding time per spectrum (batched).
    
    Returns:
        Dictionary with encoding statistics (mean, median, std, per-spectrum times)
    """
    print(f"\n[Benchmark] Encoding time (batch_size={batch_size})")
    print(f"  Warmup batches: {num_warmup}, Trial batches: {num_trials}")
    
    # Warmup
    for i in range(num_warmup):
        batch = [dataset[j] for j in range(i * batch_size, min((i + 1) * batch_size, len(dataset)))]
        if len(batch) == 0:
            break
        encode_batch(batch, dreams, mol_adapter, featurizer, spec_bins, device)
    
    # Synchronize GPU if available
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark trials
    batch_times = []
    per_spectrum_times = []
    
    for trial in range(num_trials):
        start_idx = (trial * batch_size) % len(dataset)
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = [dataset[i] for i in range(start_idx, end_idx)]
        if len(batch) == 0:
            break
        
        # Measure batch encoding time
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        encode_batch(batch, dreams, mol_adapter, featurizer, spec_bins, device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        batch_time_ms = (end - start) * 1000
        batch_times.append(batch_time_ms)
        per_spectrum_times.append(batch_time_ms / len(batch[0]))
    
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
        "per_spectrum_time_ms": {
            "mean": statistics.mean(per_spectrum_times),
            "median": statistics.median(per_spectrum_times),
            "std": statistics.stdev(per_spectrum_times) if len(per_spectrum_times) > 1 else 0.0,
            "min": min(per_spectrum_times),
            "max": max(per_spectrum_times),
        },
    }
    
    print(f"  Batch encoding time: {stats['batch_time_ms']['mean']:.2f} ± {stats['batch_time_ms']['std']:.2f} ms")
    print(f"  Per-spectrum encoding time: {stats['per_spectrum_time_ms']['mean']:.3f} ± {stats['per_spectrum_time_ms']['std']:.3f} ms")
    print(f"  Throughput: {1000 / stats['per_spectrum_time_ms']['mean']:.1f} spectra/second")
    
    return stats


def benchmark_retrieval(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    device: torch.device,
    num_warmup: int = 10,
    num_trials: int = 1000,
) -> dict:
    """
    Benchmark retrieval time per query (dot product against pre-computed gallery).
    
    Args:
        query_embeddings: [N, D] query embeddings (normalized)
        gallery_embeddings: [M, D] gallery embeddings (normalized, pre-computed)
        device: torch device
        num_warmup: number of warmup queries
        num_trials: number of trial queries
    
    Returns:
        Dictionary with retrieval statistics
    """
    print(f"\n[Benchmark] Retrieval time (gallery size={gallery_embeddings.size(0)})")
    print(f"  Warmup queries: {num_warmup}, Trial queries: {num_trials}")
    
    # Normalize embeddings
    query_embeddings = unit_normalize(query_embeddings.to(device))
    gallery_embeddings = unit_normalize(gallery_embeddings.to(device))
    
    # Warmup
    for i in range(min(num_warmup, query_embeddings.size(0))):
        q = query_embeddings[i:i+1]
        sims = (q @ gallery_embeddings.T).cpu()
        _ = torch.topk(sims, k=min(10, gallery_embeddings.size(0)), dim=1)
    
    # Synchronize GPU if available
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark trials
    query_times = []
    
    for trial in range(num_trials):
        idx = trial % query_embeddings.size(0)
        q = query_embeddings[idx:idx+1]
        
        # Measure retrieval time
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        sims = (q @ gallery_embeddings.T).cpu()
        topk = torch.topk(sims, k=min(10, gallery_embeddings.size(0)), dim=1)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        query_time_ms = (end - start) * 1000
        query_times.append(query_time_ms)
    
    stats = {
        "gallery_size": gallery_embeddings.size(0),
        "num_trials": len(query_times),
        "per_query_time_ms": {
            "mean": statistics.mean(query_times),
            "median": statistics.median(query_times),
            "std": statistics.stdev(query_times) if len(query_times) > 1 else 0.0,
            "min": min(query_times),
            "max": max(query_times),
        },
    }
    
    print(f"  Per-query retrieval time: {stats['per_query_time_ms']['mean']:.4f} ± {stats['per_query_time_ms']['std']:.4f} ms")
    print(f"  Throughput: {1000 / stats['per_query_time_ms']['mean']:.0f} queries/second")
    
    return stats


def benchmark_end_to_end(
    dataset: MassSpecGymDataset,
    dreams: DreamsAdapter,
    mol_adapter: MolAdapter,
    featurizer: MolFeaturizer,
    spec_bins: int,
    device: torch.device,
    batch_size: int,
    gallery_size: int = 1500,
    num_queries: int = 100,
) -> dict:
    """
    Benchmark end-to-end throughput (encoding + retrieval).
    
    Args:
        gallery_size: Size of candidate pool (average from Spectraverse test set)
        num_queries: Number of queries to process
    
    Returns:
        Dictionary with end-to-end statistics
    """
    print(f"\n[Benchmark] End-to-end throughput")
    print(f"  Gallery size: {gallery_size}, Queries: {num_queries}, Batch size: {batch_size}")
    
    # Pre-compute gallery embeddings (simulating FAISS index) - encode in batches
    print("  Pre-computing gallery embeddings...")
    gallery_embeddings = []
    gallery_batch_size = min(batch_size, gallery_size)  # Use same batch size for consistency
    for i in range(0, gallery_size, gallery_batch_size):
        end_idx = min(i + gallery_batch_size, gallery_size)
        gallery_batch = [dataset[j % len(dataset)] for j in range(i, end_idx)]
        z_batch, _, _ = encode_batch(gallery_batch, dreams, mol_adapter, featurizer, spec_bins, device)
        gallery_embeddings.append(unit_normalize(z_batch).cpu())  # Move to CPU to save GPU memory
        if device.type == "cuda":
            torch.cuda.empty_cache()  # Clear cache after each batch
    z_gallery = torch.cat(gallery_embeddings, dim=0).to(device)  # [gallery_size, cond_dim]
    
    # Warmup
    for i in range(3):
        start_idx = (i * batch_size) % len(dataset)
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = [dataset[j] for j in range(start_idx, end_idx)]
        if len(batch) > 0:
            z_q, _, _ = encode_batch(batch, dreams, mol_adapter, featurizer, spec_bins, device)
            z_q = unit_normalize(z_q)
            sims = (z_q @ z_gallery.T).cpu()
    
    # Synchronize GPU if available
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark end-to-end
    total_times = []
    encoding_times = []
    retrieval_times = []
    
    num_batches = (num_queries + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_queries)
        batch = [dataset[i % len(dataset)] for i in range(start_idx, end_idx)]
        
        if len(batch) == 0:
            break
        
        # Measure end-to-end time
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Encoding
        encode_start = time.perf_counter()
        z_q, _, _ = encode_batch(batch, dreams, mol_adapter, featurizer, spec_bins, device)
        z_q = unit_normalize(z_q)
        if device.type == "cuda":
            torch.cuda.synchronize()
        encode_end = time.perf_counter()
        
        # Retrieval (batched for efficiency)
        retrieve_start = time.perf_counter()
        # Compute similarity for entire batch at once
        sims = (z_q @ z_gallery.T).cpu()  # [batch_size, gallery_size]
        _ = torch.topk(sims, k=min(10, gallery_size), dim=1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        retrieve_end = time.perf_counter()
        
        end = time.perf_counter()
        
        batch_total_time_ms = (end - start) * 1000
        batch_encode_time_ms = (encode_end - encode_start) * 1000
        batch_retrieve_time_ms = (retrieve_end - retrieve_start) * 1000
        
        # Per-query times
        num_queries_in_batch = len(batch)
        total_times.append(batch_total_time_ms / num_queries_in_batch)
        encoding_times.append(batch_encode_time_ms / num_queries_in_batch)
        retrieval_times.append(batch_retrieve_time_ms / num_queries_in_batch)
    
    stats = {
        "gallery_size": gallery_size,
        "num_queries": num_queries,
        "batch_size": batch_size,
        "per_query_total_time_ms": {
            "mean": statistics.mean(total_times),
            "median": statistics.median(total_times),
            "std": statistics.stdev(total_times) if len(total_times) > 1 else 0.0,
        },
        "per_query_encoding_time_ms": {
            "mean": statistics.mean(encoding_times),
            "median": statistics.median(encoding_times),
            "std": statistics.stdev(encoding_times) if len(encoding_times) > 1 else 0.0,
        },
        "per_query_retrieval_time_ms": {
            "mean": statistics.mean(retrieval_times),
            "median": statistics.median(retrieval_times),
            "std": statistics.stdev(retrieval_times) if len(retrieval_times) > 1 else 0.0,
        },
    }
    
    total_throughput = 1000 / stats["per_query_total_time_ms"]["mean"]
    
    print(f"  Per-query total time: {stats['per_query_total_time_ms']['mean']:.3f} ms")
    print(f"    - Encoding: {stats['per_query_encoding_time_ms']['mean']:.3f} ms")
    print(f"    - Retrieval: {stats['per_query_retrieval_time_ms']['mean']:.4f} ms")
    print(f"  Total throughput: {total_throughput:.1f} queries/second")
    
    return stats


def main():
    ap = argparse.ArgumentParser(description="Benchmark SpecBridge inference throughput")
    ap.add_argument("--mgf", type=str, required=True, help="Path to MGF file (e.g., Spectraverse test set)")
    ap.add_argument("--dreams-ckpt", type=str, default=None, help="Path to DreaMS checkpoint")
    ap.add_argument("--adapter-ckpt", type=str, default=None, help="Path to trained adapter checkpoint")
    ap.add_argument("--spec-bins", type=int, default=2048)
    ap.add_argument("--fp-bits", type=int, default=2048)
    ap.add_argument("--cond-dim", type=int, default=512)
    ap.add_argument("--mapper-hidden", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding")
    ap.add_argument("--gallery-size", type=int, default=1500, help="Size of candidate pool (avg from Spectraverse)")
    ap.add_argument("--num-queries", type=int, default=1000, help="Number of queries for end-to-end benchmark")
    ap.add_argument("--fold-query", type=str, default="test", help="Fold to use (e.g., 'test')")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    ap.add_argument("--skip-encoding", action="store_true", help="Skip encoding benchmark")
    ap.add_argument("--skip-retrieval", action="store_true", help="Skip retrieval benchmark")
    ap.add_argument("--skip-end-to-end", action="store_true", help="Skip end-to-end benchmark")
    
    args = ap.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    set_seed(args.seed)
    
    print("=" * 80)
    print("SpecBridge Inference Throughput Benchmark")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"MGF file: {args.mgf}")
    print(f"Adapter checkpoint: {args.adapter_ckpt}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gallery size: {args.gallery_size}")
    print("=" * 80)
    
    # Load dataset
    print("\n[Loading] Dataset...")
    if args.fold_query:
        folds = {f.strip().lower() for f in args.fold_query.split(",") if f.strip()}
        print(f"  Filtering to fold(s): {folds}")
    else:
        folds = None
        print("  Loading all folds")
    
    ds_full = MassSpecGymDataset(args.mgf, meta_json=None, folds=folds)
    print(f"  Loaded {len(ds_full)} spectra")
    
    if len(ds_full) == 0:
        print("[ERROR] No spectra found in dataset!")
        print(f"  Available folds in dataset: {ds_full.fold_counts if hasattr(ds_full, 'fold_counts') else 'unknown'}")
        return
    
    # Convert to list for indexing
    ds_list = [ds_full[i] for i in range(len(ds_full))]
    
    # Build models
    print("\n[Loading] Models...")
    dreams_backbone = load_dreams_encoder(args.dreams_ckpt, d_in=args.spec_bins, d_out=1024)
    dreams = DreamsAdapter(dreams_backbone, d_out=args.cond_dim, hidden=0, freeze_backbone=True).to(device).eval()
    mol_encoder = MolEncoder(d_in=args.fp_bits, embed_dim=512)
    mol = MolAdapter(mol_encoder, d_out=args.cond_dim, hidden=0).to(device).eval()
    mapper = MapperB(d_in=args.cond_dim, d_out=args.cond_dim, hidden=args.mapper_hidden, gaussian=True).to(device).eval()
    
    if args.adapter_ckpt:
        print(f"  Loading adapter from {args.adapter_ckpt}")
        ck = torch.load(args.adapter_ckpt, map_location="cpu")
        sd = ck.get("model", ck)
        dreams.load_state_dict({k.replace("spec.", ""): v for k, v in sd.items() if k.startswith("spec.")}, strict=False)
        mol.load_state_dict({k.replace("mol.", ""): v for k, v in sd.items() if k.startswith("mol.")}, strict=False)
        mapper.load_state_dict({k.replace("mapB.", ""): v for k, v in sd.items() if k.startswith("mapB.")}, strict=False)
    
    featurizer = MolFeaturizer(fp_bits=args.fp_bits)
    
    # Create a simple dataset wrapper
    class SimpleDataset:
        def __init__(self, data_list):
            self.data = data_list
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]
    
    ds_wrapped = SimpleDataset(ds_list)
    
    results = {}
    
    # Benchmark 1: Encoding time
    if not args.skip_encoding:
        # Use a subset for encoding benchmark to avoid memory issues
        encoding_subset_size = min(1000, len(ds_list))
        ds_encoding = SimpleDataset(ds_list[:encoding_subset_size])
        results["encoding"] = benchmark_encoding(
            ds_encoding, dreams, mol, featurizer, args.spec_bins, device, args.batch_size
        )
    
    # Benchmark 2: Retrieval time
    if not args.skip_retrieval:
        # Pre-encode some queries and gallery in batches
        print("\n[Preparing] Query and gallery embeddings for retrieval benchmark...")
        num_queries = min(1000, len(ds_list))
        
        # Encode queries in batches
        query_embeddings = []
        for i in range(0, num_queries, args.batch_size):
            end_idx = min(i + args.batch_size, num_queries)
            query_batch = [ds_list[j % len(ds_list)] for j in range(i, end_idx)]
            z_batch, _, _ = encode_batch(query_batch, dreams, mol, featurizer, args.spec_bins, device)
            query_embeddings.append(unit_normalize(z_batch).cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
        z_queries = torch.cat(query_embeddings, dim=0).to(device)
        
        # Encode gallery in batches
        gallery_embeddings = []
        for i in range(0, args.gallery_size, args.batch_size):
            end_idx = min(i + args.batch_size, args.gallery_size)
            gallery_batch = [ds_list[j % len(ds_list)] for j in range(i, end_idx)]
            z_batch, _, _ = encode_batch(gallery_batch, dreams, mol, featurizer, args.spec_bins, device)
            gallery_embeddings.append(unit_normalize(z_batch).cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
        z_gallery = torch.cat(gallery_embeddings, dim=0).to(device)
        
        results["retrieval"] = benchmark_retrieval(z_queries, z_gallery, device)
    
    # Benchmark 3: End-to-end throughput
    if not args.skip_end_to_end:
        # Use full dataset for end-to-end
        ds_e2e = SimpleDataset(ds_list)
        results["end_to_end"] = benchmark_end_to_end(
            ds_e2e, dreams, mol, featurizer, args.spec_bins, device,
            args.batch_size, args.gallery_size, args.num_queries
        )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if "encoding" in results:
        enc = results["encoding"]
        print(f"\nEncoding Time (batched, batch_size={enc['batch_size']}):")
        print(f"  Mean: {enc['per_spectrum_time_ms']['mean']:.3f} ms per spectrum")
        print(f"  Expected: 3.2 ms per spectrum")
        print(f"  Ratio: {enc['per_spectrum_time_ms']['mean'] / 3.2:.2f}x")
    
    if "retrieval" in results:
        ret = results["retrieval"]
        print(f"\nRetrieval Time (gallery_size={ret['gallery_size']}):")
        print(f"  Mean: {ret['per_query_time_ms']['mean']:.4f} ms per query")
        print(f"  Expected: <0.1 ms per query")
        print(f"  Status: {'✓ PASS' if ret['per_query_time_ms']['mean'] < 0.1 else '✗ FAIL'}")
    
    if "end_to_end" in results:
        e2e = results["end_to_end"]
        throughput = 1000 / e2e["per_query_total_time_ms"]["mean"]
        print(f"\nEnd-to-End Throughput:")
        print(f"  Measured: {throughput:.1f} queries/second")
        print(f"  Expected: ~300 queries/second")
        print(f"  Ratio: {throughput / 300:.2f}x")
        print(f"  Breakdown:")
        print(f"    - Encoding: {e2e['per_query_encoding_time_ms']['mean']:.3f} ms")
        print(f"    - Retrieval: {e2e['per_query_retrieval_time_ms']['mean']:.4f} ms")
        print(f"    - Total: {e2e['per_query_total_time_ms']['mean']:.3f} ms")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
