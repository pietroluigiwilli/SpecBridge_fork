#!/usr/bin/env python3
"""
Check overlap between MassSpecGym test set and DreaMS pretrain data (GeMS).

This script:
1. Loads MassSpecGym test set spectra
2. Downloads and loads GeMS pretrain data (HDF5)
3. Compares spectra using spectral similarity (cosine similarity on binned spectra)
4. Reports overlap statistics
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import h5py
from urllib.request import urlretrieve

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from specbridge.data.massspecgym import MassSpecGymDataset, bin_peaks


def download_gems_data(url: str, output_path: str) -> str:
    """Download GeMS pretrain data from HuggingFace."""
    if os.path.exists(output_path):
        print(f"GeMS data already exists at {output_path}")
        return output_path
    
    print(f"Downloading GeMS data from {url}...")
    print(f"Output path: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading: {e}")
        raise
    
    return output_path


def load_gems_spectra(hdf5_path: str, max_spectra: int = None) -> List[Dict[str, Any]]:
    """Load spectra from GeMS HDF5 file.
    
    GeMS HDF5 format (DreaMS): shape (num_spectra, 2, num_peaks) where 
    spectrum[i, 0, :] = m/z values and spectrum[i, 1, :] = intensity values.
    """
    print(f"Loading GeMS spectra from {hdf5_path}...")
    spectra = []
    
    with h5py.File(hdf5_path, 'r') as f:
        # Explore the structure
        print(f"HDF5 keys: {list(f.keys())}")
        
        # DreaMS format: look for 'spectrum' key
        spec_key = None
        for key in ['spectrum', 'spectra', 'spectrum_array']:
            if key in f.keys():
                spec_key = key
                break
        
        if spec_key is None:
            # Try to find any dataset that looks like spectra
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    shape = f[key].shape
                    if len(shape) == 3 and shape[1] == 2:
                        spec_key = key
                        print(f"Found potential spectrum dataset: {key}, shape: {shape}")
                        break
        
        if spec_key is None:
            raise ValueError(f"Could not find spectrum dataset in HDF5. Available keys: {list(f.keys())}")
        
        spec_data = f[spec_key]
        print(f"Loading spectrum data from '{spec_key}', shape: {spec_data.shape}")
        
        # Expected shape: (num_spectra, 2, num_peaks)
        if len(spec_data.shape) != 3 or spec_data.shape[1] != 2:
            raise ValueError(f"Unexpected spectrum shape: {spec_data.shape}. Expected (N, 2, M)")
        
        num_spectra = spec_data.shape[0]
        if max_spectra is not None:
            num_spectra = min(num_spectra, max_spectra)
        
        print(f"Loading {num_spectra} spectra...")
        
        # Load in batches for better performance
        batch_size = 10000
        for start_idx in tqdm(range(0, num_spectra, batch_size), desc="Loading GeMS spectra"):
            end_idx = min(start_idx + batch_size, num_spectra)
            batch_size_actual = end_idx - start_idx
            
            # Load batch: shape (batch_size, 2, num_peaks)
            batch_data = spec_data[start_idx:end_idx, :, :]  # Vectorized HDF5 read
            
            # Extract m/z and intensity for all spectra in batch
            mz_batch = batch_data[:, 0, :]  # (batch_size, num_peaks)
            intensity_batch = batch_data[:, 1, :]  # (batch_size, num_peaks)
            
            # Filter out zero intensities for each spectrum
            for i in range(batch_size_actual):
                intensity_array = intensity_batch[i, :]
                mask = intensity_array > 1e-10
                
                if mask.sum() > 0:
                    mz = torch.tensor(mz_batch[i, mask], dtype=torch.float32)
                    intensity = torch.tensor(intensity_array[mask], dtype=torch.float32)
                    spectra.append({
                        'mz': mz,
                        'intensity': intensity,
                    })
    
    print(f"Loaded {len(spectra)} spectra from GeMS")
    return spectra


def bin_spectrum(mz: torch.Tensor, intensity: torch.Tensor, num_bins: int = 20000, max_mz: float = 2000.0) -> torch.Tensor:
    """Bin a spectrum into a fixed-size vector."""
    return bin_peaks(mz, intensity, num_bins=num_bins, max_mz=max_mz)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two binned spectra."""
    a_norm = a / (a.norm() + 1e-8)
    b_norm = b / (b.norm() + 1e-8)
    return float((a_norm * b_norm).sum())


def find_overlapping_spectra_streaming(
    test_spectra: List[Dict[str, Any]],
    hdf5_path: str,
    similarity_threshold: float = 0.99,
    num_bins: int = 20000,
    max_mz: float = 2000.0,
    pretrain_batch_size: int = 10000,
    max_pretrain: int = None
) -> Tuple[List[int], List[float]]:
    """
    Find test spectra that overlap with pretrain spectra using streaming approach.
    Processes pretrain data in batches from HDF5 to avoid loading everything into memory.
    
    Returns:
        overlapping_indices: List of indices in test_spectra that have matches
        similarities: List of maximum similarities for each overlapping spectrum
    """
    print(f"Comparing {len(test_spectra)} test spectra with pretrain data (streaming)...")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Bin all test spectra first (small dataset, fits in memory)
    print("Binning test spectra...")
    test_binned = []
    for spec in tqdm(test_spectra, desc="Binning test"):
        binned = bin_spectrum(spec['mz'], spec['intensity'], num_bins=num_bins, max_mz=max_mz)
        test_binned.append(binned)
    
    test_binned = torch.stack(test_binned)  # Shape: [N_test, num_bins]
    test_norm = test_binned / (test_binned.norm(dim=1, keepdim=True) + 1e-8)
    
    # Track maximum similarity for each test spectrum
    max_similarities = torch.zeros(len(test_spectra), dtype=torch.float32)
    
    # Process pretrain data in batches from HDF5
    print("Processing pretrain data in batches...")
    with h5py.File(hdf5_path, 'r') as f:
        spec_data = f['spectrum']
        total_pretrain = spec_data.shape[0]
        if max_pretrain is not None:
            total_pretrain = min(total_pretrain, max_pretrain)
        
        num_batches = (total_pretrain + pretrain_batch_size - 1) // pretrain_batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing pretrain batches"):
            start_idx = batch_idx * pretrain_batch_size
            end_idx = min(start_idx + pretrain_batch_size, total_pretrain)
            batch_size_actual = end_idx - start_idx
            
            # Load batch from HDF5
            batch_data = spec_data[start_idx:end_idx, :, :]  # (batch_size, 2, num_peaks)
            
            # Extract and bin pretrain batch
            pretrain_batch_binned = []
            for i in range(batch_size_actual):
                mz_array = batch_data[i, 0, :]
                intensity_array = batch_data[i, 1, :]
                mask = intensity_array > 1e-10
                
                if mask.sum() > 0:
                    mz = torch.tensor(mz_array[mask], dtype=torch.float32)
                    intensity = torch.tensor(intensity_array[mask], dtype=torch.float32)
                    binned = bin_spectrum(mz, intensity, num_bins=num_bins, max_mz=max_mz)
                    pretrain_batch_binned.append(binned)
            
            if len(pretrain_batch_binned) == 0:
                continue
            
            # Stack and normalize pretrain batch
            pretrain_batch = torch.stack(pretrain_batch_binned)  # (batch_size, num_bins)
            pretrain_batch_norm = pretrain_batch / (pretrain_batch.norm(dim=1, keepdim=True) + 1e-8)
            
            # Compute similarities: (N_test, num_bins) @ (num_bins, batch_size) = (N_test, batch_size)
            cos_sims = test_norm @ pretrain_batch_norm.T  # (N_test, batch_size)
            
            # Update maximum similarities
            batch_max = cos_sims.max(dim=1)[0]  # (N_test,)
            max_similarities = torch.maximum(max_similarities, batch_max)
    
    # Find overlapping spectra
    overlapping_mask = max_similarities >= similarity_threshold
    overlapping_indices = torch.where(overlapping_mask)[0].tolist()
    similarities = max_similarities[overlapping_mask].tolist()
    
    return overlapping_indices, similarities


def find_overlapping_spectra(
    test_spectra: List[Dict[str, Any]],
    pretrain_spectra: List[Dict[str, Any]],
    similarity_threshold: float = 0.99,
    num_bins: int = 20000,
    max_mz: float = 2000.0,
    batch_size: int = 1000
) -> Tuple[List[int], List[float]]:
    """
    Find test spectra that overlap with pretrain spectra based on spectral similarity.
    For small pretrain datasets that fit in memory.
    
    Returns:
        overlapping_indices: List of indices in test_spectra that have matches
        similarities: List of maximum similarities for each overlapping spectrum
    """
    print(f"Comparing {len(test_spectra)} test spectra with {len(pretrain_spectra)} pretrain spectra...")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Bin all pretrain spectra
    print("Binning pretrain spectra...")
    pretrain_binned = []
    for spec in tqdm(pretrain_spectra, desc="Binning pretrain"):
        binned = bin_spectrum(spec['mz'], spec['intensity'], num_bins=num_bins, max_mz=max_mz)
        pretrain_binned.append(binned)
    
    pretrain_binned = torch.stack(pretrain_binned)  # Shape: [N_pretrain, num_bins]
    
    # Normalize pretrain spectra
    pretrain_norm = pretrain_binned / (pretrain_binned.norm(dim=1, keepdim=True) + 1e-8)
    
    # Bin all test spectra
    print("Binning test spectra...")
    test_binned = []
    for spec in tqdm(test_spectra, desc="Binning test"):
        binned = bin_spectrum(spec['mz'], spec['intensity'], num_bins=num_bins, max_mz=max_mz)
        test_binned.append(binned)
    
    test_binned = torch.stack(test_binned)  # Shape: [N_test, num_bins]
    test_norm = test_binned / (test_binned.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute all similarities in batches
    print("Computing similarities...")
    overlapping_indices = []
    similarities = []
    
    # Process in batches to manage memory
    for i in tqdm(range(0, len(test_spectra), batch_size), desc="Comparing batches"):
        end_idx = min(i + batch_size, len(test_spectra))
        test_batch = test_norm[i:end_idx]  # Shape: [batch_size, num_bins]
        
        # Compute cosine similarities: [batch_size, num_bins] @ [num_bins, N_pretrain] = [batch_size, N_pretrain]
        cos_sims = test_batch @ pretrain_norm.T  # Shape: [batch_size, N_pretrain]
        max_sims = cos_sims.max(dim=1)[0].cpu().numpy()  # Shape: [batch_size]
        
        # Find overlapping spectra in this batch
        for j, max_sim in enumerate(max_sims):
            if max_sim >= similarity_threshold:
                overlapping_indices.append(i + j)
                similarities.append(float(max_sim))
    
    return overlapping_indices, similarities


def main():
    parser = argparse.ArgumentParser(description="Check overlap between MassSpecGym test and GeMS pretrain data")
    parser.add_argument("--msgym-mgf", type=str, 
                       default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym.mgf",
                       help="Path to MassSpecGym MGF file")
    parser.add_argument("--gems-url", type=str,
                       default="https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/GeMS_A/GeMS_A10.hdf5",
                       help="URL to GeMS pretrain HDF5 file")
    parser.add_argument("--gems-path", type=str,
                       default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/GeMS_A10.hdf5",
                       help="Local path to save/load GeMS HDF5 file")
    parser.add_argument("--mona-pkl", type=str,
                       default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/MoNA_A_Murcko_split_neighbours_%5BM%2BH%5D%2B_0.05Da.pkl",
                       help="Path to MoNA fine-tuning pickle file")
    parser.add_argument("--similarity-threshold", type=float, default=0.99,
                       help="Cosine similarity threshold for considering spectra as overlapping")
    parser.add_argument("--num-bins", type=int, default=20000,
                       help="Number of bins for spectral binning")
    parser.add_argument("--max-mz", type=float, default=2000.0,
                       help="Maximum m/z value for binning")
    parser.add_argument("--check-finetune", action="store_true",
                       help="Also check overlap with fine-tuning data (MoNA)")
    parser.add_argument("--max-pretrain", type=int, default=None,
                       help="Maximum number of pretrain spectra to process (for testing)")
    parser.add_argument("--pretrain-batch-size", type=int, default=10000,
                       help="Batch size for processing pretrain data in streaming mode")
    
    args = parser.parse_args()
    
    # Load MassSpecGym test set
    print("=" * 80)
    print("Loading MassSpecGym test set...")
    print("=" * 80)
    test_dataset = MassSpecGymDataset(args.msgym_mgf, folds={'test'})
    print(f"Loaded {len(test_dataset)} test spectra")
    print(f"Dataset stats: {test_dataset.stats_str()}")
    
    # Extract test spectra
    test_spectra = []
    for i in range(len(test_dataset)):
        item = test_dataset[i]
        test_spectra.append({
            'mz': item['mz'],
            'intensity': item['intensity'],
            'title': item['title'],
            'smiles': item.get('smiles', ''),
        })
    
    # Download GeMS pretrain data (if needed)
    print("\n" + "=" * 80)
    print("Checking GeMS pretrain data...")
    print("=" * 80)
    gems_path = download_gems_data(args.gems_url, args.gems_path)
    
    # Check dataset size to decide on streaming vs in-memory
    with h5py.File(gems_path, 'r') as f:
        total_pretrain = f['spectrum'].shape[0]
        if args.max_pretrain is not None:
            total_pretrain = min(total_pretrain, args.max_pretrain)
    
    print(f"Total pretrain spectra: {total_pretrain:,}")
    
    # Use streaming approach for large datasets (>100k spectra)
    use_streaming = total_pretrain > 100000 or args.max_pretrain is None
    
    if use_streaming:
        print("Using streaming approach (processing in batches from HDF5)...")
        # Find overlaps using streaming
        print("\n" + "=" * 80)
        print("Finding overlapping spectra...")
        print("=" * 80)
        overlapping_indices, similarities = find_overlapping_spectra_streaming(
            test_spectra,
            gems_path,
            similarity_threshold=args.similarity_threshold,
            num_bins=args.num_bins,
            max_mz=args.max_mz,
            pretrain_batch_size=args.pretrain_batch_size,
            max_pretrain=args.max_pretrain
        )
        # Store total for reporting
        num_pretrain_processed = total_pretrain
    else:
        print("Using in-memory approach (loading all pretrain data)...")
        # Load pretrain data into memory
        pretrain_spectra = load_gems_spectra(gems_path, max_spectra=args.max_pretrain)
        
        if len(pretrain_spectra) == 0:
            print("ERROR: No spectra loaded from GeMS. Please check the HDF5 structure.")
            return 1
        
        # Find overlaps
        print("\n" + "=" * 80)
        print("Finding overlapping spectra...")
        print("=" * 80)
        overlapping_indices, similarities = find_overlapping_spectra(
            test_spectra,
            pretrain_spectra,
            similarity_threshold=args.similarity_threshold,
            num_bins=args.num_bins,
            max_mz=args.max_mz
        )
        # Store total for reporting
        num_pretrain_processed = len(pretrain_spectra)
    
    # Report results
    print("\n" + "=" * 80)
    print("RESULTS: Overlap with Pretrain Data (GeMS)")
    print("=" * 80)
    print(f"Total test spectra: {len(test_spectra)}")
    print(f"Total pretrain spectra: {num_pretrain_processed:,}")
    print(f"Overlapping test spectra: {len(overlapping_indices)}")
    print(f"Overlap percentage: {100.0 * len(overlapping_indices) / len(test_spectra):.2f}%")
    if len(similarities) > 0:
        print(f"Mean similarity: {np.mean(similarities):.4f}")
        print(f"Median similarity: {np.median(similarities):.4f}")
        print(f"Min similarity: {np.min(similarities):.4f}")
        print(f"Max similarity: {np.max(similarities):.4f}")
    
    # Optionally check fine-tuning data overlap
    if args.check_finetune:
        print("\n" + "=" * 80)
        print("Loading MoNA fine-tuning data...")
        print("=" * 80)
        mona_df = pd.read_pickle(args.mona_pkl)
        print(f"MoNA dataframe shape: {mona_df.shape}")
        print(f"Columns: {mona_df.columns.tolist()}")
        
        # Filter to training data (val == False)
        if 'val' in mona_df.columns:
            train_df = mona_df[mona_df['val'] == False]
            print(f"Training spectra (val=False): {len(train_df)}")
            
            # Extract spectra from MoNA
            # PARSED PEAKS has shape (2, N) where first row is m/z, second is intensity
            mona_spectra = []
            for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Extracting MoNA spectra"):
                peaks = row['PARSED PEAKS']
                if peaks is not None and peaks.shape[0] == 2:
                    mz_array = peaks[0, :]
                    intensity_array = peaks[1, :]
                    # Filter out zero intensities
                    mask = intensity_array > 1e-10
                    if mask.sum() > 0:
                        mz = torch.tensor(mz_array[mask], dtype=torch.float32)
                        intensity = torch.tensor(intensity_array[mask], dtype=torch.float32)
                        mona_spectra.append({
                            'mz': mz,
                            'intensity': intensity,
                            'id': row.get('ID', f'MoNA_{idx}'),
                        })
            
            print(f"Extracted {len(mona_spectra)} spectra from MoNA training data")
            
            # Check overlap with test set
            print("\n" + "=" * 80)
            print("Finding overlaps with MoNA fine-tuning data...")
            print("=" * 80)
            mona_overlapping_indices, mona_similarities = find_overlapping_spectra(
                test_spectra,
                mona_spectra,
                similarity_threshold=args.similarity_threshold,
                num_bins=args.num_bins,
                max_mz=args.max_mz
            )
            
            print("\n" + "=" * 80)
            print("RESULTS: Overlap with Fine-tuning Data (MoNA)")
            print("=" * 80)
            print(f"Total test spectra: {len(test_spectra)}")
            print(f"Total MoNA training spectra: {len(mona_spectra)}")
            print(f"Overlapping test spectra: {len(mona_overlapping_indices)}")
            print(f"Overlap percentage: {100.0 * len(mona_overlapping_indices) / len(test_spectra):.2f}%")
            if len(mona_similarities) > 0:
                print(f"Mean similarity: {np.mean(mona_similarities):.4f}")
                print(f"Median similarity: {np.median(mona_similarities):.4f}")
                print(f"Min similarity: {np.min(mona_similarities):.4f}")
                print(f"Max similarity: {np.max(mona_similarities):.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

