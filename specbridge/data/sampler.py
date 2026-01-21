from torch.utils.data import Sampler
from collections import defaultdict
import random
import math
from typing import List

class BalancedBatchSampler(Sampler[List[int]]):
    """Ensures K replicates per identity (by canonical SMILES key) per batch.
    Expects each record in ds._records to have a string field "smi_key".
    """
    def __init__(self, records, batch_size: int = 256, K: int = 4, shuffle: bool = True):
        self.groups = defaultdict(list)
        for i, rec in enumerate(records):
            self.groups[rec["smi_key"]].append(i)
        # Only include identities with at least K samples to ensure we can always take K
        total_identities = len(self.groups)
        self.ids = [k for k, v in self.groups.items() if len(v) >= K]
        if len(self.ids) < total_identities:
            filtered = total_identities - len(self.ids)
            print(f"[BalancedBatchSampler] Filtered out {filtered}/{total_identities} identities with < {K} samples")
        self.batch_size = batch_size
        self.K = max(1, int(K))
        if self.batch_size < self.K:
            raise ValueError(f"batch_size ({batch_size}) must be >= K ({self.K})")
        self.shuffle = shuffle
        # Pre-compute num_ids per batch for __len__
        self.num_ids_per_batch = max(1, batch_size // self.K)

    def __iter__(self):
        ids = self.ids[:]
        if self.shuffle:
            random.shuffle(ids)
        
        # Prepare shuffled sample pools for each identity (we'll cycle through ALL samples)
        pools = {}
        for pid in ids:
            g = self.groups[pid][:]  # Copy the list
            if self.shuffle:
                random.shuffle(g)
            pools[pid] = g  # Store all samples, not just K
        
        B = self.batch_size
        num_ids = self.num_ids_per_batch
        
        # Track position in each identity's sample pool (for cycling)
        positions = {pid: 0 for pid in ids}
        
        # Continue generating batches until we can't form a valid one
        while True:
            batch: List[int] = []
            # Select which identities to use in this batch
            # Shuffle identity order for diversity
            batch_ids = ids[:]
            if self.shuffle:
                random.shuffle(batch_ids)
            
            # Fill batch with K samples from each identity (cycling through all samples)
            for pid in batch_ids:
                if len(batch) >= B:
                    break
                
                pool = pools[pid]
                if len(pool) == 0:
                    continue
                
                pos = positions[pid]
                samples_to_add = []
                
                # Take K samples starting from current position (with wrapping)
                for _ in range(self.K):
                    sample_idx = pool[pos % len(pool)]
                    samples_to_add.append(sample_idx)
                    pos = (pos + 1) % len(pool)
                
                # Check if we have room for all K samples
                if len(batch) + len(samples_to_add) <= B:
                    batch.extend(samples_to_add)
                    positions[pid] = pos  # Update position for next time
                else:
                    # Not enough room - skip this identity for this batch
                    break
            
            # Yield batch if it's valid (has at least K samples)
            if len(batch) >= self.K:
                yield batch[:B]
            else:
                # Can't form a valid batch anymore
                break

    def __len__(self) -> int:
        # Calculate based on total samples available
        if len(self.ids) == 0:
            return 0
        
        # Total samples across all valid identities
        total_samples = sum(len(self.groups[pid]) for pid in self.ids)
        num_ids = self.num_ids_per_batch
        
        # Each batch uses K samples per identity, up to num_ids identities
        # So each batch uses at most: num_ids * K samples (which should be <= batch_size)
        samples_per_batch = min(num_ids * self.K, self.batch_size)
        
        # Approximate number of batches: total samples / samples per batch
        # This is an approximation since we cycle through samples
        return max(1, math.ceil(total_samples / samples_per_batch))


class ReplicateBatchSampler(Sampler):
    def __init__(self, records, batch_size, K=4, seed=0):
        groups = defaultdict(list)
        for i, rec in enumerate(records):
            groups[rec["smi_key"]].append(i)
        # Filter to groups with at least K samples (consistent with BalancedBatchSampler)
        total_groups = len(groups)
        rep_groups = [g for g in groups.values() if len(g) >= K]
        if len(rep_groups) < total_groups:
            filtered = total_groups - len(rep_groups)
            print(f"[ReplicateBatchSampler] Filtered out {filtered}/{total_groups} groups with < {K} samples")

        self.rep_groups = rep_groups
        self.batch_size = batch_size
        self.K = K
        if self.batch_size < self.K:
            raise ValueError(f"batch_size ({batch_size}) must be >= K ({K})")
        self.rng = random.Random(seed)

    def __iter__(self):
        pool = self.rep_groups[:]
        self.rng.shuffle(pool)
        buf = []
        for g in pool:
            glist = g[:]
            self.rng.shuffle(glist)
            # Since we filtered to groups with >= K samples, this should always give K samples
            take = glist[:self.K]
            buf.extend(take)
            if len(buf) >= self.batch_size:
                yield buf[:self.batch_size]
                buf = []
        if buf:
            # Yield remaining items even if less than batch_size
            # (Better than padding with duplicates which can cause issues)
            yield buf

    def __len__(self):
        # Approximate: accounts for full batches + potentially one partial batch
        per_batch_ids = max(1, self.batch_size // max(1, self.K))
        full_batches = len(self.rep_groups) // per_batch_ids
        remaining_items = (len(self.rep_groups) % per_batch_ids) * self.K
        # Add 1 if there's a partial batch at the end
        partial_batch = 1 if remaining_items > 0 else 0
        return max(1, full_batches + partial_batch)


class UniqueSMILESBatchSampler(Sampler[List[int]]):
    """Ensures no duplicate SMILES in each batch for fair contrastive learning.
    This prevents same-SMILES samples from being treated as negatives in InfoNCE loss.
    Includes ALL data points - makes multiple passes to include all samples.
    """
    def __init__(self, records, batch_size: int = 256, shuffle: bool = True):
        self.records = records
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group samples by SMILES
        self.groups = defaultdict(list)
        for i, rec in enumerate(records):
            self.groups[rec["smi_key"]].append(i)
        
        self.smiles_list = list(self.groups.keys())
        print(f"[UniqueSMILESBatchSampler] {len(self.smiles_list)} unique SMILES, {len(records)} total samples")
        
        # Find max samples per SMILES to know how many passes we need
        self.max_samples_per_smiles = max(len(samples) for samples in self.groups.values()) if self.groups else 0

    def __iter__(self):
        # Shuffle SMILES order once per epoch
        smiles_order = self.smiles_list[:]
        if self.shuffle:
            random.shuffle(smiles_order)
        
        # Make multiple passes: in each pass, take one sample from each SMILES
        # This ensures all samples are included across all batches
        for pass_num in range(self.max_samples_per_smiles):
            batch: List[int] = []
            used_smiles_in_batch = set()
            
            # Shuffle SMILES order for each batch
            if self.shuffle:
                random.shuffle(smiles_order)
            
            for smiles in smiles_order:
                # Skip if this SMILES is already in the current batch
                if smiles in used_smiles_in_batch:
                    continue
                
                samples = self.groups[smiles]
                # If this SMILES has a sample at this pass number, add it
                if pass_num < len(samples):
                    sample_idx = samples[pass_num]
                    batch.append(sample_idx)
                    used_smiles_in_batch.add(smiles)
                    
                    # If batch is full, yield it and start new batch
                    if len(batch) >= self.batch_size:
                        yield batch[:self.batch_size]
                        batch = []
                        used_smiles_in_batch = set()
            
            # Yield remaining samples in batch if any
            if batch:
                yield batch

    def __len__(self) -> int:
        # Approximate: number of passes * (unique SMILES / batch_size)
        passes = self.max_samples_per_smiles
        batches_per_pass = max(1, math.ceil(len(self.smiles_list) / self.batch_size))
        return passes * batches_per_pass
