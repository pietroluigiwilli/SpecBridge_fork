from torch.utils.data import Sampler
from collections import defaultdict
import random

from collections import defaultdict
from torch.utils.data import Sampler
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
        self.ids = [k for k, v in self.groups.items() if len(v) >= 2]
        self.batch_size = batch_size
        self.K = max(1, int(K))
        self.shuffle = shuffle

    def __iter__(self):
        ids = self.ids[:]
        if self.shuffle:
            random.shuffle(ids)
        B = self.batch_size
        num_ids = max(1, B // self.K)
        for i in range(0, len(ids), num_ids):
            pick = ids[i:i+num_ids]
            batch: List[int] = []
            for pid in pick:
                g = self.groups[pid]
                if self.shuffle:
                    random.shuffle(g)
                batch.extend(g[:self.K])
            if len(batch) >= self.K:
                yield batch[:B]

    def __len__(self) -> int:
        return math.ceil(len(self.ids) * self.K / max(1, self.batch_size))


class ReplicateBatchSampler(Sampler):
    def __init__(self, records, batch_size, K=4, seed=0):
        groups = defaultdict(list)
        for i, rec in enumerate(records):
            groups[rec["smi_key"]].append(i)
        rep_groups = [g for g in groups.values() if len(g) >= 2]

        self.rep_groups = rep_groups
        self.batch_size = batch_size
        self.K = K
        self.rng = random.Random(seed)

    def __iter__(self):
        pool = self.rep_groups[:]
        self.rng.shuffle(pool)
        buf = []
        for g in pool:
            glist = g[:]
            self.rng.shuffle(glist)
            take = glist[:self.K]
            if len(take) < self.K:
                continue
            buf.extend(take)
            if len(buf) >= self.batch_size:
                yield buf[:self.batch_size]
                buf = []
        if buf:
            # pad (rare)
            while len(buf) < self.batch_size:
                buf.append(buf[self.rng.randrange(len(buf))])
            yield buf

    def __len__(self):
        per_batch_ids = max(1, self.batch_size // max(1, self.K))
        return max(1, len(self.rep_groups) // per_batch_ids)
