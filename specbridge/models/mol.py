from __future__ import annotations
from typing import Iterable
import numpy as np
import torch
import torch.nn as nn

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _HAS_RDKIT = True
except Exception:
    _HAS_RDKIT = False

class MolFeaturizer:
    def __init__(self, fp_bits: int = 2048, fp_radius: int = 2):
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
    def _det_random(self, s: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(abs(hash(s)) % (2**31))
        return (torch.rand(self.fp_bits, generator=g) > 0.5).float()
    def featurize(self, smiles_list: Iterable[str]) -> torch.Tensor:
        fps = []
        if _HAS_RDKIT:
            for s in smiles_list:
                m = Chem.MolFromSmiles(s)
                if m is None:
                    vec = self._det_random(s)
                else:
                    fp = AllChem.GetMorganFingerprintAsBitVect(m, self.fp_radius, nBits=self.fp_bits)
                    arr = np.zeros((self.fp_bits,), dtype=np.float32)
                    DataStructs.ConvertToNumpyArray(fp, arr)
                    vec = torch.from_numpy(arr)
                fps.append(vec)
        else:
            for s in smiles_list:
                fps.append(self._det_random(s))
        return torch.stack(fps, dim=0)

class MolEncoder(nn.Module):
    def __init__(self, d_in: int = 2048, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(d_in, 1024), nn.GELU(),
            nn.Linear(1024, embed_dim)
        )
    def forward(self, x_fp: torch.Tensor):
        return self.net(x_fp)

class MolAdapter(nn.Module):
    def __init__(self, mol_encoder: nn.Module, d_out: int, hidden: int = 0):
        super().__init__()
        self.mol_encoder = mol_encoder
        d_in = getattr(mol_encoder, 'embed_dim', 512)
        proj = nn.Linear(d_in, d_out) if hidden == 0 else nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)
        )
        self.proj = nn.Sequential(proj, nn.LayerNorm(d_out))
    def forward(self, mol_feats):
        from specbridge.utils.common import unit_normalize
        z0 = self.mol_encoder(mol_feats)
        z = self.proj(z0)
        return unit_normalize(z)
