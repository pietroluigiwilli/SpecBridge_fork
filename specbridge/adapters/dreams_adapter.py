from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from specbridge.utils.common import unit_normalize

class DummyDreams(nn.Module):
    def __init__(self, d_in: int = 2048, d_out: int = 1024):
        super().__init__()
        self.embed_dim = d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, 1024), nn.GELU(),
            nn.Linear(1024, d_out)
        )
    def forward(self, spectra_binned: torch.Tensor, meta: dict):
        return self.net(spectra_binned)


def load_dreams_encoder(dreams_ckpt: Optional[str] = None, d_in: int = 2048, d_out: int = 1024) -> nn.Module:
    if dreams_ckpt is not None:
        try:
            from dreams.api import PreTrainedModel  # type: ignore
            from dreams.models.dreams.dreams import DreaMS as DreaMSModel  # type: ignore
            ptm = PreTrainedModel.from_ckpt(
                ckpt_path=dreams_ckpt,
                ckpt_cls=DreaMSModel,
                n_highest_peaks=60,
            )
            model = ptm.model
            return model.eval()
        except Exception:
            pass
        try:
            from dreams import DreamsEncoder  # type: ignore
            model = DreamsEncoder.load_from_checkpoint(dreams_ckpt)
            return model.eval()
        except Exception:
            pass
    model = DummyDreams(d_in=d_in, d_out=d_out)
    if dreams_ckpt is None:
        return model
    try:
        sd = torch.load(dreams_ckpt, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        model.load_state_dict(sd, strict=False)
    except Exception:
        pass
    return model


class DreamsAdapter(nn.Module):
    def __init__(self, dreams_encoder, d_out, hidden=0, freeze_backbone=True,
                 pool: str = "mean", peak_dropout: float = 0.0):
        super().__init__()
        self.dreams = dreams_encoder
        self.pool = pool
        self.peak_dropout = peak_dropout
        d_in = getattr(dreams_encoder, 'embed_dim', 1024)
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out) if hidden == 0 else nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)
            ),
            nn.LayerNorm(d_out),
        )
        if freeze_backbone:
            for p in self.dreams.parameters():
                p.requires_grad = False
        if pool == "attn":
            self.attn = nn.Linear(d_in, 1)

    def _pool(self, seq, peaks):
        # peaks: [B,N,2]; mask true where mz>0
        mask = (peaks[..., 0] > 0).float()  # [B,N]
        if self.training and self.peak_dropout > 0:
            keep = (torch.rand_like(mask) > self.peak_dropout).float()
            mask = mask * keep
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        if self.pool == "max":
            # masked max
            v = seq.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            pooled = torch.nan_to_num(v.max(dim=1).values, nan=0.0, neginf=0.0)
        elif self.pool == "attn":
            w = self.attn(seq).squeeze(-1)  # [B,N]
            w = w + (mask + 1e-6).log()     # mask out invalid peaks
            w = torch.softmax(w, dim=1)
            pooled = (seq * w.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = (seq * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def forward(self, spectra_binned, meta):
        with torch.no_grad():
            if isinstance(meta, dict) and 'peaks' in meta and isinstance(meta['peaks'], torch.Tensor):
                out = self.dreams(meta['peaks'], meta)  # could be [B,N,D] or [B,D]
                if out.dim() == 3:
                    z0 = self._pool(out, meta['peaks'])
                else:
                    z0 = out
            else:
                z0 = self.dreams(spectra_binned, meta)
        z = self.proj(z0)
        return unit_normalize(z)

    def unfreeze_last(self, n_layers: int = 1):
        if hasattr(self.dreams, 'unfreeze_last') and callable(getattr(self.dreams, 'unfreeze_last')):
            try:
                self.dreams.unfreeze_last(n_layers=n_layers)  # type: ignore
                return
            except Exception:
                pass
        children = list(self.dreams.children())
        if len(children) == 0:
            for p in list(self.dreams.parameters())[-1 * max(1, n_layers)]:
                p.requires_grad = True
            return
        for mod in children[-n_layers:]:
            for p in mod.parameters():
                p.requires_grad = True
