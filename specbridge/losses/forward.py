from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ForwardSpecLossConfig:
    intensity_weight: float = 1.0

class ForwardSpectralLoss(nn.Module):
    def __init__(self, cfg: ForwardSpecLossConfig = ForwardSpecLossConfig()):
        super().__init__()
        self.cfg = cfg
    def forward(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        s_pred = F.normalize(s_pred, p=2, dim=-1)
        s_true = F.normalize(s_true, p=2, dim=-1)
        return self.cfg.intensity_weight * ((s_pred - s_true) ** 2).mean()
