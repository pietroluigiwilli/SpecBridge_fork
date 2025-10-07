from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple
from specbridge.utils.common import unit_normalize

class MapperB(nn.Module):
    def __init__(self, d: int, hidden: int = 0, gaussian: bool = True, ortho_weight: float = 1e-3):
        super().__init__()
        self.d = d
        self.gaussian = gaussian
        self.ortho_weight = ortho_weight
        if hidden > 0:
            self.mu = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
            self.lv = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)) if gaussian else None
        else:
            self.mu = nn.Linear(d, d)
            self.lv = nn.Linear(d, d) if gaussian else None
        if isinstance(self.mu, nn.Linear):
            nn.init.orthogonal_(self.mu.weight)
            nn.init.zeros_(self.mu.bias)
    def forward(self, z_s: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mu = self.mu(z_s)
        logvar = self.lv(z_s).clamp(min=-6.0, max=6.0) if self.lv is not None else None
        return mu, logvar
    def sample(self, mu: torch.Tensor, logvar: Optional[torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        if logvar is None or deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        return unit_normalize(z)
    def orthogonality_penalty(self) -> torch.Tensor:
        if isinstance(self.mu, nn.Linear):
            W = self.mu.weight
            I = torch.eye(W.size(0), device=W.device)
            return self.ortho_weight * ((W.T @ W - I)**2).mean()
        elif isinstance(self.mu, nn.Sequential) and isinstance(self.mu[0], nn.Linear):
            W = self.mu[0].weight
            I = torch.eye(W.size(0), device=W.device)
            return self.ortho_weight * ((W.T @ W - I)**2).mean()
        return torch.tensor(0.0, device=next(self.parameters()).device)
