import torch
import torch.nn as nn

class ToySpecPredictor(nn.Module):
    def __init__(self, graph_dim: int, spec_bins: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim, 1024), nn.GELU(),
            nn.Linear(1024, spec_bins)
        )
    def forward(self, graph_feats: torch.Tensor, meta: dict) -> torch.Tensor:
        return self.net(graph_feats)
