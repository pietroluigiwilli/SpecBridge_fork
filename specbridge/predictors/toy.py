"""Toy spectrum predictor for demo training."""
import torch
import torch.nn as nn

class ToySpecPredictor(nn.Module):
    """Simple MLP that predicts spectra from graph features."""
    def __init__(self, graph_dim: int = 512, spec_bins: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(graph_dim, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, spec_bins)
        )
    
    def forward(self, graph_feats: torch.Tensor, meta: dict) -> torch.Tensor:
        """Predict spectrum from graph features.
        
        Args:
            graph_feats: [B, graph_dim] graph embedding
            meta: dict with metadata (not used in toy predictor)
        
        Returns:
            [B, spec_bins] predicted spectrum
        """
        return self.net(graph_feats)
