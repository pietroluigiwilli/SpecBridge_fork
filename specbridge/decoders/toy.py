import torch
import torch.nn as nn

class ToyDecoder(nn.Module):
    def __init__(self, cond_dim: int, graph_dim: int = 512):
        super().__init__()
        self.graph_dim = graph_dim
        self.net = nn.Sequential(
            nn.LazyLinear(1024), nn.GELU(),
            nn.Linear(1024, graph_dim)
        )
        self.target_head = nn.Sequential(
            nn.LazyLinear(512), nn.GELU(), nn.Linear(512, graph_dim)
        )
    def forward(self, target_graph_feats: torch.Tensor, cond: torch.Tensor, formula: torch.Tensor, adduct: torch.Tensor, charge: torch.Tensor):
        meta_embed = torch.cat([formula, adduct, charge], dim=-1)
        x = torch.cat([cond, meta_embed], dim=-1)
        pred_graph = self.net(x)
        target = self.target_head(x).detach()
        loss = ((pred_graph - target_graph_feats)**2).mean() + ((pred_graph - target)**2).mean()
        return {"loss": loss, "graph": pred_graph}
