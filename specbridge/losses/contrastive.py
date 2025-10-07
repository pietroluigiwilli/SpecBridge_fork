from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from specbridge.utils.common import unit_normalize

class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07, learnable_temp: bool = True):
        super().__init__()
        if learnable_temp:
            self.log_tau = nn.Parameter(torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('tau', torch.tensor(temperature))
        self.learnable = learnable_temp
    def forward(self, z_s: torch.Tensor, z_m: torch.Tensor) -> torch.Tensor:
        if z_s.dim() == 1: z_s = z_s.unsqueeze(0)
        if z_m.dim() == 1: z_m = z_m.unsqueeze(0)
        z_s = unit_normalize(z_s)
        z_m = unit_normalize(z_m)
        logits = z_s @ z_m.T
        tau = torch.exp(self.log_tau) if self.learnable else self.tau
        logits = logits / tau.clamp(min=1e-6)
        logp_i = F.log_softmax(logits, dim=1)
        logp_t = F.log_softmax(logits.T, dim=1)
        loss_i = -torch.diagonal(logp_i).mean()
        loss_t = -torch.diagonal(logp_t).mean()
        return 0.5 * (loss_i + loss_t)

@torch.no_grad()
def _embed_smiles_list(smiles, featurizer, mol_encoder, mol_adapter, device):
    if not smiles: 
        return torch.empty(0, mol_adapter.proj.out_features, device=device)
    fp = featurizer.featurize(smiles).to(device)
    z0 = mol_encoder(fp); z = F.normalize(mol_adapter.proj(z0), dim=-1)
    return z

def isomer_ce(mu_s, z_m_pos, negs_per_anchor, temperature=0.07):
    mu = F.normalize(mu_s, dim=-1)
    pos = F.normalize(z_m_pos, dim=-1)
    losses = []
    for i in range(mu.size(0)):
        neg_i = negs_per_anchor[i]  # [n_i, D] (can be empty)
        if neg_i.numel() == 0:
            sims = (mu[i:i+1] @ pos[i:i+1].T) / temperature  # [1,1]
            losses.append(F.cross_entropy(sims, torch.zeros(1, dtype=torch.long, device=mu.device)))
            continue
        cand = torch.cat([pos[i:i+1], F.normalize(neg_i,dim=-1)], dim=0)  # [1+n_i, D]
        sims = (mu[i:i+1] @ cand.T) / temperature                         # [1,1+n_i]
        losses.append(F.cross_entropy(sims, torch.zeros(1, dtype=torch.long, device=mu.device)))
    return torch.stack(losses).mean()
