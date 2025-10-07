import torch, torch.nn.functional as F

def supcon_loss(z_q, z_k, keys, temperature=0.07):
    z_q = F.normalize(z_q, dim=-1); z_k = F.normalize(z_k, dim=-1)
    sim = (z_q @ z_k.T) / max(1e-6, temperature)        # [B,B]
    B = sim.size(0); device = sim.device

    # build positive mask from keys
    key_to_idx = {}
    for i, k in enumerate(keys):
        key_to_idx.setdefault(k, []).append(i)
    P = torch.zeros(B, B, dtype=torch.bool, device=device)
    for idxs in key_to_idx.values():
        for i in idxs:
            for j in idxs:
                if i != j: P[i, j] = True

    logp = sim.log_softmax(dim=1)
    pos_counts = P.sum(dim=1).clamp(min=1)
    loss = -(logp[P].sum() / pos_counts.sum())
    return loss
