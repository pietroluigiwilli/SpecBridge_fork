from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple
from specbridge.utils.common import unit_normalize
from specbridge.adapters.dreams_adapter import DreamsAdapter
from specbridge.losses.contrastive import InfoNCELoss
# Unused losses (commented out in align_losses):
# from specbridge.losses.contrastive import _embed_smiles_list, isomer_ce
# from specbridge.losses.supcon import supcon_loss

import torch.nn.functional as F

class DreamsToMolCondition(nn.Module):
    def __init__(
        self,
        dreams_encoder: nn.Module,
        d_out: int = 512,
        mapper_hidden: int = 0,
        gaussian: bool = True,
        mol_space: str = "adapter",
        chemberta_model: str | None = None,
        args=None,
        freeze_backbone: bool = True,
        init_mol_from_scratch: bool = False
    ):
        super().__init__()
        self.mol_space = mol_space
        self.args = args
        # spec branch (frozen backbone, learnable proj already inside DreamsAdapter)
        self.spec = DreamsAdapter(dreams_encoder, d_out=d_out, hidden=mapper_hidden, freeze_backbone=freeze_backbone)

        # mol branch: either learnable adapter (legacy) or frozen pretrained
        if mol_space == "chemberta":
            assert chemberta_model is not None, "Provide --chemberta-model for --mol-space chemberta"
            from transformers import AutoTokenizer, AutoModel, AutoConfig
            self.chem_tok = AutoTokenizer.from_pretrained(chemberta_model)
            
            if init_mol_from_scratch:
                # Initialize from config with random weights
                config = AutoConfig.from_pretrained(chemberta_model)
                self.chem_mdl = AutoModel.from_config(config)
                # Initialize all parameters randomly
                self.chem_mdl.init_weights()
                print(f"[mol-adapter] Initialized ChemBERTa from scratch (random init) with config from {chemberta_model}")
            else:
                # Load pretrained weights
                self.chem_mdl = AutoModel.from_pretrained(chemberta_model)
            
            # Only freeze if not initialized from scratch
            if not init_mol_from_scratch:
                for p in self.chem_mdl.parameters():
                    p.requires_grad = False  # freeze ChemBERTa
            else:
                # When initialized from scratch, all parameters are trainable
                for p in self.chem_mdl.parameters():
                    p.requires_grad = True
                print(f"[mol-adapter] ChemBERTa is trainable (random init)")
            
            hid = int(self.chem_mdl.config.hidden_size)
            # Trainable projection into your conditioning space
            self.chem_proj = nn.Sequential(
             nn.Sequential(
                nn.Linear(hid, mapper_hidden), nn.GELU(), nn.Linear(mapper_hidden, d_out)
            ),
            nn.LayerNorm(d_out),
        )
            self.mol = None

        else:
            raise ValueError(f"Unknown mol_space: {mol_space}")

        # mapper and contrastive loss
        # self.mapB = MapperB(d_in = d_out, d_out=hid, hidden=mapper_hidden, gaussian=gaussian)
        random_mapper_init = getattr(args, 'random_mapper_init', False) if args is not None else False
        if random_mapper_init:
            print(f"[mapper] Using random (Xavier uniform) initialization instead of orthogonal/Procrustes init")
        self.mapB = ProcrustesResidualMapper(d_in = d_out, d_out=hid, n_blocks=args.n_blocks, hidden=mapper_hidden, gaussian=gaussian, random_init=random_mapper_init)
        self.contrast = InfoNCELoss(temperature=0.07, learnable_temp=True)

    def _chemberta_embed(self, smiles: list[str], device: torch.device) -> torch.Tensor:
        assert self.chem_tok is not None and self.chem_mdl is not None and self.chem_proj is not None
        toks = self.chem_tok(smiles, padding=True, truncation=True, return_tensors="pt").to(device)
        # Only use no_grad if all parameters are frozen
        use_no_grad = not any(p.requires_grad for p in self.chem_mdl.parameters())
        if use_no_grad:
            with torch.no_grad():
                h = self.chem_mdl(**toks).last_hidden_state[:, 0]  # CLS [B, hidden]
        else:
            h = self.chem_mdl(**toks).last_hidden_state[:, 0]  # CLS [B, hidden]
        return  h

    def unfreeze_mol_last(self, n_layers: int = 1):
        """Unfreeze the last N layers of the molecule encoder (ChemBERTa)."""
        if self.mol_space == "chemberta" and hasattr(self, "chem_mdl"):
            # For BERT-based models, layers are in encoder.layer
            if hasattr(self.chem_mdl, "encoder") and hasattr(self.chem_mdl.encoder, "layer"):
                layers = list(self.chem_mdl.encoder.layer)
                n_total = len(layers)
                n_unfreeze = min(n_layers, n_total)
                for layer in layers[-n_unfreeze:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                print(f"[mol-adapter] unfroze last {n_unfreeze} layer(s) of ChemBERTa (total {n_total} layers)")
            else:
                # Fallback: try to unfreeze last N children modules
                children = list(self.chem_mdl.children())
                if len(children) > 0:
                    for mod in children[-n_layers:]:
                        for p in mod.parameters():
                            p.requires_grad = True
                    print(f"[mol-adapter] unfroze last {n_layers} module(s) of ChemBERTa")
        else:
            print(f"[mol-adapter] unfreeze_mol_last ignored (mol_space='{self.mol_space}' has no ChemBERTa)")

    def train(self, mode: bool = True):
        super().train(mode)
        # Only keep chemBERTa in eval if all layers are frozen
        if getattr(self, "chem_mdl", None) is not None:
            if not any(p.requires_grad for p in self.chem_mdl.parameters()):
                self.chem_mdl.eval()
        return self
    def forward(self, spectra_binned, meta, mol_feats, inference: bool = False):
        """
        meta must include a list[str] SMILES under key 'smi_key' (or 'smiles').
        mol_feats is only used when mol_space == 'ecfp' (0/1 fingerprint tensor).
        """
        z_s = self.spec(spectra_binned, meta)

        if self.mol_space == "chemberta":
            smiles = meta.get("smi_key", None) or meta.get("smiles", None)
            if smiles is None:
                raise RuntimeError("Need meta['smi_key'] or meta['smiles'] (list[str]) for ChemBERTa embedding.")
            z_m = self._chemberta_embed(smiles, z_s.device)
            # z_m = F.normalize(self.chem_proj(z_m), dim=-1)

        else:
            raise RuntimeError("unreachable")

        mu_s, lv_s = self.mapB(z_s)
        # z_hat = self.mapB.sample(mu_s, lv_s, deterministic=inference)
        return z_s, z_m, None , mu_s, lv_s

    def align_losses(
        self,
        z_s, z_m, mu_s, lv_s,
        w_con=1.0, w_map=1.0, w_ortho=1e-3, w_con_mapped=1.0,
        stop_mol_in_con=True,
        supcon_keys=None, w_sup=0.0, sup_temp=0.07,
    ):
        # z_s_n = F.normalize(z_s, dim=-1)
        # z_m_n = F.normalize(z_m, dim=-1)
        # mu_n  = F.normalize(mu_s, dim=-1)

        z_s_n = z_s
        z_m_n = z_m
        mu_n  = mu_s
        z_m_for_con = z_m_n.detach() if stop_mol_in_con else z_m_n
        # L_con   = self.contrast(z_s_n, z_m_for_con)
        
        # InfoNCE loss: matches mu_s[i] with z_m[i] (1-to-1 diagonal matching)
        # Note: With K replicates per SMILES (from BalancedBatchSampler), z_m will have 
        # duplicate embeddings for samples with the same SMILES. This is fine - InfoNCE 
        # will learn that multiple mu_s (from different spectra) should match the same z_m.
        # Skip computation if weight is 0 to avoid unnecessary work.
        L_con_m = self.contrast(mu_n, z_m_n.detach()) if w_con_mapped > 0 else torch.tensor(0.0, device=mu_n.device)
        
        # MSE loss: each spectrum (mu_s[i]) tries to match its molecule embedding (z_m[i])
        # With K replicates, this gives K training examples per molecule per batch,
        # which is beneficial for learning robust spectrum→molecule mappings.
        L_map = F.mse_loss(mu_n, z_m_n, reduction='mean') # * mu_n.size(1)
        # L_map = 1 - torch.mean(F.cosine_similarity(mu_n, z_m_n, dim=1), dim=0)
        L_ortho = self.mapB.orthogonality_penalty() * w_ortho

        # L_sup = torch.tensor(0.0, device=z_s.device)
        # if (w_sup > 0.0) and (supcon_keys is not None):
        #     L_sup = supcon_loss(z_s, z_m, supcon_keys, temperature=sup_temp)

        L =  (w_con_mapped * L_con_m)  + L_ortho + (w_map * L_map) # + (w_sup * L_sup) # + (w_map * L_map)
        logs = { "L_con_m": L_con_m.detach() if isinstance(L_con_m, torch.Tensor) else torch.tensor(0.0, device=mu_n.device),
                "L_ortho": L_ortho.detach(), "L_map": L_map.detach()}
        return L, logs





class MapperB(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 0, gaussian: bool = True, ortho_weight: float = 1e-3):
        super().__init__()
        self.gaussian = gaussian
        self.ortho_weight = ortho_weight
        if hidden > 0:
            self.mu = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))
            self.lv = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)) if gaussian else None
        else:
            self.mu = nn.Linear(d_in, d_out)
            self.lv = nn.Linear(d_in, d_out) if gaussian else None
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

class ResidualBlock(nn.Module):
    def __init__(self, d, hidden, drop=0.0):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        # small init on residual branch
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.7)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.ln(x)
        h = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + h

class ProcrustesResidualMapper(nn.Module):
    """
    x_s -> z_m: W0 (near-orthogonal) + small residual MLP.
    """
    def __init__(self, d_in, d_out, n_blocks=4, hidden=0, drop=0.0, gaussian=False, random_init=False):
        super().__init__()
        self.gaussian = gaussian
        # main linear
        self.W = nn.Linear(d_in, d_out, bias=True)
        if random_init:
            # Random initialization (Xavier uniform)
            nn.init.xavier_uniform_(self.W.weight)
        else:
            # Procrustes initialization (orthogonal)
            nn.init.orthogonal_(self.W.weight)  # good default
            nn.init.zeros_(self.W.bias)
        # residual stack (light capacity)
        hidden = hidden if hidden > 0 else max(256, min(d_out, 1024))
        self.blocks = nn.ModuleList([ResidualBlock(d_out, hidden, drop) for _ in range(n_blocks)])
        # optional Gaussian head, if you still want it:
        self.lv = nn.Linear(d_out, d_out) if gaussian else None

    @torch.no_grad()
    def procrustes_init(self, X, Y, center=True):
        """
        One-shot orthogonal LS fit W*: minimize ||X W - Y||_F with orthogonality on W.
        X: [B, d_in], Y: [B, d_out]. If d_in!=d_out we fit rectangular via SVD of cross-cov.
        """
        if center:
            X = X - X.mean(0, keepdim=True)
            Y = Y - Y.mean(0, keepdim=True)
        # cross-cov
        C = X.T @ Y  # [d_in, d_out]
        # SVD
        U, S, Vh = torch.linalg.svd(C, full_matrices=False)
        W_ortho = U @ Vh  # [d_in, d_out] if shapes align; PyTorch will broadcast appropriately
        # copy into weight (transpose because Linear stores [out,in])
        with torch.no_grad():
            if self.W.weight.shape == W_ortho.T.shape:
                self.W.weight.copy_(W_ortho.T)
            else:
                # fallback: solve unconstrained LS and orthogonalize row-space
                W_ls, *_ = torch.linalg.lstsq(X, Y)  # [d_in, d_out]
                self.W.weight.copy_(W_ls.T)
        # bias as mean offset
        self.W.bias.zero_()

    def forward(self, x):
        mu = self.W(x)
        for blk in self.blocks:
            mu = blk(mu)
        if self.gaussian:
            lv = self.lv(mu)
            return mu, lv
        return mu, None

    def orthogonality_penalty(self, strength=1e-3):
        # encourage W.weight to be orthogonal (rows or cols depending on shape)
        W = self.W.weight  # [d_out, d_in]
        if W.size(0) <= W.size(1):
            G = W @ W.T
            I = torch.eye(G.size(0), device=W.device, dtype=W.dtype)
        else:
            G = W.T @ W
            I = torch.eye(G.size(0), device=W.device, dtype=W.dtype)
        return strength * ((G - I).pow(2).mean())


