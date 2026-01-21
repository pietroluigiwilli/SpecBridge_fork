from __future__ import annotations
from typing import Optional
from argparse import Namespace
from pathlib import Path
import types
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


def _create_default_dreams_args(n_highest_peaks: int = 60):
    """Create default arguments for DreaMS model initialization.
    
    Default values are based on the DreaMS pre-training configuration (ssl_model.ckpt).
    These match the architecture used in the pretrained checkpoint.
    """
    from dreams.utils.dformats import DataFormatA  # type: ignore
    
    dformat = DataFormatA()
    
    args = Namespace(
        gains_dir=Path('.'),
        n_layers=7,
        n_heads=8,
        train_objective='mask_mz_hot',
        lr=1e-4,
        weight_decay=0.0,
        charge_feature=False,
        d_fourier=980,
        d_peak=44,
        d_mz_token=0,
        dformat=dformat,
        hot_mz_bin_size=0.05,
        n_warmup_steps=5000,
        vanilla_transformer=False,
        batch_size=32,
        log_figs=False,
        entropy_label_smoothing=0.0,
        graphormer_mz_diffs=True,
        graphormer_parametrized=False,
        fourier_strategy='lin_float_int',
        ret_order_loss_w=0.0,
        cos_reg_alpha=0.0,
        cos_reg_reduction=None,
        mask_val=-1.0,
        fourier_num_freqs=None,  # Matches ssl_model.ckpt (None for lin_float_int)
        fourier_trainable=False,
        fourier_min_freq=None,
        dropout=0.1,
        focal_loss_gamma=5.0,  # Used for mask_mz_hot objective
        focal_loss_alpha=None,
        att_dropout=0.1,
        residual_dropout=0.1,
        ff_dropout=0.1,
        ff_fourier_depth=5,
        ff_fourier_d=512,
        ff_peak_depth=1,
        ff_out_depth=1,
        no_ffs_bias=False,
        no_transformer_bias=True,
        pre_norm=True,
        scnorm=False,
        attn_mech='dot-product',
    )
    return args


def load_dreams_encoder(dreams_ckpt: Optional[str] = None, d_in: int = 2048, d_out: int = 1024, 
                        init_from_scratch: bool = False, n_highest_peaks: int = 60) -> nn.Module:
    """
    Load DreaMS encoder from checkpoint or initialize from scratch.
    
    Args:
        dreams_ckpt: Path to checkpoint file. If None and init_from_scratch=False, uses dummy encoder.
        d_in: Input dimension for dummy encoder (only used if no checkpoint and not init_from_scratch).
        d_out: Output dimension for dummy encoder (only used if no checkpoint and not init_from_scratch).
        init_from_scratch: If True, initialize DreaMS model from scratch without loading checkpoint.
        n_highest_peaks: Number of highest peaks to use (for initialization from scratch or checkpoint loading).
    
    Returns:
        DreaMS encoder model in eval mode.
    """
    # Try to initialize from scratch if requested
    if init_from_scratch:
        try:
            from dreams.models.dreams.dreams import DreaMS as DreaMSModel  # type: ignore
            import dreams.utils.data as du  # type: ignore
            import dreams.utils.dformats as dformats  # type: ignore
            
            args = _create_default_dreams_args(n_highest_peaks=n_highest_peaks)
            spec_preproc = du.SpectrumPreprocessor(
                dformat=dformats.DataFormatA(),
                n_highest_peaks=n_highest_peaks
            )
            model = DreaMSModel(args, spec_preproc)
            # Set embed_dim for compatibility with DreamsAdapter
            model.embed_dim = model.d_model
            
            # Remove task-specific heads to match pretrained checkpoint architecture
            # This ensures consistency: when loading from checkpoint, PreTrainedModel.from_ckpt()
            # calls remove_unused_backbone_parameters() which removes these heads.
            # We use the same function here so from-scratch and from-checkpoint have identical architectures.
            from dreams.api import PreTrainedModel  # type: ignore
            model = PreTrainedModel.remove_unused_backbone_parameters(model)
            
            print(f"Initialized DreaMS model from scratch (n_highest_peaks={n_highest_peaks}, embed_dim={model.embed_dim})")
            return model
        except Exception as e:
            print(f"Error initializing DreaMS from scratch: {e}")
            if init_from_scratch:
                raise
            # Fall through to checkpoint loading or dummy
    
    # Try to load from checkpoint
    if dreams_ckpt is not None:
        try:
            from dreams.api import PreTrainedModel  # type: ignore
            from dreams.models.dreams.dreams import DreaMS as DreaMSModel  # type: ignore
            ptm = PreTrainedModel.from_ckpt(
                ckpt_path=dreams_ckpt,
                ckpt_cls=DreaMSModel,
                n_highest_peaks=n_highest_peaks,
            )
            model = ptm.model
            print(f"Using dreams encoder from checkpoint {dreams_ckpt}")
            return model.eval()
        except Exception as e:
            print(f"Error using dreams encoder from checkpoint {dreams_ckpt}: {e}")
            pass
        try:
            from dreams import DreamsEncoder  # type: ignore
            model = DreamsEncoder.load_from_checkpoint(dreams_ckpt)
            print(f"Using dreams encoder from checkpoint {dreams_ckpt} using DreamsEncoder")
            return model.eval()
        except Exception as e:
            print(f"Error using dreams encoder from checkpoint {dreams_ckpt} using DreamsEncoder: {e}")
            pass
    
    # Fall back to dummy encoder
    print(f"Using dummy dreams encoder")
    model = DummyDreams(d_in=d_in, d_out=d_out)
    if dreams_ckpt is not None:
        try:
            sd = torch.load(dreams_ckpt, map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"Warning: Could not load DreaMS checkpoint {dreams_ckpt}: {e}")
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
        # Only use no_grad if all parameters are frozen
        use_no_grad = not any(p.requires_grad for p in self.dreams.parameters())
        
        # Patch __normalize_spec to use input tensor's device/dtype instead of self.device/self.dtype
        # This fixes device mismatch issues when model is called directly (not through PyTorch Lightning)
        if not hasattr(self.dreams, '_normalize_spec_patched'):
            def patched_normalize(self, spec):
                # Use spec's device and dtype instead of self.device/self.dtype
                return spec / torch.tensor([self.dformat.max_mz, 1.], device=spec.device, dtype=spec.dtype)
            # Bind the patched method to the instance
            self.dreams._DreaMS__normalize_spec = types.MethodType(patched_normalize, self.dreams)
            self.dreams._normalize_spec_patched = True
        
        if use_no_grad:
            with torch.no_grad():
                if isinstance(meta, dict) and 'peaks' in meta and isinstance(meta['peaks'], torch.Tensor):
                    out = self.dreams(meta['peaks'], meta)  # could be [B,N,D] or [B,D]
                    if out.dim() == 3:
                        z0 = self._pool(out, meta['peaks'])
                    else:
                        z0 = out
                else:
                    z0 = self.dreams(spectra_binned, meta)
        else:
            # When encoder is trainable, don't use no_grad
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
        """Unfreeze the last N transformer layers in DreaMS model.
        
        This method specifically targets transformer layers in the DreaMS architecture,
        which has transformer_encoder.atts and transformer_encoder.ffs as ModuleLists.
        """
        # First, try if the model has its own unfreeze_last method
        if hasattr(self.dreams, 'unfreeze_last') and callable(getattr(self.dreams, 'unfreeze_last')):
            try:
                self.dreams.unfreeze_last(n_layers=n_layers)  # type: ignore
                return
            except Exception:
                pass
        
        # For DreaMS models, we need to target transformer_encoder layers specifically
        if hasattr(self.dreams, 'transformer_encoder'):
            te = self.dreams.transformer_encoder
            
            # Find transformer layers (atts and ffs are ModuleLists)
            atts = None
            ffs = None
            if hasattr(te, 'atts') and isinstance(te.atts, torch.nn.ModuleList):
                atts = te.atts
            if hasattr(te, 'ffs') and isinstance(te.ffs, torch.nn.ModuleList):
                ffs = te.ffs
            
            if atts is not None or ffs is not None:
                # Determine how many layers to unfreeze
                n_att_layers = len(atts) if atts is not None else 0
                n_ff_layers = len(ffs) if ffs is not None else 0
                n_total_layers = max(n_att_layers, n_ff_layers)
                
                if n_total_layers == 0:
                    # No transformer layers found, fall back to generic method
                    pass
                else:
                    n_unfreeze = min(n_layers, n_total_layers)
                    
                    # Unfreeze last N attention layers
                    if atts is not None and len(atts) > 0:
                        for layer in atts[-n_unfreeze:]:
                            for p in layer.parameters():
                                p.requires_grad = True
                    
                    # Unfreeze last N feedforward layers
                    if ffs is not None and len(ffs) > 0:
                        for layer in ffs[-n_unfreeze:]:
                            for p in layer.parameters():
                                p.requires_grad = True
                    
                    return
        
        # Fallback: generic unfreezing of last N children
        children = list(self.dreams.children())
        if len(children) == 0:
            # If no children, try to unfreeze last N parameters (this is rarely correct)
            params = list(self.dreams.parameters())
            if len(params) > 0:
                # This is a fallback - not ideal but better than nothing
                n_unfreeze = min(n_layers, len(params))
                for p in params[-n_unfreeze:]:
                    p.requires_grad = True
            return
        
        # Unfreeze last N child modules
        n_unfreeze = min(n_layers, len(children))
        for mod in children[-n_unfreeze:]:
            for p in mod.parameters():
                p.requires_grad = True

