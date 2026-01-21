#!/usr/bin/env python3
"""
Calculate and display parameter counts for SpecBridge model components.
"""
import torch
import torch.nn as nn
from specbridge.adapters.dreams_adapter import load_dreams_encoder, DummyDreams
from specbridge.models.mapper import DreamsToMolCondition
import argparse

def count_parameters(model, name="Model"):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def format_params(num):
    """Format parameter count as K, M, etc."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f} M"
    elif num >= 1_000:
        return f"{num / 1_000:.0f} K"
    else:
        return str(num)

def main():
    parser = argparse.ArgumentParser(description="Calculate SpecBridge model parameters")
    parser.add_argument("--spec-bins", type=int, default=2048, help="Spectral bins")
    parser.add_argument("--fp-bits", type=int, default=2048, help="Fingerprint bits")
    parser.add_argument("--cond-dim", type=int, default=2048, help="Conditioning dimension")
    parser.add_argument("--mapper-hidden", type=int, default=2048, help="Mapper hidden dim (0=linear)")
    parser.add_argument("--n-blocks", type=int, default=8, help="Number of residual blocks in mapper")
    parser.add_argument("--no-gaussian", action="store_true", help="Disable Gaussian head")
    parser.add_argument("--mol-space", choices=["ecfp", "chemberta", "adapter"], default="chemberta")
    parser.add_argument("--chemberta-model", type=str, default="Derify/ChemBERTa_augmented_pubchem_13m")
    parser.add_argument("--dreams-ckpt", type=str, default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt", help="Path to DreaMS checkpoint")
    parser.add_argument("--init-spec-from-scratch", action="store_true", help="Initialize spec encoder from scratch")
    parser.add_argument("--init-mol-from-scratch", action="store_true", help="Initialize mol encoder from scratch")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze DreaMS backbone")
    parser.add_argument("--unfreeze-last", type=int, default=2, help="Unfreeze last N layers in DreaMS backbone")
    parser.add_argument("--unfreeze-after", type=int, default=0, help="Unfreeze dreams after N steps (0=immediate)")
    
    args = parser.parse_args()
    
    # Default to match training command: training uses --no-gaussian
    # So we default no_gaussian=True (gaussian disabled) to match training behavior
    # Check if user explicitly provided the flag
    import sys
    if '--no-gaussian' not in sys.argv:
        args.no_gaussian = True  # Default to match training command
    
    device = torch.device("cpu")
    
    # Create a mock args object for DreamsToMolCondition
    class MockArgs:
        def __init__(self):
            self.n_blocks = args.n_blocks
            self.random_mapper_init = False
    
    mock_args = MockArgs()
    
    print("=" * 60)
    print("SpecBridge Model Parameter Count")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  spec_bins: {args.spec_bins}")
    print(f"  cond_dim: {args.cond_dim}")
    print(f"  mapper_hidden: {args.mapper_hidden}")
    print(f"  n_blocks: {args.n_blocks}")
    print(f"  gaussian: {not args.no_gaussian}")
    print(f"  mol_space: {args.mol_space}")
    print(f"  chemberta_model: {args.chemberta_model}")
    print(f"  unfreeze_last: {args.unfreeze_last}")
    print(f"  unfreeze_after: {args.unfreeze_after}")
    print("=" * 60)
    print()
    
    # Load DreaMS encoder
    if args.dreams_ckpt:
        dreams_backbone = load_dreams_encoder(
            args.dreams_ckpt,
            d_in=args.spec_bins,
            d_out=1024,
            init_from_scratch=args.init_spec_from_scratch
        )
        # DreaMS models are typically 20-50M parameters
        dreams_size = sum(p.numel() for p in dreams_backbone.parameters())
        if dreams_size < 1_000_000:
            print(f"[WARNING] DreaMS model seems very small ({format_params(dreams_size)})")
            print(f"[WARNING] Expected DreaMS to be ~20-50M parameters. Check if checkpoint loaded correctly.")
    else:
        print("[WARNING] No --dreams-ckpt provided, using DummyDreams (very small model)")
        print("[WARNING] For accurate parameter counts, provide --dreams-ckpt path")
        print("[WARNING] Example: --dreams-ckpt /path/to/ssl_model.ckpt")
        dreams_backbone = DummyDreams(d_in=args.spec_bins, d_out=1024)
    
    # Create main model
    model = DreamsToMolCondition(
        dreams_backbone,
        d_out=args.cond_dim,
        mapper_hidden=args.mapper_hidden,
        gaussian=not args.no_gaussian,
        mol_space=args.mol_space,
        chemberta_model=args.chemberta_model,
        args=mock_args,
        freeze_backbone=args.freeze_backbone,
        init_mol_from_scratch=args.init_mol_from_scratch
    ).to(device)
    
    # Handle unfreezing logic (matching training script exactly)
    if args.unfreeze_last > 0 and args.unfreeze_after == 0:
        model.spec.unfreeze_last(n_layers=args.unfreeze_last)
        print(f"[adapter] unfroze last {args.unfreeze_last} layer(s) of DreaMS (immediate)")
    
    # Count parameters for each component
    components = []
    
    # 1. DreaMS backbone (may be frozen)
    if hasattr(model.spec, 'dreams'):
        dreams_total, dreams_trainable = count_parameters(model.spec.dreams, "DreaMS backbone")
        components.append(("dreams_backbone", "DreaMS", dreams_total, dreams_trainable))
    
    # 2. DreamsAdapter projection
    spec_proj_total, spec_proj_trainable = count_parameters(model.spec.proj, "Spec projection")
    components.append(("spec_proj", "SpecProj", spec_proj_total, spec_proj_trainable))
    
    # 3. ChemBERTa (if used)
    if args.mol_space == "chemberta" and hasattr(model, 'chem_mdl'):
        chem_total, chem_trainable = count_parameters(model.chem_mdl, "ChemBERTa")
        components.append(("chemberta", "ChemBERTa", chem_total, chem_trainable))
        
        # NOTE: chem_proj exists but is NOT used in forward pass (mapper.py line 131 is commented out)
        # Skipping chem_proj from parameter count since it's unused
    
    # 4. Mapper (ProcrustesResidualMapper)
    mapper_total, mapper_trainable = count_parameters(model.mapB, "Mapper")
    components.append(("mapper", "Mapper", mapper_total, mapper_trainable))
    
    # NOTE: Contrastive loss (InfoNCE) is a loss function, not a model component
    # Skipping from parameter count
    
    # Print table
    print(" | Name          | Type              | Total Params | Trainable Params")
    print("-" * 70)
    
    total_all = 0
    trainable_all = 0
    
    for idx, (key, name, total, trainable) in enumerate(components):
        total_all += total
        trainable_all += trainable
        print(f"{idx} | {name:14s} | {key:17s} | {format_params(total):>12s} | {format_params(trainable):>15s}")
    
    print("-" * 70)
    print(f"  | TOTAL         |                  | {format_params(total_all):>12s} | {format_params(trainable_all):>15s}")
    print()
    
    # Additional breakdown
    print("=" * 60)
    print("Detailed Breakdown:")
    print("=" * 60)
    
    # Mapper details
    if hasattr(model.mapB, 'W'):
        w_params = sum(p.numel() for p in model.mapB.W.parameters())
        print(f"  Mapper.W: {format_params(w_params)} params")
    
    if hasattr(model.mapB, 'blocks'):
        blocks_params = sum(p.numel() for p in model.mapB.blocks.parameters())
        print(f"  Mapper.blocks ({len(model.mapB.blocks)} blocks): {format_params(blocks_params)} params")
    
    if hasattr(model.mapB, 'lv') and model.mapB.lv is not None:
        lv_params = sum(p.numel() for p in model.mapB.lv.parameters())
        print(f"  Mapper.lv: {format_params(lv_params)} params")
    
    print()
    print(f"Total trainable parameters: {trainable_all:,} ({format_params(trainable_all)})")
    print(f"Total parameters (including frozen): {total_all:,} ({format_params(total_all)})")

if __name__ == "__main__":
    main()
