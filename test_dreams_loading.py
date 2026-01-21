#!/usr/bin/env python3
"""Test DreaMS model loading and unfreezing."""
import sys
import os
sys.path.insert(0, '/cluster/tufts/liulab/yiwan01/SpecBridge/DreaMS')
sys.path.insert(0, '/cluster/tufts/liulab/yiwan01/SpecBridge')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from specbridge.adapters.dreams_adapter import load_dreams_encoder, DreamsAdapter
import torch

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Try to load the model
print("=" * 80)
print("Testing DreaMS model loading")
print("=" * 80)
model = load_dreams_encoder('/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt', d_in=2048, d_out=1024)
print(f'Model type: {type(model)}')
print(f'Model has unfreeze_last method: {hasattr(model, "unfreeze_last")}')

# Check model structure
print("\nModel structure:")
if hasattr(model, 'transformer_encoder'):
    print(f'  Has transformer_encoder: True')
    if hasattr(model.transformer_encoder, 'atts'):
        print(f'  transformer_encoder.atts: ModuleList with {len(model.transformer_encoder.atts)} attention layers')
    if hasattr(model.transformer_encoder, 'ffs'):
        print(f'  transformer_encoder.ffs: ModuleList with {len(model.transformer_encoder.ffs)} feedforward layers')
    
    # List all children
    print(f'\n  All children of transformer_encoder:')
    for i, child in enumerate(model.transformer_encoder.children()):
        print(f'    {i}: {type(child).__name__}')

# Count parameters before freezing
total, trainable = count_params(model)
print(f'\nBefore freezing: Total={total:,}, Trainable={trainable:,}')

# Create adapter and freeze
print("\n" + "=" * 80)
print("Testing DreamsAdapter with freezing")
print("=" * 80)
adapter = DreamsAdapter(model, d_out=2048, freeze_backbone=True)
total_after_freeze, trainable_after_freeze = count_params(model)
print(f'After freezing: Total={total_after_freeze:,}, Trainable={trainable_after_freeze:,}')

# Test unfreezing
print("\n" + "=" * 80)
print("Testing unfreeze_last(2)")
print("=" * 80)
adapter.unfreeze_last(n_layers=2)
total_after_unfreeze, trainable_after_unfreeze = count_params(model)
print(f'After unfreezing 2 layers: Total={total_after_unfreeze:,}, Trainable={trainable_after_unfreeze:,}')
print(f'Additional trainable params: {trainable_after_unfreeze - trainable_after_freeze:,}')

if trainable_after_unfreeze == total_after_unfreeze:
    print("\n[ERROR] All parameters are trainable! This is wrong - should only unfreeze 2 layers.")
elif trainable_after_unfreeze == trainable_after_freeze:
    print("\n[ERROR] No parameters were unfrozen! The unfreezing logic failed.")
else:
    print(f"\n[SUCCESS] Unfreezing worked: {trainable_after_unfreeze - trainable_after_freeze:,} params unfrozen")
