#!/usr/bin/env python3
"""Verify which layers are actually trainable after unfreezing last 2."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'DreaMS'))

import torch
import torch.nn as nn
from specbridge.adapters.dreams_adapter import load_dreams_encoder, DreamsAdapter

def count_by_layer(model, prefix=""):
    """Count parameters by layer name."""
    layer_counts = {}
    for name, param in model.named_parameters():
        if prefix and not name.startswith(prefix):
            continue
        layer_name = name.split('.')[0] if '.' in name else name
        if layer_name not in layer_counts:
            layer_counts[layer_name] = {'total': 0, 'trainable': 0}
        layer_counts[layer_name]['total'] += param.numel()
        if param.requires_grad:
            layer_counts[layer_name]['trainable'] += param.numel()
    return layer_counts

# Load DreaMS
dreams_ckpt = "/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
dreams_encoder = load_dreams_encoder(dreams_ckpt)

print("=" * 80)
print("DreaMS Model Structure")
print("=" * 80)
print(f"Model type: {type(dreams_encoder)}")
print(f"Has transformer_encoder: {hasattr(dreams_encoder, 'transformer_encoder')}")

if hasattr(dreams_encoder, 'transformer_encoder'):
    encoder = dreams_encoder.transformer_encoder
    print(f"Encoder type: {type(encoder)}")
    
    # Check if it has atts and ffs (the actual layer components)
    if hasattr(encoder, 'atts') and hasattr(encoder, 'ffs'):
        n_layers = len(encoder.atts)
        print(f"Number of transformer layers: {n_layers}")
        
        # Count parameters before unfreezing
        print("\nBefore unfreezing last 2:")
        total_before = sum(p.numel() for p in dreams_encoder.parameters())
        trainable_before = sum(p.numel() for p in dreams_encoder.parameters() if p.requires_grad)
        print(f"Total: {total_before:,}, Trainable: {trainable_before:,}")
        
        # Count parameters in each layer before unfreezing
        print("\nParameter counts by transformer layer (before unfreezing):")
        layer_totals = []
        for i in range(n_layers):
            att_total = sum(p.numel() for p in encoder.atts[i].parameters())
            ff_total = sum(p.numel() for p in encoder.ffs[i].parameters())
            # Also count the corresponding scale/norm layers
            scale_idx_att = 2 * i
            scale_idx_ff = 2 * i + 1
            scale_att_total = sum(p.numel() for p in encoder.scales[scale_idx_att].parameters()) if scale_idx_att < len(encoder.scales) else 0
            scale_ff_total = sum(p.numel() for p in encoder.scales[scale_idx_ff].parameters()) if scale_idx_ff < len(encoder.scales) else 0
            layer_total = att_total + ff_total + scale_att_total + scale_ff_total
            layer_totals.append(layer_total)
            print(f"  Layer {i}: {layer_total:,} params (att: {att_total:,}, ff: {ff_total:,}, scales: {scale_att_total + scale_ff_total:,})")
        
        # Unfreeze last 2 layers
        print("\nUnfreezing last 2 layers (layers {} and {})...".format(n_layers-2, n_layers-1))
        for i in [n_layers-2, n_layers-1]:
            for param in encoder.atts[i].parameters():
                param.requires_grad = True
            for param in encoder.ffs[i].parameters():
                param.requires_grad = True
            # Also unfreeze the corresponding scale layers
            scale_idx_att = 2 * i
            scale_idx_ff = 2 * i + 1
            if scale_idx_att < len(encoder.scales):
                for param in encoder.scales[scale_idx_att].parameters():
                    param.requires_grad = True
            if scale_idx_ff < len(encoder.scales):
                for param in encoder.scales[scale_idx_ff].parameters():
                    param.requires_grad = True
        
        # Count parameters after unfreezing
        print("\nAfter unfreezing last 2:")
        total_after = sum(p.numel() for p in dreams_encoder.parameters())
        trainable_after = sum(p.numel() for p in dreams_encoder.parameters() if p.requires_grad)
        print(f"Total: {total_after:,}, Trainable: {trainable_after:,}")
        
        # Count last 2 layers specifically
        last_2_total = layer_totals[-2] + layer_totals[-1]
        last_2_trainable = 0
        for i in [n_layers-2, n_layers-1]:
            last_2_trainable += sum(p.numel() for p in encoder.atts[i].parameters() if p.requires_grad)
            last_2_trainable += sum(p.numel() for p in encoder.ffs[i].parameters() if p.requires_grad)
            scale_idx_att = 2 * i
            scale_idx_ff = 2 * i + 1
            if scale_idx_att < len(encoder.scales):
                last_2_trainable += sum(p.numel() for p in encoder.scales[scale_idx_att].parameters() if p.requires_grad)
            if scale_idx_ff < len(encoder.scales):
                last_2_trainable += sum(p.numel() for p in encoder.scales[scale_idx_ff].parameters() if p.requires_grad)
        print(f"\nLast 2 layers combined: {last_2_total:,} total, {last_2_trainable:,} trainable")
        
        # Count other components (everything except last 2 transformer layers)
        print("\nOther DreaMS components (frozen):")
        other_total = total_before - last_2_total
        other_trainable = trainable_after - last_2_trainable
        print(f"  Total: {other_total:,}, Trainable: {other_trainable:,}")
        
        print(f"\n=== SUMMARY ===")
        print(f"DreaMS total parameters: {total_before:,}")
        print(f"DreaMS trainable (last 2 layers only): {last_2_trainable:,}")
        print(f"DreaMS frozen: {total_before - last_2_trainable:,}")

