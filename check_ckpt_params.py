#!/usr/bin/env python3
"""
Check parameter count directly from DreaMS checkpoint file.
Uses the existing load_dreams_encoder function to properly load the model.
"""
import torch
import sys
from specbridge.adapters.dreams_adapter import load_dreams_encoder

def format_params(num):
    """Format parameter count as K, M, etc."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f} K"
    else:
        return str(num)

def count_model_params(model):
    """Count parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_ckpt_params(ckpt_path):
    """Count parameters in checkpoint file by loading the model."""
    print(f"Loading checkpoint: {ckpt_path}")
    
    try:
        # Try to use PreTrainedModel.from_ckpt directly
        import os
        import sys
        sys.path.insert(0, '/cluster/tufts/liulab/yiwan01/SpecBridge/DreaMS')
        
        try:
            from dreams.api import PreTrainedModel
            from dreams.models.dreams.dreams import DreaMS as DreaMSModel
            
            print("Attempting to load using PreTrainedModel.from_ckpt...")
            ptm = PreTrainedModel.from_ckpt(
                ckpt_path=ckpt_path,
                ckpt_cls=DreaMSModel,
                n_highest_peaks=60,
            )
            model = ptm.model
            print(f"Successfully loaded model using PreTrainedModel: {type(model)}")
        except Exception as e1:
            print(f"PreTrainedModel.from_ckpt failed: {e1}")
            print("Falling back to load_dreams_encoder...")
            # Use the existing load_dreams_encoder function which handles dependencies
            model = load_dreams_encoder(ckpt_path, d_in=2048, d_out=1024, init_from_scratch=False)
            print(f"Model type: {type(model)}")
            
            # Check if it's a dummy model
            if isinstance(model, type(load_dreams_encoder(None, d_in=2048, d_out=1024))):
                from specbridge.adapters.dreams_adapter import DummyDreams
                if isinstance(model, DummyDreams):
                    print("\n[WARNING] Loaded DummyDreams instead of real model!")
                    print("[WARNING] This means the checkpoint could not be loaded properly.")
                    print("[WARNING] The parameter count below is for DummyDreams, not the actual checkpoint.")
                    print()
        
        # Count parameters
        total_params, trainable_params = count_model_params(model)
        
        print("\n" + "=" * 80)
        print(f"Total parameters: {total_params:,} ({format_params(total_params)})")
        print(f"Trainable parameters: {trainable_params:,} ({format_params(trainable_params)})")
        print(f"Frozen parameters: {total_params - trainable_params:,} ({format_params(total_params - trainable_params)})")
        print("=" * 80)
        
        # Get detailed breakdown by module
        print("\nParameter breakdown by module:")
        print("-" * 80)
        print(f"{'Module':<50} {'Params':>15} {'Trainable':>15}")
        print("-" * 80)
        
        module_counts = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if total > 0:
                    module_counts.append((name, total, trainable))
        
        # Sort by parameter count
        module_counts.sort(key=lambda x: x[1], reverse=True)
        
        for name, total, trainable in module_counts[:30]:  # Top 30 modules
            print(f"{name:<50} {total:>15,} {trainable:>15,}")
        
        if len(module_counts) > 30:
            print(f"\n... and {len(module_counts) - 30} more modules")
        
        # Group by prefix
        print("\n" + "=" * 80)
        print("Parameter breakdown by component (prefix):")
        print("=" * 80)
        
        component_counts = {}
        for name, total, trainable in module_counts:
            prefix = name.split('.')[0] if '.' in name else name
            if prefix not in component_counts:
                component_counts[prefix] = {'total': 0, 'trainable': 0}
            component_counts[prefix]['total'] += total
            component_counts[prefix]['trainable'] += trainable
        
        component_counts = sorted(component_counts.items(), key=lambda x: x[1]['total'], reverse=True)
        print(f"{'Component':<30} {'Total':>15} {'Trainable':>15} {'Frozen':>15}")
        print("-" * 75)
        for prefix, counts in component_counts:
            frozen = counts['total'] - counts['trainable']
            print(f"{prefix:<30} {counts['total']:>15,} {counts['trainable']:>15,} {frozen:>15,}")
        
        return total_params, trainable_params
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
        # Handle different checkpoint formats
        if isinstance(ckpt, dict):
            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
                print("Found 'state_dict' key in checkpoint")
            elif 'model' in ckpt:
                state_dict = ckpt['model']
                print("Found 'model' key in checkpoint")
            else:
                state_dict = ckpt
                print("Using checkpoint dict directly as state_dict")
        else:
            state_dict = ckpt
            print("Checkpoint is state_dict directly")
        
        # Count parameters
        total_params = 0
        param_info = []
        
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                num_params = value.numel()
                total_params += num_params
                param_info.append((key, num_params, value.shape))
        
        # Sort by parameter count (descending)
        param_info.sort(key=lambda x: x[1], reverse=True)
        
        print("\n" + "=" * 80)
        print(f"Total parameters in checkpoint: {total_params:,} ({format_params(total_params)})")
        print("=" * 80)
        
        print("\nTop 20 largest parameter tensors:")
        print("-" * 80)
        print(f"{'Key':<50} {'Params':>15} {'Shape':<20}")
        print("-" * 80)
        
        for key, num_params, shape in param_info[:20]:
            print(f"{key:<50} {num_params:>15,} {str(shape):<20}")
        
        if len(param_info) > 20:
            print(f"\n... and {len(param_info) - 20} more tensors")
        
        # Group by prefix to see component breakdown
        print("\n" + "=" * 80)
        print("Parameter breakdown by component (prefix):")
        print("=" * 80)
        
        component_counts = {}
        for key, num_params, _ in param_info:
            # Get prefix (first part before first dot)
            prefix = key.split('.')[0] if '.' in key else key
            if prefix not in component_counts:
                component_counts[prefix] = 0
            component_counts[prefix] += num_params
        
        component_counts = sorted(component_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"{'Component':<30} {'Params':>15} {'Percentage':>15}")
        print("-" * 60)
        for prefix, count in component_counts:
            pct = (count / total_params) * 100 if total_params > 0 else 0
            print(f"{prefix:<30} {count:>15,} ({format_params(count):>10}) {pct:>6.2f}%")
        
        return total_params, param_info
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    ckpt_path = "/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
    
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    
    total, info = count_ckpt_params(ckpt_path)
    
    if total is not None:
        print(f"\n{'='*80}")
        print(f"SUMMARY: {total:,} total parameters ({format_params(total)})")
        print(f"{'='*80}")
