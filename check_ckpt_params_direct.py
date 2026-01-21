#!/usr/bin/env python3
"""
Check parameter count directly from DreaMS checkpoint file by reading it as a zip.
"""
import zipfile
import pickle
import io
import torch
import sys

def format_params(num):
    """Format parameter count as K, M, etc."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f} K"
    else:
        return str(num)

def count_ckpt_params(ckpt_path):
    """Count parameters in checkpoint file by reading it as zip."""
    print(f"Reading checkpoint as zip file: {ckpt_path}")
    
    try:
        with zipfile.ZipFile(ckpt_path, 'r') as zip_ref:
            # List all files in the zip
            files = zip_ref.namelist()
            print(f"Files in checkpoint: {files}")
            
            # Find the data/pickle file
            data_file = None
            for f in files:
                if 'data' in f.lower() or f.endswith('.pkl'):
                    data_file = f
                    break
            
            if data_file is None:
                # Try 'data' or 'data.pkl'
                for candidate in ['data', 'data.pkl', 'pytorch_model.bin']:
                    if candidate in files:
                        data_file = candidate
                        break
            
            if data_file is None:
                print(f"Could not find data file. Available files: {files}")
                return None, None
            
            print(f"Reading data from: {data_file}")
            data = zip_ref.read(data_file)
            
            # Try to unpickle with a custom unpickler that handles missing classes
            class SafeUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    # Allow torch types
                    if module.startswith('torch'):
                        return super().find_class(module, name)
                    # Allow collections
                    if module == 'collections' and name in ['OrderedDict', 'defaultdict']:
                        return super().find_class(module, name)
                    # Allow builtins
                    if module == 'builtins':
                        return super().find_class(module, name)
                    # For unknown classes, return a dict-like object
                    if name in ['DreaMS', 'PreTrainedModel', 'LightningModule']:
                        return type('Dummy', (), {})
                    # Return a generic dict for everything else
                    return dict
            
            # PyTorch checkpoints use persistent storage - need to use torch.load
            # But we can't use torch.load directly on the bytes, need to reconstruct the zip structure
            # Instead, let's use torch.load on the file directly with proper error handling
            print("Attempting to load using torch.load with CPU map_location...")
            try:
                # Set environment to avoid CUDA
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                # Use torch.load with weights_only=False but catch module errors
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"torch.load failed: {e}")
                # Try with a custom unpickler that handles persistent storage
                try:
                    class PersistentUnpickler(pickle.Unpickler):
                        def persistent_load(self, pid):
                            # For persistent storage, return a dummy tensor
                            # We'll count parameters from the actual tensors in the state_dict
                            return torch.tensor([])
                    
                    unpickler = PersistentUnpickler(io.BytesIO(data))
                    ckpt = unpickler.load()
                except Exception as e2:
                    print(f"Persistent unpickling also failed: {e2}")
                    return None, None
            
            print(f"Checkpoint type: {type(ckpt)}")
            
            # Extract state_dict
            state_dict = None
            if isinstance(ckpt, dict):
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                else:
                    # Assume the dict itself is the state_dict
                    state_dict = ckpt
            else:
                state_dict = ckpt
            
            if state_dict is None:
                print("Could not extract state_dict from checkpoint")
                return None, None
            
            # Count parameters
            total_params = 0
            param_info = []
            
            for key, value in state_dict.items():
                if isinstance(value, torch.Tensor):
                    num_params = value.numel()
                    total_params += num_params
                    param_info.append((key, num_params, tuple(value.shape)))
            
            # Sort by parameter count (descending)
            param_info.sort(key=lambda x: x[1], reverse=True)
            
            print("\n" + "=" * 80)
            print(f"Total parameters in checkpoint: {total_params:,} ({format_params(total_params)})")
            print("=" * 80)
            
            print("\nTop 30 largest parameter tensors:")
            print("-" * 80)
            print(f"{'Key':<60} {'Params':>15} {'Shape':<20}")
            print("-" * 80)
            
            for key, num_params, shape in param_info[:30]:
                print(f"{key:<60} {num_params:>15,} {str(shape):<20}")
            
            if len(param_info) > 30:
                print(f"\n... and {len(param_info) - 30} more tensors")
            
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
            print(f"{'Component':<40} {'Params':>20} {'Percentage':>15}")
            print("-" * 75)
            for prefix, count in component_counts:
                pct = (count / total_params) * 100 if total_params > 0 else 0
                print(f"{prefix:<40} {count:>20,} ({format_params(count):>10}) {pct:>6.2f}%")
            
            return total_params, param_info
            
    except Exception as e:
        print(f"Error reading checkpoint: {e}")
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
