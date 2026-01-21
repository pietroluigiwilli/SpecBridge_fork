#!/usr/bin/env python3
"""
Ablation Study Configuration Runner

This script provides a flexible way to run individual ablation study configurations
or all configurations in sequence.
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path

# Configuration definitions
ABLATION_CONFIGS = {
    "1_random_align": {
        "name": "Random Init + Alignment Loss",
        "init_spec_from_scratch": True,
        "init_mol_from_scratch": True,
        "w_map": 5.0,
        "w_con_mapped": 0.0,
        "unfreeze_last": 0,
        "unfreeze_mol_last": 0,
    },
    "2_random_contrastive": {
        "name": "Random Init + Contrastive Loss",
        "init_spec_from_scratch": True,
        "init_mol_from_scratch": True,
        "w_map": 0.0,
        "w_con_mapped": 1.0,
        "unfreeze_last": 0,
        "unfreeze_mol_last": 0,
    },
    "3_pretrained_contrastive": {
        "name": "Pre-trained + Contrastive Loss",
        "init_spec_from_scratch": False,
        "init_mol_from_scratch": False,
        "w_map": 0.0,
        "w_con_mapped": 1.0,
        "unfreeze_last": 0,
        "unfreeze_mol_last": 0,
    },
    "4_freeze_align": {
        "name": "Freeze + Alignment Loss",
        "init_spec_from_scratch": False,
        "init_mol_from_scratch": False,
        "w_map": 5.0,
        "w_con_mapped": 0.0,
        "unfreeze_last": 0,
        "unfreeze_mol_last": 0,
    },
    "5_freeze_contrastive": {
        "name": "Freeze + Contrastive Loss",
        "init_spec_from_scratch": False,
        "init_mol_from_scratch": False,
        "w_map": 0.0,
        "w_con_mapped": 1.0,
        "unfreeze_last": 0,
        "unfreeze_mol_last": 0,
    },
}


def build_command(config_key, config, base_args):
    """Build the training command for a given configuration."""
    cmd = ["python", "dreams_condition_adapter.py"]
    
    # Required arguments
    cmd.extend(["--mgf", base_args.mgf])
    cmd.extend(["--dreams-ckpt", base_args.dreams_ckpt])
    cmd.extend(["--fold", "train"])
    cmd.extend(["--outdir", os.path.join(base_args.outdir, config_key)])
    cmd.extend(["--mol-space", "chemberta"])
    cmd.extend(["--chemberta-model", base_args.chemberta_model])
    
    # Training hyperparameters
    cmd.extend(["--batch-size", str(base_args.batch_size)])
    cmd.extend(["--epochs", str(base_args.epochs)])
    cmd.extend(["--lr", str(base_args.lr)])
    cmd.extend(["--cond-dim", str(base_args.cond_dim)])
    cmd.extend(["--mapper-hidden", str(base_args.mapper_hidden)])
    cmd.extend(["--spec-bins", str(base_args.spec_bins)])
    cmd.extend(["--fp-bits", str(base_args.fp_bits)])
    cmd.extend(["--n-blocks", str(base_args.n_blocks)])
    cmd.extend(["--log-every", str(base_args.log_every)])
    cmd.extend(["--save-every", str(base_args.save_every)])
    
    # Loss weights
    cmd.extend(["--w-map", str(config["w_map"])])
    cmd.extend(["--w-con-mapped", str(config["w_con_mapped"])])
    cmd.extend(["--w-ortho", str(base_args.w_ortho)])
    
    # Note: We don't use --supcon-k anymore to avoid data loss from ReplicateBatchSampler
    # Standard random sampling is used for all configurations for fair comparison
    
    # Ablation-specific flags
    if config["init_spec_from_scratch"]:
        cmd.append("--init-spec-from-scratch")
    if config["init_mol_from_scratch"]:
        cmd.append("--init-mol-from-scratch")
    
    cmd.extend(["--unfreeze-last", str(config["unfreeze_last"])])
    cmd.extend(["--unfreeze-mol-last", str(config["unfreeze_mol_last"])])
    
    # Optional flags
    if base_args.no_gaussian:
        cmd.append("--no-gaussian")
    if base_args.amp:
        cmd.append("--amp")
    if base_args.early_stop:
        cmd.append("--early-stop")
        cmd.extend(["--val-every", str(base_args.val_every)])
        cmd.extend(["--patience", str(base_args.patience)])
        cmd.extend(["--early-metric", base_args.early_metric])
    
    return cmd


def run_config(config_key, config, base_args):
    """Run a single ablation configuration."""
    print("=" * 60)
    print(f"Running: {config['name']}")
    print(f"Config Key: {config_key}")
    print("=" * 60)
    print(f"  Spec encoder: {'Random init' if config['init_spec_from_scratch'] else 'Pre-trained'}")
    print(f"  Mol encoder: {'Random init' if config['init_mol_from_scratch'] else 'Pre-trained'}")
    print(f"  Loss: {'Alignment (MSE)' if config['w_map'] > 0 else 'Contrastive (InfoNCE)'}")
    print(f"  Output: {os.path.join(base_args.outdir, config_key)}")
    print("=" * 60)
    
    cmd = build_command(config_key, config, base_args)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Completed: {config_key}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {config_key} (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run SpecBridge ablation study configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all configurations (uses defaults: spectraverse dataset)
  python run_ablation_config.py

  # Run all configurations with custom paths
  python run_ablation_config.py --mgf data.mgf --dreams-ckpt ssl_model.ckpt

  # Run a specific configuration
  python run_ablation_config.py --config 1_random_align

  # List available configurations
  python run_ablation_config.py --list-configs
        """
    )
    
    # Required arguments
    parser.add_argument("--mgf", type=str, 
                       default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf",
                       help="Path to MGF file (default: spectraverse_clean.mgf)")
    parser.add_argument("--dreams-ckpt", type=str, 
                       default="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt",
                       help="Path to DreaMS checkpoint")
    parser.add_argument("--chemberta-model", type=str, 
                       default="Derify/ChemBERTa_augmented_pubchem_13m",
                       help="ChemBERTa model name")
    parser.add_argument("--outdir", type=str, default="runs/ablation_study_spectraverse",
                       help="Base output directory")
    
    # Configuration selection
    parser.add_argument("--config", type=str, choices=list(ABLATION_CONFIGS.keys()) + ["all"],
                       default="all", help="Which configuration to run (default: all)")
    parser.add_argument("--list-configs", action="store_true",
                       help="List available configurations and exit")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cond-dim", type=int, default=2048)
    parser.add_argument("--mapper-hidden", type=int, default=2048)
    parser.add_argument("--spec-bins", type=int, default=2048)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--w-ortho", type=float, default=1e-3)
    parser.add_argument("--supcon-k", type=int, default=4,
                       help="Number of replicates per SMILES per batch (for contrastive learning sampler)")
    
    # Optional flags
    parser.add_argument("--no-gaussian", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-metric", type=str, default="val_acc",
                       choices=["val_total", "val_L_con_m", "val_L_con", "val_sup", "val_acc"])
    
    args = parser.parse_args()
    
    # List configurations if requested
    if args.list_configs:
        print("Available ablation study configurations:")
        print()
        for key, config in ABLATION_CONFIGS.items():
            print(f"  {key}: {config['name']}")
            print(f"    - Spec: {'Random init' if config['init_spec_from_scratch'] else 'Pre-trained'}")
            print(f"    - Mol: {'Random init' if config['init_mol_from_scratch'] else 'Pre-trained'}")
            print(f"    - Loss: {'Alignment' if config['w_map'] > 0 else 'Contrastive'}")
        sys.exit(0)
    
    # Validate required arguments (now have defaults, but check if files exist)
    if not args.mgf:
        parser.error("--mgf is required (unless --list-configs)")
    if not args.dreams_ckpt:
        parser.error("--dreams-ckpt is required (unless --list-configs)")
    
    # Check if files exist
    if not os.path.isfile(args.mgf):
        print(f"Error: MGF file not found: {args.mgf}")
        sys.exit(1)
    if not os.path.isfile(args.dreams_ckpt):
        print(f"Error: DreaMS checkpoint not found: {args.dreams_ckpt}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Determine which configurations to run
    if args.config == "all":
        configs_to_run = list(ABLATION_CONFIGS.items())
    else:
        configs_to_run = [(args.config, ABLATION_CONFIGS[args.config])]
    
    # Run configurations
    results = {}
    for config_key, config in configs_to_run:
        success = run_config(config_key, config, args)
        results[config_key] = success
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for config_key, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {config_key}: {status}")
    print("=" * 60)
    
    # Exit with error if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

