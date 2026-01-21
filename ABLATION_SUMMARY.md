# SpecBridge Ablation Study - Quick Reference

## Overview

This ablation study examines 5 configurations to understand the impact of:
- **Encoder initialization**: Random vs. Pre-trained
- **Encoder training**: Trainable vs. Frozen  
- **Loss function**: Alignment (MSE) vs. Contrastive (InfoNCE)

## Configuration Matrix

| Config | Spec Encoder | Mol Encoder | Loss Type | Key Flags |
|--------|-------------|-------------|-----------|-----------|
| 1 | Random init, trainable | Random init, trainable | Alignment (MSE) | `--init-spec-from-scratch --init-mol-from-scratch --w-map 5.0 --w-con-mapped 0` |
| 2 | Random init, trainable | Random init, trainable | Contrastive (InfoNCE) | `--init-spec-from-scratch --init-mol-from-scratch --w-map 0 --w-con-mapped 1.0` |
| 3 | Pre-trained, frozen | Pre-trained, frozen | Contrastive (InfoNCE) | `--w-map 0 --w-con-mapped 1.0` |
| 4 | Pre-trained, frozen | Pre-trained, frozen | Alignment (MSE) | `--w-map 5.0 --w-con-mapped 0 --unfreeze-last 0 --unfreeze-mol-last 0` |
| 5 | Pre-trained, frozen | Pre-trained, frozen | Contrastive (InfoNCE) | `--w-map 0 --w-con-mapped 1.0 --unfreeze-last 0 --unfreeze-mol-last 0` |

## Quick Start

**Note:** Scripts are configured to use the SpectraVerse dataset by default.

### Run All Configurations

```bash
# Using bash script (uses SpectraVerse by default)
./run_ablation_study.sh

# Using Python script (uses SpectraVerse by default)
python run_ablation_config.py

# Or with custom paths
python run_ablation_config.py \
    --mgf /path/to/data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt
```

### Run Single Configuration

```bash
# Uses SpectraVerse by default
python run_ablation_config.py --config 1_random_align

# Or with custom paths
python run_ablation_config.py \
    --mgf /path/to/data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --config 1_random_align
```

### List Available Configurations

```bash
python run_ablation_config.py --list-configs
```

## Implementation Details

### Encoder Freezing Logic

**Spec Encoder (DreaMS):**
- **Frozen by default** when loading from checkpoint (`--dreams-ckpt`)
- **Trainable** when using `--init-spec-from-scratch`
- Can be partially unfrozen with `--unfreeze-last N`

**Mol Encoder (ChemBERTa):**
- **Frozen by default** when loading pretrained model
- **Trainable** when using `--init-mol-from-scratch`
- Can be partially unfrozen with `--unfreeze-mol-last N`

### Loss Functions

**Alignment Loss (MSE):**
- `L_map = MSE(mu_s, z_m)`
- Direct regression from spectrum embedding to molecule embedding
- Controlled by `--w-map` (default: 5.0)

**Contrastive Loss (InfoNCE):**
- InfoNCE loss with in-batch negatives
- Pulls positive pairs together, pushes negatives apart
- Controlled by `--w-con-mapped` (default: 1.0)

## Expected Results Location

All results will be saved in:
```
runs/ablation_study_spectraverse/
├── 1_random_align/
├── 2_random_contrastive/
├── 3_pretrained_contrastive/
├── 4_freeze_align/
└── 5_freeze_contrastive/
```

Each directory contains:
- `ckpt_*.pt`: Periodic checkpoints
- `best_val.pt`: Best validation checkpoint
- `last.pt`: Final checkpoint

## Evaluation

After training, evaluate each configuration:

```bash
python -m specbridge.eval.candidates \
    --mgf /path/to/test.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --adapter-ckpt runs/ablation_study_spectraverse/{config_name}/best_val.pt \
    --candidates /path/to/candidates.pkl \
    --fold-query test \
    --use-mapped \
    --deterministic-map \
    --no-gaussian \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m
```

## Notes

- All configurations use identical hyperparameters for fair comparison
- The mapper network is always trainable
- Orthogonality penalty (`--w-ortho 1e-3`) is applied in all configurations
- Configurations 4 and 5 explicitly set `--unfreeze-last 0` and `--unfreeze-mol-last 0` to ensure encoders remain frozen
- **Sampling strategy**: 
  - **For contrastive learning (configs 2, 3, 5)**: Uses `UniqueSMILESBatchSampler` which:
    - Ensures no duplicate SMILES in each batch (prevents same-SMILES samples from being treated as negatives in InfoNCE)
    - Includes ALL data points (even if same SMILES - they have different spectra)
    - Makes multiple passes to cycle through all samples
  - **For alignment loss (configs 1, 4)**: Uses standard random sampling (no sampler)
  - The `ReplicateBatchSampler` was not used because it filters out samples with fewer than K replicates, causing significant data loss.

