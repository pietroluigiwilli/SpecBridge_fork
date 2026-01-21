# SpecBridge Ablation Study

This document describes the ablation study configurations for SpecBridge, examining the impact of encoder initialization and loss functions on model performance.

## Ablation Study Configurations

### Overview

The ablation study consists of 5 configurations that systematically vary:
1. **Encoder initialization**: Random initialization vs. Pre-trained
2. **Encoder training**: Trainable vs. Frozen
3. **Loss function**: Alignment (MSE) vs. Contrastive (InfoNCE)

### Configuration Details

#### Configuration 1: Random Init + Alignment Loss
- **Spec Encoder**: Random initialization (from scratch), trainable
- **Mol Encoder**: Random initialization (from scratch), trainable
- **Loss**: Alignment loss (MSE between mapped spectrum and molecule embeddings)
- **Flags**: `--init-spec-from-scratch --init-mol-from-scratch --w-map 5.0 --w-con-mapped 0`

#### Configuration 2: Random Init + Contrastive Loss
- **Spec Encoder**: Random initialization (from scratch), trainable
- **Mol Encoder**: Random initialization (from scratch), trainable
- **Loss**: Contrastive loss (InfoNCE between mapped spectrum and molecule embeddings)
- **Flags**: `--init-spec-from-scratch --init-mol-from-scratch --w-map 0 --w-con-mapped 1.0`

#### Configuration 3: Pre-trained + Contrastive Loss
- **Spec Encoder**: Pre-trained (DreaMS checkpoint), frozen
- **Mol Encoder**: Pre-trained (ChemBERTa), frozen
- **Loss**: Contrastive loss (InfoNCE)
- **Flags**: `--w-map 0 --w-con-mapped 1.0` (default: pretrained and frozen)

#### Configuration 4: Freeze + Alignment Loss
- **Spec Encoder**: Pre-trained (DreaMS checkpoint), frozen
- **Mol Encoder**: Pre-trained (ChemBERTa), frozen
- **Loss**: Alignment loss (MSE)
- **Flags**: `--w-map 5.0 --w-con-mapped 0 --unfreeze-last 0 --unfreeze-mol-last 0`

#### Configuration 5: Freeze + Contrastive Loss
- **Spec Encoder**: Pre-trained (DreaMS checkpoint), frozen
- **Mol Encoder**: Pre-trained (ChemBERTa), frozen
- **Loss**: Contrastive loss (InfoNCE)
- **Flags**: `--w-map 0 --w-con-mapped 1.0 --unfreeze-last 0 --unfreeze-mol-last 0`

## Loss Functions

### Alignment Loss (MSE)
- **Formula**: `L_map = MSE(mu_s, z_m)`
- **Description**: Direct mean squared error between the mapped spectrum embedding (`mu_s`) and the molecule embedding (`z_m`)
- **Weight**: Controlled by `--w-map` (default: 5.0)

### Contrastive Loss (InfoNCE)
- **Formula**: InfoNCE loss between `mu_s` and `z_m` with in-batch negatives
- **Description**: Contrastive learning objective that pulls positive pairs together and pushes negatives apart
- **Weight**: Controlled by `--w-con-mapped` (default: 1.0)

## Running the Ablation Study

### Option 1: Run All Configurations (Bash Script)

```bash
# Set environment variables (optional)
export MGF_FILE=/path/to/your/data.mgf
export DREAMS_CKPT=/path/to/ssl_model.ckpt
export CHEMBERTA_MODEL=Derify/ChemBERTa_augmented_pubchem_13m
export BASE_OUTDIR=runs/ablation_study

# Run all configurations
./run_ablation_study.sh
```

### Option 2: Run Individual Configuration

```bash
# Configuration 1: Random init + Alignment
python dreams_condition_adapter.py \
    --mgf /path/to/data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --init-spec-from-scratch \
    --init-mol-from-scratch \
    --w-map 5.0 \
    --w-con-mapped 0 \
    --outdir runs/ablation_study/1_random_align \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m

# Configuration 2: Random init + Contrastive
python dreams_condition_adapter.py \
    --mgf /path/to/data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --init-spec-from-scratch \
    --init-mol-from-scratch \
    --w-map 0 \
    --w-con-mapped 1.0 \
    --outdir runs/ablation_study/2_random_contrastive \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m

# Configuration 3: Pre-trained + Contrastive
python dreams_condition_adapter.py \
    --mgf /path/to/data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --w-map 0 \
    --w-con-mapped 1.0 \
    --outdir runs/ablation_study/3_pretrained_contrastive \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m

# Configuration 4: Freeze + Alignment
python dreams_condition_adapter.py \
    --mgf /path/to/data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --w-map 5.0 \
    --w-con-mapped 0 \
    --unfreeze-last 0 \
    --unfreeze-mol-last 0 \
    --outdir runs/ablation_study/4_freeze_align \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m

# Configuration 5: Freeze + Contrastive
python dreams_condition_adapter.py \
    --mgf /path/to/data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --w-map 0 \
    --w-con-mapped 1.0 \
    --unfreeze-last 0 \
    --unfreeze-mol-last 0 \
    --outdir runs/ablation_study/5_freeze_contrastive \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m
```

## Expected Outputs

Each configuration will create:
- Training checkpoints in `{outdir}/ckpt_*.pt`
- Best validation checkpoint: `{outdir}/best_val.pt`
- Final checkpoint: `{outdir}/last.pt`
- Wandb logs (if enabled)

## Evaluation

After training, evaluate each configuration using:

```bash
python -m specbridge.eval.candidates \
    --mgf /path/to/test_data.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --adapter-ckpt runs/ablation_study/{config_name}/best_val.pt \
    --candidates /path/to/candidates.pkl \
    --fold-query test \
    --use-mapped \
    --deterministic-map \
    --no-gaussian \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m
```

## Key Implementation Details

### Encoder Initialization

- **Random Init**: When `--init-spec-from-scratch` or `--init-mol-from-scratch` is used, encoders are initialized with random weights (no pretrained weights loaded)
- **Pre-trained**: Default behavior loads pretrained weights:
  - Spec encoder: DreaMS checkpoint (`--dreams-ckpt`)
  - Mol encoder: ChemBERTa model (`--chemberta-model`)

### Encoder Freezing

- **Frozen**: Encoders are frozen by default when using pretrained weights
- **Trainable**: Encoders are trainable when:
  - Randomly initialized (`--init-*-from-scratch`)
  - Explicitly unfrozen (`--unfreeze-last`, `--unfreeze-mol-last`)

### Loss Configuration

- **Alignment only**: `--w-map > 0`, `--w-con-mapped 0`
- **Contrastive only**: `--w-map 0`, `--w-con-mapped > 0`
- **Both**: Both weights > 0 (not used in this ablation study)

## Notes

- All configurations use the same hyperparameters (batch size, learning rate, etc.) for fair comparison
- The mapper network is always trainable
- Orthogonality penalty is applied in all configurations (`--w-ortho 1e-3`)


