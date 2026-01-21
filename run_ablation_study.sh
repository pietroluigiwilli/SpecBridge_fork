#!/bin/bash
# Ablation Study Script for SpecBridge
# 
# This script runs 5 ablation study configurations:
# 1. Random init two encoders, alignment loss
# 2. Random init two encoders, contrastive loss
# 3. Pre-trained two encoders, contrastive loss
# 4. Freeze two encoders, alignment loss
# 5. Freeze two encoders, contrastive loss

set -e

# Base configuration (modify these paths as needed)
MGF_FILE="${MGF_FILE:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf}"
DREAMS_CKPT="${DREAMS_CKPT:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt}"
CHEMBERTA_MODEL="${CHEMBERTA_MODEL:-Derify/ChemBERTa_augmented_pubchem_13m}"
BASE_OUTDIR="${BASE_OUTDIR:-runs/ablation_study_spectraverse}"

# Common training hyperparameters
BATCH_SIZE=32
EPOCHS=2
LR=1e-4
COND_DIM=2048
MAPPER_HIDDEN=2048
SPEC_BINS=2048
FP_BITS=2048
N_BLOCKS=8
LOG_EVERY=50
SAVE_EVERY=200

# Loss weights
W_ORTHO=1e-3
SUPCON_K=4  # Number of replicates per SMILES per batch (for contrastive learning)

echo "=========================================="
echo "SpecBridge Ablation Study - SpectraVerse Dataset"
echo "=========================================="
echo "MGF File: $MGF_FILE"
echo "DreaMS Checkpoint: $DREAMS_CKPT"
echo "ChemBERTa Model: $CHEMBERTA_MODEL"
echo "Base Output Directory: $BASE_OUTDIR"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Function to run a single ablation configuration
run_ablation() {
    local config_name=$1
    local init_spec=$2
    local init_mol=$3
    local w_map=$4
    local w_con_mapped=$5
    local outdir="${BASE_OUTDIR}/${config_name}"
    
    echo "----------------------------------------"
    echo "Running: $config_name"
    echo "  Spec encoder: $([ "$init_spec" = "true" ] && echo "Random init" || echo "Pre-trained")"
    echo "  Mol encoder: $([ "$init_mol" = "true" ] && echo "Random init" || echo "Pre-trained")"
    echo "  Loss: $([ "$w_map" != "0" ] && echo "Alignment (MSE)" || echo "Contrastive (InfoNCE)")"
    echo "  Output: $outdir"
    echo "----------------------------------------"
    
    # Build command
    local cmd="python dreams_condition_adapter.py"
    cmd="$cmd --mgf $MGF_FILE"
    cmd="$cmd --dreams-ckpt $DREAMS_CKPT"
    cmd="$cmd --fold train"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --cond-dim $COND_DIM"
    cmd="$cmd --mapper-hidden $MAPPER_HIDDEN"
    cmd="$cmd --spec-bins $SPEC_BINS"
    cmd="$cmd --fp-bits $FP_BITS"
    cmd="$cmd --n-blocks $N_BLOCKS"
    cmd="$cmd --no-gaussian"
    cmd="$cmd --w-map $w_map"
    cmd="$cmd --w-con-mapped $w_con_mapped"
    cmd="$cmd --w-ortho $W_ORTHO"
    
    # Note: We don't use --supcon-k anymore to avoid data loss from ReplicateBatchSampler
    # Standard random sampling is used for all configurations for fair comparison
    cmd="$cmd --log-every $LOG_EVERY"
    cmd="$cmd --save-every $SAVE_EVERY"
    cmd="$cmd --outdir $outdir"
    cmd="$cmd --mol-space chemberta"
    cmd="$cmd --chemberta-model $CHEMBERTA_MODEL"
    cmd="$cmd --lr $LR"
    
    # Ablation-specific flags
    if [ "$init_spec" = "true" ]; then
        cmd="$cmd --init-spec-from-scratch"
    fi
    
    if [ "$init_mol" = "true" ]; then
        cmd="$cmd --init-mol-from-scratch"
    fi
    
    # For frozen encoders (configs 4 and 5), ensure no unfreezing
    if [[ "$config_name" == "4_freeze_align" || "$config_name" == "5_freeze_contrastive" ]]; then
        cmd="$cmd --unfreeze-last 0"
        cmd="$cmd --unfreeze-mol-last 0"
    fi
    
    echo "Command: $cmd"
    echo ""
    
    # Run the command
    eval $cmd
    
    echo ""
    echo "Completed: $config_name"
    echo ""
}

# Configuration 1: Random init two encoders, alignment loss
run_ablation "1_random_align" "true" "true" "5.0" "0"

# Configuration 2: Random init two encoders, contrastive loss
run_ablation "2_random_contrastive" "true" "true" "0" "1.0"

# Configuration 3: Pre-trained two encoders, contrastive loss
run_ablation "3_pretrained_contrastive" "false" "false" "0" "1.0"

# Configuration 4: Freeze two encoders, alignment loss
run_ablation "4_freeze_align" "false" "false" "5.0" "0"

# Configuration 5: Freeze two encoders, contrastive loss
run_ablation "5_freeze_contrastive" "false" "false" "0" "1.0"

echo "=========================================="
echo "All ablation study configurations completed!"
echo "Results saved in: $BASE_OUTDIR"
echo "=========================================="

