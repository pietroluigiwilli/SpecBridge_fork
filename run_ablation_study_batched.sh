#!/bin/bash
# Ablation Study Script for SpecBridge - Batched Execution
# 
# Runs configurations in explicit batches to manage GPU usage:
# Option 1: 3 jobs, then 2 jobs (default)
# Option 2: 2 jobs, then 2 jobs, then 1 job
# Option 3: 4 jobs, then 1 job

set -e

# Base configuration
MGF_FILE="${MGF_FILE:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf}"
DREAMS_CKPT="${DREAMS_CKPT:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt}"
CHEMBERTA_MODEL="${CHEMBERTA_MODEL:-Derify/ChemBERTa_augmented_pubchem_13m}"
BASE_OUTDIR="${BASE_OUTDIR:-runs/ablation_study_spectraverse}"

# GPU configuration
NUM_GPUS="${NUM_GPUS:-4}"  # Number of GPUs available
BATCH_MODE="${BATCH_MODE:-3+2}"  # Batch mode: "3+2", "2+2+1", or "4+1"

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
W_ORTHO=1e-3
SUPCON_K=4

echo "=========================================="
echo "SpecBridge Ablation Study - Batched Execution"
echo "=========================================="
echo "MGF File: $MGF_FILE"
echo "DreaMS Checkpoint: $DREAMS_CKPT"
echo "ChemBERTa Model: $CHEMBERTA_MODEL"
echo "Base Output Directory: $BASE_OUTDIR"
echo "Batch Size: $BATCH_SIZE"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch Mode: $BATCH_MODE"
echo ""

# Function to run a single ablation configuration on a specific GPU
run_ablation() {
    local config_name=$1
    local init_spec=$2
    local init_mol=$3
    local w_map=$4
    local w_con_mapped=$5
    local gpu_id=$6
    local outdir="${BASE_OUTDIR}/${config_name}"
    local log_file="${BASE_OUTDIR}/${config_name}.log"
    
    echo "[GPU $gpu_id] Starting: $config_name"
    echo "  Spec encoder: $([ "$init_spec" = "true" ] && echo "Random init" || echo "Pre-trained")"
    echo "  Mol encoder: $([ "$init_mol" = "true" ] && echo "Random init" || echo "Pre-trained")"
    echo "  Loss: $([ "$w_map" != "0" ] && echo "Alignment (MSE)" || echo "Contrastive (InfoNCE)")"
    echo "  Output: $outdir"
    echo "  Log: $log_file"
    
    # Build command
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python dreams_condition_adapter.py"
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
    
    if [ "$w_con_mapped" != "0" ] && [ "$w_map" = "0" ]; then
        cmd="$cmd --supcon-k $SUPCON_K"
    fi
    cmd="$cmd --log-every $LOG_EVERY"
    cmd="$cmd --save-every $SAVE_EVERY"
    cmd="$cmd --outdir $outdir"
    cmd="$cmd --mol-space chemberta"
    cmd="$cmd --chemberta-model $CHEMBERTA_MODEL"
    cmd="$cmd --lr $LR"
    
    if [ "$init_spec" = "true" ]; then
        cmd="$cmd --init-spec-from-scratch"
    fi
    if [ "$init_mol" = "true" ]; then
        cmd="$cmd --init-mol-from-scratch"
    fi
    if [[ "$config_name" == "4_freeze_align" || "$config_name" == "5_freeze_contrastive" ]]; then
        cmd="$cmd --unfreeze-last 0"
        cmd="$cmd --unfreeze-mol-last 0"
    fi
    
    echo "[GPU $gpu_id] Command: $cmd"
    echo "[GPU $gpu_id] Logging to: $log_file"
    echo ""
    
    mkdir -p "$outdir"
    nohup bash -c "$cmd" > "$log_file" 2>&1 &
    
    local pid=$!
    echo "$pid" > "${BASE_OUTDIR}/${config_name}.pid"
    echo "[GPU $gpu_id] Started $config_name (PID: $pid)"
    
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[ERROR] Failed to start $config_name on GPU $gpu_id" >&2
        return 1
    fi
    return 0
}

# Function to wait for specific PIDs
wait_for_pids() {
    local pids=("$@")
    echo "Waiting for PIDs: ${pids[*]}"
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "Batch completed!"
    echo ""
}

mkdir -p "$BASE_OUTDIR"

# Define all configurations
declare -a configs=(
    "1_random_align:true:true:5.0:0"
    "2_random_contrastive:true:true:0:1.0"
    "3_pretrained_contrastive:false:false:0:1.0"
    "4_freeze_align:false:false:5.0:0"
    "5_freeze_contrastive:false:false:0:1.0"
)

# Parse batch mode
IFS='+' read -ra BATCH_SIZES <<< "$BATCH_MODE"

num_configs=${#configs[@]}
echo "Starting $num_configs configurations in batches: ${BATCH_MODE}"
echo ""

config_idx=0
batch_num=1

for batch_size in "${BATCH_SIZES[@]}"; do
    batch_size=$(echo "$batch_size" | tr -d ' ')
    echo "=========================================="
    echo "Batch $batch_num: Starting $batch_size jobs"
    echo "=========================================="
    
    declare -a batch_pids=()
    gpu_idx=0
    
    for ((i=0; i<batch_size && config_idx<${#configs[@]}; i++)); do
        IFS=':' read -r config_name init_spec init_mol w_map w_con_mapped <<< "${configs[$config_idx]}"
        gpu_id=$((gpu_idx % NUM_GPUS))
        
        echo "[INFO] Launching configuration $((config_idx+1))/${#configs[@]}: $config_name on GPU $gpu_id"
        
        if run_ablation "$config_name" "$init_spec" "$init_mol" "$w_map" "$w_con_mapped" "$gpu_id"; then
            pid=$(cat "${BASE_OUTDIR}/${config_name}.pid" 2>/dev/null)
            batch_pids+=($pid)
            echo "[SUCCESS] Configuration $config_name started on GPU $gpu_id (PID: $pid)"
        else
            echo "[ERROR] Failed to start configuration $config_name" >&2
        fi
        
        ((config_idx++))
        ((gpu_idx++))
        sleep 2
    done
    
    echo ""
    echo "Batch $batch_num: All $batch_size jobs started. Waiting for completion..."
    wait_for_pids "${batch_pids[@]}"
    
    ((batch_num++))
done

echo "=========================================="
echo "All ablation study configurations completed!"
echo "Results saved in: $BASE_OUTDIR"
echo "=========================================="

rm -f "${BASE_OUTDIR}"/*.pid

