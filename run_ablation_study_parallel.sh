#!/bin/bash
# Ablation Study Script for SpecBridge - Parallel Execution
# 
# This script runs 5 ablation study configurations in parallel across multiple GPUs:
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

# GPU configuration
NUM_GPUS="${NUM_GPUS:-4}"  # Number of GPUs to use (default: 4)
BATCH_SIZE_JOBS="${BATCH_SIZE_JOBS:-3}"  # Number of jobs to run simultaneously per batch (default: 3)
# Options: 3 (then 2), 2 (then 2, then 1), or 4 (then 1)

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
echo "SpecBridge Ablation Study - Parallel Execution"
echo "=========================================="
echo "MGF File: $MGF_FILE"
echo "DreaMS Checkpoint: $DREAMS_CKPT"
echo "ChemBERTa Model: $CHEMBERTA_MODEL"
echo "Base Output Directory: $BASE_OUTDIR"
echo "Batch Size: $BATCH_SIZE"
echo "Number of GPUs: $NUM_GPUS"
echo "Jobs per batch: $BATCH_SIZE_JOBS"
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
    
    # For contrastive learning, set supcon_k to ensure multiple positive pairs per batch
    if [ "$w_con_mapped" != "0" ] && [ "$w_map" = "0" ]; then
        cmd="$cmd --supcon-k $SUPCON_K"
    fi
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
    
    # For pretrained_contrastive (config 3), fine-tune last 2 layers of spectra encoder
    if [[ "$config_name" == "3_pretrained_contrastive" ]]; then
        cmd="$cmd --unfreeze-last 2"
        cmd="$cmd --unfreeze-after 0"
    fi
    
    # Run the command in background and redirect output to log file
    echo "[GPU $gpu_id] Command: $cmd"
    echo "[GPU $gpu_id] Logging to: $log_file"
    echo ""
    
    # Create output directory
    mkdir -p "$outdir"
    
    # Run in background, redirecting both stdout and stderr to log file
    # Use nohup to ensure the process continues even if the parent shell exits
    nohup bash -c "$cmd" > "$log_file" 2>&1 &
    
    local pid=$!
    echo "[GPU $gpu_id] Started $config_name (PID: $pid)"
    echo "$pid" > "${BASE_OUTDIR}/${config_name}.pid"
    
    # Verify the process started
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[ERROR] Failed to start $config_name on GPU $gpu_id" >&2
        return 1
    fi
    
    return 0
}

# Create output directory
mkdir -p "$BASE_OUTDIR"

# Define all configurations
declare -a configs=(
    # "1_random_align:true:true:5.0:0"
    # "2_random_contrastive:true:true:0:1.0"
    "3_pretrained_contrastive:false:false:0:1.0"
    # "4_freeze_align:false:false:5.0:0"
    # "5_freeze_contrastive:false:false:0:1.0"
)

# Start jobs in batches
declare -a pids=()
num_configs=${#configs[@]}

echo "Starting $num_configs configurations in batches of $BATCH_SIZE_JOBS across $NUM_GPUS GPUs..."
echo ""

# Function to wait for jobs to complete
wait_for_jobs() {
    local batch_num=$1
    echo ""
    echo "=========================================="
    echo "Batch $batch_num: Waiting for jobs to complete..."
    echo "=========================================="
    
    local all_pids=()
    for pid_file in "${BASE_OUTDIR}"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if kill -0 "$pid" 2>/dev/null; then
                all_pids+=($pid)
            fi
        fi
    done
    
    if [ ${#all_pids[@]} -gt 0 ]; then
        echo "Waiting for PIDs: ${all_pids[*]}"
        for pid in "${all_pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
    
    echo "Batch $batch_num completed!"
    echo ""
}

# Function to count running jobs
count_running_jobs() {
    local count=0
    for pid_file in "${BASE_OUTDIR}"/*.pid; do
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if kill -0 "$pid" 2>/dev/null; then
                ((count++))
            fi
        fi
    done
    echo $count
}

# Launch jobs in batches
batch_num=1
for i in "${!configs[@]}"; do
    IFS=':' read -r config_name init_spec init_mol w_map w_con_mapped <<< "${configs[$i]}"
    
    # Wait if we've reached the batch size limit
    while [ $(count_running_jobs) -ge $BATCH_SIZE_JOBS ]; do
        echo "[INFO] Batch limit reached ($BATCH_SIZE_JOBS jobs). Waiting for jobs to complete..."
        sleep 10
    done
    
    # Find available GPU (round-robin within available GPUs)
    running_count=$(count_running_jobs)
    gpu_id=$((running_count % NUM_GPUS))
    
    echo "[INFO] Launching configuration $((i+1))/$num_configs: $config_name on GPU $gpu_id"
    echo "[INFO] Currently running: $(count_running_jobs) jobs"
    
    if run_ablation "$config_name" "$init_spec" "$init_mol" "$w_map" "$w_con_mapped" "$gpu_id"; then
        pid=$(cat "${BASE_OUTDIR}/${config_name}.pid" 2>/dev/null)
        pids+=($pid)
        echo "[SUCCESS] Configuration $config_name started on GPU $gpu_id (PID: $pid)"
    else
        echo "[ERROR] Failed to start configuration $config_name on GPU $gpu_id" >&2
    fi
    
    # Small delay to avoid race conditions and allow GPU initialization
    sleep 3
done

echo ""
echo "=========================================="
echo "All jobs launched! Waiting for completion..."
echo "=========================================="
echo "Running configurations:"
for i in "${!configs[@]}"; do
    IFS=':' read -r config_name init_spec init_mol w_map w_con_mapped <<< "${configs[$i]}"
    pid_file="${BASE_OUTDIR}/${config_name}.pid"
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file" 2>/dev/null || echo "unknown")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  $config_name: Running (PID: $pid)"
        else
            echo "  $config_name: Completed"
        fi
    fi
done
echo ""
echo "Monitor progress with:"
echo "  tail -f ${BASE_OUTDIR}/*.log"
echo ""
echo "Check job status with:"
echo "  ps aux | grep dreams_condition_adapter | grep -v grep"
echo ""

# Wait for all remaining jobs to complete
wait_for_jobs "final"

echo ""
echo "=========================================="
echo "All ablation study configurations completed!"
echo "Results saved in: $BASE_OUTDIR"
echo "=========================================="

# Clean up PID files
rm -f "${BASE_OUTDIR}"/*.pid

