#!/bin/bash
# Optional Ablations Script for SpecBridge
# 
# This script runs three types of optional ablations:
# 1. Mapper capacity (varying n_blocks and mapper_hidden)
# 2. Spectrum unfreezing depth (frozen, last-1, last-2, last-4)
# 3. Procrustes warm start (orthogonal init vs random init)

set -e

# Base configuration
MGF_FILE="${MGF_FILE:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf}"
DREAMS_CKPT="${DREAMS_CKPT:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt}"
CHEMBERTA_MODEL="${CHEMBERTA_MODEL:-Derify/ChemBERTa_augmented_pubchem_13m}"
BASE_OUTDIR="${BASE_OUTDIR:-runs/optional_ablations_spectraverse}"

# GPU configuration
NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE_JOBS="${BATCH_SIZE_JOBS:-1}"

# Common training hyperparameters
BATCH_SIZE=16
EPOCHS=2
LR=1e-4
COND_DIM=2048
LOG_EVERY=50
SAVE_EVERY=200

# Loss weights (standard alignment configuration)
W_MAP=5.0
W_CON_MAPPED=0
W_ORTHO=1e-3

echo "=========================================="
echo "SpecBridge Optional Ablations"
echo "=========================================="
echo "MGF File: $MGF_FILE"
echo "Base Output Directory: $BASE_OUTDIR"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# Function to run a single ablation configuration
run_ablation() {
    local config_name=$1
    local gpu_id=$2
    shift 2
    local extra_args="$@"
    
    local outdir="${BASE_OUTDIR}/${config_name}"
    local log_file="${BASE_OUTDIR}/${config_name}.log"
    
    echo "[GPU $gpu_id] Starting: $config_name"
    echo "  Output: $outdir"
    
    # Build base command
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python dreams_condition_adapter.py"
    cmd="$cmd --mgf $MGF_FILE"
    cmd="$cmd --dreams-ckpt $DREAMS_CKPT"
    cmd="$cmd --fold train"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --cond-dim $COND_DIM"
    cmd="$cmd --no-gaussian"
    cmd="$cmd --w-map $W_MAP"
    cmd="$cmd --w-con-mapped $W_CON_MAPPED"
    cmd="$cmd --w-ortho $W_ORTHO"
    cmd="$cmd --log-every $LOG_EVERY"
    cmd="$cmd --save-every $SAVE_EVERY"
    cmd="$cmd --outdir $outdir"
    cmd="$cmd --mol-space chemberta"
    cmd="$cmd --chemberta-model $CHEMBERTA_MODEL"
    cmd="$cmd --lr $LR"
    cmd="$cmd $extra_args"
    
    # Create output directory
    mkdir -p "$outdir"
    
    # Run in background
    nohup bash -c "$cmd" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "${BASE_OUTDIR}/${config_name}.pid"
    
    sleep 1
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "[ERROR] Failed to start $config_name on GPU $gpu_id" >&2
        return 1
    fi
    
    echo "[GPU $gpu_id] Started $config_name (PID: $pid)"
    return 0
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

mkdir -p "$BASE_OUTDIR"

# ========== Ablation 1: Mapper Capacity ==========
echo ""
echo "=========================================="
echo "Ablation 1: Mapper Capacity"
echo "=========================================="
echo "Varying n_blocks ∈ {0, 2, 4, 8} and mapper_hidden ∈ {512, 1024, 2048}"
echo ""

declare -a mapper_configs=(
    # "mapper_n0_h512:--n-blocks 0 --mapper-hidden 512"
    # "mapper_n0_h1024:--n-blocks 0 --mapper-hidden 1024"
    # "mapper_n0_h2048:--n-blocks 0 --mapper-hidden 2048"
    # "mapper_n2_h512:--n-blocks 2 --mapper-hidden 512"
    # "mapper_n2_h1024:--n-blocks 2 --mapper-hidden 1024"
    # "mapper_n2_h2048:--n-blocks 2 --mapper-hidden 2048"
    # "mapper_n4_h512:--n-blocks 4 --mapper-hidden 512"
    # "mapper_n4_h1024:--n-blocks 4 --mapper-hidden 1024"
    # "mapper_n4_h2048:--n-blocks 4 --mapper-hidden 2048"
    # "mapper_n8_h512:--n-blocks 8 --mapper-hidden 512"
    # "mapper_n8_h1024:--n-blocks 8 --mapper-hidden 1024"
    # "mapper_n8_h2048:--n-blocks 8 --mapper-hidden 2048"
)

# ========== Ablation 2: Spectrum Unfreezing Depth ==========
echo ""
echo "=========================================="
echo "Ablation 2: Spectrum Unfreezing Depth"
echo "=========================================="
echo "Varying unfreeze_last ∈ {0, 1, 2, 4}"
echo ""

declare -a unfreeze_configs=(
    # "unfreeze_0:--unfreeze-last 0"
    # "unfreeze_1:--unfreeze-last 1 --unfreeze-after 0"
    # "unfreeze_2:--unfreeze-last 2 --unfreeze-after 0"
    # "unfreeze_4:--unfreeze-last 4 --unfreeze-after 0"
)

# ========== Ablation 3: Procrustes Warm Start ==========
echo ""
echo "=========================================="
echo "Ablation 3: Procrustes Warm Start"
echo "=========================================="
echo "Comparing orthogonal/Procrustes init vs random init"
echo "Note: Procrustes init is the default. Random init requires code modification."
echo ""

declare -a procrustes_configs=(
    # "procrustes_warmstart:--n-blocks 8 --mapper-hidden 2048 --unfreeze-last 2 --unfreeze-after 0"
    "procrustes_random_init:--n-blocks 8 --mapper-hidden 2048 --unfreeze-last 2 --unfreeze-after 0 --random-mapper-init"
)

# Combine all configurations
declare -a all_configs=(
    "${mapper_configs[@]}"
    "${unfreeze_configs[@]}"
    "${procrustes_configs[@]}"
)

# Launch all configurations
batch_num=1
for i in "${!all_configs[@]}"; do
    IFS=':' read -r config_name extra_args <<< "${all_configs[$i]}"
    
    # Add default mapper settings for unfreeze and procrustes configs if not specified
    if [[ "$config_name" == unfreeze_* ]]; then
        extra_args="$extra_args --n-blocks 8 --mapper-hidden 2048"
    fi
    
    # Wait if we've reached the batch size limit
    while [ $(count_running_jobs) -ge $BATCH_SIZE_JOBS ]; do
        echo "[INFO] Batch limit reached ($BATCH_SIZE_JOBS jobs). Waiting..."
        sleep 10
    done
    
    # Find available GPU (round-robin)
    running_count=$(count_running_jobs)
    gpu_id=$((running_count % NUM_GPUS))
    
    echo "[INFO] Launching configuration $((i+1))/${#all_configs[@]}: $config_name on GPU $gpu_id"
    
    if run_ablation "$config_name" "$gpu_id" "$extra_args"; then
        echo "[SUCCESS] Configuration $config_name started"
    else
        echo "[ERROR] Failed to start configuration $config_name" >&2
    fi
    
    sleep 3
done

echo ""
echo "=========================================="
echo "All optional ablation jobs launched!"
echo "=========================================="
echo "Monitor progress with:"
echo "  tail -f ${BASE_OUTDIR}/*.log"
echo ""
echo "Results will be saved in: $BASE_OUTDIR"
echo ""

# Check if unfreeze_4 was actually launched
if [ ! -f "${BASE_OUTDIR}/unfreeze_4.pid" ] && [ ! -f "${BASE_OUTDIR}/unfreeze_4.log" ]; then
    echo ""
    echo "=========================================="
    echo "WARNING: unfreeze_4 was not launched!"
    echo "=========================================="
    echo "This may be because:"
    echo "  1. The script was run when unfreeze_4 was commented out"
    echo "  2. The script was interrupted before reaching unfreeze_4"
    echo "  3. There was an error launching unfreeze_4"
    echo ""
    echo "To run unfreeze_4 manually, use:"
    echo "  ${BASE_OUTDIR}/run_unfreeze_4.sh"
    echo ""
    
    # Create a helper script to run unfreeze_4 manually
    cat > "${BASE_OUTDIR}/run_unfreeze_4.sh" << 'RUNSCRIPT'
#!/bin/bash
# Manual script to run unfreeze_4 ablation
# This is created automatically if unfreeze_4 wasn't launched

MGF_FILE="${MGF_FILE:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf}"
DREAMS_CKPT="${DREAMS_CKPT:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt}"
CHEMBERTA_MODEL="${CHEMBERTA_MODEL:-Derify/ChemBERTa_augmented_pubchem_13m}"
BASE_OUTDIR="${BASE_OUTDIR:-runs/optional_ablations_spectraverse}"

BATCH_SIZE=16
EPOCHS=2
LR=1e-4
COND_DIM=2048
LOG_EVERY=50
SAVE_EVERY=200
W_MAP=5.0
W_CON_MAPPED=0
W_ORTHO=1e-3

config_name="unfreeze_4"
outdir="${BASE_OUTDIR}/${config_name}"
log_file="${BASE_OUTDIR}/${config_name}.log"

mkdir -p "$outdir"

# Use GPU 0 by default, or set CUDA_VISIBLE_DEVICES
gpu_id="${CUDA_VISIBLE_DEVICES:-0}"

cmd="CUDA_VISIBLE_DEVICES=3 python dreams_condition_adapter.py"
cmd="$cmd --mgf $MGF_FILE"
cmd="$cmd --dreams-ckpt $DREAMS_CKPT"
cmd="$cmd --fold train"
cmd="$cmd --batch-size $BATCH_SIZE"
cmd="$cmd --epochs $EPOCHS"
cmd="$cmd --cond-dim $COND_DIM"
cmd="$cmd --no-gaussian"
cmd="$cmd --w-map $W_MAP"
cmd="$cmd --w-con-mapped $W_CON_MAPPED"
cmd="$cmd --w-ortho $W_ORTHO"
cmd="$cmd --log-every $LOG_EVERY"
cmd="$cmd --save-every $SAVE_EVERY"
cmd="$cmd --outdir $outdir"
cmd="$cmd --mol-space chemberta"
cmd="$cmd --chemberta-model $CHEMBERTA_MODEL"
cmd="$cmd --lr $LR"
cmd="$cmd --n-blocks 8 --mapper-hidden 2048 --unfreeze-last 4 --unfreeze-after 0"

echo "Running: $config_name"
echo "Output: $outdir"
echo "Log: $log_file"
echo ""

nohup bash -c "$cmd" > "$log_file" 2>&1 &
pid=$!
echo "$pid" > "${BASE_OUTDIR}/${config_name}.pid"
echo "Started with PID: $pid"
echo "Monitor with: tail -f $log_file"
RUNSCRIPT
    chmod +x "${BASE_OUTDIR}/run_unfreeze_4.sh"
fi
