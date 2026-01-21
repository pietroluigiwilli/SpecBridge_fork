#!/usr/bin/env bash
set -uo pipefail

# Evaluation script for optional ablations

BASE_DIR="runs/optional_ablations_spectraverse"
MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
CANDS="${CANDS:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl}"
FOLD="val"
BATCH=128

# GPU configuration
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE_JOBS="${BATCH_SIZE_JOBS:-4}"

# Embedding space
MOL_SPACE="chemberta"
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"

echo "=========================================="
echo "Optional Ablations Evaluation"
echo "=========================================="

# Master summary file
MASTER_CSV="${BASE_DIR}/optional_ablations_summary_${FOLD}.csv"
echo "config,ckpt,step,R@1,R@5,R@20,MRR,median_rank,total_queries,evaluated" > "${MASTER_CSV}"

# Collect all evaluation tasks
declare -a eval_tasks=()

# Find all configuration directories
shopt -s nullglob
for CONFIG_DIR in "${BASE_DIR}"/mapper_* "${BASE_DIR}"/unfreeze_* "${BASE_DIR}"/procrustes_*; do
    [[ -d "${CONFIG_DIR}" ]] || continue
    CONFIG=$(basename "${CONFIG_DIR}")
    
    OUTCSV="${CONFIG_DIR}/eval_summary_${FOLD}_all.csv"
    LOGDIR="${CONFIG_DIR}/eval_logs"
    mkdir -p "${LOGDIR}"
    
    if [[ ! -f "${OUTCSV}" ]]; then
        echo "ckpt,step,R@1,R@5,R@20,MRR,median_rank,total_queries,evaluated" > "${OUTCSV}"
    fi
    
    # Use cache for all (frozen mol encoder)
    USE_CACHE=true
    
    # Find all checkpoints
    for CK in "${CONFIG_DIR}"/ckpt_*.pt "${CONFIG_DIR}"/best_val.pt "${CONFIG_DIR}"/last.pt; do
        [[ -f "${CK}" ]] || continue
        
        if [[ "${CK}" == *"best_val.pt" ]]; then
            STEP="best_val"
        elif [[ "${CK}" == *"last.pt" ]]; then
            STEP="last"
        else
            STEP=$(basename "${CK}" | sed -E 's/ckpt_0*([0-9]+)\.pt/\1/')
            # Skip checkpoints before step 10000
            if [[ "${STEP}" =~ ^[0-9]+$ ]] && [[ "${STEP}" -lt 20000 ]]; then
                continue
            fi
        fi
        
        # Skip if already evaluated
        if grep -q ",${STEP}," "${OUTCSV}" 2>/dev/null; then
            continue
        fi
        
        eval_tasks+=("${CONFIG}:${CK}:${STEP}:${OUTCSV}:${LOGDIR}:${USE_CACHE}")
    done
done

echo ">>> Found ${#eval_tasks[@]} evaluation tasks"
echo ""

if [[ ${#eval_tasks[@]} -eq 0 ]]; then
    echo "No evaluation tasks found."
    exit 0
fi

# Function to run evaluation
run_eval() {
    local task=$1
    local gpu_id=$2
    
    IFS=':' read -r CONFIG CK STEP OUTCSV LOGDIR USE_CACHE <<< "${task}"
    local LOG="${LOGDIR}/eval_${STEP}.log"
    
    echo "[GPU $gpu_id] Evaluating ${CONFIG} - ${STEP}"
    
    # Extract n_blocks and mapper_hidden from config name if present
    N_BLOCKS=8
    MAPPER_HIDDEN=2048
    COND_DIM=2048
    
    if [[ "${CONFIG}" == mapper_n* ]]; then
        # Parse n_blocks and mapper_hidden from config name
        if [[ "${CONFIG}" =~ mapper_n([0-9]+)_h([0-9]+) ]]; then
            N_BLOCKS="${BASH_REMATCH[1]}"
            MAPPER_HIDDEN="${BASH_REMATCH[2]}"
        fi
    fi
    
    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python -m specbridge.eval.candidates"
    cmd="$cmd --adapter-ckpt ${CK}"
    cmd="$cmd --mgf ${MGF}"
    cmd="$cmd --dreams-ckpt ${DREAMS}"
    cmd="$cmd --candidates ${CANDS}"
    cmd="$cmd --fold-query ${FOLD}"
    cmd="$cmd --use-mapped --deterministic-map"
    cmd="$cmd --batch-size ${BATCH}"
    cmd="$cmd --cond-dim ${COND_DIM} --mapper-hidden ${MAPPER_HIDDEN}"
    cmd="$cmd --mol-space ${MOL_SPACE}"
    cmd="$cmd --no-gaussian --n-blocks ${N_BLOCKS}"
    cmd="$cmd --chemberta-model ${CHEMBERTA_MODEL}"
    
    if [[ "${USE_CACHE}" == "true" ]]; then
        cmd="$cmd --cache-cand-emb ${CACHE}"
    fi
    
    set +e
    eval "$cmd" > "${LOG}" 2>&1
    EC=$?
    set -e
    
    if [[ ${EC} -ne 0 ]]; then
        echo "[GPU $gpu_id] ERROR (${EC}) on ${CONFIG} - ${STEP}; see ${LOG}"
        echo "${CK},${STEP},NA,NA,NA,NA,NA,NA,NA" >> "${OUTCSV}"
        echo "${CONFIG},${CK},${STEP},NA,NA,NA,NA,NA,NA,NA" >> "${MASTER_CSV}"
        return 1
    fi
    
    # Parse metrics
    R1=$(awk '/R@1:/  {print $2}'  "${LOG}" 2>/dev/null || echo "NA")
    R5=$(awk '/R@5:/  {print $2}'  "${LOG}" 2>/dev/null || echo "NA")
    R20=$(awk '/R@20:/ {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    MRR=$(awk '/MRR:/  {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    MED=$(awk '/median_rank:/ {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    TOT=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /total_queries/) print $(i+1)}' "${LOG}" 2>/dev/null || echo "NA")
    EVD=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /evaluated/) print $(i+1)}' "${LOG}" 2>/dev/null || echo "NA")
    
    echo "${CK},${STEP},${R1},${R5},${R20},${MRR},${MED},${TOT},${EVD}" >> "${OUTCSV}"
    echo "${CONFIG},${CK},${STEP},${R1},${R5},${R20},${MRR},${MED},${TOT},${EVD}" >> "${MASTER_CSV}"
    
    echo "[GPU $gpu_id] Completed ${CONFIG} - ${STEP}: R@1=${R1}, R@5=${R5}, MRR=${MRR}"
    return 0
}

# Function to count running jobs
count_running_jobs() {
    local count=0
    for pid_file in "${BASE_DIR}"/*.eval.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if kill -0 "$pid" 2>/dev/null; then
                ((count++))
            fi
        fi
    done
    echo $count
}

# Function to find available GPU
find_available_gpu() {
    local min_jobs=999999
    local best_gpu=0
    
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        local jobs=0
        for pid_file in "${BASE_DIR}"/*.eval.pid; do
            if [[ -f "$pid_file" ]]; then
                local pid=$(cat "$pid_file" 2>/dev/null)
                if kill -0 "$pid" 2>/dev/null; then
                    local gpu_file="${pid_file%.eval.pid}.gpu"
                    if [[ -f "$gpu_file" ]]; then
                        local assigned_gpu=$(cat "$gpu_file" 2>/dev/null)
                        if [[ "$assigned_gpu" == "$gpu" ]]; then
                            ((jobs++))
                        fi
                    fi
                fi
            fi
        done
        if [[ $jobs -lt $min_jobs ]]; then
            min_jobs=$jobs
            best_gpu=$gpu
        fi
    done
    echo $best_gpu
}

# Process evaluation tasks
declare -a pids=()

for task in "${eval_tasks[@]}"; do
    IFS=':' read -r CONFIG CK STEP OUTCSV LOGDIR USE_CACHE <<< "${task}"
    
    while [ $(count_running_jobs) -ge $BATCH_SIZE_JOBS ]; do
        sleep 5
    done
    
    gpu_id=$(find_available_gpu)
    echo "${gpu_id}" > "${BASE_DIR}/${CONFIG}_${STEP}.gpu"
    
    echo "[INFO] Launching: ${CONFIG} - ${STEP} on GPU ${gpu_id}"
    
    (
        run_eval "${task}" "${gpu_id}"
    ) &
    
    pid=$!
    echo "${pid}" > "${BASE_DIR}/${CONFIG}_${STEP}.eval.pid"
    pids+=($pid)
    
    sleep 0.5
done

# Wait for all jobs
for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# Clean up
rm -f "${BASE_DIR}"/*.eval.pid
rm -f "${BASE_DIR}"/*.gpu

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Summary: ${MASTER_CSV}"
echo "=========================================="


