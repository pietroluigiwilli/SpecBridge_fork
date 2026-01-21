#!/usr/bin/env bash
set -uo pipefail

# ========= Configuration =========
BASE_DIR="runs/optional_ablations_spectraverse"
MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
CANDS="${CANDS:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl}"
FOLD="test"
BATCH=32
LIMIT=100000

# GPU configuration
NUM_GPUS="${NUM_GPUS:-4}"  # Number of GPUs to use (default: 4)
BATCH_SIZE_JOBS="${BATCH_SIZE_JOBS:-4}"  # Number of evaluation jobs to run simultaneously (default: 4)

# Embedding space
MOL_SPACE="chemberta"
COND_DIM=2048
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"

echo "=========================================="
echo "Optional Ablations Test Set Evaluation with MCES"
echo "=========================================="
echo "Fold: ${FOLD}"
echo "Number of GPUs: $NUM_GPUS"
echo "Jobs per batch: $BATCH_SIZE_JOBS"
echo ""

# ========= Function to find best checkpoint by R@5 =========
find_best_ckpt() {
    local config=$1
    local val_csv="${BASE_DIR}/${config}/eval_summary_val_all.csv"
    
    if [[ ! -f "${val_csv}" ]]; then
        echo ""
        return 1
    fi
    
    # Check if file has data (more than just header)
    if [[ $(wc -l < "${val_csv}") -le 1 ]]; then
        echo ""
        return 1
    fi
    
    # Sort by R@5 (column 4) descending, get first line, extract checkpoint path (column 1)
    local best_line=$(tail -n +2 "${val_csv}" | grep -v "^$" | sort -t',' -k4,4gr | head -1)
    if [[ -z "${best_line}" ]]; then
        echo ""
        return 1
    fi
    
    # Extract checkpoint path (first column)
    echo "${best_line}" | cut -d',' -f1
}

# ========= Function to parse config parameters =========
parse_config_params() {
    local config=$1
    local n_blocks=8
    local mapper_hidden=2048
    
    # Parse mapper_nX_hY format
    if [[ "${config}" =~ mapper_n([0-9]+)_h([0-9]+) ]]; then
        n_blocks="${BASH_REMATCH[1]}"
        mapper_hidden="${BASH_REMATCH[2]}"
    # Parse unfreeze_X format (default: n_blocks=8, hidden=2048)
    elif [[ "${config}" =~ unfreeze_[0-9]+ ]]; then
        n_blocks=8
        mapper_hidden=2048
    # Parse procrustes_* format (default: n_blocks=8, hidden=2048)
    elif [[ "${config}" =~ procrustes_ ]]; then
        n_blocks=8
        mapper_hidden=2048
    fi
    
    echo "${n_blocks}:${mapper_hidden}"
}

# ========= Find all configurations =========
declare -a CONFIGS=()
for dir in "${BASE_DIR}"/*/; do
    config=$(basename "${dir}")
    # Skip if it's not a directory or doesn't have eval_summary_val_all.csv
    if [[ ! -f "${BASE_DIR}/${config}/eval_summary_val_all.csv" ]]; then
        continue
    fi
    # Skip if the CSV is empty (just header)
    if [[ $(wc -l < "${BASE_DIR}/${config}/eval_summary_val_all.csv") -le 1 ]]; then
        continue
    fi
    CONFIGS+=("${config}")
done

echo ">>> Found ${#CONFIGS[@]} configurations with validation results"
echo ""

# ========= Find best checkpoints for each configuration =========
declare -A BEST_CKPTS=()
declare -A CONFIG_PARAMS=()

for CONFIG in "${CONFIGS[@]}"; do
    CKPT=$(find_best_ckpt "${CONFIG}")
    if [[ -z "${CKPT}" ]] || [[ ! -f "${CKPT}" ]]; then
        echo ">>> WARNING: No valid checkpoint found for ${CONFIG}"
        continue
    fi
    
    BEST_CKPTS["${CONFIG}"]="${CKPT}"
    CONFIG_PARAMS["${CONFIG}"]=$(parse_config_params "${CONFIG}")
    
    # Extract R@5 from validation results for info
    VAL_CSV="${BASE_DIR}/${CONFIG}/eval_summary_val_all.csv"
    BEST_R5=$(tail -n +2 "${VAL_CSV}" | sort -t',' -k4,4gr | head -1 | cut -d',' -f4)
    echo ">>> ${CONFIG}: best R@5=${BEST_R5} -> ${CKPT}"
done

echo ""
echo ">>> Found ${#BEST_CKPTS[@]} valid checkpoints to evaluate"
echo ""

# ========= Master summary file =========
MASTER_CSV="${BASE_DIR}/optional_ablations_test_summary_mces.csv"
echo "config,ckpt,step,R@1,R@5,R@20,MRR,median_rank,total_queries,evaluated,mces@1_mean,mces@1_median" > "${MASTER_CSV}"

# ========= Collect all evaluation tasks =========
declare -a eval_tasks=()

for CONFIG in "${!BEST_CKPTS[@]}"; do
    CKPT="${BEST_CKPTS[$CONFIG]}"
    
    RUN_DIR="${BASE_DIR}/${CONFIG}"
    
    # Per-config summary
    OUTCSV="${RUN_DIR}/eval_test_summary_mces.csv"
    LOGDIR="${RUN_DIR}/eval_test_logs"
    mkdir -p "${LOGDIR}"
    
    # Write header if new
    if [[ ! -f "${OUTCSV}" ]]; then
        echo "ckpt,step,R@1,R@5,R@20,MRR,median_rank,total_queries,evaluated,mces@1_mean,mces@1_median" > "${OUTCSV}"
    fi
    
    # Extract step number
    if [[ "${CKPT}" == *"best_val.pt" ]]; then
        STEP="best_val"
    elif [[ "${CKPT}" == *"last.pt" ]]; then
        STEP="last"
    else
        STEP=$(basename "${CKPT}" | sed -E 's/ckpt_0*([0-9]+)\.pt/\1/')
    fi
    
    # Skip if already present in CSV (idempotent)
    if grep -q ",${STEP}," "${OUTCSV}" 2>/dev/null; then
        echo ">>> Skipping ${CONFIG} - ${STEP}: already evaluated"
        continue
    fi
    
    # Parse config parameters
    IFS=':' read -r N_BLOCKS MAPPER_HIDDEN <<< "${CONFIG_PARAMS[$CONFIG]}"
    
    # All configs use cache (frozen/pre-trained mol encoder)
    USE_CACHE=true
    
    # Add to task list: "CONFIG:CKPT_PATH:STEP:OUTCSV:LOGDIR:USE_CACHE:N_BLOCKS:MAPPER_HIDDEN"
    eval_tasks+=("${CONFIG}:${CKPT}:${STEP}:${OUTCSV}:${LOGDIR}:${USE_CACHE}:${N_BLOCKS}:${MAPPER_HIDDEN}")
done

echo ">>> Found ${#eval_tasks[@]} evaluation tasks"
echo ""

# ========= Function to run a single evaluation =========
run_eval() {
    local task=$1
    local gpu_id=$2
    
    IFS=':' read -r CONFIG CK STEP OUTCSV LOGDIR USE_CACHE N_BLOCKS MAPPER_HIDDEN <<< "${task}"
    local LOG="${LOGDIR}/eval_test_${STEP}.log"
    
    echo "[GPU $gpu_id] Evaluating ${CONFIG} - ${STEP} on test set (n_blocks=${N_BLOCKS}, hidden=${MAPPER_HIDDEN})"
    
    # Build command
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
    cmd="$cmd --compute-mces"
    
    if [[ "${USE_CACHE}" == "true" ]]; then
        cmd="$cmd --cache-cand-emb ${CACHE}"
    fi
    
    # Run eval and redirect to log
    set +e
    eval "$cmd" > "${LOG}" 2>&1
    EC=$?
    set -e
    
    if [[ ${EC} -ne 0 ]]; then
        echo "[GPU $gpu_id] ERROR (${EC}) on ${CONFIG} - ${STEP}; see ${LOG}"
        echo "${CK},${STEP},NA,NA,NA,NA,NA,NA,NA,NA,NA" >> "${OUTCSV}"
        echo "${CONFIG},${CK},${STEP},NA,NA,NA,NA,NA,NA,NA,NA,NA" >> "${MASTER_CSV}"
        return 1
    fi
    
    # Parse metrics from the log
    R1=$(awk '/R@1:/  {print $2}'  "${LOG}" 2>/dev/null || echo "NA")
    R5=$(awk '/R@5:/  {print $2}'  "${LOG}" 2>/dev/null || echo "NA")
    R20=$(awk '/R@20:/ {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    MRR=$(awk '/MRR:/  {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    MED=$(awk '/median_rank:/ {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    TOT=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /total_queries/) print $(i+1)}' "${LOG}" 2>/dev/null || echo "NA")
    EVD=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /evaluated/) print $(i+1)}' "${LOG}" 2>/dev/null || echo "NA")
    MCES_MEAN=$(awk '/mces@1_mean:/ {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    MCES_MEDIAN=$(awk '/mces@1_median:/ {print $2}' "${LOG}" 2>/dev/null || echo "NA")
    
    # Write to per-config CSV
    echo "${CK},${STEP},${R1},${R5},${R20},${MRR},${MED},${TOT},${EVD},${MCES_MEAN},${MCES_MEDIAN}" >> "${OUTCSV}"
    
    # Write to master CSV
    echo "${CONFIG},${CK},${STEP},${R1},${R5},${R20},${MRR},${MED},${TOT},${EVD},${MCES_MEAN},${MCES_MEDIAN}" >> "${MASTER_CSV}"
    
    echo "[GPU $gpu_id] Completed ${CONFIG} - ${STEP}: R@1=${R1}, R@5=${R5}, MRR=${MRR}, MCES_mean=${MCES_MEAN}"
    return 0
}

# ========= Function to count running jobs =========
count_running_jobs() {
    local count=0
    for pid_file in "${BASE_DIR}"/*.test_eval.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if kill -0 "$pid" 2>/dev/null; then
                ((count++))
            fi
        fi
    done
    echo $count
}

# ========= Function to get jobs per GPU =========
get_jobs_per_gpu() {
    local gpu_id=$1
    local count=0
    
    # Count from .gpu files for running jobs
    for pid_file in "${BASE_DIR}"/*.test_eval.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if kill -0 "$pid" 2>/dev/null; then
                local gpu_file="${pid_file%.test_eval.pid}.test_gpu"
                if [[ -f "$gpu_file" ]]; then
                    local assigned_gpu=$(cat "$gpu_file" 2>/dev/null)
                    if [[ "$assigned_gpu" == "$gpu_id" ]]; then
                        ((count++))
                    fi
                fi
            fi
        fi
    done
    
    echo $count
}

# ========= Function to find available GPU =========
find_available_gpu() {
    # Find GPU with fewest running jobs
    local min_jobs=999999
    local best_gpu=0
    
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        local jobs=$(get_jobs_per_gpu $gpu)
        if [[ $jobs -lt $min_jobs ]]; then
            min_jobs=$jobs
            best_gpu=$gpu
        fi
    done
    
    echo $best_gpu
}

# ========= Function to wait for jobs =========
wait_for_jobs() {
    local all_pids=()
    for pid_file in "${BASE_DIR}"/*.test_eval.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file" 2>/dev/null)
            if kill -0 "$pid" 2>/dev/null; then
                all_pids+=($pid)
            fi
        fi
    done
    
    if [[ ${#all_pids[@]} -gt 0 ]]; then
        for pid in "${all_pids[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
    fi
}

# ========= Process evaluation tasks in parallel =========
declare -a pids=()
task_idx=0

for task in "${eval_tasks[@]}"; do
    IFS=':' read -r CONFIG CK STEP OUTCSV LOGDIR USE_CACHE N_BLOCKS MAPPER_HIDDEN <<< "${task}"
    
    # Wait if we've reached the batch size limit
    while [ $(count_running_jobs) -ge $BATCH_SIZE_JOBS ]; do
        echo "[INFO] Batch limit reached ($BATCH_SIZE_JOBS jobs). Waiting..."
        sleep 5
    done
    
    # Find available GPU (one with fewest running jobs)
    # Create GPU assignment file FIRST to avoid race condition
    gpu_id=$(find_available_gpu)
    echo "${gpu_id}" > "${BASE_DIR}/${CONFIG}_${STEP}.test_gpu"
    
    ((task_idx++))
    echo "[INFO] Launching task ${task_idx}/${#eval_tasks[@]}: ${CONFIG} - ${STEP} on GPU ${gpu_id} (jobs on GPU: $(get_jobs_per_gpu $gpu_id))"
    
    # Run evaluation in background
    (
        run_eval "${task}" "${gpu_id}"
    ) &
    
    pid=$!
    echo "${pid}" > "${BASE_DIR}/${CONFIG}_${STEP}.test_eval.pid"
    pids+=($pid)
    
    # Small delay to allow GPU assignment file to be written and counted
    sleep 0.5
done

echo ""
echo "=========================================="
echo "All evaluation tasks launched! Waiting for completion..."
echo "=========================================="

# Wait for all jobs to complete
wait_for_jobs

# Clean up PID and GPU files
rm -f "${BASE_DIR}"/*.test_eval.pid
rm -f "${BASE_DIR}"/*.test_gpu

# ========= Show results per configuration =========
echo ""
echo "=========================================="
echo "Test Set Results by Configuration (with MCES)"
echo "=========================================="

for CONFIG in "${!BEST_CKPTS[@]}"; do
    OUTCSV="${BASE_DIR}/${CONFIG}/eval_test_summary_mces.csv"
    if [[ -f "${OUTCSV}" ]]; then
        echo ""
        echo ">>> Results for ${CONFIG}:"
        tail -n +2 "${OUTCSV}" | grep -v "NA" | column -s, -t
    fi
done

# ========= Final summary =========
echo ""
echo "=========================================="
echo "Optional Ablations Test Set Summary (with MCES)"
echo "=========================================="
echo "Master summary: ${MASTER_CSV}"
echo ""
echo "Results for best checkpoints:"
echo ""

for CONFIG in "${!BEST_CKPTS[@]}"; do
    CONFIG_CSV="${BASE_DIR}/${CONFIG}/eval_test_summary_mces.csv"
    if [[ -f "${CONFIG_CSV}" ]]; then
        BEST=$(tail -n +2 "${CONFIG_CSV}" | grep -v "NA" | head -n1)
        if [[ -n "${BEST}" ]]; then
            STEP=$(echo "${BEST}" | cut -d',' -f2)
            R1=$(echo "${BEST}" | cut -d',' -f3)
            R5=$(echo "${BEST}" | cut -d',' -f4)
            R20=$(echo "${BEST}" | cut -d',' -f5)
            MRR=$(echo "${BEST}" | cut -d',' -f6)
            MCES_MEAN=$(echo "${BEST}" | cut -d',' -f10)
            MCES_MEDIAN=$(echo "${BEST}" | cut -d',' -f11)
            echo "${CONFIG}:"
            echo "  Step: ${STEP}"
            echo "  R@1: ${R1}, R@5: ${R5}, R@20: ${R20}, MRR: ${MRR}"
            echo "  MCES@1_mean: ${MCES_MEAN}, MCES@1_median: ${MCES_MEDIAN}"
            echo ""
        fi
    fi
done

echo "=========================================="
echo "Test set evaluation complete!"
echo "=========================================="
