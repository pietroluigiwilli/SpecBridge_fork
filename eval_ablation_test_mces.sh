#!/usr/bin/env bash
set -uo pipefail

# ========= Configuration =========
BASE_DIR="runs/ablation_study_spectraverse"
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
N_BLOCKS=8
COND_DIM=2048
MAPPER_HIDDEN=2048
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"

echo "=========================================="
echo "Ablation Study Test Set Evaluation with MCES"
echo "=========================================="
echo "Fold: ${FOLD}"
echo "Number of GPUs: $NUM_GPUS"
echo "Jobs per batch: $BATCH_SIZE_JOBS"
echo ""

# ========= Best checkpoints from validation results (ranked by R@5) =========
declare -A BEST_CKPTS=(
    ["1_random_align"]="runs/ablation_study_spectraverse/1_random_align/ckpt_001200.pt"
    ["2_random_contrastive"]="runs/ablation_study_spectraverse/2_random_contrastive/ckpt_007600.pt"
    ["3_pretrained_contrastive"]="runs/ablation_study_spectraverse/3_pretrained_contrastive/ckpt_018200.pt"
    ["4_freeze_align"]="runs/ablation_study_spectraverse/4_freeze_align/ckpt_020200.pt"
    ["5_freeze_contrastive"]="runs/ablation_study_spectraverse/5_freeze_contrastive/ckpt_016200.pt"
)

# ========= Ablation configurations =========
declare -a CONFIGS=(
    # "1_random_align"
    # "2_random_contrastive"
    "3_pretrained_contrastive"
    # "4_freeze_align"
    # "5_freeze_contrastive"
)

# ========= Master summary file =========
MASTER_CSV="${BASE_DIR}/ablation_test_summary_mces.csv"
echo "config,ckpt,step,R@1,R@5,R@20,MRR,median_rank,total_queries,evaluated,mces@1_mean,mces@1_median" > "${MASTER_CSV}"

# ========= Collect all evaluation tasks =========
declare -a eval_tasks=()

for CONFIG in "${CONFIGS[@]}"; do
    CKPT="${BEST_CKPTS[$CONFIG]}"
    
    if [[ ! -f "${CKPT}" ]]; then
        echo ">>> WARNING: Checkpoint not found: ${CKPT}"
        continue
    fi
    
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
    
    # Determine if mol encoder is frozen (configs 3, 4, 5 use frozen/pre-trained mol encoder)
    USE_CACHE=false
    if [[ "${CONFIG}" == "3_pretrained_contrastive" ]] || \
       [[ "${CONFIG}" == "4_freeze_align" ]] || \
       [[ "${CONFIG}" == "5_freeze_contrastive" ]]; then
        USE_CACHE=true
    fi
    
    # Add to task list: "CONFIG:CKPT_PATH:STEP:OUTCSV:LOGDIR:USE_CACHE"
    eval_tasks+=("${CONFIG}:${CKPT}:${STEP}:${OUTCSV}:${LOGDIR}:${USE_CACHE}")
done

echo ">>> Found ${#eval_tasks[@]} evaluation tasks"
echo ""

# ========= Function to run a single evaluation =========
run_eval() {
    local task=$1
    local gpu_id=$2
    
    IFS=':' read -r CONFIG CK STEP OUTCSV LOGDIR USE_CACHE <<< "${task}"
    local LOG="${LOGDIR}/eval_test_${STEP}.log"
    
    echo "[GPU $gpu_id] Evaluating ${CONFIG} - ${STEP} on test set"
    
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
    IFS=':' read -r CONFIG CK STEP OUTCSV LOGDIR USE_CACHE <<< "${task}"
    
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

for CONFIG in "${CONFIGS[@]}"; do
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
echo "Ablation Study Test Set Summary (with MCES)"
echo "=========================================="
echo "Master summary: ${MASTER_CSV}"
echo ""
echo "Results for best checkpoints:"
echo ""

for CONFIG in "${CONFIGS[@]}"; do
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

