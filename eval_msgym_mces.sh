#!/usr/bin/env bash
set -uo pipefail

# Script to evaluate a specific checkpoint on MassSpecGym with MCES computation

# ========= Configuration =========
CKPT="/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/ckpt_020600.pt"
MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
FOLD="test"
BATCH=32
LIMIT=100000

# Embedding space configuration (matching training config)
MOL_SPACE="chemberta"
N_BLOCKS=8
COND_DIM=2048
MAPPER_HIDDEN=2048
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"

# ========= Output files =========
RUN_DIR="runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec"
LOGDIR="${RUN_DIR}/eval_logs"
mkdir -p "${LOGDIR}"

STEP=$(basename "${CKPT}" | sed -E 's/ckpt_0*([0-9]+)\.pt/\1/')
LOG="${LOGDIR}/eval_${STEP}_mces.log"

echo ">>> Evaluating ${CKPT} on SpectraVerse with MCES computation"
echo ">>> Log file: ${LOG}"

# Build common args
COMMON=(
  --mgf "${MGF}"
  --dreams-ckpt "${DREAMS}"
  --adapter-ckpt "${CKPT}"
  --candidates "${CANDS}"
  --fold-query "${FOLD}"
  --use-mapped --deterministic-map
  --batch-size "${BATCH}" 
  --cond-dim "${COND_DIM}" --mapper-hidden "${MAPPER_HIDDEN}"
  --mol-space "${MOL_SPACE}"
  --cache-cand-emb "${CACHE}"
  --no-gaussian --n-blocks "${N_BLOCKS}"
  --chemberta-model "${CHEMBERTA_MODEL}"
  --compute-mces
)

# Run eval and tee to log
set +e
python -m specbridge.eval.candidates \
  "${COMMON[@]}" 2>&1 | tee "${LOG}"
EC=$?
set -e

if [[ ${EC} -ne 0 ]]; then
  echo "    ERROR (${EC}) on ${CKPT}; see ${LOG}"
  exit ${EC}
fi

echo ""
echo "Evaluation complete. Results logged to: ${LOG}"
echo "MCES metrics should be visible in the log above."


