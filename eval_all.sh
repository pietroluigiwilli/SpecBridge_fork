#!/usr/bin/env bash
set -uo pipefail

# ========= Edit these to match your run =========
RUN_DIR="runs/specbridge_align_chemberta_pub_v3g_nist_contrafintune"            # folder with ckpt_*.pt
MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/nist23.mgf"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
# CANDS="/cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl"
CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/cand_dict_large_smiles.pkl"
CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym_retrieval_candidates_formula.json"
CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
CANDS="/cluster/tufts/liulab/yiwan01//SpecBridge/data/cand_dict_merged.pkl"
FOLD="test"
BATCH=16
LIMIT=100000

# Embedding space (pick ONE block)
MOL_SPACE="chemberta"                                 # chemberta | ecfp | adapter
COND_DIM=2048
MAPPER_HIDDEN=2048
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_nist23.pt"
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"
# CHEMBERTA_MODEL="Derify/ChemBERTa-druglike"
# CHEMBERTA_MODEL="laituan245/molt5-base"
# If you switch to ECFP:
# MOL_SPACE="ecfp"
# COND_DIM=1024
# MAPPER_HIDDEN=1024
# CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_test_ecfp_r2_2048.pt"
# ECFP_BITS=2048
# ECFP_RADIUS=2

# If you switch to adapter space:
# MOL_SPACE="adapter"
# COND_DIM=1024
# MAPPER_HIDDEN=1024
# CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_test_adapter.pt"

# ========= Output files =========
OUTCSV="${RUN_DIR}/eval_summary_${FOLD}_all.csv"
LOGDIR="${RUN_DIR}/eval_logs"
mkdir -p "${LOGDIR}"

# Write header if new
if [[ ! -f "${OUTCSV}" ]]; then
  echo "ckpt,step,R@1,R@5,R@20,MRR,median_rank,total_queries,evaluated" > "${OUTCSV}"
fi

# Build common args
COMMON=(
  --mgf "${MGF}"
  --dreams-ckpt "${DREAMS}"
  --candidates "${CANDS}"
  --fold-query "${FOLD}"
  --use-mapped --deterministic-map
  --batch-size "${BATCH}" 
  --cond-dim "${COND_DIM}" --mapper-hidden "${MAPPER_HIDDEN}"
  --mol-space "${MOL_SPACE}"
  # --cache-cand-emb "${CACHE}"
  --no-gaussian
)

# Add mol-space specific args
if [[ "${MOL_SPACE}" == "chemberta" ]]; then
  COMMON+=( --chemberta-model "${CHEMBERTA_MODEL}" )
elif [[ "${MOL_SPACE}" == "ecfp" ]]; then
  COMMON+=( --fp-bits "${ECFP_BITS:-2048}" --ecfp-radius "${ECFP_RADIUS:-2}" )
fi

# Loop checkpoints
shopt -s nullglob
for CK in "${RUN_DIR}"/*.pt; do
  STEP=$(basename "${CK}" | sed -E 's/ckpt_0*([0-9]+)\.pt/\1/')
  LOG="${LOGDIR}/eval_${STEP}.log"

  echo ">>> Evaluating ${CK}"
  # Skip if already present in CSV (idempotent)
  if grep -q ",${STEP}," "${OUTCSV}"; then
    echo "    (already summarized; skipping)"
    continue
  fi

  # Run eval and tee to log (don’t stop the whole sweep on error)
  set +e
  python -m specbridge.eval.candidates \
    --adapter-ckpt "${CK}" \
    "${COMMON[@]}" 2>&1 | tee "${LOG}"
  EC=$?
  set -e

  if [[ ${EC} -ne 0 ]]; then
    echo "    ERROR (${EC}) on ${CK}; see ${LOG}"
    # write a placeholder row so we know it failed
    echo "${CK},${STEP},NA,NA,NA,NA,NA,NA,NA" >> "${OUTCSV}"
    continue
  fi

  # Parse metrics from the log
  R1=$(awk '/R@1:/  {print $2}'  "${LOG}")
  R5=$(awk '/R@5:/  {print $2}'  "${LOG}")
  R20=$(awk '/R@20:/ {print $2}' "${LOG}")
  MRR=$(awk '/MRR:/  {print $2}' "${LOG}")
  MED=$(awk '/median_rank:/ {print $2}' "${LOG}")
  # parse the [eval] line
  # [eval] total_queries=1000 | evaluated=1000 | ...
  TOT=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /total_queries/) print $(i+1)}' "${LOG}")
  EVD=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /evaluated/) print $(i+1)}' "${LOG}")

  echo "${CK},${STEP},${R1},${R5},${R20},${MRR},${MED},${TOT},${EVD}" >> "${OUTCSV}"
done

echo "Summary written to: ${OUTCSV}"
echo "Top 5 by R@5:"
# pretty print top 5 by R@1
( head -n1 "${OUTCSV}"; tail -n +2 "${OUTCSV}" | sort -t',' -k4,4gr ) | head -n6 | column -s, -t
