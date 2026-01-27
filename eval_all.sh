#!/usr/bin/env bash
set -uo pipefail

# ========= Edit these to match your run =========
RUN_DIR="runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec"            # folder with ckpt_*.pt
# MGF="data/MSnLib/combined_ms2_with_folds.mgf"  # MSnLib dataset
MGF="data/spectraverse_clean.mgf"  # Spectraverse dataset
DREAMS="data/ssl_model.ckpt"
# Candidate files (uncomment the one matching your MGF dataset):
# CANDS="../massspecgym/cand_dict_large_form.pkl"  # MassSpecGym (external)
CANDS="data/cand_dict_large_smiles.pkl"  # MSGYM dataset
# CANDS="data/MassSpecGym_retrieval_candidates_formula.json"  # MassSpecGym (JSON format)
CANDS="data/candidates_test_val.pkl"  # Spectraverse dataset
# CANDS="data/cand_dict_merged.pkl"  # Merged dataset
# CANDS="data/candidates_msnlib.pkl"  # MSnLib dataset
FOLD="val"
BATCH=128
LIMIT=100000

# Embedding space (pick ONE block)
MOL_SPACE="chemberta"                                 # chemberta | ecfp | adapter
N_BLOCKS=8
COND_DIM=2048
MAPPER_HIDDEN=2048
CACHE="cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"
# CHEMBERTA_MODEL="Derify/ChemBERTa-druglike"
# CHEMBERTA_MODEL="laituan245/molt5-base"
# If you switch to ECFP:
# MOL_SPACE="ecfp"
# COND_DIM=1024
# MAPPER_HIDDEN=1024
# CACHE="cache/cands_test_ecfp_r2_2048.pt"
# ECFP_BITS=2048
# ECFP_RADIUS=2

# If you switch to adapter space:
# MOL_SPACE="adapter"
# COND_DIM=1024
# MAPPER_HIDDEN=1024
# CACHE="cache/cands_test_adapter.pt"

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
  --cache-cand-emb "${CACHE}"
  --no-gaussian --n-blocks "${N_BLOCKS}"
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
  
  # Skip checkpoints before step 30000
#   if [[ ${STEP} -lt 30000 ]]; then
#     echo ">>> Skipping ${CK} (step ${STEP} < 30000)"
#     continue
#   fi
  
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
  R1=$(awk '/R@1:/  {print $2}'  "${LOG}" || echo "NA")
  R5=$(awk '/R@5:/  {print $2}'  "${LOG}" || echo "NA")
  R20=$(awk '/R@20:/ {print $2}' "${LOG}" || echo "NA")
  MRR=$(awk '/MRR:/  {print $2}' "${LOG}" || echo "NA")
  MED=$(awk '/median_rank:/ {print $2}' "${LOG}" || echo "NA")
  # parse the [eval] line
  # [eval] total_queries=1000 | evaluated=1000 | ...
  TOT=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /total_queries/) print $(i+1)}' "${LOG}" || echo "NA")
  EVD=$(awk -F'[=| ]+' '/^\[eval\]/ {for(i=1;i<=NF;i++) if($i ~ /evaluated/) print $(i+1)}' "${LOG}" || echo "NA")

  # Ensure all variables are set (default to NA if empty)
  R1="${R1:-NA}"
  R5="${R5:-NA}"
  R20="${R20:-NA}"
  MRR="${MRR:-NA}"
  MED="${MED:-NA}"
  TOT="${TOT:-NA}"
  EVD="${EVD:-NA}"

  echo "${CK},${STEP},${R1},${R5},${R20},${MRR},${MED},${TOT},${EVD}" >> "${OUTCSV}"
done

echo "Summary written to: ${OUTCSV}"
echo "Top 5 by R@5:"
# pretty print top 5 by R@5 (column 4)
# Note: The best checkpoint is typically the one with highest R@5 or MRR
( head -n1 "${OUTCSV}"; tail -n +2 "${OUTCSV}" | sort -t',' -k4,4gr ) | head -n6 | column -s, -t
echo ""
echo "To find the best checkpoint, check the CSV file and look for highest R@5 or MRR values."