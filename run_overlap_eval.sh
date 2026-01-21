#!/usr/bin/env bash
set -uo pipefail

# Script to run overlap-filtered evaluation on a specific checkpoint
# Only checks overlap with self-supervised pretrain dataset (GeMS), not fine-tuning data

# ========= Edit these to match your run =========
CKPT="/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_msgym_mapper_spec/ckpt_001200.pt"
MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
# CANDS="/cluster/tufts/liulab/yiwan01/massspecgym/cand_dict_large_form.pkl"
# CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/cand_dict_large_smiles.pkl"
# CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym_retrieval_candidates_formula.json"
CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
FOLD="test"

# GeMS pretrain data (self-supervised training dataset only)
GEMS_PATH="/cluster/tufts/liulab/yiwan01/SpecBridge/data/GeMS_A10.hdf5"
GEMS_URL="https://huggingface.co/datasets/roman-bushuiev/GeMS/resolve/main/data/GeMS_A/GeMS_A10.hdf5"

# Output directory
OUTPUT_DIR="/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec"
OUTPUT_CSV="${OUTPUT_DIR}/eval_overlap_filtered_ckpt_001200.csv"

# Evaluation parameters (matching eval_all.sh exactly)
BATCH=32
COND_DIM=2048
MAPPER_HIDDEN=2048
MOL_SPACE="chemberta"
N_BLOCKS=8
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"
CHEMBERTA_MODEL="Derify/ChemBERTa_augmented_pubchem_13m"

# Overlap check parameters (for GeMS pretrain data only)
NUM_BINS=20000
MAX_MZ=2000.0
PRETRAIN_BATCH_SIZE=50000

# Thresholds: 1.0 (no filter), 0.99, 0.97, 0.95, 0.9, 0.8, 0.7
THRESHOLDS="1.0,0.99,0.97,0.95,0.9,0.8,0.7"

echo "Running overlap-filtered evaluation..."
echo "Checkpoint: ${CKPT}"
echo "Output: ${OUTPUT_CSV}"
echo "Thresholds: ${THRESHOLDS}"
echo "Note: Only checking overlap with self-supervised pretrain dataset (GeMS), not fine-tuning data"

python eval_overlap_filtered.py \
    --ckpt "${CKPT}" \
    --msgym-mgf "${MGF}" \
    --gems-url "${GEMS_URL}" \
    --gems-path "${GEMS_PATH}" \
    --candidates "${CANDS}" \
    --dreams-ckpt "${DREAMS}" \
    --thresholds "${THRESHOLDS}" \
    --output-csv "${OUTPUT_CSV}" \
    --batch-size "${BATCH}" \
    --spec-bins 2048 \
    --fp-bits 2048 \
    --cond-dim "${COND_DIM}" \
    --mapper-hidden "${MAPPER_HIDDEN}" \
    --mol-space "${MOL_SPACE}" \
    --cache-cand-emb "${CACHE}" \
    --n-blocks "${N_BLOCKS}" \
    --chemberta-model "${CHEMBERTA_MODEL}" \
    --num-bins "${NUM_BINS}" \
    --max-mz "${MAX_MZ}" \
    --pretrain-batch-size "${PRETRAIN_BATCH_SIZE}"

echo "Evaluation complete. Results saved to: ${OUTPUT_CSV}"

