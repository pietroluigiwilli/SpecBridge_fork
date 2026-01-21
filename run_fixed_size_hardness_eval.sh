#!/usr/bin/env bash
set -uo pipefail

# Disentangle Pool Size from Chemical Hardness
# Analyze retrieval performance at fixed pool size (N=128) stratified by max similarity

MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
ADAPTER_CKPT="/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/ckpt_020600.pt"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
FOLD="test"
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"
OUTPUT_DIR="figs"
POOL_CAP=128

python eval_fixed_size_hardness.py \
    --mgf "${MGF}" \
    --candidates "${CANDS}" \
    --adapter-ckpt "${ADAPTER_CKPT}" \
    --dreams-ckpt "${DREAMS}" \
    --fold-query "${FOLD}" \
    --cond-dim 2048 \
    --mapper-hidden 2048 \
    --n-blocks 8 \
    --batch-size 64 \
    --mol-space chemberta \
    --chemberta-model "Derify/ChemBERTa_augmented_pubchem_13m" \
    --spec-bins 2048 \
    --cache-cand-emb "${CACHE}" \
    --seed 1234 \
    --pool-cap "${POOL_CAP}" \
    --output-dir "${OUTPUT_DIR}"

echo "Done! Results saved to ${OUTPUT_DIR}/"
echo "  - fixed_size_${POOL_CAP}_hardness_stratified.csv: Binned results"
echo "  - fixed_size_${POOL_CAP}_per_query_results.csv: Per-query data"
echo "  - fixed_size_${POOL_CAP}_hardness_stratified.pdf: Visualization"
