#!/usr/bin/env bash
set -uo pipefail

# Generate UMAP visualizations for mapped and molecular embeddings
# Following parameters from run_candidate_pool_size_eval.sh

MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"
ADAPTER_CKPT="/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/ckpt_020600.pt"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
FOLD="test"
CACHE="/cluster/tufts/liulab/yiwan01/SpecBridge/cache/cands_${FOLD}_chemberta_pub_v3g_spectraverse.pt"
OUTPUT_DIR="figs"

python specbridge_analysis.py \
    --mgf "${MGF}" \
    --candidates "${CANDS}" \
    --adapter-ckpt "${ADAPTER_CKPT}" \
    --dreams-ckpt "${DREAMS}" \
    --fold-query "${FOLD}" \
    --cond-dim 2048 \
    --mapper-hidden 2048 \
    --n-blocks 8 \
    --no-gaussian \
    --batch-size 64 \
    --mol-space chemberta \
    --chemberta-model "Derify/ChemBERTa_augmented_pubchem_13m" \
    --spec-bins 2048 \
    --seed 1234 \
    --outdir "${OUTPUT_DIR}" \
    --umap \
    --use-mapped \
    --cache-cand-emb "${CACHE}"

echo "Done! UMAP visualizations saved to ${OUTPUT_DIR}/figs/"
echo "  - umap_mapped.{png,pdf}"
echo "  - umap_mol.{png,pdf}"
echo "  - umap_mapped_vs_mol_combined.{png,pdf}"

