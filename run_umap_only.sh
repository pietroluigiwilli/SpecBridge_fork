#!/usr/bin/env bash
set -uo pipefail

# Simple script to compute UMAP by computing embeddings directly
# Follows the same parameters as run_umap_analysis.sh

# Default paths (same as run_umap_analysis.sh)
MGF="/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf"
ADAPTER_CKPT="/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/ckpt_020600.pt"
DREAMS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt"
FOLD="test"
OUTPUT_DIR="figs"

# Run the script - computes embeddings directly, then UMAP
python compute_umap.py \
    --mgf "${MGF}" \
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
    --use-mapped \
    --plot-all

echo ""
echo "Done! UMAP visualizations saved to ${OUTPUT_DIR}/figs/"
echo "  - umap_mapped.{png,pdf}"
echo "  - umap_mol.{png,pdf}"
echo "  - umap_mapped_vs_mol_combined.{png,pdf}"
