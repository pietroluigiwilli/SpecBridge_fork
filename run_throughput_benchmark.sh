#!/usr/bin/env bash
set -uo pipefail

# Benchmark script to validate inference throughput numbers for SpecBridge
# Expected results:
# - Encoding time: 3.2 ms per spectrum (batched)
# - Retrieval time: <0.1 ms per query
# - Total throughput: ~300 queries per second

MGF="${MGF:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/spectraverse_clean.mgf}"
DREAMS="${DREAMS:-/cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt}"
ADAPTER_CKPT="${ADAPTER_CKPT:-/cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse_mapper_spec/ckpt_020600.pt}"

# Default parameters (can be overridden via environment variables)
BATCH_SIZE="${BATCH_SIZE:-128}"
GALLERY_SIZE="${GALLERY_SIZE:-1500}"  # Average pool size from Spectraverse test set
NUM_QUERIES="${NUM_QUERIES:-1000}"
COND_DIM="${COND_DIM:-2048}"
MAPPER_HIDDEN="${MAPPER_HIDDEN:-2048}"
FOLD="${FOLD:-test}"

echo "=========================================="
echo "SpecBridge Inference Throughput Benchmark"
echo "=========================================="
echo "MGF file: ${MGF}"
echo "DreaMS checkpoint: ${DREAMS}"
echo "Adapter checkpoint: ${ADAPTER_CKPT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gallery size: ${GALLERY_SIZE}"
echo "Number of queries: ${NUM_QUERIES}"
echo "Condition dimension: ${COND_DIM}"
echo "Mapper hidden: ${MAPPER_HIDDEN}"
echo "Fold: ${FOLD}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=2 python benchmark_inference_throughput.py \
    --mgf "${MGF}" \
    --dreams-ckpt "${DREAMS}" \
    --adapter-ckpt "${ADAPTER_CKPT}" \
    --spec-bins 2048 \
    --fp-bits 2048 \
    --cond-dim "${COND_DIM}" \
    --mapper-hidden "${MAPPER_HIDDEN}" \
    --batch-size "${BATCH_SIZE}" \
    --gallery-size "${GALLERY_SIZE}" \
    --num-queries "${NUM_QUERIES}" \
    --fold-query "${FOLD}" \
    --seed 1234 \
    "$@"

echo ""
echo "Benchmark completed!"
