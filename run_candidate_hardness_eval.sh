#!/usr/bin/env bash
set -uo pipefail

# Evaluate candidate hardness by comparing Tanimoto similarity between
# ground truth and candidates for MassSpecGym vs Spectraverse/PubChem

# MassSpecGym candidate pool
MSGYM_CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/MassSpecGym_retrieval_candidates_formula.json"

# Spectraverse candidate pool (same as used in pool size evaluation)
SPECTRAVERSE_CANDS="/cluster/tufts/liulab/yiwan01/SpecBridge/data/candidates_test_val.pkl"

OUTPUT_DIR="figs"

# Optional: sample size to limit computation (set to None or remove to use all queries)
# For faster testing, you can set this to e.g., 1000
SAMPLE_SIZE=""

CUDA_VISIBLE_DEVICES=0 python eval_candidate_hardness.py \
    --msgym-candidates "${MSGYM_CANDS}" \
    --spectraverse-candidates "${SPECTRAVERSE_CANDS}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 1234 \
    ${SAMPLE_SIZE:+--sample-size "${SAMPLE_SIZE}"}

echo "Done! Results saved to ${OUTPUT_DIR}/"
echo "  - candidate_hardness_stats.json: Summary statistics"
echo "  - candidate_hardness_data.csv: Full data"
echo "  - candidate_hardness_histogram.pdf: Distribution comparison"
echo "  - candidate_hardness_boxplot.pdf: Box plot comparison"
