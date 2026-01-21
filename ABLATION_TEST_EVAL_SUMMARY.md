# Ablation Study Test Set Evaluation Summary

## Overview
This document summarizes the process of finding the best checkpoints from validation results and setting up test set evaluation with MCES calculation.

## Best Checkpoints Identified (from Validation Set)

Based on validation results in `runs/ablation_study_spectraverse/ablation_summary_all.csv`, the best checkpoints (ranked by **R@5 / top-5 accuracy**) for each ablation configuration are:

| Configuration | Step | Checkpoint | Val R@1 | Val R@5 | Val R@20 | Val MRR |
|--------------|------|------------|---------|---------|----------|---------|
| 1_random_align | 1200 | `ckpt_001200.pt` | 0.22737 | 0.30603 | 0.40790 | 0.27064 |
| 2_random_contrastive | 7600 | `ckpt_007600.pt` | 0.30398 | 0.34662 | 0.43191 | 0.33320 |
| 3_pretrained_contrastive | 16200 | `ckpt_016200.pt` | 0.25765 | 0.35185 | 0.49068 | 0.31006 |
| 4_freeze_align | 20200 | `ckpt_020200.pt` | 0.29133 | **0.38965** | 0.52560 | 0.34468 |
| 5_freeze_contrastive | 16200 | `ckpt_016200.pt` | 0.25765 | 0.35185 | 0.49068 | 0.31006 |

**Note:** Checkpoints are selected based on highest R@5 (top-5 accuracy), with MRR as a tiebreaker.

## Test Set Evaluation Script

**Script:** `eval_ablation_test_mces.sh`

**Features:**
- Evaluates all 5 best checkpoints on the test set
- Calculates MCES (Maximum Common Edge Subgraph) distance for top-1 predictions
- Uses parallel execution across multiple GPUs
- Generates per-configuration and master summary CSV files

**Key Configuration:**
- Fold: `test`
- Candidates file: `data/candidates_test_val.pkl`
- MGF file: `data/spectraverse_clean.mgf`
- MCES calculation: Enabled via `--compute-mces` flag
- Embedding space: `chemberta`
- Batch size: 32

**Output Files:**
- Master summary: `runs/ablation_study_spectraverse/ablation_test_summary_mces.csv`
- Per-config summaries: `runs/ablation_study_spectraverse/{config}/eval_test_summary_mces.csv`
- Logs: `runs/ablation_study_spectraverse/{config}/eval_test_logs/eval_test_{step}.log`

**Metrics Reported:**
- R@1, R@5, R@20 (Recall at K)
- MRR (Mean Reciprocal Rank)
- Median rank
- Total queries and evaluated count
- **MCES@1_mean**: Mean MCES distance for top-1 predictions
- **MCES@1_median**: Median MCES distance for top-1 predictions

## Usage

```bash
cd /cluster/tufts/liulab/yiwan01/SpecBridge
./eval_ablation_test_mces.sh
```

**Environment Variables:**
- `NUM_GPUS`: Number of GPUs to use (default: 4)
- `BATCH_SIZE_JOBS`: Number of jobs to run simultaneously (default: 4)
- `CANDS`: Path to candidates file (default: `data/candidates_test_val.pkl`)

## Notes

1. The script uses the same evaluation pipeline as `eval_ablation_study.sh` but:
   - Targets the `test` fold instead of `val`
   - Includes MCES calculation
   - Only evaluates the best checkpoint per configuration

2. MCES calculation:
   - MCES distance = 0 for correct top-1 predictions
   - MCES distance > 0 for incorrect top-1 predictions (actual distance between true and predicted molecules)

3. Cache files:
   - Test set candidate embeddings are cached at: `cache/cands_test_chemberta_pub_v3g_spectraverse.pt`
   - Configs 3, 4, and 5 use cached embeddings (frozen/pre-trained mol encoder)

## Next Steps

After running the evaluation:
1. Check the master summary CSV for overall results
2. Review per-configuration logs for detailed metrics
3. Compare test set performance with validation set performance
4. Analyze MCES distances to understand structural similarity of incorrect predictions

