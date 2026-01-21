# Optional Ablations for SpecBridge

This directory contains scripts and documentation for running optional ablation studies to validate design choices.

## Overview

Three types of ablations are supported:

1. **Mapper Capacity**: Varying `n_blocks` ∈ {0, 2, 4, 8} and `mapper_hidden` ∈ {512, 1024, 2048}
2. **Spectrum Unfreezing Depth**: Varying `unfreeze_last` ∈ {0, 1, 2, 4}
3. **Procrustes Warm Start**: Comparing orthogonal/Procrustes initialization vs random initialization

## Running Ablations

### 1. Training

Run all optional ablations:

```bash
bash run_optional_ablations.sh
```

This will:
- Launch all 12 mapper capacity configurations
- Launch all 4 unfreezing depth configurations  
- Launch 2 Procrustes warm start configurations (default + random init)
- Distribute jobs across available GPUs

Configuration:
- Set `NUM_GPUS` to control number of GPUs (default: 4)
- Set `BATCH_SIZE_JOBS` to control concurrent jobs per batch (default: 3)
- Set `BASE_OUTDIR` to change output directory (default: `runs/optional_ablations_spectraverse`)

### 2. Evaluation

After training completes, evaluate all checkpoints:

```bash
bash eval_optional_ablations.sh
```

This will:
- Find all checkpoints in each configuration directory
- Evaluate on validation set
- Generate summary CSV files

### 3. Analysis

Generate plots and summary tables:

```bash
python analyze_optional_ablations.py --base-dir runs/optional_ablations_spectraverse --fold val
```

This will:
- Create heatmap for mapper capacity ablation
- Create plots for unfreezing depth ablation
- Generate summary tables
- Save plots to `figs/` directory

## Output Structure

```
runs/optional_ablations_spectraverse/
├── mapper_n0_h512/
│   ├── ckpt_*.pt
│   ├── eval_summary_val_all.csv
│   └── eval_logs/
├── mapper_n0_h1024/
├── ...
├── unfreeze_0/
├── unfreeze_1/
├── unfreeze_2/
├── unfreeze_4/
├── procrustes_warmstart/
├── procrustes_random_init/
└── optional_ablations_summary_val.csv  # Master summary
```

## LaTeX Integration

The results can be integrated into the paper using `optional_ablations_section.tex`. After running evaluations and analysis:

1. Fill in the TODO placeholders in the LaTeX file with actual results
2. The analysis script generates plots that can be referenced
3. Tables can be generated from the CSV summaries

## Notes

- **Procrustes Warm Start**: The default uses orthogonal initialization. The `--random-mapper-init` flag enables random (Xavier uniform) initialization for comparison.
- **Mapper Capacity**: `n=0` corresponds to linear-only mapper (no residual blocks)
- **Unfreezing Depth**: `unfreeze_last=0` means the spectrum encoder is completely frozen


