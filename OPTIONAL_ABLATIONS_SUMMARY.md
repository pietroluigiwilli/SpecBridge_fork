# Optional Ablations - Implementation Summary

## Files Created

### 1. Training Script
- **`run_optional_ablations.sh`**: Main script to run all optional ablation experiments
  - Mapper capacity: 12 configurations (n ∈ {0,2,4,8} × h ∈ {512,1024,2048})
  - Unfreezing depth: 4 configurations (unfreeze_last ∈ {0,1,2,4})
  - Procrustes warm start: 2 configurations (default + random init)
  - Parallel execution across multiple GPUs

### 2. Evaluation Script
- **`eval_optional_ablations.sh`**: Evaluates all trained checkpoints
  - Finds all checkpoints in each configuration directory
  - Runs evaluation on validation set
  - Generates summary CSV files
  - Parallel execution for efficiency

### 3. Analysis Script
- **`analyze_optional_ablations.py`**: Analyzes results and generates plots
  - Creates heatmap for mapper capacity ablation
  - Creates plots for unfreezing depth ablation
  - Generates summary tables
  - Saves plots to `figs/` directory

### 4. LaTeX Section
- **`optional_ablations_section.tex`**: Complete LaTeX section ready for paper
  - Includes all three ablation studies
  - Tables and figure placeholders
  - Compute and efficiency section
  - TODO markers for results to be filled in

### 5. Documentation
- **`README_optional_ablations.md`**: User guide for running ablations

## Code Modifications

### 1. Mapper Initialization Support
- **`specbridge/models/mapper.py`**: Added `random_init` parameter to `ProcrustesResidualMapper`
  - Default: orthogonal/Procrustes initialization (existing behavior)
  - Random init: Xavier uniform initialization (for ablation)

### 2. Training Script Support
- **`dreams_condition_adapter.py`**: Added `--random-mapper-init` flag
  - Enables random initialization for Procrustes warm start ablation

## Usage

### Step 1: Run Training
```bash
bash run_optional_ablations.sh
```

### Step 2: Run Evaluation
```bash
bash eval_optional_ablations.sh
```

### Step 3: Analyze Results
```bash
python analyze_optional_ablations.py --base-dir runs/optional_ablations_spectraverse --fold val
```

### Step 4: Fill in LaTeX
- Open `optional_ablations_section.tex`
- Replace TODO placeholders with actual results from analysis
- Figures will be generated in `figs/` directory

## Ablation Configurations

### Mapper Capacity (12 configs)
- n_blocks: 0, 2, 4, 8
- mapper_hidden: 512, 1024, 2048
- All other settings fixed (default alignment loss, last-2 unfrozen)

### Spectrum Unfreezing Depth (4 configs)
- unfreeze_last: 0 (frozen), 1, 2 (default), 4
- All other settings fixed (default mapper, alignment loss)

### Procrustes Warm Start (2 configs)
- Default: Orthogonal/Procrustes initialization
- Random: Xavier uniform initialization
- All other settings fixed (default mapper, alignment loss, last-2 unfrozen)

## Output Files

After running all steps:
- Training checkpoints: `runs/optional_ablations_spectraverse/*/ckpt_*.pt`
- Evaluation summaries: `runs/optional_ablations_spectraverse/*/eval_summary_val_all.csv`
- Master summary: `runs/optional_ablations_spectraverse/optional_ablations_summary_val.csv`
- Plots: `figs/mapper_capacity_ablation.pdf`, `figs/unfreezing_depth_ablation.pdf`

## Next Steps

1. Run the training script (may take several hours/days depending on GPU availability)
2. Run evaluation script after training completes
3. Run analysis script to generate plots
4. Fill in TODO placeholders in `optional_ablations_section.tex` with actual results
5. Add the LaTeX section to your paper


