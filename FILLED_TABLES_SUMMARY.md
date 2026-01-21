# Filled Tables for SpecBridge Paper

## Table 1: Hyperparameters

Replace the TODO entries with:

```latex
Alignment weight & $5.0$ \\
Contrastive weights & $w_{\text{con}}=1.0$, $w_{\text{con\_mapped}}=1.0$ (defaults; standard alignment runs use $w_{\text{con}}=0$, $w_{\text{con\_mapped}}=0$) \\
Temperature (contrastive) & $0.07$ \\
```

**Sources:**
- Alignment weight: `dreams_condition_adapter.py:811` (default=5.0)
- Contrastive weights: `dreams_condition_adapter.py:810,814` (defaults w_con=1.0, w_con_mapped=1.0)
- Temperature: `mapper.py:74` and `dreams_condition_adapter.py:809` (default=0.07)

## Table 2: Parameter Counts

### Confirmed Values:

```latex
Molecule encoder (ChemBERTa) & 43,961,088 & 0 \\
Spectrum projection head & 6,299,648 & 6,299,648 \\
Mapper $g_\theta$ & 26,774,272 & 26,774,272 \\
```

### Values Needing Verification:

```latex
Spectrum encoder (DreaMS total) & \textcolor{red}{TBD} & \textcolor{red}{TBD (last-2 blocks only)} \\
\midrule
\textbf{Total} & \textcolor{red}{TBD} & \textcolor{red}{TBD} \\
```

**Note:** To get exact DreaMS parameter counts, you need to:
1. Load the checkpoint with proper environment (requires `msml` and `dreams` modules)
2. Count total parameters in the DreaMS model
3. Count parameters in the last 2 transformer layers only

**Estimated values** (based on architecture: 7 layers, d_model=1024):
- Total DreaMS: ~30,000,000 parameters
- Last 2 layers: ~8,500,000 parameters
- Total trainable: ~41,000,000 parameters

**Calculation method:**
- Run the `count_parameters.py` script in an environment with all dependencies installed
- Or manually load the checkpoint and count parameters using PyTorch


