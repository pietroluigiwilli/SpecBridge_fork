# SpecBridge Table Values - Final

## Table 1: Hyperparameters

### Values to fill in:

- **Alignment weight** $w_{\text{map}}$: **5.0** (confirmed from `dreams_condition_adapter.py:811`)
- **Contrastive weights**: 
  - Defaults: $w_{\text{con}}=1.0$, $w_{\text{con\_mapped}}=1.0$ (from `dreams_condition_adapter.py:810,814`)
  - Standard alignment runs use: $w_{\text{con}}=0$, $w_{\text{con\_mapped}}=0$ (from README.md examples)
  - For contrastive runs: $w_{\text{con\_mapped}}$ can be 0.005 or other values
- **Temperature (contrastive)**: **0.07** (confirmed from `mapper.py:74` and `dreams_condition_adapter.py:809`)

### LaTeX entry for alignment weight:
```
Alignment weight & $5.0$ \\
```

### LaTeX entry for contrastive weights:
```
Contrastive weights & $w_{\text{con}}=1.0$, $w_{\text{con\_mapped}}=1.0$ (defaults; standard alignment runs use $w_{\text{con}}=0$, $w_{\text{con\_mapped}}=0$) \\
```

### LaTeX entry for temperature:
```
Temperature (contrastive) & $0.07$ \\
```

## Table 2: Parameter Counts

### Values confirmed from code:

1. **Molecule encoder (ChemBERTa)**: 
   - Total: **43,961,088**
   - Trainable: **0**

2. **Spectrum projection head**:
   - Total: **6,299,648**
   - Trainable: **6,299,648**

3. **Mapper $g_\theta$**:
   - Total: **26,774,272**
   - Trainable: **26,774,272**

### Values that need checkpoint loading:

4. **Spectrum encoder (DreaMS total)**:
   - Architecture: 7 transformer layers, d_model=1024, n_heads=8
   - Estimated total: ~30,000,000 (needs verification from checkpoint)
   - Estimated trainable (last-2 layers): ~8,500,000 (needs verification)

### LaTeX entries:

```
Molecule encoder (ChemBERTa) & 43,961,088 & 0 \\
Spectrum encoder (DreaMS total) & \textcolor{red}{TBD} & \textcolor{red}{TBD (last-2 blocks only)} \\
Spectrum projection head & 6,299,648 & 6,299,648 \\
Mapper $g_\theta$ & 26,774,272 & 26,774,272 \\
\midrule
\textbf{Total} & \textcolor{red}{TBD} & \textcolor{red}{TBD} \\
```

### Note:
To get exact DreaMS parameter counts, the checkpoint needs to be loaded with proper dependencies (`msml`, `dreams` modules). The estimated values are based on architecture:
- 7 transformer layers with d_model=1024
- Each layer ~4.2M parameters
- Last 2 layers ~8.4M parameters


