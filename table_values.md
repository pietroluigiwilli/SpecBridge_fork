# SpecBridge Table Values

## Hyperparameter Table

Based on code analysis:

- **Alignment weight** (`w_map`): Default is `5.0` (from `dreams_condition_adapter.py` line 811)
- **Contrastive weights**: 
  - `w_con_mapped`: Default is `1.0` (from `dreams_condition_adapter.py` line 814)
  - `w_con`: Default is `1.0` (from `dreams_condition_adapter.py` line 810)
- **Temperature (contrastive)**: Default is `0.07` (from `mapper.py` line 74, and `dreams_condition_adapter.py` line 809)

## Parameter Count Table

Based on model architecture and calculations:

### Component Breakdown:

1. **Molecule encoder (ChemBERTa)**: 
   - Total: 43,961,088 parameters
   - Trainable: 0 (fully frozen)

2. **Spectrum encoder (DreaMS total)**:
   - Architecture: 7 transformer layers, d_model=1024, n_heads=8
   - Total: ~30,000,000 parameters (estimated, needs verification from actual checkpoint)
   - Trainable: Last 2 layers only (~8,500,000 parameters, needs verification)

3. **Spectrum projection head**:
   - Architecture: Linear(1024, 2048) -> GELU -> Linear(2048, 2048) -> LayerNorm(2048)
   - Total: 6,299,648 parameters
   - Trainable: 6,299,648 parameters (all trainable)

4. **Mapper $g_\theta$**:
   - Architecture: ProcrustesResidualMapper with n_blocks=8, hidden=2048
   - Input: 2048, Output: 768 (ChemBERTa hidden size)
   - Total: 26,774,272 parameters
   - Trainable: 26,774,272 parameters (all trainable)

### Notes:
- DreaMS parameter counts need to be verified by loading the actual checkpoint
- The exact count for last-2 layers of DreaMS depends on the specific layer architecture
- Total trainable parameters: ~41-42M (depending on DreaMS last-2 layers count)


