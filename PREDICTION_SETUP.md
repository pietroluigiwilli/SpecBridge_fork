# SpecBridge Prediction Setup for all_spectra.mgf

## Summary

I've created a new script `specbridge/eval/predict_smiles.py` that outputs SMILES predictions for every spectrum in your MGF file.

## Answers to Your Questions

### 1. **Normalization: Is it already in the scripts?**

**YES**, normalization is already handled in the code:
- The embeddings are normalized using `F.normalize()` during similarity computation
- The candidate embeddings are normalized before computing cosine similarity
- The query embeddings are also normalized
- This is consistent with how `candidates.py` handles it

### 2. **Duplicate Identifiers: Combine or Keep One?**

The script handles duplicates with the `--handle-duplicates` flag:
- **`keep_first`** (default): Uses the first occurrence of each feature_id, ignores duplicates
- **`combine`**: Could combine candidates from duplicate feature_ids (currently just keeps first)

**Recommendation**: Since your candidates JSON already has one entry per feature_id, and duplicates in the MGF are likely the same spectrum measured multiple times, I recommend using `keep_first` to avoid processing the same spectrum multiple times.

## Usage

```bash
python -m specbridge.eval.predict_smiles \
    --mgf /cluster/tufts/liulab/yiwan01/SpecBridge/data/all_spectra.mgf \
    --dreams-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/data/ssl_model.ckpt \
    --adapter-ckpt /cluster/tufts/liulab/yiwan01/SpecBridge/runs/specbridge_align_chemberta_pub_v3g_spectraverse/ckpt_003800.pt \
    --candidates-json /cluster/tufts/liulab/yiwan01/SpecBridge/data/feature_id_to_candidates.json \
    --output predictions_spectraverse.json \
    --use-mapped \
    --deterministic-map \
    --cond-dim 2048 \
    --mapper-hidden 2048 \
    --mol-space chemberta \
    --chemberta-model Derify/ChemBERTa_augmented_pubchem_13m \
    --batch-size 32 \
    --top-k 1 \
    --handle-duplicates keep_first
```

## Output Format

The script outputs a JSON file with predictions:

```json
[
  {
    "feature_id": "0.00_126.9867m/z",
    "title": "0.00_126.9867m/z",
    "predicted_smiles": "CCO",
    "top_k_smiles": ["CCO"],
    "top_k_scores": [0.95],
    "num_candidates": 5,
    "status": "success"
  },
  ...
]
```

Or use `--output predictions.tsv` for TSV format.

## Key Differences from candidates.py

1. **Input format**: Uses `feature_id_to_candidates.json` (keyed by FEATURE_ID) instead of pickle (keyed by SMILES)
2. **Output**: Predicts SMILES for each spectrum instead of computing ranking metrics
3. **Duplicate handling**: Option to handle duplicate feature_ids in MGF
4. **No ground truth needed**: Doesn't require true SMILES in the MGF file

## Requirements

- MGF file with `FEATURE_ID` in headers
- Candidates JSON file: `{feature_id: [candidate_smiles_list]}`
- DreaMS checkpoint
- Trained adapter checkpoint

## Notes

- The script automatically handles normalization (L2 normalization for cosine similarity)
- Embeddings are computed using the model's `chem_proj` to ensure they're in the same space
- For inference, dummy SMILES are used in the forward pass (not used in prediction)
- The script processes spectra in batches for efficiency

