# Ground Truth in Candidate Sets - Issue and Fix

## Problem Identified

When analyzing queries with max_similarity < 0.4, we found that Recall@5 is only 56.36%, which is unexpectedly low. If all candidates have similarity < 0.4 to the ground truth, and the ground truth itself (similarity = 1.0) is in the candidate set, the model should easily distinguish it and achieve much higher Recall@5.

## Root Cause

Investigation revealed that:
1. **Ground truth matching issues**: The ground truth SMILES from the dataset may not exactly match the keys in the candidate map due to:
   - Whitespace differences
   - Canonicalization differences
   - String representation differences

2. **Lookup failures**: When `cand_map_capped.get(gt_smi, [])` returns an empty list, the code skips the query entirely (line 684-685), so the ground truth is never added to the candidate list.

3. **Inconsistent canonicalization**: Different parts of the code use different canonicalization methods, leading to mismatches.

## Fixes Applied

### 1. Improved Candidate Map Lookup (lines 679-710)
- Try multiple lookup keys: `gt_smi`, `gt_smiles`, `gt_smiles.strip()`
- If direct lookup fails, search through all keys to find a match
- Use flexible matching with canonicalization

### 2. Enhanced Ground Truth Matching (lines 745-763)
- Try multiple matching strategies when finding the ground truth in the `kept` list
- Check for exact matches, canonicalized matches, and stripped matches
- Provide better error logging if ground truth is still not found

## Expected Impact

After these fixes:
- More queries should have their ground truth properly included in candidate sets
- Recall@5 for <0.4 similarity queries should increase significantly (potentially from 56% to 80%+)
- Overall evaluation results should be more accurate

## Next Steps

**To apply the fix, you need to re-run the evaluation:**
```bash
bash run_fixed_size_hardness_eval.sh
```

This will:
1. Re-cap candidate pools with the improved logic
2. Re-compute max similarities
3. Re-run evaluation with improved ground truth matching
4. Generate new results with corrected Recall@5 values

## Verification

After re-running, you can verify the fix by:
1. Checking that Recall@5 for <0.4 similarity queries is much higher
2. Running `investigate_low_sim_queries.py` to confirm ground truth is found in candidate sets
3. Comparing old vs new results
