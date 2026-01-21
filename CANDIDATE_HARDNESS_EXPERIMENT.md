# Candidate Hardness Analysis Experiment

## Overview

This experiment analyzes the **intrinsic hardness** of candidate pools by computing the maximum Tanimoto similarity between ground truth molecules and their candidate sets. This quantifies how difficult it is to distinguish the correct answer from the decoys.

## Hypothesis

- **MassSpecGym candidates**: Likely contain structurally distinct decoys (different scaffolds, possibly different formulas)
- **Spectraverse/PubChem candidates**: Retrieved by exact molecular formula, meaning **every candidate is an isomer** (constitutional or stereoisomer)

We expect Spectraverse/PubChem candidates to have **higher maximum Tanimoto similarity** to the ground truth, indicating they are harder to distinguish.

## Method

For each query:
1. Compute ECFP4 fingerprints for ground truth and all candidates
2. Compute Tanimoto similarity between ground truth and each candidate
3. Record multiple metrics:
   - **Maximum similarity**: The single hardest decoy (excluding ground truth if present)
   - **Mean Top-K similarity**: Average of top-K most similar candidates (default K=10)
     - This measures the **density of the hard neighborhood**, not just the single hardest decoy
   - **Number of hard candidates**: Count of candidates with Tanimoto > threshold (default 0.8)
     - This is the **effective hard candidate count** per query
   - **Pool size**: Total number of valid candidates
4. Aggregate statistics across all queries

### Important Note: Pool Size Bias

**Larger pools are statistically more likely to contain high-similarity decoys by chance.** This is a fundamental statistical effect: if you sample more molecules, you're more likely to find one that happens to be similar.

However, the observed differences between MassSpecGym and Spectraverse are **much larger** than what would be expected from pool size alone. The combination of:
- Higher max similarity
- Higher mean top-K similarity (dense hard neighborhood)
- Many more hard candidates per query

...indicates that Spectraverse/PubChem pools are **intrinsically harder** due to isomerism, not just larger pool sizes.

## Usage

### Run the experiment:

```bash
./run_candidate_hardness_eval.sh
```

### Or run directly:

```bash
python eval_candidate_hardness.py \
    --msgym-candidates data/MassSpecGym_retrieval_candidates_formula.json \
    --spectraverse-candidates data/candidates_test_val.pkl \
    --output-dir figs \
    --seed 1234
```

### Optional arguments:

- `--sample-size N`: Sample N queries per dataset (for faster computation during testing)
- `--fp-radius R`: Morgan fingerprint radius (default: 2 for ECFP4)
- `--fp-nbits B`: Number of fingerprint bits (default: 4096)
- `--hard-threshold T`: Tanimoto threshold for "hard" candidates (default: 0.8)
- `--top-k K`: Number of top candidates to average for mean top-K similarity (default: 10)

## Outputs

The experiment generates:

1. **`candidate_hardness_stats.json`**: Summary statistics for all metrics (mean, median, std, percentiles)
2. **`candidate_hardness_data.csv`**: Full data for all queries including:
   - Max Tanimoto similarity
   - Mean Top-K Tanimoto similarity
   - Number of hard candidates (Tanimoto > threshold)
   - Pool size
3. **`candidate_hardness_histogram.pdf`**: Distribution of max similarity (histogram)
4. **`candidate_hardness_boxplot.pdf`**: Distribution of max similarity (box plot)
5. **`candidate_hardness_topk_boxplot.pdf`**: Distribution of mean top-K similarity (box plot)
6. **`candidate_hardness_nhard_boxplot.pdf`**: Distribution of number of hard candidates (box plot, log scale)

## Expected Results

If the hypothesis is correct, we should see:
- **Higher mean/median max Tanimoto similarity** for Spectraverse compared to MassSpecGym
- **Higher mean top-K similarity** for Spectraverse (dense hard neighborhood)
- **Many more hard candidates per query** for Spectraverse (e.g., 50 vs 1-2 for MassSpecGym)
- This confirms that PubChem candidates are **intrinsically harder** (more similar to ground truth = more isomers)

### Example Expected Values

- **MassSpecGym**: 
  - Mean max Tanimoto: ~0.3-0.5
  - Mean top-10 Tanimoto: ~0.2-0.4
  - Mean hard candidates (Tanimoto > 0.8): ~1-2
  
- **Spectraverse**:
  - Mean max Tanimoto: ~0.7-0.9
  - Mean top-10 Tanimoto: ~0.6-0.8
  - Mean hard candidates (Tanimoto > 0.8): ~20-50

## Interpretation

### Max Tanimoto Similarity
- **Low** (< 0.3): Candidates are structurally distinct, easier to distinguish
- **Medium** (0.3-0.7): Some structural similarity, moderate difficulty
- **High** (> 0.7): Candidates are very similar (isomers), harder to distinguish
- **Very high** (> 0.9): Candidates are nearly identical (stereoisomers or constitutional isomers with minor differences)

### Mean Top-K Similarity
- Measures the **density of the hard neighborhood**
- High values indicate many similar candidates, not just one outlier
- More robust than max similarity alone (less sensitive to single outliers)

### Number of Hard Candidates
- The **effective hard candidate count** per query
- MassSpecGym: Typically 1-2 hard candidates (if any)
- Spectraverse: Often 20-50+ hard candidates
- This is a **powerful metric** to quote in the paper: "Spectraverse queries have an average of X hard candidates (Tanimoto > 0.8) compared to Y for MassSpecGym"

## Integration with Paper

This analysis supports the claim in Section 4.3 that:

> "The drop in performance [on Spectraverse] reflects the fundamental difficulty of resolving isomerism using only MS/MS spectra and 2D molecular graphs, rather than a failure of the model to scale."

The similarity analysis provides quantitative evidence that PubChem-derived candidate pools are harder than standard benchmark pools.
