# Results Evaluation: Fixed-Size Hardness Analysis

## Summary

After applying fixes to ensure ground truth is in candidate sets, the evaluation shows:

### Overall Performance
- **Recall@1**: 29.55%
- **Recall@5**: 47.39%
- **Recall@10**: 57.58%
- **Median Rank**: 6.0
- **Mean Rank**: 20.4

### Key Findings

#### 1. Ground Truth Fix Verification
- ✅ Verification shows **0 missing ground truth** in candidate sets (after capping)
- ✅ The fix successfully ensures ground truth is included during capping phase

#### 2. Performance by Hardness (Max Similarity)

| MaxSim Bin | N Queries | R@1    | R@5    | R@10   | Median Rank |
|------------|-----------|--------|--------|--------|-------------|
| <0.4       | 10,704    | 0.3998 | 0.5636 | 0.6177 | 3.0         |
| 0.4-0.5    | 3,859     | 0.3001 | 0.4504 | 0.5095 | 9.0         |
| 0.5-0.6    | 4,223     | 0.3370 | 0.5030 | 0.6334 | 4.0         |
| 0.6-0.7    | 4,272     | 0.2158 | 0.4007 | 0.5466 | 8.0         |
| 0.7-0.8    | 3,880     | 0.1905 | 0.4791 | 0.5729 | 6.0         |
| 0.8-0.9    | 2,462     | 0.1076 | 0.2555 | 0.5301 | 8.0         |
| 0.9-0.99   | 568       | 0.1215 | 0.1901 | 0.2447 | 41.0        |

**Key Observation**: Higher similarity (harder cases) generally shows lower performance, as expected.

#### 3. Deep Dive: <0.4 Similarity Bin (Easy Cases)

Despite all candidates having similarity < 0.4 to ground truth:
- **Recall@5**: 56.36% (still lower than expected)
- **Recall@1**: 39.98%
- **Median Rank**: 3.0 (good)
- **Mean Rank**: 19.5 (concerning - suggests many outliers)

**Bimodal Distribution**:
- ✅ **40.0%** get perfect predictions (rank = 0)
- ❌ **43.6%** fail to get rank ≤ 4 (Recall@5 failure)
- ❌ **14.1%** have rank > 50 (very poor performance)

This suggests the model either works very well OR very poorly, even for "easy" cases.

#### 4. Pool Size Effect
- **Small pools (< 100)**: Recall@5 = 67.82% (better)
- **Large pools (≥ 100)**: Recall@5 = 56.08% (worse)

Larger pools make the task harder, as expected.

### Issues Identified

#### 1. Queries with max_similarity ≥ 1.0
- **16,187 queries** (35% of total) have max_similarity ≥ 1.0
- These should have been deduplicated but weren't
- They are excluded from analysis
- **Action needed**: Fix deduplication to catch these cases

#### 2. Low Recall@5 for Easy Cases
Even when all candidates have similarity < 0.4:
- Only 56.36% Recall@5
- 43.6% of queries fail to rank ground truth in top 5
- 14.1% have rank > 50

**Possible reasons**:
1. **Model limitations**: Some queries are genuinely difficult even with different candidates
2. **Embedding quality**: Some molecules may have poor embeddings
3. **Spectrum quality**: Some query spectra may be noisy or ambiguous
4. **Evaluation issues**: Ground truth might still not be found during evaluation in some cases

### Recommendations

1. **Investigate poor performers**: Analyze the 14.1% of <0.4 queries with rank > 50
   - Check if ground truth is actually found during evaluation
   - Examine spectrum quality
   - Check embedding quality for these molecules

2. **Fix deduplication**: Address the 16,187 queries with max_similarity ≥ 1.0
   - These are likely duplicate molecules that should be removed
   - May require improving fingerprint-based deduplication

3. **Top-5 similarity analysis**: Complete the top-5 similarity computation
   - This will provide a more accurate hardness metric for Recall@5
   - Currently running in background

4. **Model improvement**: Consider model improvements for cases where even "easy" queries fail
   - May need better embedding methods
   - May need better spectrum processing

### Files Generated

- `figs/fixed_size_128_hardness_stratified.csv`: Binned results
- `figs/fixed_size_128_per_query_results.csv`: Per-query data
- `figs/fixed_size_128_hardness_stratified.pdf`: Visualization (R@1)
- `figs/fixed_size_128_hardness_stratified_r5.csv`: R@5 results
- `figs/fixed_size_128_hardness_stratified_r5.pdf`: R@5 visualization

### Next Steps

1. ✅ Ground truth fix applied and verified
2. ⏳ Complete top-5 similarity analysis (running in background)
3. 🔍 Investigate why 43.6% of easy cases fail
4. 🔧 Fix deduplication for max_similarity ≥ 1.0 cases
