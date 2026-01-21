# Inference Throughput Benchmark

This document describes the experiment design to validate the inference throughput numbers reported in the SpecBridge paper.

## Reported Numbers

From the paper:
- **Encoding time:** 3.2 ms per spectrum (batched)
- **Retrieval time:** <0.1 ms per query (dot product against pre-computed FAISS index)
- **Total throughput:** ~300 queries per second

## Experiment Design

The benchmark script (`benchmark_inference_throughput.py`) measures three key metrics:

### 1. Encoding Time Benchmark

**Purpose:** Measure the time to encode a batch of spectra into embeddings.

**Method:**
- Load test spectra from Spectraverse test set
- Encode batches of spectra using the trained SpecBridge model
- Measure wall-clock time for each batch (with GPU synchronization)
- Report per-spectrum encoding time (batch time / batch size)

**Parameters:**
- Batch size: 256 (default, can be adjusted)
- Number of warmup batches: 3
- Number of trial batches: 10
- Device: GPU (CUDA) if available, else CPU

**Output:**
- Mean, median, std, min, max encoding time per spectrum
- Throughput in spectra/second

### 2. Retrieval Time Benchmark

**Purpose:** Measure the time to retrieve candidates from a pre-computed gallery.

**Method:**
- Pre-compute gallery embeddings (simulating FAISS index)
- For each query embedding:
  - Compute dot product similarity with all gallery embeddings
  - Find top-k candidates
- Measure wall-clock time per query (with GPU synchronization)

**Parameters:**
- Gallery size: 1500 (average pool size from Spectraverse test set)
- Number of warmup queries: 10
- Number of trial queries: 1000
- Top-k: 10

**Output:**
- Mean, median, std, min, max retrieval time per query
- Throughput in queries/second

**Note:** The benchmark uses dot product similarity, which is what FAISS IndexFlatIP uses internally. The reported <0.1 ms includes the similarity computation and top-k selection.

### 3. End-to-End Throughput Benchmark

**Purpose:** Measure total throughput including both encoding and retrieval.

**Method:**
- Pre-compute gallery embeddings (candidate pool)
- For each query:
  1. Encode spectrum to embedding (batched)
  2. Compute similarity with gallery
  3. Retrieve top-k candidates
- Measure total time per query

**Parameters:**
- Gallery size: 1500
- Number of queries: 1000
- Batch size: 256

**Output:**
- Total time per query (encoding + retrieval)
- Breakdown: encoding time vs retrieval time
- Total throughput in queries/second

## Running the Benchmark

### Quick Start

```bash
./run_throughput_benchmark.sh
```

### Custom Parameters

```bash
# Set environment variables
export MGF="/path/to/spectraverse_clean.mgf"
export ADAPTER_CKPT="/path/to/checkpoint.pt"
export BATCH_SIZE=256
export GALLERY_SIZE=1500
export NUM_QUERIES=1000

./run_throughput_benchmark.sh
```

### Direct Python Call

```bash
python benchmark_inference_throughput.py \
    --mgf /path/to/spectraverse_clean.mgf \
    --dreams-ckpt /path/to/ssl_model.ckpt \
    --adapter-ckpt /path/to/checkpoint.pt \
    --batch-size 256 \
    --gallery-size 1500 \
    --num-queries 1000 \
    --fold-query test \
    --cond-dim 512
```

### Options

- `--skip-encoding`: Skip encoding benchmark (only run retrieval and end-to-end)
- `--skip-retrieval`: Skip retrieval benchmark
- `--skip-end-to-end`: Skip end-to-end benchmark
- `--cpu`: Force CPU usage (default: use GPU if available)

## Expected Results

The benchmark will output:

1. **Encoding benchmark results:**
   - Per-spectrum encoding time should be close to 3.2 ms
   - Ratio to expected value is reported

2. **Retrieval benchmark results:**
   - Per-query retrieval time should be <0.1 ms
   - PASS/FAIL status is reported

3. **End-to-end throughput:**
   - Total throughput should be ~300 queries/second
   - Ratio to expected value is reported
   - Breakdown of encoding vs retrieval time

## Validation Criteria

The numbers are considered validated if:
- Encoding time per spectrum ≈ 3.2 ms (within reasonable variance, e.g., ±20%)
- Retrieval time per query < 0.1 ms
- Total throughput ≈ 300 queries/second (within reasonable variance, e.g., ±20%)

## Notes

1. **GPU vs CPU:** The benchmark should be run on GPU for accurate results, as the reported numbers are for GPU inference.

2. **Batch size:** The encoding time is measured with batching (default batch_size=256). Single-spectrum encoding would be slower due to overhead.

3. **FAISS vs Dot Product:** The benchmark uses direct dot product computation rather than FAISS. FAISS would be slightly faster for very large galleries (>10k), but for ~1500 candidates, the difference is negligible.

4. **Warmup:** The benchmark includes warmup runs to ensure consistent timing (avoiding cold start effects).

5. **Synchronization:** GPU operations are synchronized before timing to ensure accurate measurements.

## Troubleshooting

- **Out of memory:** Reduce `--batch-size` or `--num-queries`
- **Slow results:** Ensure GPU is being used (check device output)
- **No data:** Verify MGF file path and fold name are correct
