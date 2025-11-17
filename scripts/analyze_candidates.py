#!/usr/bin/env python3
"""
Analyze candidate distribution in a pickle file.
Handles pickle protocol 5 by using pickle5 or Python 3.8+.
"""
import sys
import pickle
from collections import Counter

# Try to import numpy, but work without it if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def load_pickle_safe(filepath):
    """Load pickle file, handling protocol 5."""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except ValueError as e:
        if "unsupported pickle protocol: 5" in str(e):
            # Try using pickle5 if available
            try:
                import pickle5
                with open(filepath, 'rb') as f:
                    return pickle5.load(f)
            except ImportError:
                print("ERROR: Pickle file uses protocol 5, which requires Python 3.8+ or pickle5 module.")
                print(f"Current Python version: {sys.version}")
                print("\nTo fix this, you can:")
                print("1. Use Python 3.8+ to run this script")
                print("2. Install pickle5: pip install pickle5")
                print("3. Re-save the pickle file with a lower protocol")
                sys.exit(1)
        raise

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_candidates.py <candidates.pkl>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    print(f"Loading candidates from: {filepath}")
    
    try:
        cand_map = load_pickle_safe(filepath)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Get candidate counts
    cand_counts = [len(cands) for cands in cand_map.values() if cands]
    
    print(f'\n=== Summary ===')
    print(f'Total entries: {len(cand_map)}')
    print(f'Entries with candidates: {len(cand_counts)}')
    print(f'Entries with 0 candidates: {sum(1 for c in cand_map.values() if not c)}')
    
    if not cand_counts:
        print("\nNo candidates found in the file!")
        return
    
    print(f'\n=== Statistics ===')
    print(f'Min candidates: {min(cand_counts)}')
    print(f'Max candidates: {max(cand_counts)}')
    
    if HAS_NUMPY:
        print(f'Mean candidates: {np.mean(cand_counts):.2f}')
        print(f'Median candidates: {np.median(cand_counts):.2f}')
        print(f'Std candidates: {np.std(cand_counts):.2f}')
        
        # Percentiles
        print(f'\n=== Percentiles ===')
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(cand_counts, p)
            print(f'{p:>3}th percentile: {val:>8.2f}')
    else:
        # Compute basic stats without numpy
        mean = sum(cand_counts) / len(cand_counts)
        sorted_counts = sorted(cand_counts)
        n = len(sorted_counts)
        median = sorted_counts[n//2] if n % 2 == 1 else (sorted_counts[n//2-1] + sorted_counts[n//2]) / 2
        variance = sum((x - mean) ** 2 for x in cand_counts) / len(cand_counts)
        std = variance ** 0.5
        print(f'Mean candidates: {mean:.2f}')
        print(f'Median candidates: {median:.2f}')
        print(f'Std candidates: {std:.2f}')
        
        # Percentiles without numpy
        print(f'\n=== Percentiles ===')
        for p in [10, 25, 50, 75, 90, 95, 99]:
            idx = int(p / 100 * (n - 1))
            val = sorted_counts[idx]
            print(f'{p:>3}th percentile: {val:>8.2f}')
    
    # Distribution bins
    print(f'\n=== Distribution by bins ===')
    bins = [
        (0, 1), (1, 5), (5, 10), (10, 20), (20, 50), 
        (50, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))
    ]
    for low, high in bins:
        if high == float('inf'):
            count = sum(1 for c in cand_counts if c >= low)
            label = f'{low}+'
        else:
            count = sum(1 for c in cand_counts if low <= c < high)
            label = f'{low}-{high-1}'
        pct = 100 * count / len(cand_counts)
        print(f'  {label:>10s}: {count:>6d} ({pct:>5.1f}%)')
    
    # Show some examples
    print(f'\n=== Examples ===')
    sorted_items = sorted(cand_map.items(), key=lambda x: len(x[1]) if x[1] else 0, reverse=True)
    print(f'Top 5 entries with most candidates:')
    for i, (smi, cands) in enumerate(sorted_items[:5]):
        smi_display = smi[:60] + "..." if len(smi) > 60 else smi
        print(f'  {i+1}. {smi_display:>63s} : {len(cands):>6d} candidates')
    
    print(f'\n=== Additional Stats ===')
    print(f'Entries with exactly 1 candidate: {sum(1 for c in cand_counts if c == 1)}')
    print(f'Entries with >50 candidates: {sum(1 for c in cand_counts if c > 50)}')
    print(f'Entries with >100 candidates: {sum(1 for c in cand_counts if c > 100)}')
    print(f'Entries with >200 candidates: {sum(1 for c in cand_counts if c > 200)}')
    print(f'Entries with >500 candidates: {sum(1 for c in cand_counts if c > 500)}')
    print(f'Entries with >1000 candidates: {sum(1 for c in cand_counts if c > 1000)}')
    
    # Histogram data for potential plotting
    print(f'\n=== Raw counts (first 20) ===')
    count_freq = Counter(cand_counts)
    for count, freq in sorted(count_freq.items())[:20]:
        print(f'  {count:>6d} candidates: {freq:>6d} entries')

if __name__ == "__main__":
    main()

