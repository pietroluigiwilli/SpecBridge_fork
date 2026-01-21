#!/bin/bash
# Monitor progress of top-5 similarity computation

echo "=== Top-5 Similarity Computation Progress ==="
echo ""

# Check if process is running
if pgrep -f "compute_top5_similarity.py" > /dev/null; then
    echo "✓ Process is running"
    ps aux | grep compute_top5_similarity | grep -v grep | awk '{print "  PID:", $2, "CPU:", $3"%", "Memory:", $4"%"}'
else
    echo "✗ Process is not running"
fi

echo ""

# Check log file
if [ -f compute_top5.log ]; then
    echo "=== Latest Log Output ==="
    tail -20 compute_top5.log
    echo ""
fi

# Check checkpoint file
if [ -f figs/top5_similarity_checkpoint.pkl ]; then
    echo "✓ Checkpoint file exists"
    ls -lh figs/top5_similarity_checkpoint.pkl
else
    echo "✗ No checkpoint file yet (still in capping phase)"
fi

echo ""
echo "To view full log: tail -f compute_top5.log"
echo "To check progress: python -c \"import pickle; d=pickle.load(open('figs/top5_similarity_checkpoint.pkl','rb')); print(f'Processed: {len(d.get(\"processed_indices\",[]))} queries')\" 2>/dev/null || echo 'Checkpoint not ready yet'"
