#!/bin/bash

# Evaluate DVIS Online R50 on YTVIS 2022 validation set

GT_DIR="/home/li.yu/code/mymnt/DVIS/datasets/ytvis_2022"
RES_FILE="/home/li.yu/code/mymnt/DVIS/output_DVIS_Online_R50_YTVIS22/inference/results.json"
OUTPUT_DIR="/home/li.yu/code/mymnt/DVIS/output_DVIS_Online_R50_YTVIS22/eval_results"

mkdir -p "$OUTPUT_DIR"

# Symlink results.json into gt directory (evaluate.py expects both in same dir)
ln -sf "$RES_FILE" "$GT_DIR/results.json"

# Activate the conda environment used for training
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate torch2100_mask2former

# Run evaluation
python "$GT_DIR/evaluate.py" "$GT_DIR" "$OUTPUT_DIR"

# Clean up symlink
rm -f "$GT_DIR/results.json"

# Print results
echo ""
echo "=== Evaluation Results ==="
cat "$OUTPUT_DIR/scores.txt"
