#!/bin/bash

# Evaluate DVIS Online Swin-L on YTVIS 2022 validation set

GT_DIR="/home/li.yu/code/mymnt/DVIS/datasets/ytvis_2022"
OUTPUT_DIR="/home/li.yu/code/mymnt/DVIS/output_DVIS_Online_SwinL_YTVIS22"
CONFIG_FILE="/home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/DVIS_Online_SwinL.yaml"

mkdir -p "$OUTPUT_DIR"

# Activate the conda environment used for training
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate torch2100_mask2former

# Check if inference results already exist
if [ -f "$OUTPUT_DIR/inference/results.json" ]; then
    echo "Inference results already exist, skipping inference step."
else
    echo "Running inference..."
    cd /home/li.yu/code/mymnt/DVIS
    python train_net_video.py \
      --num-gpus 4 \
      --config-file "$CONFIG_FILE" \
      --eval-only \
      MODEL.WEIGHTS "$OUTPUT_DIR/model_final.pth" \
      OUTPUT_DIR "$OUTPUT_DIR"
fi

# Symlink results.json into gt directory (evaluate.py expects both in same dir)
ln -sf "$OUTPUT_DIR/inference/results.json" "$GT_DIR/results.json"

# Create eval_results directory
mkdir -p "$OUTPUT_DIR/eval_results"

# Run evaluation
python "$GT_DIR/evaluate.py" "$GT_DIR" "$OUTPUT_DIR/eval_results"

# Clean up symlink
rm -f "$GT_DIR/results.json"

# Print results
echo ""
echo "=== Evaluation Results ==="
cat "$OUTPUT_DIR/eval_results/scores.txt"
