#!/bin/bash

# Evaluate DVIS Online Swin-L on YTVIS 2022 validation set

EXP_DIR="/mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_dvis_m2f_swl_0514"
OUTPUT_DIR="$EXP_DIR"
MODEL_PATH="$EXP_DIR/model_final.pth"
CONFIG_FILE="/home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/DVIS_Online_SwinL.yaml"
DATASET_DIR="/home/li.yu/code/mymnt/DVIS/datasets/ytvis_2022/valid"

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
      --num-gpus 1 \
      --config-file "$CONFIG_FILE" \
      --eval-only \
      MODEL.WEIGHTS "$MODEL_PATH" \
      DATASETS.TEST '("ytvis_2022_val",)' \
      OUTPUT_DIR "$OUTPUT_DIR"
fi

# Evaluation against YTVIS 2022 val GT
GT_DIR="/home/li.yu/code/mymnt/DVIS/datasets/ytvis_2022"
ln -sf "$OUTPUT_DIR/inference/results.json" "$GT_DIR/results.json"
mkdir -p "$OUTPUT_DIR/eval_results"
python "$GT_DIR/evaluate.py" "$GT_DIR" "$OUTPUT_DIR/eval_results"
rm -f "$GT_DIR/results.json"
echo ""
echo "=== Evaluation Results ==="
cat "$OUTPUT_DIR/eval_results/scores.txt"

# # Visualize predictions as videos with instance masks, class names, and IDs
# echo ""
# echo "=== Generating Visualization Videos ==="
# cd /home/li.yu/code/mymnt/DVIS
# python visualize_predictions.py \
#   --results-json "$OUTPUT_DIR/inference/results.json" \
#   --annotations "$DATASET_DIR/instances.json" \
#   --images-dir "$DATASET_DIR/JPEGImages" \
#   --output-dir "$OUTPUT_DIR/vis_videos" \
#   --score-thr 0.3 \
#   --fps 3
