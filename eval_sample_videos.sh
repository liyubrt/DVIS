#!/bin/bash
#SBATCH --job-name=eval_dvis
#SBATCH --output=/home/li.yu/code/scripts/eval_sample_videos_%j.txt
#SBATCH --partition=gen4,gen5,sxm5,gen3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=10-00:00:00

# Run DVIS inference on sample videos

VIDEO_DIR="/home/li.yu/code/mymnt/Wan2.2/sample_videos_snowboard"
DATASET_DIR="/home/li.yu/code/mymnt/DVIS/datasets/sample_videos_snowboard"

# downloaded model
MODEL_PATH=pretrained_models/DVIS_offline_ytvis21_swinl.pth
OUTPUT_DIR=/home/li.yu/code/mymnt/DVIS/output_Downloaded_DVIS_Offline_SwinL_YTVIS21/sample_videos_snowboard

# # trained model
# EXP_DIR="/mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_dvis_rfdetr_sl_0519"
# MODEL_PATH="$EXP_DIR/model_final.pth"
# OUTPUT_DIR="$EXP_DIR"

CONFIG_FILE="/home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/DVIS_Online_SwinL.yaml"


mkdir -p "$OUTPUT_DIR"

# Activate conda
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate torch2100_mask2former

# Step 1: Extract frames and create instances.json
echo "=== Preparing dataset ==="
cd /home/li.yu/code/mymnt/DVIS
python prepare_videos_for_inference.py \
  --input-dir "$VIDEO_DIR" \
  --output-dir "$DATASET_DIR" \
  --fps 0

# Step 2: Run DVIS inference
if [ -f "$OUTPUT_DIR/inference/results.json" ]; then
    echo "Inference results already exist, skipping."
else
    echo ""
    echo "=== Running DVIS inference ==="
    python train_net_video.py \
      --num-gpus 4 \
      --config-file "$CONFIG_FILE" \
      --eval-only \
      MODEL.WEIGHTS "$MODEL_PATH" \
      DATASETS.TEST '("sample_videos_snowboard",)' \
      SOLVER.IMS_PER_BATCH 4 \
      OUTPUT_DIR "$OUTPUT_DIR"
fi

# Step 3: Visualize predictions
echo ""
echo "=== Generating Visualization Videos ==="
python visualize_predictions.py \
  --results-json "$OUTPUT_DIR/inference/results.json" \
  --annotations "$DATASET_DIR/instances.json" \
  --images-dir "$DATASET_DIR/JPEGImages" \
  --output-dir "$OUTPUT_DIR/vis_videos" \
  --score-thr 0.3 \
  --fps 16

echo ""
echo "Done. Visualization videos saved to $OUTPUT_DIR/vis_videos"
