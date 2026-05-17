#!/bin/bash
#SBATCH --job-name=eval_dvis
#SBATCH --output=/home/li.yu/code/scripts/eval_dvis.txt
#SBATCH --partition=gen4,gen5,sxm5,gen3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=10-00:00:00
# #SBATCH --exclude=amxnl007
# #SBATCH --nodelist=stc01sppamxnl021
# #SBATCH --exclude=stc01spplmdanl006,stc01sppamxnl016
# #SBATCH --exclude=stc01sppamxnl[001-008]

# Evaluate DVIS model on Jupiter humans-on-path test set

EXP_DIR="/mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_jupiter6khuman_dvis_m2f_swl_0515"
OUTPUT_DIR="$EXP_DIR/eval_brt"
MODEL_PATH="$EXP_DIR/model_final.pth"
# OUTPUT_DIR="/home/li.yu/code/mymnt/DVIS/output_Downloaded_DVIS_Online_SwinL_YTVIS21_on_jupiter_humans"
# MODEL_PATH="pretrained_models/DVIS_online_ytvis21_swinl.pth"
CONFIG_FILE="/home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/DVIS_Online_SwinL.yaml"
DATASET_DIR="/home/li.yu/code/mymnt/DVIS/datasets/jupiter_humans/test"

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
      MODEL.WEIGHTS "$MODEL_PATH" \
      DATASETS.TEST '("jupiter_humans_test",)' \
      OUTPUT_DIR "$OUTPUT_DIR"
fi

# Evaluation against Jupiter humans test GT
GT_DIR="$DATASET_DIR"
ln -sf "$OUTPUT_DIR/inference/results.json" "$GT_DIR/results.json"
mkdir -p "$OUTPUT_DIR/eval_results"
python /home/li.yu/code/mymnt/DVIS/evaluate_jupiter_humans.py "$GT_DIR" "$OUTPUT_DIR/eval_results"
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
#   --annotations "$GT_DIR/instances.json" \
#   --images-dir "$GT_DIR/JPEGImages" \
#   --output-dir "$OUTPUT_DIR/vis_videos" \
#   --score-thr 0.3 \
#   --fps 3
