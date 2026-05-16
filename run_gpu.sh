#!/bin/bash
#SBATCH --job-name=dvis_training
#SBATCH --output=/home/li.yu/code/scripts/ytvis2022_coco_jupiter6khuman_dvis_m2f_swl_0515.txt
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


# activate virtual env
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate torch2100_mask2former  # for mask2former, rtdetr training

# # train R50 based DVIS online
# python train_net_video.py \
#   --num-gpus 8 \
#   --config-file ./configs/youtubevis_2022/DVIS_Online_R50.yaml \
#   MODEL.WEIGHTS ./pretrained_models/minvis_ytvis21_R50.pth \
#   OUTPUT_DIR /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_dvis_m2f_r50_0509

# # Finetune MinVIS segmenter on COCO + YTVIS 2022
# cd /home/li.yu/code/mymnt/DVIS
# python train_net_video.py \
#   --num-gpus 4 \
#   --config-file /home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/MinVIS_SwinL.yaml \
#   OUTPUT_DIR /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_m2f_swl_ft_0506

# train Swin-L based DVIS online using finetuned segmenter
python train_net_video.py \
  --num-gpus 4 \
  --config-file /home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/DVIS_Online_SwinL.yaml \
  SOLVER.IMS_PER_BATCH 4 \
  MODEL.WEIGHTS /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_m2f_swl_ft_0506/model_final.pth \
  OUTPUT_DIR /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_jupiter6khuman_dvis_m2f_swl_0515

# # Finetune RF-DETR on COCO + YTVIS 2022
# cd /home/li.yu/code/mymnt/DVIS
# python train_rfdetr_ytvis2022.py --epochs 20 --batch-size 4 --devices 4

# # train RF-DETR based DVIS online using finetuned segmenter
# python train_net_video.py \
#   --num-gpus 4 \
#   --config-file configs/youtubevis_2022/rfdetr/DVIS_Online_RFDETR.yaml \
#   MODEL.RFDETR.PRETRAIN_WEIGHTS /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_rfdetr_sl_ft_0506/rfdetr_finetuned.pt \
#   OUTPUT_DIR /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_dvis_rfdetr_sl_0508


# # convert brt train sequence dataset to ytvis format
# # Single split
# python convert_custom_to_ytvis.py \
#   --input-dir /mnt/data3/jupiter/datasets/sequence_data/Jupiter_train_rev2_frames_in_seqs_labeled/raw_seqs \
#   --output-dir /mnt/data3/jupiter/datasets/sequence_data/Jupiter_train_rev2_frames_in_seqs_labeled/ \
#   --split train \
#   --symlink

# # Train + valid splits (90/10)
# python convert_custom_to_ytvis.py \
#   --input-dir .../raw_seqs --output-dir .../output --split train_valid \
#   --train-ratio 0.9 --symlink


# # convert brt test sequence dataset to ytvis format
# python convert_custom_to_ytvis.py \
#   --input-dir /mnt/data3/jupiter/datasets/sequence_data/humans-on_path_forward-day-sequences-core-jupiter_rev11_8338/raw_seqs \
#   --output-dir /mnt/data3/jupiter/datasets/sequence_data/humans-on_path_forward-day-sequences-core-jupiter_rev11_8338 \
#   --split test \
#   --symlink


# deactivate virtual env
conda deactivate
conda deactivate

# leave working directory
cd /home/li.yu/code/scripts
