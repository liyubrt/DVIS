#!/bin/bash
#SBATCH --job-name=gpu_task
#SBATCH --output=/home/li.yu/code/scripts/ytvis2022_coco_dvis_m2f_r50_0509.txt
#SBATCH --partition=gen4,gen5,sxm5,gen3
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=10-00:00:00
# #SBATCH --exclude=amxnl007
# #SBATCH --nodelist=stc01sppamxnl021
# #SBATCH --exclude=stc01spplmdanl006,stc01sppamxnl016
# #SBATCH --exclude=stc01sppamxnl[001-008]


# activate virtual env
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate torch221_mask2former  # on lambda, for mask2former, rtdetr training
# conda activate torch2100_mask2former  # on harvset, for mask2former, rtdetr training

<<<<<<< HEAD:run_gpu_local.sh
# run DVIS
cd /home/li.yu/code/DVIS

# use R50
python train_net_video.py \
  --num-gpus 2 \
  --config-file /home/li.yu/code/DVIS/configs/youtubevis_2022/DVIS_Online_R50.yaml \
  --resume MODEL.WEIGHTS /home/li.yu/code/DVIS/pretrained_models/minvis_ytvis21_r50.pth

# # use swin large
=======
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

# # train Swin-L based DVIS online using finetuned segmenter
>>>>>>> 52a4a6a70252fd9adfa5c9d7bd845b1fb2625b0a:run_gpu.sh
# python train_net_video.py \
#   --num-gpus 4 \
#   --config-file /home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/DVIS_Online_SwinL.yaml \
#   MODEL.WEIGHTS /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_m2f_swl_ft_0506/model_final.pth \
#   OUTPUT_DIR /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_dvis_m2f_swl_0508

# # Finetune RF-DETR on COCO + YTVIS 2022
# cd /home/li.yu/code/mymnt/DVIS
# python train_rfdetr_ytvis2022.py --epochs 20 --batch-size 4 --devices 4

# # train RF-DETR based DVIS online using finetuned segmenter
# python train_net_video.py \
#   --num-gpus 4 \
#   --config-file configs/youtubevis_2022/rfdetr/DVIS_Online_RFDETR.yaml \
#   MODEL.RFDETR.PRETRAIN_WEIGHTS /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_rfdetr_sl_ft_0506/rfdetr_finetuned.pt \
#   OUTPUT_DIR /mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_coco_dvis_rfdetr_sl_0508


# convert brt sequence dataset to ytvis format
python convert_custom_to_ytvis.py \
  --input-dir /mnt/data3/jupiter/datasets/sequence_data/humans-on_path_forward-day-sequences-core-jupiter_rev11_8338/raw_seqs \
  --output-dir /mnt/data3/jupiter/datasets/sequence_data/humans-on_path_forward-day-sequences-core-jupiter_rev11_8338 \
  --split test \
  --symlink

<<<<<<< HEAD:run_gpu_local.sh
# # use rfdetr
# python train_net_video.py \
#   --num-gpus 4 \
#   --config-file configs/youtubevis_2022/rfdetr/DVIS_Online_RFDETR.yaml \
#   SOLVER.IMS_PER_BATCH 64
=======
>>>>>>> 52a4a6a70252fd9adfa5c9d7bd845b1fb2625b0a:run_gpu.sh

# deactivate virtual env
conda deactivate
conda deactivate

# leave working directory
cd /home/li.yu/code/scripts
