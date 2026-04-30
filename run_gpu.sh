#!/bin/bash
#SBATCH --job-name=gpu_task
#SBATCH --output=/home/li.yu/code/scripts/ytvis2022_rfdetr_l_0429.txt
#SBATCH --partition=gen4,gen5,sxm5,gen3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-gpu=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=5-00:00:00
# #SBATCH --exclude=amxnl007
# #SBATCH --nodelist=stc01sppamxnl021
# #SBATCH --exclude=stc01spplmdanl006,stc01sppamxnl016
# #SBATCH --exclude=stc01sppamxnl[001-008]


# activate virtual env
eval "$(/home/li.yu/anaconda3/bin/conda shell.bash hook)"
conda activate torch2100_mask2former  # for mask2former, rtdetr training

# run DVIS
cd /home/li.yu/code/mymnt/DVIS
# python train_net_video.py \
#   --num-gpus 4 \
#   --config-file /home/li.yu/code/mymnt/DVIS/configs/youtubevis_2022/swin/DVIS_Online_SwinL.yaml \
#   --resume MODEL.WEIGHTS /home/li.yu/code/mymnt/DVIS/pretrained_models/minvis_ytvis21_swin_large.pth

python train_net_video.py \
  --num-gpus 4 \
  --config-file configs/youtubevis_2022/rfdetr/DVIS_Online_RFDETR.yaml \
  SOLVER.IMS_PER_BATCH 64

# deactivate virtual env
conda deactivate
conda deactivate

# leave working directory
cd /home/li.yu/code/scripts
