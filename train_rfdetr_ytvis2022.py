#!/usr/bin/env python3
"""
Finetune RF-DETR SegLarge on YTVIS 2022 (converted to COCO format).

Usage:
    python train_rfdetr_ytvis2022.py [--epochs 20] [--batch-size 4] [--devices 4]

Prerequisite:
    Run datasets/utils/convert_ytvis2022_to_coco.py first to create the
    COCO-format dataset at DATASET_DIR.
"""

import argparse
import os

# Prevent PyTorch Lightning from auto-detecting SLURM environment.
# This lets PTL use its own DDP subprocess launcher so we can run
# devices=4 from a single SLURM task (ntasks-per-node=1).
for _key in ("SLURM_NTASKS", "SLURM_JOB_NAME", "SLURM_NTASKS_PER_NODE"):
    os.environ.pop(_key, None)

from rfdetr import RFDETRSegLarge


DATASET_DIR = "/mnt/data3/jupiter/datasets/public_datasets/ytvis2022_coco"
OUTPUT_DIR = "/mnt/data2/jupiter/li.yu/exps/driveable_terrain_model/ytvis2022_rfdetr_sl_ft_0504"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=1.5e-5)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    model = RFDETRSegLarge()

    # Resume from last checkpoint if it exists and no explicit resume path given
    resume_path = args.resume
    if resume_path is None:
        last_ckpt = os.path.join(OUTPUT_DIR, "last.ckpt")
        if os.path.exists(last_ckpt):
            resume_path = last_ckpt
            print(f"Auto-resuming from {resume_path}")

    model.train(
        dataset_dir=DATASET_DIR,
        dataset_file="roboflow",
        output_dir=OUTPUT_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        devices=args.devices,
        strategy="ddp" if args.devices > 1 else "auto",
        num_workers=4,
        checkpoint_interval=5,
        eval_interval=999,  # effectively disable val eval (caused NCCL hang)
        compute_val_loss=False,
        use_ema=True,
        resume=resume_path,
    )


if __name__ == "__main__":
    main()
