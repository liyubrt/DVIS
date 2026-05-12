#!/usr/bin/env python3
"""
Convert a PyTorch Lightning RF-DETR checkpoint (.ckpt) to the RF-DETR .pt
format expected by RFDETRSegmenter._load_pretrain_weights.

Lightning keys:  model.backbone.0.xxx, model.transformer.xxx, ...
RF-DETR keys:    backbone.0.xxx, transformer.xxx, ...

Usage:
    python convert_rfdetr_ckpt.py \
        --input /path/to/last.ckpt \
        --output /path/to/rfdetr_finetuned.pt
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="Path to Lightning .ckpt file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output .pt file")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    ckpt = torch.load(args.input, map_location="cpu")

    src_sd = ckpt["state_dict"]
    print(f"Source state_dict: {len(src_sd)} keys")
    if "epoch" in ckpt:
        print(f"Epoch: {ckpt['epoch']}")

    # Strip 'model.' prefix
    new_sd = {}
    for k, v in src_sd.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
        else:
            new_sd[k] = v

    print(f"Converted state_dict: {len(new_sd)} keys")
    torch.save({"model": new_sd}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
