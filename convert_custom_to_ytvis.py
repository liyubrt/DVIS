"""
Convert custom humans-on-path sequence dataset to YTVIS 2022 JSON format.

Each sequence subfolder becomes one video. All pixels with value 11 in the
mask are treated as a single "humans" instance per video.

Usage:
    python convert_custom_to_ytvis.py \
        --input-dir /mnt/data3/jupiter/datasets/sequence_data/humans-on_path_forward-day-sequences-core-jupiter_rev11_8338/raw_seqs \
        --output-dir /path/to/output \
        --split train
"""

import argparse
import csv
import json
import os
import random
import shutil

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# Full YTVIS 2021/2022 categories (40 classes). "person" is id=26.
YTVIS_CATEGORIES = [
    {"id": 1, "name": "airplane", "supercategory": "vehicle"},
    {"id": 2, "name": "bear", "supercategory": "animal"},
    {"id": 3, "name": "bird", "supercategory": "animal"},
    {"id": 4, "name": "boat", "supercategory": "vehicle"},
    {"id": 5, "name": "car", "supercategory": "vehicle"},
    {"id": 6, "name": "cat", "supercategory": "animal"},
    {"id": 7, "name": "cow", "supercategory": "animal"},
    {"id": 8, "name": "deer", "supercategory": "animal"},
    {"id": 9, "name": "dog", "supercategory": "animal"},
    {"id": 10, "name": "duck", "supercategory": "animal"},
    {"id": 11, "name": "earless_seal", "supercategory": "animal"},
    {"id": 12, "name": "elephant", "supercategory": "animal"},
    {"id": 13, "name": "fish", "supercategory": "animal"},
    {"id": 14, "name": "flying_disc", "supercategory": "object"},
    {"id": 15, "name": "fox", "supercategory": "animal"},
    {"id": 16, "name": "frog", "supercategory": "animal"},
    {"id": 17, "name": "giant_panda", "supercategory": "animal"},
    {"id": 18, "name": "giraffe", "supercategory": "animal"},
    {"id": 19, "name": "horse", "supercategory": "animal"},
    {"id": 20, "name": "leopard", "supercategory": "animal"},
    {"id": 21, "name": "lizard", "supercategory": "animal"},
    {"id": 22, "name": "monkey", "supercategory": "animal"},
    {"id": 23, "name": "motorbike", "supercategory": "vehicle"},
    {"id": 24, "name": "mouse", "supercategory": "animal"},
    {"id": 25, "name": "parrot", "supercategory": "animal"},
    {"id": 26, "name": "person", "supercategory": "person"},
    {"id": 27, "name": "rabbit", "supercategory": "animal"},
    {"id": 28, "name": "shark", "supercategory": "animal"},
    {"id": 29, "name": "skateboard", "supercategory": "object"},
    {"id": 30, "name": "snake", "supercategory": "animal"},
    {"id": 31, "name": "snowboard", "supercategory": "object"},
    {"id": 32, "name": "squirrel", "supercategory": "animal"},
    {"id": 33, "name": "surfboard", "supercategory": "object"},
    {"id": 34, "name": "tennis_racket", "supercategory": "object"},
    {"id": 35, "name": "tiger", "supercategory": "animal"},
    {"id": 36, "name": "train", "supercategory": "vehicle"},
    {"id": 37, "name": "truck", "supercategory": "vehicle"},
    {"id": 38, "name": "turtle", "supercategory": "animal"},
    {"id": 39, "name": "whale", "supercategory": "animal"},
    {"id": 40, "name": "zebra", "supercategory": "animal"},
]

# Map label mask pixel value to YTVIS category id
LABEL_TO_YTVIS_CAT = {11: 26}  # pixel value 11 (humans) -> YTVIS category 26 (person)


def encode_mask(mask_arr, label_val=11):
    """Encode binary mask (label_val pixels) to COCO uncompressed RLE."""
    binary = (mask_arr == label_val).astype(np.uint8, order="F")
    if binary.sum() == 0:
        return None, None, None
    rle = mask_utils.encode(binary)
    # Convert bytes counts to string for JSON serialization
    rle["counts"] = rle["counts"].decode("utf-8")
    area = int(mask_utils.area(rle))
    bbox = mask_utils.toBbox(rle).tolist()  # [x, y, w, h]
    return rle, area, bbox


def process_sequence(seq_dir):
    """Read data.csv and return list of (unique_id, height, width) sorted by unique_id."""
    csv_path = os.path.join(seq_dir, "data.csv")
    if not os.path.exists(csv_path):
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    frames = []
    for r in rows:
        uid = r["unique_id"]
        h = int(r["rectified_stereo_output_height"])
        w = int(r["rectified_stereo_output_width"])
        frames.append((uid, h, w))
    return frames


def convert_sequences(seq_names, input_dir, split_dir, symlink=False):
    """Convert a list of sequences to YTVIS format and save to split_dir."""
    images_dir = os.path.join(split_dir, "JPEGImages")
    os.makedirs(images_dir, exist_ok=True)

    videos = []
    annotations = []
    video_id = 0
    ann_id = 0

    for seq_name in seq_names:
        seq_dir = os.path.join(input_dir, seq_name)
        frames = process_sequence(seq_dir)
        if not frames:
            print(f"  [SKIP] {seq_name}: no data.csv or empty")
            continue

        video_id += 1
        height, width = frames[0][1], frames[0][2]
        file_names = []
        segmentations = []
        areas = []
        bboxes = []
        has_any_mask = False

        # Create video image directory
        vid_img_dir = os.path.join(images_dir, seq_name)
        os.makedirs(vid_img_dir, exist_ok=True)

        for uid, h, w in frames:
            # Copy/symlink image
            src_img = os.path.join(seq_dir, uid + ".png")
            dst_img = os.path.join(vid_img_dir, uid + ".png")
            if not os.path.exists(dst_img):
                if symlink:
                    os.symlink(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)

            file_names.append(os.path.join(seq_name, uid + ".png"))

            # Encode mask
            mask_path = os.path.join(seq_dir, uid + "_label.png")
            if os.path.exists(mask_path):
                mask_arr = np.array(Image.open(mask_path))
                rle, area, bbox = encode_mask(mask_arr, label_val=11)  # pixel value 11 = humans
                if rle is not None:
                    has_any_mask = True
                    segmentations.append(rle)
                    areas.append(area)
                    bboxes.append(bbox)
                else:
                    segmentations.append(None)
                    areas.append(None)
                    bboxes.append(None)
            else:
                segmentations.append(None)
                areas.append(None)
                bboxes.append(None)

        videos.append({
            "id": video_id,
            "width": width,
            "height": height,
            "length": len(frames),
            "file_names": file_names,
        })

        if has_any_mask:
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "video_id": video_id,
                "category_id": LABEL_TO_YTVIS_CAT[11],  # 26 = person
                "segmentations": segmentations,
                "areas": areas,
                "bboxes": bboxes,
                "iscrowd": 0,
            })

        if video_id % 100 == 0:
            print(f"  Processed {video_id}/{len(seq_names)} sequences")

    dataset = {
        "info": {
            "description": "Custom humans-on-path dataset in YTVIS format",
            "version": "1.0",
        },
        "licenses": [],
        "videos": videos,
        "categories": YTVIS_CATEGORIES,
        "annotations": annotations,
    }

    out_json = os.path.join(split_dir, "instances.json")
    with open(out_json, "w") as f:
        json.dump(dataset, f)

    print(f"  Videos: {len(videos)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Saved to: {out_json}")
    print(f"  Images in: {images_dir}")
    return len(videos), len(annotations)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Path to raw_seqs directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for YTVIS-format dataset")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test", "train_valid"],
                        help="Split to generate. Use 'train_valid' to generate both train and valid splits.")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Fraction of sequences for training (only used with --split train_valid)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/valid split")
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying images")
    args = parser.parse_args()

    seqs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print(f"Found {len(seqs)} sequences")

    if args.split == "train_valid":
        random.seed(args.seed)
        shuffled = seqs[:]
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * args.train_ratio)
        train_seqs = sorted(shuffled[:n_train])
        valid_seqs = sorted(shuffled[n_train:])

        print(f"\nSplitting: {len(train_seqs)} train, {len(valid_seqs)} valid "
              f"(ratio={args.train_ratio}, seed={args.seed})")

        print(f"\n--- Train split ---")
        convert_sequences(train_seqs, args.input_dir,
                          os.path.join(args.output_dir, "train"), args.symlink)

        print(f"\n--- Valid split ---")
        convert_sequences(valid_seqs, args.input_dir,
                          os.path.join(args.output_dir, "valid"), args.symlink)
    else:
        print(f"\n--- {args.split} split ---")
        convert_sequences(seqs, args.input_dir,
                          os.path.join(args.output_dir, args.split), args.symlink)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
