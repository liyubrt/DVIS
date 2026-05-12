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
import shutil

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Path to raw_seqs directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for YTVIS-format dataset")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--symlink", action="store_true", help="Use symlinks instead of copying images")
    args = parser.parse_args()

    split_dir = os.path.join(args.output_dir, args.split)
    images_dir = os.path.join(split_dir, "JPEGImages")
    os.makedirs(images_dir, exist_ok=True)

    seqs = sorted([
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    ])
    print(f"Found {len(seqs)} sequences")

    videos = []
    annotations = []
    video_id = 0
    ann_id = 0

    for seq_name in seqs:
        seq_dir = os.path.join(args.input_dir, seq_name)
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
                if args.symlink:
                    os.symlink(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)

            file_names.append(os.path.join(seq_name, uid + ".png"))

            # Encode mask
            mask_path = os.path.join(seq_dir, uid + "_label.png")
            if os.path.exists(mask_path):
                mask_arr = np.array(Image.open(mask_path))
                rle, area, bbox = encode_mask(mask_arr, label_val=11)
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
                "category_id": 1,
                "segmentations": segmentations,
                "areas": areas,
                "bboxes": bboxes,
                "iscrowd": 0,
            })

        if video_id % 100 == 0:
            print(f"  Processed {video_id}/{len(seqs)} sequences")

    dataset = {
        "info": {
            "description": "Custom humans-on-path dataset in YTVIS format",
            "version": "1.0",
        },
        "licenses": [],
        "videos": videos,
        "categories": [
            {"id": 1, "name": "humans", "supercategory": "person"}
        ],
        "annotations": annotations,
    }

    out_json = os.path.join(split_dir, "instances.json")
    with open(out_json, "w") as f:
        json.dump(dataset, f)

    print(f"\nDone!")
    print(f"  Videos: {len(videos)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Saved to: {out_json}")
    print(f"  Images in: {images_dir}")


if __name__ == "__main__":
    main()
