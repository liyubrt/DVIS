#!/usr/bin/env python3
"""
Convert a custom video-sequence dataset to YTVIS 2022 format.

Source layout:
    raw_seqs/
        seq_0/
            <unique_id>.png
            data.csv          # "unique_id" column = filename without extension
        seq_1/
        ...

Output layout (saved under <parent_dir>/ytvis_format/):
    JPEGImages/
        seq_0/00000.jpg
        seq_0/00001.jpg
        ...
    instances.json            # YTVIS-style annotation file (no GT annotations)

Usage:
    python convert_to_ytvis_format.py \
        --src /mnt/data3/jupiter/datasets/sequence_data/humans-on_path_forward-day-sequences-core-jupiter_rev11_8338/raw_seqs \
        --use-symlinks
"""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime

import cv2


def main():
    parser = argparse.ArgumentParser(description="Convert sequence dataset to YTVIS format")
    parser.add_argument("--src", required=True, help="Path to raw_seqs directory")
    parser.add_argument("--out", default=None,
                        help="Output directory. Defaults to <parent of src>/ytvis_format")
    parser.add_argument("--use-symlinks", action="store_true",
                        help="Symlink images instead of copying (saves disk space)")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        out_dir = os.path.join(os.path.dirname(src_dir), "ytvis_format")

    jpeg_dir = os.path.join(out_dir, "JPEGImages")
    os.makedirs(jpeg_dir, exist_ok=True)

    # Discover sequences
    seq_names = sorted(
        d for d in os.listdir(src_dir)
        if os.path.isdir(os.path.join(src_dir, d))
    )
    print(f"Found {len(seq_names)} sequences in {src_dir}")

    # Category: single class "person" (matching YTVIS convention, id=1)
    categories = [{"id": 1, "name": "person", "supercategory": "person"}]

    videos = []

    for vid_id, seq_name in enumerate(seq_names, start=1):
        seq_path = os.path.join(src_dir, seq_name)
        csv_path = os.path.join(seq_path, "data.csv")

        # Read data.csv to get ordered unique_ids
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        unique_ids = [row["unique_id"] for row in rows]

        # Determine image dimensions from first frame
        first_img_path = os.path.join(seq_path, unique_ids[0] + ".png")
        img = cv2.imread(first_img_path)
        if img is None:
            print(f"  WARNING: cannot read {first_img_path}, skipping {seq_name}")
            continue
        h, w = img.shape[:2]

        # Create output image directory for this sequence
        seq_jpeg_dir = os.path.join(jpeg_dir, seq_name)
        os.makedirs(seq_jpeg_dir, exist_ok=True)

        file_names = []
        for frame_idx, uid in enumerate(unique_ids):
            src_img = os.path.join(seq_path, uid + ".png")
            # Name frames as 00000.jpg, 00001.jpg, ... (like YTVIS)
            dst_name = f"{frame_idx:05d}.jpg"
            dst_img = os.path.join(seq_jpeg_dir, dst_name)

            if not os.path.exists(dst_img):
                if args.use_symlinks:
                    os.symlink(os.path.abspath(src_img), dst_img)
                else:
                    shutil.copy2(src_img, dst_img)

            file_names.append(f"{seq_name}/{dst_name}")

        video_entry = {
            "license": 1,
            "coco_url": "",
            "height": h,
            "width": w,
            "length": len(file_names),
            "date_captured": datetime.now().isoformat(),
            "flickr_url": "",
            "file_names": file_names,
            "id": vid_id,
        }
        videos.append(video_entry)
        print(f"  [{vid_id}/{len(seq_names)}] {seq_name}: {len(file_names)} frames, {w}x{h}")

    # Build instances.json
    instances = {
        "info": {
            "description": "Custom sequence dataset in YTVIS format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "",
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [
            {"url": "", "id": 1, "name": "Unknown"}
        ],
        "categories": categories,
        "videos": videos,
    }

    json_path = os.path.join(out_dir, "instances.json")
    with open(json_path, "w") as f:
        json.dump(instances, f)

    print(f"\nDone. Output saved to: {out_dir}")
    print(f"  instances.json: {json_path}")
    print(f"  JPEGImages/:    {jpeg_dir}")
    print(f"  Total videos:   {len(videos)}")
    total_frames = sum(v["length"] for v in videos)
    print(f"  Total frames:   {total_frames}")


if __name__ == "__main__":
    main()
