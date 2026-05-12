#!/usr/bin/env python3
"""
Convert YTVIS 2022 video-level annotations to COCO image-level format.

This flattens per-video annotations into per-frame annotations so that
image-based models (e.g. RF-DETR) can be trained on YTVIS 2022 data.

Output directory layout (roboflow-style, compatible with RF-DETR):
    <output_dir>/
        train/
            _annotations.coco.json
            <video_id>/ -> symlink to original JPEGImages/<video_id>/
        valid/
            _annotations.coco.json
            <video_id>/ -> symlink to original JPEGImages/<video_id>/

Usage:
    python convert_ytvis2022_to_coco.py \
        --ytvis-root /mnt/data3/jupiter/datasets/public_datasets/YouTube_VIS2022 \
        --output-dir /mnt/data3/jupiter/datasets/public_datasets/ytvis2022_coco
"""

import argparse
import json
import os
from pathlib import Path

from pycocotools import mask as mask_util


def convert_split(ytvis_dir: Path, output_split_dir: Path, split: str,
                  ann_json_override: Path = None):
    """Convert one YTVIS split (train or valid) to COCO format.

    Args:
        ytvis_dir: Path to the YTVIS split directory (e.g. .../YouTube_VIS2022/train)
        output_split_dir: Path to the output split directory (e.g. .../ytvis2022_coco/train)
        split: Split name for logging ("train" or "valid")
        ann_json_override: Optional path to a separate annotation JSON (e.g. gt.json)
    """
    ann_path = ann_json_override if ann_json_override else ytvis_dir / "instances.json"
    jpeg_dir = ytvis_dir / "JPEGImages"

    with open(ann_path) as f:
        ytvis = json.load(f)

    vid_map = {v["id"]: v for v in ytvis["videos"]}

    coco = {
        "images": [],
        "annotations": [],
        "categories": ytvis["categories"],
    }

    # Assign global image IDs and build (video_id, frame_idx) -> image_id map
    frame_to_image_id = {}
    image_id = 0

    for video in ytvis["videos"]:
        for frame_idx, file_name in enumerate(video["file_names"]):
            image_id += 1
            coco["images"].append({
                "id": image_id,
                "file_name": file_name,  # e.g. "0043f083b5/00000.jpg"
                "height": video["height"],
                "width": video["width"],
            })
            frame_to_image_id[(video["id"], frame_idx)] = image_id

    # Flatten video annotations into per-frame COCO annotations
    ann_id = 0
    skipped = 0

    for ann in ytvis.get("annotations", []):
        video_id = ann["video_id"]
        video = vid_map[video_id]
        h, w = video["height"], video["width"]

        for frame_idx in range(len(video["file_names"])):
            seg = ann["segmentations"][frame_idx] if frame_idx < len(ann["segmentations"]) else None
            bbox = ann["bboxes"][frame_idx] if frame_idx < len(ann["bboxes"]) else None

            if seg is None or bbox is None:
                skipped += 1
                continue

            # Convert uncompressed RLE (counts as int list) to compressed RLE string
            if isinstance(seg, dict) and isinstance(seg.get("counts"), list):
                rle = mask_util.frPyObjects(seg, h, w)
                area = float(mask_util.area(rle))
                seg_out = {
                    "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
                    "size": rle["size"],
                }
            elif isinstance(seg, dict) and isinstance(seg.get("counts"), str):
                # Already compressed RLE
                seg_out = seg
                rle = mask_util.frPyObjects(seg, h, w)
                area = float(mask_util.area(rle))
            elif isinstance(seg, list):
                # Polygon format
                seg_out = seg
                rle = mask_util.frPyObjects(seg, h, w)
                rle = mask_util.merge(rle)
                area = float(mask_util.area(rle))
            else:
                skipped += 1
                continue

            img_id = frame_to_image_id[(video_id, frame_idx)]
            ann_id += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": ann["category_id"],
                "segmentation": seg_out,
                "bbox": bbox,  # [x, y, w, h]
                "area": area,
                "iscrowd": ann.get("iscrowd", 0),
            })

    # Write annotation JSON
    output_split_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_split_dir / "_annotations.coco.json"
    with open(out_json, "w") as f:
        json.dump(coco, f)

    # Create symlinks for each video directory so images are accessible
    video_dirs = set()
    for img in coco["images"]:
        video_dir = img["file_name"].split("/")[0]
        video_dirs.add(video_dir)

    linked = 0
    for vdir in sorted(video_dirs):
        src = jpeg_dir / vdir
        dst = output_split_dir / vdir
        if not dst.exists():
            dst.symlink_to(src)
            linked += 1

    print(f"[{split}] {len(ytvis['videos'])} videos -> {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations ({skipped} empty frames skipped)")
    print(f"[{split}] Symlinked {linked} video directories")
    print(f"[{split}] Saved to {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Convert YTVIS 2022 to COCO format")
    parser.add_argument("--ytvis-root", type=str,
                        default="/mnt/data3/jupiter/datasets/public_datasets/YouTube_VIS2022",
                        help="Path to YTVIS 2022 root directory")
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/data3/jupiter/datasets/public_datasets/ytvis2022_coco",
                        help="Output directory for COCO-format dataset")
    args = parser.parse_args()

    ytvis_root = Path(args.ytvis_root)
    output_dir = Path(args.output_dir)

    assert (ytvis_root / "train" / "instances.json").exists(), \
        f"Train annotations not found at {ytvis_root / 'train' / 'instances.json'}"

    # Convert train split
    convert_split(ytvis_root / "train", output_dir / "train", "train")

    # Convert valid split — valid/instances.json has no annotations,
    # so use the root-level gt.json which contains validation ground truth.
    valid_dir = ytvis_root / "valid"
    gt_json = ytvis_root / "gt.json"
    if gt_json.exists():
        convert_split(valid_dir, output_dir / "valid", "valid", ann_json_override=gt_json)
    elif (valid_dir / "instances.json").exists():
        convert_split(valid_dir, output_dir / "valid", "valid")
    else:
        print("Valid split not found, skipping.")

    print(f"\nDone! Output at: {output_dir}")
    print(f"Use with RF-DETR: model.train(dataset_dir='{output_dir}', dataset_file='roboflow', ...)")


if __name__ == "__main__":
    main()
