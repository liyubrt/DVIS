#!/usr/bin/env python3
"""
Merge COCO (mapped to YTVIS 2022 categories) and YTVIS 2022 (COCO format)
into a single roboflow-style dataset for RF-DETR training.

COCO categories are filtered/remapped to the YTVIS 2021/2022 40-class set.
The merged dataset keeps the YTVIS 2022 category IDs (1-40).

Output layout:
    <output_dir>/
        train/
            _annotations.coco.json   (merged annotations)
            <coco images via symlink to train2017/>
            <ytvis video dirs via symlinks>
        valid/
            _annotations.coco.json   (ytvis valid only)
            <ytvis video dirs via symlinks>

Usage:
    python merge_coco_ytvis2022_for_rfdetr.py \
        --coco-dir /mnt/data3/jupiter/datasets/public_datasets/COCO2017 \
        --ytvis-coco-dir /mnt/data3/jupiter/datasets/public_datasets/ytvis2022_coco \
        --output-dir /mnt/data3/jupiter/datasets/public_datasets/ytvis2022_coco_merged
"""

import argparse
import json
import os
import shutil
from pathlib import Path

# COCO 80-class category_id -> YTVIS 2021/2022 40-class category_id
# Only categories that exist in YTVIS are included.
COCO_TO_YTVIS_2021 = {
    1: 26, 2: 23, 3: 5, 4: 23, 5: 1, 7: 36, 8: 37, 9: 4,
    16: 3, 17: 6, 18: 9, 19: 19, 21: 7, 22: 12, 23: 2, 24: 40,
    25: 18, 34: 14, 35: 31, 36: 31, 41: 29, 42: 33, 43: 34
}


def main():
    parser = argparse.ArgumentParser(
        description="Merge COCO + YTVIS2022 for RF-DETR")
    parser.add_argument("--coco-dir", type=str,
                        default="/mnt/data3/jupiter/datasets/public_datasets/COCO2017")
    parser.add_argument("--ytvis-coco-dir", type=str,
                        default="/mnt/data3/jupiter/datasets/public_datasets/ytvis2022_coco")
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/data3/jupiter/datasets/public_datasets/ytvis2022_coco_merged")
    args = parser.parse_args()

    coco_dir = Path(args.coco_dir)
    ytvis_dir = Path(args.ytvis_coco_dir)
    out_dir = Path(args.output_dir)

    # ---- Load YTVIS 2022 COCO-format train ----
    with open(ytvis_dir / "train" / "_annotations.coco.json") as f:
        ytvis_train = json.load(f)

    ytvis_categories = ytvis_train["categories"]  # 40 YTVIS categories
    print(f"YTVIS train: {len(ytvis_train['images'])} images, "
          f"{len(ytvis_train['annotations'])} annotations")

    # ---- Load COCO train ----
    with open(coco_dir / "annotations" / "instances_train2017.json") as f:
        coco_train = json.load(f)
    print(f"COCO train:  {len(coco_train['images'])} images, "
          f"{len(coco_train['annotations'])} annotations")

    # ---- Filter & remap COCO annotations ----
    coco_anns_mapped = []
    for ann in coco_train["annotations"]:
        if ann["category_id"] in COCO_TO_YTVIS_2021:
            ann_copy = dict(ann)
            ann_copy["category_id"] = COCO_TO_YTVIS_2021[ann["category_id"]]
            coco_anns_mapped.append(ann_copy)
    print(f"COCO filtered/remapped: {len(coco_anns_mapped)} annotations")

    # Collect COCO image IDs that have at least one mapped annotation
    coco_img_ids_with_anns = set(a["image_id"] for a in coco_anns_mapped)
    coco_images_filtered = [
        img for img in coco_train["images"]
        if img["id"] in coco_img_ids_with_anns
    ]
    print(f"COCO images with mapped anns: {len(coco_images_filtered)}")

    # ---- Merge: offset COCO IDs to avoid collision with YTVIS ----
    max_ytvis_img_id = max(img["id"] for img in ytvis_train["images"])
    max_ytvis_ann_id = max(ann["id"] for ann in ytvis_train["annotations"])

    img_id_offset = max_ytvis_img_id + 1
    ann_id_offset = max_ytvis_ann_id + 1

    old_to_new_img_id = {}
    merged_images = list(ytvis_train["images"])  # YTVIS images as-is
    for img in coco_images_filtered:
        new_id = img["id"] + img_id_offset
        old_to_new_img_id[img["id"]] = new_id
        # Prefix file_name with "coco/" so we can symlink coco/ -> train2017/
        merged_images.append({
            "id": new_id,
            "file_name": "coco/" + img["file_name"],
            "height": img["height"],
            "width": img["width"],
        })

    merged_anns = list(ytvis_train["annotations"])  # YTVIS annotations as-is
    for ann in coco_anns_mapped:
        if ann["image_id"] not in old_to_new_img_id:
            continue
        merged_anns.append({
            "id": ann["id"] + ann_id_offset,
            "image_id": old_to_new_img_id[ann["image_id"]],
            "category_id": ann["category_id"],
            "segmentation": ann["segmentation"],
            "bbox": ann["bbox"],
            "area": ann["area"],
            "iscrowd": ann.get("iscrowd", 0),
        })

    merged = {
        "images": merged_images,
        "annotations": merged_anns,
        "categories": ytvis_categories,
    }
    print(f"\nMerged train: {len(merged['images'])} images, "
          f"{len(merged['annotations'])} annotations, "
          f"{len(merged['categories'])} categories")

    # ---- Write merged train ----
    train_out = out_dir / "train"
    train_out.mkdir(parents=True, exist_ok=True)

    with open(train_out / "_annotations.coco.json", "w") as f:
        json.dump(merged, f)
    print(f"Saved {train_out / '_annotations.coco.json'}")

    # Symlink YTVIS video directories
    ytvis_train_dir = ytvis_dir / "train"
    linked = 0
    for entry in sorted(ytvis_train_dir.iterdir()):
        if entry.is_dir() or (entry.is_symlink() and not entry.name.startswith("_")):
            dst = train_out / entry.name
            if not dst.exists():
                dst.symlink_to(entry.resolve())
                linked += 1
    print(f"Symlinked {linked} YTVIS video directories")

    # Symlink COCO images: train/coco/ -> COCO2017/train2017/
    coco_link = train_out / "coco"
    if not coco_link.exists():
        coco_link.symlink_to((coco_dir / "train2017").resolve())
        print(f"Symlinked coco/ -> {coco_dir / 'train2017'}")

    # ---- Copy valid split as-is (YTVIS only) ----
    valid_out = out_dir / "valid"
    valid_out.mkdir(parents=True, exist_ok=True)

    ytvis_valid_dir = ytvis_dir / "valid"
    if (ytvis_valid_dir / "_annotations.coco.json").exists():
        shutil.copy2(
            ytvis_valid_dir / "_annotations.coco.json",
            valid_out / "_annotations.coco.json"
        )
        for entry in sorted(ytvis_valid_dir.iterdir()):
            if entry.is_dir() or (entry.is_symlink() and not entry.name.startswith("_")):
                dst = valid_out / entry.name
                if not dst.exists():
                    dst.symlink_to(entry.resolve())
        print(f"Copied valid split from {ytvis_valid_dir}")

    print(f"\nDone! Merged dataset at: {out_dir}")


if __name__ == "__main__":
    main()
