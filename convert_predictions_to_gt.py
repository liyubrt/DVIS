#!/usr/bin/env python3
"""
Convert DVIS inference results.json into training-ready YTVIS instances.json.

Reads:
  - <dataset_dir>/instances.json  (inference-only, has videos+categories but empty annotations)
  - <results_json>                (DVIS predictions with segmentations, scores, category_ids)

Writes:
  - <dataset_dir>/instances.json  (overwritten with GT-format annotations)

Usage:
    python convert_predictions_to_gt.py \
        --dataset-dir /path/to/dataset \
        --results-json /path/to/inference/results.json \
        --score-thr 0.5
"""

import argparse
import json
import os

from pycocotools import mask as mask_utils


def rle_to_bbox_area(rle):
    """Compute bounding box [x, y, w, h] and area from a compressed RLE mask."""
    # mask_utils.toBbox returns [x, y, w, h]
    bbox = mask_utils.toBbox(rle).tolist()
    area = int(mask_utils.area(rle))
    return bbox, area


def convert_prediction_to_annotation(pred, ann_id, video_info):
    """Convert a single prediction dict to a YTVIS GT annotation dict."""
    h, w = video_info["height"], video_info["width"]
    length = video_info["length"]

    segmentations = []
    bboxes = []
    areas = []

    for seg in pred["segmentations"]:
        if seg is None:
            segmentations.append(None)
            bboxes.append(None)
            areas.append(None)
        else:
            # Ensure counts is a string (compressed RLE)
            if isinstance(seg["counts"], list):
                # Uncompressed RLE -> compress
                rle = mask_utils.frPyObjects(seg, seg["size"][0], seg["size"][1])
            else:
                rle = seg

            bbox, area = rle_to_bbox_area(rle)

            # Convert counts to uncompressed (list of ints) for YTVIS GT format
            binary_mask = mask_utils.decode(rle)
            uncompressed_rle = mask_utils.encode(binary_mask)
            # Store as compressed RLE string (both formats are accepted by DVIS)
            segmentations.append(rle)
            bboxes.append(bbox)
            areas.append(area)

    annotation = {
        "id": ann_id,
        "video_id": pred["video_id"],
        "category_id": pred["category_id"],
        "segmentations": segmentations,
        "bboxes": bboxes,
        "areas": areas,
        "height": h,
        "width": w,
        "length": length,
        "iscrowd": 0,
    }
    return annotation


def main():
    parser = argparse.ArgumentParser(
        description="Convert DVIS inference results to YTVIS training annotations."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to dataset directory containing instances.json and JPEGImages/",
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Path to DVIS inference results.json",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.5,
        help="Score threshold for filtering predictions (default: 0.5)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output path. Defaults to <dataset-dir>/instances.json (overwrites).",
    )
    args = parser.parse_args()

    # Load existing dataset metadata
    instances_path = os.path.join(args.dataset_dir, "instances.json")
    with open(instances_path, "r") as f:
        dataset = json.load(f)

    video_lookup = {v["id"]: v for v in dataset["videos"]}

    # Load predictions
    with open(args.results_json, "r") as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} predictions from {args.results_json}")

    # Filter by score
    filtered = [p for p in predictions if p["score"] >= args.score_thr]
    print(f"After score threshold ({args.score_thr}): {len(filtered)} predictions")

    # Convert to annotations
    annotations = []
    for ann_id, pred in enumerate(filtered, start=1):
        video_info = video_lookup[pred["video_id"]]
        ann = convert_prediction_to_annotation(pred, ann_id, video_info)
        annotations.append(ann)

    # Count stats
    from collections import Counter
    cat_names = {c["id"]: c["name"] for c in dataset["categories"]}
    cat_counts = Counter(a["category_id"] for a in annotations)
    vid_counts = len(set(a["video_id"] for a in annotations))

    print(f"\nGenerated {len(annotations)} annotations across {vid_counts} videos:")
    for cat_id, count in sorted(cat_counts.items()):
        print(f"  {cat_names.get(cat_id, cat_id):20s} (id={cat_id:2d}): {count}")

    # Build output
    output = {
        "info": dataset.get("info", {}),
        "licenses": dataset.get("licenses", []),
        "videos": dataset["videos"],
        "categories": dataset["categories"],
        "annotations": annotations,
    }

    output_path = args.output_json or instances_path
    # Backup original if overwriting
    if output_path == instances_path:
        backup_path = instances_path.replace(".json", "_inference_only.json")
        if not os.path.exists(backup_path):
            os.rename(instances_path, backup_path)
            print(f"\nBacked up original to {backup_path}")

    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"Saved training annotations to {output_path}")


if __name__ == "__main__":
    main()
