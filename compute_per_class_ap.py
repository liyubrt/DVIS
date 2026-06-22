#!/usr/bin/env python3
"""
Compute per-class AP scores for YTVIS 2022 evaluation.

Reads gt.json and results.json, runs YTVOSeval, and prints/saves per-class AP.

Usage:
    python compute_per_class_ap.py <gt_dir> <output_dir>

  gt_dir:     directory containing gt.json and results.json (symlinked)
  output_dir: directory to write per_class_ap.txt
"""

import json
import os
import sys

import numpy as np
from pycocotools.ytvos import YTVOS
from pycocotools.ytvoseval import YTVOSeval


def compute_per_class_ap(gt_file, results_file, categories):
    """Run evaluation and extract per-class AP from the precision tensor."""
    gts = YTVOS(gt_file)
    res = gts.loadRes(results_file)

    ytvosEval = YTVOSeval(gts, res, "segm")
    ytvosEval.evaluate()
    ytvosEval.accumulate()
    ytvosEval.summarize()

    # precision shape: [T x R x K x A x M]
    # T = IoU thresholds, R = recall thresholds, K = categories, A = area ranges, M = max dets
    precision = ytvosEval.eval["precision"]  # (10, 101, K, 4, 10)

    # AP @ IoU=0.50:0.95, area=all, maxDets=100
    # areaRng index 0 = 'all', maxDets index -1 = max
    per_class_ap = []
    cat_ids = ytvosEval.params.catIds
    for k_idx, cat_id in enumerate(cat_ids):
        # precision[:, :, k_idx, 0, -1] -> shape (T, R)
        p = precision[:, :, k_idx, 0, -1]
        if len(p[p > -1]) == 0:
            ap = -1.0
        else:
            ap = np.mean(p[p > -1])
        cat_name = categories.get(cat_id, str(cat_id))
        per_class_ap.append((cat_id, cat_name, ap))

    return per_class_ap


def main():
    gt_dir = sys.argv[1]
    output_dir = sys.argv[2]

    gt_file = os.path.join(gt_dir, "gt.json")
    results_file = os.path.join(gt_dir, "results.json")

    # Load category names
    with open(gt_file, "r") as f:
        gt_data = json.load(f)
    categories = {c["id"]: c["name"] for c in gt_data["categories"]}

    print("\n=== Per-Class AP (IoU=0.50:0.95) ===")
    per_class_ap = compute_per_class_ap(gt_file, results_file, categories)

    # Sort by AP ascending to highlight weak categories
    per_class_ap_sorted = sorted(per_class_ap, key=lambda x: x[2])

    # Print and save
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "per_class_ap.txt")
    with open(output_file, "w") as f:
        header = f"{'Category':<20s} {'ID':>4s} {'AP':>8s}"
        print(header)
        print("-" * len(header))
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for cat_id, cat_name, ap in per_class_ap_sorted:
            line = f"{cat_name:<20s} {cat_id:4d} {ap:8.3f}"
            print(line)
            f.write(line + "\n")

        # Overall mAP
        valid_aps = [ap for _, _, ap in per_class_ap if ap > -1]
        if valid_aps:
            mAP = np.mean(valid_aps)
        else:
            mAP = -1.0
        summary = f"\n{'mAP':<20s} {'':>4s} {mAP:8.3f}"
        print(summary)
        f.write(summary + "\n")

    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
