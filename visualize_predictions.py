#!/usr/bin/env python3
"""
Visualize DVIS predictions on the YTVIS 2022 validation set.

For each video, overlays predicted instance masks (color-coded), class names,
and instance IDs on the original frames, then writes an MP4 video.

Usage:
    python visualize_predictions.py \
        --results-json output_Downloaded_DVIS_Online_SwinL_YTVIS21_on_YTVIS22/inference/results.json \
        --annotations datasets/ytvis_2022/valid/instances.json \
        --images-dir  datasets/ytvis_2022/valid/JPEGImages \
        --output-dir  output_Downloaded_DVIS_Online_SwinL_YTVIS21_on_YTVIS22/vis_videos \
        --score-thr 0.3 \
        --fps 6
"""

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_util


# ---------- colour palette (distinguishable, up to 80 instances) ----------
def _generate_palette(n=80):
    """Generate n visually distinct BGR colours."""
    rng = np.random.RandomState(42)
    palette = []
    for _ in range(n):
        palette.append(tuple(int(c) for c in rng.randint(60, 220, size=3)))
    return palette


PALETTE = _generate_palette()


def overlay_mask(image, binary_mask, color, alpha=0.45):
    """Overlay a binary mask on an image with transparency."""
    overlay = image.copy()
    overlay[binary_mask > 0] = color
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    # draw contour
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, 2)
    return image


def put_label(image, text, position, color, font_scale=0.6, thickness=2):
    """Put a text label with a background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    # clamp to image bounds
    x = max(0, min(x, image.shape[1] - tw - 4))
    y = max(th + 4, min(y, image.shape[0] - 4))
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 4, y + 4), color, -1)
    # text colour: white or black depending on brightness
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    txt_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    cv2.putText(image, text, (x + 2, y), font, font_scale, txt_color, thickness)
    return image


def mask_to_centroid(binary_mask):
    """Get (x, y) centroid of a binary mask."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def main():
    parser = argparse.ArgumentParser(description="Visualize DVIS predictions on YTVIS 2022")
    parser.add_argument("--results-json", required=True, help="Path to results.json from inference")
    parser.add_argument("--annotations", required=True, help="Path to valid/instances.json")
    parser.add_argument("--images-dir", required=True, help="Path to valid/JPEGImages")
    parser.add_argument("--output-dir", required=True, help="Directory to save output videos")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold for predictions")
    parser.add_argument("--fps", type=int, default=6, help="FPS for output videos")
    parser.add_argument("--max-videos", type=int, default=None, help="Max number of videos to visualize (for debugging)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load annotations (for video metadata and category names) ----
    with open(args.annotations) as f:
        ann = json.load(f)

    cat_id_to_name = {c["id"]: c["name"] for c in ann["categories"]}
    video_id_to_info = {v["id"]: v for v in ann["videos"]}

    # ---- Load predictions ----
    with open(args.results_json) as f:
        predictions = json.load(f)

    # Group predictions by video_id
    preds_by_video = defaultdict(list)
    for pred in predictions:
        if pred["score"] >= args.score_thr:
            preds_by_video[pred["video_id"]].append(pred)

    # Sort each video's predictions by score (descending)
    for vid in preds_by_video:
        preds_by_video[vid].sort(key=lambda p: p["score"], reverse=True)

    video_ids = sorted(video_id_to_info.keys())
    if args.max_videos is not None:
        video_ids = video_ids[: args.max_videos]

    print(f"Visualizing {len(video_ids)} videos, score_thr={args.score_thr}")

    for vi, video_id in enumerate(video_ids):
        vinfo = video_id_to_info[video_id]
        file_names = vinfo["file_names"]
        h, w = vinfo["height"], vinfo["width"]
        video_name = file_names[0].split("/")[0]

        preds = preds_by_video.get(video_id, [])

        out_path = os.path.join(args.output_dir, f"{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

        for fi, fname in enumerate(file_names):
            img_path = os.path.join(args.images_dir, fname)
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"  WARNING: could not read {img_path}")
                continue

            # overlay each instance
            for inst_idx, pred in enumerate(preds):
                seg = pred["segmentations"][fi]
                if seg is None:
                    continue

                # decode RLE mask
                binary_mask = mask_util.decode(seg)  # (H, W) uint8

                if binary_mask.sum() == 0:
                    continue

                color = PALETTE[inst_idx % len(PALETTE)]
                cat_name = cat_id_to_name.get(pred["category_id"], "?")
                score = pred["score"]
                label = f"#{inst_idx} {cat_name} {score:.2f}"

                frame = overlay_mask(frame, binary_mask, color)

                centroid = mask_to_centroid(binary_mask)
                if centroid is not None:
                    frame = put_label(frame, label, centroid, color)

            writer.write(frame)

        writer.release()
        print(f"  [{vi+1}/{len(video_ids)}] {video_name}: {len(file_names)} frames, {len(preds)} instances -> {out_path}")

    print(f"\nDone. Videos saved to {args.output_dir}")


if __name__ == "__main__":
    main()
