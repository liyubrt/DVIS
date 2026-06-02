#!/usr/bin/env python3
"""
Extract specific frames and the first "person" instance mask from DVIS predictions.

Class usage:
    extractor = DVISFrameExtractor(
        dataset_dir="datasets/sample_videos",
        results_json="output/.../inference/results.json",
    )
    frames, masks = extractor.get_frames(
        video_name="0_A_man_walking_across_a_flat_open_grassy_field_duri",
        frame_ids=[0, 10, 20, 40, 60],
    )

CLI usage:
    python extract_frames_and_masks.py \
        --video-name "0_A_man_walking_across_a_flat_open_grassy_field_duri" \
        --frame-ids 0 10 20 40 60 \
        --dataset-dir datasets/sample_videos \
        --results-json output/.../inference/results.json \
        --output-dir /tmp/extracted
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
from pycocotools import mask as mask_util

PERSON_CATEGORY_ID = 26


class DVISFrameExtractor:
    """Load DVIS predictions and extract frames + person masks by video name."""

    def __init__(self, dataset_dir: str, results_json: str, score_thr: float = 0.3):
        self.dataset_dir = dataset_dir
        self.jpeg_dir = os.path.join(dataset_dir, "JPEGImages")
        self.score_thr = score_thr

        # Load annotations
        ann_path = os.path.join(dataset_dir, "instances.json")
        with open(ann_path) as f:
            ann = json.load(f)

        # Build name -> video_info lookup
        self.name_to_video = {}
        for v in ann["videos"]:
            name = v["file_names"][0].split("/")[0]
            self.name_to_video[name] = v

        # Load predictions and index by video_id
        with open(results_json) as f:
            predictions = json.load(f)

        self.preds_by_video = {}
        for p in predictions:
            vid = p["video_id"]
            if vid not in self.preds_by_video:
                self.preds_by_video[vid] = []
            self.preds_by_video[vid].append(p)

        print(f"DVISFrameExtractor: {len(self.name_to_video)} videos, "
              f"{len(predictions)} predictions loaded")

    def video_names(self) -> List[str]:
        """Return list of available video names."""
        return sorted(self.name_to_video.keys())

    def get_frames(
        self,
        video_name: str,
        frame_ids: List[int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Extract frames and the first person mask for the given video and frame indices.

        Args:
            video_name: Video folder name (under JPEGImages/).
            frame_ids: List of 0-indexed frame indices to extract.

        Returns:
            frames: List of BGR images (H, W, 3) as np.uint8.
            masks: List of binary masks (H, W) as np.uint8, 255=person, 0=background.
        """
        if video_name not in self.name_to_video:
            raise ValueError(
                f"Video '{video_name}' not found. "
                f"Available: {self.video_names()[:5]}..."
            )

        video_info = self.name_to_video[video_name]
        video_id = video_info["id"]
        file_names = video_info["file_names"]
        num_frames = len(file_names)

        for fid in frame_ids:
            if fid < 0 or fid >= num_frames:
                raise ValueError(f"frame_id {fid} out of range [0, {num_frames - 1}]")

        # Find first (highest-scoring) person prediction
        preds = self.preds_by_video.get(video_id, [])
        person_preds = [
            p for p in preds
            if p["category_id"] == PERSON_CATEGORY_ID
            and p["score"] >= self.score_thr
        ]
        person_preds.sort(key=lambda p: p["score"], reverse=True)
        person_pred = person_preds[0] if person_preds else None

        frames = []
        masks = []

        for fid in frame_ids:
            img_path = os.path.join(self.jpeg_dir, file_names[fid])
            image = (imageio.imread(img_path) / 255.0).astype(np.float32)
            if image is None:
                raise FileNotFoundError(f"Cannot read {img_path}")

            frames.append(image)

            if person_pred is not None:
                seg = person_pred["segmentations"][fid]
                if seg is not None:
                    binary_mask = mask_util.decode(seg)  # (H, W) uint8, 0/1
                    mask_img = (binary_mask * 11).astype(np.uint8)
                else:
                    mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                mask_img = np.zeros(image.shape[:2], dtype=np.uint8)

            masks.append(mask_img)

        return frames, masks


def video_cutnpaste(
    images: List[np.ndarray],
    frames: List[np.ndarray],
    masks: List[np.ndarray],
    zoom_ratio: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Copy person regions from frames onto images using binary masks.

    Args:
        images: List of background images (H, W, 3) uint8.
        frames: List of source images (H, W, 3) uint8 containing the person.
        masks: List of binary masks (H, W) uint8, 255=person region in frames.
        zoom_ratio: Scale factor for the person region. >1 zooms in (larger),
                    <1 zooms out (smaller). The center of the person stays fixed.

    Returns:
        results: List of composited images (H, W, 3) uint8 with person from
                 frames pasted onto images.
        scaled_masks: List of binary masks (H, W) uint8 after resizing and zoom.
    """
    assert len(images) == len(frames) == len(masks), (
        f"Length mismatch: images={len(images)}, frames={len(frames)}, masks={len(masks)}"
    )

    results = []
    scaled_masks = []
    for img, frame, mask in zip(images, frames, masks):
        h, w = img.shape[:2]
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        result = img.copy()

        if zoom_ratio != 1.0 and np.any(mask > 0):
            # Find person center
            ys, xs = np.where(mask > 0)
            cy, cx = float(ys.mean()), float(xs.mean())

            # Affine transform: scale around (cx, cy)
            M = np.array([
                [zoom_ratio, 0, cx * (1 - zoom_ratio)],
                [0, zoom_ratio, cy * (1 - zoom_ratio)],
            ], dtype=np.float32)

            frame = cv2.warpAffine(frame, M, (w, h),
                                   flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            mask = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        person_region = mask > 0
        result[person_region] = frame[person_region]
        results.append(result)
        scaled_masks.append(mask)

    return results, scaled_masks


def main():
    parser = argparse.ArgumentParser(description="Extract frames and person masks")
    parser.add_argument("--video-name", required=True, help="Video folder name (under JPEGImages/)")
    parser.add_argument("--frame-ids", type=int, nargs="+", required=True,
                        help="Frame indices to extract (0-indexed)")
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to YTVIS-format dataset (with JPEGImages/ and instances.json)")
    parser.add_argument("--results-json", required=True, help="Path to results.json from inference")
    parser.add_argument("--output-dir", required=True, help="Directory to save extracted frames/masks")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    extractor = DVISFrameExtractor(args.dataset_dir, args.results_json, args.score_thr)
    frames, masks = extractor.get_frames(args.video_name, args.frame_ids)

    for fid, image, mask_img in zip(args.frame_ids, frames, masks):
        img_out = os.path.join(args.output_dir, f"{args.video_name}_frame{fid:04d}_image.png")
        mask_out = os.path.join(args.output_dir, f"{args.video_name}_frame{fid:04d}_mask.png")
        imageio.imwrite(img_out, image)
        imageio.imwrite(mask_out, mask_img)
        print(f"  frame {fid:4d}: {img_out}  mask pixels={mask_img.sum() // 255}")

    print(f"\nDone. {len(args.frame_ids)} frames saved to {args.output_dir}")


if __name__ == "__main__":
    main()
