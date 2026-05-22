#!/usr/bin/env python3
"""
Convert raw video files (.mp4) into YTVIS-format dataset for DVIS inference.

Creates:
  <output-dir>/JPEGImages/<video_name>/<frame_index>.jpg
  <output-dir>/instances.json  (minimal, no annotations)

Usage:
    python prepare_videos_for_inference.py \
        --input-dir /path/to/videos \
        --output-dir /path/to/output \
        --fps 0  # 0 = use original fps, otherwise subsample
"""

import argparse
import glob
import json
import os

import cv2


def extract_frames(video_path, out_dir, target_fps=0):
    """Extract frames from a video file. Returns (width, height, num_frames, file_names)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: cannot open {video_path}")
        return None

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # If target_fps <= 0, keep all frames
    if target_fps > 0 and target_fps < orig_fps:
        frame_interval = orig_fps / target_fps
    else:
        frame_interval = 1.0

    os.makedirs(out_dir, exist_ok=True)
    file_names = []
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Decide whether to keep this frame
        if frame_idx >= saved_idx * frame_interval:
            fname = f"{saved_idx:06d}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), frame)
            file_names.append(fname)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    return width, height, len(file_names), file_names


def main():
    parser = argparse.ArgumentParser(description="Prepare videos for DVIS inference")
    parser.add_argument("--input-dir", required=True, help="Directory containing .mp4 files")
    parser.add_argument("--output-dir", required=True, help="Output directory (YTVIS format)")
    parser.add_argument("--fps", type=float, default=0, help="Target FPS (0 = keep original)")
    parser.add_argument("--ext", default="mp4", help="Video file extension")
    args = parser.parse_args()

    video_files = sorted(glob.glob(os.path.join(args.input_dir, f"*.{args.ext}")))
    if not video_files:
        print(f"No .{args.ext} files found in {args.input_dir}")
        return

    print(f"Found {len(video_files)} videos")

    jpeg_dir = os.path.join(args.output_dir, "JPEGImages")
    os.makedirs(jpeg_dir, exist_ok=True)

    # YTVIS 2022 categories (40 classes)
    categories = [
        {"id": 1, "name": "airplane"}, {"id": 2, "name": "bear"},
        {"id": 3, "name": "bird"}, {"id": 4, "name": "boat"},
        {"id": 5, "name": "car"}, {"id": 6, "name": "cat"},
        {"id": 7, "name": "cow"}, {"id": 8, "name": "deer"},
        {"id": 9, "name": "dog"}, {"id": 10, "name": "duck"},
        {"id": 11, "name": "earless_seal"}, {"id": 12, "name": "elephant"},
        {"id": 13, "name": "fish"}, {"id": 14, "name": "flying_disc"},
        {"id": 15, "name": "fox"}, {"id": 16, "name": "frog"},
        {"id": 17, "name": "giant_panda"}, {"id": 18, "name": "giraffe"},
        {"id": 19, "name": "horse"}, {"id": 20, "name": "leopard"},
        {"id": 21, "name": "lizard"}, {"id": 22, "name": "monkey"},
        {"id": 23, "name": "motorbike"}, {"id": 24, "name": "mouse"},
        {"id": 25, "name": "parrot"}, {"id": 26, "name": "person"},
        {"id": 27, "name": "rabbit"}, {"id": 28, "name": "shark"},
        {"id": 29, "name": "skateboard"}, {"id": 30, "name": "snake"},
        {"id": 31, "name": "snowboard"}, {"id": 32, "name": "squirrel"},
        {"id": 33, "name": "surfboard"}, {"id": 34, "name": "tennis_racket"},
        {"id": 35, "name": "tiger"}, {"id": 36, "name": "train"},
        {"id": 37, "name": "truck"}, {"id": 38, "name": "turtle"},
        {"id": 39, "name": "whale"}, {"id": 40, "name": "zebra"},
    ]

    videos = []
    for vid_idx, vpath in enumerate(video_files):
        video_name = os.path.splitext(os.path.basename(vpath))[0]
        out_frames_dir = os.path.join(jpeg_dir, video_name)

        result = extract_frames(vpath, out_frames_dir, args.fps)
        if result is None:
            continue

        width, height, num_frames, file_names = result
        # YTVIS format: file_names are relative to JPEGImages dir
        file_names_rel = [f"{video_name}/{fn}" for fn in file_names]

        videos.append({
            "id": vid_idx + 1,
            "width": width,
            "height": height,
            "length": num_frames,
            "file_names": file_names_rel,
        })
        print(f"  [{vid_idx+1}/{len(video_files)}] {video_name}: {width}x{height}, {num_frames} frames")

    instances = {
        "videos": videos,
        "annotations": [],
        "categories": categories,
    }

    out_json = os.path.join(args.output_dir, "instances.json")
    with open(out_json, "w") as f:
        json.dump(instances, f)

    print(f"\nDone. {len(videos)} videos prepared.")
    print(f"  Frames:  {jpeg_dir}")
    print(f"  JSON:    {out_json}")


if __name__ == "__main__":
    main()
