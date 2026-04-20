"""
Extract frames from 3 camera videos into route_3 dataset structure.

Videos:
  cam0_20260413_142148.mp4  →  raw_left/   (left camera,  Alpamayo index 0)
  cam1_20260413_142148.mp4  →  raw/        (front camera, Alpamayo index 1)
  cam2_20260413_142148.mp4  →  raw_right/  (right camera, Alpamayo index 2)

Output structure:
  datasets/route_3/
    segment_00/
      raw_left/000000.png ... 000174.png
      raw/000000.png      ... 000174.png
      raw_right/000000.png ... 000174.png
    segment_01/
      ...
    segment_09/
      ...

Usage:
  python extract_3cam_route.py
  python extract_3cam_route.py --segments 10 --frames-per-segment 175
  python extract_3cam_route.py --dry-run   # preview without writing
"""

import os
import sys
import argparse
import cv2
import numpy as np

# Paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")

# Camera mapping: (video filename prefix, output directory name)
CAMERAS = [
    ("cam0", "raw_left"),   # left camera  → Alpamayo index 0
    ("cam1", "raw_front"),  # front camera → Alpamayo index 1
    ("cam2", "raw_right"),  # right camera → Alpamayo index 2
]


def find_video(prefix):
    """Find the video file matching a camera prefix."""
    for f in os.listdir(VIDEOS_DIR):
        if f.startswith(prefix) and f.endswith((".mp4", ".mov", ".avi", ".mkv")):
            return os.path.join(VIDEOS_DIR, f)
    return None


def get_video_info(path):
    """Get frame count and FPS of a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total, fps, w, h


def extract_frames(video_path, frame_indices):
    """Extract specific frames from a video by index. Returns list of BGR numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    sorted_indices = sorted(frame_indices)
    idx_set = set(sorted_indices)
    max_idx = sorted_indices[-1]

    current = 0
    while current <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if current in idx_set:
            frames.append((current, frame))
        current += 1

    cap.release()

    # Return in the original requested order
    frame_map = {idx: f for idx, f in frames}
    return [frame_map.get(i) for i in frame_indices]


def main():
    parser = argparse.ArgumentParser(
        description="Extract 3-camera frames into route_3 dataset"
    )
    parser.add_argument(
        "--route", type=str, default="route_3",
        help="Route directory name under datasets/ (default: route_3)"
    )
    parser.add_argument(
        "--segments", type=int, default=10,
        help="Number of segments (default: 10)"
    )
    parser.add_argument(
        "--frames-per-segment", type=int, default=175,
        help="Frames per segment (default: 175)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview what would be done without writing files"
    )
    args = parser.parse_args()

    route_dir = os.path.join(DATASETS_DIR, args.route)

    # Determine segments and frames per segment based on existing raw folder
    seg_00_raw = os.path.join(route_dir, "segment_00", "raw")
    if os.path.exists(seg_00_raw):
        # Count frames in segment_00/raw
        frames = [f for f in os.listdir(seg_00_raw) if f.endswith(('.png', '.jpg'))]
        args.frames_per_segment = len(frames)
        print(f"Detected {args.frames_per_segment} frames per segment based on {seg_00_raw}")
        
        # Count existing segments
        segments = [d for d in os.listdir(route_dir) if d.startswith("segment_") and os.path.isdir(os.path.join(route_dir, d))]
        args.segments = len(segments)
        print(f"Detected {args.segments} segments based on {route_dir}")

    total_frames_needed = args.segments * args.frames_per_segment

    # Discover and validate videos
    print("=" * 60)
    print("3-Camera Frame Extractor")
    print("=" * 60)

    video_paths = {}
    video_totals = {}
    for prefix, dir_name in CAMERAS:
        path = find_video(prefix)
        if path is None:
            print(f"ERROR: No video found for '{prefix}' in {VIDEOS_DIR}")
            sys.exit(1)
        total, fps, w, h = get_video_info(path)
        video_paths[prefix] = path
        video_totals[prefix] = total
        print(f"  {prefix}: {os.path.basename(path)}  "
              f"{total} frames, {fps:.1f} FPS, {w}x{h}")

    # Use the shortest video to determine sampling
    min_frames = min(video_totals.values())
    print(f"\nShortest video: {min_frames} frames")
    print(f"Frames needed:  {total_frames_needed} "
          f"({args.segments} segments × {args.frames_per_segment} frames)")

    if min_frames < total_frames_needed:
        print(f"\nWARNING: Not enough frames ({min_frames}) for "
              f"{total_frames_needed} requested. Will use all available frames "
              f"and distribute evenly.")
        total_frames_needed = min(total_frames_needed, min_frames)
        args.frames_per_segment = total_frames_needed // args.segments
        total_frames_needed = args.segments * args.frames_per_segment
        print(f"Adjusted: {args.frames_per_segment} frames per segment")

    # Compute evenly spaced frame indices across the video
    frame_indices = np.linspace(
        0, min_frames - 1, total_frames_needed, dtype=int
    ).tolist()

    step = min_frames / total_frames_needed
    effective_fps = (video_totals[list(video_totals.keys())[0]] /
                     (min_frames / 20.0)) / step  # assuming 20fps source
    print(f"Sampling every ~{step:.2f} frames (effective ~{20/step:.1f} Hz "
          f"from 20 FPS source)")

    if args.dry_run:
        print("\n[DRY RUN] Would create:")
        for seg_idx in range(args.segments):
            seg_start = seg_idx * args.frames_per_segment
            seg_end = seg_start + args.frames_per_segment - 1
            src_start = frame_indices[seg_start]
            src_end = frame_indices[seg_end]
            print(f"  segment_{seg_idx:02d}/  "
                  f"frames {seg_start}-{seg_end} "
                  f"(source frames {src_start}-{src_end})")
            for _, dir_name in CAMERAS:
                print(f"    {dir_name}/000000.png ... "
                      f"{args.frames_per_segment - 1:06d}.png")
        print("\nRe-run without --dry-run to extract.")
        return

    # Extract and save
    for prefix, dir_name in CAMERAS:
        path = video_paths[prefix]
        print(f"\nExtracting {prefix} -> {dir_name}/ ...")

        # Extract all needed frames at once (much faster than seeking)
        all_frames = extract_frames(path, frame_indices)

        for seg_idx in range(args.segments):
            seg_dir = os.path.join(
                route_dir, f"segment_{seg_idx:02d}", dir_name
            )
            os.makedirs(seg_dir, exist_ok=True)

            seg_start = seg_idx * args.frames_per_segment
            seg_end = seg_start + args.frames_per_segment

            for local_idx, global_idx in enumerate(
                range(seg_start, seg_end)
            ):
                frame = all_frames[global_idx]
                if frame is None:
                    print(f"  WARNING: missing frame at index "
                          f"{frame_indices[global_idx]}, writing black")
                    # Get dimensions from any available frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                out_path = os.path.join(seg_dir, f"{local_idx:06d}.png")
                cv2.imwrite(out_path, frame)

            print(f"  segment_{seg_idx:02d}/{dir_name}: "
                  f"{args.frames_per_segment} frames saved")

    print(f"\nDone! Route written to: {route_dir}")
    print(f"\nNote: route_3 does not have telemetry/ folders yet.")
    print(f"If you have telemetry from the comma 3x for this route,")
    print(f"copy it into each segment's telemetry/ directory.")


if __name__ == "__main__":
    main()
