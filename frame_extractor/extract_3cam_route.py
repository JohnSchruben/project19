"""
Extract frames from the three dashcam videos into a route dataset structure.

This script supports two alignment strategies:
  1. index: align by route frame index, with a front pad and a linear sample
     across the chosen source-video frame window.
  2. telemetry: align by route telemetry timestamps and one or two sync anchors.

Examples:
  python extract_3cam_route.py
  python extract_3cam_route.py --sync-mode index --sync-offset 23
  python extract_3cam_route.py --sync-mode telemetry --route-sync-frame 86 --video-sync-frame 63
  python extract_3cam_route.py --sync-mode telemetry --route-sync-frame 86 --video-sync-frame 63 \
      --route-sync-frame-2 1500 --video-sync-frame-2 4960
  python extract_3cam_route.py --dry-run
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
VIDEOS_DIR = os.path.join(SCRIPT_DIR, "videos")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Camera mapping: (video filename prefix, output directory name)
CAMERAS = [
    ("cam0", "raw_left"),
    ("cam1", "raw_front"),
    ("cam2", "raw_right"),
]


def find_video(prefix: str) -> Optional[str]:
    """Find the first video file that matches a camera prefix."""
    for name in sorted(os.listdir(VIDEOS_DIR)):
        if name.startswith(prefix) and name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            return os.path.join(VIDEOS_DIR, name)
    return None


def get_video_info(path: str) -> Tuple[int, float, int, int]:
    """Return total frames, fps, width, and height for a video."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total <= 0 or fps <= 0:
        raise RuntimeError(f"Video metadata is invalid for {path}")

    return total, fps, width, height


def extract_frames(video_path: str, frame_indices: List[Optional[int]]) -> Dict[int, np.ndarray]:
    """
    Extract a unique set of frames from a video by index.

    Returns a map of frame_index -> BGR frame.
    """
    requested = sorted({idx for idx in frame_indices if idx is not None})
    if not requested:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_map: Dict[int, np.ndarray] = {}
    request_set = set(requested)
    max_idx = requested[-1]
    current = 0

    while current <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break

        if current in request_set:
            frame_map[current] = frame

        current += 1

    cap.release()
    return frame_map


def count_frames(directory: str) -> int:
    """Count image files in a directory."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1
        for name in os.listdir(directory)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )


def count_json(directory: str) -> int:
    """Count json files in a directory."""
    if not os.path.isdir(directory):
        return 0
    return sum(1 for name in os.listdir(directory) if name.lower().endswith(".json"))


def list_route_segments(route_dir: str) -> List[dict]:
    """Discover route segments and their real frame counts."""
    segments: List[dict] = []

    for name in sorted(os.listdir(route_dir)):
        seg_dir = os.path.join(route_dir, name)
        if not (name.startswith("segment_") and os.path.isdir(seg_dir)):
            continue

        raw_dir = os.path.join(seg_dir, "raw")
        telemetry_dir = os.path.join(seg_dir, "telemetry")
        raw_count = count_frames(raw_dir)
        telemetry_count = count_json(telemetry_dir)

        if raw_count and telemetry_count and raw_count != telemetry_count:
            print(
                f"WARNING: {name} raw count ({raw_count}) does not match telemetry "
                f"count ({telemetry_count}); using the smaller count."
            )

        frame_count = raw_count or telemetry_count
        if raw_count and telemetry_count:
            frame_count = min(raw_count, telemetry_count)

        if frame_count == 0:
            print(f"WARNING: skipping {name} because it has no raw frames or telemetry")
            continue

        segments.append(
            {
                "name": name,
                "dir": seg_dir,
                "frame_count": frame_count,
                "telemetry_dir": telemetry_dir if telemetry_count else None,
            }
        )

    if not segments:
        raise RuntimeError(f"No usable segment_* directories found in {route_dir}")

    return segments


def load_route_times(segments: List[dict], fallback_route_fps: Optional[float]) -> Tuple[List[float], str]:
    """
    Load route-relative timestamps in seconds for every target frame.

    Uses telemetry when available. If telemetry is missing for every segment,
    falls back to a constant frame rate when requested.
    """
    if all(segment["telemetry_dir"] for segment in segments):
        timestamps: List[float] = []

        for segment in segments:
            telemetry_dir = segment["telemetry_dir"]
            for local_idx in range(segment["frame_count"]):
                path = os.path.join(telemetry_dir, f"{local_idx:06d}.json")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing telemetry file for {segment['name']} frame {local_idx:06d}: {path}"
                    )

                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)

                if "timestamp_seconds" not in data:
                    raise KeyError(f"timestamp_seconds missing from {path}")

                timestamps.append(float(data["timestamp_seconds"]))

        first_timestamp = timestamps[0]
        route_times = [value - first_timestamp for value in timestamps]
        for idx in range(1, len(route_times)):
            if route_times[idx] < route_times[idx - 1]:
                raise RuntimeError(
                    "Route telemetry timestamps are not monotonic. "
                    f"Time went backwards at global frame {idx}."
                )
        return route_times, "telemetry timestamps"

    if fallback_route_fps is None or fallback_route_fps <= 0:
        raise RuntimeError(
            "Telemetry is missing for part of the route. Pass --fallback-route-fps "
            "to use a fixed route frame rate."
        )

    total_frames = sum(segment["frame_count"] for segment in segments)
    return (
        [frame_idx / fallback_route_fps for frame_idx in range(total_frames)],
        f"uniform {fallback_route_fps:.3f} FPS fallback",
    )


def validate_anchor(name: str, frame_idx: int, total_frames: int) -> None:
    """Ensure a sync anchor points at a valid route frame."""
    if frame_idx < 0 or frame_idx >= total_frames:
        raise ValueError(f"{name} must be between 0 and {total_frames - 1}, got {frame_idx}")


def build_video_frame_map(
    route_times: List[float],
    video_fps: float,
    video_total_frames: int,
    route_anchor_time_1: float,
    video_anchor_frame_1: int,
    route_anchor_time_2: Optional[float] = None,
    video_anchor_frame_2: Optional[int] = None,
) -> Tuple[List[Optional[int]], int, float, float]:
    """
    Map each route frame time to a video frame index.

    Returns:
      frame_indices
      black_frame_count
      time_scale
      video_time_at_route_start
    """
    video_anchor_time_1 = video_anchor_frame_1 / video_fps
    time_scale = 1.0

    if route_anchor_time_2 is not None and video_anchor_frame_2 is not None:
        route_span = route_anchor_time_2 - route_anchor_time_1
        if route_span == 0:
            raise ValueError("Route sync anchors resolve to the same timestamp")

        video_anchor_time_2 = video_anchor_frame_2 / video_fps
        time_scale = (video_anchor_time_2 - video_anchor_time_1) / route_span

    max_video_time = (video_total_frames - 1) / video_fps
    frame_indices: List[Optional[int]] = []
    black_frames = 0

    for route_time in route_times:
        video_time = video_anchor_time_1 + (route_time - route_anchor_time_1) * time_scale

        if video_time < 0.0 or video_time > max_video_time:
            frame_indices.append(None)
            black_frames += 1
            continue

        frame_idx = int(round(video_time * video_fps))
        frame_idx = min(max(frame_idx, 0), video_total_frames - 1)
        frame_indices.append(frame_idx)

    video_time_at_route_start = video_anchor_time_1 - (route_anchor_time_1 * time_scale)
    return frame_indices, black_frames, time_scale, video_time_at_route_start


def build_index_frame_map(
    total_route_frames: int,
    video_total_frames: int,
    sync_offset: int,
    source_start_frame: int,
    source_end_frame: Optional[int],
) -> Tuple[List[Optional[int]], int, int, int, float]:
    """
    Map route frame indices directly onto a source-video frame window.

    Returns:
      frame_indices
      black_frame_count
      source_start_frame
      source_end_frame
      average_step
    """
    if sync_offset < 0:
        raise ValueError("--sync-offset must be non-negative")

    active_frames = total_route_frames - sync_offset
    if active_frames <= 0:
        raise ValueError("--sync-offset is larger than the number of route frames")

    if source_end_frame is None:
        source_end_frame = video_total_frames - 1

    if source_start_frame < 0 or source_start_frame >= video_total_frames:
        raise ValueError(
            f"--source-start-frame must be between 0 and {video_total_frames - 1}"
        )

    if source_end_frame < 0 or source_end_frame >= video_total_frames:
        raise ValueError(
            f"--source-end-frame must be between 0 and {video_total_frames - 1}"
        )

    if source_end_frame < source_start_frame:
        raise ValueError("--source-end-frame must be greater than or equal to --source-start-frame")

    sampled = np.linspace(
        source_start_frame,
        source_end_frame,
        active_frames,
        dtype=int,
    ).tolist()
    average_step = 0.0 if active_frames <= 1 else (source_end_frame - source_start_frame) / (active_frames - 1)
    return ([None] * sync_offset) + sampled, sync_offset, source_start_frame, source_end_frame, average_step


def clear_output_frames(directory: str) -> int:
    """Remove previously generated image files from an output directory."""
    os.makedirs(directory, exist_ok=True)
    removed = 0

    for name in os.listdir(directory):
        if name.lower().endswith(IMAGE_EXTENSIONS):
            os.remove(os.path.join(directory, name))
            removed += 1

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 3-camera frames into a route dataset."
    )
    parser.add_argument(
        "--route",
        type=str,
        default="route_3",
        help="Route directory name under datasets/ (default: route_3)",
    )
    parser.add_argument(
        "--sync-mode",
        choices=("index", "telemetry"),
        default="index",
        help="Alignment mode to use (default: index)",
    )
    parser.add_argument(
        "--sync-offset",
        type=int,
        default=23,
        help="Leading black frames for index mode (default: 23)",
    )
    parser.add_argument(
        "--source-start-frame",
        type=int,
        default=0,
        help="First source video frame to sample in index mode (default: 0)",
    )
    parser.add_argument(
        "--source-end-frame",
        type=int,
        default=None,
        help="Last source video frame to sample in index mode (default: last frame)",
    )
    parser.add_argument(
        "--route-sync-frame",
        type=int,
        default=86,
        help="Global route frame index for the primary sync anchor (default: 86)",
    )
    parser.add_argument(
        "--video-sync-frame",
        type=int,
        default=63,
        help="Video frame index for the primary sync anchor (default: 63)",
    )
    parser.add_argument(
        "--route-sync-frame-2",
        type=int,
        default=None,
        help="Optional second global route frame anchor for drift correction",
    )
    parser.add_argument(
        "--video-sync-frame-2",
        type=int,
        default=None,
        help="Optional second video frame anchor for drift correction",
    )
    parser.add_argument(
        "--fallback-route-fps",
        type=float,
        default=None,
        help="Use a fixed route frame rate when telemetry is unavailable",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the sync mapping without writing files",
    )
    args = parser.parse_args()

    route_dir = os.path.join(DATASETS_DIR, args.route)
    if not os.path.isdir(route_dir):
        print(f"ERROR: route directory does not exist: {route_dir}")
        sys.exit(1)

    segments = list_route_segments(route_dir)
    total_frames = sum(segment["frame_count"] for segment in segments)

    print("=" * 60)
    print("3-Camera Route Extraction")
    print("=" * 60)
    print(f"Route: {route_dir}")
    print(f"Segments: {len(segments)}")
    print(
        "Frames per segment: "
        + ", ".join(f"{segment['name']}={segment['frame_count']}" for segment in segments)
    )
    print(f"Total route frames: {total_frames}")
    print(f"Sync mode: {args.sync_mode}")

    video_paths: Dict[str, str] = {}
    video_infos: Dict[str, Tuple[int, float, int, int]] = {}

    for prefix, _ in CAMERAS:
        path = find_video(prefix)
        if path is None:
            print(f"ERROR: no video found for '{prefix}' in {VIDEOS_DIR}")
            sys.exit(1)

        info = get_video_info(path)
        video_paths[prefix] = path
        video_infos[prefix] = info

        total, fps, width, height = info
        print(
            f"  {prefix}: {os.path.basename(path)}  "
            f"{total} frames, {fps:.3f} FPS, {width}x{height}, {total / fps:.3f}s"
        )

    mapping_by_camera: Dict[str, List[Optional[int]]] = {}

    if args.sync_mode == "telemetry":
        second_anchor_supplied = (
            args.route_sync_frame_2 is not None or args.video_sync_frame_2 is not None
        )
        if second_anchor_supplied and (
            args.route_sync_frame_2 is None or args.video_sync_frame_2 is None
        ):
            print("ERROR: pass both --route-sync-frame-2 and --video-sync-frame-2 together")
            sys.exit(1)

        route_times, time_source = load_route_times(segments, args.fallback_route_fps)
        total_duration = route_times[-1] if route_times else 0.0
        validate_anchor("--route-sync-frame", args.route_sync_frame, total_frames)
        route_anchor_time_1 = route_times[args.route_sync_frame]

        route_anchor_time_2 = None
        if args.route_sync_frame_2 is not None:
            validate_anchor("--route-sync-frame-2", args.route_sync_frame_2, total_frames)
            route_anchor_time_2 = route_times[args.route_sync_frame_2]

        print(f"Route timing source: {time_source}")
        print(f"Route duration: {total_duration:.3f} s")
        print(
            f"Primary sync anchor: route frame {args.route_sync_frame} "
            f"at {route_anchor_time_1:.3f}s -> video frame {args.video_sync_frame}"
        )
        if route_anchor_time_2 is not None:
            print(
                f"Secondary sync anchor: route frame {args.route_sync_frame_2} "
                f"at {route_anchor_time_2:.3f}s -> video frame {args.video_sync_frame_2}"
            )

        for prefix, dir_name in CAMERAS:
            total, fps, _, _ = video_infos[prefix]
            frame_map, black_count, scale, video_time_at_route_start = build_video_frame_map(
                route_times=route_times,
                video_fps=fps,
                video_total_frames=total,
                route_anchor_time_1=route_anchor_time_1,
                video_anchor_frame_1=args.video_sync_frame,
                route_anchor_time_2=route_anchor_time_2,
                video_anchor_frame_2=args.video_sync_frame_2,
            )
            mapping_by_camera[prefix] = frame_map

            used_indices = [idx for idx in frame_map if idx is not None]
            first_used = min(used_indices) if used_indices else None
            last_used = max(used_indices) if used_indices else None
            unique_used = len(set(used_indices))

            print(f"\n{prefix} -> {dir_name}: time scale {scale:.6f}")
            print(f"  route frame 0 maps to video t={video_time_at_route_start:.3f}s")
            print(
                f"  uses {unique_used} unique video frames, "
                f"black frames required: {black_count}"
            )
            if first_used is not None:
                print(f"  mapped video frame range: {first_used}..{last_used}")
    else:
        print(
            f"Index mode settings: sync_offset={args.sync_offset}, "
            f"source_start_frame={args.source_start_frame}, "
            f"source_end_frame={'last' if args.source_end_frame is None else args.source_end_frame}"
        )

        for prefix, dir_name in CAMERAS:
            total, fps, _, _ = video_infos[prefix]
            frame_map, black_count, first_used, last_used, average_step = build_index_frame_map(
                total_route_frames=total_frames,
                video_total_frames=total,
                sync_offset=args.sync_offset,
                source_start_frame=args.source_start_frame,
                source_end_frame=args.source_end_frame,
            )
            mapping_by_camera[prefix] = frame_map

            unique_used = len({idx for idx in frame_map if idx is not None})
            print(f"\n{prefix} -> {dir_name}:")
            print(
                f"  black frames required: {black_count}, "
                f"average source step: {average_step:.3f}"
            )
            print(f"  mapped video frame range: {first_used}..{last_used} at {fps:.3f} FPS")
            print(f"  uses {unique_used} unique video frames")

    if args.dry_run:
        print("\n[DRY RUN] No files were written.")
        return

    for prefix, dir_name in CAMERAS:
        video_path = video_paths[prefix]
        _, _, width, height = video_infos[prefix]
        mapped_indices = mapping_by_camera[prefix]

        print(f"\nExtracting {prefix} -> {dir_name}/ ...")
        extracted_frames = extract_frames(video_path, mapped_indices)
        blank_frame = np.zeros((height, width, 3), dtype=np.uint8)
        global_idx = 0

        for segment in segments:
            out_dir = os.path.join(segment["dir"], dir_name)
            removed = clear_output_frames(out_dir)
            if removed:
                print(f"  cleared {removed} old frames from {segment['name']}/{dir_name}")

            for local_idx in range(segment["frame_count"]):
                video_frame_idx = mapped_indices[global_idx]
                frame = extracted_frames.get(video_frame_idx) if video_frame_idx is not None else None
                out_path = os.path.join(out_dir, f"{local_idx:06d}.png")
                cv2.imwrite(out_path, blank_frame if frame is None else frame)
                global_idx += 1

            print(f"  {segment['name']}/{dir_name}: {segment['frame_count']} frames saved")

    print(f"\nDone. Route written to: {route_dir}")


if __name__ == "__main__":
    main()
