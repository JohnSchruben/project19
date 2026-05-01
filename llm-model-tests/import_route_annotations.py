#!/usr/bin/env python3
"""
Alternate database importer for route datasets and local YOLO annotations.

This intentionally does not replace import_annotations.py, which is the original
aspave/CVAT-era importer.

Typical usage from /workspace/project19:

  python3 llm-model-tests/import_route_annotations.py \
    --frames-dir datasets/route_3/segment_00/raw \
    --db llm-model-tests/annotations.db \
    --source route_3_segment_00 \
    --yolo-labels-dir datasets/route_3/segment_00/local_yolo_annotations/labels \
    --classes-file datasets/route_3/segment_00/local_yolo_annotations/classes.txt \
    --create-placeholder-annotations

The existing DB schema stores high-level label categories, not bounding boxes.
This script converts YOLO boxes into "present" booleans per frame.
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from dataset_manager import DatasetManager, VALID_CATEGORIES


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import route frames and optional local YOLO labels into annotations.db"
    )
    parser.add_argument("--frames-dir", required=True, help="Directory containing route images")
    parser.add_argument("--db", default="llm-model-tests/annotations.db", help="SQLite DB path")
    parser.add_argument("--source", required=True, help="Source name, e.g. route_3_segment_00")
    parser.add_argument(
        "--annotations-json",
        default=None,
        help="Optional driving annotation JSON with filename/scene/steering/throttle/brake/labels",
    )
    parser.add_argument(
        "--yolo-labels-dir",
        default=None,
        help="Optional YOLO labels directory produced by local_yolo_annotate.py",
    )
    parser.add_argument(
        "--classes-file",
        default=None,
        help="Optional classes.txt for YOLO labels",
    )
    parser.add_argument(
        "--create-placeholder-annotations",
        action="store_true",
        help="Create one annotation per frame using default driving values so YOLO labels can be stored",
    )
    parser.add_argument("--default-description", default="", help="Placeholder scene_description")
    parser.add_argument("--default-steering", type=float, default=0.0)
    parser.add_argument("--default-throttle", type=float, default=0.0)
    parser.add_argument("--default-brake", type=float, default=0.0)
    parser.add_argument(
        "--keep-original-filenames",
        action="store_true",
        help="Store bare filenames like 000000.png. Default stores source/filename to avoid route collisions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def image_dimensions(image_path):
    if not PIL_AVAILABLE:
        return None, None
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as exc:
        print(f"  Warning: could not read dimensions for {image_path}: {exc}")
        return None, None


def frame_number_from_name(filename):
    stem = Path(filename).stem
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def list_images(frames_dir):
    frames_path = Path(frames_dir).resolve()
    if not frames_path.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_path}")

    images = sorted(
        p for p in frames_path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(f"No images found in: {frames_path}")
    return images


def stored_filename(image_path, source, keep_original):
    if keep_original:
        return image_path.name
    return f"{source}/{image_path.name}"


def relative_path_from_db(image_path, db_path):
    db_dir = Path(db_path).resolve().parent
    return os.path.relpath(image_path.resolve(), db_dir).replace("\\", "/")


def import_frames(db, images, db_path, source, keep_original):
    frame_map = {}
    for image_path in images:
        width, height = image_dimensions(image_path)
        filename = stored_filename(image_path, source, keep_original)
        frame_id = db.add_frame(
            filename=filename,
            relative_path=relative_path_from_db(image_path, db_path),
            width=width,
            height=height,
            source=source,
            frame_number=frame_number_from_name(image_path.name),
        )
        frame_map[image_path.name] = frame_id
        frame_map[filename] = frame_id
        print(f"  [frame={frame_id:4d}] {filename}")
    return frame_map


def load_classes(classes_file):
    if not classes_file:
        return []
    path = Path(classes_file)
    if not path.exists():
        raise FileNotFoundError(f"Classes file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_yolo_presence(image_name, labels_dir, classes):
    if not labels_dir:
        return {}

    label_path = Path(labels_dir) / f"{Path(image_name).stem}.txt"
    presence = {}
    if not label_path.exists():
        return presence

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            print(f"  Warning: skipping malformed YOLO line {label_path}:{line_number}")
            continue
        class_id = int(parts[0])
        if class_id >= len(classes):
            print(f"  Warning: class id {class_id} missing from classes file")
            continue
        category = classes[class_id]
        if category in VALID_CATEGORIES:
            presence[category] = True
    return presence


def load_annotation_json(json_path):
    if not json_path:
        return {}
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("annotations JSON must contain a top-level array")
    return {entry["filename"]: entry for entry in data if "filename" in entry}


def normalize_label_value(value):
    if isinstance(value, dict):
        return bool(value.get("present", False)), value.get("confidence")
    return bool(value), None


def add_labels(db, annotation_id, labels):
    for category, value in labels.items():
        if category not in VALID_CATEGORIES:
            print(f"  Warning: skipping unsupported category {category!r}")
            continue
        present, confidence = normalize_label_value(value)
        db.add_label_category(annotation_id, category, present, confidence)


def create_annotation_for_frame(db, frame_id, entry, args, yolo_presence):
    if entry:
        annotation_id = db.add_annotation(
            frame_id=frame_id,
            scene_description=entry.get("scene_description", ""),
            steering_angle_deg=float(entry["steering_angle_deg"]),
            throttle=float(entry["throttle"]),
            brake=float(entry["brake"]),
            annotation_source=entry.get("annotation_source", "manual"),
            annotated_at=entry.get("annotated_at"),
        )
        labels = dict(entry.get("labels", {}))
    else:
        annotation_id = db.add_annotation(
            frame_id=frame_id,
            scene_description=args.default_description,
            steering_angle_deg=args.default_steering,
            throttle=args.default_throttle,
            brake=args.default_brake,
            annotation_source="local_yolo",
        )
        labels = {}

    for category in yolo_presence:
        labels[category] = {"present": True, "confidence": None}

    add_labels(db, annotation_id, labels)
    return annotation_id


def main():
    args = parse_args()

    if args.yolo_labels_dir and not args.classes_file:
        raise SystemExit("--classes-file is required when --yolo-labels-dir is used")

    images = list_images(args.frames_dir)
    classes = load_classes(args.classes_file)
    annotation_entries = load_annotation_json(args.annotations_json)

    print("=" * 60)
    print("Route Dataset Database Import")
    print("=" * 60)
    print(f"Database   : {args.db}")
    print(f"Frames dir : {Path(args.frames_dir).resolve()}")
    print(f"Source     : {args.source}")
    if args.yolo_labels_dir:
        print(f"YOLO labels: {Path(args.yolo_labels_dir).resolve()}")
    if args.annotations_json:
        print(f"JSON labels: {Path(args.annotations_json).resolve()}")
    if args.dry_run:
        print("DRY RUN -- no changes will be written.")
        print(f"Would import {len(images)} image(s).")
        return

    with DatasetManager(args.db) as db:
        print("\n--- Registering frames ---")
        frame_map = import_frames(
            db=db,
            images=images,
            db_path=args.db,
            source=args.source,
            keep_original=args.keep_original_filenames,
        )

        should_create_annotations = (
            bool(annotation_entries)
            or bool(args.yolo_labels_dir and args.create_placeholder_annotations)
        )

        inserted_annotations = 0
        if should_create_annotations:
            print("\n--- Importing annotations ---")
            for image_path in images:
                entry = annotation_entries.get(image_path.name)
                yolo_presence = load_yolo_presence(image_path.name, args.yolo_labels_dir, classes)

                if not entry and not yolo_presence and not args.create_placeholder_annotations:
                    continue

                frame_id = frame_map[image_path.name]
                annotation_id = create_annotation_for_frame(db, frame_id, entry, args, yolo_presence)
                inserted_annotations += 1
                print(f"  [ann={annotation_id:4d}] {image_path.name}")

        print("\n--- Database summary ---")
        stats = db.get_stats()
        print(f"  Frames      : {stats['total_frames']}")
        print(f"  Annotations : {stats['total_annotations']}")
        if stats["by_source"]:
            print(f"  By source   : {stats['by_source']}")
        if stats["label_counts"]:
            print("  Label counts (present/total):")
            for category, counts in stats["label_counts"].items():
                print(f"    {category:<15}: {counts['present']}/{counts['total']}")

    print(f"\nDone. Imported {len(images)} frame(s) and {inserted_annotations} annotation row(s).")


if __name__ == "__main__":
    main()
