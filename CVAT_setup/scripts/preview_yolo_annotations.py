#!/usr/bin/env python3
"""
Draw YOLO annotation boxes onto images for quick visual inspection.

Example:
  python3 CVAT_setup/scripts/preview_yolo_annotations.py \
    datasets/route_3/segment_00
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CAMERA_DIRS = ("raw", "raw_front", "raw_left", "raw_right")
COLORS = [
    "#e53935",
    "#1e88e5",
    "#43a047",
    "#fb8c00",
    "#8e24aa",
    "#00acc1",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Render YOLO annotations onto image copies.")
    parser.add_argument("images_dir", help="Image folder or segment folder containing raw camera folders")
    parser.add_argument("--labels-dir", default=None, help="Folder containing YOLO .txt labels")
    parser.add_argument("--classes-file", default=None, help="classes.txt file")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write preview images. Default: <labels parent>/preview",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only render the first N images",
    )
    return parser.parse_args()


def images_in_dir(images_dir):
    return sorted(
        path for path in Path(images_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def discover_image_sets(input_dir):
    input_dir = Path(input_dir)
    direct_images = images_in_dir(input_dir)
    if direct_images:
        camera_name = input_dir.name if input_dir.name in CAMERA_DIRS else "images"
        return input_dir, [(camera_name, input_dir, direct_images)], False

    image_sets = []
    for camera_name in CAMERA_DIRS:
        camera_dir = input_dir / camera_name
        if camera_dir.is_dir():
            images = images_in_dir(camera_dir)
            if images:
                image_sets.append((camera_name, camera_dir, images))

    if not image_sets:
        raise SystemExit(f"No images found in {input_dir} or camera folders")
    return input_dir, image_sets, True


def resolve_annotation_dirs(args, base_dir, camera_name, multi_camera):
    if args.labels_dir:
        labels_dir = Path(args.labels_dir)
    elif multi_camera:
        labels_dir = base_dir / "local_yolo_annotations" / camera_name / "labels"
    else:
        labels_dir = base_dir.parent / "local_yolo_annotations" / "labels"

    if args.classes_file:
        classes_file = Path(args.classes_file)
    elif multi_camera:
        root_classes = base_dir / "local_yolo_annotations" / "classes.txt"
        camera_classes = base_dir / "local_yolo_annotations" / camera_name / "classes.txt"
        classes_file = root_classes if root_classes.exists() else camera_classes
    else:
        classes_file = labels_dir.parent / "classes.txt"

    if args.output_dir:
        output_root = Path(args.output_dir)
        output_dir = output_root / camera_name if multi_camera else output_root
    elif multi_camera:
        output_dir = base_dir / "local_yolo_annotations" / camera_name / "preview"
    else:
        output_dir = labels_dir.parent / "preview"

    return labels_dir, classes_file, output_dir


def load_classes(classes_file):
    path = Path(classes_file)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_images(images_dir, limit):
    images = images_in_dir(images_dir)
    if limit is not None:
        images = images[:limit]
    return images


def read_yolo_labels(label_path):
    if not label_path.exists():
        return []

    labels = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            print(f"[WARN] Skipping malformed label line {label_path}:{line_number}")
            continue

        class_id, x_center, y_center, width, height = parts
        labels.append(
            (
                int(class_id),
                float(x_center),
                float(y_center),
                float(width),
                float(height),
            )
        )
    return labels


def yolo_to_xyxy(label, image_width, image_height):
    class_id, x_center, y_center, width, height = label
    box_width = width * image_width
    box_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    x1 = center_x - box_width / 2.0
    y1 = center_y - box_height / 2.0
    x2 = center_x + box_width / 2.0
    y2 = center_y + box_height / 2.0
    return class_id, x1, y1, x2, y2


def draw_label(draw, xy, text, color, font):
    x1, y1, _, _ = xy
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    label_width = right - left + 8
    label_height = bottom - top + 6
    label_y = max(0, y1 - label_height)
    draw.rectangle((x1, label_y, x1 + label_width, label_y + label_height), fill=color)
    draw.text((x1 + 4, label_y + 3), text, fill="white", font=font)


def render_preview(image_path, labels_dir, classes, output_dir, font):
    label_path = labels_dir / f"{image_path.stem}.txt"
    labels = read_yolo_labels(label_path)

    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size

        for label in labels:
            class_id, x1, y1, x2, y2 = yolo_to_xyxy(label, image_width, image_height)
            color = COLORS[class_id % len(COLORS)]
            class_name = classes[class_id] if class_id < len(classes) else str(class_id)

            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            draw_label(draw, (x1, y1, x2, y2), class_name, color, font)

        out_path = output_dir / f"{image_path.stem}_annotated.jpg"
        image.save(out_path, quality=95)
        return out_path, len(labels)


def main():
    args = parse_args()
    base_dir, image_sets, multi_camera = discover_image_sets(Path(args.images_dir))
    font = ImageFont.load_default()
    total_boxes = 0
    total_images = 0

    for camera_name, images_dir, images in image_sets:
        labels_dir, classes_file, output_dir = resolve_annotation_dirs(
            args, base_dir, camera_name, multi_camera
        )
        classes = load_classes(classes_file)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.limit is not None:
            images = images[:args.limit]
        total_images += len(images)

        print(f"[*] Rendering {len(images)} {camera_name} preview images to {output_dir}")
        for image_path in images:
            out_path, box_count = render_preview(image_path, labels_dir, classes, output_dir, font)
            total_boxes += box_count
            print(f"{camera_name}/{image_path.name}: {box_count} boxes -> {out_path}")

    print(f"\n[SUCCESS] Wrote {total_images} preview images with {total_boxes} boxes.")


if __name__ == "__main__":
    main()
