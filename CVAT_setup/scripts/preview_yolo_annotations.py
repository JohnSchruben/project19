#!/usr/bin/env python3
"""
Draw YOLO annotation boxes onto images for quick visual inspection.

Example:
  python3 CVAT_setup/scripts/preview_yolo_annotations.py \
    datasets/route_3/segment_00/raw \
    --labels-dir datasets/route_3/segment_00/local_yolo_annotations/labels \
    --classes-file datasets/route_3/segment_00/local_yolo_annotations/classes.txt
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
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
    parser.add_argument("images_dir", help="Folder containing the original images")
    parser.add_argument("--labels-dir", required=True, help="Folder containing YOLO .txt labels")
    parser.add_argument("--classes-file", required=True, help="classes.txt file")
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


def load_classes(classes_file):
    path = Path(classes_file)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_images(images_dir, limit):
    images = sorted(
        path for path in Path(images_dir).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
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
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    classes = load_classes(args.classes_file)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = labels_dir.parent / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(images_dir, args.limit)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    font = ImageFont.load_default()
    total_boxes = 0

    print(f"[*] Rendering {len(images)} preview images to {output_dir}")
    for image_path in images:
        out_path, box_count = render_preview(image_path, labels_dir, classes, output_dir, font)
        total_boxes += box_count
        print(f"{image_path.name}: {box_count} boxes -> {out_path}")

    print(f"\n[SUCCESS] Wrote {len(images)} preview images with {total_boxes} boxes.")


if __name__ == "__main__":
    main()
