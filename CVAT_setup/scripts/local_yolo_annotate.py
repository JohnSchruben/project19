#!/usr/bin/env python3
"""
Run YOLO locally on a folder of images and export annotations without CVAT.

Outputs:
  - YOLO label txt files
  - COCO JSON
  - CVAT-style XML

Default target labels:
  0 pedestrian
  1 vehicle
  2 traffic_light
  3 stop_sign
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install pillow", file=sys.stderr)
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is required. Install with: pip install ultralytics", file=sys.stderr)
    sys.exit(1)


TARGET_LABELS = ["pedestrian", "vehicle", "traffic_light", "stop_sign"]
TARGET_LABEL_TO_ID = {name: idx for idx, name in enumerate(TARGET_LABELS)}

CLASS_MAP = {
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "stop sign": "stop_sign",
    "stop_sign": "stop_sign",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate images locally with YOLO and export YOLO/COCO/CVAT files."
    )
    parser.add_argument("folder", help="Folder containing images to annotate")
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Ultralytics model path/name. Default: yolov8s.pt",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory. Default: <image folder>/../local_yolo_annotations",
    )
    parser.add_argument(
        "--confidence",
        "--conf",
        type=float,
        default=0.5,
        help="Detection confidence threshold. Default: 0.5",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold. Default: 0.7",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. 0, cpu, cuda. Default: ultralytics auto-select.",
    )
    parser.add_argument(
        "--no-empty-labels",
        action="store_true",
        help="Do not create empty YOLO txt files for images with no detections.",
    )
    return parser.parse_args()


def list_images(folder):
    image_dir = Path(folder).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    return image_dir, images


def make_output_dir(image_dir, output_dir):
    if output_dir:
        out_dir = Path(output_dir).expanduser().resolve()
    else:
        out_dir = image_dir.parent / "local_yolo_annotations"

    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, labels_dir


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def detect_image(model, image_path, confidence, iou, device):
    kwargs = {
        "source": str(image_path),
        "conf": confidence,
        "iou": iou,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device

    results = model.predict(**kwargs)
    if not results:
        return []

    result = results[0]
    if result.boxes is None:
        return []

    detections = []
    names = result.names

    with Image.open(image_path) as img:
        width, height = img.size

    for box in result.boxes:
        raw_class_id = int(box.cls.item())
        raw_label = names.get(raw_class_id, str(raw_class_id))
        target_label = CLASS_MAP.get(raw_label)
        if target_label is None:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = clamp(float(x1), 0.0, float(width))
        y1 = clamp(float(y1), 0.0, float(height))
        x2 = clamp(float(x2), 0.0, float(width))
        y2 = clamp(float(y2), 0.0, float(height))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "label": target_label,
                "class_id": TARGET_LABEL_TO_ID[target_label],
                "confidence": float(box.conf.item()),
                "bbox_xyxy": [x1, y1, x2, y2],
                "width": width,
                "height": height,
            }
        )

    return detections


def write_yolo_label(label_path, detections):
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        width = det["width"]
        height = det["height"]

        box_w = x2 - x1
        box_h = y2 - y1
        x_center = x1 + box_w / 2.0
        y_center = y1 + box_h / 2.0

        lines.append(
            f"{det['class_id']} "
            f"{x_center / width:.8f} "
            f"{y_center / height:.8f} "
            f"{box_w / width:.8f} "
            f"{box_h / height:.8f}"
        )

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build_coco(images, detections_by_image):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": idx, "name": name, "supercategory": "object"}
            for idx, name in enumerate(TARGET_LABELS)
        ],
    }

    ann_id = 1
    for image_id, image_path in enumerate(images, start=1):
        detections = detections_by_image[image_path]
        if detections:
            width = detections[0]["width"]
            height = detections[0]["height"]
        else:
            with Image.open(image_path) as img:
                width, height = img.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            box_w = x2 - x1
            box_h = y2 - y1
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": det["class_id"],
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "score": det["confidence"],
                }
            )
            ann_id += 1

    return coco


def build_cvat_xml(images, detections_by_image):
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = "local_yolo_annotations"
    labels_node = ET.SubElement(task, "labels")
    for label_name in TARGET_LABELS:
        label_node = ET.SubElement(labels_node, "label")
        ET.SubElement(label_node, "name").text = label_name
        ET.SubElement(label_node, "color").text = "#000000"
        ET.SubElement(label_node, "type").text = "rectangle"
        ET.SubElement(label_node, "attributes")

    for image_id, image_path in enumerate(images):
        detections = detections_by_image[image_path]
        if detections:
            width = detections[0]["width"]
            height = detections[0]["height"]
        else:
            with Image.open(image_path) as img:
                width, height = img.size

        image_node = ET.SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": image_path.name,
                "width": str(width),
                "height": str(height),
            },
        )

        for det in detections:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            ET.SubElement(
                image_node,
                "box",
                {
                    "label": det["label"],
                    "source": "auto",
                    "occluded": "0",
                    "xtl": f"{x1:.2f}",
                    "ytl": f"{y1:.2f}",
                    "xbr": f"{x2:.2f}",
                    "ybr": f"{y2:.2f}",
                    "z_order": "0",
                },
            )

    ET.indent(root, space="  ")
    return ET.ElementTree(root)


def write_labels_file(output_dir):
    labels_path = output_dir / "classes.txt"
    labels_path.write_text("\n".join(TARGET_LABELS) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    image_dir, images = list_images(args.folder)
    output_dir, labels_dir = make_output_dir(image_dir, args.output_dir)

    print(f"[*] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[*] Found {len(images)} images in {image_dir}")
    print(f"[*] Writing annotations to {output_dir}")

    detections_by_image = {}
    total_boxes = 0

    for index, image_path in enumerate(images, start=1):
        detections = detect_image(
            model=model,
            image_path=image_path,
            confidence=args.confidence,
            iou=args.iou,
            device=args.device,
        )
        detections_by_image[image_path] = detections
        total_boxes += len(detections)

        label_path = labels_dir / f"{image_path.stem}.txt"
        if detections or not args.no_empty_labels:
            write_yolo_label(label_path, detections)

        print(f"[{index:04d}/{len(images):04d}] {image_path.name}: {len(detections)} boxes")

    write_labels_file(output_dir)

    coco = build_coco(images, detections_by_image)
    coco_path = output_dir / "annotations_coco.json"
    coco_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    cvat_xml_path = output_dir / "annotations_cvat.xml"
    cvat_tree = build_cvat_xml(images, detections_by_image)
    cvat_tree.write(cvat_xml_path, encoding="utf-8", xml_declaration=True)

    summary = {
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "confidence": args.confidence,
        "images": len(images),
        "boxes": total_boxes,
        "labels": TARGET_LABELS,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[SUCCESS] Local annotation complete")
    print(f"Images:       {len(images)}")
    print(f"Boxes:        {total_boxes}")
    print(f"YOLO labels:  {labels_dir}")
    print(f"COCO JSON:    {coco_path}")
    print(f"CVAT XML:     {cvat_xml_path}")
    print(f"Classes:      {output_dir / 'classes.txt'}")


if __name__ == "__main__":
    main()
