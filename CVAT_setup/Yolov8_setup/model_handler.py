from typing import Dict, List

import numpy as np
from ultralytics import YOLO


class ModelHandler:
    def __init__(self, labels: Dict[int, str], weights_path: str):
        self.labels = labels
        self.model = YOLO(weights_path)

        self.class_map = {
            "traffic light": "traffic_light",
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle",
            "motorcycle": "vehicle",
            "bicycle": "vehicle",
            "person": "pedestrian",
            "stop sign": "stop_sign",
        }

    def infer(self, image, threshold: float) -> List[dict]:
        results = self.model.predict(image, conf=threshold, verbose=False)
        output = []

        if not results:
            return output

        boxes = results[0].boxes
        if boxes is None:
            return output

        names = results[0].names

        for box in boxes:
            cls_idx = int(box.cls.item())
            raw_label = names.get(cls_idx, str(cls_idx))
            mapped_label = self.class_map.get(raw_label)

            if mapped_label is None:
                continue

            score = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            output.append(
                {
                    "confidence": f"{score}",
                    "label": mapped_label,
                    "points": [
                        int(np.clip(x1, 0, image.width)),
                        int(np.clip(y1, 0, image.height)),
                        int(np.clip(x2, 0, image.width)),
                        int(np.clip(y2, 0, image.height)),
                    ],
                    "type": "rectangle",
                }
            )

        return output
