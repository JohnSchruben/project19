import base64
import io
import json
import os

import yaml
from model_handler import ModelHandler
from PIL import Image


def init_context(context):
    context.logger.info("Init context... 0%")

    with open("/opt/nuclio/function.yaml", "rb") as function_file:
        functionconfig = yaml.safe_load(function_file)

    labels_spec = functionconfig["metadata"]["annotations"]["spec"]
    labels = {item["id"]: item["name"] for item in json.loads(labels_spec)}

    weights_path = os.getenv("WEIGHTS_PATH", "/opt/nuclio/weights/yolov8s.pt")
    context.user_data.model = ModelHandler(labels, weights_path)

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YOLOv8 custom traffic model")
    data = event.body

    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf).convert("RGB")

    results = context.user_data.model.infer(image, threshold)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
