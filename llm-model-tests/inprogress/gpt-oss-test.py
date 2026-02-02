import base64
import io
import json
import os
from dataclasses import dataclass
from typing import Optional

from PIL import Image
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    # Transformers-serve default per cookbook: http://localhost:8000/v1/responses
    base_url: str = "http://localhost:8000/v1"
    model: str = "openai/gpt-oss-20b"

    # Resize to keep your pipeline snappy (also useful even if you later add a vision model)
    max_side_px: int = 768
    jpeg_quality: int = 85

    # If you have a separate vision step, put its caption/labels here (or pass via CLI)
    default_scene_text: str = ""


def load_and_resize_image(path: str, max_side_px: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side_px / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)))
    return img


def image_to_jpeg_bytes(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def ask_gpt_oss(
    client: OpenAI,
    model: str,
    scene_text: str,
) -> str:
    """
    gpt-oss is text-only: we send a textual scene description and ask for a driving decision.
    """
    system = (
        "You are a driving-scene reasoning assistant. "
        "Given a scene description, suggest a safe, lawful driving action. "
        "Be concise."
    )

    user = (
        "Scene description:\n"
        f"{scene_text}\n\n"
        "Return:\n"
        "1) A one-sentence recommended action (e.g., keep lane, slow down, stop, turn left/right).\n"
        "2) A short bullet list of the top 3 reasons."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_output_tokens=300,
    )

    # `responses` API returns output chunks; simplest is output_text helper:
    return resp.output_text


import argparse

def main():
    cfg = Config()

    parser = argparse.ArgumentParser(description="Run GPT-OSS test")
    parser.add_argument("--image", type=str, default="car-on-road.png", help="Path to the image file")
    parser.add_argument("--prompt", type=str, default=cfg.default_scene_text, help="Scene text/prompt")
    args = parser.parse_args()

    image_path = args.image
    scene_text = args.prompt.strip()

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize + compress (useful for your overall pipeline even though gpt-oss won't consume pixels)
    img = load_and_resize_image(image_path, cfg.max_side_px)
    jpg = image_to_jpeg_bytes(img, cfg.jpeg_quality)

    out_path = os.path.splitext(image_path)[0] + f".small_{cfg.max_side_px}.jpg"
    with open(out_path, "wb") as f:
        f.write(jpg)

    print(f"[OK] Wrote resized image: {out_path}  ({len(jpg)/1024:.1f} KB)")

    if not scene_text:
        print(
            "\n[NOTE] gpt-oss cannot read images directly.\n"
            "Set SCENE_TEXT to a caption/labels (from a vision model or your own notes), e.g.:\n"
            '  set SCENE_TEXT="Forward lane, solid lines, gentle right curve, no pedestrians, one car ahead."\n'
        )
        return

    client = OpenAI(
        base_url=cfg.base_url,
        api_key="ollama"  # not used by local transformers serve, but OpenAI SDK requires something
    )

    answer = ask_gpt_oss(client, cfg.model, scene_text)
    print("\n--- gpt-oss response ---")
    print(answer)


if __name__ == "__main__":
    main()
