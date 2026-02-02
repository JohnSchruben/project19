import base64
import io
import time
from PIL import Image
import ollama

# --- Tuning knobs ---
MODEL = "minicpm-v" # or "llava"
MAX_DIM = 512 
JPEG_QUALITY = 70
NUM_PREDICT = 120 
NUM_CTX = 1024
TEMPERATURE = 0.2

PROMPT = (
    "Describe what you see in this dashcam image.\n"
    "Focus on: lanes, vehicles, pedestrians, traffic controls, and potential hazards.\n"
    "Return a short, structured answer with bullet points."
)

def load_and_shrink_image(path: str) -> bytes:
    """
    Loads an image, resizes it to fit within MAX_DIM, and returns JPEG bytes.
    This dramatically reduces vision-model latency.
    """
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()

def b64_jpeg_bytes(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("utf-8")

import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run Ollama test")
    parser.add_argument("--image", type=str, default="car-on-road-2.jpg", help="Path to the image file")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Prompt text")
    args = parser.parse_args()

    image_path = args.image  # change to your image path
    current_prompt = args.prompt

    jpeg_bytes = load_and_shrink_image(image_path)
    img_b64 = b64_jpeg_bytes(jpeg_bytes)

    client = ollama.Client()

    # Optional: warm-up (first call is often slow due to model load)
    # client.generate(model=MODEL, prompt="hi", options={"num_predict": 5})

    t0 = time.time()
    resp = client.generate(
        model=MODEL,
        prompt=current_prompt,
        images=[img_b64],  # Ollama Python accepts base64 image strings here
        options={
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
        },
        keep_alive="10m",  # keep model in memory so subsequent calls are faster
    )
    dt = time.time() - t0

    print(f"\n--- response in {dt:.2f}s ---\n")
    print(resp["response"])

if __name__ == "__main__":
    main()
