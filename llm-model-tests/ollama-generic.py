import base64
import io
import time
from PIL import Image
import ollama
import argparse
import sys
import re

# --- Tuning knobs (default values) ---
MAX_DIM = 512 
JPEG_QUALITY = 70
NUM_PREDICT = 1024 
NUM_CTX = 2048
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
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print(f"Error processing image '{path}': {e}")
        sys.exit(1)

def b64_jpeg_bytes(jpeg_bytes: bytes) -> str:
    return base64.b64encode(jpeg_bytes).decode("utf-8")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run a generic Ollama test with any model")
    parser.add_argument("--model", type=str, required=True, help="The Ollama model to use (e.g., 'llama3.2', 'phi3.5', 'gemma2')")
    parser.add_argument("--image", type=str, default="car-on-road-2.jpg", help="Path to the image file")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Prompt text")
    parser.add_argument("--ctx", type=int, default=NUM_CTX, help="Context size")
    parser.add_argument("--predict", type=int, default=NUM_PREDICT, help="Max tokens to predict")
    
    args = parser.parse_args()

    model = args.model
    image_path = args.image
    current_prompt = args.prompt
    
    # Check if model likely supports vision (heuristic) or if user just wants text
    # For simplicity, we just pass the image if provided and it's physically present.
    # If the model is text-only (like Llama 3), Ollama often just ignores the 'images' field or complains lightly, 
    # but for safety we can try to detect if image file exists.
    
    img_b64 = None
    if image_path:
        try:
            jpeg_bytes = load_and_shrink_image(image_path)
            img_b64 = b64_jpeg_bytes(jpeg_bytes)
        except Exception:
            pass # Handle gracefully if file doesn't exist, though load_and_shrink handles exit

    client = ollama.Client()

    print(f"--- Running Generic Ollama Test ---")
    print(f"Directory: Running {model}")
    print(f"Image:  {image_path}")
    # print(f"Prompt: {current_prompt}")

    t0 = time.time()
    try:
        # Prepare arguments
        kwargs = {
            "model": model,
            "prompt": current_prompt,
            "options": {
                "num_predict": args.predict,
                "num_ctx": args.ctx,
                "temperature": TEMPERATURE,
            },
            "keep_alive": "10m",
        }
        
        # Only add images list if we successfully processed an image
        if img_b64:
            kwargs["images"] = [img_b64]

        resp = client.generate(**kwargs)
        
        dt = time.time() - t0

        print(f"\n--- response in {dt:.2f}s ---\n")
        response_text = resp["response"]
        
        # Post-processing to remove chatty intros
        # Look for the start of the structured output (e.g., "1. **")
        match = re.search(r"(1\.\s+\*\*.*)", response_text, re.DOTALL)
        if match:
            clean_response = match.group(1)
            print(clean_response)
        else:
            print(response_text)
        
    except ollama.ResponseError as e:
        print(f"Error from Ollama: {e.error}")
        if e.status_code == 404:
            print(f"Model '{model}' not found. Try running: ollama pull {model}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
