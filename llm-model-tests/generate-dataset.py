import ollama
import os
import glob
import json
import base64
import time
from PIL import Image
import io

MODEL_NAME = "llava"
IMAGE_DIR = "../dataset/images"
OUTPUT_FILE = "../dataset/" + MODEL_NAME + "-results.jsonl"
PROMPT_FILE = "driving_prompt.txt"

def load_prompt():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(base_dir, PROMPT_FILE)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading prompt file: {e}")
        return ""

PROMPT = load_prompt()
PROMPT = load_prompt()
def encode_image(image_path):
    """Encodes an image to base64, similar to ollama-generic.py but simpler."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            pass

        with open(image_path, "rb") as f:
            return f.read() 
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        return None

def run_generation(model_name=MODEL_NAME):
    # Resolve absolute path to correspond to run location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir_abs = os.path.join(base_dir, IMAGE_DIR)
    
    # Update output file based on model name
    output_file_name = f"{model_name}-results.jsonl"
    output_file_path = f"../dataset/{output_file_name}"
    output_file_abs = os.path.join(base_dir, output_file_path)

    print(f"--- Generating dataset using model: {model_name} ---")
    print(f"Looking for images in: {image_dir_abs}")
    
    # Get all files (jpg, png, jpeg, etc. - assume jpg based on previous listing)
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    all_images = []
    for pattern in patterns:
        all_images.extend(glob.glob(os.path.join(image_dir_abs, pattern)))
    
    # Sort
    all_images.sort()
    
    # Process ALL images (removed MAX_IMAGES limit)
    selected_images = all_images
    
    print(f"Found {len(all_images)} images. Processing ALL of them.")

    results = []

    for img_path in selected_images:
        img_name = os.path.basename(img_path)
        print(f"Processing {img_name}...")
        
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()

            response = ollama.generate(
                model=model_name,
                prompt=PROMPT,
                images=[img_bytes],
                format='json', # Force JSON mode if model supports it (Ollama feature)
                options={
                    "temperature": 0.2 # Low temperature for consistent formatting
                }
            )
            
            response_text = response.get('response', '')
            print(f"  > Raw response: {response_text[:100]}...") # Debug print

            # Try to parse the JSON
            # Clean up potential markdown code blocks if the model ignores the instruction
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            try:
                data = json.loads(cleaned_text)
                
                # Ensure all required keys exist, fill with defaults if missing
                final_entry = {
                    "image": img_name,
                    "scene_description": data.get("scene_description", "No description provided"),
                    "steering_angle_deg": float(data.get("steering_angle_deg", 0.0)),
                    "throttle": float(data.get("throttle", 0.0)),
                    "brake": float(data.get("brake", 0.0))
                }
                
                results.append(final_entry)
                print(f"  > Success.")
                
            except json.JSONDecodeError:
                print(f"  > Failed to parse JSON. Saving raw text as description.")
                # Fallback: create a dummy entry with the error
                results.append({
                    "image": img_name,
                    "scene_description": f"JSON PARSE ERROR. Raw: {cleaned_text}",
                    "steering_angle_deg": 0.0,
                    "throttle": 0.0,
                    "brake": 0.0
                })

        except Exception as e:
            print(f"  > Error processing {img_name}: {e}")
            results.append({
                "image": img_name,
                "scene_description": f"SYSTEM ERROR: {str(e)}",
                "steering_angle_deg": 0.0,
                "throttle": 0.0,
                "brake": 0.0
            })

    # Write to JSONL
    print(f"Writing results to {output_file_abs}...")
    with open(output_file_abs, 'w') as f:
        for entry in results:
            json.dump(entry, f)
            f.write('\n')
    
    print("Done.")

def main():
    import argparse
    import ollama_utils
    
    parser = argparse.ArgumentParser(description="Generate dataset using a specific Ollama model")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="The Ollama model to use")
    args = parser.parse_args()
    
    with ollama_utils.OllamaService():
        run_generation(args.model)

if __name__ == "__main__":
    main()
