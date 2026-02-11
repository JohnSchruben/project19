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
PROMPT = """You are analyzing a dashcam image.

Describe ONLY what is clearly visible in the image.

Return your answer EXACTLY in the following structured format.
Do not add extra sections.
Do not change the section names.
If a category is not present, write "None".

1. **Traffic Lights**:
   - State color (red/yellow/green) and position (left/right/center/background).
   - If none visible, write: None

2. **Traffic Signs**:
   - List each visible sign:
     - Type (stop sign, yield sign, speed limit, pedestrian crossing, no turn, etc.)
     - Relative position (right side of road, overhead, near intersection, sidewalk left, etc.)
   - If none visible, write: None

3. **Vehicles**:
   - Total vehicle count: <number>
   - List each vehicle as:
     - Type (car, truck, SUV, bus, motorcycle, etc.)
     - Relative position (front, left lane, right lane, oncoming, parked right side, behind, etc.)

4. **Pedestrians**:
   - Total pedestrian count: <number>
   - For each pedestrian:
     - Relative position (crosswalk, sidewalk right, near intersection, etc.)
   - If none visible, write: None

5. **Lane Markings**:
   - Describe visible lane lines (solid, dashed, double yellow, crosswalk, turn arrows, ONLY markings, etc.)
   - If none visible, write: None

6. **Hazards / Recommended Action**:
   - List any immediate hazards (stopped vehicle, pedestrian crossing, red light, obstruction, etc.)
   - Provide one short recommended driving action (e.g., "Maintain lane", "Prepare to stop", "Yield to pedestrian", "Slow down", "Proceed straight").
   - If no hazards, write: No immediate hazards. Maintain normal driving."""
MAX_IMAGES = 10
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

def main():
    # Resolve absolute path to correspond to run location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir_abs = os.path.join(base_dir, IMAGE_DIR)
    output_file_abs = os.path.join(base_dir, OUTPUT_FILE)

    print(f"Looking for images in: {image_dir_abs}")
    
    # Get all files (jpg, png, jpeg, etc. - assume jpg based on previous listing)
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    all_images = []
    for pattern in patterns:
        all_images.extend(glob.glob(os.path.join(image_dir_abs, pattern)))
    
    # Sort and take top 10
    all_images.sort()
    selected_images = all_images[:MAX_IMAGES]

    print(f"Found {len(all_images)} images. Processing the first {len(selected_images)}.")

    results = []

    for img_path in selected_images:
        img_name = os.path.basename(img_path)
        print(f"Processing {img_name}...")
        
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()

            response = ollama.generate(
                model=MODEL_NAME,
                prompt=PROMPT,
                images=[img_bytes] 
            )
            
            response_text = response.get('response', '')
            
            results.append({
                "image": img_name,
                "result": response_text
            })
            print(f"  > Response length: {len(response_text)} chars")

        except Exception as e:
            print(f"  > Error processing {img_name}: {e}")
            results.append({
                "image": img_name,
                "result": f"Error: {str(e)}"
            })

    # Write to JSONL
    print(f"Writing results to {output_file_abs}...")
    with open(output_file_abs, 'w') as f:
        for entry in results:
            json.dump(entry, f)
            f.write('\n')
    
    print("Done.")

if __name__ == "__main__":
    main()
