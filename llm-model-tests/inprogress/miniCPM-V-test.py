from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Run MiniCPM-V-2_6 test")
parser.add_argument("--image", type=str, default="car-on-road.png", help="Path to the image file")
parser.add_argument("--prompt", type=str, default="Describe the scene in the image.", help="Prompt text")
args = parser.parse_args()

model_id = "openbmb/MiniCPM-V-2_6"

model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)
processor = AutoProcessor.from_pretrained(model_id)

image = Image.open(args.image).convert("RGB")
prompt = args.prompt

inputs = processor(text=prompt, images=image, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100
)

print(processor.decode(outputs[0], skip_special_tokens=True))
