import moondream as md
from PIL import Image
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Run Moondream test")
parser.add_argument("--image", type=str, default="car-on-road.png", help="Path to the image file")
parser.add_argument("--prompt", type=str, default="What is happening in this image?", help="Prompt text")
args = parser.parse_args()

model = md.vl()
image = Image.open(args.image)
answer = model.query(image, args.prompt)
print(answer)
