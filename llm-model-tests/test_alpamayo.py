import argparse
import sys
import torch
import numpy as np
from PIL import Image

# Try importing from the installed alpamayo package or local clone
try:
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1 import helper
except ImportError:
    # Fallback: check if local 'alpamayo/src' exists and add to path
    import os
    local_src = os.path.join(os.path.dirname(__file__), "alpamayo", "src")
    if os.path.exists(local_src):
        print(f"Adding local source to path: {local_src}")
        sys.path.append(local_src)
        try:
            from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
            from alpamayo_r1 import helper
        except ImportError as e:
            print(f"Error importing from local source: {e}")
            sys.exit(1)
    else:
        print("Error: alpamayo_r1 package not found and local 'alpamayo/src' not found.")
        print("Please run ./setup_alpamayo.sh to clone the repository.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Test NVIDIA Alpamayo Model")
    parser.add_argument("--image", type=str, default="car-on-road-3.jpg", help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this driving situation in detail.", help="Text prompt")
    parser.add_argument("--model-id", type=str, default="nvidia/Alpamayo-R1-10B", help="Hugging Face Model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")

    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    print(f"Device: {args.device}")

    try:
        # Load Model using official class
        model = AlpamayoR1.from_pretrained(
            args.model_id, 
            dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
        ).to(args.device)
        
        processor = helper.get_processor(model.tokenizer)

        # Load Image
        image = Image.open(args.image).convert("RGB")
        
        # Prepare inputs using Alpamayo helper structure
        # The helper expects 'messages' which usually contain image/text.
        # Based on typical VLA usage, we construct a simple message.
        # Note: 'helper.create_message' in the repo takes flattened image frames.
        # We need to see how it handles a single image.
        # Assuming simple chat template structure for now as fallback if helper is complex.
        
        # Construct a standardized message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": args.prompt}
                ]
            }
        ]

        # Use processor to format
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=500,
                do_sample=False
            )

        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("\n--- Output ---")
        print(generated_text)

    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you have run setup_alpamayo.sh and logged in to Hugging Face.")

if __name__ == "__main__":
    main()
