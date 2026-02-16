import argparse
import sys
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

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
        # Load Processor and Model
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
            device_map="auto" if args.device == "cuda" else None
        )
        
        if args.device != "cuda" and model.device.type != "cuda":
             model.to(args.device)

        # Load Image
        image = Image.open(args.image).convert("RGB")

        # Prepare inputs
        # Note: Exact prompt format depends on the model. Assuming standard chat template or simple text.
        # Alpamayo might use specific tokens. For now, we use a simple format.
        text = f"<image>\n{args.prompt}" 
        
        inputs = processor(text=[text], images=[image], return_tensors="pt")
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
        print("Note: You may need to log in to Hugging Face (`huggingface-cli login`) to access this model.")

if __name__ == "__main__":
    main()
