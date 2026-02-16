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

# Monkey-patch to fix 'tie_weights' compatibility issue with newer transformers
# Newer transformers call tie_weights(recompute_mapping=False), which ReasoningVLA might not accept.
try:
    if hasattr(AlpamayoR1, 'tie_weights'):
        original_tie_weights = AlpamayoR1.tie_weights
        
        def safe_tie_weights(self, *args, **kwargs):
            # Remove the arguments that cause the crash
            if 'recompute_mapping' in kwargs:
                kwargs.pop('recompute_mapping')
            if 'missing_keys' in kwargs:
                kwargs.pop('missing_keys')
            return original_tie_weights(self, *args, **kwargs)
        
        # Apply the patch to the class
        AlpamayoR1.tie_weights = safe_tie_weights
        print("Patched AlpamayoR1.tie_weights for compatibility.")
except Exception as e:
    print(f"Warning: Failed to patch tie_weights: {e}")

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
        
        # Construct message for processor
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
        
        # Prepare inputs with dummy history (required by model forward)
        # Assuming history length of 1 (current frame) or short history.
        # Based on typical AV models, history might be ~2-10Hz for T seconds.
        # We'll create a minimal dummy history.
        
        # Check model config for history length if possible, or assume a standard shape.
        # alpamayo usually expects [batch, T_hist, 3] for xyz
        batch_size = 1
        hist_len = 10 # reasonable guess, can be adjusted
        
        ego_history_xyz = torch.zeros((batch_size, hist_len, 3), dtype=model.dtype, device=args.device)
        ego_history_rot = torch.eye(3, dtype=model.dtype, device=args.device).unsqueeze(0).unsqueeze(0).repeat(batch_size, hist_len, 1, 1)

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        
        model_inputs = helper.to_device(model_inputs, args.device)

        # Generate
        print("Generating response...")
        with torch.no_grad():
            # Using the VLM rollout method as in the official script
            pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=1, 
                max_generation_length=512,
                return_extra=True,
            )

        print("\n--- Chain-of-Causation (Reasoning) ---")
        # extra["cot"] contains the text reasoning
        print(extra["cot"][0][0]) # [batch, sample] -> text

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("Ensure you have run setup_alpamayo.sh and logged in to Hugging Face.")

if __name__ == "__main__":
    main()
