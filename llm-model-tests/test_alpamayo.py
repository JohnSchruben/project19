import argparse
import sys
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

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

except Exception as e:
    print(f"Warning: Failed to patch tie_weights: {e}")

# Monkey-patch torch.linalg.cholesky to support BFloat16 (by upcasting to Float32)
# The Alpamayo action space utils use cholesky which fails on BFloat16.
original_cholesky = torch.linalg.cholesky

def safe_cholesky(input, *args, **kwargs):
    if input.dtype == torch.bfloat16:
        return original_cholesky(input.to(torch.float32), *args, **kwargs).to(torch.bfloat16)
    return original_cholesky(input, *args, **kwargs)

torch.linalg.cholesky = safe_cholesky
print("Patched torch.linalg.cholesky for BFloat16 compatibility.")

# Monkey-patch torch.cholesky_solve as well
original_cholesky_solve = torch.cholesky_solve

def safe_cholesky_solve(b, u, *args, **kwargs):
    if b.dtype == torch.bfloat16 or u.dtype == torch.bfloat16:
        return original_cholesky_solve(b.to(torch.float32), u.to(torch.float32), *args, **kwargs).to(torch.bfloat16)
    return original_cholesky_solve(b, u, *args, **kwargs)

torch.cholesky_solve = safe_cholesky_solve
print("Patched torch.cholesky_solve for BFloat16 compatibility.")

import json

def process_image(model, processor, image_path, prompt, device, batch_size=1, num_groups=1, hist_len=10):
    try:
        # Load Image
        image = Image.open(image_path).convert("RGB")
        
        # Construct message for processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
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
        
        # Prepare inputs with dummy history
        # Shape: (Batch, Num_Groups, Hist_Len, 3)
        ego_history_xyz = torch.zeros((batch_size, num_groups, hist_len, 3), dtype=model.dtype, device=device)
        
        # Shape: (Batch, Num_Groups, Hist_Len, 3, 3)
        ego_history_rot = torch.eye(3, dtype=model.dtype, device=device).view(1, 1, 1, 3, 3).repeat(batch_size, num_groups, hist_len, 1, 1)

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        
        model_inputs = helper.to_device(model_inputs, device)

        # Generate
        torch.cuda.manual_seed_all(42)
        autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        with torch.no_grad():
            with torch.autocast(device, dtype=autocast_dtype):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1, 
                    max_generation_length=512,
                    return_extra=True,
                )

        # Extract raw object
        raw_obj = extra["cot"][0][0]
        
        # Handle numpy/list wrapping
        if hasattr(raw_obj, 'item'):
            raw_obj = raw_obj.item()
        elif hasattr(raw_obj, 'tolist'):
            raw_obj = raw_obj.tolist()
            
        if isinstance(raw_obj, (list, tuple)):
            if len(raw_obj) > 0:
                raw_obj = raw_obj[0]
                
        reasoning = str(raw_obj)

        # Parse out assistant response
        # The prompt usually follows ChatML format
        search_term = "<|im_start|>assistant\n"
        if search_term in reasoning:
            reasoning = reasoning.split(search_term)[-1]
        
        # Remove end token
        reasoning = reasoning.split("<|im_end|>")[0].strip()
        
        # Process Trajectory
        # pred_xyz shape is likely (batch, samples, time, 3) or similar
        # We want the first sample's trajectory
        trajectory = []
        if pred_xyz is not None:
            # Assuming batch=1, samples=1
            # Take the first element
            traj_tensor = pred_xyz[0][0] # shape (T, 3)
            trajectory = traj_tensor.float().cpu().numpy().tolist()
        
        return {
            "reasoning": reasoning,
            "trajectory": trajectory
        }

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Test NVIDIA Alpamayo Model")
    parser.add_argument("--image", type=str, default="car-on-road-3.jpg", help="Path to input image or directory")
    parser.add_argument("--prompt", type=str, default="Describe this driving situation in detail.", help="Text prompt")
    parser.add_argument("--model-id", type=str, default="nvidia/Alpamayo-R1-10B", help="Hugging Face Model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output", type=str, default="alpamayo_results.json", help="Output JSON file")

    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    print(f"Device: {args.device}")

    try:
        # Load Model
        model = AlpamayoR1.from_pretrained(
            args.model_id, 
            dtype=torch.bfloat16 if args.device == "cuda" else torch.float32
        ).to(args.device)
        
        processor = helper.get_processor(model.tokenizer)

        # Determine input images
        input_path = args.image
        image_files = []
        
        if os.path.isdir(input_path):
            print(f"Processing directory: {input_path}")
            # Get common image extensions
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(os.path.join(root, file))
            image_files.sort() # Ensure sequential order
        elif os.path.isfile(input_path):
            image_files.append(input_path)
        else:
            print(f"Error: Input path '{input_path}' not found.")
            sys.exit(1)
            
        print(f"Found {len(image_files)} images.")
        
        results = []

        for img_path in tqdm(image_files, desc="Processing Images"):
            result_data = process_image(model, processor, img_path, args.prompt, args.device)
            if result_data:
                # Store relative path for portability
                try:
                    rel_path = os.path.relpath(img_path, start=os.getcwd())
                except ValueError:
                    rel_path = img_path

                results.append({
                    "image_path": rel_path,
                    "reasoning": result_data["reasoning"],
                    "trajectory": result_data["trajectory"]
                })
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Fatal Error: {e}")
        print("Ensure you have run setup_alpamayo.sh and logged in to Hugging Face.")

if __name__ == "__main__":
    main()
