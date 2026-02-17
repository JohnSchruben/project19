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
    # Check current directory "alpamayo/src"
    local_src = os.path.join(os.path.dirname(__file__), "alpamayo", "src")
    # Check parent directory "../alpamayo/src" (project root structure)
    parent_src = os.path.join(os.path.dirname(__file__), "..", "alpamayo", "src")
    
    if os.path.exists(local_src):
        print(f"Adding local source to path: {local_src}")
        sys.path.append(local_src)
    elif os.path.exists(parent_src):
        # Resolve to absolute path
        parent_src = os.path.abspath(parent_src)
        print(f"Adding parent source to path: {parent_src}")
        sys.path.append(parent_src)
    else:
        # Fallback to try importing anyway, maybe it is installed globally
        pass
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

def process_image(model, processor, image_paths, prompt, device, speed=0.0, batch_size=1, num_groups=1, hist_len=10):
    try:
        # Load Images
        # image_paths can be a single string or a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        images = []
        for p in image_paths:
            images.append(Image.open(p).convert("RGB"))
        
        # Construct message for processor
        # We interleave images first, then text
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        
        messages = [
            {
                "role": "user",
                "content": content
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
        
        # Prepare inputs with synthetic history
        # Shape: (Batch, Num_Groups, Hist_Len, 3)
        # Assuming 10Hz sampling (0.1s per step)
        dt = 0.1
        ego_history_xyz = torch.zeros((batch_size, num_groups, hist_len, 3), dtype=model.dtype, device=device)
        
        if speed > 0:
            # Generate linear backward history: we were at -x before.
            # Index i: 0 is oldest? Or newest? 
            # Usually transformers expect chronological order: 0=oldest, -1=newest (current).
            # Current pos is (0,0,0).
            # Previous pos (-1) was at (-v*dt, 0, 0).
            # Oldest pos (0) was at (-(hist_len-1)*v*dt, 0, 0).
            
            # Create a sequence of offsets
            # offsets = [-(hist_len - 1 - i) * dt * speed for i in range(hist_len)]
            
            # Implementation using torch
            steps = torch.arange(hist_len, dtype=model.dtype, device=device) # 0, 1, ..., N-1
            # We want 0 to be furthest back: -(N-1)
            # steps - (N-1) gives -(N-1), ..., 0
            steps = steps - (hist_len - 1)
            
            x_offsets = steps * dt * speed
            
            # Assign to X channel (0)
            ego_history_xyz[:, :, :, 0] = x_offsets.view(1, 1, -1)
            
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
        # pred_xyz shape is likely (batch, num_samples, time, 3)
        trajectory = []
        if pred_xyz is not None:
             # Move to CPU first
            traj_tensor = pred_xyz.float().cpu() # (B, S, T, 3)
            
            # Flatten to (Total_Points, 3) to strictly ensure we have a list of points
            # We assume we only want the FIRST sample of the FIRST batch.
            # If shape is (1, 1, T, 3), we want (T, 3).
            # If shape is (1, S, T, 3), we pick index 0.
            
            if traj_tensor.numel() > 0:
                # Reshape to (-1, 3) to flatten batch/sample dims
                flat_traj = traj_tensor.reshape(-1, 3)
                
                # However, if batch>1 or samples>1, we might merge multiple trajectories if we just flatten.
                # So we should be careful. 
                # Let's trust unwrapping by index if we know dimensions.
                
                # Safe unwrap:
                while traj_tensor.dim() > 2:
                    traj_tensor = traj_tensor[0]
                    
                trajectory = traj_tensor.numpy().tolist()
        
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
    parser.add_argument("--history-len", type=int, default=1, help="Number of frames to use as context (including current)")
    parser.add_argument("--speed", type=float, default=10.0, help="Simulated vehicle speed in m/s (default: 10.0)")

    args = parser.parse_args()

    print(f"Loading model: {args.model_id}")
    print(f"Device: {args.device}")

    # Run inference
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
        
        # History buffer for multi-frame support
        from collections import deque
        history_buffer = deque(maxlen=args.history_len)

        try:
            for img_path in tqdm(image_files, desc="Processing Images"):
                # Add current image to history
                history_buffer.append(img_path)
                
                # Pass the full history as list of paths
                # Convert deque to list
                current_context = list(history_buffer)
                
                # Get speed for current frame
                filename = os.path.basename(img_path)
                current_speed = args.speed
                current_telemetry = {}
                
                # Check for per-frame telemetry file
                # Assumes structure: segment/raw/image.png -> segment/telemetry/image.json
                img_dir = os.path.dirname(img_path)
                if os.path.basename(img_dir) == "raw":
                    telemetry_dir = os.path.join(os.path.dirname(img_dir), "telemetry")
                    json_name = os.path.splitext(filename)[0] + ".json"
                    telemetry_path = os.path.join(telemetry_dir, json_name)
                    
                    if os.path.exists(telemetry_path):
                        try:
                            with open(telemetry_path, 'r') as f:
                                t_data = json.load(f)
                                current_telemetry = t_data
                                current_speed = t_data.get("v_ego", args.speed)
                        except Exception:
                            pass # Fail silently gracefully to default speed
                
                result_data = process_image(model, processor, current_context, args.prompt, args.device, speed=current_speed)
                
                if result_data:
                    # Store relative path for portability
                    try:
                        rel_path = os.path.relpath(img_path, start=os.getcwd())
                    except ValueError:
                        rel_path = img_path

                    results.append({
                        "image_path": rel_path,
                        "speed": current_speed,
                        "telemetry_data": current_telemetry,
                        "reasoning": result_data["reasoning"],
                        "trajectory": result_data["trajectory"]
                    })
        except KeyboardInterrupt:
            print("\n\nProcess interrupted by user. Saving results processed so far...")
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    except Exception as e:
        print(f"Fatal Error: {e}")
        print("Ensure you have run setup_alpamayo.sh and logged in to Hugging Face.")

if __name__ == "__main__":
    main()
