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

def process_image(model, processor, image_paths, prompt, device, speed=0.0, yaw_rate=0.0, batch_size=1, num_groups=1, hist_len=10):
    try:
        # Load Images
        # image_paths can be a single string or a list
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        images = []
        for p in image_paths:
            images.append(Image.open(p).convert("RGB"))
        
        # Construct message for processor - Alpamayo Template from helper.py
        # We interleave images first, then text
        # Make num_traj_token dynamic based on history length (assuming 3 tokens per step)
        num_traj_token = hist_len * 3
        hist_traj_placeholder = f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
        
        # We start with the prompt text required by the model
        # "output the chain-of-thought reasoning of the driving process, then output the future trajectory."
        query_text = f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory."
        
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a driving assistant that generates safe and accurate actions.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "image", "image": img} for img in images]
                + [{"type": "text", "text": query_text}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "<|cot_start|>",
                    }
                ],
            },
        ]

        # Use processor to format
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True, # This might double-add assistant start if not careful, but apply_chat_template usually handles it. 
            # Actually, standard apply_chat_template might not handle the pre-filled "assistant" role correctly for all models.
            # Qwen usually handles it.
            return_dict=True,
            return_tensors="pt",
        )
        
        # Prepare inputs with synthetic history
        # Shape: (Batch, Num_Groups, Hist_Len, 3)
        # Assuming 20Hz sampling (0.05s per step)
        dt = 0.05
        ego_history_xyz = torch.zeros((batch_size, num_groups, hist_len, 3), dtype=model.dtype, device=device)
        
        # Shape: (Batch, Num_Groups, Hist_Len, 3, 3)
        ego_history_rot = torch.eye(3, dtype=model.dtype, device=device).view(1, 1, 1, 3, 3).repeat(batch_size, num_groups, hist_len, 1, 1)

        if speed > 0 or abs(yaw_rate) > 1e-4:
            # Generate history sequence
            # Index (N-1) is current time (t=0), Index 0 is oldest (t = -(N-1)*dt)
            steps = torch.arange(hist_len, dtype=model.dtype, device=device)
            steps = steps - (hist_len - 1)
            t = steps * dt # Negative time values
            
            if abs(yaw_rate) < 1e-4:
                # Straight line motion
                x_offsets = t * speed
                ego_history_xyz[:, :, :, 0] = x_offsets.view(1, 1, -1).contiguous()
                # Rotation remains identity
            else:
                # Curved motion (Circular Arc)
                # theta = w * t
                theta = t * yaw_rate
                r = speed / yaw_rate
                
                # Alpamayo Coordinate System: +X Forward, +Y Left
                # x(t) = (v/w) * sin(w*t)
                # y(t) = (v/w) * (1 - cos(w*t))
                
                # prevent numerical instability if r is too large
                r = torch.clamp(r, min=-10000.0, max=10000.0)

                x_offsets = r * torch.sin(theta)
                y_offsets = r * (1.0 - torch.cos(theta))
                
                ego_history_xyz[:, :, :, 0] = x_offsets.view(1, 1, -1).contiguous()
                ego_history_xyz[:, :, :, 1] = y_offsets.view(1, 1, -1).contiguous()
                
                # Update Rotation Matrix (Rotation around Z)
                c = torch.cos(theta).view(1, 1, -1)
                s = torch.sin(theta).view(1, 1, -1)
                zeros = torch.zeros_like(c)
                ones = torch.ones_like(c)
                
                # Build rotation matrix per time step
                # [ cos  -sin   0 ]
                # [ sin   cos   0 ]
                # [  0     0    1 ]
                
                # We need to assign these to the tensor
                # Using slicing to broadcast
                
                # R[0,0] = cos
                ego_history_rot[..., 0, 0] = c
                # R[0,1] = -sin
                ego_history_rot[..., 0, 1] = -s
                # R[1,0] = sin
                ego_history_rot[..., 1, 0] = s
                # R[1,1] = cos
                ego_history_rot[..., 1, 1] = c

        model_inputs = {
            "tokenized_data": inputs,
            "ego_history_xyz": ego_history_xyz.contiguous(),
            "ego_history_rot": ego_history_rot.contiguous(),
        }
        
        # print(f"DEBUG: Processing {image_paths[0] if isinstance(image_paths, list) else image_paths}")
        # print(f"DEBUG: xyz shape: {ego_history_xyz.shape}, rot shape: {ego_history_rot.shape}")
        
        model_inputs = helper.to_device(model_inputs, device)

        # Generate
        torch.cuda.manual_seed_all(42)
        autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        if device == "cuda":
            torch.cuda.synchronize()

        with torch.no_grad():
            with torch.autocast(device, dtype=autocast_dtype):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1, # Reduced from 4 to 1 as per user request
                    max_generation_length=512,
                    return_extra=True,
                )

        # Extract raw object for reasoning (CoT)
        # extra["cot"] shape: (Batch, Num_Samples) or (Batch, Num_Samples, Sequence)
        # We'll just take the first sample's reasoning for display
        raw_obj = extra["cot"][0][0]
        
        # Robust string extraction
        reasoning = ""
        try:
            # Convert numpy to python object (list or scalar)
            if hasattr(raw_obj, 'tolist'):
                raw_obj = raw_obj.tolist()
            
            if isinstance(raw_obj, (list, tuple)):
                # If it's a list, it might be a list of tokens/strings
                # Join them if they are strings
                if len(raw_obj) > 0 and isinstance(raw_obj[0], str):
                    reasoning = "".join(raw_obj)
                elif len(raw_obj) > 0:
                     # Maybe list of 1 element which is the string
                     reasoning = str(raw_obj[0])
                else:
                    reasoning = ""
            else:
                # Scalar
                reasoning = str(raw_obj)
        except Exception as e:
            print(f"Error parsing reasoning: {e}")
            reasoning = str(raw_obj)

        # Parse out assistant response
        search_term = "<|im_start|>assistant\n"
        if search_term in reasoning:
            reasoning = reasoning.split(search_term)[-1]
        
        # Remove end token
        reasoning = reasoning.split("<|im_end|>")[0].strip()
        
        # Process Trajectories
        # pred_xyz shape: (batch_size, num_traj_samples, time_steps, 3)
        trajectories = []
        if pred_xyz is not None:
             # Move to CPU first
            traj_tensor = pred_xyz.float().cpu() # (B, S, T, 3)
            
            # We assume batch_size=1
            # We want all samples: (S, T, 3)
            if traj_tensor.shape[0] > 0:
                samples = traj_tensor[0] # (S, T, 3)
                # Convert to list of lists
                trajectories = samples.numpy().tolist() # List[List[List[float]]]
        
        return {
            "reasoning": reasoning,
            "trajectories": trajectories # Return list of trajectories
        }

    except Exception as e:
        print(f"Error processing {image_paths[0]}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description="Test NVIDIA Alpamayo Model")
    parser.add_argument("--image", type=str, default="car-on-road-3.jpg", help="Path to input image or directory")
    parser.add_argument("--prompt", type=str, default="You are an autonomous driving agent. Analyze the scene and plan a safe trajectory for the ego vehicle.", help="Text prompt")
    parser.add_argument("--model-id", type=str, default="nvidia/Alpamayo-R1-10B", help="Hugging Face Model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--output", type=str, default="alpamayo_results.json", help="Output JSON file")
    parser.add_argument("--history-len", type=int, default=1, help="Number of visual frames to use as context (default: 1)")
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

        import signal
        
        # Robust Signal Handling
        stop_execution = False
        def signal_handler(sig, frame):
            nonlocal stop_execution
            print("\nSignal received. Stopping gracefully...")
            stop_execution = True

        signal.signal(signal.SIGINT, signal_handler)

        try:
            for img_path in tqdm(image_files, desc="Processing Images"):
                if stop_execution:
                    break
                    
                # Add current image to history
                history_buffer.append(img_path)
                
                # Pass the full history as list of paths
                # Convert deque to list
                current_context = list(history_buffer)
                
                # Get speed for current frame
                # Get speed for current frame
                filename = os.path.basename(img_path)
                current_speed = args.speed
                current_yaw_rate = 0.0
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
                                current_yaw_rate = t_data.get("yaw_rate", 0.0)
                        except Exception:
                            pass # Fail silently gracefully to default speed
                
                # Decouple visual history (args.history_len) from kinematic history (must be 16 for model compatibility)
                # Visual history: determines how many images are processed (affects speed)
                # Kinematic history: determines ego_history tensor size (affects crash/stability)
                result_data = process_image(model, processor, current_context, args.prompt, args.device, speed=current_speed, yaw_rate=current_yaw_rate, hist_len=args.history_len)
                
                # Extract Future Ground Truth Trajectory
                # We look ahead 50 steps (approx 5 seconds if 10Hz, or 2.5s if 20Hz)
                # Alpamayo likely outputs 3-5 seconds. Let's try to get 50 points.
                gt_trajectory = []
                x_gt, y_gt, theta_gt = 0.0, 0.0, 0.0
                
                # We need to know the frame index to find future frames
                # Assuming filenames are sequential numbers: 000000.png, 000001.png
                try:
                    base_name = os.path.splitext(filename)[0]
                    frame_idx = int(base_name)
                    
                    last_ts = current_telemetry.get('timestamp_eof', None)
                    
                    for t in range(1, 51): # 50 future steps
                        next_idx = frame_idx + t
                        next_mid = f"{next_idx:06d}"
                        
                        # Construct path to next telemetry
                        # Re-use telemetry_dir from earlier
                        if os.path.basename(img_dir) == "raw":
                            # Standard openpilot directory structure: .../segment/raw/000000.png
                            # Telemetry is at .../segment/telemetry/000000.json
                            telemetry_dir = os.path.join(os.path.dirname(img_dir), "telemetry")
                            next_json_path = os.path.join(telemetry_dir, next_mid + ".json")
                            
                            v_next = 0.0
                            w_next = 0.0
                            dt_step = 0.05 # Default to 20Hz
                            
                            if os.path.exists(next_json_path):
                                try:
                                    with open(next_json_path, 'r') as f:
                                        nd = json.load(f)
                                        v_next = nd.get("v_ego", 0.0)
                                        w_next = nd.get("yaw_rate", 0.0)
                                        next_ts = nd.get("timestamp_eof", None)
                                        
                                        if last_ts is not None and next_ts is not None:
                                            # Convert nanoseconds/microseconds to seconds
                                            # Openpilot logs usually use nanoseconds (1e9) or microseconds (1e6)
                                            # Timestamp generally increases.
                                            diff = next_ts - last_ts
                                            if diff > 1e8: # likely nanoseconds
                                                dt_step = diff / 1e9
                                            elif diff > 1e5: # likely microseconds
                                                dt_step = diff / 1e6
                                            
                                            # Sanity check dt
                                            if dt_step <= 0 or dt_step > 1.0:
                                                dt_step = 0.05
                                        
                                        last_ts = next_ts
                                except:
                                    pass
                            
                            # Integrate
                            # Coordinate system: +X Forward, +Y Left
                            # dx = v * cos(theta) * dt
                            # dy = v * sin(theta) * dt
                            # dtheta = w * dt
                            
                            dx = v_next * dt_step * np.cos(theta_gt)
                            dy = v_next * dt_step * np.sin(theta_gt)
                            dtheta = w_next * dt_step
                            
                            x_gt += dx
                            y_gt += dy
                            theta_gt += dtheta
                            
                            gt_trajectory.append([x_gt, y_gt])
                except ValueError:
                    # Filename not an integer, skip GT
                    pass

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
                        "trajectories": result_data["trajectories"], # List of trajectories
                        "gt_trajectory": gt_trajectory # Ground Truth
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
