
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
from scipy.spatial.transform import Rotation as R

# Alpamayo Imports
# Ensuring we can import from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

def load_openpilot_data(
    segment_dir: str,
    frame_idx: int,
    num_history_steps: int = 16, # Kinematic history length (must align with model config)
    num_future_steps: int = 64,  # GT future length
    time_step: float = 0.1,      # Assumed dt if timestamps missing
    device: str = "cuda"
):
    """
    Loads openpilot data and converts to Alpamayo format.
    Mimics load_physical_aiavdataset but uses relative integration for kinematics.
    """
    
    raw_dir = os.path.join(segment_dir, "raw")
    telemetry_dir = os.path.join(segment_dir, "telemetry")
    
    # 1. Kinematic Integration (History & Future) & 2. Timestamps
    # We start at t0 (frame_idx) as Origin (0,0,0, identity rot).
    
    # History Integration (Backwards from t0)
    hist_xyz = []
    hist_rot = []
    hist_xyz.append(np.zeros(3)) # t0 is 0,0,0
    hist_rot.append(np.eye(3))   # t0 is identity
    
    x, y, theta = 0.0, 0.0, 0.0
    
    # We need to look back num_history_steps - 1 times (total points = num_history_steps including t0)
    # Timestamps for image loading
    frame_indices_history = []
    
    # Current State at t0
    current_json = os.path.join(telemetry_dir, f"{frame_idx:06d}.json")
    t0_us = 0
    if os.path.exists(current_json):
        with open(current_json, 'r') as f:
            d = json.load(f)
            t0_us = d.get('timestamp_eof', 0) / 1000 # convert ns to us? No, timestamp_eof is ns usually. Alpamayo wants us.
            # But we only use it relative.
    
    # Iterate backwards
    for i in range(1, num_history_steps):
        prev_idx = frame_idx - i
        if prev_idx < 0: prev_idx = 0 # Clamp to start
        
        json_path = os.path.join(telemetry_dir, f"{prev_idx:06d}.json")
        v = 0.0
        w = 0.0
        dt = time_step
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                v = data.get('v_ego', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                steer_deg = data.get('steering_angle_deg', 0.0)
                
                # Yaw Rate Fallback (Bicycle Model)
                if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
                     steer_rad = np.deg2rad(steer_deg) / 15.0
                     yaw_rate = v * np.tan(steer_rad) / 2.7
                
                w = yaw_rate
        
        # Integrate Backwards: We are moving FROM prev TO current.
        # So to find prev relative to current:
        # Actually easier: Integrate velocity backwards.
        # x_{t-1} = x_t - v * cos(theta) * dt
        # theta_{t-1} = theta_t - w * dt
        
        x -= v * np.cos(theta) * dt
        y -= v * np.sin(theta) * dt
        theta -= w * dt
        
        hist_xyz.append(np.array([x, y, 0.0]))
        
        # Rotation Matrix from theta
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        hist_rot.append(R_mat)
        
    # Reverse to be chronological [t-N, ..., t0]
    hist_xyz = hist_xyz[::-1]
    hist_rot = hist_rot[::-1]
    
    # Future Integration (Forwards from t0)
    fut_xyz = []
    fut_rot = []
    
    x, y, theta = 0.0, 0.0, 0.0 # Reset to t0
    
    for i in range(1, num_future_steps + 1):
        next_idx = frame_idx + i
        json_path = os.path.join(telemetry_dir, f"{next_idx:06d}.json")
        v = 0.0
        w = 0.0
        dt = time_step
        
        if os.path.exists(json_path):
             with open(json_path, 'r') as f:
                data = json.load(f)
                v = data.get('v_ego', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                steer_deg = data.get('steering_angle_deg', 0.0)
                if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
                     steer_rad = np.deg2rad(steer_deg) / 15.0
                     yaw_rate = v * np.tan(steer_rad) / 2.7
                w = yaw_rate
                
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += w * dt
        
        fut_xyz.append(np.array([x, y, 0.0]))
        c, s = np.cos(theta), np.sin(theta)
        R_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        fut_rot.append(R_mat)
        
    # Convert to Tensors
    # Shape: (1, 1, Steps, 3)
    ego_history_xyz = torch.tensor(np.stack(hist_xyz), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ego_history_rot = torch.tensor(np.stack(hist_rot), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ego_future_xyz = torch.tensor(np.stack(fut_xyz), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    ego_future_rot = torch.tensor(np.stack(fut_rot), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # 3. Load Images
    # Alpamayo expects (N_cameras, num_frames, 3, H, W)
    # We map 'raw' to 'camera_front_wide_120fov' (Index 1)
    # We'll use just the current frame duplicated for 'num_frames' for simplicity, 
    # OR we can load history images if available. Let's load history.
    
    num_visual_frames = 4 # Default in load_physical_aiavdataset
    images = []
    
    # Indices for visual frames: [t-3, t-2, t-1, t0]
    # Assuming 10Hz stepping for visual frames too? Or whatever the dataset is.
    # Openpilot dataset is 20Hz usually. Alpamayo trained on 10Hz?
    # Let's just grab the last 4 frames available.
    
    for i in range(num_visual_frames):
        # Index: frame_idx - (3 - i)
        idx = frame_idx - (num_visual_frames - 1 - i)
        if idx < 0: idx = 0
        img_path = os.path.join(raw_dir, f"{idx:06d}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            # Resize? Alpamayo likely expects specific size or processor handles it.
            # Processor handles resizing usually.
            # But here we need to return tensor.
            # Let's use numpy -> tensor.
            img_np = np.array(img)
        else:
            img_np = np.zeros((224, 224, 3), dtype=np.uint8) # Placeholder
            
        images.append(img_np)
        
    # Stack: (num_frames, H, W, 3) -> (num_frames, 3, H, W)
    images_tensor = torch.tensor(np.stack(images), dtype=torch.uint8).permute(0, 3, 1, 2)
    
    # Wrap in Camera Dimension (N_cameras=1)
    # But wait, Alpamayo prompt expects specific cameras?
    # load_physical_aiavdataset returns N_cameras=4 by default.
    # If we only provide 1, will it break?
    # The prompt construction iterates over cameras.
    # helper.create_message iterates over camera indices.
    # We should probably padding other cameras with zeros or just provide 1 and hope helper handles it.
    # Let's try providing just 1 camera first.
    # Index 1 = Front Wide.
    
    image_frames = images_tensor.unsqueeze(0) # (1, num_frames, 3, H, W)
    camera_indices = torch.tensor([1], dtype=torch.int64) # Front Wide
    
    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": ego_future_xyz,
        "ego_future_rot": ego_future_rot,
        "t0_us": t0_us,
        "clip_id": "openpilot_custom"
    }


import argparse
import glob

def main():
    parser = argparse.ArgumentParser(description="Run Alpamayo on Openpilot Data")
    parser.add_argument("--image", type=str, required=True, help="Path to raw images directory (e.g. segment/raw)")
    parser.add_argument("--output", type=str, default="results.json", help="Path to save output JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames to process")
    parser.add_argument("--visualize", action="store_true", help="Save visualization plots")
    args = parser.parse_args()

    # Determine paths
    # User passes ".../raw". We need parent for telemetry.
    raw_dir = args.image
    if not os.path.isdir(raw_dir):
        print(f"Error: {raw_dir} is not a directory")
        return
    
    # Assuming standard structure: segment/raw/ -> segment/
    segment_dir = os.path.dirname(raw_dir.rstrip("/\\"))
    if os.path.basename(raw_dir.rstrip("/\\")) != "raw":
         # Maybe user passed the segment dir?
         if os.path.exists(os.path.join(raw_dir, "raw")):
             segment_dir = raw_dir
             raw_dir = os.path.join(segment_dir, "raw")
    
    print(f"Processing Data in: {segment_dir}")
    print(f"Images: {raw_dir}")
    print(f"Output: {args.output}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Model...")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to(device)
    processor = helper.get_processor(model.tokenizer)
    
    # Get List of Images
    image_files = sorted(glob.glob(os.path.join(raw_dir, "*.png")))
    if args.limit:
        image_files = image_files[:args.limit]
        
    results = []
    
    print(f"Found {len(image_files)} images.")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        try:
            frame_idx = int(os.path.splitext(filename)[0])
        except ValueError:
            continue
            
        print(f"Processing Frame {frame_idx}...")
        
        # Load Data
        # We try to load history. If beginning of segment, history might be partial (padded with first frame/zeros).
        # Our load_openpilot_data handles clamping to 0.
        try:
            data = load_openpilot_data(segment_dir, frame_idx, num_history_steps=16, device=device)
        except Exception as e:
            print(f"Error loading data for frame {frame_idx}: {e}")
            continue

        # Move to device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)
                if v.dtype == torch.float32:
                    data[k] = v.to(torch.bfloat16)

        # Inference
        try:
            messages = helper.create_message(data["image_frames"].flatten(0, 1))
            
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }
            
            with torch.no_grad():
                with torch.autocast(device, dtype=torch.bfloat16):
                    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                        data=model_inputs,
                        top_p=0.98,
                        temperature=0.6,
                        num_traj_samples=1,
                        max_generation_length=256,
                        return_extra=True,
                    )
            
            reasoning = extra["cot"][0]
            
            # Extract Prediction for JSON
            # pred_xyz shape: (1, 1, 1, Steps, 3) 
            pred_path = pred_xyz.float().cpu().numpy()[0, 0, 0, :, :3].tolist()
            
            # Get Ground Truth for JSON
            gt_path = data["ego_future_xyz"].float().cpu().numpy()[0, 0, :, :3].tolist()
            
            result_entry = {
                "filename": filename,
                "reasoning": reasoning,
                "pred_xyz": pred_path,
                "gt_xyz": gt_path,
                # "v_ego": ??? (Need to extract from load_openpilot_data or re-read)
            }
            results.append(result_entry)
            
            # Optional Visualization
            if args.visualize:
                # Reuse visualization logic but maybe save to a folder based on frame_idx
                pass 
                
        except Exception as e:
            print(f"Error inferencing frame {frame_idx}: {e}")
            continue

    # Save Results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    main()
