# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Custom dataset loader for Alpamayo R1

import os
import json
import torch
import numpy as np
from PIL import Image, ImageOps

def load_custom_dataset(
    segment_dir: str,
    frame_idx: int,
    num_history_steps: int = 16, # Kinematic history length at 10Hz
    num_future_steps: int = 64,  # GT future length at 10Hz
    time_step: float = 0.1,      # Target framerate matching Alpamayo (10Hz)
    frame_stride: int = 2,       # Kinematic dataset stride (Openpilot is 20Hz, so stride 2 = exactly 10Hz physics)
    visual_stride: int = 1,      # Artificially halved optical flow stride to fix GOPRO FOV distortion without breaking physics.
):
    """
    Loads custom openpilot-style data and converts to Alpamayo format.
    Expects segment_dir to contain:
      - raw/ (containing .png images named like 000000.png)
      - telemetry/ (containing .json files named like 000000.json with v_ego, yaw_rate, steering_angle_deg)
    """
    
    raw_dir = os.path.join(segment_dir, "raw")
    telemetry_dir = os.path.join(segment_dir, "telemetry")
    
    # 1. Kinematic Integration (History & Future)
    # start at t0 (frame_idx) as Origin (0,0,0, identity rot).
    
    # History Integration (Backwards from t0)
    hist_xyz = []
    hist_rot = []
    hist_xyz.append(np.zeros(3)) # t0 is 0,0,0
    hist_rot.append(np.eye(3))   # t0 is identity
    
    x, y, theta = 0.0, 0.0, 0.0
    
    # Current State at t0
    current_json = os.path.join(telemetry_dir, f"{frame_idx:06d}.json")
    t0_us = 0
    if os.path.exists(current_json):
        with open(current_json, 'r') as f:
            d = json.load(f)
            t0_us = d.get('timestamp_eof', 0) / 1000 # convert ns to us if needed based on your telemetry
    
    # Iterate backwards
    for i in range(1, num_history_steps):
        prev_idx = frame_idx - (i * frame_stride)
        if prev_idx < 0: prev_idx = 0 # Clamp to start
        
        json_path = os.path.join(telemetry_dir, f"{prev_idx:06d}.json")
        v = 0.0
        w = 0.0
        dt = time_step
        is_reverse = False
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                v = data.get('v_ego', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                steer_deg = data.get('steering_angle_deg', 0.0)
                is_reverse = data.get('gear_shifter') == 'reverse'
                
                # Dynamic dt calculation
                t_prev_us = data.get('timestamp_eof', 0) / 1000
                if t0_us > 0 and t_prev_us > 0:
                    dt = (t0_us - t_prev_us) / 1000000.0 / i # approximate avg dt
                
                
                # Yaw Rate Fallback (Bicycle Model)
                if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
                     steer_rad = np.deg2rad(steer_deg) / 15.49
                     yaw_rate = v * np.tan(steer_rad) / 2.7
                
                w = yaw_rate
        
        # Integrate Backwards
        if is_reverse:
             x -= (-v) * np.cos(theta) * dt
             y -= (-v) * np.sin(theta) * dt
             theta -= (-w) * dt
        else:
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
        next_idx = frame_idx + (i * frame_stride)
        json_path = os.path.join(telemetry_dir, f"{next_idx:06d}.json")
        v = 0.0
        w = 0.0
        dt = time_step
        is_reverse = False
        
        if os.path.exists(json_path):
             with open(json_path, 'r') as f:
                data = json.load(f)
                v = data.get('v_ego', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                steer_deg = data.get('steering_angle_deg', 0.0)
                is_reverse = data.get('gear_shifter') == 'reverse'
                
                # Dynamic dt calculation
                t_next_us = data.get('timestamp_eof', 0) / 1000
                if t0_us > 0 and t_next_us > 0:
                    dt = (t_next_us - t0_us) / 1000000.0 / i # approximate avg dt

                if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
                     steer_rad = np.deg2rad(steer_deg) / 15.49
                     yaw_rate = v * np.tan(steer_rad) / 2.7
                w = yaw_rate
                
        if is_reverse:
             x += (-v) * np.cos(theta) * dt
             y += (-v) * np.sin(theta) * dt
             theta += (-w) * dt
        else:
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
    #
    # Camera directory mapping – indices match Alpamayo's convention:
    #   0 = camera_cross_left_120fov   → raw_left/
    #   1 = camera_front_wide_120fov   → raw/
    #   2 = camera_cross_right_120fov  → raw_right/
    camera_dirs = [
        ("raw_left",  0),  # Left camera
        ("raw",       1),  # Front-wide camera
        ("raw_right", 2),  # Right camera
        ("raw_front", 6),  # Front-tele (narrow) camera
    ]

    # Auto-detect which camera directories are present
    available_cameras = []
    for dir_name, cam_idx in camera_dirs:
        cam_dir = os.path.join(segment_dir, dir_name)
        if os.path.isdir(cam_dir) and any(
            f.endswith(".png") for f in os.listdir(cam_dir)
        ):
            available_cameras.append((cam_dir, cam_idx))

    if not available_cameras:
        raise FileNotFoundError(
            f"No camera directories with images found in {segment_dir}. "
            "Expected at least 'raw/' containing .png frames."
        )

    num_visual_frames = 4  # Default in load_physical_aiavdataset

    all_camera_frames = []
    all_camera_indices = []

    for cam_dir, cam_idx in available_cameras:
        images = []
        for i in range(num_visual_frames):
            idx = frame_idx - (num_visual_frames - 1 - i) * visual_stride
            if idx < 0:
                idx = 0
            img_path = os.path.join(cam_dir, f"{idx:06d}.png")
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                # Use aspect-ratio preserving padding instead of crude stretching
                # Standardizing to 640x480 (W, H) prevents horizontal stretching that destroys spatial intersection curves
                img = ImageOps.pad(img, (640, 480), method=Image.Resampling.BILINEAR)
                img_np = np.array(img)
            else:
                img_np = np.zeros((480, 640, 3), dtype=np.uint8)
            images.append(img_np)

        # (num_frames, H, W, 3) -> (num_frames, 3, H, W)
        cam_tensor = torch.tensor(
            np.stack(images), dtype=torch.uint8
        ).permute(0, 3, 1, 2)
        all_camera_frames.append(cam_tensor)
        all_camera_indices.append(cam_idx)

    # Stack: (N_cameras, num_frames, 3, H, W)
    image_frames = torch.stack(all_camera_frames, dim=0)
    camera_indices = torch.tensor(all_camera_indices, dtype=torch.int64)

    # Sort by camera index for consistent ordering (matches official loader)
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]

    return {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "ego_future_xyz": ego_future_xyz,  # useful for eval, omit if you don't have future telemetry
        "ego_future_rot": ego_future_rot,  # useful for eval
        "t0_us": t0_us,
        "clip_id": f"custom_segment_frame_{frame_idx}"
    }

# ---
# How to use
# ---
if __name__ == "__main__":
    # Example usage:
    segment_dir = "../../datasets/route_1/segment_00" # path relative to notebook
    frame_idx = 100 
    
    # data = load_custom_dataset(segment_dir, frame_idx)
    #
    # messages = helper.create_message(data["image_frames"].flatten(0, 1))
    # ... rest of the code ...
