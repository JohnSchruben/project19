import os
import sys
import glob
import json
import argparse
import copy
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from alpamayo_r1.load_custom_dataset import load_custom_dataset
from alpamayo_r1 import helper
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1

def get_default_route():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if dirs:
        dirs.sort()
        return dirs[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="Headless batch video exporter with Alpamayo inference.")
    parser.add_argument("--route", type=str, default=get_default_route(), 
                        help="Path to a route directory containing segments")
    parser.add_argument("--frames", type=int, default=16, 
                        help="Number of future frames to graph for predictions")
    parser.add_argument("--command", type=str, default=None, 
                        help="Optional navigation command to inject into Alpamayo prompt (e.g., 'Turn Right')")
    args = parser.parse_args()

    if not args.route or not os.path.exists(args.route):
        print(f"Error: Route directory '{args.route}' not found.")
        return

    print("Loading Alpamayo model... (This will take a moment)")
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)

    all_segments = sorted([d for d in glob.glob(os.path.join(args.route, 'segment_*')) if os.path.isdir(d)])
    if not all_segments:
        print("No segments found in route.")
        return

    print(f"Found {len(all_segments)} segments in {args.route}.")

    for seg_dir in all_segments:
        seg_name = os.path.basename(seg_dir)
        print(f"\nProcessing {seg_name}...")
        
        telemetry_dir = os.path.join(seg_dir, "telemetry")
        raw_dir = os.path.join(seg_dir, "raw")
        if not os.path.exists(telemetry_dir) or not os.path.exists(raw_dir):
            print(f"Skipping {seg_name} due to missing data.")
            continue
            
        num_frames_seg = len(glob.glob(os.path.join(telemetry_dir, "*.json")))
        
        output_video_path = f"{seg_name}_inference.mp4"
        out = None
        
        fig_export = plt.figure(figsize=(4, 4), dpi=100)
        ax_export = fig_export.add_subplot(111)
        overlay_size = (300, 300)

        for local_idx in range(num_frames_seg):
            # Load basic image for background
            img_path = os.path.join(raw_dir, f"{local_idx:06d}.png")
            if os.path.exists(img_path):
                img_np = np.array(Image.open(img_path).convert('RGB'))
            else:
                img_np = np.zeros((224, 224, 3), dtype=np.uint8)
                
            if out is None:
                h, w, _ = img_np.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (w, h))

            try:
                data = load_custom_dataset(seg_dir, local_idx)
            except Exception as e:
                print(f"Error loading data for {seg_name} frame {local_idx}: {e}")
                out.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                continue

            # Determine navigation command dynamically if not provided
            nav_cmd = args.command
            if nav_cmd is None:
                gt_xyz = data["ego_future_xyz"][0, 0].numpy()
                n_frames = min(args.frames, gt_xyz.shape[0])
                if n_frames > 0:
                    # Determine target angle from displacement vectors 
                    # gt_xyz[:, 0] is forward (X), gt_xyz[:, 1] is left (Y)
                    last_x = gt_xyz[n_frames - 1, 0]
                    last_y = gt_xyz[n_frames - 1, 1]
                    turn_angle = np.degrees(np.arctan2(last_y, last_x))
                    
                    if turn_angle > 45: # Left turn threshold (~90 deg turn)
                        nav_cmd = "Turn Left"
                    elif turn_angle < -45: # Right turn threshold
                        nav_cmd = "Turn Right"
                    else:
                        nav_cmd = "Go Straight"
                else:
                    nav_cmd = "Go Straight"

            # Process images for Alpamayo
            messages = helper.create_message(data["image_frames"].flatten(0, 1), nav_command=nav_cmd)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            model_inputs = {
                "tokenized_data": inputs,
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            }
            model_inputs = helper.to_device(model_inputs, "cuda")

            # Inference
            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=copy.deepcopy(model_inputs),
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=1,
                    max_generation_length=256,
                    return_extra=True,
                )
                
            cot = extra["cot"][0][0] # first sample, first batch
            if isinstance(cot, np.ndarray):
                cot = cot.item() # Extract string from numpy 0d array or array size 1
            if isinstance(cot, list):
                if len(cot) > 0: cot = str(cot[0])
                else: cot = ""
            cot = str(cot).strip()
            print(f"[{seg_name} | Frame {local_idx}] Reasoning: {cot}")

            # Plotting GT vs Pred
            ax_export.clear()
            
            gt_xyz = data["ego_future_xyz"][0, 0].numpy()
            prd_xyz = pred_xyz.cpu().numpy()[0, 0, 0] # shape (seq_len, 3)
            
            n_frames = min(args.frames, gt_xyz.shape[0])
            prd_len = min(args.frames, prd_xyz.shape[0])

            gt_x_forward = np.concatenate(([0.0], gt_xyz[:n_frames, 0]))
            gt_y_left = np.concatenate(([0.0], gt_xyz[:n_frames, 1]))
            
            prd_x_forward = np.concatenate(([0.0], prd_xyz[:prd_len, 0]))
            prd_y_left = np.concatenate(([0.0], prd_xyz[:prd_len, 1]))

            # Map coordinates: -Y vs X
            ax_export.plot([-y for y in gt_y_left], gt_x_forward, marker='o', color='red', linewidth=2, label="GT")
            ax_export.plot([-y for y in prd_y_left], prd_x_forward, marker='x', color='blue', linewidth=2, label="Pred")
            
            ax_export.plot(0, 0, marker='*', color='black', markersize=15)
            
            ax_export.set_aspect('equal')
            cur_xlim = ax_export.get_xlim()
            cur_ylim = ax_export.get_ylim()
            max_range = max(cur_xlim[1]-cur_xlim[0], cur_ylim[1]-cur_ylim[0], 10.0) / 2.0
            x_c = (cur_xlim[1]+cur_xlim[0]) / 2.0
            y_c = (cur_ylim[1]+cur_ylim[0]) / 2.0
            ax_export.set_xlim(x_c - max_range, x_c + max_range)
            ax_export.set_ylim(y_c - max_range, y_c + max_range)
            ax_export.axis('off')
            
            fig_export.canvas.draw()
            rgba = np.asarray(fig_export.canvas.buffer_rgba())
            overlay_img = rgba[:, :, :3].copy()
            overlay_img = cv2.resize(overlay_img, overlay_size)
            
            gray = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2GRAY)
            mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
            
            h, w, _ = img_np.shape
            roi = img_np[h-overlay_size[1]:h, w-overlay_size[0]:w]
            bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)
            img_np[h-overlay_size[1]:h, w-overlay_size[0]:w] = cv2.add(bg, fg)

            # Draw wrapped CoT reasoning text onto image
            wrapped_text = textwrap.wrap(cot, width=70) # Wrap text cleanly
            y_text = 40
            # Background rectangle for easy text reading
            cv2.rectangle(img_np, (10, 10), (w - 10, y_text + len(wrapped_text)*30 + 10), (0, 0, 0), -1)
            for line in wrapped_text:
                cv2.putText(img_np, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_text += 30
                
            out.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
        if out is not None:
            out.release()
            print(f"Finished exporting {output_video_path}")
            
        plt.close(fig_export)

if __name__ == "__main__":
    main()
