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

from alpamayo1_5.load_custom_dataset import load_custom_dataset
from alpamayo1_5 import helper, nav_utils
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

def get_default_route():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if dirs:
        dirs.sort()
        return dirs[0]
    return None

import signal

def main():
    parser = argparse.ArgumentParser(description="Headless batch video exporter with Alpamayo inference.")
    parser.add_argument("--route", type=str, default=get_default_route(), 
                        help="Path to a route directory containing segments")
    parser.add_argument("--frames", type=int, default=16, 
                        help="Number of future frames to graph for predictions")
    parser.add_argument("--command", type=str, default=None, 
                        help="Optional navigation command to inject into Alpamayo prompt (e.g., 'Turn Right')")
    parser.add_argument("--segment", type=str, default=None,
                        help="Process only a specific segment (e.g., 'segment_00')")
    args = parser.parse_args()
    
    # Store exporters globally to gracefully shut them down if the user hits Ctrl+C
    _active_exporters = []
    def graceful_exit(sig, frame):
        print("\n\n[Ctrl+C Detected] Gracefully shutting down and saving video progress...")
        for exporter in _active_exporters:
            if exporter is not None:
                exporter.release()
        sys.exit(0)
    signal.signal(signal.SIGINT, graceful_exit)

    if not args.route or not os.path.exists(args.route):
        print(f"Error: Route directory '{args.route}' not found.")
        return

    print("Loading Alpamayo model... (This will take a moment)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", 
        dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    processor = helper.get_processor(model.tokenizer)

    all_segments = sorted([d for d in glob.glob(os.path.join(args.route, 'segment_*')) if os.path.isdir(d)])
    if args.segment:
        all_segments = [s for s in all_segments if os.path.basename(s) == args.segment]

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

        active_turn_cmd = "Go Straight"
        turn_cmd_frames_left = 0
        turn_dist_m = 0.0
        turn_active_frames = 0

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
                # Encoding at 40 FPS to force media players to respect 1x playback speed without speeding it up
                out = cv2.VideoWriter(output_video_path, fourcc, 40.0, (w, h))
                _active_exporters.append(out)

            try:
                data = load_custom_dataset(seg_dir, local_idx)
            except Exception as e:
                print(f"Error loading data for {seg_name} frame {local_idx}: {e}")
                out.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                continue

            # Determine navigation command dynamically if not provided
            nav_cmd = args.command
            if nav_cmd is None:
                nav_cmd = "Go Straight" # Default
                gt_rot = data["ego_future_rot"][0, 0].numpy()
                gt_xyz = data["ego_future_xyz"][0, 0].numpy()
                full_frames = gt_rot.shape[0]
                if full_frames > 0:
                    # Use XY displacement instead of rotation metrics to find path curvature
                    check_frames = min(200, full_frames) # Look heavily into the future
                    xs = gt_xyz[:check_frames, 0]
                    ys = gt_xyz[:check_frames, 1]
                    
                    path_angles = np.degrees(np.arctan2(ys, xs))
                    distances = np.hypot(xs, ys)
                    
                    far_idx = distances > 5.0
                    raw_nav_cmd = "Go Straight"
                    if np.any(far_idx):
                        if np.max(path_angles[far_idx]) > 15:
                            raw_nav_cmd = "Turn left"
                        elif np.min(path_angles[far_idx]) < -15:
                            raw_nav_cmd = "Turn right"
                            
                    if raw_nav_cmd != "Go Straight":
                        # Find exactly where the turn begins (first point deviating > 2 degrees)
                        turn_start_idx = 0
                        for i in range(check_frames):
                            if distances[i] < 1.0: continue
                            if (raw_nav_cmd == "Turn left" and path_angles[i] > 2) or \
                               (raw_nav_cmd == "Turn right" and path_angles[i] < -2):
                                turn_start_idx = i
                                break
                                
                        new_dist = max(0.0, float(xs[turn_start_idx]))
                        
                        if active_turn_cmd != raw_nav_cmd:
                            if new_dist < 80.0:
                                active_turn_cmd = raw_nav_cmd
                                turn_dist_m = new_dist
                                turn_active_frames = 60 # Sticky hold for 3 seconds
                        else:
                            # Update the absolute distance to the turn junction.
                            # Because relative trajectory geometry plateaus as the car steers into the curve,
                            # we ensure it actively counts down to 0m mathematically by bleeding it.
                            turn_dist_m = min(turn_dist_m, new_dist)
                            turn_dist_m = max(0.0, turn_dist_m - 0.3) 
                            turn_active_frames = 60
                    else:
                        if turn_active_frames > 0:
                            turn_active_frames -= 1
                            turn_dist_m = 0.0 # Pin to 0m because we are literally riding out the apex
                        else:
                            active_turn_cmd = "Go Straight"
                            
                    if active_turn_cmd != "Go Straight":
                        if turn_dist_m <= 5.0:
                            # If we are physically inside the junction (< 5m), drop the distance naturally to just 'Turn left'
                            nav_cmd = active_turn_cmd
                        else:
                            nav_cmd = f"{active_turn_cmd} in {int(turn_dist_m)}m"
                    else:
                        nav_cmd = "Go Straight"

            # Inference
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            else:
                torch.manual_seed(42)
                
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                nav_result = nav_utils.compare_nav_conditions(
                    model=model,
                    processor=processor,
                    data=data,
                    nav_text=nav_cmd,
                    num_traj_samples=1,
                    top_p=0.98,
                    temperature=0.6,
                    max_generation_length=256,
                    return_extra=True,
                    nav_inference_fn=model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav,
                    additional_nav_inference_kwargs={
                        "diffusion_kwargs": {
                            "use_classifier_free_guidance": True,
                            "inference_guidance_weight": 1.5,
                            "temperature": 0.6,
                        }
                    },
                )
                
            cot = nav_result.extra_with_nav["cot"][0][0] # first sample, first batch
            if isinstance(cot, np.ndarray):
                cot = cot.item() # Extract string from numpy 0d array or array size 1
            if isinstance(cot, list):
                if len(cot) > 0: cot = str(cot[0])
                else: cot = ""
            cot = str(cot).strip()
            print(f"[{seg_name} | Frame {local_idx}] Cmd: {nav_cmd} | Reasoning: {cot}")

            # Plotting GT vs Pred
            ax_export.clear()
            
            def extract_prds(pred_tensor):
                prd_xyz = pred_tensor.cpu().numpy()[0, 0, 0] # shape (seq_len, 3)
                n_frames = min(args.frames, prd_xyz.shape[0])
                p_x = np.concatenate(([0.0], prd_xyz[:n_frames, 0]))
                p_y = np.concatenate(([0.0], prd_xyz[:n_frames, 1]))
                return p_x, p_y
                
            n_gt_frames = min(args.frames, gt_xyz.shape[0])
            gt_x = np.concatenate(([0.0], gt_xyz[:n_gt_frames, 0]))
            gt_y = np.concatenate(([0.0], gt_xyz[:n_gt_frames, 1]))
            
            ax_export.plot([-y for y in gt_y], gt_x, marker='o', color='black', linewidth=2, label="GT")
            
            # with nav (Blue)
            p_xw, p_yw = extract_prds(nav_result.pred_with_nav)
            ax_export.plot([-y for y in p_yw], p_xw, marker='x', color='blue', linewidth=2, label="Nav")
            
            # counterfactual (Green)
            p_xc, p_yc = extract_prds(nav_result.pred_counterfactual)
            ax_export.plot([-y for y in p_yc], p_xc, marker='^', color='green', linewidth=2, label="Opp_Nav")

            # no nav (Red)
            p_xn, p_yn = extract_prds(nav_result.pred_no_nav)
            ax_export.plot([-y for y in p_yn], p_xn, marker='.', color='red', linewidth=2, label="No_Nav")
            
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
            
            # Calculate total height for the background rectangle (CoT + Nav Command)
            total_bg_height = y_text + len(wrapped_text)*30 + 40
            
            # Background rectangle for easy text reading (black)
            cv2.rectangle(img_np, (10, 10), (w - 10, total_bg_height), (0, 0, 0), -1)
            
            # Draw CoT lines (green)
            for line in wrapped_text:
                cv2.putText(img_np, line, (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_text += 30
            
            # Draw Nav Command line (orange in OpenCV BGR/RGB logic - since image is currently RGB via PIL, we use RGB values: (255, 165, 0))
            # Wait, `cv2` deals with colors in BGR order usually when saving, but the frame is currently an RGB numpy array because of `Image.open().convert('RGB')`
            # and finally `cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)` is called below.
            # Thus, the color tuple here MUST be RGB! Orange in RGB is (255, 165, 0).
            cv2.putText(img_np, f"Nav Command: {nav_cmd}", (20, y_text + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
            # Convert once and write twice to effectively play 20Hz frames at 40 FPS seamlessly
            bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            out.write(bgr_frame)
            
        if out is not None:
            out.release()
            print(f"Finished exporting {output_video_path}")
            
        plt.close(fig_export)

if __name__ == "__main__":
    main()
