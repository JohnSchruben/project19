import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from alpamayo1_5.load_custom_dataset import load_custom_dataset
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


TURN_COLOR_MAP = {
    "Ground Truth": "red",
    "Nav Command": "lime",
    "Command": "lime",
}


def get_default_route():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if dirs:
        dirs.sort()
        return dirs[0]
    return None

import signal


def extract_cot(extra):
    cot = extra["cot"][0][0]
    if isinstance(cot, np.ndarray):
        if cot.size == 1:
            cot = cot.item()
        elif cot.size == 0:
            cot = ""
        else:
            cot = " ".join(map(str, cot.flatten()))
    if isinstance(cot, list):
        cot = " ".join(map(str, cot)) if cot else ""
    return str(cot).strip()


def nav_label(nav_cmd: str) -> str:
    nav_lower = nav_cmd.lower()
    if "left" in nav_lower or "right" in nav_lower or "straight" in nav_lower:
        return "Nav Command"
    return "Command"


def run_nav_inference(
    model,
    processor,
    data,
    device,
    nav_cmd: str,
    num_traj_samples: int,
    guidance_weight: float,
):
    messages_nav = helper.create_message(
        data["image_frames"].flatten(0, 1),
        camera_indices=data.get("camera_indices"),
        nav_text=nav_cmd,
    )
    inputs_nav = processor.apply_chat_template(
        messages_nav,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs_nav = helper.to_device(
        {
            "tokenized_data": inputs_nav,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        device,
    )

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        pred_xyz_nav, pred_rot_nav, extra_nav = (
            model.sample_trajectories_from_data_with_vlm_rollout_cfg_nav(
                data=model_inputs_nav,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                max_generation_length=256,
                return_extra=True,
                diffusion_kwargs={
                    "use_classifier_free_guidance": True,
                    "inference_guidance_weight": guidance_weight,
                    "temperature": 0.6,
                },
            )
        )

    return pred_xyz_nav, pred_rot_nav, extra_nav


def select_prediction_xy(
    pred_tensor,
    nav_cmd: str,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    pred_np = pred_tensor.detach().cpu().numpy()[0, 0]
    if pred_np.shape[0] == 0:
        return np.array([0.0]), np.array([0.0]), 0

    nav_lower = nav_cmd.lower()
    final_lateral = pred_np[:, -1, 1]
    final_forward = pred_np[:, -1, 0]

    if "left" in nav_lower:
        sample_idx = int(np.argmax(final_lateral))
    elif "right" in nav_lower:
        sample_idx = int(np.argmin(final_lateral))
    elif "straight" in nav_lower:
        sample_idx = int(np.argmin(np.abs(final_lateral)))
    else:
        # Fallback to the sample that goes furthest forward while staying centered.
        sample_idx = int(np.argmax(final_forward - np.abs(final_lateral)))

    selected = pred_np[sample_idx]
    n_frames = min(num_frames, selected.shape[0])
    pred_x = np.concatenate(([0.0], selected[:n_frames, 0]))
    pred_y = np.concatenate(([0.0], selected[:n_frames, 1]))
    return pred_x, pred_y, n_frames


def plot_dotted_path(ax, xs: np.ndarray, ys: np.ndarray, color: str, label: str):
    ax.plot([-y for y in ys], xs, marker='o', color=color, linewidth=2.0, label=label)


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
    parser.add_argument("--start-frame", type=int, default=0,
                        help="First frame index to process within the segment (inclusive)")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Last frame index to process within the segment (inclusive)")
    parser.add_argument("--num-traj-samples", type=int, default=6,
                        help="Number of trajectory samples to draw per condition")
    parser.add_argument("--guidance-weight", type=float, default=1.5,
                        help="Classifier-free guidance weight for nav-conditioned inference")
    args = parser.parse_args()
    
    # Safe interrupt flag to gracefully finish the current frame and save the video seamlessly
    interrupt_flag = [False]
    def graceful_exit(sig, frame):
        print("\n\n[Stop Signal Detected] Finishing the current frame and cleanly saving video, please wait...")
        interrupt_flag[0] = True
    signal.signal(signal.SIGINT, graceful_exit)
    signal.signal(signal.SIGTERM, graceful_exit)

    if not args.route or not os.path.exists(args.route):
        print(f"Error: Route directory '{args.route}' not found.")
        return

    print("Loading Alpamayo model... (This will take a moment)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Alpamayo1_5.from_pretrained(
        "nvidia/Alpamayo-1.5-10B", 
        dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    if device == "cuda":
        print("Compiling model for faster inference (this may take a few minutes on the first run)...")
        model = torch.compile(model)
    
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
        if num_frames_seg == 0:
            print(f"Skipping {seg_name} because it has no telemetry frames.")
            continue

        start_frame = args.start_frame
        end_frame = num_frames_seg - 1 if args.end_frame is None else args.end_frame

        if start_frame < 0:
            print(f"Skipping {seg_name}: start frame {start_frame} must be >= 0.")
            continue
        if end_frame < 0:
            print(f"Skipping {seg_name}: end frame {end_frame} must be >= 0.")
            continue
        if start_frame >= num_frames_seg:
            print(
                f"Skipping {seg_name}: start frame {start_frame} is outside the segment "
                f"(0-{num_frames_seg - 1})."
            )
            continue
        if end_frame >= num_frames_seg:
            print(
                f"Skipping {seg_name}: end frame {end_frame} is outside the segment "
                f"(0-{num_frames_seg - 1})."
            )
            continue
        if end_frame < start_frame:
            print(
                f"Skipping {seg_name}: end frame {end_frame} is before start frame {start_frame}."
            )
            continue
        
        if start_frame == 0 and end_frame == num_frames_seg - 1:
            output_video_path = f"{seg_name}_inference.mp4"
        else:
            output_video_path = f"{seg_name}_inference_{start_frame:06d}_{end_frame:06d}.mp4"
        out = None
        
        
        fig_export = plt.figure(figsize=(4, 4), dpi=100)
        ax_export = fig_export.add_subplot(111)
        overlay_size = (300, 300)

        print(
            f"Processing frames {start_frame} through {end_frame} "
            f"(inclusive) out of 0-{num_frames_seg - 1}."
        )

        active_turn_cmd = "Go Straight"
        turn_dist_m = 0.0
        turn_active_frames = 0

        for local_idx in range(start_frame, end_frame + 1):
            if interrupt_flag[0]:
                break
            
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

            try:
                data = load_custom_dataset(seg_dir, local_idx)
            except Exception as e:
                print(f"Error loading data for {seg_name} frame {local_idx}: {e}")
                out.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                continue

            gt_rot = data["ego_future_rot"][0, 0].numpy()
            gt_xyz = data["ego_future_xyz"][0, 0].numpy()
            
            # Determine navigation command dynamically if not provided
            nav_cmd = args.command
            if nav_cmd is None:
                nav_cmd = "Go Straight" # Default
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

            # Set fixed seed to match the nav notebook exactly for deterministic conditional inference
            torch.cuda.manual_seed_all(42)

            nav_runs = []
            overlay_summary = ""
            cot = ""
            pred_xyz_nav, _, extra_nav = run_nav_inference(
                model=model,
                processor=processor,
                data=data,
                device=device,
                nav_cmd=nav_cmd,
                num_traj_samples=args.num_traj_samples,
                guidance_weight=args.guidance_weight,
            )
            nav_runs.append((nav_label(nav_cmd), nav_cmd, pred_xyz_nav))
            cot = extract_cot(extra_nav)
            overlay_summary = f"Nav Command: {nav_cmd}"
            print(
                f"[{seg_name} | Frame {local_idx}] Cmd: \033[92m{nav_cmd}\033[0m | "
                f"Reasoning: \033[38;2;255;165;0m{cot}\033[0m"
            )

            # Plotting GT vs Pred
            ax_export.clear()
            pred_plot_data = []
            max_common_frames = gt_xyz.shape[0]
            for label, cmd_text, pred_xyz in nav_runs:
                pred_x, pred_y, pred_frames = select_prediction_xy(pred_xyz, cmd_text, args.frames)
                pred_plot_data.append((label, pred_x, pred_y, pred_frames))
                max_common_frames = min(max_common_frames, pred_frames)

            n_plot_frames = min(args.frames, max_common_frames)
            gt_x = np.concatenate(([0.0], gt_xyz[:n_plot_frames, 0]))
            gt_y = np.concatenate(([0.0], gt_xyz[:n_plot_frames, 1]))
            plot_dotted_path(ax_export, gt_x, gt_y, TURN_COLOR_MAP["Ground Truth"], "Ground Truth")

            for label, pred_x, pred_y, _ in pred_plot_data:
                plot_dotted_path(
                    ax_export,
                    pred_x[: n_plot_frames + 1],
                    pred_y[: n_plot_frames + 1],
                    TURN_COLOR_MAP.get(label, TURN_COLOR_MAP["Command"]),
                    label,
                )

            ax_export.plot(0, 0, marker='*', color='black', markersize=15)
            ax_export.legend(loc="best", fontsize=8)
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

            # Draw wrapped text onto image. In compare mode, keep the overlay concise and
            # rely on the colored trajectory groups plus console logs for the per-command CoT.
            overlay_text = cot
            # Dynamically compute wrap width based on image width and font metrics
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_margin = 20  # px padding on each side
            usable_width = w - (text_margin * 2)
            # Measure average character width with a reference string
            (ref_w, _), _ = cv2.getTextSize("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", font_face, font_scale, font_thickness)
            char_width = ref_w / 52.0
            wrap_chars = max(20, int(usable_width / char_width))
            wrapped_text = textwrap.wrap(overlay_text, width=wrap_chars) if overlay_text else []
            y_text = 40
            
            # Calculate total height for the background rectangle (CoT + Nav Command)
            total_bg_height = y_text + len(wrapped_text)*30 + 40
            
            # Background rectangle for easy text reading (black)
            cv2.rectangle(img_np, (10, 10), (w - 10, total_bg_height), (0, 0, 0), -1)
            
            # Draw CoT lines (green)
            for line in wrapped_text:
                cv2.putText(img_np, line, (text_margin, y_text), font_face, font_scale, (0, 255, 0), font_thickness)
                y_text += 30
            
            # The frame is RGB until the final conversion to BGR, so use RGB tuples here.
            cv2.putText(img_np, overlay_summary, (20, y_text + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
            # Convert once and write twice to effectively play 20Hz frames at 40 FPS seamlessly
            bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            out.write(bgr_frame)
            
        if out is not None:
            out.release()
            print(f"Finished exporting {output_video_path}")
            
        plt.close(fig_export)
        
        if interrupt_flag[0]:
            print("Processing stopped early by user.")
            break

if __name__ == "__main__":
    main()
