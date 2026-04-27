import os
import sys
import glob
import argparse

def get_default_route():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if dirs:
        dirs.sort()
        return dirs[0]
    return None

parser = argparse.ArgumentParser(description="Headless batch video exporter with Alpamayo inference.")
parser.add_argument("--route", type=str, default=get_default_route(), 
                    help="Path to a route directory containing segments")
parser.add_argument("--frames", type=int, default=64, 
                    help="Number of future frames to graph for predictions")
parser.add_argument("--command", type=str, default=None, 
                    help="Optional navigation command to inject into Alpamayo prompt (e.g., 'Turn Right')")
parser.add_argument("--segment", type=str, default=None,
                    help="Process only a specific segment (e.g., 'segment_00')")
parser.add_argument("--start-frame", type=int, default=0,
                    help="First frame index to process within the segment (inclusive)")
parser.add_argument("--end-frame", type=int, default=None,
                    help="Last frame index to process within the segment (inclusive)")
parser.add_argument("--num-traj-samples", type=int, default=16,
                    help="Number of trajectory samples to draw per condition")
parser.add_argument("--selection-mode", choices=["heuristic", "mean", "median"], default="heuristic",
                    help="How to collapse sampled trajectories into the displayed path.")
parser.add_argument("--guidance-weight", type=float, default=1.5,
                    help="Classifier-free guidance weight for nav-conditioned inference")
parser.add_argument("--cameras", nargs="+", choices=["wide", "left", "right", "front"],
                    default=["wide", "left", "right", "front"],
                    help="Cameras to include (wide, left, right, front). Unlisted cameras will be excluded.")
parser.add_argument("--max-gen-length", type=int, default=256,
                    help="Maximum generation length for the trajectory diffusion model. Lower speeds it up but reduces max distance.")
parser.add_argument("--dataset-fps", type=float, default=10.0,
                    help="Dataset frame rate used for export timing (default: 10.0).")
parser.add_argument("--output-fps", type=float, default=40.0,
                    help="Output video frame rate (default: 40.0).")
parser.add_argument("--plot-all-samples", action="store_true",
                    help="Plot all trajectory samples instead of just the selected best path")
global_args = parser.parse_args()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import cv2
import torch
import re
import signal
from PIL import Image
import matplotlib.pyplot as plt
import textwrap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from alpamayo1_5.load_custom_dataset import load_custom_dataset
from alpamayo1_5.navigation_command import infer_navigation_command
from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


TURN_COLOR_MAP = {
    "Ground Truth": "red",
    "Nav Command": "lime",
    "Command": "lime",
    "Sample": "gray",
}

import signal


def extract_cot(extra, idx=0):
    try:
        # Alpamayo wraps extra["cot"] in nested dimensions [batch=1, beam=1, samples=16]
        cot_data = extra.get("cot", [])
        
        # Unwrap extraneous [1] outer dimensions until we hit the actual array of 16 options
        while isinstance(cot_data, (list, tuple, np.ndarray)) and len(cot_data) == 1:
            cot_data = cot_data[0]
            
        # Select the exact string that matches the statistically chosen trajectory index
        if isinstance(cot_data, (list, tuple, np.ndarray)):
            if len(cot_data) > idx:
                return str(cot_data[idx]).strip()
            elif len(cot_data) > 0:
                return str(cot_data[0]).strip()
        
        return str(cot_data).strip()
    except Exception:
        return ""


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
    max_gen_length: int = 256,
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
                max_generation_length=max_gen_length,
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
    selection_mode: str = "heuristic",
) -> tuple[np.ndarray, np.ndarray, int, int]:
    pred_np = pred_tensor.detach().cpu().numpy()[0, 0]
    if pred_np.shape[0] == 0:
        return np.array([0.0]), np.array([0.0]), 0, 0

    if selection_mode == "mean":
        selected = pred_np.mean(axis=0)
        sample_idx = int(
            np.argmin(
                np.linalg.norm(pred_np[:, :, :2] - selected[None, :, :2], axis=-1).mean(axis=1)
            )
        )
    elif selection_mode == "median":
        selected = np.median(pred_np, axis=0)
        sample_idx = int(
            np.argmin(
                np.linalg.norm(pred_np[:, :, :2] - selected[None, :, :2], axis=-1).mean(axis=1)
            )
        )
    else:
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
    return pred_x, pred_y, n_frames, sample_idx


def plot_dotted_path(ax, xs: np.ndarray, ys: np.ndarray, color: str, label: str):
    ax.plot([-y for y in ys], xs, marker='o', color=color, linewidth=2.0, label=label)


def main():
    args = global_args
    
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

    cam_mapping = {"left": 0, "wide": 1, "right": 2, "front": 6}
    frame_repeat = 1
    if args.dataset_fps > 0 and args.output_fps > 0:
        frame_repeat = max(1, int(round(args.output_fps / args.dataset_fps)))
    excluded_cameras = []
    for name, idx in cam_mapping.items():
        if name not in args.cameras:
            excluded_cameras.append(idx)

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
        
        # raw folder is front wide, raw_front is front narrow
        raw_dir = os.path.join(seg_dir, "raw_front")
        if not os.path.exists(raw_dir):
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
        
        route_name = os.path.basename(os.path.abspath(args.route))
        if start_frame == 0 and end_frame == num_frames_seg - 1:
            output_video_path = f"{seg_name}_{route_name}_inference.mp4"
        else:
            output_video_path = f"{seg_name}_{route_name}_inference_{start_frame:06d}_{end_frame:06d}.mp4"
        out = None
        
        fig_export = plt.figure(figsize=(4, 4), dpi=100)
        
        def signal_handler(sig, frame):
            print("\nInference stopped by user. Saving current video progress...")
            if out is not None:
                out.release()
                print(f"Saved partial export for {output_video_path}")
            if fig_export is not None:
                plt.close(fig_export)
            sys.exit(0)
            
        signal.signal(signal.SIGINT, signal_handler)
        
        ax_export = fig_export.add_subplot(111)
        overlay_size = (300, 300)

        print(
            f"Processing frames {start_frame} through {end_frame} "
            f"(inclusive) out of 0-{num_frames_seg - 1}."
        )

        for local_idx in range(start_frame, end_frame + 1):
            if interrupt_flag[0]:
                break
            
            single_camera_mode = len(args.cameras) == 1
            
            def load_cam_img(cam_dir_name, tw, th):
                cam_dir = os.path.join(seg_dir, cam_dir_name)
                img_path_png = os.path.join(cam_dir, f"{local_idx:06d}.png")
                img_path_jpg = os.path.join(cam_dir, f"{local_idx:06d}.jpg")
                img_path = img_path_png if os.path.exists(img_path_png) else (img_path_jpg if os.path.exists(img_path_jpg) else None)
                if img_path:
                    img = np.array(Image.open(img_path).convert('RGB'))
                else:
                    img = np.zeros((th, tw, 3), dtype=np.uint8)
                if img.shape[:2] != (th, tw):
                    img = cv2.resize(img, (tw, th))
                cv2.putText(img, cam_dir_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return img

            if single_camera_mode:
                target_w, target_h = 1280, 960
                
                cam_name = "raw"
                if "front" in args.cameras: cam_name = "raw_front"
                elif "left" in args.cameras: cam_name = "raw_left"
                elif "right" in args.cameras: cam_name = "raw_right"
                
                img_np = load_cam_img(cam_name, target_w, target_h)
            else:
                # Construct 2x2 grid of images for the background
                target_w, target_h = 640, 480
                img_np = np.zeros((target_h * 2, target_w * 2, 3), dtype=np.uint8)
                
                img_np[0:target_h, 0:target_w] = load_cam_img("raw", target_w, target_h)           # Top Left
                img_np[target_h:target_h*2, 0:target_w] = load_cam_img("raw_left", target_w, target_h) # Bottom Left
                img_np[target_h:target_h*2, target_w:target_w*2] = load_cam_img("raw_right", target_w, target_h) # Bottom Right

            if out is None:
                h, w, _ = img_np.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, args.output_fps, (w, h))

            try:
                data = load_custom_dataset(seg_dir, local_idx, exclude_cameras=excluded_cameras)
            except Exception as e:
                print(f"Error loading data for {seg_name} frame {local_idx}: {e}")
                if out is not None:
                    out.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                continue

            gt_xyz = data["ego_future_xyz"][0, 0].numpy()
            
            # Determine navigation command dynamically if not provided
            nav_cmd = args.command
            if nav_cmd is None:
                nav_cmd = infer_navigation_command(gt_xyz)

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
                max_gen_length=args.max_gen_length,
            )
            nav_runs.append((nav_label(nav_cmd), nav_cmd, pred_xyz_nav, extra_nav))
            overlay_summary = f"Nav Command: {nav_cmd}"

            # Plotting GT vs Pred
            ax_export.clear()
            pred_plot_data = []
            max_common_frames = gt_xyz.shape[0]
            for label, cmd_text, pred_xyz, extra in nav_runs:
                pred_x, pred_y, pred_frames, sample_idx = select_prediction_xy(
                    pred_xyz,
                    cmd_text,
                    args.frames,
                    selection_mode=args.selection_mode,
                )
                
                cot = extract_cot(extra, sample_idx)
                if args.selection_mode == "heuristic":
                    print(
                        f"[{seg_name} | Frame {local_idx}] Cmd: \033[92m{cmd_text}\033[0m | "
                        f"Reasoning: \033[38;2;255;165;0m{cot}\033[0m"
                    )
                else:
                    print(
                        f"[{seg_name} | Frame {local_idx}] Cmd: \033[92m{cmd_text}\033[0m | "
                        f"Display Path: \033[96m{args.selection_mode}\033[0m | "
                        f"Representative Reasoning: \033[38;2;255;165;0m{cot}\033[0m"
                    )
                
                if args.plot_all_samples:
                    pred_np = pred_xyz.detach().cpu().numpy()[0, 0]
                    for i in range(pred_np.shape[0]):
                        selected = pred_np[i]
                        n_f = min(args.frames, selected.shape[0])
                        px = np.concatenate(([0.0], selected[:n_f, 0]))
                        py = np.concatenate(([0.0], selected[:n_f, 1]))
                        pred_plot_data.append(("Sample", px, py, n_f))

                pred_plot_data.append((label, pred_x, pred_y, pred_frames))
                max_common_frames = min(max_common_frames, pred_frames)

            n_plot_frames = min(args.frames, max_common_frames)
            gt_x = np.concatenate(([0.0], gt_xyz[:n_plot_frames, 0]))
            gt_y = np.concatenate(([0.0], gt_xyz[:n_plot_frames, 1]))
            plot_dotted_path(ax_export, gt_x, gt_y, TURN_COLOR_MAP["Ground Truth"], "Ground Truth")

            for label, pred_x, pred_y, _ in pred_plot_data:
                # If plotting background samples, make them thinner and distinct
                is_sample = label == "Sample"
                alpha = 0.4 if is_sample else 1.0
                lw = 1.0 if is_sample else 2.0
                ax_export.plot(
                    [-y for y in pred_y[: n_plot_frames + 1]],
                    pred_x[: n_plot_frames + 1],
                    marker='o',
                    color=TURN_COLOR_MAP.get(label, TURN_COLOR_MAP["Command"]),
                    linewidth=lw,
                    alpha=alpha,
                    label=label if not is_sample else ""
                )

            ax_export.plot(0, 0, marker='*', color='black', markersize=15)
            # Remove duplicate labels
            handles, labels = ax_export.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_export.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)
            ax_export.set_aspect('equal')
            cur_xlim = ax_export.get_xlim()
            cur_ylim = ax_export.get_ylim()
            max_range = max(cur_xlim[1]-cur_xlim[0], cur_ylim[1]-cur_ylim[0], 10.0) / 2.0
            x_c = (cur_xlim[1]+cur_xlim[0]) / 2.0
            y_c = (cur_ylim[1]+cur_ylim[0]) / 2.0
            ax_export.set_xlim(x_c - max_range, x_c + max_range)
            ax_export.set_ylim(y_c - max_range, y_c + max_range)
            ax_export.axis('off')
            
            # Draw text
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6  # smaller text
            font_thickness = 1
            
            wrapped_cot = textwrap.wrap(f"Reasoning: {cot}", width=60)
            
            lines_to_draw = [
                f"[{seg_name} | Frame {local_idx}]",
                f"Cmd: {nav_cmd}",
                ""
            ] + wrapped_cot
            
            if single_camera_mode:
                text_x = 20
                text_y = 60
            else:
                text_x = target_w + 20
                text_y = 40
                
            for line in lines_to_draw:
                # Black outline for visibility
                cv2.putText(img_np, line, (text_x, text_y), font_face, font_scale, (0, 0, 0), font_thickness + 2)
                # White text
                cv2.putText(img_np, line, (text_x, text_y), font_face, font_scale, (255, 255, 255), font_thickness)
                text_y += 25
            
            fig_export.canvas.draw()
            rgba = np.asarray(fig_export.canvas.buffer_rgba())
            overlay_img = rgba[:, :, :3].copy()
            
            overlay_size = (340, 340)
            overlay_img = cv2.resize(overlay_img, overlay_size)
            
            gray = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2GRAY)
            mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
            
            if single_camera_mode:
                # Place graph in top right of the single image
                roi_y1 = 40
                roi_x1 = 1280 - overlay_size[0] - 40
            else:
                # Place graph in TOP RIGHT quadrant below the text
                roi_y1 = max(140, text_y + 10)
                roi_x1 = target_w + (target_w - overlay_size[0]) // 2
                
            roi_y2 = roi_y1 + overlay_size[1]
            roi_x2 = roi_x1 + overlay_size[0]
            
            if roi_y2 <= img_np.shape[0] and roi_x2 <= img_np.shape[1]:
                roi = img_np[roi_y1:roi_y2, roi_x1:roi_x2]
                bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
                fg = cv2.bitwise_and(overlay_img, overlay_img, mask=mask)
                img_np[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.add(bg, fg)
                
            # Repeat each dataset frame so playback speed matches the source dataset FPS.
            bgr_frame = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            for _ in range(frame_repeat):
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
