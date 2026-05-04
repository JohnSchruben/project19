import os
import cv2
import glob
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Create a 4-quadrant video from all cameras in a route.")
    parser.add_argument("--route", type=str, default="../datasets/route_3", help="Path to the route directory")
    parser.add_argument("--output", type=str, default="quadrant_preview.mp4", help="Output video path")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS")
    args = parser.parse_args()

    # Find all segments
    segments = sorted([d for d in glob.glob(os.path.join(args.route, "segment_*")) if os.path.isdir(d)])
    if not segments:
        print(f"No segments found in {args.route}")
        return

    # To ensure consistent grid, resize all to the same resolution
    # Standardizing to 640x480 or 800x600? 640x480 seems standard for these dashcams.
    target_w, target_h = 640, 480
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (target_w * 2, target_h * 2))

    if not out.isOpened():
        print(f"Failed to open video writer for {args.output}")
        return

    print(f"Generating video: {args.output}")

    for seg_dir in segments:
        seg_name = os.path.basename(seg_dir)
        print(f"Processing {seg_name}...")
        
        # We assume sequence lengths are defined by raw_front or whatever comes first
        cameras = ["raw_left", "raw_front", "raw_right", "raw"]
        
        # Discover how many frames exist by finding the max index in any camera folder
        max_idx = 0
        for cam in cameras:
            cam_dir = os.path.join(seg_dir, cam)
            if os.path.exists(cam_dir):
                frames = glob.glob(os.path.join(cam_dir, "*.png")) + glob.glob(os.path.join(cam_dir, "*.jpg"))
                if frames:
                    # Extracts exactly the numbers from strings like '000000.png'
                    indices = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames if os.path.basename(f).split('.')[0].isdigit()]
                    if indices:
                        max_idx = max(max_idx, max(indices))

        if max_idx == 0:
            print(f"No valid frames found in {seg_name}, skipping.")
            continue

        for local_idx in range(max_idx + 1):
            quadrants = []
            
            for cam in cameras:
                cam_dir = os.path.join(seg_dir, cam)
                img_path_png = os.path.join(cam_dir, f"{local_idx:06d}.png")
                img_path_jpg = os.path.join(cam_dir, f"{local_idx:06d}.jpg")
                
                img_path = None
                if os.path.exists(img_path_png):
                    img_path = img_path_png
                elif os.path.exists(img_path_jpg):
                    img_path = img_path_jpg
                
                if img_path:
                    img = cv2.imread(img_path)
                else:
                    img = None
                    
                if img is None:
                    # placeholder black frame with white text
                    img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    cv2.putText(img, f"{cam} missing", (50, target_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                else:
                    if img.shape[:2] != (target_h, target_w):
                        img = cv2.resize(img, (target_w, target_h))
                        
                # Label the camera
                cv2.putText(img, cam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                quadrants.append(img)
                
            # Layout:
            # Top-Left: raw_left      Top-Right: raw_front
            # Bottom-Left: raw_right  Bottom-Right: raw (wide)
            top_row = np.hstack((quadrants[0], quadrants[1]))
            bottom_row = np.hstack((quadrants[2], quadrants[3]))
            grid = np.vstack((top_row, bottom_row))
            
            # Put segment + frame text at the very bottom center
            text = f"{seg_name} | Frame {local_idx:06d}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(grid, text, (target_w - tw//2, target_h * 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            out.write(grid)

    out.release()
    print(f"Done! Video saved to {args.output}")

if __name__ == "__main__":
    main()
