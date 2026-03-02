import os
import sys
import glob
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'alpamayo', 'src')))
from alpamayo_r1.load_custom_dataset import load_custom_dataset

def main():
    segment_dir = r"c:\Users\johns\OneDrive\Documents\OU\Classes\Spring 2026\CS-4273-001-cap-stone\project19\datasets\route_1\segment_05" # assuming route_1
    if not os.path.exists(segment_dir):
        # find segment 05
        base = r"c:\Users\johns\OneDrive\Documents\OU\Classes\Spring 2026\CS-4273-001-cap-stone\project19\datasets"
        segs = glob.glob(os.path.join(base, "*", "segment_05"))
        if segs:
            segment_dir = segs[0]
        else:
            print("segment 05 not found")
            return
            
    telemetry_dir = os.path.join(segment_dir, "telemetry")
    num_frames = len(glob.glob(os.path.join(telemetry_dir, "*.json")))
    
    with open("debug_out.txt", "w") as f:
        f.write(f"Testing {segment_dir}\n")
        
        for local_idx in range(0, num_frames, 10): # skip every 10 frames for speed
            try:
                data = load_custom_dataset(segment_dir, local_idx)
                gt_rot = data["ego_future_rot"][0, 0].numpy()
                full_frames = gt_rot.shape[0]
                if full_frames > 0:
                    headings = np.degrees(np.arctan2(gt_rot[:, 1, 0], gt_rot[:, 0, 0]))
                    max_left_turn = np.max(headings)
                    max_right_turn = np.min(headings)
                    f.write(f"Frame {local_idx}: Left={max_left_turn:.1f} Right={max_right_turn:.1f}\n")
            except Exception as e:
                pass

if __name__ == '__main__':
    main()
