import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Add path to alpamayo modules so we can import load_custom_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../alpamayo/src/alpamayo_r1')))
from load_custom_dataset import load_custom_dataset

class GTVisualizer:
    def __init__(self, segment_dir, num_future_frames=10):
        self.segment_dir = segment_dir
        self.frame_idx = 0
        self.num_future_frames = num_future_frames
        
        # Setup matplotlib figure
        self.fig, (self.ax_img, self.ax_graph) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.canvas.manager.set_window_title("Ground Truth Path Visualizer")
        plt.subplots_adjust(bottom=0.2)
        
        # Add buttons
        self.ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
        self.ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
        
        self.btn_prev = Button(self.ax_prev, 'Previous Frame')
        self.btn_next = Button(self.ax_next, 'Next Frame')
        
        self.btn_prev.on_clicked(self.prev_frame)
        self.btn_next.on_clicked(self.next_frame)
        
        self.update_display()
        
    def get_data(self):
        try:
            # Load dataset for the current frame
            # num_future_steps is passed to restrict the future window size
            data = load_custom_dataset(
                segment_dir=self.segment_dir,
                frame_idx=self.frame_idx,
                num_future_steps=self.num_future_frames
            )
            return data
        except Exception as e:
            print(f"Error loading frame {self.frame_idx}: {e}")
            return None

    def update_display(self):
        data = self.get_data()
        if not data:
            self.fig.suptitle(f"Error loading Frame {self.frame_idx} (might be end of segment)", color='red')
            self.fig.canvas.draw()
            return
            
        self.fig.suptitle("")
            
        # 1. Update Image (the most recent frame in the history buffer)
        # 'image_frames' shape: (1, num_visual_frames, 3, H, W)
        img_tensor = data["image_frames"][0, -1]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        self.ax_img.clear()
        self.ax_img.imshow(img_np)
        self.ax_img.set_title(f"Raw Frame (Index: {self.frame_idx})")
        self.ax_img.axis('off')
        
        # 2. Update Graph (Trajectory)
        # 'ego_future_xyz' shape: (1, 1, steps, 3)
        future_xyz = data["ego_future_xyz"][0, 0].numpy()
        
        self.ax_graph.clear()
        
        # Plot future trajectory
        # Typically openpilot x is forward, y is left
        x_forward = future_xyz[:self.num_future_frames, 0]
        y_left = future_xyz[:self.num_future_frames, 1]
        
        # Plot Y (horizontal space) vs X (forward space)
        self.ax_graph.plot(y_left, x_forward, marker='o', color='blue', label='Ground Truth Trajectory')
        
        # Mark Ego Car start position
        self.ax_graph.plot(0, 0, marker='*', color='red', markersize=15, label='Ego Vehicle')
        
        self.ax_graph.set_title(f"Next {self.num_future_frames} Frames (Kinematic GT)")
        self.ax_graph.set_xlabel("Lateral Displacement (m) [Left is Positive]")
        self.ax_graph.set_ylabel("Forward Displacement (m)")
        self.ax_graph.grid(True)
        self.ax_graph.legend()
        
        # Keeps the X/Y scale 1:1 so turns look realistic
        self.ax_graph.set_aspect('equal', 'box')
        
        # Redraw
        self.fig.canvas.draw_idle()
        
    def next_frame(self, event):
        self.frame_idx += 1
        self.update_display()
        
    def prev_frame(self, event):
        self.frame_idx = max(0, self.frame_idx - 1)
        self.update_display()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GUI to visualize dataset raw frames and kinematic ground truth")
    parser.add_argument("--segment", type=str, default="../datasets/route_1/segment_00", 
                        help="Path to segment directory (default: ../datasets/route_1/segment_00)")
    parser.add_argument("--frames", type=int, default=10, 
                        help="Number of future frames to graph (default: 10)")
    
    args = parser.parse_args()
    
    print(f"Loading segment from: {args.segment}")
    app = GTVisualizer(segment_dir=args.segment, num_future_frames=args.frames)
    plt.show()
