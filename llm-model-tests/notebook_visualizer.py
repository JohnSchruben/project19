
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import ipywidgets as widgets
from IPython.display import display, clear_output

class NotebookVisualizer:
    def __init__(self, json_path):
        self.json_path = json_path
        self.data = []
        self.current_index = 0
        
        self.load_data()
        self.setup_ui()
        self.show_current_item()

    def load_data(self):
        if not os.path.exists(self.json_path):
            print(f"Error: File not found: {self.json_path}")
            return
            
        try:
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
            
            if not self.data:
                print("Warning: JSON file is empty.")
        except Exception as e:
            print(f"Error: Failed to load JSON: {e}")

    def setup_ui(self):
        if not self.data:
            return

        # Navigation Buttons
        self.btn_prev = widgets.Button(description="<< Previous")
        self.btn_next = widgets.Button(description="Next >>")
        self.lbl_counter = widgets.Label(value=f"Frame 1 / {len(self.data)}")
        
        self.btn_prev.on_click(self.on_prev)
        self.btn_next.on_click(self.on_next)
        
        self.controls = widgets.HBox([self.btn_prev, self.lbl_counter, self.btn_next])
        
        # Output Area for Image and Text
        self.out_image = widgets.Output() # For the image
        self.out_plot = widgets.Output()  # For the trajectory plot
        self.out_text = widgets.Textarea(
            value="",
            placeholder="Reasoning will appear here...",
            description="Reasoning:",
            disabled=True,
            layout=widgets.Layout(width='100%', height='200px')
        )

        display(self.controls)
        
        # Layout: Image Left, Plot Right, Text Bottom
        self.visuals = widgets.HBox([self.out_image, self.out_plot])
        display(self.visuals)
        display(self.out_text)

    def on_prev(self, b):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_item()

    def on_next(self, b):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.show_current_item()

    def show_current_item(self):
        if not self.data:
            return

        item = self.data[self.current_index]
        image_path = item.get("image_path", "")
        reasoning = item.get("reasoning", "")
        trajectory = item.get("trajectory", [])
        
        # Update Counter
        self.lbl_counter.value = f"Frame {self.current_index + 1} / {len(self.data)}"
        
        # Update Text
        # Formatting for readability
        formatted_text = f"Image: {image_path}\n\n{reasoning}"
        self.out_text.value = formatted_text
        
        # Update Image
        with self.out_image:
            clear_output(wait=True)
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    # Resize for display uniqueness if needed, or just display
                    # Constrain size
                    img.thumbnail((600, 600))
                    display(img)
                except Exception as e:
                    print(f"Error loading image: {e}")
            else:
                print(f"Image not found: {image_path}")
        
        # Update Plot
        with self.out_plot:
            clear_output(wait=True)
            self.plot_trajectory(trajectory)

    def plot_trajectory(self, trajectory):
        # Create figure
        # We use 'ioff' to prevent it from displaying automatically outside the widget
        plt.ioff()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title("Vehicle Trajectory (BEV)")
        ax.set_xlabel("Lateral (Y) [m]")
        ax.set_ylabel("Longitudinal (X) [m]")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        
        if trajectory and len(trajectory) > 0:
            # Flatten logic same as visualize_alpamayo.py
            depth_limit = 5
            curr_depth = 0
            while isinstance(trajectory, list) and len(trajectory) > 0 and curr_depth < depth_limit:
                first_item = trajectory[0]
                if isinstance(first_item, list) and len(first_item) == 3 and all(isinstance(x, (int, float)) for x in first_item):
                    break
                if isinstance(first_item, list):
                    trajectory = first_item
                else:
                    break
                curr_depth += 1

            try:
                xs = [p[0] for p in trajectory]
                ys = [p[1] for p in trajectory]
                
                ax.plot(ys, xs, 'b.-', label='Path')
                ax.plot(ys[0], xs[0], 'go', label='Start')
                
                # Fixed limits
                min_lat = min(ys)
                max_lat = max(ys)
                buffer = 5.0
                center_lat = (min_lat + max_lat) / 2.0
                ax.set_xlim(center_lat - buffer, center_lat + buffer)
                
                min_long = min(xs)
                max_long = max(xs)
                if max_long - min_long < 10.0:
                    ax.set_ylim(min_long - 2.0, min_long + 12.0)
                    
                ax.legend()
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "No Trajectory Data", ha='center', va='center')
            
        # Display the figure in the output widget
        display(fig)
        # Close the figure to free memory
        plt.close(fig)

if __name__ == "__main__":
    print("This module is intended to be imported in a Jupyter Notebook.")
    print("Usage:")
    print("  from notebook_visualizer import NotebookVisualizer")
    print("  viz = NotebookVisualizer('route_1_seg_00_results.json')")
