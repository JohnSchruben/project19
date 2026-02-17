
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
        
        self.slider = widgets.IntSlider(
            value=0,
            min=0,
            max=max(0, len(self.data) - 1),
            step=1,
            description='Frame:',
            continuous_update=False
        )
        
        self.btn_prev.on_click(self.on_prev)
        self.btn_next.on_click(self.on_next)
        self.slider.observe(self.on_slider_change, names='value')
        
        self.controls = widgets.HBox([self.btn_prev, self.slider, self.lbl_counter, self.btn_next])
        
        # Output Area for Image and Text
        # Image width doubled to 300px
        self.out_image = widgets.Output(layout=widgets.Layout(width='300px', flex='0 0 auto')) 
        
        self.out_plot = widgets.Output(layout=widgets.Layout(flex='1 1 auto', width='auto', min_width='400px'))  
        
        self.lbl_reasoning = widgets.Label(value="Reasoning:")
        # Use HTML for better text wrapping and formatting
        self.out_text = widgets.HTML(
            value="",
            layout=widgets.Layout(width='200px', height='380px', overflow='auto')
        )
        self.lbl_telemetry = widgets.Label(value="Telemetry: N/A")
        self.reasoning_box = widgets.VBox([self.lbl_reasoning, self.out_text, self.lbl_telemetry], layout=widgets.Layout(width='200px', flex='0 0 auto'))

        display(self.controls)
        
        # Layout: Image | Text (w/ Label) | Plot (Grid)
        # flex_flow='row' ensures they stay in a row
        self.visuals = widgets.HBox(
            [self.out_image, self.reasoning_box, self.out_plot], 
            layout=widgets.Layout(width='100%', flex_flow='row', align_items='flex-start')
        )
        display(self.visuals)

    def on_prev(self, b):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_item()

    def on_next(self, b):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.show_current_item()

    def on_slider_change(self, change):
        self.current_index = change['new']
        self.show_current_item()

    def show_current_item(self):
        if not self.data:
            return

        # Update slider without triggering check loop if needed (observe handles it fine usually)
        self.slider.value = self.current_index

        item = self.data[self.current_index]
        image_path = item.get("image_path", "")
        reasoning = item.get("reasoning", "")
        trajectory = item.get("trajectory", [])
        speed = item.get("speed", "N/A")
        
        # Update Counter
        self.lbl_counter.value = f"Frame {self.current_index + 1} / {len(self.data)}"
        
        # Update Telemetry
        if isinstance(speed, (int, float)):
            self.lbl_telemetry.value = f"Speed: {speed:.2f} m/s"
        else:
            self.lbl_telemetry.value = f"Speed: {speed}"

        # Update Text
        # Formatting for readability with HTML
        formatted_text = f"<div style='word-wrap: break-word;'>{reasoning}</div>"
        self.out_text.value = formatted_text
        
        # Update Image
        with self.out_image:
            clear_output(wait=True)
            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    # Resize for display uniqueness if needed, or just display
                    # Constrain size width to match widget
                    img.thumbnail((300, 300))
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
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title("Vehicle Trajectory (BEV)")
        ax.set_xlabel("Lateral (Y) [m]")
        ax.set_ylabel("Longitudinal (X) [m]")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='datalim')
        
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
                
                # Dynamic limits to show full path
                min_lat = min(ys)
                max_lat = max(ys)
                min_long = min(xs)
                max_long = max(xs)
                
                # Enforce minimum width/height to avoid thin lines
                lat_span = max(10.0, max_lat - min_lat)
                long_span = max(15.0, max_long - min_long)
                
                # Center view
                center_lat = (min_lat + max_lat) / 2.0
                center_long = (min_long + max_long) / 2.0
                
                # Add buffer
                buffer = 2.0
                ax.set_xlim(center_lat - lat_span/2.0 - buffer, center_lat + lat_span/2.0 + buffer)
                ax.set_ylim(center_long - long_span/2.0 - buffer, center_long + long_span/2.0 + buffer)
                    
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
