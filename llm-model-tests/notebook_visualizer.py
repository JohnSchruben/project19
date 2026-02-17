
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
        self.reasoning_box = widgets.VBox([self.lbl_reasoning, self.out_text], layout=widgets.Layout(width='200px', flex='0 0 auto'))

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
        
        # New Data Structures
        trajectories = item.get("trajectories", [])
        gt_trajectory = item.get("gt_trajectory", [])
        
        # Backward Compatibility
        old_trajectory = item.get("trajectory", [])
        if not trajectories and old_trajectory:
            trajectories = [old_trajectory]

        # Support both old 'speed' and new 'telemetry_data'
        telemetry_data = item.get("telemetry_data", {})
        speed = item.get("speed", "N/A")
        
        # Update Counter
        self.lbl_counter.value = f"Frame {self.current_index + 1} / {len(self.data)}"
        
        # Format Telemetry HTML
        telemetry_html = "<hr><b>Telemetry:</b><br>"
        if telemetry_data:
            for k, v in telemetry_data.items():
                # Filter out verbose fields if needed, or show all
                if k not in ["filename", "timestamp_eof"]:
                    val_str = f"{v:.2f}" if isinstance(v, float) else str(v)
                    telemetry_html += f"{k}: {val_str}<br>"
        else:
             # Fallback for old data
             telemetry_html += f"Speed: {speed}<br>"

        # Update Text
        # Formatting for readability with HTML
        # Combine Reasoning + Telemetry
        formatted_text = f"<div style='word-wrap: break-word;'>{reasoning}<br>{telemetry_html}</div>"
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
            self.plot_trajectory(trajectories, gt_trajectory)

    def plot_trajectory(self, trajectories, gt_trajectory=None):
        # Create figure
        # We use 'ioff' to prevent it from displaying automatically outside the widget
        plt.ioff()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_title("Vehicle Trajectory (BEV)")
        ax.set_xlabel("Lateral (Y) [m]")
        ax.set_ylabel("Longitudinal (X) [m]")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='datalim')
        
        all_points_x = []
        all_points_y = []

        # Plot Ground Truth (Red)
        if gt_trajectory and len(gt_trajectory) > 0:
            try:
                xs = [p[0] for p in gt_trajectory]
                ys = [p[1] for p in gt_trajectory]
                ax.plot(ys, xs, 'r-', linewidth=2, label='Ground Truth')
                all_points_x.extend(xs)
                all_points_y.extend(ys)
            except Exception as e:
                print(f"Error plotting GT: {e}")

        # Plot Predictions
        colors = ['b', 'c', 'm', 'orange', 'purple']
        if trajectories and len(trajectories) > 0:
            for i, traj in enumerate(trajectories):
                # Flatten/Unwrap logic to handle various nestings
                # We expect traj to be List[[x,y,z], [x,y,z], ...]
                # If it is List[List[List...]], unwrap it
                
                temp_traj = traj
                depth = 0
                while isinstance(temp_traj, (list, tuple)) and len(temp_traj) > 0 and depth < 3:
                     # Check if the first element is a point [x, y, z] (numbers)
                     first_elem = temp_traj[0]
                     if isinstance(first_elem, (list, tuple)):
                         # It's a list, check if it's a point
                         if len(first_elem) >= 2 and isinstance(first_elem[0], (int, float)):
                             # Found the list of points!
                             break
                         else:
                             # It's a list of lists (not points), drill down
                             # Assuming we want the first trajectory if multiple are wrapped?
                             # Actually for "trajectories" list, each item SHOULD be a trajectory.
                             # If "traj" itself is a list of trajectories, we might be iterating wrong.
                             # But let's assume valid data is just nested.
                             temp_traj = temp_traj[0] # Take first element? Or flatten?
                             # Taking first element might lose data if it was a batch.
                             # But here we are inside "for traj in trajectories".
                             pass 
                     else:
                         # It's not a list, maybe it's numbers?
                         # If temp_traj is [x, y, z], then we shouldn't be here in this loop unless we treat single point as traj?
                         pass
                     
                     depth += 1
                     
                # Re-verify we have points
                # If temp_traj is [ [x,y], [x,y] ]
                
                try:
                    # Validate first point
                    if not isinstance(temp_traj, (list, tuple)) or len(temp_traj) == 0:
                        continue
                        
                    p0 = temp_traj[0]
                    if not (isinstance(p0, (list, tuple)) and len(p0) >= 2 and isinstance(p0[0], (int, float))):
                        # print(f"Skipping invalid traj structure: {type(p0)}")
                        continue

                    xs = [float(p[0]) for p in temp_traj]
                    ys = [float(p[1]) for p in temp_traj]
                    
                    color = colors[i % len(colors)]
                    label = f'Pred {i+1}' if i < 3 else None
                    
                    ax.plot(ys, xs, color=color, marker='.', markersize=2, linestyle='-', linewidth=1, label=label)
                    
                    if i == 0:
                        ax.plot(ys[0], xs[0], 'go', label='Start')
                    
                    all_points_x.extend(xs)
                    all_points_y.extend(ys)
                except Exception as e:
                    print(f"Error plotting traj {i}: {e}")

        if len(all_points_x) > 0:
             # Dynamic limits to show full path
            try:
                min_lat = min(all_points_y)
                max_lat = max(all_points_y)
                min_long = min(all_points_x)
                max_long = max(all_points_x)
                
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
                
                ax.legend(loc='upper right', fontsize='small')
            except Exception as e:
                print(f"Error setting plot limits: {e}")
                # Fallback limits
                ax.set_xlim(-10, 10)
                ax.set_ylim(0, 50)
        else:
            ax.text(0.5, 0.5, "No Valid Trajectory Data", ha='center', va='center')
            
        # Display the figure in the output widget
        display(fig)
        # Close the figure to free memory
        plt.close(fig)

if __name__ == "__main__":
    print("This module is intended to be imported in a Jupyter Notebook.")
    print("Usage:")
    print("  from notebook_visualizer import NotebookVisualizer")
    print("  viz = NotebookVisualizer('route_1_seg_00_results.json')")
