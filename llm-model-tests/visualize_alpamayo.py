import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import os
import sys
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AlpamayoViewer:
    def __init__(self, root, json_path):
        self.root = root
        self.root.title("Alpamayo Results Viewer")
        self.root.geometry("1600x900")
        
        self.data = []
        self.current_index = 0
        self.json_path = json_path
        
        self.load_data()
        self.setup_ui()
        self.show_current_item()
        
        # Bind arrow keys
        self.root.bind('<Left>', lambda e: self.prev_item())
        self.root.bind('<Right>', lambda e: self.next_item())

    def load_data(self):
        if not os.path.exists(self.json_path):
            messagebox.showerror("Error", f"File not found: {self.json_path}")
            sys.exit(1)
            
        try:
            with open(self.json_path, 'r') as f:
                self.data = json.load(f)
            
            if not self.data:
                messagebox.showwarning("Warning", "JSON file is empty.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {e}")
            sys.exit(1)

    def setup_ui(self):
        # Main layout: Split pane (Image Left, Analysis Right)
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Frame (Image)
        self.image_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.image_frame, weight=3) # More weight to image
        
        self.image_label = ttk.Label(self.image_frame, text="No Image")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right Frame (Text + Plot)
        self.right_main_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_main_frame, weight=2)
        
        # Split Right Frame vertically: Top=Text, Bottom=Plot
        self.right_pane = ttk.PanedWindow(self.right_main_frame, orient=tk.VERTICAL)
        self.right_pane.pack(fill=tk.BOTH, expand=True)
        
        # Text Output Frame
        self.text_frame = ttk.Frame(self.right_pane)
        self.right_pane.add(self.text_frame, weight=1)
        
        self.text_label = ttk.Label(self.text_frame, text="Reasoning:", font=("Arial", 12, "bold"))
        self.text_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.text_output = tk.Text(self.text_frame, wrap=tk.WORD, font=("Consolas", 11), height=15)
        self.text_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Plot Frame
        self.plot_frame = ttk.Frame(self.right_pane)
        self.right_pane.add(self.plot_frame, weight=1)
        
        self.plot_label = ttk.Label(self.plot_frame, text="Predicted Trajectory (Top-Down):", font=("Arial", 12, "bold"))
        self.plot_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Controls Frame (at bottom of right main frame)
        self.controls_frame = ttk.Frame(self.right_main_frame)
        self.controls_frame.pack(fill=tk.X, pady=10)
        
        self.prev_btn = ttk.Button(self.controls_frame, text="<< Previous", command=self.prev_item)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.counter_label = ttk.Label(self.controls_frame, text="0 / 0")
        self.counter_label.pack(side=tk.LEFT, padx=20)
        
        self.next_btn = ttk.Button(self.controls_frame, text="Next >>", command=self.next_item)
        self.next_btn.pack(side=tk.LEFT, padx=5)

    def show_current_item(self):
        if not self.data:
            return
            
        item = self.data[self.current_index]
        image_path = item.get("image_path", "")
        reasoning = item.get("reasoning", "")
        trajectory = item.get("trajectory", [])
        
        # Update Counter
        self.counter_label.config(text=f"Frame {self.current_index + 1} / {len(self.data)}")
        
        # Update Text
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, f"Image: {image_path}\n\n{reasoning}")
        
        # Update Image
        self.display_image(image_path)
        
        # Update Plot
        self.update_plot(trajectory)

    def update_plot(self, trajectory):
        self.ax.clear()
        self.ax.set_title("Vehicle Trajectory (BEV)")
        self.ax.set_xlabel("Lateral (Y) [m]")
        self.ax.set_ylabel("Longitudinal (X) [m]")
        self.ax.grid(True)
        # Ensure aspect ratio is equal so turns look correct
        self.ax.set_aspect('equal', adjustable='box')
        
        if trajectory and len(trajectory) > 0:
            # Handle potential extra nesting from JSON (e.g. if shape was (1, T, 3))
            # If trajectory[0][0] is a list, it means we have [[ [x,y,z], ... ]]
            if isinstance(trajectory[0], list) and len(trajectory[0]) > 0 and isinstance(trajectory[0][0], list):
                trajectory = trajectory[0]

            # Trajectory is likely list of [x, y, z]
            # X is usually forward, Y is left/right. 
            # We want to plot X on vertical axis (Up) and Y on horizontal.
            
            try:
                xs = [p[0] for p in trajectory] # Forward
                ys = [p[1] for p in trajectory] # Lateral
                
                # Plot
                self.ax.plot(ys, xs, 'b.-', label='Predicted Path')
                
                # Mark start
                self.ax.plot(ys[0], xs[0], 'go', label='Start')
                
                self.ax.legend()
            except Exception as e:
                print(f"Plot error: {e}")
                self.ax.text(0.5, 0.5, "Data Format Error", ha='center', va='center')
        else:
            self.ax.text(0.5, 0.5, "No Trajectory Data", ha='center', va='center')
        
        self.canvas.draw()

    def display_image(self, rel_path):
        # Combine with current working directory logic if needed, but assuming relative to script execution
        if not os.path.exists(rel_path):
            self.image_label.config(text=f"Image not found:\n{rel_path}", image="")
            return
            
        try:
            img = Image.open(rel_path)
            
            # Resize logic to fit frame
            # Get current frame size
            frame_width = self.image_frame.winfo_width() or 800
            frame_height = self.image_frame.winfo_height() or 800
            
            # Calculate aspect ratio
            img_ratio = img.width / img.height
            frame_ratio = frame_width / frame_height
            
            if img_ratio > frame_ratio:
                new_width = frame_width
                new_height = int(frame_width / img_ratio)
            else:
                new_height = frame_height
                new_width = int(frame_height * img_ratio)
                
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(img) # Keep reference!
            
            self.image_label.config(image=self.tk_image, text="")
        except Exception as e:
            # self.image_label.config(text=f"Error loading image:\n{e}", image="")
            # Fallback for initial load when size is 1x1
            pass

    def next_item(self):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.show_current_item()

    def prev_item(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_item()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "alpamayo_results.json"
        
    root = tk.Tk()
    app = AlpamayoViewer(root, json_file)
    root.mainloop()
