import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import os
import sys

class AlpamayoViewer:
    def __init__(self, root, json_path):
        self.root = root
        self.root.title("Alpamayo Results Viewer")
        self.root.geometry("1400x900")
        
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
        # Main layout: Split pane (Image Left, Text Right)
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left Frame (Image)
        self.image_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.image_frame, weight=1)
        
        self.image_label = ttk.Label(self.image_frame, text="No Image")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Right Frame (Text + Controls)
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=1)
        
        # Text Output
        self.text_label = ttk.Label(self.right_frame, text="Model Output:", font=("Arial", 12, "bold"))
        self.text_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.text_output = tk.Text(self.right_frame, wrap=tk.WORD, font=("Consolas", 11))
        self.text_output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls Frame
        self.controls_frame = ttk.Frame(self.right_frame)
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
        
        # Update Counter
        self.counter_label.config(text=f"Frame {self.current_index + 1} / {len(self.data)}")
        
        # Update Text
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, f"Image: {image_path}\n\n{reasoning}")
        
        # Update Image
        self.display_image(image_path)

    def display_image(self, rel_path):
        # Combine with current working directory logic if needed, but assuming relative to script execution
        if not os.path.exists(rel_path):
            self.image_label.config(text=f"Image not found:\n{rel_path}", image="")
            return
            
        try:
            img = Image.open(rel_path)
            
            # Resize logic to fit frame
            # Get current frame size
            frame_width = self.image_frame.winfo_width() or 600
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
            self.image_label.config(text=f"Error loading image:\n{e}", image="")

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
