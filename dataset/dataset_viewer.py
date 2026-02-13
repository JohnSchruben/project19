import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import os
import glob

class DatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Viewer")
        self.root.geometry("1000x700")

        self.data_files = []
        self.current_data = []
        self.current_file_path = None
        self.current_index = 0
        self.image_cache = None

        
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Select Dataset File:").pack(side=tk.LEFT, padx=5)
        
        self.file_combo = ttk.Combobox(top_frame, state="readonly", width=50)
        self.file_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_selected)
        
        self.refresh_btn = ttk.Button(top_frame, text="Refresh", command=self.scan_files)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)

        # Main Content: Split Pane
        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Image
        self.image_frame = ttk.Frame(self.paned_window, relief=tk.SUNKEN)
        self.paned_window.add(self.image_frame, width=500)
        
        self.image_label = ttk.Label(self.image_frame, text="No Image")
        self.image_label.pack(fill=tk.BOTH, expand=True)
   
        # Right: Data/Editing
        self.edit_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.edit_frame, width=400)

        ttk.Label(self.edit_frame, text="Description / Result:").pack(anchor=tk.W, pady=5)
        
        self.text_editor = tk.Text(self.edit_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        self.text_editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bottom Bar: Navigation and Save
        bottom_frame = ttk.Frame(self.root, padding=10)
        bottom_frame.pack(fill=tk.X)

        self.prev_btn = ttk.Button(bottom_frame, text="<< Previous", command=self.prev_item)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(bottom_frame, text="0 / 0")
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.next_btn = ttk.Button(bottom_frame, text="Next >>", command=self.next_item)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(bottom_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)

        self.save_btn = ttk.Button(bottom_frame, text="Save JSON", command=self.save_file)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Initial scan
        self.scan_files()

    def scan_files(self):
        """Scans for .json and .jsonl files in the current directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        files = glob.glob(os.path.join(current_dir, "*.json")) + glob.glob(os.path.join(current_dir, "*.jsonl"))
        self.data_files = [os.path.basename(f) for f in files]
        self.file_combo['values'] = self.data_files
        if self.data_files:
            self.file_combo.current(0)
            self.on_file_selected(None)
        else:
            self.file_combo.set("No JSON files found")
            self.clear_ui()

    def on_file_selected(self, event):
        filename = self.file_combo.get()
        if not filename: return
        
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        self.load_file(filepath)

    def load_file(self, filepath):
        try:
            if filepath.endswith('.jsonl'):
                data = []
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            if isinstance(data, list):
                self.current_data = data
                self.current_file_path = filepath
                self.current_index = 0
                self.show_item()
            else:
                messagebox.showerror("Error", "JSON root must be a list of objects.")
                self.current_data = []
                self.clear_ui()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.current_data = []
            self.clear_ui()

    def clear_ui(self):
        self.image_label.configure(image='', text="No Data")
        self.text_editor.delete("1.0", tk.END)
        self.status_label.config(text="0 / 0")

    def show_item(self):
        if not self.current_data:
            self.clear_ui()
            return

        item = self.current_data[self.current_index]
        
        # Update Text
        self.text_editor.delete("1.0", tk.END)
        
        if "result" in item:
            text_val = item["result"]
            self.text_editor.insert(tk.END, text_val)
        else:
            # New format: Display as pretty-printed JSON (excluding image key)
            display_data = {k: v for k, v in item.items() if k != "image"}
            text_val = json.dumps(display_data, indent=4)
            self.text_editor.insert(tk.END, text_val)

        # Update Image
        img_filename = item.get("image", "")
        self.display_image(img_filename)
        
        # Update Status
        self.status_label.config(text=f"{self.current_index + 1} / {len(self.current_data)}")

    def display_image(self, filename):
        if not filename:
            self.image_label.configure(image='', text="No Image Name")
            return
            
        # Look for image in ./images/
        base_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(base_dir, "images", filename)
        
        if not os.path.exists(img_path):
             self.image_label.configure(image='', text=f"Image not found:\n{filename}")
             return

        try:
            pil_img = Image.open(img_path)
            
            # Smart Resize
            # Get window size (approx) or fixed logic
            # Let's fit to a box of 500x500 approx, or better yet, read the frame size?
            # Frame size might be 1x1 if not packed yet.
            
            max_w, max_h = 600, 600
            pil_img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            
            self.image_cache = ImageTk.PhotoImage(pil_img)
            self.image_label.configure(image=self.image_cache, text="")
        except Exception as e:
            self.image_label.configure(image='', text=f"Error loading image:\n{e}")

    def save_current_text_to_memory(self):
        if not self.current_data: return
        
        txt = self.text_editor.get("1.0", tk.END).strip()
        item = self.current_data[self.current_index]
        
        if "result" in item:
            item["result"] = txt
        else:
            try:
                # Parse JSON edits
                new_data = json.loads(txt)
                if isinstance(new_data, dict):
                    item.update(new_data)
                else:
                    print("Warning: Content must be a JSON object.")
            except json.JSONDecodeError:
                print("Warning: Invalid JSON content - changes not saved to memory.")


    def next_item(self):
        self.save_current_text_to_memory()
        if self.current_index < len(self.current_data) - 1:
            self.current_index += 1
            self.show_item()

    def prev_item(self):
        self.save_current_text_to_memory()
        if self.current_index > 0:
            self.current_index -= 1
            self.show_item()

    def save_file(self):
        self.save_current_text_to_memory()
        if self.current_file_path and self.current_data is not None:
            try:
                if self.current_file_path.endswith('.jsonl'):
                     with open(self.current_file_path, 'w', encoding='utf-8') as f:
                        for entry in self.current_data:
                            json.dump(entry, f)
                            f.write('\n')
                else:
                    with open(self.current_file_path, 'w', encoding='utf-8') as f:
                        json.dump(self.current_data, f, indent=4)
                messagebox.showinfo("Saved", f"Successfully saved to {os.path.basename(self.current_file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetViewer(root)
    root.mainloop()
