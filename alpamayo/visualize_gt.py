import sys
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
from PIL import Image
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src/alpamayo1_5')))
from load_custom_dataset import load_custom_dataset

class GTVisualizerApp:
    def __init__(self, root, route_dir=None, initial_frames=4):
        self.root = root
        self.root.title("Ground Truth Visualizer")
        self.root.geometry("1400x900")
        
        self.route_dir = route_dir
        self.all_segments = []
        self.segment_vars = []
        self.segment_dirs = []
            
        self.segment_lens = []
        self.num_total_frames = 0
        
        self.frame_idx = 0
        self.num_frames = initial_frames
        
        # Setup Styles for considerably larger UI elements
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 28))
        style.configure('TLabel', font=('Helvetica', 28))
        style.configure('TEntry', font=('Helvetica', 28))
        style.configure('TNotebook.Tab', font=('Helvetica', 28, 'bold'), padding=[20, 10])
        
        # Top Global Controls
        top_frame = ttk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        ttk.Button(top_frame, text="Select Route", command=self.select_route).pack(side=tk.LEFT, padx=5)
        
        self.btn_export = ttk.Button(top_frame, text="Export MP4", command=self.export_video, state=tk.DISABLED)
        self.btn_export.pack(side=tk.LEFT, padx=5)
        
        self.lbl_segments = ttk.Label(top_frame, text="Loaded Segments: 0 | Total Frames: 0")
        self.lbl_segments.pack(side=tk.LEFT, padx=20)
        
        # Checkbox panel
        self.chk_frame_container = ttk.Frame(root)
        self.chk_frame_container.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        
        # Tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.tab_seq = ttk.Frame(self.notebook)
        self.tab_full = ttk.Frame(self.notebook)
        self.tab_telemetry = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_seq, text="Frames Sequence")
        self.notebook.add(self.tab_full, text="Full Segments Path")
        self.notebook.add(self.tab_telemetry, text="Telemetry Graphs")
        
        self.setup_seq_tab()
        self.setup_full_tab()
        self.setup_telemetry_tab()
        
        if self.route_dir:
            self.load_route(self.route_dir)
        
    def load_segments_data(self):
        self.segment_lens = []
        self.num_total_frames = 0
        for seg in self.segment_dirs:
            telemetry_dir = os.path.join(seg, "telemetry")
            if os.path.exists(telemetry_dir):
                length = len(glob.glob(os.path.join(telemetry_dir, "*.json")))
            else:
                length = 0
            self.segment_lens.append(length)
            self.num_total_frames += length
            
        if hasattr(self, 'slider_frame'):
            # Allow jumping anywhere within the full sequence
            self.slider_frame.config(to=max(0, self.num_total_frames - 1))
            
        if hasattr(self, 'lbl_segments'):
            self.lbl_segments.config(text=f"Loaded Segments: {len(self.segment_dirs)} | Total Frames: {self.num_total_frames}")
            
        if hasattr(self, 'btn_export'):
            if self.num_total_frames > 0:
                self.btn_export.config(state=tk.NORMAL)
            else:
                self.btn_export.config(state=tk.DISABLED)

    def select_route(self):
        initial_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
        folder = filedialog.askdirectory(title="Select Route Directory", initialdir=initial_dir)
        if folder:
            self.load_route(folder)
            
    def load_route(self, route_dir):
        self.route_dir = route_dir
        self.all_segments = sorted([d for d in glob.glob(os.path.join(self.route_dir, 'segment_*')) if os.path.isdir(d)])
        self.segment_vars = [tk.BooleanVar(value=(i == 0)) for i in range(len(self.all_segments))]
        self.update_checkbox_ui()
        self.on_segment_toggled()
        
    def update_checkbox_ui(self):
        for widget in self.chk_frame_container.winfo_children():
            widget.destroy()
            
        ttk.Label(self.chk_frame_container, text="Toggle Segments:").pack(side=tk.LEFT, padx=5)
        ttk.Button(self.chk_frame_container, text="All", command=self.check_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.chk_frame_container, text="None", command=self.uncheck_all).pack(side=tk.LEFT, padx=2)
        
        chk_inner = tk.Frame(self.chk_frame_container)
        chk_inner.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        for i, (seg, var) in enumerate(zip(self.all_segments, self.segment_vars)):
            name = os.path.basename(seg)
            cb = tk.Checkbutton(chk_inner, text=name, variable=var, command=self.on_segment_toggled, font=('Helvetica', 24))
            cb.grid(row=i//10, column=i%10, sticky='w', padx=10, pady=5)
            
    def check_all(self):
        for var in self.segment_vars: var.set(True)
        self.on_segment_toggled()
        
    def uncheck_all(self):
        for var in self.segment_vars: var.set(False)
        self.on_segment_toggled()
        
    def on_segment_toggled(self):
        self.segment_dirs = [s for s, v in zip(self.all_segments, self.segment_vars) if v.get()]
        self.frame_idx = 0
        self.load_segments_data()
        self.plot_full_path_and_telemetry()
        self.update_seq_display()

    def map_global_to_local(self, global_idx):
        if not self.segment_dirs or global_idx < 0:
            return None, 0
            
        curr = 0
        for seg_dir, length in zip(self.segment_dirs, self.segment_lens):
            if global_idx < curr + length:
                return seg_dir, global_idx - curr
            curr += length
            
        if self.segment_dirs:
            return self.segment_dirs[-1], max(0, self.segment_lens[-1] - 1)
        return None, 0

    def setup_seq_tab(self):
        # Controls frame
        ctrl_frame = ttk.Frame(self.tab_seq)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        ttk.Label(ctrl_frame, text="Number of Frames (N):").pack(side=tk.LEFT, padx=5)
        self.entry_n = tk.Entry(ctrl_frame, width=5, font=('Helvetica', 28))
        self.entry_n.insert(0, str(self.num_frames))
        self.entry_n.pack(side=tk.LEFT, padx=15, pady=5)
        
        ttk.Button(ctrl_frame, text="Apply N", command=self.apply_n).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(ctrl_frame, text="Prev Set", command=self.prev_set).pack(side=tk.LEFT, padx=20)
        ttk.Button(ctrl_frame, text="Next Set", command=self.next_set).pack(side=tk.LEFT, padx=5)
        
        # Add Slider
        self.slider_var = tk.DoubleVar()
        self.slider_frame = tk.Scale(ctrl_frame, from_=0, to=0, orient=tk.HORIZONTAL, 
                                     variable=self.slider_var, command=self.on_slider_moved, 
                                     length=300, width=40, sliderlength=40)
        self.slider_frame.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        def on_slider_click(event):
            part = self.slider_frame.identify(event.x, event.y)
            if part in ('trough1', 'trough2'):
                min_val = self.slider_frame.cget("from")
                max_val = self.slider_frame.cget("to")
                width = self.slider_frame.winfo_width()
                pad = 16
                clickable_width = width - 2 * pad
                x = max(0, min(event.x - pad, clickable_width))
                if clickable_width > 0:
                    val = min_val + (x / clickable_width) * (max_val - min_val)
                    self.slider_var.set(int(val))
                    self.on_slider_moved(val)
                return "break"

        self.slider_frame.bind("<Button-1>", on_slider_click)
        
        self.lbl_status = ttk.Label(ctrl_frame, text="")
        self.lbl_status.pack(side=tk.LEFT, padx=20)
        
        # Matplotlib figure
        self.fig_seq = plt.figure(figsize=(15, 8))
        self.canvas_seq = FigureCanvasTkAgg(self.fig_seq, master=self.tab_seq)
        self.canvas_seq.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def setup_full_tab(self):
        self.fig_full = plt.figure(figsize=(10, 8))
        self.ax_full = self.fig_full.add_subplot(111)
        self.canvas_full = FigureCanvasTkAgg(self.fig_full, master=self.tab_full)
        self.canvas_full.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
    def setup_telemetry_tab(self):
        self.fig_telem, ((self.ax_v, self.ax_steer), (self.ax_accel, self.ax_pedal)) = plt.subplots(2, 2, figsize=(12, 10))
        self.canvas_telem = FigureCanvasTkAgg(self.fig_telem, master=self.tab_telemetry)
        self.canvas_telem.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Calculate full path & telemetry curves
        # will be called at end of init if args provided
        
    def on_slider_moved(self, val):
        new_idx = int(float(val))
        if new_idx != self.frame_idx:
            self.frame_idx = new_idx
            self.update_seq_display()

    def apply_n(self):
        try:
            new_n = int(self.entry_n.get())
            if new_n > 0:
                self.num_frames = new_n
                self.update_seq_display()
        except ValueError:
            pass
            
    def next_set(self):
        self.frame_idx = min(self.frame_idx + self.num_frames, max(0, self.num_total_frames - 1))
        self.slider_var.set(self.frame_idx)
        self.update_seq_display()
        
    def prev_set(self):
        self.frame_idx = max(0, self.frame_idx - self.num_frames)
        self.slider_var.set(self.frame_idx)
        self.update_seq_display()
        
    def get_frame_data(self, global_idx):
        seg_dir, local_idx = self.map_global_to_local(global_idx)
        if not seg_dir: return np.zeros((224, 224, 3), dtype=np.uint8), 0.0, 0.0, 0.0
        
        # image
        img_path = os.path.join(seg_dir, "raw", f"{local_idx:06d}.png")
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)
        else:
            img_np = np.zeros((224, 224, 3), dtype=np.uint8)
            
        telemetry_dir = os.path.join(seg_dir, "telemetry")
        json_path = os.path.join(telemetry_dir, f"{local_idx:06d}.json")
        v_ego, yaw_rate, steer = 0.0, 0.0, 0.0
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                v_ego = data.get('v_ego', 0.0)
                yaw_rate = data.get('yaw_rate', 0.0)
                steer = data.get('steering_angle_deg', 0.0)
                if data.get('gear_shifter') == 'reverse':
                    v_ego = -v_ego
        return img_np, v_ego, yaw_rate, steer

    def update_seq_display(self):
        self.fig_seq.clf()
        
        self.slider_var.set(self.frame_idx)
        
        seg_dir, local_idx = self.map_global_to_local(self.frame_idx)
        
        data = None
        if seg_dir:
            try:
                data = load_custom_dataset(
                    segment_dir=seg_dir,
                    frame_idx=local_idx,
                    num_future_steps=self.num_frames
                )
            except Exception as e:
                pass
            
        seg_name = os.path.basename(seg_dir) if seg_dir else 'None'
        self.lbl_status.config(text=f"Frames: {self.frame_idx} - {self.frame_idx + self.num_frames - 1}  [{seg_name}]")
        
        if not data:
            self.fig_seq.suptitle(f"Error loading Frame {self.frame_idx} (might be end of segment)", color='red')
            self.canvas_seq.draw()
            return

        gs = gridspec.GridSpec(3, self.num_frames, height_ratios=[3, 1, 4])
        
        for i in range(self.num_frames):
            frame_offset = self.frame_idx + i
            img_np, v_ego, yaw_rate, steer = self.get_frame_data(frame_offset)
            
            ax_img = self.fig_seq.add_subplot(gs[0, i])
            ax_img.imshow(img_np)
            ax_img.set_title(f"idx: {frame_offset}", fontsize=28, fontweight='bold')
            ax_img.axis('off')
            
            ax_txt = self.fig_seq.add_subplot(gs[1, i])
            ax_txt.axis('off')
            text_str = f"v: {v_ego:.1f}\n\nst: {steer:.1f}"
            ax_txt.text(0.5, 0.5, text_str, ha='center', va='center', fontsize=28, fontweight='bold')
            
        ax_graph = self.fig_seq.add_subplot(gs[2, :])
        
        # 'ego_future_xyz' shape: (1, 1, steps, 3)
        future_xyz = data["ego_future_xyz"][0, 0].numpy()
        x_forward = np.concatenate(([0.0], future_xyz[:self.num_frames, 0]))
        y_left = np.concatenate(([0.0], future_xyz[:self.num_frames, 1]))
        
        ax_graph.plot([-y for y in y_left], x_forward, marker='o', color='blue', label='Ground Truth Trajectory')
        ax_graph.plot(0, 0, marker='*', color='red', markersize=15, label='Ego Vehicle')
        ax_graph.set_title(f"Next {self.num_frames} Frames (GT)", fontsize=24)
        ax_graph.set_xlabel("Lateral Displacement", fontsize=20)
        ax_graph.set_ylabel("Forward Displacement", fontsize=20)
        ax_graph.grid(True)
        ax_graph.legend(fontsize=20)
        ax_graph.set_aspect('auto')
        
        # Force the X-axis (Lateral) to be somewhat wide, but not too wide 
        # so turning is still visible (using a minimum of 1 meter padding instead of 10)
        cur_xlim = ax_graph.get_xlim()
        x_center = (cur_xlim[0] + cur_xlim[1]) / 2.0
        width = cur_xlim[1] - cur_xlim[0]
        if width < 2.0:
            ax_graph.set_xlim([x_center - 1.0, x_center + 1.0])
            
        cur_ylim = ax_graph.get_ylim()
        y_center = (cur_ylim[0] + cur_ylim[1]) / 2.0
        height = cur_ylim[1] - cur_ylim[0]
        if height < 2.0:
            ax_graph.set_ylim([y_center - 1.0, y_center + 1.0])
        
        self.fig_seq.subplots_adjust(bottom=0.1, top=0.9, hspace=0.3, left=0.05, right=0.95)
        self.canvas_seq.draw()

    def plot_full_path_and_telemetry(self):
        self.ax_full.clear()
        self.ax_v.clear()
        self.ax_steer.clear()
        self.ax_accel.clear()
        self.ax_pedal.clear()
        
        if not self.segment_dirs:
            self.canvas_full.draw()
            self.canvas_telem.draw()
            return
            
        x, y, theta = 0.0, 0.0, 0.0
        path_x = [0.0]
        path_y = [0.0]
        
        v_ego_list = []
        steer_list = []
        a_ego_list = []
        gas_list = []
        brake_list = []
        time_list = []
        curr_time = 0.0
        
        dt_default = 0.05 # matched from load_custom_dataset default
        
        for seg_dir in self.segment_dirs:
            telemetry_dir = os.path.join(seg_dir, "telemetry")
            if not os.path.exists(telemetry_dir):
                continue
                
            json_files = sorted(glob.glob(os.path.join(telemetry_dir, "*.json")))
            prev_t = -1
            
            for jf in json_files:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    v = data.get('v_ego', 0.0)
                    yaw_rate = data.get('yaw_rate', 0.0)
                    steer_deg = data.get('steering_angle_deg', 0.0)
                    cur_t = data.get('timestamp_eof', 0) / 1000
                    
                    a_ego = data.get('a_ego', 0.0)
                    gas = data.get('gas', 0.0)
                    brake = data.get('brake', 0.0)
                    
                    if data.get('gear_shifter') == 'reverse':
                        v = -v
                    
                    dt = dt_default
                    if prev_t > 0 and cur_t > 0:
                        dt = (cur_t - prev_t) / 1000000.0
                    
                    if abs(yaw_rate) < 1e-4 and abs(steer_deg) > 0.5:
                        steer_rad = np.deg2rad(steer_deg) / 15.49
                        yaw_rate = v * np.tan(steer_rad) / 2.7
                        
                x += v * np.cos(theta) * dt
                y += v * np.sin(theta) * dt
                theta += yaw_rate * dt
                
                path_x.append(x)
                path_y.append(y)
                prev_t = cur_t
                
                v_ego_list.append(v)
                steer_list.append(steer_deg)
                a_ego_list.append(a_ego)
                gas_list.append(gas)
                brake_list.append(brake)
                time_list.append(curr_time)
                curr_time += dt
                
        self.ax_full.plot([-py for py in path_y], path_x, color='green', linewidth=2, label="Full Trajectory")
        self.ax_full.plot(0, 0, marker='*', color='red', markersize=15, label='Start')
        self.ax_full.plot(-path_y[-1], path_x[-1], marker='o', color='black', markersize=8, label='End')
        
        self.ax_full.set_title(f"Full Integration ({self.num_total_frames} frames)", fontsize=24)
        self.ax_full.set_xlabel("Lateral Displacement", fontsize=20)
        self.ax_full.set_ylabel("Forward Displacement", fontsize=20)
        self.ax_full.grid(True)
        self.ax_full.legend(fontsize=20)
        self.ax_full.set_aspect('auto')
        
        # Force the X-axis (Lateral) wider on the full segment too
        cur_xlim = self.ax_full.get_xlim()
        x_center = (cur_xlim[0] + cur_xlim[1]) / 2.0
        width = cur_xlim[1] - cur_xlim[0]
        if width < 10.0:
            self.ax_full.set_xlim([x_center - 5.0, x_center + 5.0])
            
        cur_ylim = self.ax_full.get_ylim()
        y_center = (cur_ylim[0] + cur_ylim[1]) / 2.0
        height = cur_ylim[1] - cur_ylim[0]
        if height < 10.0:
            self.ax_full.set_ylim([y_center - 5.0, y_center + 5.0])
        
        self.canvas_full.draw()
        
        # Plot Telemetry
        self.ax_v.plot(time_list, v_ego_list, color='blue', linewidth=2)
        self.ax_v.set_title("Ego Velocity over Time", fontsize=24)
        self.ax_v.set_ylabel("Speed (m/s)", fontsize=20)
        self.ax_v.grid(True)
        
        self.ax_steer.plot(time_list, steer_list, color='orange', linewidth=2)
        self.ax_steer.set_title("Steering Angle over Time", fontsize=24)
        self.ax_steer.set_ylabel("Angle (deg)", fontsize=20)
        self.ax_steer.grid(True)
        
        self.ax_accel.plot(time_list, a_ego_list, color='purple', linewidth=2)
        self.ax_accel.set_title("Acceleration over Time", fontsize=24)
        self.ax_accel.set_xlabel("Time (s)", fontsize=20)
        self.ax_accel.set_ylabel("Accel (m/s^2)", fontsize=20)
        self.ax_accel.grid(True)
        
        self.ax_pedal.plot(time_list, gas_list, color='green', linewidth=2, label="Gas")
        self.ax_pedal.plot(time_list, brake_list, color='red', linewidth=2, label="Brake")
        self.ax_pedal.set_title("Gas and Brake Actuation", fontsize=24)
        self.ax_pedal.set_xlabel("Time (s)", fontsize=20)
        self.ax_pedal.set_ylabel("Value", fontsize=20)
        self.ax_pedal.grid(True)
        self.ax_pedal.legend(fontsize=16)
        
        self.fig_telem.subplots_adjust(hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.9, bottom=0.1)
        self.canvas_telem.draw()

    def export_video(self):
        if not self.segment_dirs: return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
            title="Save Export MP4"
        )
        if not save_path:
            return
            
        self.btn_export.config(state=tk.DISABLED)
        self.lbl_status.config(text="Exporting MP4... Please Wait...")
        self.root.update()
        
        # We will use an off-screen layout for drawing the overlay
        fig_export = plt.figure(figsize=(4, 4), dpi=100)
        ax_export = fig_export.add_subplot(111)
        
        out = None
        overlay_size = (300, 300)
        
        for g_idx in range(self.num_total_frames):
            img_np, v_ego, yaw_rate, steer = self.get_frame_data(g_idx)
            seg_dir, local_idx = self.map_global_to_local(g_idx)
            
            data = None
            if seg_dir:
                 try:
                     data = load_custom_dataset(segment_dir=seg_dir, frame_idx=local_idx, num_future_steps=self.num_frames)
                 except Exception:
                     pass
                     
            if out is None:
                h, w, _ = img_np.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(save_path, fourcc, 20.0, (w, h))

            ax_export.clear()
            if data is not None:
                future_xyz = data["ego_future_xyz"][0, 0].numpy()
                x_forward = np.concatenate(([0.0], future_xyz[:self.num_frames, 0]))
                y_left = np.concatenate(([0.0], future_xyz[:self.num_frames, 1]))
                
                ax_export.plot([-y for y in y_left], x_forward, marker='o', color='blue', linewidth=2)
                ax_export.plot(0, 0, marker='*', color='red', markersize=15)
                
            ax_export.set_aspect('equal') # Square aspect
            cur_xlim = ax_export.get_xlim()
            cur_ylim = ax_export.get_ylim()
            # dynamically resize bounds
            max_range = max(cur_xlim[1]-cur_xlim[0], cur_ylim[1]-cur_ylim[0], 10.0) / 2.0
            x_c = (cur_xlim[1]+cur_xlim[0]) / 2.0
            y_c = (cur_ylim[1]+cur_ylim[0]) / 2.0
            ax_export.set_xlim(x_c - max_range, x_c + max_range)
            ax_export.set_ylim(y_c - max_range, y_c + max_range)
            ax_export.axis('off') # Hide axes
            fig_export.canvas.draw()
            rgba = np.asarray(fig_export.canvas.buffer_rgba())
            overlay = rgba[:, :, :3].copy()
            
            overlay = cv2.resize(overlay, overlay_size)
            
            # Alpha blend white background
            gray = cv2.cvtColor(overlay, cv2.COLOR_RGB2GRAY)
            mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]
            
            h, w, _ = img_np.shape
            roi = img_np[h-overlay_size[1]:h, w-overlay_size[0]:w]
            
            bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
            fg = cv2.bitwise_and(overlay, overlay, mask=mask)
            img_np[h-overlay_size[1]:h, w-overlay_size[0]:w] = cv2.add(bg, fg)
            
            out.write(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            if g_idx % 20 == 0:
                self.lbl_status.config(text=f"Exporting MP4... {g_idx}/{self.num_total_frames} Frames")
                self.root.update()
                
        if out is not None:
             out.release()
             
        plt.close(fig_export)
        self.btn_export.config(state=tk.NORMAL)
        self.lbl_status.config(text=f"Export Completed: {os.path.basename(save_path)}")
        self.root.update()


def get_default_route():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets')
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if os.path.isdir(d)]
    if dirs:
        dirs.sort()
        return dirs[0]
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GUI to visualize dataset raw frames and kinematic ground truth")
    parser.add_argument("--route", type=str, default=get_default_route(), 
                        help="Path to a route directory containing segments")
    parser.add_argument("--frames", type=int, default=4, 
                        help="Number of future frames to graph (default: 4)")
    args = parser.parse_args()
    
    # Initialize Tkinter
    root = tk.Tk()
    app = GTVisualizerApp(root, route_dir=args.route, initial_frames=args.frames)
    
    # Properly handle window close to terminate process
    def on_closing():
        root.quit()
        root.destroy()
        sys.exit(0)
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run loop
    root.mainloop()
