#!/usr/bin/env python3
import argparse
import os
import time
import zipfile
from pathlib import Path
from cvat_sdk import Client
from cvat_sdk.api_client import models

def main():
    parser = argparse.ArgumentParser(description="Uploads a folder to CVAT, runs AI annotation, and downloads the results.")
    parser.add_argument("folder", type=str, help="Path to the folder containing image files to annotate")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="CVAT URL")
    parser.add_argument("--user", type=str, default="admin", help="CVAT admin username")
    parser.add_argument("--password", type=str, default="admin123", help="CVAT admin password")
    parser.add_argument("--model", type=str, default="ultralytics-yolov8", help="YOLO function name (e.g. ultralytics-yolov8 or onnx-wongkinyiu-yolov7)")
    parser.add_argument("--output", "-o", type=str, default="annotations.zip", help="Output zip filename")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.folder)
    if not os.path.isdir(data_dir):
        print(f"Error: Directory '{data_dir}' not found.")
        return

    # Find images
    valid_exts = ('.png', '.jpg', '.jpeg')
    image_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) 
        if f.lower().endswith(valid_exts)
    ]
    if not image_files:
        print(f"Error: No images found {valid_exts} in '{data_dir}'")
        return

    print(f"[*] Found {len(image_files)} images in {args.folder}")
    print(f"[*] Connecting to CVAT at {args.url}...")

    with Client(args.url) as client:
        client.login((args.user, args.password))

        # 1. Fetch the Model ID from the Serverless function list
        functions = client.functions.list()
        model_id = None
        for func in functions:
             if func.name == args.model:
                 model_id = func.id
                 break
        
        if not model_id:
             raise ValueError(f"Model '{args.model}' not found in CVAT serverless functions! Is Nuclio running?")

        # 2. Define standard labels
        labels = [
            models.PatchedLabelRequest(name="Car"),
            models.PatchedLabelRequest(name="Truck"),
            models.PatchedLabelRequest(name="Motorcycle"),
            models.PatchedLabelRequest(name="Stop sign"),
        ]

        # 3. Create Project
        print("[*] Creating Project...")
        proj_name = f"AutoAnnotate_{Path(data_dir).name}_{int(time.time())}"
        project_spec = models.ProjectWriteRequest(name=proj_name, labels=labels)
        project = client.projects.create(project_spec)

        # 4. Create Task
        print("[*] Creating Task...")
        task_spec = models.TaskWriteRequest(name=f"Task_01", project_id=project.id)
        task = client.tasks.create(task_spec)

        # 5. Upload files
        print(f"[*] Uploading {len(image_files)} images (this might take a moment)...")
        task.upload_data(resources=image_files, image_quality=100)
        print("[+] Upload complete.")

        # 6. Trigger AI Annotation
        print(f"[*] Triggering {args.model} auto-annotation...")
        import_job = task.annotations.auto_annotate(
            function_id=model_id,
            mapping={} # Auto maps identical YOLO/CVAT label names
        )
        
        print("[*] Waiting for inference to finish...")
        while not import_job.is_completed:
            time.sleep(2)
            import_job.fetch()
            
        print("[+] Inference finished!")

        # 7. Export Dataset
        print(f"[*] Downloading annotations to '{args.output}'...")
        task.export_dataset(
            format_name="CVAT for images 1.1", # Options: "YOLO 1.1", "COCO 1.0", etc.
            filename=args.output,
            include_images=False
        )
        print("\n===============================================")
        print(f"[SUCCESS] Annotations saved to {os.path.abspath(args.output)}")
        print("===============================================\n")

if __name__ == "__main__":
    main()
