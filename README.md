# Group G Capstone Project19 - CS-4273-001 Spring 2026
### Team Members
- Dasanie Le (Sprint Master)
- Trevor Lietch (QA)
- Aleeza Malik (Project Owner)
- John Schruben (QA)
- Blake Shore (QA)
---
### Project Overview
Main Task: Our project is to train an existing LLM on dashcam image data so that the LLM can identify elements in dashcam footage and give driving instructions in response to the footage. This project is LLM Selfdrive for the OU Mobility Intelligence Lab (MiLa).
Side Task: We will program an automatic data preparation pipeline for autonomous driving. 

### Main Task Objectives
1. Collect dashcam image data using test vehicles at MiLa.
2. Annotate the image data in terms of scene interpretation using ChatGPT and manual correction. For each image, the LLM needs to explain the scene and how the vehicle should drive in order to avoid pedestrians, other vehicles, etc., and to follow traffic laws.
3. Train the lightweight LLMs with the collected data. Run the LLM in real-time with at least 20FPS in test vehicle. (i.e., each BEV space has to be generated within 50ms in a regular computer).
4. Test the trained LLM in vehicle closed course and on-road testing.

### Side Task Objectives
1. Data collection: Integrate the MiLa repo to automatically generate images and some features (features will be used in later E2E learning training) based on MiLa test vehicle dashcam data collected on Comma 3x device: https://github.com/OUMiLa/Openpilot_Custom/tree/main
2. Data automatic annotation: Integrate CVAT opensource code to achieve automatic image annotation for object detection (pedestrian, vehicle, traffic light, stop sign, road boundary, traffic signs, etc.), segmentation, depth estimation, etc. Modify CVAT codebase and customize it to fit our needs as CVAT is mainly for object detection.
3. Data label manual correction: Allow human experts to check the automatic annotated labels and manually correct them using the mouse if necessary.
4. Database (Labelled data storage): Build a database and integrated into your program to store the annotated data in step 3. In this database, it should be conveniently to save and extract certain data and labels.
5. Data export for training (annotated data filtering based on different training needs): Allow user to easily filter the data of interest based on labels.
---
### Technologies Used
- Python / pip - pipeline scripts, dataset import, notebooks, and utility tooling
- Bash - setup scripts and full pipeline orchestration
- MiLa Openpilot / Openpilot replay - route playback and dashcam data capture
- NVIDIA Alpamayo - trajectory prediction, navigation commands, and reasoning output
- Hugging Face - Alpamayo model access and downloads
- CUDA 12.4 + NVIDIA GPU - accelerated Alpamayo inference, with at least 40 GB VRAM recommended
- CVAT + Docker - automatic annotation review and manual label correction
- YOLO / Ultralytics - local object detection annotation export
- SQLite - `pipeline/annotations.db` storage for frames, annotations, predictions, commands, and reasoning
- Jupyter notebooks - querying and visualizing route data, annotations, and prediction paths

### Setup Instructions
1. Download and install the technologies listed above from their sources.
2. Clone the repo to your local machine.
3. See instuctions below for the pipeline.

### Tests

Run the lightweight tests from the repo root:

```bash
python3 run_tests.py
```

This covers pipeline helpers, database CRUD/export, prediction import, route capture frame counting, and Alpamayo navigation-command logic.

Check installed setup dependencies with:

```bash
python3 run_tests.py --dependencies
```

The default tests do not require Openpilot, CVAT, CUDA, or Alpamayo weights.
---
### Roadmap
Sprint 1
- Meet with mentor to receive technical info about how the MiLa self-driving cars work so we know how to program for them. Document info and update roadmap as necessary.
- Research best LLM for our project.
- Get the local LLM running on our computers.
- **Side**: Integrate the MiLa repo to automatically generate images based on MiLa test vehicle dashcam data collected.
- **Side**: Integrate CVAT opensource code to achieve automatic image annotation for object detection.

Sprint 2
- Build a program for training the LLM on detecting necessary elements in the dashcam images.
- Train the LLM to detect the elements. It should output a description of the scene.
- Test LLM's ability to detect elements correctly.
- **Side**: Build manual correction capability for annotated dashcam images.

Sprint 3
- Build a program for training the LLM on making correct driving instructions.
- Train the LLM to make driving instructions in response to dashcam images. It should output a trajectory of where to go and the accelerator, brake, and steering actions needed. (There is no specific format needed.)
- Test LLM's ability to give correct driving instructions.
- **Side**: Build a database and integrate it to store the annotated dashcam images. Build data export for training.

Sprint 4
- Build a program for running the trained LLM with live dashcam footage from the MiLa vehicles.
- Test the latency of the LLM output, and optimize code to reduce it.
- Test the LLM on the MiLa vehicles in closed course and on-road testing.

---
### MiLa Openpilot Data Generation

All generated route data is written into `datasets/` folders, then imported into `pipeline/annotations.db` for training.
We use MiLa/Openpilot routes to create dataset folders, run Alpamayo inference, annotate frames, and query the results.

#### Setup

Requirements: Hugging Face account/token registered for model downloads, CUDA 12.4, an NVIDIA GPU with at least 40 GB VRAM, Ubuntu/Linux for Openpilot, Git, Python/pip, Docker for CVAT, and enough disk space for routes, frames, model weights, and generated videos.

Run the two setup scripts from the repo root:

```bash
./setup_openpilot.sh
./setup_alpamayo_env.sh
```

#### Full Pipeline

Use `run_full_pipeline.sh` to generate a dataset and run the main processing steps:

```bash
./run_full_pipeline.sh \
  --openpilot-route "d34c14daa88a1e86/000000ca--7c5d326170" \
  datasets/route_1
```

#### Annotations

Edit annotations in CVAT. CVAT setup and manual annotation instructions are in `CVAT_setup/`, including `CVAT_Manual_Annotations_Guide.pdf`.

#### Querying Data

Use `pipeline/database_explorer.ipynb` to inspect images, annotations, commands, reasoning, and prediction paths.

#### Individual Steps

Each step can also be run with the scripts in `pipeline/`. Most scripts accept either a whole route folder like `datasets/route_1` or one segment like `datasets/route_1/segment_00`.

Capture Openpilot route data with `pipeline/route_caputure.py`:

```bash
./pipeline/route_caputure.py \
  --route "d34c14daa88a1e86/000000ca--7c5d326170" \
  --dataset-dir datasets/route_1
```

By default this moves on after capture starts and no new raw frames appear for 45 seconds. Use `--auto-stop-idle-seconds 0` to wait forever, or `--new-terminal-modeld` to debug modeld in a separate terminal.

Run local YOLO annotation export with `pipeline/annotate_route.py`:

```bash
./pipeline/annotate_route.py datasets/route_1
./pipeline/annotate_route.py datasets/route_1/segment_00
```

Run Alpamayo prediction JSON export with `pipeline/run_alpamayo.py`:

```bash
./pipeline/run_alpamayo.py datasets/route_1
./pipeline/run_alpamayo.py datasets/route_1/segment_00
```

Import annotations and predictions into SQLite with `pipeline/import_route_db.py`:

```bash
./pipeline/import_route_db.py datasets/route_1 --overwrite
./pipeline/import_route_db.py datasets/route_1/segment_00 --overwrite
```

Create the prediction video with `pipeline/create_alpamayo_video.py`:

```bash
./pipeline/create_alpamayo_video.py datasets/route_1
./pipeline/create_alpamayo_video.py datasets/route_1/segment_00
```

---
### Tooling

We added project-specific tooling for MiLa route datasets:

- `alpamayo/src/alpamayo1_5/load_custom_dataset.py` loads our `datasets/route_*/segment_*` folders into Alpamayo's expected camera, telemetry, history, and future trajectory format.
- `alpamayo/batch_export_inference.py` runs batch Alpamayo inference and writes per-frame prediction JSON with command, reasoning, ground truth path, and selected prediction path.
- `alpamayo/notebooks/inference_nav_custom.ipynb` is the custom navigation notebook for testing route frames, navigation commands, prediction selection modes, and reasoning output.
- `frame_extractor/extract_3cam_route.py` creates `raw_left`, `raw_front`, and `raw_right` camera folders from `cam0`, `cam1`, and `cam2` videos in `frame_extractor/videos/`.

If you are running the pipeline for this route d34c14daa88a1e86/00000019--ab71b8e01d, you can add the additional camera frames by adding the .mp4 files to the `frame_extractor/videos/` folder. Then run the frame extractor script. 
```bash
python3 frame_extractor/extract_3cam_route.py \
  --route route_3
```

