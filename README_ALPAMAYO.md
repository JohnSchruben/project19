# Running Alpamayo on a Custom Dataset

This guide walks you through the complete pipeline for setting up your environment, downloading a custom route using OpenPilot, and running Alpamayo inference on the extracted frames to generate a final video.

## 1. Build OpenPilot

First, ensure your OpenPilot environment is initialized and built correctly.

```bash
./setup_openpilot.sh
```

## 2. Create Dataset from Route ID

Extract your specified route and establish the dataset directory structure.

```bash
python3 run_pipeline.py --route "d34c14daa88a1e86/00000019--ab71b8e01d" --dataset-dir "./datasets/route_3"
```

## 3. Install HuggingFace and Setup Alpamayo Environment

Navigate to the `alpamayo` directory, configure your HuggingFace authentication to access the model weights, and set up the localized Python environment using `uv`.

```bash
cd alpamayo
pip install huggingface_hub
hf auth login --token [YOUR_TOKEN]

uv venv a1_5_venv
source a1_5_venv/bin/activate
uv sync --active
```
*(**Note**: Replace `[YOUR_TOKEN]` with your personal HuggingFace access token.)*

## 4. Run Inference on the Dataset

Generate inference results over the dataset segments you downloaded.

```bash
python batch_export_inference.py --route ../datasets/route_3
```

## 5. Collect Segments and Output Single Video

Stitch all the individual segment video exports into a single continuous file using `ffmpeg`.

```bash
ls segment_*_inference.mp4 | sed 's/^/file /' > filelist.txt
ffmpeg -f concat -safe 0 -i filelist.txt -c copy full_inference.mp4
```
