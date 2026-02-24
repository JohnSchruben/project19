# LLM Model Tests

## Setup
Install dependencies and Ollama:
```bash
python install_dependencies.py
```

## Running Tests
Run all models (automatically uses `driving_prompt.txt`):
```bash
python runall.py --image <path_to_image>
```

## Generating Dataset
Generate training data using a base model (LLaVA), using `driving_prompt.txt`:
```bash
python generate-dataset.py
```

## Training LoRA
Fine-tune a model using the generated dataset:
```bash
python train_lora.py
```

## Viewing Dataset
Verify the dataset and images:
```bash
python ../dataset/dataset_viewer.py
```

## Performance Benchmarking
Test real-time performance constraints (20 FPS requirement):
```bash

# Install benchmark dependencies
pip install psutil GPUtil

# Run benchmark on all models
python benchmark.py

# Run automated performance tests (fails if more than 50ms latency)
python test_performance.py
```

Results saved to 'benchmarks/' folder


## End-to-End Pipeline Testing
Test the complete video → LLM → navigation pipeline:
```bash

# Install OpenCV if not already installed
pip install opencv-python

# Run pipeline test (!!replace with your video file path!!)
python pipeline_integration_test.py --video "[video file path here]" --model moondream --max-frames 20


# Test full video
python pipeline_integration_test.py --video "[video file path here]" --model llava

# Adjust extraction rate (frames per second)
python pipeline_integration_test.py --video "[video file path here]" --model moondream --fps-extract 1.0
```

What it tests:
- Video frame extraction
- LLM inference on each frame
- Navigation output validation (steering -180° to 180°, throttle/brake 0.0 to 1.0)
- End-to-end pipeline performance

Results saved to `pipeline_results/pipeline_test_*.json`
