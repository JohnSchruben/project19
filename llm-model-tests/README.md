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
pip instal psutil GPUtil

# Run benchmark on all models
python benchmark.py

# Run automated performance tests (fails if more than 50ms latency)
python test_performance.py
```

Restults saved to 'benchmarks/' folder
