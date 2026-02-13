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
