
import os
import torch
import json
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- Configuration ---
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DATASET_FILE = "../dataset/llava-results.jsonl"
IMAGE_DIR = "../dataset/images/"
OUTPUT_DIR = "../dataset/lora_adapter"

# LoRA Config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"] 

# Training Config
BATCH_SIZE = 1 # Small batch size for VRAM constrained environments
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_STEPS = 100 # Demo steps, increase for real training
LOGGING_STEPS = 10

def load_data():
    """Loads JSONL data and prepares it for the dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, DATASET_FILE)
    image_dir_abs = os.path.join(base_dir, IMAGE_DIR)
    
    data = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    prompt_text = """You are analyzing a dashcam image.

Describe ONLY what is clearly visible in the image.

Return your answer EXACTLY in the following structured format.
Do not add extra sections.
Do not change the section names.
If a category is not present, write "None".

1. **Traffic Lights**:
   - State color (red/yellow/green) and position (left/right/center/background).
   - If none visible, write: None

2. **Traffic Signs**:
   - List each visible sign:
     - Type (stop sign, yield sign, speed limit, pedestrian crossing, no turn, etc.)
     - Relative position (right side of road, overhead, near intersection, sidewalk left, etc.)
   - If none visible, write: None

3. **Vehicles**:
   - Total vehicle count: <number>
   - List each vehicle as:
     - Type (car, truck, SUV, bus, motorcycle, etc.)
     - Relative position (front, left lane, right lane, oncoming, parked right side, behind, etc.)

4. **Pedestrians**:
   - Total pedestrian count: <number>
   - For each pedestrian:
     - Relative position (crosswalk, sidewalk right, near intersection, etc.)
   - If none visible, write: None

5. **Lane Markings**:
   - Describe visible lane lines (solid, dashed, double yellow, crosswalk, turn arrows, ONLY markings, etc.)
   - If none visible, write: None

6. **Hazards / Recommended Action**:
   - List any immediate hazards (stopped vehicle, pedestrian crossing, red light, obstruction, etc.)
   - Provide one short recommended driving action (e.g., "Maintain lane", "Prepare to stop", "Yield to pedestrian", "Slow down", "Proceed straight").
   - If no hazards, write: No immediate hazards. Maintain normal driving."""
                    
                    data.append({
                        "image_path": os.path.join(image_dir_abs, entry["image"]),
                        "conversations": [
                            {
                                "from": "human",
                                "value": f"<image>\n{prompt_text}"
                            },
                            {
                                "from": "gpt",
                                "value": entry["result"]
                            }
                        ]
                    })
                except Exception as e:
                    print(f"Skipping line due to error: {e}")
    return data

def main():
    print(f"Loading Base Model: {MODEL_ID}")
    
    # Quantization Config for 4-bit loading
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load Processor and Model
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load Dataset
    raw_data = load_data()
    print(f"Loaded {len(raw_data)} training examples.")
    
    def format_example(example):
        # We need to load image and tokenize text
        img = Image.open(example["image_path"]).convert("RGB")
        conversation = example["conversations"]
        
        # Create text for the model
        text = f"USER: <image>\n{conversation[0]['value'].replace('<image>', '').strip()}\nASSISTANT: {conversation[1]['value']}</s>"
        
        # Process
        inputs = processor(text=text, images=img, return_tensors="pt", padding="max_length", truncation=True, max_length=512)

        return inputs
    
    pass 

    dataset = Dataset.from_list(raw_data)
    
    def transform_data(batch):
        # Batch processing
        images = [Image.open(path).convert("RGB") for path in batch["image_path"]]
        texts = []
        for conv in batch["conversations"]:
            # Format: USER: ... ASSISTANT: ...
             texts.append(f"USER: <image>\n{conv[0]['value'].replace('<image>', '').strip()}\nASSISTANT: {conv[1]['value']}")
        
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    dataset.set_transform(transform_data)
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        save_strategy="no", # Demo
        report_to="none",
        remove_unused_columns=False
    )
    
    # Custom Data Collator to stack tensors
    def data_collator(features):
        import torch
        
        batch = {}
        for key in features[0].keys():
            if key == "pixel_values":
                 batch[key] = torch.stack([f[key].squeeze(0) for f in features])
            else:
                 batch[key] = torch.stack([f[key].squeeze(0) for f in features])
        return batch

    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Training finished. Saving adapter to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
