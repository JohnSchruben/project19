# Centralized list of models to ensure consistency across scripts
# Only include models that support vision (multimodal)

OLLAMA_VISION_MODELS = [
    "mistral",     # Text-only (will ignore image in generic script)
    "tinyllama",   # Text-only (will ignore image in generic script)
    "phi3.5",      # Text-only (will ignore image in generic script)
    "llama3.2-vision", # Vision capable
    "minicpm-v",       # Vision capable
    "llava",           # Vision capable
    "moondream",       # Vision capable
]
