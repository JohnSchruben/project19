import sys
import subprocess
import ollama_utils
import time

# Try to import models, fallback if missing
try:
    from models import OLLAMA_VISION_MODELS
    MODELS = OLLAMA_VISION_MODELS
except ImportError:
    print("Error: Could not import 'models.py'. Using fallback list.")
    MODELS = ["llava", "minicpm-v"]

def main():
    print(f"--- Starting Dataset Generation for {len(MODELS)} Models ---")
    
    # We use the context manager here to ensure Ollama is running for the entire duration
    # This prevents starting/stopping it for each individual script call if possible,
    # though generate-dataset.py also uses the context manager (which handles nested calls gracefully).
    with ollama_utils.OllamaService():
        for model in MODELS:
            print(f"\n[Generate All] Processing model: {model}")
            
            cmd = [sys.executable, "generate-dataset.py", "--model", model]
            
            try:
                subprocess.check_call(cmd)
                print(f"[Generate All] Successfully generated dataset for {model}")
            except subprocess.CalledProcessError as e:
                print(f"[Generate All] Failed to generate dataset for {model}: {e}")
            except Exception as e:
                print(f"[Generate All] Unexpected error for {model}: {e}")
            
            print("-" * 40)
            
    print("\n--- All generations completed ---")

if __name__ == "__main__":
    main()
