import json
import time
import sys
import argparse
import subprocess
import re
from typing import List, Dict, Tuple

try:
    from models import OLLAMA_VISION_MODELS
    MODELS = OLLAMA_VISION_MODELS
except ImportError:
    print("Error: Could not import 'models.py'. Using fallback list.")
    MODELS = ["llava", "minicpm-v"]

def load_dataset(path: str) -> List[Dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading dataset '{path}': {e}")
        sys.exit(1)

def run_model(model: str, image: str, prompt: str) -> str:
    """
    Runs the ollama-generic.py script and returns the output.
    """
    cmd = [sys.executable, "ollama-generic.py", "--model", model, "--image", image, "--prompt", prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"Error running {model}: {result.stderr}"
            print(msg)
            with open("debug_log.txt", "a") as log:
                log.write(msg + "\n")
            return ""
        return result.stdout
    except Exception as e:
        msg = f"Exception running {model}: {e}"
        print(msg)
        with open("debug_log.txt", "a") as log:
            log.write(msg + "\n")
        return ""

def calculate_accuracy(output: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """
    Calculates accuracy based on the presence of keywords in the output.
    Returns the accuracy percentage and a list of matched keywords.
    """
    if not output:
        return 0.0, []
    
    output_lower = output.lower()
    matched = [kw for kw in keywords if kw.lower() in output_lower]
    
    if not keywords:
        return 0.0, []
        
    return (len(matched) / len(keywords)) * 100, matched


def parse_model_response(full_output: str) -> str:
    """
    Extracts the actual model response from ollama-generic.py output.
    It looks for the line '--- response in X.XXs ---' and takes everything after it.
    """
    # Regex to find the timing line
    match = re.search(r"--- response in \d+\.\d+s ---\s*", full_output)
    if match:
        # Return everything after the match
        return full_output[match.end():].strip()
    return full_output.strip()

def main():
    with open("debug_log.txt", "w") as log:
        log.write(f"Python Executable: {sys.executable}\n")
    
    parser = argparse.ArgumentParser(description="Run accuracy tests for LLM models")
    parser.add_argument("--dataset", type=str, default="accuracy_dataset.json", help="Path to the test dataset JSON")
    parser.add_argument("--model", type=str, help="Run only a specific model (optional)")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    
    # Filter models if argument is provided
    models_to_test = [args.model] if args.model else MODELS
    
    results = []

    print(f"--- Starting Accuracy Benchmark of {len(models_to_test)} Models ---\n")

    with open("debug_log.txt", "a") as log:
        log.write(f"Models: {models_to_test}\n")
        log.write(f"Dataset len: {len(dataset)}\n")

    for model in models_to_test:
        msg = f"Testing Model: {model}"
        print(msg, flush=True)
        with open("debug_log.txt", "a") as log:
            log.write(msg + "\n")

        total_accuracy = 0
        total_time = 0
        
        for case in dataset:
            image = case.get("image")
            prompt = case.get("prompt")
            keywords = case.get("expected_keywords", [])
            
            msg = f"  Image: {image}"
            print(msg)
            with open("debug_log.txt", "a") as log:
                log.write(msg + "\n")
            
            start_time = time.time()
            output = run_model(model, image, prompt)
            duration = time.time() - start_time
            
            # Extract the actual response part
            model_response = parse_model_response(output)
            
            accuracy, matched = calculate_accuracy(output, keywords) # Start search on full output just in case
            
            msg = f"    Time: {duration:.2f}s"
            print(msg)
            with open("debug_log.txt", "a") as log:
                log.write(msg + "\n")

            print("    Output:")
            # Indent the response for better readability
            indented_response = "\n".join(["      " + line for line in model_response.splitlines()])
            print(indented_response)
            
            with open("debug_log.txt", "a") as log: # encoding='utf-8' might be needed
                log.write("    Output:\n" + indented_response + "\n")

            msg = f"    Keywords Matched: {len(matched)}/{len(keywords)} ({accuracy:.1f}%) {matched}"
            print(msg)
            with open("debug_log.txt", "a") as log:
                log.write(msg + "\n")
            
            total_accuracy += accuracy
            total_time += duration
            
        avg_accuracy = total_accuracy / len(dataset) if dataset else 0
        avg_time = total_time / len(dataset) if dataset else 0
        
        results.append({
            "model": model,
            "accuracy": avg_accuracy,
            "time": avg_time
        })
        print("-" * 40 + "\n")
        with open("debug_log.txt", "a") as log:
            log.write("-" * 40 + "\n\n")

    # Sort results by accuracy (highest first)
    results.sort(key=lambda x: x["accuracy"], reverse=True)

    print("\n" + "="*50)
    print("ACCURACY RANKING (Highest to Lowest)")
    print("="*50)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Avg Time':<10}")
    print("-" * 46)
    
    for r in results:
        print(f"{r['model']:<20} | {r['accuracy']:<6.1f}%    | {r['time']:.2f}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
