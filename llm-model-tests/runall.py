import subprocess
import argparse
import sys
import re
import time

# List of models to benchmark
# These are the models you are currently using or testing
try:
    from models import OLLAMA_VISION_MODELS
    MODELS = OLLAMA_VISION_MODELS
except ImportError:
    print("Error: Could not import 'models.py'. Using fallback list.")
    MODELS = ["llava", "minicpm-v"]

def parse_time_from_output(output):
    """
    Looks for the string '--- response in X.XXs ---' and returns the float X.XX
    """
    match = re.search(r"--- response in (\d+\.\d+)s ---", output)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Run all LLM model tests using ollama-generic.py")
    parser.add_argument("--image", type=str, default="car-on-road.png", help="Path to the image file to use for all tests")
    parser.add_argument("--prompt", type=str, help="Prompt text to use for all tests")
    args = parser.parse_args()

    results = []

    print(f"--- Starting Benchmark of {len(MODELS)} Models ---\n")

    for model in MODELS:
        print(f"Directory: Running {model}...")
        
        cmd = [sys.executable, "ollama-generic.py", "--model", model]
        
        if args.image:
            cmd.extend(["--image", args.image])
        
        if args.prompt:
            cmd.extend(["--prompt", args.prompt])
        
        # We manually time the subprocess as a fallback, but prefer the internal metric
        start_time = time.time()
        
        try:
            # Capture output to parse time
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Print the output to the console so the user sees progress/results
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            
            if result.returncode != 0:
                print(f"Error running {model}")
                results.append((model, float('inf')))
                continue

            # Try to parse the specific response time from the script output
            duration = parse_time_from_output(result.stdout)
            
            # Fallback to wall clock if parsing failed
            if duration is None:
                duration = time.time() - start_time
                
            results.append((model, duration))
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to run {model}: {e}")
            results.append((model, float('inf')))
        except Exception as e:
            print(f"An unexpected error occurred while running {model}: {e}")
            results.append((model, float('inf')))
            
        print("-" * 40 + "\n")

    # Sort results by duration (fastest first)
    # Inf durations (errors) will be at the end
    results.sort(key=lambda x: x[1])

    print("\n" + "="*40)
    print("SPEED RANKING (Fastest to Slowest)")
    print("="*40)
    print(f"{'Model':<20} | {'Time (s)':<10}")
    print("-" * 33)
    
    for model, duration in results:
        time_str = f"{duration:.2f}s" if duration != float('inf') else "FAILED"
        print(f"{model:<20} | {time_str:<10}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
