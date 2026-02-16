import time
import json
import os
import sys
import base64
import io
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from PIL import Image
import ollama
import ollama_utils

# Check for optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. CPU/memory monitoring disabled.")
    print("Install with: pip install psutil")

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False
    print("Warning: GPUtil not installed. GPU monitoring disabled.")
    print("Install with: pip install gputil")

# Import model list
try:
    from models import OLLAMA_VISION_MODELS
    MODELS = OLLAMA_VISION_MODELS
except ImportError:
    print("Warning: Could not import models.py. Using fallback.")
    MODELS = ["llava"]

# Configuration
MAX_DIM = 512
JPEG_QUALITY = 70
NUM_ITERATIONS = 100  # Number of frames to benchmark per model
OUTPUT_DIR = Path("benchmarks")
PROMPT_FILE = "driving_prompt.txt"

def load_prompt() -> str:
    """Load prompt from driving_prompt.txt"""
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return "Describe the traffic situation and provide driving instructions."

def measure_preprocessing(image_path: str) -> Tuple[bytes, float]:
    """
    Measure image loading and preprocessing time.
    Returns: (processed_image_bytes, preprocessing_time_ms)
    """
    start = time.perf_counter()

    try:
        # Load and resize image (same logic as ollama-generic.py)
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
        
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        img_bytes = buf.getvalue()
        
        end = time.perf_counter()
        preprocessing_ms = (end - start) * 1000
        
        return img_bytes, preprocessing_ms
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, 0.0
    
def measure_inference(model: str, img_bytes: bytes, prompt: str) -> Tuple[str, float]:
    """
    Measure pure LLM inference time.
    Returns: (response_text, inference_time_ms)
    """
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    client = ollama.Client()
    
    start = time.perf_counter()

    try:
        resp = client.generate(
            model=model,
            prompt=prompt,
            images=[img_b64],
            format='json',
            options={
                "temperature": 0.2,
                "num_predict": 1024,
                "num_ctx": 2048,
            },
            keep_alive="10m"
        )
        
        end = time.perf_counter()
        inference_ms = (end - start) * 1000
        
        response_text = resp.get("response", "")
        return response_text, inference_ms
    except Exception as e:
        print(f"Error in inference: {e}")
        return "", 0.0
    
def measure_postprocessing(response_text: str) -> Tuple[Dict, float]:
    """
    Measure JSON parsing and validation time.
    Returns: (parsed_data, postprocessing_time_ms)
    """
    start = time.perf_counter()

    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Parse JSON
        data = json.loads(cleaned)

        # Validate required fields
        required_fields = ["scene_description", "steering_angle_deg", "throttle", "brake"]
        for field in required_fields:
            if field not in data:
                data[field] = None
        
        end = time.perf_counter()
        postprocessing_ms = (end - start) * 1000
        
        return data, postprocessing_ms
    except json.JSONDecodeError:
        end = time.perf_counter()
        postprocessing_ms = (end - start) * 1000
        return {"error": "JSON parse failed"}, postprocessing_ms
    except Exception as e:
        end = time.perf_counter()
        postprocessing_ms = (end - start) * 1000
        return {"error": str(e)}, postprocessing_ms

def get_system_resources() -> Dict[str, float]:
    """
    Capture current system resource usage.
    Returns: dict with cpu_percent, memory_mb, gpu_percent
    """
    resources = {
        "cpu_percent": 0.0,
        "memory_mb": 0.0,
        "gpu_percent": 0.0,
        "gpu_memory_mb": 0.0
    }

    if HAS_PSUTIL:
        try:
            resources["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            resources["memory_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            pass
    
    if HAS_GPUTIL:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                resources["gpu_percent"] = gpu.load * 100
                resources["gpu_memory_mb"] = gpu.memoryUsed
        except Exception:
            pass
    
    return resources

def calculate_statistics(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate latency statistics.
    Returns: dict with min, max, mean, p95, p99, fps
    """
    if not latencies:
        return {
            "min_ms": 0.0,
            "max_ms": 0.0,
            "mean_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "fps": 0.0
        }
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)
    
    mean_ms = statistics.mean(latencies)
    
    return {
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "mean_ms": mean_ms,
        "p95_ms": sorted_latencies[p95_idx] if p95_idx < n else sorted_latencies[-1],
        "p99_ms": sorted_latencies[p99_idx] if p99_idx < n else sorted_latencies[-1],
        "fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0
    }

# Main benchmark function
def run_benchmark(model: str, image_path: str, num_iterations: int = NUM_ITERATIONS) -> Dict:
    """
    Run complete benchmark for a single model.
    Returns: aggregated results with statistics
    """
    print(f"\nBenchmarking {model}...")
    print(f"  Image: {image_path}")
    print(f"  Iterations: {num_iterations}")
    
    prompt = load_prompt()
    
    # Storage for all measurements
    preprocessing_times = []
    inference_times = []
    postprocessing_times = []
    total_times = []
    
    success_count = 0
    error_count = 0
    
    for i in range(num_iterations):
        # Phase 1: Preprocessing
        img_bytes, preproc_ms = measure_preprocessing(image_path)
        if img_bytes is None:
            error_count += 1
            continue
        
        # Phase 2: Inference
        response_text, inference_ms = measure_inference(model, img_bytes, prompt)
        if not response_text:
            error_count += 1
            continue
        
        # Phase 3: Postprocessing
        parsed_data, postproc_ms = measure_postprocessing(response_text)
        if "error" in parsed_data:
            error_count += 1
            # Still count the time even if parsing failed
        
        # Calculate total
        total_ms = preproc_ms + inference_ms + postproc_ms
        
        # Store measurements
        preprocessing_times.append(preproc_ms)
        inference_times.append(inference_ms)
        postprocessing_times.append(postproc_ms)
        total_times.append(total_ms)
        
        success_count += 1
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{num_iterations} iterations")
    
    # Calculate statistics
    preproc_stats = calculate_statistics(preprocessing_times)
    inference_stats = calculate_statistics(inference_times)
    postproc_stats = calculate_statistics(postprocessing_times)
    total_stats = calculate_statistics(total_times)
    
    # Get resource usage
    resources = get_system_resources()
    
    # Compile results
    results = {
        "model": model,
        "image": os.path.basename(image_path),
        "iterations": num_iterations,
        "success_count": success_count,
        "error_count": error_count,
        "timestamp": datetime.now().isoformat(),
        
        # Preprocessing metrics
        "preprocessing_mean_ms": preproc_stats["mean_ms"],
        "preprocessing_p95_ms": preproc_stats["p95_ms"],
        
        # Inference metrics
        "inference_min_ms": inference_stats["min_ms"],
        "inference_max_ms": inference_stats["max_ms"],
        "inference_mean_ms": inference_stats["mean_ms"],
        "inference_p95_ms": inference_stats["p95_ms"],
        "inference_p99_ms": inference_stats["p99_ms"],
        
        # Postprocessing metrics
        "postprocessing_mean_ms": postproc_stats["mean_ms"],
        "postprocessing_p95_ms": postproc_stats["p95_ms"],
        
        # Total metrics
        "total_min_ms": total_stats["min_ms"],
        "total_max_ms": total_stats["max_ms"],
        "total_mean_ms": total_stats["mean_ms"],
        "total_p95_ms": total_stats["p95_ms"],
        "total_p99_ms": total_stats["p99_ms"],
        "fps": total_stats["fps"],
        
        # Resource usage
        "cpu_percent": resources["cpu_percent"],
        "memory_mb": resources["memory_mb"],
        "gpu_percent": resources["gpu_percent"],
        "gpu_memory_mb": resources["gpu_memory_mb"],
        
        # Performance check
        "meets_20fps_requirement": total_stats["fps"] >= 20.0,
        "meets_50ms_requirement": total_stats["mean_ms"] <= 50.0
    }
    
    # Print summary
    print(f"\n Results for {model}:")
    print(f"Total Mean Latency: {total_stats['mean_ms']:.2f}ms")
    print(f"Total P95 Latency: {total_stats['p95_ms']:.2f}ms")
    print(f"FPS: {total_stats['fps']:.2f}")
    print(f"Breakdown:")
    print(f"Preprocessing: {preproc_stats['mean_ms']:.2f}ms")
    print(f"Inference: {inference_stats['mean_ms']:.2f}ms")
    print(f"Postprocessing: {postproc_stats['mean_ms']:.2f}ms")
    
    if total_stats["fps"] >= 20.0:
        print(f"✓ PASS: Meets 20 FPS requirement")
    else:
        print(f"✗ FAIL: Does not meet 20 FPS requirement ({total_stats['fps']:.2f} FPS)")
    
    return results

def export_results(all_results: List[Dict]):
    """
    Export results to CSV and JSON files.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export JSON
    json_path = OUTPUT_DIR / f"benchmark_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")
    
    # Export CSV
    csv_path = OUTPUT_DIR / f"benchmark_results_{timestamp}.csv"
    if all_results:
        with open(csv_path, 'w') as f:
            # Header
            keys = all_results[0].keys()
            f.write(",".join(keys) + "\n")
            
            # Data rows
            for result in all_results:
                values = [str(result[k]) for k in keys]
                f.write(",".join(values) + "\n")
        print(f"CSV results saved to: {csv_path}")

def print_summary_table(all_results: List[Dict]):
    """
    Print a summary table of all benchmarks.
    """
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Model':<20} | {'Mean (ms)':<10} | {'P95 (ms)':<10} | {'FPS':<10} | {'Status':<10}")
    print("-" * 80)
    
    for r in sorted(all_results, key=lambda x: x["total_mean_ms"]):
        status = "✓ PASS" if r["meets_20fps_requirement"] else "✗ FAIL"
        print(f"{r['model']:<20} | {r['total_mean_ms']:<10.2f} | {r['total_p95_ms']:<10.2f} | {r['fps']:<10.2f} | {status:<10}")
    
    print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark LLM performance for real-time constraints")
    parser.add_argument("--model", type=str, help="Benchmark specific model (default: all models)")
    parser.add_argument("--image", type=str, default="car-on-road-3.jpg", help="Image path")
    parser.add_argument("--iterations", type=int, default=NUM_ITERATIONS, help=f"Number of iterations (default: {NUM_ITERATIONS})")
    args = parser.parse_args()
    
    # Determine which models to test
    models_to_test = [args.model] if args.model else MODELS
    
    # Verify image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    print("="*80)
    print("REAL-TIME PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"Target: 20 FPS (≤50ms mean latency)")
    print(f"Models: {', '.join(models_to_test)}")
    print(f"Iterations per model: {args.iterations}")
    print(f"Image: {args.image}")
    print("="*80)
    
    all_results = []
    
    with ollama_utils.OllamaService():
        for model in models_to_test:
            try:
                result = run_benchmark(model, args.image, args.iterations)
                all_results.append(result)
            except Exception as e:
                print(f"Error benchmarking {model}: {e}")
    
    if all_results:
        print_summary_table(all_results)
        export_results(all_results)
    else:
        print("\nNo results to export.")

if __name__ == "__main__":
    main()