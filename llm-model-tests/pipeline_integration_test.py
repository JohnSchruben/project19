#!/usr/bin/env python3
"""
End-to-End Pipeline Integration Test

Tests the complete pipeline: Video → Frame Extraction → LLM Inference → Navigation Outputs
Validates that all components (frame extraction, LLM harness, output validation) work together.

Usage:
    python pipeline_integration_test.py --video test_video.MOV --model llava
    python pipeline_integration_test.py --video test_video.MOV --model moondream --max-frames 50
"""

import cv2
import os
import sys
import time
import json
import base64
import io
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from PIL import Image
import ollama
import ollama_utils

# Import validation function from existing test
try:
    from Driving_Instructions_Angle_Test import validate_turn_angle
except ImportError:
    # Fallback validation if import fails
    def validate_turn_angle(angle):
        return isinstance(angle, (float, int)) and -180 <= angle <= 180

# Configuration
MAX_DIM = 512
JPEG_QUALITY = 70
PROMPT_FILE = "driving_prompt.txt"
OUTPUT_DIR = Path("pipeline_results")

def load_prompt() -> str:
    """Load driving prompt from file"""
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error loading prompt: {e}")
        return "Analyze this dashcam image and provide driving instructions in JSON format."

def extract_frames_from_video(video_path: str, max_frames: int = None, fps_extract: float = 2.0) -> List[Tuple[int, any]]:
    """
    Extract frames from video file.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to extract (None = all)
        fps_extract: Frames per second to extract
    
    Returns:
        List of (frame_number, frame_data) tuples
    """
    print(f"Extracting frames from: {video_path}")
    print(f"Target extraction rate: {fps_extract} FPS")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {fps_video:.2f} FPS, {total_frames} total frames")
    
    # Calculate frame interval
    frame_interval = int(fps_video / fps_extract) if fps_extract < fps_video else 1
    
    frames = []
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at interval
        if frame_count % frame_interval == 0:
            frames.append((frame_count, frame))
            extracted_count += 1
            
            # Check max frames limit
            if max_frames and extracted_count >= max_frames:
                break
        
        frame_count += 1
    
    cap.release()
    
    print(f"Extracted {len(frames)} frames (every {frame_interval} frames)")
    return frames

def preprocess_frame(frame) -> bytes:
    """
    Convert OpenCV frame to JPEG bytes for LLM.
    
    Args:
        frame: OpenCV frame (BGR format)
    
    Returns:
        JPEG bytes
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    img = Image.fromarray(frame_rgb)
    
    # Resize to reduce latency
    img.thumbnail((MAX_DIM, MAX_DIM), Image.Resampling.LANCZOS)
    
    # Convert to JPEG bytes
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()

def run_llm_inference(model: str, img_bytes: bytes, prompt: str) -> Tuple[str, float]:
    """
    Run LLM inference on frame.
    
    Args:
        model: Model name
        img_bytes: Image as JPEG bytes
        prompt: Prompt text
    
    Returns:
        (response_text, inference_time_ms)
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
        
        return resp.get("response", ""), inference_ms
    except Exception as e:
        print(f"Error in inference: {e}")
        return "", 0.0

def parse_and_validate_output(response_text: str) -> Tuple[Dict, bool, List[str]]:
    """
    Parse JSON response and validate navigation outputs.
    
    Args:
        response_text: LLM response
    
    Returns:
        (parsed_data, is_valid, validation_errors)
    """
    errors = []
    
    # Clean response
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"error": "JSON parse failed"}, False, [f"JSON decode error: {e}"]
    
    # Validate required fields
    required_fields = ["scene_description", "steering_angle_deg", "throttle", "brake"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return data, False, errors
    
    # Validate steering angle
    if not validate_turn_angle(data["steering_angle_deg"]):
        errors.append(f"Invalid steering_angle_deg: {data['steering_angle_deg']} (must be -180 to 180)")
    
    # Validate throttle (0.0 to 1.0)
    throttle = data["throttle"]
    if not isinstance(throttle, (int, float)) or not (0.0 <= throttle <= 1.0):
        errors.append(f"Invalid throttle: {throttle} (must be 0.0 to 1.0)")
    
    # Validate brake (0.0 to 1.0)
    brake = data["brake"]
    if not isinstance(brake, (int, float)) or not (0.0 <= brake <= 1.0):
        errors.append(f"Invalid brake: {brake} (must be 0.0 to 1.0)")
    
    is_valid = len(errors) == 0
    return data, is_valid, errors

def run_pipeline_test(video_path: str, model: str, max_frames: int = None, fps_extract: float = 2.0):
    """
    Run complete end-to-end pipeline test.
    
    Args:
        video_path: Path to video file
        model: LLM model name
        max_frames: Maximum frames to process
        fps_extract: Frames per second to extract
    """
    print("="*80)
    print("END-TO-END PIPELINE INTEGRATION TEST")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Model: {model}")
    print(f"Max frames: {max_frames if max_frames else 'All'}")
    print("="*80)
    
    # Load prompt
    prompt = load_prompt()
    
    # Step 1: Extract frames
    print("\n[STEP 1] Extracting frames from video...")
    try:
        frames = extract_frames_from_video(video_path, max_frames, fps_extract)
    except Exception as e:
        print(f"ERROR: Frame extraction failed: {e}")
        return
    
    if not frames:
        print("ERROR: No frames extracted")
        return
    
    # Step 2: Process each frame through pipeline
    print(f"\n[STEP 2] Processing {len(frames)} frames through LLM pipeline...")
    
    results = []
    valid_count = 0
    invalid_count = 0
    
    total_preprocess_time = 0
    total_inference_time = 0
    total_postprocess_time = 0
    
    for idx, (frame_num, frame) in enumerate(frames):
        print(f"\nProcessing frame {idx + 1}/{len(frames)} (video frame #{frame_num})...")
        
        # Preprocess
        start = time.perf_counter()
        img_bytes = preprocess_frame(frame)
        preprocess_ms = (time.perf_counter() - start) * 1000
        total_preprocess_time += preprocess_ms
        
        # Inference
        response_text, inference_ms = run_llm_inference(model, img_bytes, prompt)
        total_inference_time += inference_ms
        
        if not response_text:
            invalid_count += 1
            results.append({
                "frame_index": idx,
                "video_frame_number": frame_num,
                "valid": False,
                "error": "No response from LLM"
            })
            continue
        
        # Parse and validate
        start = time.perf_counter()
        parsed_data, is_valid, validation_errors = parse_and_validate_output(response_text)
        postprocess_ms = (time.perf_counter() - start) * 1000
        total_postprocess_time += postprocess_ms
        
        # Store result
        result = {
            "frame_index": idx,
            "video_frame_number": frame_num,
            "valid": is_valid,
            "scene_description": parsed_data.get("scene_description", ""),
            "steering_angle_deg": parsed_data.get("steering_angle_deg"),
            "throttle": parsed_data.get("throttle"),
            "brake": parsed_data.get("brake"),
            "preprocessing_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocessing_ms": postprocess_ms,
            "total_ms": preprocess_ms + inference_ms + postprocess_ms
        }
        
        if not is_valid:
            result["validation_errors"] = validation_errors
            invalid_count += 1
            print(f"  ✗ INVALID - Errors: {validation_errors}")
        else:
            valid_count += 1
            print(f"  ✓ VALID - Steering: {result['steering_angle_deg']:.1f}°, Throttle: {result['throttle']:.2f}, Brake: {result['brake']:.2f}")
        
        results.append(result)
    
    # Step 3: Calculate statistics
    print("\n[STEP 3] Calculating pipeline statistics...")
    
    total_frames = len(frames)
    avg_preprocess = total_preprocess_time / total_frames
    avg_inference = total_inference_time / total_frames
    avg_postprocess = total_postprocess_time / total_frames
    avg_total = (total_preprocess_time + total_inference_time + total_postprocess_time) / total_frames
    
    pipeline_fps = 1000.0 / avg_total if avg_total > 0 else 0.0
    
    # Step 4: Save results
    print("\n[STEP 4] Saving results...")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"pipeline_test_{model}_{timestamp}.json"
    
    summary = {
        "test_info": {
            "video": str(video_path),
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "frames_processed": total_frames
        },
        "validation_results": {
            "valid_frames": valid_count,
            "invalid_frames": invalid_count,
            "success_rate": (valid_count / total_frames * 100) if total_frames > 0 else 0
        },
        "performance_metrics": {
            "avg_preprocessing_ms": avg_preprocess,
            "avg_inference_ms": avg_inference,
            "avg_postprocessing_ms": avg_postprocess,
            "avg_total_ms": avg_total,
            "pipeline_fps": pipeline_fps,
            "meets_20fps_requirement": pipeline_fps >= 20.0
        },
        "frame_results": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Step 5: Print summary
    print("\n" + "="*80)
    print("PIPELINE TEST SUMMARY")
    print("="*80)
    print(f"Total Frames Processed: {total_frames}")
    print(f"Valid Outputs: {valid_count} ({valid_count/total_frames*100:.1f}%)")
    print(f"Invalid Outputs: {invalid_count} ({invalid_count/total_frames*100:.1f}%)")
    print()
    print("Performance Breakdown:")
    print(f"  Preprocessing:  {avg_preprocess:>8.2f}ms")
    print(f"  Inference:      {avg_inference:>8.2f}ms")
    print(f"  Postprocessing: {avg_postprocess:>8.2f}ms")
    print(f"  ─────────────────────────")
    print(f"  Total:          {avg_total:>8.2f}ms")
    print(f"  Pipeline FPS:   {pipeline_fps:>8.2f}")
    print()
    
    if pipeline_fps >= 20.0:
        print("✓ PASS: Meets 20 FPS real-time requirement")
    else:
        print(f"✗ FAIL: Does not meet 20 FPS requirement ({pipeline_fps:.2f} FPS)")
    
    print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Integration Test")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--model", type=str, default="llava", help="LLM model to use (default: llava)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process (default: all)")
    parser.add_argument("--fps-extract", type=float, default=2.0, help="Frames per second to extract (default: 2.0)")
    
    args = parser.parse_args()
    
    # Verify video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Run pipeline test with Ollama service
    with ollama_utils.OllamaService():
        run_pipeline_test(
            video_path=args.video,
            model=args.model,
            max_frames=args.max_frames,
            fps_extract=args.fps_extract
        )

if __name__ == "__main__":
    main()