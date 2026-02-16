#!/usr/bin/env python3
import argparse
import os
import subprocess
import signal
import sys
import time
from pathlib import Path

def run_pipeline(args):
    """
    Runs the Openpilot replay tool and modeld detection script concurrently.
    """
    
    # Paths
    replay_path = Path(args.replay_path)
    modeld_path = Path(args.modeld_path)
    dataset_dir = Path(args.dataset_dir)

    # Check if tools exist (warn if not, but allow proceed if user insists or just for dry run)
    if not replay_path.exists() and not args.dry_run:
        print(f"WARNING: Replay tool not found at {replay_path}. Ensure it is built.")
    if not modeld_path.exists() and not args.dry_run:
        print(f"WARNING: Modeld script not found at {modeld_path}. Check path.")

    # Expand user home in dataset dir
    dataset_dir = dataset_dir.expanduser()

    # Environment variables for modeld
    modeld_env = os.environ.copy()
    modeld_env["MODELD_DATASET_DIR"] = str(dataset_dir)
    modeld_env["MODELD_MAX_SEGMENT"] = str(args.max_segment)

    # Construct commands
    replay_cmd = [str(replay_path), args.route] + args.replay_flags.split()
    modeld_cmd = ["python3", str(modeld_path)]

    print("="*40)
    print("Openpilot Pipeline Runner")
    print("="*40)
    print(f"Replay Command: {' '.join(replay_cmd)}")
    print(f"Modeld Command: {' '.join(modeld_cmd)}")
    print(f"Modeld Env: MODELD_DATASET_DIR={dataset_dir}, MODELD_MAX_SEGMENT={args.max_segment}")
    print("="*40)

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return

    # Start modeld in background
    print("\nStarting modeld...")
    try:
        modeld_process = subprocess.Popen(modeld_cmd, env=modeld_env)
    except Exception as e:
        print(f"Error starting modeld: {e}")
        return

    # Give modeld a moment to initialize (optional, but good practice)
    time.sleep(2)

    # Start replay in foreground
    print("\nStarting replay...")
    try:
        subprocess.run(replay_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Replay tool failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Terminate modeld when replay finishes or is interrupted
        print("\nStopping modeld...")
        modeld_process.terminate()
        try:
            modeld_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            modeld_process.kill()
        print("Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Openpilot replay and modeld detection.")
    
    # Replay arguments
    parser.add_argument("--route", type=str, default="d34c14daa88a1e86/000000ca--7c5d326170",
                        help="Route to replay (default: d34c14daa88a1e86/000000ca--7c5d326170)")
    parser.add_argument("--replay-flags", type=str, default="--no-loop --dcam --ecam",
                        help="Flags for replay tool (default: --no-loop --dcam --ecam)")
    parser.add_argument("--replay-path", type=str, default="./tools/replay/replay",
                        help="Path to replay executable (default: ./tools/replay/replay)")

    # Modeld arguments
    parser.add_argument("--dataset-dir", type=str, default="~/project19/datasets/leaf_run",
                        help="Directory for modeld output (default: ~/project19/datasets/leaf_run)")
    parser.add_argument("--max-segment", type=int, default=12,
                        help="Max segment for modeld (default: 12)")
    parser.add_argument("--modeld-path", type=str, default="selfdrive/modeld_detection_second.py",
                        help="Path to modeld script (default: selfdrive/modeld_detection_second.py)")

    # General arguments
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")

    args = parser.parse_args()
    
    # Ensure dataset directory exists
    Path(args.dataset_dir).expanduser().mkdir(parents=True, exist_ok=True)

    run_pipeline(args)
