#!/usr/bin/env python3
import argparse
import os
import subprocess
import signal
import sys
import time
from pathlib import Path

def get_modeld_cmd(op_dir, modeld_path, args):
    """
    Constructs the modeld command, attempting to detect the environment runner.
    """
    cmd = []
    
    # Check if user specified a python command wrapper
    if args.python_cmd:
        cmd.extend(args.python_cmd.split())
    else:
        # Auto-detect environment
        if (op_dir / "poetry.lock").exists():
            print("Detected poetry environment.")
            cmd.extend(["poetry", "run", "python3"])
        elif (op_dir / "Pipfile").exists():
            print("Detected pipenv environment.")
            cmd.extend(["pipenv", "run", "python3"])
        elif (op_dir / "uv.lock").exists():
            print("Detected uv environment.")
            cmd.extend(["uv", "run", "python3"])
        else:
            # Default to system python3
            # If inside a venv (virtualenv), python3 should work if activated.
            # If not activated, we might be able to find venv/bin/python
            venv_python = op_dir / "venv" / "bin" / "python3"
            if venv_python.exists():
                print(f"Detected venv at {venv_python}")
                cmd.append(str(venv_python))
            else:
                 cmd.append("python3")

    cmd.append(str(modeld_path))
    return cmd

def get_terminal_cmd(cmd_list, env, block=False):
    """
    Wraps a command to run in a new terminal window.
    Returns (terminal_cmd_list, is_blocking)
    """
    # Convert command list to string for shell execution
    env_str = " ".join([f"{k}={v}" for k, v in env.items() if k.startswith("MODELD_") or k == "PYTHONPATH"])
    cmd_str = f"{env_str} {' '.join(cmd_list)}"
    
    # Common terminal emulators
    # Format: (terminal, non_blocking_args, blocking_args)
    terminals = [
        ("gnome-terminal", 
         ["--", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["--wait", "--", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]),
        ("konsole", 
         ["-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["--nofork", "-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]),
        ("xfce4-terminal", 
         ["-x", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["--disable-server", "-x", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]),
        ("xterm", 
         ["-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"],
         ["-e", "bash", "-c", f"{cmd_str}; echo 'Process finished. Press enter to exit.'; read"]), # xterm -e blocks by default
    ]

    for term, args_nb, args_b in terminals:
        if shutil.which(term):
            if block:
                # gnome-terminal --wait blocks. xterm blocks by default.
                return [term] + args_b, True
            else:
                return [term] + args_nb, False # terminals usually return immediately (daemonize) or block.
            
    return None, False

def run_pipeline(args):
    """
    Runs the Openpilot replay tool and modeld detection script concurrently.
    """
    
    # Resolve openpilot directory
    op_dir = Path(args.openpilot_dir).resolve()
    if not op_dir.exists() and not args.dry_run:
        print(f"ERROR: Openpilot directory not found at {op_dir}")
        print("Please specify correct path with --openpilot-dir")
        return

    # Paths (relative to openpilot_dir unless absolute)
    replay_path = Path(args.replay_path)
    if not replay_path.is_absolute():
        replay_path = op_dir / replay_path

    modeld_path = Path(args.modeld_path)
    if not modeld_path.is_absolute():
        modeld_path = op_dir / modeld_path

    dataset_dir = Path(args.dataset_dir).expanduser()

    # Check if tools exist
    if not replay_path.exists() and not args.dry_run:
        print(f"WARNING: Replay tool not found at {replay_path}. Ensure it is built.")
    if not modeld_path.exists() and not args.dry_run:
        print(f"WARNING: Modeld script not found at {modeld_path}.")

    # Environment variables for modeld
    modeld_env = os.environ.copy()
    modeld_env["MODELD_DATASET_DIR"] = str(dataset_dir)
    modeld_env["MODELD_MAX_SEGMENT"] = str(args.max_segment)
    modeld_env["MODELD_SEGMENT_FRAMES"] = str(args.segment_frames)
    # Ensure python path includes openpilot dir
    modeld_env["PYTHONPATH"] = f"{op_dir}:{modeld_env.get('PYTHONPATH', '')}"

    # Construct commands
    replay_cmd = [str(replay_path), args.route] + args.replay_flags.split()
    modeld_cmd = get_modeld_cmd(op_dir, modeld_path, args)

    print("="*40)
    print("Openpilot Pipeline Runner")
    print("="*40)
    print(f"Working Directory: {op_dir}")
    print(f"Replay Command: {' '.join(replay_cmd)}")
    print(f"Modeld Command: {' '.join(modeld_cmd)}")
    print(f"Modeld Env: MODELD_DATASET_DIR={dataset_dir}, MODELD_MAX_SEGMENT={args.max_segment}, MODELD_SEGMENT_FRAMES={args.segment_frames}")
    if args.new_terminal:
        print("Modeld will run in a NEW TERMINAL window.")
    if args.replay_new_terminal:
        print("Replay will run in a NEW TERMINAL window.")
    print("="*40)

    if args.dry_run:
        print("Dry run complete. Exiting.")
        return

    # Start modeld
    modeld_process = None
    use_terminal_modeld = False
    
    if args.new_terminal:
        term_cmd, _ = get_terminal_cmd(modeld_cmd, modeld_env, block=False)
        if term_cmd:
            print(f"\nLaunching modeld in new terminal: {' '.join(term_cmd)}")
            subprocess.Popen(term_cmd, cwd=op_dir) 
            use_terminal_modeld = True
            print("Modeld launched in separate window. Check it for output/errors.")
        else:
            print("\nWARNING: No supported terminal emulator found (gnome-terminal, konsole, xfce4-terminal, xterm).")
            print("Falling back to same-terminal execution.")
    
    if not use_terminal_modeld:
        print("\nStarting modeld in background...")
        try:
            # Run from op_dir so relative imports/paths work
            modeld_process = subprocess.Popen(modeld_cmd, env=modeld_env, cwd=op_dir)
        except Exception as e:
            print(f"Error starting modeld: {e}")
            return

    # Give modeld a moment to initialize
    time.sleep(2)

    # Start replay
    print("\nStarting replay...")
    try:
        if args.replay_new_terminal:
            term_cmd, is_blocking = get_terminal_cmd(replay_cmd, os.environ.copy(), block=True)
            if term_cmd:
                print(f"Launching replay in new terminal: {' '.join(term_cmd)}")
                if is_blocking:
                    subprocess.run(term_cmd, check=True, cwd=op_dir)
                else:
                    print("WARNING: Terminal emulator does not support blocking wait. Pipeline might finish prematurely.")
                    print("Attempting to run anyway, but modeld might be killed early.")
                    subprocess.run(term_cmd, check=True, cwd=op_dir)
            else:
                 print("\nWARNING: No supported terminal emulator found for replay.")
                 print("Falling back to same-terminal execution.")
                 subprocess.run(replay_cmd, check=True, cwd=op_dir)
        else:
            # Run from op_dir
            subprocess.run(replay_cmd, check=True, cwd=op_dir)
            
    except subprocess.CalledProcessError as e:
        print(f"Replay tool failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Terminate modeld when replay finishes or is interrupted
        if modeld_process:
            print("\nStopping modeld...")
            modeld_process.terminate()
            try:
                modeld_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                modeld_process.kill()
        elif use_terminal_modeld:
            print("\nReplay finished.")
            print("NOTE: Modeld running in the separate terminal may keep running or wait for input.")
            print("Please close that window manually if it remains open.")
            
        print("Pipeline finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Openpilot Replay and Modeld Pipeline")
    
    parser.add_argument("--route", type=str, required=True,
                        help="Route ID (e.g., 'd34c14daa88a1e86/000000ca--7c5d326170')")
    parser.add_argument("--replay-flags", type=str, default="--dcamera --ecamera --no-vipc",
                        help="Flags for replay (default: --dcamera --ecamera --no-vipc)")
    parser.add_argument("--replay-path", type=str, default="./tools/replay/replay",
                        help="Path to replay executable (default: ./tools/replay/replay)")

    # Modeld arguments
    parser.add_argument("--dataset-dir", type=str, default="~/project19/datasets/leaf_run_2",
                        help="Directory for modeld output (default: ~/project19/datasets/leaf_run_2)")
    parser.add_argument("--max-segment", type=int, default=12,
                        help="Max segment for modeld (default: 12)")
    parser.add_argument("--segment-frames", type=int, default=175,
                        help="Frames per segment (MODELD_SEGMENT_FRAMES) (default: 175)")
    parser.add_argument("--modeld-path", type=str, default="selfdrive/modeld/modeld_detection_second.py",
                        help="Path to modeld script (default: selfdrive/modeld/modeld_detection_second.py)")

    # General arguments
    parser.add_argument("--openpilot-dir", type=str, default="../openpilot",
                        help="Path to openpilot directory (default: ../openpilot)")
    parser.add_argument("--python-cmd", type=str, default=None,
                        help="Python command/wrapper to use (e.g. 'poetry run python3'). Auto-detected if not set.")
    parser.add_argument("--new-terminal", action="store_true", default=True,
                        help="Open modeld in a new terminal window to avoid output interleaving (default: True)")
    parser.add_argument("--no-new-terminal", action="store_false", dest="new_terminal",
                        help="Run modeld in the same terminal (background)")
    parser.add_argument("--replay-new-terminal", action="store_true", default=True,
                        help="Open replay tool in a new terminal window (default: True)")
    parser.add_argument("--no-replay-new-terminal", action="store_false", dest="replay_new_terminal",
                        help="Run replay in the same terminal")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    
    args = parser.parse_args()
    
    # Ensure dataset directory exists
    Path(args.dataset_dir).expanduser().mkdir(parents=True, exist_ok=True)

    run_pipeline(args)
