import glob
import os
import subprocess
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Run all LLM model tests")
    parser.add_argument("--image", type=str, help="Path to the image file to use for all tests")
    parser.add_argument("--prompt", type=str, help="Prompt text to use for all tests")
    args = parser.parse_args()

    # Find all python files ending with -test.py in the current directory
    scripts = glob.glob("*-test.py")

    for script in scripts:
        print(f"--- Running {script} ---")
        cmd = [sys.executable, script]
        
        if args.image:
            cmd.extend(["--image", args.image])
        
        if args.prompt:
            cmd.extend(["--prompt", args.prompt])
        
        try:
            result = subprocess.run(cmd, check=True)
            if result.returncode != 0:
                print(f"Error running {script}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to run {script}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while running {script}: {e}")
            
        print("\n")

if __name__ == "__main__":
    main()
