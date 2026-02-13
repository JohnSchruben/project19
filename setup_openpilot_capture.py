import os
import shutil
import argparse
import sys

def setup_openpilot(openpilot_path):
    """
    Copies the modified modeld files from this repo's external/Openpilot_Custom
    to the target openpilot repository.
    """
    openpilot_path = os.path.abspath(openpilot_path)
    if not os.path.exists(openpilot_path):
        print(f"Error: Openpilot path does not exist: {openpilot_path}")
        return False
    
    # Verify it looks like an openpilot repo
    if not os.path.exists(os.path.join(openpilot_path, "selfdrive", "modeld")):
        print(f"Error: {openpilot_path} does not strictly look like an openpilot repo (missing selfdrive/modeld)")
        return False
        
    print(f"Target Openpilot found at: {openpilot_path}")
    
    # Source files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "Openpilot_Custom", "openpilot_files", "selfdrive")
    
    # Files to copy (as per MiLa README)
    # They provide modeld_detection_first.py and modeld_detection_second.py
    # We want to replace selfdrive/modeld/modeld.py (or add it as a variant)
    # The instructions say: cp openpilot_files/selfdrive/modeld_detection_second.py ~/openpilot/selfdrive/modeld/
    
    source_file = os.path.join(source_dir, "modeld_detection_second.py")
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        print("Did you clone the Openpilot_Custom repo correctly?")
        return False

    target_dir = os.path.join(openpilot_path, "selfdrive", "modeld")
    target_file = os.path.join(target_dir, "modeld_detection_second.py")
    
    print(f"Copying {source_file} -> {target_file}")
    try:
        shutil.copy2(source_file, target_file)
        print("Success! File copied.")
        print("\nNext Steps (in your Openpilot environment):")
        print(f"1. Open {target_file}")
        print("2. Modify run_modeld.py or run this script manually instead of the standard modeld.")
        print("   (Refer to external/Openpilot_Custom/docs/DATA_PREPARATION_GUIDE.md for details)")
    except Exception as e:
        print(f"Error copying file: {e}")
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup Openpilot for MiLa Data Capture")
    parser.add_argument("openpilot_path", help="Path to your local openpilot repository")
    args = parser.parse_args()
    
    setup_openpilot(args.openpilot_path)
