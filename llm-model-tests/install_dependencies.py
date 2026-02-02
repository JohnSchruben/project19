import subprocess
import sys

def install(package):
    print(f"--- Installing {package} ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}\n")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}\n")

def main():
    dependencies = [
        "transformers",
        "accelerate",
        "torch",
        "pillow",
        "moondream",
        "ollama",
        "openai",
        "timm"
    ]

    print("Starting dependency installation...\n")
    for dep in dependencies:
        install(dep)
    
    print("All dependency checks completed.")

if __name__ == "__main__":
    main()
