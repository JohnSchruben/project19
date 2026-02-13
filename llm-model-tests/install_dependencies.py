import subprocess
import sys

def install_package(package):
    print(f"--- Installing python package: {package} ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}\n")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}\n")

def check_ollama_installed():
    try:
        subprocess.check_output(["ollama", "--version"], stderr=subprocess.STDOUT)
        print("Ollama is already installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_ollama():
    print("--- Ollama not found. Attempting to install... ---")
    if sys.platform == "win32":
        try:
            print("Attempting to install via winget...")
            subprocess.check_call(["winget", "install", "-e", "--id", "Ollama.Ollama"])
            print("Ollama installed successfully. You may need to restart your terminal.")
        except Exception as e:
            print(f"Failed to install Ollama via winget: {e}")
            print("Please download and install Ollama manually from https://ollama.com/download")
    else:
        # Linux/Mac fallback (curl)
        try:
             print("Attempting to install via curl script...")
             subprocess.check_call("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
        except Exception as e:
            print(f"Failed to install Ollama: {e}")
            print("Please install manually from https://ollama.com")

def pull_model(model_name):
    print(f"--- Pulling Ollama model: {model_name} ---")
    try:
        subprocess.check_call(["ollama", "pull", model_name])
        print(f"Successfully pulled {model_name}\n")
    except subprocess.CalledProcessError:
        print(f"Failed to pull {model_name}. Make sure Ollama is installed and running.\n")
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Is it installed and in your PATH?\n")

def main():
    # 0. Check for Ollama binary
    if not check_ollama_installed():
        install_ollama()

    # 1. Install Python libraries for generic test
    py_dependencies = [
        "ollama",
        "pillow",
        "datasets",
        "transformers",
        "peft",
        "bitsandbytes",
        "trl",
        "torch",
        "accelerate"
    ]

    print("=== Phase 1: Installing Python Dependencies ===\n")
    for dep in py_dependencies:
        install_package(dep)
    
    # 2. Pull Ollama models
    # Models list from shared config
    try:
        from models import OLLAMA_VISION_MODELS
        models = OLLAMA_VISION_MODELS
    except ImportError:
        print("Error: Could not import 'OLLAMA_VISION_MODELS' from 'models.py'.")
        return

    print("=== Phase 2: Pulling Ollama Models ===\n")
    print("Make sure 'ollama serve' is running in another terminal if needed (though 'ollama pull' usually starts it).\n")
    
    for model in models:
        pull_model(model)
    
    print("All dependency checks and model pulls completed.")

if __name__ == "__main__":
    main()
