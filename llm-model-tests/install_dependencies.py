import subprocess
import sys
import time
import urllib.request

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

def is_ollama_running():
    try:
        with urllib.request.urlopen("http://localhost:11434") as response:
            return response.status == 200
    except Exception:
        return False

def start_ollama_service():
    if is_ollama_running():
        # print("Ollama is already running check passed.")
        return

    print("--- Starting Ollama service... ---")
    try:
        if sys.platform == "win32":
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Waiting for Ollama to initialize...")
        for i in range(10):
            if is_ollama_running():
                print("Ollama started successfully.\n")
                return
            time.sleep(1)
            
        print("Warning: Timed out waiting for Ollama to start. Proceeding anyway...\n")
    except Exception as e:
        print(f"Failed to start Ollama: {e}\n")

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
    
    start_ollama_service()
    
    for model in models:
        pull_model(model)
    
    print("All dependency checks and model pulls completed.")

if __name__ == "__main__":
    main()
