import subprocess
import sys
import time
import urllib.request
import os
import signal

def is_ollama_running():
    """Checks if the Ollama service is currently running."""
    try:
        with urllib.request.urlopen("http://localhost:11434") as response:
            return response.status == 200
    except Exception:
        return False

def start_ollama_service():
    """
    Starts the Ollama service if it's not already running.
    Returns True if it started the service, False if it was already running.
    """
    if is_ollama_running():
        return False

    print("--- Starting Ollama service... ---")
    try:
        # Start in a new process group/console so it doesn't die immediately if we are in a script
        if sys.platform == "win32":
            # creationflags=subprocess.CREATE_NEW_CONSOLE creates a visible window. 
            # If we want it hidden, we can use other flags, but visible is often better for debugging.
            subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Waiting for Ollama to initialize...")
        for i in range(20):  # Wait up to 20 seconds
            if is_ollama_running():
                print("Ollama started successfully.\n")
                return True
            time.sleep(1)
            
        print("Warning: Timed out waiting for Ollama to start. Proceeding anyway...\n")
        return True # We attempted to start it
    except Exception as e:
        print(f"Failed to start Ollama: {e}\n")
        return False

def stop_ollama_service():
    """Stops the Ollama service."""
    print("--- Stopping Ollama service... ---")
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "ollama_app.exe"], capture_output=True) # Just in case
        else:
            subprocess.run(["pkill", "ollama"], capture_output=True)
        print("Ollama service stopped.\n")
    except Exception as e:
        print(f"Error stopping Ollama: {e}\n")

class OllamaService:
    """
    Context manager to ensure Ollama is running during a block of code.
    It stops Ollama on exit ONLY if it was the one that started it.
    """
    def __init__(self):
        self.started_by_me = False

    def __enter__(self):
        self.started_by_me = start_ollama_service()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.started_by_me:
            stop_ollama_service()
        # Else: it was already running, so we leave it running.
