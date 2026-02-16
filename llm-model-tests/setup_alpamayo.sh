#!/bin/bash
echo "Setting up Alpamayo Test Environment..."

# Check if hf is installed
if ! command -v hf &> /dev/null; then
    echo "hf command not found. Installing 'huggingface_hub'..."
    pip install huggingface_hub
    
    # Re-check (Note: standard pip install gives huggingface-cli, user might have alias 'hf')
    if ! command -v hf &> /dev/null; then
        echo "Warning: 'hf' command not found after installing huggingface_hub."
        echo "Checking for 'huggingface-cli' fallback..."
        if command -v huggingface-cli &> /dev/null; then
            echo "Found 'huggingface-cli'. Creating alias 'hf'..."
            alias hf=huggingface-cli
            shopt -s expand_aliases
        else
            echo "Error: Neither 'hf' nor 'huggingface-cli' found. Please install manually."
            exit 1
        fi
    fi
fi

# Install dependencies mentioned in Alpamayo README
echo "Installing Alpamayo specific dependencies..."
echo "Installing uv..."
pip install uv

echo "Installing physical_ai_av..."
pip install physical_ai_av

echo "Installing accelerate and transformers (ensure up to date)..."
pip install --upgrade accelerate transformers

echo "Installing pillow and torch (required for image processing)..."
pip install pillow torch torchvision

echo "Attempting to install flash-attn (optional but recommended)..."
pip install flash-attn --no-build-isolation || echo "Warning: flash-attn failed to install. Continuing without it."

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Warning: ffmpeg not found. It is recommended for handling video data."
    echo "Install with: sudo apt-get install ffmpeg"
fi

# Check if token is provided as argument
if [ -n "$1" ]; then
    echo "Logging in with provided token..."
    hf auth login --token "$1"
else
    echo "Please log in to Hugging Face to access the model."
    echo "You will need an access token from https://huggingface.co/settings/tokens"
    echo "Usage: ./setup_alpamayo.sh [HF_TOKEN] to avoid interactive login."
    hf auth login
fi

echo "Cloning Alpamayo repository..."
if [ ! -d "alpamayo" ]; then
    git clone https://github.com/NVlabs/alpamayo.git
else
    echo "alpamayo directory already exists. Pulling latest..."
    cd alpamayo && git pull && cd ..
fi

echo "Installing package dependencies manually..."
# Dependencies from pyproject.toml
# Enforcing numpy<2 due to compatibility issues with current torch/transformers stack
# Enforcing transformers>=4.57.1 as required by Alpamayo (for Qwen3VL support)
pip install "numpy<2" "transformers>=4.57.1" "accelerate>=1.12.0" "av>=16.0.1" "einops>=0.8.1" "hydra-colorlog>=1.2.0" "hydra-core>=1.3.2" "pandas>=2.3.3" "pillow>=12.0.0" "scipy" "tqdm" "matplotlib" --upgrade 

echo "Attempting to install physical_ai_av (optional/manual)..."
pip install physical_ai_av || echo "Warning: physical_ai_av failed to install. Proceeding without it." 

echo "Downloading NVIDIA Alpamayo-R1-10B model..."
# Explicitly download the model (snapshot) to the cache
hf download nvidia/Alpamayo-R1-10B --exclude "*.bin"

echo ""
echo "Setup complete."
echo "Since we skipped installing the package globally, 'test_alpamayo.py' handles the path."
echo "You can now run the test script:"
echo "python3 test_alpamayo.py --image <path_to_image>"

echo ""
echo "Setup complete. The model has been downloaded."
echo "You can now run the test script:"
echo "python3 test_alpamayo.py --image <path_to_image>"
