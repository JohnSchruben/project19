#!/bin/bash
echo "Setting up Alpamayo Test Environment..."

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Installing 'huggingface_hub'..."
    pip install huggingface_hub
    
    # Re-check
    if ! command -v huggingface-cli &> /dev/null; then
        echo "Error: Failed to install huggingface-cli. Please install manually: pip install huggingface_hub"
        exit 1
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
    huggingface-cli login --token "$1"
else
    echo "Please log in to Hugging Face to access the model."
    echo "You will need an access token from https://huggingface.co/settings/tokens"
    echo "Usage: ./setup_alpamayo.sh [HF_TOKEN] to avoid interactive login."
    huggingface-cli login
fi

echo "Downloading NVIDIA Alpamayo-R1-10B model..."
# Explicitly download the model (snapshot) to the cache
huggingface-cli download nvidia/Alpamayo-R1-10B --exclude "*.bin"

echo ""
echo "Setup complete. The model has been downloaded."
echo "You can now run the test script:"
echo "python3 test_alpamayo.py --image <path_to_image>"
