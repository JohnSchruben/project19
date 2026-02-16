#!/bin/bash

# Define paths
# Assumes this script is run from project19 root
PROJECT_ROOT=$(pwd)
OPENPILOT_DIR="../openpilot"
CUSTOM_MODELD_DIR="Openpilot_Custom/openpilot_files/selfdrive"

# Ensure .local/bin is in PATH for uv
export PATH="$HOME/.local/bin:$PATH"

# Check if openpilot directory exists
if [ ! -d "$OPENPILOT_DIR" ]; then
    echo "Openpilot directory not found at $OPENPILOT_DIR"
    echo "Cloning openpilot..."
    git clone https://github.com/commaai/openpilot.git "$OPENPILOT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error cloning openpilot."
        exit 1
    fi
else
    echo "Found openpilot directory at $OPENPILOT_DIR"
fi

# Checkout v0.9.8
echo "Checking out openpilot v0.9.8..."
cd "$OPENPILOT_DIR"
git fetch --tags
git checkout v0.9.8
if [ $? -ne 0 ]; then
    echo "Error checking out v0.9.8. You might have local changes."
    exit 1
fi

# Initialize submodules
echo "Initializing submodules..."
git submodule update --init --recursive
if [ $? -ne 0 ]; then
    echo "Error initializing submodules."
    exit 1
fi

# Fix metadrive-simulator hash mismatch in uv.lock
# The upstream package hash changed, causing uv sync to fail. We patch it with the new hash.
echo "Patching uv.lock for metadrive-simulator..."
if [ -f "uv.lock" ]; then
    sed -i 's/fbf0ea9be67e65cd45d38ff930e3d49f705dd76c9ddbd1e1482e3f87b61efcef/d0afaf3b005e35e14b929d5491d2d5b64562d0c1cd5093ba969fb63908670dd4/g' uv.lock
fi

# Run ubuntu setup script
echo "Running ubuntu setup script..."
tools/ubuntu_setup.sh
if [ $? -ne 0 ]; then
    echo "Error running ubuntu_setup.sh. Please install dependencies manually."
    exit 1
fi

# Reload profile to ensure uv is in PATH
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Pull LFS files (needed for catch2 and others)
# Pull LFS files (needed for catch2 and others)
echo "Pulling Git LFS files..."
# We are already in openpilot directory from line 23
git lfs install
git lfs pull
if [ $? -ne 0 ]; then
    echo "Error pulling LFS files."
    exit 1
fi

# Return to project root to find custom files
echo "Returning to project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# Setup Depth Anything V2
DEPTH_DIR="$OPENPILOT_DIR/DepV2"
if [ ! -d "$DEPTH_DIR" ]; then
    echo "Cloning Depth Anything V2 into openpilot/DepV2..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2 "$DEPTH_DIR"
    if [ $? -ne 0 ]; then
        echo "Error cloning Depth Anything V2."
        exit 1
    fi
    
    # Try to install dependencies
    echo "Installing Depth Anything V2 dependencies..."
    if [ -f "$DEPTH_DIR/requirements.txt" ]; then
        # Use uv to install into the openpilot environment
        echo "Installing requirements using uv..."
        cd "$OPENPILOT_DIR"
        uv pip install -r "DepV2/requirements.txt"
        cd "$PROJECT_ROOT"
    else
        echo "WARNING: requirements.txt not found in Depth Anything V2 repo."
    fi
else
    echo "Depth Anything V2 directory already exists."
fi

# Copy files
echo "Copying custom modeld files..."
cp "$CUSTOM_MODELD_DIR/modeld_detection_second.py" "$OPENPILOT_DIR/selfdrive/modeld/"
cp "$CUSTOM_MODELD_DIR/modeld_detection_first.py" "$OPENPILOT_DIR/selfdrive/modeld/"

if [ $? -eq 0 ]; then
    echo "Files copied successfully."
    
    # Build Openpilot
    echo "Building Openpilot..."
    cd "$OPENPILOT_DIR"
    # Use uv run to execute scons within the environment
    uv run scons -u -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Error building Openpilot."
        exit 1
    fi
    # No need to cd back, script is ending
    
    echo "Setup and Build Complete."
    echo "You can now run the pipeline:"
    echo "  python3 run_pipeline.py"
else
    echo "Error copying files."
    exit 1
fi
