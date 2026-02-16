#!/bin/bash

# Define paths
# Assumes this script is run from project19 root and openpilot is .../openpilot
OPENPILOT_DIR="../openpilot"
CUSTOM_MODELD_DIR="Openpilot_Custom/openpilot_files/selfdrive"

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

# Run ubuntu setup script
echo "Running ubuntu setup script..."
tools/ubuntu_setup.sh
if [ $? -ne 0 ]; then
    echo "Error running ubuntu_setup.sh. Please install dependencies manually."
    exit 1
fi
cd - > /dev/null

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
        pip install -r "$DEPTH_DIR/requirements.txt"
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
    scons -u -j$(nproc)
    if [ $? -ne 0 ]; then
        echo "Error building Openpilot."
        exit 1
    fi
    cd - > /dev/null
    
    echo "Setup and Build Complete."
    echo "You can now run the pipeline:"
    echo "  python3 run_pipeline.py"
else
    echo "Error copying files."
    exit 1
fi
