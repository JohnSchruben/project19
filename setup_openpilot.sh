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

# Copy files
echo "Copying custom modeld files..."
cp "$CUSTOM_MODELD_DIR/modeld_detection_second.py" "$OPENPILOT_DIR/selfdrive/modeld/"
cp "$CUSTOM_MODELD_DIR/modeld_detection_first.py" "$OPENPILOT_DIR/selfdrive/modeld/"

if [ $? -eq 0 ]; then
    echo "Files copied successfully."
    echo "You can now run the pipeline:"
    echo "  python3 run_pipeline.py"
else
    echo "Error copying files."
    exit 1
fi
