#!/bin/bash
# Wrapper script to run Alpamayo on Openpilot data

# Default arguments
IMAGE_DIR=""
OUTPUT_FILE="results.json"
LIMIT=""
VISUALIZE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --image) IMAGE_DIR="$2"; shift ;;
        --output) OUTPUT_FILE="$2"; shift ;;
        --limit) LIMIT="--limit $2"; shift ;;
        --visualize) VISUALIZE="--visualize" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$IMAGE_DIR" ]; then
    echo "Usage: ./run_alpamayo.sh --image <path/to/raw> --output <results.json> [--limit <N>] [--visualize]"
    exit 1
fi

# Run the python script
echo "Running Alpamayo on $IMAGE_DIR..."
python alpamayo/run_on_openpilot.py --image "$IMAGE_DIR" --output "$OUTPUT_FILE" $LIMIT $VISUALIZE
