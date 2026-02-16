#!/bin/bash
# Wrapper to run test_alpamayo.py using the python interpreter associated with the active 'pip'
# This fixes issues where 'python3' points to system python while 'pip' points to a venv.

# Wrapper to run test_alpamayo.py using the correct python environment
# Priority:
# 1. uv run (if available and configured)
# 2. Python associated with the active 'pip'
# 3. System python3

echo "Detecting Python environment..."

# 1. Try uv run
if command -v uv &> /dev/null; then
    echo "Found 'uv'. Attempting to run with 'uv run'..."
    uv run python test_alpamayo.py "$@"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        exit 0
    else
        echo "'uv run' failed or exited with error. Trying fallback method..."
    fi
fi

# 2. Try Python associated with pip
PIP_EXE=$(command -v pip)
if [ -n "$PIP_EXE" ]; then
    DIR_PIP=$(dirname "$PIP_EXE")
    # Try finding python in same dir
    if [ -x "$DIR_PIP/python3" ]; then
        CANDIDATE="$DIR_PIP/python3"
    elif [ -x "$DIR_PIP/python" ]; then
        CANDIDATE="$DIR_PIP/python"
    fi
    
    if [ -n "$CANDIDATE" ]; then
        echo "Checking candidate: $CANDIDATE"
        if "$CANDIDATE" -c "import torch" &> /dev/null; then
            echo "Success! $CANDIDATE has torch installed."
            "$CANDIDATE" test_alpamayo.py "$@"
            exit $?
        else
            echo "Candidate $CANDIDATE missing torch."
        fi
    fi
fi

# 3. Last ditch: python3
echo "Falling back to system python3..."
python3 test_alpamayo.py "$@"
