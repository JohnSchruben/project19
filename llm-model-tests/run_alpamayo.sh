#!/bin/bash
# Wrapper to run test_alpamayo.py using the python interpreter associated with the active 'pip'
# This fixes issues where 'python3' points to system python while 'pip' points to a venv.

PIP_EXE=$(command -v pip)
PYTHON_EXE=$(command -v python3)

echo "Resolved pip: $PIP_EXE"
echo "Resolved python3: $PYTHON_EXE"

# Check if pip and python are in the same directory
DIR_PIP=$(dirname "$PIP_EXE")
DIR_PY=$(dirname "$PYTHON_EXE")

TARGET_PYTHON="$PYTHON_EXE"

if [ "$DIR_PIP" != "$DIR_PY" ]; then
    echo "Warning: pip and python3 are in different directories."
    echo "Attempting to use python from pip's directory: $DIR_PIP"
    
    if [ -x "$DIR_PIP/python3" ]; then
        TARGET_PYTHON="$DIR_PIP/python3"
        echo "Found: $TARGET_PYTHON"
    elif [ -x "$DIR_PIP/python" ]; then
        TARGET_PYTHON="$DIR_PIP/python"
        echo "Found: $TARGET_PYTHON"
    else
        echo "Could not find python exe in pip directory. Getting sys.executable from pip..."
        # Fallback: ask pip which python it uses (if possible) or just try running module
        # Running pip -V usually implies the python version/path
    fi
fi

echo "Running test with: $TARGET_PYTHON"
"$TARGET_PYTHON" test_alpamayo.py "$@"
