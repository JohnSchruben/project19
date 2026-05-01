#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ALPAMAYO_VENV:-$PROJECT_ROOT/a1_5_venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv is required. Install uv first, then rerun this script." >&2
  exit 1
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
  echo "[INFO] Reusing Alpamayo venv: $VENV_DIR"
else
  echo "[INFO] Creating Alpamayo venv: $VENV_DIR"
  uv venv "$VENV_DIR" --python 3.12
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[INFO] Syncing Alpamayo dependencies into active root venv"
(
  cd "$PROJECT_ROOT/alpamayo"
  uv sync --active
)

echo "[SUCCESS] Alpamayo environment is ready"
echo ""
echo "Activate it from the project root with:"
echo "  source a1_5_venv/bin/activate"
echo ""
echo "Then run Alpamayo from the root with:"
echo "  python pipeline/run_alpamayo.py datasets/route_1/segment_00"
