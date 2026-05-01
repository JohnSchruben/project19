#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_full_pipeline.sh [options] <route-folder-or-segment-folder>

Examples:
  ./run_full_pipeline.sh datasets/route_1
  ./run_full_pipeline.sh datasets/route_1/segment_00
  ./run_full_pipeline.sh --openpilot-route "d34c14daa88a1e86/000000ca--7c5d326170" datasets/route_1

Options:
  --openpilot-route ROUTE_ID   Run pipeline/route_caputure.py first and write data into the route folder.
  --num-traj-samples N         Alpamayo trajectory samples per frame. Default: 1.
  --help                       Show this help.
EOF
}

OPENPILOT_ROUTE=""
NUM_TRAJ_SAMPLES=1
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --openpilot-route)
      OPENPILOT_ROUTE="${2:-}"
      shift 2
      ;;
    --num-traj-samples)
      NUM_TRAJ_SAMPLES="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -n "$TARGET" ]]; then
        echo "Only one route or segment folder may be provided." >&2
        usage
        exit 1
      fi
      TARGET="$1"
      shift
      ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  usage
  exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

pick_host_python() {
  local venv_bin=""
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    venv_bin="$(cd "$VIRTUAL_ENV/bin" 2>/dev/null && pwd || true)"
  fi

  local path_dir candidate resolved
  IFS=':' read -r -a path_dirs <<< "$PATH"
  for path_dir in "${path_dirs[@]}"; do
    [[ -z "$path_dir" ]] && continue
    resolved="$(cd "$path_dir" 2>/dev/null && pwd || true)"
    [[ -z "$resolved" ]] && continue
    [[ -n "$venv_bin" && "$resolved" == "$venv_bin" ]] && continue
    [[ "$resolved" == "$PROJECT_ROOT/alpamayo/a1_5_venv/bin" ]] && continue
    [[ "$resolved" == "$PROJECT_ROOT/a1_5_venv/bin" ]] && continue

    candidate="$resolved/python3"
    if [[ -x "$candidate" ]] && "$candidate" -m pip --version >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi

    candidate="$resolved/python"
    if [[ -x "$candidate" ]] && "$candidate" -m pip --version >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  echo "python3"
}

HOST_PYTHON="$(pick_host_python)"
echo "[INFO] Host Python: $HOST_PYTHON"

reset_pipeline_db() {
  echo "[INFO] Resetting pipeline/annotations.db"
  "$HOST_PYTHON" - <<'PY'
import sqlite3
from pathlib import Path

db_path = Path("pipeline/annotations.db")
schema_path = Path("pipeline/schema.sql")

if db_path.exists():
    db_path.unlink()

conn = sqlite3.connect(db_path)
conn.executescript(schema_path.read_text(encoding="utf-8"))
conn.commit()
conn.close()
PY
}

real_path() {
  "$HOST_PYTHON" - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
}

setup_alpamayo_env() {
  local venv_dir="$PROJECT_ROOT/a1_5_venv"

  if [[ -f "$venv_dir/bin/activate" ]]; then
    echo "[INFO] Reusing existing Alpamayo venv: $venv_dir"
  else
    uv venv "$venv_dir" --python 3.12
  fi

  # shellcheck source=/dev/null
  source "$venv_dir/bin/activate"
  (
    cd "$PROJECT_ROOT/alpamayo"
    uv sync --active
  )
}

if [[ -n "$OPENPILOT_ROUTE" ]]; then
  DATASET_DIR="$TARGET"
  if [[ "$(basename "$TARGET")" == segment_* ]]; then
    DATASET_DIR="$(dirname "$TARGET")"
  fi
  echo "[INFO] Running route capture into $DATASET_DIR"
  "$HOST_PYTHON" pipeline/route_caputure.py --route "$OPENPILOT_ROUTE" --dataset-dir "$DATASET_DIR"
fi

if [[ ! -d "$TARGET" ]]; then
  echo "[ERROR] Folder does not exist: $TARGET" >&2
  exit 1
fi

TARGET_ABS="$(real_path "$TARGET")"

if [[ "$(basename "$TARGET_ABS")" == segment_* ]]; then
  ROUTE_DIR="$(dirname "$TARGET_ABS")"
  SEGMENTS=("$TARGET_ABS")
else
  ROUTE_DIR="$TARGET_ABS"
  SEGMENTS=()
  while IFS= read -r seg; do
    SEGMENTS+=("$seg")
  done < <(find "$ROUTE_DIR" -maxdepth 1 -type d -name 'segment_*' | sort)
fi

if [[ ${#SEGMENTS[@]} -eq 0 ]]; then
  echo "[ERROR] No segment_* folders found under $ROUTE_DIR" >&2
  exit 1
fi

echo "[INFO] Route folder: $ROUTE_DIR"
echo "[INFO] Segments:"
for seg in "${SEGMENTS[@]}"; do
  echo "  - $seg"
done

reset_pipeline_db

echo "[INFO] Installing local annotation dependencies"
"$HOST_PYTHON" -m pip install ultralytics pillow

echo "[INFO] Running local YOLO annotation"
for seg in "${SEGMENTS[@]}"; do
  "$HOST_PYTHON" pipeline/annotate_route.py "$seg"
done

echo "[INFO] Preparing Alpamayo environment"
setup_alpamayo_env

for seg in "${SEGMENTS[@]}"; do
  echo "[INFO] Running Alpamayo for $(basename "$seg")"
  python pipeline/run_alpamayo.py "$seg" --num-traj-samples "$NUM_TRAJ_SAMPLES"
done

deactivate

echo "[INFO] Importing annotations and Alpamayo predictions into pipeline DB"
"$HOST_PYTHON" pipeline/import_route_db.py "$TARGET_ABS" --overwrite

echo "[SUCCESS] Built pipeline/annotations.db"
