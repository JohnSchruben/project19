#!/usr/bin/env bash
set -euo pipefail

# Headless CVAT + Nuclio + custom YOLOv8 setup.
# Run from anywhere on Ubuntu/Linux/WSL:
#   bash CVAT_setup/scripts/setup_cvat_yolov8_headless.sh
#
# Optional overrides:
#   CVAT_DIR=$HOME/cvat
#   CVAT_USER=admin
#   CVAT_PASSWORD=admin123
#   CVAT_EMAIL=admin@example.com
#   ANNOTATION_VENV=/path/to/venv
#   SKIP_DOCKER_INSTALL=1

CVAT_DIR="${CVAT_DIR:-$HOME/cvat}"
CVAT_USER="${CVAT_USER:-admin}"
CVAT_PASSWORD="${CVAT_PASSWORD:-admin123}"
CVAT_EMAIL="${CVAT_EMAIL:-admin@example.com}"
NUCLIO_VERSION="${NUCLIO_VERSION:-1.13.0}"
FUNCTION_NAME="${FUNCTION_NAME:-ultralytics-yolov8s-custom-v2}"
SKIP_DOCKER_INSTALL="${SKIP_DOCKER_INSTALL:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
YOLO_SRC="$REPO_ROOT/CVAT_setup/Yolov8_setup"
YOLO_DEST="$CVAT_DIR/serverless/ultralytics/yolov8s_custom/nuclio"
ANNOTATION_VENV="${ANNOTATION_VENV:-$REPO_ROOT/CVAT_setup/.venv}"

log() {
  printf '\n[INFO] %s\n' "$*"
}

warn() {
  printf '\n[WARN] %s\n' "$*" >&2
}

die() {
  printf '\n[ERROR] %s\n' "$*" >&2
  exit 1
}

require_linux() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    die "This script is intended for Ubuntu/Linux/WSL. Run it inside your Linux CLI environment."
  fi
}

install_base_packages() {
  log "Installing base packages"
  sudo apt update
  sudo apt install -y ca-certificates curl gnupg git wget python3 python3-venv python3-pip
}

install_docker() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    log "Docker and Docker Compose are already installed"
    return
  fi

  if [[ "$SKIP_DOCKER_INSTALL" == "1" ]]; then
    die "Docker or Docker Compose is missing, and SKIP_DOCKER_INSTALL=1 was set."
  fi

  log "Installing Docker and Docker Compose plugin"
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg

  # shellcheck source=/dev/null
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" |
    sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  if ! id -nG "$USER" | grep -qw docker; then
    sudo usermod -aG docker "$USER"
    warn "Added $USER to the docker group. This script will use sudo for Docker now; open a new shell later to use Docker without sudo."
  fi
}

start_docker_daemon() {
  if docker ps >/dev/null 2>&1 || sudo docker ps >/dev/null 2>&1; then
    return
  fi

  log "Starting Docker daemon"
  sudo systemctl start docker >/dev/null 2>&1 || sudo service docker start >/dev/null 2>&1 || true

  if ! docker ps >/dev/null 2>&1 && ! sudo docker ps >/dev/null 2>&1; then
    die "Docker is installed but the daemon is not running. Start Docker, then rerun this script."
  fi
}

set_docker_wrappers() {
  if docker ps >/dev/null 2>&1; then
    DOCKER=(docker)
    NUCTL=(env DOCKER_API_VERSION=1.52 nuctl)
  else
    warn "Docker is not available to this shell without sudo; using sudo for Docker and Nuclio commands."
    DOCKER=(sudo docker)
    NUCTL=(sudo env DOCKER_API_VERSION=1.52 nuctl)
  fi
}

clone_cvat() {
  if [[ -d "$CVAT_DIR/.git" ]]; then
    log "CVAT already exists at $CVAT_DIR"
    return
  fi

  log "Cloning CVAT into $CVAT_DIR"
  git clone https://github.com/cvat-ai/cvat.git "$CVAT_DIR"
}

start_cvat() {
  log "Starting CVAT with serverless support"
  cd "$CVAT_DIR"
  "${DOCKER[@]}" compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

  log "Waiting for cvat_server to accept Django commands"
  for _ in $(seq 1 60); do
    if "${DOCKER[@]}" exec cvat_server python3 manage.py check >/dev/null 2>&1; then
      return
    fi
    sleep 5
  done

  die "CVAT did not become ready after 5 minutes. Check: docker logs cvat_server"
}

create_admin_user() {
  log "Creating/updating CVAT admin user '$CVAT_USER'"
"${DOCKER[@]}" exec -e CVAT_USER="$CVAT_USER" -e CVAT_PASSWORD="$CVAT_PASSWORD" -e CVAT_EMAIL="$CVAT_EMAIL" cvat_server python3 manage.py shell -c '
import os
from django.contrib.auth import get_user_model

User = get_user_model()
user, created = User.objects.get_or_create(username=os.environ["CVAT_USER"], defaults={"email": os.environ["CVAT_EMAIL"], "is_staff": True, "is_superuser": True})
user.email = os.environ["CVAT_EMAIL"]
user.is_staff = True
user.is_superuser = True
user.set_password(os.environ["CVAT_PASSWORD"])
user.save()
print("created" if created else "updated")
' || die "Failed to create/update CVAT admin user"
}

install_nuctl() {
  if command -v nuctl >/dev/null 2>&1; then
    log "Nuclio CLI already installed"
    return
  fi

  log "Installing Nuclio CLI v$NUCLIO_VERSION"
  tmp_file="$(mktemp)"
  wget "https://github.com/nuclio/nuclio/releases/download/${NUCLIO_VERSION}/nuctl-${NUCLIO_VERSION}-linux-amd64" -O "$tmp_file"
  sudo mv "$tmp_file" /usr/local/bin/nuctl
  sudo chmod +x /usr/local/bin/nuctl
}

copy_yolov8_function() {
  [[ -f "$YOLO_SRC/function.yaml" ]] || die "Missing $YOLO_SRC/function.yaml"
  [[ -f "$YOLO_SRC/main.py" ]] || die "Missing $YOLO_SRC/main.py"
  [[ -f "$YOLO_SRC/model_handler.py" ]] || die "Missing $YOLO_SRC/model_handler.py"

  log "Copying custom YOLOv8 Nuclio function into CVAT"
  mkdir -p "$YOLO_DEST"
  cp "$YOLO_SRC/function.yaml" "$YOLO_SRC/main.py" "$YOLO_SRC/model_handler.py" "$YOLO_DEST/"
}

deploy_yolov8_function() {
  log "Deploying Nuclio function $FUNCTION_NAME"
  cd "$CVAT_DIR"

  "${NUCTL[@]}" create project cvat --platform local >/dev/null 2>&1 || true

  if "${NUCTL[@]}" get function "$FUNCTION_NAME" --platform local >/dev/null 2>&1; then
    warn "Existing function $FUNCTION_NAME found; deleting it before redeploy."
    "${NUCTL[@]}" delete function "$FUNCTION_NAME" --platform local
  fi

  "${NUCTL[@]}" deploy \
    --project-name cvat \
    --path serverless/ultralytics/yolov8s_custom/nuclio \
    --file serverless/ultralytics/yolov8s_custom/nuclio/function.yaml \
    --platform local \
    --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
    --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
    --platform-config '{"attributes":{"network":"cvat_cvat"}}'
}

verify_serverless() {
  log "Verifying CVAT serverless mode and Nuclio function"
  serverless="$("${DOCKER[@]}" exec cvat_server bash -lc 'echo "$CVAT_SERVERLESS"' | tr -d '\r')"
  [[ "$serverless" == "1" ]] || die "CVAT_SERVERLESS is '$serverless', expected '1'."

  "${NUCTL[@]}" get function --platform local
}

setup_annotation_venv() {
  log "Creating Python venv for headless annotation script"
  python3 -m venv "$ANNOTATION_VENV"
  "$ANNOTATION_VENV/bin/python" -m pip install --upgrade pip
  "$ANNOTATION_VENV/bin/python" -m pip install cvat-sdk
}

print_done() {
  cat <<EOF

==========================================
CVAT headless setup complete.

CVAT URL:       http://localhost:8080
Username:       $CVAT_USER
Password:       $CVAT_PASSWORD
YOLO function:  $FUNCTION_NAME
Annotation venv:
  $ANNOTATION_VENV

Run annotations from the project root like this:

  "$ANNOTATION_VENV/bin/python" CVAT_setup/scripts/auto_annotate.py \\
    datasets/route_1/segment_00/raw \\
    --url http://localhost:8080 \\
    --user "$CVAT_USER" \\
    --password "$CVAT_PASSWORD" \\
    --output route_1_segment_00_annotations.zip

==========================================
EOF
}

main() {
  require_linux
  install_base_packages
  install_docker
  start_docker_daemon
  set_docker_wrappers
  clone_cvat
  start_cvat
  create_admin_user
  install_nuctl
  copy_yolov8_function
  deploy_yolov8_function
  verify_serverless
  setup_annotation_venv
  print_done
}

main "$@"
