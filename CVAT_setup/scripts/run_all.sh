#!/usr/bin/env bash
set -e

# ==========================================
# CVAT + YOLOv7 Automated Setup Script
# ==========================================
echo "[INFO] Starting Automated Setup for CVAT!"

# 1. Install Docker & Prerequisites (from docker_setup.sh)
echo "[INFO] Running docker setup..."
if ! command -v docker &> /dev/null; then
    sudo apt update
    sudo apt install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    . /etc/os-release
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin git wget
    sudo usermod -aG docker $USER
else
    echo "[INFO] Docker already installed, skipping installation."
fi

# Group permission trick: Run everything else within the 'docker' group context
# This avoids needing to log out and log back in!
sg docker -c '
set -e

# 2. Clone CVAT
echo "[INFO] Cloning CVAT repository..."
if [ ! -d "$HOME/cvat" ]; then
    cd ~
    git clone https://github.com/cvat-ai/cvat.git
else
    echo "[INFO] ~/cvat already exists. Skipping clone."
fi

# 3. Start CVAT with Serverless Compose
echo "[INFO] Starting CVAT containers..."
cd ~/cvat
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d

# Wait for CVAT server container to actually be ready
echo "[INFO] Waiting 15 seconds for CVAT server to spin up..."
sleep 15

# 4. Create Admin User (Non-interactively)
echo "[INFO] Creating admin superuser..."
docker exec -i cvat_server python3 manage.py createsuperuser \
    --noinput \
    --username "admin" \
    --email "admin@example.com" || echo "[WARN] User might already exist, ignoring error."

# Set the password to "admin123"
docker exec -i cvat_server python3 manage.py shell -c "from django.contrib.auth import get_user_model; User = get_user_model(); u = User.objects.get(username=\"admin\"); u.set_password(\"admin123\"); u.save()"
echo "[INFO] Admin user created with username: admin and password: admin123"

# 5. Install Nuclio CLI (nuctl)
echo "[INFO] Installing Nuclio CLI..."
if ! command -v nuctl &> /dev/null; then
    wget https://github.com/nuclio/nuclio/releases/download/1.13.0/nuctl-1.13.0-linux-amd64
    sudo mv nuctl-1.13.0-linux-amd64 /usr/local/bin/nuctl
    sudo chmod +x /usr/local/bin/nuctl
else
    echo "[INFO] Nuclio CLI already installed."
fi

# 6. Deploy YOLOv7 Serverless Function
echo "[INFO] Deploying YOLOv7 via Nuclio. This may take a few minutes..."
cd ~/cvat
DOCKER_API_VERSION=1.52 nuctl deploy \
  --project-name cvat \
  --path serverless/onnx/WongKinYiu/yolov7/nuclio \
  --file serverless/onnx/WongKinYiu/yolov7/nuclio/function.yaml \
  --platform local \
  --env CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk \
  --env CVAT_FUNCTIONS_REDIS_PORT=6666 \
  --platform-config "{\"attributes\":{\"network\":\"cvat_cvat\"}}" || {
    echo "[WARN] Deployment failed or function already exists. If it fails due to existing function, ignore this."
}

echo "=========================================="
echo "CVAT Setup Complete!"
echo "Access CVAT at http://localhost:8080"
echo "Username: admin"
echo "Password: admin123"
echo "=========================================="
'
