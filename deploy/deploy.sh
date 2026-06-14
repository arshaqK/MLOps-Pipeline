#!/bin/bash
# deploy/deploy.sh — Build, push, and deploy the ML inference container to EC2.
#
# Usage:
#   Local:          bash deploy/deploy.sh
#   GitHub Actions: called automatically with secrets injected as env vars
#
# Required environment variables:
#   DOCKERHUB_USERNAME   — Docker Hub username
#   DOCKERHUB_TOKEN      — Docker Hub access token
#   EC2_HOST             — public IP of your EC2 instance
#   EC2_USER             — SSH user (default: ubuntu)
#   EC2_SSH_KEY          — path to .pem file locally, or key content in CI

set -euo pipefail

set -a
source .env
set +a

DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:?DOCKERHUB_USERNAME is required}"
DOCKERHUB_TOKEN="${DOCKERHUB_TOKEN:?DOCKERHUB_TOKEN is required}"
EC2_HOST="${EC2_HOST:?EC2_HOST is required}"
EC2_USER="${EC2_USER:-ubuntu}"
EC2_SSH_KEY="${EC2_SSH_KEY:-~/.ssh/MLOps.pem}"
IMAGE_NAME="${DOCKERHUB_USERNAME}/mlops-pipeline"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="mlops-pipeline"
APP_PORT="${APP_PORT:-8000}"

echo "========================================"
echo " MLOps Pipeline — Deploy Script"
echo " Image : ${IMAGE_NAME}:${IMAGE_TAG}"
echo " Target: ${EC2_USER}@${EC2_HOST}"
echo "========================================"

# ---------------------------------------------------------------------------
# Step 1 — Build Docker image
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Building Docker image..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "      Build complete."

# ---------------------------------------------------------------------------
# Step 2 — Push to Docker Hub
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Pushing image to Docker Hub..."
echo "${DOCKERHUB_TOKEN}" | docker login --username "${DOCKERHUB_USERNAME}" --password-stdin
docker push "${IMAGE_NAME}:${IMAGE_TAG}"
echo "      Push complete."

# ---------------------------------------------------------------------------
# Step 3 — SSH into EC2 and run container
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Deploying to EC2 (${EC2_HOST})..."

# If key is passed as content (CI), write it to a temp file
SSH_KEY_PATH="${EC2_SSH_KEY}"
if [[ "${EC2_SSH_KEY}" == -----BEGIN* ]]; then
    SSH_KEY_PATH=$(mktemp /tmp/ec2_key.XXXXXX)
    echo "${EC2_SSH_KEY}" > "${SSH_KEY_PATH}"
    chmod 600 "${SSH_KEY_PATH}"
fi

ssh -o StrictHostKeyChecking=no \
    -i "${SSH_KEY_PATH}" \
    "${EC2_USER}@${EC2_HOST}" << REMOTE
set -e

echo "  -> Logging into Docker Hub on EC2..."
echo "${DOCKERHUB_TOKEN}" | docker login --username "${DOCKERHUB_USERNAME}" --password-stdin

echo "  -> Pulling latest image..."
docker pull "${IMAGE_NAME}:${IMAGE_TAG}"

echo "  -> Stopping old container if running..."
docker stop "${CONTAINER_NAME}" 2>/dev/null || true
docker rm   "${CONTAINER_NAME}" 2>/dev/null || true

echo "  -> Starting new container..."
docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    -p "${APP_PORT}:8000" \
    -e MODEL_DIR=model/artifacts \
    -e LABEL_COL=label \
    -e SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL}" \
    -e POLL_INTERVAL="${POLL_INTERVAL}" \
    -e DRIFT_THRESHOLD="${DRIFT_THRESHOLD}" \
    -e RETRAIN_ROW_COUNT="${RETRAIN_ROW_COUNT}" \
    -e ACCURACY_THRESHOLD="${ACCURACY_THRESHOLD}" \
    -v /home/ubuntu/data:/app/data \
    "${IMAGE_NAME}:${IMAGE_TAG}"

echo "  -> Container started."
docker ps --filter "name=${CONTAINER_NAME}"
REMOTE

# Clean up temp key if created
if [[ "${EC2_SSH_KEY}" == -----BEGIN* ]]; then
    rm -f "${SSH_KEY_PATH}"
fi

# ---------------------------------------------------------------------------
# Step 4 — Smoke test
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Running smoke tests..."
sleep 5

HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "http://${EC2_HOST}:${APP_PORT}/health")
if [[ "${HEALTH}" == "200" ]]; then
    echo "      /health -> HTTP 200 OK"
else
    echo "      /health -> HTTP ${HEALTH} FAILED"
    exit 1
fi

PREDICT=$(curl -s -X POST "http://${EC2_HOST}:${APP_PORT}/predict" \
    -H "Content-Type: application/json" \
    -d '{"features": [1.5, 2.0]}')
echo "      /predict -> ${PREDICT}"

echo ""
echo "========================================"
echo " Deployment complete!"
echo " Service: http://${EC2_HOST}:${APP_PORT}"
echo "========================================"