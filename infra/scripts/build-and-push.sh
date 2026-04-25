#!/usr/bin/env bash
# =============================================================================
# Build Docker images and push to ECR
# =============================================================================
# Usage: ./infra/scripts/build-and-push.sh
# Prerequisites: AWS CLI configured, Docker running, terraform applied
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Get AWS account and region from Terraform outputs
cd "$SCRIPT_DIR/../terraform"
REGION=$(terraform output -raw cluster_region 2>/dev/null || echo "us-east-1")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD)

echo "================================================"
echo "MedExRAG — Build & Push to ECR"
echo "================================================"
echo "Region:     $REGION"
echo "Account:    $ACCOUNT_ID"
echo "ECR Base:   $ECR_BASE"
echo "Git SHA:    $GIT_SHA"
echo "================================================"

# Authenticate Docker with ECR
echo ""
echo "[1/5] Authenticating with ECR..."
aws ecr get-login-password --region "$REGION" | \
  docker login --username AWS --password-stdin "$ECR_BASE"

# Build base image
echo ""
echo "[2/5] Building base image..."
cd "$PROJECT_ROOT"
docker build \
  -f docker/Dockerfile.base \
  -t medexrag-base:latest \
  .

# Build and push Streamlit
echo ""
echo "[3/5] Building & pushing Streamlit image..."
docker build \
  -f docker/Dockerfile.streamlit \
  -t "${ECR_BASE}/medexrag-streamlit:${GIT_SHA}" \
  -t "${ECR_BASE}/medexrag-streamlit:latest" \
  .
docker push "${ECR_BASE}/medexrag-streamlit:${GIT_SHA}"
docker push "${ECR_BASE}/medexrag-streamlit:latest"

# Build and push VLM Worker
echo ""
echo "[4/5] Building & pushing VLM Worker image..."
docker build \
  -f docker/Dockerfile.vlm-worker \
  -t "${ECR_BASE}/medexrag-vlm-worker:${GIT_SHA}" \
  -t "${ECR_BASE}/medexrag-vlm-worker:latest" \
  .
docker push "${ECR_BASE}/medexrag-vlm-worker:${GIT_SHA}"
docker push "${ECR_BASE}/medexrag-vlm-worker:latest"

# Build and push CLI
echo ""
echo "[5/5] Building & pushing CLI image..."
docker build \
  -f docker/Dockerfile.cli \
  -t "${ECR_BASE}/medexrag-cli:${GIT_SHA}" \
  -t "${ECR_BASE}/medexrag-cli:latest" \
  .
docker push "${ECR_BASE}/medexrag-cli:${GIT_SHA}"
docker push "${ECR_BASE}/medexrag-cli:latest"

echo ""
echo "================================================"
echo "All images pushed successfully!"
echo "  ${ECR_BASE}/medexrag-streamlit:${GIT_SHA}"
echo "  ${ECR_BASE}/medexrag-vlm-worker:${GIT_SHA}"
echo "  ${ECR_BASE}/medexrag-cli:${GIT_SHA}"
echo "================================================"
