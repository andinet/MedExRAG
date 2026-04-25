#!/usr/bin/env bash
# =============================================================================
# Deploy MedExRAG to EKS
# =============================================================================
# Usage: ./infra/scripts/deploy.sh
# Prerequisites: terraform applied, images pushed to ECR, kubectl configured
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_DIR="$SCRIPT_DIR/../k8s"

# Get cluster info from Terraform
cd "$SCRIPT_DIR/../terraform"
CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || echo "medexrag")
REGION=$(terraform output -raw cluster_region 2>/dev/null || echo "us-east-1")
EFS_ID=$(terraform output -raw efs_filesystem_id 2>/dev/null || echo "<EFS_ID>")
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

echo "================================================"
echo "MedExRAG — Deploy to EKS"
echo "================================================"
echo "Cluster:  $CLUSTER_NAME"
echo "Region:   $REGION"
echo "EFS ID:   $EFS_ID"
echo "ECR Base: $ECR_BASE"
echo "================================================"

# Configure kubectl
echo ""
echo "[1/6] Configuring kubectl..."
aws eks update-kubeconfig --region "$REGION" --name "$CLUSTER_NAME"

# Replace placeholders in manifests with actual values
echo ""
echo "[2/6] Preparing manifests..."
TEMP_DIR=$(mktemp -d)
cp -r "$K8S_DIR"/* "$TEMP_DIR/"

# Replace placeholders
find "$TEMP_DIR" -name "*.yaml" -exec sed -i \
  -e "s|<ACCOUNT_ID>|${ACCOUNT_ID}|g" \
  -e "s|<REGION>|${REGION}|g" \
  -e "s|<EFS_FILESYSTEM_ID>|${EFS_ID}|g" \
  {} +

# Apply namespace
echo ""
echo "[3/6] Creating namespace..."
kubectl apply -f "$TEMP_DIR/namespace.yaml"

# Apply storage
echo ""
echo "[4/6] Setting up storage (EFS PV/PVC)..."
kubectl apply -f "$TEMP_DIR/efs-pv.yaml"

# Apply observability stack
echo ""
echo "[5/6] Deploying observability stack..."
kubectl apply -f "$TEMP_DIR/observability/"

# Apply application workloads
echo ""
echo "[6/6] Deploying application..."
kubectl apply -f "$TEMP_DIR/streamlit/"
kubectl apply -f "$TEMP_DIR/vlm-worker/"

# Wait for Streamlit to be ready
echo ""
echo "Waiting for Streamlit to be ready..."
kubectl -n medexrag rollout status deployment/streamlit --timeout=120s

# Get ingress URL
echo ""
echo "================================================"
echo "Deployment complete!"
echo "================================================"
echo ""
echo "Streamlit pods:"
kubectl -n medexrag get pods -l app=streamlit
echo ""
echo "VLM Worker pods (should be 0 until first analysis):"
kubectl -n medexrag get pods -l app=vlm-worker
echo ""
echo "Ingress:"
kubectl -n medexrag get ingress
echo ""
echo "To get the ALB URL:"
echo "  kubectl -n medexrag get ingress medexrag-ingress -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'"
echo ""

# Cleanup temp dir
rm -rf "$TEMP_DIR"
