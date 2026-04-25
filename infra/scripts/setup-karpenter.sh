#!/usr/bin/env bash
# =============================================================================
# Install Karpenter + KEDA on EKS
# =============================================================================
# Run after terraform apply and before deploying the GPU nodepool.
# Usage: ./infra/scripts/setup-karpenter.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get cluster info
cd "$SCRIPT_DIR/../terraform"
CLUSTER_NAME=$(terraform output -raw cluster_name 2>/dev/null || echo "medexrag")

echo "================================================"
echo "Installing Karpenter + KEDA on ${CLUSTER_NAME}"
echo "================================================"

# Install Karpenter
echo ""
echo "[1/3] Installing Karpenter..."
helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --version "0.35.0" \
  --namespace kube-system \
  --set "settings.clusterName=${CLUSTER_NAME}" \
  --set "settings.interruptionQueue=${CLUSTER_NAME}" \
  --wait

echo "Karpenter installed."

# Install KEDA (for scale-to-zero)
echo ""
echo "[2/3] Installing KEDA..."
helm repo add kedacore https://kedacore.github.io/charts 2>/dev/null || true
helm repo update
helm upgrade --install keda kedacore/keda \
  --namespace keda \
  --create-namespace \
  --wait

echo "KEDA installed."

# Apply GPU NodePool
echo ""
echo "[3/3] Applying GPU NodePool..."
kubectl apply -f "$SCRIPT_DIR/../k8s/karpenter/gpu-nodepool.yaml"

echo ""
echo "================================================"
echo "Karpenter + KEDA setup complete!"
echo "GPU nodes will auto-provision when GPU pods are pending."
echo "================================================"
