#!/usr/bin/env bash
# =============================================================================
# Teardown — Destroy ALL AWS Resources (stop all costs)
# =============================================================================
# WARNING: This destroys everything! EFS data will be lost.
# Usage: ./infra/scripts/teardown.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "   WARNING: FULL TEARDOWN"
echo "================================================"
echo "This will destroy:"
echo "  - EKS cluster and all pods"
echo "  - All node groups (CPU + GPU)"
echo "  - EFS filesystem (all stored data!)"
echo "  - ECR repositories (all images!)"
echo "  - VPC and networking"
echo "  - ALB and ingress"
echo ""
read -p "Type 'destroy' to confirm: " CONFIRM

if [ "$CONFIRM" != "destroy" ]; then
  echo "Aborted."
  exit 1
fi

echo ""
echo "[1/3] Removing Kubernetes resources..."
kubectl delete namespace medexrag --ignore-not-found 2>/dev/null || true

echo ""
echo "[2/3] Removing Helm releases..."
helm uninstall karpenter -n kube-system 2>/dev/null || true
helm uninstall keda -n keda 2>/dev/null || true
helm uninstall aws-load-balancer-controller -n kube-system 2>/dev/null || true

echo ""
echo "[3/3] Destroying Terraform infrastructure..."
cd "$SCRIPT_DIR/../terraform"
terraform destroy -auto-approve

echo ""
echo "================================================"
echo "Teardown complete. All AWS resources destroyed."
echo "================================================"
