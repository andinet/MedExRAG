# CI/CD Guide

GitHub Actions workflows for MedExRAG. For Actions fundamentals see the [official docs](https://docs.github.com/actions).

## Workflows

| File | Triggers | Purpose |
|------|----------|---------|
| `.github/workflows/ci.yml` | push, pull_request | Lint, unit tests, Docker compose validation |
| `.github/workflows/deploy-eks.yml` | push to `main`, manual dispatch | Terraform apply, build/push to ECR, kubectl apply |
| `.github/workflows/mlops.yml` | push, schedule (nightly), manual | DVC pipeline, evaluation, MLflow logging |

## ci.yml

Three jobs on `ubuntu-latest`:

- **lint** — `black --check`, `isort --check-only`, `flake8` against `src/medexrag/pipeline.py` and `src/medexrag/agents.py`.
- **test** — `pytest tests/ -v` with `pytest-cov`. Needs `lint`.
- **docker-build** — `docker compose -f docker/docker-compose.yml config --quiet` to validate compose syntax. Needs `lint`.

No secrets required.

## deploy-eks.yml

Provisions infra and deploys to EKS. See `infra/terraform/` and `infra/k8s/`.

Pipeline:
1. `aws-actions/configure-aws-credentials` — assumes role via OIDC (no static keys).
2. `terraform init && terraform apply` against `infra/terraform/` (creates VPC, EKS, ECR, EFS, IAM).
3. `infra/scripts/build-and-push.sh` — builds `Dockerfile.streamlit`, `Dockerfile.cli`, `Dockerfile.vlm` and pushes to ECR.
4. `aws eks update-kubeconfig` then `kubectl apply -f infra/k8s/`.
5. Optional: `infra/scripts/setup-karpenter.sh` for GPU node autoscaling.

Teardown: `infra/scripts/teardown.sh` (or run via manual dispatch).

### Required secrets

| Secret | Purpose |
|--------|---------|
| `AWS_DEPLOY_ROLE_ARN` | IAM role assumed via OIDC for the workflow |
| `AWS_REGION` | Region for EKS/ECR (e.g. `us-east-1`) |
| `AWS_ACCOUNT_ID` | For ECR registry URL |

### Setting up OIDC

In AWS:
1. Create an IAM OIDC provider for `token.actions.githubusercontent.com` (audience `sts.amazonaws.com`).
2. Create a role with a trust policy restricting `token.actions.githubusercontent.com:sub` to `repo:<owner>/<repo>:ref:refs/heads/main` (or environment-scoped).
3. Attach policies for EKS, ECR, EC2, IAM, EFS, S3 (for Terraform state) — minimum needed by `infra/terraform/`.
4. Set `AWS_DEPLOY_ROLE_ARN` repository secret to the role ARN.

The workflow uses:
```yaml
permissions:
  id-token: write
  contents: read
```

## mlops.yml

Runs evaluation and tracks metrics. See `docs/deployment/MLOPS_GUIDE.md` for details.

Jobs:
- `evaluation-tests` — `pytest tests/test_evaluation.py`
- `e2e-benchmarks` — runs `src/medexrag/evaluation/e2e_benchmarks.py` (mock pipeline)
- `retrieval-evaluation` — computes Precision@k, MRR, NDCG
- `llm-judge` — optional, gated on `OPENAI_API_KEY` secret
- `quality-gate` — fails build if metrics drop below thresholds in `tests/test_evaluation.py::TestQualityGates`
- `dvc-pipeline` — `dvc repro` on `main` only
- `generate-report` — markdown summary uploaded as artifact

### Optional secrets

| Secret | Used by |
|--------|---------|
| `OPENAI_API_KEY` | `llm-judge` job (skipped if absent) |
| `MLFLOW_TRACKING_URI` | MLflow server URL (otherwise local file backend) |
| AWS/GCS/Azure creds for DVC remote | `dvc-pipeline` job (see `.dvc/config`) |

## Running locally

```bash
# CI checks
black --check src/medexrag/
isort --check-only src/medexrag/
PYTHONPATH=src pytest tests/ -v
docker compose -f docker/docker-compose.yml config --quiet

# MLOps pipeline
PYTHONPATH=src dvc repro
```
