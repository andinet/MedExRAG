# MedExRAG AWS EKS Deployment

Deploy MedExRAG on AWS EKS with scale-to-zero GPU support.

## Architecture

```
ALB (HTTPS) → Streamlit (CPU node, always-on)
                   ↓ HTTP
              VLM Worker (GPU node, scale-to-zero via KEDA/Karpenter)
                   ↓
              EFS (ChromaDB + HuggingFace cache + medical literature)
```

- **Streamlit** runs on a cheap CPU node (`t3.xlarge`, ~$120/mo)
- **VLM Worker** runs on a GPU node (`g4dn.xlarge` spot, ~$0.16/hr) that scales to zero when idle
- **EFS** provides shared persistent storage across all pods
- **Full observability**: Prometheus, Grafana, Jaeger, OpenTelemetry

## Prerequisites

1. **AWS Account** with billing enabled
2. **AWS CLI** v2 ([install](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html))
3. **Terraform** >= 1.5 ([install](https://developer.hashicorp.com/terraform/install))
4. **kubectl** ([install](https://kubernetes.io/docs/tasks/tools/))
5. **Helm** >= 3.0 ([install](https://helm.sh/docs/intro/install/))
6. **Docker** for building images

## AWS Setup

### Step 1: Create an IAM User (first-time only)

1. Go to [AWS IAM Console](https://console.aws.amazon.com/iam/) → **Users** → **Create user**
2. Name: `medexrag-deployer`
3. Attach these **managed policies** (or create a custom policy with equivalent permissions):
   - `AmazonEKSClusterPolicy`
   - `AmazonEKSWorkerNodePolicy`
   - `AmazonEKS_CNI_Policy`
   - `AmazonEC2FullAccess` (for Karpenter node provisioning)
   - `AmazonElasticFileSystemFullAccess`
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonVPCFullAccess`
   - `IAMFullAccess` (for creating EKS/Karpenter roles)
   - `ElasticLoadBalancingFullAccess`

   > **Tip:** For a quick start, you can use `AdministratorAccess` and narrow permissions later.

4. Go to the user → **Security credentials** → **Create access key**
5. Choose **CLI** use case → download the Access Key ID and Secret Access Key

### Step 2: Configure AWS CLI

```bash
aws configure
# AWS Access Key ID:     <paste your access key>
# AWS Secret Access Key: <paste your secret key>
# Default region:        us-east-1
# Default output format: json
```

Verify it works:

```bash
aws sts get-caller-identity
# Should show your account ID and user ARN
```

### Step 3: (Optional) S3 Backend for Terraform State

For team use, create an S3 bucket for remote Terraform state:

```bash
aws s3 mb s3://medexrag-terraform-state --region us-east-1
aws s3api put-bucket-versioning --bucket medexrag-terraform-state \
  --versioning-configuration Status=Enabled
```

Then uncomment the `backend "s3"` block in `infra/terraform/main.tf`.

For solo use, you can skip this — Terraform will store state locally.

### Step 4: (Optional) GitHub Actions CI/CD Secrets

If using the GitHub Actions workflow (`.github/workflows/deploy-eks.yml`):

1. **Create an OIDC Identity Provider** in IAM for GitHub Actions:
   - Go to IAM → **Identity providers** → **Add provider**
   - Provider type: **OpenID Connect**
   - Provider URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`

2. **Create an IAM Role** for GitHub Actions:
   - Trusted entity: **Web identity** → select the OIDC provider above
   - Condition: `token.actions.githubusercontent.com:sub` → `repo:<your-org>/MedExRAG:*`
   - Attach the same policies listed in Step 1
   - Note the **Role ARN** (e.g., `arn:aws:iam::123456789012:role/medexrag-github-deploy`)

3. **Add GitHub Repository Secret**:
   - Go to your repo → **Settings** → **Secrets and variables** → **Actions**
   - Add secret: `AWS_DEPLOY_ROLE_ARN` = the Role ARN from above

### AWS Permissions Summary

| Component | AWS Permissions Needed |
|-----------|----------------------|
| Terraform (VPC, subnets) | `ec2:*Vpc*`, `ec2:*Subnet*`, `ec2:*RouteTable*`, `ec2:*InternetGateway*`, `ec2:*NatGateway*` |
| Terraform (EKS) | `eks:*` |
| Terraform (ECR) | `ecr:*` |
| Terraform (EFS) | `elasticfilesystem:*` |
| Terraform (IAM roles) | `iam:*Role*`, `iam:*Policy*`, `iam:*InstanceProfile*` |
| Terraform (ALB) | `elasticloadbalancing:*` |
| Karpenter (runtime) | `ec2:RunInstances`, `ec2:TerminateInstances`, `ec2:CreateFleet` |
| Docker push to ECR | `ecr:GetAuthorizationToken`, `ecr:BatchCheckLayerAvailability`, `ecr:PutImage` |

## Quick Start

### 1. Provision Infrastructure

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

terraform init
terraform plan
terraform apply
```

### 2. Build & Push Docker Images

```bash
./infra/scripts/build-and-push.sh
```

### 3. Install Karpenter + KEDA (GPU auto-scaling)

```bash
./infra/scripts/setup-karpenter.sh
```

### 4. Deploy to EKS

```bash
./infra/scripts/deploy.sh
```

### 5. Access the Application

```bash
# Get the ALB URL
kubectl -n medexrag get ingress medexrag-ingress \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
```

## Cost Estimates (us-east-1)

| Resource | Monthly Cost |
|----------|-------------|
| EKS control plane | $73 |
| CPU node (t3.xlarge, always-on) | $120 |
| GPU node (g4dn.xlarge spot, ~2h/day) | $10-15 |
| EFS (20GB) | $6 |
| ALB | $22 |
| NAT Gateway | $32 |
| **Total (light GPU use)** | **~$265/mo** |

## Scale-to-Zero Flow

1. User opens Streamlit UI (instant — CPU node always running)
2. User uploads X-ray and clicks "Analyze"
3. Streamlit calls VLM Worker via HTTP
4. KEDA sees pending request → scales VLM Worker from 0 to 1 replica
5. Karpenter provisions a `g4dn.xlarge` spot instance (~60-90s)
6. VLM Worker loads model from EFS cache (~60-90s)
7. Analysis runs (~5-10s)
8. After 10 minutes idle → KEDA scales to 0 → Karpenter terminates GPU node

**Cold start: ~2-3 minutes** (subsequent requests while warm: ~5-10s)

## Teardown (Stop All Costs)

```bash
./infra/scripts/teardown.sh
```

## File Structure

```
infra/
├── terraform/          # AWS infrastructure (VPC, EKS, ECR, EFS, IAM)
├── k8s/                # Kubernetes manifests
│   ├── streamlit/      # Web UI deployment
│   ├── vlm-worker/     # GPU inference service
│   ├── cli-worker/     # Batch processing jobs
│   ├── observability/  # Prometheus, Grafana, Jaeger, OTel
│   └── karpenter/      # GPU node auto-scaling
├── scripts/            # Deployment automation
└── README.md           # This file
```

## Local Development

The local Docker Compose workflow is unchanged:

```bash
docker compose -f docker/docker-compose.yml up streamlit
# → http://localhost:8501 (VLM runs locally)
```

The `VLM_ENDPOINT` environment variable controls routing:
- **Not set** (default): VLM loads locally in the Streamlit process
- **Set** (e.g., `http://vlm-worker:8080`): VLM calls the remote GPU worker
