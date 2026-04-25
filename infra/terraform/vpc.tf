# =============================================================================
# VPC — Networking for EKS Cluster
# =============================================================================
# 2 public subnets (ALB) + 2 private subnets (EKS nodes) across 2 AZs
# NAT Gateway for private subnet internet access (model downloads, ECR pulls)
# =============================================================================

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = local.azs
  public_subnets  = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 8, k)]
  private_subnets = [for k, v in local.azs : cidrsubnet(var.vpc_cidr, 8, k + 10)]

  enable_nat_gateway   = true
  single_nat_gateway   = true  # Cost optimization: 1 NAT GW instead of per-AZ
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Tags required for EKS auto-discovery
  public_subnet_tags = {
    "kubernetes.io/role/elb"                      = 1
    "kubernetes.io/cluster/${var.cluster_name}"    = "owned"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"              = 1
    "kubernetes.io/cluster/${var.cluster_name}"    = "owned"
    "karpenter.sh/discovery"                       = var.cluster_name
  }

  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}
