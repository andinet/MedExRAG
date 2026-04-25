# =============================================================================
# EKS Cluster + Node Groups
# =============================================================================
# CPU node group: always-on, runs Streamlit + observability
# GPU node group: scale-to-zero via Karpenter, runs VLM worker
# =============================================================================

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version

  # Networking
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster access
  cluster_endpoint_public_access = true  # Allow kubectl from outside VPC

  # EKS Addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa.iam_role_arn
    }
    aws-efs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.efs_csi_irsa.iam_role_arn
    }
  }

  # --- Managed Node Groups ---

  eks_managed_node_groups = {
    # CPU nodes: Streamlit, observability, system pods
    cpu = {
      name           = "${var.cluster_name}-cpu"
      instance_types = [var.cpu_instance_type]
      ami_type       = "AL2_x86_64"

      min_size     = var.cpu_min_size
      max_size     = var.cpu_max_size
      desired_size = var.cpu_desired_size

      labels = {
        "node-type" = "cpu"
      }

      tags = {
        NodeGroup = "cpu"
      }
    }

    # GPU nodes: VLM inference (scale-to-zero managed by Karpenter)
    gpu = {
      name           = "${var.cluster_name}-gpu"
      instance_types = [var.gpu_instance_type]
      ami_type       = "AL2_x86_64_GPU"  # Includes NVIDIA drivers

      min_size     = 0  # Scale to zero!
      max_size     = var.gpu_max_size
      desired_size = 0  # Start with zero GPU nodes

      capacity_type = "SPOT"  # 70% cheaper than on-demand

      labels = {
        "node-type"        = "gpu"
        "nvidia.com/gpu"   = "present"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "present"
        effect = "NO_SCHEDULE"
      }]

      tags = {
        NodeGroup = "gpu"
      }
    }
  }

  tags = {
    Name = var.cluster_name
  }
}

# --- IRSA for EBS CSI Driver ---

module "ebs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-ebs-csi"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }
}

# --- IRSA for EFS CSI Driver ---

module "efs_csi_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${var.cluster_name}-efs-csi"
  attach_efs_csi_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:efs-csi-controller-sa"]
    }
  }
}

# --- AWS Load Balancer Controller (for ALB Ingress) ---

module "alb_controller_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                              = "${var.cluster_name}-alb-controller"
  attach_load_balancer_controller_policy = true

  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }
}

resource "helm_release" "aws_load_balancer_controller" {
  name       = "aws-load-balancer-controller"
  repository = "https://aws.github.io/eks-charts"
  chart      = "aws-load-balancer-controller"
  namespace  = "kube-system"
  version    = "1.7.1"

  set {
    name  = "clusterName"
    value = module.eks.cluster_name
  }

  set {
    name  = "serviceAccount.create"
    value = "true"
  }

  set {
    name  = "serviceAccount.name"
    value = "aws-load-balancer-controller"
  }

  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.alb_controller_irsa.iam_role_arn
  }

  depends_on = [module.eks]
}
