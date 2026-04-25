# =============================================================================
# EFS — Shared Persistent Storage
# =============================================================================
# Replaces Docker named volumes for Kubernetes. Provides ReadWriteMany storage
# shared across pods for ChromaDB, HuggingFace cache, and medical literature.
# =============================================================================

resource "aws_efs_file_system" "medexrag" {
  creation_token = "${var.cluster_name}-efs"
  encrypted      = true
  throughput_mode = var.efs_throughput_mode

  tags = {
    Name = "${var.cluster_name}-efs"
  }
}

# Security group: allow NFS from EKS nodes only
resource "aws_security_group" "efs" {
  name_prefix = "${var.cluster_name}-efs-"
  description = "Allow NFS traffic from EKS nodes"
  vpc_id      = module.vpc.vpc_id

  ingress {
    description     = "NFS from EKS nodes"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }

  tags = {
    Name = "${var.cluster_name}-efs-sg"
  }
}

# Mount targets in each private subnet
resource "aws_efs_mount_target" "medexrag" {
  for_each = toset(module.vpc.private_subnets)

  file_system_id  = aws_efs_file_system.medexrag.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}
