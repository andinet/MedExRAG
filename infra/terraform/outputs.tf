# =============================================================================
# Terraform Outputs
# =============================================================================
# Values needed for kubectl, Docker push, and k8s manifest configuration
# =============================================================================

output "cluster_name" {
  description = "EKS cluster name"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster API endpoint"
  value       = module.eks.cluster_endpoint
}

output "cluster_region" {
  description = "AWS region"
  value       = var.region
}

output "ecr_repository_urls" {
  description = "ECR repository URLs for Docker push"
  value = {
    for name, repo in aws_ecr_repository.repos : name => repo.repository_url
  }
}

output "efs_filesystem_id" {
  description = "EFS filesystem ID (for k8s PersistentVolume)"
  value       = aws_efs_file_system.medexrag.id
}

output "configure_kubectl" {
  description = "Command to configure kubectl for this cluster"
  value       = "aws eks update-kubeconfig --region ${var.region} --name ${module.eks.cluster_name}"
}
