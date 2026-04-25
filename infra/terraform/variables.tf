# =============================================================================
# Terraform Variables for MedExRAG EKS Deployment
# =============================================================================

variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "medexrag"
}

variable "cluster_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.29"
}

variable "environment" {
  description = "Environment name (e.g., production, staging)"
  type        = string
  default     = "production"
}

# --- Networking ---

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# --- Node Groups ---

variable "cpu_instance_type" {
  description = "EC2 instance type for CPU node group"
  type        = string
  default     = "t3.xlarge"  # 4 vCPU, 16GB RAM — $120/mo
}

variable "cpu_min_size" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "cpu_max_size" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 3
}

variable "cpu_desired_size" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 1
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU node group"
  type        = string
  default     = "g4dn.xlarge"  # T4 16GB GPU, 4 vCPU, 16GB RAM — $0.53/hr
}

variable "gpu_max_size" {
  description = "Maximum number of GPU nodes (min is always 0 for scale-to-zero)"
  type        = number
  default     = 1
}

# --- Storage ---

variable "efs_throughput_mode" {
  description = "EFS throughput mode (bursting or provisioned)"
  type        = string
  default     = "bursting"
}

# --- Tags ---

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "MedExRAG"
    ManagedBy   = "Terraform"
  }
}
