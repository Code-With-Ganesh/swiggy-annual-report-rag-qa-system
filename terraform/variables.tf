variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name prefix for resources"
  type        = string
  default     = "swiggy-chatbot"
}

variable "allowed_ssh_cidr" {
  description = "Restricted CIDR allowed to SSH"
  type        = string
  default     = "203.0.113.10/32"

  validation {
    condition     = var.allowed_ssh_cidr != "0.0.0.0/0"
    error_message = "Do not expose SSH to the internet. Use a specific trusted CIDR."
  }
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.micro"
}
