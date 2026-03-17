output "instance_public_ip" {
  description = "Public IP of web instance"
  value       = aws_instance.web.public_ip
}

output "instance_id" {
  description = "EC2 instance id of web instance"
  value       = aws_instance.web.id
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}
