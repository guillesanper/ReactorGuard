# variables.tf — Módulo VPC
# Define todos los parámetros configurables del módulo.
# Los valores concretos se pasan desde environments/dev/main.tf.

variable "project_id" {
  description = "ID del proyecto GCP donde se crearán los recursos."
  type        = string
}

variable "region" {
  description = "Región GCP para la subnet y Cloud NAT."
  type        = string
  default     = "europe-southwest1"
}

variable "vpc_name" {
  description = "Nombre de la VPC. Se usará también como prefijo para subnets y router."
  type        = string
  default     = "reactorguard-vpc"
}

variable "subnet_cidr" {
  description = "CIDR principal de la subnet de nodos GKE."
  type        = string
  default     = "10.0.1.0/24"
}

variable "pods_cidr" {
  description = "CIDR del secondary range para pods GKE."
  type        = string
  default     = "10.0.16.0/20"
}

variable "services_cidr" {
  description = "CIDR del secondary range para services GKE."
  type        = string
  default     = "10.0.32.0/20"
}

variable "env" {
  description = "Nombre del entorno (dev, staging, prod). Se usa en labels."
  type        = string
  default     = "dev"
}
