# variables.tf — Módulo GKE
# Parámetros del cluster y node pools. Se inyectan desde environments/dev/main.tf.

variable "project_id" {
  description = "ID del proyecto GCP."
  type        = string
}

variable "region" {
  description = "Región GCP donde se desplegará el cluster regional (multi-zona)."
  type        = string
  default     = "europe-southwest1"
}

variable "cluster_name" {
  description = "Nombre del cluster GKE."
  type        = string
  default     = "reactorguard-cluster"
}

variable "network" {
  description = "Self-link o nombre de la VPC donde vivirá el cluster."
  type        = string
}

variable "subnetwork" {
  description = "Self-link o nombre de la subnet de nodos GKE."
  type        = string
}

variable "pods_range_name" {
  description = "Nombre del secondary range reservado para pods GKE."
  type        = string
}

variable "services_range_name" {
  description = "Nombre del secondary range reservado para services GKE."
  type        = string
}

variable "master_ipv4_cidr" {
  description = "CIDR /28 para el plano de control privado del cluster."
  type        = string
  default     = "172.16.0.0/28"
}

variable "env" {
  description = "Entorno (dev, staging, prod). Se usa en labels."
  type        = string
  default     = "dev"
}

# --- Node Pool: platform ---

variable "platform_machine_type" {
  description = "Tipo de máquina para el node pool de plataforma (cargas generales)."
  type        = string
  default     = "e2-standard-4"
}

variable "platform_min_nodes" {
  description = "Número mínimo de nodos en el pool platform."
  type        = number
  default     = 1
}

variable "platform_max_nodes" {
  description = "Número máximo de nodos en el pool platform."
  type        = number
  default     = 3
}

# --- Node Pool: ml-serving ---

variable "ml_machine_type" {
  description = "Tipo de máquina para el node pool de ML serving (safety-critical)."
  type        = string
  default     = "n2-standard-8"
}

variable "ml_min_nodes" {
  description = "Número mínimo de nodos en el pool ml-serving."
  type        = number
  default     = 1
}

variable "ml_max_nodes" {
  description = "Número máximo de nodos en el pool ml-serving."
  type        = number
  default     = 5
}
