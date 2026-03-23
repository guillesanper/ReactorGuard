# variables.tf — Módulo IAM
# Parámetros de entrada para la creación de Service Accounts y bindings IAM.

variable "project_id" {
  description = "ID del proyecto GCP donde se crean las Service Accounts."
  type        = string
}

variable "region" {
  description = "Región GCP (informativo, no afecta recursos IAM globales)."
  type        = string
  default     = "europe-southwest1"
}
