# variables.tf — Módulo Security
# Parámetros para Secret Manager, KMS, Cloud Armor, Load Balancer e IAP.

variable "project_id" {
  description = "ID del proyecto GCP."
  type        = string
}

variable "region" {
  description = "Región GCP para el KMS key ring. Debe coincidir con la región de los buckets GCS."
  type        = string
  default     = "europe-southwest1"
}

variable "env" {
  description = "Entorno (dev, staging, prod). Se usa como label en los secretos."
  type        = string
  default     = "dev"
}

variable "iap_support_email" {
  description = <<-EOT
    Dirección de correo que aparece en la pantalla de consentimiento OAuth de IAP.
    Debe ser una cuenta de Google válida (personal o grupo de Google Workspace)
    con acceso al proyecto GCP. Se muestra a los usuarios cuando IAP les pide
    autenticarse antes de acceder a la aplicación.
  EOT
  type    = string
  default = "admin@reactorguard.internal"
}
