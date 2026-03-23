# variables.tf — Módulo Storage
# Define los parámetros comunes de todos los buckets GCS.

variable "project_id" {
  description = "ID del proyecto GCP."
  type        = string
}

variable "region" {
  description = "Región GCP donde se crearán los buckets. Debe coincidir con el cluster GKE."
  type        = string
  default     = "europe-southwest1"
}

variable "env" {
  description = "Entorno (dev, staging, prod). Se añade como sufijo a los nombres de buckets."
  type        = string
  default     = "dev"
}

variable "cmek_key" {
  description = <<-EOT
    URI completo de la clave KMS para CMEK (Customer-Managed Encryption Key).
    Si es null, se usará GMEK (Google-Managed Encryption Key).
    Formato: projects/PROJECT/locations/REGION/keyRings/RING/cryptoKeys/KEY
  EOT
  type        = string
  default     = null
}
