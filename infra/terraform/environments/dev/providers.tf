# providers.tf
# Declara y configura los providers de GCP que usará este entorno.
#
# - google       : provider principal para recursos estables (VPC, GKE, IAM…)
# - google-beta  : provider para recursos en beta o flags experimentales
#                  (p.ej. algunas opciones avanzadas de GKE Autopilot).
#   Ambos providers deben apuntar al mismo proyecto y región para evitar
#   discrepancias entre recursos stable y beta.

provider "google" {
  project = "sentinel-platform-485714"
  region  = "europe-southwest1"
}

provider "google-beta" {
  project = "sentinel-platform-485714"
  region  = "europe-southwest1"
}

provider "tls" {}
