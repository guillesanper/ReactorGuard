# versions.tf
# Fija las versiones mínimas de Terraform y de cada provider.
# Pinear versiones evita que una actualización automática del registry
# rompa el plan sin previo aviso.
#
# Convención de constraints:
#   ~> 5.0  →  >= 5.0, < 6.0  (acepta patches y minors, bloquea major)

terraform {
  # Versión mínima del binario de Terraform.
  # 1.6.0 introdujo el bloque `import` declarativo y mejoras en tests.
  required_version = ">= 1.6.0"

  required_providers {
    # Provider principal de GCP (recursos GA)
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }

    # Provider beta de GCP (recursos en preview/beta)
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }

    # Provider de Helm para desplegar charts en GKE
    # (p.ej. cert-manager, ingress-nginx, Kafka Operator)
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }

    # Provider de Kubernetes para gestionar recursos del clúster
    # directamente desde Terraform (namespaces, secrets, config maps…)
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }
}
