# outputs.tf — Módulo GKE
# Expone los datos del cluster que necesitan otros módulos (IAM, Kafka, K8s providers).

output "cluster_name" {
  description = "Nombre del cluster GKE."
  value       = google_container_cluster.main.name
}

output "cluster_endpoint" {
  description = "Endpoint HTTPS del API server de Kubernetes. Usado por los providers helm y kubernetes."
  value       = google_container_cluster.main.endpoint
  sensitive   = true # Contiene IP del master — no exponer en logs de CI/CD
}

output "cluster_ca_certificate" {
  description = "Certificado CA del cluster (base64). Requerido para autenticar contra el API server."
  value       = google_container_cluster.main.master_auth[0].cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "Región o zona del cluster."
  value       = google_container_cluster.main.location
}
