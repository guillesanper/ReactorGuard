# outputs.tf — Módulo VPC
# Expone los identificadores que otros módulos (GKE, IAM…) necesitan referenciar.

output "vpc_name" {
  description = "Nombre de la VPC."
  value       = google_compute_network.vpc.name
}

output "vpc_self_link" {
  description = "Self-link de la VPC (URI completo). Usado por recursos que requieren referencia directa."
  value       = google_compute_network.vpc.self_link
}

output "subnet_name" {
  description = "Nombre de la subnet de nodos GKE."
  value       = google_compute_subnetwork.gke_nodes.name
}

output "subnet_self_link" {
  description = "Self-link de la subnet de nodos GKE. Requerido por google_container_cluster."
  value       = google_compute_subnetwork.gke_nodes.self_link
}

output "pods_range_name" {
  description = "Nombre del secondary range para pods GKE. Requerido en ip_allocation_policy del cluster."
  value       = "gke-pods"
}

output "services_range_name" {
  description = "Nombre del secondary range para services GKE. Requerido en ip_allocation_policy del cluster."
  value       = "gke-services"
}
