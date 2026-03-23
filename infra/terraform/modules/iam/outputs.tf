# outputs.tf — Módulo IAM
# Expone los emails de las Service Accounts para que otros módulos
# (storage, security) puedan crear bindings adicionales sin hardcodear emails.

output "ingestion_sa_email" {
  description = "Email de la SA del microservicio de ingestión de sensores."
  value       = google_service_account.ingestion.email
}

output "ingestion_sa_name" {
  description = "Nombre completo (resource ID) de la SA de ingestión."
  value       = google_service_account.ingestion.name
}

output "ml_sa_email" {
  description = "Email de la SA del servidor PINN y microservicios ML."
  value       = google_service_account.ml.email
}

output "ml_sa_name" {
  description = "Nombre completo (resource ID) de la SA de ML."
  value       = google_service_account.ml.name
}

output "mlflow_sa_email" {
  description = "Email de la SA de MLflow tracking server."
  value       = google_service_account.mlflow.email
}

output "mlflow_sa_name" {
  description = "Nombre completo (resource ID) de la SA de MLflow."
  value       = google_service_account.mlflow.name
}

output "cicd_sa_email" {
  description = "Email de la SA de CI/CD (GitHub Actions). Usar para generar clave JSON."
  value       = google_service_account.cicd.email
}
