# outputs.tf — Módulo Storage
# Expone nombres y URLs de cada bucket para que otros módulos (IAM, ML serving)
# puedan referenciarlos sin hardcodear strings.

output "data_raw_bucket_name" {
  description = "Nombre del bucket de datos crudos."
  value       = google_storage_bucket.data_raw.name
}

output "data_raw_bucket_url" {
  description = "URL gs:// del bucket de datos crudos."
  value       = google_storage_bucket.data_raw.url
}

output "data_processed_bucket_name" {
  description = "Nombre del bucket de features procesadas."
  value       = google_storage_bucket.data_processed.name
}

output "data_processed_bucket_url" {
  description = "URL gs:// del bucket de features procesadas."
  value       = google_storage_bucket.data_processed.url
}

output "models_bucket_name" {
  description = "Nombre del bucket de modelos serializados."
  value       = google_storage_bucket.models.name
}

output "models_bucket_url" {
  description = "URL gs:// del bucket de modelos. Usado por MLflow y el servicio de serving."
  value       = google_storage_bucket.models.url
}

output "mlflow_bucket_name" {
  description = "Nombre del bucket de artefactos MLflow."
  value       = google_storage_bucket.mlflow.name
}

output "mlflow_bucket_url" {
  description = "URL gs:// del bucket de MLflow. Se pasa como MLFLOW_ARTIFACT_URI."
  value       = google_storage_bucket.mlflow.url
}
