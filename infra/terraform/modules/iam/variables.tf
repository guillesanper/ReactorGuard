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

# ---------------------------------------------------------------------------
# Nombres de buckets GCS (inyectados desde el módulo storage).
# Se usan para conceder IAM POR BUCKET (google_storage_bucket_iam_member) en
# lugar de a nivel de proyecto, respetando least-privilege: cada SA solo toca
# los buckets que necesita.
# ---------------------------------------------------------------------------

variable "data_raw_bucket" {
  description = "Nombre del bucket de datos crudos de sensores (module.storage.data_raw_bucket_name)."
  type        = string
}

variable "data_processed_bucket" {
  description = "Nombre del bucket de features procesadas (module.storage.data_processed_bucket_name)."
  type        = string
}

variable "models_bucket" {
  description = "Nombre del bucket de modelos serializados (module.storage.models_bucket_name)."
  type        = string
}

variable "mlflow_bucket" {
  description = "Nombre del bucket de artefactos MLflow (module.storage.mlflow_bucket_name)."
  type        = string
}
