# main.tf — Módulo IAM: Service Accounts con Least Privilege
# ---------------------------------------------------------------------------
# Cada microservicio tiene su propia Google Service Account (GSA) con
# exactamente los permisos que necesita. No se comparten SAs entre servicios.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SERVICE ACCOUNTS
# ---------------------------------------------------------------------------

resource "google_service_account" "ingestion" {
  project      = var.project_id
  account_id   = "reactorguard-ingestion-sa"
  display_name = "ReactorGuard Ingestion SA"
  description  = "Ingestión de datos de sensores — lee raw, escribe raw+processed"
}

resource "google_service_account" "ml" {
  project      = var.project_id
  account_id   = "reactorguard-ml-sa"
  display_name = "ReactorGuard ML SA"
  description  = "PINN server — lee modelos y processed, accede a Secret Manager"
}

resource "google_service_account" "mlflow" {
  project      = var.project_id
  account_id   = "reactorguard-mlflow-sa"
  display_name = "ReactorGuard MLflow SA"
  description  = "MLflow tracking server — admin en buckets mlflow y models"
}

resource "google_service_account" "cicd" {
  project      = var.project_id
  account_id   = "reactorguard-cicd-sa"
  display_name = "ReactorGuard CI/CD SA"
  description  = "GitHub Actions — push a Artifact Registry, deploy en GKE"
}

# ---------------------------------------------------------------------------
# IAM BINDINGS — reactorguard-ingestion-sa
# Lee datos crudos, escribe en raw y processed, publica en Pub/Sub
# ---------------------------------------------------------------------------

resource "google_project_iam_member" "ingestion_gcs_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_project_iam_member" "ingestion_gcs_creator" {
  project = var.project_id
  role    = "roles/storage.objectCreator"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

resource "google_project_iam_member" "ingestion_pubsub_publisher" {
  project = var.project_id
  role    = "roles/pubsub.publisher"
  member  = "serviceAccount:${google_service_account.ingestion.email}"
}

# ---------------------------------------------------------------------------
# IAM BINDINGS — reactorguard-ml-sa
# Lee modelos y features, accede a secrets, escribe métricas de monitoring
# ---------------------------------------------------------------------------

resource "google_project_iam_member" "ml_gcs_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.ml.email}"
}

resource "google_project_iam_member" "ml_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.ml.email}"
}

resource "google_project_iam_member" "ml_metric_writer" {
  project = var.project_id
  role    = "roles/monitoring.metricWriter"
  member  = "serviceAccount:${google_service_account.ml.email}"
}

# ---------------------------------------------------------------------------
# IAM BINDINGS — reactorguard-mlflow-sa
# Admin completo en buckets de artefactos MLflow y modelos serializados
# ---------------------------------------------------------------------------

resource "google_project_iam_member" "mlflow_gcs_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.mlflow.email}"
}

# ---------------------------------------------------------------------------
# IAM BINDINGS — reactorguard-cicd-sa
# Pipeline CI/CD: push de imágenes, despliegue en GKE, lectura/escritura GCS
# ---------------------------------------------------------------------------

resource "google_project_iam_member" "cicd_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

resource "google_project_iam_member" "cicd_gcs_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}

resource "google_project_iam_member" "cicd_gke_developer" {
  project = var.project_id
  role    = "roles/container.developer"
  member  = "serviceAccount:${google_service_account.cicd.email}"
}
