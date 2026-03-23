# workload_identity.tf — Módulo IAM
# ---------------------------------------------------------------------------
# CÓMO FUNCIONA WORKLOAD IDENTITY
# ---------------------------------------------------------------------------
#
#   Workload Identity elimina la necesidad de montar archivos JSON con
#   credenciales dentro de los pods. El flujo es:
#
#   1. Pod arranca usando una Kubernetes Service Account (KSA)
#   2. GKE intercepta las llamadas a la metadata API del pod
#   3. El binding IAM de este archivo autoriza a la KSA a impersonar la GSA
#   4. GKE emite credenciales temporales de la Google Service Account (GSA)
#   5. El pod llama a la GCP API con esas credenciales temporales (rotadas cada hora)
#
#   Flujo completo: Pod → KSA → Workload Identity Pool → GSA → GCP API
#
#   El formato del member es:
#     serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/KSA_NAME]
#
#   Requisito en Kubernetes: la KSA debe tener la anotación:
#     iam.gke.io/gcp-service-account: GSA_EMAIL
#   (Se añade en T2.2 cuando se crean los manifiestos K8s)
# ---------------------------------------------------------------------------

# reactorguard-ingestion-sa ↔ namespace reactorguard-ingestion / KSA sensor-validator
resource "google_service_account_iam_member" "ingestion_workload_identity" {
  service_account_id = google_service_account.ingestion.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[reactorguard-ingestion/sensor-validator]"
}

# reactorguard-ml-sa ↔ namespace reactorguard-ml / KSA pinn-server
resource "google_service_account_iam_member" "ml_workload_identity" {
  service_account_id = google_service_account.ml.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[reactorguard-ml/pinn-server]"
}

# reactorguard-mlflow-sa ↔ namespace reactorguard-ml / KSA mlflow-server
resource "google_service_account_iam_member" "mlflow_workload_identity" {
  service_account_id = google_service_account.mlflow.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[reactorguard-ml/mlflow-server]"
}

# NOTA: reactorguard-cicd-sa NO tiene Workload Identity.
# Corre en GitHub Actions (fuera del cluster GKE). Usa una clave JSON
# almacenada en GitHub Secrets → GCP_SA_KEY. Rotar cada 90 días.
