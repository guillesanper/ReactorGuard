# secrets.tf — Módulo Security: Secret Manager
#
# ===========================================================================
# ⚠️  IMPORTANTE — VALORES PLACEHOLDER
# ===========================================================================
# Los secret_data definidos aquí son PLACEHOLDERS. No contienen credenciales reales.
#
# Antes de desplegar en cualquier entorno, ejecutar:
#
#   .\infra\scripts\Load-Secrets.ps1 `
#       -ScadaUsername  "usuario_real" `
#       -ScadaPassword  "password_real" `
#       -ScadaEndpoint  "https://scada.ejemplo.com" `
#       -JwtSecret      "cadena_aleatoria_32_bytes" `
#       -MlflowDbUrl    "postgresql://user:pass@host/db" `
#       -GcpApiKey      "AIza..."
#
# El script cargará las versiones reales sobrescribiendo los placeholders.
# NUNCA commitees credenciales reales en este repositorio.
# ===========================================================================

# ---------------------------------------------------------------------------
# Secreto 1: Credenciales SCADA
# Contiene endpoint, usuario y contraseña del sistema SCADA de la planta.
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "scada_credentials" {
  project   = var.project_id
  secret_id = "reactorguard-scada-credentials"

  replication {
    auto {}
  }

  labels = {
    env     = var.env
    service = "ingestion"
    project = "reactorguard"
  }
}

resource "google_secret_manager_secret_version" "scada_credentials_placeholder" {
  secret = google_secret_manager_secret.scada_credentials.id
  secret_data = jsonencode({
    username = "REPLACE_ME"
    password = "REPLACE_ME"
    endpoint = "REPLACE_ME"
  })

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# ---------------------------------------------------------------------------
# Secreto 2: JWT Secret
# Clave de firma para tokens JWT de la API REST. Mínimo 32 bytes aleatorios.
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "jwt_secret" {
  project   = var.project_id
  secret_id = "reactorguard-jwt-secret"

  replication {
    auto {}
  }

  labels = {
    env     = var.env
    service = "api"
    project = "reactorguard"
  }
}

resource "google_secret_manager_secret_version" "jwt_secret_placeholder" {
  secret      = google_secret_manager_secret.jwt_secret.id
  secret_data = "REPLACE_WITH_32_BYTE_RANDOM_STRING"

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# ---------------------------------------------------------------------------
# Secreto 3: MLflow DB URL
# Cadena de conexión PostgreSQL para el backend de MLflow tracking.
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "mlflow_db_url" {
  project   = var.project_id
  secret_id = "reactorguard-mlflow-db-url"

  replication {
    auto {}
  }

  labels = {
    env     = var.env
    service = "mlflow"
    project = "reactorguard"
  }
}

resource "google_secret_manager_secret_version" "mlflow_db_url_placeholder" {
  secret      = google_secret_manager_secret.mlflow_db_url.id
  secret_data = "postgresql://REPLACE_ME"

  lifecycle {
    ignore_changes = [secret_data]
  }
}

# ---------------------------------------------------------------------------
# Secreto 4: GCP API Key
# API Key para servicios GCP que no soportan Workload Identity.
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "gcp_api_key" {
  project   = var.project_id
  secret_id = "reactorguard-gcp-api-key"

  replication {
    auto {}
  }

  labels = {
    env     = var.env
    service = "platform"
    project = "reactorguard"
  }
}

resource "google_secret_manager_secret_version" "gcp_api_key_placeholder" {
  secret      = google_secret_manager_secret.gcp_api_key.id
  secret_data = "REPLACE_ME"

  lifecycle {
    ignore_changes = [secret_data]
  }
}
