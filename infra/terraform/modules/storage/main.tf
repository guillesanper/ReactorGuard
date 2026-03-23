# main.tf — Módulo Storage
# Crea los cuatro buckets GCS de ReactorGuard con políticas de retención,
# versionado y lifecycle diferenciadas según el tipo de dato almacenado.
#
# Política de lifecycle común:
#   - Día 30: mover a NEARLINE (acceso infrecuente, 20% del coste de STANDARD)
#   - Día 90: mover a COLDLINE (acceso muy infrecuente, 5% del coste de STANDARD)
# Las retenciones específicas por bucket solo afectan al borrado, no al tier.

locals {
  # Prefijo de nombre incluye entorno para separar dev/staging/prod
  # en caso de que se decida usar el mismo proyecto GCP para todos.
  bucket_prefix = "reactorguard"
}

# ---------------------------------------------------------------------------
# Bucket 1: datos crudos de sensores y simulación OpenMC
# Sin versionado (los datos de sensor son inmutables — nunca se sobrescriben).
# Retención de 90 días para datos de simulación; sin retención mínima para
# datos reales de planta (pueden necesitar borrarse por regulación GDPR).
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "data_raw" {
  name          = "${local.bucket_prefix}-data-raw-${var.env}"
  project       = var.project_id
  location      = var.region
  storage_class = "STANDARD"

  # Acceso uniforme a nivel de bucket — deshabilita las ACLs por objeto.
  # Toda la gestión de acceso se hace mediante IAM (más auditable y consistente).
  uniform_bucket_level_access = true

  # Bloquea el acceso público incondicionalmente (datos de sensor son confidenciales)
  public_access_prevention = "enforced"

  versioning {
    enabled = false # Datos de sensor: inmutables, no tiene sentido versionar
  }

  # Ciclo de vida: degradar clase de almacenamiento progresivamente
  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 30 # días sin acceso → mover a NEARLINE
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age = 90 # días sin acceso → mover a COLDLINE
    }
  }

  dynamic "encryption" {
    for_each = var.cmek_key != null ? [1] : []
    content {
      default_kms_key_name = var.cmek_key
    }
  }

  labels = {
    env     = var.env
    purpose = "data-raw"
    project = "reactorguard"
  }
}

# ---------------------------------------------------------------------------
# Bucket 2: features calculadas y splits de dataset (train/val/test)
# Versionado activado: los features pueden recalcularse (DVC los gestiona).
# Retención de 30 días — los features se regeneran frecuentemente.
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "data_processed" {
  name          = "${local.bucket_prefix}-data-processed-${var.env}"
  project       = var.project_id
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true # Features son regenerables — versionar para reproducibilidad
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 30
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age = 90
    }
  }

  # Limpiar versiones antiguas para controlar coste
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age                = 30
      with_state         = "ARCHIVED" # Solo aplica a versiones no-current
    }
  }

  dynamic "encryption" {
    for_each = var.cmek_key != null ? [1] : []
    content {
      default_kms_key_name = var.cmek_key
    }
  }

  labels = {
    env     = var.env
    purpose = "data-processed"
    project = "reactorguard"
  }
}

# ---------------------------------------------------------------------------
# Bucket 3: modelos serializados (PINN, BNN, conformal predictors)
# Retención de 365 días — CRÍTICO para auditoría regulatoria nuclear.
#
# Por qué 365 días:
# La normativa nuclear (IAEA SSG-39, NRC 10 CFR 50.59) exige que cualquier
# modelo usado en sistemas de instrumentación y control pueda auditarse
# retrospectivamente. Si un modelo hace una predicción incorrecta, el
# regulador necesita acceder a la versión exacta del modelo que estaba en
# producción en ese momento.
# Versionado activado: NUNCA se sobreescribe un modelo — solo se añaden versiones.
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "models" {
  name          = "${local.bucket_prefix}-models-${var.env}"
  project       = var.project_id
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true # CRÍTICO: historial completo de versiones de modelos para auditoría
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 30
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age = 90
    }
  }

  # NO añadir regla de Delete — los modelos se conservan 365 días mínimo
  # El borrado manual requiere aprobación del equipo de seguridad nuclear

  dynamic "encryption" {
    for_each = var.cmek_key != null ? [1] : []
    content {
      default_kms_key_name = var.cmek_key
    }
  }

  labels = {
    env     = var.env
    purpose = "models"
    project = "reactorguard"
  }
}

# ---------------------------------------------------------------------------
# Bucket 4: artefactos y metadatos de experimentos MLflow
# Retención de 180 días — balance entre trazabilidad y coste de almacenamiento.
# Versionado activado: los runs de MLflow son inmutables una vez registrados.
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "mlflow" {
  name          = "${local.bucket_prefix}-mlflow-${var.env}"
  project       = var.project_id
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
    condition {
      age = 30
    }
  }

  lifecycle_rule {
    action {
      type          = "SetStorageClass"
      storage_class = "COLDLINE"
    }
    condition {
      age = 90
    }
  }

  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age        = 180
      with_state = "ARCHIVED"
    }
  }

  dynamic "encryption" {
    for_each = var.cmek_key != null ? [1] : []
    content {
      default_kms_key_name = var.cmek_key
    }
  }

  labels = {
    env     = var.env
    purpose = "mlflow"
    project = "reactorguard"
  }
}
