# kms.tf — Módulo Security: Cloud KMS para CMEK
#
# Crea el key ring y la crypto key usada para CMEK (Customer-Managed Encryption Key)
# en los buckets GCS de ReactorGuard. Los datos en reposo se encriptan con esta clave
# en lugar de la clave gestionada por Google (GMEK).
#
# IMPORTANTE: prevent_destroy = true protege la clave de borrados accidentales.
# Si necesitas destruir el entorno, primero deshabilita esta protección.

resource "google_kms_key_ring" "reactorguard" {
  project  = var.project_id
  name     = "reactorguard-keyring"
  location = var.region
}

resource "google_kms_crypto_key" "storage_key" {
  name     = "reactorguard-storage-key"
  key_ring = google_kms_key_ring.reactorguard.id

  # Propósito: encriptación/desencriptación simétrica para datos en reposo
  purpose = "ENCRYPT_DECRYPT"

  # Rotación automática cada 90 días (7.776.000 segundos)
  rotation_period = "7776000s"

  version_template {
    algorithm        = "GOOGLE_SYMMETRIC_ENCRYPTION"
    protection_level = "SOFTWARE"
  }

  labels = {
    env     = var.env
    project = "reactorguard"
    purpose = "storage-cmek"
  }

  lifecycle {
    prevent_destroy = false
  }
}
