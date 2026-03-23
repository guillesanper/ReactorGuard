# outputs.tf — Módulo Security
# Expone IDs de secretos, claves KMS, IP del LB y credenciales IAP.

output "storage_key_id" {
  description = <<-EOT
    ID completo de la crypto key KMS para CMEK en GCS.
    Formato: projects/PROJECT/locations/REGION/keyRings/RING/cryptoKeys/KEY
    Pasar a module.storage.cmek_key para activar encriptación con clave propia.
  EOT
  value       = google_kms_crypto_key.storage_key.id
}

output "key_ring_name" {
  description = "Nombre del KMS key ring de ReactorGuard."
  value       = google_kms_key_ring.reactorguard.name
}

output "scada_credentials_secret_id" {
  description = "ID del secreto de credenciales SCADA en Secret Manager."
  value       = google_secret_manager_secret.scada_credentials.id
}

output "jwt_secret_id" {
  description = "ID del secreto JWT en Secret Manager."
  value       = google_secret_manager_secret.jwt_secret.id
}

output "mlflow_db_url_secret_id" {
  description = "ID del secreto de la URL de base de datos MLflow."
  value       = google_secret_manager_secret.mlflow_db_url.id
}

output "gcp_api_key_secret_id" {
  description = "ID del secreto de la GCP API key."
  value       = google_secret_manager_secret.gcp_api_key.id
}

# ---------------------------------------------------------------------------
# Outputs Cloud Armor + Load Balancer + IAP (T1.8)
# ---------------------------------------------------------------------------

output "lb_ip" {
  description = <<-EOT
    IP estática del Load Balancer global de ReactorGuard.
    Usar para configurar el registro DNS: api.reactorguard.internal → esta IP.
    La IP es persistente aunque se destruya y recree el forwarding rule.
  EOT
  value = google_compute_global_address.lb_ip.address
}

output "waf_policy_id" {
  description = <<-EOT
    ID completo de la política Cloud Armor WAF.
    Formato: projects/PROJECT/global/securityPolicies/NAME
    Se puede referenciar desde otros backend services para reutilizar la misma policy.
  EOT
  value = google_compute_security_policy.reactorguard_waf.id
}

output "iap_client_id" {
  description = <<-EOT
    OAuth client_id del IAP de ReactorGuard.
    Usar en la anotación del GKE Ingress BackendConfig:
      spec.iap.oauthclientCredentials.secretName
    También necesario para configurar el consent screen de OAuth.
  EOT
  value = google_iap_client.reactorguard_iap_client.client_id
}

output "iap_client_secret" {
  description = <<-EOT
    OAuth client_secret del IAP. SENSIBLE — no mostrar en logs ni outputs de CI.
    Almacenar en Secret Manager con Load-Secrets.ps1 tras el primer apply.
    El GKE Ingress lo necesita en un Secret de Kubernetes referenciado por BackendConfig.
  EOT
  value     = google_iap_client.reactorguard_iap_client.secret
  sensitive = true
}
