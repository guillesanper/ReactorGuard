# iap.tf — Módulo Security: Identity-Aware Proxy (IAP)
#
# IAP añade autenticación OAuth 2.0 obligatoria delante de las aplicaciones GCP.
# En lugar de exponer servicios directamente a internet, IAP:
#   1. Intercepta cada petición HTTPS (ya desencriptada por el LB)
#   2. Verifica la identidad del usuario con Google OAuth 2.0
#   3. Comprueba que el usuario tiene el rol IAP-secured Web App User en el proyecto
#   4. Sólo entonces reenvía la petición al backend (GKE Ingress)
#
# Por qué IAP se sitúa DESPUÉS del Load Balancer:
#   IAP opera sobre la conexión HTTP ya desencriptada. El LB termina TLS primero,
#   exponiendo la petición en claro, y entonces IAP puede leer las cabeceras
#   de autorización OAuth 2.0 e inspeccionar el token de identidad.
#   Si IAP estuviese antes del LB, vería tráfico HTTPS opaco y no podría
#   extraer ni validar el token.
#
# Configuración necesaria en el GKE Ingress (después del apply):
#   annotations:
#     kubernetes.io/ingress.allow-http: "false"
#     beta.cloud.google.com/backend-config: '{"default": "reactorguard-backend-config"}'
#   El BackendConfig debe referenciar el client_id del output iap_client_id.

# ---------------------------------------------------------------------------
# IAP Brand (marca OAuth)
# ---------------------------------------------------------------------------
# Representa la "marca" de la aplicación en la pantalla de consentimiento OAuth.
# Sólo puede existir UNA brand por proyecto GCP.
# Si el proyecto ya tiene una brand (creada manualmente o por otra herramienta),
# hay que importarla: terraform import google_iap_brand.reactorguard_brand <brand_name>
resource "google_iap_brand" "reactorguard_brand" {
  project = var.project_id

  application_title = "ReactorGuard Platform"

  # Dirección de correo que aparece en la pantalla de consentimiento OAuth
  # y que los usuarios pueden contactar con dudas sobre los permisos solicitados.
  support_email = var.iap_support_email
}

# ---------------------------------------------------------------------------
# IAP OAuth Client
# ---------------------------------------------------------------------------
# Genera el par (client_id, client_secret) que el backend service usará
# para validar los tokens OAuth de IAP.
# Los valores se exponen como outputs sensibles para:
#   - client_id  → anotación del GKE Ingress / BackendConfig
#   - secret     → almacenarlo en Secret Manager (script Load-Secrets.ps1)
resource "google_iap_client" "reactorguard_iap_client" {
  display_name = "ReactorGuard IAP Client"
  brand        = google_iap_brand.reactorguard_brand.name
}
