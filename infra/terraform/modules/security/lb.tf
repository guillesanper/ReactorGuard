# lb.tf — Módulo Security: HTTPS Load Balancer global
#
# Arquitectura del flujo de tráfico:
#
#   [Cliente]
#      │
#      ├── puerto 80 (HTTP)
#      │   └── google_compute_global_forwarding_rule (http)
#      │        └── google_compute_target_http_proxy
#      │             └── google_compute_url_map (http_redirect)  ←─ devuelve 301
#      │                  └── [Redirige a HTTPS — sin backend real]
#      │
#      └── puerto 443 (HTTPS)
#           └── google_compute_global_forwarding_rule (https)
#                └── google_compute_target_https_proxy  ←─ termina TLS con SSL cert gestionado
#                     └── google_compute_url_map (main)
#                          └── google_compute_backend_service
#                               ├── Cloud Armor WAF  ←─ filtra ataques OWASP
#                               ├── IAP              ←─ verifica identidad OAuth
#                               └── [NEG del GKE Ingress — se añade al desplegar]
#
# Por qué IAP se sitúa DESPUÉS del LB:
#   IAP opera sobre conexiones HTTP ya desencriptadas. El LB termina TLS primero,
#   exponiendo la petición HTTP en claro, y entonces IAP puede inspeccionar
#   las cabeceras de autenticación OAuth y validar la identidad del usuario.
#   Si IAP estuviese antes del LB, sólo vería tráfico HTTPS cifrado y no podría
#   extraer el token de identidad.

# ---------------------------------------------------------------------------
# IP estática global
# ---------------------------------------------------------------------------
# Una IP estática garantiza que el DNS no necesita actualizarse si se recrea el LB.
resource "google_compute_global_address" "lb_ip" {
  project = var.project_id
  name    = "reactorguard-lb-ip"

  description  = "IP estática del Load Balancer global de ReactorGuard"
  address_type = "EXTERNAL"
}

# ---------------------------------------------------------------------------
# Certificado SSL gestionado por Google
# ---------------------------------------------------------------------------
# Google aprovisiona y renueva automáticamente el certificado TLS.
# El dominio debe resolver a la IP del forwarding rule para que la validación
# DV (Domain Validation) funcione. En un entorno interno se usaría un cert
# auto-gestionado; aquí usamos Google-managed para simplificar operaciones.
# Dev: self-signed cert (Google-managed certs require a publicly resolvable domain)
resource "google_compute_ssl_certificate" "reactorguard_cert" {
  project = var.project_id
  name    = "reactorguard-ssl-cert"

  # Self-signed certificate for dev — replace with a real cert in staging/prod
  private_key = tls_private_key.reactorguard_dev.private_key_pem
  certificate = tls_self_signed_cert.reactorguard_dev.cert_pem

  lifecycle {
    create_before_destroy = true
  }
}

resource "tls_private_key" "reactorguard_dev" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "tls_self_signed_cert" "reactorguard_dev" {
  private_key_pem = tls_private_key.reactorguard_dev.private_key_pem

  subject {
    common_name  = "reactorguard.dev"
    organization = "ReactorGuard Dev"
  }

  validity_period_hours = 8760 # 1 year

  allowed_uses = [
    "key_encipherment",
    "digital_signature",
    "server_auth",
  ]
}

# ---------------------------------------------------------------------------
# Health Check HTTP (requerido por el backend service)
# ---------------------------------------------------------------------------
# GCP requiere un health check para poder asociar backends al backend service.
# El path /health es el endpoint de liveness que expone la API FastAPI del
# proyecto (router montado en /health, ver api/main.py y api/routers/health.py).
resource "google_compute_health_check" "reactorguard_http" {
  project = var.project_id
  name    = "reactorguard-http-health-check"

  description = "Health check HTTP para el backend service de ReactorGuard"

  check_interval_sec  = 10
  timeout_sec         = 5
  healthy_threshold   = 2
  unhealthy_threshold = 3

  http_health_check {
    port         = 8000
    request_path = "/health"
  }
}

# ---------------------------------------------------------------------------
# Backend Service
# ---------------------------------------------------------------------------
# Punto de integración central: conecta Cloud Armor WAF e IAP con el tráfico
# que llega al cluster GKE. Los backends reales (NEG del Ingress) se añaden
# en el módulo de despliegue de la aplicación (semana 2+).
resource "google_compute_backend_service" "reactorguard_backend" {
  project = var.project_id
  name    = "reactorguard-backend"

  description = "Backend service de ReactorGuard: Cloud Armor WAF + IAP habilitado"

  protocol    = "HTTP"
  port_name   = "http"
  timeout_sec = 30

  health_checks = [google_compute_health_check.reactorguard_http.id]

  # Cloud Armor WAF: todas las peticiones son filtradas por la policy OWASP
  # antes de llegar al backend. Las reglas están definidas en armor.tf.
  security_policy = google_compute_security_policy.reactorguard_waf.id

  # IAP: intercepta cada petición HTTPS post-TLS y exige autenticación Google
  # OAuth 2.0 antes de reenviarla al GKE Ingress.
  # Las credenciales (client_id + secret) se generan en iap.tf.
  iap {
    oauth2_client_id     = google_iap_client.reactorguard_iap_client.client_id
    oauth2_client_secret = google_iap_client.reactorguard_iap_client.secret
  }

  # Logging al 100% en dev para facilitar el debugging.
  # En prod se reduciría a 0.1–0.5 para controlar costes.
  log_config {
    enable      = true
    sample_rate = 1.0
  }
}

# ---------------------------------------------------------------------------
# URL Map principal (HTTPS → backend service)
# ---------------------------------------------------------------------------
resource "google_compute_url_map" "reactorguard_https" {
  project = var.project_id
  name    = "reactorguard-url-map"

  description     = "URL map principal: enruta todo el tráfico HTTPS al backend service"
  default_service = google_compute_backend_service.reactorguard_backend.id
}

# ---------------------------------------------------------------------------
# Target HTTPS Proxy (termina TLS)
# ---------------------------------------------------------------------------
resource "google_compute_target_https_proxy" "reactorguard_https_proxy" {
  project = var.project_id
  name    = "reactorguard-https-proxy"

  url_map          = google_compute_url_map.reactorguard_https.id
  ssl_certificates = [google_compute_ssl_certificate.reactorguard_cert.id]
}

# ---------------------------------------------------------------------------
# Forwarding Rule HTTPS (puerto 443)
# ---------------------------------------------------------------------------
resource "google_compute_global_forwarding_rule" "reactorguard_https" {
  project = var.project_id
  name    = "reactorguard-https-forwarding-rule"

  target     = google_compute_target_https_proxy.reactorguard_https_proxy.id
  ip_address = google_compute_global_address.lb_ip.address
  port_range = "443"
}

# ===========================================================================
# Redirección HTTP → HTTPS
# ===========================================================================
# Las peticiones al puerto 80 se redirigen con 301 Moved Permanently al puerto 443.
# Esto garantiza que todo el tráfico usa HTTPS sin necesidad de que los clientes
# recuerden escribir https://.
# IMPORTANTE: Esta URL map NO tiene backend — sólo emite una redirección 301.
# Por eso no expone datos y no requiere Cloud Armor ni IAP.

resource "google_compute_url_map" "reactorguard_http_redirect" {
  project = var.project_id
  name    = "reactorguard-http-redirect"

  description = "Redirige todo el tráfico HTTP a HTTPS (301 Moved Permanently)"

  default_url_redirect {
    https_redirect         = true
    redirect_response_code = "MOVED_PERMANENTLY_DEFAULT"
    strip_query            = false
  }
}

resource "google_compute_target_http_proxy" "reactorguard_http_proxy" {
  project = var.project_id
  name    = "reactorguard-http-proxy"

  url_map = google_compute_url_map.reactorguard_http_redirect.id
}

resource "google_compute_global_forwarding_rule" "reactorguard_http" {
  project = var.project_id
  name    = "reactorguard-http-forwarding-rule"

  target     = google_compute_target_http_proxy.reactorguard_http_proxy.id
  ip_address = google_compute_global_address.lb_ip.address
  port_range = "80"
}
