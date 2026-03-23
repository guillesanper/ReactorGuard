# armor.tf — Módulo Security: Cloud Armor WAF
#
# Cloud Armor actúa como escudo perimetral delante del Load Balancer global.
# Todas las peticiones externas pasan primero por Cloud Armor antes de llegar
# al cluster GKE.
#
# Orden de evaluación de reglas (menor prioridad = mayor precedencia):
#   Prioridad 1000 → Bloquea SQLi  (deny 403)
#   Prioridad 1001 → Bloquea XSS   (deny 403)
#   Prioridad 2000 → Rate limiting  (throttle → deny 429 si supera 1000 req/min)
#   Prioridad 2147483647 → Default: permite lo que no fue denegado explícitamente
#
# Por qué ALLOW en default:
#   El modelo de Cloud Armor es "deny-specific": se denegan ataques conocidos
#   y se permite el resto. Esto es menos restrictivo que un allowlist, pero
#   compatible con APIs públicas donde no conocemos el rango de IPs clientes.

resource "google_compute_security_policy" "reactorguard_waf" {
  project = var.project_id
  name    = "reactorguard-waf-policy"

  description = "WAF policy de ReactorGuard: OWASP CRS v3.3 + rate limiting + Adaptive Protection DDoS"

  # ---------------------------------------------------------------------------
  # Regla 1: SQL Injection (OWASP ModSecurity CRS v3.3)
  # ---------------------------------------------------------------------------
  # evaluatePreconfiguredExpr usa reglas pre-compiladas por Google basadas en
  # las firmas de ModSecurity. Bloquea patrones clásicos de SQLi:
  #   UNION SELECT, -- comentarios SQL, stacked queries, blind SQLi, etc.
  rule {
    description = "OWASP ModSecurity CRS v3.3: SQL Injection detection"
    priority    = 1000
    action      = "deny(403)"

    match {
      expr {
        expression = "evaluatePreconfiguredExpr('sqli-v33-stable')"
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Regla 2: Cross-Site Scripting (OWASP ModSecurity CRS v3.3)
  # ---------------------------------------------------------------------------
  # Detecta intentos de inyección de scripts en parámetros y cabeceras HTTP:
  #   <script>, onerror=, javascript:, data:text/html, etc.
  rule {
    description = "OWASP ModSecurity CRS v3.3: Cross-Site Scripting detection"
    priority    = 1001
    action      = "deny(403)"

    match {
      expr {
        expression = "evaluatePreconfiguredExpr('xss-v33-stable')"
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Regla 3: Rate Limiting por IP
  # ---------------------------------------------------------------------------
  # Limita cada IP a 1000 requests por minuto. Las peticiones que superan
  # el umbral reciben 429 (Too Many Requests) en lugar de 403, para que los
  # clientes legítimos puedan distinguir rate-limit de bloqueo por seguridad.
  rule {
    description = "Rate limiting: máximo 1000 requests/min por IP"
    priority    = 2000
    action      = "throttle"

    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }

    rate_limit_options {
      rate_limit_threshold {
        count        = 1000
        interval_sec = 60
      }
      conform_action = "allow"
      exceed_action  = "deny(429)"
      enforce_on_key = "IP"
    }
  }

  # ---------------------------------------------------------------------------
  # Regla default: permite todo lo no denegado explícitamente
  # ---------------------------------------------------------------------------
  # NOTA: Esta regla siempre debe existir y tener la prioridad más baja (2147483647).
  # Sin ella, Cloud Armor denegaría todo el tráfico por defecto.
  rule {
    description = "Default rule: allow remaining traffic"
    priority    = 2147483647
    action      = "allow"

    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Adaptive Protection: detección DDoS en capa 7 basada en ML
  # ---------------------------------------------------------------------------
  # Google analiza el tráfico en tiempo real y sugiere reglas automáticamente
  # cuando detecta patrones anómalos (volumetric DDoS, slowloris, etc.).
  # Las sugerencias aparecen en la consola GCP y deben aplicarse manualmente
  # o integrarse con alertas de Cloud Monitoring.
  adaptive_protection_config {
    layer_7_ddos_defense_config {
      enable = true
    }
  }
}
