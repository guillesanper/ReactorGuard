# main.tf — Módulo VPC
# Crea la red privada de ReactorGuard sobre la que corren todos los servicios.
# La VPC es custom (no auto-subnetworks) para tener control total sobre los CIDRs.

# ---------------------------------------------------------------------------
# Red principal (custom mode = sin subnets automáticas)
# ---------------------------------------------------------------------------
resource "google_compute_network" "vpc" {
  name                    = var.vpc_name
  project                 = var.project_id
  auto_create_subnetworks = false # Creamos las subnets manualmente para controlar CIDRs

  description = "VPC privada de ReactorGuard. Sin subnets automáticas."
}

# ---------------------------------------------------------------------------
# Subnet principal para nodos GKE
# Los secondary ranges son obligatorios para GKE en modo VPC-native (alias IPs).
# - gke-pods: espacio de IPs que GKE asignará a los pods dentro de cada nodo.
# - gke-services: espacio de IPs para los ClusterIP de los Services de K8s.
# Sin estos rangos secundarios, el cluster no puede crearse en modo private.
# ---------------------------------------------------------------------------
resource "google_compute_subnetwork" "gke_nodes" {
  name          = "${var.vpc_name}-nodes"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.vpc.self_link
  ip_cidr_range = var.subnet_cidr

  # Habilita Private Google Access para que los nodos (sin IP pública) puedan
  # acceder a APIs de GCP (GCR, Artifact Registry, Secret Manager…)
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "gke-pods"
    ip_cidr_range = var.pods_cidr
  }

  secondary_ip_range {
    range_name    = "gke-services"
    ip_cidr_range = var.services_cidr
  }
}

# ---------------------------------------------------------------------------
# Cloud Router — necesario para que Cloud NAT pueda anunciar rutas de salida
# ---------------------------------------------------------------------------
resource "google_compute_router" "router" {
  name    = "${var.vpc_name}-router"
  project = var.project_id
  region  = var.region
  network = google_compute_network.vpc.self_link
}

# ---------------------------------------------------------------------------
# Cloud NAT — permite que los nodos privados hagan pull de imágenes Docker
# y accedan a internet de salida (pip install, apt-get…) sin IP pública.
# NAT_IP_ALLOCATE_OPTION_UNSPECIFIED = GCP gestiona las IPs NAT automáticamente.
# ---------------------------------------------------------------------------
resource "google_compute_router_nat" "nat" {
  name                               = "${var.vpc_name}-nat"
  project                            = var.project_id
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# ---------------------------------------------------------------------------
# Firewall: deny-all ingress (regla base de mínimo privilegio)
# Toda conexión entrante está bloqueada salvo que una regla más específica
# (con menor priority number) la permita explícitamente.
# ---------------------------------------------------------------------------
resource "google_compute_firewall" "deny_all_ingress" {
  name      = "${var.vpc_name}-deny-all-ingress"
  project   = var.project_id
  network   = google_compute_network.vpc.name
  direction = "INGRESS"
  priority  = 65534 # Prioridad más alta = última en evaluarse

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
  description   = "Bloquea todo tráfico de entrada no autorizado explícitamente."
}

# ---------------------------------------------------------------------------
# Firewall: allow-internal (tráfico entre nodos dentro de la VPC)
# Permite que GKE nodes, pods y services se comuniquen entre sí.
# Cubre el CIDR de nodos + el de pods + el de services.
# ---------------------------------------------------------------------------
resource "google_compute_firewall" "allow_internal" {
  name      = "${var.vpc_name}-allow-internal"
  project   = var.project_id
  network   = google_compute_network.vpc.name
  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
  }
  allow {
    protocol = "udp"
  }
  allow {
    protocol = "icmp"
  }

  # Permite tráfico entre los rangos internos: nodos, pods y services
  source_ranges = [
    var.subnet_cidr,
    var.pods_cidr,
    var.services_cidr,
  ]

  description = "Permite comunicación interna entre nodos, pods y services dentro de la VPC."
}

# ---------------------------------------------------------------------------
# Firewall: allow-health-checks (GCP Load Balancer health probes)
# Los Load Balancers de GCP prueban la salud de los backends desde estos CIDRs.
# Sin esta regla, el LB marcaría todos los pods como unhealthy y no enrutaría tráfico.
# Rangos oficiales: https://cloud.google.com/load-balancing/docs/health-checks#fw-rule
# ---------------------------------------------------------------------------
resource "google_compute_firewall" "allow_health_checks" {
  name      = "${var.vpc_name}-allow-health-checks"
  project   = var.project_id
  network   = google_compute_network.vpc.name
  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
  }

  source_ranges = [
    "130.211.0.0/22", # Rango de health checkers de GCP (clásico + global LB)
    "35.191.0.0/16",  # Rango adicional de health checkers (HTTP/HTTPS LB)
  ]

  description = "Permite health checks de los Google Cloud Load Balancers."
}
