# main.tf — Módulo GKE
# Crea el cluster GKE Standard privado y sus dos node pools.
#
# GKE Standard (NO Autopilot) es obligatorio porque:
# - Strimzi (Kafka operator) requiere permisos de cluster-admin y admission webhooks.
# - Autopilot restringe DaemonSets y capacidades de seguridad que Strimzi necesita.

# ---------------------------------------------------------------------------
# Cluster GKE regional (alta disponibilidad multi-zona)
# ---------------------------------------------------------------------------
resource "google_container_cluster" "main" {
  name     = var.cluster_name
  project  = var.project_id
  location = var.region # Regional = réplica del control plane en 3 zonas

  # Eliminar el node pool default que GKE crea automáticamente.
  # Usamos node pools separados con configuración explícita.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = var.network
  subnetwork = var.subnetwork

  # --- Cluster privado ---
  # Los nodos no tienen IPs públicas; solo pueden comunicarse con el master
  # y con internet a través de Cloud NAT.
  private_cluster_config {
    enable_private_nodes    = true  # CRÍTICO: nodos sin IP pública
    enable_private_endpoint = false # El master endpoint es accesible desde la VPC (y opcionalmente desde authorized_networks)
    master_ipv4_cidr_block  = var.master_ipv4_cidr
  }

  # --- VPC-native networking (alias IPs) ---
  # Requerido para usar secondary ranges de la subnet como IPs de pods/services.
  ip_allocation_policy {
    cluster_secondary_range_name  = var.pods_range_name
    services_secondary_range_name = var.services_range_name
  }

  # --- Canal de versiones ---
  # REGULAR = versiones estables con ~2 semanas de lag respecto a Rapid.
  release_channel {
    channel = "REGULAR"
  }

  # --- Workload Identity ---
  # Permite que las Service Accounts de K8s se autentiquen en GCP
  # sin necesidad de montar credenciales JSON en los pods.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # --- Binary Authorization ---
  # Solo permite desplegar imágenes firmadas por nuestra cadena de CI/CD.
  # Configuración de las policies se hace en el módulo security.
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # --- Network Policy (Calico) ---
  # Habilita NetworkPolicy de Kubernetes para aislar namespaces y pods.
  # Sin esto, todos los pods pueden comunicarse entre sí libremente.
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # --- Google Cloud Managed Prometheus ---
  # Recolección de métricas gestionada sin necesitar un Prometheus propio.
  monitoring_config {
    managed_prometheus {
      enabled = true
    }
  }

  # --- Addons ---
  addons_config {
    # GcePersistentDiskCsiDriver: necesario para PersistentVolumes en GKE moderno
    gce_persistent_disk_csi_driver_config {
      enabled = true
    }
    # HorizontalPodAutoscaling: necesario para que el cluster autoscaler funcione
    horizontal_pod_autoscaling {
      disabled = false
    }
  }

  # --- Logging y Monitoring ---
  logging_service    = "logging.googleapis.com/kubernetes"
  monitoring_service = "monitoring.googleapis.com/kubernetes"

  resource_labels = {
    env     = var.env
    project = "reactorguard"
  }
}

# ---------------------------------------------------------------------------
# Node Pool 1: platform
# Para cargas generales: API, ingestion, Kafka brokers, observabilidad.
# Preemptible = SÍ → ahorro de ~80% de coste en dev; en prod considerar standard.
# ---------------------------------------------------------------------------
resource "google_container_node_pool" "platform" {
  name     = "platform"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.main.name

  autoscaling {
    min_node_count = var.platform_min_nodes
    max_node_count = var.platform_max_nodes
  }

  node_config {
    machine_type = var.platform_machine_type
    preemptible  = true # Preemptible para reducir coste en cargas tolerantes a interrupciones

    # Workload Identity en el node pool
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Shielded nodes: arranque verificado + vTPM + Integrity Monitoring
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    labels = {
      node-pool = "platform"
      env       = var.env
    }

    # Sin taints: cualquier pod puede schedularse aquí salvo que tenga toleraciones específicas
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

# ---------------------------------------------------------------------------
# Node Pool 2: ml-serving
# Exclusivo para inferencia de modelos PINN/BNN/conformal (safety-critical).
#
# CRÍTICO — por qué NO puede ser preemptible:
# En producción, este pool sirve predicciones de anomalías en tiempo real.
# Un nodo preemptible puede ser eliminado por GCP con solo 30 segundos de aviso.
# Si el nodo es interrumpido mientras procesa una alerta de fallo de reactor,
# el sistema podría perder la predicción y no emitir la alarma a tiempo.
# Para cargas safety-critical, la disponibilidad supera el ahorro de coste.
#
# El taint ml-workload=true:NoSchedule hace que SOLO los pods con la
# toleración correspondiente se schedeulen aquí. Esto evita que pods de
# plataforma consuman recursos de los nodos de ML serving.
# ---------------------------------------------------------------------------
resource "google_container_node_pool" "ml_serving" {
  name     = "ml-serving"
  project  = var.project_id
  location = var.region
  cluster  = google_container_cluster.main.name

  autoscaling {
    min_node_count = var.ml_min_nodes
    max_node_count = var.ml_max_nodes
  }

  node_config {
    machine_type = var.ml_machine_type
    preemptible  = false # NO preemptible — safety-critical serving

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }

    labels = {
      node-pool = "ml-serving"
      env       = var.env
    }

    # Taint que impide que pods sin la toleración explícita lleguen a este pool.
    # Los pods de ML deben añadir:
    #   tolerations:
    #     - key: "ml-workload"
    #       operator: "Equal"
    #       value: "true"
    #       effect: "NoSchedule"
    taint {
      key    = "ml-workload"
      value  = "true"
      effect = "NO_SCHEDULE"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
