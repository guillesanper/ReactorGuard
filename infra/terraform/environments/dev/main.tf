# main.tf
# Punto de entrada del entorno dev.
# Orquesta los módulos reutilizables definidos en infra/terraform/modules/.
# Cada módulo encapsula un dominio de infraestructura independiente;
# este fichero solo los conecta pasando outputs de unos como inputs de otros.
#
# ESTADO ACTUAL: esqueleto con bloques vacíos — las variables se rellenarán
# en subtareas sucesivas conforme se implementen los módulos.

# ---------------------------------------------------------------------------
# Red (VPC, subnets, Cloud NAT, firewall rules)
# ---------------------------------------------------------------------------
module "vpc" {
  source = "../../modules/vpc"

  # Variables que se definirán al implementar el módulo vpc:
  # project_id = "reactorguard-platform"
  # region     = "europe-west1"
  # env        = "dev"
}

# ---------------------------------------------------------------------------
# Clúster GKE (Autopilot o Standard según decisión de arquitectura)
# ---------------------------------------------------------------------------
module "gke" {
  source = "../../modules/gke"

  # Dependerá del output de vpc:
  # network    = module.vpc.network_name
  # subnetwork = module.vpc.subnetwork_name
}

# ---------------------------------------------------------------------------
# Almacenamiento (buckets GCS para datos crudos, modelos, artefactos)
# ---------------------------------------------------------------------------
module "storage" {
  source = "../../modules/storage"

  # project_id = "reactorguard-platform"
  # env        = "dev"
}

# ---------------------------------------------------------------------------
# IAM (service accounts, bindings de roles, Workload Identity)
# ---------------------------------------------------------------------------
module "iam" {
  source = "../../modules/iam"

  # project_id    = "reactorguard-platform"
  # gke_sa_email  = module.gke.service_account_email
}

# ---------------------------------------------------------------------------
# Seguridad (KMS keys, Binary Authorization, Secret Manager secrets base)
# ---------------------------------------------------------------------------
module "security" {
  source = "../../modules/security"

  # project_id = "reactorguard-platform"
  # region     = "europe-west1"
}

# ---------------------------------------------------------------------------
# Kafka (Confluent o self-hosted en GKE mediante Helm/Strimzi)
# ---------------------------------------------------------------------------
module "kafka" {
  source = "../../modules/kafka"

  # Se configurará una vez el módulo gke esté operativo
  # cluster_endpoint = module.gke.endpoint
}
