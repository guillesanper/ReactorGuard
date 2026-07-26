# main.tf — Entorno dev
# Orquesta los módulos reutilizables de ReactorGuard.
# Los outputs de un módulo se pasan como inputs de los módulos dependientes.
#
# Orden de dependencias:
#   vpc → gke (necesita la subnet y los secondary ranges)
#   vpc, gke → iam (necesita el cluster SA y los buckets)
#   (storage es independiente de vpc/gke)

locals {
  project_id = "sentinel-platform-485714"
  region     = "europe-southwest1"
  env        = "dev"
}

# ---------------------------------------------------------------------------
# Red (VPC, subnets, Cloud NAT, firewall rules)
# ---------------------------------------------------------------------------
module "vpc" {
  source = "../../modules/vpc"

  project_id    = local.project_id
  region        = local.region
  env           = local.env
  vpc_name      = "reactorguard-vpc"
  subnet_cidr   = "10.0.1.0/24"
  pods_cidr     = "10.0.16.0/20"
  services_cidr = "10.0.32.0/20"
}

# ---------------------------------------------------------------------------
# Clúster GKE Standard privado con dos node pools
# Depende de vpc para obtener la subnet y los secondary ranges.
# ---------------------------------------------------------------------------
module "gke" {
  source = "../../modules/gke"

  project_id          = local.project_id
  region              = local.region
  env                 = local.env
  cluster_name        = "reactorguard-cluster"
  network             = module.vpc.vpc_self_link
  subnetwork          = module.vpc.subnet_self_link
  pods_range_name     = module.vpc.pods_range_name
  services_range_name = module.vpc.services_range_name
  master_ipv4_cidr    = "172.16.0.0/28"

  # Node Pool platform (cargas generales, preemptible)
  platform_machine_type = "e2-standard-4"
  platform_min_nodes    = 1
  platform_max_nodes    = 3

  # Node Pool ml-serving (safety-critical, NO preemptible)
  ml_machine_type = "n2-standard-8"
  ml_min_nodes    = 1
  ml_max_nodes    = 5
}

# ---------------------------------------------------------------------------
# Seguridad (KMS para CMEK, Secret Manager, Cloud Armor WAF, LB, IAP)
# Debe aplicarse ANTES de storage para poder pasar el ID de la KMS key.
# ---------------------------------------------------------------------------
module "security" {
  source = "../../modules/security"

  project_id        = local.project_id
  region            = local.region
  env               = local.env
  iap_support_email = "admin@reactorguard.internal"
}

# ---------------------------------------------------------------------------
# Almacenamiento GCS (4 buckets: raw, processed, models, mlflow)
# Depende de security para obtener el ID de la KMS key (CMEK).
# ---------------------------------------------------------------------------
module "storage" {
  source = "../../modules/storage"

  project_id = local.project_id
  region     = local.region
  env        = local.env
  cmek_key   = module.security.storage_key_id
}

# ---------------------------------------------------------------------------
# IAM (service accounts, bindings, Workload Identity)
# Depende de vpc y gke (cluster SA) y de storage: los bindings de GCS se
# conceden POR BUCKET (least-privilege), así que necesita los nombres de bucket.
# ---------------------------------------------------------------------------
module "iam" {
  source = "../../modules/iam"

  project_id = local.project_id
  region     = local.region

  # Nombres de bucket para bindings IAM por-bucket (no a nivel de proyecto).
  data_raw_bucket       = module.storage.data_raw_bucket_name
  data_processed_bucket = module.storage.data_processed_bucket_name
  models_bucket         = module.storage.models_bucket_name
  mlflow_bucket         = module.storage.mlflow_bucket_name
}

# ---------------------------------------------------------------------------
# Kafka: desplegado via Install-Kafka.ps1 después del terraform apply.
# No se gestiona aquí porque Helm necesita el cluster GKE activo.
# ---------------------------------------------------------------------------
