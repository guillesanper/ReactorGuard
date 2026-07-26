#!/usr/bin/env bash
# bootstrap.sh
# Prepara el proyecto GCP para que Terraform pueda ejecutarse por primera vez.
#
# Ejecución:
#   chmod +x infra/scripts/bootstrap.sh
#   ./infra/scripts/bootstrap.sh
#
# Requisitos previos:
#   - gcloud instalado y en el PATH
#   - gcloud auth login + gcloud auth application-default login completados
#   - Rol Owner (o roles/storage.admin + roles/serviceusage.serviceUsageAdmin)
#     sobre el proyecto sentinel-platform-485714

set -euo pipefail  # Salir en cualquier error, variables sin definir o pipe roto

# ---------------------------------------------------------------------------
# Variables de configuración
# ---------------------------------------------------------------------------
PROJECT_ID="sentinel-platform-485714"
REGION="europe-southwest1"
STATE_BUCKET="reactorguard-terraform-state"

# APIs necesarias para la plataforma ReactorGuard
APIS=(
  "container.googleapis.com"          # GKE
  "compute.googleapis.com"            # VPC, subnets, firewall
  "storage.googleapis.com"            # GCS buckets
  "secretmanager.googleapis.com"      # Secret Manager
  "pubsub.googleapis.com"             # Pub/Sub (mensajería / Kafka bridge)
  "cloudkms.googleapis.com"           # Cloud KMS (cifrado de datos en reposo)
  "binaryauthorization.googleapis.com" # Binary Authorization (supply-chain security)
)

# ---------------------------------------------------------------------------
# Colores para output más legible
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # Sin color

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Verificar que gcloud está disponible
# ---------------------------------------------------------------------------
info "Verificando dependencias..."
command -v gcloud >/dev/null 2>&1 || error "gcloud no encontrado. Instálalo desde https://cloud.google.com/sdk/docs/install"

# Asegurarse de que el proyecto activo es el correcto
gcloud config set project "${PROJECT_ID}"
info "Proyecto activo: ${PROJECT_ID}"

# ---------------------------------------------------------------------------
# 1. Crear el bucket de estado de Terraform si no existe
# ---------------------------------------------------------------------------
info "Comprobando bucket de estado: gs://${STATE_BUCKET}"

if gsutil ls -b "gs://${STATE_BUCKET}" >/dev/null 2>&1; then
  warning "El bucket gs://${STATE_BUCKET} ya existe — no se vuelve a crear."
else
  info "Creando bucket gs://${STATE_BUCKET} en ${REGION}..."
  gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${STATE_BUCKET}"

  # Habilitar versionado del objeto para poder recuperar estados anteriores
  info "Habilitando versionado en el bucket de estado..."
  gsutil versioning set on "gs://${STATE_BUCKET}"

  # Bloquear acceso público por seguridad
  gsutil pap set enforced "gs://${STATE_BUCKET}"

  info "Bucket creado correctamente."
fi

# ---------------------------------------------------------------------------
# 2. Habilitar las APIs de GCP necesarias
# ---------------------------------------------------------------------------
info "Habilitando APIs del proyecto (puede tardar 1-2 min la primera vez)..."

for API in "${APIS[@]}"; do
  info "  → ${API}"
  gcloud services enable "${API}" --project="${PROJECT_ID}"
done

info "Todas las APIs habilitadas."

# ---------------------------------------------------------------------------
# 3. Instrucciones de los siguientes pasos
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================================${NC}"
echo -e "${GREEN}  Bootstrap completado correctamente${NC}"
echo -e "${GREEN}========================================================${NC}"
echo ""
echo "Próximos pasos:"
echo ""
echo "  1. Asegúrate de tener Terraform >= 1.6.0 instalado:"
echo "     terraform version"
echo ""
echo "  2. Inicializa el backend remoto en el entorno dev:"
echo "     cd infra/terraform/environments/dev"
echo "     terraform init"
echo ""
echo "  3. Revisa qué recursos se crearán:"
echo "     terraform plan"
echo ""
echo "  4. Aplica cuando estés conforme:"
echo "     terraform apply"
echo ""
echo "  Bucket de estado : gs://${STATE_BUCKET}"
echo "  Proyecto GCP     : ${PROJECT_ID}"
echo "  Región           : ${REGION}"
echo ""
