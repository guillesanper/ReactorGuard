# bootstrap.ps1
# Prepara el proyecto GCP para que Terraform pueda ejecutarse por primera vez.
#
# Ejecucion:
#   $env:GOOGLE_APPLICATION_CREDENTIALS = "C:\ruta\key.json"
#   .\infra\scripts\bootstrap.ps1

# Añadir gcloud al PATH si no está (instalacion por defecto en Windows)
$gcloudBin = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin"
if (Test-Path $gcloudBin) {
    $env:PATH = "$gcloudBin;$env:PATH"
}

$PROJECT_ID   = "sentinel-platform-485714"
$REGION       = "europe-west1"
$STATE_BUCKET = "reactorguard-terraform-state"

$APIS = @(
    "container.googleapis.com",
    "compute.googleapis.com",
    "storage.googleapis.com",
    "secretmanager.googleapis.com",
    "pubsub.googleapis.com",
    "cloudkms.googleapis.com",
    "binaryauthorization.googleapis.com"
)

function log_info    { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Green }
function log_warning { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function log_error   { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# 0. Verificar gcloud
log_info "Verificando dependencias..."
$gcloudPath = where.exe gcloud 2>$null
if (-not $gcloudPath) {
    log_error "gcloud no encontrado. Instalalo desde https://cloud.google.com/sdk/docs/install"
}

# Autenticar con service account si hay JSON configurado
if ($env:GOOGLE_APPLICATION_CREDENTIALS) {
    log_info "Autenticando con: $env:GOOGLE_APPLICATION_CREDENTIALS"
    gcloud auth activate-service-account --key-file="$env:GOOGLE_APPLICATION_CREDENTIALS"
} else {
    log_warning "GOOGLE_APPLICATION_CREDENTIALS no definida."
}

gcloud config set project $PROJECT_ID
log_info "Proyecto activo: $PROJECT_ID"

# 1. Crear bucket de estado si no existe
log_info "Comprobando bucket: gs://$STATE_BUCKET"

gsutil ls -b "gs://$STATE_BUCKET" 2>$null
if ($LASTEXITCODE -eq 0) {
    log_warning "El bucket gs://$STATE_BUCKET ya existe, no se vuelve a crear."
} else {
    log_info "Creando bucket gs://$STATE_BUCKET en $REGION..."
    gsutil mb -p $PROJECT_ID -l $REGION "gs://$STATE_BUCKET"
    gsutil versioning set on "gs://$STATE_BUCKET"
    gsutil pap set enforced "gs://$STATE_BUCKET"
    log_info "Bucket creado correctamente."
}

# 2. Habilitar APIs
log_info "Habilitando APIs (puede tardar 1-2 min)..."
foreach ($API in $APIS) {
    log_info "  -> $API"
    gcloud services enable $API --project=$PROJECT_ID
}
log_info "Todas las APIs habilitadas."

# 3. Siguientes pasos
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Bootstrap completado" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Siguientes pasos:"
Write-Host "  1. terraform version   (verificar >= 1.6.0)"
Write-Host "  2. cd infra\terraform\environments\dev"
Write-Host "  3. terraform init"
Write-Host "  4. terraform plan"
Write-Host ""
Write-Host "  Bucket : gs://$STATE_BUCKET"
Write-Host "  Proyecto: $PROJECT_ID"
Write-Host "  Region  : $REGION"
Write-Host ""
