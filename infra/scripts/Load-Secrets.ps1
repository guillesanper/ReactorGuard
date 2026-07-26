#Requires -Version 7.0
<#
.SYNOPSIS
    Carga los valores reales de los secretos en Google Secret Manager.

.DESCRIPTION
    Sustituye los placeholders creados por Terraform con los valores reales
    de las credenciales de ReactorGuard. Ejecutar ANTES de terraform apply
    si el entorno ya existe, o DESPUÉS del primer apply para cargar los valores.

    ⚠️  IMPORTANTE: Los valores que pases aquí son credenciales reales.
    - Nunca los escribas directamente en la línea de comandos en un terminal compartido.
    - Usa variables de entorno o un gestor de credenciales para pasarlos.
    - Este script NO guarda los valores en disco ni en logs.

.EXAMPLE
    .\Load-Secrets.ps1 `
        -ScadaUsername  "operador_planta" `
        -ScadaPassword  (Read-Host "Password SCADA" -AsSecureString | ConvertFrom-SecureString -AsPlainText) `
        -ScadaEndpoint  "https://scada.reactorguard.internal" `
        -JwtSecret      "$(openssl rand -hex 32)" `
        -MlflowDbUrl    "postgresql://mlflow:pass@10.0.1.10/mlflow" `
        -GcpApiKey      "AIzaSy..."
#>

# =============================================================================
# CAPA 1 — CONFIGURACIÓN
# =============================================================================
[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory, HelpMessage = "Nombre de usuario del sistema SCADA")]
    [string]$ScadaUsername,

    [Parameter(Mandatory, HelpMessage = "Contraseña del sistema SCADA")]
    [string]$ScadaPassword,

    [Parameter(Mandatory, HelpMessage = "URL del endpoint SCADA (https://...)")]
    [string]$ScadaEndpoint,

    [Parameter(Mandatory, HelpMessage = "Clave JWT de 32+ bytes aleatorios")]
    [string]$JwtSecret,

    [Parameter(Mandatory, HelpMessage = "Cadena de conexión PostgreSQL para MLflow")]
    [string]$MlflowDbUrl,

    [Parameter(Mandatory, HelpMessage = "GCP API Key")]
    [string]$GcpApiKey,

    [string]$ProjectId = "sentinel-platform-485714"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# CAPA 2 — UTILIDADES
# =============================================================================

function Assert-NotEmpty {
    <#
    .SYNOPSIS
        Valida que un valor no sea vacío ni contenga el placeholder REPLACE_ME.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Value,
        [Parameter(Mandatory)][string]$ParamName
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "El parámetro '$ParamName' no puede estar vacío."
    }
    if ($Value -like "*REPLACE_ME*" -or $Value -like "*REPLACE_WITH*") {
        throw "El parámetro '$ParamName' contiene un placeholder. Proporciona el valor real."
    }
}

function Write-Step {
    <#
    .SYNOPSIS
        Imprime un mensaje de progreso con timestamp.
    #>
    param([Parameter(Mandatory)][string]$Message)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $Message" -ForegroundColor Cyan
}

function ConvertTo-SecretPayload {
    <#
    .SYNOPSIS
        Serializa un hashtable a JSON compacto para Secret Manager.
    #>
    param([Parameter(Mandatory)][hashtable]$Data)
    return ($Data | ConvertTo-Json -Compress -Depth 5)
}

# =============================================================================
# CAPA 3 — SERVICIO
# =============================================================================

function Set-GcpSecret {
    <#
    .SYNOPSIS
        Añade una nueva versión a un secreto existente en Secret Manager.
    .DESCRIPTION
        Usa gcloud secrets versions add para actualizar el secreto.
        Si el secreto no existe (Terraform no se ha aplicado), lanza error.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)][string]$SecretName,
        [Parameter(Mandatory)][string]$Payload,
        [Parameter(Mandatory)][string]$Project
    )

    if ($PSCmdlet.ShouldProcess($SecretName, "Cargar nueva versión en Secret Manager")) {
        Write-Step "Cargando secreto: $SecretName"

        # Pasar el payload por stdin para evitar que aparezca en el historial de comandos
        $Payload | gcloud secrets versions add $SecretName `
            --project=$Project `
            --data-file=- 2>&1

        if ($LASTEXITCODE -ne 0) {
            throw "Error al cargar el secreto '$SecretName'. ¿Se aplicó terraform apply primero?"
        }

        Write-Host "  ✅ $SecretName — versión añadida correctamente" -ForegroundColor Green
    }
}

# =============================================================================
# CAPA 4 — ORQUESTACIÓN
# =============================================================================

function Invoke-LoadSecrets {
    <#
    .SYNOPSIS
        Valida todos los parámetros y carga los 4 secretos de ReactorGuard.
    #>
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host " ReactorGuard — Carga de Secretos en Secret Manager" -ForegroundColor Yellow
    Write-Host " Proyecto: $ProjectId" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""

    # --- Validar todos los parámetros antes de enviar nada ---
    Write-Step "Validando parámetros..."
    Assert-NotEmpty -Value $ScadaUsername  -ParamName "ScadaUsername"
    Assert-NotEmpty -Value $ScadaPassword  -ParamName "ScadaPassword"
    Assert-NotEmpty -Value $ScadaEndpoint  -ParamName "ScadaEndpoint"
    Assert-NotEmpty -Value $JwtSecret      -ParamName "JwtSecret"
    Assert-NotEmpty -Value $MlflowDbUrl    -ParamName "MlflowDbUrl"
    Assert-NotEmpty -Value $GcpApiKey      -ParamName "GcpApiKey"

    if ($JwtSecret.Length -lt 32) {
        throw "JwtSecret debe tener al menos 32 caracteres. Longitud actual: $($JwtSecret.Length)"
    }
    if ($ScadaEndpoint -notmatch '^https?://') {
        throw "ScadaEndpoint debe comenzar con https:// o http://. Valor: $ScadaEndpoint"
    }
    if ($MlflowDbUrl -notmatch '^postgresql://') {
        throw "MlflowDbUrl debe comenzar con postgresql://. Valor: $MlflowDbUrl"
    }

    Write-Host "  ✅ Validación completada — todos los parámetros son válidos" -ForegroundColor Green
    Write-Host ""

    # --- Cargar secreto 1: Credenciales SCADA ---
    $scadaPayload = ConvertTo-SecretPayload @{
        username = $ScadaUsername
        password = $ScadaPassword
        endpoint = $ScadaEndpoint
    }
    Set-GcpSecret -SecretName "reactorguard-scada-credentials" -Payload $scadaPayload -Project $ProjectId

    # --- Cargar secreto 2: JWT Secret ---
    Set-GcpSecret -SecretName "reactorguard-jwt-secret" -Payload $JwtSecret -Project $ProjectId

    # --- Cargar secreto 3: MLflow DB URL ---
    Set-GcpSecret -SecretName "reactorguard-mlflow-db-url" -Payload $MlflowDbUrl -Project $ProjectId

    # --- Cargar secreto 4: GCP API Key ---
    Set-GcpSecret -SecretName "reactorguard-gcp-api-key" -Payload $GcpApiKey -Project $ProjectId

    # --- Resumen ---
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " ✅ Los 4 secretos se cargaron correctamente" -ForegroundColor Green
    Write-Host ""
    Write-Host " Verificar con:" -ForegroundColor White
    Write-Host "   gcloud secrets list --project=$ProjectId --filter='name:reactorguard'" -ForegroundColor White
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""
}

# Punto de entrada
Invoke-LoadSecrets
