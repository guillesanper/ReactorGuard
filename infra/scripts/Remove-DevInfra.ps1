#Requires -Version 7.0
<#
.SYNOPSIS
    Destruye la infraestructura del entorno dev con confirmación explícita.

.DESCRIPTION
    Ejecuta terraform destroy en infra/terraform/environments/dev con protecciones
    para evitar destrucciones accidentales:
      1. Solicita confirmación escribiendo exactamente "DESTROY-DEV"
      2. Muestra un resumen de lo que se va a destruir (terraform plan -destroy)
      3. Requiere confirmación final antes de ejecutar el destroy
      4. Limpia el directorio .terraform local tras la destrucción

    ⚠️  ADVERTENCIA: Esta operación es IRREVERSIBLE. Todos los recursos GCP
    del entorno dev serán eliminados, incluyendo datos en GCS y secretos.
    Asegúrate de tener backups antes de ejecutar.

.PARAMETER TfDir
    Ruta al directorio del entorno Terraform. Por defecto: infra/terraform/environments/dev

.PARAMETER SkipPlan
    Si se especifica, omite el terraform plan -destroy previo y va directo al destroy.
    Úsalo sólo si ya has revisado el plan anteriormente.

.EXAMPLE
    # Desde la raíz del repositorio:
    .\infra\scripts\Remove-DevInfra.ps1

    # Especificando el directorio explícitamente:
    .\infra\scripts\Remove-DevInfra.ps1 -TfDir "infra\terraform\environments\dev"
#>

# =============================================================================
# CAPA 1 — CONFIGURACIÓN
# =============================================================================
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$TfDir     = "infra\terraform\environments\dev",
    [switch]$SkipPlan
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# CAPA 2 — UTILIDADES
# =============================================================================

function Write-Step {
    <#
    .SYNOPSIS
        Imprime un mensaje de progreso con timestamp y color.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Message,
        [string]$Color = "Cyan"
    )
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $Message" -ForegroundColor $Color
}

function Write-Warning-Banner {
    <#
    .SYNOPSIS
        Imprime el banner de advertencia de destrucción.
    #>
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
    Write-Host "!!                                                          !!" -ForegroundColor Red
    Write-Host "!!   ⚠️  DESTRUCCIÓN DE INFRAESTRUCTURA DEV — ReactorGuard  !!" -ForegroundColor Red
    Write-Host "!!                                                          !!" -ForegroundColor Red
    Write-Host "!!   Esta operación es IRREVERSIBLE. Se eliminarán:         !!" -ForegroundColor Red
    Write-Host "!!     • Cluster GKE (reactorguard-cluster)                !!" -ForegroundColor Red
    Write-Host "!!     • Buckets GCS y TODOS los datos almacenados          !!" -ForegroundColor Red
    Write-Host "!!     • Secretos en Secret Manager                         !!" -ForegroundColor Red
    Write-Host "!!     • KMS key ring (protección prevent_destroy activa)   !!" -ForegroundColor Red
    Write-Host "!!     • VPC, subnets, Cloud NAT, firewall rules            !!" -ForegroundColor Red
    Write-Host "!!     • Service Accounts e IAM bindings                    !!" -ForegroundColor Red
    Write-Host "!!     • Load Balancer, Cloud Armor policy, IAP             !!" -ForegroundColor Red
    Write-Host "!!                                                          !!" -ForegroundColor Red
    Write-Host "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" -ForegroundColor Red
    Write-Host ""
}

function Get-Confirmation {
    <#
    .SYNOPSIS
        Solicita que el usuario escriba "DESTROY-DEV" para confirmar la operación.
    .OUTPUTS
        $true si la confirmación es correcta, termina el script si no lo es.
    #>
    [CmdletBinding()]
    param()

    Write-Host "  Para continuar, escribe exactamente: " -NoNewline -ForegroundColor Yellow
    Write-Host "DESTROY-DEV" -ForegroundColor Red
    Write-Host "  (o presiona Ctrl+C para cancelar)" -ForegroundColor Yellow
    Write-Host ""
    $input = Read-Host "  Confirmación"

    if ($input -ne "DESTROY-DEV") {
        Write-Host ""
        Write-Host "  Confirmación incorrecta. Operación cancelada." -ForegroundColor Green
        Write-Host ""
        exit 0
    }
    return $true
}

function Resolve-TfDir {
    <#
    .SYNOPSIS
        Resuelve y valida la ruta del directorio Terraform.
    .OUTPUTS
        Ruta absoluta al directorio Terraform.
    #>
    [CmdletBinding()]
    param()

    # Intentar resolver desde el directorio actual (raíz del repo)
    $absolutePath = Resolve-Path -Path $TfDir -ErrorAction SilentlyContinue
    if ($null -eq $absolutePath) {
        # Intentar desde la raíz del repo (un nivel arriba del script)
        $scriptDir  = Split-Path -Parent $PSCommandPath
        $repoRoot   = Split-Path -Parent (Split-Path -Parent $scriptDir)
        $absolutePath = Join-Path $repoRoot $TfDir
    }
    else {
        $absolutePath = $absolutePath.Path
    }

    if (-not (Test-Path $absolutePath -PathType Container)) {
        throw "Directorio Terraform no encontrado: $absolutePath"
    }
    return $absolutePath
}

# =============================================================================
# CAPA 3 — SERVICIO
# =============================================================================

function Invoke-TerraformPlanDestroy {
    <#
    .SYNOPSIS
        Ejecuta terraform plan -destroy para mostrar qué se va a eliminar.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$WorkDir
    )

    Write-Step "Ejecutando terraform plan -destroy para revisar recursos a eliminar..."
    Push-Location $WorkDir
    try {
        & terraform plan -destroy -out=destroy.tfplan 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "terraform plan -destroy falló con código $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Invoke-TerraformDestroy {
    <#
    .SYNOPSIS
        Ejecuta terraform destroy -auto-approve usando el plan generado.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$WorkDir
    )

    Write-Step "Ejecutando terraform destroy..." "Red"
    Push-Location $WorkDir
    try {
        if (Test-Path "destroy.tfplan") {
            & terraform apply -destroy destroy.tfplan 2>&1
        }
        else {
            & terraform destroy -auto-approve 2>&1
        }
        if ($LASTEXITCODE -ne 0) {
            throw "terraform destroy falló con código $LASTEXITCODE"
        }
    }
    finally {
        # Limpiar el plan file
        if (Test-Path (Join-Path $WorkDir "destroy.tfplan")) {
            Remove-Item (Join-Path $WorkDir "destroy.tfplan") -Force
        }
        Pop-Location
    }
}

function Remove-TerraformCache {
    <#
    .SYNOPSIS
        Elimina el directorio .terraform local tras la destrucción.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$WorkDir
    )

    $tfCacheDir = Join-Path $WorkDir ".terraform"
    if (Test-Path $tfCacheDir) {
        Write-Step "Limpiando directorio .terraform local: $tfCacheDir"
        Remove-Item -Recurse -Force $tfCacheDir
        Write-Step "  → Directorio .terraform eliminado." "Gray"
    }
    else {
        Write-Step "  → Directorio .terraform no encontrado (ya limpio)." "Gray"
    }
}

# =============================================================================
# CAPA 4 — ORQUESTACIÓN
# =============================================================================

function Invoke-RemoveDevInfra {
    <#
    .SYNOPSIS
        Orquesta el flujo completo de destrucción: confirmación → plan → destroy → limpieza.
    #>
    [CmdletBinding()]
    param()

    # --- Banner de advertencia ---
    Write-Warning-Banner

    # --- Confirmación 1: escribir DESTROY-DEV ---
    Get-Confirmation

    # --- Resolver directorio Terraform ---
    $tfAbsPath = Resolve-TfDir
    Write-Step "Directorio Terraform: $tfAbsPath"

    # --- Plan de destrucción (a menos que -SkipPlan) ---
    if (-not $SkipPlan) {
        Invoke-TerraformPlanDestroy -WorkDir $tfAbsPath

        Write-Host ""
        Write-Host "  Revisa el plan anterior." -ForegroundColor Yellow
        Write-Host "  ¿Confirmas que quieres destruir TODOS los recursos listados? (s/N): " -NoNewline -ForegroundColor Yellow
        $finalConfirm = Read-Host
        if ($finalConfirm -notin @("s", "S", "si", "SI", "sí", "SÍ")) {
            Write-Host ""
            Write-Host "  Operación cancelada por el usuario." -ForegroundColor Green
            exit 0
        }
    }

    # --- Ejecutar destroy ---
    Write-Step "Iniciando destrucción de infraestructura dev..." "Red"
    Invoke-TerraformDestroy -WorkDir $tfAbsPath

    Write-Step "Destrucción completada." "Green"

    # --- Limpiar .terraform ---
    Remove-TerraformCache -WorkDir $tfAbsPath

    # --- Resumen ---
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host "  ✅  Infraestructura dev eliminada correctamente." -ForegroundColor Green
    Write-Host ""
    Write-Host "  Para recrear el entorno:" -ForegroundColor White
    Write-Host "    cd $tfAbsPath" -ForegroundColor White
    Write-Host "    terraform init" -ForegroundColor White
    Write-Host "    terraform apply" -ForegroundColor White
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
}

# Punto de entrada
Invoke-RemoveDevInfra
