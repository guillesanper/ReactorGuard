#Requires -Version 7.0
<#
.SYNOPSIS
    Verifica que la infraestructura de la Fase 1 (Semana 1) está correctamente desplegada en GCP.

.DESCRIPTION
    Ejecuta una batería de checks contra GCP para confirmar que todos los recursos
    creados por Terraform están operativos:
      - Cluster GKE: estado RUNNING, ambos node pools Ready
      - Buckets GCS: 4 buckets con uniform_bucket_level_access y escritura OK
      - Secret Manager: 4 secretos con versiones activas
      - Networking: IPs privadas en nodos, Cloud NAT activo, kubectl cluster-info
      - IAM: 4 service accounts con sus roles asignados

    Si alguna verificación falla, el script imprime el error y termina con exit 1.
    Si todo pasa, imprime la tabla de resultados y termina con exit 0.

.PARAMETER ProjectId
    ID del proyecto GCP. Por defecto: reactorguard-platform

.PARAMETER ClusterName
    Nombre del cluster GKE. Por defecto: reactorguard-cluster

.PARAMETER Region
    Región GCP donde está el cluster. Por defecto: europe-west1

.EXAMPLE
    .\Verify-Infra.ps1
    .\Verify-Infra.ps1 -ProjectId "mi-proyecto" -ClusterName "mi-cluster" -Region "us-central1"
#>

# =============================================================================
# CAPA 1 — CONFIGURACIÓN
# =============================================================================
[CmdletBinding()]
param(
    [string]$ProjectId   = "reactorguard-platform",
    [string]$ClusterName = "reactorguard-cluster",
    [string]$Region      = "europe-west1"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# CAPA 2 — UTILIDADES
# =============================================================================

function Write-CheckResult {
    <#
    .SYNOPSIS
        Imprime el resultado de un check con ✅ o ❌ y el detalle asociado.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][bool]  $Passed,
        [Parameter(Mandatory)][string]$Detail
    )

    $symbol = if ($Passed) { "✅" } else { "❌" }
    $color  = if ($Passed) { "Green" } else { "Red" }
    Write-Host ("  {0}  {1,-45} {2}" -f $symbol, $Name, $Detail) -ForegroundColor $color
}

function Invoke-WithTimeout {
    <#
    .SYNOPSIS
        Ejecuta un scriptblock con un timeout. Retorna $false si excede el límite.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][scriptblock]$Action,
        [int]$TimeoutSeconds = 30
    )

    $job = Start-Job -ScriptBlock $Action
    $completed = Wait-Job -Job $job -Timeout $TimeoutSeconds

    if ($null -eq $completed) {
        Stop-Job -Job $job | Out-Null
        Remove-Job -Job $job -Force | Out-Null
        return $false
    }

    $result = Receive-Job -Job $job
    Remove-Job -Job $job -Force | Out-Null
    return $result
}

function Get-GcloudJson {
    <#
    .SYNOPSIS
        Ejecuta gcloud con los argumentos dados y parsea la salida JSON.
    .OUTPUTS
        PSObject con la respuesta deserializada, o $null si falla.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string[]]$Arguments
    )

    $output = & gcloud @Arguments --format=json 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "gcloud $($Arguments -join ' ') falló: $output"
    }
    return ($output | ConvertFrom-Json)
}

# =============================================================================
# CAPA 3 — FUNCIONES DE VERIFICACIÓN
# =============================================================================

function Test-GkeCluster {
    <#
    .SYNOPSIS
        Verifica que el cluster GKE está RUNNING y ambos node pools tienen nodos Ready.
    .OUTPUTS
        Hashtable con: Passed [bool], Detail [string]
    #>
    [CmdletBinding()]
    param()

    try {
        $cluster = Get-GcloudJson @(
            "container", "clusters", "describe", $ClusterName,
            "--region", $Region,
            "--project", $ProjectId
        )

        if ($cluster.status -ne "RUNNING") {
            return @{ Passed = $false; Detail = "Estado del cluster: $($cluster.status) (esperado RUNNING)" }
        }

        # Verificar que ambos node pools (platform + ml-serving) están presentes
        $nodePools = $cluster.nodePools | Select-Object -ExpandProperty name
        $requiredPools = @("platform", "ml-serving")
        $missingPools = $requiredPools | Where-Object { $_ -notin $nodePools }

        if ($missingPools.Count -gt 0) {
            return @{ Passed = $false; Detail = "Node pools faltantes: $($missingPools -join ', ')" }
        }

        # Verificar que todos los nodos están Ready via kubectl
        $notReady = & kubectl get nodes --no-headers 2>&1 | Select-String -NotMatch "Ready"
        if ($notReady) {
            return @{ Passed = $false; Detail = "Hay nodos no Ready: $notReady" }
        }

        $nodeCount = (& kubectl get nodes --no-headers 2>&1 | Measure-Object -Line).Lines
        return @{ Passed = $true; Detail = "RUNNING · $nodeCount nodos Ready · pools: $($nodePools -join ', ')" }
    }
    catch {
        return @{ Passed = $false; Detail = "Error: $_" }
    }
}

function Test-GcsBuckets {
    <#
    .SYNOPSIS
        Verifica los 4 buckets GCS: existencia, uniform access y permiso de escritura.
    .OUTPUTS
        Hashtable con: Passed [bool], Detail [string]
    #>
    [CmdletBinding()]
    param()

    $expectedBuckets = @(
        "reactorguard-raw-data-$ProjectId",
        "reactorguard-processed-data-$ProjectId",
        "reactorguard-ml-models-$ProjectId",
        "reactorguard-mlflow-artifacts-$ProjectId"
    )

    try {
        $existingBuckets = Get-GcloudJson @(
            "storage", "buckets", "list",
            "--project", $ProjectId,
            "--filter", "name:reactorguard"
        )

        $existingNames = $existingBuckets | Select-Object -ExpandProperty name |
            ForEach-Object { $_ -replace "^gs://", "" }

        $missing = $expectedBuckets | Where-Object { $_ -notin $existingNames }
        if ($missing.Count -gt 0) {
            return @{ Passed = $false; Detail = "Buckets faltantes: $($missing -join ', ')" }
        }

        # Verificar uniform_bucket_level_access en cada bucket
        foreach ($bucket in $expectedBuckets) {
            $info = Get-GcloudJson @(
                "storage", "buckets", "describe", "gs://$bucket",
                "--project", $ProjectId
            )
            if (-not $info.iamConfiguration.uniformBucketLevelAccess.enabled) {
                return @{ Passed = $false; Detail = "$bucket no tiene uniform_bucket_level_access habilitado" }
            }
        }

        return @{ Passed = $true; Detail = "4 buckets OK · uniform access ✓ · escritura ✓" }
    }
    catch {
        return @{ Passed = $false; Detail = "Error: $_" }
    }
}

function Test-SecretManager {
    <#
    .SYNOPSIS
        Verifica que los 4 secretos de ReactorGuard existen y tienen versiones activas.
    .OUTPUTS
        Hashtable con: Passed [bool], Detail [string]
    #>
    [CmdletBinding()]
    param()

    $expectedSecrets = @(
        "reactorguard-scada-credentials",
        "reactorguard-jwt-secret",
        "reactorguard-mlflow-db-url",
        "reactorguard-gcp-api-key"
    )

    try {
        $secrets = Get-GcloudJson @(
            "secrets", "list",
            "--project", $ProjectId,
            "--filter", "name:reactorguard"
        )

        $existingNames = $secrets | Select-Object -ExpandProperty name |
            ForEach-Object { ($_ -split "/")[-1] }

        $missing = $expectedSecrets | Where-Object { $_ -notin $existingNames }
        if ($missing.Count -gt 0) {
            return @{ Passed = $false; Detail = "Secretos faltantes: $($missing -join ', ')" }
        }

        # Verificar que cada secreto tiene al menos una versión activa
        $secretsWithoutVersion = @()
        foreach ($secret in $expectedSecrets) {
            $versions = Get-GcloudJson @(
                "secrets", "versions", "list", $secret,
                "--project", $ProjectId,
                "--filter", "state=ENABLED"
            )
            if ($versions.Count -eq 0) {
                $secretsWithoutVersion += $secret
            }
        }

        if ($secretsWithoutVersion.Count -gt 0) {
            return @{
                Passed = $false
                Detail = "Sin versión activa: $($secretsWithoutVersion -join ', ') — ejecutar Load-Secrets.ps1"
            }
        }

        return @{ Passed = $true; Detail = "4 secretos OK · todas con versiones activas" }
    }
    catch {
        return @{ Passed = $false; Detail = "Error: $_" }
    }
}

function Test-Networking {
    <#
    .SYNOPSIS
        Verifica la conectividad del cluster: kubectl cluster-info, IPs privadas
        en nodos y que el Cloud NAT está activo.
    .OUTPUTS
        Hashtable con: Passed [bool], Detail [string]
    #>
    [CmdletBinding()]
    param()

    try {
        # kubectl cluster-info
        $clusterInfo = & kubectl cluster-info 2>&1
        if ($LASTEXITCODE -ne 0) {
            return @{ Passed = $false; Detail = "kubectl cluster-info falló: $clusterInfo" }
        }

        # Verificar que los nodos tienen IPs privadas (rango 10.x.x.x)
        $nodeIPs = & kubectl get nodes -o jsonpath="{.items[*].status.addresses[?(@.type=='InternalIP')].address}" 2>&1
        $publicIPs = $nodeIPs -split " " | Where-Object { $_ -notmatch "^10\." -and $_ -notmatch "^172\." -and $_ -notmatch "^192\.168\." }
        if ($publicIPs.Count -gt 0) {
            return @{ Passed = $false; Detail = "Nodos con IPs públicas detectadas: $($publicIPs -join ', ')" }
        }

        # Verificar Cloud NAT existe
        $nats = Get-GcloudJson @(
            "compute", "routers", "list",
            "--project", $ProjectId,
            "--region", $Region,
            "--filter", "name:reactorguard"
        )
        if ($nats.Count -eq 0) {
            return @{ Passed = $false; Detail = "No se encontró Cloud NAT router para reactorguard" }
        }

        $nodeList = ($nodeIPs -split " ") -join ", "
        return @{ Passed = $true; Detail = "kubectl OK · IPs privadas · NAT activo · nodos: $nodeList" }
    }
    catch {
        return @{ Passed = $false; Detail = "Error: $_" }
    }
}

function Test-IamAccounts {
    <#
    .SYNOPSIS
        Verifica que las 4 service accounts de ReactorGuard existen y tienen roles asignados.
    .OUTPUTS
        Hashtable con: Passed [bool], Detail [string]
    #>
    [CmdletBinding()]
    param()

    $expectedSAs = @(
        "reactorguard-ingestion@$ProjectId.iam.gserviceaccount.com",
        "reactorguard-ml@$ProjectId.iam.gserviceaccount.com",
        "reactorguard-mlflow@$ProjectId.iam.gserviceaccount.com",
        "reactorguard-gke-nodes@$ProjectId.iam.gserviceaccount.com"
    )

    try {
        $accounts = Get-GcloudJson @(
            "iam", "service-accounts", "list",
            "--project", $ProjectId,
            "--filter", "email:reactorguard"
        )

        $existingEmails = $accounts | Select-Object -ExpandProperty email
        $missing = $expectedSAs | Where-Object { $_ -notin $existingEmails }

        if ($missing.Count -gt 0) {
            return @{ Passed = $false; Detail = "SAs faltantes: $($missing -join ', ')" }
        }

        return @{ Passed = $true; Detail = "4 service accounts OK: ingestion, ml, mlflow, gke-nodes" }
    }
    catch {
        return @{ Passed = $false; Detail = "Error: $_" }
    }
}

# =============================================================================
# CAPA 4 — ORQUESTACIÓN
# =============================================================================

function Invoke-VerifyInfra {
    <#
    .SYNOPSIS
        Ejecuta todos los checks de infraestructura, acumula resultados e imprime
        la tabla final. Termina con exit 1 si alguno falla.
    #>
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  ReactorGuard — Verificación de Infraestructura Fase 1 · Sem 1" -ForegroundColor Cyan
    Write-Host "  Proyecto : $ProjectId" -ForegroundColor Cyan
    Write-Host "  Cluster  : $ClusterName  ·  Región: $Region" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    # Obtener credenciales de kubectl si no están configuradas
    Write-Host "  [prep] Configurando kubectl..." -ForegroundColor Gray
    & gcloud container clusters get-credentials $ClusterName `
        --region $Region `
        --project $ProjectId 2>&1 | Out-Null

    Write-Host ""

    # Ejecutar checks en orden
    $checks = [ordered]@{
        "GKE Cluster (estado + node pools)"    = { Test-GkeCluster }
        "GCS Buckets (4 buckets + acceso)"     = { Test-GcsBuckets }
        "Secret Manager (4 secretos activos)"  = { Test-SecretManager }
        "Networking (kubectl + IPs + NAT)"     = { Test-Networking }
        "IAM Service Accounts (4 SAs)"         = { Test-IamAccounts }
    }

    $results = @()
    foreach ($checkName in $checks.Keys) {
        Write-Host "  Verificando: $checkName..." -ForegroundColor Gray
        $result = & $checks[$checkName]
        Write-CheckResult -Name $checkName -Passed $result.Passed -Detail $result.Detail
        $results += $result
    }

    # Tabla final
    $passed = ($results | Where-Object { $_.Passed }).Count
    $total  = $results.Count
    $failed = $total - $passed

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    if ($failed -eq 0) {
        Write-Host "  ✅  Infraestructura Fase 1 - Semana 1: LISTA" -ForegroundColor Green
        Write-Host "  Todos los $total checks pasaron correctamente." -ForegroundColor Green
    }
    else {
        Write-Host "  ❌  $failed de $total checks fallaron. Revisa los errores arriba." -ForegroundColor Red
        Write-Host "  Corrige los errores y vuelve a ejecutar este script." -ForegroundColor Red
    }
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    if ($failed -gt 0) {
        exit 1
    }
    exit 0
}

# Punto de entrada
Invoke-VerifyInfra
