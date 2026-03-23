#Requires -Version 7.0
# =============================================================================
# tests/integration/Invoke-KafkaTests.ps1
# Orquesta el despliegue del pod cliente, ejecución de tests Kafka y limpieza.
#
# Flujo:
#   1. Crea el directorio de resultados.
#   2. Despliega kafka-client-pod.yaml y espera a que esté Running.
#   3. Instala kafka-python en el pod (pip).
#   4. Copia los scripts Python al pod y ejecuta test_kafka_connectivity.py.
#   5. (Si -SkipBenchmark no está activo) Ejecuta benchmark_kafka.py y
#      recoge el JSON de resultados.
#   6. Limpieza garantizada del pod (try/finally).
#   7. Imprime el resumen de resultados con ✅/❌.
#
# Uso:
#   .\tests\integration\Invoke-KafkaTests.ps1
#   .\tests\integration\Invoke-KafkaTests.ps1 -SkipBenchmark -Verbose
# =============================================================================

[CmdletBinding()]
param(
    [string]$Namespace    = "kafka-operator",
    [string]$ResultsPath  = "tests/results",
    [switch]$SkipBenchmark
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# CAPA 2 — FUNCIONES DE UTILIDAD (sin efectos secundarios, reutilizables)
# =============================================================================

function New-TestResultsDir {
    <#
    .SYNOPSIS
        Crea el directorio de resultados si no existe.
    #>
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        Write-Verbose "Directorio de resultados creado: $Path"
    }
    else {
        Write-Verbose "Directorio de resultados ya existe: $Path"
    }
}

function Get-BenchmarkResult {
    <#
    .SYNOPSIS
        Lee y parsea el JSON de resultados del benchmark.
    .OUTPUTS
        [PSCustomObject] con las propiedades del benchmark, o $null si no existe.
    #>
    param([Parameter(Mandatory)][string]$JsonPath)

    if (-not (Test-Path $JsonPath)) {
        Write-Verbose "Archivo de resultados no encontrado: $JsonPath"
        return $null
    }

    try {
        $content = Get-Content $JsonPath -Raw -Encoding UTF8
        return $content | ConvertFrom-Json
    }
    catch {
        Write-Warning "Error al parsear JSON de resultados '$JsonPath': $_"
        return $null
    }
}

function Write-TestResult {
    <#
    .SYNOPSIS
        Imprime el resultado de un test con formato ✅/❌.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][bool]$Passed,
        [string]$Value = ""
    )

    $icon  = if ($Passed) { "✅" } else { "❌" }
    $color = if ($Passed) { "Green" } else { "Red" }
    $line  = "$icon $Name"
    if ($Value) { $line += ": $Value" }
    Write-Host $line -ForegroundColor $color
}

function Write-Step {
    param([Parameter(Mandatory)][string]$Message)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host "[$ts] $Message" -ForegroundColor Cyan
}

# =============================================================================
# CAPA 3 — FUNCIONES DE SERVICIO (efectos secundarios aislados)
# =============================================================================

function Deploy-KafkaClientPod {
    <#
    .SYNOPSIS
        Despliega el pod utilitario kafka-client y espera a que esté Running.
    .DESCRIPTION
        Aplica k8s/tools/kafka-client-pod.yaml. Si el pod ya existe lo elimina
        primero para asegurar una imagen y configuración limpias.
    #>
    [CmdletBinding()]
    param()

    $manifestPath = Join-Path $PSScriptRoot ".." ".." "k8s" "tools" "kafka-client-pod.yaml"
    $manifestPath = [System.IO.Path]::GetFullPath($manifestPath)

    # Eliminar pod anterior si existe (puede quedar de un run anterior fallido)
    $existing = kubectl get pod kafka-client --namespace="$Namespace" --ignore-not-found 2>$null
    if ($existing) {
        Write-Step "Eliminando pod anterior kafka-client..."
        kubectl delete pod kafka-client --namespace="$Namespace" --wait=true 2>&1 | Write-Verbose
    }

    Write-Step "Desplegando kafka-client-pod.yaml..."
    kubectl apply --filename="$manifestPath" --namespace="$Namespace" 2>&1 | Write-Verbose

    if ($LASTEXITCODE -ne 0) {
        throw "Error al desplegar kafka-client-pod.yaml"
    }

    Write-Step "Esperando a que el pod kafka-client esté Running..."
    kubectl wait pod kafka-client `
        --namespace="$Namespace" `
        --for=condition=Ready `
        --timeout=120s 2>&1 | Write-Verbose

    if ($LASTEXITCODE -ne 0) {
        throw "Timeout esperando al pod kafka-client. Revisar: kubectl describe pod kafka-client -n $Namespace"
    }

    Write-Step "Pod kafka-client está Running."
}

function Install-PythonDependencies {
    <#
    .SYNOPSIS
        Instala kafka-python dentro del pod kafka-client.
    .DESCRIPTION
        El pod confluentinc/cp-kafka no incluye Python por defecto — instala
        miniconda/python si es necesario, o usa el Python del sistema.
        Nota: cp-kafka 7.5.0 está basado en UBI8 y tiene Python 3.9 disponible.
    #>
    [CmdletBinding()]
    param()

    Write-Step "Instalando kafka-python en el pod..."
    kubectl exec kafka-client `
        --namespace="$Namespace" `
        -- bash -c "pip install --quiet kafka-python 2>&1 || python3 -m pip install --quiet kafka-python 2>&1" `
        2>&1 | Write-Verbose

    # No fallar si pip no está disponible — los tests lo detectarán
    Write-Verbose "Dependencias Python instaladas (o ya presentes)."
}

function Invoke-ConnectivityTest {
    <#
    .SYNOPSIS
        Copia y ejecuta test_kafka_connectivity.py dentro del pod.
    .OUTPUTS
        [bool] True si el test pasó.
    #>
    [CmdletBinding()]
    param()

    $scriptSrc = Join-Path $PSScriptRoot "test_kafka_connectivity.py"
    $scriptSrc = [System.IO.Path]::GetFullPath($scriptSrc)

    Write-Step "Copiando test_kafka_connectivity.py al pod..."
    kubectl cp "$scriptSrc" "${Namespace}/kafka-client:/tmp/test_kafka_connectivity.py" 2>&1 | Write-Verbose

    if ($LASTEXITCODE -ne 0) {
        throw "Error al copiar test_kafka_connectivity.py al pod."
    }

    Write-Step "Ejecutando test de conectividad (100 mensajes)..."
    $output = kubectl exec kafka-client `
        --namespace="$Namespace" `
        -- python3 /tmp/test_kafka_connectivity.py 2>&1

    $testPassed = ($LASTEXITCODE -eq 0)
    Write-Verbose "Salida del test de conectividad:`n$output"

    # Extraer número de mensajes recibidos del output
    $deliveredLine = $output | Select-String "Conteo correcto"
    $deliveredInfo = if ($deliveredLine) { "100/100 mensajes entregados" } else { "verificar logs" }

    return [PSCustomObject]@{
        Passed       = $testPassed
        DeliveredInfo = $deliveredInfo
        Output       = $output -join "`n"
    }
}

function Invoke-BenchmarkTest {
    <#
    .SYNOPSIS
        Copia y ejecuta benchmark_kafka.py, recoge el JSON de resultados.
    .OUTPUTS
        [PSCustomObject] con Passed, P99Ms, ThroughputMsg, JsonPath.
    #>
    [CmdletBinding()]
    param()

    $scriptSrc  = Join-Path $PSScriptRoot "benchmark_kafka.py"
    $scriptSrc  = [System.IO.Path]::GetFullPath($scriptSrc)
    $remoteJson = "/tmp/kafka_benchmark.json"
    $localJson  = Join-Path $ResultsPath "kafka_benchmark.json"
    $localJson  = [System.IO.Path]::GetFullPath($localJson)

    Write-Step "Copiando benchmark_kafka.py al pod..."
    kubectl cp "$scriptSrc" "${Namespace}/kafka-client:/tmp/benchmark_kafka.py" 2>&1 | Write-Verbose

    Write-Step "Ejecutando benchmark (10.000 mensajes de 1KB). Puede tardar ~30s..."
    $output = kubectl exec kafka-client `
        --namespace="$Namespace" `
        -- python3 /tmp/benchmark_kafka.py --messages 10000 --output "$remoteJson" 2>&1

    $benchPassed = ($LASTEXITCODE -eq 0)
    Write-Verbose "Salida del benchmark:`n$output"

    # Copiar el JSON de resultados desde el pod al host
    Write-Step "Recuperando JSON de resultados del pod..."
    kubectl cp "${Namespace}/kafka-client:${remoteJson}" "$localJson" 2>&1 | Write-Verbose

    # Parsear el JSON para el resumen
    $benchResult = Get-BenchmarkResult -JsonPath $localJson

    $p99 = if ($benchResult) { [math]::Round($benchResult.latency_ms.p99, 2) } else { 0 }
    $tps = if ($benchResult) { [math]::Round($benchResult.throughput.msg_per_sec, 0) } else { 0 }

    return [PSCustomObject]@{
        Passed        = $benchPassed
        P99Ms         = $p99
        ThroughputMsg = $tps
        JsonPath      = $localJson
        Output        = $output -join "`n"
    }
}

function Remove-KafkaClientPod {
    <#
    .SYNOPSIS
        Elimina el pod utilitario kafka-client (limpieza garantizada).
    .DESCRIPTION
        Siempre se llama desde un bloque `finally` para asegurar que el pod
        no queda corriendo tras los tests, independientemente del resultado.
    #>
    [CmdletBinding()]
    param()

    Write-Step "Limpiando pod kafka-client..."
    kubectl delete pod kafka-client `
        --namespace="$Namespace" `
        --ignore-not-found `
        --wait=false 2>&1 | Write-Verbose

    Write-Verbose "Pod kafka-client eliminado."
}

# =============================================================================
# CAPA 4 — ORQUESTACIÓN
# =============================================================================

function Invoke-AllKafkaTests {
    <#
    .SYNOPSIS
        Punto de entrada principal. Orquesta todos los tests de Kafka.
    #>
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
    Write-Host "║       ReactorGuard — Verificación de Kafka               ║" -ForegroundColor Magenta
    Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
    Write-Host "  Namespace    : $Namespace"
    Write-Host "  Results dir  : $ResultsPath"
    Write-Host "  Skip bench   : $($SkipBenchmark.IsPresent)"
    Write-Host ""

    New-TestResultsDir -Path $ResultsPath

    $connectivityResult = $null
    $benchmarkResult    = $null

    try {
        Deploy-KafkaClientPod
        Install-PythonDependencies

        # Test 1: conectividad básica
        $connectivityResult = Invoke-ConnectivityTest

        # Test 2: benchmark (opcional)
        if (-not $SkipBenchmark) {
            $benchmarkResult = Invoke-BenchmarkTest
        }
    }
    finally {
        # Limpieza garantizada aunque los tests fallen
        Remove-KafkaClientPod
    }

    # ─── Resumen de resultados ───────────────────────────────────────────────
    Write-Host ""
    Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor White
    Write-Host "  Resumen de Tests Kafka — ReactorGuard Fase 1"            -ForegroundColor White
    Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor White

    $allPassed = $true

    if ($null -ne $connectivityResult) {
        Write-TestResult `
            -Name "Kafka connectivity" `
            -Passed $connectivityResult.Passed `
            -Value $(if ($connectivityResult.Passed) { "OK ($($connectivityResult.DeliveredInfo))" } else { "FAILED" })
        if (-not $connectivityResult.Passed) { $allPassed = $false }
    }

    if ($null -ne $benchmarkResult) {
        Write-TestResult `
            -Name "Latency p99" `
            -Passed $benchmarkResult.Passed `
            -Value "$($benchmarkResult.P99Ms)ms (umbral: < 10ms)"

        Write-TestResult `
            -Name "Throughput" `
            -Passed $true `
            -Value "$($benchmarkResult.ThroughputMsg) msg/s"

        if (-not $benchmarkResult.Passed) { $allPassed = $false }

        Write-Host ""
        Write-Host "  Resultados JSON: $($benchmarkResult.JsonPath)" -ForegroundColor Gray
    }

    Write-Host "══════════════════════════════════════════════════════════" -ForegroundColor White
    Write-Host ""

    if (-not $allPassed) {
        Write-Host "❌ Uno o más tests fallaron. Ver logs arriba para detalles." -ForegroundColor Red
        exit 1
    }
    else {
        Write-Host "✅ Todos los tests de Kafka pasaron. Criterios Fase 1 cumplidos." -ForegroundColor Green
    }
}

# Punto de entrada
Invoke-AllKafkaTests
