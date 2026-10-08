#Requires -Version 7.0
<#
.SYNOPSIS
    Mide Kafka en el cluster: latencia produce -> consume y throughput sostenido.

.DESCRIPTION
    Es la unica forma de cerrar el criterio 1 de la Fase 2 (throughput > 50.000 msg/s) y
    el de latencia de la Fase 1 (p99 < 10 ms): una medicion en local es informativa.

    Como llega el codigo al cluster (decision de M6):
      1. Se levanta un pod efimero (k8s/tools/kafka-benchmark-pod.yaml) con la IMAGEN DEL
         PROYECTO, que ya trae data.* y las dependencias con la version fijada en el build.
      2. Se copian al pod, con kubectl cp, SOLO el codigo de tests/integration y el parquet
         de d00 (los mensajes del benchmark son lecturas reales del TEP). No se instala
         nada en caliente ni se lleva codigo de test a la imagen que se despliega.
      3. Se ejecutan los dos benchmarks como modulos (python -m tests.integration...).
      4. Se recuperan los JSON a tests/results/ y se borra el pod (siempre, aunque falle).

    Lo que se mide va a un topic PROPIO, bench-throughput, con un KafkaUser propio
    (reactorguard-benchmark). Medir en sensor-readings-raw inyectaria cientos de miles de
    lecturas sinteticas en el topic que consume el validador.

    Este script NO juzga el criterio: imprime la cifra, el umbral y el entorno. Quien lo
    da por CUMPLIDO / FALLADO / NO MEDIDO es Verify-Phase2.ps1.

    Cifra de throughput: un proceso de kafka-python tiene un techo medido de unos 12.000
    msg/s, asi que el resultado depende de ese techo y no solo del broker. Se imprime
    junto al limite de CPU del pod para que se pueda interpretar.

    Prerrequisitos:
      - Cluster operativo y kubectl apuntando a el.
      - Install-Kafka.ps1 ejecutado (topic bench-throughput y KafkaUser benchmark) y
        Sync-KafkaCredentials.ps1 ejecutado despues (copia las credenciales al namespace).
      - data/processed/tep/fault_type=00/readings.parquet (Invoke-Pipeline.ps1).
      - La imagen reactorguard-api publicada en el registro (CI).

.PARAMETER Namespace
    Namespace donde corre el pod de benchmark.

.PARAMETER DurationSeconds
    Duracion de la fase medida del throughput.

.PARAMETER Messages
    Mensajes medidos del test de latencia (ademas del calentamiento).

.PARAMETER SkipThroughput
    No ejecuta el benchmark de throughput.

.PARAMETER SkipLatency
    No ejecuta el test de latencia.

.EXAMPLE
    .\tests\integration\Invoke-KafkaTests.ps1
    .\tests\integration\Invoke-KafkaTests.ps1 -SkipThroughput -Verbose
#>

# =============================================================================
# CAPA 1 - CONFIGURACION
# =============================================================================
[CmdletBinding()]
param(
    [string]$Namespace      = "reactorguard-ingestion",
    [string]$Topic          = "bench-throughput",
    [string]$ResultsPath    = "tests/results",
    [int]$DurationSeconds   = 20,
    [int]$Messages          = 100,
    [switch]$SkipThroughput,
    [switch]$SkipLatency
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PodName         = "kafka-benchmark"
$RemoteWorkDir   = "/work"
$RemoteParams    = "/app/params.yaml"
$ThroughputJson  = "kafka_throughput.json"
$LatencyJson     = "kafka_latency.json"
$PayloadRelative = "data/processed/tep/fault_type=00/readings.parquet"
$CredentialSecrets = @("reactorguard-benchmark", "reactorguard-cluster-cluster-ca-cert")
$KafkaNamespace  = "kafka-operator"

# Codigo que viaja al pod (rutas relativas a la raiz del repositorio). Lista explicita: lo
# que no esta aqui no se copia.
$StagedFiles = @(
    "tests/__init__.py",
    "tests/integration/__init__.py",
    "tests/integration/benchmark_kafka.py",
    "tests/integration/benchmark_report.py",
    "tests/integration/test_kafka_connectivity.py"
)

# =============================================================================
# CAPA 2 - UTILIDADES PURAS (sin kubectl)
# =============================================================================

function Get-RepoRoot {
    <#
    .SYNOPSIS
        Devuelve la raiz del repositorio (dos niveles por encima de este script).
    #>
    [CmdletBinding()]
    param()

    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".." ".."))
}

function Get-ThroughputArguments {
    <#
    .SYNOPSIS
        Argumentos de python para el benchmark de throughput del cluster.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$TopicName,
        [Parameter(Mandatory)][int]$Duration,
        [Parameter(Mandatory)][string]$OutputPath,
        [Parameter(Mandatory)][string]$ParamsPath
    )

    return @(
        "python", "-m", "tests.integration.benchmark_kafka",
        "--environment", "cluster",
        "--topic", $TopicName,
        "--duration", "$Duration",
        "--output", $OutputPath,
        "--params", $ParamsPath
    )
}

function Get-LatencyArguments {
    <#
    .SYNOPSIS
        Argumentos de python para el test de latencia del cluster.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$TopicName,
        [Parameter(Mandatory)][int]$Count,
        [Parameter(Mandatory)][string]$OutputPath
    )

    return @(
        "python", "-m", "tests.integration.test_kafka_connectivity",
        "--environment", "cluster",
        "--topic", $TopicName,
        "--messages", "$Count",
        "--output", $OutputPath
    )
}

function Read-BenchmarkReport {
    <#
    .SYNOPSIS
        Lee un informe JSON del formato comun de benchmarks.
    .OUTPUTS
        [psobject] con el informe, o $null si no existe o no se puede leer.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$JsonPath)

    if (-not (Test-Path $JsonPath)) {
        return $null
    }
    try {
        return (Get-Content $JsonPath -Raw -Encoding UTF8 | ConvertFrom-Json)
    }
    catch {
        Write-Warning "No se pudo parsear '$JsonPath': $_"
        return $null
    }
}

function Format-ReportLine {
    <#
    .SYNOPSIS
        Una linea legible por informe: cifra, umbral, entorno y si el valor lo cumple.
    .DESCRIPTION
        Solo describe la medicion; no da el criterio por cumplido. Un valor medido fuera
        del cluster se marca como informativo.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][psobject]$Report)

    $comparison = if ($Report.direction -eq "at_least") { ">=" } else { "<=" }
    $verdict = if ($Report.passed) { "dentro del umbral" } else { "FUERA del umbral" }
    $scope = if ($Report.environment -eq "cluster") { "cluster" } else { "INFORMATIVO (no es cluster)" }
    return ("{0}: {1} {2} (umbral {3} {4}; entorno: {5}) -> {6}" -f
        $Report.criterion, $Report.value, $Report.unit, $comparison, $Report.threshold, $scope, $verdict)
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

# =============================================================================
# CAPA 3 - SERVICIO (kubectl; efectos secundarios aislados)
# =============================================================================

function Assert-Prerequisites {
    <#
    .SYNOPSIS
        Falla pronto y con un mensaje util si falta algo, antes de crear ningun pod.
    #>
    [CmdletBinding()]
    param()

    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        throw "kubectl no esta en el PATH."
    }
    $payload = Join-Path (Get-RepoRoot) $PayloadRelative
    if (-not (Test-Path $payload)) {
        throw "Falta $PayloadRelative. Generalo con .\infra\scripts\Invoke-Pipeline.ps1."
    }
    foreach ($secret in $CredentialSecrets) {
        kubectl get secret $secret --namespace=$Namespace --output=name 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Falta el Secret '$secret' en '$Namespace'. Ejecuta .\infra\scripts\Sync-KafkaCredentials.ps1."
        }
    }
    kubectl get kafkatopic $Topic --namespace=$KafkaNamespace --output=name 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Falta el KafkaTopic '$Topic' en '$KafkaNamespace'. Ejecuta .\infra\scripts\Install-Kafka.ps1."
    }
}

function Deploy-BenchmarkPod {
    <#
    .SYNOPSIS
        Crea el pod de benchmark (borrando uno anterior) y espera a que este Ready.
    #>
    [CmdletBinding()]
    param()

    $manifest = Join-Path (Get-RepoRoot) "k8s" "tools" "kafka-benchmark-pod.yaml"
    kubectl delete pod $PodName --namespace=$Namespace --ignore-not-found --wait=true 2>&1 | Write-Verbose

    Write-Step "Creando el pod $PodName en $Namespace..."
    kubectl apply --filename=$manifest 2>&1 | Write-Verbose
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl apply de kafka-benchmark-pod.yaml fallo."
    }

    # La imagen del proyecto es grande: el primer pull puede tardar varios minutos.
    kubectl wait pod $PodName --namespace=$Namespace --for=condition=Ready --timeout=600s 2>&1 | Write-Verbose
    if ($LASTEXITCODE -ne 0) {
        throw "El pod $PodName no llego a Ready. Revisa: kubectl describe pod $PodName -n $Namespace"
    }
}

function Copy-BenchmarkCode {
    <#
    .SYNOPSIS
        Copia al pod el codigo de tests/integration y el parquet de d00.
    .DESCRIPTION
        Se prepara un directorio temporal con solo los ficheros de $StagedFiles (sin
        __pycache__) y se copia con rutas RELATIVAS: kubectl cp interpreta "C:\..." como
        pod:ruta y falla en Windows.
    #>
    [CmdletBinding()]
    param()

    $root = Get-RepoRoot
    $staging = Join-Path ([System.IO.Path]::GetTempPath()) "reactorguard-bench-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $staging | Out-Null
    Push-Location $staging
    try {
        foreach ($file in $StagedFiles) {
            $target = Join-Path $staging $file
            New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
            Copy-Item (Join-Path $root $file) $target
        }
        Write-Step "Copiando el codigo del benchmark a ${PodName}:$RemoteWorkDir/tests ..."
        kubectl cp "tests" "${Namespace}/${PodName}:${RemoteWorkDir}/tests" 2>&1 | Write-Verbose
        if ($LASTEXITCODE -ne 0) {
            throw "kubectl cp del codigo fallo."
        }

        $remoteDir = "$RemoteWorkDir/" + (Split-Path $PayloadRelative -Parent).Replace("\", "/")
        kubectl exec $PodName --namespace=$Namespace -- mkdir -p $remoteDir 2>&1 | Write-Verbose
        Copy-Item (Join-Path $root $PayloadRelative) (Join-Path $staging "readings.parquet")
        Write-Step "Copiando el parquet de d00 a ${PodName}:$remoteDir ..."
        kubectl cp "readings.parquet" "${Namespace}/${PodName}:${remoteDir}/readings.parquet" 2>&1 | Write-Verbose
        if ($LASTEXITCODE -ne 0) {
            throw "kubectl cp del parquet fallo."
        }
    }
    finally {
        Pop-Location
        Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Invoke-RemoteCommand {
    <#
    .SYNOPSIS
        Ejecuta un comando dentro del pod y devuelve su codigo de salida.
    .DESCRIPTION
        La salida del comando (logs de Python en stderr) se muestra tal cual; el exito
        se decide por el codigo de salida, no por el texto.
    .OUTPUTS
        [int] codigo de salida.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Command)

    kubectl exec $PodName --namespace=$Namespace -- @Command 2>&1 | ForEach-Object { Write-Host "  $_" }
    return $LASTEXITCODE
}

function Receive-Report {
    <#
    .SYNOPSIS
        Copia un informe JSON del pod a tests/results/.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$FileName)

    $resultsDir = Join-Path (Get-RepoRoot) $ResultsPath
    New-Item -ItemType Directory -Path $resultsDir -Force | Out-Null
    Push-Location $resultsDir
    try {
        kubectl cp "${Namespace}/${PodName}:${RemoteWorkDir}/${FileName}" $FileName 2>&1 | Write-Verbose
        if ($LASTEXITCODE -ne 0) {
            throw "No se pudo recuperar $FileName del pod."
        }
    }
    finally {
        Pop-Location
    }
    return (Join-Path $resultsDir $FileName)
}

function Get-PodCpuLimit {
    <#
    .SYNOPSIS
        Limite de CPU del pod de benchmark, para interpretar la cifra de throughput.
    #>
    [CmdletBinding()]
    param()

    $limit = kubectl get pod $PodName --namespace=$Namespace --output=jsonpath='{.spec.containers[0].resources.limits.cpu}' 2>$null
    return $(if ($limit) { $limit } else { "desconocido" })
}

function Remove-BenchmarkPod {
    <#
    .SYNOPSIS
        Borra el pod de benchmark. Se llama siempre desde un finally.
    #>
    [CmdletBinding()]
    param()

    Write-Step "Eliminando el pod $PodName..."
    kubectl delete pod $PodName --namespace=$Namespace --ignore-not-found --wait=false 2>&1 | Write-Verbose
}

# =============================================================================
# CAPA 4 - ORQUESTACION
# =============================================================================

function Invoke-AllKafkaTests {
    <#
    .SYNOPSIS
        Punto de entrada: prepara el pod, ejecuta las mediciones y resume.
    .OUTPUTS
        Codigo de salida: 0 si todas las mediciones pedidas se ejecutaron, 1 si alguna fallo.
    #>
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Magenta
    Write-Host " ReactorGuard - Medicion de Kafka en el cluster" -ForegroundColor Magenta
    Write-Host " Namespace: $Namespace   Topic: $Topic" -ForegroundColor Magenta
    Write-Host "============================================================" -ForegroundColor Magenta

    Assert-Prerequisites

    $failures = [System.Collections.Generic.List[string]]::new()
    $reports = [System.Collections.Generic.List[string]]::new()
    $cpuLimit = "desconocido"

    try {
        Deploy-BenchmarkPod
        Copy-BenchmarkCode
        $cpuLimit = Get-PodCpuLimit

        if (-not $SkipLatency) {
            Write-Step "Midiendo la latencia produce -> consume ($Messages mensajes)..."
            $code = Invoke-RemoteCommand (Get-LatencyArguments `
                    -TopicName $Topic -Count $Messages -OutputPath "$RemoteWorkDir/$LatencyJson")
            if ($code -ne 0) { $failures.Add("latencia (codigo $code)") }
            else { $reports.Add((Receive-Report -FileName $LatencyJson)) }
        }

        if (-not $SkipThroughput) {
            Write-Step "Midiendo el throughput sostenido ($DurationSeconds s)..."
            $code = Invoke-RemoteCommand (Get-ThroughputArguments `
                    -TopicName $Topic -Duration $DurationSeconds `
                    -OutputPath "$RemoteWorkDir/$ThroughputJson" -ParamsPath $RemoteParams)
            if ($code -ne 0) { $failures.Add("throughput (codigo $code)") }
            else { $reports.Add((Receive-Report -FileName $ThroughputJson)) }
        }
    }
    finally {
        Remove-BenchmarkPod
    }

    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor White
    foreach ($path in $reports) {
        $report = Read-BenchmarkReport -JsonPath $path
        if ($null -ne $report) {
            Write-Host ("  " + (Format-ReportLine -Report $report))
            Write-Host "    informe: $path" -ForegroundColor Gray
        }
    }
    Write-Host "  Limite de CPU del pod de benchmark: $cpuLimit (un productor Python usa como mucho ~1,5 nucleos)." -ForegroundColor Gray
    Write-Host "  Este script no da los criterios por cumplidos: eso lo decide Verify-Phase2.ps1." -ForegroundColor Gray
    Write-Host "------------------------------------------------------------" -ForegroundColor White

    if ($failures.Count -gt 0) {
        Write-Host "ERROR: fallaron las mediciones: $($failures -join ', ')" -ForegroundColor Red
        return 1
    }
    return 0
}

# Punto de entrada. No se ejecuta si el fichero se carga con dot-sourcing (pruebas de la capa 2).
if ($MyInvocation.InvocationName -ne ".") {
    try {
        exit (Invoke-AllKafkaTests)
    }
    catch {
        Write-Host ""
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
}
