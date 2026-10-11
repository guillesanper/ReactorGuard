#Requires -Version 7.0
<#
.SYNOPSIS
    Verifica los cuatro criterios de exito de la Fase 2 y los entregables que la acompanan.

.DESCRIPTION
    Cada criterio recibe uno de TRES estados, no dos:

        CUMPLIDO   hay evidencia medida y cumple el umbral
        FALLADO    hay evidencia medida y NO cumple el umbral (o la evidencia es
                   incoherente: un hash que no coincide, un test que falla)
        NO MEDIDO  no hay evidencia valida: falta el dato, el test se salto, la
                   medicion es local y el criterio exige el cluster

    NO MEDIDO no es un fallo y tampoco un exito. Confundirlo con FALLADO afirma una
    precision que nadie ha calculado; confundirlo con CUMPLIDO es la deuda que la
    Fase 1 arrastro (criterios declarados cerrados sin haberse medido).

    Criterios (bloquean el codigo de salida):
        1  Throughput Kafka > 50.000 msg/s    tests/results/kafka_throughput.json
           Solo CUMPLIDO o FALLADO si environment == cluster. Una medicion local es
           informativa y se muestra, pero deja el criterio en NO MEDIDO.
        2  Precision de stuck > 95 %          pytest de T3.6 e invariantes de streaming
           El exit code de pytest NO basta: un skipif devuelve 0. Se ejecuta con
           --junitxml y se cuentan passed y skipped; con skipped > 0 el criterio
           queda NO MEDIDO.
        3  Latencia de features < 100 ms      tests/results/feature_latency.json
           Misma regla de entorno que el criterio 1.
        4  `dvc repro` reproducible           dvc status + dvc.lock (+ -FromScratch)
           Estado limpio y dvc.lock completo -> CUMPLIDO. Una salida modificada o un
           dvc.lock sin etapas -> FALLADO. Dependencias cambiadas o salidas ausentes
           -> NO MEDIDO (el pipeline esta desactualizado, no demostrado roto).

    Entregables de aviso (se reportan con los mismos tres estados y NO bloquean):
        Pods streamer y validador Running, `feast apply` registrado, dashboard.

    Codigos de salida:
        0  todos los criterios CUMPLIDOS
        1  algun criterio FALLADO
        2  ninguno FALLADO, pero alguno NO MEDIDO
        3  error interno del verificador (no es un veredicto sobre la Fase 2)

    Informe: tests/results/phase2_verification.json (esta en .gitignore).

    Cuatro capas:
        Capa 1 - Configuracion : parametros, umbrales y rutas
        Capa 2 - Utilidades    : funciones puras que deciden el estado de cada criterio
        Capa 3 - Servicio      : pytest, dvc, kubectl, git y el registro de Feast
        Capa 4 - Orquestacion  : evalua, imprime, escribe el informe y devuelve el codigo

    Este script no toca GCP: con -SkipCluster ni siquiera llama a kubectl, y sin ese
    parametro solo consulta el contexto actual si pertenece al proyecto $ProjectId.

.PARAMETER ProjectId
    Proyecto GCP. El contexto de kubectl debe ser gke_<ProjectId>_... para que se
    consulte el cluster.

.PARAMETER ProjectRoot
    Raiz del repositorio. Por defecto, dos niveles por encima de este script.

.PARAMETER ResultsDir
    Directorio de los informes de medicion (relativo a ProjectRoot).

.PARAMETER PythonPath
    Interprete con el que se ejecuta pytest. Por defecto, el del venv del proyecto.

.PARAMETER Namespace
    Namespace de los pods del streamer y del validador.

.PARAMETER SkipPytest
    No ejecuta pytest: usa el informe JUnit que haya en ResultsDir. Sin informe, el
    criterio 2 queda NO MEDIDO.

.PARAMETER SkipCluster
    No llama a kubectl. Los entregables que dependen del cluster quedan NO MEDIDO.

.PARAMETER FromScratch
    Ademas, rehace el pipeline entero en una copia temporal (sin dvc.lock, con
    descarga) y compara los hashes de las salidas. Tarda minutos y necesita red.
    Nunca toca data/ del repositorio.

.EXAMPLE
    .\infra\scripts\Verify-Phase2.ps1
    .\infra\scripts\Verify-Phase2.ps1 -SkipCluster
    .\infra\scripts\Verify-Phase2.ps1 -SkipPytest -SkipCluster -FromScratch
#>

# =============================================================================
# CAPA 1 - CONFIGURACION
# =============================================================================
[CmdletBinding()]
param(
    [string]$ProjectId   = "sentinel-platform-485714",
    [string]$ProjectRoot = "",
    [string]$ResultsDir  = "tests/results",
    [string]$PythonPath  = "",
    [string]$Namespace   = "reactorguard-ingestion",
    [switch]$SkipPytest,
    [switch]$SkipCluster,
    [switch]$FromScratch
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Estados. NO MEDIDO es ausencia de evidencia, no un fallo.
$StateMet         = "CUMPLIDO"
$StateFailed      = "FALLADO"
$StateNotMeasured = "NO MEDIDO"

# Codigos de salida.
$ExitMet         = 0
$ExitFailed      = 1
$ExitNotMeasured = 2
$ExitError       = 3

# Umbrales de los criterios (Plan_Fases, Fase 2).
$ThroughputThreshold     = 50000.0   # mensajes por segundo, sostenidos
$LatencyThresholdMs      = 100.0     # p99 de FeaturePipeline.transform sobre una ventana
$StuckPrecisionThreshold = 0.95

# Evidencia que cada criterio lee, relativa a ResultsDir.
$ThroughputReportFile = "kafka_throughput.json"
$LatencyReportFile    = "feature_latency.json"
$MetricsFile          = "validator_metrics_tep.json"
$JUnitFile            = "phase2_stuck.junit.xml"
$ReportFile           = "phase2_verification.json"

# Suite del criterio 2, relativa a ProjectRoot. El primer fichero escribe $MetricsFile;
# el segundo comprueba que el streaming conserva el orden por sensor.
$StuckTestFiles = @(
    "tests/unit/test_validator_on_tep.py",
    "tests/unit/test_stream_invariants.py"
)
$MetricsTestName = "test_metrics_report_is_written"
$StuckDataFiles  = @(
    "data/processed/tep/fault_type=00/readings.parquet",
    "data/processed/tep/fault_type=21/readings.parquet"
)

# Entregables.
$ExpectedDeployments = @("tep-streamer", "sensor-validator")
$DashboardFile       = "observability/grafana/dashboards/data-pipeline.json"
$DashboardNamespace  = "reactorguard-observability"
$DashboardLabel      = "grafana_dashboard=1"
$FeastConfigFile     = "ml/features/store/feature_store.local.yaml"
$FeastView           = "sensor_features"
$FeastService        = "sensor_analysis"

# =============================================================================
# CAPA 2 - UTILIDADES PURAS (deciden estados; no llaman a ningun programa)
# =============================================================================

function Get-JsonProperty {
    <#
    .SYNOPSIS
        Lee una propiedad de un objeto JSON sin fallar si falta (StrictMode lo haria).
    #>
    [CmdletBinding()]
    param(
        [object]$Object,
        [Parameter(Mandatory)][string]$Name,
        [object]$Default = $null
    )

    if ($null -eq $Object) { return $Default }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Test-IsNumber {
    <#
    .SYNOPSIS
        Verdadero si el valor es un numero JSON (no un booleano ni una cadena).
    #>
    [CmdletBinding()]
    param([object]$Value)

    return ($Value -is [double]) -or ($Value -is [int]) -or ($Value -is [long]) -or ($Value -is [decimal])
}

function Format-Number {
    <#
    .SYNOPSIS
        Formatea un numero con punto decimal, independiente de la cultura del sistema.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][double]$Value,
        [string]$Format = "0.####"
    )

    return $Value.ToString($Format, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Resolve-ProjectPath {
    <#
    .SYNOPSIS
        Resuelve una ruta contra la raiz del proyecto si es relativa.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Path
    )

    if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Path))
}

function New-Verdict {
    <#
    .SYNOPSIS
        Construye el veredicto de un criterio o entregable.
    .OUTPUTS
        [ordered] con id, name, blocking, state, detail, evidence y warnings.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Id,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][bool]$Blocking,
        [Parameter(Mandatory)][string]$State,
        [Parameter(Mandatory)][string]$Detail,
        [System.Collections.IDictionary]$Evidence = @{},
        [string[]]$Warnings = @()
    )

    if ($State -notin @($StateMet, $StateFailed, $StateNotMeasured)) {
        throw "Estado desconocido '$State'."
    }
    return [ordered]@{
        id       = $Id
        name     = $Name
        blocking = $Blocking
        state    = $State
        detail   = $Detail
        evidence = $Evidence
        warnings = @($Warnings)
    }
}

function Read-JsonFile {
    <#
    .SYNOPSIS
        Lee un fichero JSON y distingue ausente, ilegible y correcto.
    .OUTPUTS
        [hashtable] con Found [bool], Data [object], Error [string] y WrittenUtc [datetime].
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return @{ Found = $false; Data = $null; Error = "no existe $Path"; WrittenUtc = $null }
    }
    $written = (Get-Item -LiteralPath $Path).LastWriteTimeUtc
    try {
        $data = Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    catch {
        return @{ Found = $false; Data = $null; Error = "no se pudo parsear ${Path}: $($_.Exception.Message)"; WrittenUtc = $written }
    }
    if ($null -eq $data) {
        return @{ Found = $false; Data = $null; Error = "$Path esta vacio"; WrittenUtc = $written }
    }
    return @{ Found = $true; Data = $data; Error = ""; WrittenUtc = $written }
}

function Get-HeadBase {
    <#
    .SYNOPSIS
        Devuelve un hash de commit sin el sufijo -dirty, o cadena vacia si es desconocido.
    #>
    [CmdletBinding()]
    param([string]$Commit)

    if ([string]::IsNullOrWhiteSpace($Commit) -or $Commit -eq "unknown") { return "" }
    return ($Commit -replace "-dirty$", "")
}

function Resolve-BenchmarkVerdict {
    <#
    .SYNOPSIS
        Estado de un criterio que lee un informe de benchmark (criterios 1 y 3).
    .DESCRIPTION
        El valor se compara con el umbral del SCRIPT, no con el que declare el
        informe, y `passed` se recalcula: un informe no puede aprobarse a si mismo.
        Solo environment == cluster da CUMPLIDO o FALLADO.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Id,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Criterion,
        [Parameter(Mandatory)][double]$Threshold,
        [Parameter(Mandatory)][ValidateSet("at_least", "at_most")][string]$Direction,
        [Parameter(Mandatory)][string]$Unit,
        [Parameter(Mandatory)][string]$HowToMeasure,
        [Parameter(Mandatory)][hashtable]$Read,
        [string]$HeadCommit = ""
    )

    if (-not $Read.Found) {
        return New-Verdict -Id $Id -Name $Name -Blocking $true -State $StateNotMeasured `
            -Detail "Sin medicion: $($Read.Error). $HowToMeasure"
    }
    $report = $Read.Data

    $missing = @("schema_version", "criterion", "environment", "value") |
        Where-Object { $null -eq (Get-JsonProperty -Object $report -Name $_) }
    if (@($missing).Count -gt 0) {
        return New-Verdict -Id $Id -Name $Name -Blocking $true -State $StateNotMeasured `
            -Detail "Informe invalido: faltan los campos $(@($missing) -join ', '). $HowToMeasure"
    }
    $reportedCriterion = [string](Get-JsonProperty -Object $report -Name "criterion")
    if ($reportedCriterion -ne $Criterion) {
        return New-Verdict -Id $Id -Name $Name -Blocking $true -State $StateNotMeasured `
            -Detail "Informe de otro criterio ('$reportedCriterion', se esperaba '$Criterion')."
    }
    $value = Get-JsonProperty -Object $report -Name "value"
    if (-not (Test-IsNumber $value)) {
        return New-Verdict -Id $Id -Name $Name -Blocking $true -State $StateNotMeasured `
            -Detail "Informe invalido: value no es numerico."
    }

    $environment = [string](Get-JsonProperty -Object $report -Name "environment")
    $commit      = [string](Get-JsonProperty -Object $report -Name "commit" -Default "unknown")
    $relation    = if ($Direction -eq "at_least") { ">=" } else { "<=" }
    $met         = if ($Direction -eq "at_least") { [double]$value -ge $Threshold } else { [double]$value -le $Threshold }
    $measured    = "$(Format-Number ([double]$value)) $Unit (umbral $relation $(Format-Number $Threshold))"

    $warnings = [System.Collections.Generic.List[string]]::new()
    $reportThreshold = Get-JsonProperty -Object $report -Name "threshold"
    if ((Test-IsNumber $reportThreshold) -and ([double]$reportThreshold -ne $Threshold)) {
        $warnings.Add("El informe declara umbral $(Format-Number ([double]$reportThreshold)); se usa $(Format-Number $Threshold).")
    }
    $reportPassed = Get-JsonProperty -Object $report -Name "passed"
    if (($reportPassed -is [bool]) -and ($reportPassed -ne $met)) {
        $warnings.Add("El informe dice passed=$reportPassed pero value y umbral dan $met; se usa el recalculo.")
    }
    $headBase = Get-HeadBase -Commit $HeadCommit
    $measuredBase = Get-HeadBase -Commit $commit
    if ($headBase -and $measuredBase -and -not ($headBase.StartsWith($measuredBase) -or $measuredBase.StartsWith($headBase))) {
        $warnings.Add("Medido en el commit $commit; HEAD es $HeadCommit.")
    }
    if ($commit -like "*-dirty") {
        $warnings.Add("Medido con el arbol sin commitear ($commit).")
    }

    $evidence = [ordered]@{
        environment = $environment
        value       = $value
        threshold   = $Threshold
        unit        = $Unit
        commit      = $commit
        measured_at = [string](Get-JsonProperty -Object $report -Name "measured_at" -Default "")
    }

    switch ($environment) {
        "cluster" {
            $details = Get-JsonProperty -Object $report -Name "details"
            $servers = @(Get-JsonProperty -Object $details -Name "bootstrap_servers" -Default @())
            $loopback = @($servers | Where-Object { [string]$_ -match '^(127\.0\.0\.1|localhost|\[?::1\]?)(:|$)' })
            if ($loopback.Count -gt 0) {
                return New-Verdict -Id $Id -Name $Name -Blocking $true -State $StateNotMeasured `
                    -Detail "El informe dice cluster pero el bootstrap es local ($($loopback -join ', ')): $measured." `
                    -Evidence $evidence -Warnings $warnings
            }
            $state = if ($met) { $StateMet } else { $StateFailed }
            return New-Verdict -Id $Id -Name $Name -Blocking $true -State $state `
                -Detail "Medido en el cluster: $measured." -Evidence $evidence -Warnings $warnings
        }
        "local" {
            return New-Verdict -Id $Id -Name $Name -Blocking $true -State $StateNotMeasured `
                -Detail "Medicion LOCAL, informativa: $measured. Solo una medicion en el cluster cierra el criterio. $HowToMeasure" `
                -Evidence $evidence -Warnings $warnings
        }
        default {
            return New-Verdict -Id $Id -Name $Name -Blocking $true -State $StateNotMeasured `
                -Detail "Entorno '$environment' no reconocido (se espera local o cluster)." `
                -Evidence $evidence -Warnings $warnings
        }
    }
}

function Read-JUnitSummary {
    <#
    .SYNOPSIS
        Cuenta passed, failed, error y skipped de un informe JUnit de pytest.
    .OUTPUTS
        [pscustomobject], o $null si el fichero no existe.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    [xml]$document = Get-Content -LiteralPath $Path -Raw -Encoding UTF8

    $passedNames = [System.Collections.Generic.List[string]]::new()
    $failedNames = [System.Collections.Generic.List[string]]::new()
    $skipReasons = [System.Collections.Generic.List[string]]::new()
    $errors = 0
    $skipped = 0
    foreach ($case in $document.SelectNodes("//testcase")) {
        $name = "$($case.GetAttribute('classname')).$($case.GetAttribute('name'))"
        $skip = $case.SelectSingleNode("skipped")
        if ($null -ne $case.SelectSingleNode("error")) {
            $errors++
            $failedNames.Add($name)
        }
        elseif ($null -ne $case.SelectSingleNode("failure")) {
            $failedNames.Add($name)
        }
        elseif ($null -ne $skip) {
            $skipped++
            $skipReasons.Add($skip.GetAttribute("message"))
        }
        else {
            $passedNames.Add($name)
        }
    }

    # pytest escribe el instante de arranque con desfase horario; sin el, se asume local.
    $startUtc = $null
    $suite = $document.SelectSingleNode("//testsuite")
    if ($null -ne $suite) {
        $stamp = $suite.GetAttribute("timestamp")
        $parsed = [System.DateTimeOffset]::MinValue
        if ($stamp -and [System.DateTimeOffset]::TryParse(
                $stamp, [System.Globalization.CultureInfo]::InvariantCulture,
                [System.Globalization.DateTimeStyles]::AssumeLocal, [ref]$parsed)) {
            $startUtc = $parsed.UtcDateTime
        }
    }

    return [pscustomobject]@{
        Passed      = $passedNames.Count
        Failed      = $failedNames.Count - $errors
        Errors      = $errors
        Skipped     = $skipped
        PassedNames = $passedNames.ToArray()
        FailedNames = $failedNames.ToArray()
        SkipReasons = $skipReasons.ToArray()
        StartUtc    = $startUtc
    }
}

function Resolve-StuckPrecisionVerdict {
    <#
    .SYNOPSIS
        Estado del criterio 2: distingue passed de skipped y no se fia del exit code.
    .PARAMETER DataPresent
        Si el parquet adaptado del TEP esta poblado.
    .PARAMETER JUnit
        Resultado de Read-JUnitSummary, o $null si pytest no dejo informe.
    .PARAMETER Metrics
        Resultado de Read-JsonFile sobre validator_metrics_tep.json.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][bool]$DataPresent,
        [AllowNull()][object]$JUnit,
        [Parameter(Mandatory)][hashtable]$Metrics
    )

    $id = "2"
    $name = "Precision del detector de stuck > 95 %"
    $howTo = "Ejecuta .\infra\scripts\Invoke-Pipeline.ps1 y vuelve a lanzar este script."

    if (-not $DataPresent) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Faltan los datos (data/processed/tep). Los tests se saltarian, no fallarian. $howTo"
    }
    if ($null -eq $JUnit) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Sin informe JUnit: pytest no se ejecuto (-SkipPytest sin un informe previo)."
    }
    if (($JUnit.Failed + $JUnit.Errors) -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
            -Detail "$($JUnit.Failed + $JUnit.Errors) tests fallan: $($JUnit.FailedNames -join '; ')" `
            -Evidence ([ordered]@{ passed = $JUnit.Passed; failed = $JUnit.Failed; errors = $JUnit.Errors; skipped = $JUnit.Skipped })
    }
    if ($JUnit.Skipped -gt 0) {
        $reason = if ($JUnit.SkipReasons.Count -gt 0) { " Motivo: $($JUnit.SkipReasons[0])" } else { "" }
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "$($JUnit.Skipped) tests SALTADOS de $($JUnit.Passed + $JUnit.Skipped): no se midio nada.$reason" `
            -Evidence ([ordered]@{ passed = $JUnit.Passed; skipped = $JUnit.Skipped })
    }
    if ($JUnit.Passed -eq 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "El informe JUnit no contiene ningun test ejecutado."
    }
    $wroteMetrics = @($JUnit.PassedNames | Where-Object { $_ -like "*$MetricsTestName" }).Count -gt 0
    if (-not $wroteMetrics) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "El test $MetricsTestName no se ejecuto: no hay metricas de esta ejecucion."
    }
    if (-not $Metrics.Found) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Los tests pasan pero falta el informe de metricas: $($Metrics.Error)."
    }
    if (($null -ne $JUnit.StartUtc) -and ($null -ne $Metrics.WrittenUtc) -and
        ($Metrics.WrittenUtc -lt $JUnit.StartUtc.AddSeconds(-2))) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "El informe de metricas es anterior a esta ejecucion de pytest: es una cifra vieja."
    }

    $gate = Get-JsonProperty -Object $Metrics.Data -Name "gate"
    $measured = Get-JsonProperty -Object $gate -Name "measured"
    if (-not (Test-IsNumber $measured)) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "El informe de metricas no trae gate.measured."
    }
    $cross = Get-JsonProperty -Object $Metrics.Data -Name "cross_validation"
    $evidence = [ordered]@{
        precision             = $measured
        threshold             = $StuckPrecisionThreshold
        tests_passed          = $JUnit.Passed
        cross_validation_pairs = Get-JsonProperty -Object $cross -Name "stuck_pairs_detected"
    }
    $text = "precision de stuck $(Format-Number ([double]$measured) '0.0000') (umbral >= $(Format-Number $StuckPrecisionThreshold '0.00')), $($JUnit.Passed) tests ejecutados y 0 saltados"
    $state = if ([double]$measured -ge $StuckPrecisionThreshold) { $StateMet } else { $StateFailed }
    return New-Verdict -Id $id -Name $name -Blocking $true -State $state -Detail $text -Evidence $evidence
}

function Get-OutputTail {
    <#
    .SYNOPSIS
        Ultimas lineas utiles de la salida de un programa, sin barras de progreso (tqdm).
    #>
    [CmdletBinding()]
    param(
        [AllowEmptyString()][Parameter(Mandatory)][string]$Text,
        [int]$Lines = 8
    )

    $useful = @(($Text -split "\r?\n") | ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and $_ -notmatch '%\||it/s\]|B/s\]|^\S+\.dat:' })
    return ($useful | Select-Object -Last $Lines) -join " | "
}

function Test-NetworkFailure {
    <#
    .SYNOPSIS
        Verdadero si la salida describe un fallo de red y no un fallo del pipeline.
    #>
    [CmdletBinding()]
    param([AllowEmptyString()][Parameter(Mandatory)][string]$Text)

    return ($Text -match '(?i)ConnectionError|Max retries exceeded|NewConnectionError|getaddrinfo|Name or service not known|Temporary failure in name resolution|ReadTimeout|ConnectTimeout|SSLError|URLError|No such host|actively refused|timed out')
}

function Get-StageNames {
    <#
    .SYNOPSIS
        Nombres de etapa de un dvc.yaml o dvc.lock (claves de segundo nivel de `stages:`).
    #>
    [CmdletBinding()]
    param([AllowEmptyString()][Parameter(Mandatory)][string]$Text)

    $names = [System.Collections.Generic.List[string]]::new()
    $inStages = $false
    foreach ($line in ($Text -split "\r?\n")) {
        if ($line -match '^stages:\s*$') { $inStages = $true; continue }
        if ($inStages -and $line -match '^\S') { $inStages = $false }
        if ($inStages -and $line -match '^  ([A-Za-z0-9_-]+):\s*$') { $names.Add($Matches[1]) }
    }
    return $names.ToArray()
}

function Get-MissingLockStages {
    <#
    .SYNOPSIS
        Etapas de dvc.yaml que dvc.lock no registra.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$DvcYamlText,
        [AllowEmptyString()][string]$DvcLockText = ""
    )

    $declared = @(Get-StageNames -Text $DvcYamlText)
    $locked = @(Get-StageNames -Text $DvcLockText)
    return @($declared | Where-Object { $_ -notin $locked })
}

function Get-LockOutputHashes {
    <#
    .SYNOPSIS
        Mapa "etapa:ruta" -> md5 de las salidas registradas en un dvc.lock.
    #>
    [CmdletBinding()]
    param([AllowEmptyString()][Parameter(Mandatory)][string]$Text)

    $hashes = @{}
    $stage = ""
    $inOuts = $false
    $path = ""
    $inStages = $false
    foreach ($line in ($Text -split "\r?\n")) {
        if ($line -match '^stages:\s*$') { $inStages = $true; continue }
        if (-not $inStages) { continue }
        if ($line -match '^  ([A-Za-z0-9_-]+):\s*$') { $stage = $Matches[1]; $inOuts = $false; continue }
        if ($line -match '^    outs:\s*$') { $inOuts = $true; continue }
        # Las entradas de lista de dvc.lock van al mismo nivel que su clave ("    - path:"):
        # solo una clave nueva ("    deps:", "    params:") cierra la seccion de salidas.
        if ($line -match '^    [A-Za-z_]') { $inOuts = $false; continue }
        if (-not $inOuts) { continue }
        if ($line -match '^    - path: (.+?)\s*$') { $path = $Matches[1]; continue }
        if ($path -and $line -match '^\s+md5: (\S+)\s*$') {
            $hashes["${stage}:${path}"] = $Matches[1]
            $path = ""
        }
    }
    return $hashes
}

function Compare-LockHashes {
    <#
    .SYNOPSIS
        Salidas cuyo hash difiere entre el dvc.lock del repositorio y el de una copia.
    .OUTPUTS
        [string[]] con "etapa:ruta (esperado X, obtenido Y)"; vacio si coinciden todas.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Expected,
        [Parameter(Mandatory)][hashtable]$Actual
    )

    $differences = [System.Collections.Generic.List[string]]::new()
    foreach ($key in ($Expected.Keys | Sort-Object)) {
        if (-not $Actual.ContainsKey($key)) {
            $differences.Add("$key (esperado $($Expected[$key]), no se produjo)")
        }
        elseif ($Actual[$key] -ne $Expected[$key]) {
            $differences.Add("$key (esperado $($Expected[$key]), obtenido $($Actual[$key]))")
        }
    }
    return $differences.ToArray()
}

function Get-DvcStatusSummary {
    <#
    .SYNOPSIS
        Clasifica la salida de `dvc status --json` (ya parseada como hashtable).
    .OUTPUTS
        [pscustomobject] con ChangedDeps, OutsModified y OutsMissing (listas de "etapa: ruta").
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][System.Collections.IDictionary]$Status)

    $deps = [System.Collections.Generic.List[string]]::new()
    $modified = [System.Collections.Generic.List[string]]::new()
    $missing = [System.Collections.Generic.List[string]]::new()
    foreach ($stage in $Status.Keys) {
        foreach ($entry in @($Status[$stage])) {
            if ($entry -isnot [System.Collections.IDictionary]) {
                # Cadenas como "always changed": la etapa no es verificable.
                $deps.Add("${stage}: $entry")
                continue
            }
            foreach ($section in $entry.Keys) {
                $items = $entry[$section]
                foreach ($path in $items.Keys) {
                    $label = "${stage}: $path ($($items[$path]))"
                    if ($section -eq "changed deps") { $deps.Add($label) }
                    elseif ($items[$path] -eq "modified") { $modified.Add($label) }
                    else { $missing.Add($label) }
                }
            }
        }
    }
    return [pscustomobject]@{
        ChangedDeps = $deps.ToArray()
        OutsModified = $modified.ToArray()
        OutsMissing = $missing.ToArray()
    }
}

function Resolve-DvcVerdict {
    <#
    .SYNOPSIS
        Estado del criterio 4.
    .PARAMETER MissingLockStages
        Etapas de dvc.yaml ausentes de dvc.lock.
    .PARAMETER StatusSummary
        Resultado de Get-DvcStatusSummary, o $null si dvc status no se pudo ejecutar.
    .PARAMETER StatusProblem
        Motivo por el que dvc status no se pudo ejecutar (cuando StatusSummary es $null).
    .PARAMETER Scratch
        Resultado de Invoke-FromScratch, o $null si no se pidio.
    #>
    [CmdletBinding()]
    param(
        [string[]]$MissingLockStages = @(),
        [AllowNull()][object]$StatusSummary,
        [string]$StatusProblem = "",
        [AllowNull()][object]$Scratch = $null
    )

    $id = "4"
    $name = "dvc repro reproducible desde cero"
    $howTo = "Ejecuta .\infra\scripts\Invoke-Pipeline.ps1 (nunca dvc repro a pelo) y repite."

    if (@($MissingLockStages).Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
            -Detail "dvc.lock no registra las etapas: $(@($MissingLockStages) -join ', '). $howTo"
    }
    if ($null -eq $StatusSummary) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "No se pudo ejecutar dvc status: $StatusProblem"
    }
    if (@($StatusSummary.OutsModified).Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
            -Detail "Salidas que no corresponden a su dvc.lock: $(@($StatusSummary.OutsModified) -join '; '). $howTo"
    }
    if ((@($StatusSummary.ChangedDeps).Count + @($StatusSummary.OutsMissing).Count) -gt 0) {
        $all = @($StatusSummary.ChangedDeps) + @($StatusSummary.OutsMissing)
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Pipeline desactualizado o sin datos, no demostrado roto: $($all -join '; '). $howTo"
    }

    $evidence = [ordered]@{ dvc_status = "clean"; from_scratch = "not requested" }
    if ($null -eq $Scratch) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateMet `
            -Detail "dvc status limpio y dvc.lock registra todas las etapas." -Evidence $evidence
    }
    $evidence["from_scratch"] = $Scratch.Status
    switch ($Scratch.Status) {
        "matched" {
            return New-Verdict -Id $id -Name $name -Blocking $true -State $StateMet `
                -Detail "dvc status limpio y la reproduccion desde cero coincide bit a bit: $($Scratch.Detail)" `
                -Evidence $evidence
        }
        "mismatch" {
            return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
                -Detail "La reproduccion desde cero produce salidas distintas: $($Scratch.Detail)" -Evidence $evidence
        }
        "failed" {
            return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
                -Detail "dvc repro desde cero fallo: $($Scratch.Detail)" -Evidence $evidence
        }
        default {
            return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
                -Detail "No se pudo completar la reproduccion desde cero: $($Scratch.Detail)" -Evidence $evidence
        }
    }
}

function Resolve-DeploymentsVerdict {
    <#
    .SYNOPSIS
        Estado del entregable "streamer y validador Running" a partir de los Deployments.
    .PARAMETER Deployments
        Salida de `kubectl get deployments -o json` ya parseada, o $null si no hay cluster.
    .PARAMETER Problem
        Por que no se consulto el cluster (cuando Deployments es $null).
    #>
    [CmdletBinding()]
    param(
        [AllowNull()][object]$Deployments,
        [string]$Problem = ""
    )

    $id = "pods"
    $name = "Pods del streamer y del validador Running"
    if ($null -eq $Deployments) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured -Detail $Problem
    }
    $items = @(Get-JsonProperty -Object $Deployments -Name "items" -Default @())
    $byName = @{}
    foreach ($item in $items) {
        $itemName = [string](Get-JsonProperty -Object (Get-JsonProperty -Object $item -Name "metadata") -Name "name")
        $byName[$itemName] = $item
    }

    $absent = [System.Collections.Generic.List[string]]::new()
    $notReady = [System.Collections.Generic.List[string]]::new()
    $ready = [System.Collections.Generic.List[string]]::new()
    foreach ($expected in $ExpectedDeployments) {
        if (-not $byName.ContainsKey($expected)) { $absent.Add($expected); continue }
        $item = $byName[$expected]
        $desired = Get-JsonProperty -Object (Get-JsonProperty -Object $item -Name "spec") -Name "replicas" -Default 0
        $have = Get-JsonProperty -Object (Get-JsonProperty -Object $item -Name "status") -Name "readyReplicas" -Default 0
        if (($desired -gt 0) -and ($have -eq $desired)) { $ready.Add("$expected $have/$desired") }
        else { $notReady.Add("$expected $have/$desired") }
    }

    $evidence = [ordered]@{ ready = $ready.ToArray(); not_ready = $notReady.ToArray(); absent = $absent.ToArray() }
    if ($absent.Count -eq $ExpectedDeployments.Count) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured `
            -Detail "No hay Deployments $($ExpectedDeployments -join ', '): el streaming no esta desplegado." -Evidence $evidence
    }
    if (($absent.Count + $notReady.Count) -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateFailed `
            -Detail "No estan todos listos. Ausentes: $($absent -join ', ' ). Sin replicas listas: $($notReady -join ', ')." -Evidence $evidence
    }
    return New-Verdict -Id $id -Name $name -Blocking $false -State $StateMet `
        -Detail "Replicas listas: $($ready -join ', ')." -Evidence $evidence
}

function Resolve-DashboardVerdict {
    <#
    .SYNOPSIS
        Estado del entregable "dashboard del pipeline de datos".
    .PARAMETER File
        Resultado de Read-JsonFile sobre el JSON del dashboard.
    .PARAMETER ClusterConfigMaps
        Salida de `kubectl get configmap -l grafana_dashboard=1 -o json` parseada, o $null.
    .PARAMETER ClusterProblem
        Por que no se consulto el cluster (cuando ClusterConfigMaps es $null).
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$File,
        [AllowNull()][object]$ClusterConfigMaps = $null,
        [string]$ClusterProblem = ""
    )

    $id = "dashboard"
    $name = "Dashboard de Grafana del pipeline de datos"
    if (-not $File.Found) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateFailed `
            -Detail "El dashboard no es valido: $($File.Error)."
    }
    $uid = [string](Get-JsonProperty -Object $File.Data -Name "uid")
    $panels = @(Get-JsonProperty -Object $File.Data -Name "panels" -Default @())
    $real = @($panels | Where-Object { (Get-JsonProperty -Object $_ -Name "type") -ne "row" })
    if (-not $uid -or $real.Count -eq 0) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateFailed `
            -Detail "El dashboard no tiene uid o no tiene paneles."
    }
    $evidence = [ordered]@{ uid = $uid; panels = $real.Count }
    if ($null -eq $ClusterConfigMaps) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured `
            -Detail "Dashboard valido ($($real.Count) paneles) pero no verificado en el cluster: $ClusterProblem" -Evidence $evidence
    }
    $items = @(Get-JsonProperty -Object $ClusterConfigMaps -Name "items" -Default @())
    if ($items.Count -eq 0) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured `
            -Detail "Dashboard valido ($($real.Count) paneles); ningun ConfigMap con la etiqueta $DashboardLabel en $DashboardNamespace (deuda de despliegue: no hay Grafana en el cluster)." `
            -Evidence $evidence
    }
    return New-Verdict -Id $id -Name $name -Blocking $false -State $StateMet `
        -Detail "Dashboard valido ($($real.Count) paneles) y desplegado como ConfigMap." -Evidence $evidence
}

function Resolve-FeastVerdict {
    <#
    .SYNOPSIS
        Estado del entregable "feast apply" a partir del registro local.
    .PARAMETER RegistryExists
        Si el fichero de registro existe.
    .PARAMETER Probe
        Objeto con feature_views y feature_services leidos del registro, o $null.
    .PARAMETER Problem
        Por que no se pudo leer el registro (cuando Probe es $null).
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][bool]$RegistryExists,
        [AllowNull()][object]$Probe = $null,
        [string]$Problem = ""
    )

    $id = "feast"
    $name = "feast apply registrado"
    if (-not $RegistryExists) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured `
            -Detail "No hay registro de Feast: feast apply no se ha ejecutado en este entorno."
    }
    if ($null -eq $Probe) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured -Detail $Problem
    }
    $views = @(Get-JsonProperty -Object $Probe -Name "feature_views" -Default @())
    $services = @(Get-JsonProperty -Object $Probe -Name "feature_services" -Default @())
    $evidence = [ordered]@{ feature_views = $views; feature_services = $services }
    if (($FeastView -notin $views) -or ($FeastService -notin $services)) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateFailed `
            -Detail "El registro existe pero no tiene $FeastView y $FeastService (tiene vistas: $($views -join ', '))." `
            -Evidence $evidence
    }
    return New-Verdict -Id $id -Name $name -Blocking $false -State $StateMet `
        -Detail "El registro contiene la vista $FeastView y el servicio $FeastService." -Evidence $evidence
}

function Get-ExitCode {
    <#
    .SYNOPSIS
        Codigo de salida a partir de los criterios que bloquean.
    .OUTPUTS
        [int]: 1 si alguno FALLADO, 2 si alguno NO MEDIDO, 0 si todos CUMPLIDOS.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object[]]$Verdicts)

    $blocking = @($Verdicts | Where-Object { $_.blocking })
    if (@($blocking | Where-Object { $_.state -eq $StateFailed }).Count -gt 0) { return $ExitFailed }
    if (@($blocking | Where-Object { $_.state -eq $StateNotMeasured }).Count -gt 0) { return $ExitNotMeasured }
    return $ExitMet
}

function Get-StateCounts {
    <#
    .SYNOPSIS
        Cuenta veredictos por estado.
    #>
    [CmdletBinding()]
    param([AllowEmptyCollection()][object[]]$Verdicts = @())

    return [ordered]@{
        $StateMet         = @($Verdicts | Where-Object { $_.state -eq $StateMet }).Count
        $StateFailed      = @($Verdicts | Where-Object { $_.state -eq $StateFailed }).Count
        $StateNotMeasured = @($Verdicts | Where-Object { $_.state -eq $StateNotMeasured }).Count
    }
}

function New-ReportDocument {
    <#
    .SYNOPSIS
        Documento JSON del informe de verificacion.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][object[]]$Criteria,
        [Parameter(Mandatory)][object[]]$Deliverables,
        [Parameter(Mandatory)][int]$ExitCode,
        [Parameter(Mandatory)][string]$Commit,
        [Parameter(Mandatory)][string]$Project,
        [Parameter(Mandatory)][string]$GeneratedAt
    )

    return [ordered]@{
        schema_version = 1
        generated_at   = $GeneratedAt
        project_id     = $Project
        commit         = $Commit
        exit_code      = $ExitCode
        summary        = [ordered]@{
            criteria     = (Get-StateCounts -Verdicts $Criteria)
            deliverables = (Get-StateCounts -Verdicts $Deliverables)
        }
        criteria       = @($Criteria)
        deliverables   = @($Deliverables)
    }
}

# =============================================================================
# CAPA 3 - SERVICIO (programas externos; efectos secundarios aislados)
# =============================================================================

function Invoke-Native {
    <#
    .SYNOPSIS
        Ejecuta un programa y devuelve su codigo de salida y su salida combinada.
    .DESCRIPTION
        stderr de un programa no es un error de PowerShell: se mezcla en la salida y
        el exito se decide por el codigo de salida. Un programa ausente devuelve -1.
    .OUTPUTS
        [pscustomobject] con ExitCode y Output.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = ""
    )

    $previous = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = 0
    $lines = @()
    $code = 0
    try {
        if ($WorkingDirectory) { Push-Location -LiteralPath $WorkingDirectory }
        try {
            $lines = @(& $FilePath @Arguments 2>&1 | ForEach-Object { "$_" })
            $code = $LASTEXITCODE
        }
        finally {
            if ($WorkingDirectory) { Pop-Location }
        }
    }
    catch {
        $lines = @($_.Exception.Message)
        $code = -1
    }
    finally {
        $ErrorActionPreference = $previous
    }
    return [pscustomobject]@{ ExitCode = $code; Output = ($lines -join "`n") }
}

function Get-GitCommit {
    <#
    .SYNOPSIS
        Commit abreviado de HEAD, con sufijo -dirty si el arbol tiene cambios.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root)

    $head = Invoke-Native -FilePath "git" -Arguments @("-C", $Root, "rev-parse", "--short", "HEAD")
    if ($head.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($head.Output)) { return "unknown" }
    $status = Invoke-Native -FilePath "git" -Arguments @("-C", $Root, "status", "--porcelain")
    $suffix = if ($status.ExitCode -eq 0 -and -not [string]::IsNullOrWhiteSpace($status.Output)) { "-dirty" } else { "" }
    return "$($head.Output.Trim())$suffix"
}

function Invoke-StuckSuite {
    <#
    .SYNOPSIS
        Ejecuta la suite del criterio 2 dejando un informe JUnit.
    .OUTPUTS
        [pscustomobject] con ExitCode, Output y Seconds.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string]$JUnitPath
    )

    $parent = Split-Path -Parent $JUnitPath
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    if (Test-Path -LiteralPath $JUnitPath) { Remove-Item -LiteralPath $JUnitPath -Force }

    $arguments = @("-m", "pytest") + $StuckTestFiles + @("--junitxml=$JUnitPath", "-q", "-p", "no:cacheprovider")
    $started = Get-Date
    $previousPath = $env:PYTHONPATH
    $env:PYTHONPATH = $Root
    try {
        $run = Invoke-Native -FilePath $Python -Arguments $arguments -WorkingDirectory $Root
    }
    finally {
        $env:PYTHONPATH = $previousPath
    }
    return [pscustomobject]@{
        ExitCode = $run.ExitCode
        Output   = $run.Output
        Seconds  = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    }
}

function Get-DvcStatus {
    <#
    .SYNOPSIS
        Ejecuta `dvc status --json` con el venv del proyecto al principio del PATH.
    .OUTPUTS
        [pscustomobject] con Summary (o $null) y Problem.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root)

    # dvc.exe no activa el venv: anteponer, nunca sustituir (igual que Invoke-Pipeline.ps1).
    $scripts = Join-Path $Root ".venv\Scripts"
    $previousPath = $env:PATH
    if (Test-Path -LiteralPath $scripts) { $env:PATH = "$scripts$([System.IO.Path]::PathSeparator)$env:PATH" }
    try {
        if ($null -eq (Get-Command dvc -ErrorAction SilentlyContinue)) {
            return [pscustomobject]@{ Summary = $null; Problem = "dvc no esta en el PATH (instala el venv con Setup-DevEnv.ps1)." }
        }
        $run = Invoke-Native -FilePath "dvc" -Arguments @("status", "--json") -WorkingDirectory $Root
    }
    finally {
        $env:PATH = $previousPath
    }
    if ($run.ExitCode -ne 0) {
        return [pscustomobject]@{ Summary = $null; Problem = "dvc status devolvio $($run.ExitCode): $($run.Output)" }
    }
    try {
        $parsed = $run.Output | ConvertFrom-Json -AsHashtable
    }
    catch {
        return [pscustomobject]@{ Summary = $null; Problem = "dvc status --json no devolvio JSON: $($run.Output)" }
    }
    if ($null -eq $parsed) { $parsed = @{} }
    return [pscustomobject]@{ Summary = (Get-DvcStatusSummary -Status $parsed); Problem = "" }
}

function Invoke-FromScratch {
    <#
    .SYNOPSIS
        Rehace el pipeline en una copia temporal y compara los hashes de las salidas.
    .DESCRIPTION
        Copia los ficheros que git sigue (sin dvc.lock ni .dvc), inicializa DVC sin
        SCM y ejecuta `dvc repro` con todas las etapas pendientes, descarga incluida.
        Jamas toca data/ del repositorio: todo ocurre bajo el directorio temporal,
        que se borra siempre.
    .OUTPUTS
        [pscustomobject] con Status (matched, mismatch, failed, unavailable) y Detail.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root)

    $lockPath = Join-Path $Root "dvc.lock"
    if (-not (Test-Path -LiteralPath $lockPath)) {
        return [pscustomobject]@{ Status = "unavailable"; Detail = "no hay dvc.lock con el que comparar." }
    }
    $listing = Invoke-Native -FilePath "git" -Arguments @("-C", $Root, "ls-files")
    if ($listing.ExitCode -ne 0) {
        return [pscustomobject]@{ Status = "unavailable"; Detail = "git ls-files fallo: $($listing.Output)" }
    }

    $scripts = Join-Path $Root ".venv\Scripts"
    # Nombre CORTO a proposito: la run-cache de DVC crea .dvc\cache\runs\xx\<64>\<64> y con un
    # temporal largo la ruta supera MAX_PATH en Windows (WinError 3 tras ejecutar la etapa).
    $temp = Join-Path ([System.IO.Path]::GetTempPath()) ("rg-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    $previousPath = $env:PATH
    $previousPythonPath = $env:PYTHONPATH
    try {
        New-Item -ItemType Directory -Path $temp | Out-Null
        foreach ($relative in ($listing.Output -split "\r?\n" | Where-Object { $_ })) {
            if ($relative -eq "dvc.lock" -or $relative -like ".dvc/*") { continue }
            $source = Join-Path $Root $relative
            if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { continue }
            $target = Join-Path $temp $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
            Copy-Item -LiteralPath $source -Destination $target
        }

        if (Test-Path -LiteralPath $scripts) { $env:PATH = "$scripts$([System.IO.Path]::PathSeparator)$env:PATH" }
        $env:PYTHONPATH = $temp
        $init = Invoke-Native -FilePath "dvc" -Arguments @("init", "--no-scm", "--quiet") -WorkingDirectory $temp
        if ($init.ExitCode -ne 0) {
            return [pscustomobject]@{ Status = "unavailable"; Detail = "dvc init fallo: $($init.Output)" }
        }
        # --no-run-cache: una etapa con las mismas entradas no debe darse por ejecutada.
        $repro = Invoke-Native -FilePath "dvc" -Arguments @("repro", "--no-run-cache") -WorkingDirectory $temp
        if ($repro.ExitCode -ne 0) {
            $tail = Get-OutputTail -Text $repro.Output
            # Una descarga que falla por la red no demuestra que el pipeline no reproduzca;
            # cualquier otro fallo, incluido uno de download_tep, si.
            if (Test-NetworkFailure -Text $repro.Output) {
                return [pscustomobject]@{ Status = "unavailable"; Detail = "sin red para descargar el dataset: $tail" }
            }
            return [pscustomobject]@{ Status = "failed"; Detail = $tail }
        }
        $newLock = Join-Path $temp "dvc.lock"
        $expected = Get-LockOutputHashes -Text (Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8)
        $actual = Get-LockOutputHashes -Text (Get-Content -LiteralPath $newLock -Raw -Encoding UTF8)
        $differences = @(Compare-LockHashes -Expected $expected -Actual $actual)
        if ($differences.Count -gt 0) {
            return [pscustomobject]@{ Status = "mismatch"; Detail = ($differences -join "; ") }
        }
        return [pscustomobject]@{ Status = "matched"; Detail = "$($expected.Count) salidas con el mismo hash." }
    }
    finally {
        $env:PATH = $previousPath
        $env:PYTHONPATH = $previousPythonPath
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-FeastRegistryPath {
    <#
    .SYNOPSIS
        Ruta absoluta del registro de Feast declarado en la configuracion local.
    .OUTPUTS
        [string], o cadena vacia si la configuracion no existe o no declara registro.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Root)

    $config = Join-Path $Root $FeastConfigFile
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) { return "" }
    $line = Get-Content -LiteralPath $config -Encoding UTF8 | Where-Object { $_ -match '^registry:\s*(\S+)' } | Select-Object -First 1
    if ($null -eq $line) { return "" }
    $null = $line -match '^registry:\s*(\S+)'
    return (Resolve-ProjectPath -Root $Root -Path $Matches[1])
}

function Get-FeastRegistryContents {
    <#
    .SYNOPSIS
        Lee del registro local las vistas y servicios registrados (solo lectura).
    .OUTPUTS
        [pscustomobject] con Probe (o $null) y Problem.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Python
    )

    $script = @'
import json
import sys
from pathlib import Path

from feast import FeatureStore

root = Path(sys.argv[1])
store = FeatureStore(repo_path=str(root), fs_yaml_file=root / sys.argv[2])
print(json.dumps({
    "feature_views": sorted(view.name for view in store.list_feature_views()),
    "feature_services": sorted(service.name for service in store.list_feature_services()),
}))
'@
    $file = Join-Path ([System.IO.Path]::GetTempPath()) ("reactorguard-feast-probe-" + [guid]::NewGuid().ToString("N") + ".py")
    try {
        Set-Content -LiteralPath $file -Value $script -Encoding UTF8
        $run = Invoke-Native -FilePath $Python -Arguments @($file, $Root, $FeastConfigFile) -WorkingDirectory $Root
    }
    finally {
        Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue
    }
    if ($run.ExitCode -ne 0) {
        $tail = (($run.Output -split "\r?\n") | Select-Object -Last 3) -join " | "
        return [pscustomobject]@{ Probe = $null; Problem = "No se pudo leer el registro de Feast: $tail" }
    }
    $json = ($run.Output -split "\r?\n" | Where-Object { $_.TrimStart().StartsWith("{") } | Select-Object -Last 1)
    try {
        return [pscustomobject]@{ Probe = ($json | ConvertFrom-Json); Problem = "" }
    }
    catch {
        return [pscustomobject]@{ Probe = $null; Problem = "El lector del registro no devolvio JSON: $($run.Output)" }
    }
}

function Get-ClusterAccess {
    <#
    .SYNOPSIS
        Decide si se puede consultar el cluster del proyecto con el contexto actual.
    .OUTPUTS
        [pscustomobject] con Available [bool] y Reason.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Project,
        [switch]$Skip
    )

    if ($Skip) {
        return [pscustomobject]@{ Available = $false; Reason = "omitido con -SkipCluster." }
    }
    if ($null -eq (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        return [pscustomobject]@{ Available = $false; Reason = "kubectl no esta en el PATH." }
    }
    $context = Invoke-Native -FilePath "kubectl" -Arguments @("config", "current-context")
    if ($context.ExitCode -ne 0 -or [string]::IsNullOrWhiteSpace($context.Output)) {
        return [pscustomobject]@{ Available = $false; Reason = "kubectl no tiene un contexto actual." }
    }
    # Un contexto ajeno (docker-desktop, otro proyecto) no es evidencia de este despliegue.
    $name = $context.Output.Trim()
    if ($name -notlike "gke_${Project}_*") {
        return [pscustomobject]@{ Available = $false; Reason = "el contexto de kubectl '$name' no pertenece al proyecto $Project." }
    }
    return [pscustomobject]@{ Available = $true; Reason = "" }
}

function Get-KubectlJson {
    <#
    .SYNOPSIS
        Ejecuta `kubectl get <recurso> -o json` y devuelve el objeto, o $null con el motivo.
    .OUTPUTS
        [pscustomobject] con Data y Problem.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string[]]$Arguments)

    $run = Invoke-Native -FilePath "kubectl" -Arguments (@("get") + $Arguments + @("--output", "json"))
    if ($run.ExitCode -ne 0) {
        return [pscustomobject]@{ Data = $null; Problem = "kubectl get $($Arguments -join ' ') fallo: $($run.Output)" }
    }
    try {
        return [pscustomobject]@{ Data = ($run.Output | ConvertFrom-Json); Problem = "" }
    }
    catch {
        return [pscustomobject]@{ Data = $null; Problem = "kubectl no devolvio JSON." }
    }
}

function Write-ReportFile {
    <#
    .SYNOPSIS
        Escribe el informe en disco, creando el directorio si hace falta.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][object]$Document
    )

    $parent = Split-Path -Parent $Path
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    $Document | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Path -Encoding UTF8
}

# =============================================================================
# CAPA 4 - ORQUESTACION
# =============================================================================

function Invoke-Guarded {
    <#
    .SYNOPSIS
        Evalua un criterio y convierte una excepcion en NO MEDIDO con su motivo.
    .DESCRIPTION
        Que no se pueda medir un criterio no debe impedir medir los demas, ni
        declararlo FALLADO: no hay evidencia, no hay veredicto.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Id,
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][bool]$Blocking,
        [Parameter(Mandatory)][scriptblock]$Evaluate
    )

    try {
        return (& $Evaluate)
    }
    catch {
        return New-Verdict -Id $Id -Name $Name -Blocking $Blocking -State $StateNotMeasured `
            -Detail "Error al medir: $($_.Exception.Message)"
    }
}

function Write-VerdictLine {
    <#
    .SYNOPSIS
        Imprime un veredicto en una linea con su color y sus avisos.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][object]$Verdict)

    $color = switch ($Verdict.state) {
        $StateMet { "Green" }
        $StateFailed { "Red" }
        default { "Yellow" }
    }
    Write-Host ("  [{0,-9}] {1,-9} {2}" -f $Verdict.state, $Verdict.id, $Verdict.name) -ForegroundColor $color
    Write-Host "              $($Verdict.detail)" -ForegroundColor Gray
    foreach ($warning in $Verdict.warnings) {
        Write-Host "              aviso: $warning" -ForegroundColor DarkYellow
    }
}

function Invoke-VerifyPhase2 {
    <#
    .SYNOPSIS
        Punto de entrada: mide, imprime, escribe el informe y devuelve el codigo de salida.
    .OUTPUTS
        [int] 0, 1 o 2 segun los criterios que bloquean.
    #>
    [CmdletBinding()]
    param()

    $root = if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
        [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".." ".."))
    } else {
        [System.IO.Path]::GetFullPath($ProjectRoot)
    }
    $results = Resolve-ProjectPath -Root $root -Path $ResultsDir
    $python = if ([string]::IsNullOrWhiteSpace($PythonPath)) { Join-Path $root ".venv\Scripts\python.exe" } else { $PythonPath }

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host " ReactorGuard - Verificacion de la Fase 2" -ForegroundColor Cyan
    Write-Host " Proyecto: $ProjectId   Raiz: $root" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan

    $commit = Get-GitCommit -Root $root

    # --- Criterio 1 ---
    $criterion1 = Invoke-Guarded -Id "1" -Name "Throughput Kafka > 50.000 msg/s" -Blocking $true -Evaluate {
        Resolve-BenchmarkVerdict -Id "1" -Name "Throughput Kafka > 50.000 msg/s" `
            -Criterion "kafka_throughput" -Threshold $ThroughputThreshold -Direction "at_least" `
            -Unit "messages_per_second" `
            -HowToMeasure "Midelo en el cluster con .\tests\integration\Invoke-KafkaTests.ps1." `
            -Read (Read-JsonFile -Path (Join-Path $results $ThroughputReportFile)) -HeadCommit $commit
    }

    # --- Criterio 2 ---
    Write-Host ""
    Write-Host "[criterio 2] suite de stuck: $($StuckTestFiles -join ', ')" -ForegroundColor Gray
    $criterion2 = Invoke-Guarded -Id "2" -Name "Precision del detector de stuck > 95 %" -Blocking $true -Evaluate {
        $present = (@($StuckDataFiles | Where-Object { Test-Path -LiteralPath (Join-Path $root $_) }).Count -eq $StuckDataFiles.Count)
        $junitPath = Join-Path $results $JUnitFile
        if ($present -and -not $SkipPytest) {
            if (-not (Test-Path -LiteralPath $python)) { throw "No existe el interprete $python (Setup-DevEnv.ps1)." }
            $run = Invoke-StuckSuite -Root $root -Python $python -JUnitPath $junitPath
            Write-Host "[criterio 2] pytest termino con codigo $($run.ExitCode) en $($run.Seconds) s" -ForegroundColor Gray
            if ($run.ExitCode -notin @(0, 1) -and -not (Test-Path -LiteralPath $junitPath)) {
                $tail = (($run.Output -split "\r?\n") | Select-Object -Last 5) -join " | "
                throw "pytest no dejo informe (codigo $($run.ExitCode)): $tail"
            }
        }
        Resolve-StuckPrecisionVerdict -DataPresent $present `
            -JUnit (Read-JUnitSummary -Path $junitPath) `
            -Metrics (Read-JsonFile -Path (Join-Path $results $MetricsFile))
    }

    # --- Criterio 3 ---
    $criterion3 = Invoke-Guarded -Id "3" -Name "Latencia del pipeline de features < 100 ms" -Blocking $true -Evaluate {
        Resolve-BenchmarkVerdict -Id "3" -Name "Latencia del pipeline de features < 100 ms" `
            -Criterion "feature_latency" -Threshold $LatencyThresholdMs -Direction "at_most" `
            -Unit "milliseconds" `
            -HowToMeasure "Midelo en el cluster con tests/integration/benchmark_feature_latency.py --environment cluster." `
            -Read (Read-JsonFile -Path (Join-Path $results $LatencyReportFile)) -HeadCommit $commit
    }

    # --- Criterio 4 ---
    Write-Host "[criterio 4] dvc status" -ForegroundColor Gray
    $criterion4 = Invoke-Guarded -Id "4" -Name "dvc repro reproducible desde cero" -Blocking $true -Evaluate {
        $yamlPath = Join-Path $root "dvc.yaml"
        if (-not (Test-Path -LiteralPath $yamlPath)) { throw "No existe dvc.yaml en $root." }
        $lockPath = Join-Path $root "dvc.lock"
        $lockText = if (Test-Path -LiteralPath $lockPath) { Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 } else { "" }
        $missing = @(Get-MissingLockStages -DvcYamlText (Get-Content -LiteralPath $yamlPath -Raw -Encoding UTF8) -DvcLockText $lockText)
        $status = Get-DvcStatus -Root $root
        $scratch = $null
        if ($FromScratch -and $missing.Count -eq 0 -and $null -ne $status.Summary) {
            Write-Host "[criterio 4] reproduccion desde cero en una copia temporal (minutos)..." -ForegroundColor Gray
            $scratch = Invoke-FromScratch -Root $root
        }
        Resolve-DvcVerdict -MissingLockStages $missing -StatusSummary $status.Summary `
            -StatusProblem $status.Problem -Scratch $scratch
    }

    # --- Entregables ---
    $access = Get-ClusterAccess -Project $ProjectId -Skip:$SkipCluster

    $pods = Invoke-Guarded -Id "pods" -Name "Pods del streamer y del validador Running" -Blocking $false -Evaluate {
        if (-not $access.Available) {
            Resolve-DeploymentsVerdict -Deployments $null -Problem "Cluster no consultado: $($access.Reason)"
        }
        else {
            $got = Get-KubectlJson -Arguments @("deployments", "--namespace", $Namespace)
            Resolve-DeploymentsVerdict -Deployments $got.Data -Problem $got.Problem
        }
    }

    $feast = Invoke-Guarded -Id "feast" -Name "feast apply registrado" -Blocking $false -Evaluate {
        $registry = Get-FeastRegistryPath -Root $root
        $exists = [bool]$registry -and (Test-Path -LiteralPath $registry -PathType Leaf)
        if (-not $exists) {
            Resolve-FeastVerdict -RegistryExists $false
        }
        elseif (-not (Test-Path -LiteralPath $python)) {
            Resolve-FeastVerdict -RegistryExists $true -Probe $null -Problem "No existe el interprete $python."
        }
        else {
            $probe = Get-FeastRegistryContents -Root $root -Python $python
            Resolve-FeastVerdict -RegistryExists $true -Probe $probe.Probe -Problem $probe.Problem
        }
    }

    $dashboard = Invoke-Guarded -Id "dashboard" -Name "Dashboard de Grafana del pipeline de datos" -Blocking $false -Evaluate {
        $file = Read-JsonFile -Path (Join-Path $root $DashboardFile)
        if (-not $access.Available) {
            Resolve-DashboardVerdict -File $file -ClusterConfigMaps $null -ClusterProblem "Cluster no consultado: $($access.Reason)"
        }
        else {
            $got = Get-KubectlJson -Arguments @("configmap", "--namespace", $DashboardNamespace, "--selector", $DashboardLabel)
            Resolve-DashboardVerdict -File $file -ClusterConfigMaps $got.Data -ClusterProblem $got.Problem
        }
    }

    $criteria = @($criterion1, $criterion2, $criterion3, $criterion4)
    $deliverables = @($pods, $feast, $dashboard)
    $exitCode = Get-ExitCode -Verdicts $criteria

    Write-Host ""
    Write-Host "CRITERIOS (bloquean el codigo de salida)" -ForegroundColor White
    foreach ($verdict in $criteria) { Write-VerdictLine -Verdict $verdict }
    Write-Host ""
    Write-Host "ENTREGABLES (aviso, no bloquean)" -ForegroundColor White
    foreach ($verdict in $deliverables) { Write-VerdictLine -Verdict $verdict }

    $document = New-ReportDocument -Criteria $criteria -Deliverables $deliverables -ExitCode $exitCode `
        -Commit $commit -Project $ProjectId -GeneratedAt ([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"))
    $reportPath = Join-Path $results $ReportFile
    Write-ReportFile -Path $reportPath -Document $document

    $counts = Get-StateCounts -Verdicts $criteria
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host (" Criterios: {0} CUMPLIDO, {1} FALLADO, {2} NO MEDIDO   (salida {3})" -f `
            $counts[$StateMet], $counts[$StateFailed], $counts[$StateNotMeasured], $exitCode) -ForegroundColor Cyan
    Write-Host " Informe  : $reportPath" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    return $exitCode
}

# Punto de entrada. No se ejecuta si el fichero se carga con dot-sourcing (pruebas).
if ($MyInvocation.InvocationName -ne ".") {
    try {
        exit (Invoke-VerifyPhase2)
    }
    catch {
        Write-Host ""
        Write-Host "ERROR del verificador: $($_.Exception.Message)" -ForegroundColor Red
        exit $ExitError
    }
}
