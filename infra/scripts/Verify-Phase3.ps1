#Requires -Version 7.0
<#
.SYNOPSIS
    Verifica los cuatro criterios de exito de la Fase 3 (capa de simulacion).

.DESCRIPTION
    Cada criterio recibe uno de TRES estados, no dos:

        CUMPLIDO   hay evidencia medida y cumple el umbral
        FALLADO    hay evidencia medida y NO cumple el umbral (o la evidencia es
                   incoherente: un hash que no coincide, un test que falla)
        NO MEDIDO  no hay evidencia valida: falta el dataset, el test se salto, el
                   pipeline esta desactualizado

    NO MEDIDO no es un fallo y tampoco un exito (la misma regla que Verify-Phase2.ps1).

    Criterios (bloquean el codigo de salida; Plan_Fases, Fase 3, y plan de Fase 3, seccion 10):
        1  Cobertura de fault types           metrics/simulation_metrics.json
           Los 8 tipos del TDD 2.2 presentes y >= 95 % de la rejilla de modos (D6).
           La fraccion se recalcula con covered / grid_size: el `passed` del informe
           no se usa para decidir.
        2  Cumplimiento fisico                metricas + suite tests/safety (JUnit)
           Conservacion al 100 % (D7: residuo bajo la tolerancia en todos los runs) y
           el estado verdadero solo viola las reglas de la matriz esperada. La suite
           se ejecuta con --junitxml y se cuentan passed y skipped: un skip devuelve 0.
        3  Volumen > 1.000.000 lecturas       manifest.json + los parquet
           Lecturas con verdad-terreno: las que llevan los dos ejes de fallo
           (`process_fault`, `instrument_fault`) informados, sea "none" o un fallo.
           Se cuentan en los PARQUET, no en el manifiesto, y se contrastan con el:
           un sha256, unas filas o unas etiquetas que no coinciden son FALLADO.
        4  Reproducibilidad con la semilla    dvc status simulate + dvc.lock (+ -FromScratch)
           Con -FromScratch rehace SOLO la etapa simulate en una copia temporal y
           compara los hashes de sus salidas con los de dvc.lock. La comparacion es
           bit a bit y por tanto vale dentro de una plataforma (physics_constraints.md,
           14.4): el lock del repositorio se genera en Windows.

    Entregables de aviso (mismos tres estados, NO bloquean):
        fase2  Sin regresion de Fase 2: el ultimo informe de Verify-Phase2.ps1 tiene
               los criterios 2 y 4 CUMPLIDOS.
        docs   Documentos de la fase presentes y no vacios.

    Codigos de salida:
        0  todos los criterios CUMPLIDOS
        1  algun criterio FALLADO
        2  ninguno FALLADO, pero alguno NO MEDIDO
        3  error interno del verificador (no es un veredicto sobre la Fase 3)

    Informe: tests/results/phase3_verification.json (esta en .gitignore).

    Cuatro capas:
        Capa 1 - Configuracion : parametros, umbrales y rutas
        Capa 2 - Utilidades    : funciones puras que deciden el estado de cada criterio
        Capa 3 - Servicio      : pytest, dvc, git y el lector de parquet (Python)
        Capa 4 - Orquestacion  : evalua, imprime, escribe el informe y devuelve el codigo

    Las utilidades comunes con Verify-Phase2.ps1 estan COPIADAS, no compartidas: el
    script de Fase 2 lo ejecutan sus propios tests y no se refactoriza.

.PARAMETER ProjectRoot
    Raiz del repositorio. Por defecto, dos niveles por encima de este script.

.PARAMETER ResultsDir
    Directorio de los informes (relativo a ProjectRoot).

.PARAMETER PythonPath
    Interprete con el que se ejecutan pytest y el lector de parquet. Por defecto, el
    del venv del proyecto.

.PARAMETER SkipPytest
    No ejecuta la suite de seguridad: usa el informe JUnit que haya en ResultsDir.
    Sin informe, el criterio 2 queda NO MEDIDO.

.PARAMETER FromScratch
    Ademas, rehace la etapa simulate en una copia temporal (sin dvc.lock) y compara
    los hashes. Tarda lo que el stage (~11 min con 8 procesos en el portatil de
    referencia). Necesita los ficheros nuevos COMMITEADOS: la copia sale de
    `git ls-files`. Nunca toca data/ del repositorio.

.EXAMPLE
    .\infra\scripts\Verify-Phase3.ps1
    .\infra\scripts\Verify-Phase3.ps1 -SkipPytest
    .\infra\scripts\Verify-Phase3.ps1 -FromScratch
#>

# =============================================================================
# CAPA 1 - CONFIGURACION
# =============================================================================
[CmdletBinding()]
param(
    [string]$ProjectRoot = "",
    [string]$ResultsDir  = "tests/results",
    [string]$PythonPath  = "",
    [switch]$SkipPytest,
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

# Umbrales de los criterios (Plan_Fases, Fase 3; D6 y D7 del plan de Fase 3).
$FaultTypesRequired    = 8           # tipos de fallo del TDD 2.2
$CoverageGate          = 0.95        # fraccion minima de la rejilla de modos (D6)
$VolumeThreshold       = 1000000     # lecturas con verdad-terreno, estrictamente mas
$ConservationTolerance = 1.0e-6      # residuo relativo de energia admitido (D7)

# Evidencia, relativa a ProjectRoot (rutas de la seccion simulation: de params.yaml).
$MetricsFile  = "metrics/simulation_metrics.json"
$DatasetDir   = "data/raw/simulation"
$ManifestName = "manifest.json"

# Etapa de DVC que produce el dataset.
$SimulateStage = "simulate"

# Suite del criterio 2, relativa a ProjectRoot, y los tests que deben haberse ejecutado.
$SafetyTestFiles = @(
    "tests/safety/test_safety_constraints.py"
)
$ComplianceTestNames = @(
    "test_conservation_under_tolerance_in_every_run",
    "test_expected_violation_matrix"
)

# Verdad-terreno por lectura (TDD 2.2: tipo, severidad e inicio del fallo, en dos ejes).
$GroundTruthColumns = @(
    "scenario",
    "true_value",
    "corrupted",
    "process_fault",
    "process_severity",
    "instrument_fault",
    "instrument_severity",
    "injected_at"
)
$FaultAxisColumns = @("process_fault", "instrument_fault")
$NoFaultLabel     = "none"

# Informes, relativos a ResultsDir.
$JUnitFile        = "phase3_safety.junit.xml"
$ReportFile       = "phase3_verification.json"
$Phase2ReportFile = "phase2_verification.json"

# Entregables.
$Phase2GuardCriteria = @("2", "4")
$RequiredDocs = @(
    "Docs/physics_constraints.md",
    "Docs/ReactorGuard_Fase3_Simulacion.md"
)

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

function Get-MissingFields {
    <#
    .SYNOPSIS
        Nombres de `Fields` que faltan en un objeto JSON o que no son numericos.
    #>
    [CmdletBinding()]
    param(
        [object]$Object,
        [Parameter(Mandatory)][string[]]$Fields
    )

    return @($Fields | Where-Object { -not (Test-IsNumber (Get-JsonProperty -Object $Object -Name $_)) })
}

function Resolve-CoverageVerdict {
    <#
    .SYNOPSIS
        Estado del criterio 1: los 8 tipos presentes y la rejilla de modos cubierta.
    .DESCRIPTION
        La fraccion se recalcula con covered / grid_size y se compara con el umbral del
        SCRIPT: ni el `gate` ni el `passed` del informe deciden (un informe no puede
        aprobarse a si mismo). La cobertura de las particiones se informa en la
        evidencia y no bloquea: D6 es una puerta sobre el dataset entero.
    .PARAMETER Metrics
        Resultado de Read-JsonFile sobre metrics/simulation_metrics.json.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][hashtable]$Metrics)

    $id = "1"
    $name = "Cobertura: 8/8 tipos de fallo y >= 95 % de la rejilla"
    $howTo = "Ejecuta .\infra\scripts\Invoke-Pipeline.ps1 -Stage simulate."

    if (-not $Metrics.Found) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Sin metricas del dataset: $($Metrics.Error). $howTo"
    }
    $coverage = Get-JsonProperty -Object $Metrics.Data -Name "coverage"
    $overall = Get-JsonProperty -Object $coverage -Name "overall"
    $missing = @(Get-MissingFields -Object $coverage -Fields @("n_types")) +
        @(Get-MissingFields -Object $overall -Fields @("covered", "grid_size"))
    if ($missing.Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Metricas invalidas: faltan coverage.n_types, coverage.overall.covered o grid_size ($($missing -join ', '))."
    }
    $types = [int](Get-JsonProperty -Object $coverage -Name "n_types")
    $covered = [int](Get-JsonProperty -Object $overall -Name "covered")
    $gridSize = [int](Get-JsonProperty -Object $overall -Name "grid_size")
    if ($gridSize -le 0 -or $covered -lt 0 -or $covered -gt $gridSize) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Metricas incoherentes: $covered modos cubiertos de una rejilla de $gridSize."
    }
    $fraction = $covered / $gridSize
    $met = ($types -ge $FaultTypesRequired) -and ($fraction -ge $CoverageGate)

    $warnings = [System.Collections.Generic.List[string]]::new()
    $reportGate = Get-JsonProperty -Object $coverage -Name "gate"
    if ((Test-IsNumber $reportGate) -and ([double]$reportGate -ne $CoverageGate)) {
        $warnings.Add("Las metricas declaran gate $(Format-Number ([double]$reportGate)); se usa $(Format-Number $CoverageGate).")
    }
    $reportPassed = Get-JsonProperty -Object $coverage -Name "passed"
    if (($reportPassed -is [bool]) -and ($reportPassed -ne $met)) {
        $warnings.Add("Las metricas dicen passed=$reportPassed pero tipos y rejilla dan $met; se usa el recalculo.")
    }

    $partitions = [ordered]@{}
    $scopes = Get-JsonProperty -Object $coverage -Name "partitions"
    if ($null -ne $scopes) {
        foreach ($property in $scopes.PSObject.Properties) {
            $scopeFraction = Get-JsonProperty -Object $property.Value -Name "grid_fraction"
            if (Test-IsNumber $scopeFraction) { $partitions[$property.Name] = [double]$scopeFraction }
        }
    }
    $evidence = [ordered]@{
        fault_types   = $types
        required      = $FaultTypesRequired
        covered_modes = $covered
        grid_size     = $gridSize
        grid_fraction = $fraction
        gate          = $CoverageGate
        partitions    = $partitions
    }
    $text = "$types/$FaultTypesRequired tipos; rejilla $covered/$gridSize ($(Format-Number (100 * $fraction) '0.0') %, umbral >= $(Format-Number (100 * $CoverageGate) '0') %)"
    $state = if ($met) { $StateMet } else { $StateFailed }
    return New-Verdict -Id $id -Name $name -Blocking $true -State $state -Detail $text `
        -Evidence $evidence -Warnings $warnings
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

    return [pscustomobject]@{
        Passed      = $passedNames.Count
        Failed      = $failedNames.Count - $errors
        Errors      = $errors
        Skipped     = $skipped
        PassedNames = $passedNames.ToArray()
        FailedNames = $failedNames.ToArray()
        SkipReasons = $skipReasons.ToArray()
    }
}

function Resolve-ComplianceVerdict {
    <#
    .SYNOPSIS
        Estado del criterio 2: conservacion, matriz esperada y suite de seguridad.
    .DESCRIPTION
        Toda la evidencia de fallo se reune antes de decidir: un FALLADO de las
        metricas no se oculta tras una suite sin ejecutar, ni al reves. Solo sin
        ningun fallo cuenta lo que falte por medir.
    .PARAMETER Metrics
        Resultado de Read-JsonFile sobre metrics/simulation_metrics.json.
    .PARAMETER JUnit
        Resultado de Read-JUnitSummary, o $null si pytest no dejo informe.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Metrics,
        [AllowNull()][object]$JUnit
    )

    $id = "2"
    $name = "Cumplimiento fisico: conservacion 100 % y matriz esperada"
    $failures = [System.Collections.Generic.List[string]]::new()
    $unmeasured = [System.Collections.Generic.List[string]]::new()
    $warnings = [System.Collections.Generic.List[string]]::new()
    $evidence = [ordered]@{ tolerance = $ConservationTolerance }

    $compliance = if ($Metrics.Found) { Get-JsonProperty -Object $Metrics.Data -Name "compliance" } else { $null }
    $fields = @("conservation_failures", "max_instant_residual", "max_interval_residual")
    if (-not $Metrics.Found) {
        $unmeasured.Add("sin metricas del dataset ($($Metrics.Error))")
    }
    elseif (@(Get-MissingFields -Object $compliance -Fields $fields).Count -gt 0 -or
            $null -eq (Get-JsonProperty -Object $compliance -Name "unexpected_violations")) {
        $unmeasured.Add("metricas invalidas: faltan campos de compliance")
    }
    else {
        $conservationFailures = [long](Get-JsonProperty -Object $compliance -Name "conservation_failures")
        $instant = [double](Get-JsonProperty -Object $compliance -Name "max_instant_residual")
        $interval = [double](Get-JsonProperty -Object $compliance -Name "max_interval_residual")
        $evidence["conservation_failures"] = $conservationFailures
        $evidence["max_instant_residual"] = $instant
        $evidence["max_interval_residual"] = $interval
        if ($conservationFailures -gt 0) {
            $failures.Add("$conservationFailures instantes o intervalos sobre la tolerancia de conservacion")
        }
        $worst = [math]::Max($instant, $interval)
        if ($worst -gt $ConservationTolerance) {
            $failures.Add("residuo de conservacion $(Format-Number $worst '0.###e+0') > $(Format-Number $ConservationTolerance '0.###e+0')")
        }
        $reportTolerance = Get-JsonProperty -Object $compliance -Name "conservation_tolerance"
        if ((Test-IsNumber $reportTolerance) -and ([double]$reportTolerance -ne $ConservationTolerance)) {
            $warnings.Add("Las metricas declaran tolerancia $(Format-Number ([double]$reportTolerance) '0.###e+0'); se usa $(Format-Number $ConservationTolerance '0.###e+0').")
        }

        $unexpected = [System.Collections.Generic.List[string]]::new()
        foreach ($run in (Get-JsonProperty -Object $compliance -Name "unexpected_violations").PSObject.Properties) {
            foreach ($rule in $run.Value.PSObject.Properties) {
                $unexpected.Add("$($run.Name) $($rule.Name)=$($rule.Value)")
            }
        }
        $evidence["unexpected_violations"] = $unexpected.ToArray()
        if ($unexpected.Count -gt 0) {
            $failures.Add("el estado verdadero viola reglas fuera de la matriz esperada: $($unexpected -join ', ')")
        }
        $reportAsExpected = Get-JsonProperty -Object $compliance -Name "violations_as_expected"
        if (($reportAsExpected -is [bool]) -and ($reportAsExpected -ne ($unexpected.Count -eq 0))) {
            $warnings.Add("Las metricas dicen violations_as_expected=$reportAsExpected pero listan $($unexpected.Count) violaciones; se usa la lista.")
        }
    }

    if ($null -eq $JUnit) {
        $unmeasured.Add("sin informe JUnit de la suite de seguridad (-SkipPytest sin un informe previo)")
    }
    else {
        $evidence["tests_passed"] = $JUnit.Passed
        $evidence["tests_skipped"] = $JUnit.Skipped
        if (($JUnit.Failed + $JUnit.Errors) -gt 0) {
            $failures.Add("$($JUnit.Failed + $JUnit.Errors) tests de seguridad fallan: $($JUnit.FailedNames -join '; ')")
        }
        elseif ($JUnit.Skipped -gt 0) {
            $reason = if ($JUnit.SkipReasons.Count -gt 0) { " (motivo: $($JUnit.SkipReasons[0]))" } else { "" }
            $unmeasured.Add("$($JUnit.Skipped) tests de seguridad SALTADOS de $($JUnit.Passed + $JUnit.Skipped)$reason")
        }
        else {
            $notRun = @($ComplianceTestNames | Where-Object {
                    $test = $_
                    @($JUnit.PassedNames | Where-Object { $_ -like "*.$test" }).Count -eq 0
                })
            if ($notRun.Count -gt 0) {
                $unmeasured.Add("la suite no ejecuto $($notRun -join ', ')")
            }
        }
    }

    if ($failures.Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
            -Detail (($failures -join "; ") + ".") -Evidence $evidence -Warnings $warnings
    }
    if ($unmeasured.Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Falta evidencia: $($unmeasured -join '; ')." -Evidence $evidence -Warnings $warnings
    }
    $worstText = Format-Number ([math]::Max($evidence["max_instant_residual"], $evidence["max_interval_residual"])) '0.##e+0'
    $text = "0 fallos de conservacion (peor residuo $worstText, tolerancia $(Format-Number $ConservationTolerance '0.###e+0')), " +
        "el estado verdadero solo viola la matriz esperada y $($JUnit.Passed) tests de seguridad pasan sin saltos"
    return New-Verdict -Id $id -Name $name -Blocking $true -State $StateMet -Detail $text `
        -Evidence $evidence -Warnings $warnings
}

function Resolve-VolumeVerdict {
    <#
    .SYNOPSIS
        Estado del criterio 3: lecturas con verdad-terreno contadas en los parquet.
    .DESCRIPTION
        El manifiesto dice que deberia haber; los parquet dicen que hay. Un sha256,
        un numero de filas o unas etiquetas que no coinciden con el manifiesto son
        evidencia incoherente (FALLADO). Un parquet ausente deja el dataset
        incompleto: NO MEDIDO, como una salida de DVC borrada.
    .PARAMETER Manifest
        Resultado de Read-JsonFile sobre data/raw/simulation/manifest.json.
    .PARAMETER Probe
        Salida del lector de parquet (Get-ParquetFacts): objeto con `files`, o $null.
    .PARAMETER ProbeProblem
        Por que no se pudieron leer los parquet (cuando Probe es $null).
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Manifest,
        [AllowNull()][object]$Probe = $null,
        [string]$ProbeProblem = ""
    )

    $id = "3"
    $name = "Volumen: > 1.000.000 lecturas con verdad-terreno"
    $howTo = "Ejecuta .\infra\scripts\Invoke-Pipeline.ps1 -Stage simulate."

    if (-not $Manifest.Found) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Sin manifiesto del dataset: $($Manifest.Error). $howTo"
    }
    $runs = @(Get-JsonProperty -Object $Manifest.Data -Name "runs" -Default @())
    if ($runs.Count -eq 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "El manifiesto no lista ningun run. $howTo"
    }
    if ($null -eq $Probe) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "No se pudieron leer los parquet: $ProbeProblem"
    }

    $facts = @{}
    foreach ($file in @(Get-JsonProperty -Object $Probe -Name "files" -Default @())) {
        $facts[[string](Get-JsonProperty -Object $file -Name "path")] = $file
    }

    $incoherent = [System.Collections.Generic.List[string]]::new()
    $absent = [System.Collections.Generic.List[string]]::new()
    $invalid = [System.Collections.Generic.List[string]]::new()
    [long]$groundTruth = 0
    [long]$faultLabelled = 0
    [long]$rows = 0
    foreach ($run in $runs) {
        $path = [string](Get-JsonProperty -Object $run -Name "path")
        $declaredSha = [string](Get-JsonProperty -Object $run -Name "sha256")
        $declared = @(Get-MissingFields -Object $run -Fields @("rows", "labelled"))
        if (-not $path -or -not $declaredSha -or $declared.Count -gt 0) {
            $invalid.Add($(if ($path) { $path } else { "(run sin path)" }))
            continue
        }
        $fact = $facts[$path]
        if ($null -eq $fact -or -not [bool](Get-JsonProperty -Object $fact -Name "exists" -Default $false)) {
            $absent.Add($path)
            continue
        }
        $actualSha = [string](Get-JsonProperty -Object $fact -Name "sha256")
        if ($actualSha -ne $declaredSha) {
            $incoherent.Add("${path}: sha256 distinto del manifiesto")
            continue
        }
        $actualRows = [long](Get-JsonProperty -Object $fact -Name "rows" -Default 0)
        $rows += $actualRows
        if ($actualRows -ne [long](Get-JsonProperty -Object $run -Name "rows")) {
            $incoherent.Add("${path}: $actualRows filas frente a $(Get-JsonProperty -Object $run -Name 'rows') del manifiesto")
        }
        $columns = @(Get-JsonProperty -Object $fact -Name "columns" -Default @())
        $lacking = @($GroundTruthColumns | Where-Object { $_ -notin $columns })
        if ($lacking.Count -gt 0) {
            $incoherent.Add("${path}: sin las columnas de verdad-terreno $($lacking -join ', ')")
            continue
        }
        $truth = [long](Get-JsonProperty -Object $fact -Name "ground_truth" -Default 0)
        $labelled = [long](Get-JsonProperty -Object $fact -Name "labelled" -Default 0)
        $groundTruth += $truth
        $faultLabelled += $labelled
        if ($truth -ne $actualRows) {
            $incoherent.Add("${path}: $($actualRows - $truth) lecturas sin etiqueta en algun eje")
        }
        if ($labelled -ne [long](Get-JsonProperty -Object $run -Name "labelled")) {
            $incoherent.Add("${path}: $labelled lecturas con fallo frente a $(Get-JsonProperty -Object $run -Name 'labelled') del manifiesto")
        }
    }
    $declaredTotal = Get-JsonProperty -Object $Manifest.Data -Name "total_readings"
    if ($absent.Count -eq 0 -and $invalid.Count -eq 0 -and (Test-IsNumber $declaredTotal) -and [long]$declaredTotal -ne $rows) {
        $incoherent.Add("el manifiesto declara $declaredTotal lecturas y los parquet suman $rows")
    }

    $evidence = [ordered]@{
        runs              = $runs.Count
        readings          = $rows
        ground_truth      = $groundTruth
        fault_labelled    = $faultLabelled
        threshold         = $VolumeThreshold
        absent_files      = $absent.ToArray()
    }
    if ($incoherent.Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
            -Detail "El dataset no coincide con su manifiesto: $($incoherent -join '; ')." -Evidence $evidence
    }
    if (($absent.Count + $invalid.Count) -gt 0) {
        $parts = @()
        if ($absent.Count -gt 0) { $parts += "$($absent.Count) parquet ausentes ($($absent[0])...)" }
        if ($invalid.Count -gt 0) { $parts += "$($invalid.Count) entradas del manifiesto sin path, sha256, rows o labelled" }
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
            -Detail "Dataset incompleto: $($parts -join '; '). $howTo" -Evidence $evidence
    }

    $share = if ($groundTruth -gt 0) { 100.0 * $faultLabelled / $groundTruth } else { 0.0 }
    $text = "$(Format-Number $groundTruth '#,0') lecturas con verdad-terreno en $($runs.Count) runs " +
        "(umbral > $(Format-Number $VolumeThreshold '#,0')); $(Format-Number $faultLabelled '#,0') con fallo ($(Format-Number $share '0.00') %)"
    $state = if ($groundTruth -gt $VolumeThreshold) { $StateMet } else { $StateFailed }
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
        Where-Object { $_ -and $_ -notmatch '%\||it/s\]|B/s\]' })
    return ($useful | Select-Object -Last $Lines) -join " | "
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

function Select-StageHashes {
    <#
    .SYNOPSIS
        Filtra un mapa de Get-LockOutputHashes a las salidas de una etapa.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Hashes,
        [Parameter(Mandatory)][string]$Stage
    )

    $selected = @{}
    foreach ($key in $Hashes.Keys) {
        if ($key.StartsWith("${Stage}:")) { $selected[$key] = $Hashes[$key] }
    }
    return $selected
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

function Resolve-ReproducibilityVerdict {
    <#
    .SYNOPSIS
        Estado del criterio 4 sobre la etapa simulate.
    .PARAMETER Declared
        Si dvc.yaml declara la etapa.
    .PARAMETER Locked
        Si dvc.lock la registra.
    .PARAMETER StatusSummary
        Resultado de Get-DvcStatusSummary para la etapa, o $null si dvc status no se pudo ejecutar.
    .PARAMETER StatusProblem
        Motivo por el que dvc status no se pudo ejecutar (cuando StatusSummary es $null).
    .PARAMETER Scratch
        Resultado de Invoke-FromScratch, o $null si no se pidio.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][bool]$Declared,
        [Parameter(Mandatory)][bool]$Locked,
        [AllowNull()][object]$StatusSummary,
        [string]$StatusProblem = "",
        [AllowNull()][object]$Scratch = $null
    )

    $id = "4"
    $name = "Reproducibilidad del stage simulate con la misma semilla"
    $howTo = "Ejecuta .\infra\scripts\Invoke-Pipeline.ps1 -Stage $SimulateStage (nunca dvc repro a pelo) y repite."

    if (-not $Declared) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
            -Detail "dvc.yaml no declara la etapa $SimulateStage."
    }
    if (-not $Locked) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateFailed `
            -Detail "dvc.lock no registra la etapa ${SimulateStage}: no hay hashes con los que comparar. $howTo"
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
            -Detail "Etapa desactualizada o sin datos, no demostrada irreproducible: $($all -join '; '). $howTo"
    }

    $evidence = [ordered]@{ dvc_status = "clean"; from_scratch = "not requested" }
    if ($null -eq $Scratch) {
        return New-Verdict -Id $id -Name $name -Blocking $true -State $StateMet `
            -Detail "dvc status de $SimulateStage limpio y la etapa esta en dvc.lock (sin -FromScratch)." `
            -Evidence $evidence
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
                -Detail "dvc repro $SimulateStage desde cero fallo: $($Scratch.Detail)" -Evidence $evidence
        }
        default {
            return New-Verdict -Id $id -Name $name -Blocking $true -State $StateNotMeasured `
                -Detail "No se pudo completar la reproduccion desde cero: $($Scratch.Detail)" -Evidence $evidence
        }
    }
}

function Resolve-Phase2RegressionVerdict {
    <#
    .SYNOPSIS
        Estado del entregable "sin regresion de Fase 2" a partir del informe de Verify-Phase2.ps1.
    .PARAMETER Report
        Resultado de Read-JsonFile sobre phase2_verification.json.
    .PARAMETER HeadCommit
        Commit actual (con -dirty si procede), para avisar de un informe de otro commit.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][hashtable]$Report,
        [string]$HeadCommit = ""
    )

    $id = "fase2"
    $name = "Sin regresion de Fase 2 (criterios 2 y 4)"
    $howTo = "Ejecuta .\infra\scripts\Verify-Phase2.ps1 -SkipCluster."
    if (-not $Report.Found) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured `
            -Detail "Sin informe de Fase 2: $($Report.Error). $howTo"
    }

    $states = @{}
    foreach ($criterion in @(Get-JsonProperty -Object $Report.Data -Name "criteria" -Default @())) {
        $states[[string](Get-JsonProperty -Object $criterion -Name "id")] = [string](Get-JsonProperty -Object $criterion -Name "state")
    }
    $warnings = [System.Collections.Generic.List[string]]::new()
    $commit = [string](Get-JsonProperty -Object $Report.Data -Name "commit" -Default "unknown")
    $headBase = Get-HeadBase -Commit $HeadCommit
    $reportBase = Get-HeadBase -Commit $commit
    if ($headBase -and $reportBase -and -not ($headBase.StartsWith($reportBase) -or $reportBase.StartsWith($headBase))) {
        $warnings.Add("El informe de Fase 2 es del commit $commit; HEAD es $HeadCommit. $howTo")
    }

    $evidence = [ordered]@{ commit = $commit }
    $failed = [System.Collections.Generic.List[string]]::new()
    $unknown = [System.Collections.Generic.List[string]]::new()
    foreach ($guard in $Phase2GuardCriteria) {
        $state = if ($states.ContainsKey($guard)) { $states[$guard] } else { "" }
        $evidence["criterion_$guard"] = $state
        if ($state -eq $StateFailed) { $failed.Add($guard) }
        elseif ($state -ne $StateMet) { $unknown.Add($guard) }
    }
    if ($failed.Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateFailed `
            -Detail "Verify-Phase2.ps1 da FALLADO en el criterio $($failed -join ' y ')." -Evidence $evidence -Warnings $warnings
    }
    if ($unknown.Count -gt 0) {
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateNotMeasured `
            -Detail "El informe de Fase 2 no da CUMPLIDO en el criterio $($unknown -join ' y '). $howTo" `
            -Evidence $evidence -Warnings $warnings
    }
    return New-Verdict -Id $id -Name $name -Blocking $false -State $StateMet `
        -Detail "Verify-Phase2.ps1 da CUMPLIDO en los criterios $($Phase2GuardCriteria -join ' y ')." `
        -Evidence $evidence -Warnings $warnings
}

function Resolve-DocsVerdict {
    <#
    .SYNOPSIS
        Estado del entregable "documentos de la fase".
    .PARAMETER Sizes
        Ruta relativa -> tamano en bytes, o -1 si el fichero no existe.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][System.Collections.IDictionary]$Sizes)

    $id = "docs"
    $name = "Documentos de la Fase 3"
    $absent = @($Sizes.Keys | Where-Object { $Sizes[$_] -lt 0 } | Sort-Object)
    $empty = @($Sizes.Keys | Where-Object { $Sizes[$_] -eq 0 } | Sort-Object)
    $evidence = [ordered]@{ documents = @($Sizes.Keys | Sort-Object) }
    if (($absent.Count + $empty.Count) -gt 0) {
        $parts = @()
        if ($absent.Count -gt 0) { $parts += "ausentes: $($absent -join ', ')" }
        if ($empty.Count -gt 0) { $parts += "vacios: $($empty -join ', ')" }
        return New-Verdict -Id $id -Name $name -Blocking $false -State $StateFailed `
            -Detail "Documentos $($parts -join '; ')." -Evidence $evidence
    }
    return New-Verdict -Id $id -Name $name -Blocking $false -State $StateMet `
        -Detail "Presentes: $(@($Sizes.Keys | Sort-Object) -join ', ')." -Evidence $evidence
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
        [Parameter(Mandatory)][string]$GeneratedAt
    )

    return [ordered]@{
        schema_version = 1
        phase          = 3
        generated_at   = $GeneratedAt
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

function Invoke-SafetySuite {
    <#
    .SYNOPSIS
        Ejecuta la suite de seguridad dejando un informe JUnit.
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

    $arguments = @("-m", "pytest") + $SafetyTestFiles + @("--junitxml=$JUnitPath", "-q", "-p", "no:cacheprovider")
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

function Get-ParquetFacts {
    <#
    .SYNOPSIS
        Lee de cada parquet su sha256, sus filas, sus columnas y sus etiquetas (solo lectura).
    .DESCRIPTION
        `ground_truth` cuenta las lecturas con los dos ejes de fallo informados (no
        nulos) y `labelled` las que tienen un fallo en alguno ("none" es sin fallo).
        Solo se leen las dos columnas de eje; el resto sale de los metadatos.
    .OUTPUTS
        [pscustomobject] con Probe (objeto con `files`, o $null) y Problem.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$DatasetRoot,
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string[]]$RelativePaths
    )

    $script = @'
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

root = Path(sys.argv[1])
axes = sys.argv[2].split(",")
no_fault = sys.argv[3]
files = []
for relative in sys.argv[4:]:
    path = root / relative
    entry = {"path": relative, "exists": path.is_file()}
    if entry["exists"]:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        entry.update(sha256=digest.hexdigest(), rows=parquet.metadata.num_rows, columns=names)
        if all(axis in names for axis in axes):
            table = parquet.read(columns=axes)
            truth = pc.is_valid(table[axes[0]])
            fault = pc.fill_null(pc.not_equal(table[axes[0]], no_fault), False)
            for axis in axes[1:]:
                truth = pc.and_(truth, pc.is_valid(table[axis]))
                fault = pc.or_(fault, pc.fill_null(pc.not_equal(table[axis], no_fault), False))
            entry.update(
                ground_truth=int(pc.sum(pc.cast(truth, "int64")).as_py() or 0),
                labelled=int(pc.sum(pc.cast(fault, "int64")).as_py() or 0),
            )
    files.append(entry)
print(json.dumps({"files": files}))
'@
    $file = Join-Path ([System.IO.Path]::GetTempPath()) ("reactorguard-parquet-probe-" + [guid]::NewGuid().ToString("N") + ".py")
    try {
        Set-Content -LiteralPath $file -Value $script -Encoding UTF8
        $arguments = @($file, $DatasetRoot, ($FaultAxisColumns -join ","), $NoFaultLabel) + $RelativePaths
        $run = Invoke-Native -FilePath $Python -Arguments $arguments
    }
    finally {
        Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue
    }
    if ($run.ExitCode -ne 0) {
        return [pscustomobject]@{ Probe = $null; Problem = "el lector de parquet fallo: $(Get-OutputTail -Text $run.Output -Lines 3)" }
    }
    $json = ($run.Output -split "\r?\n" | Where-Object { $_.TrimStart().StartsWith("{") } | Select-Object -Last 1)
    try {
        return [pscustomobject]@{ Probe = ($json | ConvertFrom-Json); Problem = "" }
    }
    catch {
        return [pscustomobject]@{ Probe = $null; Problem = "el lector de parquet no devolvio JSON: $(Get-OutputTail -Text $run.Output -Lines 3)" }
    }
}

function Get-DvcStatus {
    <#
    .SYNOPSIS
        Ejecuta `dvc status <etapa> --json` con el venv del proyecto al principio del PATH.
    .OUTPUTS
        [pscustomobject] con Summary (o $null) y Problem.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Stage
    )

    # dvc.exe no activa el venv: anteponer, nunca sustituir (igual que Invoke-Pipeline.ps1).
    $scripts = Join-Path $Root ".venv\Scripts"
    $previousPath = $env:PATH
    if (Test-Path -LiteralPath $scripts) { $env:PATH = "$scripts$([System.IO.Path]::PathSeparator)$env:PATH" }
    try {
        if ($null -eq (Get-Command dvc -ErrorAction SilentlyContinue)) {
            return [pscustomobject]@{ Summary = $null; Problem = "dvc no esta en el PATH (instala el venv con Setup-DevEnv.ps1)." }
        }
        $run = Invoke-Native -FilePath "dvc" -Arguments @("status", $Stage, "--json") -WorkingDirectory $Root
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
        Rehace una etapa en una copia temporal y compara los hashes de sus salidas.
    .DESCRIPTION
        Copia los ficheros que git sigue (sin dvc.lock ni .dvc), inicializa DVC sin
        SCM y ejecuta `dvc repro --no-run-cache <etapa>`. La etapa simulate no
        descarga nada, de modo que cualquier fallo es del pipeline. Jamas toca data/
        del repositorio: todo ocurre bajo el directorio temporal, que se borra siempre.
    .OUTPUTS
        [pscustomobject] con Status (matched, mismatch, failed, unavailable) y Detail.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Root,
        [Parameter(Mandatory)][string]$Stage
    )

    $lockPath = Join-Path $Root "dvc.lock"
    if (-not (Test-Path -LiteralPath $lockPath)) {
        return [pscustomobject]@{ Status = "unavailable"; Detail = "no hay dvc.lock con el que comparar." }
    }
    $expected = Select-StageHashes -Stage $Stage `
        -Hashes (Get-LockOutputHashes -Text (Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8))
    if ($expected.Count -eq 0) {
        return [pscustomobject]@{ Status = "unavailable"; Detail = "dvc.lock no registra salidas de $Stage." }
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
        $repro = Invoke-Native -FilePath "dvc" -Arguments @("repro", "--no-run-cache", $Stage) -WorkingDirectory $temp
        if ($repro.ExitCode -ne 0) {
            return [pscustomobject]@{ Status = "failed"; Detail = (Get-OutputTail -Text $repro.Output) }
        }
        $newLock = Join-Path $temp "dvc.lock"
        if (-not (Test-Path -LiteralPath $newLock)) {
            return [pscustomobject]@{ Status = "failed"; Detail = "dvc repro no escribio dvc.lock." }
        }
        $actual = Select-StageHashes -Stage $Stage `
            -Hashes (Get-LockOutputHashes -Text (Get-Content -LiteralPath $newLock -Raw -Encoding UTF8))
        $differences = @(Compare-LockHashes -Expected $expected -Actual $actual)
        if ($differences.Count -gt 0) {
            return [pscustomobject]@{ Status = "mismatch"; Detail = ($differences -join "; ") }
        }
        return [pscustomobject]@{ Status = "matched"; Detail = "$($expected.Count) salidas de $Stage con el mismo hash." }
    }
    finally {
        $env:PATH = $previousPath
        $env:PYTHONPATH = $previousPythonPath
        Remove-Item -LiteralPath $temp -Recurse -Force -ErrorAction SilentlyContinue
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

function Invoke-VerifyPhase3 {
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
    $metricsPath = Join-Path $root $MetricsFile
    $datasetRoot = Join-Path $root $DatasetDir

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host " ReactorGuard - Verificacion de la Fase 3 (simulacion)" -ForegroundColor Cyan
    Write-Host " Raiz: $root" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan

    $commit = Get-GitCommit -Root $root

    # --- Criterio 1 ---
    $criterion1 = Invoke-Guarded -Id "1" -Name "Cobertura: 8/8 tipos de fallo y >= 95 % de la rejilla" -Blocking $true -Evaluate {
        Resolve-CoverageVerdict -Metrics (Read-JsonFile -Path $metricsPath)
    }

    # --- Criterio 2 ---
    Write-Host ""
    Write-Host "[criterio 2] suite de seguridad: $($SafetyTestFiles -join ', ')" -ForegroundColor Gray
    $criterion2 = Invoke-Guarded -Id "2" -Name "Cumplimiento fisico: conservacion 100 % y matriz esperada" -Blocking $true -Evaluate {
        $junitPath = Join-Path $results $JUnitFile
        if (-not $SkipPytest) {
            if (-not (Test-Path -LiteralPath $python)) { throw "No existe el interprete $python (Setup-DevEnv.ps1)." }
            $run = Invoke-SafetySuite -Root $root -Python $python -JUnitPath $junitPath
            Write-Host "[criterio 2] pytest termino con codigo $($run.ExitCode) en $($run.Seconds) s" -ForegroundColor Gray
            if ($run.ExitCode -notin @(0, 1) -and -not (Test-Path -LiteralPath $junitPath)) {
                throw "pytest no dejo informe (codigo $($run.ExitCode)): $(Get-OutputTail -Text $run.Output -Lines 5)"
            }
        }
        Resolve-ComplianceVerdict -Metrics (Read-JsonFile -Path $metricsPath) -JUnit (Read-JUnitSummary -Path $junitPath)
    }

    # --- Criterio 3 ---
    Write-Host "[criterio 3] parquet de $DatasetDir" -ForegroundColor Gray
    $criterion3 = Invoke-Guarded -Id "3" -Name "Volumen: > 1.000.000 lecturas con verdad-terreno" -Blocking $true -Evaluate {
        $manifest = Read-JsonFile -Path (Join-Path $datasetRoot $ManifestName)
        $paths = @()
        if ($manifest.Found) {
            $paths = @(@(Get-JsonProperty -Object $manifest.Data -Name "runs" -Default @()) |
                ForEach-Object { [string](Get-JsonProperty -Object $_ -Name "path") } | Where-Object { $_ })
        }
        if (-not $manifest.Found -or $paths.Count -eq 0) {
            Resolve-VolumeVerdict -Manifest $manifest
        }
        elseif (-not (Test-Path -LiteralPath $python)) {
            Resolve-VolumeVerdict -Manifest $manifest -Probe $null -ProbeProblem "no existe el interprete $python."
        }
        else {
            $facts = Get-ParquetFacts -DatasetRoot $datasetRoot -Python $python -RelativePaths $paths
            Resolve-VolumeVerdict -Manifest $manifest -Probe $facts.Probe -ProbeProblem $facts.Problem
        }
    }

    # --- Criterio 4 ---
    Write-Host "[criterio 4] dvc status $SimulateStage" -ForegroundColor Gray
    $criterion4 = Invoke-Guarded -Id "4" -Name "Reproducibilidad del stage simulate con la misma semilla" -Blocking $true -Evaluate {
        $yamlPath = Join-Path $root "dvc.yaml"
        if (-not (Test-Path -LiteralPath $yamlPath)) { throw "No existe dvc.yaml en $root." }
        $lockPath = Join-Path $root "dvc.lock"
        $lockText = if (Test-Path -LiteralPath $lockPath) { Get-Content -LiteralPath $lockPath -Raw -Encoding UTF8 } else { "" }
        $declared = $SimulateStage -in @(Get-StageNames -Text (Get-Content -LiteralPath $yamlPath -Raw -Encoding UTF8))
        $locked = $SimulateStage -in @(Get-StageNames -Text $lockText)
        $status = if ($declared -and $locked) { Get-DvcStatus -Root $root -Stage $SimulateStage } else { $null }
        $scratch = $null
        if ($FromScratch -and $null -ne $status -and $null -ne $status.Summary) {
            Write-Host "[criterio 4] reproduccion de $SimulateStage desde cero en una copia temporal (minutos)..." -ForegroundColor Gray
            $scratch = Invoke-FromScratch -Root $root -Stage $SimulateStage
        }
        Resolve-ReproducibilityVerdict -Declared $declared -Locked $locked `
            -StatusSummary $(if ($null -ne $status) { $status.Summary } else { $null }) `
            -StatusProblem $(if ($null -ne $status) { $status.Problem } else { "" }) -Scratch $scratch
    }

    # --- Entregables ---
    $phase2 = Invoke-Guarded -Id "fase2" -Name "Sin regresion de Fase 2 (criterios 2 y 4)" -Blocking $false -Evaluate {
        Resolve-Phase2RegressionVerdict -Report (Read-JsonFile -Path (Join-Path $results $Phase2ReportFile)) -HeadCommit $commit
    }
    $docs = Invoke-Guarded -Id "docs" -Name "Documentos de la Fase 3" -Blocking $false -Evaluate {
        $sizes = [ordered]@{}
        foreach ($relative in $RequiredDocs) {
            $path = Join-Path $root $relative
            $sizes[$relative] = if (Test-Path -LiteralPath $path -PathType Leaf) { (Get-Item -LiteralPath $path).Length } else { -1 }
        }
        Resolve-DocsVerdict -Sizes $sizes
    }

    $criteria = @($criterion1, $criterion2, $criterion3, $criterion4)
    $deliverables = @($phase2, $docs)
    $exitCode = Get-ExitCode -Verdicts $criteria

    Write-Host ""
    Write-Host "CRITERIOS (bloquean el codigo de salida)" -ForegroundColor White
    foreach ($verdict in $criteria) { Write-VerdictLine -Verdict $verdict }
    Write-Host ""
    Write-Host "ENTREGABLES (aviso, no bloquean)" -ForegroundColor White
    foreach ($verdict in $deliverables) { Write-VerdictLine -Verdict $verdict }

    $document = New-ReportDocument -Criteria $criteria -Deliverables $deliverables -ExitCode $exitCode `
        -Commit $commit -GeneratedAt ([DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ"))
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
        exit (Invoke-VerifyPhase3)
    }
    catch {
        Write-Host ""
        Write-Host "ERROR del verificador: $($_.Exception.Message)" -ForegroundColor Red
        exit $ExitError
    }
}
