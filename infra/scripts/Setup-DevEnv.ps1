#Requires -Version 7.0
<#
.SYNOPSIS
    Reconstruye el entorno de desarrollo local de ReactorGuard y captura la linea
    base de calidad (ruff, mypy, pytest) del arbol de trabajo actual.

.DESCRIPTION
    Pensado para una maquina recien clonada: .venv/, data/raw/ y data/processed/
    estan en .gitignore, asi que tras el clon no hay entorno ejecutable.

    Pasos que ejecuta:
      1. Localiza el interprete Python 3.12 (py -3.12). El 3.14 por defecto no
         tiene ruedas de torch, por lo que la version es obligatoria, no preferente.
      2. Crea .venv (o la reutiliza; -Force la recrea desde cero).
      3. Actualiza pip/setuptools/wheel e instala el proyecto con pip install -e ".[dev]".
      4. Ejecuta ruff, mypy y pytest y resume el resultado de cada uno.

    El script NO falla si ruff/mypy/pytest encuentran errores: su proposito es
    medir el estado real del repo, no imponerlo. Solo termina con exit 1 si el
    entorno no se pudo construir (paso 1-3).

.PARAMETER Force
    Elimina .venv si ya existe y la recrea desde cero.

.PARAMETER SkipChecks
    Construye el entorno pero omite la fase de ruff/mypy/pytest.

.PARAMETER PythonVersion
    Version del interprete a resolver via el lanzador py. Por defecto: 3.12

.EXAMPLE
    .\Setup-DevEnv.ps1
    .\Setup-DevEnv.ps1 -Force
    .\Setup-DevEnv.ps1 -SkipChecks
#>

# =============================================================================
# CAPA 1 - CONFIGURACION
# =============================================================================
[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipChecks,
    [string]$PythonVersion = "3.12"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$VenvPath   = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

# =============================================================================
# CAPA 2 - UTILIDADES
# =============================================================================

function Write-Step {
    <#
    .SYNOPSIS
        Imprime el encabezado de una fase del script.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Message
    )

    Write-Host ""
    Write-Host "  [paso] $Message" -ForegroundColor Cyan
}

function Write-CheckResult {
    <#
    .SYNOPSIS
        Imprime el resultado de un check con marca OK/FAIL y su detalle.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][bool]  $Passed,
        [Parameter(Mandatory)][string]$Detail
    )

    $mark  = if ($Passed) { "OK  " } else { "FAIL" }
    $color = if ($Passed) { "Green" } else { "Red" }
    Write-Host ("  {0}  {1,-28} {2}" -f $mark, $Name, $Detail) -ForegroundColor $color
}

function Resolve-PythonInterpreter {
    <#
    .SYNOPSIS
        Resuelve la ruta al interprete de la version pedida via el lanzador py.
    .OUTPUTS
        String con la ruta absoluta al ejecutable.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Version
    )

    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw "El lanzador de Python (py) no esta disponible en PATH."
    }

    $listing = & py -0p 2>&1
    $match = $listing | Select-String -Pattern "-V:$([regex]::Escape($Version))\s" | Select-Object -First 1

    if ($null -eq $match) {
        throw "Python $Version no esta instalado. Versiones disponibles:`n$($listing -join "`n")"
    }

    # Formato de la linea: " -V:3.12          C:\ruta\python.exe"
    $path = ($match.Line -split "\s{2,}")[-1].Trim()
    if (-not (Test-Path $path)) {
        throw "El lanzador reporta Python $Version en '$path' pero la ruta no existe."
    }

    return $path
}

function Invoke-QualityCheck {
    <#
    .SYNOPSIS
        Ejecuta un comando del venv y devuelve si termino con exit code 0.
    .OUTPUTS
        Hashtable con: Passed [bool], Detail [string]
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]  $Name,
        [Parameter(Mandatory)][string[]]$Arguments
    )

    Write-Host "  Ejecutando: $Name..." -ForegroundColor Gray
    $output = & $VenvPython @Arguments 2>&1
    $exitCode = $LASTEXITCODE

    $output | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }

    # Preferir la linea de recuento ("Found N errors", "N failed, M passed") sobre la
    # ultima linea, que en ruff es la sugerencia de --fix y no el resumen.
    $nonEmpty = @($output | Where-Object { $_ -match "\S" })
    $summary  = $nonEmpty | Where-Object { $_ -match "Found \d+ error|\d+ (failed|passed)|Success" } |
        Select-Object -Last 1
    if ($null -eq $summary) {
        $summary = $nonEmpty | Select-Object -Last 1
    }

    return @{
        Passed = ($exitCode -eq 0)
        Detail = if ($null -eq $summary) { "sin salida" } else { ($summary.ToString().Trim() -replace "=+", "").Trim() }
    }
}

# =============================================================================
# CAPA 3 - FASES
# =============================================================================

function New-VirtualEnv {
    <#
    .SYNOPSIS
        Crea el virtualenv .venv con el interprete indicado, recreandolo si -Force.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Interpreter
    )

    if (Test-Path $VenvPath) {
        if ($Force) {
            Write-Host "  .venv existe y se pidio -Force: eliminando..." -ForegroundColor Yellow
            Remove-Item -Recurse -Force $VenvPath
        }
        else {
            Write-Host "  .venv ya existe, se reutiliza (usa -Force para recrearla)." -ForegroundColor Yellow
            return
        }
    }

    & $Interpreter -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        throw "La creacion del virtualenv fallo con exit code $LASTEXITCODE."
    }
}

function Install-Dependencies {
    <#
    .SYNOPSIS
        Actualiza pip y instala el proyecto en modo editable con el extra dev.
    #>
    [CmdletBinding()]
    param()

    & $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "La actualizacion de pip/setuptools/wheel fallo con exit code $LASTEXITCODE."
    }

    & $VenvPython -m pip install -e "$RepoRoot[dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "pip install -e '.[dev]' fallo con exit code $LASTEXITCODE."
    }
}

function Invoke-Baseline {
    <#
    .SYNOPSIS
        Ejecuta ruff, mypy y pytest e imprime el resumen por herramienta.
    .OUTPUTS
        Int con el numero de herramientas que reportaron fallos.
    #>
    [CmdletBinding()]
    param()

    $checks = [ordered]@{
        "ruff (E,W,F,I)" = @("-m", "ruff", "check", ".", "--select", "E,W,F,I", "--output-format=concise")
        "mypy"           = @("-m", "mypy", "ml/", "api/", "data/", "--ignore-missing-imports")
        "pytest"         = @("-m", "pytest", "tests/unit", "tests/safety", "-q", "--no-header")
    }

    $results = [ordered]@{}
    foreach ($name in $checks.Keys) {
        $results[$name] = Invoke-QualityCheck -Name $name -Arguments $checks[$name]
    }

    Write-Host ""
    Write-Host "  Linea base del arbol de trabajo:" -ForegroundColor Cyan
    foreach ($name in $results.Keys) {
        Write-CheckResult -Name $name -Passed $results[$name].Passed -Detail $results[$name].Detail
    }

    return ($results.Values | Where-Object { -not $_.Passed }).Count
}

# =============================================================================
# CAPA 4 - ORQUESTACION
# =============================================================================

function Invoke-SetupDevEnv {
    <#
    .SYNOPSIS
        Orquesta la reconstruccion del entorno y la captura de la linea base.
    #>
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  ReactorGuard - Reconstruccion del entorno de desarrollo" -ForegroundColor Cyan
    Write-Host "  Repo   : $RepoRoot" -ForegroundColor Cyan
    Write-Host "  Python : $PythonVersion (obligatorio: torch no publica ruedas para 3.14)" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan

    Write-Step "Resolviendo interprete Python $PythonVersion"
    $interpreter = Resolve-PythonInterpreter -Version $PythonVersion
    Write-Host "  Interprete: $interpreter" -ForegroundColor Gray

    Write-Step "Creando virtualenv en .venv"
    New-VirtualEnv -Interpreter $interpreter

    Write-Step "Instalando dependencias (pip install -e '.[dev]')"
    Install-Dependencies

    if ($SkipChecks) {
        Write-Host ""
        Write-Host "  Entorno listo. Checks omitidos (-SkipChecks)." -ForegroundColor Green
        Write-Host "  Activar con: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
        Write-Host ""
        exit 0
    }

    Write-Step "Midiendo la linea base (ruff, mypy, pytest)"
    $failed = Invoke-Baseline

    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host "  Entorno reconstruido correctamente." -ForegroundColor Green
    if ($failed -gt 0) {
        Write-Host "  $failed de 3 herramientas reportan fallos en el arbol actual." -ForegroundColor Yellow
        Write-Host "  Es la linea base esperada antes de unificar el schema (Paso 1)." -ForegroundColor Yellow
    }
    else {
        Write-Host "  ruff, mypy y pytest en verde." -ForegroundColor Green
    }
    Write-Host "  Activar con: .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
    Write-Host "================================================================" -ForegroundColor Cyan
    Write-Host ""

    exit 0
}

# Punto de entrada
Invoke-SetupDevEnv
