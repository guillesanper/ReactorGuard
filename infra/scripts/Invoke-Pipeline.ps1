<#
.SYNOPSIS
    Run the ReactorGuard DVC data pipeline with a correctly resolved interpreter.

.DESCRIPTION
    Sustituye a data/generators/prepare_tep.ps1, que orquestaba a mano la misma
    secuencia que hoy describe dvc.yaml. Aquel script tenia dos problemas de
    fondo: duplicaba el grafo de dependencias (invocando descarga, exploracion y
    adaptacion en orden fijo, sin saltarse lo que ya estaba al dia) y su
    OutputDir por defecto escribia en data/processed/tep, que es un `out` del
    stage adapt_tep, de modo que ejecutarlo dejaba el estado de DVC sucio.

    Aqui el grafo lo posee dvc.yaml y este script solo se ocupa de lo que DVC no
    puede resolver por si mismo: que `dvc repro` lance el interprete del venv.
    dvc.exe no activa el entorno virtual, asi que `python` dentro de un `cmd:`
    resuelve al Python del sistema y los stages fallan con
    ModuleNotFoundError (tqdm, pyarrow, ...) pese a estar todo instalado.

    Cuatro capas:
        Capa 1 - Configuracion : resolver ProjectRoot y el interprete del venv,
                                 y anteponerlo al PATH
        Capa 2 - Preflight     : verificar que dvc responde y que params.yaml parsea
        Capa 3 - Ejecucion     : dvc repro, opcionalmente acotado o forzado
        Capa 4 - Reporte       : dvc status y resumen de particiones y conteos

.PARAMETER Stage
    Reproduce only this stage and its dependencies. Defaults to the whole graph.

.PARAMETER Force
    Reproduce even if DVC considers the stages up to date.

.EXAMPLE
    .\infra\scripts\Invoke-Pipeline.ps1
    .\infra\scripts\Invoke-Pipeline.ps1 -Stage adapt_tep
    .\infra\scripts\Invoke-Pipeline.ps1 -Stage adapt_tep -Force
#>

[CmdletBinding()]
param (
    [string]$Stage,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===========================================================================
# Capa 1 - Configuracion
# ===========================================================================

Write-Host "[Capa 1/4] Configuracion"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvPython  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$VenvScripts = Join-Path $ProjectRoot ".venv\Scripts"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Interprete del venv no encontrado en $VenvPython. Ejecuta infra/scripts/Setup-DevEnv.ps1 primero."
    exit 1
}

# Anteponer, nunca sustituir: dvc.exe invoca `python` para cada `cmd:` y debe
# encontrar el del venv antes que el del sistema. Sin esto los stages fallan con
# ModuleNotFoundError aunque las dependencias esten instaladas.
$env:PATH        = "$VenvScripts;$env:PATH"
$env:PYTHONPATH  = $ProjectRoot

Set-Location $ProjectRoot

$resolvedPython = (Get-Command python).Source
if ($resolvedPython -ne $VenvPython) {
    Write-Error "`python` resuelve a $resolvedPython en lugar de $VenvPython. Aborto antes de ensuciar el estado de DVC."
    exit 1
}

Write-Host "  Project root : $ProjectRoot"
Write-Host "  Python       : $resolvedPython"

# ===========================================================================
# Capa 2 - Preflight
# ===========================================================================

Write-Host ""
Write-Host "[Capa 2/4] Preflight"

$dvcCommand = Get-Command dvc -ErrorAction SilentlyContinue
if ($null -eq $dvcCommand) {
    Write-Error "dvc no esta disponible en el PATH tras anteponer el venv. Ejecuta infra/scripts/Setup-DevEnv.ps1."
    exit 1
}
Write-Host "  dvc          : $($dvcCommand.Source)"

& $VenvPython -c "import yaml,sys; yaml.safe_load(open('params.yaml',encoding='utf-8')) or sys.exit('params.yaml esta vacio')"
if ($LASTEXITCODE -ne 0) {
    Write-Error "params.yaml no parsea. Revisa la cabecera del fichero antes de depurar los stages."
    exit 1
}
Write-Host "  params.yaml  : parsea correctamente"

$spansPath = Join-Path $ProjectRoot "configs\sensor_spans.yaml"
if (-not (Test-Path $spansPath)) {
    Write-Error "Falta configs/sensor_spans.yaml. Generalo con: python data/generators/derive_sensor_spans.py"
    exit 1
}
Write-Host "  spans        : configs/sensor_spans.yaml presente"

# ===========================================================================
# Capa 3 - Ejecucion
# ===========================================================================

Write-Host ""
Write-Host "[Capa 3/4] Ejecucion - dvc repro"

$reproArgs = @("repro")
if ($Force) { $reproArgs += "--force" }
if ($Stage) { $reproArgs += $Stage }

Write-Host "  dvc $($reproArgs -join ' ')"
Write-Host ""

& dvc @reproArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "dvc repro fallo con codigo $LASTEXITCODE."
    exit 1
}

# ===========================================================================
# Capa 4 - Reporte
# ===========================================================================

Write-Host ""
Write-Host "[Capa 4/4] Reporte"
Write-Host ""

& dvc status

$processedDir = Join-Path $ProjectRoot "data\processed\tep"
if (Test-Path $processedDir) {
    Write-Host ""
    Write-Host "  Particiones en data/processed/tep:"

    $summary = & $VenvPython -c @"
import pathlib, sys
import pandas as pd
root = pathlib.Path(r'$processedDir')
total = 0
for part in sorted(root.glob('fault_type=*/readings.parquet')):
    n = len(pd.read_parquet(part, columns=['value']))
    total += n
    label = 'normal' if part.parent.name.endswith('=00') else 'fault'
    print(f"    {part.parent.name} ({label:6}): {n:>8,} lecturas")
print(f"    {'TOTAL':<20}: {total:>8,} lecturas")
"@
    if ($LASTEXITCODE -eq 0) { $summary | ForEach-Object { Write-Host $_ } }
}

Write-Host ""
Write-Host "Pipeline completado."
