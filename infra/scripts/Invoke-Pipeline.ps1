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
        Capa 2 - Preflight     : verificar que dvc responde, que params.yaml parsea
                                 y que existen las tablas que leen los stages
        Capa 3 - Ejecucion     : dvc repro, opcionalmente acotado o forzado
        Capa 4 - Reporte       : dvc status y resumen de particiones y conteos
                                 (TEP y simulacion)

    El stage `simulate` (F3-M6) es el mas caro del grafo y lo domina el vacio
    (F3-M5 midio ~840 s por run en Windows; Docs/physics_constraints.md, 14.1).
    Reparte los runs entre un proceso por CPU; el resultado no depende de cuantos.

    El stage `prepare_sim` (F4-M1) cuelga de `simulate` y escribe los datos de la
    PINN repartidos por run en data/processed/sim_pinn (segundos).

.PARAMETER Stage
    Reproduce only this stage and its dependencies. Defaults to the whole graph.

.PARAMETER Force
    Reproduce even if DVC considers the stages up to date.

.EXAMPLE
    .\infra\scripts\Invoke-Pipeline.ps1
    .\infra\scripts\Invoke-Pipeline.ps1 -Stage adapt_tep
    .\infra\scripts\Invoke-Pipeline.ps1 -Stage adapt_tep -Force
    .\infra\scripts\Invoke-Pipeline.ps1 -Stage simulate
    .\infra\scripts\Invoke-Pipeline.ps1 -Stage prepare_sim
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

# Las tablas de la simulacion solo se exigen si el stage simulate puede correr
# (grafo entero, -Stage simulate o -Stage prepare_sim, que depende de el y lee el
# catalogo y plant_parameters.yaml): reproducir solo la rama TEP no las necesita.
if (-not $Stage -or $Stage -in @("simulate", "prepare_sim")) {
    $simConfigs = @(
        @{ Path = "configs\sim_sensors.yaml";            Hint = "catalogo de sensores (F3-M1), versionado en git" },
        @{ Path = "configs\sim_sensor_spans.yaml";       Hint = "python data/generators/derive_sim_spans.py" },
        @{ Path = "configs\plant_parameters.yaml";       Hint = "parametros de planta (F3-M1), versionado en git" },
        @{ Path = "configs\neutronics_coefficients.yaml"; Hint = "coeficientes neutronicos (F3-M1), versionado en git" },
        @{ Path = "configs\water_table.yaml";            Hint = "python -m data.simulation.build_water_table --output configs/water_table.yaml" }
    )
    foreach ($config in $simConfigs) {
        if (-not (Test-Path (Join-Path $ProjectRoot $config.Path))) {
            Write-Error "Falta $($config.Path) ($($config.Hint)). El stage simulate no puede correr sin ella."
            exit 1
        }
    }
    Write-Host "  simulacion   : $($simConfigs.Count) tablas de configs/ presentes"
}

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

$simulationDir = Join-Path $ProjectRoot "data\raw\simulation"
if (Test-Path $simulationDir) {
    Write-Host ""
    Write-Host "  Runs en data/raw/simulation:"

    $summary = & $VenvPython -c @"
import json, pathlib
root = pathlib.Path(r'$simulationDir')
manifest = root / 'manifest.json'
runs = json.loads(manifest.read_text(encoding='utf-8'))['runs'] if manifest.exists() else []
by_type = {}
for run in runs:
    entry = by_type.setdefault(run['scenario'], [0, 0, 0])
    entry[0] += 1
    entry[1] += run['rows']
    entry[2] += run['labelled']
for scenario in sorted(by_type):
    n_runs, rows, labelled = by_type[scenario]
    print(f"    fault_type={scenario:<24}: {n_runs:>3} runs, {rows:>9,} lecturas, {labelled:>9,} etiquetadas")
total = sum(e[1] for e in by_type.values())
labelled = sum(e[2] for e in by_type.values())
share = 100.0 * labelled / total if total else 0.0
print(f"    {'TOTAL':<35}: {len(runs):>3} runs, {total:>9,} lecturas, {labelled:>9,} etiquetadas ({share:.2f} %)")
if not manifest.exists():
    print('    (sin manifest.json: el stage simulate no termino)')
"@
    if ($LASTEXITCODE -eq 0) { $summary | ForEach-Object { Write-Host $_ } }
}

$simPinnMetrics = Join-Path $ProjectRoot "metrics\sim_pinn_metrics.json"
if (Test-Path $simPinnMetrics) {
    Write-Host ""
    Write-Host "  Datos de la PINN en data/processed/sim_pinn (reparto por run, D4-7):"

    $summary = & $VenvPython -c @"
import json, pathlib
metrics = json.loads(pathlib.Path(r'$simPinnMetrics').read_text(encoding='utf-8'))
for name in ('train', 'val', 'test'):
    s = metrics['splits'][name]
    print(f"    {name:<5}: {s['runs']:>2} runs, {s['instants']:>6,} instantes, "
          f"{s['critical_instants']:>5,} CRITICAL, {s['trainable_pairs']:>6,} pares, "
          f"rejilla {100.0 * s['coverage_fraction']:.1f} %")
"@
    if ($LASTEXITCODE -eq 0) { $summary | ForEach-Object { Write-Host $_ } }
}

Write-Host ""
Write-Host "Pipeline completado."
