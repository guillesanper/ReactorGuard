<#
.SYNOPSIS
    Prepare the Tennessee Eastman Process dataset for ReactorGuard.

.DESCRIPTION
    Orchestrates the full TEP data pipeline across four logical layers:

        Layer 1 - Configuration  : resolve project root, Python interpreter, output paths
        Layer 2 - Acquisition    : download raw .dat files from the Prof. Braatz GitHub repo
        Layer 3 - Transformation : produce exploration report and adapt to SensorReading schema
        Layer 4 - Persistence    : write fault_type-partitioned Parquet files and print statistics

    On success the following artefacts are produced:
        data/raw/tep/d00.dat .. d21.dat           raw TEP files
        data/raw/tep/tep_checksums.json           MD5 checksums of raw files
        data/reports/tep_exploration.json         per-file descriptive statistics
        data/reports/tep_correlations.csv         Pearson correlation matrix
        data/processed/tep/fault_type=00/readings.parquet  adapted readings, normal
        data/processed/tep/fault_type=01/readings.parquet  adapted readings, fault 1
        ...

    El parquet se escribe fuera de data/raw/tep para no anidarlo dentro del out
    del stage download_tep de dvc.yaml, que provocaria una colision de outs.

.PARAMETER DataDir
    Directory for raw TEP .dat files. Defaults to data/raw/tep.

.PARAMETER OutputDir
    Root directory for partitioned Parquet output. Defaults to data/processed/tep.

.EXAMPLE
    .\prepare_tep.ps1
    .\prepare_tep.ps1 -DataDir D:\datasets\tep -OutputDir D:\datasets\tep_parquet
#>

param (
    [string]$DataDir   = "data/raw/tep",
    [string]$OutputDir = "data/processed/tep"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===========================================================================
# Layer 1 - Configuration
# ===========================================================================

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$VenvPython  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Error "Virtual environment not found at $VenvPython. Run infra/scripts/Setup-DevEnv.ps1 first."
    exit 1
}

$env:PYTHONPATH = $ProjectRoot

Write-Host "[Layer 1/4] Configuration"
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Python       : $VenvPython"
Write-Host "  Data dir     : $DataDir"
Write-Host "  Output dir   : $OutputDir"

# ===========================================================================
# Layer 2 - Acquisition
# ===========================================================================

Write-Host ""
Write-Host "[Layer 2/4] Acquisition - downloading TEP dataset"

$downloadScript = @"
import sys, logging
sys.path.insert(0, r'$ProjectRoot')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from data.generators.tep_downloader import download_tep
download_tep(r'$DataDir')
"@

$tmpDownload = [System.IO.Path]::GetTempFileName() + ".py"
Set-Content -Path $tmpDownload -Value $downloadScript -Encoding UTF8
try {
    & $VenvPython $tmpDownload
    if ($LASTEXITCODE -ne 0) { Write-Error "Download step failed (exit code $LASTEXITCODE)." ; exit 1 }
} finally {
    Remove-Item -Path $tmpDownload -Force -ErrorAction SilentlyContinue
}

# ===========================================================================
# Layer 3 - Transformation
# ===========================================================================

Write-Host ""
Write-Host "[Layer 3/4] Transformation - exploration report"

$exploreScript = @"
import sys, logging
sys.path.insert(0, r'$ProjectRoot')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from data.generators.tep_explorer import explore_tep
explore_tep(r'$DataDir')
"@

$tmpExplore = [System.IO.Path]::GetTempFileName() + ".py"
Set-Content -Path $tmpExplore -Value $exploreScript -Encoding UTF8
try {
    & $VenvPython $tmpExplore
    if ($LASTEXITCODE -ne 0) { Write-Error "Exploration step failed (exit code $LASTEXITCODE)." ; exit 1 }
} finally {
    Remove-Item -Path $tmpExplore -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "[Layer 3/4] Transformation - adapting to SensorReading schema"

$adaptScript = @"
import sys, logging
sys.path.insert(0, r'$ProjectRoot')
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from data.generators.tep_adapter import TEPAdapter, save_to_parquet

adapter = TEPAdapter()
df = adapter.adapt_all(r'$DataDir')
save_to_parquet(df, r'$OutputDir')

# Store summary for Layer 4 output
import json, pathlib
summary = {
    'total_readings': len(df),
    'distribution': {
        str(int(ft)): int(count)
        for ft, count in df.groupby('fault_type').size().items()
    },
}
pathlib.Path(r'$OutputDir', 'adapt_summary.json').write_text(
    json.dumps(summary, indent=2), encoding='utf-8'
)
"@

$tmpAdapt = [System.IO.Path]::GetTempFileName() + ".py"
Set-Content -Path $tmpAdapt -Value $adaptScript -Encoding UTF8
try {
    & $VenvPython $tmpAdapt
    if ($LASTEXITCODE -ne 0) { Write-Error "Adaptation step failed (exit code $LASTEXITCODE)." ; exit 1 }
} finally {
    Remove-Item -Path $tmpAdapt -Force -ErrorAction SilentlyContinue
}

# ===========================================================================
# Layer 4 - Persistence (summary statistics)
# ===========================================================================

Write-Host ""
Write-Host "[Layer 4/4] Persistence - final statistics"

$summaryPath = Join-Path $OutputDir "adapt_summary.json"
if (Test-Path $summaryPath) {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    Write-Host ("  Total readings : {0:N0}" -f $summary.total_readings)
    Write-Host "  Distribution by fault_type:"
    $summary.distribution.PSObject.Properties | Sort-Object { [int]$_.Name } | ForEach-Object {
        $ft    = [int]$_.Name
        $count = $_.Value
        $label = if ($ft -eq 0) { "normal" } else { "fault_{0:D2}" -f $ft }
        Write-Host ("    fault_type={0,2:D} ({1,-10}): {2,8:N0} readings" -f $ft, $label, $count)
    }
    Remove-Item $summaryPath -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "TEP dataset preparation complete."
Write-Host "  Raw files  : $DataDir"
Write-Host "  Parquet    : $OutputDir"
Write-Host "  Reports    : data/reports/"
