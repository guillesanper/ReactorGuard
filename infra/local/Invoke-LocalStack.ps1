#Requires -Version 7.0
<#
.SYNOPSIS
    Levanta, para o inspecciona el stack local de ReactorGuard (Kafka y Redis).

.DESCRIPTION
    Envuelve infra/local/docker-compose.yaml. El stack es de UN broker PLAINTEXT sin
    replicacion: sirve para las pruebas de integracion y los benchmarks, y una cifra
    medida aqui es informativa (solo el cluster cierra los criterios 1 y 3).

    Cuatro capas:
        Capa 1 - Configuracion : rutas, nombre de proyecto compose, tiempos de espera
        Capa 2 - Utilidades    : funciones puras (argumentos de compose, estado)
        Capa 3 - Servicio      : llamadas a docker y esperas sobre contenedores
        Capa 4 - Orquestacion  : -Up, -Down y -Status

    -Up espera a que Kafka y Redis esten sanos y a que kafka-init termine de crear
    los tres topics, y al final muestra las variables de entorno que usan los tests.

.PARAMETER Up
    Levanta el stack y espera a que este listo.

.PARAMETER Down
    Detiene y elimina los contenedores (incluidos los del perfil observability). El
    stack no declara volumenes: bajarlo borra los datos.

.PARAMETER Status
    Muestra el estado de cada contenedor del stack.

.PARAMETER Observability
    Con -Up, anade Prometheus (127.0.0.1:9091) y Grafana (127.0.0.1:3000).

.EXAMPLE
    .\infra\local\Invoke-LocalStack.ps1 -Up
    $env:KAFKA_BOOTSTRAP = "127.0.0.1:9092"
    .\.venv\Scripts\python.exe -m pytest tests/integration -m integration
    .\infra\local\Invoke-LocalStack.ps1 -Down

.NOTES
    Codigos de salida: 0 correcto; 1 error (Docker ausente, timeout o fallo de compose).
#>

[CmdletBinding(DefaultParameterSetName = "Status")]
param (
    [Parameter(Mandatory, ParameterSetName = "Up")]
    [switch]$Up,

    [Parameter(Mandatory, ParameterSetName = "Down")]
    [switch]$Down,

    [Parameter(ParameterSetName = "Status")]
    [switch]$Status,

    [Parameter(ParameterSetName = "Up")]
    [switch]$Observability
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ===========================================================================
# Capa 1 - Configuracion
# ===========================================================================

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ComposeFile = Join-Path $ScriptDir "docker-compose.yaml"
$ProjectName = "reactorguard-local"

$KafkaContainer = "reactorguard-kafka"
$InitContainer  = "reactorguard-kafka-init"
$RedisContainer = "reactorguard-redis"

# 127.0.0.1 y no localhost: en Windows localhost resuelve primero a ::1 y el puerto
# solo esta publicado en IPv4.
$KafkaBootstrap = "127.0.0.1:9092"
$RedisPort      = if ($env:REACTORGUARD_REDIS_PORT) { $env:REACTORGUARD_REDIS_PORT } else { "6380" }

# El arranque de Kafka KRaft en frio tarda decenas de segundos; 180 s deja margen
# para una maquina cargada sin ocultar un fallo real.
$HealthTimeoutSeconds = 180
$PollSeconds          = 3

# ===========================================================================
# Capa 2 - Utilidades (puras: sin efectos secundarios)
# ===========================================================================

function Get-ComposeBaseArgs {
    <#
    .SYNOPSIS
        Devuelve los argumentos comunes de `docker compose`.
    .OUTPUTS
        [string[]]
    #>
    param(
        [Parameter(Mandatory)][string]$File,
        [Parameter(Mandatory)][string]$Project,
        [switch]$WithObservability
    )

    $composeArgs = @("compose", "--file", $File, "--project-name", $Project)
    if ($WithObservability) {
        $composeArgs += @("--profile", "observability")
    }
    return $composeArgs
}

function Get-ContainerVerdict {
    <#
    .SYNOPSIS
        Clasifica el estado de un contenedor de larga vida como Ready, Waiting o Failed.
    .OUTPUTS
        [string]
    #>
    param(
        [Parameter(Mandatory)][string]$State,
        [string]$Health
    )

    if ($State -in @("exited", "dead")) { return "Failed" }
    if ($State -ne "running") { return "Waiting" }
    switch ($Health) {
        "healthy"   { return "Ready" }
        "unhealthy" { return "Failed" }
        default     { return "Waiting" }
    }
}

function Get-InitVerdict {
    <#
    .SYNOPSIS
        Clasifica el estado del contenedor one-shot kafka-init.
    .OUTPUTS
        [string]
    #>
    param(
        [Parameter(Mandatory)][string]$State,
        [int]$ExitCode
    )

    if ($State -eq "exited") {
        if ($ExitCode -eq 0) { return "Ready" }
        return "Failed"
    }
    if ($State -eq "dead") { return "Failed" }
    return "Waiting"
}

# ===========================================================================
# Capa 3 - Servicio (docker)
# ===========================================================================

function Assert-DockerAvailable {
    if ($null -eq (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "docker no esta en el PATH. Instala Docker Desktop."
    }
    docker info --format "{{.ServerVersion}}" *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "El demonio de Docker no responde. Arranca Docker Desktop y reintenta."
    }
}

function Get-ContainerInspect {
    <#
    .SYNOPSIS
        Lee estado, salud y codigo de salida de un contenedor; $null si no existe.
    #>
    param([Parameter(Mandatory)][string]$Name)

    $format = "{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{end}}|{{.State.ExitCode}}"
    $raw = docker inspect --format $format $Name 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($raw)) { return $null }
    $parts = $raw.Trim().Split("|")
    return [pscustomobject]@{
        State    = $parts[0]
        Health   = $parts[1]
        ExitCode = [int]$parts[2]
    }
}

function Wait-ForVerdict {
    <#
    .SYNOPSIS
        Espera a que un contenedor llegue a Ready; lanza si falla o se agota el tiempo.
    #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][ValidateSet("Service", "Init")][string]$Kind
    )

    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $info = Get-ContainerInspect -Name $Name
        if ($null -ne $info) {
            $verdict = if ($Kind -eq "Init") {
                Get-InitVerdict -State $info.State -ExitCode $info.ExitCode
            } else {
                Get-ContainerVerdict -State $info.State -Health $info.Health
            }
            if ($verdict -eq "Ready") { return }
            if ($verdict -eq "Failed") {
                docker logs --tail 40 $Name
                throw "El contenedor $Name fallo (estado=$($info.State), salud=$($info.Health), salida=$($info.ExitCode))."
            }
        }
        Start-Sleep -Seconds $PollSeconds
    }
    docker logs --tail 40 $Name
    throw "Tiempo agotado ($HealthTimeoutSeconds s) esperando a $Name."
}

function Invoke-Compose {
    param(
        [Parameter(Mandatory)][string[]]$BaseArgs,
        [Parameter(Mandatory)][string[]]$Command
    )

    & docker @BaseArgs @Command
    if ($LASTEXITCODE -ne 0) {
        throw "docker $($Command -join ' ') fallo con codigo $LASTEXITCODE."
    }
}

# ===========================================================================
# Capa 4 - Orquestacion
# ===========================================================================

function Start-Stack {
    param([switch]$WithObservability)

    Write-Host "Levantando el stack local ($ComposeFile)"
    $baseArgs = Get-ComposeBaseArgs -File $ComposeFile -Project $ProjectName -WithObservability:$WithObservability
    Invoke-Compose -BaseArgs $baseArgs -Command @("up", "--detach")

    Write-Host "Esperando a Kafka..."
    Wait-ForVerdict -Name $KafkaContainer -Kind Service
    Write-Host "Esperando a que kafka-init cree los topics..."
    Wait-ForVerdict -Name $InitContainer -Kind Init
    Write-Host "Esperando a Redis..."
    Wait-ForVerdict -Name $RedisContainer -Kind Service

    Write-Host ""
    Write-Host "Stack listo. Variables para los tests y benchmarks:"
    Write-Host "  `$env:KAFKA_BOOTSTRAP = `"$KafkaBootstrap`""
    Write-Host "  Redis en 127.0.0.1:$RedisPort"
    if ($WithObservability) {
        Write-Host "  Prometheus http://localhost:9091   Grafana http://localhost:3000"
    }
}

function Stop-Stack {
    Write-Host "Bajando el stack local"
    $baseArgs = Get-ComposeBaseArgs -File $ComposeFile -Project $ProjectName -WithObservability
    Invoke-Compose -BaseArgs $baseArgs -Command @("down", "--remove-orphans")
}

function Show-Status {
    $baseArgs = Get-ComposeBaseArgs -File $ComposeFile -Project $ProjectName -WithObservability
    Invoke-Compose -BaseArgs $baseArgs -Command @("ps", "--all")
}

try {
    Assert-DockerAvailable
    switch ($PSCmdlet.ParameterSetName) {
        "Up"     { Start-Stack -WithObservability:$Observability }
        "Down"   { Stop-Stack }
        default  { Show-Status }
    }
    exit 0
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
