#Requires -Version 7.0
# =============================================================================
# infra/scripts/Install-Kafka.ps1
# Instala el operador Strimzi y despliega el cluster Kafka de ReactorGuard.
#
# Diferencia clave entre el operador y los recursos custom:
#   - El OPERADOR (Strimzi) es un Deployment K8s que corre continuamente y
#     observa recursos de tipo Kafka, KafkaTopic, KafkaUser.
#   - Los RECURSOS CUSTOM (kafka-cluster.yaml, kafka-topics.yaml, kafka-users.yaml)
#     son declaraciones de la intención deseada. El operador los lee y actúa.
#   Sin el operador activo, apply de los CRDs no tiene ningún efecto.
#
# Orden de aplicación (importa):
#   1. kafka-metrics.yaml  ConfigMap que kafka-cluster.yaml referencia; sin él los
#                          brokers no arrancan.
#   2. kafka-cluster.yaml  brokers, ZooKeeper y Entity Operator.
#   3. kafka-topics.yaml   los 4 topics (3 del flujo y bench-throughput).
#   4. kafka-users.yaml    los 4 KafkaUser (mTLS + ACLs). El User Operator publica el
#                          Secret de cada uno en este namespace; después hay que
#                          copiarlos a reactorguard-ingestion con
#                          infra/scripts/Sync-KafkaCredentials.ps1.
#
# Uso:
#   .\infra\scripts\Install-Kafka.ps1
#   .\infra\scripts\Install-Kafka.ps1 -Namespace kafka-operator -Verbose
# =============================================================================

[CmdletBinding()]
param(
    [string]$Namespace      = "kafka-operator",
    [string]$StrimziVersion = "0.39.0",
    [int]$TimeoutSeconds    = 300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =============================================================================
# CAPA 2 — FUNCIONES DE UTILIDAD (sin efectos secundarios, reutilizables)
# =============================================================================

function Wait-KubernetesPod {
    <#
    .SYNOPSIS
        Espera a que los pods que coincidan con un LabelSelector estén Ready.
    .DESCRIPTION
        Wrapper de `kubectl wait` que lanza una excepción si el timeout expira
        antes de que los pods alcancen el estado Running/Ready.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$LabelSelector,
        [Parameter(Mandatory)][string]$Namespace,
        [int]$TimeoutSec = 120
    )

    Write-Verbose "Esperando pods con label '$LabelSelector' en ns '$Namespace' (timeout ${TimeoutSec}s)..."
    $result = kubectl wait pod `
        --selector="$LabelSelector" `
        --namespace="$Namespace" `
        --for=condition=Ready `
        --timeout="${TimeoutSec}s" 2>&1

    if ($LASTEXITCODE -ne 0) {
        throw "Timeout esperando pods ($LabelSelector): $result"
    }
    Write-Verbose "Pods Ready: $result"
}

function Test-KafkaClusterReady {
    <#
    .SYNOPSIS
        Verifica que el KafkaCluster esté en estado READY y todos los brokers Running.
    .OUTPUTS
        [bool] True si el cluster está Ready, False en caso contrario.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Namespace
    )

    try {
        # Verificar el status del recurso Kafka custom
        $kafkaStatus = kubectl get kafka reactorguard-cluster `
            --namespace="$Namespace" `
            --output=jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>$null

        if ($kafkaStatus -ne "True") {
            return $false
        }

        # Verificar que los brokers (pods) estén Running
        $brokerPods = kubectl get pods `
            --namespace="$Namespace" `
            --selector="strimzi.io/name=reactorguard-cluster-kafka" `
            --output=jsonpath='{.items[*].status.phase}' 2>$null

        $phases = $brokerPods -split ' ' | Where-Object { $_ -ne '' }
        $allRunning = ($phases.Count -eq 3) -and ($phases | ForEach-Object { $_ -eq "Running" } | Where-Object { -not $_ }).Count -eq 0

        return $allRunning
    }
    catch {
        return $false
    }
}

function Write-Step {
    <#
    .SYNOPSIS
        Imprime un paso con timestamp formateado.
    #>
    param([Parameter(Mandatory)][string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor Cyan
}

# =============================================================================
# CAPA 3 — FUNCIONES DE SERVICIO (efectos secundarios aislados)
# =============================================================================

function Install-StrimziOperator {
    <#
    .SYNOPSIS
        Aplica el módulo Terraform de kafka y espera a que el operador esté Ready.
    .DESCRIPTION
        Usa `terraform apply -target` para aplicar sólo el módulo kafka sin
        afectar otros módulos. Luego espera al pod del operador.
    #>
    [CmdletBinding()]
    param()

    Write-Step "Instalando Strimzi Operator v$StrimziVersion en namespace '$Namespace'..."

    $tfDir = Join-Path $PSScriptRoot ".." "terraform" "environments" "dev"
    $tfDir = [System.IO.Path]::GetFullPath($tfDir)

    if (-not (Test-Path $tfDir)) {
        throw "Directorio Terraform no encontrado: $tfDir"
    }

    Push-Location $tfDir
    try {
        Write-Verbose "Ejecutando terraform init..."
        terraform init -reconfigure 2>&1 | Write-Verbose

        Write-Verbose "Ejecutando terraform apply -target=module.kafka..."
        terraform apply `
            -target=module.kafka `
            -auto-approve `
            -var="kafka_namespace=$Namespace" 2>&1 | Write-Verbose

        if ($LASTEXITCODE -ne 0) {
            throw "terraform apply falló para el módulo kafka."
        }
    }
    finally {
        Pop-Location
    }

    Write-Step "Esperando a que el pod del operador Strimzi esté Ready..."
    Wait-KubernetesPod `
        -LabelSelector "name=strimzi-cluster-operator" `
        -Namespace $Namespace `
        -TimeoutSec $TimeoutSeconds

    Write-Step "Operador Strimzi instalado correctamente."
}

function Deploy-KafkaCluster {
    <#
    .SYNOPSIS
        Aplica kafka-cluster.yaml y espera a que los brokers estén listos.
    .DESCRIPTION
        El recurso `kind: Kafka` es un CRD de Strimzi. Al aplicarlo, el operador
        crea los StatefulSets de Kafka y ZooKeeper. La espera puede tardar 3-5
        minutos porque K8s debe crear los PersistentVolumeClaims y descargar
        la imagen Docker de Kafka.
    #>
    [CmdletBinding()]
    param()

    $metricsPath = Join-Path $PSScriptRoot ".." ".." "k8s" "base" "kafka" "kafka-metrics.yaml"
    $metricsPath = [System.IO.Path]::GetFullPath($metricsPath)

    Write-Step "Aplicando el ConfigMap kafka-metrics (debe existir antes que el cluster)..."
    kubectl apply --filename="$metricsPath" --namespace="$Namespace" 2>&1 | Write-Verbose

    if ($LASTEXITCODE -ne 0) {
        throw "kubectl apply de kafka-metrics.yaml falló."
    }

    $manifestPath = Join-Path $PSScriptRoot ".." ".." "k8s" "base" "kafka" "kafka-cluster.yaml"
    $manifestPath = [System.IO.Path]::GetFullPath($manifestPath)

    Write-Step "Desplegando KafkaCluster reactorguard-cluster..."
    kubectl apply --filename="$manifestPath" --namespace="$Namespace" 2>&1 | Write-Verbose

    if ($LASTEXITCODE -ne 0) {
        throw "kubectl apply de kafka-cluster.yaml falló."
    }

    Write-Step "Esperando a que ZooKeeper esté listo (puede tardar 2-3 minutos)..."
    Wait-KubernetesPod `
        -LabelSelector "strimzi.io/name=reactorguard-cluster-zookeeper" `
        -Namespace $Namespace `
        -TimeoutSec $TimeoutSeconds

    Write-Step "Esperando a que los brokers Kafka estén listos..."
    Wait-KubernetesPod `
        -LabelSelector "strimzi.io/name=reactorguard-cluster-kafka" `
        -Namespace $Namespace `
        -TimeoutSec $TimeoutSeconds

    Write-Step "Esperando a que el Entity Operator esté listo..."
    Wait-KubernetesPod `
        -LabelSelector "strimzi.io/name=reactorguard-cluster-entity-operator" `
        -Namespace $Namespace `
        -TimeoutSec 120

    # Verificar estado del cluster
    $ready = Test-KafkaClusterReady -Namespace $Namespace
    if (-not $ready) {
        throw "El KafkaCluster no alcanzó el estado READY. Revisar: kubectl describe kafka reactorguard-cluster -n $Namespace"
    }

    Write-Step "KafkaCluster reactorguard-cluster READY con 3 brokers."
}

function Deploy-KafkaTopics {
    <#
    .SYNOPSIS
        Aplica kafka-topics.yaml y verifica que los 4 topics existen.
    .DESCRIPTION
        El Entity Operator (parte de Strimzi) observa los recursos KafkaTopic y
        los crea dentro del cluster Kafka. La verificación comprueba que los 4
        topics esperados existen en el namespace.
    #>
    [CmdletBinding()]
    param()

    $manifestPath = Join-Path $PSScriptRoot ".." ".." "k8s" "base" "kafka" "kafka-topics.yaml"
    $manifestPath = [System.IO.Path]::GetFullPath($manifestPath)

    Write-Step "Desplegando KafkaTopics (sensor-readings-raw, sensor-validated, anomaly-alerts, bench-throughput)..."
    kubectl apply --filename="$manifestPath" --namespace="$Namespace" 2>&1 | Write-Verbose

    if ($LASTEXITCODE -ne 0) {
        throw "kubectl apply de kafka-topics.yaml falló."
    }

    # Esperar a que los topics sean reconciliados por el Entity Operator
    Write-Step "Esperando reconciliación de topics (10s)..."
    Start-Sleep -Seconds 10

    # Verificar que los 4 topics existen
    $expectedTopics = @("sensor-readings-raw", "sensor-validated", "anomaly-alerts", "bench-throughput")
    $existingTopics = kubectl get kafkatopic `
        --namespace="$Namespace" `
        --output=jsonpath='{.items[*].metadata.name}' 2>$null

    $topicList = $existingTopics -split ' ' | Where-Object { $_ -ne '' }

    foreach ($topic in $expectedTopics) {
        if ($topicList -notcontains $topic) {
            throw "Topic '$topic' no encontrado tras el apply. Topics actuales: $($topicList -join ', ')"
        }
        Write-Host "  ✅ Topic: $topic" -ForegroundColor Green
    }

    Write-Step "Los 4 topics KafkaTopic están creados y reconciliados."
}

function Deploy-KafkaUsers {
    <#
    .SYNOPSIS
        Aplica kafka-users.yaml y espera a que los 4 KafkaUser estén Ready.
    .DESCRIPTION
        Con authorization simple el acceso es DENY por defecto: sin estos usuarios
        ningún cliente puede producir ni consumir. El User Operator emite el
        certificado de cada usuario y lo publica en un Secret homónimo de este
        namespace; ese Secret es lo que copia Sync-KafkaCredentials.ps1.
    #>
    [CmdletBinding()]
    param()

    $manifestPath = Join-Path $PSScriptRoot ".." ".." "k8s" "base" "kafka" "kafka-users.yaml"
    $manifestPath = [System.IO.Path]::GetFullPath($manifestPath)

    Write-Step "Desplegando KafkaUsers (ingestion, validator, detector, benchmark)..."
    kubectl apply --filename="$manifestPath" --namespace="$Namespace" 2>&1 | Write-Verbose

    if ($LASTEXITCODE -ne 0) {
        throw "kubectl apply de kafka-users.yaml falló."
    }

    $expectedUsers = @("reactorguard-ingestion", "reactorguard-validator", "reactorguard-detector", "reactorguard-benchmark")
    foreach ($user in $expectedUsers) {
        Write-Verbose "Esperando a que KafkaUser/$user esté Ready..."
        $result = kubectl wait kafkauser/$user `
            --namespace="$Namespace" `
            --for=condition=Ready `
            --timeout="120s" 2>&1

        if ($LASTEXITCODE -ne 0) {
            throw "KafkaUser '$user' no alcanzó Ready: $result"
        }
        Write-Host "  OK KafkaUser: $user" -ForegroundColor Green
    }

    Write-Step "Los 4 KafkaUser están Ready. Siguiente paso: .\infra\scripts\Sync-KafkaCredentials.ps1"
}

# =============================================================================
# CAPA 4 — ORQUESTACIÓN
# =============================================================================

function Invoke-KafkaInstall {
    <#
    .SYNOPSIS
        Punto de entrada principal. Orquesta la instalación completa de Kafka.
    #>
    [CmdletBinding()]
    param()

    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
    Write-Host "║     ReactorGuard — Instalación de Kafka con Strimzi     ║" -ForegroundColor Magenta
    Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Magenta
    Write-Host "  Namespace : $Namespace"
    Write-Host "  Strimzi   : v$StrimziVersion"
    Write-Host "  Timeout   : ${TimeoutSeconds}s"
    Write-Host ""

    try {
        Install-StrimziOperator
        Deploy-KafkaCluster
        Deploy-KafkaTopics
        Deploy-KafkaUsers

        Write-Host ""
        Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
        Write-Host "║              Instalación completada con éxito           ║" -ForegroundColor Green
        Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Green
        Write-Host ""
        Write-Host "Verificación manual:" -ForegroundColor Yellow
        Write-Host "  kubectl get pods -n $Namespace        # strimzi-operator + 3 brokers + 3 zookeepers" -ForegroundColor Yellow
        Write-Host "  kubectl get kafka -n $Namespace       # reactorguard-cluster, READY=True" -ForegroundColor Yellow
        Write-Host "  kubectl get kafkatopic -n $Namespace  # 4 topics" -ForegroundColor Yellow
        Write-Host "  kubectl get kafkauser -n $Namespace   # 4 usuarios, READY=True" -ForegroundColor Yellow
    }
    catch {
        Write-Host ""
        Write-Host "❌ ERROR durante la instalación de Kafka:" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host ""
        Write-Host "Diagnóstico:" -ForegroundColor Yellow
        Write-Host "  kubectl get events -n $Namespace --sort-by=.lastTimestamp" -ForegroundColor Yellow
        Write-Host "  kubectl describe kafka reactorguard-cluster -n $Namespace" -ForegroundColor Yellow
        exit 1
    }
}

# Punto de entrada
Invoke-KafkaInstall
