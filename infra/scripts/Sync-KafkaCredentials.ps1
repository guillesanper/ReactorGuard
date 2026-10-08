#Requires -Version 7.0
<#
.SYNOPSIS
    Copia las credenciales mTLS de Kafka de kafka-operator a reactorguard-ingestion.

.DESCRIPTION
    Los KafkaUser de Strimzi viven en el namespace kafka-operator y el User Operator
    publica el certificado de cada uno en un Secret homonimo de ESE namespace. Un pod
    no puede montar un Secret de otro namespace, asi que el streamer y el validador (en
    reactorguard-ingestion) no pueden usarlos tal cual. Este script copia:

      - El Secret de cada KafkaUser (por defecto reactorguard-ingestion,
        reactorguard-validator y reactorguard-benchmark), SOLO las claves user.crt y user.key. No se copian
        user.p12 ni user.password: ningun cliente de ReactorGuard las usa.
      - El CA del cluster (<cluster>-cluster-ca-cert), SOLO la clave ca.crt, que es lo
        que verifica los certificados de los brokers.

    Los Secrets destino conservan el nombre del origen: los Deployments los montan con
    ese nombre (tep-streamer-deployment.yaml y sensor-validator-deployment.yaml), y un
    test comprueba que los nombres coinciden.

    Seguridad: los valores viajan por stdin de kubectl y nunca se escriben en disco ni en
    la salida. El script solo imprime nombres y la fecha de caducidad de cada
    certificado.

    ROTACION (paso manual). Strimzi renueva los certificados en kafka-operator por su
    cuenta (por defecto 30 dias antes de que caduquen) y el CA del cluster tambien. Las
    COPIAS de este namespace NO se actualizan solas. Con un certificado caducado el
    handshake TLS falla y los pods entran en CrashLoopBackOff. Procedimiento:
      1. Antes de la fecha NotAfter que imprime este script, vuelve a ejecutarlo.
      2. Reinicia las cargas, porque los clientes Python leen los ficheros al arrancar y
         no los recargan: kubectl rollout restart deployment/tep-streamer
         deployment/sensor-validator -n reactorguard-ingestion
    Si el CA del cluster rota, repetir ambos pasos ANTES de que caduque el CA antiguo.
    Automatizarlo (por ejemplo con un CronJob o external-secrets) es trabajo futuro.

    Requisitos: kubectl con contexto en el cluster, el operador Strimzi y los KafkaUser
      en Ready (Install-Kafka.ps1) y el namespace destino creado (kubectl apply -k
      k8s/base/, o al menos namespaces.yaml).

.PARAMETER SourceNamespace
    Namespace de los KafkaUser y del CA (el de Strimzi).

.PARAMETER TargetNamespace
    Namespace donde corren el streamer y el validador.

.PARAMETER ClusterName
    Nombre del recurso Kafka; el CA se llama <ClusterName>-cluster-ca-cert.

.PARAMETER KafkaUsers
    KafkaUser cuyo Secret se copia. El de benchmark lo usa el pod efimero de
    tests/integration/Invoke-KafkaTests.ps1. El detector no se incluye: no tiene carga
    en este namespace todavia.

.PARAMETER TimeoutSeconds
    Espera maxima a que cada Secret de origen exista.

.EXAMPLE
    .\infra\scripts\Sync-KafkaCredentials.ps1
    .\infra\scripts\Sync-KafkaCredentials.ps1 -WhatIf
#>

# =============================================================================
# CAPA 1 - CONFIGURACION
# =============================================================================
[CmdletBinding(SupportsShouldProcess)]
param(
    [string]$SourceNamespace = "kafka-operator",
    [string]$TargetNamespace = "reactorguard-ingestion",
    [string]$ClusterName     = "reactorguard-cluster",
    [string[]]$KafkaUsers    = @("reactorguard-ingestion", "reactorguard-validator", "reactorguard-benchmark"),
    [int]$TimeoutSeconds     = 120
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Claves que se copian de cada tipo de Secret (minimo privilegio).
$UserSecretKeys = @("user.crt", "user.key")
$CaSecretKeys   = @("ca.crt")
$ManagedByLabel = "sync-kafkacredentials"

# =============================================================================
# CAPA 2 - UTILIDADES PURAS (sin kubectl, sin efectos secundarios)
# =============================================================================

function Get-ClusterCaSecretName {
    <#
    .SYNOPSIS
        Devuelve el nombre del Secret con el CA del cluster que crea Strimzi.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Cluster)

    return "$Cluster-cluster-ca-cert"
}

function Test-PemBase64 {
    <#
    .SYNOPSIS
        Indica si un valor base64 decodifica a texto PEM (empieza por -----BEGIN).
    .DESCRIPTION
        Detecta que se ha copiado la clave equivocada (por ejemplo un .p12 binario)
        antes de publicar un Secret que haria fallar el handshake mas tarde.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Base64)

    try {
        $bytes = [System.Convert]::FromBase64String($Base64)
    }
    catch {
        return $false
    }
    $text = [System.Text.Encoding]::ASCII.GetString($bytes)
    return $text.TrimStart().StartsWith("-----BEGIN ", [System.StringComparison]::Ordinal)
}

function Select-SecretData {
    <#
    .SYNOPSIS
        Extrae de un Secret de Kubernetes solo las claves pedidas, validando que son PEM.
    .OUTPUTS
        [ordered] clave -> valor base64, en el orden pedido.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][psobject]$Secret,
        [Parameter(Mandatory)][string[]]$Keys
    )

    $name = $Secret.metadata.name
    $data = $Secret.PSObject.Properties["data"]
    if ($null -eq $data -or $null -eq $data.Value) {
        throw "El Secret '$name' no tiene datos."
    }

    $selected = [ordered]@{}
    foreach ($key in $Keys) {
        $property = $data.Value.PSObject.Properties[$key]
        if ($null -eq $property -or [string]::IsNullOrWhiteSpace([string]$property.Value)) {
            $available = ($data.Value.PSObject.Properties.Name) -join ", "
            throw "El Secret '$name' no tiene la clave '$key'. Claves presentes: $available"
        }
        if (-not (Test-PemBase64 -Base64 ([string]$property.Value))) {
            throw "La clave '$key' del Secret '$name' no es PEM."
        }
        $selected[$key] = [string]$property.Value
    }
    return $selected
}

function New-SecretManifestJson {
    <#
    .SYNOPSIS
        Construye el manifiesto JSON de un Secret Opaque a partir de datos ya en base64.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Namespace,
        [Parameter(Mandatory)][System.Collections.IDictionary]$Data,
        [Parameter(Mandatory)][string]$SourceNamespace
    )

    $manifest = [ordered]@{
        apiVersion = "v1"
        kind       = "Secret"
        type       = "Opaque"
        metadata   = [ordered]@{
            name        = $Name
            namespace   = $Namespace
            labels      = [ordered]@{
                "app.kubernetes.io/part-of"    = "reactorguard"
                "app.kubernetes.io/managed-by" = $ManagedByLabel
            }
            annotations = [ordered]@{
                "reactorguard/copied-from" = "$SourceNamespace/$Name"
            }
        }
        data       = $Data
    }
    return ($manifest | ConvertTo-Json -Depth 6 -Compress)
}

function Get-CertificateNotAfter {
    <#
    .SYNOPSIS
        Devuelve la fecha de caducidad (UTC) del primer certificado de un valor PEM en base64.
    .OUTPUTS
        [datetime] o $null si no se puede leer.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Base64)

    try {
        $pem = [System.Text.Encoding]::ASCII.GetString([System.Convert]::FromBase64String($Base64))
        $certificate = [System.Security.Cryptography.X509Certificates.X509Certificate2]::CreateFromPem($pem)
        return $certificate.NotAfter.ToUniversalTime()
    }
    catch {
        return $null
    }
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

function Assert-Namespace {
    <#
    .SYNOPSIS
        Falla con un mensaje util si un namespace no existe.
    #>
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Namespace)

    kubectl get namespace $Namespace --output=name 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "El namespace '$Namespace' no existe. Aplica k8s/base/namespaces.yaml primero."
    }
}

function Get-SourceSecret {
    <#
    .SYNOPSIS
        Lee un Secret del namespace de origen, esperando a que exista.
    .DESCRIPTION
        El User Operator tarda unos segundos en publicar el Secret tras crear el
        KafkaUser, asi que se sondea hasta TimeoutSeconds.
    .OUTPUTS
        [psobject] el Secret parseado.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Namespace,
        [Parameter(Mandatory)][int]$TimeoutSec
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ($true) {
        $json = kubectl get secret $Name --namespace=$Namespace --output=json 2>$null
        if ($LASTEXITCODE -eq 0 -and $json) {
            return ($json -join "`n" | ConvertFrom-Json)
        }
        if ((Get-Date) -ge $deadline) {
            throw "El Secret '$Name' no existe en '$Namespace' tras ${TimeoutSec}s. " +
                "Comprueba: kubectl get kafkauser -n $Namespace"
        }
        Start-Sleep -Seconds 3
    }
}

function Publish-Secret {
    <#
    .SYNOPSIS
        Aplica un Secret en el namespace destino pasando el manifiesto por stdin.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Namespace,
        [Parameter(Mandatory)][string]$ManifestJson
    )

    if ($PSCmdlet.ShouldProcess("$Namespace/$Name", "Crear o actualizar el Secret")) {
        $ManifestJson | kubectl apply --filename=- 2>&1 | Write-Verbose
        if ($LASTEXITCODE -ne 0) {
            throw "kubectl apply del Secret '$Namespace/$Name' fallo."
        }
    }
}

function Copy-KafkaSecret {
    <#
    .SYNOPSIS
        Copia un Secret entre namespaces, quedandose solo con las claves indicadas.
    .OUTPUTS
        [ordered] clave -> valor base64 de lo copiado (para informar de la caducidad).
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string[]]$Keys
    )

    $secret = Get-SourceSecret -Name $Name -Namespace $SourceNamespace -TimeoutSec $TimeoutSeconds
    $data = Select-SecretData -Secret $secret -Keys $Keys
    $json = New-SecretManifestJson `
        -Name $Name -Namespace $TargetNamespace -Data $data -SourceNamespace $SourceNamespace
    Publish-Secret -Name $Name -Namespace $TargetNamespace -ManifestJson $json
    return $data
}

# =============================================================================
# CAPA 4 - ORQUESTACION
# =============================================================================

function Invoke-CredentialSync {
    <#
    .SYNOPSIS
        Copia el CA del cluster y el Secret de cada KafkaUser al namespace destino.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param()

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host " ReactorGuard - Sincronizacion de credenciales Kafka (mTLS)" -ForegroundColor Yellow
    Write-Host " Origen : $SourceNamespace" -ForegroundColor Yellow
    Write-Host " Destino: $TargetNamespace" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""

    if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
        throw "kubectl no esta en el PATH."
    }
    Assert-Namespace -Namespace $SourceNamespace
    Assert-Namespace -Namespace $TargetNamespace

    $report = [System.Collections.Generic.List[object]]::new()

    $caName = Get-ClusterCaSecretName -Cluster $ClusterName
    Write-Step "Copiando el CA del cluster: $caName"
    $caData = Copy-KafkaSecret -Name $caName -Keys $CaSecretKeys
    $report.Add([pscustomobject]@{
            Secret   = $caName
            Keys     = ($CaSecretKeys -join ", ")
            NotAfter = Get-CertificateNotAfter -Base64 $caData["ca.crt"]
        })

    foreach ($user in $KafkaUsers) {
        Write-Step "Copiando las credenciales del KafkaUser: $user"
        $userData = Copy-KafkaSecret -Name $user -Keys $UserSecretKeys
        $report.Add([pscustomobject]@{
                Secret   = $user
                Keys     = ($UserSecretKeys -join ", ")
                NotAfter = Get-CertificateNotAfter -Base64 $userData["user.crt"]
            })
    }

    Write-Host ""
    Write-Host "Secrets sincronizados en '$TargetNamespace' (los valores no se muestran):" -ForegroundColor Green
    $report | ForEach-Object {
        $expiry = if ($null -ne $_.NotAfter) { $_.NotAfter.ToString("yyyy-MM-dd") } else { "no determinada" }
        Write-Host ("  {0,-40} claves: {1,-18} caduca (UTC): {2}" -f $_.Secret, $_.Keys, $expiry)
    }
    Write-Host ""
    Write-Host "Rotacion (manual): repite este script antes de la fecha de caducidad y reinicia:" -ForegroundColor Yellow
    Write-Host "  kubectl rollout restart deployment/tep-streamer deployment/sensor-validator -n $TargetNamespace" -ForegroundColor Yellow
    Write-Host ""
}

# Punto de entrada. No se ejecuta si el fichero se carga con dot-sourcing (pruebas de las
# funciones puras de la capa 2).
if ($MyInvocation.InvocationName -ne ".") {
    try {
        Invoke-CredentialSync
    }
    catch {
        Write-Host ""
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Diagnostico:" -ForegroundColor Yellow
        Write-Host "  kubectl get kafkauser -n $SourceNamespace" -ForegroundColor Yellow
        Write-Host "  kubectl get secret -n $SourceNamespace" -ForegroundColor Yellow
        exit 1
    }
}
