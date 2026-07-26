#Requires -Version 7.0
<#
.SYNOPSIS
    Verifica que Workload Identity está configurado correctamente en ReactorGuard.

.DESCRIPTION
    Crea un pod efímero con la KSA anotada, obtiene un token GCP desde la GSA
    impersonada y consulta la Secret Manager API. Un HTTP 200 confirma que el
    flujo completo funciona: pod → KSA → GSA → GCP API.

    Flujo Workload Identity:
      1. Pod se inicia usando la KSA especificada
      2. GKE detecta la anotación iam.gke.io/gcp-service-account
      3. GKE Metadata Server emite un token de la GSA mapeada
      4. El pod usa ese token para llamar a GCP APIs sin credenciales estáticas
      5. Si el binding IAM (T1.6) está correcto → HTTP 200
         Si no está configurado             → HTTP 403

.EXAMPLE
    .\Test-WorkloadIdentity.ps1
    .\Test-WorkloadIdentity.ps1 -Namespace reactorguard-ingestion -KsaName sensor-validator

.NOTES
    Prerequisitos: T1.6 (IAM bindings) y T2.1 (namespaces) completadas.
    El cluster debe estar accesible: kubectl get nodes debe funcionar.
#>

# ---------------------------------------------------------------------------
# CAPA 1 — Configuración
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$ProjectId  = "sentinel-platform-485714",
    [string]$Namespace  = "reactorguard-ml",
    [string]$KsaName    = "pinn-server",
    [string]$SecretName = "reactorguard/jwt-secret"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# CAPA 2 — Utilidades
# ---------------------------------------------------------------------------

function New-TestPodManifest {
    <#
    .SYNOPSIS Genera el manifiesto YAML para un pod de prueba efímero.
    El pod usa la KSA anotada con Workload Identity y contiene el SDK de GCP
    para poder obtener un token de acceso desde el Metadata Server de GKE.
    #>
    param(
        [Parameter(Mandatory)][string]$PodName,
        [Parameter(Mandatory)][string]$Namespace,
        [Parameter(Mandatory)][string]$Ksa
    )
    return @"
apiVersion: v1
kind: Pod
metadata:
  name: $PodName
  namespace: $Namespace
  labels:
    app: workload-identity-test
    managed-by: Test-WorkloadIdentity
spec:
  serviceAccountName: $Ksa
  containers:
    - name: wi-test
      image: google/cloud-sdk:slim
      command: ["sleep", "120"]
      resources:
        requests:
          cpu: "100m"
          memory: "128Mi"
        limits:
          cpu: "200m"
          memory: "256Mi"
  restartPolicy: Never
  terminationGracePeriodSeconds: 5
"@
}

function Wait-PodReady {
    <#
    .SYNOPSIS Espera hasta que el pod esté en estado Running o hasta que expire el timeout.
    Hace polling cada 3 segundos para no sobrecargar la API de Kubernetes.
    #>
    param(
        [Parameter(Mandatory)][string]$PodName,
        [Parameter(Mandatory)][string]$Namespace,
        [int]$TimeoutSec = 60
    )
    Write-Host "  Esperando pod '$PodName' en '$Namespace' (timeout: ${TimeoutSec}s)..." -ForegroundColor Cyan

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $phase = kubectl get pod $PodName -n $Namespace -o jsonpath='{.status.phase}' 2>$null
        if ($phase -eq "Running") {
            Write-Host "  Pod '$PodName' esta Running." -ForegroundColor Green
            return
        }
        if ($phase -eq "Failed" -or $phase -eq "Unknown") {
            throw "Pod '$PodName' entro en estado '$phase'. Revisar: kubectl describe pod $PodName -n $Namespace"
        }
        Write-Host "    Estado actual: $phase — reintentando en 3s..." -ForegroundColor Gray
        Start-Sleep -Seconds 3
    }
    throw "Timeout (${TimeoutSec}s) esperando pod '$PodName' en estado Running."
}

# ---------------------------------------------------------------------------
# CAPA 3 — Servicio
# ---------------------------------------------------------------------------

function Invoke-SecretManagerTest {
    <#
    .SYNOPSIS Ejecuta dentro del pod: obtiene token GCP y consulta Secret Manager API.

    El flujo dentro del pod:
      gcloud auth print-access-token
        → llama al GKE Metadata Server (169.254.169.254)
        → obtiene token firmado por la GSA impersonada
        → devuelve el Bearer token

    Luego hace GET a la Secret Manager REST API con ese token.
    HTTP 200 = acceso autorizado (bindings IAM T1.6 correctos)
    HTTP 403 = acceso denegado (revisar workload_identity.tf en T1.6)
    #>
    param(
        [Parameter(Mandatory)][string]$PodName,
        [Parameter(Mandatory)][string]$ProjectId,
        [Parameter(Mandatory)][string]$SecretName
    )

    Write-Host "  Obteniendo token de acceso GCP desde el pod via Workload Identity..." -ForegroundColor Cyan

    # El Metadata Server de GKE devuelve el token de la GSA asociada a la KSA
    $token = kubectl exec $PodName -n $Namespace -- sh -c "gcloud auth print-access-token 2>/dev/null"

    if (-not $token -or $token.Trim() -eq "") {
        throw "No se obtuvo token GCP desde el pod. Verificar: (1) KSA anotada correctamente, (2) Workload Identity habilitado en el node pool, (3) binding IAM en T1.6."
    }
    Write-Host "  Token GCP obtenido ($($token.Trim().Length) caracteres)." -ForegroundColor Green

    # Secret Manager REST API — verifica que la GSA tiene secretmanager.secretAccessor
    $encodedSecret = $SecretName.Replace("/", "%2F")
    $secretUrl = "https://secretmanager.googleapis.com/v1/projects/$ProjectId/secrets/$encodedSecret"
    Write-Host "  Consultando Secret Manager: $secretUrl" -ForegroundColor Cyan

    $response = Invoke-WebRequest `
        -Uri         $secretUrl `
        -Method      GET `
        -Headers     @{ Authorization = "Bearer $($token.Trim())" } `
        -UseBasicParsing `
        -ErrorAction Stop

    if ($response.StatusCode -eq 200) {
        Write-Host "  Secret Manager respondio HTTP 200 — acceso autorizado." -ForegroundColor Green
    }
    else {
        throw "Secret Manager respondio HTTP $($response.StatusCode) (esperado 200). Verificar roles IAM de la GSA."
    }
}

function Remove-TestPod {
    <#
    .SYNOPSIS Elimina el pod de prueba. Llamado desde el bloque finally para garantizar limpieza
    incluso si el test falla o es interrumpido con Ctrl+C.
    #>
    param(
        [Parameter(Mandatory)][string]$PodName,
        [Parameter(Mandatory)][string]$Namespace
    )
    Write-Host "  Eliminando pod de prueba '$PodName'..." -ForegroundColor Gray
    kubectl delete pod $PodName -n $Namespace --ignore-not-found --grace-period=0 2>$null | Out-Null
    Write-Host "  Pod eliminado." -ForegroundColor Gray
}

# ---------------------------------------------------------------------------
# CAPA 4 — Orquestacion
# ---------------------------------------------------------------------------

function Invoke-WorkloadIdentityTest {
    <#
    .SYNOPSIS Orquesta el test completo de Workload Identity para ReactorGuard.

    Pasos:
      1. Genera un nombre de pod unico para evitar colisiones
      2. Crea el pod con la KSA especificada (iam.gke.io/gcp-service-account anotada)
      3. Espera a que el pod este Running (hasta 60s)
      4. Dentro del pod: obtiene token GCP de la GSA via Metadata Server
      5. Con el token: llama a Secret Manager API y verifica HTTP 200
      6. Limpia el pod (garantizado en finally aunque falle el test)

    Si recibes HTTP 403:
      - Verificar que workload_identity.tf (T1.6) esta aplicado en GCP
      - Verificar que la KSA tiene la anotacion correcta: kubectl describe sa $KsaName -n $Namespace
      - Verificar el binding: gcloud iam service-accounts get-iam-policy GSA_EMAIL
    #>

    $podName = "wi-test-$(Get-Random -Maximum 99999)"

    Write-Host ""
    Write-Host "========================================================" -ForegroundColor Yellow
    Write-Host "  ReactorGuard — Workload Identity Verification Test" -ForegroundColor Yellow
    Write-Host "========================================================" -ForegroundColor Yellow
    Write-Host "  Proyecto  : $ProjectId"
    Write-Host "  Namespace : $Namespace"
    Write-Host "  KSA       : $KsaName"
    Write-Host "  Secreto   : $SecretName"
    Write-Host ""

    # Verificacion previa: la KSA existe
    $ksaAnnotation = kubectl get serviceaccount $KsaName -n $Namespace `
        -o jsonpath='{.metadata.annotations.iam\.gke\.io/gcp-service-account}' 2>$null
    if (-not $ksaAnnotation) {
        throw "KSA '$KsaName' no encontrada en namespace '$Namespace', o no tiene anotacion Workload Identity. Ejecutar: kubectl apply -k k8s/base/"
    }
    Write-Host "  KSA anotada con GSA: $ksaAnnotation" -ForegroundColor Cyan

    $manifest = New-TestPodManifest -PodName $podName -Namespace $Namespace -Ksa $KsaName

    try {
        # Paso 1: Crear pod efimero
        Write-Host "[1/4] Creando pod de prueba '$podName'..." -ForegroundColor White
        $manifest | kubectl apply -f - | Out-Null
        Write-Host "  Pod creado." -ForegroundColor Green

        # Paso 2: Esperar Running
        Write-Host "[2/4] Esperando que el pod este Running..." -ForegroundColor White
        Wait-PodReady -PodName $podName -Namespace $Namespace -TimeoutSec 60

        # Paso 3: Verificar acceso GCP via Workload Identity
        Write-Host "[3/4] Verificando acceso a Secret Manager via Workload Identity..." -ForegroundColor White
        Invoke-SecretManagerTest -PodName $podName -ProjectId $ProjectId -SecretName $SecretName

        # Paso 4: Resultado
        Write-Host "[4/4] Test completado exitosamente." -ForegroundColor Green
        Write-Host ""
        Write-Host "========================================================" -ForegroundColor Green
        Write-Host "  RESULTADO: Workload Identity VERIFICADO" -ForegroundColor Green
        Write-Host "  $KsaName --> $ksaAnnotation" -ForegroundColor Green
        Write-Host "========================================================" -ForegroundColor Green

    }
    finally {
        # Limpieza garantizada: el pod se elimina siempre, incluso si hay error
        Write-Host ""
        Write-Host "[limpieza] Eliminando pod temporal..." -ForegroundColor Gray
        Remove-TestPod -PodName $podName -Namespace $Namespace
    }
}

# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
Invoke-WorkloadIdentityTest
