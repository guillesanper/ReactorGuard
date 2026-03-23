# ReactorGuard — Fase 1: Plan Detallado con Prompts
**Foundation: GKE + Kafka · Semanas 1–2**

---

## Cómo usar este documento

Cada subtarea tiene:
- **Qué hace**: descripción del resultado esperado
- **Prerequisitos**: qué debe estar listo antes
- **Prompt**: listo para pegar directamente en Claude
- **Verificación**: cómo comprobar que está bien hecho

El orden de las tareas es el orden en que deben ejecutarse. No saltes tareas — cada una es prerequisito de la siguiente.

**Plataforma de desarrollo**: Windows 11. Todos los scripts son **PowerShell 7+** (`.ps1`).
Ejecutar scripts: `.\infra\scripts\nombre.ps1` desde una terminal PowerShell 7.
Instalar PowerShell 7 si no está disponible: `winget install Microsoft.PowerShell`.

**Patrón de arquitectura de scripts**: Todos los scripts siguen una estructura de 4 capas:
1. **Capa de configuración**: `#Requires`, `[CmdletBinding()]`, `param()`, `Set-StrictMode`
2. **Capa de utilidades**: funciones helper reutilizables, sin efectos secundarios
3. **Capa de servicio**: funciones que llaman a herramientas externas (gcloud, kubectl, terraform)
4. **Capa de orquestación**: función principal que coordina el flujo

---

## SEMANA 1 — Infraestructura GCP con Terraform

---

### T1.1 — Estructura del repositorio y configuración inicial

**Qué hace**: Crea la estructura de carpetas del repositorio, `pyproject.toml`, `.gitignore` y los archivos raíz de configuración. Es la base sobre la que se construye todo.

**Prerequisitos**: Repositorio Git vacío creado en GitHub.

**Prompt**:
```
Crea la estructura completa de directorios y archivos de configuración base para el proyecto ReactorGuard.

El proyecto es una plataforma cloud-native de detección de anomalías en reactores nucleares.
Stack: Python 3.11, GCP, Terraform, Kubernetes, Kafka, PyTorch, FastAPI.
Plataforma de desarrollo: Windows 11. Todos los scripts deben ser PowerShell 7+ (.ps1), nunca .sh.

Necesito que generes:

1. El árbol completo de directorios del repositorio exactamente como aparece en la sección 12 del TDD:
   - .github/workflows/ (vacío por ahora, solo el directorio)
   - infra/terraform/modules/ (vpc, gke, kafka, storage, iam, security)
   - infra/terraform/environments/ (dev, staging, prod)
   - k8s/base/ (ingestion, ml, observability, rbac)
   - k8s/overlays/ (dev, staging, prod)
   - api/routers/
   - ml/features/, ml/models/, ml/training/, ml/validation/, ml/serving/
   - data/schemas/, data/generators/, data/validation/
   - observability/prometheus/, observability/grafana/dashboards/, observability/alertmanager/
   - tests/unit/, tests/integration/, tests/safety/
   - docs/runbooks/

2. pyproject.toml con:
   - Python 3.11
   - Dependencias: torch, fastapi, uvicorn, kafka-python, feast, mlflow, dvc[gcs], mapie, shap, pydantic, opentelemetry-sdk, prometheus-client
   - Dev dependencies: pytest, pytest-asyncio, ruff, mypy, trivy

3. .gitignore apropiado para Python + Terraform + Kubernetes + PowerShell (incluir *.ps1~ y PSReadline history)

4. README.md con secciones: Overview, Prerequisites (incluir PowerShell 7+), Quick Start (los comandos de la sección 13 del TDD adaptados a Windows/PowerShell), Architecture

5. params.yaml vacío con estructura comentada para: simulation.yaml params (n_samples, fault_injection_rate, openmc_seed) y training params (learning_rate, physics_lambda, hidden_size)

Genera los archivos con contenido real, no placeholders vacíos. Los archivos de código deben tener al menos los imports y el skeleton de las clases/funciones principales.
```

**Verificación**:
```powershell
(Get-ChildItem -Recurse -File).Count   # Debe haber al menos 25 archivos
python -m pytest tests/                # No debe fallar (0 tests = ok)
```

---

### T1.2 — Módulo Terraform: Backend de estado y proyecto GCP

**Qué hace**: Configura el backend remoto de Terraform en GCS y el bloque `provider` de GCP. Sin esto, el estado de Terraform se guarda en local y no es compartible ni seguro.

**Prerequisitos**: T1.1 completada. Tener `gcloud` autenticado localmente con una cuenta con permisos de Owner en el proyecto GCP.

**Prompt**:
```
Crea los archivos Terraform para configurar el backend remoto y el provider de GCP para ReactorGuard.

Contexto del proyecto:
- GCP Project ID: reactorguard-platform
- Región principal: europe-west1
- El bucket de estado se llama: reactorguard-terraform-state
- Usaremos Terraform >= 1.6.0
- Plataforma de desarrollo: Windows 11, PowerShell 7+. NO generes scripts .sh.

Necesito exactamente estos archivos:

1. infra/terraform/environments/dev/backend.tf
   - Backend GCS con bucket reactorguard-terraform-state
   - Prefix: terraform/state/dev
   - Locking habilitado (GCS lo hace automáticamente)

2. infra/terraform/environments/dev/providers.tf
   - Provider google y google-beta, versión ~> 5.0
   - Región europe-west1
   - Project reactorguard-platform

3. infra/terraform/environments/dev/versions.tf
   - required_version = ">= 1.6.0"
   - required_providers: google ~> 5.0, google-beta ~> 5.0, helm ~> 2.12, kubernetes ~> 2.25

4. infra/terraform/environments/dev/main.tf
   - Skeleton que llamará a los módulos (vpc, gke, storage, iam, security, kafka)
   - Por ahora los module blocks pueden tener source y variables vacías — los rellenaremos en subtareas siguientes

5. Un script PowerShell infra/scripts/bootstrap.ps1 siguiendo arquitectura de 4 capas:

   CAPA 1 — Configuración y parámetros:
   #Requires -Version 7.0
   [CmdletBinding()]
   param(
     [string]$ProjectId   = "reactorguard-platform",
     [string]$Region      = "europe-west1",
     [string]$StateBucket = "reactorguard-terraform-state"
   )
   Set-StrictMode -Version Latest
   $ErrorActionPreference = "Stop"

   CAPA 2 — Funciones de utilidad (sin efectos secundarios, reutilizables):
   - function Invoke-Gcloud([string[]]$Arguments) — wrapper que ejecuta gcloud, captura stdout/stderr y lanza excepción si falla
   - function Test-GcsBucketExists([string]$BucketName) — devuelve [bool], no lanza excepciones
   - function Write-Step([string]$Message) — imprime paso con timestamp formateado

   CAPA 3 — Funciones de servicio (llaman a GCP, efectos secundarios aislados):
   - function Initialize-StateBucket — crea el bucket GCS si no existe, idempotente
   - function Enable-RequiredApis — habilita las APIs: container, compute, storage, secretmanager, pubsub, cloudkms, binaryauthorization

   CAPA 4 — Orquestación (punto de entrada, coordina el flujo):
   - function Invoke-Bootstrap — llama a Initialize-StateBucket y Enable-RequiredApis con try/catch, imprime resumen final
   Invoke-Bootstrap   # Llamada al final del script

Añade comentarios en cada archivo explicando qué hace cada bloque.
```

**Verificación**:
```powershell
.\infra\scripts\bootstrap.ps1
Push-Location infra\terraform\environments\dev
terraform init
terraform validate   # Debe pasar sin errores
Pop-Location
```

---

### T1.3 — Módulo Terraform: VPC y Networking

**Qué hace**: Crea la red privada sobre la que correrán todos los servicios. GKE nodes, pods y servicios tendrán rangos IP separados. Sin VPC privada, los nodos serían accesibles desde internet.

**Prerequisitos**: T1.2 completada y `terraform init` ejecutado.

**Prompt**:
```
Crea el módulo Terraform para la VPC de ReactorGuard en GCP.

Requisitos de red del proyecto:
- VPC name: reactorguard-vpc
- Región: europe-west1
- Subnet principal para GKE nodes: 10.0.1.0/24, nombre: gke-nodes (private)
- Secondary range para GKE pods: 10.0.16.0/20, nombre: gke-pods
- Secondary range para GKE services: 10.0.32.0/20, nombre: gke-services
- Cloud NAT + Router para que los nodos privados puedan hacer pull de imágenes Docker
- Firewall rules: denegar todo por defecto, permitir solo tráfico interno entre subnets del proyecto

Genera estos archivos:

1. infra/terraform/modules/vpc/main.tf con:
   - google_compute_network (custom, auto_create_subnetworks=false)
   - google_compute_subnetwork con secondary_ip_range para pods y services
   - google_compute_router
   - google_compute_router_nat (NAT automático para todas las IPs de la subnet)
   - google_compute_firewall para deny-all ingress por defecto
   - google_compute_firewall para allow-internal (tráfico dentro del VPC)
   - google_compute_firewall para allow-health-checks (GCP Load Balancer health checks: 130.211.0.0/22, 35.191.0.0/16)

2. infra/terraform/modules/vpc/variables.tf con todas las variables necesarias (project_id, region, vpc_name, etc.)

3. infra/terraform/modules/vpc/outputs.tf exponiendo: vpc_name, vpc_self_link, subnet_name, subnet_self_link, pods_range_name, services_range_name

4. Actualiza infra/terraform/environments/dev/main.tf para llamar al módulo vpc con los valores concretos del proyecto

Añade comentarios explicando por qué cada recurso es necesario, especialmente las secondary ranges de GKE.
```

**Verificación**:
```powershell
Push-Location infra\terraform\environments\dev
terraform plan   # Debe mostrar ~8 recursos a crear, 0 errores
# Revisar que no aparezca ningún recurso con IP pública en los outputs
Pop-Location
```

---

### T1.4 — Módulo Terraform: GKE Cluster

**Qué hace**: Crea el cluster de Kubernetes privado con dos node pools diferenciados: uno para cargas generales y uno dedicado exclusivamente a ML serving con nodos no-preemptible.

**Prerequisitos**: T1.3 completada. VPC y subnets deben existir (o el módulo VPC debe estar en el plan).

**Prompt**:
```
Crea el módulo Terraform para el cluster GKE de ReactorGuard.

Especificaciones exactas del TDD:
- Cluster name: reactorguard-cluster
- Tipo: GKE Standard (NO Autopilot — Strimzi requiere control granular)
- Privado: nodos sin IPs públicas, master endpoint accesible solo desde la VPC
- Región: europe-west1 (regional, no zonal — para HA)
- Release channel: REGULAR

Node Pool 1 — platform:
- Máquina: e2-standard-4 (4 vCPU, 16GB RAM)
- Autoscaling: min 1, max 3 nodos
- Preemptible: SÍ (para reducir coste)
- Labels: node-pool=platform
- Taints: ninguno

Node Pool 2 — ml-serving:
- Máquina: n1-standard-8 (8 vCPU, 30GB RAM)
- Autoscaling: min 1, max 5 nodos
- Preemptible: NO (CRÍTICO — safety-critical serving)
- Labels: node-pool=ml-serving
- Taints: ml-workload=true:NoSchedule (para que solo los pods ML vayan a estos nodos)

Configuraciones de seguridad obligatorias:
- Workload Identity habilitado
- Binary Authorization habilitado
- Network Policy (Calico) habilitado
- Managed Prometheus (Google Cloud Managed Service for Prometheus) habilitado
- Shielded nodes habilitados
- Private cluster: master_ipv4_cidr_block = 172.16.0.0/28

Genera:
1. infra/terraform/modules/gke/main.tf con google_container_cluster y dos google_container_node_pool
2. infra/terraform/modules/gke/variables.tf
3. infra/terraform/modules/gke/outputs.tf (cluster_name, cluster_endpoint, cluster_ca_certificate)
4. Actualiza environments/dev/main.tf para llamar al módulo gke

Incluye comentarios explicando por qué ml-serving no puede ser preemptible y qué hace el taint ml-workload.
```

**Verificación**:
```powershell
Push-Location infra\terraform\environments\dev
terraform plan   # Debe mostrar el cluster y los 2 node pools, sin errores
# Verificar en el plan que enable_private_nodes = true
# Verificar que ml-serving node pool tiene preemptible = false
Pop-Location
```

---

### T1.5 — Módulo Terraform: GCS Buckets y Storage

**Qué hace**: Crea los cuatro buckets de GCS con la estructura de almacenamiento del proyecto, retención, versionado y CMEK preparado.

**Prerequisitos**: T1.2 completada.

**Prompt**:
```
Crea el módulo Terraform para los buckets GCS de ReactorGuard.

Los cuatro buckets necesarios según el TDD sección 4.4:

1. reactorguard-data-raw
   - Propósito: datos crudos de sensores y simulación
   - Estructura de particionamiento (solo documentar, no Terraform): plant=X/year=X/month=X/day=X/hour=X/
   - Retención: 90 días para datos de simulación, sin retención para datos de planta reales
   - Versionado: desactivado (los datos de sensor son inmutables)

2. reactorguard-data-processed
   - Propósito: features calculadas y splits train/val/test
   - Retención: 30 días
   - Versionado: activado (los features pueden recalcularse)

3. reactorguard-models
   - Propósito: modelos serializados (PINN, BNN, conformal)
   - Retención: 365 días (auditoría regulatoria)
   - Versionado: activado (CRÍTICO — necesitamos historial completo de versiones de modelos)

4. reactorguard-mlflow
   - Propósito: artefactos y metadatos de experimentos MLflow
   - Retención: 180 días
   - Versionado: activado

Para todos los buckets:
- Región: europe-west1 (mismo que el cluster)
- Clase de almacenamiento: STANDARD
- Acceso uniforme a nivel de bucket (no ACLs)
- Encriptación: por ahora con clave gestionada por Google (GMEK); preparar variable para CMEK futuro
- Lifecycle rule: mover a NEARLINE después de 30 días, a COLDLINE después de 90 días
- Bloquear acceso público: sí

Genera:
1. infra/terraform/modules/storage/main.tf con los 4 google_storage_bucket
2. infra/terraform/modules/storage/variables.tf
3. infra/terraform/modules/storage/outputs.tf (nombres y URLs de cada bucket)
4. Actualiza environments/dev/main.tf

Añade comentarios explicando la política de lifecycle y por qué models tiene retención de 365 días.
```

**Verificación**:
```powershell
Push-Location infra\terraform\environments\dev
terraform plan   # 4 buckets, sin errores
# Verificar que uniform_bucket_level_access = true en todos
# Verificar que ningún bucket tiene public_access_prevention = "inherited"
Pop-Location
```

---

### T1.6 — Módulo Terraform: IAM y Workload Identity

**Qué hace**: Crea las cuentas de servicio GCP para cada microservicio y configura Workload Identity para que los pods de K8s accedan a GCP sin credenciales en texto plano.

**Prerequisitos**: T1.4 (GKE) y T1.5 (GCS) completadas o en el mismo plan.

**Prompt**:
```
Crea el módulo Terraform de IAM y Workload Identity para ReactorGuard.

El proyecto tiene estos microservicios, cada uno necesita su propia service account con permisos mínimos (principio de least privilege):

1. reactorguard-ingestion-sa
   - Lee de: reactorguard-data-raw (Storage Object Viewer)
   - Escribe en: reactorguard-data-raw, reactorguard-data-processed (Storage Object Creator)
   - Publica en Pub/Sub: roles/pubsub.publisher (para bridge Kafka→PubSub si se necesita)
   - Workload Identity: namespace reactorguard-ingestion, KSA sensor-validator

2. reactorguard-ml-sa
   - Lee de: reactorguard-data-processed, reactorguard-models (Storage Object Viewer)
   - Lee secretos: roles/secretmanager.secretAccessor
   - Escribe métricas: roles/monitoring.metricWriter
   - Workload Identity: namespace reactorguard-ml, KSA pinn-server

3. reactorguard-mlflow-sa
   - Lee y escribe en: reactorguard-mlflow, reactorguard-models (Storage Object Admin en ambos)
   - Workload Identity: namespace reactorguard-ml, KSA mlflow-server

4. reactorguard-cicd-sa
   - Para GitHub Actions: puede hacer push a GCR, leer/escribir en todos los buckets, desplegar en GKE
   - NO necesita Workload Identity (corre fuera del cluster)
   - Necesita generar una key JSON para GitHub Secrets

Para cada service account genera el binding de Workload Identity:
  serviceAccount:PROJECT_ID.svc.id.goog[NAMESPACE/KSA_NAME]

Genera:
1. infra/terraform/modules/iam/main.tf con google_service_account y google_project_iam_member para cada SA
2. infra/terraform/modules/iam/workload_identity.tf con google_service_account_iam_member para los bindings
3. infra/terraform/modules/iam/variables.tf
4. infra/terraform/modules/iam/outputs.tf (service account emails)
5. Actualiza environments/dev/main.tf

Añade un comentario prominente explicando cómo funciona Workload Identity: el flujo pod → KSA → GSA → GCP API.
```

**Verificación**:
```powershell
Push-Location infra\terraform\environments\dev
terraform plan   # ~12 recursos IAM, sin errores
Pop-Location
# Después de apply:
gcloud iam service-accounts list --filter="reactorguard"   # Debe mostrar las 4 SAs
```

---

### T1.7 — Módulo Terraform: Secret Manager y KMS

**Qué hace**: Crea los secretos iniciales en Secret Manager y la clave KMS para CMEK. Los pods leerán credenciales desde aquí en tiempo de ejecución, nunca desde variables de entorno hardcodeadas.

**Prerequisitos**: T1.6 completada.

**Prompt**:
```
Crea el módulo Terraform para Secret Manager y KMS en ReactorGuard.

Secretos a crear en Secret Manager (con valores placeholder — los reales se cargan manualmente):

1. reactorguard/scada-credentials → placeholder: {"username": "REPLACE_ME", "password": "REPLACE_ME", "endpoint": "REPLACE_ME"}
2. reactorguard/jwt-secret → placeholder: "REPLACE_WITH_32_BYTE_RANDOM_STRING"
3. reactorguard/mlflow-db-url → placeholder: "postgresql://REPLACE_ME"
4. reactorguard/gcp-api-key → placeholder: "REPLACE_ME"

Para KMS (CMEK):
- Key ring: reactorguard-keyring, región europe-west1
- Key: reactorguard-storage-key para encriptación de GCS buckets
- Rotation period: 90 días
- Propósito: ENCRYPT_DECRYPT

Genera:
1. infra/terraform/modules/security/secrets.tf con google_secret_manager_secret y google_secret_manager_secret_version para cada secreto
2. infra/terraform/modules/security/kms.tf con google_kms_key_ring y google_kms_crypto_key
3. infra/terraform/modules/security/variables.tf
4. infra/terraform/modules/security/outputs.tf (secret IDs, key ring name)
5. Actualiza environments/dev/main.tf

También genera un script PowerShell infra/scripts/Load-Secrets.ps1 con arquitectura de 4 capas:

   CAPA 1 — Configuración:
   #Requires -Version 7.0
   [CmdletBinding()]
   param(
     [Parameter(Mandatory)][string]$ScadaUsername,
     [Parameter(Mandatory)][string]$ScadaPassword,
     [Parameter(Mandatory)][string]$ScadaEndpoint,
     [Parameter(Mandatory)][string]$JwtSecret,
     [Parameter(Mandatory)][string]$MlflowDbUrl,
     [Parameter(Mandatory)][string]$GcpApiKey,
     [string]$ProjectId = "reactorguard-platform"
   )
   Set-StrictMode -Version Latest
   $ErrorActionPreference = "Stop"

   CAPA 2 — Utilidades:
   - function Assert-NotEmpty([string]$Value, [string]$ParamName) — valida que el valor no sea vacío ni placeholder
   - function Write-Step([string]$Message) — output con timestamp
   - function ConvertTo-SecretPayload([hashtable]$Data) — serializa a JSON

   CAPA 3 — Servicio:
   - function Set-GcpSecret([string]$SecretName, [string]$Payload) — llama a gcloud secrets versions add con manejo de errores

   CAPA 4 — Orquestación:
   - function Invoke-LoadSecrets — valida todos los parámetros, llama a Set-GcpSecret para cada secreto, imprime resumen
   Invoke-LoadSecrets

IMPORTANTE: Añade una nota grande en comments indicando que los valores placeholder del terraform NO son los valores reales y que hay que ejecutar Load-Secrets.ps1 antes de desplegar.
```

**Verificación**:
```powershell
Push-Location infra\terraform\environments\dev
terraform plan   # Secrets + KMS, sin errores
Pop-Location
# Después de apply:
gcloud secrets list                                        # Debe mostrar los 4 secretos
gcloud kms keyrings list --location=europe-west1           # Debe mostrar reactorguard-keyring
```

---

### T1.8 — Módulo Terraform: Cloud Armor WAF + Load Balancer + IAP

**Qué hace**: Crea la capa de seguridad perimetral: el WAF bloquea ataques OWASP antes de que lleguen al cluster, e IAP asegura que solo usuarios autenticados con Google pueden acceder.

**Prerequisitos**: T1.4 (GKE) completada.

**Prompt**:
```
Crea el módulo Terraform para Cloud Armor WAF e Identity-Aware Proxy (IAP) de ReactorGuard.

Requisitos de seguridad perimetral del TDD:
- Cloud Armor policy con reglas OWASP ModSecurity Core Rule Set
- HTTPS Load Balancer que termina TLS y redirige HTTP → HTTPS
- IAP para autenticación antes de llegar al cluster
- DDoS protection habilitado

Para Cloud Armor:
1. google_compute_security_policy con:
   - Regla default: allow (las reglas específicas deniegan)
   - Regla OWASP SQLi: expresión evaluatePreconfigs("sqli-v33-stable"), priority 1000, action deny(403)
   - Regla OWASP XSS: expresión evaluatePreconfigs("xss-v33-stable"), priority 1001, action deny(403)
   - Regla rate limiting: 1000 requests/min por IP, action throttle, priority 2000
   - Adaptive Protection habilitado (ML-based DDoS detection)

Para el Load Balancer:
2. google_compute_global_address para IP estática del LB
3. google_compute_managed_ssl_certificate (Google-managed, domain: api.reactorguard.internal)
4. google_compute_url_map redirigiendo todo a un backend service
5. google_compute_backend_service apuntando al NEG (Network Endpoint Group) del GKE Ingress
6. google_compute_target_https_proxy con el SSL cert
7. google_compute_global_forwarding_rule (puerto 443)
8. Redirección HTTP→HTTPS: google_compute_url_map separado + forwarding_rule puerto 80

Para IAP:
9. google_iap_brand (marca OAuth)
10. google_iap_client
11. Outputs de client_id y client_secret para configurar en GKE Ingress annotations

Genera:
1. infra/terraform/modules/security/armor.tf
2. infra/terraform/modules/security/lb.tf
3. infra/terraform/modules/security/iap.tf
4. Actualiza outputs.tf con lb_ip, iap_client_id

Añade comentarios explicando el orden de evaluación de reglas de Cloud Armor y por qué IAP se sitúa después del LB.
```

**Verificación**:
```powershell
Push-Location infra\terraform\environments\dev
terraform plan   # LB + Armor + IAP resources, sin errores
# Revisar que security_policy está asociado al backend_service
# Revisar que el forwarding_rule HTTP redirige a HTTPS (no termina en el backend)
Pop-Location
```

---

### T1.9 — Aplicar Terraform y verificar infraestructura completa

**Qué hace**: Ejecuta el plan completo de Terraform y verifica que todos los recursos están creados correctamente antes de pasar a Kubernetes.

**Prerequisitos**: T1.2 a T1.8 completadas. Terraform plan sin errores.

**Prompt**:
```
Genera scripts PowerShell de verificación y destrucción de infraestructura para ReactorGuard.
Plataforma: Windows 11, PowerShell 7+. No generes scripts .sh.

SCRIPT 1: infra/scripts/Verify-Infra.ps1
Arquitectura de 4 capas:

CAPA 1 — Configuración:
#Requires -Version 7.0
[CmdletBinding()]
param(
  [string]$ProjectId    = "reactorguard-platform",
  [string]$ClusterName  = "reactorguard-cluster",
  [string]$Region       = "europe-west1"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

CAPA 2 — Utilidades:
- function Write-CheckResult([string]$Name, [bool]$Passed, [string]$Detail) — imprime ✅ / ❌ con detalle
- function Invoke-WithTimeout([scriptblock]$Action, [int]$TimeoutSeconds = 30) — ejecuta con timeout
- function Get-GcloudJson([string[]]$Arguments) — ejecuta gcloud y parsea JSON con ConvertFrom-Json

CAPA 3 — Funciones de verificación (una por criterio, separación de responsabilidades):
- function Test-GkeCluster — verifica que el cluster está RUNNING y ambos node pools están Ready
- function Test-GcsBuckets — verifica los 4 buckets, uniform_bucket_level_access y permisos de escritura
- function Test-SecretManager — verifica que los 4 secretos existen y tienen versiones activas
- function Test-Networking — kubectl cluster-info + verifica IPs privadas en nodos + Cloud NAT
- function Test-IamAccounts — lista las 4 service accounts y sus roles

CAPA 4 — Orquestación:
- function Invoke-VerifyInfra — ejecuta todos los Test-* en orden, acumula resultados, imprime tabla final
  - Si alguna verificación falla: imprime el error y exit 1
  - Si todo pasa: imprime "Infraestructura Fase 1 - Semana 1: LISTA" y exit 0
Invoke-VerifyInfra

SCRIPT 2: infra/scripts/Remove-DevInfra.ps1
Destruye la infraestructura de dev con confirmación explícita:
- Solicitar confirmación escribiendo "DESTROY-DEV" para evitar destrucciones accidentales
- Ejecutar terraform destroy en infra/terraform/environments/dev
- Borrar el directorio .terraform local después de la destrucción
- Loggear cada paso con timestamp
```

**Verificación**:
```powershell
.\infra\scripts\Verify-Infra.ps1
# Debe imprimir tabla con todos los checks en ✅
# Exit code 0
```

---

## SEMANA 2 — Plataforma Base, Seguridad y Kafka

---

### T2.1 — Namespaces K8s y NetworkPolicies

**Qué hace**: Crea los cuatro namespaces de Kubernetes con las políticas de red que controlan exactamente qué puede comunicarse con qué. Sin NetworkPolicies, cualquier pod puede hablar con cualquier otro.

**Prerequisitos**: T1.9 completada. `kubectl` apuntando al cluster.

**Prompt**:
```
Crea los manifiestos Kubernetes para los namespaces y NetworkPolicies de ReactorGuard.

Según la sección 8.1 del TDD, la matriz de comunicación es:

Namespace reactorguard-ingestion:
- Acepta tráfico ENTRANTE desde: namespace kafka-operator (topics Kafka)
- Envía tráfico SALIENTE hacia: namespace reactorguard-ml
- Puertos expuestos internamente: 8000 (app), 9090 (metrics)

Namespace reactorguard-ml:
- Acepta tráfico ENTRANTE desde: reactorguard-ingestion, reactorguard-observability
- Envía tráfico SALIENTE hacia: GCP APIs (Secret Manager, GCS) — requiere egress a internet vía NAT
- Puertos expuestos internamente: 8000 (app), 9090 (metrics)

Namespace reactorguard-observability:
- Acepta tráfico ENTRANTE desde: todos los namespaces en puertos 8000 y 9090 (scraping)
- Envía tráfico SALIENTE hacia: internet (alerting a PagerDuty/Slack)
- Puertos expuestos: 3000 (Grafana), 9090 (Prometheus)

Namespace kafka-operator:
- Solo tráfico interno entre pods del namespace
- Expone puerto 9092 (Kafka bootstrap) hacia reactorguard-ingestion

Genera estos archivos:

1. k8s/base/namespaces.yaml — los 4 namespaces con labels estándar:
   app.kubernetes.io/managed-by: kustomize
   environment: dev (para dev overlay)

2. k8s/base/rbac/network-policies.yaml — una NetworkPolicy por namespace:
   - Política "deny-all" base (denegar todo ingress y egress por defecto)
   - Políticas de allow específicas según la matriz anterior
   - Usar namespaceSelector con los labels de los namespaces para referenciarlos

3. k8s/base/rbac/kustomization.yaml

4. k8s/base/kustomization.yaml que incluya todos los recursos base

Añade comentarios en cada NetworkPolicy explicando en lenguaje natural qué permite y qué deniega.
```

**Verificación**:
```powershell
kubectl apply -k k8s/base/
kubectl get namespaces | Select-String "reactorguard"   # 3 namespaces
kubectl get networkpolicies -A                           # Al menos 8 policies (deny-all + allows por namespace)
# Test de conectividad (debería fallar):
kubectl run test-pod --image=busybox -n reactorguard-ml --rm -it -- wget -qO- http://service.reactorguard-ingestion:8000
```

---

### T2.2 — Workload Identity en Kubernetes (KSAs)

**Qué hace**: Crea las Kubernetes Service Accounts y las anota con las Google Service Accounts creadas en T1.6. Esto activa el mecanismo de Workload Identity para que los pods accedan a GCP.

**Prerequisitos**: T1.6 (IAM) y T2.1 (namespaces) completadas.

**Prompt**:
```
Crea los manifiestos Kubernetes para las Service Accounts con Workload Identity en ReactorGuard.

Workload Identity funciona anotando la Kubernetes Service Account (KSA) con el email de la Google Service Account (GSA). El pod usa la KSA, y GKE lo mapea automáticamente a la GSA sin credenciales.

Las anotaciones necesarias según la documentación de GCP:
  annotations:
    iam.gke.io/gcp-service-account: GSA_EMAIL

Los bindings que creamos en Terraform (T1.6) ya dan permiso a la KSA para impersonar la GSA.

Genera estos archivos:

1. k8s/base/ingestion/serviceaccount.yaml
   - Namespace: reactorguard-ingestion
   - Name: sensor-validator
   - Annotation: iam.gke.io/gcp-service-account: reactorguard-ingestion-sa@reactorguard-platform.iam.gserviceaccount.com

2. k8s/base/ml/serviceaccount.yaml
   - Dos KSAs en el mismo archivo separadas por ---:
   - pinn-server: anotada con reactorguard-ml-sa@...
   - mlflow-server: anotada con reactorguard-mlflow-sa@...

3. k8s/base/rbac/roles.yaml
   - Role en cada namespace que permite a las KSAs leer ConfigMaps y Secrets del mismo namespace
   - RoleBinding correspondiente

4. Un script PowerShell infra/scripts/Test-WorkloadIdentity.ps1 con arquitectura de 4 capas:

   CAPA 1 — Configuración:
   #Requires -Version 7.0
   [CmdletBinding()]
   param(
     [string]$ProjectId   = "reactorguard-platform",
     [string]$Namespace   = "reactorguard-ml",
     [string]$KsaName     = "pinn-server",
     [string]$SecretName  = "reactorguard/jwt-secret"
   )
   Set-StrictMode -Version Latest
   $ErrorActionPreference = "Stop"

   CAPA 2 — Utilidades:
   - function New-TestPodManifest([string]$PodName, [string]$Namespace, [string]$Ksa) — genera YAML del pod temporal
   - function Wait-PodReady([string]$PodName, [string]$Namespace, [int]$TimeoutSec = 60)

   CAPA 3 — Servicio:
   - function Invoke-SecretManagerTest([string]$PodName, [string]$ProjectId, [string]$SecretName) — ejecuta dentro del pod:
     $token = kubectl exec $PodName -n $Namespace -- sh -c "gcloud auth print-access-token"
     Invoke-WebRequest con header Authorization: Bearer $token hacia la Secret Manager API
     Verifica que StatusCode == 200
   - function Remove-TestPod([string]$PodName, [string]$Namespace) — limpieza garantizada

   CAPA 4 — Orquestación:
   - function Invoke-WorkloadIdentityTest — crea pod, ejecuta test, limpia (con try/finally para garantizar limpieza)
   Invoke-WorkloadIdentityTest

Añade comentarios explicando el flujo completo: pod → KSA annotation → GSA impersonation → GCP API.
```

**Verificación**:
```powershell
kubectl apply -k k8s/base/
kubectl get serviceaccounts -n reactorguard-ml   # pinn-server, mlflow-server
.\infra\scripts\Test-WorkloadIdentity.ps1        # HTTP 200, no 403
```

---

### T2.3 — Instalación Strimzi Kafka Operator

**Qué hace**: Instala el operador de Strimzi que gestiona el ciclo de vida de Kafka en K8s. El operador es el "controlador" que traduce recursos custom (`kind: Kafka`) en Deployments, Services y ConfigMaps reales.

**Prerequisitos**: T2.1 completada. Helm instalado localmente.

**Prompt**:
```
Crea los manifiestos y scripts para instalar y configurar el operador Strimzi en ReactorGuard.
Plataforma: Windows 11, PowerShell 7+. No generes scripts .sh.

Versión target: Strimzi 0.39.0 (especificada en el TDD sección 7.2)
Namespace destino: kafka-operator

Genera:

1. infra/terraform/modules/kafka/main.tf con:
   - helm_release "strimzi" apuntando a https://strimzi.io/charts/, chart strimzi-kafka-operator, version 0.39.0
   - Namespace kafka-operator, create_namespace=true
   - Values: watchNamespaces=["kafka-operator"], logLevel=INFO
   - wait=true, timeout=300

2. k8s/base/kafka/kafka-cluster.yaml — el recurso custom Kafka según el TDD sección 7.2:
   apiVersion: kafka.strimzi.io/v1beta2
   kind: Kafka
   Con spec exactamente como el TDD: 3 replicas kafka, 3 replicas zookeeper, y además:
   - config: log.retention.hours=168, num.partitions=12, default.replication.factor=3, min.insync.replicas=2
   - storage kafka: type persistent-claim, size 50Gi, class standard-rwo
   - storage zookeeper: type persistent-claim, size 10Gi
   - listeners: plain (9092) y tls (9093) dentro del cluster, sin external listener
   - resources kafka: request 1CPU/2Gi, limit 2CPU/4Gi
   - resources zookeeper: request 0.5CPU/1Gi, limit 1CPU/2Gi

3. k8s/base/kafka/kafka-topics.yaml — los 3 KafkaTopic resources:
   - sensor-readings-raw: partitions=12, replicas=3, retention.ms=604800000 (7 días)
   - sensor-validated: partitions=12, replicas=3, retention.ms=604800000
   - anomaly-alerts: partitions=3, replicas=3, retention.ms=2592000000 (30 días, alertas duran más)

4. Un script PowerShell infra/scripts/Install-Kafka.ps1 con arquitectura de 4 capas:

   CAPA 1 — Configuración:
   #Requires -Version 7.0
   [CmdletBinding()]
   param(
     [string]$Namespace       = "kafka-operator",
     [string]$StrimziVersion  = "0.39.0",
     [int]$TimeoutSeconds     = 300
   )
   Set-StrictMode -Version Latest
   $ErrorActionPreference = "Stop"

   CAPA 2 — Utilidades:
   - function Wait-KubernetesPod([string]$LabelSelector, [string]$Namespace, [int]$TimeoutSec) — wrapper de kubectl wait
   - function Test-KafkaClusterReady([string]$Namespace) — verifica que todos los brokers están Running

   CAPA 3 — Servicio:
   - function Install-StrimziOperator — aplica el módulo Terraform de kafka y espera a que el operador esté Ready
   - function Deploy-KafkaCluster — aplica kafka-cluster.yaml y espera a que los brokers estén listos
   - function Deploy-KafkaTopics — aplica kafka-topics.yaml y verifica que los 3 topics existen

   CAPA 4 — Orquestación:
   - function Invoke-KafkaInstall — llama a las funciones de servicio en orden con manejo de errores
   Invoke-KafkaInstall

Añade comentarios explicando la diferencia entre el Strimzi operator y los recursos custom que gestiona.
```

**Verificación**:
```powershell
.\infra\scripts\Install-Kafka.ps1
kubectl get pods -n kafka-operator       # strimzi-operator Running
kubectl get kafka -n kafka-operator      # reactorguard-cluster, READY=True
kubectl get kafkatopic -n kafka-operator # 3 topics
```

---

### T2.4 — Verificación de Kafka: latencia y throughput

**Qué hace**: Prueba funcional de Kafka: produce y consume mensajes para verificar que el cluster responde dentro de los criterios de éxito (latency p99 < 10ms en red interna).

**Prerequisitos**: T2.3 completada. Kafka cluster en estado READY.

**Prompt**:
```
Crea un script de benchmarking y verificación funcional de Kafka para ReactorGuard.
Plataforma: Windows 11, PowerShell 7+. No generes scripts .sh.

El criterio de éxito de la Fase 1 es: latencia produce/consume p99 < 10ms en red interna del cluster.

Genera:

1. tests/integration/test_kafka_connectivity.py — test de conectividad básica:
   - Usa kafka-python (pip install kafka-python)
   - Bootstrap servers: reactorguard-cluster-kafka-bootstrap.kafka-operator:9092
   - Produce 100 mensajes al topic sensor-readings-raw con un schema de sensor reading simplificado
   - Consume esos 100 mensajes desde el inicio (offset=earliest)
   - Verifica que todos los mensajes llegan y el contenido es correcto
   - Imprime latencia promedio

2. tests/integration/benchmark_kafka.py — benchmark de latencia y throughput:
   - Produce 10,000 mensajes de 1KB (tamaño típico de una lectura de sensor con su metadata)
   - Mide latencia end-to-end por mensaje: tiempo desde produce() hasta que el consumer lo recibe
   - Calcula y reporta: p50, p95, p99, p99.9 de latencia
   - Calcula throughput: mensajes/segundo y MB/segundo
   - El test FALLA si p99 > 10ms
   - Genera un archivo JSON con los resultados: tests/results/kafka_benchmark.json

3. Un pod de utilidad k8s/tools/kafka-client-pod.yaml:
   - Imagen: confluentinc/cp-kafka:7.5.0
   - En namespace kafka-operator
   - Command: sleep infinity (para poder ejecutar comandos interactivos)
   - Útil para depuración con kafka-console-producer y kafka-console-consumer

4. Script PowerShell tests/integration/Invoke-KafkaTests.ps1 con arquitectura de 4 capas:

   CAPA 1 — Configuración:
   #Requires -Version 7.0
   [CmdletBinding()]
   param(
     [string]$Namespace    = "kafka-operator",
     [string]$ResultsPath  = "tests/results",
     [switch]$SkipBenchmark
   )
   Set-StrictMode -Version Latest
   $ErrorActionPreference = "Stop"

   CAPA 2 — Utilidades:
   - function New-TestResultsDir([string]$Path) — crea el directorio de resultados si no existe
   - function Get-BenchmarkResult([string]$JsonPath) — lee y parsea el JSON de resultados
   - function Write-TestResult([string]$Name, [bool]$Passed, [string]$Value) — output formateado con ✅/❌

   CAPA 3 — Servicio:
   - function Deploy-KafkaClientPod — despliega el pod de utilidad y espera a que esté Running
   - function Invoke-ConnectivityTest([string]$PodName) — copia y ejecuta test_kafka_connectivity.py dentro del pod
   - function Invoke-BenchmarkTest([string]$PodName) — copia y ejecuta benchmark_kafka.py, recoge resultados
   - function Remove-KafkaClientPod([string]$PodName) — limpieza garantizada

   CAPA 4 — Orquestación:
   - function Invoke-AllKafkaTests — despliega pod, ejecuta tests, recoge resultados (try/finally para limpieza)
   Invoke-AllKafkaTests

El schema del mensaje de prueba debe ser compatible con el sensor_reading.py del TDD sección 4.2 (aunque simplificado).
```

**Verificación**:
```powershell
.\tests\integration\Invoke-KafkaTests.ps1
# Debe imprimir:
# ✅ Kafka connectivity: OK (100/100 messages delivered)
# ✅ Latency p99: X.Xms (< 10ms)
# ✅ Throughput: XXXX msg/s
```

---

### T2.5 — Skeleton CI/CD con GitHub Actions

**Qué hace**: Crea el pipeline de CI/CD básico que se ejecuta en cada push: lint, type check, tests y build de imagen Docker. Es el esqueleto sobre el que en la Fase 7 añadiremos los ML safety gates.

**Prerequisitos**: T1.1 completada. Repositorio en GitHub con secrets configurados: `GCP_SA_KEY` (JSON key de `reactorguard-cicd-sa`), `GCP_PROJECT_ID`.

**Prompt**:
```
Crea el pipeline CI/CD base de GitHub Actions para ReactorGuard.

Este es el skeleton de la Fase 1. En la Fase 7 añadiremos los ML safety gates. Por ahora debe cubrir: lint, type check, unit tests, security scans, build y push.

Genera estos archivos:

1. .github/workflows/ci.yaml — Pipeline principal, se dispara en push a cualquier rama y en PR a main:

   Jobs en orden (cada uno depende del anterior):

   Job 1: lint-and-typecheck
   - Python 3.11
   - pip install ruff mypy
   - ruff check . --select E,W,F,I (imports, style, errors)
   - mypy ml/ api/ data/ --ignore-missing-imports

   Job 2: unit-tests (depends-on: lint-and-typecheck)
   - pip install -r requirements.txt
   - pytest tests/unit/ -v --tb=short --junitxml=test-results.xml
   - Upload test results as artifact

   Job 3: security-scan (depends-on: lint-and-typecheck, puede correr en paralelo con unit-tests)
   - Semgrep: uses returntocorp/semgrep-action con ruleset p/python y p/secrets
   - Gitleaks: uses gitleaks/gitleaks-action para detectar secretos en el código
   - pip-audit: pip install pip-audit && pip-audit (vulnerabilidades en dependencias Python)

   Job 4: build-and-push (depends-on: unit-tests, security-scan)
   - Solo en push a main o tags v*
   - Autentica en GCR con la SA key de GitHub Secrets
   - docker build -t gcr.io/$GCP_PROJECT_ID/reactorguard-api:$GITHUB_SHA .
   - Trivy scan de la imagen: uses aquasecurity/trivy-action, fail si severity CRITICAL
   - docker push solo si Trivy pasa

2. .github/workflows/terraform.yaml — se dispara en cambios en infra/terraform/:
   - terraform fmt -check (falla si hay archivos no formateados)
   - terraform init
   - terraform validate
   - tfsec (security scan de Terraform): uses aquasecurity/tfsec-action
   - terraform plan (solo muestra el plan, no aplica)
   - En PR: comenta el plan en el PR usando actions como setup-terraform

3. api/Dockerfile — imagen base para la API (skeleton, se completará en Fase 6):
   - Multi-stage: builder (instala deps) + runtime (copia solo lo necesario)
   - Base: python:3.11-slim
   - Usuario no-root: crea usuario appuser
   - HEALTHCHECK con /health endpoint
   - Expone puerto 8000

4. .github/dependabot.yml — actualizaciones automáticas de dependencias Python y GitHub Actions

Añade comentarios en cada job explicando por qué ese step es necesario para un proyecto safety-critical.
```

**Verificación**:
```powershell
git add .
git commit -m "feat: add CI/CD skeleton"
git push
# Ir a GitHub Actions y verificar que el workflow se dispara
# Todos los jobs deben estar en verde (o amarillo si no hay tests todavía)
# El job build-and-push debe saltar si no estamos en main
```

---

### T2.6 — Managed Prometheus en GKE

**Qué hace**: Configura Google Cloud Managed Service for Prometheus (GMP) para que Prometheus esté disponible sin operar un servidor Prometheus propio. Establece el scraping de métricas básicas del cluster.

**Prerequisitos**: T1.4 (GKE con Managed Prometheus habilitado) y T2.1 (namespaces) completadas.

**Prompt**:
```
Configura Google Cloud Managed Service for Prometheus (GMP) en el cluster de ReactorGuard.
Plataforma: Windows 11, PowerShell 7+. No generes scripts .sh.

GMP está habilitado a nivel de cluster desde Terraform (T1.4). Ahora necesitamos configurar qué métricas recoger y dónde guardarlas.

Genera:

1. k8s/base/observability/prometheus-config.yaml — PodMonitoring resources (recurso custom de GMP):
   - PodMonitoring para namespace reactorguard-ml: scraping del label app: pinn-server en puerto 9090, intervalo 15s
   - PodMonitoring para namespace reactorguard-ingestion: scraping del label app: sensor-validator en puerto 9090
   - ClusterPodMonitoring para Kafka: scraping de pods en kafka-operator con label strimzi.io/name, puerto 9404 (JMX exporter)

2. k8s/base/observability/grafana-datasource-configmap.yaml — ConfigMap con la configuración de datasource para que Grafana apunte a GMP:
   - Type: prometheus
   - URL: https://monitoring.googleapis.com/v1/projects/reactorguard-platform/location/global/prometheus
   - Auth: using Google service account (Workload Identity de Grafana)

3. k8s/base/observability/kustomization.yaml

4. observability/prometheus/rules.yaml — PrometheusRule con las alertas críticas del TDD sección 10.1:
   - reactorguard_critical_alerts_total > 0 por 0m → severity: critical, página inmediata
   - reactorguard_prediction_latency_seconds p99 > 0.05 por 5m → severity: warning
   - reactorguard_physics_violations_total > 0 por 0m → severity: critical
   - reactorguard_sensor_faults_total{fault_type="stuck"} > 3 en 1h → severity: warning
   - reactorguard_uncertainty_calibration_ece > 0.05 por 15m → severity: warning + trigger retrain

5. Un script PowerShell infra/scripts/Test-Prometheus.ps1 con arquitectura de 4 capas:

   CAPA 1 — Configuración:
   #Requires -Version 7.0
   [CmdletBinding()]
   param(
     [string]$ProjectId = "reactorguard-platform"
   )
   Set-StrictMode -Version Latest
   $ErrorActionPreference = "Stop"

   CAPA 2 — Utilidades:
   - function Invoke-GmpQuery([string]$ProjectId, [string]$PromQL) — ejecuta una query PromQL contra la API de GMP usando Invoke-WebRequest con el token de gcloud

   CAPA 3 — Servicio:
   - function Test-PodMonitoringResources — kubectl get podmonitoring -A y verifica que existen los 3 esperados
   - function Test-GmpMetricsAvailable([string]$ProjectId) — hace una query de prueba a GMP y devuelve el número de time series activas

   CAPA 4 — Orquestación:
   - function Invoke-PrometheusVerification — ejecuta los tests y muestra resumen con ✅/❌
   Invoke-PrometheusVerification

Incluye comentarios explicando la diferencia entre GMP (managed) y un Prometheus self-hosted, y por qué elegimos GMP para este proyecto.
```

**Verificación**:
```powershell
kubectl apply -k k8s/base/observability/
kubectl get podmonitoring -A      # Los 3 PodMonitoring
.\infra\scripts\Test-Prometheus.ps1   # Debe mostrar métricas activas
# En Cloud Console → Monitoring → Metrics Explorer: buscar kubernetes.io/container/cpu_request_cores
```

---

### T2.7 — Verificación completa de la Fase 1

**Qué hace**: Script de cierre de fase que ejecuta todas las verificaciones de los criterios de éxito del TDD y genera un reporte de estado.

**Prerequisitos**: Todas las tareas T1.1 a T2.6 completadas.

**Prompt**:
```
Crea el script de verificación y cierre de la Fase 1 de ReactorGuard.
Plataforma: Windows 11, PowerShell 7+. No generes scripts .sh.

SCRIPT: infra/scripts/Test-Phase1Completion.ps1
Arquitectura de 4 capas:

CAPA 1 — Configuración:
#Requires -Version 7.0
[CmdletBinding()]
param(
  [string]$ProjectId       = "reactorguard-platform",
  [string]$ClusterName     = "reactorguard-cluster",
  [string]$Region          = "europe-west1",
  [string]$KafkaBenchmark  = "tests/results/kafka_benchmark.json",
  [string]$ReportOutput    = "docs/phase1_completion_report.md"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

CAPA 2 — Utilidades:
- class CheckResult { [string]$Name; [bool]$Passed; [string]$Value; [bool]$IsBlocking }
- function New-CheckResult([string]$Name, [bool]$Passed, [string]$Value, [bool]$IsBlocking) → [CheckResult]
- function Write-CheckTable([CheckResult[]]$Results) — imprime tabla formateada con ✅/❌ y marca BLOCKING
- function Export-MarkdownReport([CheckResult[]]$Results, [string]$OutputPath) — genera el .md con timestamp

CAPA 3 — Funciones de verificación (cada criterio del TDD en su propia función, separación de responsabilidades):
Criterio 1 (BLOCKING):   function Test-GkeClusterRunning — kubectl get nodes, verificar STATUS=Ready para todos
Criterio 2 (BLOCKING):   function Test-KafkaLatency([string]$BenchmarkJson) — lee el JSON y compara p99 contra 10ms
Criterio 3 (BLOCKING):   function Test-TerraformValidate — ejecuta terraform validate en infra/terraform/environments/dev
Criterio 4 (BLOCKING):   function Test-TrivyImageScan — ejecuta Trivy sobre la imagen Docker base, cuenta CRITICAL
Criterio 5 (BLOCKING):   function Test-GcsBucketsAccessible — intenta leer y escribir en cada bucket con las SAs correspondientes
Criterio 6 (WARNING):    function Test-SecretManagerLoaded — verifica que los 4 secretos existen y tienen versiones activas
Criterio 7 (BLOCKING):   function Test-NetworkPoliciesApplied — kubectl get networkpolicies -A, al menos 8 policies
Criterio 8 (WARNING):    function Test-CiCdLastRunSuccessful — verifica con gh CLI que el último run de ci.yaml pasó

CAPA 4 — Orquestación:
function Invoke-Phase1Verification:
  - Ejecuta cada Test-* con Invoke-WithTimeout 30s
  - Acumula resultados en [CheckResult[]]
  - Llama a Write-CheckTable y Export-MarkdownReport
  - Imprime resumen:
    === FASE 1 FOUNDATION ===
    X/8 criterios cumplidos
    Estado: LISTA PARA FASE 2 | BLOQUEADA (si algún criterio BLOCKING falla)
  - Exit 0 si todos los criterios BLOCKING pasan; exit 1 si alguno falla
Invoke-Phase1Verification

Define en un bloque de comentarios al principio cuáles criterios son BLOCKING vs WARNING y por qué.
```

**Verificación**:
```powershell
.\infra\scripts\Test-Phase1Completion.ps1
# Salida esperada:
# ✅ GKE cluster RUNNING (2/2 node pools Ready)           [BLOCKING]
# ✅ Kafka p99 latency: 7.3ms (< 10ms)                   [BLOCKING]
# ✅ terraform validate: 0 errors                         [BLOCKING]
# ✅ Trivy: 0 CRITICAL vulnerabilities                    [BLOCKING]
# ✅ GCS buckets: 4/4 accesibles                          [BLOCKING]
# ✅ Secret Manager: 4/4 secretos activos                 [WARNING]
# ✅ NetworkPolicies: 8 policies aplicadas                [BLOCKING]
# ✅ CI/CD: último run exitoso                            [WARNING]
# === FASE 1 FOUNDATION ===
# 8/8 criterios cumplidos
# Estado: LISTA PARA FASE 2
```

---

## Resumen de la Fase 1

| # | Tarea | Semana | Output principal | Script |
|---|-------|--------|-----------------|--------|
| T1.1 | Estructura del repositorio | 1 | Repo con todos los directorios y configs | — |
| T1.2 | Terraform backend + provider | 1 | Backend GCS + provider GCP configurados | `bootstrap.ps1` |
| T1.3 | Módulo VPC + networking | 1 | VPC privada con subnets GKE | — |
| T1.4 | Módulo GKE cluster | 1 | Cluster con 2 node pools | — |
| T1.5 | Módulo GCS buckets | 1 | 4 buckets con lifecycle policies | — |
| T1.6 | Módulo IAM + Workload Identity | 1 | 4 service accounts con permisos mínimos | — |
| T1.7 | Secret Manager + KMS | 1 | 4 secretos + clave KMS | `Load-Secrets.ps1` |
| T1.8 | Cloud Armor WAF + LB + IAP | 1 | Perímetro de seguridad completo | — |
| T1.9 | Apply Terraform + verificación | 1 | Infraestructura completa en GCP | `Verify-Infra.ps1` / `Remove-DevInfra.ps1` |
| T2.1 | Namespaces K8s + NetworkPolicies | 2 | Aislamiento de red entre servicios | — |
| T2.2 | Workload Identity KSAs | 2 | KSAs anotadas, pods con acceso GCP | `Test-WorkloadIdentity.ps1` |
| T2.3 | Strimzi Kafka Operator | 2 | Kafka cluster 3 brokers + 3 topics | `Install-Kafka.ps1` |
| T2.4 | Verificación Kafka latencia | 2 | Benchmark p99 < 10ms confirmado | `Invoke-KafkaTests.ps1` |
| T2.5 | CI/CD GitHub Actions | 2 | Pipeline lint + test + scan + build | — |
| T2.6 | Managed Prometheus | 2 | Métricas del cluster en GMP | `Test-Prometheus.ps1` |
| T2.7 | Verificación cierre Fase 1 | 2 | Reporte 8/8 criterios cumplidos | `Test-Phase1Completion.ps1` |

### Scripts PowerShell de la Fase 1

| Script | Capa de arquitectura aplicada | Propósito |
|--------|------------------------------|-----------|
| `infra/scripts/bootstrap.ps1` | Config → Utils → Service → Orchestration | Crear bucket de estado y habilitar APIs GCP |
| `infra/scripts/Load-Secrets.ps1` | Config → Utils → Service → Orchestration | Cargar secretos reales en Secret Manager |
| `infra/scripts/Verify-Infra.ps1` | Config → Utils → Service → Orchestration | Verificar infraestructura Semana 1 completa |
| `infra/scripts/Remove-DevInfra.ps1` | Config → Utils → Service → Orchestration | Destruir infraestructura dev con confirmación |
| `infra/scripts/Test-WorkloadIdentity.ps1` | Config → Utils → Service → Orchestration | Verificar acceso GCP desde pods K8s |
| `infra/scripts/Install-Kafka.ps1` | Config → Utils → Service → Orchestration | Instalar Strimzi y desplegar Kafka cluster |
| `tests/integration/Invoke-KafkaTests.ps1` | Config → Utils → Service → Orchestration | Benchmark latencia/throughput Kafka |
| `infra/scripts/Test-Prometheus.ps1` | Config → Utils → Service → Orchestration | Verificar GMP y PodMonitoring resources |
| `infra/scripts/Test-Phase1Completion.ps1` | Config → Utils → Service → Orchestration | Reporte final de cierre de Fase 1 |
