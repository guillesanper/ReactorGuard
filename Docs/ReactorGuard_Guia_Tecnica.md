# ReactorGuard — Guía Técnica de Conceptos y Tecnologías
*Documento de referencia personal · Basado en TDD v1.0*

---

## Parte I — El Problema: ¿Por qué detectar anomalías en un reactor es difícil?

### El contexto físico

Un reactor nuclear de agua a presión (PWR, *Pressurized Water Reactor*) funciona así en términos simplificados: el núcleo genera calor por fisión. Ese calor lo absorbe el agua del **circuito primario**, que nunca hierve porque está bajo altísima presión (~155 bar). Esa agua caliente transfiere su energía a un circuito secundario en un generador de vapor, y ese vapor mueve una turbina.

Durante todo ese proceso, hay **miles de sensores** midiendo constantemente: temperatura del coolant (refrigerante), flujo de agua, posición de barras de control, flujo de neutrones, presión... Aproximadamente uno o varios miles de lecturas por segundo por planta.

El reto de ReactorGuard es monitorizar todo eso en tiempo real y distinguir: ¿ese cambio en la temperatura es un evento real que requiere atención, o es simplemente que el sensor está fallando?

---

### Los tres tipos de "alarma" que se confunden constantemente

**1. Anomalía de proceso real** — Algo está pasando físicamente en el reactor. Por ejemplo, el flujo de coolant está bajando. Hay que actuar.

**2. Fallo de sensor** — El sensor está roto, deriva, o está "congelado" en un valor. Los datos son mentira, pero el reactor está bien.

**3. Ruido** — Interferencia electromagnética, vibración mecánica. Un pico de un solo sample que no tiene sentido físico.

El problema crítico: los tres pueden producir exactamente los mismos valores numéricos en los datos. Un sensor de temperatura que mide 320°C puede estar bien, estar fallando por deriva, o estar marcando una excursión térmica real. **No se puede distinguir mirando el número solo.**

Por eso ReactorGuard necesita física + ML + incertidumbre. Ninguno de los tres por separado es suficiente.

---

### El coste asimétrico del error

En la mayoría de sistemas de ML, un falso positivo y un falso negativo tienen costes similares. Aquí no:

- **Falso negativo** (no detectar una anomalía real): Potencialmente catastrófico. Es el fallo que no queremos bajo ninguna circunstancia.
- **Falso positivo** (alertar cuando no hay problema): Caro operacionalmente — paras la planta, movilizas equipos, produces pérdidas económicas — pero nadie muere.

Esta asimetría es la razón por la que el TDD fija **< 0.1% de falsos negativos en fallos CRITICAL** como requisito no negociable, aunque eso signifique aceptar más falsas alarmas.

---

### El catálogo de fallos (Fault Taxonomy)

Estos son los ocho tipos de fallo que ReactorGuard debe detectar:

**Thermocouple drift (deriva)** — Un termopar (sensor de temperatura) pierde calibración gradualmente por el daño de la radiación. La lectura se va alejando del valor real a razón de ~0.1°C por día. Es casi indetectable con métodos simples porque los valores siempre parecen "razonables".

**Stuck sensor (sensor congelado)** — El sensor se queda fijo en su último valor válido. No da NaN, no da cero, da un número perfectamente plausible... que no cambia nunca. Es el fallo más peligroso porque parece normal durante horas.

**Noise spike (pico de ruido)** — Una interferencia electromagnética (EMI) produce un valor aberrante en un único sample. Fácil de confundir con un evento real de corta duración.

**Bias fault (sesgo)** — El sensor empieza a medir con un offset constante desde un punto en el tiempo, como si alguien hubiera recalibrado mal. Difícil de detectar sin un sensor de referencia independiente.

**Intermittent fault (fallo intermitente)** — El sensor alterna entre lecturas válidas e inválidas aleatoriamente. Estadísticamente muy difícil de separar de variabilidad legítima.

**Core temperature excursion** — Evento físico real: la temperatura del núcleo sube de forma anómala. Nunca debe perderse. Severidad CRITICAL.

**Coolant flow reduction** — La bomba de refrigerante está degradándose. El síntoma temprano es una oscilación sutil en el flujo, semanas antes del fallo. Severidad HIGH.

**Void formation** — Se está formando vapor (burbujas) en el circuito primario de un PWR, que debería mantenerse siempre en fase líquida. Muy peligroso. Severidad CRITICAL.

---

## Parte II — La Arquitectura de Solución

### ¿Por qué no sirve un modelo ML convencional?

Un modelo convencional (Isolation Forest, XGBoost, etc.) aprende correlaciones estadísticas en los datos. El problema es que esas correlaciones **no conocen física**. Puede predecir que la temperatura de salida del núcleo puede ser menor que la de entrada, lo cual es físicamente imposible (el núcleo es una fuente de calor). Si el modelo acepta predicciones físicamente absurdas como válidas, perderá anomalías reales.

ReactorGuard resuelve esto con tres capas:

```
Capa 1: Pre-filtro de sensores     → ¿El sensor está funcionando bien?
Capa 2: ML con física incorporada  → ¿El estado del reactor es normal?
Capa 3: Reglas físicas duras       → ¿Se violan leyes termodinámicas?
```

Solo si las tres capas coinciden en que "todo está bien", se descarta la anomalía.

---

## Parte III — Tecnologías del Stack

### Infraestructura Cloud: GCP + Terraform + GKE

**GCP (Google Cloud Platform)** es el proveedor cloud elegido. El proyecto usa principalmente tres servicios:

- **GKE (Google Kubernetes Engine)**: Es Kubernetes gestionado. Google se encarga de mantener el plano de control del cluster. Nosotros solo gestionamos los nodos y lo que corre encima.
- **GCS (Google Cloud Storage)**: Almacenamiento de objetos (como S3 de AWS). Aquí van los datos crudos, los datos procesados, los modelos entrenados y los artefactos de MLflow.
- **Secret Manager**: Bóveda de secretos. Las credenciales del SCADA, las API keys y el JWT secret nunca van en código ni en variables de entorno en texto plano; se leen desde aquí en tiempo de ejecución.

**Terraform** es la herramienta de Infraestructura como Código (IaC). En lugar de crear recursos en la consola de GCP a mano, defines todo en archivos `.tf` y Terraform se encarga de crear, modificar o destruir los recursos para que el estado real coincida con lo que describes. Ventaja principal: si el cluster se destruye, lo recreas en minutos con `terraform apply`. El estado de Terraform se guarda en GCS para que el equipo comparta el mismo estado.

**Kubernetes (K8s)** es el orquestador de contenedores. Un contenedor Docker es una aplicación empaquetada con todas sus dependencias. Kubernetes gestiona dónde y cómo corren esos contenedores, cuántas réplicas hay, cómo se reinician si fallan, cómo se enrutan las peticiones, etc. Conceptos clave que usa ReactorGuard:

- **Namespace**: Separación lógica dentro del cluster. ReactorGuard tiene cuatro: ingestion, ml, observability, kafka-operator. Los namespaces tienen NetworkPolicies que controlan qué puede hablar con qué.
- **Deployment**: Define cuántas réplicas de un pod (contenedor) quieres y cómo actualizarlas. ReactorGuard usa 3 réplicas para el PINN server.
- **PodDisruptionBudget (PDB)**: Garantía de disponibilidad durante mantenimiento. Con `minAvailable: 2`, Kubernetes no puede matar más de 1 réplica a la vez aunque esté haciendo una actualización del nodo.
- **HPA (Horizontal Pod Autoscaler)**: Escala automáticamente el número de réplicas según carga (CPU, latencia).
- **Workload Identity**: Mecanismo de GKE por el que un pod puede asumir una identidad de IAM de GCP sin necesidad de credenciales en texto plano. El pod del PINN server puede leer de GCS porque tiene una Workload Identity asociada a una cuenta de servicio con los permisos correctos.

---

#### Los módulos Terraform de ReactorGuard (Fase 1, Semana 1)

Toda la infraestructura está organizada en módulos reutilizables bajo `infra/terraform/modules/`. Cada módulo hace exactamente una cosa. El entorno `environments/dev/main.tf` los orquesta pasando outputs de unos como inputs de otros.

##### T1.1 — Estructura del repositorio

El primer paso no crea ningún recurso en GCP. Establece el esqueleto del repositorio: árbol de directorios, `pyproject.toml` (gestión de dependencias Python con grupos opcionales por componente), `.gitignore` (excluye `*.tfstate`, `*.json` de credenciales, `__pycache__`), y `README.md`.

Un repositorio mal organizado desde el principio es deuda técnica que nunca se paga. La estructura separa `infra/` (Terraform, scripts) de `src/` (Python), `k8s/` (manifiestos Kubernetes) y `docs/` para que equipos distintos puedan trabajar en paralelo sin conflictos.

##### T1.2 — Backend de estado y configuración base

Tres piezas fundamentales antes del primer recurso real:

**Backend remoto** (`backend.tf`): Terraform necesita guardar en algún sitio qué recursos ya creó. Si ese estado fuera local, solo una persona podría aplicar cambios y perder el archivo significaría perder el control de toda la infraestructura. Con el backend en el bucket `reactorguard-terraform-state`:
- Cualquier miembro del equipo o el pipeline CI/CD puede aplicar cambios.
- Hay historial de versiones del estado (el bucket tiene versionado activado).
- El *state locking* previene que dos `terraform apply` simultáneos corrompan el estado.

**Providers con versión fija** (`providers.tf`, `versions.tf`): Se fija `~> 5.0` para el provider de Google y el archivo `.terraform.lock.hcl` guarda hashes criptográficos del binario descargado. Esto garantiza que en tu máquina, en la del compañero y en CI/CD se ejecuta exactamente el mismo código de Terraform, eliminando el "en mi máquina funciona".

**`bootstrap.ps1`**: Los recursos del backend no pueden ser creados por Terraform porque Terraform necesita ese bucket para existir *antes* de ejecutarse — un problema de huevo y gallina. El script lo resuelve creando ese bucket inicial con `gcloud` directamente. Es el único paso manual de toda la infraestructura.

##### T1.3 — VPC y Networking

La red privada virtual donde vive toda la infraestructura. Componentes:

- **VPC** (`reactorguard-vpc`): Red completamente aislada. No se usa la VPC por defecto de GCP (que tiene configuraciones permisivas heredadas de años atrás) porque en producción eso es un riesgo de seguridad.
- **Subnet** con tres rangos IP separados:
  ```
  Nodos del cluster:  10.0.1.0/24     (máx. ~254 nodos)
  Pods Kubernetes:    10.0.16.0/20    (máx. ~4.000 pods)
  Servicios K8s:      10.0.32.0/20    (máx. ~4.000 servicios internos)
  ```
  Kubernetes necesita estos tres rangos separados para su red overlay interna. Si usaras el mismo rango para todo, habría colisiones de IP.
- **Cloud NAT**: Los nodos del cluster **no tienen IPs públicas**. Nadie desde internet puede conectarse directamente a un nodo. Cloud NAT actúa como intermediario: los nodos pueden *salir* a internet (descargar imágenes Docker, llamar a APIs de GCP), pero nadie de fuera puede *entrar* directamente. Es el equivalente al router de casa en términos conceptuales.
- **Firewall rules**: Tráfico denegado por defecto. Solo se permiten las comunicaciones explícitamente declaradas.

##### T1.4 — Cluster GKE

El cerebro de la plataforma. Configuración relevante:

- **Cluster privado**: El control plane no tiene IP pública. Solo accesible desde la red interna definida en T1.3.
- **Workload Identity habilitado**: Prerequisito para T1.6. Sin esta opción activada en el cluster, los bindings de Workload Identity no funcionan.
- **Shielded nodes**: VMs con Secure Boot para resistir rootkits a nivel de sistema operativo.

La decisión de diseño más importante son los **dos node pools**:

```
pool: platform
  Máquina:   e2-standard-4 (4 vCPU, 16 GB RAM)
  Tipo:      PREEMPTIBLE (spot instances, ~70% más barato)
  Escala:    1–3 nodos
  Uso:       Kafka, Prometheus, MLflow, cargas no críticas

pool: ml-serving
  Máquina:   n1-standard-8 (8 vCPU, 30 GB RAM)
  Tipo:      ESTÁNDAR (no preemptible, siempre disponible)
  Escala:    1–5 nodos
  Taint:     ml-serving=true:NoSchedule
  Uso:       PINN server (inferencia en tiempo real)
```

Los pods de inferencia ML no pueden interrumpirse: si un reactor está en anomalía y el pod de detección muere porque GCP reclamó la VM spot, es un problema de seguridad real. Los pods de Kafka y Prometheus toleran reinicios (se recuperan solos). Separar las cargas permite optimizar coste sin sacrificar safety.

El **taint** `ml-serving=true:NoSchedule` hace que Kubernetes no coloque ningún pod en el pool ML a menos que ese pod tenga explícitamente la toleration correspondiente. Garantiza que los nodos de alta memoria no se llenen con pods de Prometheus que no los necesitan.

##### T1.5 — GCS Buckets y Storage

Cuatro buckets con roles distintos en el pipeline de datos:

```
[Sensores SCADA] ──→ reactorguard-data-raw
                           │
                           ▼
                   reactorguard-data-processed   (features engineered)
                           │
                           ▼
                   reactorguard-models           (modelos serializados)
                           ▲
                   reactorguard-mlflow           (artefactos de experimentos)
```

Todos comparten la misma configuración base:
- **Versionado**: Si se sobrescribe un modelo entrenado por error, se puede recuperar la versión anterior.
- **Uniform bucket-level access**: Sin ACLs por objeto. Más simple y más seguro.
- **Lifecycle rules**: Las versiones antiguas se borran automáticamente después de N días. Evita que el coste de almacenamiento crezca indefinidamente.
- **`cmek_key`**: Encriptación con clave propia (conecta directamente con T1.7).

Usar cuatro buckets en vez de uno permite permisos granulares por etapa: el servicio de ingestión puede escribir en `data-raw` pero no puede tocar `models`. Lifecycle policies distintas: los datos crudos pueden guardarse 30 días, los modelos indefinidamente.

##### T1.6 — IAM y Workload Identity

La identidad y los permisos de cada componente. Cuatro Google Service Accounts con **principio de least privilege**:

| Service Account | Puede hacer | No puede hacer |
|-----------------|-------------|----------------|
| `reactorguard-ingestion-sa` | Leer `data-raw`, escribir `raw`+`processed`, publicar Pub/Sub | Leer secretos, tocar modelos |
| `reactorguard-ml-sa` | Leer `processed`+`models`, leer secretos, escribir métricas | Escribir datos, borrar nada |
| `reactorguard-mlflow-sa` | Admin completo en `mlflow`+`models` | Tocar datos crudos, leer secretos SCADA |
| `reactorguard-cicd-sa` | Push Artifact Registry, deploy GKE, admin GCS | No tiene Workload Identity |

Si el pod de ingestión queda comprometido, el atacante solo puede acceder a los buckets de datos. No puede leer credenciales SCADA, no puede borrar modelos, no puede hacer deploy de código malicioso. Eso es least privilege en la práctica.

**Workload Identity** es la pieza más sofisticada del módulo. El problema que resuelve:

Sin Workload Identity (lo que hace el 80% de los proyectos):
```
# Se monta un archivo JSON con credenciales permanentes dentro del pod.
# Si alguien accede al pod, tiene las credenciales para siempre.
# Rotar credenciales requiere recrear el Secret de K8s y reiniciar pods.
kubectl create secret generic gcp-key --from-file=key.json
```

Con Workload Identity (lo que hace ReactorGuard):
```
Pod usa KSA "sensor-validator"
    ↓
GKE detecta que esa KSA está anotada con "reactorguard-ingestion-sa@..."
    ↓
El binding IAM en workload_identity.tf autoriza esa KSA a impersonar la GSA
    ↓
Pod recibe credenciales temporales (duran ~1 hora, rotadas automáticamente)
    ↓
Llama a GCS, Pub/Sub, etc. sin ningún archivo de credenciales
```

Ventajas concretas:
- Las credenciales son temporales. Si se filtran, expiran en ~1 hora.
- No hay archivos JSON en el cluster. Se elimina el vector de ataque más común en Kubernetes.
- Rotación automática. Cero trabajo operativo.

Workload Identity está en el CISA Kubernetes Hardening Guide y en el CIS GKE Security Benchmark. Usarlo desde el inicio (en vez de "cuando tengamos tiempo") es una señal de que el equipo entiende seguridad en profundidad.

##### T1.7 — Secret Manager y KMS

Dos servicios separados que juntos resuelven la gestión de secretos y el cifrado en reposo.

**Secret Manager** almacena cuatro secretos:

| Secreto | Contiene | Lo usa |
|---------|----------|--------|
| `reactorguard-scada-credentials` | `{username, password, endpoint}` del SCADA | `ingestion-sa` |
| `reactorguard-jwt-secret` | Clave de firma JWT (mínimo 32 bytes) | API REST |
| `reactorguard-mlflow-db-url` | `postgresql://user:pass@host/db` | `mlflow-sa` |
| `reactorguard-gcp-api-key` | API key de GCP para servicios sin Workload Identity | Varios |

El patrón **placeholder + `Load-Secrets.ps1`** resuelve un problema habitual en IaC: si pones las credenciales reales en el `.tf`, acaban en el repositorio. La solución es que Terraform crea la *estructura* del secreto con un valor inofensivo, y las credenciales reales se cargan manualmente con el script una sola vez:

```
terraform apply    →  crea el secreto con "REPLACE_ME"
Load-Secrets.ps1   →  carga el valor real
terraform apply    →  ignora secret_data (lifecycle ignore_changes), no lo sobreescribe
```

**KMS (Cloud Key Management Service)** crea la clave de cifrado:
- **Key Ring** `reactorguard-keyring`: Contenedor lógico de claves en `europe-west1`.
- **Crypto Key** `reactorguard-storage-key`: Clave AES-256 usada para CMEK en los 4 buckets de T1.5.

Sin KMS, los buckets usan GMEK (Google-Managed Encryption Key): GCP cifra los datos, pero GCP también tiene la clave. Con CMEK:
- Tú controlas la clave. Si revocas el acceso a la clave, los datos quedan ilegibles aunque GCP tenga las copias físicas.
- Rotación automática cada **90 días**, reduciendo la ventana de exposición si la clave se compromromete.
- `prevent_destroy = true`: Terraform se niega a borrar la clave accidentalmente. Borrar la clave equivale a perder acceso permanente a todos los datos cifrados con ella.

##### Cómo se interrelacionan los módulos

```
T1.2 (Backend)
  └── habilita → todos los módulos (sin estado remoto no hay Terraform colaborativo)

T1.3 (VPC)
  └── provee subnet + secondary ranges → T1.4 (GKE los necesita para crear el cluster)

T1.4 (GKE)
  └── habilita Workload Identity → T1.6 (los bindings WI solo funcionan con WI activo en GKE)

T1.7 (KMS)
  └── provee storage_key_id → T1.5 (buckets usan esa clave como CMEK)

T1.6 (IAM)
  └── provee SA emails → rest of platform (cualquier binding futuro referencia estos emails)
  └── provee bindings WI → Semana 2 (los pods K8s T2.2 usan esas identidades)

T1.5 (Storage)
  └── provee bucket names/URLs → T1.6 (IAM sabe en qué buckets dar permisos)
  └── provee bucket URLs → Semana 2+ (MLflow, feature pipeline, modelo serving)
```

La cadena completa, en orden de aplicación:

```
bootstrap.ps1 → terraform init → terraform apply:
  1. security (KMS key ring + key)
  2. storage  (buckets con esa KMS key)
  3. vpc      (red y subnets)
  4. gke      (cluster sobre esa VPC)
  5. iam      (service accounts + bindings + WI)
```

---

### Streaming: Apache Kafka + Strimzi

**Apache Kafka** es un broker de mensajes distribuido diseñado para alto throughput. La diferencia con una cola de mensajes convencional es que Kafka **retiene los mensajes** en disco durante un tiempo configurable (en ReactorGuard, 168 horas = 7 días). Esto permite que múltiples consumidores lean el mismo stream, y que se pueda "rebobinar" para re-procesar datos históricos, algo crítico para debugging.

Conceptos clave:

- **Topic**: Canal nombrado donde se publican mensajes. ReactorGuard tiene tres: `sensor-readings-raw`, `sensor-validated`, `anomaly-alerts`.
- **Partición**: Cada topic se divide en N particiones (12 en este caso). Cada partición es una secuencia ordenada de mensajes. El número de particiones limita el paralelismo máximo: no puedes tener más consumidores que particiones en un topic.
- **Replication factor = 3**: Cada mensaje se replica en 3 brokers. Si uno cae, los otros dos tienen la copia.
- **min.insync.replicas = 2**: Un mensaje solo se confirma como "escrito" cuando al menos 2 réplicas lo han persistido. Garantía de durabilidad.

**Strimzi** es un operador de Kubernetes para Kafka. Un operador en K8s es un patrón de extensión: defines un recurso custom (en este caso, `kind: Kafka`) y el operador se encarga de crear y gestionar todos los Deployments, Services y ConfigMaps necesarios para que Kafka corra en K8s. Sin Strimzi, tendrías que gestionar manualmente docenas de manifiestos YAML.

---

### La capa de ML: PINN, BNN y Conformal Prediction

#### Physics-Informed Neural Network (PINN)

Una red neuronal estándar aprende una función `f(x) → y` minimizando el error en los datos de entrenamiento. Una PINN hace lo mismo, pero añade **términos de penalización en la función de pérdida** que castigan al modelo cuando sus predicciones violan ecuaciones físicas conocidas.

En ReactorGuard, la PINN aprende a predecir el estado esperado de todos los sensores dado el estado actual. Si la predicción viola la conservación de energía (Q_generado ≠ Q_extraído), la pérdida sube. El modelo aprende que esas predicciones no son aceptables.

La función de pérdida es:
```
Loss_total = Loss_datos + λ_energía · Loss_energía + λ_masa · Loss_masa
```

Donde:
- `Loss_datos`: Error cuadrático entre predicción y lectura real (MSE estándar).
- `Loss_energía`: Penaliza cuando Q_generado ≠ Q_extraído por el coolant.
- `Loss_masa`: Penaliza cuando el flujo de entrada ≠ flujo de salida en estado estacionario.
- `λ`: Hiperparámetros que controlan cuánto pesa la física vs. los datos.

El **residual de la PINN** (diferencia entre lo que predice y lo que mide el sensor) es la señal de anomalía: si la PINN predice 310°C y el sensor marca 320°C, hay un residual de 10°C que puede indicar un evento real o un fallo del sensor.

#### Bayesian Neural Network (BNN)

Una red neuronal convencional da un único número como predicción: "la temperatura será 312.4°C". Una BNN da una **distribución de probabilidad**: "la temperatura estará entre 309 y 315°C con 95% de probabilidad". 

Técnicamente, la diferencia es que en una BNN los **pesos de la red son distribuciones** (gaussianas, normalmente), no valores fijos. Para hacer una predicción, se samplea múltiples veces de esas distribuciones y se obtiene un intervalo. El ancho del intervalo mide la incertidumbre del modelo.

Esto es importante para ReactorGuard porque un operador necesita saber no solo "esto es una anomalía con score 0.73", sino "este score tiene un intervalo de confianza de ±0.02" (alta certeza) vs "±0.40" (el modelo no está seguro). En el segundo caso, conviene tener más cautela.

#### Conformal Prediction (con MAPIE)

El problema con la BNN es que sus intervalos de confianza dependen de asunciones sobre la distribución de los datos. Si los datos reales no siguen exactamente esa distribución, los intervalos pueden estar mal calibrados.

**Conformal Prediction** es un marco matemático que provee intervalos de predicción con **garantía de cobertura libre de distribución**: sin importar qué distribución tengan los datos, si dices "95% de cobertura", el intervalo verdaderamente contendrá el valor real el 95% de las veces.

**MAPIE** (*Model Agnostic Prediction Interval Estimator*) es la librería Python que implementa esto. Funciona así: entrenas cualquier modelo, luego pasas un conjunto de calibración separado para ajustar el ancho de los intervalos hasta que la cobertura real sea la prometida.

**ECE (Expected Calibration Error)** mide qué tan bien calibrado está un modelo de probabilidad. Si dices "95% de confianza" en 1000 predicciones, idealmente el valor real debería estar dentro del intervalo en ~950 de ellas. Si solo ocurre en 800, el ECE es alto (modelo sobreconfiado). ReactorGuard requiere ECE < 0.05.

---

### Feature Engineering: Procesado de Series Temporales

Los sensores producen series temporales: secuencias de valores con timestamps. Los modelos ML no pueden "ver" el tiempo directamente, así que transformamos esas series en **features** (características numéricas) que capturan información relevante:

- **rolling_mean / rolling_std**: Media y desviación estándar en ventana deslizante (60s, 5min, 1h). Capturan el "estado base" y la "volatilidad" de cada sensor.
- **rate_of_change (dX/dt)**: Derivada discreta del sensor. Un stuck sensor tendrá dX/dt ≈ 0 indefinidamente. Un noise spike tendrá un dX/dt gigantesco en un solo step.
- **Kalman residual**: El **filtro de Kalman** es un algoritmo que, dado un modelo del sistema (cómo evoluciona el estado), combina las predicciones del modelo con las medidas ruidosas para dar la mejor estimación del estado real. El residual (diferencia entre la medida y la predicción del Kalman) detecta desviaciones graduales como la deriva de un sensor.
- **cross_correlation**: Correlación de Pearson entre pares de sensores en ventana deslizante, comparada con la correlación de baseline (estado normal). Si dos sensores que normalmente están correlacionados dejan de estarlo, uno de ellos probablemente está fallando.
- **stuck_score**: Varianza de rolling muy pequeña indica que el sensor lleva N muestras sin moverse — síntoma directo de stuck sensor.

---

### Feast — Feature Store

Un **feature store** es una capa de infraestructura que gestiona features de ML en dos modos:

- **Offline (batch)**: Recuperar el valor de una feature en un timestamp histórico para entrenar modelos. Por ejemplo, "dame el rolling_std_60s del sensor TC-CORE-12 el 15 de enero a las 14:32".
- **Online (tiempo real)**: Recuperar el valor actual de una feature con baja latencia para hacer inferencia. Por ejemplo, "dame las features actuales de todos los sensores del reactor".

Sin un feature store, hay duplicación de lógica: el código de entrenamiento calcula las features de una manera, y el código de producción las calcula de otra. Los bugs de inconsistencia entre entrenamiento y producción son una de las causas más comunes de degradación de modelos en producción. **Feast** resuelve esto unificando ambos accesos bajo la misma definición.

---

### MLflow + DVC — Experimentos y Versiones de Datos

**MLflow** es la plataforma de seguimiento de experimentos ML. Cada vez que entrenas el PINN con distintos hiperparámetros (distinto λ_energía, distinto learning rate), MLflow registra automáticamente: los parámetros usados, las métricas resultantes (ECE, recall, physics satisfaction) y los artefactos (el modelo serializado). Así puedes comparar 50 runs y ver qué configuración fue mejor.

**DVC (Data Version Control)** hace lo mismo pero para **datos y pipelines**. Git versiona código, DVC versiona datasets y los pasos del pipeline. Define un DAG (grafo acíclico dirigido) de transformaciones: `simulate → featurize → train`. Si cambias el parámetro `fault_injection_rate` en `params.yaml`, DVC sabe exactamente qué steps del pipeline hay que re-ejecutar y cuáles no. Los datos reales se almacenan en GCS; DVC solo guarda los hashes y metadatos en Git.

Esto es especialmente importante en ReactorGuard porque los modelos dependen de parámetros de simulación. Si cambias la tasa de inyección de fallos, el modelo entrenado ya no es comparable con el anterior. DVC fuerza a que esa relación sea explícita y trazable.

---

### OpenMC + PyDy — Simulación Física

**OpenMC** es un código de Monte Carlo para simulación de neutrónica. Resuelve la ecuación de transporte de neutrones: modela cómo los neutrones se mueven, se absorben y producen fisión dentro del núcleo. Es el mismo tipo de herramienta que usan los ingenieros nucleares reales. En ReactorGuard se usa para generar datos sintéticos con física real que sirven de ground truth para entrenar los modelos.

**PyDy** (Python Dynamics) es una librería para modelar sistemas dinámicos descritos por ODEs (ecuaciones diferenciales ordinarias). En ReactorGuard modela el circuito térmico del reactor: cómo la temperatura del coolant evoluciona en función del calor generado por la fisión y las propiedades de los intercambiadores. Acoplado con OpenMC, da una simulación completa neutrónica + térmica.

Por qué necesitamos simulación: los fallos **CRITICAL** son eventos raros en plantas reales. No hay suficientes ejemplos etiquetados de "esto fue una excursión térmica real" en datos públicos. La simulación permite generar tantos ejemplos como se necesiten, con los labels perfectamente conocidos.

---

### FastAPI — API de Inferencia

**FastAPI** es un framework web Python de alto rendimiento para construir APIs REST. Su ventaja principal sobre alternativas como Flask es que usa **async/await** nativo de Python, lo que permite manejar muchas peticiones concurrentes sin bloquear hilos. Para el target de < 50ms p99 bajo carga concurrente, esto es relevante.

FastAPI genera automáticamente documentación interactiva (Swagger UI) a partir de los type hints y modelos Pydantic, lo que es útil para que el equipo de operaciones entienda y pruebe la API sin leer código.

**Uvicorn** es el servidor ASGI (Asynchronous Server Gateway Interface) que ejecuta la aplicación FastAPI. Es el equivalente a Gunicorn para aplicaciones síncronas, pero diseñado para código asíncrono.

---

### SHAP — Explicabilidad

**SHAP (SHapley Additive exPlanations)** es un método para explicar predicciones de modelos ML. Responde a la pregunta: "¿cuánto contribuyó cada feature al score de anomalía de esta predicción?". Los SHAP values tienen fundamento matemático sólido (teoría de juegos cooperativos) y son independientes del tipo de modelo.

En ReactorGuard, cuando el sistema detecta una anomalía, el endpoint `/v1/explain` devuelve las top 3 features que más contribuyeron. Por ejemplo: "el score de anomalía de 0.87 está impulsado principalmente por rolling_std_60s (+0.31), kalman_residual (+0.28) y rate_of_change (+0.19)". Esto permite al operador entender por qué el sistema alertó, en lugar de confiar ciegamente en un número.

Esto también es un requisito regulatorio en entornos nucleares: los sistemas de soporte a la decisión deben ser **auditables** y **explicables**.

---

### CI/CD + Security: GitHub Actions + Trivy + Semgrep

**GitHub Actions** es la plataforma de CI/CD integrada en GitHub. Cada push o pull request dispara un workflow que ejecuta automáticamente: tests, análisis de seguridad, build de imagen Docker, y despliegue.

**Trivy** es un escáner de vulnerabilidades para imágenes Docker y código. Analiza las dependencias del sistema operativo y las librerías Python en busca de CVEs (vulnerabilidades conocidas públicamente) catalogadas en bases de datos como NVD. El pipeline de ReactorGuard bloquea el deploy si hay alguna vulnerabilidad CRITICAL.

**Semgrep** es una herramienta de análisis estático (SAST). Busca patrones de código inseguro: credenciales hardcodeadas, uso de funciones peligrosas, inyecciones de código, etc. Es como un linter pero enfocado en seguridad.

**Gitleaks** escanea el historial de Git buscando secretos que hayan podido colarse accidentalmente en el código: API keys, contraseñas, tokens.

**tfsec** hace lo mismo con código Terraform: busca configuraciones de infraestructura inseguras (buckets GCS públicos, reglas de firewall demasiado permisivas, etc.).

**Binary Authorization** es una política de GKE que solo permite ejecutar imágenes Docker que hayan sido firmadas digitalmente por el pipeline de CI/CD. Previene que alguien despliegue una imagen que no ha pasado por los controles de seguridad.

---

### Observability: Prometheus + Grafana + Jaeger

**Prometheus** es un sistema de monitorización basado en pull: cada 15 segundos, Prometheus hace scraping de los endpoints `/metrics` de todos los servicios y almacena las métricas en su base de datos de series temporales. Las métricas están en formato clave-valor con labels: `reactorguard_prediction_latency_seconds{quantile="0.99"} 0.042`.

**Grafana** es la capa de visualización sobre Prometheus. Permite crear dashboards con gráficas, tablas y alertas. ReactorGuard tiene cuatro dashboards: Reactor Overview, ML Model Health, Sensor Analytics y Operations.

**Jaeger** implementa **tracing distribuido**. En un sistema de microservicios, una petición a `/v1/analyze` pasa por varios servicios: Kafka consumer → Feature pipeline → PINN → BNN → Fusion → Alert. Jaeger registra el tiempo que tarda cada hop, lo que permite identificar exactamente dónde se está perdiendo el tiempo cuando el p99 sube. Sin tracing, solo sabes que "la latencia total fue 80ms", pero no sabes si el problema está en la feature extraction o en la inferencia del PINN.

**OpenTelemetry** es el estándar abierto de instrumentación. En lugar de usar la librería de Jaeger directamente en el código, se usa la API de OpenTelemetry, que es agnóstica al backend. Si mañana se cambia Jaeger por Tempo (de Grafana), no hay que tocar el código de aplicación.

---

## Parte IV — Conceptos Transversales

### SLO / SLI / SLA

- **SLI (Service Level Indicator)**: La métrica que mides. Por ejemplo, "latencia p99 de /v1/analyze".
- **SLO (Service Level Objective)**: El objetivo para esa métrica. "El SLI debe ser < 50ms el 99.9% del tiempo".
- **SLA (Service Level Agreement)**: El contrato formal con consecuencias si el SLO no se cumple.

ReactorGuard tiene un SLO de 99.99% de disponibilidad, lo que equivale a < 52.6 minutos de downtime al año. El **error budget** es el margen que te queda: si en lo que va de año ya has consumido 30 minutos de downtime, te quedan 22.6 minutos de margen antes de violar el SLO.

### p99, p50, p95 (percentiles de latencia)

Cuando se habla de latencia, la media es engañosa. Si el 99% de las peticiones tardan 10ms y el 1% tarda 5 segundos, la media puede ser 60ms — pero ese 1% representa miles de peticiones al día en un sistema de alta frecuencia.

- **p50 (mediana)**: El 50% de las peticiones tarda menos que este valor.
- **p95**: El 95% tarda menos. Captura la experiencia de la mayoría.
- **p99**: El 99% tarda menos. Captura los "outliers frecuentes".
- **p99.9**: Solo el 0.1% tarda más. Para sistemas safety-critical.

ReactorGuard tiene target de **p99 < 50ms** para el endpoint de análisis completo. Eso significa que en 99 de cada 100 peticiones, la respuesta llega en menos de 50ms.

### Canary Deployment

En lugar de actualizar todos los pods de golpe (lo que podría introducir un bug crítico en producción), un canary deployment envía solo el **10% del tráfico** a la nueva versión mientras el 90% sigue en la versión estable. Si las métricas de la versión nueva son buenas después de un período de observación, se completa el rollout. Si hay problemas, se hace rollback instantáneo afectando solo al 10% del tráfico.

### Kustomize

**Kustomize** es una herramienta para gestionar variaciones de manifiestos YAML de Kubernetes sin duplicarlos. Defines una configuración base (`k8s/base/`) y luego **overlays** por entorno (`dev/`, `staging/`, `prod/`) que solo describen las diferencias. Por ejemplo, en dev tienes 1 réplica del PINN server y en prod tienes 3. Sin Kustomize, tendrías tres copias casi idénticas del mismo YAML.

### Workload Identity

En GKE, cada pod puede tener asociada una **Kubernetes Service Account**. Con Workload Identity, esa service account de K8s se vincula a una **Google Service Account** de IAM. El pod puede entonces llamar a APIs de GCP (leer de GCS, acceder a Secret Manager) con los permisos de esa Google Service Account, sin que haya credenciales en variables de entorno ni en el código.

---

## Glosario Rápido

| Término | Definición breve |
|---------|-----------------|
| **PWR** | Pressurized Water Reactor. Tipo de reactor nuclear de agua a presión. |
| **SCADA** | Supervisory Control and Data Acquisition. Sistema de control industrial que agrega los datos de sensores de la planta. |
| **PINN** | Physics-Informed Neural Network. Red neuronal cuya función de pérdida incluye restricciones de ecuaciones físicas. |
| **BNN** | Bayesian Neural Network. Red neuronal donde los pesos son distribuciones de probabilidad, no valores fijos. |
| **ECE** | Expected Calibration Error. Mide si la confianza declarada de un modelo coincide con su precisión real. |
| **UQ** | Uncertainty Quantification. Representación explícita de la incertidumbre en las predicciones. |
| **Conformal Prediction** | Método de ML que garantiza matemáticamente que los intervalos de predicción tienen la cobertura declarada. |
| **MAPIE** | Model Agnostic Prediction Interval Estimator. Librería Python para Conformal Prediction. |
| **SHAP** | SHapley Additive exPlanations. Método para explicar qué features impulsaron cada predicción. |
| **DVC** | Data Version Control. Control de versiones para datasets y pipelines de ML. |
| **TEP** | Tennessee Eastman Process. Dataset benchmark estándar para detección de anomalías industriales. |
| **EMI** | Electromagnetic Interference. Interferencia electromagnética que puede causar ruido en sensores. |
| **Kalman filter** | Algoritmo que combina predicción del modelo con medidas ruidosas para estimar el estado real del sistema. |
| **Feast** | Feature store open-source para servir features a modelos ML tanto en entrenamiento como en producción. |
| **Strimzi** | Operador de Kubernetes que gestiona clusters de Apache Kafka en K8s. |
| **GKE** | Google Kubernetes Engine. Kubernetes gestionado por GCP. |
| **PDB** | PodDisruptionBudget. Garantía de K8s de que un número mínimo de réplicas siempre estará disponible. |
| **HPA** | Horizontal Pod Autoscaler. Escala automáticamente el número de réplicas según carga. |
| **IaC** | Infrastructure as Code. Gestión de infraestructura mediante código versionado (Terraform). |
| **CVE** | Common Vulnerabilities and Exposures. Identificador estándar de vulnerabilidades de seguridad conocidas. |
| **SAST** | Static Application Security Testing. Análisis de seguridad del código fuente sin ejecutarlo. |
| **CMEK** | Customer-Managed Encryption Key. Clave de cifrado que gestiona el cliente, no GCP. |
| **IAP** | Identity-Aware Proxy. Proxy de GCP que autentica usuarios antes de llegar a la aplicación. |
| **SLO** | Service Level Objective. Objetivo de nivel de servicio (ej. 99.99% disponibilidad). |
| **Monte Carlo** | Método de simulación basado en sampling aleatorio repetido para resolver problemas complejos. |
| **ODE** | Ordinary Differential Equation. Ecuación diferencial que modela la evolución temporal de un sistema. |
| **DAG** | Directed Acyclic Graph. Grafo de dependencias sin ciclos. Lo usa DVC para el pipeline de datos. |
| **ASGI** | Asynchronous Server Gateway Interface. Estándar Python para servidores web asíncronos. |

---

*ReactorGuard Guía Técnica · v1.0 · Documento de referencia personal*
