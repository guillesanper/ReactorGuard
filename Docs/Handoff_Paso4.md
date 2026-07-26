# Prompt de continuacion — ReactorGuard Paso 4 (infraestructura: Kafka, GCP, Feast, Grafana)

Trabajo en ReactorGuard (`c:\Users\guill\Desktop\proyectos\reactorGuard\ReactorGuard`),
una plataforma de deteccion de anomalias en reactores nucleares. Windows 11, PowerShell 7+.
La documentacion de fases esta en `Docs/` (Plan_Fases, Fase1_Prompts, Fase2_Prompts,
Progreso, Guia_Tecnica). El plan completo esta en
`C:\Users\guill\.claude\plans\compiled-inventing-scroll.md` — leelo primero.

Handoffs anteriores, vigentes como contexto heredado y que NO hay que re-investigar:
`Docs/Handoff_Paso3.md` (Bloque 1) y `Docs/Handoff_Paso3_Bloque3.md` (Bloques 2 y 3).
Este documento continua a partir de ellos.

**Los Pasos 0, 1, 2 y 3 estan CERRADOS y verificados.** Todo lo local funciona sin GCP.
Este Paso 4 es el que despliega infraestructura, y por eso **requiere intervencion humana
real**: hay que autenticarse en GCP, aplicar Terraform con coste economico asociado, y
operar un cluster. Las secciones 6 y 7 son la guia paso a paso.

---

## 1. Linea base medida al terminar el Paso 3 — punto de partida

```
pytest tests/unit tests/safety --cov --cov-fail-under=70
  545 passed, 3 skipped        cobertura TOTAL 98,79%
ruff check .                   All checks passed
mypy ml/ api/ data/            no issues found in 37 source files

.\infra\scripts\Invoke-Pipeline.ps1     download_tep -> explore_tep -> adapt_tep
                                        -> featurize -> split   (5 stages)
.\infra\scripts\Invoke-Pipeline.ps1     2a vez: los cinco "didn't change, skipping"

data/processed/tep         22 particiones   550.160 lecturas
data/processed/features    22 particiones   550.160 filas x 31 columnas
data/processed/splits      train 385.112 / val 82.524 / test 82.524 = 550.160
```

Criterios de Fase 2, estado real:

| # | Criterio | Estado |
|---|----------|--------|
| 1 | Throughput Kafka > 50.000 msg/s sostenidos | **PENDIENTE — es este Paso 4** |
| 2 | Stuck sensor precision > 95% | **CUMPLIDO: 1,0000** (`tests/results/validator_metrics_tep.json`) |
| 3 | Feature pipeline latency < 100 ms por ventana bajo carga | **PENDIENTE — sin medir bajo carga** |
| 4 | `dvc repro` reproducible desde cero | **CUMPLIDO y demostrado** (ver seccion 3.C) |

**NADA ESTA COMMITEADO.** El arbol acumula tres sesiones de trabajo. Ver seccion 8.

---

## 2. Estado del codigo — que existe y en que condicion

### Funcionando y con tests (NO lo toques salvo que la tarea lo pida)

| Modulo | Cobertura | Que hace |
|---|---|---|
| `data/schemas/sensor_reading.py` | 100% | Contrato canonico. Schema anidado del TDD 4.2 |
| `data/schemas/sensor_spans.py` | 100% | Span calibrado + sobre de alarma por sensor |
| `data/generators/tep_*.py` | 93-100% | Descarga, exploracion y adaptacion del TEP |
| `data/generators/derive_sensor_spans.py` | 97% | Genera `configs/sensor_spans.yaml` (a mano) |
| `data/generators/train_val_test_split.py` | 100% | Stage `split` |
| `data/validation/sensor_validator.py` | 99% | Los 5 detectores + orquestador |
| `data/validation/sensor_fault.py` | 100% | `FaultType`, `Severity`, `SensorFault` |
| `data/validation/fault_injector.py` | 100% | Inyeccion sintetica con semilla fija (T3.6) |
| `ml/features/kalman.py` | 100% | `OnlineKalmanFilter`, `KalmanFilterBank` |
| `ml/features/feature_params.py` | 100% | Seccion `features:` de params.yaml |
| `ml/features/pipeline.py` | 99% | Los 7 grupos de features del TDD 5.5 |
| `ml/features/batch_featurizer.py` | 94% | Stage `featurize` |

### Andamiaje que NO se toca en este paso

- `api/` — 2 de 4 endpoints devuelven 501 (`predict.py:27`, `explain.py:29`). Fase 6.
- `ml/models/pinn.py` — `physics_residual` NO implementa la fisica que documenta. Fase 4 (T6.1).
- `ml/training/train.py` — sin loop de entrenamiento. Fase 4.
- `ml/serving/predictor.py` — declara MAPIE y nunca lo ajusta. Fase 4.
- `data/generators/reactor_simulator.py` — ROTO contra el schema canonico. Excluido de mypy
  y en el `omit` de cobertura. Su suite `tests/safety/test_safety_constraints.py` entera
  esta en `pytest.mark.skip("Fase 3")`. **NO los toques.**

### Retirado

- `ml/features/extractor.py` — eliminado con `git rm` en T4.2. Superado por `pipeline.py`.
  Recuperable con `git show 3801d0b:ml/features/extractor.py`.
- `data/generators/prepare_tep.ps1` — sustituido por `Invoke-Pipeline.ps1`.

---

## 3. Hallazgos medidos que condicionan el Paso 4 — NO los re-investigues

### A. La cadencia del TEP son 180 s, y eso ha roto ya TRES defaults distintos

Es el patron de fallo recurrente del proyecto. Cada componente escrito asumiendo el sondeo
SCADA de 1 s ha fallado en silencio al llegar a los 180 s del TEP:

1. **`process_noise=0.1` del Kalman.** Q[0,0] escala con dt^3: a 180 s se dispara a 194.400,
   la sigma de innovacion sube a 602 y un salto de 25 unidades queda en 0,042 sigmas.
   INVISIBLE. Corregido a `1e-6` en el detector y en el featurizer.
2. **Umbral de deriva de 1,0 anchos de sobre/hora.** Disparaba en **11.969 de 25.012**
   muestras de operacion normal, el **48%**. Recalibrado a **11,0** (ver B).
3. **Ventanas de features en segundos** (60 s, 5 min, 1 h del TDD 5.5). A 180 s la de 60 s
   no contiene NI UNA muestra. Reescritas en MUESTRAS: `window_samples: [10, 20, 60]`.

**Consecuencia para el Paso 4: cualquier umbral, ventana o timeout nuevo se calibra a 180 s
y se mide, no se hereda de un documento.** El TDD y los prompts de Fase 2 estan escritos
para 1 s en varios sitios.

### B. Limitacion medida del detector de deriva (declarada, no pendiente)

Ver `DRIFT_DETECTION_FLOOR_NOTE` en `data/validation/sensor_validator.py`. La banda util es
solo **10-18 anchos de sobre por hora**:

- **Suelo 10,13**: es el movimiento propio del proceso en d00, no ruido del filtro. No baja
  con q: medido, satura en 9,13 para q <= 1e-8 mientras los falsos positivos del residual
  suben del 2,1% al 6,4%.
- **Techo 18**: por encima ya es violacion de tasa y la reporta antes `RateOfChangeDetector`.

Una deriva de instrumento LENTA, que es el caso realista, **no la detecta este mecanismo a
esta cadencia**. Haria falta un estimador de linea base larga. Esta documentado, no es deuda
oculta.

### C. `adapt_tep` era NO REPRODUCIBLE y se arreglo

`reading_id` usaba `uuid.uuid4()`: cada ejecucion producia 550.160 identificadores nuevos,
dos personas obtenian parquets con hash distinto y la cache de DVC no se podia compartir.
Sustituido por `uuid5(namespace, plant_id|fault_type|sensor_id|timestamp)`.

**El `fault_type` en la clave NO es redundante**: los 22 ficheros del TEP reinician el reloj
en `start_time`, asi que sin el la muestra 0 de XMEAS-01 en d00 y en d01 colisionarian.
Verificado: 550.160 identificadores, 550.160 unicos, y la huella sha256 sale identica tras
forzar una re-adaptacion desde cero.

### D. Los positivos reales de fallo de INSTRUMENTO en el TEP son casi inexistentes

- Solo **XMV-04 en d21** esta realmente congelado (480 muestras, `nunique=1`, `std=0`).
  480 pares positivos de 550.160: el **0,087%**.
- **El plan original decia "d14/d15 positivos" y ESO ES FALSO** para un detector de stuck:
  d14 es sticking de valvula (respuesta lenta, la lectura sigue variando) y d15 es
  indetectable. El run-length maximo de ambos es 5, el mismo que el de d00.
- Suelo empirico: tiradas de 5-6 valores repetidos aparecen en operacion normal
  (cuantizacion del analizador). La ventana de stuck debe ser > 6 (default 8).
- Por eso T3.6 usa **inyeccion sintetica** con semilla fija sobre d00, y reserva XMV-04/d21
  como validacion cruzada del unico positivo real.

### E. Falsos positivos del residual de Kalman sobre datos reales

**3,59%** sobre sensores intactos de d00, a 3 sigmas. Un residual gaussiano daria 0,27%: los
residuales del proceso real tienen colas mas pesadas que el modelo cinematico. Por eso
`kalman_anomaly` es senal de apoyo y **no lleva gate**. Tenlo en cuenta al dimensionar las
alertas de `anomaly-alerts` en T3.5: a 550.160 lecturas, un 3,59% son ~19.700 alertas.

---

## 4. Decisiones de diseno tomadas y ya implementadas — NO las re-discutas

1. **`quality` es ortogonal a `fault_type`.** El del TEP describe una perturbacion del
   PROCESO; `quality` describe la fiabilidad del INSTRUMENTO. El adaptador emite `GOOD` para
   los 22 ficheros. Confundirlos convierte cualquier evaluacion en tautologia.
2. **Spans con DOS rangos.** `[min, max]` = span calibrado del ADC (margen 200%);
   `[alarm_min, alarm_max]` = sobre de operacion normal (margen 20%). El detector de rango
   usa `in_alarm_envelope`, NUNCA `contains`.
3. **`features.sensor_selection` no enumera tags.** Los resuelve de los datos y los contrasta
   contra `expected_sensor_count`, para que una deriva de schema falle en voz alta.
4. **T3.6 con inyeccion sintetica de semilla fija** (ver 3.D).
5. **El validador ESCRIBE `quality`, nunca la lee.** `ValidationResult.enriched_reading`
   lleva SUSPECT si hay fallos no criticos y BAD si hay alguno CRITICAL (BAD hace
   `is_usable=False`). El reading original NUNCA se muta.
6. **Umbrales dependientes de escala en fracciones del ancho del sobre**, no en unidades de
   ingenieria: un umbral absoluto no significa lo mismo en 52 canales heterogeneos.
7. **`SensorValidator.get_metrics()` devuelve un dict plano, NO registra en Prometheus.**
   Un registro global colisiona entre tests y ata el modulo a un backend. **T3.5 es donde
   esas cifras se convierten en metricas** — es tuyo ahora.
8. **El pivot largo->ancho vive en `batch_featurizer`, por particion.** La frontera de
   `fault_type` es la unidad de independencia: cada fichero reinicia el reloj, y rodar una
   ventana a traves de ella contaminaria una clase con la historia de otra. `pipeline.py`
   queda agnostico al particionado, que es lo que le permite servir tambien al camino en
   linea de T3.5.
9. **El featurizer NO llama al validador.** Acoplarlos haria que recalibrar un umbral
   invalidase todas las features. Quien escribe `quality` es el consumidor de T3.5.
10. **La salida de features es LARGA** (una fila por `timestep, sensor_id`) y los nombres de
    feature NO llevan el tag dentro. Es ademas lo que Feast espera: su `sensor_entity` tiene
    `join_key=sensor_id`. **Relevante directo para T4.3.**
11. **El split es temporal DENTRO de cada `fault_type`.** Satisface estratificacion y ausencia
    de fuga a la vez. `temporal_order_holds` en las metricas lo verifica clase por clase:
    en AGREGADO la propiedad no se cumple ni puede (d00 tiene 500 timesteps y los ficheros de
    fallo 480, asi que `train_max_timestep`=349 > `val_min_timestep`=336) y comprobarlo en
    agregado daria un falso negativo.

---

## 5. Aviso critico ANTES de tocar nada: incoherencias conocidas que bloquean el despliegue

### 5.1 EL PROJECT ID ESTA MAL EN 8 SITIOS — arreglalo lo primero

El proyecto GCP real es **`sentinel-platform-485714`**. Terraform lo tiene bien
(`infra/terraform/environments/dev/main.tf:11`, parametrizado via `var.project_id`), pero
estos ficheros referencian `reactorguard-platform`, **que no existe**:

```
k8s/base/ingestion/serviceaccount.yaml:17   (comentario)
k8s/base/ingestion/serviceaccount.yaml:32   iam.gke.io/gcp-service-account
k8s/base/ml/serviceaccount.yaml:35          iam.gke.io/gcp-service-account
k8s/base/ml/serviceaccount.yaml:53          iam.gke.io/gcp-service-account
infra/scripts/Load-Secrets.ps1:49           $ProjectId default
infra/scripts/Test-WorkloadIdentity.ps1:34  $ProjectId default
infra/scripts/Verify-Infra.ps1:19,37        $ProjectId default y docstring
```

**Consecuencia si no se arregla:** las anotaciones de Workload Identity apuntan a cuentas de
servicio inexistentes, los pods no obtienen credenciales y todo lo que hable con GCS o Secret
Manager falla con un 403 dificil de diagnosticar. Terraform, en cambio, aplicaria sin quejarse.

Los nombres correctos que Terraform SI crea (`infra/terraform/modules/iam/main.tf`):

```
reactorguard-ingestion-sa@sentinel-platform-485714.iam.gserviceaccount.com
reactorguard-ml-sa@sentinel-platform-485714.iam.gserviceaccount.com
reactorguard-mlflow-sa@sentinel-platform-485714.iam.gserviceaccount.com
reactorguard-cicd-sa@sentinel-platform-485714.iam.gserviceaccount.com
```

Los bindings de Workload Identity ya estan bien porque usan `${var.project_id}`
(`infra/terraform/modules/iam/workload_identity.tf:29,36,43`).

### 5.2 Aviso heredado del Paso 1

Si ves errores de parseo TOML inexplicables que rompen ruff/mypy/pytest a la vez, **mira la
cabecera de `pyproject.toml` antes de depurar otra cosa** — se corrompio una vez con texto de
prompt incrustado.

### 5.3 `dvc repro` a pelo NO funciona

`dvc.exe` no activa el venv, asi que `python` dentro de un `cmd:` de un stage resuelve al
Python del sistema y falla con `ModuleNotFoundError`. **Peor: DVC borra los `outs` ANTES de
ejecutar el stage, asi que un fallo asi te deja sin `data/processed/tep`.** (Paso ya una vez;
se recupera regenerando, pero son minutos.)

**Usa siempre `.\infra\scripts\Invoke-Pipeline.ps1`**, que antepone el venv al PATH.

### 5.4 La regla `*.json` de `.gitignore`

Es un comodin deliberado contra claves de GCP descargadas. **Cada directorio que versione
JSON de codigo fuente necesita su excepcion explicita.** Ya existen dos:

```
!observability/**/*.json    # dashboards de Grafana (T4.6)
!metrics/*.json             # metricas de DVC
```

Si anades JSON fuente en otro sitio y no pones excepcion, **desaparecera del commit en
silencio**.

---

## 6. GUIA PASO A PASO — que tienes que hacer TU (el humano) y que hago yo

Esta es la parte que requiere intervencion humana. Sigue el orden; cada bloque depende del
anterior. Los bloques marcados **[TU]** los ejecutas tu porque necesitan credenciales,
aprobacion de gasto o decisiones de cuenta. Los marcados **[IA]** los hago yo.

### Bloque 0 — Antes de nada, saneamiento **[IA, luego TU verificas]**

1. Yo corrijo los 8 sitios del project ID (seccion 5.1).
2. Yo verifico que la suite sigue verde (`545 passed`).
3. **[TU]** revisas el diff antes de continuar.

### Bloque 1 — Prerrequisitos de tu maquina **[TU]**

Comprueba que tienes instalado y en el PATH. Yo no puedo instalarlos por ti:

```powershell
gcloud version          # Google Cloud SDK
terraform version       # >= 1.5
kubectl version --client
helm version             # necesario para el operador de Kafka (Strimzi)
```

Si falta algo:
- gcloud: https://cloud.google.com/sdk/docs/install
- terraform: `winget install HashiCorp.Terraform`
- kubectl: `gcloud components install kubectl`
- helm: `winget install Helm.Helm`

### Bloque 2 — Autenticacion y proyecto GCP **[TU]**

```powershell
gcloud auth login
gcloud auth application-default login       # Terraform usa ESTAS credenciales
gcloud config set project sentinel-platform-485714
gcloud config get-value project             # debe imprimir sentinel-platform-485714
```

Habilita las APIs necesarias (tarda unos minutos):

```powershell
gcloud services enable `
  container.googleapis.com `
  compute.googleapis.com `
  storage.googleapis.com `
  secretmanager.googleapis.com `
  cloudkms.googleapis.com `
  iam.googleapis.com `
  artifactregistry.googleapis.com
```

**Comprueba que hay facturacion activa.** Sin ella `container.googleapis.com` falla:

```powershell
gcloud billing projects describe sentinel-platform-485714
```

### Bloque 3 — Bucket del estado de Terraform **[TU]**

`infra/terraform/environments/dev/backend.tf` espera el bucket
`reactorguard-terraform-state`, que **tiene que existir ANTES del primer `terraform init`**.
Es el problema del huevo y la gallina del estado remoto.

```powershell
gsutil ls -b gs://reactorguard-terraform-state 2>$null
# Si no existe:
gsutil mb -p sentinel-platform-485714 -l europe-southwest1 gs://reactorguard-terraform-state
gsutil versioning set on gs://reactorguard-terraform-state
```

El versionado NO es opcional: es lo que te permite recuperar el estado si un `apply` lo
corrompe.

Existe `infra/scripts/bootstrap.ps1`; **pideme que lo lea antes de ejecutarlo** para
comprobar si ya hace esto y no duplicarlo.

### Bloque 4 — AVISO DE COSTE, y luego `terraform apply` **[TU decide, IA acompana]**

**LEE ESTO ANTES DE APLICAR.** Un cluster GKE con los nodos que declara el modulo, mas
balanceador, mas KMS, **cuesta dinero real por hora**, este o no en uso. Si lo dejas
encendido un mes te llega una factura de tres cifras en euros.

```powershell
cd infra\terraform\environments\dev
terraform init
terraform validate
terraform plan -out=tfplan          # LEE EL PLAN. Cuenta cuantos recursos crea.
```

**Parate aqui y dime que dice el plan.** Yo lo reviso contigo antes de aplicar. Cuando lo
apruebes:

```powershell
terraform apply tfplan
```

Tarda **15-25 minutos**, casi todo creando el cluster GKE.

> **Compromiso de coste:** cuando termines la sesion, ejecuta
> `.\infra\scripts\Remove-DevInfra.ps1` (pideme que lo lea antes, para confirmar que destruye
> lo que crees) o `terraform destroy`. El estado en GCS sobrevive, asi que puedes recrear todo
> mañana con otro `apply`.

### Bloque 5 — Credenciales del cluster y verificacion **[TU ejecuta, IA diagnostica]**

```powershell
gcloud container clusters get-credentials <NOMBRE_CLUSTER> --region europe-southwest1
kubectl get nodes                          # deben aparecer Ready
.\infra\scripts\Verify-Infra.ps1 -ProjectId sentinel-platform-485714
```

El nombre del cluster sale de los outputs de Terraform (`terraform output`). **Cierra los 4
criterios pendientes de la Fase 1**, que llevan sin medir desde el principio.

### Bloque 6 — Kafka **[TU ejecuta, IA diagnostica]**

```powershell
.\infra\scripts\Install-Kafka.ps1          # pideme que lo lea antes
kubectl apply -k k8s/base/
kubectl get kafka -n kafka-operator
kubectl get kafkatopic -n kafka-operator
```

Los 3 topics ya estan declarados en `k8s/base/kafka/kafka-topics.yaml`:

| Topic | Particiones | Replicas | `min.insync.replicas` |
|---|---:|---:|---:|
| `sensor-readings-raw` | 12 | 3 | 2 |
| `sensor-validated` | 12 | 3 | 2 |
| `anomaly-alerts` | 3 | 3 | 2 |

Namespaces ya declarados: `reactorguard-ingestion`, `reactorguard-ml`,
`reactorguard-observability`, `kafka-operator`.

### Bloque 7 en adelante — desarrollo **[IA]**

A partir de aqui vuelvo a escribir codigo yo, con el cluster vivo para poder probarlo.
Ver seccion 7.

---

## 7. Las tareas del Paso 4, en orden de dependencia

### T3.3 — TEP Streamer -> Kafka producer

`data/generators/tep_streamer.py` + deployment K8s. Publica en `sensor-readings-raw`.

- **Reutiliza `readings_from_frame` de `data/generators/tep_adapter.py`**, que ya reconstruye
  `SensorReading` desde el parquet largo. No dupliques esa lectura.
- Serializacion: `SensorReading.to_kafka_bytes()` / `from_kafka_bytes()`, ya existen y tienen
  tests.
- **CLAVE DE PARTICION: `sensor_id`. No es un detalle, es una condicion de correccion.**

  Kafka garantiza orden SOLO dentro de una particion, y en un consumer group cada particion
  la lee EXACTAMENTE UN consumidor. La clave decide la particion: `hash(clave) % 12`.

  Los cinco detectores NO son funciones puras, acumulan estado por sensor:

  | Detector | Estado |
  |---|---|
  | `StuckValueDetector` | `_last_value` y `_run_length` por sensor |
  | `RateOfChangeDetector` | `_last` = (valor, timestamp) por sensor |
  | `KalmanResidualDetector` | un `OnlineKalmanFilter` completo por sensor |
  | `CrossCorrelationChecker` | un `deque` de 100 muestras por sensor |

  **Sin clave (round-robin), las 480 lecturas de `TEP-XMV-04` se reparten entre las 12
  particiones**, unas 40 por particion, y las leen consumidores distintos. Cada uno ve una de
  cada 12 lecturas, el `run_length` NUNCA llega a 8, y **el unico sensor realmente congelado
  del dataset deja de detectarse**. Sin error, sin log, sin excepcion: el criterio de Fase 2
  que hoy vale 1,0000 se desploma y nada te dice donde mirar. Ademas el orden se rompe, el
  `elapsed` sale negativo (lo descarta `sensor_validator.py:356`) y el `dt` del Kalman queda mal.

  **Dos consecuencias operativas que se derivan de esto:**

  1. **El paralelismo maximo util son 12 consumidores**, uno por particion. El decimotercero
     se queda sin asignacion y no acelera nada.
  2. **UN REBALANCEO PIERDE EL ESTADO EN MEMORIA.** Cuando un consumidor entra o sale, Kafka
     reasigna particiones y el nuevo dueno arranca con los detectores vacios. Un sensor
     congelado necesita **otras 8 muestras = 24 minutos** a cadencia TEP para volver a
     detectarse. Es asumible, pero hay que saberlo y NO confundirlo con un fallo del detector.
     Si esos 24 minutos fuesen inaceptables en operacion, habria que persistir el estado, y eso
     es un rediseno, no un ajuste: preguntame antes de acometerlo.

- Modo "fast" para el benchmark (sin respetar la cadencia real de 3 min) y modo "realtime".
- **Criterio 1 de la Fase 2 (>50.000 msg/s) se mide aqui**, con
  `tests/integration/benchmark_kafka.py`, que ya existe.

### T3.5 — Validation Consumer

`data/validation/validation_consumer.py` + `validation_service.py` + deployment.

- Consume de `sensor-readings-raw`, valida con `SensorValidator`, publica en
  `sensor-validated` y las alertas en `anomaly-alerts`.
- **`SensorValidator.get_metrics()` devuelve un dict plano a proposito (decision 7).
  ESTE es el modulo que lo convierte en metricas Prometheus** y expone `/metrics`.
- **Commit del offset SOLO despues de publicar** todos los mensajes del batch. Al reves se
  pierden mensajes en un reinicio.
- **Dimensiona las alertas con el 3,59% del hallazgo 3.E**, no con una estimacion optimista.
- Ojo con las replicas y el estado por sensor: ver la nota de particionado en T3.3.

### T4.3 — Feast + Redis

- **La salida del featurizer ya es LARGA con `sensor_id` como clave** (decision 10), que es
  justo lo que espera `sensor_entity` con `join_key=sensor_id`. No hay que transformar nada.
- **El Project ID que aparece en los prompts de Fase 2 para `feature_store.yaml` es
  `reactorguard-platform`: ESTA MAL.** Usa `sentinel-platform-485714`.
- **Los TTL del prompt original (1 h, 30 s, 1 min, 5 min) estan pensados para cadencia de 1 s.**
  Un TTL en Feast es cuanto tiempo un valor sigue sirviendose online. Con el de 30 s que el
  prompt propone para el residual de Kalman, y muestras cada 180 s:

  ```
  t=0        llega la lectura, feature escrita, TTL 30 s
  t=0..30    get_online_features() devuelve el valor      <-  30 s utiles
  t=30..180  devuelve null: caducado, y no hay lectura nueva  <- 150 s muertos
  t=180      llega la siguiente
  ```

  **El 83% del tiempo la feature devuelve `null`.** El servicio no fallaria: recibiria nulos y
  o bien imputaria (fabricando la evidencia) o bien descartaria la prediccion. Silencioso en
  ambos casos.

  Regla: **TTL > intervalo de muestreo, con margen** (a 180 s, del orden de 360-540 s). Pero
  el numero se mide contra la cadencia real del stream, no se hereda. Recalibralos y dimelo.

### T4.5 — Particionamiento GCS

- `data/storage/gcs_client.py` con `GCSStorageClient` y **`LocalCache` con la misma interfaz**
  (backend en disco), que es lo que permite testear sin credenciales.
- Buckets que Terraform ya crea: `data-raw`, `data-processed`, `models`, `mlflow`
  (prefijados y con sufijo `-${var.env}`).

### T4.6 — Dashboard Grafana

- `observability/grafana/dashboards/data-pipeline.json`. **La excepcion de `.gitignore` ya
  esta puesta** (`!observability/**/*.json`), asi que se versionara bien.
- Los paneles de latencia asumen metricas que T3.5 debe exponer antes.

### T4.7 — `Verify-Phase2.ps1`

- **CRITERIO 2 lee `tests/results/validator_metrics_tep.json`**, que esta **gitignored a
  proposito**: lleva `latency_ms` (media, p50, p99), medidas de reloj que cambian en cada
  ejecucion y ensuciarian el arbol despues de cada `pytest`. Es el INFORME, no la puerta:
  quien hace cumplir el criterio es el propio test.

- **LA TRAMPA DEL SKIPIF — leela antes de escribir el script.** La suite de T3.6 arranca con:

  ```python
  pytestmark = pytest.mark.skipif(
      not _NORMAL_PARTITION.exists() or not _STUCK_PARTITION.exists(), ...)
  ```

  En un clon nuevo sin `data/processed/tep` poblado, esos 15 tests **no fallan: se SALTAN**.
  `pytest` devuelve **exit code 0**, todo parece verde, y el JSON no se escribe jamas. Un
  script ingenuo haria:

  ```
  pytest                                  -> exit 0, "todo bien"
  leer validator_metrics_tep.json         -> no existe
  -> "CRITERIO 2: FALLO"
  ```

  Y eso es **una mentira en la direccion peligrosa**: reporta "la precision no llega al 95%"
  cuando lo cierto es "no se ha medido nada". Son cosas distintas, y confundirlas es
  EXACTAMENTE la deuda que este proyecto ya arrastro una vez: `Progreso.md` declara la Fase 1
  cerrada con 4 criterios sin medir, por no separar "codigo escrito" de "criterio verificado".

  **Lo que el script tiene que hacer:**

  1. Comprobar que el parquet esta poblado; si no, correr `Invoke-Pipeline.ps1` o abortar
     diciendo "no medible: faltan los datos".
  2. Ejecutar `pytest tests/unit/test_validator_on_tep.py` y **distinguir passed de skipped**
     (`--junitxml` y parsear, o contar sobre `-q --tb=no`). El exit code NO basta.
  3. Solo entonces leer el JSON.
  4. **Reportar TRES estados, no dos: CUMPLIDO / FALLADO / NO MEDIDO.**

- CRITERIO 4 ya esta cumplido y demostrado; el script solo tiene que comprobarlo.

---

## 8. Nota de commit

**El arbol acumula tres sesiones sin commitear.** Si te lo pido, agrupa en commits coherentes
por bloque; **no commitees sin que te lo pida**:

1. Spans por sensor (`configs/`, `data/schemas/sensor_spans.py`, `derive_sensor_spans.py`,
   `Invoke-Pipeline.ps1`, borrado de `prepare_tep.ps1`) + sus tests.
2. Kalman (`ml/features/kalman.py`) + tests.
3. Validador (`data/validation/sensor_fault.py`, `sensor_validator.py`) + tests.
4. T3.6 (`fault_injector.py`, `test_validator_on_tep.py`, recalibracion del umbral de deriva,
   `readings_from_frame`).
5. T4.2 (`feature_params.py`, `pipeline.py`, `batch_featurizer.py`, retirada de
   `extractor.py`, seccion `features:` de params.yaml).
6. T4.4 (`train_val_test_split.py`, stage `split`, excepciones de `.gitignore`) +
   determinismo de `reading_id`.

`Docs/ReactorGuard_Progreso.md` sigue **desactualizado**: declara la Fase 1 cerrada con 4
criterios sin medir y la Fase 2 vacia cuando esta casi entera. Es el Paso 5 del plan.

---

## 9. Como quiero trabajar

**Paso a paso.** Antes de borrar o sobrescribir cualquier archivo, muestrame que contiene y
por que. Antes de regenerar el parquet o lanzar cualquier cosa que escriba en `data/`, dime
que va a escribir y donde. **Antes de aplicar o destruir infraestructura en GCP, parate y
dime exactamente que recursos toca y que coste implica.** Antes de tocar el contrato de datos
(`data/schemas/sensor_reading.py`) o el significado de `quality`/`is_usable`, preguntame.

Al terminar, dime que encontraste y como quedo la linea base, **con numeros medidos, no
estimados**.

**Si algo que te he dicho aqui no coincide con lo que ves en el codigo, dimelo en vez de
adaptarte en silencio.**

Principios obligatorios: sin emojis en codigo, comentarios, logs ni docs. Type hints
completos. Docstrings con Args/Returns/Raises. Tests unitarios en cada modulo nuevo. Scripts
de desarrollo local en PowerShell 7+ (.ps1) con arquitectura de 4 capas; .sh solo dentro de
contenedores o GitHub Actions. La cobertura no debe bajar del 70% y los modulos nuevos entran
en el alcance medido (no los anadas al `omit` para hacer pasar el gate).

---

## 10. Objetivo verificable al terminar el Paso 4

```powershell
ruff check .                                                  # 0 findings
mypy ml/ api/ data/ --ignore-missing-imports                  # 0 errores
pytest tests/unit tests/safety --cov --cov-fail-under=70      # verde
.\infra\scripts\Invoke-Pipeline.ps1                           # up to date

kubectl get nodes                                             # Ready
kubectl get kafka,kafkatopic -n kafka-operator                # 3 topics
kubectl get pods -n reactorguard-ingestion                    # streamer + validator Running
.\infra\scripts\Verify-Infra.ps1 -ProjectId sentinel-platform-485714
.\tests\integration\Invoke-KafkaTests.ps1                     # > 50.000 msg/s
.\infra\scripts\Verify-Phase2.ps1                             # 4/4 criterios blocking
```

Y al acabar, **destruye la infraestructura** para no acumular coste:

```powershell
.\infra\scripts\Remove-DevInfra.ps1
```

**NO avances a la Fase 3 (simulacion) ni a la Fase 4 (PINN) sin que yo te lo diga.**
