# ReactorGuard — Registro de Progreso
**Nuclear Reactor Anomaly Detection Platform**
*Actualizado: 2026-10-11 — Fase 2 verificada contra el commit `d35f802`; Fase 3: F3-M1 (`b343fb7`) y F3-M2 (`36c9d3a`) commiteados, F3-M3 hecho sin commitear*

Este documento distingue tres cosas que antes se confundian: **codigo escrito**, **verificado en local**
y **criterio medido en el entorno real**. Una tarea puede estar completada como codigo y aun asi no
tener su criterio de exito medido.

---

## Estado General

| Fase | Nombre | Semanas | Estado |
|------|--------|---------|--------|
| 1 | Foundation – GKE + Kafka | 1–2 | Codigo completo, **sin desplegar**; criterios sin medir |
| 2 | Data Pipeline + Sensor Validator | 3–4 | **Codigo completo y verificado en local**; criterios 2 y 4 CUMPLIDOS, 1 y 3 NO MEDIDOS hasta medir en el cluster |
| 3 | Simulation Layer (OpenMC + PyDy) | 5 | **En curso**: F3-M1 (modelo termico, tabla de agua, catalogo), F3-M2 (constraints fisicos) y F3-M3 (simulador en operacion normal, spans y estudio de cadencia) hechos; F3-M4 a F3-M7 pendientes. Plan en `ReactorGuard_Fase3_Plan.md` |
| 4 | ML Core – PINN Training | 6–7 | No iniciada (solo andamiaje) |
| 5 | Uncertainty – BNN + Conformal Prediction | 8 | No iniciada |
| 6 | API + K8s Serving | 9–10 | No iniciada (solo andamiaje, 2 de 4 endpoints devuelven 501) |
| 7 | CI/CD + Security Gates | 11 | Solo esqueleto (`ci.yaml`, `terraform.yaml`, `dependabot.yml`) |
| 8 | Observability Stack | 12 | No iniciada; `observability/` solo contiene el dashboard `data-pipeline` (T4.6, adelantado desde Fase 2) |

### Linea base medida (2026-10-11, cierre de M9)

```
pytest tests/unit tests/safety --cov --cov-fail-under=70
  1836 passed, 3 skipped       cobertura TOTAL 99,41%   (3747 sentencias, 22 sin cubrir)   458 s
ruff check .                   All checks passed
mypy ml/ api/ data/            no issues found in 66 source files
dvc status                     Data and pipelines are up to date

dvc.yaml                       download_tep -> explore_tep -> adapt_tep -> featurize -> {feast_source, split}

data/processed/tep             22 particiones   550.160 lecturas
data/processed/features        22 particiones   550.160 filas x 31 columnas
data/processed/feast_source    22 runs          550.160 filas (linea temporal sintetica)
data/processed/splits          train 385.112 / val 82.524 / test 82.524
```

Evolucion de la suite: 585 passed al abrir la Fase 2 (M0) -> 1803 al cerrar M8 -> 1836 con los 32 tests de
`Verify-Phase2.ps1` y el contraste de `$ProjectId`. Los 3 omitidos de abajo no han cambiado.

Los 3 tests omitidos son la suite de `tests/safety/test_safety_constraints.py`, marcada `skip`
a proposito hasta la Fase 3 (depende de `reactor_simulator.py`, roto contra el schema canonico).

### Linea base al cierre de F3-M1 (2026-10-11, commit `b343fb7`)

```
pytest tests/unit tests/safety --cov --cov-fail-under=70
  2053 passed, 4 skipped       cobertura TOTAL 99,47%   (4507 sentencias, 24 sin cubrir)   525 s
ruff check .                   All checks passed
mypy ml/ api/ data/            no issues found in 72 source files
dvc status                     Data and pipelines are up to date
```

El omitido nuevo es `test_nodes_match_iapws` (`importorskip("iapws")`): `iapws` es dependencia de
desarrollo y no esta en el venv. Cobertura por modulo nuevo de `data/simulation`: `thermal_model` 100 %,
`water_properties` 100 %, `plant_params` 100 %, `sensor_catalog` 100 %, `build_water_table` 98 % (solo el
guard `__main__`).

### Linea base al cierre de F3-M2 (2026-10-11, sin commitear)

```
pytest tests/unit tests/safety --cov --cov-fail-under=70
  2184 passed, 4 skipped       cobertura TOTAL 99,50%   (4755 sentencias, 24 sin cubrir)   457 s
ruff check .                   All checks passed
mypy ml/ api/ data/            no issues found in 73 source files
dvc status                     Data and pipelines are up to date
```

Cobertura del modulo nuevo: `physics_constraints` 100 %; `plant_params` sigue en 100 % con sus 6 claves
nuevas. Los 24 sin cubrir son los mismos que al cierre de F3-M1 (el guard `__main__` de
`build_water_table` y codigo de Fase 2).

### Linea base al cierre de F3-M3 (2026-10-11, sin commitear)

```
pytest tests/unit tests/safety --cov --cov-fail-under=70
  2399 passed, 3 failed, 4 skipped   cobertura TOTAL 99,48%   (5625 sentencias, 29 sin cubrir)
ruff check .                   All checks passed
mypy ml/ api/ data/            no issues found in 78 source files
dvc status                     Data and pipelines are up to date
```

**Los 3 fallos son `TestRepositoryParams` de `tests/unit/test_simulation_params.py`**: leen la seccion
`simulation:` del `params.yaml` del repo, que **todavia es la antigua**. La edicion fue denegada por el
clasificador de permisos de la sesion y queda para el usuario (el diff esta en el informe del hito). Con la
seccion nueva aplicada son 2402 passed. El tiempo de la suite (2043 s) NO es comparable con los 457-477 s de
las lineas base anteriores: Kernel-Power registro suspensiones (eventos 506/507 a las 20:04, 20:06 y 20:32)
dentro de ese intervalo; la misma suite tardo 477 s en la corrida anterior. Las mediciones del informe de
F3-M3 (19:35-19:41) no tienen eventos de suspension en su intervalo.

Cobertura de los modulos nuevos: `simulation_params` 100 %, `scenario` 100 %, `dataset_builder` 100 %,
`physics_constraints` 100 % (con `rule_margins`), `reactor_simulator` 99 %, `simulation_study` 99 %,
`derive_sim_spans` 97 % (los tres con el guard `__main__` sin cubrir). `reactor_simulator.py` salio del `omit`
de cobertura y del override de mypy. Omitidos: los mismos 4 (3 de `tests/safety` hasta F3-M6 y el contraste
con `iapws`).

Mediciones de F3-M3 (detalle en `Docs/physics_constraints.md`, seccion 11): sigma de lazo 0,15 K (con 0,30 K
la regla de energia daba 22 violaciones en 51.000 instantes normales; con 0,15 K, 0), peor residuo de energia
3,92 % y de masa 4,41 %; cadencia 1/10/60/180 s: 0 violaciones sobre valores verdaderos, max abs(dT/dt) del
estado 0,22 C/s; regla de masa evaluable en 15,0 / 94,9 / 99,7 / 99,2 % de las muestras medidas; validador
de Fase 2 a 180 s: 90 alertas por 1000 lecturas, todas `kalman_anomaly` en los 5 canales de caudal;
determinismo identico entre dos construcciones y entre procesos; 414 pasos/s extremo a extremo, dataset
completo (1.377.000 lecturas) en 146 s y 75,2 MB.

---

## Fase 1 — Foundation: GKE + Kafka
**Semanas 1–2** · Codigo completo y validado; **nunca desplegado**

El historial de commits lo declara asi: "a comprobar despliegue con terraform". Todo lo siguiente esta
escrito y pasa `terraform validate` y los tests de manifiestos
(`tests/unit/test_k8s_manifests.py`), pero **ningun recurso se ha creado en GCP**.

### Semana 1 — Infraestructura GCP con Terraform

| ID | Tarea | Codigo | Desplegado | Notas |
|----|-------|--------|------------|-------|
| T1.1 | Estructura del repositorio y configuracion inicial | Completado | n/a | Arbol de dirs, pyproject.toml, .gitignore, README |
| T1.2 | Terraform: backend de estado y proyecto GCP | Completado | No | `backend.tf` espera el bucket `reactorguard-terraform-state`, que debe existir antes del primer `init` |
| T1.3 | Terraform: VPC y networking | Completado | No | VPC, subnet, Cloud NAT, firewall |
| T1.4 | Terraform: GKE cluster | Completado | No | Cluster privado, node pools `platform` y `ml-serving` con taints |
| T1.5 | Terraform: GCS buckets | Completado | No | 4 buckets (raw, processed, models, mlflow), lifecycle, versionado |
| T1.6 | Terraform: IAM y Workload Identity | Completado | No | 4 SAs con minimo privilegio |
| T1.7 | Terraform: Secret Manager y KMS | Completado | No | 4 secretos placeholder, CMEK con rotacion 90 d |
| T1.8 | Terraform: Cloud Armor + LB + IAP | Completado | No | `armor.tf`, `lb.tf`, `iap.tf` |
| T1.9 | Scripts de verificacion y destruccion | Completado | No | `Verify-Infra.ps1`, `Remove-DevInfra.ps1` (confirmacion `DESTROY-DEV`) |

### Semana 2 — Plataforma Base, Seguridad y Kafka

| ID | Tarea | Codigo | Desplegado | Notas |
|----|-------|--------|------------|-------|
| T2.1 | Namespaces K8s y NetworkPolicies | Completado | No | 4 namespaces + NetworkPolicies deny-all + allow por namespace |
| T2.2 | Workload Identity en Kubernetes (KSAs) | Completado | No | Project ID corregido a `sentinel-platform-485714` (antes `reactorguard-platform`, que no existe) |
| T2.3 | Strimzi Kafka Operator v0.39.0 | Completado | No | KafkaCluster 3 brokers + 3 zookeepers; 3 topics (raw / validated / alerts); `Install-Kafka.ps1` |
| T2.4 | Verificacion de Kafka: latencia y throughput | Completado | No | `test_kafka_connectivity.py`, `benchmark_kafka.py`, `Invoke-KafkaTests.ps1` |
| T2.5 | Skeleton CI/CD con GitHub Actions | Completado | n/a | `ci.yaml`, `terraform.yaml`, `api/Dockerfile` multi-stage no-root, `dependabot.yml` |

### Endurecimiento de seguridad (commit `98cd321`, auditoria)

Aplicado tras una auditoria: ajustes en IAM (modulos `iam`), `gke`, `security/kms.tf`, `security/lb.tf`,
Kafka (nuevo `k8s/base/kafka/kafka-users.yaml` con usuarios y ACLs), NetworkPolicies, Dockerfile y
`ci.yaml`. Todo sin desplegar.

### Criterios de Exito Fase 1

| Metrica | Umbral | Estado |
|---------|--------|--------|
| Kafka produce/consume latency p99 | < 10 ms en red interna | **No medido** (requiere clúster) |
| GKE cluster status | RUNNING, todos los nodos Ready | **No medido** |
| `terraform validate` en CI | 0 errores | Cumplido en local; no verificado en GitHub Actions |
| Trivy scan imagenes base | 0 vulnerabilidades CRITICAL | **No medido** |

### Deuda conocida de Fase 1

- ~~`k8s/base/kustomization.yaml` no incluye `kafka/`~~ **Resuelto en M6**: ahora incluye el cluster, los topics,
  los usuarios y el ConfigMap `kafka-metrics` que `kafka-cluster.yaml` referenciaba y no existia. Un test
  (`test_k8s_manifests.py`) lo mantiene. Sigue sin desplegarse.
- **Region**: Terraform y `bootstrap.ps1` usan `europe-southwest1`. Este documento decia `europe-west1`
  hasta hoy; el valor correcto es `europe-southwest1`.
- **Project ID**: el ID real es `sentinel-platform-485714`. `reactorguard-platform` es solo el nombre del
  proyecto y no existe como ID de GCP.

---

## Fase 2 — Data Pipeline + Sensor Validator
**Semanas 3–4** · Codigo completo y verificado en local; sin desplegar. Criterios 1 y 3 pendientes del cluster

### SEMANA 3 — TEP Dataset + Sensor Validator

| ID | Tarea | Estado | Notas |
|----|-------|--------|-------|
| T3.1 | Schema Pydantic de sensor reading | Completado | `data/schemas/sensor_reading.py`, cobertura 100%. Serializacion `to_kafka_bytes` / `from_kafka_bytes` |
| T3.2 | Descarga y exploracion del TEP | Completado | `tep_downloader`, `tep_explorer`, `tep_loader`, `tep_adapter`, `adapt_tep`. 22 ficheros, 550.160 lecturas. `reading_id` determinista (uuid5) |
| T3.3 | `tep_streamer.py`: parquet/GCS -> productor Kafka | Completado (local) | `data/generators/tep_streamer.py` + `tep_streamer_service.py`. Clave `sensor_id`, cabecera `run-id`, modos `fast` y `realtime` (plazos absolutos, sin deriva). Local, modo fast: 550.160 mensajes en 70 s (~7,9 mil msg/s, 0 errores, un proceso; informativo). Sin desplegar |
| T3.4 | `sensor_validator.py`: 5 detectores | Completado | stuck, rate-of-change, range, cross-correlation, Kalman. Cobertura 99%. Spans por sensor con dos rangos (`configs/sensor_spans.yaml`) |
| T3.5 | Consumidor Kafka: validacion -> `sensor-validated` | Completado (local) | `validation_consumer.py`, `validation_service.py`, `alert_event.py`. At-least-once (flush y despues commit), veneno saltado con commit, reset de estado por `run-id`, retroceso de reloj y rebalanceo. 15 metricas `reactorguard_*` en `:9090`. Local: ~2,2 mil lecturas/s. Con los 22 runs emite **100.934 alertas (18,3 %)**, no las ~19.700 estimadas (ver hallazgo 9). Sin desplegar |
| T3.6 | Tests del validador con datos TEP reales | Completado | `test_validator_on_tep.py` (15 tests). Inyeccion sintetica con semilla fija sobre d00 y validacion cruzada con el unico positivo real (XMV-04 en d21) |

### SEMANA 4 — Feature Engineering + Feast + DVC

| ID | Tarea | Estado | Notas |
|----|-------|--------|-------|
| T4.1 | `kalman.py`: filtro Kalman online por sensor | Completado | `OnlineKalmanFilter`, `KalmanFilterBank`, cobertura 100% |
| T4.2 | `features/pipeline.py` + `batch_featurizer.py` | Completado | 7 grupos de features del TDD 5.5, ventanas en muestras (`[10, 20, 60]`), salida larga con `sensor_id` |
| T4.3 | Feast feature store | Completado (local) | `ml/features/store/`: vista `sensor_features` con campos generados desde `FeaturePipeline.feature_names()`, stage DVC `feast_source`, offline parquet, online SQLite en local y Redis en cluster (sin BigQuery). TTL = 3 intervalos = **540 s**. El TTL de Feast **no caduca** nada en lectura online (medido en 0.64): la frescura la aplica `FeastClient` y Redis lleva `key_ttl_seconds` (hallazgo 7). Sin desplegar |
| T4.4 | DVC: `download_tep -> explore_tep -> adapt_tep -> featurize -> {feast_source, split}` | Completado | 6 stages, reproducible desde cero (bit a bit, ver criterio 4); la segunda ejecucion las salta todas. No incluye `simulate` (Fase 3). Usar siempre `Invoke-Pipeline.ps1`, no `dvc repro` a pelo |
| T4.5 | Particionamiento GCS (`GCSStorageClient` + `LocalCache`) | Completado (local) | `data/storage/`: layout hive `plant=/year=/month=/day=/hour=`, parquet con esquema Arrow, partes con nombre determinista y escritura `if_generation_match=0` (412 = ya escrito: `ingestion-sa` solo tiene `objectCreator`). Nada del pipeline lo llama todavia: desbloquea la fuente del streamer en el cluster y el archivo futuro. Sin desplegar |
| T4.6 | Dashboard Grafana del pipeline de datos | Completado (local) | `observability/grafana/dashboards/data-pipeline.json`: 18 paneles, variable `rate_window` (defecto 15m), 158 tests que contrastan cada metrica y etiqueta con `StreamMetrics`. Validado **sin inspeccion visual** (sin renderizador). Sin paneles de latencia de features ni de GCS (no hay metrica que los alimente). **No desplegado: no hay Grafana ni Prometheus en el cluster** |
| T4.7 | `Verify-Phase2.ps1` | Completado | `infra/scripts/Verify-Phase2.ps1`: cuatro capas, tres estados (CUMPLIDO / FALLADO / NO MEDIDO), salidas 0/1/2 (3 = error interno), informe `tests/results/phase2_verification.json`. 32 tests ejecutan PowerShell sobre artefactos sinteticos. Distingue `passed` de `skipped` con JUnit y exige `environment: cluster` en los criterios 1 y 3 |

### Criterios de Exito Fase 2

Estado dado por `.\infra\scripts\Verify-Phase2.ps1` (informe en `tests/results/phase2_verification.json`).
Un criterio solo es CUMPLIDO con evidencia del entorno que exige; una medicion local es informativa.

| Metrica | Umbral | Estado | Evidencia |
|---------|--------|--------|-----------|
| 1. Throughput Kafka pipeline | > 50.000 readings/s sostenidos | **NO MEDIDO** (exige cluster) | Local, informativo: **13.993 msg/s** (280.800 mensajes en 20,07 s, un proceso Python, un broker de un nodo, ventanas de 9.270 a 17.465 msg/s). Medido en `a428e71-dirty` |
| 2. Stuck sensor precision | > 95% | **CUMPLIDO** | **1,0000**; 27 tests ejecutados y 0 saltados (T3.6 + invariantes de streaming), 473 de 480 pares de XMV-04/d21 detectados tambien por el camino de streaming con 1, 2 y 3 consumidores |
| 3. Feature pipeline latency | < 100 ms por ventana bajo carga | **NO MEDIDO** (exige cluster) | Local, informativo: **p99 82,1 ms** (p50 48,0, p95 69,9, max 89,9; 441 llamadas, 60 muestras x 52 sensores, con 4 procesos de validacion en paralelo). La cifra anterior de 98,8 ms era previa a la vectorizacion de `transform` |
| 4. `dvc repro` reproducible desde cero | Sin errores | **CUMPLIDO** | `dvc status` limpio, `dvc.lock` registra las 6 etapas y la reproduccion desde cero en una copia temporal (con descarga) dio las **8 salidas con el mismo hash** (`-FromScratch`, 80 s) |

Los criterios 1 y 3 **no se pueden cerrar sin el cluster**. Para el 1 hay ademas una limitacion conocida: el
techo medido de un proceso de kafka-python (12.000 a 17.000 msg/s) esta por debajo de 50.000, de modo que
la medicion en el cluster podria exigir fragmentar sensores entre procesos productores (cada sensor siempre
en uno solo). No se ha anticipado: se decide con esa medicion.

### Metricas del validador sobre TEP (medidas, d00 con inyeccion sintetica)

| Detector | Precision | Recall | Tasa de falsos positivos |
|----------|-----------|--------|--------------------------|
| stuck | 1,0000 | 0,8141 | 0,0000 |
| bias_out_of_range | 1,0000 | 1,0000 | 0,0000 |
| noise_spike | 1,0000 | 0,7778 | 0,0000 |
| sensor_drift | 0,9926 | 0,9302 | 0,0001 |

Latencia del validador por par: media 0,078 ms, p50 0,066 ms, p99 0,230 ms (maquina de desarrollo, sin carga).
Validacion cruzada sobre el unico positivo real de stuck (XMV-04, d21): 473 de 480 pares detectados, 0 falsos
positivos en otros sensores.

### Hallazgos medidos que condicionan lo que queda

1. **La cadencia del TEP es de 180 s, no 1 s.** Ha invalidado tres defaults heredados del TDD:
   `process_noise` del Kalman (0,1 -> 1e-6), umbral de deriva (1,0 -> 11,0 anchos de sobre por hora; con 1,0
   disparaba en el 48% de la operacion normal) y ventanas de features en segundos (ahora en muestras).
   **Cualquier umbral, ventana, TTL o timeout nuevo se calibra a 180 s y se mide.**
2. **Deriva lenta de instrumento: no detectable** con este mecanismo a esta cadencia. Banda util de solo
   10–18 anchos de sobre/hora. Documentado en `DRIFT_DETECTION_FLOOR_NOTE`. Haria falta un estimador de linea
   base larga.
3. **Falsos positivos del residual de Kalman: 3,59%** sobre sensores intactos de d00 (colas pesadas). Por eso
   `kalman_anomaly` es senal de apoyo, sin gate. **El dimensionado de `anomaly-alerts` con ~19.700 alertas
   quedo superado por la medicion con los 22 runs: 100.934 (ver hallazgo 9).**
4. **Positivos reales de fallo de instrumento en el TEP: casi inexistentes** (solo XMV-04 en d21, 0,087%).
   El plan original que daba d14/d15 como positivos de stuck era incorrecto.
5. **Estado en memoria por sensor.** Los detectores acumulan estado: sin particionar por `sensor_id`, el
   sensor congelado deja de detectarse en silencio. Un rebalanceo de Kafka pierde ese estado (unos 24 min a
   cadencia TEP para volver a detectar). Persistirlo seria un rediseno: decision pendiente.
6. **El split es temporal dentro de cada `fault_type`.** `temporal_order_holds` es true por clase; en
   agregado no se cumple ni puede (d00 tiene 500 timesteps y el resto 480).
7. **El TTL de Feast no caduca nada en lectura online** (medido en Feast 0.64). El calculo del 83 % de
   `Handoff_Paso4.md` sigue siendo el motivo de que el TTL sea de 3 intervalos (540 s) y no de 30 s, pero la
   caducidad la aplica `FeastClient` con el `event_timestamp` y, en el cluster, Redis con
   `key_ttl_seconds` = 540 s. El 83 % es aritmetico (150 s de 180 s sin valor vigente con un TTL de 30 s),
   no una medicion.
8. **`rate(...[1m])` con cadencia de 180 s y scrape de 15 s no devuelve vacio: devuelve un valor
   enganoso.** Medido en M8: da 0 en el 80 % de las evaluaciones y picos de 1,16 msg/s frente a una media
   real de 0,29; con `[5m]` da 0 en el 44 %. Solo queda vacio con scrape >= 60 s. Por eso el dashboard usa
   `rate_window` (defecto 15m) y `increase()`. Corrige la redaccion previa ("a 180 s una consulta
   `rate(...[1m])` devuelve vacio").
9. **El validador emite 100.934 alertas (18,3 %) sobre los 22 runs**, no ~19.700 (3,59 %, medido solo en
   operacion normal sobre sensores intactos). Por tipo: `bias_out_of_range` 70.225, `stuck` 15.765,
   `kalman_anomaly` 11.518, `sensor_drift` 1.920, `noise_spike` 1.506. Por `quality`: good 462.478,
   suspect 55.905, bad 31.777. **Causa no investigada**: probable excursion de proceso en los 21 runs con
   fallo; `stuck` merece revision, porque d21 solo tiene 480 positivos reales. Es una cifra medida, no un
   defecto diagnosticado.
10. **Con 52 claves el particionador deja vacias 2 de las 12 particiones** (la 5 y la 9; la mas cargada
    recibe 8 sensores): el paralelismo util del validador son 10 consumidores, no 12.
11. **El camino en linea de features no existe**: `FeaturePipeline.transform()` es por lotes sobre una
    ventana contigua. El criterio 3 se mide sobre `transform()` de una ventana y T3.5 no calcula features.

---

## Fase 3 — Simulation Layer: OpenMC + PyDy
**Semana 5** · En curso. Plan aprobado por hitos en [ReactorGuard_Fase3_Plan.md](ReactorGuard_Fase3_Plan.md)
(decisiones D1-D13 y registro de progreso en su seccion 15). F3-M0 (preflight), F3-M1, F3-M2 y F3-M3 hechos.

`data/generators/reactor_simulator.py` estaba roto contra el schema canonico; **F3-M3 lo reescribio** y ya cuenta en mypy y en la cobertura.

**F3-M1 (modelo fisico, tabla de agua y catalogo)** aporta `data/simulation/` (`thermal_model`,
`water_properties`, `build_water_table`, `plant_params`, `sensor_catalog`) y cuatro YAML en `configs/`.
Medido: equilibrio nominal exacto (n = 1 tras 24 h, max |dy/dt| relativo 5,9e-17), cierre de energia 2,0e-15
sobre 2000 pasos de rampa, 4,2 a 6,7 ms por paso de 180 s, LSODA frente a Radau 6,5e-8, tabla frente a
IAPWS 0,16 % (liquido, rejilla completa) y 0,43 % (region de operacion), misma traza byte a byte entre procesos.
El motor es scipy/sympy: PyDy se descarto (D4). Los coeficientes neutronicos son
`LITERATURE_PLACEHOLDER` hasta F3-M8.

**F3-M2 (constraints fisicos)** aporta `data/simulation/physics_constraints.py` (las 6 reglas de TDD 5.4 mas la
ley de masa de App. B, `evaluate_rules` por roles del catalogo, `RuleReport`, `ConservationReport` y
`min_observable_rate`), `Docs/physics_constraints.md` (ecuaciones, limites, inconsistencias del TDD y matriz de
violaciones esperadas) y seis claves nuevas en `limits:`. Medido: el estado verdadero del modelo (equilibrio y
rampas de 0,5 a 10 %/min) viola 0 de las 7 reglas y conserva la energia (instantaneo 1,9e-20, por intervalo
1,8e-15). Con el ruido inicial del catalogo la regla de energia tiene sigma ~1,2 % a plena carga: 4,2 sigma
frente al 5 % del TDD, y 0,11 % de falsos positivos a 70 % de carga (decision para F3-M3). A 180 s la regla
dT/dt exige un salto de 1800 K entre muestras (C5): no es un detector a esa cadencia.

**F3-M3 (simulador en operacion normal, spans y estudio de cadencia)** aporta `data/simulation/{simulation_params,scenario,dataset_builder}.py`, la reescritura de `data/generators/reactor_simulator.py`, `data/generators/derive_sim_spans.py` + `configs/sim_sensor_spans.yaml` (27 sensores, 5 runs normales, semilla propia) y `data/generators/simulation_study.py` (cadencia, ruido, determinismo y rendimiento, repetible en F3-M8). Genera 34 runs normales como frame largo con las 11 columnas de `readings_from_frame` mas la verdad-terreno del TDD 2.2, validables offline con el mismo codigo del TEP. **Decision cerrada con datos**: sigma de los termopares de lazo 0,15 K (la regla de energia a baja carga). **Hallazgo para Fase 2**: el detector Kalman del validador usa `observation_noise = 1,0` fija en unidades de ingenieria y alerta en ~49 % de las lecturas normales de los canales de caudal (deuda, no se toca). Pendiente del usuario: aplicar el diff de la seccion `simulation:` de `params.yaml`.

| ID | Tarea | Estado |
|----|-------|--------|
| T5.1 | OpenMC configurado: modelo PWR simplificado | Pendiente (F3-M8, opcional, corre en Docker; no esta en PyPI ni en conda-forge win-64) |
| T5.2 | Modelo termodinamico: ODEs coolant/core | **Completado en F3-M1** (scipy LSODA, no PyDy); `data/simulation/thermal_model.py` |
| T5.3 | `reactor_simulator.py`: neutronica + termica integradas | **Completado en F3-M3** en operacion normal (modelo ODE, 34 runs, 1.377.000 lecturas, determinista entre procesos, 414 pasos/s); los fallos llegan en F3-M4/M5 |
| T5.4 | `fault_injector.py`: drift, stuck, noise spike, bias + 4 tipos adicionales | **Parcial**: `data/validation/fault_injector.py` ya cubre los 4 tipos base sobre TEP (T3.6); faltan los 4 fisicos |
| T5.5 | DVC stage `simulate` con labels | Pendiente (F3-M6); `ReactorSimulator.write` y `main` ya escriben el layout del TDD 4.4 |
| T5.6 | Validacion de constraints fisicos (Appendix B TDD) | **Parcial, F3-M2**: reglas y conservacion implementadas y verificadas contra el modelo (`physics_constraints.py`, `Docs/physics_constraints.md`). Pendiente: la matriz de violaciones esperadas (M4-M6) y la suite de `tests/safety`, que sigue en `skip` hasta F3-M6 |

---

## Fase 4 — ML Core: PINN Training
**Semanas 6–7** · No iniciada (solo andamiaje)

Existe `ml/models/pinn.py`, `ml/training/train.py` y `ml/serving/predictor.py`, pero:
`physics_residual` **no implementa la fisica que documenta**, `train.py` no tiene loop de entrenamiento y
`predictor.py` declara MAPIE sin ajustarlo.

| ID | Tarea | Estado |
|----|-------|--------|
| T6.1 | `ml/models/pinn.py`: arquitectura + energy/mass balance loss | Pendiente (esqueleto con forward verificado en `test_pinn_forward.py`) |
| T6.2 | Training loop PyTorch: Adam, LR scheduler, early stopping | Pendiente |
| T6.3 | Integracion MLflow | Pendiente |
| T6.4 | DVC stage `train` | Pendiente |
| T6.5 | `ml/validation/physics_check.py` | Pendiente |
| T7.1 | Reentrenamiento con dataset de simulacion | Pendiente |
| T7.2 | Grid search lambda_energy y lambda_mass | Pendiente |
| T7.3 | `ml/models/rules.py`: 6 constraints del TDD | Pendiente |
| T7.4 | Fusion PINN + Rules | Pendiente |
| T7.5 | `ml/validation/safety_check.py` | Pendiente |
| T7.6 | SHAP: top 3 features por prediccion | Pendiente |
| T7.7 | Benchmark latencia PINN < 50 ms p99 CPU | Pendiente |

---

## Fase 5 — Uncertainty: BNN + Conformal Prediction
**Semana 8** · No iniciada

| ID | Tarea | Estado |
|----|-------|--------|
| T8.1 | `ml/models/bnn.py` | Pendiente |
| T8.2 | `ml/models/uncertainty.py` (MAPIE, method='plus', alpha=0.05) | Pendiente |
| T8.3 | `calibrate_threshold` | Pendiente |
| T8.4 | `ml/validation/calibration_check.py` | Pendiente |
| T8.5 | `ml/models/fusion.py` completo | Pendiente |
| T8.6 | Logging de calibracion en MLflow | Pendiente |

---

## Fase 6 — API + K8s Serving
**Semanas 9–10** · No iniciada (andamiaje)

Existen `api/main.py`, `routers/health.py`, `routers/predict.py` y `routers/explain.py`. `predict` y `explain`
devuelven HTTP 501 ([predict.py:27](../api/routers/predict.py#L27), [explain.py:29](../api/routers/explain.py#L29)).
`api/Dockerfile` ya esta (multi-stage, no-root).

| ID | Tarea | Estado |
|----|-------|--------|
| T9.1 | `api/main.py`: carga de modelos desde GCS en startup | Pendiente |
| T9.2 | Endpoints del TDD 6.1 (analyze, batch, validate, constraints, explain, health) | Pendiente (solo `health` existe) |
| T9.3 | Autenticacion JWT con Secret Manager | Pendiente |
| T9.4 | OpenTelemetry | Pendiente |
| T9.5 | Tests de integracion de endpoints | Pendiente |
| T10.1 | Dockerfile multi-stage Python slim, no-root | **Completado** (en T2.5) |
| T10.2 | Trivy scan: 0 CRITICAL | Pendiente (no medido) |
| T10.3 | `deployment.yaml` del pinn-server | Pendiente |
| T10.4 | PodDisruptionBudget minAvailable=2 | Pendiente |
| T10.5 | HPA 3–10 replicas | Pendiente |
| T10.6 | Canary 10% | Pendiente |
| T10.7 | Kustomize overlays dev / staging / prod | Pendiente (`k8s/overlays/*` vacios) |

---

## Fase 7 — CI/CD + Security Gates
**Semana 11** · Solo esqueleto

| ID | Tarea | Estado |
|----|-------|--------|
| T11.1 | `ci.yaml` completo: Semgrep + Gitleaks + pip-audit + Trivy | Parcial (lint, tests, scan, build; endurecido en `98cd321`) |
| T11.2 | `terraform.yaml`: fmt + validate + tfsec + plan | Parcial (workflow existe) |
| T11.3 | `ml-training.yaml` (semanal) con safety gates | Pendiente |
| T11.4 | `ml-promote.yaml` con aprobacion manual | Pendiente |
| T11.5 | Binary Authorization | Pendiente |
| T11.6 | GitHub environments dev / staging / prod | Pendiente |
| T11.7 | `tests/safety/test_critical_faults.py` | Pendiente |

---

## Fase 8 — Observability Stack
**Semana 12** · No iniciada. `observability/` solo contiene el dashboard `data-pipeline` (T4.6) y su
`kustomization.yaml`; no hay Prometheus, reglas de alerta ni Alertmanager.

| ID | Tarea | Estado |
|----|-------|--------|
| T12.1 | Prometheus: 8 metricas custom + reglas de alerting | Pendiente |
| T12.2 | Grafana: 4 dashboards | Parcial: 1 de 4 (`data-pipeline`, hecho como T4.6) |
| T12.3 | Alertmanager: CRITICAL -> PagerDuty, WARNING -> Slack | Pendiente |
| T12.4 | Jaeger + OpenTelemetry Collector | Pendiente |
| T12.5 | Runbooks en `docs/runbooks/` | Pendiente |
| T12.6 | SLO dashboard 99,99% | Pendiente |

---

## Deuda de despliegue de la Fase 2

Lo que impide dar por cerrada la Fase 2 en el cluster y que **no** se resuelve en local:

| Deuda | Detalle | Donde se resuelve |
|-------|---------|-------------------|
| Criterios 1 y 3 sin medir en cluster | Solo una medicion con `environment: cluster` los cierra | Despliegue + `Invoke-KafkaTests.ps1` y `benchmark_feature_latency.py --environment cluster` |
| **No hay Grafana ni Prometheus en el cluster** | T2.6 de la Fase 1 nunca se hizo. El dashboard `data-pipeline` (ConfigMap con etiqueta `grafana_dashboard: "1"` en `reactorguard-observability`) no tiene donde cargarse, y no hay scrape de los pods en `:9090` | Fase 8 (T12.1, T12.2) o antes si se quiere observar el despliegue |
| Fuente del streamer en el cluster | No hay remoto DVC; hay que subir el parquet con `gsutil rsync data/processed/tep gs://reactorguard-data-raw-dev/tep/` | Paso manual del despliegue |
| Secretos copiados entre namespaces | Un pod no monta Secrets de otro namespace: `Sync-KafkaCredentials.ps1` copia usuario y CA; la rotacion es manual | Paso manual |
| Contrasena de Redis y copia en `reactorguard-ml` | Se crea fuera de banda, nunca en el repo | Paso manual |
| `feast apply` en el cluster | El registro vive en `data-processed`; solo `cicd-sa` tiene `objectAdmin` | CI/CD (sin cambio de Terraform en esta fase) |
| Imagen unica y grande | `api/Dockerfile` arrastra torch para streamer y validador | Fase 6 |
| Sin DLQ ni persistencia del estado | Mensajes venenosos solo en el log; cada rebalanceo cuesta 24 min de recalentamiento de stuck | Decision pendiente |

## Siguiente paso recomendado

1. **Despliegue en GCP con intervencion humana** (autenticacion, bucket de estado, revisar `terraform plan`
   y coste antes de `apply`): cierra los criterios pendientes de Fase 1 y los criterios 1 y 3 de Fase 2.
   Destruir la infraestructura al terminar. Guia en [Handoff_Paso4.md](Handoff_Paso4.md) (secciones 6 y 7).
2. Antes del despliegue, decidir: persistencia del estado de los detectores, DLQ, y que hacer con las
   100.934 alertas (hallazgo 9).
3. Fases 3 y 4 solo cuando se indique explicitamente.

Referencia tecnica del camino de streaming (invariantes, contrato de mensajes, commit, rebalanceo, metricas y
runbook): [ReactorGuard_Fase2_Streaming.md](ReactorGuard_Fase2_Streaming.md).

---

## Leyenda

| Estado | Significado |
|--------|-------------|
| Completado | Codigo escrito, con tests, verificado |
| Parcial | Existe una parte; la nota indica que falta |
| Pendiente | No iniciado |
| No medido | El criterio de exito requiere un entorno que no existe todavia |

---

## Notas Tecnicas

- **Entorno**: Windows 11, scripts en `.ps1` (PowerShell 7+); `.sh` solo dentro de contenedores o GitHub Actions
- **Python**: venv en 3.12 (`requires-python >= 3.11`)
- **Pipeline de datos**: ejecutar siempre `.\infra\scripts\Invoke-Pipeline.ps1`. `dvc repro` a pelo usa el
  Python del sistema y, peor, DVC borra los `outs` antes de ejecutar la stage
- **Stack local** (Kafka KRaft de un nodo y Redis en Docker): `.\infra\local\Invoke-LocalStack.ps1 -Up` /
  `-Down`; con `-Observability` anade Prometheus y Grafana
- **Verificacion de la Fase 2**: `.\infra\scripts\Verify-Phase2.ps1` (salida 0 todos cumplidos, 1 algun
  fallado, 2 alguno sin medir, 3 error del verificador). `-SkipCluster` evita kubectl; `-FromScratch` rehace
  el pipeline en una copia temporal (minutos, necesita red)
- **Puerto 8000 del host**: puede estar ocupado por otro proyecto; los servicios en el host fallan con
  `EXIT_RUNTIME`. Usar una copia de `params.yaml` con otro `health_port` y `--params`
- **Suspension de Windows**: la suspension moderna (Kernel-Power 506/507) falsea las mediciones largas;
  contrastarlas con esos eventos antes de darlas por buenas
- **`.gitignore` y JSON**: la regla `*.json` es deliberada (claves GCP). Todo directorio con JSON fuente
  necesita su excepcion explicita (`observability/**`, `metrics/*.json`)
- **Terraform**: los errores del LSP en VSCode son falsos positivos; recargar ventana
- **Region GCP**: `europe-southwest1`
- **Project ID GCP**: `sentinel-platform-485714` (nombre del proyecto: `reactorguard-platform`)
- **Backend de estado**: bucket `reactorguard-terraform-state`

---

*ReactorGuard Progreso · Basado en TDD v1.0*
