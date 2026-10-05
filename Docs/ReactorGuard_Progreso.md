# ReactorGuard — Registro de Progreso
**Nuclear Reactor Anomaly Detection Platform**
*Actualizado: 2026-10-05 — estado verificado contra el codigo en el commit `98cd321` (2026-07-26)*

Este documento distingue tres cosas que antes se confundian: **codigo escrito**, **verificado en local**
y **criterio medido en el entorno real**. Una tarea puede estar completada como codigo y aun asi no
tener su criterio de exito medido.

---

## Estado General

| Fase | Nombre | Semanas | Estado |
|------|--------|---------|--------|
| 1 | Foundation – GKE + Kafka | 1–2 | Codigo completo, **sin desplegar**; criterios sin medir |
| 2 | Data Pipeline + Sensor Validator | 3–4 | **Parcial**: camino local completo; faltan las piezas que requieren Kafka/GCP |
| 3 | Simulation Layer (OpenMC + PyDy) | 5 | No iniciada |
| 4 | ML Core – PINN Training | 6–7 | No iniciada (solo andamiaje) |
| 5 | Uncertainty – BNN + Conformal Prediction | 8 | No iniciada |
| 6 | API + K8s Serving | 9–10 | No iniciada (solo andamiaje, 2 de 4 endpoints devuelven 501) |
| 7 | CI/CD + Security Gates | 11 | Solo esqueleto (`ci.yaml`, `terraform.yaml`, `dependabot.yml`) |
| 8 | Observability Stack | 12 | No iniciada (`observability/` vacio) |

### Linea base medida (2026-10-05)

```
pytest tests/unit tests/safety --cov --cov-fail-under=70
  585 passed, 3 skipped        cobertura TOTAL 98,79%   (1568 sentencias, 19 sin cubrir)
ruff check .                   All checks passed
mypy ml/ api/ data/            no issues found in 37 source files

dvc.yaml                       download_tep -> explore_tep -> adapt_tep -> featurize -> split

data/processed/tep             22 particiones   550.160 lecturas
data/processed/features        22 particiones   550.160 filas x 31 columnas
data/processed/splits          train 385.112 / val 82.524 / test 82.524
```

Los 3 tests omitidos son la suite de `tests/safety/test_safety_constraints.py`, marcada `skip`
a proposito hasta la Fase 3 (depende de `reactor_simulator.py`, roto contra el schema canonico).

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

- **`k8s/base/kustomization.yaml` no incluye `kafka/`** (ni `kafka-cluster.yaml`, ni `kafka-topics.yaml`, ni
  el nuevo `kafka-users.yaml`). Hoy `kubectl apply -k k8s/base/` no crearia el clúster ni los topics.
  El handoff del Paso 4 asume que si. Hay que anadir los recursos o aplicar los ficheros a mano.
- **Region**: Terraform y `bootstrap.ps1` usan `europe-southwest1`. Este documento decia `europe-west1`
  hasta hoy; el valor correcto es `europe-southwest1`.
- **Project ID**: el ID real es `sentinel-platform-485714`. `reactorguard-platform` es solo el nombre del
  proyecto y no existe como ID de GCP.

---

## Fase 2 — Data Pipeline + Sensor Validator
**Semanas 3–4** · Parcial: todo lo que corre en local esta hecho; faltan las piezas que necesitan Kafka y GCP

### SEMANA 3 — TEP Dataset + Sensor Validator

| ID | Tarea | Estado | Notas |
|----|-------|--------|-------|
| T3.1 | Schema Pydantic de sensor reading | Completado | `data/schemas/sensor_reading.py`, cobertura 100%. Serializacion `to_kafka_bytes` / `from_kafka_bytes` |
| T3.2 | Descarga y exploracion del TEP | Completado | `tep_downloader`, `tep_explorer`, `tep_loader`, `tep_adapter`, `adapt_tep`. 22 ficheros, 550.160 lecturas. `reading_id` determinista (uuid5) |
| T3.3 | `tep_streamer.py`: CSV -> productor Kafka | **Pendiente** | Clave de particion = `sensor_id` (condicion de correccion, ver Handoff_Paso4 seccion 7) |
| T3.4 | `sensor_validator.py`: 5 detectores | Completado | stuck, rate-of-change, range, cross-correlation, Kalman. Cobertura 99%. Spans por sensor con dos rangos (`configs/sensor_spans.yaml`) |
| T3.5 | Consumidor Kafka: validacion -> `sensor-validated` | **Pendiente** | Aqui `get_metrics()` pasa a metricas Prometheus y se expone `/metrics` |
| T3.6 | Tests del validador con datos TEP reales | Completado | `test_validator_on_tep.py` (15 tests). Inyeccion sintetica con semilla fija sobre d00 y validacion cruzada con el unico positivo real (XMV-04 en d21) |

### SEMANA 4 — Feature Engineering + Feast + DVC

| ID | Tarea | Estado | Notas |
|----|-------|--------|-------|
| T4.1 | `kalman.py`: filtro Kalman online por sensor | Completado | `OnlineKalmanFilter`, `KalmanFilterBank`, cobertura 100% |
| T4.2 | `features/pipeline.py` + `batch_featurizer.py` | Completado | 7 grupos de features del TDD 5.5, ventanas en muestras (`[10, 20, 60]`), salida larga con `sensor_id` |
| T4.3 | Feast feature store | **Pendiente** | TTL del prompt original estan pensados para 1 s: recalibrar a la cadencia de 180 s |
| T4.4 | DVC: `download_tep -> explore_tep -> adapt_tep -> featurize -> split` | Completado | Reproducible, la segunda ejecucion salta las 5 stages. No incluye `simulate` (Fase 3). Usar siempre `Invoke-Pipeline.ps1`, no `dvc repro` a pelo |
| T4.5 | Particionamiento GCS (`GCSStorageClient` + `LocalCache`) | **Pendiente** | |
| T4.6 | Dashboard Grafana del pipeline de datos | **Pendiente** | Depende de las metricas de T3.5. La excepcion `!observability/**/*.json` de `.gitignore` ya existe |
| T4.7 | `Verify-Phase2.ps1` | **Pendiente** | Debe reportar tres estados: CUMPLIDO / FALLADO / NO MEDIDO (los tests de T3.6 se saltan, no fallan, si falta `data/processed/tep`) |

### Criterios de Exito Fase 2

| Metrica | Umbral | Estado |
|---------|--------|--------|
| Throughput Kafka pipeline | > 50.000 readings/s sostenidos | **Pendiente**: requiere clúster y T3.3 |
| Stuck sensor precision | > 95% | **Cumplido: 1,0000** (`tests/results/validator_metrics_tep.json`, puerta en el propio test) |
| Feature pipeline latency | < 100 ms por ventana bajo carga | **Pendiente**: sin medir bajo carga |
| `dvc repro` reproducible desde cero | Sin errores | **Cumplido y demostrado** |

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
3. **Falsos positivos del residual de Kalman: 3,59%** sobre sensores intactos (colas pesadas). Por eso
   `kalman_anomaly` es senal de apoyo, sin gate. Sobre 550.160 lecturas son unas 19.700 alertas: dimensionar
   `anomaly-alerts` con este numero.
4. **Positivos reales de fallo de instrumento en el TEP: casi inexistentes** (solo XMV-04 en d21, 0,087%).
   El plan original que daba d14/d15 como positivos de stuck era incorrecto.
5. **Estado en memoria por sensor.** Los detectores acumulan estado: sin particionar por `sensor_id`, el
   sensor congelado deja de detectarse en silencio. Un rebalanceo de Kafka pierde ese estado (unos 24 min a
   cadencia TEP para volver a detectar). Persistirlo seria un rediseno: decision pendiente.
6. **El split es temporal dentro de cada `fault_type`.** `temporal_order_holds` es true por clase; en
   agregado no se cumple ni puede (d00 tiene 500 timesteps y el resto 480).

---

## Fase 3 — Simulation Layer: OpenMC + PyDy
**Semana 5** · No iniciada

`data/generators/reactor_simulator.py` existe pero esta **roto contra el schema canonico** y excluido de
mypy y de la cobertura. Hay que reescribirlo, no repararlo.

| ID | Tarea | Estado |
|----|-------|--------|
| T5.1 | OpenMC configurado: modelo PWR simplificado | Pendiente |
| T5.2 | Modelo termodinamico PyDy: ODEs coolant/core | Pendiente |
| T5.3 | `reactor_simulator.py`: neutronica + termica integradas | Pendiente (el fichero actual es un esqueleto roto) |
| T5.4 | `fault_injector.py`: drift, stuck, noise spike, bias + 4 tipos adicionales | **Parcial**: `data/validation/fault_injector.py` ya cubre los 4 tipos base sobre TEP (T3.6); faltan los 4 fisicos |
| T5.5 | DVC stage `simulate` con labels | Pendiente |
| T5.6 | Validacion de constraints fisicos (Appendix B TDD) | Pendiente (su suite en `tests/safety` esta en `skip`) |

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
**Semana 12** · No iniciada (`observability/` sin contenido salvo `.gitkeep`)

| ID | Tarea | Estado |
|----|-------|--------|
| T12.1 | Prometheus: 8 metricas custom + reglas de alerting | Pendiente |
| T12.2 | Grafana: 4 dashboards | Pendiente |
| T12.3 | Alertmanager: CRITICAL -> PagerDuty, WARNING -> Slack | Pendiente |
| T12.4 | Jaeger + OpenTelemetry Collector | Pendiente |
| T12.5 | Runbooks en `docs/runbooks/` | Pendiente |
| T12.6 | SLO dashboard 99,99% | Pendiente |

---

## Siguiente paso recomendado

1. **Trabajo local, sin coste GCP:** T3.3 (streamer), T3.5 (consumidor + metricas Prometheus), T4.5
   (`LocalCache`), T4.3 (Feast con TTL recalibrados), T4.6 (dashboard), T4.7 (`Verify-Phase2.ps1`).
2. **Despliegue en GCP con intervencion humana** (autenticacion, bucket de estado, revisar `terraform plan`
   y coste antes de `apply`): cierra los criterios pendientes de Fase 1 y los criterios 1 y 3 de Fase 2.
   Corregir antes el hueco de `kustomization.yaml` (ver deuda de Fase 1). Destruir la infraestructura al terminar.
3. Fases 3 y 4 solo cuando se indique explicitamente.

Detalle de las tareas pendientes de Fase 2 en [Handoff_Paso4.md](Handoff_Paso4.md) (secciones 6 y 7).

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
- **`.gitignore` y JSON**: la regla `*.json` es deliberada (claves GCP). Todo directorio con JSON fuente
  necesita su excepcion explicita (`observability/**`, `metrics/*.json`)
- **Terraform**: los errores del LSP en VSCode son falsos positivos; recargar ventana
- **Region GCP**: `europe-southwest1`
- **Project ID GCP**: `sentinel-platform-485714` (nombre del proyecto: `reactorguard-platform`)
- **Backend de estado**: bucket `reactorguard-terraform-state`

---

*ReactorGuard Progreso · Basado en TDD v1.0*
