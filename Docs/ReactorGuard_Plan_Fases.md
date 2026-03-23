# ReactorGuard – Plan de Fases del Proyecto
**Nuclear Reactor Anomaly Detection Platform**  
*Basado en TDD v1.0 · Enero 2025*

---

## Resumen Ejecutivo

ReactorGuard se desarrolla en **8 fases durante 12 semanas**. Cada fase construye sobre la anterior con entregables concretos, criterios de éxito medibles y riesgos identificados.

| # | Fase | Semanas | Entregable Principal | Criterio de Éxito |
|---|------|---------|----------------------|-------------------|
| 1 | Foundation – GKE + Kafka | 1–2 | Cluster GKE + Strimzi operativos | Kafka produce/consume < 10ms p99 |
| 2 | Data Pipeline + Sensor Validator | 3–4 | TEP → Kafka → Feast features | Pipeline > 50k readings/s |
| 3 | Simulation Layer (OpenMC + PyDy) | 5 | Generador de datos etiquetados con fault injection | Cobertura > 95% fault modes del TDD |
| 4 | ML Core – PINN Training | 6–7 | PINN entrenado + validación física + MLflow | Physics constraints > 99%, recall CRITICAL > 99.9% |
| 5 | Uncertainty – BNN + Conformal Prediction | 8 | BNN + Conformal Prediction calibrado | Coverage ≥ 95%, ECE < 0.05 |
| 6 | API + K8s Serving | 9–10 | FastAPI con SHAP + despliegue K8s | p99 < 50ms, 3 réplicas, PDB activo |
| 7 | CI/CD + Security Gates | 11 | GitHub Actions con ML safety gates completos | 0 deploys sin pasar safety checks |
| 8 | Observability Stack | 12 | Prometheus + Grafana + Jaeger + Alerting | SLO 99.99% monitorizado, tracing activo |

---

## Principios Transversales de Desarrollo

Estos principios aplican a todas las fases del proyecto sin excepcion. Cualquier entregable de codigo, configuracion o documentacion debe cumplirlos.

### Arquitectura y Calidad de Codigo

- **Arquitectura limpia y separacion de responsabilidades**: cada modulo tiene una unica responsabilidad bien definida. El dominio no depende de frameworks ni de infraestructura. Las capas se comunican a traves de interfaces explicitas.
- **Patrones de diseno**: usar patrones establecidos (Repository, Factory, Strategy, Adapter) donde simplifiquen el codigo. No aplicarlos mecanicamente si no aportan claridad.
- **Codigo eficiente, mantenible y escalable**: evitar optimizaciones prematuras, pero escribir codigo que no necesite reescribirse al crecer la carga o el equipo.
- **Sin emojis**: la documentacion, comentarios de codigo, logs y mensajes de commit no deben contener emojis. El tono es tecnico y profesional.
- **Documentacion suficiente**: documentar decisiones de diseno no obvias, contratos de interfaces publicas y comportamiento esperado en casos limite. No documentar lo que el codigo ya expresa con claridad.
- **Tests como ciudadanos de primera clase**: cada modulo nuevo incluye tests unitarios. Los tests de integracion cubren los flujos criticos definidos en cada fase.

### Entorno de Desarrollo

- **Sistema operativo**: Windows 11. Todos los scripts de automatizacion, setup, despliegue local y utilidades deben estar en **PowerShell (.ps1)**. No se crean scripts `.sh` para tareas que se ejecuten en la maquina de desarrollo.
- **Scripts bash**: unicamente en contextos donde el runtime es Linux de forma garantizada (dentro de contenedores Docker, GitHub Actions con `runs-on: ubuntu-latest`, o nodos GKE). En esos casos se documenta explicitamente el contexto de ejecucion.

### Identificadores del Proyecto

- **Nombre del proyecto**: `reactorguard-platform` — usado en nombres de recursos, namespaces K8s, buckets GCS y referencias internas.
- **Google Cloud Project ID**: `sentinel-platform-485714` — ID real usado en todos los comandos `gcloud`, referencias Terraform (`project = "sentinel-platform-485714"`), y URLs de consola GCP. No confundir con el nombre del proyecto.

---

## Fase 1 — Foundation: GKE + Kafka
**Semanas 1–2**

### Objetivo
Establecer la infraestructura cloud-native completa sobre GCP: redes privadas, GKE, almacenamiento, seguridad y el broker Kafka operativo vía Strimzi. Es el cimiento sobre el que correrá todo el sistema.

### Semana 1 — Infraestructura GCP con Terraform
- Usar proyecto GCP `reactorguard-platform` (ID: `sentinel-platform-485714`) y habilitar APIs: `container`, `compute`, `storage`, `secretmanager`, `pubsub`
- Terraform módulo VPC `10.0.0.0/16` con subnets privadas para GKE nodes, pods y servicios
- Terraform GKE privado `reactorguard-cluster` con dos node pools:
  - `platform`: e2-standard-4 (cargas de orquestación general)
  - `ml-serving`: n1-standard-8, **preemptible=OFF** (crítico para seguridad)
- Terraform 4 buckets GCS: raw data, processed data, models, mlflow
- Cloud NAT + Router para tráfico de salida desde nodos privados
- Cloud Armor WAF con reglas OWASP activadas
- HTTPS LB + Identity-Aware Proxy (IAP) como punto de entrada

### Semana 2 — Plataforma Base y Seguridad
- Desplegar namespaces K8s: `reactorguard-ingestion`, `reactorguard-ml`, `reactorguard-observability`, `kafka-operator`
- Aplicar NetworkPolicies por namespace según matriz TDD sección 8.1
- Configurar Secret Manager: credenciales SCADA, API keys, JWT secret
- KMS: habilitar CMEK para encriptación de buckets GCS
- IAM Workload Identity por service account (una cuenta por servicio)
- Binary Authorization: política de allow only signed images
- Skeleton CI/CD en GitHub Actions: lint → type check → unit tests → build Docker → push GCR
- Instalar Strimzi Kafka operator vía Helm v0.39.0 en `kafka-operator`
- Desplegar Kafka cluster: 3 brokers, 3 Zookeepers, replication factor 3, `min.insync.replicas=2`
- Crear topics: `sensor-readings-raw`, `sensor-validated`, `anomaly-alerts` (12 particiones c/u)
- Configurar Managed Prometheus en GKE

### Entregables
- GKE cluster operativo y accesible vía `kubectl`
- Kafka cluster con 3 brokers respondiendo a produce/consume
- 4 buckets GCS creados y accesibles con Workload Identity
- Secret Manager con secretos de prueba cargados
- Pipeline CI/CD ejecutando lint + test + build en cada push
- Terraform state en GCS con locking habilitado
- Todos los namespaces K8s con NetworkPolicies aplicadas

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| Kafka produce/consume latency p99 | < 10ms en red interna |
| GKE cluster status | RUNNING, todos los nodos Ready |
| `terraform validate` en CI | 0 errores |
| Trivy scan imágenes base | 0 vulnerabilidades CRITICAL |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| Cuotas GCP insuficientes (CPUs, IPs) | ALTO | Solicitar quota increase antes de comenzar. Tener región backup. |
| Complejidad Strimzi en GKE Autopilot | MEDIO | Usar GKE Standard, no Autopilot. Strimzi requiere control granular. |
| Costes GCP superiores a lo esperado | MEDIO | Activar billing alerts desde día 1. Usar preemptible para non-ml pools. |

---

## Fase 2 — Data Pipeline + Sensor Validator
**Semanas 3–4**

### Objetivo
Construir el pipeline completo de ingesta desde el dataset Tennessee Eastman Process (TEP) hasta el feature store Feast, incluyendo el validador de sensores que actúa como pre-filtro ante cualquier modelo ML.

### Semana 3 — TEP Dataset + Sensor Validator
- Descargar y explorar TEP: 52 variables de proceso, 21 tipos de fallo
- Implementar `tep_streamer.py`: lector CSV → productor Kafka simulando streaming continuo
- Implementar `sensor_reading.py`: schema Pydantic completo según TDD sección 4.2
- Implementar `sensor_validator.py` con los 5 detectores del TDD:
  - **Stuck value detector**: rolling variance < threshold por N samples consecutivos
  - **Rate-of-change detector**: dX/dt > physical_maximum (ej. 50°C/s)
  - **Range validator**: X < X_min o X > X_max contra límites físicos
  - **Cross-correlation checker**: Pearson(sensor_i, sensor_j) vs baseline
  - **Kalman filter residual**: |measurement – prediction| > k·σ
- Consumidor Kafka en `reactorguard-ingestion` que aplica el validador y publica en `sensor-validated`
- Tests unitarios para cada detector con casos límite explícitos

### Semana 4 — Feature Engineering + Feast + DVC
- Implementar `features/pipeline.py` con todas las features de TDD sección 5.5:
  - `rolling_mean`, `rolling_std`: ventanas 60s, 5min, 1h
  - `rate_of_change`: derivadas a 1s y 10s
  - `kalman_residual`: desviación del filtro Kalman online
  - `cross_correlation_{i,j}`: Pearson en ventana 5min
  - `variance_ratio`: ratio ventana corta/larga
  - `zero_crossing_rate`: detección de oscilaciones en 60s
  - `stuck_score`: rolling variance < epsilon en 10s
  - `hour_of_day`, `power_level_pct`: contexto operacional
- Implementar `kalman.py`: filtro Kalman por sensor con actualización online
- Configurar Feast feature store con ventanas rolling sobre GCS
- Configurar DVC pipeline (`dvc.yaml`): stages `simulate → featurize → train/val/test split`
- Particionamiento GCS: `plant/year/month/day/hour/` según TDD sección 4.4
- Tests de integración: TEP CSV → Kafka → Validator → Features → Feast en local

### Entregables
- Streamer TEP publicando en Kafka con throughput medido
- Sensor validator con 5 detectores y tests unitarios
- Feature pipeline con todas las features del TDD implementadas
- Feast sirviendo features históricas y online
- DVC pipeline reproducible: `dvc repro featurize` funciona desde cero
- Dataset TEP procesado en GCS con splits train/val/test
- Dashboard Grafana con métricas de Kafka (lag, throughput)

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| Throughput Kafka pipeline | > 50,000 readings/s sostenidos |
| Stuck sensor precision (test set TEP) | > 95% |
| Feature pipeline latency | < 100ms por ventana de 60s |
| `dvc repro featurize` | Reproducible desde cero sin errores |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| TEP no cubre todos los fault types del TDD | MEDIO | Complementar con NASA Prognostics en paralelo. Fase 3 añade simulación propia. |
| Feast latency demasiado alta para p99 < 50ms | ALTO | Usar Redis como backend online. Cachear features frecuentes en memoria. |
| Kalman filter diverge con ruido TEP | MEDIO | Ajustar matrices Q y R con datos de entrenamiento. Añadir reinicio automático. |

---

## Fase 3 — Simulation Layer: OpenMC + PyDy
**Semana 5**

### Objetivo
Construir el generador de datos sintéticos con física real que produce datos etiquetados con ground truth para los tipos de fallo que no existen en datasets públicos. Sin esta fase no se puede alcanzar el recall > 99.9% en CRITICAL faults.

### Semana 5 — Simulador + Fault Injector
- Instalar y configurar OpenMC; modelo simplificado de reactor PWR (geometría cilíndrica, UO2, agua moderadora)
- Modelo termodinámico con PyDy: ODEs para temperatura de coolant, transferencia de calor core → coolant → secundario
- Implementar `reactor_simulator.py` integrando neutrónica + térmica: genera series temporales de todos los sensores del TDD
- Implementar `fault_injector.py` con los 4 métodos base del TDD:
  - `inject_drift`: offset cumulativo ~0.001°C/step
  - `inject_stuck`: congela en último valor válido por N steps
  - `inject_noise_spike`: spike de k·std en sample aleatorio
  - `inject_bias`: offset sistemático desde t=25–50% del series
- Añadir fault types adicionales del TDD: core temperature excursion, coolant flow reduction, void formation
- Generador parametrizable vía `params.yaml`: `n_samples`, `fault_injection_rate`, `openmc_seed`
- DVC stage `simulate` produce `data/raw/simulation/` con labels incluidos
- Validar que la simulación cumple todos los constraints físicos del TDD Appendix B
- Dataset balanceado: ~60% normal, ~40% faults distribuidos por severity

### Entregables
- OpenMC + PyDy generando series temporales físicamente consistentes
- FaultInjector con los 8 fault types del TDD implementados y verificados
- Dataset simulado con > 95% cobertura de fault modes del TDD
- Labels ground truth por reading: `fault_type`, `severity`, `injected_at`
- DVC stage `simulate` reproducible con seed fijo
- Validación automática de constraints físicos en datos generados
- `docs/physics_constraints.md` con las ecuaciones implementadas

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| Cobertura de fault types | > 95% de los 8 tipos del TDD |
| Physics constraint compliance en datos generados | 100% |
| Volumen de datos generados | > 1M readings etiquetados |
| Reproducibilidad con mismo seed | Resultados idénticos |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| OpenMC lento para generar volumen suficiente | ALTO | Usar imagen Docker pre-configurada. Paralelizar con múltiples seeds. PyDy solo como fallback. |
| Simulación no realista vs. planta real | MEDIO | Validar contra KAERI NPP dataset (Kaggle) en Fase 2. Ajustar parámetros ODE. |
| Desequilibrio extremo de clases (pocos CRITICAL) | ALTO | Forzar `fault_injection_rate` mínimo del 5% para CRITICAL. SMOTE como backup. |

---

## Fase 4 — ML Core: PINN Training
**Semanas 6–7**

### Objetivo
Entrenar la Physics-Informed Neural Network (PINN), principal diferenciador técnico del sistema. Aprende correlaciones entre sensores mientras satisface constraints termodinámicos. Sus residuales son la señal primaria de anomalía.

### Semana 6 — Implementación y entrenamiento inicial
- Implementar `ml/models/pinn.py` completo:
  - Arquitectura: `Linear(n_sensors+4, 256) → Tanh × 3 → Linear(256, n_sensors)`
  - `energy_balance_loss`: Q_gen = flow · Cp · (T_out – T_in)
  - `mass_balance_loss`: flow_in = flow_out (estado estacionario)
  - `training_loss = data_loss + λ_energy·energy_loss + λ_mass·mass_loss`
- Configurar training loop PyTorch: Adam optimizer, LR scheduler, early stopping
- Integrar MLflow: hiperparámetros, pérdidas por epoch, métricas de validación
- Integrar DVC: stage `train` con deps en features, outs en `models/pinn/candidate/`
- Primera ejecución con dataset TEP: validar convergencia del loss
- Implementar `ml/validation/physics_check.py`: verificar > 99% de predicciones satisfacen constraints

### Semana 7 — PINN con simulación + Rule Engine + SHAP
- Reentrenar PINN con dataset de simulación (Fase 3): mayor cobertura de faults
- Ajustar λ_energy y λ_mass mediante grid search; registrar en MLflow
- Implementar `ml/models/rules.py` con los 6 constraints del TDD:
  - Energy balance: |Q_gen – Q_removed| / Q_gen > 0.05
  - Temperature monotonicity: T_core_outlet < T_core_inlet
  - Maximum dT/dt > 10°C/s
  - Flow rate range: flow < 0 o flow > pump_max
  - Power/flux correlation: |P_thermal – k·flux| > threshold
  - Pressure/temperature: P < P_sat(T) en primario
- Fusión parcial PINN + Rules: `pinn_residual * 0.5 + rule_violations * 0.5`
- Ejecutar `ml/validation/safety_check.py`: recall CRITICAL faults > 99.9%
- Implementar SHAP values: top 3 features por predicción
- Benchmark de latencia: inferencia PINN < 50ms p99 en CPU

### Entregables
- ReactorPINN entrenado con loss convergido y métricas en MLflow
- Physics constraint satisfaction > 99% en test set
- Rule Engine con los 6 constraints del TDD implementados
- Safety check: recall CRITICAL faults > 99.9% (stuck, drift, thermal excursion)
- SHAP values funcionando por predicción
- Latencia de inferencia PINN < 50ms p99
- `dvc repro train` funciona de extremo a extremo

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| Physics constraint satisfaction | > 99% en test set de simulación |
| Recall en CRITICAL faults | > 99.9% (false negative rate < 0.1%) |
| PINN inference latency p99 | < 50ms en CPU |
| MLflow experiment tracking | Todos los runs logeados |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| PINN no converge con constraint loss activado | ALTO | Curriculum: entrenar sin constraints primero, luego activar λ gradualmente (0→1 en 50 epochs). |
| Overfitting a fault types de simulación | MEDIO | Validar en KAERI NPP dataset real. Añadir dropout 0.1 en capas ocultas. |
| Latencia PINN > 50ms con muchos sensores | MEDIO | Cuantización INT8 con TorchScript. Reducir a los sensores más correlacionados. |

---

## Fase 5 — Uncertainty: BNN + Conformal Prediction
**Semana 8**

### Objetivo
Añadir cuantificación de incertidumbre explícita y calibrada. El BNN produce distribuciones de predicción y Conformal Prediction garantiza matemáticamente la cobertura declarada, independientemente de las asunciones del modelo.

### Semana 8 — BNN + Conformal Prediction + Fusion Layer
- Implementar `ml/models/bnn.py`: BayesianTorch o PyMC con inferencia variacional (ELBO loss)
- Implementar `ml/models/uncertainty.py` (UncertaintyQuantifier con MAPIE):
  - Base estimator: MLPRegressor (128, 64) o PINN adapter
  - `method='plus'`, `cv=5` para split conformal
  - `alpha=0.05` → cobertura objetivo 95%
- Implementar `calibrate_threshold`: umbral en percentil 95 de `interval_width` en calibration set
- Implementar `ml/validation/calibration_check.py`:
  - ECE (Expected Calibration Error) < 0.05
  - Coverage real ≥ 95% en test set
  - Correlación entre `interval_width` y error real
- Actualizar `ml/models/fusion.py` con ensemble completo:
  - `anomaly_score = pinn_residual * 0.35 + bnn_uncertainty * 0.35 + rule_violations * 0.30`
  - `IF rule_violation == 1 → severity = CRITICAL` (override ML score)
- Logging de intervalos de confianza en MLflow: coverage, ECE por epoch
- Validar `uncertainty_flag` en fault scenarios: alta incertidumbre debe correlacionar con anomalías

### Entregables
- BNN entrenado con inferencia variacional funcional
- Conformal Prediction con cobertura garantizada ≥ 95%
- ECE < 0.05 en calibration set
- Fusion layer completo: PINN + BNN + Rules con pesos del TDD
- `uncertainty_flag` activándose correctamente en alta incertidumbre
- Response JSON completo según schema TDD sección 6.2
- Calibration check en pipeline CI como validation step

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| Conformal Prediction coverage | ≥ 95% en test set |
| Expected Calibration Error (ECE) | < 0.05 |
| Correlación interval_width vs error real | > 0.6 Pearson |
| anomaly_score en CRITICAL faults | > 0.8 en > 99.9% de casos |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| BNN entrenamiento lento sin GPU | MEDIO | Usar Laplace Approximation como alternativa ligera. MAPIE es CPU-friendly. |
| Intervalos de confianza excesivamente anchos | MEDIO | Reducir ventana de calibración. Separar conformal por tipo de sensor. |
| ECE > 0.05 en datos reales vs. simulación | ALTO | Recalibrar threshold en KAERI dataset. Añadir recalibración periódica en el pipeline. |

---

## Fase 6 — API + K8s Serving
**Semanas 9–10**

### Objetivo
Exponer los modelos ML como una API REST de baja latencia desplegada en Kubernetes con alta disponibilidad garantizada mediante PodDisruptionBudgets. Convierte el código de investigación en un servicio de producción.

### Semana 9 — FastAPI Implementation
- Implementar `api/main.py` con FastAPI + Uvicorn; startup carga modelos desde GCS
- Implementar todos los endpoints del TDD sección 6.1:
  - `POST /v1/analyze`: pipeline completo (validación → features → PINN → BNN → fusion), < 50ms
  - `POST /v1/analyze/batch`: hasta 500 readings, < 500ms
  - `POST /v1/sensor/validate`: solo sensor fault classifier, < 10ms
  - `POST /v1/constraints/check`: solo physics rules, < 5ms
  - `POST /v1/explain`: SHAP values por predicción, < 200ms
  - `GET /health`, `/ready`, `/metrics`: probes y Prometheus
- Response schema completo según TDD sección 6.2 (sensor_health, anomaly, physics, explanation, recommended_actions)
- Autenticación JWT con secret desde Secret Manager
- OpenTelemetry instrumentation: trace completo por request
- Tests de integración con fixtures de muestra para todos los endpoints

### Semana 10 — K8s Deployment + PDB + HPA
- Dockerfile optimizado: Python 3.11-slim, multi-stage build
- Trivy scan en CI: 0 vulnerabilidades CRITICAL
- `k8s/base/ml/pinn-server/deployment.yaml` según spec TDD sección 8.2:
  - 3 réplicas, `nodeSelector: ml-serving`, tolerations para `ml-workload`
  - Resources: request 2CPU/4Gi, limit 4CPU/8Gi
  - `startupProbe`: `/ready`, failureThreshold=30, periodSeconds=5
  - `livenessProbe`: `/health`, periodSeconds=10
  - `configMapKeyRef` para `MODEL_VERSION`
- PodDisruptionBudget con `minAvailable: 2`
- HPA: scale entre 3–10 réplicas según CPU/latency metrics
- Canary deployment: 10% de tráfico a nueva versión antes de rollout completo
- Kustomize overlays para dev/staging/prod

### Entregables
- API con todos los endpoints del TDD respondiendo en los tiempos target
- Dockerfile con Trivy scan limpio
- Deployment K8s con 3 réplicas en node pool `ml-serving`
- PDB garantizando `minAvailable: 2` en todo momento
- HPA configurado con escalado automático
- Canary deployment funcionando en staging
- OpenTelemetry traces visibles en Jaeger

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| /v1/analyze latency p99 | < 50ms bajo carga sostenida |
| /v1/sensor/validate latency p99 | < 10ms |
| Disponibilidad durante rolling update | ≥ 2 réplicas siempre activas |
| Trivy scan | 0 vulnerabilidades CRITICAL |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| Carga de modelos desde GCS en startup > 60s | ALTO | `startupProbe` con failureThreshold=30. Pre-warm con init container. |
| p99 > 50ms bajo carga concurrente | ALTO | Batching interno en el servidor. Aumentar réplicas. Cuantización del modelo. |
| Falta de memoria en ml-serving nodes (OOM) | MEDIO | Memory limits conservadores. Monitoring de OOM kills en Prometheus. |

---

## Fase 7 — CI/CD + Security Gates
**Semana 11**

### Objetivo
Cerrar el ciclo de automatización con pipelines de CI/CD que incluyen compuertas específicas de seguridad para ML: ningún modelo puede llegar a producción sin pasar los safety checks del TDD.

### Semana 11 — GitHub Actions + ML Safety Gates
- Completar `.github/workflows/ci.yaml`: unit tests → Semgrep SAST → pip-audit → Gitleaks → Trivy → build → push GCR
- Completar `.github/workflows/terraform.yaml`: fmt → validate → plan → tfsec
- Implementar `.github/workflows/ml-training.yaml` (weekly CronJob) con gates del TDD:
  - `physics_check.py --min-satisfaction 0.99`: 99% de predicciones satisfacen constraints
  - `calibration_check.py --max-ece 0.05 --coverage-target 0.95`
  - `safety_check.py --fault-types stuck,drift,thermal_excursion --min-recall 0.999`
  - Latency benchmark: p99 < 50ms
  - **ALL pass → auto-promote a producción | ANY fail → alert + block deploy**
- Implementar `.github/workflows/ml-promote.yaml`: promoción manual con aprobación
- Binary Authorization: verificar firma de imagen antes de deploy a producción
- Configurar environments de GitHub: dev (auto-deploy), staging (auto), prod (manual approval)
- Añadir `tests/safety/test_critical_faults.py`: scenarios must-catch por fault type

### Entregables
- Pipeline CI completo ejecutando en cada push con todos los security scans
- ML Training workflow semanal con los 4 safety gates del TDD
- Promote workflow con aprobación manual para producción
- Binary Authorization verificando imágenes firmadas
- `tests/safety/test_critical_faults.py` con casos explícitos por fault type
- 0 posibilidad de deploy sin pasar physics + calibration + safety checks

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| CI pipeline tiempo total | < 15 minutos |
| Safety gates bloqueando deploys defectuosos | 100% de casos negativos bloqueados |
| Semgrep + Trivy en cada PR | 0 findings HIGH/CRITICAL ignorados |
| Reproducibilidad del training pipeline | Mismos datos + seed = mismas métricas |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| ML training semanal falla por data drift | MEDIO | Alertar sin bloquear si falla por datos. Separar gate de datos vs. gate de código. |
| CI pipeline demasiado lento (>20min) | MEDIO | Paralelizar jobs. Cachear dependencias Python y Docker layers. |
| False positive en safety gates bloquea deploy urgente | BAJO | Escape hatch con aprobación de 2 revisores documentado. |

---

## Fase 8 — Observability Stack
**Semana 12**

### Objetivo
Hacer el sistema completamente observable: métricas de negocio y técnicas en Grafana, tracing distribuido en Jaeger para debugging de latencia, y alerting proactivo antes de que los problemas afecten a operadores.

### Semana 12 — Prometheus + Grafana + Jaeger + Alerting
- Desplegar Prometheus con reglas del TDD sección 10.1; configurar todas las métricas custom:
  - `reactorguard_predictions_total` (Counter)
  - `reactorguard_anomaly_score_histogram` (Histogram) — alert si p99 > 0.8 por 5min
  - `reactorguard_critical_alerts_total` (Counter) — alert en cualquier incremento
  - `reactorguard_prediction_latency_seconds` (Histogram) — alert si p99 > 0.05s
  - `reactorguard_physics_violations_total` (Counter) — alert CRITICAL en cualquier incremento
  - `reactorguard_sensor_faults_total` (Counter by fault_type) — alert si stuck > 3/hora
  - `reactorguard_model_uncertainty_mean` (Gauge) — alert si > 2x baseline
  - `reactorguard_uncertainty_calibration_ece` (Gauge) — trigger retrain si > 0.05
- Desplegar Grafana con los 4 dashboards del TDD sección 10.2:
  - **Reactor Overview**: anomaly score heatmap, active alerts por severity, physics violations rate, sensor health grid
  - **ML Model Health**: latency p50/p95/p99, uncertainty trend, ECE trend, constraint satisfaction rate
  - **Sensor Analytics**: time series con anomaly overlay, stuck count, drift trends, cross-correlation heatmap
  - **Operations**: alert volume por severity, MTTA, false positive rate, model version activa
- Configurar Alertmanager con rutas: CRITICAL → PagerDuty, WARNING → Slack
- Desplegar Jaeger + OpenTelemetry Collector; verificar traces completos:
  - Kafka consumer → Feature pipeline → PINN → BNN → Fusion → Alert
- Crear runbooks en `docs/runbooks/` para cada tipo de alerta
- SLO dashboard: 99.99% availability con error budget burn rate

### Entregables
- Prometheus scrapeando todas las métricas del TDD con reglas de alerting configuradas
- 4 dashboards Grafana funcionales con los paneles del TDD
- Alertmanager con rutas CRITICAL → PagerDuty, WARNING → Slack
- Jaeger mostrando traces end-to-end de cada request de análisis
- Runbooks para cada alerta en `docs/runbooks/`
- SLO dashboard con 99.99% availability target y error budget

### Criterios de Éxito
| Métrica | Umbral |
|---------|--------|
| Cobertura de métricas del TDD | 100% de las 8 métricas implementadas |
| Alerting end-to-end (metric → PagerDuty) | < 2 minutos de latencia |
| Jaeger trace completeness | 100% de requests con trace completo |
| SLO availability objetivo | 99.99% (< 52.6 min downtime/año) |

### Riesgos
| Riesgo | Severidad | Mitigación |
|--------|-----------|------------|
| Alert fatigue por demasiadas alertas WARNING | MEDIO | Tunear thresholds con datos reales de la Fase 6. Agrupar alertas correlacionadas. |
| Prometheus con alta cardinalidad (muchos sensores) | MEDIO | Limitar labels de alta cardinalidad. Usar recording rules para queries costosas. |
| Jaeger overhead de tracing > 1ms | BAJO | Sampling rate al 10% en producción. 100% solo en staging y debug. |

---

## Dependencias Entre Fases

```
Fase 1 (Infra)
    └── Fase 2 (Data Pipeline) ──── requiere: Kafka, GCS, GKE
         └── Fase 3 (Simulation) ── requiere: DVC pipeline, feature schema
              └── Fase 4 (PINN) ─── requiere: datos etiquetados, feature store
                   └── Fase 5 (UQ) ─ requiere: PINN entrenado
                        └── Fase 6 (API) ── requiere: todos los modelos
                             ├── Fase 7 (CI/CD) ─ requiere: API + modelos + tests
                             └── Fase 8 (Obs.) ── requiere: API desplegada en K8s
```

## Stack de Tecnologías por Fase

| Fase | Tecnologías Principales |
|------|------------------------|
| 1 | Terraform, GKE, Strimzi, Kafka, Cloud Armor, IAP, Secret Manager, KMS |
| 2 | Kafka (consumer), Pydantic, Feast, DVC, Pandas, NumPy, FilterPy (Kalman) |
| 3 | OpenMC, PyDy, NumPy, DVC, Parquet |
| 4 | PyTorch, DeepXDE, SHAP, MLflow, DVC |
| 5 | PyMC / BayesianTorch, MAPIE, MLflow |
| 6 | FastAPI, Uvicorn, OpenTelemetry, Kustomize, Docker |
| 7 | GitHub Actions, Semgrep, Trivy, Gitleaks, tfsec, Binary Authorization |
| 8 | Prometheus, Grafana, Jaeger, Alertmanager, OpenTelemetry Collector |

---

*ReactorGuard Plan de Fases · v1.0 · Basado en TDD v1.0*
