# ReactorGuard — Registro de Progreso
**Nuclear Reactor Anomaly Detection Platform**
*Actualizado: 2026-03-23 — T2.1, T2.2, T2.3, T2.4 y T2.5 completados — Fase 1 cerrada*

---

## Estado General

| Fase | Nombre | Semanas | Estado |
|------|--------|---------|--------|
| 1 | Foundation – GKE + Kafka | 1–2 | ✅ Completado |
| 2 | Data Pipeline + Sensor Validator | 3–4 | ⬜ Pendiente |
| 3 | Simulation Layer (OpenMC + PyDy) | 5 | ⬜ Pendiente |
| 4 | ML Core – PINN Training | 6–7 | ⬜ Pendiente |
| 5 | Uncertainty – BNN + Conformal Prediction | 8 | ⬜ Pendiente |
| 6 | API + K8s Serving | 9–10 | ⬜ Pendiente |
| 7 | CI/CD + Security Gates | 11 | ⬜ Pendiente |
| 8 | Observability Stack | 12 | ⬜ Pendiente |

---

## Fase 1 — Foundation: GKE + Kafka
**Semanas 1–2**

### SEMANA 1 — Infraestructura GCP con Terraform

| ID | Tarea | Estado | Notas |
|----|-------|--------|-------|
| T1.1 | Estructura del repositorio y configuración inicial | ✅ Completado | Árbol de dirs, pyproject.toml, .gitignore, README |
| T1.2 | Módulo Terraform: Backend de estado y proyecto GCP | ✅ Completado | backend.tf, providers.tf, versions.tf, main.tf, bootstrap.ps1 |
| T1.3 | Módulo Terraform: VPC y Networking | ✅ Completado | VPC, subnet, Cloud NAT, firewall rules; `terraform validate` OK |
| T1.4 | Módulo Terraform: GKE Cluster | ✅ Completado | Cluster privado, node pools platform + ml-serving con taints |
| T1.5 | Módulo Terraform: GCS Buckets y Storage | ✅ Completado | 4 buckets con lifecycle, versionado, uniform bucket access |
| T1.6 | Módulo Terraform: IAM y Workload Identity | ✅ Completado | 4 SAs con least privilege + Workload Identity (ingestion/ml/mlflow); `terraform validate` OK |
| T1.7 | Módulo Terraform: Secret Manager y KMS | ✅ Completado | 4 secretos con placeholder + KMS key ring/key CMEK 90d rotation; `Load-Secrets.ps1`; `terraform validate` OK |
| T1.8 | Módulo Terraform: Cloud Armor WAF + Load Balancer + IAP | ✅ Completado | armor.tf + lb.tf + iap.tf; outputs lb_ip + iap_client_id; `terraform validate` OK |
| T1.9 | Aplicar Terraform y verificar infraestructura completa | ✅ Completado | `Verify-Infra.ps1` (5 checks: GKE, GCS, Secrets, Net, IAM) + `Remove-DevInfra.ps1` (con confirmación DESTROY-DEV) |

### SEMANA 2 — Plataforma Base, Seguridad y Kafka

| ID | Tarea | Estado | Notas |
|----|-------|--------|-------|
| T2.1 | Namespaces K8s y NetworkPolicies | ✅ Completado | 4 namespaces + 8 NetworkPolicies (deny-all + allow por ns); kustomization.yaml base |
| T2.2 | Workload Identity en Kubernetes (KSAs) | ✅ Completado | sensor-validator, pinn-server, mlflow-server con anotaciones WI; Roles + RoleBindings; Test-WorkloadIdentity.ps1 |
| T2.3 | Instalación Strimzi Kafka Operator v0.39.0 | ✅ Completado | helm_release Strimzi 0.39.0; KafkaCluster 3 brokers + 3 zookeepers; 3 KafkaTopics (raw/validated/alerts); `Install-Kafka.ps1` |
| T2.4 | Verificación de Kafka: latencia y throughput | ✅ Completado | `test_kafka_connectivity.py` (100 msgs); `benchmark_kafka.py` (10k msgs p99<10ms → JSON); `kafka-client-pod.yaml`; `Invoke-KafkaTests.ps1` |
| T2.5 | Skeleton CI/CD con GitHub Actions | ✅ Completado | ci.yaml (lint→tests→scan→build/push) + terraform.yaml (fmt→validate→tfsec→plan) + api/Dockerfile (multi-stage, non-root) + dependabot.yml |

### Criterios de Éxito Fase 1

| Métrica | Umbral | Estado |
|---------|--------|--------|
| Kafka produce/consume latency p99 | < 10ms en red interna | ⬜ |
| GKE cluster status | RUNNING, todos los nodos Ready | ⬜ |
| `terraform validate` en CI | 0 errores | ✅ |
| Trivy scan imágenes base | 0 vulnerabilidades CRITICAL | ⬜ |

---

## Fase 2 — Data Pipeline + Sensor Validator
**Semanas 3–4** · ⬜ Pendiente

### SEMANA 3 — TEP Dataset + Sensor Validator

| ID | Tarea | Estado |
|----|-------|--------|
| T3.1 | TEP Dataset: descarga y exploración | ⬜ Pendiente |
| T3.2 | `tep_streamer.py`: CSV → productor Kafka | ⬜ Pendiente |
| T3.3 | `sensor_reading.py`: schema Pydantic completo | ⬜ Pendiente |
| T3.4 | `sensor_validator.py`: 5 detectores (stuck, rate-of-change, range, cross-correlation, Kalman) | ⬜ Pendiente |
| T3.5 | Consumidor Kafka: validación → `sensor-validated` | ⬜ Pendiente |
| T3.6 | Tests unitarios por detector | ⬜ Pendiente |

### SEMANA 4 — Feature Engineering + Feast + DVC

| ID | Tarea | Estado |
|----|-------|--------|
| T4.1 | `features/pipeline.py`: rolling stats, rate of change, Kalman residual, correlaciones | ⬜ Pendiente |
| T4.2 | `kalman.py`: filtro Kalman online por sensor | ⬜ Pendiente |
| T4.3 | Feast feature store configurado sobre GCS | ⬜ Pendiente |
| T4.4 | DVC pipeline `dvc.yaml`: simulate → featurize → split | ⬜ Pendiente |
| T4.5 | Tests de integración end-to-end: TEP → Kafka → Features → Feast | ⬜ Pendiente |

---

## Fase 3 — Simulation Layer: OpenMC + PyDy
**Semana 5** · ⬜ Pendiente

| ID | Tarea | Estado |
|----|-------|--------|
| T5.1 | OpenMC configurado: modelo PWR simplificado | ⬜ Pendiente |
| T5.2 | Modelo termodinámico PyDy: ODEs coolant/core | ⬜ Pendiente |
| T5.3 | `reactor_simulator.py`: neutrónica + térmica integradas | ⬜ Pendiente |
| T5.4 | `fault_injector.py`: drift, stuck, noise spike, bias + 4 tipos adicionales | ⬜ Pendiente |
| T5.5 | DVC stage `simulate`: genera `data/raw/simulation/` con labels | ⬜ Pendiente |
| T5.6 | Validación de constraints físicos (Appendix B TDD) | ⬜ Pendiente |

---

## Fase 4 — ML Core: PINN Training
**Semanas 6–7** · ⬜ Pendiente

### SEMANA 6 — Implementación y entrenamiento inicial

| ID | Tarea | Estado |
|----|-------|--------|
| T6.1 | `ml/models/pinn.py`: arquitectura + energy/mass balance loss | ⬜ Pendiente |
| T6.2 | Training loop PyTorch: Adam, LR scheduler, early stopping | ⬜ Pendiente |
| T6.3 | Integración MLflow: hiperparámetros + métricas por epoch | ⬜ Pendiente |
| T6.4 | DVC stage `train` | ⬜ Pendiente |
| T6.5 | `ml/validation/physics_check.py`: > 99% constraints satisfechos | ⬜ Pendiente |

### SEMANA 7 — PINN con simulación + Rule Engine + SHAP

| ID | Tarea | Estado |
|----|-------|--------|
| T7.1 | Reentrenamiento con dataset de simulación | ⬜ Pendiente |
| T7.2 | Grid search λ_energy y λ_mass en MLflow | ⬜ Pendiente |
| T7.3 | `ml/models/rules.py`: 6 constraints físicos del TDD | ⬜ Pendiente |
| T7.4 | Fusion PINN + Rules: `pinn_residual * 0.5 + rule_violations * 0.5` | ⬜ Pendiente |
| T7.5 | `ml/validation/safety_check.py`: recall CRITICAL > 99.9% | ⬜ Pendiente |
| T7.6 | SHAP values: top 3 features por predicción | ⬜ Pendiente |
| T7.7 | Benchmark latencia: inferencia PINN < 50ms p99 CPU | ⬜ Pendiente |

---

## Fase 5 — Uncertainty: BNN + Conformal Prediction
**Semana 8** · ⬜ Pendiente

| ID | Tarea | Estado |
|----|-------|--------|
| T8.1 | `ml/models/bnn.py`: BayesianTorch / PyMC con ELBO | ⬜ Pendiente |
| T8.2 | `ml/models/uncertainty.py`: MAPIE, method='plus', alpha=0.05 | ⬜ Pendiente |
| T8.3 | `calibrate_threshold`: percentil 95 de interval_width | ⬜ Pendiente |
| T8.4 | `ml/validation/calibration_check.py`: ECE < 0.05, coverage ≥ 95% | ⬜ Pendiente |
| T8.5 | `ml/models/fusion.py` completo: PINN 0.35 + BNN 0.35 + Rules 0.30 | ⬜ Pendiente |
| T8.6 | Logging calibration en MLflow + validación `uncertainty_flag` | ⬜ Pendiente |

---

## Fase 6 — API + K8s Serving
**Semanas 9–10** · ⬜ Pendiente

### SEMANA 9 — FastAPI Implementation

| ID | Tarea | Estado |
|----|-------|--------|
| T9.1 | `api/main.py`: FastAPI + Uvicorn, carga modelos desde GCS en startup | ⬜ Pendiente |
| T9.2 | Todos los endpoints del TDD sección 6.1 (analyze, batch, validate, constraints, explain, health) | ⬜ Pendiente |
| T9.3 | Autenticación JWT con Secret Manager | ⬜ Pendiente |
| T9.4 | OpenTelemetry: trace completo por request | ⬜ Pendiente |
| T9.5 | Tests de integración para todos los endpoints | ⬜ Pendiente |

### SEMANA 10 — K8s Deployment + PDB + HPA

| ID | Tarea | Estado |
|----|-------|--------|
| T10.1 | Dockerfile multi-stage Python 3.11-slim, usuario no-root | ⬜ Pendiente |
| T10.2 | Trivy scan: 0 vulnerabilidades CRITICAL | ⬜ Pendiente |
| T10.3 | `k8s/base/ml/pinn-server/deployment.yaml`: 3 réplicas, nodeSelector, probes, resources | ⬜ Pendiente |
| T10.4 | PodDisruptionBudget: minAvailable=2 | ⬜ Pendiente |
| T10.5 | HPA: 3–10 réplicas por CPU/latency | ⬜ Pendiente |
| T10.6 | Canary deployment: 10% tráfico nueva versión | ⬜ Pendiente |
| T10.7 | Kustomize overlays: dev / staging / prod | ⬜ Pendiente |

---

## Fase 7 — CI/CD + Security Gates
**Semana 11** · ⬜ Pendiente

| ID | Tarea | Estado |
|----|-------|--------|
| T11.1 | `ci.yaml` completo: Semgrep + Gitleaks + pip-audit + Trivy | ⬜ Pendiente |
| T11.2 | `terraform.yaml`: fmt + validate + tfsec + plan en PR | ⬜ Pendiente |
| T11.3 | `ml-training.yaml` (weekly): physics_check + calibration_check + safety_check + latency gate | ⬜ Pendiente |
| T11.4 | `ml-promote.yaml`: promoción manual con aprobación a producción | ⬜ Pendiente |
| T11.5 | Binary Authorization: solo imágenes firmadas en producción | ⬜ Pendiente |
| T11.6 | GitHub environments: dev (auto) / staging (auto) / prod (manual approval) | ⬜ Pendiente |
| T11.7 | `tests/safety/test_critical_faults.py`: casos must-catch por fault type | ⬜ Pendiente |

---

## Fase 8 — Observability Stack
**Semana 12** · ⬜ Pendiente

| ID | Tarea | Estado |
|----|-------|--------|
| T12.1 | Prometheus: 8 métricas custom del TDD + reglas de alerting | ⬜ Pendiente |
| T12.2 | Grafana: 4 dashboards (Reactor Overview, ML Health, Sensor Analytics, Operations) | ⬜ Pendiente |
| T12.3 | Alertmanager: CRITICAL → PagerDuty, WARNING → Slack | ⬜ Pendiente |
| T12.4 | Jaeger + OpenTelemetry Collector: traces end-to-end | ⬜ Pendiente |
| T12.5 | Runbooks en `docs/runbooks/` por tipo de alerta | ⬜ Pendiente |
| T12.6 | SLO dashboard: 99.99% availability + error budget burn rate | ⬜ Pendiente |

---

## Leyenda

| Símbolo | Significado |
|---------|-------------|
| ✅ | Completado |
| 🔄 | En progreso |
| ⬜ | Pendiente |
| ❌ | Bloqueado |

---

## Notas Técnicas

- **Entorno**: Windows 11 — scripts en `.ps1` (PowerShell), no `.sh`
- **Terraform**: `terraform validate` pasa correctamente (errores de LSP en VSCode son falsos positivos — recargar ventana con `Ctrl+Shift+P → Developer: Reload Window`)
- **Región GCP**: `europe-west1`
- **Proyecto GCP**: `reactorguard-platform`
- **Backend state**: bucket `reactorguard-terraform-state`

---

*ReactorGuard Progreso · Basado en TDD v1.0*
