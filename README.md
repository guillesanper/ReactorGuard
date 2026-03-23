# ReactorGuard

**Cloud-native anomaly detection platform for nuclear reactors.**

ReactorGuard ingests real-time sensor streams from nuclear reactor instrumentation,
applies physics-informed machine learning to detect anomalies, and surfaces safety
alerts with uncertainty-quantified predictions. The platform is built for regulatory
auditability and operates on Google Kubernetes Engine.

---

## Overview

| Component | Technology |
|-----------|------------|
| Stream ingestion | Apache Kafka (Strimzi on GKE) |
| ML model | Physics-Informed Neural Network (PyTorch) |
| Uncertainty quantification | MAPIE (conformal prediction) |
| Feature store | Feast (GCS offline, BigQuery online) |
| Experiment tracking | MLflow |
| API | FastAPI + OpenTelemetry |
| Infrastructure | GKE + Terraform |
| Observability | Prometheus + Grafana + Alertmanager |

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Runtime |
| Terraform | ≥ 1.6.0 | Infrastructure provisioning |
| kubectl | ≥ 1.28 | Kubernetes management |
| Helm | ≥ 3.14 | Kubernetes package manager |
| gcloud CLI | latest | GCP authentication |
| Docker | ≥ 24 | Container build |
| DVC | ≥ 3.49 | Data versioning |

GCP access requirements:
- Project: `reactorguard-platform`
- Role: `Owner` (or a custom role with `container.admin`, `storage.admin`, `iam.admin`)
- Billing account linked

---

## Quick Start

### 1. Bootstrap GCP project

```bash
# Authenticate with GCP
gcloud auth application-default login
gcloud config set project reactorguard-platform

# Create Terraform state bucket and enable required APIs
bash infra/scripts/bootstrap.sh
```

### 2. Provision infrastructure

```bash
cd infra/terraform/environments/dev
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

### 3. Configure kubectl

```bash
gcloud container clusters get-credentials reactorguard-cluster \
  --region europe-west1 \
  --project reactorguard-platform
```

### 4. Install Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 5. Deploy Kafka (Strimzi)

```bash
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi-operator strimzi/strimzi-kafka-operator \
  --namespace kafka --create-namespace
kubectl apply -k k8s/overlays/dev
```

### 6. Run data simulation

```bash
python -m data.generators.reactor_simulator --config params.yaml
```

### 7. Train model

```bash
dvc repro
# or manually:
python -m ml.training.train --params params.yaml
```

### 8. Start API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8080
```

### 9. Run tests

```bash
pytest tests/unit/
pytest tests/integration/
pytest tests/safety/
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GKE Cluster                               │
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │   Reactor    │───▶│    Kafka     │───▶│  Ingestion       │   │
│  │   Sensors    │    │  (Strimzi)   │    │  Service         │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│                      ┌─────────────────────────────▼──────────┐  │
│                      │          Feature Pipeline               │  │
│                      │  (Feast · feature extraction · cache)   │  │
│                      └─────────────────────────────┬──────────┘  │
│                                                    │             │
│  ┌──────────────────────────────────────────────────▼──────────┐  │
│  │                    ML Serving (PINN)                         │  │
│  │  Physics-Informed Neural Network + MAPIE uncertainty         │  │
│  │  node-pool: ml-serving (n1-standard-8, non-preemptible)      │  │
│  └──────────────────────────────────┬───────────────────────────┘  │
│                                     │                           │
│  ┌──────────────────────────────────▼───────────────────────────┐  │
│  │                     FastAPI Gateway                           │  │
│  │         /predict · /health · /metrics · /explain             │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Observability Stack                             │   │
│  │       Prometheus → Grafana · Alertmanager → PagerDuty        │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

GCS: Terraform state · DVC data · MLflow artifacts · Feast offline store
BigQuery: Feast online store · audit logs
Secret Manager: credentials · API keys
```

---

## Repository Structure

```
.
├── .github/workflows/      # CI/CD pipelines
├── infra/
│   ├── terraform/
│   │   ├── modules/        # vpc, gke, kafka, storage, iam, security
│   │   └── environments/   # dev, staging, prod
│   └── scripts/            # bootstrap, helpers
├── k8s/
│   ├── base/               # ingestion, ml, observability, rbac
│   └── overlays/           # dev, staging, prod (kustomize)
├── api/                    # FastAPI application
├── ml/                     # ML pipeline: features, models, training, serving
├── data/                   # Schemas, simulation generators, validators
├── observability/          # Prometheus rules, Grafana dashboards, Alertmanager
├── tests/                  # unit, integration, safety
└── docs/runbooks/          # Operational runbooks
```

---

## License

Proprietary — all rights reserved.
