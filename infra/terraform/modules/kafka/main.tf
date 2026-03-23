# =============================================================================
# infra/terraform/modules/kafka/main.tf
# Módulo Kafka — Strimzi Operator v0.39.0 vía Helm
#
# Strimzi es un operador de Kubernetes que gestiona el ciclo de vida de Apache
# Kafka en K8s. En lugar de desplegar Kafka manualmente (Deployments, Services,
# ConfigMaps), Strimzi introduce recursos custom (CRDs) como `kind: Kafka` y
# `kind: KafkaTopic` que el operador traduce a los recursos K8s necesarios.
#
# Flujo de responsabilidades:
#   Helm instala el operador Strimzi → el operador observa los CRDs →
#   al aplicar kafka-cluster.yaml, el operador crea los brokers y ZooKeeper →
#   al aplicar kafka-topics.yaml, el operador crea los topics dentro del cluster.
# =============================================================================

terraform {
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }
}

# -----------------------------------------------------------------------------
# helm_release "strimzi"
# Instala el chart strimzi-kafka-operator desde el repositorio oficial de Strimzi.
# version 0.39.0 es la versión especificada en el TDD sección 7.2.
# create_namespace=true crea el namespace kafka-operator si no existe, de forma
# idempotente (no falla si el namespace ya existe).
# wait=true + timeout=300 bloquea hasta que todos los pods del operador estén Ready.
# -----------------------------------------------------------------------------
resource "helm_release" "strimzi" {
  name             = "strimzi"
  repository       = "https://strimzi.io/charts/"
  chart            = "strimzi-kafka-operator"
  version          = var.strimzi_version
  namespace        = var.namespace
  create_namespace = true

  # wait=true es crítico: si Terraform continúa antes de que el operador esté
  # Ready, la aplicación de los CRDs (kafka-cluster.yaml) fallará porque los
  # webhooks de validación de Strimzi aún no están disponibles.
  wait    = true
  timeout = var.timeout_seconds

  # watchNamespaces: el operador sólo observará recursos Kafka en este namespace.
  # Limitar el scope reduce la superficie de ataque y simplifica el RBAC.
  set {
    name  = "watchNamespaces"
    value = "{${var.namespace}}"
  }

  # logLevel INFO es suficiente para producción. DEBUG genera demasiado ruido
  # en los logs y dificulta el troubleshooting real.
  set {
    name  = "logLevel"
    value = "INFO"
  }
}
