# =============================================================================
# infra/terraform/modules/kafka/outputs.tf
# =============================================================================

output "namespace" {
  description = "Namespace donde está instalado el operador Strimzi."
  value       = helm_release.strimzi.namespace
}

output "strimzi_status" {
  description = "Estado del helm release de Strimzi (deployed, failed, etc.)."
  value       = helm_release.strimzi.status
}

output "kafka_bootstrap_server" {
  description = "Bootstrap server interno para productores/consumidores Kafka."
  value       = "reactorguard-cluster-kafka-bootstrap.${helm_release.strimzi.namespace}:9092"
}
