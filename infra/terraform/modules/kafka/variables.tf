# =============================================================================
# infra/terraform/modules/kafka/variables.tf
# =============================================================================

variable "namespace" {
  description = "Namespace de Kubernetes donde se instalará Strimzi y el cluster Kafka."
  type        = string
  default     = "kafka-operator"
}

variable "strimzi_version" {
  description = "Versión del chart strimzi-kafka-operator. TDD sección 7.2 especifica 0.39.0."
  type        = string
  default     = "0.39.0"
}

variable "timeout_seconds" {
  description = "Segundos de espera para que el operador Strimzi esté Ready tras el helm install."
  type        = number
  default     = 300
}
