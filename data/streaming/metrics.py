"""Prometheus metrics of the streaming services, in one place.

Un unico modulo con todos los nombres (D9, D14) para que el dashboard y el codigo
no puedan divergir: un test del dashboard comprueba que cada `reactorguard_*`
citado en una expresion PromQL esta en METRIC_NAMES.

Cada StreamMetrics crea su propio CollectorRegistry: el registro global de
prometheus_client colisiona entre tests (decision 7 del handoff) y ataria los
modulos a un backend. Las etiquetas son de baja cardinalidad (nunca sensor_id).
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

STREAM_MESSAGES_PRODUCED = "reactorguard_stream_messages_produced_total"
STREAM_PRODUCE_ERRORS = "reactorguard_stream_produce_errors_total"
STREAM_PRODUCE_LATENCY = "reactorguard_stream_produce_latency_seconds"
READINGS_CONSUMED = "reactorguard_readings_consumed_total"
READINGS_VALIDATED = "reactorguard_readings_validated_total"
SENSOR_FAULTS = "reactorguard_sensor_faults_total"
ALERTS_PUBLISHED = "reactorguard_alerts_published_total"
POISON_MESSAGES = "reactorguard_poison_messages_total"
VALIDATION_LATENCY = "reactorguard_validation_latency_seconds"
BATCH_SIZE = "reactorguard_batch_size"
CONSUMER_LAG = "reactorguard_consumer_lag"
CONSUMER_REBALANCES = "reactorguard_consumer_rebalances_total"
DETECTOR_STATE_RESETS = "reactorguard_detector_state_resets_total"
COMMIT_FAILURES = "reactorguard_commit_failures_total"
CONSUMER_HEARTBEAT = "reactorguard_consumer_heartbeat_timestamp_seconds"

METRIC_NAMES: tuple[str, ...] = (
    STREAM_MESSAGES_PRODUCED,
    STREAM_PRODUCE_ERRORS,
    STREAM_PRODUCE_LATENCY,
    READINGS_CONSUMED,
    READINGS_VALIDATED,
    SENSOR_FAULTS,
    ALERTS_PUBLISHED,
    POISON_MESSAGES,
    VALIDATION_LATENCY,
    BATCH_SIZE,
    CONSUMER_LAG,
    CONSUMER_REBALANCES,
    DETECTOR_STATE_RESETS,
    COMMIT_FAILURES,
    CONSUMER_HEARTBEAT,
)

# De 50 us a 250 ms. El validador mide media 0,078 ms, p50 0,066 ms y p99 0,23 ms
# por lectura: la resolucion hace falta en el extremo bajo, y 250 ms cubre una
# publicacion lenta. Por encima cae en +Inf.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.00005,
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
)

# Registros por lote: el techo es max_poll_records (500 por defecto).
BATCH_SIZE_BUCKETS: tuple[float, ...] = (1, 5, 10, 25, 52, 100, 250, 500, 1000)

# Valores de la etiqueta `reason` de detector_state_resets_total.
RESET_REASONS: tuple[str, ...] = ("run_change", "time_regression", "rebalance")


class StreamMetrics:
    """All streaming metrics, registered in a registry owned by the instance."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        """Register every metric.

        Args:
            registry: Registry to use. A fresh one is created when omitted, so
                two instances never collide.
        """
        self.registry = CollectorRegistry() if registry is None else registry

        self.messages_produced = Counter(
            STREAM_MESSAGES_PRODUCED,
            "Messages handed to the producer.",
            ["topic"],
            registry=self.registry,
        )
        self.produce_errors = Counter(
            STREAM_PRODUCE_ERRORS,
            "Messages whose delivery failed.",
            registry=self.registry,
        )
        self.produce_latency = Histogram(
            STREAM_PRODUCE_LATENCY,
            "Seconds from handing a batch to the producer until its flush returns.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.readings_consumed = Counter(
            READINGS_CONSUMED,
            "Readings consumed from the raw topic.",
            registry=self.registry,
        )
        self.readings_validated = Counter(
            READINGS_VALIDATED,
            "Readings validated, by resulting quality.",
            ["quality"],
            registry=self.registry,
        )
        self.sensor_faults = Counter(
            SENSOR_FAULTS,
            "Sensor faults detected, by fault type.",
            ["fault_type"],
            registry=self.registry,
        )
        self.alerts_published = Counter(
            ALERTS_PUBLISHED,
            "Alert events published, by severity.",
            ["severity"],
            registry=self.registry,
        )
        self.poison_messages = Counter(
            POISON_MESSAGES,
            "Messages that could not be deserialized and were skipped.",
            registry=self.registry,
        )
        self.validation_latency = Histogram(
            VALIDATION_LATENCY,
            "Seconds spent validating one reading.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            BATCH_SIZE,
            "Records processed per consumer batch.",
            buckets=BATCH_SIZE_BUCKETS,
            registry=self.registry,
        )
        self.consumer_lag = Gauge(
            CONSUMER_LAG,
            "Records still to be consumed, by partition.",
            ["partition"],
            registry=self.registry,
        )
        self.consumer_rebalances = Counter(
            CONSUMER_REBALANCES,
            "Consumer group rebalances seen by this consumer.",
            registry=self.registry,
        )
        self.detector_state_resets = Counter(
            DETECTOR_STATE_RESETS,
            "Per-sensor detector state resets, by reason.",
            ["reason"],
            registry=self.registry,
        )
        self.commit_failures = Counter(
            COMMIT_FAILURES,
            "Offset commits that failed.",
            registry=self.registry,
        )
        self.consumer_heartbeat = Gauge(
            CONSUMER_HEARTBEAT,
            "Unix time of the last loop iteration.",
            registry=self.registry,
        )

    def render(self) -> bytes:
        """Serialize the registry in the Prometheus text exposition format.

        Returns:
            The exposition payload.
        """
        return generate_latest(self.registry)
