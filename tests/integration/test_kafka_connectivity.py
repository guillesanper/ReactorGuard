"""Delivery and round-trip latency against a Kafka cluster (criterio de la Fase 1).

Criterio de la Fase 1: latencia produce -> consume con p99 por debajo de 10 ms en red
interna. Este modulo lo mide, y de paso comprueba que la conexion segura funciona de
punta a punta: handshake mTLS, ACL del KafkaUser y entrega de cada mensaje.

Sustituye al script anterior, que se conectaba al :9092 sin autenticacion (el cluster
exige SCRAM o mTLS), usaba un grupo de consumo aleatorio (que ninguna ACL puede
autorizar) y solo informaba de una latencia media. Aqui:

  - La conexion sale de `KafkaConnectionSettings.from_env()` (variables KAFKA_*), la
    misma que usan los servicios.
  - Un mensaje cada vez: se envia con acks=all y se espera su ACK, y se cronometra desde
    antes del envio hasta que el consumidor lo recibe. Es el tiempo que vive un mensaje
    aislado, no un rendimiento sostenido (de eso se ocupa benchmark_kafka.py).
  - La particion se asigna a mano (assign, sin grupo de consumo): no necesita ACL de
    grupo y el orden de llegada es el de una sola particion. Solo cuentan los mensajes
    con el identificador de ESTA ejecucion, de modo que datos previos del topic no
    cuentan.
  - Un calentamiento se descarta (conexiones, metadatos, handshake TLS).

Una medicion `--environment local` es INFORMATIVA; solo `cluster` cierra el criterio.

Uso (desde la raiz del repositorio, con el stack local levantado):

    $env:KAFKA_BOOTSTRAP = "127.0.0.1:9092"
    .\\.venv\\Scripts\\python.exe -m tests.integration.test_kafka_connectivity --create-topic

En el cluster lo lanza tests/integration/Invoke-KafkaTests.ps1. Salida:
tests/results/kafka_latency.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, KafkaProducer, TopicPartition

from data.streaming.kafka_settings import KafkaConnectionSettings
from tests.integration.benchmark_kafka import benchmark_topic
from tests.integration.benchmark_report import (
    ENVIRONMENTS,
    build_report,
    summarize,
    write_report,
)

_LOG = logging.getLogger("kafka_connectivity")

CRITERION = "kafka_latency"
UNIT = "milliseconds"
LATENCY_THRESHOLD_MS = 10.0
"""Criterio de la Fase 1 (TDD): p99 de produce -> consume, en milisegundos."""

DEFAULT_TOPIC = "bench-throughput"
DEFAULT_MESSAGES = 100
DEFAULT_WARMUP = 10
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_OUTPUT = Path("tests/results/kafka_latency.json")
PARTITION = 0
_KEY = b"connectivity"
_POLL_MS = 50


class RoundTripError(RuntimeError):
    """A message was not delivered back to the consumer in time."""


class KafkaRoundTrip:
    """Sends one message and waits until the consumer sees it.

    Attributes:
        nonce: Identifier of this run; messages from other runs are ignored.
    """

    def __init__(
        self,
        producer: Any,
        consumer: Any,
        topic: str,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Bind the round trip to a producer, a consumer and a topic.

        Args:
            producer: A kafka-python producer (send(...).get(timeout)).
            consumer: A kafka-python consumer with the partition already assigned.
            topic: Topic both clients use.
            timeout_s: Longest wait for the ACK and for the delivery.
            clock: Monotonic clock in seconds.
        """
        self.nonce = uuid.uuid4().hex
        self._producer = producer
        self._consumer = consumer
        self._topic = topic
        self._timeout_s = timeout_s
        self._clock = clock

    def __call__(self, sequence: int) -> None:
        """Send message number `sequence` and block until it comes back.

        Args:
            sequence: Position of the message in the run.

        Raises:
            RoundTripError: If the message is not received within the timeout.
        """
        value = json.dumps({"nonce": self.nonce, "seq": sequence}).encode("ascii")
        self._producer.send(self._topic, key=_KEY, value=value, partition=PARTITION).get(
            timeout=self._timeout_s
        )
        deadline = self._clock() + self._timeout_s
        while self._clock() < deadline:
            for records in self._consumer.poll(timeout_ms=_POLL_MS).values():
                for record in records:
                    if record.value == value:
                        return
        raise RoundTripError(
            f"Message {sequence} of run {self.nonce} was not delivered within "
            f"{self._timeout_s:.1f} s."
        )


def run_roundtrips(
    roundtrip: Callable[[int], None],
    count: int,
    warmup: int,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> list[float]:
    """Time `count` round trips after discarding `warmup` of them.

    Args:
        roundtrip: Sends message n and returns once it is delivered.
        count: Round trips whose latency is kept.
        warmup: Leading round trips that are run but not kept.
        clock: Monotonic clock in seconds.

    Returns:
        One latency per kept round trip, in milliseconds.

    Raises:
        ValueError: If count is not positive or warmup is negative.
    """
    if count <= 0 or warmup < 0:
        raise ValueError("count must be positive and warmup must not be negative.")
    latencies: list[float] = []
    for sequence in range(warmup + count):
        started = clock()
        roundtrip(sequence)
        elapsed_ms = (clock() - started) * 1000.0
        if sequence >= warmup:
            latencies.append(elapsed_ms)
    return latencies


def make_report(
    latencies_ms: Sequence[float],
    *,
    environment: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON document of a latency measurement.

    Args:
        latencies_ms: One latency per kept round trip.
        environment: "local" or "cluster".
        details: Extra context of the run (topic, servers, warm-up).

    Returns:
        The document in the shared benchmark format; its value is the p99.
    """
    summary = summarize(latencies_ms)
    return build_report(
        criterion=CRITERION,
        environment=environment,
        value=round(summary["p99_ms"], 3),
        unit=UNIT,
        threshold=LATENCY_THRESHOLD_MS,
        direction="at_most",
        details={
            **details,
            "messages_delivered": len(latencies_ms),
            "summary_ms": {key: round(value, 3) for key, value in summary.items()},
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments; sys.argv[1:] when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Kafka produce -> consume round-trip latency.")
    parser.add_argument("--messages", type=int, default=DEFAULT_MESSAGES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="seconds")
    parser.add_argument(
        "--create-topic",
        action="store_true",
        help="create a private topic and delete it afterwards (local stack)",
    )
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="local")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fail-above-threshold",
        action="store_true",
        help="exit 2 when the p99 exceeds the criterion (default: exit 0)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and write its report.

    Args:
        argv: Command-line arguments; sys.argv[1:] when omitted.

    Returns:
        0 when every message was delivered, 1 when one was lost, 2 when
        --fail-above-threshold is set and the p99 exceeds the criterion.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    settings = KafkaConnectionSettings.from_env()
    client = settings.to_client_config()

    with benchmark_topic(
        settings, None if args.create_topic else args.topic, create=args.create_topic
    ) as topic:
        producer = KafkaProducer(**client, acks="all", linger_ms=0, client_id="kafka-latency")
        consumer = KafkaConsumer(
            **client, group_id=None, enable_auto_commit=False, client_id="kafka-latency"
        )
        try:
            partition = TopicPartition(topic, PARTITION)
            consumer.assign([partition])
            consumer.seek_to_end(partition)
            consumer.position(partition)
            roundtrip = KafkaRoundTrip(producer, consumer, topic, timeout_s=args.timeout)
            _LOG.info("Measuring %d round trips into '%s'", args.messages, topic)
            try:
                latencies = run_roundtrips(roundtrip, args.messages, args.warmup)
            except RoundTripError as exc:
                _LOG.error("%s", exc)
                return 1
        finally:
            consumer.close()
            producer.close()

    report = make_report(
        latencies,
        environment=args.environment,
        details={
            "topic": topic,
            "bootstrap_servers": list(settings.bootstrap_servers),
            "security_protocol": settings.security_protocol,
            "warmup_messages": args.warmup,
            "acks": "all",
        },
    )
    write_report(args.output, report)
    _LOG.info(
        "p99 %.2f ms (p50 %.2f ms) over %d messages, environment=%s, threshold %.0f ms -> %s",
        report["value"],
        report["details"]["summary_ms"]["p50_ms"],
        len(latencies),
        args.environment,
        LATENCY_THRESHOLD_MS,
        "meets" if report["passed"] else "ABOVE",
    )
    if args.fail_above_threshold and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
