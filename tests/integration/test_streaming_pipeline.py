"""Streaming pipeline against a real Kafka broker (marker `integration`).

Lo que las pruebas unitarias no pueden probar porque usan fakes: que los kwargs de
kafka-python son aceptados por un broker de verdad (idempotencia, lz4, cabeceras),
que el commit tras el listener de rebalanceo funciona y que un segundo consumidor del
grupo reparte las particiones. Se ejecutan contra el stack local:

    .\\infra\\local\\Invoke-LocalStack.ps1 -Up
    $env:KAFKA_BOOTSTRAP = "127.0.0.1:9092"
    .\\.venv\\Scripts\\python.exe -m pytest tests/integration -m integration

Cada test crea sus PROPIOS topics (sufijo aleatorio) y su grupo, y los borra al
terminar, de modo que no depende de lo que haya en los topics fijos ni lo contamina.
Sin KAFKA_BOOTSTRAP los tests se saltan con un mensaje claro.
"""

from __future__ import annotations

import dataclasses
import os
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from kafka.admin import KafkaAdminClient, NewTopic

from data.generators.tep_adapter import sensor_id
from data.schemas.sensor_reading import SensorReading
from data.schemas.sensor_spans import DEFAULT_SPANS_PATH, SensorSpan, load_sensor_spans
from data.streaming import metrics as metric_names
from data.streaming.kafka_adapters import KafkaMessageConsumer, KafkaMessageProducer
from data.streaming.kafka_settings import ENV_BOOTSTRAP, KafkaConnectionSettings
from data.streaming.metrics import StreamMetrics
from data.streaming.streaming_params import StreamingParams, load_streaming_params
from data.streaming.transport import HEADER_RUN_ID, ConsumedRecord
from data.validation.alert_event import AlertEvent
from data.validation.sensor_fault import FaultType
from data.validation.sensor_validator import DEFAULT_STUCK_WINDOW, SensorValidator
from data.validation.validation_consumer import ValidationConsumer
from tests.support.readings import make_value_reading

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get(ENV_BOOTSTRAP),
        reason=(
            f"{ENV_BOOTSTRAP} no esta definido. Levanta el stack con "
            ".\\infra\\local\\Invoke-LocalStack.ps1 -Up y exporta "
            f'$env:{ENV_BOOTSTRAP} = "127.0.0.1:9092".'
        ),
    ),
]

_T0 = datetime(2000, 1, 1, tzinfo=UTC)
_STEP = timedelta(minutes=3)
_RUN_ID = b"fault_type=21/loop=0"
_FROZEN = "TEP-XMV-04"
_MOVING = "TEP-XMEAS-01"
_TIMEOUT_S = 90.0
_SETTLE_POLLS = 3
"""Polls vacios consecutivos tras los que un topic se da por leido entero."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def settings() -> KafkaConnectionSettings:
    """Connection settings from KAFKA_* (PLAINTEXT against the local stack)."""
    return KafkaConnectionSettings.from_env()


@pytest.fixture(scope="module")
def spans() -> dict[str, SensorSpan]:
    """Load the committed, calibrated span table."""
    return load_sensor_spans(DEFAULT_SPANS_PATH)


@pytest.fixture()
def streaming(settings: KafkaConnectionSettings) -> Iterator[StreamingParams]:
    """Private topics and group for one test, deleted afterwards.

    Yields:
        The streaming params with the topic names and the group id replaced.
    """
    base = load_streaming_params(Path("params.yaml"))
    suffix = uuid.uuid4().hex[:8]
    params = dataclasses.replace(
        base,
        raw_topic=f"it-raw-{suffix}",
        validated_topic=f"it-validated-{suffix}",
        alerts_topic=f"it-alerts-{suffix}",
        consumer_group=f"it-group-{suffix}",
    )
    admin = KafkaAdminClient(**settings.to_client_config())
    try:
        admin.create_topics(
            [
                NewTopic(params.raw_topic, num_partitions=12, replication_factor=1),
                NewTopic(params.validated_topic, num_partitions=12, replication_factor=1),
                NewTopic(params.alerts_topic, num_partitions=3, replication_factor=1),
            ]
        )
        yield params
    finally:
        admin.delete_topics([params.raw_topic, params.validated_topic, params.alerts_topic])
        admin.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mid(span: SensorSpan) -> tuple[float, float]:
    """Return the centre and the width of a sensor's alarm envelope."""
    return (span.alarm_min + span.alarm_max) / 2.0, span.alarm_max - span.alarm_min


def build_readings(
    spans: dict[str, SensorSpan], sensors: Sequence[str], steps: int, *, frozen: str | None
) -> list[SensorReading]:
    """Build in-envelope readings, timestep by timestep, one per sensor.

    Args:
        spans: Calibrated span table.
        sensors: Sensor tags to emit.
        steps: Number of timesteps.
        frozen: Tag held at a constant value, or None for none. El resto se mueve un
            0,05 % del ancho del sobre por muestra: dentro del sobre y sin repetir.

    Returns:
        The readings in (timestep, sensor) order.
    """
    readings: list[SensorReading] = []
    for step in range(steps):
        for tag in sensors:
            centre, width = _mid(spans[tag])
            value = centre if tag == frozen else centre + width * 0.0005 * step
            readings.append(make_value_reading(tag, _T0 + step * _STEP, value))
    return readings


def produce(
    settings: KafkaConnectionSettings, streaming: StreamingParams, readings: list[SensorReading]
) -> None:
    """Publish readings keyed by sensor with the run-id header, and verify delivery."""
    producer = KafkaMessageProducer(settings, streaming, client_id="it-producer")
    try:
        for reading in readings:
            producer.send(
                streaming.raw_topic,
                reading.sensor.id.encode("utf-8"),
                reading.to_kafka_bytes(),
                [(HEADER_RUN_ID, _RUN_ID)],
            )
        producer.flush(streaming.flush_timeout_s)
    finally:
        producer.close()


def read_topic(
    settings: KafkaConnectionSettings, streaming: StreamingParams, topic: str
) -> list[ConsumedRecord]:
    """Read a topic from the beginning until it stays quiet.

    Args:
        settings: Connection settings.
        streaming: Streaming params (poll size and offset reset).
        topic: Topic to read.

    Returns:
        Every record found, in poll order.

    Raises:
        AssertionError: If the group never gets its partitions.
    """
    consumer = KafkaMessageConsumer(
        settings, streaming, group_id=f"it-reader-{uuid.uuid4().hex[:8]}", client_id="it-reader"
    )
    consumer.subscribe([topic])
    found: list[ConsumedRecord] = []
    quiet = 0
    deadline = time.monotonic() + _TIMEOUT_S
    try:
        while time.monotonic() < deadline:
            records = consumer.poll(streaming.poll_timeout_ms, streaming.max_poll_records)
            found.extend(records)
            quiet = 0 if records else quiet + 1
            if quiet >= _SETTLE_POLLS and consumer.assignment():
                return found
    finally:
        consumer.close()
    raise AssertionError(f"Topic {topic} was never fully read within {_TIMEOUT_S} s.")


def run_until(
    consumers: Sequence[ValidationConsumer], done: Callable[[], bool], what: str
) -> None:
    """Drive consumers from this thread until a condition holds.

    Args:
        consumers: Validation consumers to poll in turn.
        done: Condition to wait for.
        what: Description for the failure message.

    Raises:
        AssertionError: If the condition does not hold within the timeout.
    """
    deadline = time.monotonic() + _TIMEOUT_S
    while time.monotonic() < deadline:
        for consumer in consumers:
            consumer.run_once()
        if done():
            return
    raise AssertionError(f"Timed out after {_TIMEOUT_S} s waiting for: {what}.")


def _build_consumer(
    settings: KafkaConnectionSettings,
    streaming: StreamingParams,
    spans: dict[str, SensorSpan],
    name: str,
) -> tuple[ValidationConsumer, StreamMetrics, KafkaMessageConsumer]:
    """Wire one validation consumer onto real Kafka clients and subscribe it."""
    metrics = StreamMetrics()
    transport = KafkaMessageConsumer(settings, streaming, client_id=f"it-{name}")
    sut = ValidationConsumer(
        transport,
        KafkaMessageProducer(settings, streaming, client_id=f"it-{name}-out"),
        SensorValidator(spans),
        streaming,
        metrics,
    )
    transport.subscribe([streaming.raw_topic], sut)
    return sut, metrics, transport


def _sample(metrics: StreamMetrics, name: str, **labels: str) -> float:
    """Read one sample of a metric, 0.0 when absent."""
    value = metrics.registry.get_sample_value(name, labels or None)
    return 0.0 if value is None else float(value)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRawToValidated:
    """100 lecturas crudas llegan validadas, sin duplicados y con lag cero."""

    def test_hundred_readings_are_validated_and_committed(
        self,
        settings: KafkaConnectionSettings,
        spans: dict[str, SensorSpan],
        streaming: StreamingParams,
    ) -> None:
        """The whole path works against a real broker, and a stuck sensor alerts."""
        steps = 50
        readings = build_readings(spans, [_FROZEN, _MOVING], steps, frozen=_FROZEN)
        assert len(readings) == 100
        produce(settings, streaming, readings)

        sut, metrics, transport = _build_consumer(settings, streaming, spans, "single")
        try:
            run_until(
                [sut],
                lambda: _sample(metrics, metric_names.READINGS_CONSUMED) >= len(readings),
                "100 readings consumed",
            )
            sut.checkpoint()
            lag = {
                partition.partition: _sample(
                    metrics, metric_names.CONSUMER_LAG, partition=str(partition.partition)
                )
                for partition in transport.assignment()
            }
        finally:
            transport.close()

        # Lag cero en todas las particiones asignadas y la metrica de consumo exacta.
        assert lag and all(value == 0.0 for value in lag.values())
        assert _sample(metrics, metric_names.READINGS_CONSUMED) == 100.0

        validated = read_topic(settings, streaming, streaming.validated_topic)
        ids = [SensorReading.from_kafka_bytes(r.value).reading_id for r in validated if r.value]
        assert len(ids) == 100
        assert len(set(ids)) == 100
        assert set(ids) == {reading.reading_id for reading in readings}
        assert {r.key for r in validated} == {_FROZEN.encode(), _MOVING.encode()}
        assert {r.header(HEADER_RUN_ID) for r in validated} == {_RUN_ID}

        alerts = [
            AlertEvent.from_kafka_bytes(r.value)
            for r in read_topic(settings, streaming, streaming.alerts_topic)
            if r.value
        ]
        assert len({alert.alert_id for alert in alerts}) == len(alerts)
        stuck = [alert for alert in alerts if alert.fault_type is FaultType.STUCK]
        assert {alert.sensor_id for alert in stuck} == {_FROZEN}
        assert len(stuck) == steps - (DEFAULT_STUCK_WINDOW - 1)

    def test_a_restarted_consumer_finds_nothing_left_to_read(
        self,
        settings: KafkaConnectionSettings,
        spans: dict[str, SensorSpan],
        streaming: StreamingParams,
    ) -> None:
        """Committed offsets survive: the same group resumes at the end, not at zero."""
        readings = build_readings(spans, [_MOVING], 20, frozen=None)
        produce(settings, streaming, readings)

        first, metrics, transport = _build_consumer(settings, streaming, spans, "first")
        try:
            run_until(
                [first],
                lambda: _sample(metrics, metric_names.READINGS_CONSUMED) >= len(readings),
                "all readings consumed",
            )
            first.checkpoint()
        finally:
            transport.close()

        second, second_metrics, second_transport = _build_consumer(
            settings, streaming, spans, "second"
        )
        try:
            for _ in range(_SETTLE_POLLS + 2):
                second.run_once()
        finally:
            second_transport.close()
        assert _sample(second_metrics, metric_names.READINGS_CONSUMED) == 0.0


class TestRebalance:
    """Dos consumidores del mismo grupo se reparten las particiones."""

    def test_two_consumers_split_the_partitions_and_lose_nothing(
        self,
        settings: KafkaConnectionSettings,
        spans: dict[str, SensorSpan],
        streaming: StreamingParams,
    ) -> None:
        """A second member takes half; when it leaves, the first takes all back."""
        tags = [sensor_id(column) for column in range(52)]
        readings = build_readings(spans, tags, 10, frozen=None)
        produce(settings, streaming, readings)

        first, first_metrics, first_transport = _build_consumer(settings, streaming, spans, "a")
        second, second_metrics, second_transport = _build_consumer(
            settings, streaming, spans, "b"
        )

        def consumed() -> float:
            return _sample(first_metrics, metric_names.READINGS_CONSUMED) + _sample(
                second_metrics, metric_names.READINGS_CONSUMED
            )

        try:
            run_until(
                [first, second],
                lambda: consumed() >= len(readings)
                and first.is_ready
                and second.is_ready
                and len(first_transport.assignment()) + len(second_transport.assignment()) == 12,
                "both members joined and every reading consumed",
            )
            owned_a = first_transport.assignment()
            owned_b = second_transport.assignment()
            assert owned_a and owned_b
            assert owned_a.isdisjoint(owned_b)
            assert len(owned_a | owned_b) == 12

            # Cuando el segundo se va, el primero recupera las 12 particiones y, al
            # perder la asignacion previa, reinicia el estado de sus sensores.
            second_transport.close()
            run_until(
                [first],
                lambda: len(first_transport.assignment()) == 12 and first.is_ready,
                "the remaining member owns all 12 partitions",
            )
            assert _sample(first_metrics, metric_names.CONSUMER_REBALANCES) >= 2.0
            assert (
                _sample(first_metrics, metric_names.DETECTOR_STATE_RESETS, reason="rebalance")
                > 0.0
            )
            first.checkpoint()
        finally:
            first_transport.close()
            second_transport.close()

        validated = read_topic(settings, streaming, streaming.validated_topic)
        counts = Counter(
            SensorReading.from_kafka_bytes(r.value).reading_id for r in validated if r.value
        )
        assert set(counts) == {reading.reading_id for reading in readings}
        assert all(count == 1 for count in counts.values()), "duplicates after rebalance"
