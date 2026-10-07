"""Tests for data/validation/validation_consumer.py.

El consumidor se ejecuta contra un broker en memoria (lado productor y lado
consumidor) y el `SensorValidator` real con la tabla de spans sintetica de conftest,
de modo que las alertas que se comprueban las genera el detector de verdad. Un fake
registra la secuencia de llamadas para probar que el commit va despues de publicar y
de verificar la entrega.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from data.generators.tep_adapter import sensor_id
from data.schemas.sensor_reading import QualityFlag, SensorReading
from data.schemas.sensor_spans import SensorSpan
from data.streaming import metrics as metric_names
from data.streaming.errors import CommitError, PublishError, StreamingError
from data.streaming.metrics import RESET_REASONS, StreamMetrics
from data.streaming.streaming_params import StreamingParams, load_streaming_params
from data.streaming.transport import HEADER_RUN_ID, ConsumedRecord, PartitionRef
from data.validation.alert_event import AlertEvent
from data.validation.sensor_fault import FaultType, SensorFault, Severity
from data.validation.sensor_validator import (
    SensorValidator,
    StuckValueDetector,
    ValidationResult,
)
from data.validation.validation_consumer import (
    RESET_REBALANCE,
    RESET_RUN_CHANGE,
    RESET_TIME_REGRESSION,
    ValidationConsumer,
)
from tests.support.in_memory_broker import InMemoryBroker, partition_for
from tests.support.in_memory_consumer import CallLog, InMemoryConsumer, RecordingProducer
from tests.support.readings import make_value_reading

T0 = datetime(2000, 1, 1, tzinfo=UTC)
STEP = timedelta(minutes=3)
RUN_0 = "fault_type=00/loop=0"
RUN_1 = "fault_type=01/loop=0"
STUCK_SENSOR = "TEP-XMV-04"
SENSOR_A = sensor_id(0)
SENSOR_B = sensor_id(1)


@pytest.fixture()
def streaming() -> StreamingParams:
    return load_streaming_params(Path("params.yaml"))


def _sample(metrics: StreamMetrics, name: str, **labels: str) -> float:
    value = metrics.registry.get_sample_value(name, labels or None)
    return 0.0 if value is None else float(value)


class Harness:
    """A validation consumer wired to in-memory brokers and a call log."""

    def __init__(
        self,
        streaming: StreamingParams,
        spans: dict[str, SensorSpan],
        *,
        validator: SensorValidator | None = None,
        warmup_seconds: float | None = None,
        partitions: Sequence[int] | None = None,
    ) -> None:
        self.streaming = streaming
        self.raw = InMemoryBroker(12)
        self.out = InMemoryBroker(12)
        self.log = CallLog()
        self.consumer = InMemoryConsumer(self.raw, partitions=partitions, log=self.log)
        self.producer = RecordingProducer(self.out, self.log)
        self.validator = validator if validator is not None else SensorValidator(spans)
        self.metrics = StreamMetrics()
        self.ticks = 0
        self.sut = ValidationConsumer(
            self.consumer,
            self.producer,
            self.validator,
            streaming,
            self.metrics,
            on_tick=self._tick,
            warmup_seconds=warmup_seconds,
        )
        # run() hace lo mismo; los tests que llaman a run_once() necesitan la asignacion.
        self.consumer.subscribe([streaming.raw_topic], self.sut)

    def _tick(self) -> None:
        self.ticks += 1

    def publish(
        self,
        sensor: str,
        step: int,
        value: float | None = None,
        *,
        run: str | None = RUN_0,
    ) -> ConsumedRecord:
        """Append one raw reading to the raw topic and return its record."""
        reading = make_value_reading(
            sensor, T0 + step * STEP, 50.0 + 0.01 * step if value is None else value
        )
        headers = [] if run is None else [(HEADER_RUN_ID, run.encode("ascii"))]
        self.raw.send(self.streaming.raw_topic, sensor.encode(), reading.to_kafka_bytes(), headers)
        return self.raw.records(self.streaming.raw_topic, partition_for(sensor.encode(), 12))[-1]

    def sent_to(self, topic: str) -> list[Any]:
        return [message for message in self.out.sent if message.topic == topic]

    def validated(self) -> list[SensorReading]:
        return [
            SensorReading.from_kafka_bytes(message.value)
            for message in self.sent_to(self.streaming.validated_topic)
        ]

    def alerts(self) -> list[AlertEvent]:
        return [
            AlertEvent.from_kafka_bytes(message.value)
            for message in self.sent_to(self.streaming.alerts_topic)
        ]

    def metric(self, name: str, **labels: str) -> float:
        return _sample(self.metrics, name, **labels)


@pytest.fixture()
def harness(
    streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
) -> Callable[..., Harness]:
    def _build(**kwargs: Any) -> Harness:
        return Harness(kwargs.pop("streaming", streaming), sensor_spans, **kwargs)

    return _build


class FlakyValidator(SensorValidator):
    """A validator whose first validate() calls raise ValueError."""

    def __init__(self, spans: dict[str, SensorSpan], fail_times: int) -> None:
        super().__init__(spans)
        self.fail_times = fail_times
        self.calls = 0

    def validate(self, reading: SensorReading) -> ValidationResult:
        self.calls += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise ValueError("Timestamp precedes the previous one.")
        return super().validate(reading)


class SpyValidator(SensorValidator):
    """A validator that remembers the readings it was given and their quality."""

    def __init__(self, spans: dict[str, SensorSpan]) -> None:
        super().__init__(spans)
        self.seen: list[tuple[SensorReading, QualityFlag]] = []

    def validate(self, reading: SensorReading) -> ValidationResult:
        self.seen.append((reading, reading.measurement.quality))
        return super().validate(reading)


class ScriptedValidator(SensorValidator):
    """A validator that reports the faults it was told to, on every reading."""

    def __init__(self, spans: dict[str, SensorSpan], faults: list[SensorFault]) -> None:
        super().__init__(spans)
        self.faults = faults

    def validate(self, reading: SensorReading) -> ValidationResult:
        return ValidationResult(False, list(self.faults), reading)


def _fault(fault_type: FaultType, detector: str, sensor: str = SENSOR_A) -> SensorFault:
    return SensorFault(
        sensor_id=sensor,
        fault_type=fault_type,
        severity=Severity.MEDIUM,
        confidence=0.6,
        detected_at=T0,
        detector=detector,
        evidence={"pair": [sensor, "other"]},
    )


def _stuck_run_length(h: Harness, sensor: str) -> int:
    detector = next(d for d in h.validator.detectors if isinstance(d, StuckValueDetector))
    return detector.run_length(sensor)


def _two_sensors_in_different_partitions() -> tuple[str, str]:
    first = sensor_id(0)
    for col in range(1, 52):
        other = sensor_id(col)
        if partition_for(other.encode(), 12) != partition_for(first.encode(), 12):
            return first, other
    raise AssertionError("every sensor maps to the same partition")


class TestCommitAfterPublish:
    def test_every_send_comes_before_the_flush_and_the_flush_before_the_commit(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in range(5):
            h.publish(SENSOR_A, step)
        assert h.sut.run_once() == 5
        assert h.log.names() == ["send"] * 5 + ["flush", "commit"]

    def test_the_commit_names_the_next_offset_to_read_of_each_partition(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        first, second = _two_sensors_in_different_partitions()
        for step in range(4):
            h.publish(first, step)
        for step in range(3):
            h.publish(second, step)
        h.sut.run_once()
        raw = h.streaming.raw_topic
        assert h.consumer.commits == [
            {
                PartitionRef(raw, partition_for(first.encode(), 12)): 4,
                PartitionRef(raw, partition_for(second.encode(), 12)): 3,
            }
        ]

    def test_a_failing_flush_prevents_the_commit(self, harness: Callable[..., Harness]) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.out.fail_flush_with = PublishError("delivery failed")
        with pytest.raises(PublishError):
            h.sut.run_once()
        assert "send" in h.log.names()
        assert "flush" in h.log.names()
        assert "commit" not in h.log.names()
        assert h.consumer.committed == {}
        assert h.metric(metric_names.STREAM_PRODUCE_ERRORS) == 1

    def test_a_send_that_cannot_be_queued_prevents_flush_and_commit(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.out.fail_send_after = 0
        with pytest.raises(PublishError):
            h.sut.run_once()
        assert "flush" not in h.log.names()
        assert "commit" not in h.log.names()
        assert h.metric(metric_names.STREAM_PRODUCE_ERRORS) == 1

    def test_the_flush_uses_the_configured_timeout(
        self, harness: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        timeouts: list[float | None] = []
        monkeypatch.setattr(h.out, "flush", lambda timeout=None: timeouts.append(timeout))
        h.sut.run_once()
        assert timeouts == [h.streaming.flush_timeout_s]

    def test_a_commit_that_fails_is_retried_until_it_succeeds(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.consumer.commit_failures = 2
        h.sut.run_once()
        assert h.log.names().count("commit") == 3
        assert len(h.consumer.commits) == 1
        assert h.metric(metric_names.COMMIT_FAILURES) == 2

    def test_a_commit_that_always_fails_aborts_without_advancing(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        attempts = h.streaming.commit_retries + 1
        h.consumer.commit_failures = attempts
        with pytest.raises(CommitError):
            h.sut.run_once()
        assert h.log.names().count("commit") == attempts
        assert h.consumer.committed == {}
        assert h.metric(metric_names.COMMIT_FAILURES) == attempts
        # El lote sigue pendiente: un nuevo intento confirma los mismos offsets.
        h.sut.checkpoint()
        assert h.consumer.commits == [{PartitionRef(h.streaming.raw_topic, partition_for(
            SENSOR_A.encode(), 12)): 1}]

    def test_an_empty_poll_neither_flushes_nor_commits_but_keeps_the_heartbeat(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        assert h.sut.run_once() == 0
        assert h.log.names() == []
        assert h.ticks == 1
        assert h.metric(metric_names.BATCH_SIZE + "_count") == 0

    def test_a_batch_beats_the_heartbeat_after_the_poll_and_after_the_work(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.sut.run_once()
        assert h.ticks == 2

    def test_the_second_batch_starts_after_the_committed_offsets(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.sut.run_once()
        h.publish(SENSOR_A, 1)
        assert h.sut.run_once() == 1
        assert [len(message.value) > 0 for message in h.sent_to(h.streaming.validated_topic)] == [
            True,
            True,
        ]
        assert h.log.names().count("commit") == 2


class TestRunLoop:
    def test_run_subscribes_consumes_until_stopped_and_commits(
        self, streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        h = Harness(streaming, sensor_spans)
        for step in range(3):
            h.publish(SENSOR_A, step)
        polls: list[int] = []

        def stop_after_the_batch() -> None:
            polls.append(1)
            if len(polls) >= 2:
                h.sut.stop()

        h.sut = ValidationConsumer(
            h.consumer, h.producer, h.validator, streaming, h.metrics, on_tick=stop_after_the_batch
        )
        h.sut.run()
        assert h.consumer.subscribed == [streaming.raw_topic]
        assert len(h.validated()) == 3
        assert len(h.consumer.commits) == 1
        assert h.sut.is_ready

    def test_stopping_before_running_consumes_nothing(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.sut.stop()
        h.sut.run()
        assert h.consumer.subscribed == [h.streaming.raw_topic]
        assert h.log.names() == []

    def test_a_failing_poll_surfaces(self, harness: Callable[..., Harness]) -> None:
        h = harness()
        h.consumer.fail_poll_with = StreamingError("broker down")
        with pytest.raises(StreamingError):
            h.sut.run_once()

    def test_the_consumer_without_a_heartbeat_callback_works(
        self, streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        h = Harness(streaming, sensor_spans)
        h.publish(SENSOR_A, 0)
        sut = ValidationConsumer(
            h.consumer, h.producer, h.validator, streaming, StreamMetrics()
        )
        h.consumer.subscribe([streaming.raw_topic], sut)
        assert sut.run_once() == 1


class TestPoisonMessages:
    def test_unreadable_messages_are_skipped_counted_and_committed(
        self, harness: Callable[..., Harness], caplog: pytest.LogCaptureFixture
    ) -> None:
        h = harness()
        raw = h.streaming.raw_topic
        key = SENSOR_A.encode()
        h.publish(SENSOR_A, 0)
        for payload in (b"not json at all", b"{}", b"\xff\xfe\xfd"):
            h.raw.send(raw, key, payload, [(HEADER_RUN_ID, RUN_0.encode())])
        h.publish(SENSOR_A, 1)
        with caplog.at_level(logging.ERROR):
            assert h.sut.run_once() == 5
        assert h.metric(metric_names.POISON_MESSAGES) == 3
        assert len(h.validated()) == 2
        assert h.metric(metric_names.READINGS_CONSUMED) == 2
        partition = PartitionRef(raw, partition_for(key, 12))
        assert h.consumer.commits == [{partition: 5}]
        assert "sha256" in caplog.text
        assert f"{raw}[{partition.partition}]@1" in caplog.text

    def test_a_batch_of_only_poison_still_advances_the_offsets(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        raw = h.streaming.raw_topic
        for _ in range(3):
            h.raw.send(raw, b"k", b"garbage")
        h.sut.run_once()
        assert h.sent_to(h.streaming.validated_topic) == []
        assert h.consumer.commits == [{PartitionRef(raw, partition_for(b"k", 12)): 3}]
        assert h.log.names() == ["flush", "commit"]

    def test_a_tombstone_is_poison(self, harness: Callable[..., Harness]) -> None:
        h = harness()
        raw = h.streaming.raw_topic
        tombstone = ConsumedRecord(raw, 0, 7, b"k", None, (), 0)
        h.sut.process([tombstone])
        h.sut.checkpoint()
        assert h.metric(metric_names.POISON_MESSAGES) == 1
        assert h.consumer.commits == [{PartitionRef(raw, 0): 8}]

    def test_a_payload_that_is_valid_json_but_not_a_reading_is_poison(self) -> None:
        with pytest.raises(ValidationError):
            SensorReading.from_kafka_bytes(b'{"sensor": 1}')


class TestRunBoundaries:
    def test_a_new_run_resets_the_sensor_state_without_failing(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in range(6):
            h.publish(STUCK_SENSOR, step, 50.0, run=RUN_0)
        # El segundo run reinicia el reloj, como los 22 ficheros del TEP.
        for step in range(6):
            h.publish(STUCK_SENSOR, step, 50.0, run=RUN_1)
        h.sut.run_once()
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_RUN_CHANGE) == 1
        assert h.alerts() == []
        assert len(h.validated()) == 12
        assert _stuck_run_length(h, STUCK_SENSOR) == 6

    def test_without_the_run_change_the_two_runs_would_have_looked_like_one_stuck_run(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in range(12):
            h.publish(STUCK_SENSOR, step, 50.0, run=RUN_0)
        h.sut.run_once()
        assert len(h.alerts()) == 5

    def test_a_timestamp_going_back_inside_the_same_run_resets_the_sensor(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in (5, 6, 7, 2, 3):
            h.publish(SENSOR_A, step, run=RUN_0)
        h.sut.run_once()
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_TIME_REGRESSION) == 1
        assert len(h.validated()) == 5

    def test_a_regression_is_detected_without_any_run_header(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in (5, 6, 1, 2):
            h.publish(SENSOR_A, step, run=None)
        h.sut.run_once()
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_TIME_REGRESSION) == 1
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_RUN_CHANGE) == 0

    def test_a_repeated_timestamp_is_not_a_regression(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in (1, 1, 2):
            h.publish(SENSOR_A, step)
        h.sut.run_once()
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_TIME_REGRESSION) == 0

    def test_a_header_that_appears_or_disappears_is_a_run_change(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 1, run=None)
        h.publish(SENSOR_A, 2, run=RUN_0)
        h.sut.run_once()
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_RUN_CHANGE) == 1

    def test_mixing_naive_and_aware_timestamps_resets_instead_of_failing(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        aware = make_value_reading(SENSOR_A, T0, 50.0)
        naive = make_value_reading(SENSOR_A, datetime(2000, 1, 1, 0, 3), 50.0)
        headers = [(HEADER_RUN_ID, RUN_0.encode())]
        for reading in (aware, naive):
            h.raw.send(h.streaming.raw_topic, SENSOR_A.encode(), reading.to_kafka_bytes(), headers)
        h.sut.run_once()
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_TIME_REGRESSION) == 1
        assert len(h.validated()) == 2

    def test_the_other_sensors_keep_their_state_when_one_changes_run(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in range(3):
            h.publish(SENSOR_A, step, 50.0, run=RUN_0)
            h.publish(SENSOR_B, step, 50.0, run=RUN_0)
        h.publish(SENSOR_A, 0, 50.0, run=RUN_1)
        h.sut.run_once()
        assert _stuck_run_length(h, SENSOR_B) == 3
        assert _stuck_run_length(h, SENSOR_A) == 1


class TestValidatorFailures:
    def test_a_value_error_is_retried_once_from_clean_state(
        self, streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        validator = FlakyValidator(sensor_spans, fail_times=1)
        h = Harness(streaming, sensor_spans, validator=validator)
        h.publish(SENSOR_A, 0)
        h.sut.run_once()
        assert validator.calls == 2
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_TIME_REGRESSION) == 1
        assert len(h.validated()) == 1
        assert h.consumer.commits

    def test_a_second_failure_propagates_and_nothing_is_committed(
        self, streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        validator = FlakyValidator(sensor_spans, fail_times=5)
        h = Harness(streaming, sensor_spans, validator=validator)
        h.publish(SENSOR_A, 0)
        with pytest.raises(ValueError, match="precedes"):
            h.sut.run_once()
        assert validator.calls == 2
        assert "commit" not in h.log.names()
        assert h.consumer.committed == {}


class TestPublishedOutput:
    def test_every_reading_is_published_with_its_key_and_headers(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        sources = [h.publish(sensor_id(col), step) for step in range(3) for col in range(5)]
        h.sut.run_once()
        validated = h.sent_to(h.streaming.validated_topic)
        assert len(validated) == 15
        assert {(m.key, m.headers) for m in validated} == {
            (r.key, r.headers) for r in sources
        }
        assert all(m.headers == ((HEADER_RUN_ID, RUN_0.encode()),) for m in validated)

    def test_a_sensor_keeps_its_order_in_the_validated_topic(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        for step in range(6):
            for col in range(8):
                h.publish(sensor_id(col), step)
        h.sut.run_once()
        for records in h.out.partitions_of(h.streaming.validated_topic).values():
            by_sensor: dict[str, list[datetime]] = {}
            for record in records:
                reading = SensorReading.from_kafka_bytes(record.value or b"")
                by_sensor.setdefault(reading.sensor.id, []).append(reading.timestamp)
            assert all(stamps == sorted(stamps) for stamps in by_sensor.values())
            assert all(len(stamps) == 6 for stamps in by_sensor.values())

    def test_a_clean_reading_leaves_unchanged_and_raises_no_alert(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.sut.run_once()
        (reading,) = h.validated()
        assert reading.measurement.quality is QualityFlag.GOOD
        assert h.alerts() == []
        assert h.metric(metric_names.READINGS_VALIDATED, quality="good") == 1

    def test_eight_equal_readings_raise_a_stuck_alert_and_leave_suspect(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        records = [h.publish(STUCK_SENSOR, step, 50.0) for step in range(8)]
        h.sut.run_once()
        validated = h.validated()
        assert [r.measurement.quality for r in validated[:7]] == [QualityFlag.GOOD] * 7
        assert validated[7].measurement.quality is QualityFlag.SUSPECT
        assert validated[7].is_usable
        (alert,) = h.alerts()
        assert alert.fault_type is FaultType.STUCK
        assert alert.sensor_id == STUCK_SENSOR
        assert alert.run_id == RUN_0
        assert alert.reading_id == validated[7].reading_id
        assert alert.source.topic == h.streaming.raw_topic
        assert alert.source.partition == records[7].partition
        assert alert.source.offset == records[7].offset
        assert alert.evidence["run_length"] == 8
        message = h.sent_to(h.streaming.alerts_topic)[0]
        assert message.key == STUCK_SENSOR.encode()
        assert message.headers == ((HEADER_RUN_ID, RUN_0.encode()),)
        assert h.metric(metric_names.SENSOR_FAULTS, fault_type="stuck") == 1
        assert h.metric(metric_names.ALERTS_PUBLISHED, severity="MEDIUM") == 1
        assert h.metric(metric_names.STREAM_MESSAGES_PRODUCED, topic="anomaly-alerts") == 1
        assert h.metric(metric_names.READINGS_VALIDATED, quality="suspect") == 1

    def test_a_critical_range_violation_is_flagged_bad_and_still_published(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0, 20_000.0)
        h.sut.run_once()
        (reading,) = h.validated()
        assert reading.measurement.quality is QualityFlag.BAD
        assert not reading.is_usable
        (alert,) = h.alerts()
        assert alert.fault_type is FaultType.BIAS_OUT_OF_RANGE
        assert alert.severity is Severity.CRITICAL
        assert h.metric(metric_names.READINGS_VALIDATED, quality="bad") == 1
        assert h.metric(metric_names.ALERTS_PUBLISHED, severity="CRITICAL") == 1

    def test_the_consumed_reading_is_never_mutated(
        self, streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        spy = SpyValidator(sensor_spans)
        h = Harness(streaming, sensor_spans, validator=spy)
        h.publish(SENSOR_A, 0, 20_000.0)
        h.sut.run_once()
        ((reading, quality_before),) = spy.seen
        assert quality_before is QualityFlag.GOOD
        assert reading.measurement.quality is QualityFlag.GOOD
        assert h.validated()[0].measurement.quality is QualityFlag.BAD

    def test_a_reading_without_a_value_passes_through_unjudged(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        raw = make_value_reading(SENSOR_A, T0, None)
        h.raw.send(h.streaming.raw_topic, SENSOR_A.encode(), raw.to_kafka_bytes())
        h.sut.run_once()
        (reading,) = h.validated()
        assert reading.measurement.value is None
        assert h.alerts() == []

    def test_a_record_without_a_key_is_published_under_its_sensor_id(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        reading = make_value_reading(SENSOR_A, T0, 50.0)
        h.raw.send(h.streaming.raw_topic, None, reading.to_kafka_bytes())
        h.sut.run_once()
        (message,) = h.sent_to(h.streaming.validated_topic)
        assert message.key == SENSOR_A.encode()

    def test_two_faults_of_the_same_kind_on_one_reading_get_different_alert_ids(
        self, streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        faults = [
            _fault(FaultType.DRIFT_CORRELATED, "CrossCorrelationChecker"),
            _fault(FaultType.DRIFT_CORRELATED, "CrossCorrelationChecker"),
            _fault(FaultType.KALMAN_ANOMALY, "KalmanResidualDetector"),
        ]
        h = Harness(
            streaming, sensor_spans, validator=ScriptedValidator(sensor_spans, faults)
        )
        h.publish(SENSOR_A, 0)
        h.sut.run_once()
        alerts = h.alerts()
        assert len(alerts) == 3
        assert len({alert.alert_id for alert in alerts}) == 3

    def test_reprocessing_the_same_reading_yields_the_same_alert_ids(
        self, streaming: StreamingParams, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        def alert_ids() -> list[Any]:
            h = Harness(streaming, sensor_spans)
            for step in range(8):
                h.publish(STUCK_SENSOR, step, 50.0)
            h.sut.run_once()
            return [alert.alert_id for alert in h.alerts()]

        assert alert_ids() == alert_ids() != []

    def test_alerts_of_the_real_detectors_serialize(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        # Una rampa suave y un salto de 0,94 anchos de sobre en una muestra (3 min).
        for step in range(30):
            h.publish(SENSOR_A, step)
        h.publish(SENSOR_A, 30, 9_500.0)
        h.publish(SENSOR_A, 31)
        h.sut.run_once()
        kinds = {alert.fault_type for alert in h.alerts()}
        assert FaultType.NOISE_SPIKE in kinds
        assert FaultType.KALMAN_ANOMALY in kinds
        assert len(h.alerts()) == len(h.sent_to(h.streaming.alerts_topic))


class TestMetrics:
    def test_counters_and_histograms_of_a_batch(self, harness: Callable[..., Harness]) -> None:
        h = harness()
        for step in range(4):
            h.publish(SENSOR_A, step)
        h.sut.run_once()
        assert h.metric(metric_names.READINGS_CONSUMED) == 4
        assert h.metric(metric_names.VALIDATION_LATENCY + "_count") == 4
        assert h.metric(metric_names.BATCH_SIZE + "_count") == 1
        assert h.metric(metric_names.BATCH_SIZE + "_sum") == 4
        assert h.metric(metric_names.STREAM_PRODUCE_LATENCY + "_count") == 1
        assert h.metric(metric_names.STREAM_MESSAGES_PRODUCED, topic="sensor-validated") == 4

    def test_the_reset_reasons_are_the_declared_ones(self) -> None:
        assert {RESET_RUN_CHANGE, RESET_TIME_REGRESSION, RESET_REBALANCE} == set(RESET_REASONS)

    def test_the_lag_of_each_partition_is_set_after_every_batch(
        self, harness: Callable[..., Harness], streaming: StreamingParams
    ) -> None:
        h = harness(streaming=dataclasses.replace(streaming, max_poll_records=4))
        for step in range(10):
            h.publish(SENSOR_A, step)
        partition = str(partition_for(SENSOR_A.encode(), 12))
        lags = []
        for _ in range(3):
            h.sut.run_once()
            lags.append(h.metric(metric_names.CONSUMER_LAG, partition=partition))
        assert lags == [6, 2, 0]

    def test_a_failure_computing_the_lag_does_not_stop_the_loop(
        self, harness: Callable[..., Harness], caplog: pytest.LogCaptureFixture
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        h.consumer.fail_end_offsets_with = StreamingError("metadata unavailable")
        with caplog.at_level(logging.WARNING):
            assert h.sut.run_once() == 1
        assert "Could not compute the consumer lag" in caplog.text
        assert h.consumer.commits

    def test_a_partition_missing_from_the_end_offsets_is_skipped(
        self, harness: Callable[..., Harness], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h = harness()
        h.publish(SENSOR_A, 0)
        monkeypatch.setattr(h.consumer, "end_offsets", lambda partitions: {})
        assert h.sut.run_once() == 1
        partition = str(partition_for(SENSOR_A.encode(), 12))
        assert h.metrics.registry.get_sample_value(
            metric_names.CONSUMER_LAG, {"partition": partition}
        ) is None

    def test_an_assignment_with_no_partitions_sets_no_lag(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness(partitions=[])
        assert h.sut.run_once() == 0
        h.sut.process([ConsumedRecord("sensor-readings-raw", 0, 0, b"k", b"garbage", (), 0)])
        h.sut._update_lag()  # noqa: SLF001
        assert h.metric(metric_names.CONSUMER_LAG, partition="0") == 0


class TestRebalance:
    def _two_partition_harness(
        self, harness: Callable[..., Harness]
    ) -> tuple[Harness, str, str, PartitionRef, PartitionRef]:
        h = harness()
        first, second = _two_sensors_in_different_partitions()
        for step in range(3):
            h.publish(first, step, 50.0)
            h.publish(second, step, 50.0)
        h.sut.run_once()
        raw = h.streaming.raw_topic
        return (
            h,
            first,
            second,
            PartitionRef(raw, partition_for(first.encode(), 12)),
            PartitionRef(raw, partition_for(second.encode(), 12)),
        )

    def test_revoking_one_partition_resets_only_the_sensors_seen_in_it(
        self, harness: Callable[..., Harness]
    ) -> None:
        h, first, second, revoked, _ = self._two_partition_harness(harness)
        assert _stuck_run_length(h, first) == 3
        h.sut.on_partitions_revoked([revoked])
        assert _stuck_run_length(h, first) == 0
        assert _stuck_run_length(h, second) == 3
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_REBALANCE) == 1

    def test_every_sensor_seen_in_the_partition_is_reset_not_only_the_last_batch(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        sensors = [sensor_id(col) for col in range(52)]
        for step in range(2):
            for sensor in sensors:
                h.publish(sensor, step, 50.0)
        h.sut.run_once()
        # Un lote posterior solo trae un sensor: el mapa recuerda a todos los demas.
        h.publish(sensors[0], 2, 50.0)
        h.sut.run_once()
        raw = h.streaming.raw_topic
        target = partition_for(sensors[0].encode(), 12)
        expected = {s for s in sensors if partition_for(s.encode(), 12) == target}
        assert len(expected) > 1
        h.sut.on_partitions_revoked([PartitionRef(raw, target)])
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_REBALANCE) == len(
            expected
        )
        for sensor in sensors:
            assert (_stuck_run_length(h, sensor) == 0) == (sensor in expected)

    def test_lost_partitions_reset_the_same_way(self, harness: Callable[..., Harness]) -> None:
        h, first, second, lost, _ = self._two_partition_harness(harness)
        h.sut.on_partitions_lost([lost])
        assert _stuck_run_length(h, first) == 0
        assert _stuck_run_length(h, second) == 3
        assert not h.sut.is_ready

    def test_the_listener_neither_flushes_nor_commits(
        self, harness: Callable[..., Harness]
    ) -> None:
        h, _, _, revoked, _ = self._two_partition_harness(harness)
        before = list(h.log.events)
        h.sut.on_partitions_revoked([revoked])
        h.sut.on_partitions_assigned([revoked])
        assert h.log.events == before

    def test_a_revoked_sensor_is_a_fresh_start_not_a_run_change(
        self, harness: Callable[..., Harness]
    ) -> None:
        h, first, _, revoked, _ = self._two_partition_harness(harness)
        h.sut.on_partitions_revoked([revoked])
        h.publish(first, 0, 50.0, run=RUN_1)
        h.sut.run_once()
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_RUN_CHANGE) == 0
        assert _stuck_run_length(h, first) == 1

    def test_the_lag_series_of_a_revoked_partition_disappears(
        self, harness: Callable[..., Harness]
    ) -> None:
        h, _, _, revoked, kept = self._two_partition_harness(harness)
        label = {"partition": str(revoked.partition)}
        assert h.metrics.registry.get_sample_value(metric_names.CONSUMER_LAG, label) == 0
        h.sut.on_partitions_revoked([revoked])
        assert h.metrics.registry.get_sample_value(metric_names.CONSUMER_LAG, label) is None
        assert h.metrics.registry.get_sample_value(
            metric_names.CONSUMER_LAG, {"partition": str(kept.partition)}
        ) == 0

    def test_uncommitted_offsets_of_a_revoked_partition_are_dropped(
        self, harness: Callable[..., Harness], caplog: pytest.LogCaptureFixture
    ) -> None:
        h = harness()
        record = h.publish(SENSOR_A, 0)
        h.sut.process([record])
        with caplog.at_level(logging.WARNING):
            h.sut.on_partitions_revoked([record.partition_ref])
        assert "uncommitted" in caplog.text
        h.sut.checkpoint()
        assert h.consumer.commits == []

    def test_readiness_follows_the_assignment_cycle(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        ref = PartitionRef(h.streaming.raw_topic, 0)
        assert not h.sut.is_ready
        h.sut.on_partitions_assigned([ref])
        assert h.sut.is_ready
        h.sut.on_partitions_revoked([ref])
        assert not h.sut.is_ready
        h.sut.on_partitions_assigned([])
        assert h.sut.is_ready

    def test_an_assignment_is_counted_as_a_rebalance_and_logs_the_warm_up(
        self, harness: Callable[..., Harness], caplog: pytest.LogCaptureFixture
    ) -> None:
        h = harness(warmup_seconds=8 * 180.0)
        with caplog.at_level(logging.INFO):
            h.sut.on_partitions_assigned([PartitionRef(h.streaming.raw_topic, 4)])
        assert h.metric(metric_names.CONSUMER_REBALANCES) == 1
        assert "24 min" in caplog.text

    def test_an_assignment_without_a_warm_up_estimate_logs_no_estimate(
        self, harness: Callable[..., Harness], caplog: pytest.LogCaptureFixture
    ) -> None:
        h = harness()
        with caplog.at_level(logging.INFO):
            h.sut.on_partitions_assigned([PartitionRef(h.streaming.raw_topic, 4)])
        assert "Assigned partitions" in caplog.text
        assert "needs" not in caplog.text

    def test_a_full_rebalance_through_the_consumer_resets_and_resumes_from_the_commit(
        self, harness: Callable[..., Harness]
    ) -> None:
        h, first, second, _, _ = self._two_partition_harness(harness)
        assert h.metric(metric_names.CONSUMER_REBALANCES) == 1
        h.consumer.rebalance([partition_for(first.encode(), 12)])
        h.publish(first, 3, 50.0)
        assert h.sut.run_once() == 1
        assert h.metric(metric_names.CONSUMER_REBALANCES) == 2
        assert h.metric(metric_names.DETECTOR_STATE_RESETS, reason=RESET_REBALANCE) == 2
        # Eager: se revocaron ambas particiones; el sensor retomado reconstruye su racha.
        assert _stuck_run_length(h, first) == 1
        assert _stuck_run_length(h, second) == 0

    def test_a_stuck_run_needs_the_full_window_again_after_a_rebalance(
        self, harness: Callable[..., Harness]
    ) -> None:
        h = harness()
        partition = partition_for(STUCK_SENSOR.encode(), 12)
        for step in range(7):
            h.publish(STUCK_SENSOR, step, 50.0)
        h.sut.run_once()
        h.consumer.rebalance([partition])
        for step in range(7, 12):
            h.publish(STUCK_SENSOR, step, 50.0)
        h.sut.run_once()
        # 7 + 5 iguales serian 5 alertas sin rebalanceo; con el estado perdido, ninguna.
        assert h.alerts() == []
