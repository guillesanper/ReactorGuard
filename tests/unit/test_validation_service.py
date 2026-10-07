"""Tests for data/validation/validation_service.py.

El servicio se ejecuta de punta a punta con un KafkaConsumer y un KafkaProducer
falsos (a traves de `consumer_factory` y `producer_factory`), la tabla de spans
sintetica de conftest, puertos HTTP libres y parametros en tmp_path. Nada toca la red
ni Kafka.
"""

from __future__ import annotations

import copy
import logging
import signal
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from kafka.errors import KafkaError
from kafka.structs import TopicPartition

from data.generators.tep_params import load_tep_params
from data.schemas.sensor_reading import QualityFlag, SensorReading
from data.streaming.lifecycle import GracefulShutdown
from data.validation import validation_service
from data.validation.alert_event import AlertEvent
from data.validation.sensor_validator import RangeValidator, SensorValidator
from data.validation.validation_service import (
    EXIT_CONFIG,
    EXIT_OK,
    EXIT_RUNTIME,
    cli,
    main,
)
from tests.support.in_memory_broker import partition_for
from tests.support.readings import make_value_reading

BASE_ENV = {"KAFKA_BOOTSTRAP": "broker:9092"}
RAW = "sensor-readings-raw"
STUCK_SENSOR = "TEP-XMV-04"
RUN = b"fault_type=21/loop=0"
T0 = datetime(2000, 1, 1, tzinfo=UTC)
PARTITION = partition_for(STUCK_SENSOR.encode(), 12)


class FakeFuture:
    """Delivery future of the fake kafka-python producer."""

    def __init__(self, error: KafkaError | None) -> None:
        self.error = error
        self.is_done = True

    def succeeded(self) -> bool:
        return self.error is None

    def get(self, timeout: float | None = None) -> str:
        del timeout
        if self.error is not None:
            raise self.error
        return "metadata"


class FakeKafkaProducer:
    """Stand-in for kafka.KafkaProducer."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.sent: list[dict[str, Any]] = []
        self.flushes = 0
        self.close_timeouts: list[float | None] = []
        self.delivery_error: KafkaError | None = None

    def send(self, topic: str, **kwargs: Any) -> FakeFuture:
        self.sent.append({"topic": topic, **kwargs})
        return FakeFuture(self.delivery_error)

    def flush(self, timeout: float | None = None) -> None:
        del timeout
        self.flushes += 1

    def close(self, timeout: float | None = None) -> None:
        self.close_timeouts.append(timeout)


@dataclass
class FakeRecord:
    """The fields of a kafka-python ConsumerRecord the adapter reads."""

    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None
    headers: list[tuple[str, bytes]]
    timestamp: int = 0


class FakeKafkaConsumer:
    """Stand-in for kafka.KafkaConsumer, driven by a script of batches."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.batches: list[list[FakeRecord]] = []
        self.on_poll: Callable[[int], None] | None = None
        self.polls = 0
        self.commits: list[dict[TopicPartition, int]] = []
        self.commit_attempts = 0
        self.commit_error: KafkaError | None = None
        self.listener: Any = None
        self.topics: list[str] = []
        self.assigned = {TopicPartition(RAW, PARTITION)}
        self.positions: dict[TopicPartition, int] = {}
        self.closed_with: dict[str, Any] | None = None

    def subscribe(self, topics: list[str], listener: Any = None) -> None:
        self.topics = topics
        self.listener = listener

    def poll(
        self, timeout_ms: int | None = None, max_records: int | None = None
    ) -> dict[TopicPartition, list[FakeRecord]]:
        self.polls += 1
        if self.polls == 1 and self.listener is not None:
            self.listener.on_partitions_assigned(set(self.assigned))
        if self.on_poll is not None:
            self.on_poll(self.polls)
        if not self.batches:
            time.sleep(0.001)
            return {}
        out: dict[TopicPartition, list[FakeRecord]] = {}
        for record in self.batches.pop(0):
            partition = TopicPartition(record.topic, record.partition)
            out.setdefault(partition, []).append(record)
            self.positions[partition] = record.offset + 1
        return out

    def commit(self, offsets: dict[TopicPartition, Any], timeout_ms: int | None = None) -> None:
        self.commit_attempts += 1
        if self.commit_error is not None:
            raise self.commit_error
        self.commits.append({tp: meta.offset for tp, meta in offsets.items()})

    def assignment(self) -> set[TopicPartition]:
        return set(self.assigned)

    def position(self, partition: TopicPartition) -> int:
        return self.positions.get(partition, 0)

    def end_offsets(self, partitions: list[TopicPartition]) -> dict[TopicPartition, int]:
        return {tp: self.positions.get(tp, 0) for tp in partitions}

    def close(self, autocommit: bool = True, timeout_ms: int | None = None) -> None:
        self.closed_with = {"autocommit": autocommit, "timeout_ms": timeout_ms}


def _record(offset: int, value: float, *, sensor: str = STUCK_SENSOR) -> FakeRecord:
    reading = make_value_reading(sensor, T0 + timedelta(minutes=3 * offset), value)
    return FakeRecord(
        RAW, partition_for(sensor.encode(), 12), offset, sensor.encode(),
        reading.to_kafka_bytes(), [("run-id", RUN)],
    )


@dataclass
class Setup:
    """A params file and the ports the service will bind."""

    params_path: Path
    health_port: int
    metrics_port: int
    document: dict[str, Any]
    producers: list[FakeKafkaProducer]
    consumers: list[FakeKafkaConsumer]
    script: list[list[FakeRecord]]
    on_poll: Callable[[int], None] | None = None

    def producer_factory(self, **config: Any) -> FakeKafkaProducer:
        producer = FakeKafkaProducer(**config)
        self.producers.append(producer)
        return producer

    def consumer_factory(self, **config: Any) -> FakeKafkaConsumer:
        consumer = FakeKafkaConsumer(**config)
        consumer.batches = [list(batch) for batch in self.script]
        consumer.on_poll = self.on_poll
        self.consumers.append(consumer)
        return consumer

    def rewrite(self) -> None:
        self.params_path.write_text(yaml.safe_dump(self.document, sort_keys=False), "utf-8")

    def run(self, env: dict[str, str] | None = None, **kwargs: Any) -> int:
        return main(
            self.params_path,
            BASE_ENV if env is None else env,
            producer_factory=kwargs.pop("producer_factory", self.producer_factory),
            consumer_factory=kwargs.pop("consumer_factory", self.consumer_factory),
            **kwargs,
        )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def setup(tmp_path: Path, spans_file: Path) -> Setup:
    document: dict[str, Any] = copy.deepcopy(
        yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    )
    document["tep"]["spans_path"] = str(spans_file)
    document["streaming"]["bind_host"] = "127.0.0.1"
    health, metrics = _free_port(), _free_port()
    document["streaming"]["health_port"] = health
    document["streaming"]["metrics_port"] = metrics
    prepared = Setup(tmp_path / "params.yaml", health, metrics, document, [], [], [])
    prepared.rewrite()
    return prepared


def _stop_after(shutdown: GracefulShutdown, polls: int) -> Callable[[int], None]:
    def hook(count: int) -> None:
        if count >= polls:
            shutdown.request()

    return hook


def _get(port: int, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:  # noqa: S310
            return int(response.status), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


class TestRun:
    def test_validates_publishes_commits_and_shuts_down_cleanly(self, setup: Setup) -> None:
        setup.script = [[_record(i, 50.0) for i in range(8)]]
        shutdown = GracefulShutdown()
        setup.on_poll = _stop_after(shutdown, 3)
        assert setup.run(shutdown=shutdown) == EXIT_OK

        (producer,) = setup.producers
        validated = [m for m in producer.sent if m["topic"] == "sensor-validated"]
        alerts = [m for m in producer.sent if m["topic"] == "anomaly-alerts"]
        assert len(validated) == 8
        assert len(alerts) == 1
        assert all(m["key"] == STUCK_SENSOR.encode() for m in validated + alerts)
        assert all(m["headers"] == [("run-id", RUN)] for m in validated + alerts)
        last = SensorReading.from_kafka_bytes(validated[-1]["value"])
        assert last.measurement.quality is QualityFlag.SUSPECT
        event = AlertEvent.from_kafka_bytes(alerts[0]["value"])
        assert event.sensor_id == STUCK_SENSOR
        assert event.run_id == RUN.decode()
        assert event.source.offset == 7

        (consumer,) = setup.consumers
        assert consumer.commits == [{TopicPartition(RAW, PARTITION): 8}]
        assert consumer.topics == [RAW]
        assert consumer.closed_with == {"autocommit": False, "timeout_ms": 5000}
        assert producer.close_timeouts == [5.0]
        assert producer.flushes >= 1

    def test_the_clients_are_built_from_the_kafka_environment(self, setup: Setup) -> None:
        shutdown = GracefulShutdown()
        setup.on_poll = _stop_after(shutdown, 1)
        env = {**BASE_ENV, "KAFKA_BOOTSTRAP": "a:9092,b:9092"}
        assert setup.run(env, shutdown=shutdown) == EXIT_OK
        consumer_config = setup.consumers[0].config
        producer_config = setup.producers[0].config
        assert consumer_config["bootstrap_servers"] == ["a:9092", "b:9092"]
        assert consumer_config["group_id"] == "sensor-validator"
        assert consumer_config["enable_auto_commit"] is False
        assert producer_config["bootstrap_servers"] == ["a:9092", "b:9092"]
        assert producer_config["acks"] == "all"
        assert consumer_config["client_id"] == producer_config["client_id"]
        assert consumer_config["client_id"].startswith("sensor-validator-")

    def test_the_endpoints_are_up_while_consuming_and_closed_afterwards(
        self, setup: Setup
    ) -> None:
        setup.script = [[_record(i, 50.0 + i) for i in range(3)]]
        shutdown = GracefulShutdown()
        observed: dict[str, tuple[int, str]] = {}

        def probe(count: int) -> None:
            if count == 2:
                observed["health"] = _get(setup.health_port, "/health")
                observed["ready"] = _get(setup.health_port, "/ready")
                observed["metrics"] = _get(setup.metrics_port, "/metrics")
                shutdown.request()

        setup.on_poll = probe
        assert setup.run(shutdown=shutdown) == EXIT_OK
        assert observed["health"][0] == 200
        assert observed["ready"][0] == 200
        status, body = observed["metrics"]
        assert status == 200
        assert "reactorguard_readings_consumed_total 3.0" in body
        assert "reactorguard_consumer_heartbeat_timestamp_seconds" in body
        with pytest.raises(urllib.error.URLError):
            _get(setup.health_port, "/health")

    def test_readiness_waits_for_the_group_assignment(self, setup: Setup) -> None:
        shutdown = GracefulShutdown()
        observed: dict[str, int] = {}

        def probe(count: int) -> None:
            if count == 1:
                observed["after_assignment"] = _get(setup.health_port, "/ready")[0]
                shutdown.request()

        def factory(**config: Any) -> FakeKafkaConsumer:
            consumer = setup.consumer_factory(**config)
            original = consumer.subscribe

            def subscribe(topics: list[str], listener: Any = None) -> None:
                original(topics, listener)
                observed["before_assignment"] = _get(setup.health_port, "/ready")[0]

            consumer.subscribe = subscribe  # type: ignore[method-assign]
            return consumer

        setup.on_poll = probe
        assert setup.run(shutdown=shutdown, consumer_factory=factory) == EXIT_OK
        assert observed == {"before_assignment": 503, "after_assignment": 200}

    def test_a_real_sigterm_stops_the_service_and_is_logged(
        self, setup: Setup, caplog: pytest.LogCaptureFixture
    ) -> None:
        setup.script = [[_record(i, 50.0 + i) for i in range(3)]]

        def deliver_sigterm(count: int) -> None:
            if count == 2:
                handler = signal.getsignal(signal.SIGTERM)
                assert callable(handler)
                handler(signal.SIGTERM, None)

        setup.on_poll = deliver_sigterm
        with caplog.at_level(logging.INFO):
            assert setup.run() == EXIT_OK
        assert "Stopped on SIGTERM" in caplog.text
        assert setup.consumers[0].commits

    def test_a_sensor_without_a_span_is_still_published(self, setup: Setup) -> None:
        setup.script = [[_record(0, 50.0, sensor="UNKNOWN-TAG")]]
        shutdown = GracefulShutdown()
        setup.on_poll = _stop_after(shutdown, 3)
        assert setup.run(shutdown=shutdown) == EXIT_OK
        assert [m["topic"] for m in setup.producers[0].sent] == ["sensor-validated"]


class TestFailures:
    def test_a_delivery_failure_exits_with_runtime_error_without_committing(
        self, setup: Setup
    ) -> None:
        setup.script = [[_record(i, 50.0 + i) for i in range(3)]]

        def factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.producer_factory(**config)
            producer.delivery_error = KafkaError("broker down")
            return producer

        assert setup.run(producer_factory=factory) == EXIT_RUNTIME
        assert setup.consumers[0].commits == []
        assert setup.consumers[0].closed_with is not None
        assert setup.producers[0].close_timeouts == [5.0]

    def test_a_commit_that_keeps_failing_exits_after_the_configured_retries(
        self, setup: Setup
    ) -> None:
        setup.script = [[_record(0, 50.0)]]

        def factory(**config: Any) -> FakeKafkaConsumer:
            consumer = setup.consumer_factory(**config)
            consumer.commit_error = KafkaError("rebalanced")
            return consumer

        assert setup.run(consumer_factory=factory) == EXIT_RUNTIME
        expected = setup.document["streaming"]["commit_retries"] + 1
        assert setup.consumers[0].commit_attempts == expected
        assert setup.consumers[0].commits == []

    def test_a_failing_poll_exits_with_runtime_error(self, setup: Setup) -> None:
        def broken_poll(count: int) -> None:
            raise KafkaError("coordinator unavailable")

        setup.on_poll = broken_poll
        assert setup.run() == EXIT_RUNTIME
        assert setup.producers[0].close_timeouts == [5.0]

    def test_a_consumer_that_cannot_be_created_exits_and_closes_the_producer(
        self, setup: Setup
    ) -> None:
        def broken(**config: Any) -> FakeKafkaConsumer:
            raise KafkaError("no brokers")

        assert setup.run(consumer_factory=broken) == EXIT_RUNTIME
        assert setup.producers[0].close_timeouts == [5.0]

    def test_a_producer_that_cannot_be_created_exits_before_the_consumer(
        self, setup: Setup
    ) -> None:
        def broken(**config: Any) -> FakeKafkaProducer:
            raise KafkaError("no brokers")

        assert setup.run(producer_factory=broken) == EXIT_RUNTIME
        assert setup.consumers == []

    def test_errors_while_closing_are_logged_not_raised(
        self, setup: Setup, caplog: pytest.LogCaptureFixture
    ) -> None:
        shutdown = GracefulShutdown()
        setup.on_poll = _stop_after(shutdown, 1)

        def consumer_factory(**config: Any) -> FakeKafkaConsumer:
            consumer = setup.consumer_factory(**config)

            def failing_close(autocommit: bool = True, timeout_ms: int | None = None) -> None:
                raise RuntimeError("close failed")

            consumer.close = failing_close  # type: ignore[method-assign]
            return consumer

        def producer_factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.producer_factory(**config)

            def failing_close(timeout: float | None = None) -> None:
                raise RuntimeError("close failed")

            producer.close = failing_close  # type: ignore[method-assign]
            return producer

        with caplog.at_level(logging.ERROR):
            code = setup.run(
                shutdown=shutdown,
                consumer_factory=consumer_factory,
                producer_factory=producer_factory,
            )
        assert code == EXIT_OK
        assert "Error closing the Kafka consumer" in caplog.text
        assert "Error closing the Kafka producer" in caplog.text

    def test_a_busy_port_exits_before_touching_kafka(self, setup: Setup) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", setup.health_port))
            sock.listen()
            code = setup.run()
        assert code == EXIT_RUNTIME
        assert setup.producers == []
        assert setup.consumers == []


class TestConfigurationErrors:
    def test_missing_kafka_bootstrap(self, setup: Setup) -> None:
        assert setup.run({}) == EXIT_CONFIG
        assert setup.producers == []

    def test_missing_params_file(self, tmp_path: Path, setup: Setup) -> None:
        code = main(
            tmp_path / "absent.yaml",
            BASE_ENV,
            producer_factory=setup.producer_factory,
            consumer_factory=setup.consumer_factory,
        )
        assert code == EXIT_CONFIG

    def test_missing_streaming_section(self, setup: Setup) -> None:
        del setup.document["streaming"]
        setup.rewrite()
        assert setup.run() == EXIT_CONFIG

    def test_inconsistent_streaming_values(self, setup: Setup) -> None:
        setup.document["streaming"]["max_poll_interval_ms"] = 1000
        setup.rewrite()
        assert setup.run() == EXIT_CONFIG

    def test_missing_span_table(self, setup: Setup, tmp_path: Path) -> None:
        setup.document["tep"]["spans_path"] = str(tmp_path / "absent.yaml")
        setup.rewrite()
        assert setup.run() == EXIT_CONFIG
        assert setup.producers == []

    def test_malformed_span_table(self, setup: Setup, tmp_path: Path) -> None:
        bad = tmp_path / "bad_spans.yaml"
        bad.write_text(yaml.safe_dump({"sensors": {"TEP-XMEAS-01": 5}}), encoding="utf-8")
        setup.document["tep"]["spans_path"] = str(bad)
        setup.rewrite()
        assert setup.run() == EXIT_CONFIG


class TestHelpers:
    def test_the_warm_up_is_the_stuck_window_times_the_sampling_interval(
        self, setup: Setup
    ) -> None:
        validator = SensorValidator({})
        tep = load_tep_params(setup.params_path)
        assert validation_service._warmup_seconds(validator, tep) == 8 * 180.0  # noqa: SLF001

    def test_a_validator_without_a_stuck_detector_has_no_warm_up_estimate(
        self, setup: Setup
    ) -> None:
        validator = SensorValidator({}, detectors=[RangeValidator({})])
        tep = load_tep_params(setup.params_path)
        assert validation_service._warmup_seconds(validator, tep) is None  # noqa: SLF001

    def test_the_client_id_carries_the_host_name(
        self, setup: Setup, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from data.streaming.streaming_params import load_streaming_params

        monkeypatch.setattr(socket, "gethostname", lambda: "pod-7")
        streaming = load_streaming_params(setup.params_path)
        assert validation_service._client_id(streaming) == "sensor-validator-pod-7"  # noqa: SLF001


@pytest.fixture()
def clean_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
    root.setLevel(level)


class TestCli:
    def test_passes_the_params_option_to_main(
        self, setup: Setup, monkeypatch: pytest.MonkeyPatch, clean_logging: None
    ) -> None:
        monkeypatch.delenv("KAFKA_BOOTSTRAP", raising=False)
        assert cli(["--params", str(setup.params_path)]) == EXIT_CONFIG

    def test_a_missing_params_file_is_a_configuration_error(
        self, tmp_path: Path, clean_logging: None
    ) -> None:
        assert cli(["--params", str(tmp_path / "absent.yaml")]) == EXIT_CONFIG

    def test_an_invalid_log_level_is_a_configuration_error(
        self,
        setup: Setup,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        clean_logging: None,
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "LOUD")
        assert cli(["--params", str(setup.params_path)]) == EXIT_CONFIG
        assert "LOG_LEVEL" in capsys.readouterr().err

