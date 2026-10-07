"""Tests for data/generators/tep_streamer_service.py.

El servicio se ejecuta de punta a punta con un KafkaProducer falso (a traves de
`producer_factory`), parquet sintetico en tmp_path, puertos HTTP libres y, para el
origen GCS, un BlobStore local. Nada toca la red ni Kafka.
"""

from __future__ import annotations

import copy
import logging
import signal
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from kafka.errors import KafkaError

from data.generators import tep_streamer_service
from data.generators.tep_streamer_service import EXIT_CONFIG, EXIT_OK, EXIT_RUNTIME, cli, main
from data.storage.blob_store import LocalBlobStore
from data.storage.storage_params import StorageParams
from data.streaming.lifecycle import GracefulShutdown
from tests.support.tep_frames import make_run_frame, write_partition

SENSORS = 52
TIMESTEPS = 3
RUNS = 2
BASE_ENV = {"KAFKA_BOOTSTRAP": "broker:9092"}


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
        self.on_send: Callable[[int], None] | None = None

    def send(self, topic: str, **kwargs: Any) -> FakeFuture:
        self.sent.append({"topic": topic, **kwargs})
        if self.on_send is not None:
            self.on_send(len(self.sent))
        return FakeFuture(self.delivery_error)

    def flush(self, timeout: float | None = None) -> None:
        del timeout
        self.flushes += 1

    def close(self, timeout: float | None = None) -> None:
        self.close_timeouts.append(timeout)


@dataclass
class Setup:
    """A params file, its data and the ports the service will bind."""

    params_path: Path
    parquet_root: Path
    health_port: int
    metrics_port: int
    document: dict[str, Any]
    producers: list[FakeKafkaProducer]

    def factory(self, **config: Any) -> FakeKafkaProducer:
        producer = FakeKafkaProducer(**config)
        self.producers.append(producer)
        return producer

    def rewrite(self) -> None:
        self.params_path.write_text(yaml.safe_dump(self.document, sort_keys=False), "utf-8")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def setup(tmp_path: Path) -> Setup:
    root = tmp_path / "tep"
    for fault_type in range(RUNS):
        write_partition(root, fault_type, make_run_frame(fault_type, TIMESTEPS))
    document: dict[str, Any] = copy.deepcopy(
        yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    )
    document["tep"]["processed_dir"] = str(root)
    document["streaming"]["bind_host"] = "127.0.0.1"
    health, metrics = _free_port(), _free_port()
    document["streaming"]["health_port"] = health
    document["streaming"]["metrics_port"] = metrics
    prepared = Setup(tmp_path / "params.yaml", root, health, metrics, document, [])
    prepared.rewrite()
    return prepared


def _get(port: int, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:  # noqa: S310
            return int(response.status), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


class TestRun:
    def test_streams_every_reading_and_exits_cleanly(self, setup: Setup) -> None:
        code = main(setup.params_path, BASE_ENV, producer_factory=setup.factory)
        assert code == EXIT_OK
        (producer,) = setup.producers
        assert len(producer.sent) == RUNS * TIMESTEPS * SENSORS
        assert {message["topic"] for message in producer.sent} == {"sensor-readings-raw"}
        assert all(message["key"] for message in producer.sent)
        assert producer.sent[0]["headers"] == [("run-id", b"fault_type=00/loop=0")]
        assert producer.flushes >= 1
        assert producer.close_timeouts == [5.0]

    def test_the_producer_is_built_from_the_kafka_environment(self, setup: Setup) -> None:
        env = {**BASE_ENV, "KAFKA_BOOTSTRAP": "a:9092,b:9092"}
        assert main(setup.params_path, env, producer_factory=setup.factory) == EXIT_OK
        config = setup.producers[0].config
        assert config["bootstrap_servers"] == ["a:9092", "b:9092"]
        assert config["security_protocol"] == "PLAINTEXT"
        assert config["client_id"] == "tep-streamer"
        assert config["acks"] == "all"

    def test_the_gcs_source_reads_the_raw_bucket_through_the_blob_store(
        self, setup: Setup, tmp_path: Path
    ) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        for fault_type in range(RUNS):
            partition = setup.parquet_root / f"fault_type={fault_type:02d}" / "readings.parquet"
            store.put_if_absent(f"tep/fault_type={fault_type:02d}/readings.parquet",
                                partition.read_bytes())
        for partition_dir in list(setup.parquet_root.iterdir()):
            for file in partition_dir.iterdir():
                file.unlink()
            partition_dir.rmdir()
        opened: list[StorageParams] = []

        def factory(storage: StorageParams) -> LocalBlobStore:
            opened.append(storage)
            return store

        code = main(
            setup.params_path,
            {**BASE_ENV, "DATA_SOURCE": "gcs"},
            producer_factory=setup.factory,
            blob_store_factory=factory,
        )
        assert code == EXIT_OK
        assert [storage.raw_bucket for storage in opened] == ["reactorguard-data-raw-dev"]
        assert len(setup.producers[0].sent) == RUNS * TIMESTEPS * SENSORS

    def test_the_endpoints_are_up_while_streaming_and_closed_afterwards(
        self, setup: Setup
    ) -> None:
        observed: dict[str, tuple[int, str]] = {}

        def probe(count: int) -> None:
            if count == SENSORS:
                observed["health"] = _get(setup.health_port, "/health")
                observed["ready"] = _get(setup.health_port, "/ready")
                observed["metrics"] = _get(setup.metrics_port, "/metrics")

        def factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.factory(**config)
            producer.on_send = probe
            return producer

        assert main(setup.params_path, BASE_ENV, producer_factory=factory) == EXIT_OK
        assert observed["health"][0] == 200
        assert observed["ready"][0] == 200
        assert observed["metrics"][0] == 200
        assert "reactorguard_stream_messages_produced_total" in observed["metrics"][1]
        assert "reactorguard_consumer_heartbeat_timestamp_seconds" in observed["metrics"][1]
        with pytest.raises(urllib.error.URLError):
            _get(setup.health_port, "/health")

    def test_sigterm_style_shutdown_stops_a_realtime_wait(self, setup: Setup) -> None:
        setup.document["streamer"]["mode"] = "realtime"
        setup.rewrite()
        shutdown = GracefulShutdown()

        def factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.factory(**config)
            producer.on_send = lambda count: shutdown.request() if count == SENSORS else None
            return producer

        code = main(setup.params_path, BASE_ENV, producer_factory=factory, shutdown=shutdown)
        assert code == EXIT_OK
        assert len(setup.producers[0].sent) == SENSORS
        assert setup.producers[0].close_timeouts == [5.0]

    def test_a_real_sigterm_stops_the_service_and_is_logged(
        self, setup: Setup, caplog: pytest.LogCaptureFixture
    ) -> None:
        setup.document["streamer"]["mode"] = "realtime"
        setup.rewrite()

        def deliver_sigterm(count: int) -> None:
            if count == SENSORS:
                handler = signal.getsignal(signal.SIGTERM)
                assert callable(handler)
                handler(signal.SIGTERM, None)

        def factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.factory(**config)
            producer.on_send = deliver_sigterm
            return producer

        with caplog.at_level(logging.INFO):
            assert main(setup.params_path, BASE_ENV, producer_factory=factory) == EXIT_OK
        assert len(setup.producers[0].sent) == SENSORS
        assert "Stopped on SIGTERM" in caplog.text

    def test_environment_overrides_reach_the_streamer(self, setup: Setup) -> None:
        setup.document["streamer"]["loop"] = True
        setup.rewrite()
        shutdown = GracefulShutdown()
        runs_seen: list[bytes] = []

        def factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.factory(**config)

            def on_send(count: int) -> None:
                runs_seen.append(producer.sent[-1]["headers"][0][1])
                if count == RUNS * TIMESTEPS * SENSORS + 1:
                    shutdown.request()

            producer.on_send = on_send
            return producer

        env = {**BASE_ENV, "STREAM_MODE": "fast", "SPEED_MULTIPLIER": "5"}
        assert main(setup.params_path, env, producer_factory=factory, shutdown=shutdown) == EXIT_OK
        assert b"fault_type=00/loop=1" in runs_seen


class TestFailures:
    def test_a_delivery_failure_exits_with_runtime_error_and_closes(self, setup: Setup) -> None:
        def factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.factory(**config)
            producer.delivery_error = KafkaError("broker down")
            return producer

        assert main(setup.params_path, BASE_ENV, producer_factory=factory) == EXIT_RUNTIME
        assert setup.producers[0].close_timeouts == [5.0]

    def test_a_missing_data_directory_exits_with_runtime_error(self, setup: Setup) -> None:
        setup.document["tep"]["processed_dir"] = str(setup.parquet_root / "absent")
        setup.rewrite()
        code = main(setup.params_path, BASE_ENV, producer_factory=setup.factory)
        assert code == EXIT_RUNTIME
        assert setup.producers[0].sent == []
        assert setup.producers[0].close_timeouts == [5.0]

    def test_a_producer_that_cannot_be_created_exits_with_runtime_error(
        self, setup: Setup
    ) -> None:
        def broken(**config: Any) -> FakeKafkaProducer:
            raise KafkaError("no brokers")

        assert main(setup.params_path, BASE_ENV, producer_factory=broken) == EXIT_RUNTIME

    def test_an_error_while_closing_the_producer_is_logged_not_raised(
        self, setup: Setup, caplog: pytest.LogCaptureFixture
    ) -> None:
        def factory(**config: Any) -> FakeKafkaProducer:
            producer = setup.factory(**config)

            def failing_close(timeout: float | None = None) -> None:
                raise RuntimeError("close failed")

            producer.close = failing_close  # type: ignore[method-assign]
            return producer

        with caplog.at_level(logging.ERROR):
            assert main(setup.params_path, BASE_ENV, producer_factory=factory) == EXIT_OK
        assert "Error closing the Kafka producer" in caplog.text

    def test_a_busy_port_exits_before_touching_kafka(self, setup: Setup) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", setup.health_port))
            sock.listen()
            code = main(setup.params_path, BASE_ENV, producer_factory=setup.factory)
        assert code == EXIT_RUNTIME
        assert setup.producers == []


class TestConfigurationErrors:
    def test_missing_kafka_bootstrap(self, setup: Setup) -> None:
        assert main(setup.params_path, {}, producer_factory=setup.factory) == EXIT_CONFIG
        assert setup.producers == []

    def test_missing_params_file(self, tmp_path: Path, setup: Setup) -> None:
        code = main(tmp_path / "absent.yaml", BASE_ENV, producer_factory=setup.factory)
        assert code == EXIT_CONFIG

    def test_missing_streamer_section(self, setup: Setup) -> None:
        del setup.document["streamer"]
        setup.rewrite()
        assert main(setup.params_path, BASE_ENV, producer_factory=setup.factory) == EXIT_CONFIG

    @pytest.mark.parametrize(
        ("name", "value"),
        [("STREAM_MODE", "warp"), ("SPEED_MULTIPLIER", "0"), ("DATA_SOURCE", "s3")],
    )
    def test_invalid_environment_override(self, setup: Setup, name: str, value: str) -> None:
        env = {**BASE_ENV, name: value}
        assert main(setup.params_path, env, producer_factory=setup.factory) == EXIT_CONFIG
        assert setup.producers == []

    def test_wait_slice_that_would_starve_the_heartbeat(self, setup: Setup) -> None:
        setup.document["streamer"]["wait_slice_s"] = 540.0
        setup.rewrite()
        assert main(setup.params_path, BASE_ENV, producer_factory=setup.factory) == EXIT_CONFIG


class TestHelpers:
    def test_the_default_blob_store_is_the_raw_bucket(self, setup: Setup) -> None:
        captured: dict[str, Any] = {}

        class FakeStore:
            def __init__(self, bucket: str, *, timeout_s: float) -> None:
                captured.update(bucket=bucket, timeout_s=timeout_s)

        original = tep_streamer_service.GCSBlobStore
        tep_streamer_service.GCSBlobStore = FakeStore  # type: ignore[misc,assignment]
        try:
            storage = StorageParams(
                bucket_prefix="reactorguard",
                env="dev",
                local_root=Path("x"),
                read_workers=4,
                request_timeout_s=7.0,
            )
            tep_streamer_service._gcs_store(storage)
        finally:
            tep_streamer_service.GCSBlobStore = original  # type: ignore[misc]
        assert captured == {"bucket": "reactorguard-data-raw-dev", "timeout_s": 7.0}


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
