"""Tests for data/streaming/kafka_adapters.py.

Los constructores de KafkaProducer/KafkaConsumer se sustituyen por fakes via el
parametro `*_factory` (sin parchear internos de la libreria). Ademas se comprueba
que cada kwarg existe en el DEFAULT_CONFIG real de kafka-python: asi un nombre mal
escrito falla aqui y no al conectar con un broker.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

import pytest
from kafka import KafkaConsumer, KafkaProducer
from kafka.consumer.fetcher import ConsumerRecord
from kafka.errors import KafkaError, KafkaTimeoutError
from kafka.structs import OffsetAndMetadata, TopicPartition

from data.streaming import kafka_adapters
from data.streaming.errors import CommitError, PublishError, StreamingError
from data.streaming.kafka_adapters import KafkaMessageConsumer, KafkaMessageProducer
from data.streaming.kafka_settings import KafkaConnectionSettings
from data.streaming.streaming_params import StreamingParams, load_streaming_params
from data.streaming.transport import PartitionRef


@pytest.fixture()
def params() -> StreamingParams:
    """Return the real streaming parameters of the repository."""
    return load_streaming_params()


@pytest.fixture()
def settings() -> KafkaConnectionSettings:
    """Return PLAINTEXT settings."""
    return KafkaConnectionSettings(bootstrap_servers=("broker:9092",))


class FakeFuture:
    """Stand-in for FutureRecordMetadata."""

    def __init__(self, error: KafkaError | None = None, done: bool = True) -> None:
        self.error = error
        self.is_done = done
        self.get_timeouts: list[float | None] = []

    def succeeded(self) -> bool:
        return self.is_done and self.error is None

    def get(self, timeout: float | None = None) -> str:
        self.get_timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return "metadata"


class FakeProducer:
    """Stand-in for KafkaProducer recording every call."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.sent: list[dict[str, Any]] = []
        self.futures: list[FakeFuture] = []
        self.next_error: KafkaError | None = None
        self.send_error: KafkaError | None = None
        self.flush_error: KafkaError | None = None
        self.flush_timeouts: list[float | None] = []
        self.closed_with: list[float | None] = []

    def send(self, topic: str, **kwargs: Any) -> FakeFuture:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append({"topic": topic, **kwargs})
        future = FakeFuture(error=self.next_error)
        self.futures.append(future)
        return future

    def flush(self, timeout: float | None = None) -> None:
        self.flush_timeouts.append(timeout)
        if self.flush_error is not None:
            raise self.flush_error

    def close(self, timeout: float | None = None) -> None:
        self.closed_with.append(timeout)


class FakeConsumer:
    """Stand-in for KafkaConsumer recording every call."""

    def __init__(self, **config: Any) -> None:
        self.config = config
        self.subscribed: dict[str, Any] = {}
        self.polled: dict[TopicPartition, list[ConsumerRecord]] = {}
        self.poll_args: dict[str, Any] = {}
        self.committed: dict[str, Any] = {}
        self.error: KafkaError | None = None
        self.assigned: set[TopicPartition] = set()
        self.positions: dict[TopicPartition, int] = {}
        self.ends: dict[TopicPartition, int] = {}
        self.close_args: dict[str, Any] = {}

    def subscribe(self, topics: list[str], listener: Any = None) -> None:
        self.subscribed = {"topics": topics, "listener": listener}

    def poll(self, timeout_ms: int, max_records: int | None) -> dict[Any, list[Any]]:
        if self.error is not None:
            raise self.error
        self.poll_args = {"timeout_ms": timeout_ms, "max_records": max_records}
        return self.polled

    def commit(self, offsets: dict[TopicPartition, OffsetAndMetadata], timeout_ms: int) -> None:
        if self.error is not None:
            raise self.error
        self.committed = {"offsets": offsets, "timeout_ms": timeout_ms}

    def assignment(self) -> set[TopicPartition]:
        return self.assigned

    def position(self, partition: TopicPartition) -> int:
        if self.error is not None:
            raise self.error
        return self.positions[partition]

    def end_offsets(self, partitions: list[TopicPartition]) -> dict[TopicPartition, int]:
        if self.error is not None:
            raise self.error
        return {tp: self.ends[tp] for tp in partitions}

    def close(self, autocommit: bool, timeout_ms: int) -> None:
        self.close_args = {"autocommit": autocommit, "timeout_ms": timeout_ms}


def _producer(
    settings: KafkaConnectionSettings, params: StreamingParams, **kwargs: Any
) -> tuple[KafkaMessageProducer, FakeProducer]:
    holder: list[FakeProducer] = []

    def factory(**config: Any) -> FakeProducer:
        fake = FakeProducer(**config)
        holder.append(fake)
        return fake

    adapter = KafkaMessageProducer(settings, params, producer_factory=factory, **kwargs)
    return adapter, holder[0]


def _consumer(
    settings: KafkaConnectionSettings, params: StreamingParams, **kwargs: Any
) -> tuple[KafkaMessageConsumer, FakeConsumer]:
    holder: list[FakeConsumer] = []

    def factory(**config: Any) -> FakeConsumer:
        fake = FakeConsumer(**config)
        holder.append(fake)
        return fake

    adapter = KafkaMessageConsumer(settings, params, consumer_factory=factory, **kwargs)
    return adapter, holder[0]


class TestProducerConfig:
    def test_delivery_guarantees_are_explicit(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, fake = _producer(settings, params)
        assert fake.config["acks"] == "all"
        assert fake.config["enable_idempotence"] is True

    def test_tunables_come_from_params(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, fake = _producer(settings, params)
        assert fake.config["compression_type"] == params.compression_type == "lz4"
        assert fake.config["linger_ms"] == params.batch_linger_ms
        assert fake.config["batch_size"] == params.producer_batch_bytes
        assert fake.config["max_in_flight_requests_per_connection"] == params.max_in_flight_requests
        assert fake.config["request_timeout_ms"] == params.request_timeout_ms
        assert fake.config["delivery_timeout_ms"] == params.delivery_timeout_ms

    def test_connection_settings_are_forwarded(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, fake = _producer(settings, params)
        assert fake.config["bootstrap_servers"] == ["broker:9092"]
        assert fake.config["security_protocol"] == "PLAINTEXT"

    def test_client_id_only_when_given(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, without = _producer(settings, params)
        _, with_id = _producer(settings, params, client_id="streamer-1")
        assert "client_id" not in without.config
        assert with_id.config["client_id"] == "streamer-1"

    def test_every_kwarg_exists_in_the_real_library(
        self, tmp_path: Path, params: StreamingParams
    ) -> None:
        ca, cert, key = tmp_path / "ca", tmp_path / "crt", tmp_path / "key"
        for path in (ca, cert, key):
            path.write_text("x", encoding="utf-8")
        tls = KafkaConnectionSettings(
            bootstrap_servers=("b:9093",),
            security_protocol="SASL_SSL",
            ssl_cafile=ca,
            ssl_certfile=cert,
            ssl_keyfile=key,
            ssl_key_password="p",
            sasl_mechanism="PLAIN",
            sasl_username="u",
            sasl_password="p",
        )
        _, fake = _producer(tls, params, client_id="c")
        assert set(fake.config) <= set(KafkaProducer.DEFAULT_CONFIG)

    def test_factory_failure_becomes_publish_error(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        def boom(**_: Any) -> None:
            raise KafkaError("no brokers")

        with pytest.raises(PublishError, match="no brokers"):
            KafkaMessageProducer(settings, params, producer_factory=boom)


class TestProducerSend:
    def test_send_forwards_key_value_and_headers(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        adapter.send("t", b"TEP-XMV-04", b"{}", [("run-id", b"fault_type=21;loop=0")])
        assert fake.sent == [
            {
                "topic": "t",
                "value": b"{}",
                "key": b"TEP-XMV-04",
                "headers": [("run-id", b"fault_type=21;loop=0")],
            }
        ]

    def test_headers_default_to_empty_list(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        adapter.send("t", None, b"v")
        assert fake.sent[0]["headers"] == []
        assert fake.sent[0]["key"] is None

    def test_send_error_becomes_publish_error(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        fake.send_error = KafkaTimeoutError("buffer full")
        with pytest.raises(PublishError, match="buffer full"):
            adapter.send("t", b"k", b"v")

    def test_resolved_futures_are_pruned_without_flush(
        self,
        settings: KafkaConnectionSettings,
        params: StreamingParams,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(kafka_adapters, "_PRUNE_THRESHOLD", 10)
        adapter, fake = _producer(settings, params)
        adapter._prune_at = 10
        for _ in range(100):
            adapter.send("t", b"k", b"v")
        assert len(adapter._pending) < 20
        assert len(fake.sent) == 100

    def test_prune_keeps_failed_and_pending_futures(
        self,
        settings: KafkaConnectionSettings,
        params: StreamingParams,
    ) -> None:
        adapter, fake = _producer(settings, params)
        failed = FakeFuture(error=KafkaError("lost"))
        pending = FakeFuture(done=False)
        ok = FakeFuture()
        adapter._pending = [failed, pending, ok]
        adapter._prune_resolved()
        assert adapter._pending == [failed, pending]
        assert fake.sent == []


class TestProducerFlush:
    def test_flush_passes_when_every_future_succeeds(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        adapter.send("t", b"k", b"v1")
        adapter.send("t", b"k", b"v2")
        adapter.flush(timeout=5.0)
        assert fake.flush_timeouts == [5.0]
        assert all(future.get_timeouts == [5.0] for future in fake.futures)

    def test_flush_raises_when_a_future_failed(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        # KafkaProducer.flush() no lanza si un registro falla: hay que mirar los futures.
        adapter, fake = _producer(settings, params)
        adapter.send("t", b"k", b"ok")
        fake.next_error = KafkaError("record too large")
        adapter.send("t", b"k", b"bad")
        with pytest.raises(PublishError, match=r"1 of 2 messages failed.*record too large"):
            adapter.flush(timeout=1.0)

    def test_flush_timeout_becomes_publish_error(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        adapter.send("t", b"k", b"v")
        fake.flush_error = KafkaTimeoutError("timed out")
        with pytest.raises(PublishError, match="timed out"):
            adapter.flush(timeout=0.1)

    def test_failures_are_reported_once(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        fake.next_error = KafkaError("boom")
        adapter.send("t", b"k", b"v")
        with pytest.raises(PublishError):
            adapter.flush()
        fake.next_error = None
        adapter.send("t", b"k", b"v")
        adapter.flush()

    def test_flush_with_nothing_pending_is_a_noop_success(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        adapter.flush()
        assert fake.flush_timeouts == [None]

    def test_close_forwards_timeout(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _producer(settings, params)
        adapter.close(timeout=2.5)
        assert fake.closed_with == [2.5]


class TestConsumerConfig:
    def test_commits_are_manual(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, fake = _consumer(settings, params)
        assert fake.config["enable_auto_commit"] is False

    def test_tunables_come_from_params(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, fake = _consumer(settings, params)
        assert fake.config["group_id"] == params.consumer_group == "sensor-validator"
        assert fake.config["auto_offset_reset"] == params.auto_offset_reset
        assert fake.config["isolation_level"] == params.isolation_level
        assert fake.config["max_poll_records"] == params.max_poll_records
        assert fake.config["max_poll_interval_ms"] == params.max_poll_interval_ms

    def test_group_and_client_override(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, fake = _consumer(settings, params, group_id="other", client_id="c1")
        assert fake.config["group_id"] == "other"
        assert fake.config["client_id"] == "c1"

    def test_every_kwarg_exists_in_the_real_library(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        _, fake = _consumer(settings, params, client_id="c1")
        assert set(fake.config) <= set(KafkaConsumer.DEFAULT_CONFIG)

    def test_factory_failure_becomes_streaming_error(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        def boom(**_: Any) -> None:
            raise KafkaError("no brokers")

        with pytest.raises(StreamingError, match="no brokers"):
            KafkaMessageConsumer(settings, params, consumer_factory=boom)


def _record(partition: int, offset: int, headers: list[tuple[str, bytes]] | None = None) -> Any:
    return ConsumerRecord(
        topic="raw",
        partition=partition,
        leader_epoch=0,
        offset=offset,
        timestamp=1_700_000_000_000 + offset,
        timestamp_type=0,
        key=b"TEP-XMEAS-01",
        value=b"payload",
        headers=headers or [],
        checksum=None,
        serialized_key_size=12,
        serialized_value_size=7,
        serialized_header_size=0,
    )


class TestConsumerPoll:
    def test_records_are_flattened_with_timestamp_mapped(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        fake.polled = {
            TopicPartition("raw", 0): [_record(0, 5, [("run-id", b"r1")]), _record(0, 6)],
            TopicPartition("raw", 3): [_record(3, 9)],
        }
        records = adapter.poll(timeout_ms=250, max_records=10)
        assert fake.poll_args == {"timeout_ms": 250, "max_records": 10}
        assert [(r.partition, r.offset) for r in records] == [(0, 5), (0, 6), (3, 9)]
        first = records[0]
        assert first.timestamp_ms == 1_700_000_000_005
        assert first.key == b"TEP-XMEAS-01"
        assert first.value == b"payload"
        assert first.headers == (("run-id", b"r1"),)
        assert first.partition_ref == PartitionRef("raw", 0)
        assert first.header("run-id") == b"r1"
        assert first.header("absent") is None

    def test_empty_poll(self, settings: KafkaConnectionSettings, params: StreamingParams) -> None:
        adapter, _ = _consumer(settings, params)
        assert adapter.poll(timeout_ms=10) == []

    def test_poll_error(self, settings: KafkaConnectionSettings, params: StreamingParams) -> None:
        adapter, fake = _consumer(settings, params)
        fake.error = KafkaError("fetch failed")
        with pytest.raises(StreamingError, match="fetch failed"):
            adapter.poll(timeout_ms=10)


class TestConsumerCommit:
    def test_commit_sends_next_offsets_with_timeout(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        adapter.commit({PartitionRef("raw", 0): 11, PartitionRef("raw", 2): 4})
        assert fake.committed["timeout_ms"] == params.commit_timeout_ms
        assert fake.committed["offsets"] == {
            TopicPartition("raw", 0): OffsetAndMetadata(11, "", -1),
            TopicPartition("raw", 2): OffsetAndMetadata(4, "", -1),
        }

    def test_commit_error(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        fake.error = KafkaError("rebalance in progress")
        with pytest.raises(CommitError, match="rebalance in progress"):
            adapter.commit({PartitionRef("raw", 0): 1})


class TestConsumerIntrospection:
    def test_assignment_position_and_end_offsets(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        tp0, tp1 = TopicPartition("raw", 0), TopicPartition("raw", 1)
        fake.assigned = {tp0, tp1}
        fake.positions = {tp0: 7, tp1: 3}
        fake.ends = {tp0: 10, tp1: 3}
        assert adapter.assignment() == frozenset({PartitionRef("raw", 0), PartitionRef("raw", 1)})
        assert adapter.position(PartitionRef("raw", 0)) == 7
        ends = adapter.end_offsets([PartitionRef("raw", 0), PartitionRef("raw", 1)])
        assert ends == {PartitionRef("raw", 0): 10, PartitionRef("raw", 1): 3}
        lag = {p: ends[p] - adapter.position(p) for p in ends}
        assert lag == {PartitionRef("raw", 0): 3, PartitionRef("raw", 1): 0}

    def test_position_error(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        fake.error = KafkaError("not assigned")
        with pytest.raises(StreamingError, match="not assigned"):
            adapter.position(PartitionRef("raw", 0))

    def test_end_offsets_error(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        fake.error = KafkaError("timeout")
        with pytest.raises(StreamingError, match="timeout"):
            adapter.end_offsets([PartitionRef("raw", 0)])

    def test_close_does_not_autocommit(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        adapter.close()
        assert fake.close_args == {"autocommit": False, "timeout_ms": params.close_timeout_ms}


class _RecordingListener:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[PartitionRef]]] = []

    def on_partitions_revoked(self, revoked: Collection[PartitionRef]) -> None:
        self.calls.append(("revoked", list(revoked)))

    def on_partitions_assigned(self, assigned: Collection[PartitionRef]) -> None:
        self.calls.append(("assigned", list(assigned)))

    def on_partitions_lost(self, lost: Collection[PartitionRef]) -> None:
        self.calls.append(("lost", list(lost)))


class TestSubscribe:
    def test_subscribe_without_listener(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        adapter.subscribe(["raw"])
        assert fake.subscribed == {"topics": ["raw"], "listener": None}

    def test_listener_callbacks_are_translated(
        self, settings: KafkaConnectionSettings, params: StreamingParams
    ) -> None:
        adapter, fake = _consumer(settings, params)
        listener = _RecordingListener()
        adapter.subscribe(("raw",), listener)
        bridge = fake.subscribed["listener"]
        bridge.on_partitions_revoked([TopicPartition("raw", 1)])
        bridge.on_partitions_assigned([TopicPartition("raw", 2), TopicPartition("raw", 3)])
        bridge.on_partitions_lost([TopicPartition("raw", 4)])
        assert listener.calls == [
            ("revoked", [PartitionRef("raw", 1)]),
            ("assigned", [PartitionRef("raw", 2), PartitionRef("raw", 3)]),
            ("lost", [PartitionRef("raw", 4)]),
        ]
