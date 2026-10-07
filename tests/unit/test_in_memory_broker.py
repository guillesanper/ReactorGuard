"""Tests for tests/support/in_memory_broker.py (the double used by the stream tests)."""

from __future__ import annotations

import pytest
from kafka.partitioner.default import DefaultPartitioner

from data.generators.tep_adapter import sensor_id
from data.streaming.errors import PublishError
from tests.support.in_memory_broker import InMemoryBroker, partition_for

TOPIC = "sensor-readings-raw"


class FakeCluster:
    """The three ClusterMetadata calls DefaultPartitioner makes."""

    def __init__(self, partitions: int) -> None:
        self._partitions = set(range(partitions))

    def topics(self) -> set[str]:
        return {TOPIC}

    def partitions_for_topic(self, topic: str) -> set[int]:
        return self._partitions

    def available_partitions_for_topic(self, topic: str) -> set[int]:
        return self._partitions


class TestPartitionFor:
    @pytest.mark.parametrize("partitions", [1, 3, 12, 50])
    def test_matches_the_real_kafka_python_partitioner(self, partitions: int) -> None:
        cluster = FakeCluster(partitions)
        partitioner = DefaultPartitioner()
        keys = [sensor_id(col).encode("utf-8") for col in range(52)] + [b"", b"a", b"\xff\x00"]
        for key in keys:
            expected = partitioner.partition(TOPIC, key, key, None, None, cluster)
            assert partition_for(key, partitions) == expected

    def test_is_deterministic_and_in_range(self) -> None:
        for col in range(52):
            key = sensor_id(col).encode("utf-8")
            assert partition_for(key, 12) == partition_for(key, 12)
            assert 0 <= partition_for(key, 12) < 12


class TestInMemoryBroker:
    def test_a_key_always_goes_to_the_same_partition_with_growing_offsets(self) -> None:
        broker = InMemoryBroker(12)
        for index in range(5):
            broker.send(TOPIC, b"TEP-XMV-04", f"v{index}".encode(), [("run-id", b"r")])
        partitions = broker.partitions_of(TOPIC)
        assert len(partitions) == 1
        (records,) = partitions.values()
        assert [record.offset for record in records] == [0, 1, 2, 3, 4]
        assert [record.value for record in records] == [f"v{i}".encode() for i in range(5)]
        assert records[0].header("run-id") == b"r"
        assert broker.records(TOPIC, records[0].partition) == records

    def test_messages_without_key_are_spread_round_robin(self) -> None:
        broker = InMemoryBroker(3)
        for _ in range(6):
            broker.send(TOPIC, None, b"x")
        assert {p: len(r) for p, r in broker.partitions_of(TOPIC).items()} == {0: 2, 1: 2, 2: 2}

    def test_unknown_partition_is_empty(self) -> None:
        assert InMemoryBroker(2).records(TOPIC, 1) == []

    def test_sent_keeps_every_message_in_order(self) -> None:
        broker = InMemoryBroker(2)
        broker.send("a", b"k", b"1")
        broker.send("b", None, b"2")
        assert [(m.topic, m.value) for m in broker.sent] == [("a", b"1"), ("b", b"2")]

    def test_flush_is_counted_and_can_fail(self) -> None:
        broker = InMemoryBroker(2)
        broker.flush(1.0)
        assert broker.flushes == 1
        broker.fail_flush_with = PublishError("boom")
        with pytest.raises(PublishError, match="boom"):
            broker.flush()
        assert broker.flushes == 2

    def test_send_fails_after_the_configured_count(self) -> None:
        broker = InMemoryBroker(2)
        broker.fail_send_after = 2
        broker.send(TOPIC, b"k", b"1")
        broker.send(TOPIC, b"k", b"2")
        with pytest.raises(PublishError):
            broker.send(TOPIC, b"k", b"3")
        assert len(broker.sent) == 2

    def test_close_is_recorded(self) -> None:
        broker = InMemoryBroker(2)
        broker.close(1.0)
        assert broker.closed

    def test_partitions_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="partitions"):
            InMemoryBroker(0)
