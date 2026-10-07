"""Tests for tests/support/in_memory_consumer.py (the doubles of the validation tests)."""

from __future__ import annotations

from collections.abc import Collection

import pytest

from data.streaming.errors import CommitError, StreamingError
from data.streaming.transport import PartitionRef
from tests.support.in_memory_broker import InMemoryBroker, partition_for
from tests.support.in_memory_consumer import CallLog, InMemoryConsumer, RecordingProducer

TOPIC = "sensor-readings-raw"


class RecordingListener:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[int]]] = []

    def on_partitions_revoked(self, revoked: Collection[PartitionRef]) -> None:
        self.calls.append(("revoked", sorted(p.partition for p in revoked)))

    def on_partitions_assigned(self, assigned: Collection[PartitionRef]) -> None:
        self.calls.append(("assigned", sorted(p.partition for p in assigned)))

    def on_partitions_lost(self, lost: Collection[PartitionRef]) -> None:
        self.calls.append(("lost", sorted(p.partition for p in lost)))


def _fill(broker: InMemoryBroker, keys: list[bytes], per_key: int) -> None:
    for index in range(per_key):
        for key in keys:
            broker.send(TOPIC, key, f"{key.decode()}-{index}".encode())


class TestPoll:
    def test_returns_records_by_partition_in_offset_order(self) -> None:
        broker = InMemoryBroker(4)
        _fill(broker, [b"a", b"b", b"c"], per_key=3)
        consumer = InMemoryConsumer(broker)
        consumer.subscribe([TOPIC])
        records = consumer.poll(1000)
        assert len(records) == 9
        partitions = [record.partition for record in records]
        assert partitions == sorted(partitions)
        for number in set(partitions):
            offsets = [r.offset for r in records if r.partition == number]
            assert offsets == list(range(len(offsets)))

    def test_the_next_poll_continues_where_the_last_stopped(self) -> None:
        broker = InMemoryBroker(1)
        _fill(broker, [b"a"], per_key=5)
        consumer = InMemoryConsumer(broker)
        consumer.subscribe([TOPIC])
        first = consumer.poll(1000, max_records=2)
        second = consumer.poll(1000, max_records=2)
        third = consumer.poll(1000)
        assert [r.offset for r in first] == [0, 1]
        assert [r.offset for r in second] == [2, 3]
        assert [r.offset for r in third] == [4]
        assert consumer.poll(1000) == []

    def test_only_the_owned_partitions_are_read(self) -> None:
        broker = InMemoryBroker(4)
        _fill(broker, [b"a", b"b", b"c", b"d"], per_key=2)
        owned = partition_for(b"a", 4)
        consumer = InMemoryConsumer(broker, partitions=[owned])
        consumer.subscribe([TOPIC])
        assert {record.partition for record in consumer.poll(1000)} == {owned}

    def test_a_failing_poll_raises(self) -> None:
        consumer = InMemoryConsumer(InMemoryBroker(1))
        consumer.subscribe([TOPIC])
        consumer.fail_poll_with = StreamingError("down")
        with pytest.raises(StreamingError):
            consumer.poll(1000)


class TestOffsets:
    def test_position_and_end_offsets_give_the_lag(self) -> None:
        broker = InMemoryBroker(1)
        _fill(broker, [b"a"], per_key=5)
        consumer = InMemoryConsumer(broker)
        consumer.subscribe([TOPIC])
        consumer.poll(1000, max_records=2)
        ref = PartitionRef(TOPIC, 0)
        assert consumer.assignment() == frozenset({ref})
        assert consumer.position(ref) == 2
        assert consumer.end_offsets([ref]) == {ref: 5}

    def test_end_offsets_can_fail(self) -> None:
        consumer = InMemoryConsumer(InMemoryBroker(1))
        consumer.fail_end_offsets_with = StreamingError("down")
        with pytest.raises(StreamingError):
            consumer.end_offsets([PartitionRef(TOPIC, 0)])

    def test_commit_is_recorded_and_can_fail_a_number_of_times(self) -> None:
        consumer = InMemoryConsumer(InMemoryBroker(1))
        ref = PartitionRef(TOPIC, 0)
        consumer.commit_failures = 2
        for _ in range(2):
            with pytest.raises(CommitError):
                consumer.commit({ref: 5})
        assert consumer.commits == []
        consumer.commit({ref: 5})
        assert consumer.commits == [{ref: 5}]
        assert consumer.committed == {ref: 5}

    def test_a_group_resumes_from_its_committed_offsets(self) -> None:
        broker = InMemoryBroker(1)
        _fill(broker, [b"a"], per_key=5)
        consumer = InMemoryConsumer(broker, committed={PartitionRef(TOPIC, 0): 3})
        consumer.subscribe([TOPIC])
        assert [r.offset for r in consumer.poll(1000)] == [3, 4]

    def test_close(self) -> None:
        consumer = InMemoryConsumer(InMemoryBroker(1))
        consumer.close()
        assert consumer.closed


class TestRebalance:
    def test_the_first_poll_assigns_everything(self) -> None:
        listener = RecordingListener()
        consumer = InMemoryConsumer(InMemoryBroker(3))
        consumer.subscribe([TOPIC], listener)
        consumer.poll(1000)
        assert listener.calls == [("assigned", [0, 1, 2])]

    def test_a_rebalance_revokes_then_assigns_at_the_next_poll(self) -> None:
        listener = RecordingListener()
        consumer = InMemoryConsumer(InMemoryBroker(3))
        consumer.subscribe([TOPIC], listener)
        consumer.poll(1000)
        consumer.rebalance([1])
        assert listener.calls == [("assigned", [0, 1, 2])]
        consumer.poll(1000)
        assert listener.calls[1:] == [("revoked", [0, 1, 2]), ("assigned", [1])]
        assert consumer.assignment() == frozenset({PartitionRef(TOPIC, 1)})

    def test_a_lost_rebalance_reports_the_loss(self) -> None:
        listener = RecordingListener()
        consumer = InMemoryConsumer(InMemoryBroker(2))
        consumer.subscribe([TOPIC], listener)
        consumer.poll(1000)
        consumer.rebalance([0], lost=True)
        consumer.poll(1000)
        assert listener.calls[1:] == [("lost", [0, 1]), ("assigned", [0])]

    def test_a_reassigned_partition_resumes_from_the_commit_not_the_read_position(self) -> None:
        broker = InMemoryBroker(1)
        _fill(broker, [b"a"], per_key=4)
        consumer = InMemoryConsumer(broker)
        consumer.subscribe([TOPIC])
        consumer.poll(1000, max_records=3)
        consumer.commit({PartitionRef(TOPIC, 0): 1})
        consumer.rebalance([0])
        assert [r.offset for r in consumer.poll(1000)] == [1, 2, 3]

    def test_works_without_a_listener(self) -> None:
        consumer = InMemoryConsumer(InMemoryBroker(1))
        consumer.subscribe([TOPIC])
        consumer.poll(1000)
        consumer.rebalance([0])
        assert consumer.poll(1000) == []


class TestRecording:
    def test_commits_sends_and_flushes_share_one_sequence(self) -> None:
        log = CallLog()
        broker = InMemoryBroker(1)
        producer = RecordingProducer(broker, log)
        consumer = InMemoryConsumer(broker, log=log)
        producer.send("out", b"k", b"v")
        producer.flush(1.0)
        consumer.commit({PartitionRef(TOPIC, 0): 1})
        producer.close()
        assert log.names() == ["send", "flush", "commit"]
        assert log.events[0] == ("send", "out")
        assert broker.flushes == 1
        assert broker.closed
