"""In-memory consumer and call recorder, for the validation consumer tests.

Completa a `InMemoryBroker` (solo el lado productor) con el lado consumidor del puerto
`MessageConsumer`: `poll` por particion en orden de offset, `commit`, `position`,
`end_offsets` y un rebalanceo que invoca el `RebalanceListener` como lo hace la
libreria (revocar lo anterior y asignar lo nuevo dentro de `poll`). Los offsets
confirmados sobreviven al rebalanceo, de modo que una particion reasignada se retoma
donde se confirmo y no donde se leyo.

`CallLog` y `RecordingProducer` registran el orden de las llamadas `send`, `flush` y
`commit` en una unica secuencia, para comprobar que el commit va siempre despues de
publicar y de verificar la entrega.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Any

from data.streaming.errors import CommitError, StreamingError
from data.streaming.transport import (
    ConsumedRecord,
    Header,
    MessageProducer,
    PartitionRef,
    RebalanceListener,
)
from tests.support.in_memory_broker import InMemoryBroker


class CallLog:
    """An ordered record of calls made on the fakes of one test."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def add(self, *event: Any) -> None:
        """Append one event."""
        self.events.append(event)

    def names(self) -> list[str]:
        """Return the kind of every event, in order."""
        return [str(event[0]) for event in self.events]


class RecordingProducer:
    """A MessageProducer that logs its calls and delegates to another one."""

    def __init__(self, inner: MessageProducer, log: CallLog) -> None:
        self._inner = inner
        self._log = log

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[Header] = (),
    ) -> None:
        self._log.add("send", topic)
        self._inner.send(topic, key, value, headers)

    def flush(self, timeout: float | None = None) -> None:
        self._log.add("flush")
        self._inner.flush(timeout)

    def close(self, timeout: float | None = None) -> None:
        self._inner.close(timeout)


class InMemoryConsumer:
    """A MessageConsumer over the partitions of an InMemoryBroker topic."""

    def __init__(
        self,
        broker: InMemoryBroker,
        *,
        partitions: Iterable[int] | None = None,
        log: CallLog | None = None,
        committed: Mapping[PartitionRef, int] | None = None,
    ) -> None:
        """Create a consumer that owns the given partitions once it polls.

        Args:
            broker: Broker holding the records.
            partitions: Partition numbers to own; every partition of the broker
                when omitted.
            log: Where commits are recorded, next to the producer calls.
            committed: Offsets already committed by the group.
        """
        self._broker = broker
        self._target = (
            set(range(broker.partitions)) if partitions is None else set(partitions)
        )
        self._log = log
        self.committed: dict[PartitionRef, int] = dict(committed or {})
        self.commits: list[dict[PartitionRef, int]] = []
        self.commit_failures = 0
        self.fail_poll_with: StreamingError | None = None
        self.fail_end_offsets_with: StreamingError | None = None
        self.closed = False
        self.subscribed: list[str] = []
        self._listener: RebalanceListener | None = None
        self._assigned: set[PartitionRef] = set()
        self._positions: dict[PartitionRef, int] = {}
        self._needs_join = False
        self._lost = False

    @property
    def topic(self) -> str:
        """Return the subscribed topic."""
        return self.subscribed[0]

    def subscribe(
        self, topics: Sequence[str], listener: RebalanceListener | None = None
    ) -> None:
        self.subscribed = list(topics)
        self._listener = listener
        self._needs_join = True

    def rebalance(self, partitions: Iterable[int], *, lost: bool = False) -> None:
        """Change the owned partitions; the listener is called by the next poll.

        Args:
            partitions: The partition numbers owned afterwards.
            lost: Report the old ones through on_partitions_lost, as the library does
                when the member was evicted from the group.
        """
        self._target = set(partitions)
        self._needs_join = True
        self._lost = lost

    def _join(self) -> None:
        """Run the revoke/assign cycle of a rebalance."""
        previous = set(self._assigned)
        if self._listener is not None and previous:
            if self._lost:
                self._listener.on_partitions_lost(previous)
            else:
                self._listener.on_partitions_revoked(previous)
        self._assigned = {PartitionRef(self.topic, number) for number in self._target}
        self._positions = {ref: self.committed.get(ref, 0) for ref in self._assigned}
        self._needs_join = False
        self._lost = False
        if self._listener is not None:
            self._listener.on_partitions_assigned(set(self._assigned))

    def poll(self, timeout_ms: int, max_records: int | None = None) -> list[ConsumedRecord]:
        del timeout_ms
        if self.fail_poll_with is not None:
            raise self.fail_poll_with
        if self._needs_join:
            self._join()
        limit = max_records if max_records is not None else 2**31
        out: list[ConsumedRecord] = []
        for ref in sorted(self._assigned):
            available = self._broker.records(ref.topic, ref.partition)[self._positions[ref] :]
            take = available[: limit - len(out)]
            out.extend(take)
            self._positions[ref] += len(take)
            if len(out) >= limit:
                break
        return out

    def commit(self, offsets: Mapping[PartitionRef, int]) -> None:
        if self._log is not None:
            self._log.add("commit", dict(offsets))
        if self.commit_failures > 0:
            self.commit_failures -= 1
            raise CommitError("Simulated: the commit failed.")
        self.commits.append(dict(offsets))
        self.committed.update(offsets)

    def assignment(self) -> frozenset[PartitionRef]:
        return frozenset(self._assigned)

    def position(self, partition: PartitionRef) -> int:
        return self._positions[partition]

    def end_offsets(self, partitions: Collection[PartitionRef]) -> dict[PartitionRef, int]:
        if self.fail_end_offsets_with is not None:
            raise self.fail_end_offsets_with
        return {
            ref: len(self._broker.records(ref.topic, ref.partition)) for ref in partitions
        }

    def close(self) -> None:
        self.closed = True
