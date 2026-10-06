"""Transport ports: what the streamer and the consumer need from a message bus.

Los servicios dependen de estos Protocol, no de kafka-python (D11). Los
adaptadores de kafka_adapters.py los implementan sobre la libreria y los tests
usan fakes en memoria, de modo que ninguna prueba unitaria necesita un broker.

Convenio de offsets: `commit` recibe, por particion, el SIGUIENTE offset a leer
(ultimo procesado mas uno), igual que Kafka y que `end_offsets`. Con ese convenio
el lag de una particion es `end_offsets[p] - position(p)`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

# Cabecera con la identidad de la pasada de datos (fault_type y numero de bucle).
# El consumidor reinicia el estado de un sensor cuando cambia (D2).
HEADER_RUN_ID = "run-id"

Header = tuple[str, bytes]


@dataclass(frozen=True, order=True)
class PartitionRef:
    """A topic partition, independent of the client library.

    Attributes:
        topic: Topic name.
        partition: Partition number.
    """

    topic: str
    partition: int


@dataclass(frozen=True)
class ConsumedRecord:
    """One record delivered to a consumer.

    Attributes:
        topic: Topic the record came from.
        partition: Partition the record came from.
        offset: Position of the record in its partition.
        key: Raw key, or None when the record has none.
        value: Raw value, or None for a tombstone.
        headers: Record headers in order.
        timestamp_ms: Record timestamp in milliseconds since the epoch.
    """

    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes | None
    headers: tuple[Header, ...]
    timestamp_ms: int

    @property
    def partition_ref(self) -> PartitionRef:
        """Return the partition this record belongs to."""
        return PartitionRef(self.topic, self.partition)

    def header(self, name: str) -> bytes | None:
        """Return the value of the last header with the given name.

        Args:
            name: Header name.

        Returns:
            The header value, or None when absent.
        """
        for key, value in reversed(self.headers):
            if key == name:
                return value
        return None


class RebalanceListener(Protocol):
    """Callbacks invoked by the consumer when its assignment changes.

    En kafka-python los callbacks sincronos corren en el hilo de IO del latido:
    deben volver rapido (muy por debajo de session_timeout_ms). El
    comportamiento concreto (flush, commit y reinicio de estado) es de T3.5.
    """

    def on_partitions_revoked(self, revoked: Collection[PartitionRef]) -> None:
        """Handle partitions about to be taken away.

        Args:
            revoked: Partitions the consumer is losing.
        """

    def on_partitions_assigned(self, assigned: Collection[PartitionRef]) -> None:
        """Handle partitions newly given to the consumer.

        Args:
            assigned: Partitions the consumer now owns.
        """

    def on_partitions_lost(self, lost: Collection[PartitionRef]) -> None:
        """Handle partitions lost without a clean revoke.

        Args:
            lost: Partitions the consumer no longer owns.
        """


class MessageProducer(Protocol):
    """Publishes messages and reports delivery failures."""

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[Header] = (),
    ) -> None:
        """Queue one message for delivery.

        Args:
            topic: Destination topic.
            key: Partitioning key; messages with the same key keep their order.
            value: Serialized payload.
            headers: Message headers.

        Raises:
            PublishError: If the message cannot even be queued.
        """

    def flush(self, timeout: float | None = None) -> None:
        """Block until every queued message is delivered, then verify them.

        Args:
            timeout: Maximum seconds to wait; None waits indefinitely.

        Raises:
            PublishError: If the wait times out or any message failed. A flush
                that returns normally guarantees every message sent before it
                was acknowledged.
        """

    def close(self, timeout: float | None = None) -> None:
        """Release the producer.

        Args:
            timeout: Maximum seconds to wait for pending messages.
        """


class MessageConsumer(Protocol):
    """Consumes messages in partition order and commits offsets explicitly."""

    def subscribe(
        self, topics: Sequence[str], listener: RebalanceListener | None = None
    ) -> None:
        """Join the consumer group for the given topics.

        Args:
            topics: Topics to subscribe to.
            listener: Optional rebalance callbacks.
        """

    def poll(self, timeout_ms: int, max_records: int | None = None) -> list[ConsumedRecord]:
        """Fetch the next records.

        Args:
            timeout_ms: Maximum wait when no record is available.
            max_records: Maximum records to return; None uses the client default.

        Returns:
            Records grouped by partition and ordered by offset within each one.
        """

    def commit(self, offsets: Mapping[PartitionRef, int]) -> None:
        """Synchronously commit offsets.

        Args:
            offsets: For each partition, the next offset to read.

        Raises:
            CommitError: If the commit fails or times out.
        """

    def assignment(self) -> frozenset[PartitionRef]:
        """Return the partitions currently assigned to this consumer."""

    def position(self, partition: PartitionRef) -> int:
        """Return the next offset that will be fetched from a partition.

        Args:
            partition: An assigned partition.

        Returns:
            The offset of the next record to be returned.
        """

    def end_offsets(self, partitions: Collection[PartitionRef]) -> dict[PartitionRef, int]:
        """Return the latest offset of each partition (last offset plus one).

        Args:
            partitions: Partitions to query.

        Returns:
            Mapping from partition to its end offset.
        """

    def close(self) -> None:
        """Leave the group and release the consumer without committing."""
