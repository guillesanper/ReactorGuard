"""In-memory stand-in for a Kafka broker, for the streaming tests.

Implementa el lado productor del puerto `MessageProducer` y reparte los mensajes en
particiones con el MISMO algoritmo que el productor real de kafka-python: murmur2 de
la clave serializada, mascara de 31 bits y modulo del numero de particiones (el
cuerpo de `DefaultPartitioner.partition`). Asi un test puede comprobar propiedades
de reparto (todas las lecturas de un sensor en una unica particion y en orden, D1)
sin broker ni red.

Los mensajes sin clave se reparten en round-robin, determinista a diferencia del
aleatorio de kafka-python, para que un test espejo reproduzca siempre lo mismo.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kafka.partitioner.default import murmur2

from data.streaming.errors import PublishError
from data.streaming.transport import ConsumedRecord, Header

_POSITIVE_MASK = 0x7FFFFFFF


def partition_for(key: bytes, num_partitions: int) -> int:
    """Return the partition kafka-python's default partitioner gives a key.

    Args:
        key: Serialized message key.
        num_partitions: Number of partitions of the topic.

    Returns:
        The partition number.
    """
    return (int(murmur2(key)) & _POSITIVE_MASK) % num_partitions


@dataclass(frozen=True)
class SentMessage:
    """One message accepted by the broker.

    Attributes:
        topic: Destination topic.
        key: Message key, or None.
        value: Serialized payload.
        headers: Message headers.
    """

    topic: str
    key: bytes | None
    value: bytes
    headers: tuple[Header, ...]


class InMemoryBroker:
    """A broker with a fixed number of partitions per topic, holding every message."""

    def __init__(self, partitions: int = 12) -> None:
        """Create an empty broker.

        Args:
            partitions: Partitions of every topic (12 is the raw topic of the
                cluster, k8s/base/kafka/kafka-topics.yaml).

        Raises:
            ValueError: If partitions is not positive.
        """
        if partitions <= 0:
            raise ValueError(f"partitions must be > 0, got {partitions}.")
        self.partitions = partitions
        self._records: dict[tuple[str, int], list[ConsumedRecord]] = {}
        self._round_robin = 0
        self.sent: list[SentMessage] = []
        self.flushes = 0
        self.closed = False
        self.fail_flush_with: PublishError | None = None
        self.fail_send_after: int | None = None

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[Header] = (),
    ) -> None:
        """Append a message to the partition its key maps to.

        Args:
            topic: Destination topic.
            key: Partitioning key; None means round-robin.
            value: Serialized payload.
            headers: Message headers.

        Raises:
            PublishError: If fail_send_after is set and that many messages were
                already accepted.
        """
        if self.fail_send_after is not None and len(self.sent) >= self.fail_send_after:
            raise PublishError("Simulated: the message could not be queued.")
        if key is None:
            partition = self._round_robin % self.partitions
            self._round_robin += 1
        else:
            partition = partition_for(key, self.partitions)
        log = self._records.setdefault((topic, partition), [])
        log.append(
            ConsumedRecord(
                topic=topic,
                partition=partition,
                offset=len(log),
                key=key,
                value=value,
                headers=tuple(headers),
                timestamp_ms=len(self.sent),
            )
        )
        self.sent.append(SentMessage(topic, key, value, tuple(headers)))

    def flush(self, timeout: float | None = None) -> None:
        """Count a flush and fail it when asked to.

        Args:
            timeout: Ignored; every message is already delivered.

        Raises:
            PublishError: If fail_flush_with is set.
        """
        del timeout
        self.flushes += 1
        if self.fail_flush_with is not None:
            raise self.fail_flush_with

    def close(self, timeout: float | None = None) -> None:
        """Mark the producer as closed.

        Args:
            timeout: Ignored.
        """
        del timeout
        self.closed = True

    def records(self, topic: str, partition: int) -> list[ConsumedRecord]:
        """Return the messages of one partition in offset order.

        Args:
            topic: Topic name.
            partition: Partition number.

        Returns:
            The records, a copy.
        """
        return list(self._records.get((topic, partition), []))

    def partitions_of(self, topic: str) -> dict[int, list[ConsumedRecord]]:
        """Return the non-empty partitions of a topic.

        Args:
            topic: Topic name.

        Returns:
            Mapping from partition number to its records.
        """
        return {
            partition: list(records)
            for (name, partition), records in sorted(self._records.items())
            if name == topic
        }
