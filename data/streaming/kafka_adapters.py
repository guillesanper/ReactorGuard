"""kafka-python adapters for the transport ports.

Adaptadores finos: traducen configuracion, tipos y errores, y no contienen logica
de negocio. Los constructores de KafkaProducer/KafkaConsumer se reciben como
parametro (por defecto, los reales) para que los tests sustituyan el constructor
y comprueben los kwargs sin broker ni parches sobre internos de la libreria.

Garantias del productor (D4): acks="all" e idempotencia EXPLICITA. En kafka-python
3.x la idempotencia ya viene activa por defecto, pero pedirla explicitamente hace
que un conflicto de configuracion lance en vez de desactivarla en silencio. La
idempotencia protege el orden por particion con <=5 peticiones en vuelo; NO
convierte el ciclo consumidor-productor-commit en exactly-once.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

from kafka import ConsumerRebalanceListener, KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from kafka.structs import OffsetAndMetadata, TopicPartition

from data.streaming.errors import CommitError, PublishError, StreamingError
from data.streaming.kafka_settings import KafkaConnectionSettings
from data.streaming.streaming_params import StreamingParams
from data.streaming.transport import (
    ConsumedRecord,
    Header,
    PartitionRef,
    RebalanceListener,
)

_LOG = logging.getLogger(__name__)

# Epoca de lider desconocida: el valor que Kafka define para "no informada".
_UNKNOWN_LEADER_EPOCH = -1
# Cada _PRUNE_THRESHOLD envios sin flush se descartan los futures ya resueltos con
# exito. Sin esto un productor que nunca hace flush acumularia un future por
# mensaje. Los fallidos y los pendientes se conservan siempre.
_PRUNE_THRESHOLD = 4096


def _to_partition_ref(partition: TopicPartition) -> PartitionRef:
    """Convert a kafka-python TopicPartition to the library-independent type.

    Args:
        partition: The kafka-python partition.

    Returns:
        The equivalent PartitionRef.
    """
    return PartitionRef(partition.topic, partition.partition)


def _to_topic_partition(partition: PartitionRef) -> TopicPartition:
    """Convert a PartitionRef to the kafka-python TopicPartition.

    Args:
        partition: The library-independent partition.

    Returns:
        The equivalent TopicPartition.
    """
    return TopicPartition(partition.topic, partition.partition)


class KafkaMessageProducer:
    """MessageProducer backed by kafka-python's KafkaProducer."""

    def __init__(
        self,
        settings: KafkaConnectionSettings,
        params: StreamingParams,
        *,
        client_id: str | None = None,
        producer_factory: Callable[..., Any] = KafkaProducer,
    ) -> None:
        """Create the producer with the delivery guarantees of D4.

        Args:
            settings: Connection and security settings.
            params: Streaming parameters (compression, batching, timeouts).
            client_id: Optional client id shown in broker logs.
            producer_factory: Constructor of the underlying producer. Tests
                replace it to inspect the keyword arguments.

        Raises:
            PublishError: If the underlying producer cannot be created.
        """
        config: dict[str, Any] = {
            **settings.to_client_config(),
            "acks": "all",
            "enable_idempotence": True,
            "compression_type": params.compression_type,
            "linger_ms": params.batch_linger_ms,
            "batch_size": params.producer_batch_bytes,
            "max_in_flight_requests_per_connection": params.max_in_flight_requests,
            "request_timeout_ms": params.request_timeout_ms,
            "delivery_timeout_ms": params.delivery_timeout_ms,
        }
        if client_id is not None:
            config["client_id"] = client_id
        try:
            self._producer = producer_factory(**config)
        except KafkaError as exc:
            raise PublishError(f"Could not create the Kafka producer: {exc}") from exc
        self._pending: list[Any] = []
        self._prune_at = _PRUNE_THRESHOLD

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[Header] = (),
    ) -> None:
        """Queue one message and remember its delivery future.

        Args:
            topic: Destination topic.
            key: Partitioning key.
            value: Serialized payload.
            headers: Message headers as (name, bytes) pairs.

        Raises:
            PublishError: If the message cannot be queued (buffer full, metadata
                timeout, broker error).
        """
        try:
            future = self._producer.send(topic, value=value, key=key, headers=list(headers))
        except KafkaError as exc:
            raise PublishError(f"Could not queue a message for '{topic}': {exc}") from exc
        self._pending.append(future)
        if len(self._pending) >= self._prune_at:
            self._prune_resolved()

    def _prune_resolved(self) -> None:
        """Drop futures already resolved with success, keep failed and pending."""
        self._pending = [
            future
            for future in self._pending
            if not (future.is_done and future.succeeded())
        ]
        self._prune_at = max(_PRUNE_THRESHOLD, 2 * len(self._pending))

    def flush(self, timeout: float | None = None) -> None:
        """Wait for delivery and verify every message sent since the last flush.

        KafkaProducer.flush() no lanza si un registro falla (solo por timeout),
        asi que cada future se comprueba despues. Tras flush() los futures estan
        resueltos: la libreria activa el latch despues de ejecutar sus callbacks.

        Despues de un PublishError el llamador no debe hacer commit y debe
        reprocesar el lote; los futures fallidos se descartan al informarlos.

        Args:
            timeout: Maximum seconds to wait; None waits indefinitely.

        Raises:
            PublishError: If the flush times out or any message failed.
        """
        pending, self._pending = self._pending, []
        self._prune_at = _PRUNE_THRESHOLD
        try:
            self._producer.flush(timeout=timeout)
        except KafkaError as exc:
            raise PublishError(f"Flush did not complete: {exc}") from exc

        failures: list[KafkaError] = []
        for future in pending:
            try:
                future.get(timeout=timeout)
            except KafkaError as exc:
                failures.append(exc)
        if failures:
            _LOG.error("%d of %d messages failed delivery", len(failures), len(pending))
            raise PublishError(
                f"{len(failures)} of {len(pending)} messages failed delivery; "
                f"first error: {failures[0]!r}"
            )

    def close(self, timeout: float | None = None) -> None:
        """Close the underlying producer.

        Args:
            timeout: Maximum seconds to wait for pending messages.
        """
        self._producer.close(timeout=timeout)


class _ListenerBridge(ConsumerRebalanceListener):  # type: ignore[misc]
    """Translate kafka-python rebalance callbacks to a RebalanceListener."""

    def __init__(self, listener: RebalanceListener) -> None:
        """Wrap a listener.

        Args:
            listener: The library-independent listener to notify.
        """
        self._listener = listener

    def on_partitions_revoked(self, revoked: Collection[TopicPartition]) -> None:
        """Forward a revoke.

        Args:
            revoked: Partitions being taken away.
        """
        self._listener.on_partitions_revoked([_to_partition_ref(tp) for tp in revoked])

    def on_partitions_assigned(self, assigned: Collection[TopicPartition]) -> None:
        """Forward an assignment.

        Args:
            assigned: Partitions newly owned.
        """
        self._listener.on_partitions_assigned([_to_partition_ref(tp) for tp in assigned])

    def on_partitions_lost(self, lost: Collection[TopicPartition]) -> None:
        """Forward a loss.

        Args:
            lost: Partitions lost without a clean revoke.
        """
        self._listener.on_partitions_lost([_to_partition_ref(tp) for tp in lost])


class KafkaMessageConsumer:
    """MessageConsumer backed by kafka-python's KafkaConsumer."""

    def __init__(
        self,
        settings: KafkaConnectionSettings,
        params: StreamingParams,
        *,
        group_id: str | None = None,
        client_id: str | None = None,
        consumer_factory: Callable[..., Any] = KafkaConsumer,
    ) -> None:
        """Create the consumer with manual commits (D4).

        Args:
            settings: Connection and security settings.
            params: Streaming parameters (poll size, offsets, timeouts).
            group_id: Consumer group; defaults to params.consumer_group.
            client_id: Optional client id shown in broker logs.
            consumer_factory: Constructor of the underlying consumer. Tests
                replace it to inspect the keyword arguments.

        Raises:
            StreamingError: If the underlying consumer cannot be created.
        """
        config: dict[str, Any] = {
            **settings.to_client_config(),
            "group_id": params.consumer_group if group_id is None else group_id,
            "enable_auto_commit": False,
            "auto_offset_reset": params.auto_offset_reset,
            "isolation_level": params.isolation_level,
            "max_poll_records": params.max_poll_records,
            "max_poll_interval_ms": params.max_poll_interval_ms,
        }
        if client_id is not None:
            config["client_id"] = client_id
        try:
            self._consumer = consumer_factory(**config)
        except KafkaError as exc:
            raise StreamingError(f"Could not create the Kafka consumer: {exc}") from exc
        self._commit_timeout_ms = params.commit_timeout_ms
        self._close_timeout_ms = params.close_timeout_ms

    def subscribe(
        self, topics: Sequence[str], listener: RebalanceListener | None = None
    ) -> None:
        """Join the consumer group for the given topics.

        Args:
            topics: Topics to subscribe to.
            listener: Optional rebalance callbacks.
        """
        bridge = None if listener is None else _ListenerBridge(listener)
        self._consumer.subscribe(topics=list(topics), listener=bridge)

    def poll(self, timeout_ms: int, max_records: int | None = None) -> list[ConsumedRecord]:
        """Fetch the next records, flattened and ordered per partition.

        Args:
            timeout_ms: Maximum wait when no record is available.
            max_records: Maximum records to return; None uses max_poll_records.

        Returns:
            The records, partition by partition, each in offset order.

        Raises:
            StreamingError: If the poll fails.
        """
        try:
            batches = self._consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
        except KafkaError as exc:
            raise StreamingError(f"Poll failed: {exc}") from exc
        records: list[ConsumedRecord] = []
        for partition_records in batches.values():
            for record in partition_records:
                records.append(
                    ConsumedRecord(
                        topic=record.topic,
                        partition=record.partition,
                        offset=record.offset,
                        key=record.key,
                        value=record.value,
                        headers=tuple((str(name), bytes(data)) for name, data in record.headers),
                        timestamp_ms=record.timestamp,
                    )
                )
        return records

    def commit(self, offsets: Mapping[PartitionRef, int]) -> None:
        """Synchronously commit the next offset to read of each partition.

        Args:
            offsets: For each partition, the next offset to read.

        Raises:
            CommitError: If the commit fails or times out. No retry happens here:
                the caller owns the retry policy (params.commit_retries).
        """
        payload = {
            _to_topic_partition(partition): OffsetAndMetadata(
                offset, "", _UNKNOWN_LEADER_EPOCH
            )
            for partition, offset in offsets.items()
        }
        try:
            self._consumer.commit(offsets=payload, timeout_ms=self._commit_timeout_ms)
        except KafkaError as exc:
            raise CommitError(f"Offset commit failed: {exc}") from exc

    def assignment(self) -> frozenset[PartitionRef]:
        """Return the partitions currently assigned to this consumer.

        Returns:
            The assigned partitions.
        """
        return frozenset(_to_partition_ref(tp) for tp in self._consumer.assignment())

    def position(self, partition: PartitionRef) -> int:
        """Return the next offset that will be fetched from a partition.

        Args:
            partition: An assigned partition.

        Returns:
            The offset of the next record to be returned.

        Raises:
            StreamingError: If the position cannot be determined.
        """
        try:
            return int(self._consumer.position(_to_topic_partition(partition)))
        except KafkaError as exc:
            raise StreamingError(f"Could not read the position of {partition}: {exc}") from exc

    def end_offsets(self, partitions: Collection[PartitionRef]) -> dict[PartitionRef, int]:
        """Return the latest offset of each partition (last offset plus one).

        Args:
            partitions: Partitions to query.

        Returns:
            Mapping from partition to its end offset.

        Raises:
            StreamingError: If the offsets cannot be fetched.
        """
        try:
            raw = self._consumer.end_offsets(
                [_to_topic_partition(partition) for partition in partitions]
            )
        except KafkaError as exc:
            raise StreamingError(f"Could not fetch end offsets: {exc}") from exc
        return {_to_partition_ref(tp): int(offset) for tp, offset in raw.items()}

    def close(self) -> None:
        """Leave the group without committing (D4: commits are explicit)."""
        self._consumer.close(autocommit=False, timeout_ms=self._close_timeout_ms)
