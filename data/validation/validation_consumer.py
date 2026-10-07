"""Validation consumer: sensor-readings-raw -> sensor-validated and anomaly-alerts.

Consume lecturas crudas por particion y en orden de offset, las pasa por el
`SensorValidator` y publica (a) TODA lectura, con su `quality` enriquecida, en el
topic validado (D3) y (b) un `AlertEvent` por cada fallo detectado en el topic de
alertas. Semantica at-least-once (D4): cada lote se procesa entero, se hace
`producer.flush()` (que lanza si algo fallo) y solo entonces `consumer.commit()`.
Tras un fallo puede haber duplicados; `reading_id` y `alert_id` son deterministas
y permiten deduplicar aguas abajo. Nunca exactly-once.

Estado por sensor. Los detectores acumulan estado (racha de stuck, ultimo valor,
filtro de Kalman), y los 22 ficheros del TEP reinician el reloj, asi que el estado
de un sensor se descarta cuando:

- la cabecera `run-id` cambia respecto a la ultima vista para ese sensor (D2);
- el timestamp retrocede (red de seguridad si falta la cabecera);
- el filtro de Kalman lanza `ValueError` (se trata como retroceso: reinicio y UNA
  revalidacion; si vuelve a fallar la excepcion sale y el lote no se confirma);
- la particion donde se vio el sensor se revoca o se pierde (D6).

Rebalanceo y commit. En kafka-python 3.x los callbacks del listener corren en el
hilo de IO del consumidor, el mismo de los latidos, y `consumer.commit()` desde ahi
lanza RuntimeError. Por eso el listener NO hace flush ni commit: no hace falta,
porque el unico punto de entrada al rejoin es `poll()` y cada lote se confirma por
completo antes del siguiente `poll()`, de modo que al revocar nunca queda trabajo
sin confirmar. El listener solo descarta estado, es rapido y no lanza (una excepcion
de listener sale por `poll()` como KafkaError). Como los callbacks solo se ejecutan
dentro de `poll()`, mientras el hilo del bucle espera en esa llamada, no hay acceso
concurrente al estado y no se usan cerrojos.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import Counter
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import datetime

from data.schemas.sensor_reading import SensorReading
from data.streaming.errors import CommitError, PublishError, StreamingError
from data.streaming.metrics import StreamMetrics
from data.streaming.streaming_params import StreamingParams
from data.streaming.transport import (
    HEADER_RUN_ID,
    ConsumedRecord,
    MessageConsumer,
    MessageProducer,
    PartitionRef,
)
from data.validation.alert_event import AlertEvent, AlertSource
from data.validation.sensor_fault import FaultType
from data.validation.sensor_validator import SensorValidator, ValidationResult

_LOG = logging.getLogger(__name__)

# Valores de la etiqueta `reason` de detector_state_resets_total (metrics.RESET_REASONS).
RESET_RUN_CHANGE = "run_change"
RESET_TIME_REGRESSION = "time_regression"
RESET_REBALANCE = "rebalance"

_SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class _SensorTrack:
    """What the consumer remembers about the last reading of a sensor.

    Attributes:
        run_id: Raw value of the run-id header of that reading, or None.
        timestamp: Timestamp of that reading.
    """

    run_id: bytes | None
    timestamp: datetime


def _went_back(previous: datetime, current: datetime) -> bool:
    """Return whether time moved backwards between two readings of a sensor.

    Args:
        previous: Timestamp of the earlier-consumed reading.
        current: Timestamp of the one just consumed.

    Returns:
        True when current precedes previous. Mezclar fechas con y sin zona horaria
        no se puede ordenar (comparar lanzaria TypeError) y el filtro de Kalman lo
        rechaza igualmente, asi que cuenta como retroceso: el estado se descarta.
    """
    if (previous.tzinfo is None) != (current.tzinfo is None):
        return True
    return current < previous


class ValidationConsumer:
    """Validates consumed readings and publishes the verdicts, at-least-once."""

    def __init__(
        self,
        consumer: MessageConsumer,
        producer: MessageProducer,
        validator: SensorValidator,
        params: StreamingParams,
        metrics: StreamMetrics,
        *,
        on_tick: Callable[[], None] | None = None,
        warmup_seconds: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Wire the consumer; nothing is consumed until run().

        Args:
            consumer: Source of raw readings (the transport port).
            producer: Destination of validated readings and alerts.
            validator: The detectors and their per-sensor state.
            params: Topics, poll size, flush timeout and commit retries.
            metrics: Prometheus metrics fed while consuming.
            on_tick: Called after every poll, empty ones included, and after every
                batch; the service uses it for the heartbeat that /health reads.
            warmup_seconds: Data time a sensor needs to rebuild its stuck run after
                its state is dropped (stuck window times the sampling interval).
                Only used in the log line of an assignment.
            clock: Monotonic time source in seconds for latencies. Tests inject a
                fake.
        """
        self._consumer = consumer
        self._producer = producer
        self._validator = validator
        self._params = params
        self._metrics = metrics
        self._on_tick = on_tick
        self._warmup_seconds = warmup_seconds
        self._clock = clock

        self._stop = threading.Event()
        self._joined = False
        self._track: dict[str, _SensorTrack] = {}
        self._sensors_by_partition: dict[PartitionRef, set[str]] = {}
        # Siguiente offset a leer por particion, de lo procesado y aun sin confirmar.
        self._pending: dict[PartitionRef, int] = {}
        self._window_started: float | None = None
        self._batch_resets: Counter[str] = Counter()
        self._validated_out = metrics.messages_produced.labels(topic=params.validated_topic)
        self._alerts_out = metrics.messages_produced.labels(topic=params.alerts_topic)

    # ------------------------------------------------------------------
    # Life cycle
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return whether the consumer has joined the group and is not rebalancing.

        Un pod sin particiones (mas pods que particiones) esta listo: esta unido al
        grupo y simplemente no tiene trabajo.
        """
        return self._joined

    def stop(self) -> None:
        """Ask run() to finish; safe to call from another thread.

        El lote en curso se completa y se confirma antes de volver.
        """
        self._stop.set()

    def run(self) -> None:
        """Subscribe and consume until stop() is called.

        Raises:
            StreamingError: If a poll fails.
            PublishError: If a message cannot be queued or a flush fails. El lote no
                se confirma: se reprocesa al reiniciar.
            CommitError: If the commit still fails after commit_retries retries. No
                se avanza: el lote se reprocesa al reiniciar.
            ValueError: If a reading makes the validator fail even from clean state.
        """
        self._consumer.subscribe([self._params.raw_topic], self)
        _LOG.info(
            "Consuming '%s' as group '%s' into '%s' and '%s'",
            self._params.raw_topic,
            self._params.consumer_group,
            self._params.validated_topic,
            self._params.alerts_topic,
        )
        while not self._stop.is_set():
            self.run_once()
        self.checkpoint()

    def run_once(self) -> int:
        """Poll one batch, validate it, publish it and confirm it.

        Returns:
            The number of records polled; zero when the poll came back empty.

        Raises:
            StreamingError: If the poll fails.
            PublishError: If a message cannot be queued or the flush fails.
            CommitError: If the commit fails after every retry.
            ValueError: If a reading makes the validator fail even from clean state.
        """
        records = self._consumer.poll(self._params.poll_timeout_ms, self._params.max_poll_records)
        self._tick()
        if not records:
            return 0
        try:
            self.process(records)
            self.checkpoint()
        except PublishError:
            self._metrics.produce_errors.inc()
            _LOG.error("Publishing failed; the batch is not committed and will be reprocessed")
            raise
        self._metrics.batch_size.observe(len(records))
        self._update_lag()
        self._tick()
        return len(records)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process(self, records: list[ConsumedRecord]) -> None:
        """Validate and publish a batch; nothing is flushed or committed here.

        Args:
            records: Records in partition order, each partition in offset order.

        Raises:
            PublishError: If a message cannot be queued.
            ValueError: If a reading makes the validator fail even from clean state.
        """
        if self._window_started is None:
            self._window_started = self._clock()
        self._batch_resets.clear()
        for record in records:
            self._handle(record)
        if self._batch_resets:
            _LOG.info("Detector state reset in this batch: %s", dict(self._batch_resets))

    def checkpoint(self) -> None:
        """Flush the producer and, only then, commit what was processed.

        Raises:
            PublishError: If the flush times out or any message failed; nothing is
                committed.
            CommitError: If the commit fails after every retry.
        """
        if not self._pending:
            return
        self._producer.flush(self._params.flush_timeout_s)
        if self._window_started is not None:
            self._metrics.produce_latency.observe(max(self._clock() - self._window_started, 0.0))
            self._window_started = None
        self._commit(dict(self._pending))
        self._pending.clear()

    def _commit(self, offsets: dict[PartitionRef, int]) -> None:
        """Commit offsets, retrying a failed commit commit_retries times.

        No hay espera entre intentos: cada intento ya bloquea hasta
        commit_timeout_ms, y un fallo inmediato (grupo reasignado) no mejora esperando.

        Args:
            offsets: For each partition, the next offset to read.

        Raises:
            CommitError: If every attempt fails.
        """
        attempts = self._params.commit_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                self._consumer.commit(offsets)
            except CommitError as exc:
                self._metrics.commit_failures.inc()
                if attempt == attempts:
                    _LOG.error("Offset commit failed %d times; not advancing: %s", attempts, exc)
                    raise
                _LOG.warning("Offset commit failed (attempt %d of %d): %s", attempt, attempts, exc)
            else:
                return

    def _handle(self, record: ConsumedRecord) -> None:
        """Process one record and mark its offset as processed.

        Args:
            record: The consumed record.

        Raises:
            PublishError: If a message cannot be queued.
            ValueError: If the validator fails even from clean state.
        """
        reading = self._deserialize(record)
        if reading is not None:
            self._validate_and_publish(record, reading)
        ref = record.partition_ref
        self._pending[ref] = max(self._pending.get(ref, 0), record.offset + 1)

    def _deserialize(self, record: ConsumedRecord) -> SensorReading | None:
        """Decode a record, skipping (and counting) a poison message (D5).

        Args:
            record: The consumed record.

        Returns:
            The reading, or None when the record cannot be deserialized. El offset
            se confirma igualmente: bloquear la particion para siempre es peor.
        """
        payload = record.value
        if payload is None:
            self._skip_poison(record, b"", "tombstone (no value)")
            return None
        try:
            return SensorReading.from_kafka_bytes(payload)
        except ValueError as exc:
            # pydantic.ValidationError y json.JSONDecodeError son ValueError.
            reason = f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
            self._skip_poison(record, payload, reason)
            return None

    def _skip_poison(self, record: ConsumedRecord, payload: bytes, reason: str) -> None:
        """Log and count a message that will not be processed.

        Args:
            record: The consumed record.
            payload: Its raw value (empty for a tombstone).
            reason: Why it was rejected.
        """
        _LOG.error(
            "Skipping undeserializable message %s[%d]@%d (%d bytes, sha256 %s): %s",
            record.topic,
            record.partition,
            record.offset,
            len(payload),
            hashlib.sha256(payload).hexdigest()[:16],
            reason,
        )
        self._metrics.poison_messages.inc()

    def _validate_and_publish(self, record: ConsumedRecord, reading: SensorReading) -> None:
        """Validate one reading and queue its validated copy and its alerts.

        Args:
            record: The record the reading came from.
            reading: The deserialized reading; never mutated.

        Raises:
            PublishError: If a message cannot be queued.
            ValueError: If the validator fails even from clean state.
        """
        sensor_id = reading.sensor.id
        run_id = record.header(HEADER_RUN_ID)
        self._sensors_by_partition.setdefault(record.partition_ref, set()).add(sensor_id)
        self._metrics.readings_consumed.inc()
        self._check_continuity(sensor_id, run_id, reading.timestamp)

        started = self._clock()
        result = self._validate(reading)
        self._metrics.validation_latency.observe(max(self._clock() - started, 0.0))
        self._track[sensor_id] = _SensorTrack(run_id, reading.timestamp)

        self._publish(record, reading, result, run_id)

    def _check_continuity(self, sensor_id: str, run_id: bytes | None, timestamp: datetime) -> None:
        """Drop a sensor's state when its stream is no longer continuous (D2).

        Args:
            sensor_id: Instrument tag.
            run_id: run-id header of the reading just consumed.
            timestamp: Timestamp of that reading.
        """
        previous = self._track.get(sensor_id)
        if previous is None:
            return
        if previous.run_id != run_id:
            self._reset(sensor_id, RESET_RUN_CHANGE, in_batch=True)
        elif _went_back(previous.timestamp, timestamp):
            self._reset(sensor_id, RESET_TIME_REGRESSION, in_batch=True)

    def _validate(self, reading: SensorReading) -> ValidationResult:
        """Run the validator, retrying once from clean state on a ValueError.

        Args:
            reading: The reading to validate.

        Returns:
            The validation result.

        Raises:
            ValueError: If the validator fails again from clean state. La excepcion
                sale, el lote no se confirma y el pod reinicia y lo reprocesa.
        """
        try:
            return self._validator.validate(reading)
        except ValueError as exc:
            _LOG.warning(
                "Validator rejected a reading of %s (%s); resetting its state and retrying once",
                reading.sensor.id,
                exc,
            )
            self._reset(reading.sensor.id, RESET_TIME_REGRESSION, in_batch=True)
            return self._validator.validate(reading)

    def _publish(
        self,
        record: ConsumedRecord,
        reading: SensorReading,
        result: ValidationResult,
        run_id: bytes | None,
    ) -> None:
        """Queue the validated reading and one alert per fault.

        Args:
            record: The record the reading came from.
            reading: The original reading.
            result: Its validation result.
            run_id: run-id header of the record, copied into the alert.

        Raises:
            PublishError: If a message cannot be queued.
        """
        sensor_key = reading.sensor.id.encode("utf-8")
        enriched = result.enriched_reading
        self._producer.send(
            self._params.validated_topic,
            record.key if record.key is not None else sensor_key,
            enriched.to_kafka_bytes(),
            record.headers,
        )
        self._validated_out.inc()
        self._metrics.readings_validated.labels(quality=enriched.measurement.quality.value).inc()

        if not result.faults:
            return
        source = AlertSource(topic=record.topic, partition=record.partition, offset=record.offset)
        run_text = None if run_id is None else run_id.decode("ascii", errors="replace")
        occurrences: Counter[tuple[FaultType, str]] = Counter()
        for fault in result.faults:
            slot = (fault.fault_type, fault.detector)
            event = AlertEvent.from_fault(
                fault, reading, source, run_id=run_text, occurrence=occurrences[slot]
            )
            occurrences[slot] += 1
            self._producer.send(
                self._params.alerts_topic, sensor_key, event.to_kafka_bytes(), record.headers
            )
            self._alerts_out.inc()
            self._metrics.sensor_faults.labels(fault_type=fault.fault_type.value).inc()
            self._metrics.alerts_published.labels(severity=fault.severity.value).inc()

    def _reset(self, sensor_id: str, reason: str, *, in_batch: bool = False) -> None:
        """Discard every detector's state for one sensor and count it.

        Args:
            sensor_id: Instrument tag.
            reason: Label of detector_state_resets_total.
            in_batch: Whether to add it to the per-batch summary log.
        """
        self._validator.reset_sensor(sensor_id)
        self._track.pop(sensor_id, None)
        self._metrics.detector_state_resets.labels(reason=reason).inc()
        if in_batch:
            self._batch_resets[reason] += 1
        _LOG.debug("Detector state of %s reset (%s)", sensor_id, reason)

    # ------------------------------------------------------------------
    # Lag and heartbeat
    # ------------------------------------------------------------------

    def _update_lag(self) -> None:
        """Set the lag gauge of every assigned partition (end offset minus position).

        Un fallo al consultar offsets no aborta el bucle: el lote ya esta confirmado
        y la metrica es de observacion.
        """
        try:
            assigned = self._consumer.assignment()
            if not assigned:
                return
            end_offsets = self._consumer.end_offsets(assigned)
            for partition in sorted(assigned):
                end = end_offsets.get(partition)
                if end is None:
                    continue
                lag = max(end - self._consumer.position(partition), 0)
                self._metrics.consumer_lag.labels(partition=str(partition.partition)).set(lag)
        except StreamingError as exc:
            _LOG.warning("Could not compute the consumer lag: %s", exc)

    def _tick(self) -> None:
        """Signal liveness to the owner of the consumer, if it asked for it."""
        if self._on_tick is not None:
            self._on_tick()

    # ------------------------------------------------------------------
    # RebalanceListener (called from the consumer's IO thread, inside poll())
    # ------------------------------------------------------------------

    def on_partitions_revoked(self, revoked: Collection[PartitionRef]) -> None:
        """Forget the state of every sensor seen in the revoked partitions (D6).

        Args:
            revoked: Partitions the consumer is losing. Se reinician TODOS los
                sensores vistos en ellas, no solo los del ultimo lote: si la
                particion vuelve mas tarde, el estado obsoleto produciria falsos
                resultados.
        """
        self._joined = False
        self._release(revoked, "revoked")

    def on_partitions_lost(self, lost: Collection[PartitionRef]) -> None:
        """Forget the state of the lost partitions; there is nothing to commit.

        Args:
            lost: Partitions the consumer no longer owns, without a clean revoke.
        """
        self._joined = False
        self._release(lost, "lost")

    def on_partitions_assigned(self, assigned: Collection[PartitionRef]) -> None:
        """Count the rebalance and log the warm-up the new partitions imply.

        Args:
            assigned: Partitions the consumer now owns.
        """
        self._joined = True
        self._metrics.consumer_rebalances.inc()
        warmup = (
            ""
            if self._warmup_seconds is None
            else f"; a frozen sensor needs {self._warmup_seconds / _SECONDS_PER_MINUTE:.0f} min "
            "of data to be detected again"
        )
        _LOG.info(
            "Assigned partitions %s; detector state starts empty%s",
            sorted((p.topic, p.partition) for p in assigned),
            warmup,
        )

    def _release(self, partitions: Collection[PartitionRef], cause: str) -> None:
        """Drop everything held for partitions the consumer no longer owns.

        Args:
            partitions: The partitions given up.
            cause: "revoked" or "lost", for the log.
        """
        sensors: set[str] = set()
        for partition in partitions:
            sensors |= self._sensors_by_partition.pop(partition, set())
            if self._pending.pop(partition, None) is not None:
                _LOG.warning(
                    "%s holds processed but uncommitted offsets; the new owner reprocesses them",
                    partition,
                )
            self._metrics.consumer_lag.remove(str(partition.partition))
        for sensor_id in sorted(sensors):
            self._reset(sensor_id, RESET_REBALANCE)
        if partitions:
            _LOG.info(
                "Partitions %s %s: state of %d sensors discarded",
                sorted((p.topic, p.partition) for p in partitions),
                cause,
                len(sensors),
            )
