"""TEP streamer: publishes the adapted TEP readings to Kafka.

Lee los runs del TEP desde un `ReadingSource` y publica cada lectura como un
mensaje del topic de lecturas crudas. Reutiliza `readings_from_frame` (no duplica
la lectura del parquet) y `SensorReading.to_kafka_bytes()`.

Invariantes de correccion que este modulo garantiza:

- D1: la clave de cada mensaje es el `sensor_id` (UTF-8). Kafka solo ordena dentro
  de una particion y los detectores acumulan estado por sensor, asi que todas las
  lecturas de un sensor deben caer en la misma particion y en orden.
- D2: cada mensaje lleva la cabecera `run-id` con el run (`fault_type=NN`) y el
  numero de bucle. Los 22 ficheros del TEP reinician el reloj; el consumidor
  reinicia el estado de un sensor cuando la cabecera cambia.
- Orden de emision: un timestep completo (sus 52 lecturas) y luego el siguiente,
  dentro de un run, y los runs UNO TRAS OTRO. Nunca se entrelazan runs: romperia la
  monotonia del timestamp por sensor.

Modos: FAST emite sin pausas (benchmark); REALTIME emite un lote por timestep y
espera `sample_interval / speed_multiplier` entre lotes con PLAZOS ABSOLUTOS
(`deadline += interval`): la espera de cada lote se calcula contra el instante en que
deberia salir, no contra el instante en que acabo el anterior, de modo que el tiempo
de proceso, los sobresueos y las pausas del GC no se acumulan como deriva.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd

from data.generators.reading_source import ReadingSource
from data.generators.tep_adapter import readings_from_frame
from data.generators.tep_streamer_params import StreamerParams, StreamMode
from data.streaming.errors import PublishError
from data.streaming.metrics import StreamMetrics
from data.streaming.transport import HEADER_RUN_ID, Header, MessageProducer

_LOG = logging.getLogger(__name__)

# Por debajo de un microsegundo un plazo se da por cumplido: las sumas repetidas de
# `deadline += interval` con un intervalo no representable (180 / 7) dejan restos de
# 1e-14 que provocarian una espera espuria, y ningun temporizador del sistema
# resuelve menos que esto.
_TIMER_RESOLUTION_S = 1e-6


def format_run_id(run_id: str, loop_index: int) -> str:
    """Return the value of the run-id header.

    Args:
        run_id: Run identifier of the source, e.g. "fault_type=21".
        loop_index: Zero-based number of the pass over the runs.

    Returns:
        The header value, e.g. "fault_type=21/loop=0". El consumidor solo necesita
        comparar igualdad: cambia en cada run y en cada vuelta (D2).
    """
    return f"{run_id}/loop={loop_index}"


def timestep_slices(frame: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Split a long-format frame into one sub-frame per timestep.

    Args:
        frame: Frame with a `timestep` column sorted in non-decreasing order, as
            adapt_tep writes it (rows of a timestep are contiguous, in sensor order).

    Yields:
        The rows of each timestep, in row order.

    Raises:
        KeyError: If the frame has no `timestep` column.
        ValueError: If timesteps go backwards. No se reordena en silencio: una
            lectura reordenada cambia lo que ven los detectores.
    """
    if "timestep" not in frame.columns:
        raise KeyError("Frame is missing the 'timestep' column needed to batch the readings.")
    steps = frame["timestep"].to_numpy()
    if steps.size == 0:
        return
    changes = np.diff(steps)
    if (changes < 0).any():
        raise ValueError("The 'timestep' column must be sorted in non-decreasing order.")
    bounds = [0, *(np.flatnonzero(changes) + 1).tolist(), int(steps.size)]
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        yield frame.iloc[start:end]


class TEPStreamer:
    """Streams TEP readings to a Kafka topic, one timestep batch at a time."""

    def __init__(
        self,
        producer: MessageProducer,
        source: ReadingSource,
        params: StreamerParams,
        metrics: StreamMetrics,
        *,
        topic: str,
        flush_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Any] | None = None,
        on_tick: Callable[[], None] | None = None,
    ) -> None:
        """Wire the streamer; nothing is read or sent until run().

        Args:
            producer: Where messages are published.
            source: Where the runs are read from.
            params: Mode, speed, batching and TEP metadata.
            metrics: Prometheus metrics fed while streaming.
            topic: Destination topic (streaming.raw_topic).
            flush_timeout_s: Timeout of every producer flush; None waits forever.
            clock: Monotonic time source in seconds. Tests inject a fake.
            sleeper: Blocks for the given seconds. Defaults to waiting on the stop
                event, so stop() wakes a sleeping streamer immediately.
            on_tick: Called after every batch and every wait slice; the service
                uses it for the heartbeat that /health reads.
        """
        self._producer = producer
        self._source = source
        self._params = params
        self._metrics = metrics
        self._topic = topic
        self._flush_timeout_s = flush_timeout_s
        self._clock = clock
        self._stop = threading.Event()
        self._sleeper: Callable[[float], Any] = self._stop.wait if sleeper is None else sleeper
        self._on_tick = on_tick

        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._messages = 0
        self._bytes = 0
        self._errors = 0
        self._timesteps = 0
        self._flushes = 0
        self._late_timesteps = 0
        self._flush_latency_total = 0.0
        self._pending = 0
        self._timesteps_since_flush = 0
        self._window_started: float | None = None

    def stop(self) -> None:
        """Ask the streamer to finish; safe to call from another thread.

        El lote en curso se completa (un timestep nunca queda a medias), se hace el
        flush final y run() vuelve.
        """
        self._stop.set()

    def run(self) -> None:
        """Stream until the source is exhausted (or forever with loop) or stop().

        Raises:
            RuntimeError: If run() was already called.
            FileNotFoundError: If the source has no runs to read.
            KeyError: If a frame has no `timestep` column.
            ValueError: If a pass over the source yields no readings, a frame is not
                sorted by timestep, or a row violates the SensorReading contract
                (this includes pydantic.ValidationError).
            PublishError: If a message could not be queued or a flush failed. Nada
                de lo enviado desde el ultimo flush correcto esta garantizado.
        """
        if self._started_at is not None:
            raise RuntimeError("TEPStreamer.run() can only be called once.")
        self._started_at = self._clock()
        _LOG.info(
            "Streaming to '%s' in %s mode (speed x%s, loop=%s)",
            self._topic,
            self._params.mode.value,
            self._params.speed_multiplier,
            self._params.loop,
        )
        try:
            self._stream()
            self._flush()
        except PublishError:
            self._errors += 1
            self._metrics.produce_errors.inc()
            _LOG.error(
                "Publishing failed after %d messages; what was sent since the last flush "
                "is not guaranteed",
                self._messages,
            )
            raise
        finally:
            self._finished_at = self._clock()
        _LOG.info("Streamer finished: %s", self.get_metrics())

    def get_metrics(self) -> dict[str, float]:
        """Return the counters of this run as a flat mapping.

        Returns:
            messages, bytes (payload bytes, keys and headers excluded), errors
            (failed send or flush operations, not messages: PublishError does not
            say how many failed), timesteps, flushes, late_timesteps (batches that
            left after their deadline), mean_flush_latency_s, elapsed_s and
            messages_per_second.
        """
        if self._started_at is None:
            elapsed = 0.0
        else:
            end = self._clock() if self._finished_at is None else self._finished_at
            elapsed = max(end - self._started_at, 0.0)
        return {
            "messages": float(self._messages),
            "bytes": float(self._bytes),
            "errors": float(self._errors),
            "timesteps": float(self._timesteps),
            "flushes": float(self._flushes),
            "late_timesteps": float(self._late_timesteps),
            "mean_flush_latency_s": (
                self._flush_latency_total / self._flushes if self._flushes else 0.0
            ),
            "elapsed_s": elapsed,
            "messages_per_second": self._messages / elapsed if elapsed > 0 else 0.0,
        }

    def _stream(self) -> None:
        """Emit every batch of every run, honouring the mode and stop().

        Raises:
            ValueError: If a whole pass over the source yields no batch (con loop
                activo seria un bucle infinito sin trabajo).
            PublishError: On a failed send or flush.
        """
        realtime = self._params.mode is StreamMode.REALTIME
        interval = self._params.step_interval_s
        deadline = self._clock()
        loop_index = 0
        while True:
            batches_in_pass = 0
            for run_id, frame in self._source.runs():
                header: tuple[Header, ...] = (
                    (HEADER_RUN_ID, format_run_id(run_id, loop_index).encode("ascii")),
                )
                _LOG.info("Run %s (loop %d): %d readings", run_id, loop_index, len(frame))
                for step in timestep_slices(frame):
                    if self._stop.is_set():
                        return
                    if realtime and not self._wait_until(deadline):
                        return
                    self._emit(step, header)
                    batches_in_pass += 1
                    deadline += interval
                    cadence_due = self._timesteps_since_flush >= self._params.flush_every_timesteps
                    if realtime or cadence_due:
                        self._flush()
                    self._tick()
                self._flush()
            if batches_in_pass == 0:
                raise ValueError("The reading source yielded no timesteps; nothing to stream.")
            if not self._params.loop:
                return
            loop_index += 1

    def _emit(self, step: pd.DataFrame, header: tuple[Header, ...]) -> None:
        """Serialize and send the readings of one timestep.

        Se serializa el timestep ENTERO antes de enviar nada: una fila invalida
        falla sin haber publicado medio timestep.

        Args:
            step: Rows of one timestep.
            header: Headers shared by every message of the run.

        Raises:
            ValueError: If a row violates the SensorReading contract (this includes
                pydantic.ValidationError).
            PublishError: If a message cannot be queued.
        """
        params = self._params
        batch = [
            (reading.sensor.id.encode("utf-8"), reading.to_kafka_bytes())
            for reading in readings_from_frame(
                step,
                calibration_date=params.calibration_date,
                last_maintenance=params.last_maintenance,
                drift_coefficient=params.drift_coefficient,
            )
        ]
        if self._window_started is None:
            self._window_started = self._clock()
        sent = 0
        sent_bytes = 0
        try:
            for key, value in batch:
                self._producer.send(self._topic, key, value, header)
                sent += 1
                sent_bytes += len(value)
        finally:
            self._messages += sent
            self._bytes += sent_bytes
            self._pending += sent
            if sent:
                self._metrics.messages_produced.labels(topic=self._topic).inc(sent)
        self._timesteps += 1
        self._timesteps_since_flush += 1

    def _flush(self) -> None:
        """Flush the producer if anything was sent since the last flush.

        Raises:
            PublishError: If the flush fails; the pending count is kept.
        """
        if self._pending == 0:
            return
        started = self._window_started if self._window_started is not None else self._clock()
        self._producer.flush(self._flush_timeout_s)
        latency = max(self._clock() - started, 0.0)
        self._flushes += 1
        self._flush_latency_total += latency
        self._metrics.produce_latency.observe(latency)
        self._pending = 0
        self._timesteps_since_flush = 0
        self._window_started = None

    def _wait_until(self, deadline: float) -> bool:
        """Sleep until an absolute deadline, in slices that keep the heartbeat alive.

        Si el plazo ya paso no se duerme y NO se reinicia: el siguiente plazo sigue
        contando desde el original, asi que un retraso se recupera en lugar de
        arrastrarse.

        Args:
            deadline: Instant on the injected clock at which the batch is due.

        Returns:
            True when the deadline was reached, False when stop() interrupted.
        """
        remaining = deadline - self._clock()
        if remaining < -_TIMER_RESOLUTION_S:
            self._late_timesteps += 1
            log = _LOG.warning if self._late_timesteps == 1 else _LOG.debug
            log("Batch %.3f s late; catching up without sleeping", -remaining)
        while remaining > _TIMER_RESOLUTION_S:
            if self._stop.is_set():
                return False
            self._sleeper(min(remaining, self._params.wait_slice_s))
            self._tick()
            remaining = deadline - self._clock()
        return not self._stop.is_set()

    def _tick(self) -> None:
        """Signal liveness to the owner of the streamer, if it asked for it."""
        if self._on_tick is not None:
            self._on_tick()
