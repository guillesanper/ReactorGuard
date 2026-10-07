"""Sustained production throughput of the Kafka pipeline (criterio 1 de la Fase 2).

Criterio: mas de 50.000 lecturas por segundo SOSTENIDAS. Este benchmark mide eso y
nada mas: cuantos mensajes por segundo confirma el broker (acks=all) a un unico
proceso productor durante N segundos, usando el MISMO adaptador que el streamer de
produccion (`KafkaMessageProducer`: idempotencia, lz4, batch y linger de params.yaml,
y verificacion del resultado de cada mensaje en cada flush).

Sustituye al benchmark anterior, que media la latencia round-trip produce-consume de
mensajes sinteticos de 1 KB contra un :9092 sin autenticacion que el cluster rechaza.
La latencia de extremo a extremo es otra pregunta (criterio de la Fase 1) y no se
mezcla con el throughput.

Metodologia:
  - Los mensajes son lecturas REALES del TEP (d00), serializadas con
    `SensorReading.to_kafka_bytes()` ANTES de medir y reutilizadas en ciclo: la
    serializacion pydantic no entra en la medida, porque este es el techo del
    transporte. El coste de serializar por mensaje se mide aparte (ver el informe).
  - Clave `sensor_id` y cabecera `run-id`, como en produccion.
  - Una fase de calentamiento (conexiones, metadatos, JIT del compresor) que se descarta.
  - Se flushea cada `flush_every_timesteps x sensores` mensajes (el cadence del
    streamer en modo FAST) y el tiempo incluye el flush final: solo cuentan los
    mensajes CONFIRMADOS.
  - Ademas de la media, se reportan las tasas por ventanas de 1 s (min, mediana, max)
    para ver si el rendimiento es estable o tiene rachas.

Una medicion `--environment local` es INFORMATIVA (un broker, sin replicacion, en la
misma maquina que el productor). Solo `--environment cluster` puede cerrar el criterio.

Uso (desde la raiz del repositorio, con el stack local levantado):

    $env:KAFKA_BOOTSTRAP = "127.0.0.1:9092"
    .\\.venv\\Scripts\\python.exe -m tests.integration.benchmark_kafka --create-topic

Se ejecuta como modulo (-m) porque comparte `benchmark_report` con el benchmark de
latencia de features. Salida: tests/results/kafka_throughput.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import os
import platform
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from kafka.admin import KafkaAdminClient, NewTopic

from data.generators.tep_adapter import readings_from_frame
from data.generators.tep_streamer import format_run_id
from data.generators.tep_streamer_params import load_streamer_params
from data.streaming.kafka_adapters import KafkaMessageProducer
from data.streaming.kafka_settings import KafkaConnectionSettings
from data.streaming.streaming_params import StreamingParams, load_streaming_params
from data.streaming.transport import HEADER_RUN_ID, MessageProducer
from tests.integration.benchmark_report import ENVIRONMENTS, build_report, write_report

_LOG = logging.getLogger("benchmark_kafka")

CRITERION = "kafka_throughput"
UNIT = "messages_per_second"
THROUGHPUT_THRESHOLD = 50_000.0
"""Criterio de la Fase 2 (TDD): lecturas por segundo sostenidas."""

DEFAULT_DURATION_S = 20.0
DEFAULT_WARMUP_S = 3.0
DEFAULT_OUTPUT = Path("tests/results/kafka_throughput.json")
DEFAULT_PARAMS = Path("params.yaml")
_PAYLOAD_PARTITION = Path("data/processed/tep/fault_type=00/readings.parquet")
_BENCH_PARTITIONS = 12
_WINDOW_S = 1.0

Payload = tuple[bytes, bytes]
"""(key, value) ya serializados."""


@dataclasses.dataclass(frozen=True)
class ThroughputResult:
    """Outcome of one timed production phase.

    Attributes:
        messages: Messages confirmed by the broker during the phase.
        payload_bytes: Bytes of those messages' values (keys and headers excluded).
        elapsed_s: Duration of the phase, final flush included.
        flushes: Number of flushes performed.
        window_rates: Messages per second in consecutive windows of about 1 s.
    """

    messages: int
    payload_bytes: int
    elapsed_s: float
    flushes: int
    window_rates: tuple[float, ...]

    @property
    def messages_per_second(self) -> float:
        """Return the mean confirmed throughput of the phase."""
        return self.messages / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def megabytes_per_second(self) -> float:
        """Return the mean confirmed payload throughput in MB/s (1 MB = 2**20 bytes)."""
        return self.payload_bytes / (1024 * 1024) / self.elapsed_s if self.elapsed_s > 0 else 0.0


def windowed_rates(marks: Sequence[tuple[float, int]], window_s: float = _WINDOW_S) -> list[float]:
    """Turn cumulative (time, messages) marks into per-window rates.

    Args:
        marks: Points (seconds since the start, messages confirmed so far), in
            increasing time. The origin (0.0, 0) is implied.
        window_s: Minimum window length. A window closes at the first mark at least
            this far from the previous window's end, so its length is a multiple of
            the flush spacing and not exactly window_s.

    Returns:
        One rate per closed window; the trailing partial window is dropped.
    """
    rates: list[float] = []
    anchor_time = 0.0
    anchor_count = 0
    for time_s, count in marks:
        if time_s - anchor_time >= window_s:
            rates.append((count - anchor_count) / (time_s - anchor_time))
            anchor_time, anchor_count = time_s, count
    return rates


def run_phase(
    producer: MessageProducer,
    topic: str,
    payloads: Sequence[Payload],
    header: tuple[tuple[str, bytes], ...],
    *,
    duration_s: float,
    flush_every: int,
    flush_timeout_s: float | None,
    clock: Callable[[], float] = time.perf_counter,
) -> ThroughputResult:
    """Produce in a loop for a fixed time and count what the broker confirmed.

    Args:
        producer: The transport under test.
        topic: Destination topic.
        payloads: Pre-serialized (key, value) pairs, sent cyclically.
        header: Headers of every message.
        duration_s: Length of the sending phase. The final flush is extra and is
            included in the elapsed time.
        flush_every: Messages between flushes.
        flush_timeout_s: Timeout of every flush.
        clock: Monotonic time source in seconds.

    Returns:
        The messages confirmed, their bytes, the elapsed time and the window rates.

    Raises:
        ValueError: If payloads is empty or a parameter is not positive.
        PublishError: If a message fails delivery (a benchmark with losses is not a
            measurement).
    """
    if not payloads:
        raise ValueError("payloads must not be empty.")
    if duration_s <= 0 or flush_every <= 0:
        raise ValueError("duration_s and flush_every must be positive.")

    sent = 0
    sent_bytes = 0
    confirmed = 0
    confirmed_bytes = 0
    flushes = 0
    marks: list[tuple[float, int]] = []
    started = clock()
    send = producer.send
    for key, value in itertools.cycle(payloads):
        send(topic, key, value, header)
        sent += 1
        sent_bytes += len(value)
        if sent % flush_every == 0:
            producer.flush(flush_timeout_s)
            flushes += 1
            confirmed, confirmed_bytes = sent, sent_bytes
            now = clock() - started
            marks.append((now, confirmed))
            if now >= duration_s:
                break
    # Si el tiempo se cumplio entre dos flushes, lo pendiente aun no cuenta.
    elapsed = clock() - started
    return ThroughputResult(
        messages=confirmed,
        payload_bytes=confirmed_bytes,
        elapsed_s=marks[-1][0] if marks else elapsed,
        flushes=flushes,
        window_rates=tuple(windowed_rates(marks)),
    )


def build_payloads(partition: Path, params_path: Path) -> tuple[list[Payload], int]:
    """Serialize the readings of one real TEP run, ahead of the measurement.

    Args:
        partition: Parquet file of a run (d00).
        params_path: params.yaml, for the streamer metadata stamped on each reading.

    Returns:
        The (key, value) pairs in emission order and the number of sensors per
        timestep.

    Raises:
        FileNotFoundError: If the parquet is not populated.
    """
    if not partition.exists():
        raise FileNotFoundError(
            f"{partition} not found. Generate it with .\\infra\\scripts\\Invoke-Pipeline.ps1."
        )
    streamer = load_streamer_params(params_path)
    frame = pd.read_parquet(partition).sort_values(["timestep", "sensor_id"], kind="stable")
    payloads = [
        (reading.sensor.id.encode("utf-8"), reading.to_kafka_bytes())
        for reading in readings_from_frame(
            frame,
            calibration_date=streamer.calibration_date,
            last_maintenance=streamer.last_maintenance,
            drift_coefficient=streamer.drift_coefficient,
        )
    ]
    return payloads, int(frame["sensor_id"].nunique())


@contextmanager
def benchmark_topic(
    settings: KafkaConnectionSettings, name: str | None, *, create: bool
) -> Iterator[str]:
    """Provide the topic to produce into, creating and removing it when asked.

    Args:
        settings: Connection settings.
        name: Topic to use; a random private name when create is set and name is None.
        create: Create the topic (12 partitions, replication 1) and delete it at the
            end. Para el stack local; en el cluster los topics ya existen y las ACLs
            no permiten crearlos.

    Yields:
        The topic name.
    """
    if not create:
        if name is None:
            raise ValueError("A topic name is required unless --create-topic is given.")
        yield name
        return
    topic = name or f"bench-throughput-{uuid.uuid4().hex[:8]}"
    admin = KafkaAdminClient(**settings.to_client_config())
    try:
        admin.create_topics([NewTopic(topic, _BENCH_PARTITIONS, 1)])
        yield topic
    finally:
        admin.delete_topics([topic])
        admin.close()


def _producer_details(streaming: StreamingParams) -> dict[str, Any]:
    """Return the producer settings that shape the throughput figure."""
    return {
        "compression_type": streaming.compression_type,
        "batch_linger_ms": streaming.batch_linger_ms,
        "producer_batch_bytes": streaming.producer_batch_bytes,
        "max_in_flight_requests": streaming.max_in_flight_requests,
        "acks": "all",
        "idempotence": True,
    }


def make_report(
    result: ThroughputResult,
    *,
    environment: str,
    streaming: StreamingParams,
    details: dict[str, Any],
) -> dict[str, Any]:
    """Build the JSON document of a throughput measurement.

    Args:
        result: The measured phase.
        environment: "local" or "cluster".
        streaming: Streaming params the producer ran with.
        details: Extra context of the run (topic, payload size, durations).

    Returns:
        The document in the shared benchmark format.
    """
    rates = list(result.window_rates)
    return build_report(
        criterion=CRITERION,
        environment=environment,
        value=round(result.messages_per_second, 2),
        unit=UNIT,
        threshold=THROUGHPUT_THRESHOLD,
        direction="at_least",
        details={
            **details,
            "messages_confirmed": result.messages,
            "elapsed_s": round(result.elapsed_s, 3),
            "flushes": result.flushes,
            "megabytes_per_second": round(result.megabytes_per_second, 3),
            "window_rates": {
                "windows": len(rates),
                "min": round(min(rates), 1) if rates else None,
                "median": round(statistics.median(rates), 1) if rates else None,
                "max": round(max(rates), 1) if rates else None,
            },
            "producer": _producer_details(streaming),
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "cpus": os.cpu_count(),
            },
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments; sys.argv[1:] when omitted.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Sustained Kafka production throughput.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="seconds")
    parser.add_argument("--warmup", type=float, default=DEFAULT_WARMUP_S, help="seconds")
    parser.add_argument("--topic", default=None, help="topic (default: streaming.raw_topic)")
    parser.add_argument(
        "--create-topic",
        action="store_true",
        help="create a private 12-partition topic and delete it afterwards (local stack)",
    )
    parser.add_argument("--environment", choices=ENVIRONMENTS, default="local")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--linger-ms", type=int, default=None, help="override batch_linger_ms")
    parser.add_argument(
        "--batch-bytes", type=int, default=None, help="override producer_batch_bytes"
    )
    parser.add_argument("--compression", default=None, help="override compression_type")
    parser.add_argument(
        "--fail-below-threshold",
        action="store_true",
        help="exit 2 when the figure does not reach the criterion (default: exit 0)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark and write its report.

    Args:
        argv: Command-line arguments; sys.argv[1:] when omitted.

    Returns:
        0 when the measurement completed, 2 when --fail-below-threshold is set and the
        figure is below the criterion.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    settings = KafkaConnectionSettings.from_env()
    streaming = load_streaming_params(args.params)
    overrides = {
        key: value
        for key, value in (
            ("batch_linger_ms", args.linger_ms),
            ("producer_batch_bytes", args.batch_bytes),
            ("compression_type", args.compression),
        )
        if value is not None
    }
    streaming = dataclasses.replace(streaming, **overrides)
    streamer = load_streamer_params(args.params)

    _LOG.info("Serializing the d00 readings ahead of the measurement")
    payloads, sensors = build_payloads(_PAYLOAD_PARTITION, args.params)
    flush_every = streamer.flush_every_timesteps * sensors
    header = ((HEADER_RUN_ID, format_run_id("fault_type=00", 0).encode("ascii")),)
    mean_bytes = sum(len(value) for _, value in payloads) / len(payloads)

    with benchmark_topic(
        settings, args.topic or (None if args.create_topic else streaming.raw_topic),
        create=args.create_topic,
    ) as topic:
        producer = KafkaMessageProducer(settings, streaming, client_id="benchmark-throughput")
        try:
            _LOG.info("Warm-up %.1f s", args.warmup)
            run_phase(
                producer, topic, payloads, header,
                duration_s=args.warmup, flush_every=flush_every,
                flush_timeout_s=streaming.flush_timeout_s,
            )
            _LOG.info("Measuring %.1f s into '%s'", args.duration, topic)
            result = run_phase(
                producer, topic, payloads, header,
                duration_s=args.duration, flush_every=flush_every,
                flush_timeout_s=streaming.flush_timeout_s,
            )
        finally:
            producer.close()

    report = make_report(
        result,
        environment=args.environment,
        streaming=streaming,
        details={
            "topic": topic,
            "bootstrap_servers": list(settings.bootstrap_servers),
            "payload_source": str(_PAYLOAD_PARTITION),
            "distinct_payloads": len(payloads),
            "mean_value_bytes": round(mean_bytes, 1),
            "flush_every_messages": flush_every,
            "duration_requested_s": args.duration,
            "warmup_s": args.warmup,
        },
    )
    write_report(args.output, report)
    _LOG.info(
        "%.0f msg/s (%.2f MB/s) over %.1f s, environment=%s, threshold %.0f -> %s. Report: %s",
        report["value"],
        result.megabytes_per_second,
        result.elapsed_s,
        args.environment,
        THROUGHPUT_THRESHOLD,
        "meets" if report["passed"] else "BELOW",
        args.output,
    )
    if args.fail_below_threshold and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
