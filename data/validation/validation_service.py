"""Entry point of the validation consumer service.

Servicio fino: resuelve la configuracion, monta las piezas de M1 (latido, /health,
/ready, /metrics, cierre ordenado) alrededor de `ValidationConsumer` y traduce el
resultado a un codigo de salida. La logica de validacion y commit vive en
validation_consumer.py.

Configuracion: la seccion `streaming:` y `tep:` de params.yaml (topics, grupo y
tiempos; tabla de spans y cadencia), KAFKA_* para la conexion y la seguridad (ver
data/streaming/kafka_settings.py) y LOG_LEVEL (solo para la linea de comandos). No
hay variables propias del servicio.

Codigos de salida: 0 parada por SIGTERM/SIGINT, 1 fallo en ejecucion (PublishError,
CommitError, lectura que el validador rechaza incluso con estado limpio, puertos
ocupados) y 2 configuracion invalida. Tras un fallo en ejecucion el lote en curso NO
se confirma: Kubernetes reinicia el pod y el lote se reprocesa (at-least-once).
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, KafkaProducer

from data.generators.tep_params import DEFAULT_PARAMS_PATH, TEPParams, load_tep_params
from data.schemas.sensor_spans import load_sensor_spans
from data.streaming.errors import ConfigurationError
from data.streaming.kafka_adapters import KafkaMessageConsumer, KafkaMessageProducer
from data.streaming.kafka_settings import KafkaConnectionSettings
from data.streaming.lifecycle import GracefulShutdown
from data.streaming.logging_setup import configure_logging
from data.streaming.metrics import StreamMetrics
from data.streaming.observability import Heartbeat, ServiceHttpServer
from data.streaming.streaming_params import StreamingParams, load_streaming_params
from data.validation.sensor_validator import SensorValidator, StuckValueDetector
from data.validation.validation_consumer import ValidationConsumer

_LOG = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_CONFIG = 2

_WATCHER_JOIN_SECONDS = 1.0
_SECONDS_PER_MINUTE = 60.0


def _client_id(streaming: StreamingParams) -> str:
    """Return the client id shown in broker logs, unique per pod.

    Args:
        streaming: Streaming parameters (the consumer group names the service).

    Returns:
        The group name plus the host name, which in Kubernetes is the pod name.
    """
    return f"{streaming.consumer_group}-{socket.gethostname()}"


def _warmup_seconds(validator: SensorValidator, tep: TEPParams) -> float | None:
    """Return the data time a sensor needs to rebuild a stuck run from scratch.

    Args:
        validator: The validator whose detectors are inspected.
        tep: TEP parameters (the sampling interval).

    Returns:
        Stuck window times the sampling interval in seconds (8 x 180 s = 24 min by
        default), or None when the validator has no stuck detector.
    """
    for detector in validator.detectors:
        if isinstance(detector, StuckValueDetector):
            return detector.window_size * tep.sample_interval_minutes * _SECONDS_PER_MINUTE
    return None


def _load_configuration(
    params_path: str | Path, env: Mapping[str, str] | None
) -> tuple[StreamingParams, KafkaConnectionSettings, TEPParams, SensorValidator]:
    """Resolve and cross-check every piece of configuration.

    Args:
        params_path: Path to params.yaml.
        env: Environment mapping; defaults to os.environ.

    Returns:
        The streaming parameters, the Kafka connection settings, the TEP
        parameters and a validator built from the calibrated span table.

    Raises:
        FileNotFoundError: If params.yaml or the span table does not exist.
        KeyError: If a section or key is missing.
        TypeError: If a span entry is not a mapping.
        ValueError: If a value is invalid or inconsistent.
        ConfigurationError: If the KAFKA_* variables are missing or inconsistent.
    """
    streaming = load_streaming_params(params_path)
    tep = load_tep_params(params_path)
    settings = KafkaConnectionSettings.from_env(env)
    validator = SensorValidator(load_sensor_spans(tep.spans_path))
    return streaming, settings, tep, validator


def main(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    env: Mapping[str, str] | None = None,
    *,
    producer_factory: Callable[..., Any] = KafkaProducer,
    consumer_factory: Callable[..., Any] = KafkaConsumer,
    shutdown: GracefulShutdown | None = None,
) -> int:
    """Run the validation service until it fails or is told to stop.

    Args:
        params_path: Path to the params file.
        env: Environment mapping; defaults to os.environ.
        producer_factory: Constructor of the underlying Kafka producer. Tests
            replace it with a fake.
        consumer_factory: Constructor of the underlying Kafka consumer. Tests
            replace it with a fake.
        shutdown: Shutdown handler. A new one (SIGTERM and SIGINT) by default.

    Returns:
        The process exit code: EXIT_OK, EXIT_RUNTIME or EXIT_CONFIG.
    """
    try:
        streaming, settings, tep, validator = _load_configuration(params_path, env)
    except (FileNotFoundError, KeyError, TypeError, ValueError, ConfigurationError) as exc:
        _LOG.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG

    metrics = StreamMetrics()
    heartbeat = Heartbeat(streaming.staleness_seconds, metrics=metrics)
    ready = threading.Event()
    validation: ValidationConsumer | None = None

    def _is_ready() -> bool:
        return ready.is_set() and validation is not None and validation.is_ready

    server = ServiceHttpServer(
        host=streaming.bind_host,
        health_port=streaming.health_port,
        metrics_port=streaming.metrics_port,
        metrics=metrics,
        liveness=heartbeat.is_alive,
        readiness=_is_ready,
    )
    try:
        server.start()
    except OSError as exc:
        _LOG.error("Could not bind the HTTP ports: %s", exc)
        return EXIT_RUNTIME

    stop = GracefulShutdown() if shutdown is None else shutdown
    consumer: KafkaMessageConsumer | None = None
    producer: KafkaMessageProducer | None = None
    watcher: threading.Thread | None = None
    exit_code = EXIT_OK
    try:
        with stop:
            client_id = _client_id(streaming)
            producer = KafkaMessageProducer(
                settings, streaming, client_id=client_id, producer_factory=producer_factory
            )
            consumer = KafkaMessageConsumer(
                settings, streaming, client_id=client_id, consumer_factory=consumer_factory
            )
            validation = ValidationConsumer(
                consumer,
                producer,
                validator,
                streaming,
                metrics,
                on_tick=heartbeat.beat,
                warmup_seconds=_warmup_seconds(validator, tep),
            )
            running = validation

            def _stop_consumer_on_shutdown() -> None:
                stop.wait()
                running.stop()

            watcher = threading.Thread(
                target=_stop_consumer_on_shutdown,
                name="reactorguard-stop-watcher",
                daemon=True,
            )
            watcher.start()
            ready.set()
            validation.run()
            if stop.signal_name is not None:
                _LOG.info("Stopped on %s", stop.signal_name)
    except Exception:
        _LOG.exception("Validation consumer failed")
        exit_code = EXIT_RUNTIME
    finally:
        ready.clear()
        stop.request()
        if watcher is not None:
            watcher.join(timeout=_WATCHER_JOIN_SECONDS)
        if consumer is not None:
            try:
                consumer.close()
            except Exception:
                _LOG.exception("Error closing the Kafka consumer")
        if producer is not None:
            try:
                producer.close(streaming.close_timeout_ms / 1000.0)
            except Exception:
                _LOG.exception("Error closing the Kafka producer")
        server.stop()
    return exit_code


def cli(argv: Sequence[str] | None = None) -> int:
    """Command-line wrapper: JSON logging and the --params option.

    Args:
        argv: Arguments to parse; defaults to sys.argv[1:].

    Returns:
        The exit code of main().
    """
    parser = argparse.ArgumentParser(
        description="Validate the raw sensor readings and publish verdicts and alerts."
    )
    parser.add_argument(
        "--params", default=str(DEFAULT_PARAMS_PATH), help="Path to params.yaml."
    )
    args = parser.parse_args(argv)
    try:
        configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    except ValueError as exc:
        sys.stderr.write(f"Invalid LOG_LEVEL: {exc}" + os.linesep)
        return EXIT_CONFIG
    return main(args.params)


if __name__ == "__main__":
    sys.exit(cli())
