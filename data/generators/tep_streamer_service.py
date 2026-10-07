"""Entry point of the TEP streamer service.

Servicio fino: resuelve la configuracion, monta las piezas de M1 (latido, /health,
/ready, /metrics, cierre ordenado) alrededor de `TEPStreamer` y traduce el resultado
a un codigo de salida. La logica de streaming vive en tep_streamer.py.

Configuracion: la seccion `streamer:` de params.yaml, mas las variables de entorno
STREAM_MODE, SPEED_MULTIPLIER y DATA_SOURCE (sobrescriben el valor de params.yaml),
KAFKA_* (conexion y seguridad, ver data/streaming/kafka_settings.py) y LOG_LEVEL
(solo para la linea de comandos).

Codigos de salida: 0 fin normal o parada por SIGTERM/SIGINT, 1 fallo en ejecucion
(PublishError, fuente ausente, dato invalido) y 2 configuracion invalida. Con
loop=false el proceso termina al agotar los datos: un Deployment lo relanzaria y
reemitiria los 22 runs, asi que en el cluster se usa loop=true (o un Job).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from kafka import KafkaProducer

from data.generators.reading_source import ParquetDirSource, ReadingSource, StorageSource
from data.generators.tep_params import DEFAULT_PARAMS_PATH
from data.generators.tep_streamer import TEPStreamer
from data.generators.tep_streamer_params import (
    DataSource,
    StreamerParams,
    apply_env_overrides,
    load_streamer_params,
    validate_against_streaming,
)
from data.storage.blob_store import BlobStore, GCSBlobStore
from data.storage.storage_params import StorageParams, load_storage_params
from data.streaming.errors import ConfigurationError
from data.streaming.kafka_adapters import KafkaMessageProducer
from data.streaming.kafka_settings import KafkaConnectionSettings
from data.streaming.lifecycle import GracefulShutdown
from data.streaming.logging_setup import configure_logging
from data.streaming.metrics import StreamMetrics
from data.streaming.observability import Heartbeat, ServiceHttpServer
from data.streaming.streaming_params import StreamingParams, load_streaming_params

_LOG = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_CONFIG = 2

BlobStoreFactory = Callable[[StorageParams], BlobStore]

_WATCHER_JOIN_SECONDS = 1.0


def _gcs_store(storage: StorageParams) -> BlobStore:
    """Open the raw-data bucket with the ambient credentials (Workload Identity).

    Args:
        storage: Storage parameters naming the bucket and the request timeout.

    Returns:
        A store bound to the raw-data bucket.
    """
    return GCSBlobStore(storage.raw_bucket, timeout_s=storage.request_timeout_s)


def _build_source(
    params: StreamerParams,
    storage: StorageParams,
    blob_store_factory: BlobStoreFactory,
) -> ReadingSource:
    """Create the reading source selected by params.data_source.

    Args:
        params: Streamer parameters.
        storage: Storage parameters (bucket name and timeout of the GCS source).
        blob_store_factory: Builds the store of the GCS source.

    Returns:
        The reading source.
    """
    if params.data_source is DataSource.PARQUET:
        return ParquetDirSource(params.parquet_dir)
    return StorageSource(blob_store_factory(storage), params.storage_prefix)


def _load_configuration(
    params_path: str | Path, env: Mapping[str, str] | None
) -> tuple[StreamerParams, StreamingParams, KafkaConnectionSettings, StorageParams]:
    """Resolve and cross-check every piece of configuration.

    Args:
        params_path: Path to params.yaml.
        env: Environment mapping; defaults to os.environ.

    Returns:
        The streamer parameters (environment overrides applied), the streaming
        parameters, the Kafka connection settings and the storage parameters.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If a section or key is missing.
        ValueError: If a value is invalid or inconsistent.
        ConfigurationError: If the KAFKA_* variables are missing or inconsistent.
    """
    streaming = load_streaming_params(params_path)
    params = apply_env_overrides(load_streamer_params(params_path), env)
    validate_against_streaming(params, streaming)
    settings = KafkaConnectionSettings.from_env(env)
    return params, streaming, settings, load_storage_params(params_path)


def main(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    env: Mapping[str, str] | None = None,
    *,
    producer_factory: Callable[..., Any] = KafkaProducer,
    blob_store_factory: BlobStoreFactory = _gcs_store,
    shutdown: GracefulShutdown | None = None,
) -> int:
    """Run the streamer service until it finishes, fails or is told to stop.

    Args:
        params_path: Path to the params file.
        env: Environment mapping; defaults to os.environ.
        producer_factory: Constructor of the underlying Kafka producer. Tests
            replace it with a fake.
        blob_store_factory: Builds the store of the GCS source. Tests replace it.
        shutdown: Shutdown handler. A new one (SIGTERM and SIGINT) by default.

    Returns:
        The process exit code: EXIT_OK, EXIT_RUNTIME or EXIT_CONFIG.
    """
    try:
        params, streaming, settings, storage = _load_configuration(params_path, env)
    except (FileNotFoundError, KeyError, ValueError, ConfigurationError) as exc:
        _LOG.error("Invalid configuration: %s", exc)
        return EXIT_CONFIG

    metrics = StreamMetrics()
    heartbeat = Heartbeat(streaming.staleness_seconds, metrics=metrics)
    ready = threading.Event()
    server = ServiceHttpServer(
        host=streaming.bind_host,
        health_port=streaming.health_port,
        metrics_port=streaming.metrics_port,
        metrics=metrics,
        liveness=heartbeat.is_alive,
        readiness=ready.is_set,
    )
    try:
        server.start()
    except OSError as exc:
        _LOG.error("Could not bind the HTTP ports: %s", exc)
        return EXIT_RUNTIME

    stop = GracefulShutdown() if shutdown is None else shutdown
    producer: KafkaMessageProducer | None = None
    watcher: threading.Thread | None = None
    exit_code = EXIT_OK
    try:
        with stop:
            source = _build_source(params, storage, blob_store_factory)
            producer = KafkaMessageProducer(
                settings,
                streaming,
                client_id=params.client_id,
                producer_factory=producer_factory,
            )
            streamer = TEPStreamer(
                producer,
                source,
                params,
                metrics,
                topic=streaming.raw_topic,
                flush_timeout_s=streaming.flush_timeout_s,
                on_tick=heartbeat.beat,
            )
            def _stop_streamer_on_shutdown() -> None:
                stop.wait()
                streamer.stop()

            watcher = threading.Thread(
                target=_stop_streamer_on_shutdown,
                name="reactorguard-stop-watcher",
                daemon=True,
            )
            watcher.start()
            ready.set()
            streamer.run()
            if stop.signal_name is not None:
                _LOG.info("Stopped on %s", stop.signal_name)
    except Exception:
        _LOG.exception("TEP streamer failed")
        exit_code = EXIT_RUNTIME
    finally:
        ready.clear()
        stop.request()
        if watcher is not None:
            watcher.join(timeout=_WATCHER_JOIN_SECONDS)
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
    parser = argparse.ArgumentParser(description="Stream the adapted TEP readings to Kafka.")
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
