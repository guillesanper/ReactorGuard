"""Typed access to the streaming: section of params.yaml.

Mismo patron que data/generators/tep_params.py: dataclass inmutable, `_require`
para claves obligatorias y validacion en la carga, de modo que un valor
incoherente falle al arrancar el servicio y no a mitad de un lote.

Los valores de tiempo se calibran a la cadencia del TEP (180 s), no a la del TDD
(1 s). Las relaciones entre ellos (peor caso de un lote frente a
max_poll_interval_ms, entrega frente a linger mas timeout de peticion) se validan
aqui porque romperlas no produce un error claro en Kafka: produce expulsiones
del grupo o envios que caducan antes de poder reintentarse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "streaming"

_COMPRESSION_TYPES = frozenset({"gzip", "snappy", "lz4", "zstd"})
_AUTO_OFFSET_RESETS = frozenset({"earliest", "latest"})
_ISOLATION_LEVELS = frozenset({"read_uncommitted", "read_committed"})
_MAX_PORT = 65535
# Maximo de peticiones en vuelo con el que la idempotencia de Kafka conserva el
# orden por particion.
_MAX_IN_FLIGHT_FOR_ORDER = 5


@dataclass(frozen=True)
class StreamingParams:
    """Resolved configuration shared by the streamer and the validation consumer.

    Attributes:
        raw_topic: Topic holding unvalidated readings.
        validated_topic: Topic receiving every validated reading.
        alerts_topic: Topic receiving one event per detected fault.
        consumer_group: Consumer group id; must match the ACL of the KafkaUser.
        poll_timeout_ms: Maximum wait of one consumer poll.
        max_poll_records: Maximum records returned by one poll (batch ceiling).
        auto_offset_reset: Where a group with no committed offset starts.
        isolation_level: Visibility of transactional records.
        commit_retries: Extra commit attempts after the first failure.
        commit_timeout_ms: Timeout of one synchronous commit.
        close_timeout_ms: Timeout of the consumer and producer shutdown.
        max_poll_interval_ms: Maximum time between polls before Kafka evicts the
            consumer from the group.
        compression_type: Producer codec, or None for no compression.
        batch_linger_ms: Time the producer waits to fill a batch.
        producer_batch_bytes: Target size of one producer batch.
        max_in_flight_requests: Unacknowledged requests per connection.
        request_timeout_ms: Timeout of one producer request.
        delivery_timeout_ms: Upper bound for a send, retries included.
        flush_timeout_s: Timeout of one producer flush.
        bind_host: Interface the HTTP servers bind to.
        health_port: Port serving /health and /ready.
        metrics_port: Port serving /metrics.
        staleness_seconds: Heartbeat age after which /health reports failure.
    """

    raw_topic: str
    validated_topic: str
    alerts_topic: str
    consumer_group: str
    poll_timeout_ms: int
    max_poll_records: int
    auto_offset_reset: str
    isolation_level: str
    commit_retries: int
    commit_timeout_ms: int
    close_timeout_ms: int
    max_poll_interval_ms: int
    compression_type: str | None
    batch_linger_ms: int
    producer_batch_bytes: int
    max_in_flight_requests: int
    request_timeout_ms: int
    delivery_timeout_ms: int
    flush_timeout_s: float
    bind_host: str
    health_port: int
    metrics_port: int
    staleness_seconds: float

    @property
    def worst_case_batch_seconds(self) -> float:
        """Return the longest a batch can take: one flush plus every commit try."""
        commit_s = (self.commit_retries + 1) * self.commit_timeout_ms / 1000.0
        return self.flush_timeout_s + commit_s


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed streaming: mapping.
        key: Key that must be present.

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
    return section[key]


def _positive_int(section: dict[str, Any], key: str) -> int:
    """Read an integer that must be strictly positive.

    Args:
        section: The parsed streaming: mapping.
        key: Key to read.

    Returns:
        The value as an int.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not positive.
    """
    value = int(_require(section, key))
    if value <= 0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be > 0, got {value}.")
    return value


def _non_negative_int(section: dict[str, Any], key: str) -> int:
    """Read an integer that must be zero or positive.

    Args:
        section: The parsed streaming: mapping.
        key: Key to read.

    Returns:
        The value as an int.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is negative.
    """
    value = int(_require(section, key))
    if value < 0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be >= 0, got {value}.")
    return value


def _positive_float(section: dict[str, Any], key: str) -> float:
    """Read a float that must be strictly positive.

    Args:
        section: The parsed streaming: mapping.
        key: Key to read.

    Returns:
        The value as a float.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not positive.
    """
    value = float(_require(section, key))
    if value <= 0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be > 0, got {value}.")
    return value


def _non_empty_str(section: dict[str, Any], key: str) -> str:
    """Read a string that must not be blank.

    Args:
        section: The parsed streaming: mapping.
        key: Key to read.

    Returns:
        The stripped value.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is blank.
    """
    value = str(_require(section, key)).strip()
    if not value:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must not be empty.")
    return value


def _choice(section: dict[str, Any], key: str, allowed: frozenset[str]) -> str:
    """Read a string restricted to a closed set.

    Args:
        section: The parsed streaming: mapping.
        key: Key to read.
        allowed: Accepted values.

    Returns:
        The value.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not in allowed.
    """
    value = str(_require(section, key))
    if value not in allowed:
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be one of {sorted(allowed)}, got '{value}'."
        )
    return value


def _port(section: dict[str, Any], key: str) -> int:
    """Read a TCP port.

    Args:
        section: The parsed streaming: mapping.
        key: Key to read.

    Returns:
        The port number.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is outside 1..65535.
    """
    value = int(_require(section, key))
    if not 1 <= value <= _MAX_PORT:
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be in 1..{_MAX_PORT}, got {value}."
        )
    return value


def _compression(section: dict[str, Any]) -> str | None:
    """Read the producer codec; 'none' and null both mean no compression.

    Args:
        section: The parsed streaming: mapping.

    Returns:
        The codec name, or None when compression is disabled.

    Raises:
        KeyError: If compression_type is missing.
        ValueError: If the codec is not supported.
    """
    raw = _require(section, "compression_type")
    if raw is None or str(raw).lower() == "none":
        return None
    value = str(raw)
    if value not in _COMPRESSION_TYPES:
        raise ValueError(
            f"params.yaml: '{_SECTION}.compression_type' must be 'none' or one of "
            f"{sorted(_COMPRESSION_TYPES)}, got '{value}'."
        )
    return value


def _validate_relations(params: StreamingParams) -> None:
    """Check the cross-field invariants of a resolved configuration.

    Args:
        params: The configuration to check.

    Raises:
        ValueError: If two topics collide, the ports collide, the in-flight limit
            breaks ordering, delivery cannot cover linger plus a request, or the
            worst-case batch does not fit in max_poll_interval_ms.
    """
    topics = (params.raw_topic, params.validated_topic, params.alerts_topic)
    if len(set(topics)) != len(topics):
        raise ValueError(f"params.yaml: '{_SECTION}' topics must be distinct, got {topics}.")
    if params.health_port == params.metrics_port:
        raise ValueError(
            f"params.yaml: '{_SECTION}.health_port' and 'metrics_port' must differ, "
            f"both are {params.health_port}."
        )
    if params.max_in_flight_requests > _MAX_IN_FLIGHT_FOR_ORDER:
        raise ValueError(
            f"params.yaml: '{_SECTION}.max_in_flight_requests' must be <= "
            f"{_MAX_IN_FLIGHT_FOR_ORDER} to keep per-sensor ordering, "
            f"got {params.max_in_flight_requests}."
        )
    minimum_delivery = params.batch_linger_ms + params.request_timeout_ms
    if params.delivery_timeout_ms < minimum_delivery:
        raise ValueError(
            f"params.yaml: '{_SECTION}.delivery_timeout_ms' ({params.delivery_timeout_ms}) "
            f"must be >= batch_linger_ms + request_timeout_ms ({minimum_delivery})."
        )
    worst_case_ms = params.worst_case_batch_seconds * 1000.0
    if worst_case_ms >= params.max_poll_interval_ms:
        raise ValueError(
            f"params.yaml: worst-case batch time {params.worst_case_batch_seconds:.1f} s "
            f"must be below '{_SECTION}.max_poll_interval_ms' "
            f"({params.max_poll_interval_ms} ms), or Kafka evicts the consumer mid-batch."
        )


def load_streaming_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> StreamingParams:
    """Load and validate the streaming: section of a params.yaml file.

    Args:
        params_path: Path to the params file. Defaults to params.yaml at the
            repository root.

    Returns:
        A fully resolved StreamingParams instance.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the streaming: section or a required key is missing.
        ValueError: If a value is out of range or the values are inconsistent.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    section = document[_SECTION]

    params = StreamingParams(
        raw_topic=_non_empty_str(section, "raw_topic"),
        validated_topic=_non_empty_str(section, "validated_topic"),
        alerts_topic=_non_empty_str(section, "alerts_topic"),
        consumer_group=_non_empty_str(section, "consumer_group"),
        poll_timeout_ms=_positive_int(section, "poll_timeout_ms"),
        max_poll_records=_positive_int(section, "max_poll_records"),
        auto_offset_reset=_choice(section, "auto_offset_reset", _AUTO_OFFSET_RESETS),
        isolation_level=_choice(section, "isolation_level", _ISOLATION_LEVELS),
        commit_retries=_non_negative_int(section, "commit_retries"),
        commit_timeout_ms=_positive_int(section, "commit_timeout_ms"),
        close_timeout_ms=_positive_int(section, "close_timeout_ms"),
        max_poll_interval_ms=_positive_int(section, "max_poll_interval_ms"),
        compression_type=_compression(section),
        batch_linger_ms=_non_negative_int(section, "batch_linger_ms"),
        producer_batch_bytes=_positive_int(section, "producer_batch_bytes"),
        max_in_flight_requests=_positive_int(section, "max_in_flight_requests"),
        request_timeout_ms=_positive_int(section, "request_timeout_ms"),
        delivery_timeout_ms=_positive_int(section, "delivery_timeout_ms"),
        flush_timeout_s=_positive_float(section, "flush_timeout_s"),
        bind_host=_non_empty_str(section, "bind_host"),
        health_port=_port(section, "health_port"),
        metrics_port=_port(section, "metrics_port"),
        staleness_seconds=_positive_float(section, "staleness_seconds"),
    )
    _validate_relations(params)
    return params
