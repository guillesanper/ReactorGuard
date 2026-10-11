"""Client for the online feature store, with freshness enforced here.

POR QUE LA CADUCIDAD SE APLICA AQUI. Medido en Feast 0.64: el `ttl` del FeatureView
NO caduca nada en la lectura online. Solo acota la ventana de `materialize` y los
joins historicos; `get_online_features` sirve el ultimo valor escrito aunque tenga
dias (con un TTL de 540 s, un valor de 2000 s de antiguedad y otro de 26 anos se
sirvieron igual). Lo contrario de lo que asumia Handoff_Paso4.md: sin esta capa, un
sensor cuyo productor se cae seguiria sirviendo su ultima feature indefinidamente,
que es el fallo silencioso que el TTL debia impedir.

FeastClient mira el event_timestamp de cada fila devuelta y, si es mas viejo que
ttl_seconds, entrega nulos y lo lista en `stale_sensors`. Funciona igual con SQLite
(local) y con Redis (cluster). En Redis, ademas, key_ttl_seconds de
feature_store.cluster.yaml hace que la clave desaparezca del almacen; entonces el
sensor aparece en `missing_sensors`.

Nulo y ausente son dos casos distintos y se reportan por separado:
  stale    el sensor tiene valor, pero mas viejo que el TTL.
  missing  el sensor no tiene ningun valor en el almacen.

Los fallos de infraestructura (Redis caido, SQLite bloqueado) se elevan como
ConnectionError, no como nulos: un almacen inalcanzable no es un sensor sin dato.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from feast import FeatureStore
from feast.errors import FeatureServiceNotFoundException
from redis.exceptions import RedisError

from ml.features.store.entities import SENSOR_JOIN_KEY
from ml.features.store.feature_store_params import FeatureStoreParams
from ml.features.store.feature_views import SENSOR_ANALYSIS_SERVICE

_LOG = logging.getLogger(__name__)

_TIMESTAMP_SUFFIX = "__ts"
_PLACEHOLDER_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _utc_now() -> datetime:
    """Return the current instant as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


@dataclass(frozen=True)
class OnlineFeatures:
    """Result of one online lookup.

    Attributes:
        values: For every requested sensor, feature name to value. A feature is
            None when it is null in the store (warm-up of a rolling window), and ALL
            of a sensor's features are None when the sensor is stale or missing.
        event_timestamps: Event time of each sensor's row, or None if missing.
        stale_sensors: Sensors whose row is older than the TTL.
        missing_sensors: Sensors with no row in the store.
        elapsed_ms: Wall time of the store call, in milliseconds.
    """

    values: dict[str, dict[str, float | None]]
    event_timestamps: dict[str, datetime | None]
    stale_sensors: tuple[str, ...]
    missing_sensors: tuple[str, ...]
    elapsed_ms: float

    @property
    def fresh_sensors(self) -> tuple[str, ...]:
        """Return the sensors that are neither stale nor missing."""
        unusable = set(self.stale_sensors) | set(self.missing_sensors)
        return tuple(sensor for sensor in self.values if sensor not in unusable)

    @property
    def complete(self) -> bool:
        """Return True when every requested sensor is fresh."""
        return not self.stale_sensors and not self.missing_sensors


def assert_placeholders_resolved(config_path: Path) -> None:
    """Fail loudly if a feature_store.yaml still holds an unset ${VAR}.

    Feast expande las variables de entorno con os.path.expandvars, que deja
    INTACTA una variable no definida. Sin esta comprobacion la cadena de conexion
    de Redis llevaria la contrasena literal "${REDIS_PASSWORD}" y el fallo
    apareceria como un AUTH rechazado, lejos de su causa. Las lineas de comentario
    se ignoran: documentan el nombre de la variable.

    Args:
        config_path: The feature_store.yaml to check.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a ${VAR} placeholder remains after expansion.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Feature store config not found: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    code = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    unresolved = sorted(set(_PLACEHOLDER_RE.findall(os.path.expandvars(code))))
    if unresolved:
        raise ValueError(
            f"{config_path} references environment variables that are not set: "
            f"{', '.join(unresolved)}. En el cluster las inyecta el Secret montado "
            "en el pod."
        )


def _clean(value: object) -> float | None:
    """Map a store value to float or None, treating NaN as null.

    Args:
        value: Raw value from OnlineResponse.to_dict().

    Returns:
        The value as a float, or None if it is None or NaN.
    """
    if value is None:
        return None
    number = float(value)  # type: ignore[arg-type]
    return None if math.isnan(number) else number


class FeastClient:
    """Reads sensor features from the online store and enforces the TTL.

    Attributes:
        ttl_seconds: Age beyond which a row counts as stale.
    """

    def __init__(
        self,
        store: FeatureStore,
        ttl_seconds: int,
        service_name: str = SENSOR_ANALYSIS_SERVICE,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Build the client.

        Args:
            store: A Feast feature store whose registry already holds the service.
            ttl_seconds: Maximum age of a served row, in seconds.
            service_name: FeatureService to read.
            clock: Source of the current time, injectable for tests.

        Raises:
            ValueError: If ttl_seconds is not positive.
        """
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}.")
        self._store = store
        self.ttl_seconds = ttl_seconds
        self._service_name = service_name
        self._clock = clock

    @classmethod
    def from_params(
        cls,
        store_params: FeatureStoreParams,
        repo_path: Path | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> FeastClient:
        """Build a client from params.yaml, using the config of the chosen backend.

        Args:
            store_params: Resolved feature_store: section.
            repo_path: Root that the config's relative paths resolve against.
                Defaults to the current directory.
            clock: Source of the current time.

        Returns:
            A client over the store described by store_params.config_path.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the config references an unset environment variable.
        """
        root = repo_path if repo_path is not None else Path.cwd()
        config = store_params.config_path
        if not config.is_absolute():
            config = root / config
        assert_placeholders_resolved(config)

        store = FeatureStore(repo_path=str(root), fs_yaml_file=config)
        return cls(store, store_params.ttl_seconds, clock=clock)

    def get_online_features(self, sensor_ids: Sequence[str]) -> OnlineFeatures:
        """Fetch the latest features of some sensors.

        Args:
            sensor_ids: Sensors to look up. Must be non-empty, non-blank and unique.

        Returns:
            Values, event times and the stale and missing sensors.

        Raises:
            ValueError: If sensor_ids is empty, has a blank entry or repeats one.
            LookupError: If the FeatureService is not registered (`feast apply`).
            ConnectionError: If the online store cannot be reached.
        """
        self._validate(sensor_ids)

        started = time.perf_counter()
        try:
            service = self._store.get_feature_service(self._service_name)
            response = self._store.get_online_features(
                features=service,
                entity_rows=[{SENSOR_JOIN_KEY: sensor} for sensor in sensor_ids],
            ).to_dict(include_event_timestamps=True)
        except FeatureServiceNotFoundException as exc:
            raise LookupError(
                f"Feature service '{self._service_name}' is not registered. "
                "Ejecuta `feast apply` (apply_definitions) antes de leer."
            ) from exc
        except (RedisError, sqlite3.Error, OSError) as exc:
            raise ConnectionError(
                f"Online feature store unreachable: {type(exc).__name__}: {exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return self._interpret(response, elapsed_ms)

    @staticmethod
    def _validate(sensor_ids: Sequence[str]) -> None:
        """Check the sensor list.

        Args:
            sensor_ids: Sensors requested.

        Raises:
            ValueError: If it is empty, has a blank entry or repeats one.
        """
        if not sensor_ids:
            raise ValueError("sensor_ids must not be empty.")
        if any(not sensor.strip() for sensor in sensor_ids):
            raise ValueError("sensor_ids must not contain blank identifiers.")
        if len(set(sensor_ids)) != len(sensor_ids):
            raise ValueError("sensor_ids must be unique: a repeated sensor is ambiguous.")

    def _interpret(self, response: dict[str, Any], elapsed_ms: float) -> OnlineFeatures:
        """Turn a raw Feast response into values plus freshness verdicts.

        Args:
            response: OnlineResponse.to_dict(include_event_timestamps=True).
            elapsed_ms: Time the store call took.

        Returns:
            The structured result.
        """
        names = [
            key
            for key in response
            if key != SENSOR_JOIN_KEY and not key.endswith(_TIMESTAMP_SUFFIX)
        ]
        # Todas las features de una fila se escriben con el mismo timestamp, asi que
        # basta el de la primera para juzgar la frescura de la fila.
        probe = response[names[0] + _TIMESTAMP_SUFFIX]
        now = self._clock().timestamp()

        values: dict[str, dict[str, float | None]] = {}
        stamps: dict[str, datetime | None] = {}
        stale: list[str] = []
        missing: list[str] = []

        for row, sensor in enumerate(response[SENSOR_JOIN_KEY]):
            event_seconds = int(probe[row])
            if event_seconds == 0:
                missing.append(sensor)
                stamps[sensor] = None
                values[sensor] = dict.fromkeys(names)
            elif now - event_seconds > self.ttl_seconds:
                stale.append(sensor)
                stamps[sensor] = datetime.fromtimestamp(event_seconds, tz=UTC)
                values[sensor] = dict.fromkeys(names)
            else:
                stamps[sensor] = datetime.fromtimestamp(event_seconds, tz=UTC)
                values[sensor] = {name: _clean(response[name][row]) for name in names}

        if stale or missing:
            _LOG.warning(
                "Online lookup of %d sensors: %d stale (older than %d s), %d missing.",
                len(values),
                len(stale),
                self.ttl_seconds,
                len(missing),
            )
        return OnlineFeatures(
            values=values,
            event_timestamps=stamps,
            stale_sensors=tuple(stale),
            missing_sensors=tuple(missing),
            elapsed_ms=elapsed_ms,
        )
