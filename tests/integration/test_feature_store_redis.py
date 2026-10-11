"""Feast against a real Redis (marker `integration`).

Lo que los tests unitarios (SQLite) no pueden probar: que el almacen online de Redis
acepta la configuracion, que `materialize` escribe los 52 sensores, que
FeastClient los lee de vuelta y que key_ttl_seconds hace desaparecer las claves de un
sensor que deja de escribirse. Se ejecutan contra el stack local:

    docker compose -f infra/local/docker-compose.yaml up -d redis
    .\\.venv\\Scripts\\python.exe -m pytest -m integration `
        tests/integration/test_feature_store_redis.py

El Redis del compose escucha en 127.0.0.1:6380 (REACTORGUARD_REDIS_PORT lo cambia). El
puerto 6379 NO se usa nunca: lo suele ocupar el Redis de otro proyecto.

Cada ejecucion usa un proyecto de Feast con sufijo aleatorio, de modo que sus claves no
se mezclan con las de otra ejecucion, y lo desmonta al terminar. Los datos son la cola
(10 muestras) de la fuente REAL de feast_source rebasada a la hora actual: sin rebasar,
los timestamps del ano 2000 estan siempre caducados frente al reloj. Si el artefacto de
feast_source no existe, los tests se saltan.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import redis
from feast import FeatureStore

from ml.features.feature_params import load_feature_params
from ml.features.pipeline import FeaturePipeline
from ml.features.store.feast_client import FeastClient
from ml.features.store.feature_store_params import load_feature_store_params
from ml.features.store.source_builder import (
    EVENT_TIMESTAMP_COLUMN,
    rebase_event_timestamps,
)
from tests.support.feast_store import CONFIG_NAME, build_store, make_params, redis_config_text

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PARAMS = REPO_ROOT / "params.yaml"
SOURCE_ARTIFACT = REPO_ROOT / load_feature_store_params(REAL_PARAMS).source_path

REDIS_HOST = "127.0.0.1"
REDIS_PORT = int(os.environ.get("REACTORGUARD_REDIS_PORT", "6380"))
TAIL_STEPS = 10


def _redis_reachable() -> bool:
    """Return True if the compose Redis answers PING."""
    try:
        return bool(
            redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=1).ping()
        )
    except redis.RedisError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _redis_reachable(),
        reason=(
            f"No hay Redis en {REDIS_HOST}:{REDIS_PORT}. Levantalo con "
            "`docker compose -f infra/local/docker-compose.yaml up -d redis`."
        ),
    ),
    pytest.mark.skipif(
        not SOURCE_ARTIFACT.is_file(),
        reason=f"{SOURCE_ARTIFACT} no existe. Ejecuta .\\infra\\scripts\\Invoke-Pipeline.ps1.",
    ),
]


def _tail_of_the_real_source(anchor: datetime) -> pd.DataFrame:
    """Return the last TAIL_STEPS timesteps of the real source, rebased to anchor.

    Args:
        anchor: Instant that the latest event is moved to.

    Returns:
        The shifted tail, one row per sensor and step.
    """
    frame = pd.read_parquet(SOURCE_ARTIFACT)
    cutoff = frame[EVENT_TIMESTAMP_COLUMN].max() - timedelta(
        seconds=(TAIL_STEPS - 1) * load_feature_store_params(REAL_PARAMS).sample_interval_seconds
    )
    tail = frame[frame[EVENT_TIMESTAMP_COLUMN] >= cutoff]
    return rebase_event_timestamps(tail, anchor)


class _Redis:
    """A materialized Redis-backed store plus the data that fed it."""

    def __init__(self, store: FeatureStore, frame: pd.DataFrame, anchor: datetime) -> None:
        self.store = store
        self.frame = frame
        self.anchor = anchor


def _open(root: Path, key_ttl_seconds: int) -> Iterator[_Redis]:
    """Create, materialize and finally tear down a Redis-backed store.

    Args:
        root: Directory for the config, registry and source.
        key_ttl_seconds: Redis key expiry configured on the online store.

    Yields:
        The store with its data.
    """
    project = f"it_{uuid.uuid4().hex[:8]}"
    anchor = datetime.now(UTC) - timedelta(seconds=30)
    frame = _tail_of_the_real_source(anchor)
    params = make_params(project)
    store = build_store(
        root,
        params,
        frame,
        frame[EVENT_TIMESTAMP_COLUMN].min().to_pydatetime(),
        anchor,
        load_feature_params(REAL_PARAMS),
        config_text=redis_config_text(project, f"{REDIS_HOST}:{REDIS_PORT}", key_ttl_seconds),
    )
    try:
        yield _Redis(store, frame, anchor)
    finally:
        store.teardown()
        leftovers = [
            key
            for key in redis.Redis(host=REDIS_HOST, port=REDIS_PORT).scan_iter()
            if project.encode() in key
        ]
        assert leftovers == [], f"teardown left {len(leftovers)} keys of {project} in Redis"


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Redis]:
    """Store whose keys carry the production expiry (540 s)."""
    ttl = load_feature_store_params(REAL_PARAMS).ttl_seconds
    yield from _open(tmp_path_factory.mktemp("feast_redis"), key_ttl_seconds=ttl)


def _client(handle: _Redis, offset_seconds: float = 0.0) -> FeastClient:
    """Build a client over the store with a clock shifted by offset_seconds.

    Args:
        handle: The materialized store.
        offset_seconds: Seconds added to the real clock.

    Returns:
        A client with the production TTL.
    """
    ttl = load_feature_store_params(REAL_PARAMS).ttl_seconds
    return FeastClient(
        handle.store, ttl, clock=lambda: datetime.now(UTC) + timedelta(seconds=offset_seconds)
    )


def _all_sensors(handle: _Redis) -> list[str]:
    """Return the sensors present in the source, sorted."""
    return sorted(handle.frame["sensor_id"].unique())


def test_materialize_and_read_return_values_for_all_52_sensors(live: _Redis) -> None:
    sensors = _all_sensors(live)
    assert len(sensors) == 52

    result = _client(live).get_online_features(sensors)

    assert result.complete, (result.stale_sensors, result.missing_sensors)
    assert result.fresh_sensors == tuple(sensors)
    names = FeaturePipeline(load_feature_params(REAL_PARAMS), 180).feature_names()
    for sensor in sensors:
        assert set(result.values[sensor]) == set(names)
        assert result.values[sensor]["value"] is not None


def test_the_served_value_is_the_latest_row_of_each_sensor(live: _Redis) -> None:
    latest = live.frame[live.frame[EVENT_TIMESTAMP_COLUMN] == live.anchor].set_index("sensor_id")
    result = _client(live).get_online_features(_all_sensors(live))

    for sensor, row in latest.iterrows():
        assert result.values[str(sensor)]["value"] == pytest.approx(row["value"])


def test_the_lookup_latency_is_reported(live: _Redis) -> None:
    result = _client(live).get_online_features(_all_sensors(live))
    assert 0.0 < result.elapsed_ms < 5_000.0


def test_a_stale_read_returns_nulls_for_every_sensor(live: _Redis) -> None:
    ttl = load_feature_store_params(REAL_PARAMS).ttl_seconds
    result = _client(live, offset_seconds=ttl + 60).get_online_features(_all_sensors(live))

    assert len(result.stale_sensors) == 52
    assert result.fresh_sensors == ()
    assert all(
        value is None for features in result.values.values() for value in features.values()
    )


def test_redis_keys_carry_the_configured_expiry(live: _Redis) -> None:
    ttl = load_feature_store_params(REAL_PARAMS).ttl_seconds
    connection = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
    project = live.store.project.encode()
    remaining = [connection.ttl(key) for key in connection.scan_iter() if project in key]

    assert remaining, "no keys of this project in Redis"
    assert all(0 < seconds <= ttl for seconds in remaining)


def test_a_sensor_that_stops_being_written_expires_to_missing(
    tmp_path: Path,
) -> None:
    # key_ttl_seconds corto SOLO en este test, para no esperar 540 s. Lo que se
    # comprueba es el mecanismo (la clave desaparece y FeastClient lo informa como
    # missing), no el valor de produccion, que ya cubre test_redis_keys_carry_...
    for handle in _open(tmp_path, key_ttl_seconds=2):
        sensors = _all_sensors(handle)
        assert _client(handle).get_online_features(sensors).complete

        time.sleep(3)

        result = _client(handle).get_online_features(sensors)
        assert len(result.missing_sensors) == 52
        assert result.stale_sensors == ()


def test_config_file_is_the_one_feast_loaded(live: _Redis) -> None:
    assert live.store.config.online_store.type == "redis"
    assert (Path(live.store.repo_path) / CONFIG_NAME).is_file()
