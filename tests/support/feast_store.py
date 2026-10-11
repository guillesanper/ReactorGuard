"""Helpers to run the real Feast against a throwaway SQLite store in tests.

No hay fakeredis: los tests unitarios ejercitan Feast de verdad (apply, materialize,
get_online_features) con el almacen SQLite en tmp_path. Es mas fiel que un doble y
no necesita red ni Docker; Redis queda para la prueba de integracion.

Las rutas del feature_store.yaml son relativas a la raiz del almacen (repo_path):
una ruta absoluta de Windows (C:\\...) se interpreta como URL con esquema "c".
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from feast import FeatureStore

from ml.features.feature_params import FeatureParams
from ml.features.store.feature_store_params import FeatureStoreParams
from ml.features.store.feature_views import apply_definitions, build_definitions
from ml.features.store.source_builder import (
    CREATED_TIMESTAMP_COLUMN,
    EVENT_TIMESTAMP_COLUMN,
)

CONFIG_NAME = "feature_store.yaml"
SOURCE_NAME = "source.parquet"
INTERVAL_SECONDS = 180


def local_config_text(project: str) -> str:
    """Return a SQLite feature_store.yaml with paths relative to the store root.

    Args:
        project: Feast project name.

    Returns:
        The YAML text.
    """
    return (
        f"project: {project}\n"
        "provider: local\n"
        "registry: registry.db\n"
        "online_store:\n"
        "  type: sqlite\n"
        "  path: online.db\n"
        "offline_store:\n"
        "  type: file\n"
        "entity_key_serialization_version: 3\n"
    )


def redis_config_text(project: str, endpoint: str, key_ttl_seconds: int) -> str:
    """Return a Redis feature_store.yaml with a local registry file.

    Replica la forma de feature_store.cluster.yaml (online Redis con
    key_ttl_seconds) apuntando a un Redis de Docker y con el registro en disco, ya
    que el bucket del cluster no es alcanzable desde las pruebas.

    Args:
        project: Feast project name (it prefixes the Redis keys).
        endpoint: "host:port" of the Redis server.
        key_ttl_seconds: Expiry of an entity's key, in seconds.

    Returns:
        The YAML text.
    """
    return (
        f"project: {project}\n"
        "provider: local\n"
        "registry: registry.db\n"
        "online_store:\n"
        "  type: redis\n"
        "  redis_type: redis\n"
        f'  connection_string: "{endpoint}"\n'
        f"  key_ttl_seconds: {key_ttl_seconds}\n"
        "offline_store:\n"
        "  type: file\n"
        "entity_key_serialization_version: 3\n"
    )


def make_params(project: str = "unit_test") -> FeatureStoreParams:
    """Build store parameters that point at files inside the store root.

    Args:
        project: Feast project name.

    Returns:
        Parameters with the production TTL (3 intervals = 540 s).
    """
    return FeatureStoreParams(
        project=project,
        ttl_intervals=3,
        run_gap_intervals=10,
        online_backend="sqlite",
        source_path=Path(SOURCE_NAME),
        local_config_path=Path(CONFIG_NAME),
        cluster_config_path=Path("cluster.yaml"),
        sample_interval_seconds=INTERVAL_SECONDS,
    )


def synthetic_source_frame(
    feature_names: Sequence[str],
    sensors: Sequence[str],
    last_event: datetime,
    steps: int = 4,
    nan_cells: Sequence[tuple[str, str]] = (),
) -> pd.DataFrame:
    """Build an offline source with one row per sensor and step.

    Args:
        feature_names: Feature columns, as FeaturePipeline.feature_names() lists.
        sensors: Sensor ids.
        last_event: Event time of the latest step.
        steps: Rows per sensor, spaced INTERVAL_SECONDS apart and ending at last_event.
        nan_cells: (sensor_id, feature) pairs set to NaN on the LATEST row, to
            reproduce the warm-up nulls of rolling windows.

    Returns:
        A frame shaped like the feast_source output. The value of the latest row of
        a sensor is 1000 x (sensor index + 1) + feature index, and earlier rows are
        smaller, so a test can tell which row was served.
    """
    rows: list[dict[str, object]] = []
    for step in range(steps):
        moment = last_event - timedelta(seconds=INTERVAL_SECONDS * (steps - 1 - step))
        for sensor_index, sensor in enumerate(sensors):
            row: dict[str, object] = {
                EVENT_TIMESTAMP_COLUMN: moment,
                CREATED_TIMESTAMP_COLUMN: moment,
                "sensor_id": sensor,
                "fault_type": 0,
                "timestep": step,
            }
            for feature_index, name in enumerate(feature_names):
                row[name] = 1000.0 * (sensor_index + 1) + feature_index - 100.0 * (steps - 1 - step)
            rows.append(row)

    frame = pd.DataFrame(rows)
    latest = frame[EVENT_TIMESTAMP_COLUMN] == last_event
    for sensor, feature in nan_cells:
        frame.loc[latest & (frame["sensor_id"] == sensor), feature] = np.nan
    return frame


def build_store(
    root: Path,
    params: FeatureStoreParams,
    frame: pd.DataFrame,
    first_event: datetime,
    last_event: datetime,
    feature_params: FeatureParams,
    config_text: str | None = None,
) -> FeatureStore:
    """Create, register and materialize a feature store under root.

    Args:
        root: Directory that holds the config, registry, online store and source.
        params: Store parameters (project, TTL, source path).
        frame: Offline source to write.
        first_event: Start of the materialization window.
        last_event: End of the materialization window.
        feature_params: Resolved features: section.
        config_text: feature_store.yaml content. Defaults to the SQLite setup.

    Returns:
        The materialized store.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / CONFIG_NAME).write_text(
        config_text if config_text is not None else local_config_text(params.project),
        encoding="utf-8",
    )
    frame.to_parquet(root / SOURCE_NAME, index=False)

    store = FeatureStore(repo_path=str(root), fs_yaml_file=root / CONFIG_NAME)
    apply_definitions(store, build_definitions(params, feature_params))
    store.materialize(first_event - timedelta(days=1), last_event + timedelta(days=1))
    return store
