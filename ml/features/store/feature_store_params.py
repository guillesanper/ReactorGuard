"""Typed access to the feature_store: section of params.yaml.

Mismo patron que data/storage/storage_params.py: dataclass inmutable, `_require`
para claves obligatorias y validacion en la carga.

EL TTL SE CALIBRA A LA CADENCIA DEL TEP, NO A LA DEL DISENO ORIGINAL. El TTL de
30 s que proponia el diseno original para el residual de Kalman asumia una muestra cada
1 s. Con una muestra nueva cada 180 s:

    t = 0          llega la lectura y se escribe la feature (TTL 30 s)
    t = 0 .. 30    la feature se sirve                       30 s utiles
    t = 30 .. 180  la feature esta caducada y no hay otra   150 s muertos
    t = 180        llega la siguiente

Es decir, la feature es nula el 150 / 180 = 83 % del tiempo. El servicio no
fallaria: recibiria nulos y o bien imputaria (fabricando la evidencia) o bien
descartaria la prediccion, en silencio en ambos casos. Por eso el TTL se expresa
en INTERVALOS DE MUESTREO (ttl_intervals) y los segundos se derivan de
tep.sample_interval_minutes: 3 intervalos x 180 s = 540 s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from data.generators.tep_params import load_tep_params

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "feature_store"

BACKEND_SQLITE = "sqlite"
BACKEND_REDIS = "redis"
ONLINE_BACKENDS = (BACKEND_SQLITE, BACKEND_REDIS)

# Con menos de 2 intervalos no queda margen: un solo segundo de retraso del
# productor ya cuenta como caducado (ver el calculo del 83 % arriba).
MIN_TTL_INTERVALS = 2

# Feast usa el nombre de proyecto como prefijo de las claves del almacen online.
_PROJECT_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class FeatureStoreParams:
    """Resolved configuration of the Feast feature store.

    Attributes:
        project: Feast project name.
        ttl_intervals: FeatureView TTL, in sampling intervals.
        run_gap_intervals: Gap between consecutive runs in the synthetic timeline of
            feast_source, in sampling intervals.
        online_backend: Online store backend, "sqlite" or "redis".
        source_path: Parquet file produced by the feast_source stage.
        local_config_path: feature_store.yaml of the SQLite (local) setup.
        cluster_config_path: feature_store.yaml of the Redis (cluster) setup.
        sample_interval_seconds: TEP sampling cadence, derived from
            tep.sample_interval_minutes.
    """

    project: str
    ttl_intervals: int
    run_gap_intervals: int
    online_backend: str
    source_path: Path
    local_config_path: Path
    cluster_config_path: Path
    sample_interval_seconds: int

    @property
    def ttl_seconds(self) -> int:
        """Return the FeatureView TTL in seconds (ttl_intervals x cadence)."""
        return self.ttl_intervals * self.sample_interval_seconds

    @property
    def config_path(self) -> Path:
        """Return the feature_store.yaml that matches online_backend."""
        if self.online_backend == BACKEND_REDIS:
            return self.cluster_config_path
        return self.local_config_path


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed feature_store: mapping.
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
        section: The parsed feature_store: mapping.
        key: Key to read.

    Returns:
        The value as an int.

    Raises:
        KeyError: If key is missing.
        TypeError: If the value is not an integer (a float would be truncated
            silently and a bool is almost certainly a mistake).
        ValueError: If the value is not positive.
    """
    raw = _require(section, key)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(
            f"params.yaml: '{_SECTION}.{key}' must be an integer number of sampling "
            f"intervals, got {raw!r}."
        )
    if raw <= 0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be > 0, got {raw}.")
    return raw


def _parse_ttl_intervals(section: dict[str, Any]) -> int:
    """Read and validate ttl_intervals.

    Args:
        section: The parsed feature_store: mapping.

    Returns:
        The TTL in sampling intervals.

    Raises:
        KeyError: If the key is missing.
        TypeError: If the value is not an integer.
        ValueError: If it is below MIN_TTL_INTERVALS.
    """
    intervals = _positive_int(section, "ttl_intervals")
    if intervals < MIN_TTL_INTERVALS:
        raise ValueError(
            f"params.yaml: '{_SECTION}.ttl_intervals' is {intervals}; a TTL of one "
            "sampling interval leaves no slack, so a producer delayed by a single "
            f"second makes the feature null. Minimum is {MIN_TTL_INTERVALS}."
        )
    return intervals


def _parse_backend(section: dict[str, Any]) -> str:
    """Read and validate online_backend.

    Args:
        section: The parsed feature_store: mapping.

    Returns:
        "sqlite" or "redis".

    Raises:
        KeyError: If the key is missing.
        ValueError: If the value is not a supported backend.
    """
    backend = str(_require(section, "online_backend"))
    if backend not in ONLINE_BACKENDS:
        raise ValueError(
            f"params.yaml: '{_SECTION}.online_backend' must be one of "
            f"{list(ONLINE_BACKENDS)}, got '{backend}'."
        )
    return backend


def _parse_project(section: dict[str, Any]) -> str:
    """Read and validate the Feast project name.

    Args:
        section: The parsed feature_store: mapping.

    Returns:
        The project name.

    Raises:
        KeyError: If the key is missing.
        ValueError: If the name is not lowercase letters, digits and underscores.
    """
    project = str(_require(section, "project"))
    if not _PROJECT_RE.match(project):
        raise ValueError(
            f"params.yaml: '{_SECTION}.project' must start with a lowercase letter "
            f"and contain only lowercase letters, digits and underscores, got "
            f"'{project}'."
        )
    return project


def load_feature_store_params(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> FeatureStoreParams:
    """Load and validate the feature_store: section of a params.yaml file.

    La cadencia se toma de la seccion tep: del mismo fichero, no se duplica.

    Args:
        params_path: Path to the params file. Defaults to params.yaml at the
            repository root.

    Returns:
        A fully resolved FeatureStoreParams instance.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the feature_store: or tep: section or a required key is missing.
        TypeError: If an interval count is not an integer.
        ValueError: If a value is out of range, or if run_gap_intervals does not
            exceed ttl_intervals.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    section = document[_SECTION]

    ttl_intervals = _parse_ttl_intervals(section)
    run_gap = _positive_int(section, "run_gap_intervals")
    if run_gap <= ttl_intervals:
        raise ValueError(
            f"params.yaml: '{_SECTION}.run_gap_intervals' ({run_gap}) must exceed "
            f"'{_SECTION}.ttl_intervals' ({ttl_intervals}); otherwise the tail of one "
            "run is still inside the TTL of the next run's first row and a "
            "point-in-time join would mix two fault classes."
        )

    interval_minutes = load_tep_params(path).sample_interval_minutes
    if interval_minutes <= 0:
        raise ValueError(
            f"params.yaml: 'tep.sample_interval_minutes' must be > 0, got "
            f"{interval_minutes}."
        )

    return FeatureStoreParams(
        project=_parse_project(section),
        ttl_intervals=ttl_intervals,
        run_gap_intervals=run_gap,
        online_backend=_parse_backend(section),
        source_path=Path(str(_require(section, "source_path"))),
        local_config_path=Path(str(_require(section, "local_config_path"))),
        cluster_config_path=Path(str(_require(section, "cluster_config_path"))),
        sample_interval_seconds=interval_minutes * 60,
    )
