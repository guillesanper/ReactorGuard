"""Typed access to the streamer: section of params.yaml.

Mismo patron que data/streaming/streaming_params.py: dataclass inmutable, `_require`
para claves obligatorias y validacion en la carga. Lo que ya declara la seccion
`tep:` NO se repite en `streamer:`: el intervalo entre muestras, el directorio del
parquet y los metadatos de planta se derivan de ella al cargar, de modo que cambiar
`tep.sample_interval_minutes` cambia el ritmo del streamer sin tocar nada mas.

Las variables de entorno STREAM_MODE, SPEED_MULTIPLIER y DATA_SOURCE sobrescriben
tres valores en el arranque del servicio (`apply_env_overrides`); el resto de la
configuracion vive solo en params.yaml.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from data.generators.tep_params import DEFAULT_PARAMS_PATH, TEPParams, load_tep_params
from data.streaming.streaming_params import StreamingParams

_SECTION = "streamer"
_E = TypeVar("_E", bound=StrEnum)

ENV_STREAM_MODE = "STREAM_MODE"
ENV_SPEED_MULTIPLIER = "SPEED_MULTIPLIER"
ENV_DATA_SOURCE = "DATA_SOURCE"

_SECONDS_PER_MINUTE = 60.0
# La espera se trocea en tramos para dar el latido: hacen falta al menos dos por
# ventana de caducidad, o un unico retraso del sleeper dejaria /health en 503.
_MIN_BEATS_PER_STALENESS = 2


class StreamMode(StrEnum):
    """How the streamer paces its output."""

    FAST = "fast"
    REALTIME = "realtime"


class DataSource(StrEnum):
    """Where the streamer reads the long-format readings from."""

    PARQUET = "parquet"
    GCS = "gcs"


@dataclass(frozen=True)
class StreamerParams:
    """Resolved configuration of the TEP streamer.

    Attributes:
        mode: FAST emits without pauses, REALTIME paces one batch per timestep.
        speed_multiplier: Time compression of REALTIME (2.0 halves the wait).
        loop: Whether to replay the runs forever.
        data_source: Backend the readings are read from.
        storage_prefix: Object prefix of the DVC layout inside the raw bucket.
        flush_every_timesteps: FAST mode flush cadence, in timesteps.
        wait_slice_s: Longest single sleep of the REALTIME wait.
        client_id: Client id of the producer.
        sample_interval_s: Seconds between TEP samples (tep.sample_interval_minutes).
        parquet_dir: Root of the fault_type partitions (tep.processed_dir).
        calibration_date: Calibration date stamped on every rebuilt reading.
        last_maintenance: Maintenance date stamped on every rebuilt reading.
        drift_coefficient: Drift coefficient stamped on every rebuilt reading.
    """

    mode: StreamMode
    speed_multiplier: float
    loop: bool
    data_source: DataSource
    storage_prefix: str
    flush_every_timesteps: int
    wait_slice_s: float
    client_id: str
    sample_interval_s: float
    parquet_dir: Path
    calibration_date: date
    last_maintenance: date
    drift_coefficient: float

    @property
    def step_interval_s(self) -> float:
        """Return the REALTIME wait between two consecutive timestep batches."""
        return self.sample_interval_s / self.speed_multiplier


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed streamer: mapping.
        key: Key that must be present.

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
    return section[key]


def _positive_float(value: object, name: str) -> float:
    """Coerce a value to a finite, strictly positive float.

    Args:
        value: Raw value (number or numeric string).
        name: Label used in the error message.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not numeric, not finite or not positive.
    """
    try:
        number = float(str(value))
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}.") from None
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}.")
    return number


def _enum_value(enum_type: type[_E], value: object, name: str) -> _E:
    """Parse a closed-set string into an enum member, case-insensitively.

    Args:
        enum_type: The StrEnum holding the allowed values.
        value: Raw value to parse.
        name: Label used in the error message.

    Returns:
        The matching member.

    Raises:
        ValueError: If the value is not one of the members.
    """
    text = str(value).strip().lower()
    try:
        return enum_type(text)
    except ValueError:
        allowed = sorted(member.value for member in enum_type)
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}.") from None


def _positive_int(section: dict[str, Any], key: str) -> int:
    """Read an integer that must be strictly positive.

    Args:
        section: The parsed streamer: mapping.
        key: Key to read.

    Returns:
        The value as an int.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not a positive integer.
    """
    raw = _require(section, key)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be an integer > 0, got {raw!r}.")
    return raw


def _non_empty_str(section: dict[str, Any], key: str) -> str:
    """Read a string that must not be blank.

    Args:
        section: The parsed streamer: mapping.
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


def _strict_bool(section: dict[str, Any], key: str) -> bool:
    """Read a YAML boolean, refusing look-alikes such as "false" or 0.

    Args:
        section: The parsed streamer: mapping.
        key: Key to read.

    Returns:
        The boolean.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not a YAML boolean.
    """
    raw = _require(section, key)
    if not isinstance(raw, bool):
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be true or false, got {raw!r}.")
    return raw


def _build(section: dict[str, Any], tep: TEPParams) -> StreamerParams:
    """Combine the streamer: section with the tep: values it derives from.

    Args:
        section: The parsed streamer: mapping.
        tep: Resolved tep: parameters.

    Returns:
        The resolved parameters.

    Raises:
        KeyError: If a required key is missing.
        ValueError: If a value is out of range.
    """
    if tep.sample_interval_minutes <= 0:
        raise ValueError(
            f"params.yaml: 'tep.sample_interval_minutes' must be > 0, "
            f"got {tep.sample_interval_minutes}."
        )
    return StreamerParams(
        mode=_enum_value(StreamMode, _require(section, "mode"), f"{_SECTION}.mode"),
        speed_multiplier=_positive_float(
            _require(section, "speed_multiplier"), f"{_SECTION}.speed_multiplier"
        ),
        loop=_strict_bool(section, "loop"),
        data_source=_enum_value(
            DataSource, _require(section, "data_source"), f"{_SECTION}.data_source"
        ),
        storage_prefix=str(_require(section, "storage_prefix")).strip().strip("/"),
        flush_every_timesteps=_positive_int(section, "flush_every_timesteps"),
        wait_slice_s=_positive_float(_require(section, "wait_slice_s"), f"{_SECTION}.wait_slice_s"),
        client_id=_non_empty_str(section, "client_id"),
        sample_interval_s=tep.sample_interval_minutes * _SECONDS_PER_MINUTE,
        parquet_dir=tep.processed_dir,
        calibration_date=tep.calibration_date,
        last_maintenance=tep.last_maintenance_date,
        drift_coefficient=tep.drift_coefficient,
    )


def load_streamer_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> StreamerParams:
    """Load and validate the streamer: section of a params.yaml file.

    Args:
        params_path: Path to the params file. Defaults to params.yaml at the
            repository root.

    Returns:
        A fully resolved StreamerParams instance.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the streamer: or tep: section or a required key is missing.
        ValueError: If a value is out of range.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    return _build(document[_SECTION], load_tep_params(path))


def apply_env_overrides(
    params: StreamerParams, env: Mapping[str, str] | None = None
) -> StreamerParams:
    """Override mode, speed and data source from the environment.

    Una variable vacia cuenta como ausente (un Deployment que declara la variable
    sin valor no debe cambiar la configuracion).

    Args:
        params: Parameters loaded from params.yaml.
        env: Variable mapping. Defaults to os.environ.

    Returns:
        A copy of params with the overrides applied.

    Raises:
        ValueError: If a variable holds an invalid value.
    """
    source = os.environ if env is None else env
    changes: dict[str, Any] = {}

    raw_mode = source.get(ENV_STREAM_MODE, "").strip()
    if raw_mode:
        changes["mode"] = _enum_value(StreamMode, raw_mode, ENV_STREAM_MODE)
    raw_speed = source.get(ENV_SPEED_MULTIPLIER, "").strip()
    if raw_speed:
        changes["speed_multiplier"] = _positive_float(raw_speed, ENV_SPEED_MULTIPLIER)
    raw_source = source.get(ENV_DATA_SOURCE, "").strip()
    if raw_source:
        changes["data_source"] = _enum_value(DataSource, raw_source, ENV_DATA_SOURCE)
    return replace(params, **changes)


def validate_against_streaming(params: StreamerParams, streaming: StreamingParams) -> None:
    """Check that the wait slice keeps /health alive.

    Args:
        params: Resolved streamer parameters.
        streaming: Resolved streaming parameters (heartbeat staleness).

    Raises:
        ValueError: If wait_slice_s leaves fewer than two heartbeats inside
            staleness_seconds.
    """
    if params.wait_slice_s * _MIN_BEATS_PER_STALENESS > streaming.staleness_seconds:
        raise ValueError(
            f"params.yaml: '{_SECTION}.wait_slice_s' ({params.wait_slice_s} s) must allow at "
            f"least {_MIN_BEATS_PER_STALENESS} heartbeats within 'streaming.staleness_seconds' "
            f"({streaming.staleness_seconds} s), or /health fails during a REALTIME wait."
        )
