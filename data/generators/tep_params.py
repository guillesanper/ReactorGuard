"""Typed access to the tep: section of params.yaml.

Existe para que los `params:` declarados en dvc.yaml sean reales. Si los stages
declarasen depender de params.yaml mientras el codigo mantiene los valores como
constantes de modulo, DVC invalidaria el stage al cambiar un parametro que
ningun ejecutable llega a leer: reproducibilidad aparente, no efectiva.

Todo lo que dvc.yaml declara bajo `params: - tep` se lee desde aqui.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "tep"


@dataclass(frozen=True)
class TEPParams:
    """Resolved configuration for the TEP data pipeline stages.

    Attributes:
        raw_dir: Directory holding the downloaded d00.dat .. d21.dat files.
        processed_dir: Root of the fault_type-partitioned Parquet output.
        reports_dir: Directory for the exploration report artefacts.
        plant_id: Plant identifier stamped on every SensorReading.
        start_time: UTC origin for the synthetic timestamp series.
        sample_interval_minutes: Spacing between consecutive TEP samples.
        adc_scale_max: Denominator used to scale values into 16-bit ADC counts.
        calibration_date: Calibration date recorded in SensorMetadata.
        last_maintenance_date: Maintenance date recorded in SensorMetadata.
        drift_coefficient: Drift coefficient recorded in SensorMetadata.
    """

    raw_dir: Path
    processed_dir: Path
    reports_dir: Path
    plant_id: str
    start_time: datetime
    sample_interval_minutes: int
    adc_scale_max: float
    calibration_date: date
    last_maintenance_date: date
    drift_coefficient: float


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed tep: mapping.
        key: Key that must be present.

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
    return section[key]


def _as_date(value: object) -> date:
    """Coerce a YAML scalar to a date.

    PyYAML already parses unquoted ISO dates into date objects; quoted ones
    arrive as strings.

    Args:
        value: A date instance or an ISO-8601 date string.

    Returns:
        The corresponding date.

    Raises:
        TypeError: If value is neither a date nor a string.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"Expected a date or ISO date string, got {type(value).__name__}.")


def load_tep_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> TEPParams:
    """Load and validate the tep: section of a params.yaml file.

    Args:
        params_path: Path to the params file. Defaults to params.yaml at the
            repository root.

    Returns:
        A fully resolved TEPParams instance.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the tep: section or a required key is missing.
        ValueError: If start_time is not a valid ISO-8601 timestamp.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    section = document[_SECTION]

    raw_start = _require(section, "start_time")
    start_time = (
        raw_start
        if isinstance(raw_start, datetime)
        else datetime.fromisoformat(str(raw_start))
    )

    return TEPParams(
        raw_dir=Path(str(_require(section, "raw_dir"))),
        processed_dir=Path(str(_require(section, "processed_dir"))),
        reports_dir=Path(str(_require(section, "reports_dir"))),
        plant_id=str(_require(section, "plant_id")),
        start_time=start_time,
        sample_interval_minutes=int(_require(section, "sample_interval_minutes")),
        adc_scale_max=float(_require(section, "adc_scale_max")),
        calibration_date=_as_date(_require(section, "calibration_date")),
        last_maintenance_date=_as_date(_require(section, "last_maintenance_date")),
        drift_coefficient=float(_require(section, "drift_coefficient")),
    )
