"""Acceso tipado a la seccion simulation: de params.yaml.

Existe por la misma razon que `tep_params.py`: que los `params:` que `dvc.yaml`
declare para el stage `simulate` sean efectivos. Todo lo que el simulador lee de
`params.yaml` pasa por aqui, y un valor fuera de rango falla al cargar con un
mensaje que nombra la clave, no tres modulos mas abajo.

`fault_injection_rate` y `critical_min_fraction` se validan aqui pero los consume
el planificador de F3-M5; en F3-M3 todos los runs son de operacion normal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "simulation"

SEVERITY_KEYS: tuple[str, ...] = ("critical", "high", "medium", "low")
"""Niveles de severidad del TDD 2.2, en orden decreciente."""

_WEIGHT_SUM_TOLERANCE = 1.0e-9
_MAX_LOAD_FRACTION = 1.5
"""Techo de carga que acepta `ReactorModel` (ver thermal_model._MAX_LOAD)."""


@dataclass(frozen=True)
class SimulationParams:
    """Configuracion resuelta del simulador.

    Attributes:
        n_samples: Pasos de tiempo totales a generar (no lecturas). Se redondea
            hacia arriba a runs completos.
        fault_injection_rate: Fraccion de lecturas etiquetadas como afectadas.
        critical_min_fraction: Fraccion minima de lecturas CRITICAL.
        openmc_seed: Semilla maestra; de ella cuelga un flujo por run.
        dt_seconds: Cadencia emitida [s].
        run_length_steps: Pasos por run.
        warmup_fraction: Fraccion inicial de cada run a carga constante.
        start_time: Instante (UTC) del primer paso del primer run.
        plant_id: Identificador de planta que llevan las lecturas.
        severity_weights: Reparto de la cuota etiquetada por severidad (suma 1).
        load_min_fraction: Carga minima de la operacion normal.
        load_max_fraction: Carga maxima de la operacion normal.
        max_ramp_percent_per_min: Rampa maxima de carga [% del nominal por min].
        load_hold_minutes_min: Minimo de minutos a carga constante entre rampas.
        load_hold_minutes_max: Maximo de minutos a carga constante entre rampas.
        output_dir: Raiz del dataset (`fault_type=<escenario>/seed=<n>/...`).
        metrics_path: Fichero de metricas (lo escribe F3-M6).
        plant_parameters_path: `configs/plant_parameters.yaml`.
        neutronics_path: `configs/neutronics_coefficients.yaml`.
        sensors_path: `configs/sim_sensors.yaml`.
        spans_path: `configs/sim_sensor_spans.yaml`.
        water_table_path: `configs/water_table.yaml`.
        calibration_date: Fecha de calibracion de SensorMetadata.
        last_maintenance_date: Fecha de mantenimiento de SensorMetadata.
        drift_coefficient: Coeficiente de deriva de SensorMetadata.
    """

    n_samples: int
    fault_injection_rate: float
    critical_min_fraction: float
    openmc_seed: int
    dt_seconds: float
    run_length_steps: int
    warmup_fraction: float
    start_time: datetime
    plant_id: str
    severity_weights: dict[str, float]
    load_min_fraction: float
    load_max_fraction: float
    max_ramp_percent_per_min: float
    load_hold_minutes_min: float
    load_hold_minutes_max: float
    output_dir: Path
    metrics_path: Path
    plant_parameters_path: Path
    neutronics_path: Path
    sensors_path: Path
    spans_path: Path
    water_table_path: Path
    calibration_date: date
    last_maintenance_date: date
    drift_coefficient: float

    @property
    def n_runs(self) -> int:
        """Return the number of runs: ceil(n_samples / run_length_steps)."""
        return math.ceil(self.n_samples / self.run_length_steps)

    @property
    def total_steps(self) -> int:
        """Return the steps actually generated (n_runs whole runs)."""
        return self.n_runs * self.run_length_steps

    @property
    def warmup_steps(self) -> int:
        """Return the leading steps of each run held at constant load."""
        return round(self.run_length_steps * self.warmup_fraction)

    @property
    def run_duration_s(self) -> float:
        """Return the simulated time one run spans [s]."""
        return self.run_length_steps * self.dt_seconds


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed simulation: mapping.
        key: Key that must be present.

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
    return section[key]


def _number(section: dict[str, Any], key: str) -> float:
    """Return a finite float.

    Raises:
        KeyError: If the key is missing.
        ValueError: If the value is not a finite number.
    """
    raw = _require(section, key)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be a number, got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be finite, got {raw!r}.")
    return value


def _positive(section: dict[str, Any], key: str) -> float:
    """Return a strictly positive finite float."""
    value = _number(section, key)
    if value <= 0.0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be > 0, got {value}.")
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    """Return a strictly positive integer (a float with integral value is accepted)."""
    value = _number(section, key)
    if value <= 0.0 or value != int(value):
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be a positive integer, got {value}."
        )
    return int(value)


def _as_date(key: str, value: object) -> date:
    """Coerce a YAML scalar to a date.

    Raises:
        TypeError: If value is neither a date nor a string.
        ValueError: If a string is not an ISO date.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"params.yaml: '{_SECTION}.{key}' is not an ISO date: {value!r}."
            ) from exc
    raise TypeError(
        f"params.yaml: '{_SECTION}.{key}' must be a date or ISO string, got "
        f"{type(value).__name__}."
    )


def _as_start_time(value: object) -> datetime:
    """Coerce a YAML scalar to a timezone-aware datetime.

    Raises:
        TypeError: If value is neither a datetime nor a string.
        ValueError: If it is not ISO-8601 or has no timezone.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"params.yaml: '{_SECTION}.start_time' is not ISO-8601: {value!r}."
            ) from exc
    else:
        raise TypeError(
            f"params.yaml: '{_SECTION}.start_time' must be a datetime or ISO string, got "
            f"{type(value).__name__}."
        )
    if parsed.tzinfo is None:
        raise ValueError(
            f"params.yaml: '{_SECTION}.start_time' needs a timezone (e.g. +00:00), got {value!r}."
        )
    return parsed


def _severity_weights(section: dict[str, Any]) -> dict[str, float]:
    """Return the severity weights, validated.

    Raises:
        KeyError: If the key is missing.
        TypeError: If it is not a mapping.
        ValueError: If the keys are not the four severities, a weight is negative
            or the weights do not sum to 1.
    """
    raw = _require(section, "severity_weights")
    if not isinstance(raw, dict):
        raise TypeError(f"params.yaml: '{_SECTION}.severity_weights' must be a mapping.")
    if set(raw) != set(SEVERITY_KEYS):
        raise ValueError(
            f"params.yaml: '{_SECTION}.severity_weights' needs exactly the keys "
            f"{list(SEVERITY_KEYS)}, got {sorted(map(str, raw))}."
        )
    weights = {key: float(raw[key]) for key in SEVERITY_KEYS}
    if any(not math.isfinite(w) or w < 0.0 for w in weights.values()):
        raise ValueError(
            f"params.yaml: '{_SECTION}.severity_weights' must be finite and >= 0, got {weights}."
        )
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"params.yaml: '{_SECTION}.severity_weights' must sum to 1, got {total}."
        )
    return weights


def load_simulation_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> SimulationParams:
    """Load and validate the simulation: section of a params.yaml file.

    Args:
        params_path: Path to the params file. Defaults to params.yaml at the
            repository root.

    Returns:
        A fully resolved SimulationParams.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the simulation: section or a required key is missing.
        TypeError: If a block has the wrong YAML type.
        ValueError: If a value is out of range: rate not in (0, 1), weights not
            summing to 1, critical_min_fraction above fault_injection_rate,
            non-positive lengths, a naive start_time or inconsistent load limits.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    section = document[_SECTION]
    if not isinstance(section, dict):
        raise TypeError(f"params.yaml: '{_SECTION}:' must be a mapping.")

    rate = _number(section, "fault_injection_rate")
    if not 0.0 < rate < 1.0:
        raise ValueError(
            f"params.yaml: '{_SECTION}.fault_injection_rate' must be in (0, 1), got {rate}."
        )
    critical_min = _number(section, "critical_min_fraction")
    if not 0.0 < critical_min <= rate:
        raise ValueError(
            f"params.yaml: '{_SECTION}.critical_min_fraction' must be in (0, "
            f"fault_injection_rate={rate}], got {critical_min}."
        )
    warmup = _number(section, "warmup_fraction")
    if not 0.0 <= warmup < 1.0:
        raise ValueError(
            f"params.yaml: '{_SECTION}.warmup_fraction' must be in [0, 1), got {warmup}."
        )
    load_min = _positive(section, "load_min_fraction")
    load_max = _positive(section, "load_max_fraction")
    if not load_min < load_max <= _MAX_LOAD_FRACTION:
        raise ValueError(
            f"params.yaml: '{_SECTION}' needs load_min_fraction < load_max_fraction <= "
            f"{_MAX_LOAD_FRACTION}, got {load_min} and {load_max}."
        )
    hold_min = _positive(section, "load_hold_minutes_min")
    hold_max = _positive(section, "load_hold_minutes_max")
    if hold_min > hold_max:
        raise ValueError(
            f"params.yaml: '{_SECTION}.load_hold_minutes_min' ({hold_min}) exceeds "
            f"load_hold_minutes_max ({hold_max})."
        )

    return SimulationParams(
        n_samples=_positive_int(section, "n_samples"),
        fault_injection_rate=rate,
        critical_min_fraction=critical_min,
        openmc_seed=int(_number(section, "openmc_seed")),
        dt_seconds=_positive(section, "dt_seconds"),
        run_length_steps=_positive_int(section, "run_length_steps"),
        warmup_fraction=warmup,
        start_time=_as_start_time(_require(section, "start_time")),
        plant_id=str(_require(section, "plant_id")),
        severity_weights=_severity_weights(section),
        load_min_fraction=load_min,
        load_max_fraction=load_max,
        max_ramp_percent_per_min=_positive(section, "max_ramp_percent_per_min"),
        load_hold_minutes_min=hold_min,
        load_hold_minutes_max=hold_max,
        output_dir=Path(str(_require(section, "output_dir"))),
        metrics_path=Path(str(_require(section, "metrics_path"))),
        plant_parameters_path=Path(str(_require(section, "plant_parameters_path"))),
        neutronics_path=Path(str(_require(section, "neutronics_path"))),
        sensors_path=Path(str(_require(section, "sensors_path"))),
        spans_path=Path(str(_require(section, "spans_path"))),
        water_table_path=Path(str(_require(section, "water_table_path"))),
        calibration_date=_as_date("calibration_date", _require(section, "calibration_date")),
        last_maintenance_date=_as_date(
            "last_maintenance_date", _require(section, "last_maintenance_date")
        ),
        drift_coefficient=_number(section, "drift_coefficient"),
    )
