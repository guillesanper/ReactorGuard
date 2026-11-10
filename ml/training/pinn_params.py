"""Acceso tipado a la seccion pinn: de params.yaml (Fase 4).

Existe por la misma razon que `data/simulation/simulation_params.py`: que los
`params:` que `dvc.yaml` declare para los stages de la PINN sean efectivos. Todo lo
que esos stages leen de la seccion `pinn:` pasa por aqui, y un valor fuera de rango
falla al cargar con un mensaje que nombra la clave completa.

F4-M1 solo resuelve las claves del stage `prepare_sim` (datos y reparto por run,
D4-7 y D4-8). Los hiperparametros del modelo y del entrenamiento entran en la
seccion cuando su consumidor exista (F4-M2 y F4-M3), no antes: una clave sin lector
es la deuda que D4-9 retira de `training:`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "pinn"


@dataclass(frozen=True)
class PinnParams:
    """Configuracion resuelta de la seccion pinn:.

    Attributes:
        data_dir: Directorio de salida de `prepare_sim` (out de DVC): un parquet
            ancho por particion y la asignacion de runs.
        metrics_path: JSON de metricas de `prepare_sim` (`dvc metrics show`).
        val_runs_per_scenario: Runs de cada escenario que van a validacion (los
            penultimos por `run_id`).
        test_runs_per_scenario: Runs de cada escenario que van a test (los ultimos
            por `run_id`).
    """

    data_dir: Path
    metrics_path: Path
    val_runs_per_scenario: int
    test_runs_per_scenario: int

    @property
    def min_runs_per_scenario(self) -> int:
        """Return the runs a scenario needs so that train keeps at least one."""
        return self.val_runs_per_scenario + self.test_runs_per_scenario + 1


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed pinn: mapping.
        key: Key that must be present.

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
    return section[key]


def _path(section: dict[str, Any], key: str) -> Path:
    """Return a non-empty relative or absolute path.

    Raises:
        KeyError: If the key is missing.
        ValueError: If the value is not a non-empty string.
    """
    raw = _require(section, key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be a non-empty path, got {raw!r}.")
    return Path(raw)


def _positive_int(section: dict[str, Any], key: str) -> int:
    """Return a strictly positive integer (a float with integral value is accepted).

    Raises:
        KeyError: If the key is missing.
        ValueError: If the value is not a positive integer.
    """
    raw = _require(section, key)
    if isinstance(raw, bool):
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be a positive integer, got {raw!r}."
        )
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be a positive integer, got {raw!r}."
        ) from exc
    if not math.isfinite(value) or value <= 0.0 or value != int(value):
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be a positive integer, got {raw!r}."
        )
    return int(value)


def load_pinn_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> PinnParams:
    """Load and validate the pinn: section of a params.yaml file.

    Args:
        params_path: Path to the params file. Defaults to params.yaml at the
            repository root.

    Returns:
        A fully resolved PinnParams.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the pinn: section or a required key is missing.
        TypeError: If the section is not a mapping.
        ValueError: If a value is out of range or the metrics file would live
            inside the data directory (two DVC outs cannot nest).
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

    data_dir = _path(section, "data_dir")
    metrics_path = _path(section, "metrics_path")
    if metrics_path.is_relative_to(data_dir):
        raise ValueError(
            f"params.yaml: '{_SECTION}.metrics_path' ({metrics_path.as_posix()}) lies inside "
            f"'{_SECTION}.data_dir' ({data_dir.as_posix()}). Son outs distintos de DVC y no "
            "pueden anidarse."
        )
    return PinnParams(
        data_dir=data_dir,
        metrics_path=metrics_path,
        val_runs_per_scenario=_positive_int(section, "val_runs_per_scenario"),
        test_runs_per_scenario=_positive_int(section, "test_runs_per_scenario"),
    )
