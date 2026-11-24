"""Acceso tipado a la seccion pinn: de params.yaml (Fase 4).

Existe por la misma razon que `data/simulation/simulation_params.py`: que los
`params:` que `dvc.yaml` declare para los stages de la PINN sean efectivos. Todo lo
que esos stages leen de la seccion `pinn:` pasa por aqui, y un valor fuera de rango
falla al cargar con un mensaje que nombra la clave completa.

Una vista de la misma seccion por stage, para que cada uno declare en
`dvc.yaml` exactamente las claves que lee:

- `PinnParams` (F4-M1): datos y reparto por run del stage `prepare_sim` (D4-7, D4-8).
- `PinnTrainParams` (F4-M3): red, optimizacion, curriculum, salidas y MLflow del
  stage `train`. Sustituye a las claves del andamiaje que D4-9 retiro de
  `training:`.
- `PinnPhysicsParams` (F4-M4): checkpoint, test y metricas de `physics_check` en el
  stage `evaluate`.
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


def _read_section(params_path: str | Path) -> dict[str, Any]:
    """Return the pinn: mapping of a params file.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the pinn: section is missing.
        TypeError: If the section is not a mapping.
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
    return section


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
    section = _read_section(params_path)
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


# ----------------------------------------------------------------------
# Stage train (F4-M3)
# ----------------------------------------------------------------------

MLFLOW_URI_SCHEME = "sqlite:///"
"""Unico backend admitido (D4-14): el de ficheros esta en mantenimiento en MLflow 3 (H3)."""


@dataclass(frozen=True)
class PinnTrainParams:
    """Configuracion resuelta del stage train (seccion pinn:).

    Attributes:
        data_dir: Directorio de `prepare_sim` del que se leen las particiones.
        seed: Semilla de los Generators (inicializacion y barajado) y del dropout.
        num_threads: Hilos intra-op de torch.
        hidden: Anchura de las capas ocultas.
        n_hidden: Capas ocultas.
        dropout: Probabilidad de dropout entre capas ocultas, en [0, 1).
        learning_rate: Tasa inicial de Adam.
        weight_decay: Decaimiento L2 de Adam (>= 0).
        batch_size: Pares por lote.
        max_epochs: Tope de epocas.
        early_stopping_patience: Epocas sin mejora de val tras el curriculum antes
            de parar.
        lr_patience: Epocas sin mejora antes de que ReduceLROnPlateau baje lr.
        lr_factor: Factor de ReduceLROnPlateau, en (0, 1).
        min_lr: Suelo de lr (> 0).
        lambda_energy: Peso objetivo del termino de energia (>= 0).
        lambda_mass: Peso objetivo del termino de masa (>= 0).
        curriculum_epochs: Epocas de la rampa lineal 0 -> lambda (0 = sin rampa).
        model_dir: Out de DVC con model.pt y model_card.json.
        train_metrics_path: JSON de metricas del stage.
        mlflow_tracking_uri: URI sqlite del store (fuera del hash de DVC).
        mlflow_experiment: Nombre del experimento de MLflow.
    """

    data_dir: Path
    seed: int
    num_threads: int
    hidden: int
    n_hidden: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    lr_patience: int
    lr_factor: float
    min_lr: float
    lambda_energy: float
    lambda_mass: float
    curriculum_epochs: int
    model_dir: Path
    train_metrics_path: Path
    mlflow_tracking_uri: str
    mlflow_experiment: str


def _number(section: dict[str, Any], key: str) -> float:
    """Return a finite real number (bool rejected).

    Raises:
        KeyError: If the key is missing.
        ValueError: If the value is not a finite number.
    """
    raw = _require(section, key)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be a number, got {raw!r}.")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be finite, got {raw!r}.")
    return value


def _positive(section: dict[str, Any], key: str) -> float:
    """Return a finite number > 0."""
    value = _number(section, key)
    if value <= 0.0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be > 0, got {value}.")
    return value


def _non_negative(section: dict[str, Any], key: str) -> float:
    """Return a finite number >= 0."""
    value = _number(section, key)
    if value < 0.0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be >= 0, got {value}.")
    return value


def _open_unit(section: dict[str, Any], key: str, *, closed_low: bool) -> float:
    """Return a number in [0, 1) (closed_low) or (0, 1)."""
    value = _number(section, key)
    low_ok = value >= 0.0 if closed_low else value > 0.0
    if not (low_ok and value < 1.0):
        interval = "[0, 1)" if closed_low else "(0, 1)"
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be in {interval}, got {value}.")
    return value


def _non_negative_int(section: dict[str, Any], key: str) -> int:
    """Return an integer >= 0 (a float with integral value is accepted)."""
    raw = _require(section, key)
    message = f"params.yaml: '{_SECTION}.{key}' must be a non-negative integer, got {raw!r}."
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise ValueError(message)
    if not math.isfinite(raw) or raw < 0 or raw != int(raw):
        raise ValueError(message)
    return int(raw)


def _text(section: dict[str, Any], key: str) -> str:
    """Return a non-empty string."""
    raw = _require(section, key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be a non-empty string, got {raw!r}."
        )
    return raw


def _check_not_nested(paths: dict[str, Path]) -> None:
    """Raise if one out path lies inside another (DVC outs cannot nest)."""
    for name, path in paths.items():
        for other_name, other in paths.items():
            if name != other_name and path.is_relative_to(other):
                raise ValueError(
                    f"params.yaml: '{_SECTION}.{name}' ({path.as_posix()}) lies inside "
                    f"'{_SECTION}.{other_name}' ({other.as_posix()}). Son outs distintos de DVC "
                    "y no pueden anidarse."
                )


def load_pinn_train_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> PinnTrainParams:
    """Load and validate the keys of the pinn: section that the train stage reads.

    Args:
        params_path: Path to the params file.

    Returns:
        A fully resolved PinnTrainParams.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the pinn: section or a required key is missing.
        TypeError: If the section is not a mapping.
        ValueError: If a value is out of range, two outs nest or the MLflow URI is
            not a sqlite URI.
    """
    section = _read_section(params_path)
    data_dir = _path(section, "data_dir")
    model_dir = _path(section, "model_dir")
    train_metrics_path = _path(section, "train_metrics_path")
    _check_not_nested(
        {"data_dir": data_dir, "model_dir": model_dir, "train_metrics_path": train_metrics_path}
    )
    uri = _text(section, "mlflow_tracking_uri")
    if not uri.startswith(MLFLOW_URI_SCHEME) or len(uri) == len(MLFLOW_URI_SCHEME):
        raise ValueError(
            f"params.yaml: '{_SECTION}.mlflow_tracking_uri' must be a {MLFLOW_URI_SCHEME}<path> "
            f"URI (D4-14), got {uri!r}."
        )
    return PinnTrainParams(
        data_dir=data_dir,
        seed=_non_negative_int(section, "seed"),
        num_threads=_positive_int(section, "num_threads"),
        hidden=_positive_int(section, "hidden"),
        n_hidden=_positive_int(section, "n_hidden"),
        dropout=_open_unit(section, "dropout", closed_low=True),
        learning_rate=_positive(section, "learning_rate"),
        weight_decay=_non_negative(section, "weight_decay"),
        batch_size=_positive_int(section, "batch_size"),
        max_epochs=_positive_int(section, "max_epochs"),
        early_stopping_patience=_positive_int(section, "early_stopping_patience"),
        lr_patience=_non_negative_int(section, "lr_patience"),
        lr_factor=_open_unit(section, "lr_factor", closed_low=False),
        min_lr=_positive(section, "min_lr"),
        lambda_energy=_non_negative(section, "lambda_energy"),
        lambda_mass=_non_negative(section, "lambda_mass"),
        curriculum_epochs=_non_negative_int(section, "curriculum_epochs"),
        model_dir=model_dir,
        train_metrics_path=train_metrics_path,
        mlflow_tracking_uri=uri,
        mlflow_experiment=_text(section, "mlflow_experiment"),
    )


# ----------------------------------------------------------------------
# Stage evaluate: validacion fisica (F4-M4)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PinnPhysicsParams:
    """Configuracion resuelta de `physics_check` en el stage evaluate (seccion pinn:).

    Attributes:
        data_dir: Directorio de `prepare_sim` del que se lee test.
        model_dir: Directorio del checkpoint (`model.pt`) del stage train.
        num_threads: Hilos intra-op de torch durante la prediccion.
        physics_metrics_path: JSON de satisfaccion fisica (D4-12).
    """

    data_dir: Path
    model_dir: Path
    num_threads: int
    physics_metrics_path: Path


def load_pinn_physics_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> PinnPhysicsParams:
    """Load and validate the keys of the pinn: section that physics_check reads.

    Args:
        params_path: Path to the params file.

    Returns:
        A fully resolved PinnPhysicsParams.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the pinn: section or a required key is missing.
        TypeError: If the section is not a mapping.
        ValueError: If a value is out of range or the metrics file would live
            inside the data or the model directory.
    """
    section = _read_section(params_path)
    data_dir = _path(section, "data_dir")
    model_dir = _path(section, "model_dir")
    physics_metrics_path = _path(section, "physics_metrics_path")
    _check_not_nested(
        {
            "data_dir": data_dir,
            "model_dir": model_dir,
            "physics_metrics_path": physics_metrics_path,
        }
    )
    return PinnPhysicsParams(
        data_dir=data_dir,
        model_dir=model_dir,
        num_threads=_positive_int(section, "num_threads"),
        physics_metrics_path=physics_metrics_path,
    )
