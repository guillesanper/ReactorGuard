"""Rejilla de lambdas de la PINN: lambda_energy x lambda_mass (F4-M5, T7.1, T7.2).

Fuera de DVC (coste y MLflow): se ejecuta a mano y el ganador se copia a mano a
`pinn.lambda_energy` y `pinn.lambda_mass` de `params.yaml`, de modo que el cambio
queda visible en el diff y el stage `train` entrena con el. En CI no se ejecuta.

Cada combinacion entrena una red con las claves del stage train (`train_model`),
sustituyendo solo los dos lambdas, sobre los mismos datos (leidos una vez), y se
registra en MLflow como un run propio con el tag `grid`. No se guarda ningun
checkpoint: de cada run quedan sus metricas en MLflow y una entrada en el
manifiesto `pinn.grid_metrics_path`.

Satisfaccion en val: la misma definicion que el criterio 1 en test
(`ml.validation.physics_check.physics_document` sobre las predicciones de
val.parquet desde x(t-1) medido, reglas votadas por run).

Regla de seleccion (decision del usuario al abrir F4-M5):

1. Admisibles: combinaciones con satisfaccion en val >= `grid_min_val_satisfaction`.
2. Provisional: la admisible de menor `val_data` (empate: orden de la rejilla).
3. Ruido entre semillas: la combinacion sin fisica (0, 0) y la provisional se
   repiten con las semillas seed + 1, ..., seed + grid_seed_repeats - 1 (la semilla
   seed ya esta en la rejilla). La tolerancia es la mayor desviacion tipica
   muestral de `val_data` entre semillas de esas dos combinaciones.
4. Candidatas: admisibles con `val_data` <= minimo + tolerancia.
5. Ganadora: la candidata de menor `val_energy + val_mass` (empate: menor
   `val_data`, despues orden de la rejilla).

Todas las cifras de la seleccion son las de la semilla `pinn.seed` (las de la
rejilla); las repeticiones solo fijan la tolerancia.

El manifiesto no lleva reloj ni rutas absolutas y sus claves van ordenadas; los
tiempos de pared y el host van solo a MLflow y al log.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from data.simulation.physics_constraints import ConstraintLimits
from data.simulation.plant_params import load_plant_params
from data.simulation.sensor_catalog import load_sensor_catalog
from data.simulation.simulation_params import load_simulation_params
from ml.training.checkpoint import LoadedModel, write_json
from ml.training.pinn_params import (
    DEFAULT_PARAMS_PATH,
    PinnGridParams,
    PinnTrainParams,
    load_pinn_grid_params,
)
from ml.training.train import (
    MlflowRunLogger,
    NullRunLogger,
    RunLogger,
    TrainingData,
    load_training_data,
    train_model,
)
from ml.validation import physics_check

_LOG = logging.getLogger(__name__)

MANIFEST_VERSION = 1
"""Version del manifiesto de la rejilla."""

GRID_TAG = "grid"
"""Tag de MLflow que marca los runs de la rejilla."""

GRID_NAME = "pinn-lambda"
"""Valor del tag `grid`."""

ROLE_GRID = "grid"
"""Run de la rejilla (semilla pinn.seed)."""

ROLE_SEED_REPEAT = "seed_repeat"
"""Repeticion con otra semilla para medir el ruido entre semillas."""

VAL_SPLIT = "val"
"""Particion de la satisfaccion de la seleccion (test queda para el criterio 1)."""


# ----------------------------------------------------------------------
# Registro
# ----------------------------------------------------------------------


class GridRunLogger(RunLogger, Protocol):
    """RunLogger de un run de la rejilla: identificador y cierre."""

    @property
    def run_id(self) -> str | None:
        """Return the MLflow run id (None without MLflow)."""

    def finish(self, status: str) -> None:
        """Close the run with a terminal status (FINISHED or FAILED)."""


class NullGridRunLogger(NullRunLogger):
    """GridRunLogger que no registra nada (tests y uso sin MLflow)."""

    @property
    def run_id(self) -> str | None:
        """Return None: no run exists."""
        return None

    def finish(self, status: str) -> None:
        """Ignore the status."""


LoggerFactory = Callable[[str], GridRunLogger]
"""Crea el logger de un run a partir de su nombre."""


def mlflow_factory(tracking_uri: str, experiment: str) -> LoggerFactory:
    """Return a factory of MlflowRunLogger on one store and experiment."""

    def create(run_name: str) -> GridRunLogger:
        return MlflowRunLogger(tracking_uri, experiment, run_name)

    return create


def null_factory(run_name: str) -> GridRunLogger:
    """Return a NullGridRunLogger (ignores the run name)."""
    return NullGridRunLogger()


# ----------------------------------------------------------------------
# Datos de la satisfaccion en val
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ValPhysics:
    """Lo que necesita la satisfaccion fisica en val, ya resuelto.

    Attributes:
        frame: val.parquet con las columnas de physics_check.
        roles: Rol -> tags.
        limits: Umbrales de las reglas.
    """

    frame: pd.DataFrame
    roles: Mapping[str, Sequence[str]]
    limits: ConstraintLimits


def load_val_physics(params_path: str | Path, train: PinnTrainParams) -> ValPhysics:
    """Read val.parquet with the columns of physics_check and the rule limits.

    Raises:
        FileNotFoundError: If val.parquet is missing.
    """
    simulation = load_simulation_params(params_path)
    catalog = load_sensor_catalog(simulation.sensors_path)
    path = train.data_dir / f"{VAL_SPLIT}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.as_posix()} not found: run the prepare_sim stage.")
    frame = pd.read_parquet(path, columns=physics_check.partition_columns(catalog.tags))
    limits = ConstraintLimits.from_plant(load_plant_params(simulation.plant_parameters_path))
    return ValPhysics(frame=frame, roles=catalog.roles, limits=limits)


def val_satisfaction(
    loaded: LoadedModel, data: TrainingData, val: ValPhysics, num_threads: int
) -> float | None:
    """Return the physics satisfaction of the predictions of val (criterion 1 on val)."""
    with physics_check.torch_threads(num_threads):
        values = physics_check.predict_next(loaded, val.frame, val.roles)
    document = physics_check.physics_document(
        val.frame, values, loaded.stats.tags, val.roles, data.table, val.limits, {}
    )
    satisfaction = document["satisfaction"]
    return None if satisfaction is None else float(satisfaction)


# ----------------------------------------------------------------------
# Un run
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GridPoint:
    """Una combinacion de la rejilla.

    Attributes:
        lambda_energy: Peso del termino de energia.
        lambda_mass: Peso del termino de masa.
    """

    lambda_energy: float
    lambda_mass: float

    @property
    def label(self) -> str:
        """Return a short, stable name (e1000_m0)."""
        return f"e{self.lambda_energy:g}_m{self.lambda_mass:g}"


@dataclass(frozen=True)
class RunEntry:
    """Lo que queda de un run de la rejilla.

    Attributes:
        point: Combinacion.
        seed: Semilla del run.
        role: ROLE_GRID o ROLE_SEED_REPEAT.
        run_id: Run de MLflow (None sin MLflow).
        best_epoch: Mejor epoca.
        epochs_run: Epocas ejecutadas.
        val_data: MSE de datos en val en la mejor epoca.
        val_energy: Termino de energia en val.
        val_mass: Termino de masa en val.
        physics_share: Peso de la fisica en val (lambdas de la combinacion).
        val_satisfaction: Satisfaccion fisica de las predicciones de val.
        seconds: Tiempo de pared del entrenamiento (fuera del manifiesto).
    """

    point: GridPoint
    seed: int
    role: str
    run_id: str | None
    best_epoch: int
    epochs_run: int
    val_data: float
    val_energy: float
    val_mass: float
    physics_share: float
    val_satisfaction: float | None
    seconds: float

    @property
    def val_physics(self) -> float:
        """Return val_energy + val_mass (desempate de la seleccion)."""
        return self.val_energy + self.val_mass


def run_point(
    point: GridPoint,
    seed: int,
    role: str,
    data: TrainingData,
    val: ValPhysics,
    params: PinnTrainParams,
    factory: LoggerFactory,
) -> RunEntry:
    """Train one combination with one seed, measure val and close its run.

    Args:
        point: Combinacion.
        seed: Semilla del run.
        role: ROLE_GRID o ROLE_SEED_REPEAT.
        data: Datos de entrenamiento.
        val: Datos de la satisfaccion en val.
        params: Claves del stage train (los lambdas y la semilla se sustituyen).
        factory: Creador del logger del run.

    Returns:
        The RunEntry.
    """
    run_params = dataclasses.replace(
        params, lambda_energy=point.lambda_energy, lambda_mass=point.lambda_mass, seed=seed
    )
    logger = factory(f"grid-{point.label}-s{seed}")
    try:
        trained = train_model(data, run_params, logger)
        satisfaction = val_satisfaction(
            LoadedModel(model=trained.model, stats=data.stats), data, val, params.num_threads
        )
        best = trained.result.best
        logger.log_summary(
            {
                **{f"best_{k}": v for k, v in best.metrics().items()},
                "best_epoch": float(best.epoch),
                "epochs_run": float(trained.result.epochs_run),
                "train_seconds": trained.seconds,
                "val_satisfaction": math.nan if satisfaction is None else satisfaction,
            },
            {
                GRID_TAG: GRID_NAME,
                "grid_point": point.label,
                "grid_role": role,
                "stopped_early": str(trained.result.stopped_early),
                "torch_version": torch.__version__,
                "host": platform.node(),
                "platform": platform.platform(),
            },
        )
    except BaseException:
        logger.finish("FAILED")
        raise
    logger.finish("FINISHED")
    _LOG.info(
        "%s seed %d (%s): %d epochs in %.1f s, val_data %.6g, val_energy %.3g, val_mass %.3g, "
        "physics share %.3g, val satisfaction %s",
        point.label,
        seed,
        role,
        trained.result.epochs_run,
        trained.seconds,
        best.val.data,
        best.val.energy,
        best.val.mass,
        best.physics_share,
        satisfaction,
    )
    return RunEntry(
        point=point,
        seed=seed,
        role=role,
        run_id=logger.run_id,
        best_epoch=best.epoch,
        epochs_run=trained.result.epochs_run,
        val_data=best.val.data,
        val_energy=best.val.energy,
        val_mass=best.val.mass,
        physics_share=best.physics_share,
        val_satisfaction=satisfaction,
        seconds=trained.seconds,
    )


# ----------------------------------------------------------------------
# Seleccion
# ----------------------------------------------------------------------


def grid_points(params: PinnGridParams) -> list[GridPoint]:
    """Return the combinations in grid order (lambda_energy outer, lambda_mass inner)."""
    return [GridPoint(e, m) for e in params.lambda_energy for m in params.lambda_mass]


def repeated_points(points: Sequence[GridPoint], first: RunEntry | None) -> list[GridPoint]:
    """Return the points repeated with other seeds: (0, 0) and the provisional winner.

    Solo se repiten combinaciones de la rejilla (asi cada una tiene la semilla base);
    sin admisibles se repite solo (0, 0).
    """
    wanted = [GridPoint(0.0, 0.0)]
    if first is not None:
        wanted.append(first.point)
    out: list[GridPoint] = []
    for point in wanted:
        if point in points and point not in out:
            out.append(point)
    return out


def admissible(entries: Sequence[RunEntry], min_satisfaction: float) -> list[RunEntry]:
    """Return the entries whose val satisfaction reaches the threshold, in order."""
    return [
        entry
        for entry in entries
        if entry.val_satisfaction is not None and entry.val_satisfaction >= min_satisfaction
    ]


def provisional(candidates: Sequence[RunEntry]) -> RunEntry | None:
    """Return the entry of minimum val_data (first in order on ties), None if empty."""
    if not candidates:
        return None
    return min(candidates, key=lambda entry: entry.val_data)


def seed_noise(entries: Sequence[RunEntry], point: GridPoint) -> float | None:
    """Return the sample standard deviation of val_data across the seeds of a point.

    Returns:
        The deviation (ddof = 1), or None with fewer than two runs of the point.
    """
    values = [entry.val_data for entry in entries if entry.point == point]
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


@dataclass(frozen=True)
class Selection:
    """Resultado de la regla de seleccion.

    Attributes:
        winner: Combinacion elegida (None si ninguna es admisible).
        provisional: Admisible de menor val_data.
        tolerance: Tolerancia de val_data (ruido entre semillas).
        noise: Desviacion de val_data entre semillas por combinacion repetida.
        candidates: Combinaciones dentro de la tolerancia, en orden de la rejilla.
        n_admissible: Combinaciones admisibles.
    """

    winner: RunEntry | None
    provisional: RunEntry | None
    tolerance: float
    noise: Mapping[str, float | None]
    candidates: Sequence[RunEntry]
    n_admissible: int


def select(
    grid: Sequence[RunEntry], repeats: Sequence[RunEntry], min_satisfaction: float
) -> Selection:
    """Apply the written selection rule (module docstring).

    Args:
        grid: Runs de la rejilla con la semilla pinn.seed, en orden de la rejilla.
        repeats: Repeticiones con otras semillas.
        min_satisfaction: Satisfaccion minima en val.

    Returns:
        The Selection.
    """
    allowed = admissible(grid, min_satisfaction)
    first = provisional(allowed)
    everything = [*grid, *repeats]
    repeated = sorted({entry.point for entry in repeats}, key=lambda p: p.label)
    noise = {point.label: seed_noise(everything, point) for point in repeated}
    tolerance = max((value for value in noise.values() if value is not None), default=0.0)
    if first is None:
        return Selection(None, None, tolerance, noise, [], 0)
    limit = first.val_data + tolerance
    candidates = [entry for entry in allowed if entry.val_data <= limit]
    winner = min(candidates, key=lambda entry: (entry.val_physics, entry.val_data))
    return Selection(winner, first, tolerance, noise, candidates, len(allowed))


# ----------------------------------------------------------------------
# Manifiesto
# ----------------------------------------------------------------------


def _round(value: float | None) -> float | None:
    """Return a float with 12 significant digits (None passes through)."""
    if value is None or not math.isfinite(value):
        return None
    return float(f"{value:.12g}")


def entry_document(entry: RunEntry) -> dict[str, Any]:
    """Return the manifest entry of one run (sin tiempos)."""
    return {
        "lambda_energy": entry.point.lambda_energy,
        "lambda_mass": entry.point.lambda_mass,
        "point": entry.point.label,
        "seed": entry.seed,
        "role": entry.role,
        "run_id": entry.run_id,
        "best_epoch": entry.best_epoch,
        "epochs_run": entry.epochs_run,
        "val_data": _round(entry.val_data),
        "val_energy": _round(entry.val_energy),
        "val_mass": _round(entry.val_mass),
        "physics_share": _round(entry.physics_share),
        "val_satisfaction": _round(entry.val_satisfaction),
    }


def manifest_document(
    params: PinnGridParams,
    grid: Sequence[RunEntry],
    repeats: Sequence[RunEntry],
    selection: Selection,
    digests: Mapping[str, str],
) -> dict[str, Any]:
    """Return the grid manifest (sin reloj ni rutas).

    Args:
        params: Configuracion de la rejilla.
        grid: Runs de la rejilla.
        repeats: Repeticiones de semilla.
        selection: Resultado de la seleccion.
        digests: sha256 de los ficheros de datos leidos, por nombre.

    Returns:
        The document written to `pinn.grid_metrics_path`.
    """
    winner = selection.winner
    return {
        "version": MANIFEST_VERSION,
        "grid": {
            "name": GRID_NAME,
            "lambda_energy": list(params.lambda_energy),
            "lambda_mass": list(params.lambda_mass),
            "seed": params.train.seed,
            "seed_repeats": params.seed_repeats,
            "max_epochs": params.train.max_epochs,
        },
        "rule": {
            "min_val_satisfaction": params.min_val_satisfaction,
            "primary": "min val_data among admissible",
            "tolerance": "max sample std of val_data across seeds",
            "tie_break": "min val_energy + val_mass, then val_data",
        },
        "inputs": dict(sorted(digests.items())),
        "runs": [entry_document(entry) for entry in [*grid, *repeats]],
        "n_runs": len(grid) + len(repeats),
        "selection": {
            "n_admissible": selection.n_admissible,
            "provisional": None
            if selection.provisional is None
            else selection.provisional.point.label,
            "seed_noise": {k: _round(v) for k, v in sorted(selection.noise.items())},
            "tolerance": _round(selection.tolerance),
            "candidates": [entry.point.label for entry in selection.candidates],
            "winner": None
            if winner is None
            else {
                "point": winner.point.label,
                "lambda_energy": winner.point.lambda_energy,
                "lambda_mass": winner.point.lambda_mass,
                "run_id": winner.run_id,
            },
        },
    }


# ----------------------------------------------------------------------
# Rejilla completa
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class GridOutcome:
    """Lo que deja una ejecucion de la rejilla.

    Attributes:
        grid: Runs de la rejilla.
        repeats: Repeticiones de semilla.
        selection: Resultado de la seleccion.
        document: Contenido del manifiesto.
        metrics_path: Manifiesto escrito.
    """

    grid: Sequence[RunEntry]
    repeats: Sequence[RunEntry]
    selection: Selection
    document: Mapping[str, Any]
    metrics_path: Path


def run_grid(
    params_path: str | Path, params: PinnGridParams, factory: LoggerFactory
) -> GridOutcome:
    """Train every combination, repeat two of them with other seeds and select.

    Args:
        params_path: Fichero de parametros (secciones pinn: y simulation:).
        params: Configuracion de la rejilla.
        factory: Creador del logger de cada run.

    Returns:
        The GridOutcome.
    """
    base = params.train
    data = load_training_data(params_path, base)
    val = load_val_physics(params_path, base)
    points = grid_points(params)
    grid = [run_point(point, base.seed, ROLE_GRID, data, val, base, factory) for point in points]
    repeated = repeated_points(points, provisional(admissible(grid, params.min_val_satisfaction)))
    repeats = [
        run_point(point, base.seed + k, ROLE_SEED_REPEAT, data, val, base, factory)
        for point in repeated
        for k in range(1, params.seed_repeats)
    ]
    selection = select(grid, repeats, params.min_val_satisfaction)
    document = manifest_document(params, grid, repeats, selection, data.digests)
    metrics_path = write_json(document, params.metrics_path)
    seconds = sum(entry.seconds for entry in [*grid, *repeats])
    if selection.winner is None:
        _LOG.warning("No combination reaches val satisfaction %s.", params.min_val_satisfaction)
    else:
        _LOG.info(
            "Grid of %d runs in %.0f s of training: winner %s (tolerance %.3g, %d candidates). "
            "Copy lambda_energy = %g and lambda_mass = %g to params.yaml by hand.",
            len(grid) + len(repeats),
            seconds,
            selection.winner.point.label,
            selection.tolerance,
            len(selection.candidates),
            selection.winner.point.lambda_energy,
            selection.winner.point.lambda_mass,
        )
    return GridOutcome(grid, repeats, selection, document, metrics_path)


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> GridOutcome:
    """Run the lambda grid end to end, logging every run to MLflow.

    Args:
        params_path: Fichero de parametros.

    Returns:
        The GridOutcome.
    """
    params = load_pinn_grid_params(params_path)
    factory = mlflow_factory(params.train.mlflow_tracking_uri, params.train.mlflow_experiment)
    return run_grid(params_path, params, factory)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
