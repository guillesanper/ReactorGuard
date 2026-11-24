"""Validacion fisica de la PINN sobre test: satisfaccion de las reglas (F4-M4, T6.5).

Criterio 1 de la Fase 4 (D4-12): fraccion de instantes de test cuyo estado
PREDICHO, desnormalizado, no viola ninguna de las 7 reglas de
`data.simulation.physics_constraints` (votadas por mediana de canales, D4-13).

Que se predice (decision del usuario al abrir F4-M4):

- todos los instantes t de test que tienen t-1 en su run, etiquetados incluidos: la
  entrada es x(t-1) tal como se midio, con su contexto (D4-2), y la salida y(t) se
  desnormaliza con los estadisticos de train del checkpoint;
- el primer instante de cada run no tiene prediccion y cuenta como NO EVALUABLE
  (fuera del denominador, informado aparte).

Como se evaluan las reglas: por run (`evaluate_rules_detailed` exige un unico run y
un `DatetimeIndex`), sobre el frame ancho de las predicciones de ese run. Las dos
reglas con derivada temporal (dT/dt y la mascara cuasi-estacionaria de la masa) usan
por tanto la diferencia entre predicciones consecutivas y(t) - y(t-1), y no son
evaluables en el primer instante predicho de cada run. dT/dt cuenta como regla
evaluable aunque a 180 s no pueda dispararse (D4-12; minima derivada observable de
`min_observable_rate`).

Un instante predicho es EVALUADO si al menos una regla es evaluable en el, y
SATISFECHO si ninguna regla evaluable se viola. Satisfaccion = satisfechos /
evaluados. Se informa ademas el subconjunto normal (instantes sin etiqueta), cada
regla y cada escenario, y la misma metrica sobre la MEDICION de los mismos instantes
como referencia (la medicion normal viola 0 de 32.161, seccion 2.3 del plan).

El JSON no lleva reloj ni rutas, con claves ordenadas: es estable en bytes.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from data.simulation.dataset_builder import file_sha256
from data.simulation.physics_constraints import (
    RULE_NAMES,
    ConstraintLimits,
    RuleEvaluation,
    evaluate_rules_detailed,
)
from data.simulation.plant_params import load_plant_params
from data.simulation.sensor_catalog import load_sensor_catalog
from data.simulation.simulation_params import load_simulation_params
from data.simulation.water_properties import WaterTable, load_water_table
from ml.training.checkpoint import MODEL_FILE, LoadedModel, load_checkpoint, write_json
from ml.training.pinn_data import context_features
from ml.training.pinn_params import (
    DEFAULT_PARAMS_PATH,
    PinnPhysicsParams,
    load_pinn_physics_params,
)
from ml.training.sim_dataset import ASSIGNMENT_NAME, ID_COLUMNS, value_columns

_LOG = logging.getLogger(__name__)

METRICS_VERSION = 1
"""Version del JSON de metricas de physics_check."""

SPLIT = "test"
"""Particion evaluada (D4-12)."""

SATISFACTION_THRESHOLD = 0.99
"""Criterio 1 de la Fase 4: satisfaccion estrictamente mayor que este valor."""

RULE_MODE = "voted"
"""Reglas sobre la mediana de los canales de cada rol (D4-13; por canal en F4-M6)."""

FRAME_COLUMNS: tuple[str, ...] = (*ID_COLUMNS, "labelled")
"""Columnas no de sensor que se leen de test."""


def partition_columns(tags: Sequence[str]) -> list[str]:
    """Return the parquet columns physics_check reads for a sensor order."""
    return [*FRAME_COLUMNS, *value_columns(tags)]


# ----------------------------------------------------------------------
# Prediccion
# ----------------------------------------------------------------------


@contextmanager
def torch_threads(num_threads: int) -> Iterator[None]:
    """Fix the intra-op threads of torch and restore the previous value on exit."""
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def first_of_run(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """Return where each row is the first instant of its run in the frame."""
    runs = frame["run_id"].to_numpy()
    first = np.ones(len(frame), dtype=np.bool_)
    first[1:] = runs[1:] != runs[:-1]
    return first


def _zscore(
    values: NDArray[np.float64], mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """Return (values - mean) / std as float32 (computed in float64), as in training."""
    z = (values - np.asarray(mean, dtype=np.float64)) / np.asarray(std, dtype=np.float64)
    return torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))


def predict_next(
    loaded: LoadedModel, frame: pd.DataFrame, roles: Mapping[str, Sequence[str]]
) -> NDArray[np.float64]:
    """Predict every instant that has t-1 in its run, in engineering units.

    Args:
        loaded: Checkpoint cargado (red en modo evaluacion y z-score de train).
        frame: Frame ancho de una particion (instantes de cada run consecutivos y
            en orden, contiguos).
        roles: Rol -> tags (contexto de D4-2).

    Returns:
        Array (len(frame), n_sensors): fila t = y(t) predicho desde x(t-1) medido;
        NaN en el primer instante de cada run.

    Raises:
        ValueError: If the instants of a run are not consecutive and in order.
    """
    stats = loaded.stats
    tags = stats.tags
    context = context_features(frame, tags, roles)
    measured = frame[value_columns(tags)].to_numpy(dtype=np.float64)
    target = np.flatnonzero(~first_of_run(frame))
    origin = target - 1
    out = np.full((len(frame), len(tags)), np.nan, dtype=np.float64)
    if len(target) == 0:
        return out
    inputs = _zscore(
        np.concatenate([measured[origin], context[origin]], axis=1),
        stats.input_mean,
        stats.input_std,
    )
    with torch.no_grad():
        z = loaded.model(inputs).double().numpy()
    mean = np.asarray(stats.sensor_mean, dtype=np.float64)
    std = np.asarray(stats.sensor_std, dtype=np.float64)
    out[target] = z * std + mean
    return out


# ----------------------------------------------------------------------
# Reglas por run
# ----------------------------------------------------------------------


def evaluate_by_run(
    frame: pd.DataFrame,
    values: NDArray[np.float64],
    rows: NDArray[np.bool_],
    tags: Sequence[str],
    roles: Mapping[str, Sequence[str]],
    table: WaterTable,
    limits: ConstraintLimits,
) -> RuleEvaluation:
    """Apply the rules run by run to some rows of a value matrix.

    Cada run se pivota a un frame ancho (una columna por tag, indice
    `DatetimeIndex` con las marcas de tiempo de las filas elegidas) y se evalua por
    separado, de modo que las derivadas temporales nunca cruzan runs.

    Args:
        frame: Frame de la particion (identidad de cada fila).
        values: Matriz (len(frame), n_sensors) en unidades de ingenieria.
        rows: Filas que entran en la evaluacion (las demas quedan no evaluables).
        tags: Orden de las columnas de values.
        roles: Rol -> tags.
        table: Tabla del agua.
        limits: Umbrales de las reglas.

    Returns:
        A RuleEvaluation indexed by the row position in frame (False outside rows).
    """
    violations = np.zeros((len(frame), len(RULE_NAMES)), dtype=np.bool_)
    evaluable = np.zeros_like(violations)
    runs = frame["run_id"].to_numpy()
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    for run in pd.unique(runs[rows]):
        positions = np.flatnonzero(rows & (runs == run))
        wide = pd.DataFrame(values[positions], columns=list(tags), index=stamps[positions])
        result = evaluate_rules_detailed(wide, roles, table, limits)
        violations[positions] = result.violations[list(RULE_NAMES)].to_numpy(dtype=np.bool_)
        evaluable[positions] = result.evaluable[list(RULE_NAMES)].to_numpy(dtype=np.bool_)
    index = pd.RangeIndex(len(frame))
    return RuleEvaluation(
        violations=pd.DataFrame(violations, index=index, columns=list(RULE_NAMES)),
        evaluable=pd.DataFrame(evaluable, index=index, columns=list(RULE_NAMES)),
    )


# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------


def _ratio(numerator: int, denominator: int) -> float | None:
    """Return numerator / denominator with 12 significant digits (None if 0/0)."""
    if denominator == 0:
        return None
    return float(f"{numerator / denominator:.12g}")


def summarize(
    evaluation: RuleEvaluation, scope: NDArray[np.bool_], predicted: NDArray[np.bool_]
) -> dict[str, Any]:
    """Count satisfied, violating and non-evaluable instants of a scope.

    Args:
        evaluation: Resultado de `evaluate_by_run` sobre toda la particion.
        scope: Instantes del alcance (todos o los normales).
        predicted: Instantes con prediccion (los demas, primeros de run).

    Returns:
        The summary block of the metrics JSON.
    """
    violated = evaluation.violations.to_numpy(dtype=np.bool_) & evaluation.evaluable.to_numpy(
        dtype=np.bool_
    )
    evaluable = evaluation.evaluable.to_numpy(dtype=np.bool_)
    in_scope = scope & predicted
    evaluated = in_scope & evaluable.any(axis=1)
    violating = evaluated & violated.any(axis=1)
    n_evaluated = int(evaluated.sum())
    n_satisfied = n_evaluated - int(violating.sum())
    rules: dict[str, dict[str, Any]] = {}
    for position, rule in enumerate(RULE_NAMES):
        n_evaluable = int((in_scope & evaluable[:, position]).sum())
        n_violations = int((in_scope & violated[:, position]).sum())
        rules[rule] = {
            "n_evaluable": n_evaluable,
            "n_not_evaluated": int(in_scope.sum()) - n_evaluable,
            "n_violations": n_violations,
            "rate": _ratio(n_violations, n_evaluable),
        }
    return {
        "n_instants": int(scope.sum()),
        "n_evaluated": n_evaluated,
        "n_not_evaluable": {
            "first_of_run": int((scope & ~predicted).sum()),
            "no_rule": int((in_scope & ~evaluable.any(axis=1)).sum()),
        },
        "n_satisfied": n_satisfied,
        "n_violating": n_evaluated - n_satisfied,
        "satisfaction": _ratio(n_satisfied, n_evaluated),
        "rules": rules,
    }


def by_scenario(
    evaluation: RuleEvaluation, frame: pd.DataFrame, predicted: NDArray[np.bool_]
) -> dict[str, dict[str, Any]]:
    """Return evaluated, violating and per-rule violations per scenario."""
    scenarios = frame["scenario"].astype(str).to_numpy()
    out: dict[str, dict[str, Any]] = {}
    for scenario in sorted(set(scenarios)):
        block = summarize(evaluation, scenarios == scenario, predicted)
        out[scenario] = {
            "n_evaluated": block["n_evaluated"],
            "n_violating": block["n_violating"],
            "satisfaction": block["satisfaction"],
            "violations": {rule: block["rules"][rule]["n_violations"] for rule in RULE_NAMES},
        }
    return out


def source_block(
    evaluation: RuleEvaluation, frame: pd.DataFrame, predicted: NDArray[np.bool_]
) -> dict[str, Any]:
    """Return the all / normal / per-scenario summary of one evaluated source."""
    normal = ~frame["labelled"].to_numpy(dtype=np.bool_)
    everything = np.ones(len(frame), dtype=np.bool_)
    return {
        "all": summarize(evaluation, everything, predicted),
        "normal": summarize(evaluation, normal, predicted),
        "by_scenario": by_scenario(evaluation, frame, predicted),
    }


def passes(satisfaction: float | None) -> bool:
    """Return whether the satisfaction meets criterion 1 (> 99 %)."""
    return satisfaction is not None and satisfaction > SATISFACTION_THRESHOLD


# ----------------------------------------------------------------------
# Stage
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsCheckOutcome:
    """Lo que deja una ejecucion de physics_check.

    Attributes:
        document: Contenido del JSON escrito.
        metrics_path: JSON escrito.
        seconds: Tiempo de pared de prediccion y reglas (solo informativo).
    """

    document: Mapping[str, Any]
    metrics_path: Path
    seconds: float


def physics_document(
    frame: pd.DataFrame,
    predicted_values: NDArray[np.float64],
    tags: Sequence[str],
    roles: Mapping[str, Sequence[str]],
    table: WaterTable,
    limits: ConstraintLimits,
    digests: Mapping[str, str],
) -> dict[str, Any]:
    """Return the metrics JSON for the predictions and the measurement of a partition.

    Args:
        frame: Frame de test.
        predicted_values: Salida de `predict_next` sobre frame.
        tags: Orden de sensores.
        roles: Rol -> tags.
        table: Tabla del agua.
        limits: Umbrales de las reglas.
        digests: sha256 de los ficheros leidos, por nombre.

    Returns:
        The document (sin reloj ni rutas).
    """
    predicted = ~first_of_run(frame)
    measured_values = frame[value_columns(tags)].to_numpy(dtype=np.float64)
    everything = np.ones(len(frame), dtype=np.bool_)
    prediction = evaluate_by_run(frame, predicted_values, predicted, tags, roles, table, limits)
    # La medicion se evalua sobre el run entero (sus derivadas usan x(t-1) medido) y
    # se resume sobre los mismos instantes que la prediccion.
    measurement = evaluate_by_run(frame, measured_values, everything, tags, roles, table, limits)
    prediction_block = source_block(prediction, frame, predicted)
    satisfaction = prediction_block["all"]["satisfaction"]
    return {
        "version": METRICS_VERSION,
        "split": SPLIT,
        "rule_mode": RULE_MODE,
        "rules": list(RULE_NAMES),
        "threshold": SATISFACTION_THRESHOLD,
        "inputs": dict(sorted(digests.items())),
        "instants": {
            "total": len(frame),
            "predicted": int(predicted.sum()),
            "runs": int(frame["run_id"].nunique()),
        },
        "prediction": prediction_block,
        "measurement": source_block(measurement, frame, predicted),
        "satisfaction": satisfaction,
        "passed": passes(satisfaction),
    }


def _read_tags(data_dir: Path) -> tuple[str, ...]:
    """Return the sensor order written by prepare_sim."""
    path = data_dir / ASSIGNMENT_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path.as_posix()} not found: run the prepare_sim stage.")
    document = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(tag) for tag in document["tags"])


def _check_tags(
    data_dir: Path, checkpoint_tags: Sequence[str], catalog_tags: Sequence[str]
) -> None:
    """Raise if prepare_sim, the checkpoint and the catalog disagree on the sensor order."""
    data_tags = _read_tags(data_dir)
    if tuple(checkpoint_tags) != tuple(catalog_tags) or data_tags != tuple(catalog_tags):
        raise ValueError(
            "physics_check: the sensor order of the checkpoint, prepare_sim and the catalog "
            "must agree."
        )


def run_physics_check(params_path: str | Path, params: PinnPhysicsParams) -> PhysicsCheckOutcome:
    """Predict test, apply the rules and write the metrics JSON.

    Args:
        params_path: Fichero de parametros (secciones pinn: y simulation:).
        params: Seccion pinn: resuelta para physics_check.

    Returns:
        The PhysicsCheckOutcome.

    Raises:
        FileNotFoundError: If the checkpoint, the partition or a table is missing.
        ValueError: If the sensor orders disagree or the data are malformed.
    """
    simulation = load_simulation_params(params_path)
    catalog = load_sensor_catalog(simulation.sensors_path)
    table = load_water_table(simulation.water_table_path)
    limits = ConstraintLimits.from_plant(load_plant_params(simulation.plant_parameters_path))
    model_path = params.model_dir / MODEL_FILE
    loaded = load_checkpoint(model_path)
    _check_tags(params.data_dir, loaded.stats.tags, catalog.tags)
    tags = loaded.stats.tags
    path = params.data_dir / f"{SPLIT}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.as_posix()} not found: run the prepare_sim stage.")
    frame = pd.read_parquet(path, columns=partition_columns(tags))
    digests = {
        MODEL_FILE: file_sha256(model_path),
        path.name: file_sha256(path),
        ASSIGNMENT_NAME: file_sha256(params.data_dir / ASSIGNMENT_NAME),
    }
    start = time.perf_counter()
    with torch_threads(params.num_threads):
        values = predict_next(loaded, frame, catalog.roles)
    document = physics_document(frame, values, tags, catalog.roles, table, limits, digests)
    seconds = time.perf_counter() - start
    metrics_path = write_json(document, params.physics_metrics_path)
    every, normal = document["prediction"]["all"], document["prediction"]["normal"]
    _LOG.info(
        "Physics check on %s in %.1f s: %d / %d instants satisfy every rule (%s); normal %d / %d; "
        "%d first-of-run instants not evaluable",
        SPLIT,
        seconds,
        every["n_satisfied"],
        every["n_evaluated"],
        "passed" if document["passed"] else "FAILED",
        normal["n_satisfied"],
        normal["n_evaluated"],
        every["n_not_evaluable"]["first_of_run"],
    )
    return PhysicsCheckOutcome(document=document, metrics_path=metrics_path, seconds=seconds)


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> PhysicsCheckOutcome:
    """Run physics_check end to end.

    Args:
        params_path: Fichero de parametros.

    Returns:
        The PhysicsCheckOutcome.
    """
    return run_physics_check(params_path, load_pinn_physics_params(params_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
