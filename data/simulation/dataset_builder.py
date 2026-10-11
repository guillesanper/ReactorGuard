"""Ensamblado del frame largo de un run de simulacion (parte 1 de F3-M3).

Cadena: entradas del modelo -> estado (LSODA) -> magnitudes verdaderas por sensor
-> ruido por sensor -> cuentas ADC con `SensorSpan.to_raw_counts` -> filas.

El frame largo lleva las 11 columnas que `readings_from_frame`
(data/generators/tep_adapter.py) exige para reconstruir `SensorReading`, de modo
que una salida de simulacion se valida offline con el mismo codigo que el TEP
(H12), mas las columnas de verdad-terreno del TDD 2.2: `true_value`, `corrupted`,
los dos ejes de fallo con su severidad, `injected_at` y `episode_id`.

Este modulo es de produccion: NO importa nada de `data/validation`. La validacion
offline vive en los tests y en `calibration_study`.

Los fallos (F3-M4 y F3-M5) se enchufan en dos puntos: las entradas del modelo
(`simulate_run` recibe una secuencia de `ModelInputs`, no solo una carga) y la
serie medida (`build_run_frame` recibe `measured` ya ensamblada). En F3-M3 solo
hay operacion normal y las etiquetas son las de "sin fallo".
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data.generators.tep_adapter import READING_ID_NAMESPACE
from data.schemas.sensor_reading import QualityFlag
from data.schemas.sensor_spans import SensorSpan
from data.simulation.scenario import FaultAxis, RunPlan, ScenarioKind
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.thermal_model import Diagnostics, ModelInputs, ReactorModel, StepResult

_LOG = logging.getLogger(__name__)

CONTRACT_COLUMNS: tuple[str, ...] = (
    "reading_id",
    "timestamp",
    "plant_id",
    "sensor_id",
    "sensor_type",
    "sensor_location",
    "elevation_m",
    "value",
    "unit",
    "quality",
    "raw_counts",
)
"""Las 11 columnas que `readings_from_frame` exige."""

LABEL_COLUMNS: tuple[str, ...] = (
    "timestep",
    "is_usable",
    "run_id",
    "scenario",
    "true_value",
    "corrupted",
    "process_fault",
    "process_severity",
    "instrument_fault",
    "instrument_severity",
    "injected_at",
    "episode_id",
)
"""Columnas de verdad-terreno y de identidad del run."""

LONG_COLUMNS: tuple[str, ...] = CONTRACT_COLUMNS + LABEL_COLUMNS
"""Orden de columnas del frame largo."""

NO_FAULT = FaultAxis.NONE.value
"""Etiqueta de `process_fault` e `instrument_fault` sin fallo."""


def _reading_key(plant_id: str, run_id: str, tag: str, iso_timestamp: str) -> str:
    """Return the identity string a reading UUID is derived from."""
    return f"{plant_id}|{run_id}|{tag}|{iso_timestamp}"


def sim_reading_uuid(plant_id: str, run_id: str, tag: str, timestamp: datetime) -> uuid.UUID:
    """Return the deterministic identifier of one simulated reading.

    UUID5 sobre la identidad de la medida, con el mismo espacio de nombres que el
    TEP. EL `run_id` ENTRA EN LA CLAVE: cada run es un experimento independiente y
    una misma terna (planta, sensor, instante) no debe repetirse entre runs de
    semillas distintas aunque coincidan los instantes.

    Args:
        plant_id: Plant identifier stamped on the reading.
        run_id: Identifier of the run (`sim_run_001`).
        tag: Sensor tag.
        timestamp: Timezone-aware instant of the sample.

    Returns:
        The reading's UUID5, stable across runs and machines.
    """
    key = _reading_key(plant_id, run_id, tag, timestamp.isoformat())
    return uuid.uuid5(READING_ID_NAMESPACE, key)


@dataclass(frozen=True)
class RunSimulation:
    """Resultado de integrar un run: estado, magnitudes verdaderas y diagnostico.

    Attributes:
        plan: Plan del run.
        dt_s: Cadencia de muestreo [s].
        inputs: Entradas del modelo en cada muestra.
        states: Estado del modelo en cada muestra, forma (n_steps, N_STATES).
        true_values: Valor verdadero (sin ruido) de cada sensor, forma
            (n_steps, n_sensors), en el orden del catalogo.
        diagnostics: Diagnostico instantaneo en cada muestra.
        steps: Resultado de integrar cada intervalo (n_steps - 1).
        timestamps: Instante de cada muestra.
    """

    plan: RunPlan
    dt_s: float
    inputs: tuple[ModelInputs, ...]
    states: NDArray[np.float64]
    true_values: NDArray[np.float64]
    diagnostics: tuple[Diagnostics, ...]
    steps: tuple[StepResult, ...]
    timestamps: tuple[datetime, ...]

    @property
    def n_steps(self) -> int:
        """Return the number of emitted samples."""
        return int(self.true_values.shape[0])

    @property
    def max_energy_residual(self) -> float:
        """Return the worst relative energy closure over the intervals."""
        if not self.steps:
            return 0.0
        return max(step.energy_residual_rel for step in self.steps)


def simulate_run(
    model: ReactorModel,
    catalog: SensorCatalog,
    plan: RunPlan,
    inputs: Sequence[ModelInputs],
    dt_s: float,
) -> RunSimulation:
    """Integrate one run from the equilibrium of its first load.

    Entre dos muestras las entradas varian linealmente (`inputs_end`), asi que un
    programa de carga muestreado es exactamente lo que el modelo integra.

    Args:
        model: Modelo termico.
        catalog: Catalogo de sensores (define las magnitudes que se guardan).
        plan: Plan del run (da los instantes y la identidad).
        inputs: Entradas en cada muestra; longitud `plan.n_steps`.
        dt_s: Cadencia de muestreo [s].

    Returns:
        The RunSimulation.

    Raises:
        ValueError: If len(inputs) != plan.n_steps, fewer than 2 samples, or
            dt_s <= 0 (the model also raises if the state leaves the water table).
    """
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    if len(inputs) != plan.n_steps:
        raise ValueError(
            f"{plan.run_id}: got {len(inputs)} input samples for {plan.n_steps} steps."
        )
    if plan.n_steps < 2:
        raise ValueError(f"{plan.run_id}: a run needs at least 2 steps, got {plan.n_steps}.")

    state = model.steady_state(inputs[0].load)
    states = [state]
    steps: list[StepResult] = []
    for k in range(plan.n_steps - 1):
        step = model.advance(states[-1], inputs[k], dt_s, inputs[k + 1])
        steps.append(step)
        states.append(step.y)

    rows: list[list[float]] = []
    for s, u in zip(states, inputs, strict=True):
        values = catalog.true_values(model.observables(s, u))
        rows.append([values[tag] for tag in catalog.tags])
    diagnostics = tuple(model.diagnostics(s, u) for s, u in zip(states, inputs, strict=True))
    return RunSimulation(
        plan=plan,
        dt_s=dt_s,
        inputs=tuple(inputs),
        states=np.vstack(states),
        true_values=np.asarray(rows, dtype=np.float64),
        diagnostics=diagnostics,
        steps=tuple(steps),
        timestamps=tuple(plan.timestamps(dt_s)),
    )


def noise_sigmas(catalog: SensorCatalog) -> NDArray[np.float64]:
    """Return the noise sigma of every sensor, in catalog order.

    Args:
        catalog: Catalogo de sensores.

    Returns:
        Array of shape (n_sensors,).
    """
    return np.array([sensor.noise_sigma for sensor in catalog.sensors], dtype=np.float64)


def measure(
    sim: RunSimulation, catalog: SensorCatalog, rng: np.random.Generator
) -> NDArray[np.float64]:
    """Return the noisy reading of every sensor at every step.

    Ruido blanco gaussiano independiente por sensor y por muestra: a 180 s entre
    muestras no hay autocorrelacion que modelar. Se genera la matriz completa de
    una vez, de modo que el flujo del generador no depende de como se consuma.

    Args:
        sim: Run integrado.
        catalog: Catalogo (aporta `noise_sigma`).
        rng: Generador del flujo de ruido del run.

    Returns:
        Array of shape (n_steps, n_sensors).
    """
    noise = rng.normal(0.0, 1.0, size=sim.true_values.shape) * noise_sigmas(catalog)
    measured: NDArray[np.float64] = sim.true_values + noise
    return measured


def _check_spans(catalog: SensorCatalog, spans: Mapping[str, SensorSpan]) -> None:
    """Raise if a catalog sensor has no span (the validator would be blind to it).

    Raises:
        KeyError: If a sensor of the catalog is missing from spans.
    """
    missing = [tag for tag in catalog.tags if tag not in spans]
    if missing:
        raise KeyError(
            f"Sensor span table is missing {len(missing)} simulation tags: "
            f"{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}. "
            "Regenerate it with data/generators/derive_sim_spans.py."
        )


def raw_counts_matrix(
    measured: NDArray[np.float64], catalog: SensorCatalog, spans: Mapping[str, SensorSpan]
) -> NDArray[np.int64]:
    """Return the ADC counts of a (n_steps, n_sensors) matrix of values.

    Usa `SensorSpan.to_raw_counts` valor a valor: es la definicion unica del
    escalado y del recorte, y repetirla vectorizada seria otra copia que mantener.

    Args:
        measured: Valores medidos en unidades de ingenieria.
        catalog: Catalogo de sensores (da el orden de las columnas).
        spans: Span calibrado por tag.

    Returns:
        Integer array in [0, 65535] with the shape of `measured`.

    Raises:
        KeyError: If a sensor has no span.
        ValueError: If a value is not finite.
    """
    _check_spans(catalog, spans)
    if not np.isfinite(measured).all():
        raise ValueError("measured values must be finite to be scaled to ADC counts.")
    counts = np.empty(measured.shape, dtype=np.int64)
    for j, tag in enumerate(catalog.tags):
        span = spans[tag]
        counts[:, j] = [span.to_raw_counts(float(v)) for v in measured[:, j]]
    return counts


def assemble_long_frame(
    sim: RunSimulation,
    measured: NDArray[np.float64],
    catalog: SensorCatalog,
    spans: Mapping[str, SensorSpan],
    plant_id: str,
) -> pd.DataFrame:
    """Return the long-format frame of one run, one row per (step, sensor).

    Filas en orden (paso, sensor del catalogo): el orden importa porque los
    detectores del validador tienen estado por sensor y lo construyen segun
    llegan las lecturas.

    Args:
        sim: Run integrado.
        measured: Valores medidos, forma (n_steps, n_sensors).
        catalog: Catalogo de sensores.
        spans: Span calibrado por tag (escala a cuentas ADC).
        plant_id: Identificador de planta de las lecturas.

    Returns:
        DataFrame with the columns of LONG_COLUMNS.

    Raises:
        NotImplementedError: If the run is not normal operation (the fault labels
            arrive with F3-M4 and F3-M5).
        ValueError: If `measured` does not match the run.
        KeyError: If a sensor has no span.
    """
    if sim.plan.scenario is not ScenarioKind.NORMAL:
        raise NotImplementedError(
            f"{sim.plan.run_id}: scenario '{sim.plan.scenario.value}' needs the fault "
            "labels of F3-M4/F3-M5; F3-M3 only assembles normal operation."
        )
    if measured.shape != sim.true_values.shape:
        raise ValueError(
            f"measured has shape {measured.shape}, expected {sim.true_values.shape}."
        )
    n_steps, n_sensors = measured.shape
    n_rows = n_steps * n_sensors
    counts = raw_counts_matrix(measured, catalog, spans)

    run_id = sim.plan.run_id
    iso = [ts.isoformat() for ts in sim.timestamps]
    reading_ids = [
        str(uuid.uuid5(READING_ID_NAMESPACE, _reading_key(plant_id, run_id, tag, stamp)))
        for stamp in iso
        for tag in catalog.tags
    ]
    sensors = catalog.sensors
    no_label = pd.array([None] * n_rows, dtype="string")
    columns = {
        "reading_id": reading_ids,
        "timestamp": pd.Series(pd.DatetimeIndex(sim.timestamps).repeat(n_sensors)),
        "plant_id": np.full(n_rows, plant_id, dtype=object),
        "sensor_id": np.tile(np.array([s.tag for s in sensors], dtype=object), n_steps),
        "sensor_type": np.tile(
            np.array([s.sensor_type.value for s in sensors], dtype=object), n_steps
        ),
        "sensor_location": np.tile(
            np.array([s.location.value for s in sensors], dtype=object), n_steps
        ),
        "elevation_m": np.tile(
            np.array([s.elevation_m for s in sensors], dtype=np.float64), n_steps
        ),
        "value": measured.reshape(-1),
        "unit": np.tile(np.array([s.unit.value for s in sensors], dtype=object), n_steps),
        "quality": np.full(n_rows, QualityFlag.GOOD.value, dtype=object),
        "raw_counts": counts.reshape(-1),
        "timestep": np.repeat(np.arange(n_steps, dtype=np.int64), n_sensors),
        "is_usable": np.ones(n_rows, dtype=np.bool_),
        "run_id": np.full(n_rows, run_id, dtype=object),
        "scenario": np.full(n_rows, sim.plan.scenario.value, dtype=object),
        "true_value": sim.true_values.reshape(-1),
        "corrupted": np.zeros(n_rows, dtype=np.bool_),
        "process_fault": np.full(n_rows, NO_FAULT, dtype=object),
        "process_severity": no_label,
        "instrument_fault": np.full(n_rows, NO_FAULT, dtype=object),
        "instrument_severity": no_label,
        "injected_at": pd.Series(pd.NaT, index=range(n_rows), dtype="datetime64[ns, UTC]"),
        "episode_id": no_label,
    }
    return pd.DataFrame(columns, columns=list(LONG_COLUMNS))


def build_run_frame(
    sim: RunSimulation,
    catalog: SensorCatalog,
    spans: Mapping[str, SensorSpan],
    plant_id: str,
    noise_rng: np.random.Generator,
) -> pd.DataFrame:
    """Measure a run with noise and assemble its long frame.

    Args:
        sim: Run integrado.
        catalog: Catalogo de sensores.
        spans: Span calibrado por tag.
        plant_id: Identificador de planta.
        noise_rng: Generador del flujo de ruido del run (`RunPlan.rngs()[1]`).

    Returns:
        The long frame (see `assemble_long_frame`).
    """
    return assemble_long_frame(sim, measure(sim, catalog, noise_rng), catalog, spans, plant_id)


def wide_frame_from_matrix(
    values: NDArray[np.float64], catalog: SensorCatalog, timestamps: Sequence[datetime]
) -> pd.DataFrame:
    """Return the wide frame (one column per tag) of a (n_steps, n_sensors) matrix.

    Es lo que `evaluate_rules` consume; evita pasar por el frame largo cuando solo
    se quieren evaluar reglas sobre valores verdaderos o medidos.

    Args:
        values: Valores, forma (n_steps, n_sensors), en el orden del catalogo.
        catalog: Catalogo de sensores.
        timestamps: Instante de cada fila.

    Returns:
        DataFrame with a DatetimeIndex and one float column per catalog tag.

    Raises:
        ValueError: If the shapes do not match the catalog and the timestamps.
    """
    if values.ndim != 2 or values.shape != (len(timestamps), len(catalog.sensors)):
        raise ValueError(
            f"values has shape {values.shape}, expected "
            f"({len(timestamps)}, {len(catalog.sensors)})."
        )
    index = pd.DatetimeIndex(list(timestamps))
    return pd.DataFrame(values, columns=list(catalog.tags), index=index, dtype=np.float64)


def wide_frame_from_long(frame: pd.DataFrame, column: str = "value") -> pd.DataFrame:
    """Pivot the long frame of ONE run into the wide frame the rules evaluate.

    Args:
        frame: Frame largo de un unico run.
        column: Columna a pivotar (`value` o `true_value`).

    Returns:
        One row per timestamp (DatetimeIndex), one float column per sensor, in
        order of first appearance.

    Raises:
        ValueError: If the frame holds more than one run.
    """
    if frame["run_id"].nunique() != 1:
        raise ValueError(
            "wide_frame_from_long needs exactly one run per frame; pivot each run "
            "separately before evaluating the rules."
        )
    order = list(dict.fromkeys(frame["sensor_id"]))
    wide = frame.pivot(index="timestamp", columns="sensor_id", values=column)
    return wide[order].astype(np.float64)


def run_parquet_path(
    output_dir: str | Path, scenario: ScenarioKind, seed: int, run_id: str
) -> Path:
    """Return the dataset path of a run (TDD 4.4 layout).

    Args:
        output_dir: Raiz del dataset.
        scenario: Escenario del run (directorio `fault_type=<escenario>`).
        seed: Semilla maestra (directorio `seed=<n>`).
        run_id: Identificador del run (nombre del fichero).

    Returns:
        `<output_dir>/fault_type=<escenario>/seed=<n>/<run_id>.parquet`.
    """
    return Path(output_dir) / f"fault_type={scenario.value}" / f"seed={seed}" / f"{run_id}.parquet"


def write_run_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Write a run frame as Parquet, creating the directory.

    `DataFrame.to_parquet(index=False)`: el mismo metodo que el TEP, reproducible
    en bytes para el mismo frame.

    Args:
        frame: Frame largo del run.
        path: Destino.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    _LOG.info("Wrote %d readings to %s", len(frame), target.name)
    return target
