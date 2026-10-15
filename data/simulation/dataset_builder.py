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

Los fallos se enchufan en dos puntos: las entradas del modelo (`simulate_run`
recibe una secuencia de `ModelInputs`, no solo una carga) y la serie medida
(`build_run_frame` recibe `measured` ya ensamblada). Los fallos de PROCESO
(F3-M4) solo cambian las entradas: las etiquetas salen de `RunPlan.process_fault`
y el valor medido es el del estado perturbado, con `corrupted = False` (corrupta
es la lectura de un instrumento, no la de una planta que de verdad cambio). Los
de INSTRUMENTO (F3-M5) cambian `measured` DESPUES del ruido
(`instrument_faults.apply_instrument_faults`): las etiquetas salen de
`RunPlan.instrument_faults`, `true_value` queda intacto y `corrupted` marca solo
las lecturas cuyo valor difiere de la medida sin fallo.
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
from data.simulation.instrument_faults import apply_instrument_faults
from data.simulation.scenario import FAULT_SEVERITY, FaultAxis, RunPlan, ScenarioKind
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
    "injected_at_step",
    "episode_id",
    "fault_level",
)
"""Columnas de verdad-terreno y de identidad del run.

`injected_at` e `injected_at_step` son el inicio del episodio ACTIVO en esa lectura;
`fault_level` es su nivel de magnitud (`low`, `medium`, `high`). Los tres, como el
resto de etiquetas de fallo, son nulos fuera de la ventana de un episodio."""

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


def advance_substeps(
    model: ReactorModel,
    y: NDArray[np.float64],
    fine_inputs: Sequence[ModelInputs],
    start: int,
    substeps: int,
    sub_dt_s: float,
) -> StepResult:
    """Integrate one sampling interval as `substeps` consecutive sub-intervals.

    Sirve para fallos con dinamica mas rapida que la cadencia (la oscilacion de
    la bomba): el estado se integra con las entradas de cada sub-instante y solo
    se emite la muestra del final del intervalo. El cierre de energia se vuelve a
    calcular sobre el intervalo completo.

    Args:
        model: Modelo termico.
        y: Estado al inicio del intervalo.
        fine_inputs: Entradas en la rejilla fina.
        start: Indice de `fine_inputs` donde empieza el intervalo.
        substeps: Sub-intervalos por intervalo de muestreo.
        sub_dt_s: Duracion de cada sub-intervalo [s].

    Returns:
        The StepResult of the whole interval.
    """
    state = y
    e_gen = e_sec = 0.0
    n_rhs = 0
    for j in range(substeps):
        result = model.advance(state, fine_inputs[start + j], sub_dt_s, fine_inputs[start + j + 1])
        e_gen += result.energy_generated_mj
        e_sec += result.energy_removed_mj
        n_rhs += result.n_rhs_evaluations
        state = result.y
    d_u = model.internal_energy_mj(state) - model.internal_energy_mj(y)
    residual = abs(d_u - (e_gen - e_sec)) / max(e_gen, float(np.finfo(float).tiny))
    return StepResult(
        y=state,
        energy_generated_mj=e_gen,
        energy_removed_mj=e_sec,
        internal_energy_change_mj=d_u,
        energy_residual_rel=float(residual),
        n_rhs_evaluations=n_rhs,
    )


def simulate_run(
    model: ReactorModel,
    catalog: SensorCatalog,
    plan: RunPlan,
    inputs: Sequence[ModelInputs],
    dt_s: float,
    *,
    fine_inputs: Sequence[ModelInputs] | None = None,
    substeps: int = 1,
) -> RunSimulation:
    """Integrate one run from the equilibrium of its first load.

    Entre dos muestras las entradas varian linealmente (`inputs_end`), asi que un
    programa de carga muestreado es exactamente lo que el modelo integra. Un fallo
    con dinamica mas rapida que la cadencia aporta ademas una rejilla fina
    (`fine_inputs`, `substeps` por intervalo): el estado se integra sobre ella y
    solo se emiten las muestras de la rejilla gruesa.

    Args:
        model: Modelo termico.
        catalog: Catalogo de sensores (define las magnitudes que se guardan).
        plan: Plan del run (da los instantes y la identidad).
        inputs: Entradas en cada muestra; longitud `plan.n_steps`.
        dt_s: Cadencia de muestreo [s].
        fine_inputs: Entradas en la rejilla fina, longitud
            `(plan.n_steps - 1) * substeps + 1`; obligatoria si `substeps > 1`.
        substeps: Sub-intervalos por intervalo de muestreo.

    Returns:
        The RunSimulation.

    Raises:
        ValueError: If len(inputs) != plan.n_steps, fewer than 2 samples,
            dt_s <= 0, substeps < 1 or the fine grid has the wrong length (the
            model also raises if the state leaves the water table).
    """
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    if len(inputs) != plan.n_steps:
        raise ValueError(
            f"{plan.run_id}: got {len(inputs)} input samples for {plan.n_steps} steps."
        )
    if plan.n_steps < 2:
        raise ValueError(f"{plan.run_id}: a run needs at least 2 steps, got {plan.n_steps}.")
    if substeps < 1:
        raise ValueError(f"substeps must be >= 1, got {substeps}.")
    if substeps > 1:
        expected = (plan.n_steps - 1) * substeps + 1
        if fine_inputs is None or len(fine_inputs) != expected:
            got = None if fine_inputs is None else len(fine_inputs)
            raise ValueError(
                f"{plan.run_id}: substeps={substeps} needs {expected} fine inputs, got {got}."
            )

    state = model.steady_state(inputs[0].load)
    states = [state]
    steps: list[StepResult] = []
    for k in range(plan.n_steps - 1):
        if substeps == 1 or fine_inputs is None:
            step = model.advance(states[-1], inputs[k], dt_s, inputs[k + 1])
        else:
            step = advance_substeps(
                model, states[-1], fine_inputs, k * substeps, substeps, dt_s / substeps
            )
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


@dataclass(frozen=True)
class _StepLabels:
    """Process-fault labels of a run, one entry per step (not per reading).

    Attributes:
        process_fault: `none` or the scenario value, per step.
        severity: Static TDD severity inside the window, None outside.
        episode: Episode identifier inside the window, None outside.
        injected_at: Start of the active episode inside the window, NaT outside.
        injected_step: Start step of the active episode, -1 outside.
        level: Magnitude level inside the window, None outside.
    """

    process_fault: NDArray[np.object_]
    severity: NDArray[np.object_]
    episode: NDArray[np.object_]
    injected_at: pd.DatetimeIndex
    injected_step: NDArray[np.int64]
    level: NDArray[np.object_]


def _process_labels(sim: RunSimulation) -> _StepLabels:
    """Return the process-fault labels of a run.

    La etiqueta cubre TODOS los sensores de cada paso de la ventana
    (`ProcessFaultSpec.is_active`), desde el instante de inicio. Una corrida
    normal o de instrumento devuelve las etiquetas de "sin fallo de proceso".

    Args:
        sim: Run integrado.

    Returns:
        The per-step labels.
    """
    n_steps = sim.n_steps
    process_fault = np.full(n_steps, NO_FAULT, dtype=object)
    severity = np.full(n_steps, None, dtype=object)
    episode = np.full(n_steps, None, dtype=object)
    level = np.full(n_steps, None, dtype=object)
    injected_step = np.full(n_steps, -1, dtype=np.int64)
    injected: list[datetime | None] = [None] * n_steps
    spec = sim.plan.process_fault
    if spec is not None:
        kind = sim.plan.scenario
        start = sim.timestamps[spec.injected_step]
        for step in range(n_steps):
            if spec.is_active(step):
                process_fault[step] = kind.value
                severity[step] = FAULT_SEVERITY[kind].value
                episode[step] = f"{sim.plan.run_id}:{kind.value}"
                injected[step] = start
                injected_step[step] = spec.injected_step
                level[step] = spec.level.value
    return _StepLabels(
        process_fault=process_fault,
        severity=severity,
        episode=episode,
        injected_at=pd.DatetimeIndex(pd.to_datetime(injected, utc=True)),
        injected_step=injected_step,
        level=level,
    )


_NAT_NS = int(np.iinfo(np.int64).min)
"""Representacion entera de NaT (ns desde la epoca)."""


@dataclass(frozen=True)
class _ReadingLabels:
    """Instrument-fault labels of a run, one entry per (step, sensor).

    Attributes:
        instrument_fault: `none` or the kind value, shape (n_steps, n_sensors).
        severity: Static TDD severity inside an episode, None outside.
        episode: Episode identifier inside an episode, None outside.
        injected_at: Start instant (ns since the epoch) of the active episode,
            `_NAT_NS` outside.
        injected_step: Start step of the active episode, -1 outside.
        level: Magnitude level inside an episode, None outside.
    """

    instrument_fault: NDArray[np.object_]
    severity: NDArray[np.object_]
    episode: NDArray[np.object_]
    injected_at: NDArray[np.int64]
    injected_step: NDArray[np.int64]
    level: NDArray[np.object_]


def _instrument_labels(sim: RunSimulation, catalog: SensorCatalog) -> _ReadingLabels:
    """Return the instrument-fault labels of a run.

    La etiqueta cubre toda la ventana de cada episodio en el sensor afectado, y
    solo en el: un fallo de instrumento no es un hecho de la planta. Un run normal
    o de proceso devuelve las etiquetas de "sin fallo de instrumento".

    Args:
        sim: Run integrado.
        catalog: Catalogo de sensores (da la columna de cada tag).

    Returns:
        The per-reading labels.

    Raises:
        KeyError: If an episode names a sensor missing from the catalog.
    """
    shape = (sim.n_steps, len(catalog.sensors))
    fault = np.full(shape, NO_FAULT, dtype=object)
    severity = np.full(shape, None, dtype=object)
    episode_id = np.full(shape, None, dtype=object)
    level = np.full(shape, None, dtype=object)
    injected_at = np.full(shape, _NAT_NS, dtype=np.int64)
    injected_step = np.full(shape, -1, dtype=np.int64)
    column = {tag: j for j, tag in enumerate(catalog.tags)}
    stamps = pd.DatetimeIndex(sim.timestamps).asi8
    for episode in sim.plan.instrument_faults:
        if episode.sensor_id not in column:
            raise KeyError(
                f"{sim.plan.run_id}: episode names unknown sensor '{episode.sensor_id}'."
            )
        j = column[episode.sensor_id]
        window = slice(episode.start_step, episode.last_step + 1)
        scenario = episode.kind.scenario
        fault[window, j] = scenario.value
        severity[window, j] = FAULT_SEVERITY[scenario].value
        episode_id[window, j] = episode.episode_id(sim.plan.run_id)
        level[window, j] = episode.level.value
        injected_at[window, j] = stamps[episode.start_step]
        injected_step[window, j] = episode.start_step
    return _ReadingLabels(
        instrument_fault=fault,
        severity=severity,
        episode=episode_id,
        injected_at=injected_at,
        injected_step=injected_step,
        level=level,
    )


def assemble_long_frame(
    sim: RunSimulation,
    measured: NDArray[np.float64],
    catalog: SensorCatalog,
    spans: Mapping[str, SensorSpan],
    plant_id: str,
    corrupted: NDArray[np.bool_] | None = None,
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
        corrupted: Lecturas cuyo valor difiere de la medida sin fallo, forma
            (n_steps, n_sensors). Obligatoria en un run de instrumento (la
            produce `apply_instrument_faults`); en los demas es todo False.

    Returns:
        DataFrame with the columns of LONG_COLUMNS.

    Raises:
        ValueError: If `measured` or `corrupted` do not match the run, or an
            instrument run is assembled without its corruption mask.
        KeyError: If a sensor has no span or an episode names an unknown sensor.
    """
    if measured.shape != sim.true_values.shape:
        raise ValueError(
            f"measured has shape {measured.shape}, expected {sim.true_values.shape}."
        )
    is_instrument = sim.plan.scenario.axis is FaultAxis.INSTRUMENT
    if is_instrument and corrupted is None:
        raise ValueError(
            f"{sim.plan.run_id}: an instrument run ('{sim.plan.scenario.value}') needs the "
            "corruption mask produced by apply_instrument_faults."
        )
    if corrupted is not None and corrupted.shape != measured.shape:
        raise ValueError(
            f"corrupted has shape {corrupted.shape}, expected {measured.shape}."
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
    labels = _process_labels(sim)
    readings = _instrument_labels(sim, catalog)
    flagged = np.zeros(n_rows, dtype=np.bool_) if corrupted is None else corrupted.reshape(-1)
    no_label = np.full(n_rows, None, dtype=object)
    if is_instrument:
        episode = readings.episode.reshape(-1)
        instrument_severity = readings.severity.reshape(-1)
        process_severity = no_label
        level = readings.level.reshape(-1)
        injected_step = readings.injected_step.reshape(-1)
        injected_ns = readings.injected_at.reshape(-1)
        injected_at = pd.Series(pd.to_datetime(injected_ns, unit="ns", utc=True))
        injected_at[injected_ns == _NAT_NS] = pd.NaT
    else:
        episode = np.repeat(labels.episode, n_sensors)
        instrument_severity = no_label
        process_severity = np.repeat(labels.severity, n_sensors)
        level = np.repeat(labels.level, n_sensors)
        injected_step = np.repeat(labels.injected_step, n_sensors)
        injected_at = pd.Series(labels.injected_at.repeat(n_sensors))
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
        "corrupted": flagged,
        "process_fault": np.repeat(labels.process_fault, n_sensors),
        "process_severity": pd.array(process_severity, dtype="string"),
        "instrument_fault": readings.instrument_fault.reshape(-1),
        "instrument_severity": pd.array(instrument_severity, dtype="string"),
        "injected_at": injected_at,
        "injected_at_step": pd.arrays.IntegerArray(injected_step, injected_step < 0),
        "episode_id": pd.array(episode, dtype="string"),
        "fault_level": pd.array(level, dtype="string"),
    }
    return pd.DataFrame(columns, columns=list(LONG_COLUMNS))


def build_run_frame(
    sim: RunSimulation,
    catalog: SensorCatalog,
    spans: Mapping[str, SensorSpan],
    plant_id: str,
    noise_rng: np.random.Generator,
    fault_rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Measure a run with noise, apply its instrument faults and assemble its frame.

    Orden fijo: estado verdadero -> ruido -> fallos de instrumento -> cuentas ADC.
    Los fallos de instrumento actuan sobre la serie YA medida, de modo que
    `true_value` es el de la planta y el valor leido es el del instrumento.

    Args:
        sim: Run integrado.
        catalog: Catalogo de sensores.
        spans: Span calibrado por tag.
        plant_id: Identificador de planta.
        noise_rng: Generador del flujo de ruido del run (`RunPlan.rngs()[1]`).
        fault_rng: Generador del flujo de fallos del run (`RunPlan.rngs()[2]`);
            solo lo consume el intermitente. Por defecto, el del plan.

    Returns:
        The long frame (see `assemble_long_frame`).
    """
    measured = measure(sim, catalog, noise_rng)
    if sim.plan.scenario.axis is not FaultAxis.INSTRUMENT:
        return assemble_long_frame(sim, measured, catalog, spans, plant_id)
    rng = sim.plan.rngs()[2] if fault_rng is None else fault_rng
    effect = apply_instrument_faults(measured, catalog, spans, sim.plan, sim.dt_s, rng)
    return assemble_long_frame(
        sim, effect.values, catalog, spans, plant_id, corrupted=effect.corrupted
    )


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
