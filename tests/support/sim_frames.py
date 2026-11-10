"""Builders compartidos de trazas, frames anchos y simuladores de F3-M1 a F3-M3.

Patron de tests/support/tep_frames.py: los tests de las constraints (F3-M2) y los
del simulador (F3-M3 en adelante) construyen aqui sus datos en vez de repetirlos.
Todo es determinista; el ruido solo entra si se pasa un Generator.

F3-M3 anade el builder de la configuracion `simulation:` y del simulador sobre una
configuracion pequena (runs de unas decenas de pasos): `write_sim_params` y
`build_simulator`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

from data.generators.reactor_simulator import ReactorSimulator
from data.simulation.dataset_builder import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    NO_FAULT,
    file_sha256,
    run_parquet_path,
    write_run_parquet,
)
from data.simulation.plant_params import load_neutronics, load_plant_params
from data.simulation.scenario import FaultAxis, ScenarioKind
from data.simulation.sensor_catalog import SensorCatalog, load_sensor_catalog
from data.simulation.simulation_params import load_simulation_params
from data.simulation.thermal_model import Diagnostics, ModelInputs, ReactorModel, StepResult
from data.simulation.water_properties import WaterTable, load_water_table

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
DEFAULT_DT_S = 180.0
DEFAULT_START = "2026-01-01 00:00:00"
SIM_SPANS_PATH = CONFIGS / "sim_sensor_spans.yaml"


@dataclass(frozen=True)
class Trace:
    """Traza del modelo muestreada cada dt_s.

    Attributes:
        times_s: Instante de cada muestra desde el inicio [s], forma (n,).
        states: Estado en cada muestra.
        inputs: Entradas vigentes en cada muestra.
        diagnostics: Diagnostico instantaneo en cada muestra.
        steps: Resultado de integrar cada intervalo, longitud n - 1.
    """

    times_s: NDArray[np.float64]
    states: list[NDArray[np.float64]]
    inputs: list[ModelInputs]
    diagnostics: list[Diagnostics]
    steps: list[StepResult]


def build_model(**kwargs: Any) -> ReactorModel:
    """Return the reference model on the versioned configuration."""
    return ReactorModel(
        load_plant_params(CONFIGS / "plant_parameters.yaml"),
        load_neutronics(CONFIGS / "neutronics_coefficients.yaml"),
        load_water_table(CONFIGS / "water_table.yaml"),
        **kwargs,
    )


def build_table() -> WaterTable:
    """Return the versioned water table."""
    return load_water_table(CONFIGS / "water_table.yaml")


def build_catalog() -> SensorCatalog:
    """Return the versioned sensor catalog."""
    return load_sensor_catalog(CONFIGS / "sim_sensors.yaml")


def ramp_loads(load_from: float, load_to: float, n_steps: int, hold_steps: int = 0) -> list[float]:
    """Return a load profile: hold_steps at load_from, then a linear ramp.

    Args:
        load_from: Carga inicial (fraccion del nominal).
        load_to: Carga final.
        n_steps: Intervalos de la rampa.
        hold_steps: Intervalos previos a carga constante.

    Returns:
        hold_steps + n_steps + 1 loads (one per sample).
    """
    ramp = np.linspace(load_from, load_to, n_steps + 1)
    return [load_from] * hold_steps + [float(v) for v in ramp]


def run_load_profile(
    model: ReactorModel, loads: list[float], dt_s: float = DEFAULT_DT_S
) -> Trace:
    """Integrate the model along a load profile starting at its equilibrium.

    Entre dos muestras la carga varia linealmente (`inputs_end`).

    Args:
        model: Modelo de referencia.
        loads: Carga en cada muestra (al menos 2).
        dt_s: Cadencia de muestreo [s].

    Returns:
        The Trace.

    Raises:
        ValueError: If fewer than two loads are given.
    """
    if len(loads) < 2:
        raise ValueError("a load profile needs at least two samples.")
    state = model.steady_state(loads[0])
    states = [state]
    inputs = [ModelInputs(load=load) for load in loads]
    steps: list[StepResult] = []
    for k in range(len(loads) - 1):
        step = model.advance(states[-1], inputs[k], dt_s, inputs[k + 1])
        steps.append(step)
        states.append(step.y)
    return Trace(
        times_s=np.arange(len(loads), dtype=np.float64) * dt_s,
        states=states,
        inputs=inputs,
        diagnostics=[model.diagnostics(s, i) for s, i in zip(states, inputs, strict=True)],
        steps=steps,
    )


def wide_frame(
    trace: Trace,
    model: ReactorModel,
    catalog: SensorCatalog,
    start: str = DEFAULT_START,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Return the wide frame of sensor values along a trace.

    Args:
        trace: Traza del modelo.
        model: Modelo que la produjo.
        catalog: Catalogo de sensores.
        start: Instante de la primera muestra.
        rng: Si se da, anade ruido blanco N(0, noise_sigma) a cada sensor.

    Returns:
        DataFrame with a DatetimeIndex and one float column per catalog tag.
    """
    rows = [
        catalog.true_values(model.observables(state, inputs))
        for state, inputs in zip(trace.states, trace.inputs, strict=True)
    ]
    frame = pd.DataFrame(rows, columns=list(catalog.tags), dtype=np.float64)
    if rng is not None:
        sigma = np.array([catalog.by_tag(tag).noise_sigma for tag in catalog.tags])
        frame = frame + rng.normal(0.0, 1.0, size=frame.shape) * sigma
    frame.index = pd.Timestamp(start) + pd.to_timedelta(trace.times_s, unit="s")
    return frame


# ----------------------------------------------------------------------
# F3-M3: configuracion `simulation:` y simulador sobre runs cortos
# ----------------------------------------------------------------------


def sim_section(work_dir: Path, **overrides: Any) -> dict[str, Any]:
    """Return a small, valid `simulation:` section.

    Runs de 30 pasos (90 min a 180 s) y esperas entre rampas de 6 a 20 min, de modo
    que cada run contiene rampas. Las rutas de configuracion son absolutas (los
    ficheros versionados) y las de salida cuelgan de `work_dir`.

    Args:
        work_dir: Directorio de trabajo para la salida (normalmente `tmp_path`).
        **overrides: Claves de la seccion a sustituir.

    Returns:
        The section as a plain mapping, ready for `yaml.safe_dump`.
    """
    section: dict[str, Any] = {
        "n_samples": 90,
        "fault_injection_rate": 0.40,
        "critical_min_fraction": 0.05,
        "openmc_seed": 7,
        "dt_seconds": 180,
        "run_length_steps": 30,
        "warmup_fraction": 0.20,
        "start_time": "2026-01-01T00:00:00+00:00",
        "plant_id": "SIM-TEST-01",
        "severity_weights": {"critical": 0.30, "high": 0.30, "medium": 0.25, "low": 0.15},
        "load_min_fraction": 0.70,
        "load_max_fraction": 1.00,
        "max_ramp_percent_per_min": 5.0,
        "load_hold_minutes_min": 6,
        "load_hold_minutes_max": 20,
        "output_dir": str(work_dir / "out"),
        "metrics_path": str(work_dir / "metrics.json"),
        "plant_parameters_path": str(CONFIGS / "plant_parameters.yaml"),
        "neutronics_path": str(CONFIGS / "neutronics_coefficients.yaml"),
        "sensors_path": str(CONFIGS / "sim_sensors.yaml"),
        "spans_path": str(SIM_SPANS_PATH),
        "water_table_path": str(CONFIGS / "water_table.yaml"),
        "calibration_date": "2025-06-01",
        "last_maintenance_date": "2025-12-01",
        "drift_coefficient": 0.0001,
    }
    section.update(overrides)
    return section


def write_sim_params(work_dir: Path, **overrides: Any) -> Path:
    """Write a params file holding a small `simulation:` section.

    Args:
        work_dir: Directorio donde escribir `params.yaml`.
        **overrides: Claves de la seccion a sustituir (ver `sim_section`).

    Returns:
        The path of the params file.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "params.yaml"
    path.write_text(
        yaml.safe_dump({"simulation": sim_section(work_dir, **overrides)}), encoding="utf-8"
    )
    return path


def build_simulator(work_dir: Path, **overrides: Any) -> ReactorSimulator:
    """Return a ReactorSimulator over the small configuration.

    Args:
        work_dir: Directorio de trabajo.
        **overrides: Claves de la seccion `simulation:` a sustituir.

    Returns:
        The simulator, built from the versioned configs.
    """
    params_path = write_sim_params(work_dir, **overrides)
    return ReactorSimulator.from_params(load_simulation_params(params_path))


# ----------------------------------------------------------------------
# F3-M6: dataset planificado pequeno (suite de seguridad y stage simulate)
# ----------------------------------------------------------------------

SMALL_DATASET_RATE = 0.12
"""Tasa etiquetada de la opcion B de D11 (la de params.yaml)."""

SMALL_DATASET_WEIGHTS: dict[str, float] = {
    "critical": 0.55,
    "high": 0.38,
    "medium": 0.06,
    "low": 0.01,
}
"""Pesos de severidad de la opcion B de D11 (los de params.yaml)."""

_SMALL_INSTRUMENT_LENGTHS: dict[str, tuple[int, int, int]] = {
    "drift": (24, 12, 6),
    "stuck": (10, 12, 16),
    "intermittent": (8, 12, 16),
}
"""Longitudes (bajo, medio, alto) de los episodios de instrumento en runs de 60 pasos.

El bloqueado debe superar la ventana de 8 del detector y crecer con el nivel."""


def write_small_plant_params(work_dir: Path) -> Path:
    """Write a copy of plant_parameters.yaml scaled to runs of 60 steps.

    El planificador exige 3 runs por tipo y ventanas de proceso de 151 pasos por
    defecto, que no caben en un dataset de segundos. Esta copia acorta las
    ventanas (4 pasos a magnitud plena y 5 de recuperacion, salvo el vacio, que
    conserva sus 20), las longitudes de instrumento y pide 2 runs por tipo, con un
    tope de picos de 0,05 para que el reparto de la opcion B quepa. Las magnitudes
    de los niveles, las rampas de entrada (30 min) y el modelo fisico no cambian.
    Las rampas no se comprimen mas (medido en F3-M6): con una entrada de 6 min o una
    retirada en 1 paso la excursion alta viola la regla de potencia/flujo sobre el
    estado verdadero, y el vacio alto con 5 pasos de recuperacion la viola un paso
    DESPUES de cerrar su ventana; con las rampas de produccion no ocurre ninguna
    de las dos cosas.

    Args:
        work_dir: Directorio donde escribir el fichero.

    Returns:
        The path of the scaled plant parameters.
    """
    document = yaml.safe_load((CONFIGS / "plant_parameters.yaml").read_text(encoding="utf-8"))
    for kind, tiers in document["fault_tiers"].items():
        tiers["hold_steps"] = 4
        if kind != "void_formation":
            tiers["recovery_steps"] = 5
    for kind, (low, medium, high) in _SMALL_INSTRUMENT_LENGTHS.items():
        levels = document["instrument_faults"][kind]["levels"]
        levels["low"]["length_steps"] = low
        levels["medium"]["length_steps"] = medium
        levels["high"]["length_steps"] = high
    document["scenario_planner"]["n_min_runs"] = 2
    document["scenario_planner"]["max_spike_rate"] = 0.05
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "plant_parameters.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def write_small_dataset_params(work_dir: Path, **overrides: Any) -> Path:
    """Write a params file whose planned dataset is feasible and takes seconds.

    34 runs de 60 pasos (2040 pasos, 55.080 lecturas: los mismos runs que el
    dataset de params.yaml) con la tasa y los pesos de la opcion B de D11 sobre la
    planta escalada de `write_small_plant_params`. El plan etiqueta el 12,65 %
    (CRITICAL 6,98 %), con los 8 tipos y 15 runs normales: los minimos por tipo
    pesan mas en runs cortos y hacen falta mas runs normales que en produccion.

    Args:
        work_dir: Directorio de trabajo (salida, metricas y planta escalada).
        **overrides: Claves de la seccion `simulation:` a sustituir.

    Returns:
        The path of the params file.
    """
    section: dict[str, Any] = {
        "n_samples": 2040,
        "run_length_steps": 60,
        "openmc_seed": 42,
        "fault_injection_rate": SMALL_DATASET_RATE,
        "severity_weights": dict(SMALL_DATASET_WEIGHTS),
        "plant_parameters_path": str(write_small_plant_params(work_dir)),
    }
    section.update(overrides)
    return write_sim_params(work_dir, **section)


# ----------------------------------------------------------------------
# F4-M1: frames largos sinteticos y dataset en disco (stage prepare_sim)
# ----------------------------------------------------------------------

SYNTHETIC_SEED = 42
"""Semilla del directorio `seed=<n>` del dataset sintetico."""


@dataclass(frozen=True)
class LabelSpan:
    """Ventana etiquetada de un run sintetico.

    Attributes:
        first_step: Primer paso etiquetado.
        last_step: Ultimo paso etiquetado (incluido).
        level: Nivel de magnitud (`low`, `medium`, `high`).
        tags: Sensores etiquetados; None etiqueta todos (fallo de proceso).
        episode: Sufijo del `episode_id` (`<run_id>:<episode>`).
    """

    first_step: int
    last_step: int
    level: str = "high"
    tags: tuple[str, ...] | None = None
    episode: str = "ep0"


def synthetic_value(sensor_index: int, step: int) -> float:
    """Return the measured value of a synthetic reading (distinct per sensor and step)."""
    return 100.0 + 10.0 * sensor_index + 0.1 * step


TRUE_OFFSET = 0.05
"""`value - true_value` de toda lectura sintetica."""


def long_run_frame(
    catalog: SensorCatalog,
    run_id: str,
    scenario: ScenarioKind,
    n_steps: int,
    spans: Sequence[LabelSpan] = (),
    start: str = "2026-01-01T00:00:00+00:00",
) -> pd.DataFrame:
    """Return a synthetic long frame of one run with the columns prepare_sim reads.

    Filas por paso y, dentro de cada paso, en el orden del catalogo, como el stage
    `simulate`. El valor medido es `synthetic_value` y el verdadero el medido menos
    `TRUE_OFFSET`. Las etiquetas siguen el eje y la severidad estatica del escenario:
    un fallo de proceso etiqueta `process_*` y uno de instrumento `instrument_*` (con
    `corrupted`). Las columnas de severidad, episodio y nivel son `string`, como en
    el parquet real.

    Args:
        catalog: Catalogo de sensores.
        run_id: Identificador del run.
        scenario: Escenario del run.
        n_steps: Pasos del run.
        spans: Ventanas etiquetadas (solo para escenarios con fallo).
        start: Instante del paso 0 (cadencia de 180 s).

    Returns:
        The long frame.
    """
    tags = catalog.tags
    types = [sensor.sensor_type.value for sensor in catalog.sensors]
    n_rows = n_steps * len(tags)
    steps = np.repeat(np.arange(n_steps), len(tags))
    sensor_index = np.tile(np.arange(len(tags)), n_steps)
    value = np.array(
        [synthetic_value(int(i), int(s)) for i, s in zip(sensor_index, steps, strict=True)]
    )
    fault = np.full(n_rows, NO_FAULT, dtype=object)
    severity: list[str | None] = [None] * n_rows
    episode: list[str | None] = [None] * n_rows
    level: list[str | None] = [None] * n_rows
    for span in spans:
        chosen = set(tags if span.tags is None else span.tags)
        for row in range(n_rows):
            in_window = span.first_step <= steps[row] <= span.last_step
            if in_window and tags[sensor_index[row]] in chosen:
                fault[row] = scenario.value
                severity[row] = None if scenario.severity is None else scenario.severity.value
                episode[row] = f"{run_id}:{span.episode}"
                level[row] = span.level
    labelled = fault != NO_FAULT
    none = np.full(n_rows, NO_FAULT, dtype=object)
    empty: list[str | None] = [None] * n_rows
    on_process = scenario.axis is FaultAxis.PROCESS
    timestamps = pd.Timestamp(start) + pd.to_timedelta(steps * 180, unit="s")
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(timestamps).tz_convert("UTC"),
            "sensor_id": np.array([tags[i] for i in sensor_index], dtype=object),
            "sensor_type": np.array([types[i] for i in sensor_index], dtype=object),
            "value": value,
            "timestep": steps.astype(np.int64),
            "run_id": np.full(n_rows, run_id, dtype=object),
            "scenario": np.full(n_rows, scenario.value, dtype=object),
            "true_value": value - TRUE_OFFSET,
            "corrupted": labelled & (scenario.axis is FaultAxis.INSTRUMENT),
            "process_fault": fault if on_process else none,
            "process_severity": pd.array(severity if on_process else empty, dtype="string"),
            "instrument_fault": none if on_process else fault,
            "instrument_severity": pd.array(empty if on_process else severity, dtype="string"),
            "episode_id": pd.array(episode, dtype="string"),
            "fault_level": pd.array(level, dtype="string"),
        }
    )


def write_long_dataset(root: Path, frames: Sequence[pd.DataFrame]) -> Path:
    """Write run frames in the `simulate` layout with a manifest.

    Cada frame va a `fault_type=<escenario>/seed=42/<run_id>.parquet` y el
    manifiesto lista `run_id`, `scenario`, `path` relativa y `sha256`, como el de
    `dataset_builder.write_dataset` (solo los campos que prepare_sim lee).

    Args:
        root: Raiz del dataset.
        frames: Un frame largo por run.

    Returns:
        The manifest path.
    """
    entries: list[dict[str, str]] = []
    for frame in frames:
        run_id = str(frame["run_id"].iloc[0])
        scenario = ScenarioKind(str(frame["scenario"].iloc[0]))
        path = write_run_parquet(frame, run_parquet_path(root, scenario, SYNTHETIC_SEED, run_id))
        entries.append(
            {
                "run_id": run_id,
                "scenario": scenario.value,
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
        )
    manifest = root / MANIFEST_NAME
    manifest.write_text(
        json.dumps({"version": MANIFEST_VERSION, "runs": entries}, indent=2), encoding="utf-8"
    )
    return manifest
