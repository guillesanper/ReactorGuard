"""Builders compartidos de trazas y frames anchos del modelo termico de F3-M1.

Patron de tests/support/tep_frames.py: los tests de las constraints (F3-M2) y los
del simulador (F3-M3 en adelante) construyen aqui sus datos en vez de repetirlos.
Todo es determinista; el ruido solo entra si se pasa un Generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data.simulation.plant_params import load_neutronics, load_plant_params
from data.simulation.sensor_catalog import SensorCatalog, load_sensor_catalog
from data.simulation.thermal_model import Diagnostics, ModelInputs, ReactorModel, StepResult
from data.simulation.water_properties import WaterTable, load_water_table

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "configs"
DEFAULT_DT_S = 180.0
DEFAULT_START = "2026-01-01 00:00:00"


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
