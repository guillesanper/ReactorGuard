"""Derive the calibrated span table of the simulation sensors.

Genera configs/sim_sensor_spans.yaml a partir de 5 runs de operacion normal
simulada. Es el analogo de `derive_sensor_spans.py` para el catalogo propio de
la simulacion (D2), con la misma logica y los mismos margenes: el sobre de alarma
(`alarm_min`, `alarm_max`) y el span del transmisor (`min`, `max`) se obtienen
ensanchando el rango observado.

Dos decisiones de procedencia, ambas deliberadas:

    - Se observan SOLO runs de operacion normal. Un span ajustado sobre datos de
      fallo dejaria al detector de rango ciego justo a lo que debe senalar.
    - Los 5 runs usan `SPANS_SEED`, DISTINTA de `simulation.openmc_seed`. Si el
      span se derivase de los mismos runs que luego se validan, el validador
      mediria contra la muestra que defino su umbral y los falsos positivos
      saldrian infraestimados por construccion.

Este modulo NO es un stage de dvc.yaml: se ejecuta a mano, el fichero resultante se
revisa y se commitea, y el stage `simulate` lo declara como `deps` (como pasa con
`configs/sensor_spans.yaml`).

Uso:
    python data/generators/derive_sim_spans.py
    python data/generators/derive_sim_spans.py --params params.yaml --output otro.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from data.generators.derive_sensor_spans import (
    DEFAULT_ALARM_MARGIN,
    DEFAULT_SPAN_MARGIN,
    MIN_ABSOLUTE_WIDTH,
)
from data.simulation.dataset_builder import measure, simulate_run
from data.simulation.plant_params import load_neutronics, load_plant_params
from data.simulation.scenario import LoadProfile, normal_run_plans
from data.simulation.sensor_catalog import SensorCatalog, load_sensor_catalog
from data.simulation.simulation_params import (
    DEFAULT_PARAMS_PATH,
    SimulationParams,
    load_simulation_params,
)
from data.simulation.thermal_model import ModelInputs, ReactorModel
from data.simulation.water_properties import load_water_table

_LOG = logging.getLogger(__name__)

SPANS_SEED = 20260905
"""Semilla maestra de los runs de los spans. Distinta de `simulation.openmc_seed`."""

N_SPAN_RUNS = 5
"""Runs de operacion normal que se observan."""

_DECIMALS = 6


def observe_normal_operation(
    params: SimulationParams,
    model: ReactorModel,
    catalog: SensorCatalog,
    n_runs: int = N_SPAN_RUNS,
    seed: int = SPANS_SEED,
) -> NDArray[np.float64]:
    """Return the measured values of n_runs normal-operation runs.

    Args:
        params: Resolved simulation parameters (cadencia, longitud, programa de carga).
        model: Thermal model.
        catalog: Sensor catalog (sigma de ruido incluido).
        n_runs: Number of normal runs to observe.
        seed: Master seed of the observation runs.

    Returns:
        Array of shape (n_runs * run_length_steps, n_sensors), runs stacked in order.
    """
    profile = LoadProfile.from_params(params)
    blocks: list[NDArray[np.float64]] = []
    for plan in normal_run_plans(params, seed=seed, n_runs=n_runs):
        load_rng, noise_rng, _ = plan.rngs()
        loads = profile.sample(load_rng, plan.n_steps, params.dt_seconds)
        inputs = [ModelInputs(load=float(load)) for load in loads]
        sim = simulate_run(model, catalog, plan, inputs, params.dt_seconds)
        blocks.append(measure(sim, catalog, noise_rng))
    return np.vstack(blocks)


def derive_spans(
    observed: NDArray[np.float64],
    catalog: SensorCatalog,
    alarm_margin: float = DEFAULT_ALARM_MARGIN,
    span_margin: float = DEFAULT_SPAN_MARGIN,
) -> dict[str, dict[str, Any]]:
    """Derive a span entry per sensor from observed normal-operation values.

    Args:
        observed: Measured values, shape (n_samples, n_sensors), catalog order.
        catalog: Sensor catalog (tags, units).
        alarm_margin: Fractional margin of the normal-operation envelope on each
            side of the observed range.
        span_margin: Fractional margin of the calibrated span; at least
            alarm_margin.

    Returns:
        Mapping of tag to {min, max, alarm_min, alarm_max, unit}, in catalog order.

    Raises:
        ValueError: If a margin is negative, span_margin < alarm_margin, or the
            observation does not match the catalog.
    """
    if alarm_margin < 0.0 or span_margin < 0.0:
        raise ValueError(
            f"Margins must be non-negative, got alarm_margin={alarm_margin}, "
            f"span_margin={span_margin}."
        )
    if span_margin < alarm_margin:
        raise ValueError(
            f"span_margin ({span_margin}) must be at least alarm_margin "
            f"({alarm_margin}): the alarm envelope lives inside the span."
        )
    if observed.ndim != 2 or observed.shape[1] != len(catalog.sensors) or observed.shape[0] < 1:
        raise ValueError(
            f"observed must have shape (n_samples, {len(catalog.sensors)}), "
            f"got {observed.shape}."
        )

    spans: dict[str, dict[str, Any]] = {}
    for j, sensor in enumerate(catalog.sensors):
        low = float(observed[:, j].min())
        high = float(observed[:, j].max())
        width = high - low
        if width <= 0.0:
            _LOG.warning(
                "%s is constant at %.6f in normal operation; assigning the minimum span "
                "width of %.1f.",
                sensor.tag,
                low,
                MIN_ABSOLUTE_WIDTH,
            )
            alarm_pad = span_pad = MIN_ABSOLUTE_WIDTH / 2.0
        else:
            alarm_pad = width * alarm_margin
            span_pad = width * span_margin
        spans[sensor.tag] = {
            "min": round(low - span_pad, _DECIMALS),
            "max": round(high + span_pad, _DECIMALS),
            "alarm_min": round(low - alarm_pad, _DECIMALS),
            "alarm_max": round(high + alarm_pad, _DECIMALS),
            "unit": sensor.unit.value,
        }
    return spans


def render_yaml(
    spans: dict[str, dict[str, Any]],
    n_runs: int,
    n_steps: int,
    seed: int,
    alarm_margin: float,
    span_margin: float,
) -> str:
    """Render the span table as a commented YAML document.

    Args:
        spans: Mapping produced by derive_spans.
        n_runs: Runs observed, recorded for traceability.
        n_steps: Steps per run.
        seed: Master seed of the observation, recorded for traceability.
        alarm_margin: Alarm margin used.
        span_margin: Span margin used.

    Returns:
        The YAML document as a string, ready to be written to disk.
    """
    header = (
        "# ==========================================================================\n"
        "# ReactorGuard - Spans calibrados de los sensores de la simulacion\n"
        "#\n"
        "# GENERADO por data/generators/derive_sim_spans.py. Se revisa a mano y se\n"
        "# commitea; no es un artefacto de `dvc repro`.\n"
        "#\n"
        f"#   Derivado de  : {n_runs} runs de operacion normal simulada x {n_steps} pasos\n"
        f"#   Semilla      : {seed} (SPANS_SEED; distinta de simulation.openmc_seed)\n"
        f"#   Margen span  : {span_margin:.0%} a cada lado del rango observado\n"
        f"#   Margen alarma: {alarm_margin:.0%} a cada lado del rango observado\n"
        "#\n"
        "# Dos rangos por sensor (misma semantica que configs/sensor_spans.yaml):\n"
        "#\n"
        "#   [min, max]              span calibrado del transmisor. Escalado a cuentas\n"
        "#                           ADC en data/simulation/dataset_builder.py.\n"
        "#   [alarm_min, alarm_max]  sobre de operacion normal. Umbral del detector de\n"
        "#                           fuera-de-rango en data/validation/sensor_validator.py.\n"
        "#\n"
        "# Si cambia el catalogo, el sigma de ruido o el programa de carga hay que\n"
        "# regenerarlo: un span derivado de otro ruido deja de ser el de esta planta.\n"
        "# ==========================================================================\n\n"
    )
    body = yaml.safe_dump(
        {"sensors": spans}, sort_keys=False, default_flow_style=False, allow_unicode=False
    )
    return header + body


def main(argv: list[str] | None = None) -> Path:
    """Derive the span table and write it to disk.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        The path the span table was written to.

    Raises:
        FileNotFoundError: If the params or a configuration file does not exist.
        KeyError: If a required key is missing.
        ValueError: If a margin is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument(
        "--output",
        default=None,
        help="Destination path (defaults to simulation.spans_path).",
    )
    parser.add_argument("--alarm-margin", type=float, default=DEFAULT_ALARM_MARGIN)
    parser.add_argument("--span-margin", type=float, default=DEFAULT_SPAN_MARGIN)
    args = parser.parse_args(argv)

    params = load_simulation_params(args.params)
    model = ReactorModel(
        load_plant_params(params.plant_parameters_path),
        load_neutronics(params.neutronics_path),
        load_water_table(params.water_table_path),
    )
    catalog = load_sensor_catalog(params.sensors_path)

    observed = observe_normal_operation(params, model, catalog)
    spans = derive_spans(observed, catalog, args.alarm_margin, args.span_margin)

    output = Path(args.output) if args.output else params.spans_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_yaml(
            spans,
            N_SPAN_RUNS,
            params.run_length_steps,
            SPANS_SEED,
            args.alarm_margin,
            args.span_margin,
        ),
        encoding="utf-8",
    )
    _LOG.info("Wrote %d sensor spans to %s.", len(spans), output)
    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
