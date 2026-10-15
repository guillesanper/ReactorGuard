"""Mediciones obligatorias de F3-M3: cadencia, ruido, determinismo y rendimiento.

Herramienta de desarrollo, no un stage. Existe para que las cifras de
`Docs/physics_constraints.md` (seccion de F3-M3) se puedan repetir con una orden y,
en particular, para que F3-M8 "repita las mediciones de M3" con los coeficientes
neutronicos calibrados.

Cuatro estudios:

    cadence_study       el mismo escenario normal a 1/10/60/180 s: violaciones
                        sobre valores verdaderos (esperado 0), maxima derivada
                        del estado, y alertas del validador SIN recalibrar.
    noise_calibration   reglas sobre mediciones normales en N runs, con la
                        distancia minima al umbral de cada una.
    determinism_check   mismo frame y mismos bytes de parquet entre dos
                        ejecuciones y entre procesos.
    measure_throughput  pasos/s extremo a extremo.
    fault_matrix        (F3-M4) por tipo de fallo de proceso y nivel: que reglas se
                        violan, con que tasa y con que retardo en muestras, sobre
                        valores verdaderos y sobre mediciones con ruido.
    instrument_matrix   (F3-M5) lo mismo para los 5 fallos de instrumento, sensor por
                        sensor, con la votacion por mediana y con el sensor aislado.
    instrument_detection (F3-M5) que ve el validador de Fase 2 de cada fallo inyectado
                        (informativo: no se recalibra nada).
    capacity_report     (F3-M5) presupuesto de D11, topes de densidad y tasa maxima.
    plan_coverage_report (F3-M5) etiquetas y cobertura del dataset planificado.

El validador de Fase 2 (`data/validation`) se importa de forma perezosa y solo
aqui: el simulador de produccion no depende de el.

Uso:
    python -m data.generators.simulation_study report --params params.yaml \
        --output metrics/m3_study.json
    python -m data.generators.simulation_study digest --params params.yaml
    python -m data.generators.simulation_study faults --params params.yaml \
        --output metrics/m4_fault_matrix.json --repeats 3
    python -m data.generators.simulation_study instrument --params params.yaml \
        --output metrics/m5_instrument.json --repeats 2
    python -m data.generators.simulation_study plan --params params.yaml \
        --output metrics/m5_plan.json [--rate 0.05 --weights capacity_a]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import subprocess  # noqa: S404 - the cross-process check spawns this same module
import sys
import tempfile
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.generators.reactor_simulator import ReactorSimulator
from data.generators.tep_adapter import readings_from_frame
from data.schemas.sensor_spans import SensorSpan
from data.simulation.coverage import CoverageCollector, CoverageScope
from data.simulation.dataset_builder import (
    RunSimulation,
    assemble_long_frame,
    build_run_frame,
    measure,
    run_parquet_path,
    simulate_run,
    wide_frame_from_matrix,
    write_run_parquet,
)
from data.simulation.instrument_faults import apply_episode, apply_instrument_faults
from data.simulation.physics_constraints import (
    RULE_NAMES,
    ConstraintLimits,
    RuleEvaluation,
    conservation_report,
    evaluate_rules_detailed,
    finite_difference_rate,
    min_observable_rate,
    rule_margins,
)
from data.simulation.process_faults import PROCESS_KINDS, LevelResponse, response_from_run
from data.simulation.scenario import (
    FAULT_KINDS,
    INSTRUMENT_KINDS,
    FaultLevel,
    InstrumentEpisode,
    InstrumentFaultKind,
    RunPlan,
    ScenarioKind,
    make_run_plan,
)
from data.simulation.scenario_planner import compute_budget, max_feasible_rate
from data.simulation.sensor_catalog import SensorCatalog, SensorDef
from data.simulation.simulation_params import (
    DEFAULT_PARAMS_PATH,
    SimulationParams,
    load_simulation_params,
)
from data.simulation.thermal_model import ModelInputs, StateIndex
from data.simulation.water_properties import WaterTable

_LOG = logging.getLogger(__name__)

DEFAULT_CADENCES_S: tuple[float, ...] = (1.0, 10.0, 60.0, 180.0)
"""Cadencias del estudio de D1."""

DEFAULT_STUDY_DURATION_S = 6 * 3600.0
"""Duracion simulada del escenario normal del estudio de cadencia [s]."""

STUDY_WARMUP_S = 1800.0
"""Tramo inicial a carga constante del escenario del estudio [s].

No es el `warmup_fraction` del dataset: aplicado a un run de 1500 pasos son 15 h,
mas que el escenario entero, y el estudio no tendria ni una rampa.
"""

STUDY_HOLD_MINUTES: tuple[float, float] = (20.0, 40.0)
"""Esperas entre rampas del escenario del estudio [min]: mas densas que las del
dataset (60-360 min) para que 6 h contengan varias rampas y el estudio las ejercite."""

CADENCE_STUDY_SEED = 20261012
"""Semilla del escenario del estudio de cadencia (distinta de las del dataset y los spans)."""

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------


def with_noise_sigma(catalog: SensorCatalog, overrides: Mapping[str, float]) -> SensorCatalog:
    """Return a copy of the catalog with some sensors' noise sigma replaced.

    Args:
        catalog: Catalogo base.
        overrides: Tag -> nuevo sigma (> 0).

    Returns:
        A new SensorCatalog; the roles are shared.

    Raises:
        KeyError: If a tag is not in the catalog.
        ValueError: If a sigma is not finite and strictly positive.
    """
    unknown = sorted(set(overrides) - set(catalog.tags))
    if unknown:
        raise KeyError(f"noise sigma override: unknown tags {unknown}.")
    for tag, sigma in overrides.items():
        if not (math.isfinite(sigma) and sigma > 0.0):
            raise ValueError(f"noise sigma override: '{tag}' needs sigma > 0, got {sigma}.")
    sensors: tuple[SensorDef, ...] = tuple(
        dataclasses.replace(s, noise_sigma=float(overrides.get(s.tag, s.noise_sigma)))
        for s in catalog.sensors
    )
    return SensorCatalog(sensors=sensors, roles=catalog.roles)


def _finite(value: float) -> float | None:
    """Return value, or None when it is not finite (JSON has no NaN)."""
    return float(value) if math.isfinite(value) else None


# ----------------------------------------------------------------------
# Resumen de reglas
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RuleMarginStats:
    """Cuanto se acerca una regla a su umbral sobre un conjunto de filas.

    Attributes:
        n_rows: Filas evaluadas por la regla (evaluables o no).
        n_evaluable: Filas en que la regla pudo evaluarse.
        n_violations: Filas en que se violo.
        min_margin: Distancia minima al umbral (negativa si se violo); en las
            unidades de la regla (ver `rule_margins`). None si nada fue evaluable.
        mean_margin: Distancia media al umbral.
        std_margin: Desviacion tipica de la distancia.
    """

    n_rows: int
    n_evaluable: int
    n_violations: int
    min_margin: float | None
    mean_margin: float | None
    std_margin: float | None

    @property
    def sigma_to_threshold(self) -> float | None:
        """Return mean_margin / std_margin: cuantas sigma separan la media del umbral."""
        if self.mean_margin is None or not self.std_margin:
            return None
        return self.mean_margin / self.std_margin

    @property
    def min_margin_sigma(self) -> float | None:
        """Return min_margin / std_margin: el peor caso medido, en sigma."""
        if self.min_margin is None or not self.std_margin:
            return None
        return self.min_margin / self.std_margin


def summarize_rules(
    evaluation: RuleEvaluation, margins: pd.DataFrame
) -> dict[str, RuleMarginStats]:
    """Aggregate violations and margins per rule.

    Args:
        evaluation: Violaciones y evaluabilidad (puede abarcar varios runs).
        margins: Distancias de `rule_margins`, mismo indice y columnas.

    Returns:
        Rule name -> RuleMarginStats, in RULE_NAMES order.
    """
    out: dict[str, RuleMarginStats] = {}
    for rule in RULE_NAMES:
        evaluable = evaluation.evaluable[rule].to_numpy()
        values = margins[rule].to_numpy()[evaluable]
        has = values.size > 0
        out[rule] = RuleMarginStats(
            n_rows=len(evaluation.violations),
            n_evaluable=int(evaluable.sum()),
            n_violations=int(evaluation.violations[rule].sum()),
            min_margin=_finite(float(values.min())) if has else None,
            mean_margin=_finite(float(values.mean())) if has else None,
            std_margin=_finite(float(values.std())) if has else None,
        )
    return out


def _evaluate(
    wide: pd.DataFrame, catalog: SensorCatalog, table: WaterTable, limits: ConstraintLimits
) -> tuple[RuleEvaluation, pd.DataFrame]:
    """Evaluate the seven rules and their margins on one wide frame."""
    evaluation = evaluate_rules_detailed(wide, catalog.roles, table, limits)
    return evaluation, rule_margins(wide, catalog.roles, table, limits)


def _concat(
    parts: Sequence[tuple[RuleEvaluation, pd.DataFrame]],
) -> tuple[RuleEvaluation, pd.DataFrame]:
    """Concatenate per-run evaluations (the rules never mix rows of two runs)."""
    evaluation = RuleEvaluation(
        violations=pd.concat([p[0].violations for p in parts], ignore_index=True),
        evaluable=pd.concat([p[0].evaluable for p in parts], ignore_index=True),
    )
    return evaluation, pd.concat([p[1] for p in parts], ignore_index=True)


# ----------------------------------------------------------------------
# Calibracion de ruido
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseCalibration:
    """Reglas sobre mediciones normales de N runs con unos sigma dados.

    Attributes:
        label: Nombre de la configuracion de ruido.
        sigma_by_tag: Sigma de cada sensor que participa en alguna regla.
        n_runs: Runs evaluados.
        n_rows: Instantes evaluados.
        n_rows_violating: Instantes con alguna regla violada.
        rules: Estadisticos por regla.
    """

    label: str
    sigma_by_tag: dict[str, float]
    n_runs: int
    n_rows: int
    n_rows_violating: int
    rules: dict[str, RuleMarginStats]

    @property
    def total_violations(self) -> int:
        """Return the violations summed over the rules."""
        return sum(stats.n_violations for stats in self.rules.values())


def _rule_tags(catalog: SensorCatalog) -> list[str]:
    """Return the tags that take part in some rule, in catalog order."""
    used = {tag for tags in catalog.roles.values() for tag in tags}
    return [tag for tag in catalog.tags if tag in used]


def noise_calibration(
    sims: Sequence[RunSimulation],
    catalog: SensorCatalog,
    table: WaterTable,
    limits: ConstraintLimits,
    label: str = "catalog",
) -> NoiseCalibration:
    """Measure the rules over noisy normal runs with the catalog's sigmas.

    La integracion no depende del ruido, asi que `sims` se integra una vez y se
    re-mide con cada configuracion de sigma: el flujo de ruido de cada run
    (`plan.rngs()[1]`) es el mismo, solo cambia su escala.

    Args:
        sims: Runs integrados (`simulate_runs`).
        catalog: Catalogo con los sigma a evaluar.
        table: Tabla de agua.
        limits: Umbrales.
        label: Etiqueta de la configuracion.

    Returns:
        The NoiseCalibration.

    Raises:
        ValueError: If sims is empty.
    """
    if not sims:
        raise ValueError("noise calibration needs at least one run.")
    parts: list[tuple[RuleEvaluation, pd.DataFrame]] = []
    for sim in sims:
        _, noise_rng, _ = sim.plan.rngs()
        measured = measure(sim, catalog, noise_rng)
        wide = wide_frame_from_matrix(measured, catalog, sim.timestamps)
        parts.append(_evaluate(wide, catalog, table, limits))
    evaluation, margins = _concat(parts)
    violating = evaluation.violations.any(axis=1)
    return NoiseCalibration(
        label=label,
        sigma_by_tag={tag: catalog.by_tag(tag).noise_sigma for tag in _rule_tags(catalog)},
        n_runs=len(sims),
        n_rows=len(evaluation.violations),
        n_rows_violating=int(violating.sum()),
        rules=summarize_rules(evaluation, margins),
    )


def simulate_runs(
    simulator: ReactorSimulator, n_runs: int | None = None, n_steps: int | None = None
) -> list[RunSimulation]:
    """Integrate the first n_runs runs of the dataset.

    Args:
        simulator: Simulador.
        n_runs: Runs a integrar; por defecto todos los del dataset.
        n_steps: Si se da, pasos por run (para estudios cortos).

    Returns:
        The RunSimulations in run order.
    """
    plans = simulator.plans if n_runs is None else simulator.plans[:n_runs]
    if n_steps is not None:
        plans = [dataclasses.replace(plan, n_steps=n_steps) for plan in plans]
    return [simulator.simulate(plan) for plan in plans]


# ----------------------------------------------------------------------
# Estudio de cadencia
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CadenceResult:
    """Resultado del escenario normal a una cadencia.

    Attributes:
        dt_s: Cadencia [s].
        n_steps: Muestras del escenario.
        n_readings: Lecturas (muestras x sensores).
        true_rows_violating: Muestras con alguna regla violada sobre valores
            verdaderos (esperado 0).
        true_violations_by_rule: Violaciones por regla sobre valores verdaderos.
        max_state_dt_dt_c_s: Maxima derivada de temperatura del estado [C/s].
        max_true_fd_rate_c_s: Maxima |dT/dt| por diferencia finita de T_out y T_in
            verdaderas, que es lo que ve la regla 3 [C/s].
        min_observable_jump_k: Salto minimo que la regla 3 puede ver a esta cadencia [K].
        measured_violations_by_rule: Violaciones sobre mediciones con el ruido del catalogo.
        mass_evaluable_true: Fraccion de muestras en que la regla de masa es
            evaluable sobre valores verdaderos.
        mass_evaluable_measured: Idem sobre mediciones: cae al bajar dt porque el
            ruido de la diferencia finita de T_avg escala con 1/dt.
        dt_avg_noise_k_s: Desviacion tipica de dT_avg/dt medida, ruido MAS senal [K/s];
            a cadencias finas domina el ruido (escala como 1/dt).
        ramp_fraction: Fraccion de muestras en que la carga varia (el escenario
            debe contener rampas; si es 0 el estudio no ejercita nada dinamico).
        validator_faults: Alertas del validador (cuenta de faltas).
        validator_alerted_readings: Lecturas con al menos una alerta.
        validator_faults_by_type: Alertas por tipo de falta.
        validator_faults_by_sensor: Alertas por sensor (solo los que alertan).
        alerts_per_1000: Alertas por 1000 lecturas.
        alerts_per_1000_hold: Idem restringido a muestras con carga constante.
        alerts_per_1000_ramp: Idem restringido a muestras con carga variando.
    """

    dt_s: float
    n_steps: int
    n_readings: int
    true_rows_violating: int
    true_violations_by_rule: dict[str, int]
    max_state_dt_dt_c_s: float
    max_true_fd_rate_c_s: float
    min_observable_jump_k: float
    measured_violations_by_rule: dict[str, int]
    mass_evaluable_true: float
    mass_evaluable_measured: float
    dt_avg_noise_k_s: float
    ramp_fraction: float
    validator_faults: int
    validator_alerted_readings: int
    validator_faults_by_type: dict[str, int]
    validator_faults_by_sensor: dict[str, int]
    alerts_per_1000: float
    alerts_per_1000_hold: float | None
    alerts_per_1000_ramp: float | None


ValidatorFactory = Callable[[dict[str, SensorSpan]], Any]


def default_validator_factory(spans: dict[str, SensorSpan]) -> Any:
    """Build the Phase 2 SensorValidator (lazy import: only the study needs it).

    Args:
        spans: Span table of the simulation sensors.

    Returns:
        A `SensorValidator`.
    """
    from data.validation.sensor_validator import SensorValidator

    return SensorValidator(spans)


def _per_1000(faults: int, readings: int) -> float | None:
    """Return faults per 1000 readings, or None without readings."""
    return None if readings == 0 else 1000.0 * faults / readings


def cadence_run(
    simulator: ReactorSimulator,
    dt_s: float,
    duration_s: float = DEFAULT_STUDY_DURATION_S,
    seed: int = CADENCE_STUDY_SEED,
    validator_factory: ValidatorFactory = default_validator_factory,
    warmup_s: float = STUDY_WARMUP_S,
    hold_minutes: tuple[float, float] = STUDY_HOLD_MINUTES,
) -> CadenceResult:
    """Run the normal scenario at one cadence and measure what is lost.

    El escenario es el MISMO en tiempo fisico para todas las cadencias (mismo
    flujo de carga, mismo calentamiento y mismas esperas en segundos); solo cambia cada cuanto se
    muestrea. Los umbrales de las reglas y del validador NO se recalibran: el
    estudio mide lo que pasa cuando la cadencia cambia y lo demas no.

    Args:
        simulator: Simulador (modelo, catalogo, spans y limites).
        dt_s: Cadencia a estudiar [s].
        duration_s: Duracion simulada del escenario [s].
        seed: Semilla del escenario.
        validator_factory: Constructor del validador (inyectable en tests).
        warmup_s: Tramo inicial a carga constante [s].
        hold_minutes: (minimo, maximo) de espera entre rampas [min].

    Returns:
        The CadenceResult.

    Raises:
        ValueError: If dt_s <= 0 or the scenario has fewer than 2 samples.
    """
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    params = simulator.params
    n_steps = int(round(duration_s / dt_s)) + 1
    if n_steps < 2:
        raise ValueError(f"duration {duration_s} s gives {n_steps} samples at {dt_s} s.")
    catalog, model = simulator.catalog, simulator.model
    table = model.table
    limits = ConstraintLimits.from_plant(model.plant)

    plan = RunPlan(
        run_index=0,
        run_id="sim_run_cadence",
        scenario=ScenarioKind.NORMAL,
        n_steps=n_steps,
        start_time=params.start_time,
        seed_entropy=seed,
        spawn_key=(0,),
    )
    profile = dataclasses.replace(
        simulator.profile,
        warmup_steps=round(warmup_s / dt_s),
        hold_minutes_min=hold_minutes[0],
        hold_minutes_max=hold_minutes[1],
    )
    load_rng, noise_rng, _ = plan.rngs()
    loads = profile.sample(load_rng, n_steps, dt_s)
    inputs = [ModelInputs(load=float(load)) for load in loads]
    sim = simulate_run(model, catalog, plan, inputs, dt_s)

    # (a) y (b): reglas y derivada sobre el estado verdadero.
    true_wide = wide_frame_from_matrix(sim.true_values, catalog, sim.timestamps)
    true_eval, _ = _evaluate(true_wide, catalog, table, limits)
    out_tags = list(catalog.roles["T_out"]) + list(catalog.roles["T_in"])
    elapsed = np.full(n_steps, dt_s, dtype=np.float64)
    elapsed[0] = np.nan
    fd_rate = np.abs(finite_difference_rate(true_wide[out_tags].to_numpy(), elapsed))
    max_fd = float(np.nanmax(fd_rate)) if np.isfinite(fd_rate).any() else 0.0

    # Mediciones con el ruido del catalogo: reglas y evaluabilidad de la regla de masa.
    measured = measure(sim, catalog, noise_rng)
    measured_wide = wide_frame_from_matrix(measured, catalog, sim.timestamps)
    measured_eval, _ = _evaluate(measured_wide, catalog, table, limits)
    t_avg = 0.5 * (
        measured_wide[list(catalog.roles["T_out"])].median(axis=1).to_numpy()
        + measured_wide[list(catalog.roles["T_in"])].median(axis=1).to_numpy()
    )
    avg_rate = finite_difference_rate(t_avg, elapsed)
    finite_rate = avg_rate[np.isfinite(avg_rate)]
    rate_noise = float(finite_rate.std()) if finite_rate.size else float("nan")

    # (c) alertas del validador sobre el frame largo, sin recalibrar nada.
    frame = assemble_long_frame(sim, measured, catalog, simulator.spans, params.plant_id)
    validator = validator_factory(simulator.spans)
    n_sensors = len(catalog.sensors)
    ramp_step = np.zeros(n_steps, dtype=np.bool_)
    ramp_step[1:] = np.abs(np.diff(loads)) > 1e-12
    faults_by_step = np.zeros(n_steps, dtype=np.int64)
    faults_by_sensor: dict[str, int] = {}
    alerted = 0
    for i, reading in enumerate(readings_from_frame(frame)):
        faults = validator.validate(reading).faults
        if faults:
            alerted += 1
            faults_by_step[i // n_sensors] += len(faults)
            faults_by_sensor[reading.sensor.id] = faults_by_sensor.get(
                reading.sensor.id, 0
            ) + len(faults)
    metrics = validator.get_metrics()
    n_readings = n_steps * n_sensors
    hold = ~ramp_step
    faults_total = int(faults_by_step.sum())

    return CadenceResult(
        dt_s=dt_s,
        n_steps=n_steps,
        n_readings=n_readings,
        true_rows_violating=int(true_eval.violations.any(axis=1).sum()),
        true_violations_by_rule={r: int(true_eval.violations[r].sum()) for r in RULE_NAMES},
        max_state_dt_dt_c_s=max(d.max_temperature_rate_c_s for d in sim.diagnostics),
        max_true_fd_rate_c_s=max_fd,
        min_observable_jump_k=min_observable_rate(dt_s, limits.max_dt_c_per_s),
        measured_violations_by_rule={r: int(measured_eval.violations[r].sum()) for r in RULE_NAMES},
        mass_evaluable_true=float(true_eval.evaluable["mass_balance"].mean()),
        mass_evaluable_measured=float(measured_eval.evaluable["mass_balance"].mean()),
        dt_avg_noise_k_s=rate_noise,
        ramp_fraction=float(ramp_step.mean()),
        validator_faults=faults_total,
        validator_alerted_readings=alerted,
        validator_faults_by_type={str(k): int(v) for k, v in metrics["faults_by_type"].items()},
        validator_faults_by_sensor=dict(sorted(faults_by_sensor.items())),
        alerts_per_1000=float(_per_1000(faults_total, n_readings) or 0.0),
        alerts_per_1000_hold=_per_1000(
            int(faults_by_step[hold].sum()), int(hold.sum()) * n_sensors
        ),
        alerts_per_1000_ramp=_per_1000(
            int(faults_by_step[ramp_step].sum()), int(ramp_step.sum()) * n_sensors
        ),
    )


def cadence_study(
    simulator: ReactorSimulator,
    cadences_s: Sequence[float] = DEFAULT_CADENCES_S,
    duration_s: float = DEFAULT_STUDY_DURATION_S,
    seed: int = CADENCE_STUDY_SEED,
    validator_factory: ValidatorFactory = default_validator_factory,
) -> list[CadenceResult]:
    """Run `cadence_run` for each cadence (D1).

    Args:
        simulator: Simulador.
        cadences_s: Cadencias a estudiar [s].
        duration_s: Duracion simulada del escenario [s].
        seed: Semilla del escenario.
        validator_factory: Constructor del validador.

    Returns:
        One CadenceResult per cadence, in the order given.
    """
    results: list[CadenceResult] = []
    for dt_s in cadences_s:
        _LOG.info("Cadence study at dt=%g s (%g s simulated).", dt_s, duration_s)
        results.append(cadence_run(simulator, dt_s, duration_s, seed, validator_factory))
    return results


# ----------------------------------------------------------------------
# Matriz de violaciones de los fallos de proceso (F3-M4)
# ----------------------------------------------------------------------


DEFAULT_FAULT_TAIL_STEPS = 30
"""Muestras de operacion normal que siguen a la ventana del fallo en la matriz."""

_LIMIT_TOLERANCE_BAR = 0.05
"""Distancia a un limite de presion bajo la cual una muestra cuenta como limitada."""


@dataclass(frozen=True)
class FaultRuleCell:
    """Cuanto dispara una regla dentro de la ventana de un fallo.

    Attributes:
        n_window_steps: Muestras de la ventana etiquetada.
        n_evaluable: Muestras de la ventana en que la regla pudo evaluarse.
        n_violations: Muestras de la ventana en que se violo.
        first_delay_steps: Muestras entre el inicio del fallo y la primera
            violacion; None si la regla no se violo.
    """

    n_window_steps: int
    n_evaluable: int
    n_violations: int
    first_delay_steps: int | None

    @property
    def rate(self) -> float | None:
        """Return n_violations / n_evaluable, or None when nothing was evaluable."""
        return None if self.n_evaluable == 0 else self.n_violations / self.n_evaluable


@dataclass(frozen=True)
class FaultRunResult:
    """Medida de un run con fallo de proceso.

    Attributes:
        scenario: Escenario (`core_temp_excursion`, ...).
        level: Nivel (`low`, `medium`, `high`).
        run_index: Posicion del run (da la carga y el ruido).
        run_steps: Muestras del run (ventana mas cola de operacion normal).
        injected_step: Primera muestra del fallo.
        last_active_step: Ultima muestra etiquetada.
        response: Respuesta verdadera desde la inyeccion hasta el final del run.
        max_fd_rate_c_s: Maxima |dT/dt| por diferencia finita de T_out y T_in
            verdaderas entre muestras (lo que ve la regla 3 a esta cadencia).
        state_rate_violation_steps: Muestras de la ventana en que la derivada
            INSTANTANEA del estado supera el limite de la regla 3 (D9).
        state_imbalance_violation_steps: Muestras de la ventana en que el
            desbalance instantaneo del nucleo supera el de la regla 1.
        max_energy_residual: Peor cierre de energia por intervalo.
        pressure_limited_steps: Muestras de la ventana en que la presion esta a
            menos de `_LIMIT_TOLERANCE_BAR` de un limite del limitador del modelo
            (el transitorio llego al borde de la tabla de agua).
        conservation_compliant: Conservacion sobre el estado completo (D7).
        true_rules: Regla -> celda sobre valores verdaderos, dentro de la ventana.
        measured_rules: Idem sobre mediciones con ruido.
        outside_true: Regla -> violaciones verdaderas FUERA de la ventana.
        outside_measured: Idem sobre mediciones.
        seconds: Tiempo de integracion y evaluacion del run.
    """

    scenario: str
    level: str
    run_index: int
    run_steps: int
    injected_step: int
    last_active_step: int
    response: LevelResponse
    max_fd_rate_c_s: float
    state_rate_violation_steps: int
    state_imbalance_violation_steps: int
    max_energy_residual: float
    pressure_limited_steps: int
    conservation_compliant: bool
    true_rules: dict[str, FaultRuleCell]
    measured_rules: dict[str, FaultRuleCell]
    outside_true: dict[str, int]
    outside_measured: dict[str, int]
    seconds: float


def _window_cells(
    evaluation: RuleEvaluation, injected_step: int, last_step: int
) -> tuple[dict[str, FaultRuleCell], dict[str, int]]:
    """Split an evaluation into the fault window and the rest.

    Args:
        evaluation: Violaciones y evaluabilidad de un run.
        injected_step: Primera muestra de la ventana.
        last_step: Ultima muestra de la ventana (incluida).

    Returns:
        (rule -> window cell, rule -> violations outside the window).
    """
    n_rows = len(evaluation.violations)
    inside = np.zeros(n_rows, dtype=np.bool_)
    inside[injected_step : last_step + 1] = True
    cells: dict[str, FaultRuleCell] = {}
    outside: dict[str, int] = {}
    for rule in RULE_NAMES:
        violated = evaluation.violations[rule].to_numpy()
        evaluable = evaluation.evaluable[rule].to_numpy()
        hits = np.flatnonzero(violated & inside)
        cells[rule] = FaultRuleCell(
            n_window_steps=int(inside.sum()),
            n_evaluable=int((evaluable & inside).sum()),
            n_violations=int(hits.size),
            first_delay_steps=int(hits[0]) - injected_step if hits.size else None,
        )
        outside[rule] = int((violated & ~inside).sum())
    return cells, outside


def fault_run(
    simulator: ReactorSimulator,
    scenario: ScenarioKind,
    level: FaultLevel,
    *,
    run_index: int = 0,
    seed: int | None = None,
    hold_steps: int | None = None,
    recovery_steps: int | None = None,
    tail_steps: int = DEFAULT_FAULT_TAIL_STEPS,
) -> FaultRunResult:
    """Run one process fault and measure which rules it violates and when.

    El run acaba `tail_steps` muestras despues de la ventana, de modo que la
    recuperacion y la vuelta a la operacion normal quedan medidas. El ruido es el
    del catalogo (los umbrales de las reglas no se tocan).

    Args:
        simulator: Simulador.
        scenario: Uno de los tres escenarios de proceso.
        level: Nivel de magnitud.
        run_index: Posicion del run (carga y ruido).
        seed: Semilla maestra; por defecto la de `params`.
        hold_steps: Pasos a magnitud plena; por defecto los de `fault_tiers`.
        recovery_steps: Pasos de recuperacion; por defecto los de `fault_tiers`.
        tail_steps: Muestras de operacion normal tras la ventana.

    Returns:
        The FaultRunResult.

    Raises:
        KeyError: If the scenario is not a process fault.
        ValueError: If the window does not fit in the run.
    """
    started = time.perf_counter()
    plan = simulator.process_plan(
        scenario,
        level,
        run_index=run_index,
        seed=seed,
        hold_steps=hold_steps,
        recovery_steps=recovery_steps,
    )
    spec = plan.process_fault
    assert spec is not None and spec.last_active_step is not None
    last = spec.last_active_step
    plan = dataclasses.replace(plan, n_steps=min(plan.n_steps, last + 1 + tail_steps))

    model, catalog = simulator.model, simulator.catalog
    limits = ConstraintLimits.from_plant(model.plant)
    sim = simulator.simulate(plan)
    measured = measure(sim, catalog, plan.rngs()[1])
    true_wide = wide_frame_from_matrix(sim.true_values, catalog, sim.timestamps)
    measured_wide = wide_frame_from_matrix(measured, catalog, sim.timestamps)
    true_eval, _ = _evaluate(true_wide, catalog, model.table, limits)
    measured_eval, _ = _evaluate(measured_wide, catalog, model.table, limits)
    true_rules, outside_true = _window_cells(true_eval, spec.injected_step, last)
    measured_rules, outside_measured = _window_cells(measured_eval, spec.injected_step, last)

    elapsed = np.full(plan.n_steps, simulator.params.dt_seconds, dtype=np.float64)
    elapsed[0] = np.nan
    tags = list(catalog.roles["T_out"]) + list(catalog.roles["T_in"])
    fd_rate = np.abs(finite_difference_rate(true_wide[tags].to_numpy(), elapsed))
    window = sim.diagnostics[spec.injected_step : last + 1]
    report = conservation_report(sim.diagnostics, sim.steps, limits)
    p_floor, p_ceiling = model.pressure_limits_bar
    pressure = sim.states[spec.injected_step : last + 1, StateIndex.PRESSURE]
    limited = int(
        np.sum(
            (pressure <= p_floor + _LIMIT_TOLERANCE_BAR)
            | (pressure >= p_ceiling - _LIMIT_TOLERANCE_BAR)
        )
    )
    return FaultRunResult(
        scenario=scenario.value,
        level=level.value,
        run_index=run_index,
        run_steps=plan.n_steps,
        injected_step=spec.injected_step,
        last_active_step=last,
        response=response_from_run(sim, model, spec.injected_step),
        max_fd_rate_c_s=float(np.nanmax(fd_rate)) if np.isfinite(fd_rate).any() else 0.0,
        state_rate_violation_steps=sum(
            d.max_temperature_rate_c_s > limits.max_dt_c_per_s for d in window
        ),
        state_imbalance_violation_steps=sum(
            d.energy_imbalance_fraction > limits.energy_imbalance_fraction for d in window
        ),
        max_energy_residual=sim.max_energy_residual,
        pressure_limited_steps=limited,
        conservation_compliant=report.compliant,
        true_rules=true_rules,
        measured_rules=measured_rules,
        outside_true=outside_true,
        outside_measured=outside_measured,
        seconds=time.perf_counter() - started,
    )


def fault_matrix(
    simulator: ReactorSimulator,
    scenarios: Sequence[ScenarioKind] = PROCESS_KINDS,
    levels: Sequence[FaultLevel] = tuple(FaultLevel),
    n_repeats: int = 3,
    hold_steps: int | None = None,
    recovery_steps: int | None = None,
) -> list[FaultRunResult]:
    """Run every (scenario, level) cell `n_repeats` times with different loads.

    Cada repeticion usa un `run_index` distinto: otra traza de carga y otro
    ruido, de modo que la celda no depende de una unica carga inicial.

    Args:
        simulator: Simulador.
        scenarios: Escenarios de proceso a medir.
        levels: Niveles a medir.
        n_repeats: Runs por celda (>= 1).
        hold_steps: Pasos a magnitud plena; por defecto los de `fault_tiers`.
        recovery_steps: Pasos de recuperacion; por defecto los de `fault_tiers`.

    Returns:
        One FaultRunResult per run, grouped by scenario, level and repetition.

    Raises:
        ValueError: If n_repeats < 1.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}.")
    results: list[FaultRunResult] = []
    for scenario in scenarios:
        for level in levels:
            for repeat in range(n_repeats):
                _LOG.info("Fault run %s/%s #%d", scenario.value, level.value, repeat)
                results.append(
                    fault_run(
                        simulator,
                        scenario,
                        level,
                        run_index=repeat,
                        hold_steps=hold_steps,
                        recovery_steps=recovery_steps,
                    )
                )
    return results


def _rule_summary(cells: Sequence[Mapping[str, FaultRuleCell]]) -> dict[str, dict[str, Any]]:
    """Aggregate the window cells of several runs, rule by rule."""
    out: dict[str, dict[str, Any]] = {}
    for rule in RULE_NAMES:
        parts = [cell[rule] for cell in cells]
        evaluable = sum(p.n_evaluable for p in parts)
        violations = sum(p.n_violations for p in parts)
        delays = [p.first_delay_steps for p in parts if p.first_delay_steps is not None]
        out[rule] = {
            "n_window_steps": sum(p.n_window_steps for p in parts),
            "n_evaluable": evaluable,
            "n_violations": violations,
            "rate": violations / evaluable if evaluable else None,
            "runs_violating": len(delays),
            "first_delay_steps_min": min(delays) if delays else None,
            "first_delay_steps_max": max(delays) if delays else None,
        }
    return out


def summarize_fault_matrix(results: Sequence[FaultRunResult]) -> dict[str, Any]:
    """Aggregate the runs into one entry per (scenario, level) cell.

    Args:
        results: Runs de `fault_matrix`.

    Returns:
        Mapping `"<scenario>/<level>"` -> counts, rates, delays and the extremes
        of the true response across the cell's runs.
    """
    groups: dict[str, list[FaultRunResult]] = {}
    for result in results:
        groups.setdefault(f"{result.scenario}/{result.level}", []).append(result)
    summary: dict[str, Any] = {}
    for key, runs in groups.items():
        peak = {
            f.name: [
                min(getattr(r.response, f.name) for r in runs),
                max(getattr(r.response, f.name) for r in runs),
            ]
            for f in dataclasses.fields(LevelResponse)
        }
        summary[key] = {
            "n_runs": len(runs),
            "window_steps": sum(r.last_active_step - r.injected_step + 1 for r in runs),
            "response_min_max": peak,
            "max_fd_rate_c_s": max(r.max_fd_rate_c_s for r in runs),
            "state_rate_violation_steps": sum(r.state_rate_violation_steps for r in runs),
            "state_imbalance_violation_steps": sum(
                r.state_imbalance_violation_steps for r in runs
            ),
            "pressure_limited_steps": sum(r.pressure_limited_steps for r in runs),
            "max_energy_residual": max(r.max_energy_residual for r in runs),
            "conservation_compliant": all(r.conservation_compliant for r in runs),
            "true": _rule_summary([r.true_rules for r in runs]),
            "measured": _rule_summary([r.measured_rules for r in runs]),
            "outside_true": {
                rule: sum(r.outside_true[rule] for r in runs) for rule in RULE_NAMES
            },
            "outside_measured": {
                rule: sum(r.outside_measured[rule] for r in runs) for rule in RULE_NAMES
            },
            "seconds": sum(r.seconds for r in runs),
        }
    return summary


# ----------------------------------------------------------------------
# Fallos de instrumento: matriz de reglas y deteccion del validador (F3-M5)
# ----------------------------------------------------------------------


DEFAULT_INSTRUMENT_START_FRACTION = 0.40
"""Instante de inicio de los episodios de la matriz, como fraccion del run.

Cae en el rango de inicio del sesgo (25-50 %) y deja tras el calentamiento una
ventana previa de operacion normal."""

@dataclass(frozen=True)
class InstrumentRuleCell:
    """Reglas que viola un episodio de instrumento en un sensor.

    Attributes:
        kind: Tipo (`drift`, `stuck`, ...).
        level: Nivel de magnitud.
        sensor_id: Sensor manipulado.
        direction: +1 o -1 (sentido del desplazamiento o carril).
        in_roles: Roles de las constraints que lee este sensor (vacio si ninguno).
        start_step: Primera muestra del episodio.
        last_step: Ultima muestra del episodio.
        voted: Regla -> celda con la votacion por mediana del rol (el Rule Engine
            de `physics_constraints` tal cual).
        per_channel: Regla -> celda con el sensor como UNICO canal de sus roles
            (lo que veria una evaluacion por canal).
        outside_voted: Regla -> violaciones votadas fuera de la ventana.
        outside_per_channel: Idem por canal.
    """

    kind: str
    level: str
    sensor_id: str
    direction: int
    in_roles: tuple[str, ...]
    start_step: int
    last_step: int
    voted: dict[str, FaultRuleCell]
    per_channel: dict[str, FaultRuleCell]
    outside_voted: dict[str, int]
    outside_per_channel: dict[str, int]


def single_channel_roles(
    roles: Mapping[str, tuple[str, ...]], tag: str
) -> dict[str, tuple[str, ...]]:
    """Return the roles with one sensor as the ONLY channel of the roles it plays.

    Quita la votacion para ese sensor: las reglas de relacion comparan su valor
    directamente con el de los demas roles, de modo que un canal averiado ya no
    queda enmascarado por la mediana de sus redundantes.

    Args:
        roles: Rol -> tags del catalogo.
        tag: Sensor aislado.

    Returns:
        The roles; those that do not contain `tag` are unchanged.
    """
    return {name: ((tag,) if tag in members else tuple(members)) for name, members in roles.items()}


def _episode_for(
    simulator: ReactorSimulator,
    kind: InstrumentFaultKind,
    level: FaultLevel,
    sensor_id: str,
    start_step: int,
    n_steps: int,
    direction: int,
) -> InstrumentEpisode:
    """Return a mid-range episode of a (kind, level) on a sensor.

    Para medir se usa el punto medio del rango de magnitud del nivel (no un
    sorteo) y el sentido se fija: la matriz es una funcion del nivel y no del azar.
    La ventana se recorta al final del run, de modo que un estudio con runs mas
    cortos que los del dataset sigue siendo posible (la ventana medida es la que
    cabe).
    """
    tiers = simulator.scenario_config.instrument.for_kind(kind)
    spec = tiers.level(level)
    length = min(tiers.window_steps(level, start_step, n_steps), n_steps - start_step)
    return InstrumentEpisode(
        sensor_id=sensor_id,
        kind=kind,
        level=level,
        start_step=start_step,
        length_steps=length,
        magnitude=spec.magnitude_mid,
        direction=direction,
        p_enter=spec.p_enter,
        p_exit=spec.p_exit,
    )


def instrument_matrix(
    simulator: ReactorSimulator,
    *,
    run_index: int = 0,
    n_steps: int | None = None,
    start_fraction: float = DEFAULT_INSTRUMENT_START_FRACTION,
    kinds: Sequence[InstrumentFaultKind] = INSTRUMENT_KINDS,
    levels: Sequence[FaultLevel] = tuple(FaultLevel),
    directions: Sequence[int] = (1, -1),
    sensors: Sequence[str] | None = None,
    control_sensors: Sequence[str] = ("TC-CORE-01",),
) -> list[InstrumentRuleCell]:
    """Measure which rules each instrument fault violates, sensor by sensor.

    Un unico run normal se integra y se mide una vez; cada celda es esa misma
    medicion con UN episodio aplicado a UN sensor. Solo los sensores que algun
    rol de las constraints lee pueden alterar una regla, asi que por defecto se
    evaluan esos y un sensor de control fuera de los roles (que debe dar 0).
    Cada celda se evalua dos veces: con la votacion por mediana del rol (el
    Rule Engine de F3-M2) y con el sensor aislado como unico canal.

    Args:
        simulator: Simulador.
        run_index: Run normal sobre el que se aplican los episodios.
        n_steps: Pasos del run; por defecto `run_length_steps`.
        start_fraction: Inicio del episodio como fraccion del run.
        kinds: Tipos a medir.
        levels: Niveles a medir.
        directions: Sentidos del desplazamiento (+1, -1).
        sensors: Sensores a manipular; por defecto los de los roles.
        control_sensors: Sensores fuera de los roles que se miden como control.

    Returns:
        One InstrumentRuleCell per (kind, level, sensor, direction).

    Raises:
        ValueError: If the run is too short for the episodes or `directions`
            is empty.
    """
    if not directions:
        raise ValueError("directions must not be empty.")
    model, catalog, spans = simulator.model, simulator.catalog, simulator.spans
    limits = ConstraintLimits.from_plant(model.plant)
    plan = make_run_plan(simulator.params, run_index, ScenarioKind.NORMAL)
    if n_steps is not None:
        plan = dataclasses.replace(plan, n_steps=n_steps)
    sim = simulator.simulate(plan)
    _, noise_rng, _ = plan.rngs()
    measured = measure(sim, catalog, noise_rng)
    start = max(1, round(start_fraction * plan.n_steps))
    role_tags = [tag for tag in catalog.tags if any(tag in m for m in catalog.roles.values())]
    targets = list(sensors) if sensors is not None else role_tags + list(control_sensors)
    column = {tag: j for j, tag in enumerate(catalog.tags)}
    rng = np.random.default_rng(0)
    cells: list[InstrumentRuleCell] = []
    for kind in kinds:
        for level in levels:
            for tag in targets:
                for direction in directions:
                    episode = _episode_for(
                        simulator, kind, level, tag, start, plan.n_steps, direction
                    )
                    effect = apply_episode(
                        measured[:, column[tag]],
                        episode,
                        spans[tag],
                        catalog.by_tag(tag).noise_sigma,
                        simulator.params.dt_seconds,
                        rng,
                    )
                    values = measured.copy()
                    values[:, column[tag]] = effect.values
                    wide = wide_frame_from_matrix(values, catalog, sim.timestamps)
                    voted_eval, _ = _evaluate(wide, catalog, model.table, limits)
                    isolated = SensorCatalog(
                        sensors=catalog.sensors,
                        roles=single_channel_roles(catalog.roles, tag),
                    )
                    single_eval, _ = _evaluate(wide, isolated, model.table, limits)
                    voted, outside_voted = _window_cells(voted_eval, start, episode.last_step)
                    single, outside_single = _window_cells(single_eval, start, episode.last_step)
                    cells.append(
                        InstrumentRuleCell(
                            kind=kind.value,
                            level=level.value,
                            sensor_id=tag,
                            direction=direction,
                            in_roles=tuple(
                                sorted(r for r, m in catalog.roles.items() if tag in m)
                            ),
                            start_step=start,
                            last_step=episode.last_step,
                            voted=voted,
                            per_channel=single,
                            outside_voted=outside_voted,
                            outside_per_channel=outside_single,
                        )
                    )
    return cells


def summarize_instrument_matrix(cells: Sequence[InstrumentRuleCell]) -> dict[str, Any]:
    """Aggregate the instrument matrix per (kind, level).

    Cada entrada separa los sensores que algun rol lee (`role_sensors`) de los
    que no (`non_role_sensors`, que no pueden alterar ninguna regla), y por cada
    una da la evaluacion votada y la por canal.

    Args:
        cells: Celdas de `instrument_matrix`.

    Returns:
        Mapping `"<kind>/<level>"` -> counts and rates per rule.
    """
    groups: dict[str, list[InstrumentRuleCell]] = {}
    for cell in cells:
        groups.setdefault(f"{cell.kind}/{cell.level}", []).append(cell)
    summary: dict[str, Any] = {}
    for key, members in groups.items():
        in_role = [c for c in members if c.in_roles]
        out_role = [c for c in members if not c.in_roles]
        by_sensor: dict[str, Any] = {}
        for tag in sorted({c.sensor_id for c in in_role}):
            mine = [c for c in in_role if c.sensor_id == tag]
            by_sensor[tag] = {
                "roles": list(mine[0].in_roles),
                "voted": _rule_summary([c.voted for c in mine]),
                "per_channel": _rule_summary([c.per_channel for c in mine]),
            }
        entry: dict[str, Any] = {
            "n_cells": len(members),
            "role_sensors": {
                "n_cells": len(in_role),
                "voted": _rule_summary([c.voted for c in in_role]) if in_role else {},
                "per_channel": (
                    _rule_summary([c.per_channel for c in in_role]) if in_role else {}
                ),
                "by_sensor": by_sensor,
            },
            "non_role_sensors": {
                "n_cells": len(out_role),
                "voted": _rule_summary([c.voted for c in out_role]) if out_role else {},
            },
            "outside_voted": {
                rule: sum(c.outside_voted[rule] for c in members) for rule in RULE_NAMES
            },
            "outside_per_channel": {
                rule: sum(c.outside_per_channel[rule] for c in members) for rule in RULE_NAMES
            },
        }
        summary[key] = entry
    return summary


# Tipo de fallo del validador que cada manipulacion deberia producir (verdad-terreno
# independiente del detector: solo para informar la tasa de acierto por tipo).
EXPECTED_VALIDATOR_FAULT: dict[str, tuple[str, ...]] = {
    "drift": ("sensor_drift",),
    "stuck": ("stuck",),
    "noise_spike": ("noise_spike",),
    "bias": ("bias_out_of_range",),
    "intermittent": ("bias_out_of_range",),
}
"""`FaultType.value` del validador de Fase 2 que cabe esperar de cada manipulacion."""


@dataclass(frozen=True)
class DetectionCell:
    """Deteccion informativa del validador de Fase 2 sobre un tipo y nivel.

    El validador no se toca ni se recalibra: esto mide lo que hace a 180 s sobre
    la simulacion. Los canales de caudal se reportan aparte porque su detector de
    Kalman alerta ~49 % en operacion normal (F3-M3, deuda de Fase 2).

    Attributes:
        kind: Tipo.
        level: Nivel.
        n_episodes: Episodios inyectados.
        n_episodes_excl_flow: Episodios fuera del grupo de caudal.
        labelled_readings: Lecturas etiquetadas.
        episodes_detected: Episodios con alguna alerta en su ventana.
        episodes_detected_excl_flow: Idem sin caudal.
        readings_flagged: Lecturas etiquetadas con alguna alerta.
        readings_flagged_expected: Lecturas etiquetadas con el tipo esperado.
        faults_by_type: Alertas por tipo del validador en las ventanas.
        first_delay_steps: Muestras hasta la primera alerta, por episodio detectado.
        false_positive_per_1000: Lecturas alertadas por mil, fuera de episodios y
            sin caudal.
        false_positive_per_1000_flow: Idem en los canales de caudal.
        episodes_by_group: Grupo -> (detectados, total).
    """

    kind: str
    level: str
    n_episodes: int
    n_episodes_excl_flow: int
    labelled_readings: int
    episodes_detected: int
    episodes_detected_excl_flow: int
    readings_flagged: int
    readings_flagged_expected: int
    faults_by_type: dict[str, int]
    first_delay_steps: list[int]
    false_positive_per_1000: float | None
    false_positive_per_1000_flow: float | None
    episodes_by_group: dict[str, tuple[int, int]]


def _one_sensor_per_group(
    catalog: SensorCatalog, repeat: int, exclude: Collection[str] = ()
) -> dict[str, str]:
    """Return one sensor of each sensor type, rotated by `repeat`.

    Args:
        catalog: Catalogo.
        repeat: Desplazamiento de la rotacion (otra repeticion, otro sensor).
        exclude: Tags que no se pueden elegir.

    Returns:
        Mapping sensor type -> tag.
    """
    members: dict[str, list[str]] = {}
    for sensor in catalog.sensors:
        if sensor.tag not in exclude:
            members.setdefault(sensor.sensor_type.value, []).append(sensor.tag)
    return {g: tags[repeat % len(tags)] for g, tags in members.items()}


def instrument_detection(
    simulator: ReactorSimulator,
    *,
    n_repeats: int = 2,
    n_steps: int | None = None,
    kinds: Sequence[InstrumentFaultKind] = INSTRUMENT_KINDS,
    levels: Sequence[FaultLevel] = tuple(FaultLevel),
    start_fraction: float = DEFAULT_INSTRUMENT_START_FRACTION,
    spikes_per_sensor: int = 10,
    validator_factory: ValidatorFactory = default_validator_factory,
) -> list[DetectionCell]:
    """Measure what the Phase 2 validator sees of each injected instrument fault.

    Cada repeticion integra un run normal y, para cada (tipo, nivel), inyecta un
    episodio en UN sensor de cada grupo (distintos entre repeticiones) y pasa el
    frame completo por un `SensorValidator` nuevo, sin recalibrar nada. Es
    informativo: los falsos negativos esperados (intermitente, sesgo dentro del
    sobre, deriva sub-detectable) son parte de la medida.

    Args:
        simulator: Simulador.
        n_repeats: Repeticiones (run normal y sensores distintos).
        n_steps: Pasos del run; por defecto `run_length_steps`.
        kinds: Tipos a medir.
        levels: Niveles a medir.
        start_fraction: Inicio de los episodios como fraccion del run.
        spikes_per_sensor: Picos por sensor del episodio de `noise_spike`
            (separados por varias muestras: un pico aislado es un caso de
            deteccion y diez son una tasa).
        validator_factory: Constructor del validador (inyectable en tests).

    Returns:
        One DetectionCell per (kind, level), aggregated over the repeats.

    Raises:
        ValueError: If n_repeats < 1.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}.")
    catalog, spans, params = simulator.catalog, simulator.spans, simulator.params
    group_of = {s.tag: s.sensor_type.value for s in catalog.sensors}
    column = {tag: j for j, tag in enumerate(catalog.tags)}
    n_sensors = len(catalog.sensors)
    totals: dict[tuple[str, str], dict[str, Any]] = {}

    for repeat in range(n_repeats):
        base_plan = make_run_plan(params, repeat, ScenarioKind.NORMAL)
        if n_steps is not None:
            base_plan = dataclasses.replace(base_plan, n_steps=n_steps)
        sim = simulator.simulate(base_plan)
        _, noise_rng, _ = base_plan.rngs()
        measured = measure(sim, catalog, noise_rng)
        start = max(1, round(start_fraction * base_plan.n_steps))
        picked = _one_sensor_per_group(catalog, repeat)
        for kind in kinds:
            for level in levels:
                episodes: list[InstrumentEpisode] = []
                for tag in picked.values():
                    if kind is InstrumentFaultKind.NOISE_SPIKE:
                        room = base_plan.n_steps - start
                        gap = max(2, room // (spikes_per_sensor + 1))
                        n_spikes = min(spikes_per_sensor, (room - 1) // gap + 1)
                        for i in range(n_spikes):
                            episodes.append(
                                _episode_for(
                                    simulator, kind, level, tag, start + i * gap,
                                    base_plan.n_steps, 1 if i % 2 == 0 else -1,
                                )
                            )
                    else:
                        episodes.append(
                            _episode_for(
                                simulator, kind, level, tag, start, base_plan.n_steps,
                                1 if repeat % 2 == 0 else -1,
                            )
                        )
                plan = dataclasses.replace(
                    base_plan,
                    scenario=ScenarioKind(kind.value),
                    instrument_faults=tuple(episodes),
                )
                effect = apply_instrument_faults(
                    measured, catalog, spans, plan, params.dt_seconds,
                    np.random.default_rng(repeat),
                )
                frame = assemble_long_frame(
                    dataclasses.replace(sim, plan=plan),
                    effect.values,
                    catalog,
                    spans,
                    params.plant_id,
                    corrupted=effect.corrupted,
                )
                faults = _validator_faults(frame, validator_factory(spans), n_sensors)
                bucket = totals.setdefault(
                    (kind.value, level.value),
                    {
                        "n_episodes": 0, "n_excl": 0, "labelled": 0, "det": 0, "det_excl": 0,
                        "flagged": 0, "flagged_expected": 0, "by_type": {}, "delays": [],
                        "fp": 0, "fp_n": 0, "fp_flow": 0, "fp_flow_n": 0, "groups": {},
                    },
                )
                expected = set(EXPECTED_VALIDATOR_FAULT[kind.value])
                labelled_mask = np.zeros((plan.n_steps, n_sensors), dtype=np.bool_)
                for episode in plan.instrument_faults:
                    j = column[episode.sensor_id]
                    window = range(episode.start_step, episode.last_step + 1)
                    labelled_mask[episode.start_step : episode.last_step + 1, j] = True
                    seen = [s for s in window if (s, j) in faults]
                    group = group_of[episode.sensor_id]
                    in_flow = group == "flow"
                    bucket["n_episodes"] += 1
                    bucket["labelled"] += episode.length_steps
                    detected = bool(seen)
                    bucket["det"] += int(detected)
                    if not in_flow:
                        bucket["n_excl"] += 1
                        bucket["det_excl"] += int(detected)
                    count = bucket["groups"].setdefault(group, [0, 0])
                    count[0] += int(detected)
                    count[1] += 1
                    if detected:
                        bucket["delays"].append(seen[0] - episode.start_step)
                    for s in window:
                        types = faults.get((s, j))
                        if not types:
                            continue
                        bucket["flagged"] += 1
                        bucket["flagged_expected"] += int(bool(expected & set(types)))
                        for name in types:
                            bucket["by_type"][name] = bucket["by_type"].get(name, 0) + 1
                for s, j in faults:
                    if labelled_mask[s, j]:
                        continue
                    key = "fp_flow" if group_of[catalog.tags[j]] == "flow" else "fp"
                    bucket[key] += 1
                flow_cols = [column[t] for t in catalog.tags if group_of[t] == "flow"]
                rest = ~labelled_mask
                bucket["fp_flow_n"] += int(rest[:, flow_cols].sum())
                bucket["fp_n"] += int(rest.sum()) - int(rest[:, flow_cols].sum())

    cells: list[DetectionCell] = []
    for (kind_name, level_name), b in sorted(totals.items()):
        cells.append(
            DetectionCell(
                kind=kind_name,
                level=level_name,
                n_episodes=b["n_episodes"],
                n_episodes_excl_flow=b["n_excl"],
                labelled_readings=b["labelled"],
                episodes_detected=b["det"],
                episodes_detected_excl_flow=b["det_excl"],
                readings_flagged=b["flagged"],
                readings_flagged_expected=b["flagged_expected"],
                faults_by_type=dict(sorted(b["by_type"].items())),
                first_delay_steps=sorted(b["delays"]),
                false_positive_per_1000=_per_1000(b["fp"], b["fp_n"]),
                false_positive_per_1000_flow=_per_1000(b["fp_flow"], b["fp_flow_n"]),
                episodes_by_group={g: (c[0], c[1]) for g, c in sorted(b["groups"].items())},
            )
        )
    return cells


def _validator_faults(
    frame: pd.DataFrame, validator: Any, n_sensors: int
) -> dict[tuple[int, int], list[str]]:
    """Run a frame through a validator; return (step, sensor column) -> fault types.

    Los readings se pasan en el orden del frame (paso y luego sensor), que es el
    orden en que los detectores construyen su estado.
    """
    found: dict[tuple[int, int], list[str]] = {}
    for i, reading in enumerate(readings_from_frame(frame)):
        result = validator.validate(reading)
        if result.faults:
            found[(i // n_sensors, i % n_sensors)] = [f.fault_type.value for f in result.faults]
    return found


# ----------------------------------------------------------------------
# Presupuesto, capacidad y plan (F3-M5)
# ----------------------------------------------------------------------


CANDIDATE_WEIGHTS: dict[str, dict[str, float]] = {
    "d11": {"critical": 0.30, "high": 0.30, "medium": 0.25, "low": 0.15},
    "capacity_a": {"critical": 0.50, "high": 0.38, "medium": 0.10, "low": 0.02},
    "capacity_b": {"critical": 0.55, "high": 0.38, "medium": 0.06, "low": 0.01},
    "process_heavy": {"critical": 0.60, "high": 0.37, "medium": 0.025, "low": 0.005},
}
"""Pesos de severidad candidatos del informe de capacidad (el de D11 y tres mas
pegados a lo que los tipos de instrumento pueden etiquetar)."""


def capacity_report(
    simulator: ReactorSimulator,
    weights: Mapping[str, Mapping[str, float]] = CANDIDATE_WEIGHTS,
) -> dict[str, Any]:
    """Return the label budget, the density caps and the reachable rates.

    Args:
        simulator: Simulador.
        weights: Nombre -> pesos de severidad a evaluar.

    Returns:
        A JSON-ready mapping: the budget at the configured rate and weights, the
        per-type capacity, and the largest feasible rate per weighting.
    """
    params, catalog, config = simulator.params, simulator.catalog, simulator.scenario_config
    budget = compute_budget(params, catalog, config)
    result: dict[str, Any] = _jsonable(
        {
            "params": {
                "n_runs": params.n_runs,
                "run_length_steps": params.run_length_steps,
                "n_sensors": len(catalog.sensors),
                "fault_injection_rate": params.fault_injection_rate,
                "severity_weights": params.severity_weights,
                "critical_min_fraction": params.critical_min_fraction,
                "planner": config.planner,
            },
            "budget": {
                "total_readings": budget.total_readings,
                "target_readings": budget.target_readings,
                "runs_available": budget.runs_available,
                "runs_needed": budget.runs_needed,
                "feasible": budget.feasible,
                "severities": [
                    {
                        "severity": s.severity.value,
                        "weight": s.weight,
                        "quota_readings": s.quota_readings,
                        "kinds": [k.value for k in s.kinds],
                        "capacity_per_run": s.capacity_per_run,
                        "runs_per_kind": s.runs_per_kind,
                        "runs": s.runs,
                    }
                    for s in budget.severities
                ],
            },
            "capacity_per_run": {
                kind.value: {
                    "severity": cap.severity.value,
                    "min_readings": cap.min_readings,
                    "max_readings": cap.max_readings,
                    "max_fraction_of_run": cap.fraction(budget.run_readings),
                    "max_episodes": cap.max_episodes,
                }
                for kind, cap in budget.capacities.items()
            },
            "max_feasible_rate": {
                name: {
                    "weights": dict(w),
                    "rate": max_feasible_rate(params, catalog, config, w),
                    "critical_share": w["critical"] * max_feasible_rate(params, catalog, config, w),
                }
                for name, w in weights.items()
            },
        }
    )
    return result


def plan_coverage_report(
    simulator: ReactorSimulator,
    *,
    fault_injection_rate: float | None = None,
    severity_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Plan the dataset and measure its labels and coverage without simulating it.

    Args:
        simulator: Simulador.
        fault_injection_rate: Sustituye al de `params` si se da.
        severity_weights: Sustituye a los pesos de `params` si se da.

    Returns:
        A JSON-ready mapping with the plan summary, the labelled share per type
        and severity, and the coverage of the whole dataset and its partitions.

    Raises:
        ValueError: If the quotas are unreachable (message with the numbers).
    """
    planned = simulator.planned(
        fault_injection_rate=fault_injection_rate, severity_weights=severity_weights
    )
    settings = simulator.scenario_config.planner
    collector = CoverageCollector()
    for plan in planned.plans:
        collector.add_plan(plan, simulator.catalog)
    report = collector.report(
        settings.k_min_readings, settings.n_min_episodes, settings.coverage_gate
    )
    total = planned.budget.total_readings

    def scope(s: CoverageScope) -> dict[str, Any]:
        return {
            "n_steps": s.n_steps,
            "labelled_readings": s.labelled_readings,
            "types_present": list(s.types_present),
            "n_types": s.n_types,
            "grid_size": s.grid_size,
            "covered": len(s.covered),
            "grid_fraction": s.grid_fraction,
            "missing": [m.label for m in s.missing],
            "by_type": {
                kind.value: list(s.type_cells(kind.value))
                for kind in FAULT_KINDS
            },
        }

    document: dict[str, Any] = _jsonable(
        {
            "rate": planned.labelled_fraction,
            "target_rate": fault_injection_rate or simulator.params.fault_injection_rate,
            "labelled_readings": planned.planned_readings,
            "total_readings": total,
            "scenarios": dict(planned.by_scenario),
            "by_severity": {
                sev: {"readings": n, "fraction": n / total}
                for sev, n in planned.readings_by_severity().items()
            },
            "kinds": {
                kind.value: {
                    "n_runs": kp.n_runs,
                    "episodes": kp.episodes,
                    "quota_readings": kp.quota_readings,
                    "planned_readings": kp.planned_readings,
                    "levels": dict(kp.levels),
                }
                for kind, kp in planned.kinds.items()
            },
            "coverage": {
                "k_min_readings": report.k_min_readings,
                "n_min_episodes": report.n_min_episodes,
                "gate": report.gate,
                "types_ok": report.types_ok,
                "grid_ok": report.grid_ok,
                "passed": report.passed,
                "overall": scope(report.overall),
                "partitions": {name: scope(s) for name, s in report.partitions.items()},
            },
        }
    )
    return document


# ----------------------------------------------------------------------
# Determinismo y rendimiento
# ----------------------------------------------------------------------


def parquet_digest(frame: pd.DataFrame, directory: Path, name: str = "run.parquet") -> str:
    """Return the SHA-256 of the Parquet bytes of a frame.

    Args:
        frame: Frame largo.
        directory: Directorio de trabajo donde se escribe.
        name: Nombre del fichero.

    Returns:
        Hex digest of the written file.
    """
    path = write_run_parquet(frame, directory / name)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DeterminismResult:
    """Resultado de la comprobacion de determinismo de un run.

    Attributes:
        run_id: Run comprobado.
        n_rows: Filas del frame.
        frames_equal: `DataFrame.equals` entre dos construcciones en este proceso.
        parquet_equal: Bytes de parquet iguales entre las dos construcciones.
        sha256: Digest del parquet.
        cross_process_equal: Digest igual en un proceso nuevo (None si no se midio).
    """

    run_id: str
    n_rows: int
    frames_equal: bool
    parquet_equal: bool
    sha256: str
    cross_process_equal: bool | None


def subprocess_digest(params_path: str | Path, run_index: int, n_steps: int | None) -> str:
    """Return the run digest computed by a fresh Python process.

    Args:
        params_path: Fichero de parametros.
        run_index: Run a generar.
        n_steps: Pasos del run (None = los de params).

    Returns:
        The hex digest the child printed.

    Raises:
        RuntimeError: If the child process fails.
    """
    command = [
        sys.executable,
        "-m",
        "data.generators.simulation_study",
        "digest",
        "--params",
        str(Path(params_path).resolve()),
        "--run-index",
        str(run_index),
    ]
    if n_steps is not None:
        command += ["--steps", str(n_steps)]
    completed = subprocess.run(  # noqa: S603 - fixed argv built from our own arguments
        command, capture_output=True, text=True, cwd=_REPO_ROOT, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"digest subprocess failed: {completed.stderr.strip()}")
    return completed.stdout.strip().splitlines()[-1]


def determinism_check(
    simulator: ReactorSimulator,
    plan: RunPlan,
    workdir: Path,
    params_path: str | Path | None = None,
) -> DeterminismResult:
    """Check that one run is reproducible in-process and across processes.

    Args:
        simulator: Simulador.
        plan: Run a comprobar.
        workdir: Directorio donde escribir los parquet temporales.
        params_path: Si se da, tambien se compara con un proceso nuevo.

    Returns:
        The DeterminismResult.
    """
    first = simulator.generate_run(plan)
    second = simulator.generate_run(plan)
    digest_a = parquet_digest(first, workdir, "a.parquet")
    digest_b = parquet_digest(second, workdir, "b.parquet")
    cross: bool | None = None
    if params_path is not None:
        n_steps = plan.n_steps if plan.n_steps != simulator.params.run_length_steps else None
        cross = subprocess_digest(params_path, plan.run_index, n_steps) == digest_a
    return DeterminismResult(
        run_id=plan.run_id,
        n_rows=len(first),
        frames_equal=bool(first.equals(second)),
        parquet_equal=digest_a == digest_b,
        sha256=digest_a,
        cross_process_equal=cross,
    )


@dataclass(frozen=True)
class ThroughputResult:
    """Rendimiento extremo a extremo (integrar + medir + ensamblar + escribir).

    Attributes:
        n_runs: Runs generados.
        n_steps: Pasos totales.
        n_readings: Lecturas totales.
        seconds_integrate: Tiempo integrando el modelo (y sus observables).
        seconds_assemble: Tiempo en ruido, cuentas ADC y frame largo.
        seconds_write: Tiempo escribiendo parquet.
        steps_per_s: Pasos por segundo extremo a extremo.
        readings_per_s: Lecturas por segundo extremo a extremo.
    """

    n_runs: int
    n_steps: int
    n_readings: int
    seconds_integrate: float
    seconds_assemble: float
    seconds_write: float
    steps_per_s: float
    readings_per_s: float


def measure_throughput(
    simulator: ReactorSimulator, workdir: Path, n_runs: int = 1, n_steps: int | None = None
) -> ThroughputResult:
    """Time the full pipeline over the first runs.

    Args:
        simulator: Simulador.
        workdir: Directorio donde escribir el parquet.
        n_runs: Runs a generar.
        n_steps: Pasos por run (por defecto los de params).

    Returns:
        The ThroughputResult.
    """
    plans = simulator.plans[:n_runs]
    if n_steps is not None:
        plans = [dataclasses.replace(plan, n_steps=n_steps) for plan in plans]
    t_integrate = t_assemble = t_write = 0.0
    steps = readings = 0
    started = time.perf_counter()
    for plan in plans:
        t0 = time.perf_counter()
        sim = simulator.simulate(plan)
        t1 = time.perf_counter()
        _, noise_rng, _ = plan.rngs()
        frame = build_run_frame(
            sim, simulator.catalog, simulator.spans, simulator.params.plant_id, noise_rng
        )
        t2 = time.perf_counter()
        write_run_parquet(
            frame,
            run_parquet_path(
                workdir, plan.scenario, simulator.params.openmc_seed, plan.run_id
            ),
        )
        t3 = time.perf_counter()
        t_integrate += t1 - t0
        t_assemble += t2 - t1
        t_write += t3 - t2
        steps += plan.n_steps
        readings += len(frame)
    total = time.perf_counter() - started
    return ThroughputResult(
        n_runs=len(plans),
        n_steps=steps,
        n_readings=readings,
        seconds_integrate=t_integrate,
        seconds_assemble=t_assemble,
        seconds_write=t_write,
        steps_per_s=steps / total,
        readings_per_s=readings / total,
    )


# ----------------------------------------------------------------------
# Informe y CLI
# ----------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and numpy scalars into plain JSON types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_report(
    simulator: ReactorSimulator,
    params_path: str | Path,
    cadences_s: Sequence[float] = DEFAULT_CADENCES_S,
    duration_s: float = DEFAULT_STUDY_DURATION_S,
    n_runs: int | None = None,
    n_steps: int | None = None,
    validator_factory: ValidatorFactory = default_validator_factory,
) -> dict[str, Any]:
    """Run the four studies and return a JSON-ready report.

    Args:
        simulator: Simulador.
        params_path: Fichero de parametros (para la comprobacion entre procesos).
        cadences_s: Cadencias del estudio de cadencia.
        duration_s: Duracion simulada del estudio de cadencia.
        n_runs: Runs de la calibracion de ruido (por defecto todos).
        n_steps: Pasos por run en calibracion, determinismo y rendimiento.
        validator_factory: Constructor del validador.

    Returns:
        Mapping with `cadence`, `noise`, `determinism` and `throughput`.
    """
    params = simulator.params
    limits = ConstraintLimits.from_plant(simulator.model.plant)
    sims = simulate_runs(simulator, n_runs, n_steps)
    noise = noise_calibration(sims, simulator.catalog, simulator.model.table, limits)
    cadence = cadence_study(simulator, cadences_s, duration_s, validator_factory=validator_factory)
    plan = dataclasses.replace(
        simulator.plans[0], n_steps=n_steps if n_steps is not None else params.run_length_steps
    )
    with tempfile.TemporaryDirectory(prefix="rg-study-") as tmp:
        work = Path(tmp)
        determinism = determinism_check(simulator, plan, work, params_path)
        throughput = measure_throughput(simulator, work, 1, n_steps)
    report: dict[str, Any] = _jsonable(
        {
            "params": {
                "dt_seconds": params.dt_seconds,
                "run_length_steps": params.run_length_steps,
                "n_runs": params.n_runs,
                "openmc_seed": params.openmc_seed,
            },
            "cadence": cadence,
            "noise": noise,
            "determinism": determinism,
            "throughput": throughput,
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("report", help="run the four studies and write a JSON report")
    report.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    report.add_argument("--output", required=True)
    report.add_argument("--duration-s", type=float, default=DEFAULT_STUDY_DURATION_S)
    report.add_argument("--n-runs", type=int, default=None)
    report.add_argument("--steps", type=int, default=None)
    digest = sub.add_parser("digest", help="print the SHA-256 of one run's parquet")
    digest.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    digest.add_argument("--run-index", type=int, default=0)
    digest.add_argument("--steps", type=int, default=None)
    faults = sub.add_parser(
        "faults", help="measure the rule-violation matrix of the process faults"
    )
    faults.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    faults.add_argument("--output", required=True)
    faults.add_argument("--repeats", type=int, default=3)
    faults.add_argument("--hold-steps", type=int, default=None)
    faults.add_argument("--recovery-steps", type=int, default=None)
    instrument = sub.add_parser(
        "instrument", help="measure the rule matrix and the validator detection of the 5 faults"
    )
    instrument.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    instrument.add_argument("--output", required=True)
    instrument.add_argument("--repeats", type=int, default=2)
    instrument.add_argument("--steps", type=int, default=None)
    plan = sub.add_parser("plan", help="budget, density caps, reachable rate and coverage")
    plan.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    plan.add_argument("--output", required=True)
    plan.add_argument("--rate", type=float, default=None)
    plan.add_argument(
        "--weights", choices=sorted(CANDIDATE_WEIGHTS), default=None,
        help="severity weights by name (default: the ones of params.yaml)",
    )
    args = parser.parse_args(argv)

    params: SimulationParams = load_simulation_params(args.params)
    simulator = ReactorSimulator.from_params(params)
    if args.command == "instrument":
        cells = instrument_matrix(simulator, n_steps=args.steps)
        detection = instrument_detection(simulator, n_repeats=args.repeats, n_steps=args.steps)
        document = _jsonable(
            {
                "params": {
                    "dt_seconds": params.dt_seconds,
                    "openmc_seed": params.openmc_seed,
                    "repeats": args.repeats,
                },
                "matrix": summarize_instrument_matrix(cells),
                "detection": detection,
                "cells": cells,
            }
        )
        text_out = json.dumps(document, indent=2, sort_keys=True)
        Path(args.output).write_text(text_out, encoding="utf-8")
        _LOG.info("Wrote the instrument study to %s", args.output)
        return 0
    if args.command == "plan":
        weights = CANDIDATE_WEIGHTS[args.weights] if args.weights else None
        plan_doc: dict[str, Any] = {"capacity": capacity_report(simulator)}
        try:
            plan_doc["plan"] = plan_coverage_report(
                simulator, fault_injection_rate=args.rate, severity_weights=weights
            )
        except ValueError as exc:
            plan_doc["plan"] = {"error": str(exc)}
        text_out = json.dumps(plan_doc, indent=2, sort_keys=True)
        Path(args.output).write_text(text_out, encoding="utf-8")
        _LOG.info("Wrote the plan report to %s", args.output)
        return 0
    if args.command == "faults":
        results = fault_matrix(
            simulator,
            n_repeats=args.repeats,
            hold_steps=args.hold_steps,
            recovery_steps=args.recovery_steps,
        )
        document = _jsonable(
            {
                "params": {
                    "dt_seconds": params.dt_seconds,
                    "openmc_seed": params.openmc_seed,
                    "repeats": args.repeats,
                },
                "summary": summarize_fault_matrix(results),
                "runs": results,
            }
        )
        text = json.dumps(document, indent=2, sort_keys=True)
        Path(args.output).write_text(text, encoding="utf-8")
        _LOG.info("Wrote the fault matrix to %s", args.output)
        return 0
    if args.command == "digest":
        plan = simulator.plans[args.run_index]
        if args.steps is not None:
            plan = dataclasses.replace(plan, n_steps=args.steps)
        with tempfile.TemporaryDirectory(prefix="rg-digest-") as tmp:
            print(parquet_digest(simulator.generate_run(plan), Path(tmp)))
        return 0

    data = build_report(
        simulator,
        args.params,
        duration_s=args.duration_s,
        n_runs=args.n_runs,
        n_steps=args.steps,
    )
    Path(args.output).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    _LOG.info("Wrote the study report to %s", args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())
