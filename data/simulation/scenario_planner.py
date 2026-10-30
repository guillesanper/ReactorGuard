"""Planificador de escenarios por presupuesto (F3-M5, D11).

Decide QUE runs tiene el dataset: cuantos de operacion normal y cuantos de cada
uno de los 8 tipos del TDD 2.2, con que niveles, en que sensores y en que
instantes. Es determinista: toda decision sale del `Generator` que se le pasa.

Presupuesto (D11):

    1. `n_runs = ceil(n_samples / run_length_steps)`.
    2. Objetivo etiquetado = `fault_injection_rate` x lecturas totales; la cuota de
       cada severidad es su peso por ese objetivo.
    3. Una severidad la cubren sus tipos (CRITICAL: stuck, excursion, vacio; HIGH:
       drift, sesgo, caudal; MEDIUM: intermitente; LOW: pico de ruido). La cuota se
       reparte entre ellos en proporcion a su CAPACIDAD por run bajo los topes de
       densidad: un tipo que etiqueta poco no puede cargar con la parte de uno que
       etiqueta mucho.
    4. Cada tipo recibe `n = max(n_min_runs, ceil(cuota / capacidad_de_la_severidad))`
       runs, iguales para los tipos de una severidad; los niveles rotan y los
       episodios se reparten por maximo resto.
    5. Topes de densidad (`scenario_planner:`): fraccion del run que puede ocupar un
       fallo de proceso, sensores que puede afectar un run de instrumento y picos
       por muestra y sensor.
    6. Si la cuota de una severidad no cabe bajo los topes, o los runs no
       alcanzan, falla con `ValueError` y el mensaje trae las cifras. NO se fuerza
       ni se redondea la cuota: se vuelve al usuario (D11).

Las lecturas etiquetadas de un run salen de `RunPlan.expected_labelled_readings`:
un fallo de proceso etiqueta todos los sensores de su ventana y uno de
instrumento solo el sensor del episodio. Por eso un tipo de instrumento etiqueta
mucho menos por run que uno de proceso, y por eso MEDIUM y LOW (un unico tipo de
instrumento cada una) son las severidades que limitan el presupuesto.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.simulation.instrument_faults import (
    InstrumentFaultTiers,
    InstrumentKindTiers,
    build_episode,
    parse_instrument_faults,
)
from data.simulation.plant_params import PlantParams
from data.simulation.process_faults import (
    FaultTiers,
    KindTiers,
    parse_fault_tiers,
    process_run_plan,
)
from data.simulation.scenario import (
    FAULT_KINDS,
    FAULT_SEVERITY,
    FaultAxis,
    FaultLevel,
    FaultSeverity,
    InstrumentEpisode,
    InstrumentFaultKind,
    RunPlan,
    ScenarioKind,
    make_run_plan,
)
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.simulation_params import SEVERITY_KEYS, SimulationParams

_LOG = logging.getLogger(__name__)

PLANNER_STREAM_KEY = 2**31 - 1
"""`spawn_key` del flujo del planificador bajo la semilla maestra.

Los runs usan `spawn_key=(i,)` con i < n_runs: esta clave no puede coincidir con
ninguna, de modo que planificar no altera el ruido ni la carga de ningun run."""

_LEVELS: tuple[FaultLevel, ...] = (FaultLevel.LOW, FaultLevel.MEDIUM, FaultLevel.HIGH)

_PLACEMENT_TRIES = 200
"""Intentos de colocar un pico a >= 2 muestras de los demas del mismo sensor."""

_SEVERITY_OF: dict[str, FaultSeverity] = {
    "critical": FaultSeverity.CRITICAL,
    "high": FaultSeverity.HIGH,
    "medium": FaultSeverity.MEDIUM,
    "low": FaultSeverity.LOW,
}

_NAME = "plant_parameters.yaml: 'scenario_planner'"


# ----------------------------------------------------------------------
# Configuracion
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerSettings:
    """Topes y minimos del planificador (`scenario_planner:`).

    Attributes:
        n_min_runs: Runs minimos por tipo de fallo.
        min_normal_runs: Runs de operacion normal que se reservan siempre.
        max_sensor_fraction: Fraccion del catalogo que un run de instrumento puede
            afectar.
        max_window_fraction: Fraccion del run que puede ocupar un fallo de proceso.
        max_spike_rate: Picos por muestra y sensor afectado, como maximo.
        k_min_readings: Lecturas etiquetadas para dar un modo de la rejilla por
            cubierto (D6).
        n_min_episodes: Episodios para dar un modo por cubierto.
        coverage_gate: Fraccion minima de la rejilla cubierta (D6).
    """

    n_min_runs: int
    min_normal_runs: int
    max_sensor_fraction: float
    max_window_fraction: float
    max_spike_rate: float
    k_min_readings: int
    n_min_episodes: int
    coverage_gate: float


def _require(section: Mapping[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Raises:
        KeyError: If key is missing.
    """
    if key not in section:
        raise KeyError(f"{_NAME}: missing required key '{key}'.")
    return section[key]


def _number(section: Mapping[str, Any], key: str) -> float:
    """Return a finite float.

    Raises:
        KeyError: If the key is missing.
        ValueError: If the value is not a finite number.
    """
    raw = _require(section, key)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{_NAME}.{key}: must be a number, got {raw!r}.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{_NAME}.{key}: must be finite, got {raw!r}.")
    return value


def _whole(section: Mapping[str, Any], key: str, minimum: int) -> int:
    """Return an int >= minimum.

    Raises:
        ValueError: If the value is not a whole number >= minimum.
    """
    value = _number(section, key)
    if value != int(value) or value < minimum:
        raise ValueError(f"{_NAME}.{key}: must be a whole number >= {minimum}, got {value}.")
    return int(value)


def _fraction(section: Mapping[str, Any], key: str) -> float:
    """Return a float in (0, 1].

    Raises:
        ValueError: If the value is outside (0, 1].
    """
    value = _number(section, key)
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{_NAME}.{key}: must be in (0, 1], got {value}.")
    return value


def parse_planner_settings(raw: Mapping[str, Any]) -> PlannerSettings:
    """Resolve the `scenario_planner:` block of plant_parameters.yaml.

    Args:
        raw: `PlantParams.scenario_planner` (the parsed mapping).

    Returns:
        The typed PlannerSettings.

    Raises:
        KeyError: If a required key is missing.
        ValueError: If a value is out of range.
    """
    return PlannerSettings(
        n_min_runs=_whole(raw, "n_min_runs", 1),
        min_normal_runs=_whole(raw, "min_normal_runs", 0),
        max_sensor_fraction=_fraction(raw, "max_sensor_fraction"),
        max_window_fraction=_fraction(raw, "max_window_fraction"),
        max_spike_rate=_fraction(raw, "max_spike_rate"),
        k_min_readings=_whole(raw, "k_min_readings", 1),
        n_min_episodes=_whole(raw, "n_min_episodes", 1),
        coverage_gate=_fraction(raw, "coverage_gate"),
    )


@dataclass(frozen=True)
class ScenarioConfig:
    """Todo lo que el planificador lee de `plant_parameters.yaml`.

    Attributes:
        process: Niveles y cronologia de los fallos de proceso.
        instrument: Niveles de los fallos de instrumento.
        planner: Topes y minimos del planificador.
    """

    process: FaultTiers
    instrument: InstrumentFaultTiers
    planner: PlannerSettings

    @classmethod
    def from_plant(cls, plant: PlantParams) -> ScenarioConfig:
        """Resolve the three blocks from the loaded plant parameters.

        Args:
            plant: Parametros de planta (`load_plant_params`).

        Returns:
            The ScenarioConfig.

        Raises:
            KeyError: If a fault, level or key is missing.
            TypeError: If a block has the wrong type.
            ValueError: If a value is out of range.
        """
        return cls(
            process=parse_fault_tiers(plant.fault_tiers),
            instrument=parse_instrument_faults(plant.instrument_faults),
            planner=parse_planner_settings(plant.scenario_planner),
        )


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------


def planner_rng(params: SimulationParams) -> np.random.Generator:
    """Return the planner's own generator, independent of every run stream.

    Args:
        params: Parametros de simulacion (`openmc_seed` es la semilla maestra).

    Returns:
        A Generator on `SeedSequence(seed, spawn_key=(PLANNER_STREAM_KEY,))`.
    """
    sequence = np.random.SeedSequence(params.openmc_seed, spawn_key=(PLANNER_STREAM_KEY,))
    return np.random.default_rng(sequence)


def apportion(total: int, weights: Sequence[float], minimum: int = 0) -> list[int]:
    """Split an integer among parts by largest remainder (Hamilton).

    Cada parte recibe `minimum` y el resto se reparte en proporcion a su peso;
    las fracciones que sobran van a los mayores restos (empates: la parte de
    menor indice). La suma es exactamente `total`.

    Args:
        total: Cantidad entera a repartir.
        weights: Pesos no negativos, no todos cero.
        minimum: Minimo que recibe cada parte.

    Returns:
        One integer per weight, summing to `total`.

    Raises:
        ValueError: If a weight is negative, all are zero, or `total` cannot
            cover the minimums.
    """
    n = len(weights)
    if n == 0:
        raise ValueError("apportion needs at least one part.")
    if any(w < 0.0 or not math.isfinite(w) for w in weights) or sum(weights) <= 0.0:
        raise ValueError(
            f"apportion needs non-negative weights with a positive sum, got {weights}."
        )
    spare = total - minimum * n
    if spare < 0:
        raise ValueError(f"apportion: {total} cannot give {minimum} to each of {n} parts.")
    scale = float(sum(weights))
    shares = [spare * w / scale for w in weights]
    parts = [minimum + math.floor(s) for s in shares]
    order = sorted(range(n), key=lambda i: (-(shares[i] - math.floor(shares[i])), i))
    for i in order[: total - sum(parts)]:
        parts[i] += 1
    return parts


def _groups(catalog: SensorCatalog) -> dict[str, tuple[str, ...]]:
    """Return sensor tags per sensor type, in catalog order of first appearance."""
    groups: dict[str, list[str]] = {}
    for sensor in catalog.sensors:
        groups.setdefault(sensor.sensor_type.value, []).append(sensor.tag)
    return {name: tuple(tags) for name, tags in groups.items()}


def _kind_of(scenario: ScenarioKind) -> InstrumentFaultKind:
    """Return the instrument manipulation of an instrument scenario."""
    return InstrumentFaultKind(scenario.value)


# ----------------------------------------------------------------------
# Capacidad y cuotas
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TypeCapacity:
    """Lecturas etiquetadas que puede dar un run de un tipo bajo los topes.

    Attributes:
        kind: Tipo de fallo.
        severity: Severidad estatica del tipo.
        min_readings: Lecturas de un run con la plantilla minima (proceso: la
            ventana por defecto de `fault_tiers`; instrumento: un episodio).
        max_readings: Lecturas de un run en el tope de densidad.
        max_episodes: Episodios que caben en un run (1 en proceso).
        mean_window: Lecturas medias de un episodio sobre los tres niveles.
    """

    kind: ScenarioKind
    severity: FaultSeverity
    min_readings: int
    max_readings: int
    max_episodes: int
    mean_window: float

    def fraction(self, run_readings: int) -> float:
        """Return max_readings as a fraction of the readings of a run."""
        return self.max_readings / run_readings


def _process_window_bounds(
    tiers: KindTiers, params: SimulationParams, settings: PlannerSettings
) -> tuple[int, int, int, int]:
    """Return (onset, recovery, min window, max window) of a process fault in steps.

    La ventana etiquetada es `onset + hold + recovery + 1`. El tope sale de
    `max_window_fraction` y de que el fallo empiece al menos tras el calentamiento.

    Raises:
        ValueError: If the cap leaves no room for the default window.
    """
    onset = tiers.onset_steps(params.dt_seconds)
    recovery = tiers.recovery_steps
    w_min = onset + tiers.hold_steps + recovery + 1
    room = params.run_length_steps - params.warmup_steps
    w_max = min(math.floor(settings.max_window_fraction * params.run_length_steps), room)
    if w_max < w_min:
        raise ValueError(
            f"{_NAME}: max_window_fraction={settings.max_window_fraction} allows a window of "
            f"{w_max} steps but '{tiers.kind.value}' needs {w_min} by default."
        )
    return onset, recovery, w_min, w_max


def _instrument_episode_cap(settings: PlannerSettings, n_sensors: int) -> int:
    """Return how many distinct sensors an instrument run may affect (>= 1)."""
    return max(1, math.floor(settings.max_sensor_fraction * n_sensors))


def type_capacity(
    kind: ScenarioKind, params: SimulationParams, catalog: SensorCatalog, config: ScenarioConfig
) -> TypeCapacity:
    """Return the labelled-readings capacity of one run of a fault type.

    Args:
        kind: Uno de los 8 tipos de fallo.
        params: Parametros de simulacion.
        catalog: Catalogo de sensores.
        config: Niveles y topes.

    Returns:
        The TypeCapacity.

    Raises:
        KeyError: If `kind` is not a fault type.
        ValueError: If the caps leave no valid window.
    """
    if kind not in FAULT_SEVERITY:
        raise KeyError(f"scenario planner: '{kind.value}' is not a fault type.")
    n_sensors = len(catalog.sensors)
    settings = config.planner
    severity = FAULT_SEVERITY[kind]
    if kind.axis is FaultAxis.PROCESS:
        _, _, w_min, w_max = _process_window_bounds(config.process.for_kind(kind), params, settings)
        return TypeCapacity(
            kind=kind,
            severity=severity,
            min_readings=n_sensors * w_min,
            max_readings=n_sensors * w_max,
            max_episodes=1,
            mean_window=float(n_sensors * w_max),
        )
    tiers = config.instrument.for_kind(_kind_of(kind))
    episodes = _instrument_episode_cap(settings, n_sensors)
    n_steps = params.run_length_steps
    if kind is ScenarioKind.NOISE_SPIKE:
        per_sensor = max(1, math.floor(settings.max_spike_rate * n_steps))
        return TypeCapacity(
            kind=kind,
            severity=severity,
            min_readings=1,
            max_readings=episodes * per_sensor,
            max_episodes=episodes * per_sensor,
            mean_window=1.0,
        )
    if kind is ScenarioKind.BIAS:
        low, high = tiers.onset_fraction
        mean = n_steps * (1.0 - 0.5 * (low + high))
    else:
        mean = tiers.mean_window_steps(n_steps)
    return TypeCapacity(
        kind=kind,
        severity=severity,
        min_readings=int(mean),
        max_readings=int(episodes * mean),
        max_episodes=episodes,
        mean_window=mean,
    )


@dataclass(frozen=True)
class SeverityBudget:
    """Cuota de una severidad y lo que cuesta cubrirla.

    Attributes:
        severity: Severidad.
        weight: Peso de D11.
        quota_readings: Lecturas etiquetadas que le corresponden.
        kinds: Tipos que la cubren.
        capacity_per_run: Suma de las capacidades maximas de un run de cada tipo.
        runs_per_kind: Runs de CADA tipo de la severidad.
        capped_by_minimum: True si `n_min_runs` (y no la cuota) fija los runs.
    """

    severity: FaultSeverity
    weight: float
    quota_readings: float
    kinds: tuple[ScenarioKind, ...]
    capacity_per_run: int
    runs_per_kind: int
    capped_by_minimum: bool

    @property
    def runs(self) -> int:
        """Return the runs the severity consumes in total."""
        return self.runs_per_kind * len(self.kinds)


@dataclass(frozen=True)
class Budget:
    """Presupuesto de D11 resuelto para una configuracion.

    Attributes:
        n_runs: Runs del dataset.
        run_readings: Lecturas de un run.
        total_readings: Lecturas del dataset.
        target_readings: Lecturas que deben quedar etiquetadas.
        severities: Presupuesto por severidad, en orden decreciente.
        capacities: Capacidad por tipo.
        runs_available: Runs disponibles para fallos (`n_runs - min_normal_runs`).
    """

    n_runs: int
    run_readings: int
    total_readings: int
    target_readings: float
    severities: tuple[SeverityBudget, ...]
    capacities: Mapping[ScenarioKind, TypeCapacity]
    runs_available: int

    @property
    def runs_needed(self) -> int:
        """Return the fault runs the quotas require under the caps."""
        return sum(s.runs for s in self.severities)

    @property
    def feasible(self) -> bool:
        """Return whether the quotas fit in the available runs."""
        return self.runs_needed <= self.runs_available


def _severity_budget(
    severity: FaultSeverity,
    weight: float,
    quota: float,
    capacities: Mapping[ScenarioKind, TypeCapacity],
    n_min_runs: int,
) -> SeverityBudget:
    """Return the budget of one severity."""
    kinds = tuple(k for k in FAULT_KINDS if FAULT_SEVERITY[k] is severity)
    capacity = sum(capacities[k].max_readings for k in kinds)
    needed = math.ceil(quota / capacity) if quota > 0 else 0
    runs = max(n_min_runs, needed)
    return SeverityBudget(
        severity=severity,
        weight=weight,
        quota_readings=quota,
        kinds=kinds,
        capacity_per_run=capacity,
        runs_per_kind=runs,
        capped_by_minimum=needed <= n_min_runs,
    )


def compute_budget(
    params: SimulationParams,
    catalog: SensorCatalog,
    config: ScenarioConfig,
    *,
    fault_injection_rate: float | None = None,
    severity_weights: Mapping[str, float] | None = None,
) -> Budget:
    """Resolve the D11 budget for a configuration (does not raise on infeasibility).

    Args:
        params: Parametros de simulacion.
        catalog: Catalogo de sensores.
        config: Niveles y topes.
        fault_injection_rate: Si se da, sustituye al de `params` (estudio de
            capacidad sin editar `params.yaml`).
        severity_weights: Si se da, sustituye a los pesos de `params`.

    Returns:
        The Budget; `feasible` says whether the quotas fit in the runs.

    Raises:
        ValueError: If the rate or the weights are out of range.
    """
    rate = params.fault_injection_rate if fault_injection_rate is None else fault_injection_rate
    weights = dict(params.severity_weights if severity_weights is None else severity_weights)
    if not 0.0 < rate < 1.0:
        raise ValueError(f"fault_injection_rate must be in (0, 1), got {rate}.")
    if set(weights) != set(SEVERITY_KEYS) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"severity weights must be {list(SEVERITY_KEYS)} summing to 1: {weights}.")
    n_sensors = len(catalog.sensors)
    run_readings = params.run_length_steps * n_sensors
    total = params.n_runs * run_readings
    capacities = {k: type_capacity(k, params, catalog, config) for k in FAULT_KINDS}
    target = rate * total
    budgets = tuple(
        _severity_budget(
            _SEVERITY_OF[key],
            weights[key],
            weights[key] * target,
            capacities,
            config.planner.n_min_runs,
        )
        for key in SEVERITY_KEYS
    )
    return Budget(
        n_runs=params.n_runs,
        run_readings=run_readings,
        total_readings=total,
        target_readings=target,
        severities=budgets,
        capacities=capacities,
        runs_available=params.n_runs - config.planner.min_normal_runs,
    )


def max_feasible_rate(
    params: SimulationParams,
    catalog: SensorCatalog,
    config: ScenarioConfig,
    severity_weights: Mapping[str, float] | None = None,
) -> float:
    """Return the largest `fault_injection_rate` the planner can satisfy.

    Busca por biseccion sobre la misma regla de reparto que `plan_runs`, de modo
    que `rate <= max_feasible_rate` implica que `plan_runs` no falla por cuota.

    Args:
        params: Parametros de simulacion.
        catalog: Catalogo de sensores.
        config: Niveles y topes.
        severity_weights: Pesos de severidad a usar (por defecto los de `params`).

    Returns:
        The largest feasible rate in (0, 1) (bisection, 60 steps); 0.0 if even
        the minimum runs per type do not fit.
    """
    low, high = 0.0, 1.0 - 1e-9
    probe = 1e-9
    if not compute_budget(
        params, catalog, config, fault_injection_rate=probe, severity_weights=severity_weights
    ).feasible:
        return 0.0
    for _ in range(60):
        mid = 0.5 * (low + high)
        feasible = compute_budget(
            params, catalog, config, fault_injection_rate=mid, severity_weights=severity_weights
        ).feasible
        low, high = (mid, high) if feasible else (low, mid)
    return low


def _infeasible_message(budget: Budget, max_rate: float, rate: float) -> str:
    """Return the ValueError text of an unreachable quota, with the numbers."""
    lines: list[str] = []
    lines.append(
        f"scenario planner: the label budget is unreachable under the density caps "
        f"(fault_injection_rate={rate:.3f})."
    )
    lines.append(
        f"  readings: {budget.total_readings} in {budget.n_runs} runs; target "
        f"{budget.target_readings:.0f} labelled; runs available for faults "
        f"{budget.runs_available}, needed {budget.runs_needed}."
    )
    for s in budget.severities:
        kinds = ", ".join(k.value for k in s.kinds)
        lines.append(
            f"  {s.severity.value:8s} weight {s.weight:.2f} quota "
            f"{s.quota_readings:10.0f} readings; "
            f"capacity per run {s.capacity_per_run:6d} over [{kinds}] -> "
            f"{s.runs_per_kind} runs per type ({s.runs} in total)"
        )
    lines.append(
        f"  largest rate this configuration can reach with these weights: {max_rate:.4f}. "
        "Lower fault_injection_rate, change severity_weights, relax a cap of "
        "scenario_planner or accept fewer labelled readings; the planner does not force "
        "unreal densities (D11)."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Planes
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class KindPlan:
    """Lo que el planificador decidio para un tipo de fallo.

    Attributes:
        kind: Tipo.
        n_runs: Runs del tipo.
        quota_readings: Cuota de lecturas asignada al tipo.
        planned_readings: Lecturas etiquetadas que los planes producen.
        episodes: Episodios (1 por run en proceso).
        levels: Episodios por nivel.
    """

    kind: ScenarioKind
    n_runs: int
    quota_readings: float
    planned_readings: int
    episodes: int
    levels: Mapping[str, int]


@dataclass(frozen=True)
class PlannedDataset:
    """Planes de todos los runs y el resumen de lo planificado.

    Attributes:
        plans: Un RunPlan por run, en orden de run.
        budget: Presupuesto de D11 resuelto.
        kinds: Decision por tipo de fallo.
        n_sensors: Sensores del catalogo.
    """

    plans: list[RunPlan]
    budget: Budget
    kinds: Mapping[ScenarioKind, KindPlan]
    n_sensors: int
    by_scenario: Mapping[str, int] = field(default_factory=dict)

    @property
    def planned_readings(self) -> int:
        """Return the labelled readings of the whole plan."""
        return sum(p.expected_labelled_readings(self.n_sensors) for p in self.plans)

    @property
    def labelled_fraction(self) -> float:
        """Return the labelled readings over all readings."""
        return self.planned_readings / self.budget.total_readings

    def readings_by_severity(self) -> dict[str, int]:
        """Return labelled readings per static severity."""
        counts: Counter[str] = Counter()
        for plan in self.plans:
            severity = plan.scenario.severity
            if severity is not None:
                counts[severity.value] += plan.expected_labelled_readings(self.n_sensors)
        return {s.value: counts.get(s.value, 0) for s in FaultSeverity}


def _plan_process_runs(
    kind: ScenarioKind,
    n_runs: int,
    quota: float,
    params: SimulationParams,
    catalog: SensorCatalog,
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> list[tuple[FaultLevel, int, int]]:
    """Return (level, injected_step, hold_steps) of each run of a process type.

    La ventana de cada run sale de repartir por maximo resto las muestras que
    exige la cuota, acotadas a [ventana por defecto, ventana maxima]. Los niveles
    rotan desde un desplazamiento sorteado, de modo que n >= 3 runs cubren los 3.

    Returns:
        One tuple per run.
    """
    tiers = config.process.for_kind(kind)
    onset, recovery, w_min, w_max = _process_window_bounds(tiers, params, config.planner)
    n_sensors = len(catalog.sensors)
    wanted = math.ceil(quota / n_sensors)
    wanted = min(max(wanted, w_min * n_runs), w_max * n_runs)
    windows = apportion(wanted, [1.0] * n_runs)
    offset = int(rng.integers(0, len(_LEVELS)))
    out: list[tuple[FaultLevel, int, int]] = []
    for i, window in enumerate(windows):
        window = min(max(window, w_min), w_max)
        hold = window - onset - recovery - 1
        latest = params.run_length_steps - window
        start = int(rng.integers(params.warmup_steps, latest + 1))
        out.append((_LEVELS[(offset + i) % len(_LEVELS)], start, hold))
    return out


def _place_spike(taken: set[int], low: int, high: int, rng: np.random.Generator) -> int | None:
    """Return a step in [low, high] at least 2 samples from every taken step."""
    for _ in range(_PLACEMENT_TRIES):
        step = int(rng.integers(low, high + 1))
        if all(abs(step - other) >= 2 for other in taken):
            return step
    return None


def _plan_instrument_episodes(
    kind: ScenarioKind,
    n_runs: int,
    quota: float,
    params: SimulationParams,
    catalog: SensorCatalog,
    config: ScenarioConfig,
    rng: np.random.Generator,
) -> list[list[InstrumentEpisode]]:
    """Return the episodes of each run of an instrument type.

    Cada episodio toma la celda (grupo de sensor x nivel) menos cubierta que aun
    tenga hueco en algun run (empates: orden barajado de la rejilla), de modo que
    con al menos `grupos x 3 x n_min_episodes` episodios cada modo queda cubierto
    y el reparto entre celdas es parejo. Va al run con menos episodios que tenga
    un sensor libre del grupo (sensores distintos por run, salvo los picos); la
    ventana de un tipo con nivel de longitud sale del nivel.

    Raises:
        ValueError: If the episodes do not fit under the density caps.
    """
    tiers: InstrumentKindTiers = config.instrument.for_kind(_kind_of(kind))
    settings = config.planner
    n_steps = params.run_length_steps
    groups = _groups(catalog)
    cells = [(g, lv) for g in groups for lv in _LEVELS]
    cells = [cells[i] for i in rng.permutation(len(cells))]
    counts = [0] * len(cells)
    n_sensors = len(catalog.sensors)
    max_sensors = _instrument_episode_cap(settings, n_sensors)
    spike = kind is ScenarioKind.NOISE_SPIKE
    per_sensor_spikes = max(1, math.floor(settings.max_spike_rate * n_steps))
    warmup = params.warmup_steps

    used: list[dict[str, int]] = [{} for _ in range(n_runs)]
    steps_taken: list[dict[str, set[int]]] = [{} for _ in range(n_runs)]
    episodes: list[list[InstrumentEpisode]] = [[] for _ in range(n_runs)]
    labelled = 0
    index = 0
    while labelled < quota or min(counts) < settings.n_min_episodes:
        chosen: tuple[int, int, str] | None = None
        for cell in sorted(range(len(cells)), key=lambda c: (counts[c], c)):
            members = groups[cells[cell][0]]
            for run in sorted(range(n_runs), key=lambda r: (len(episodes[r]), r)):
                free = _free_sensor(
                    used[run], members, spike, per_sensor_spikes, max_sensors, index
                )
                if free is not None:
                    chosen = (cell, run, free)
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise ValueError(
                f"scenario planner: no room for another '{kind.value}' episode after "
                f"{index} episodes ({labelled} readings of a quota of {quota:.0f}): every "
                f"run is at its sensor cap ({max_sensors} sensors, {per_sensor_spikes} "
                "spikes per sensor)."
            )
        cell, run, sensor = chosen
        level = cells[cell][1]
        counts[cell] += 1
        length = tiers.window_steps(level, 0, n_steps)
        if n_steps - length < warmup and kind is not ScenarioKind.BIAS:
            raise ValueError(
                f"scenario planner: a '{kind.value}' episode of {length} steps does not fit "
                f"after the {warmup}-step warm-up of a {n_steps}-step run."
            )
        if kind is ScenarioKind.BIAS:
            low, high = tiers.onset_fraction
            lo_step = max(warmup, math.ceil(low * n_steps))
            hi_step = max(lo_step, math.floor(high * n_steps))
            start = int(rng.integers(lo_step, hi_step + 1))
        elif spike:
            taken = steps_taken[run].setdefault(sensor, set())
            placed = _place_spike(taken, warmup, n_steps - 1, rng)
            if placed is None:
                raise ValueError(
                    f"scenario planner: cannot place another spike on {sensor} of run {run} "
                    f"at >= 2 samples from the {len(taken)} already there."
                )
            start = placed
            taken.add(start)
        else:
            start = int(rng.integers(warmup, n_steps - length + 1))
        episode = build_episode(tiers, level, sensor, start, n_steps, rng)
        episodes[run].append(episode)
        used[run][sensor] = used[run].get(sensor, 0) + 1
        labelled += episode.length_steps
        index += 1
    return episodes


def _free_sensor(
    used: Mapping[str, int],
    members: Sequence[str],
    spike: bool,
    per_sensor_spikes: int,
    max_sensors: int,
    rotation: int,
) -> str | None:
    """Return a sensor of the group that can take one more episode in this run.

    Un run afecta como mucho `max_sensors` sensores distintos. Salvo los picos,
    cada sensor lleva un unico episodio; un sensor admite hasta
    `per_sensor_spikes` picos.

    Returns:
        A tag, or None when the group has no room in this run.
    """
    ordered = [members[(rotation + i) % len(members)] for i in range(len(members))]
    if spike:
        for tag in ordered:
            if used.get(tag, 0) < per_sensor_spikes and (tag in used or len(used) < max_sensors):
                return tag
        return None
    if len(used) >= max_sensors:
        return None
    for tag in ordered:
        if tag not in used:
            return tag
    return None


def plan_runs(
    params: SimulationParams,
    catalog: SensorCatalog,
    rng: np.random.Generator,
    config: ScenarioConfig,
    *,
    fault_injection_rate: float | None = None,
    severity_weights: Mapping[str, float] | None = None,
) -> list[RunPlan]:
    """Plan every run of the dataset under the D11 label budget.

    Args:
        params: Parametros de simulacion.
        catalog: Catalogo de sensores.
        rng: Generador del planificador (`planner_rng(params)`).
        config: Niveles y topes (`ScenarioConfig.from_plant`).
        fault_injection_rate: Si se da, sustituye al de `params` (estudio de
            capacidad sin editar `params.yaml`).
        severity_weights: Si se da, sustituye a los pesos de `params`.

    Returns:
        One RunPlan per run, in run order (interleaved: the order is shuffled).

    Raises:
        ValueError: If the quotas do not fit under the caps and the run budget
            (the message carries the numbers), or an option is out of range.
    """
    return plan_dataset(
        params,
        catalog,
        rng,
        config,
        fault_injection_rate=fault_injection_rate,
        severity_weights=severity_weights,
    ).plans


def plan_dataset(
    params: SimulationParams,
    catalog: SensorCatalog,
    rng: np.random.Generator,
    config: ScenarioConfig,
    *,
    fault_injection_rate: float | None = None,
    severity_weights: Mapping[str, float] | None = None,
) -> PlannedDataset:
    """Plan every run and return the plans with the budget and per-type decisions.

    Args:
        params: Parametros de simulacion.
        catalog: Catalogo de sensores.
        rng: Generador del planificador (`planner_rng(params)`).
        config: Niveles y topes.
        fault_injection_rate: Sustituye al de `params` si se da.
        severity_weights: Sustituye a los de `params` si se da.

    Returns:
        The PlannedDataset.

    Raises:
        ValueError: If the quotas are unreachable, CRITICAL would fall below
            `critical_min_fraction`, or an option is out of range.
    """
    rate = params.fault_injection_rate if fault_injection_rate is None else fault_injection_rate
    weights = dict(params.severity_weights if severity_weights is None else severity_weights)
    budget = compute_budget(
        params, catalog, config, fault_injection_rate=rate, severity_weights=weights
    )
    if weights["critical"] * rate < params.critical_min_fraction:
        raise ValueError(
            f"scenario planner: CRITICAL gets {weights['critical'] * rate:.4f} of the readings "
            f"(weight {weights['critical']} x rate {rate}), below critical_min_fraction="
            f"{params.critical_min_fraction}."
        )
    if not budget.feasible:
        raise ValueError(
            _infeasible_message(budget, max_feasible_rate(params, catalog, config, weights), rate)
        )

    n_sensors = len(catalog.sensors)
    jobs: list[tuple[ScenarioKind, dict[str, Any]]] = []
    quotas: dict[ScenarioKind, float] = {}
    for severity in budget.severities:
        for kind in severity.kinds:
            capacity = budget.capacities[kind]
            quota = severity.quota_readings * capacity.max_readings / severity.capacity_per_run
            quotas[kind] = quota
            n_runs = severity.runs_per_kind
            if kind.axis is FaultAxis.PROCESS:
                process = _plan_process_runs(kind, n_runs, quota, params, catalog, config, rng)
                for level, start, hold in process:
                    jobs.append((kind, {"level": level, "start": start, "hold": hold}))
            else:
                per_run = _plan_instrument_episodes(
                    kind, n_runs, quota, params, catalog, config, rng
                )
                for run_episodes in per_run:
                    jobs.append((kind, {"episodes": tuple(run_episodes)}))

    n_normal = params.n_runs - len(jobs)
    jobs.extend((ScenarioKind.NORMAL, {}) for _ in range(n_normal))
    order = [jobs[i] for i in rng.permutation(len(jobs))]

    plans: list[RunPlan] = []
    for index, (kind, spec) in enumerate(order):
        if kind is ScenarioKind.NORMAL:
            plans.append(make_run_plan(params, index, kind))
        elif kind.axis is FaultAxis.PROCESS:
            plans.append(
                process_run_plan(
                    params,
                    config.process.for_kind(kind),
                    spec["level"],
                    run_index=index,
                    injected_step=spec["start"],
                    hold_steps=spec["hold"],
                )
            )
        else:
            plans.append(
                make_run_plan(params, index, kind, instrument_faults=tuple(spec["episodes"]))
            )
    counts = Counter(p.scenario.value for p in plans)
    _LOG.info("Planned %d runs (%d normal): %s", len(plans), n_normal, dict(sorted(counts.items())))
    return PlannedDataset(
        plans=plans,
        budget=budget,
        kinds=_kind_plans(plans, quotas, n_sensors),
        n_sensors=n_sensors,
        by_scenario=dict(sorted(counts.items())),
    )


def _kind_plans(
    plans: Sequence[RunPlan], quotas: Mapping[ScenarioKind, float], n_sensors: int
) -> dict[ScenarioKind, KindPlan]:
    """Summarise what the final plans hold per fault type."""
    out: dict[ScenarioKind, KindPlan] = {}
    for kind in FAULT_KINDS:
        runs = [p for p in plans if p.scenario is kind]
        levels: Counter[str] = Counter()
        episodes = 0
        for plan in runs:
            if plan.process_fault is not None:
                levels[plan.process_fault.level.value] += 1
                episodes += 1
            for episode in plan.instrument_faults:
                levels[episode.level.value] += 1
                episodes += 1
        out[kind] = KindPlan(
            kind=kind,
            n_runs=len(runs),
            quota_readings=quotas[kind],
            planned_readings=sum(p.expected_labelled_readings(n_sensors) for p in runs),
            episodes=episodes,
            levels={lv.value: levels.get(lv.value, 0) for lv in _LEVELS},
        )
    return out
