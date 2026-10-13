"""Escenarios, severidad, programa de carga y planes de run de la simulacion.

`FAULT_SEVERITY` es la UNICA tabla de severidad (D10): la del TDD 2.2, estatica.
Es distinta de la severidad dinamica del validador de Fase 2 (H9): aquella dice
cuanta confianza tiene un detector, esta dice cuanto importa el fallo inyectado.
Este modulo no importa nada de `data/validation`: los valores de `FaultSeverity`
coinciden con los de `Severity` por texto, no por tipo.

El programa de carga (`LoadProfile`) es lo que define la operacion normal: rampas
acotadas entre `load_min` y `load_max` con pendiente maxima `max_ramp_percent_per_min`.
Esa cota es la que justifica el umbral de la regla potencia/flujo
(`power_flux_threshold_percent`): una rampa mayor deja de ser operacion normal para
las constraints.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from data.simulation.simulation_params import SimulationParams

_MIN_RAMP_S = 1.0
"""Duracion minima de una rampa [s]: evita nodos coincidentes si el objetivo ~ la carga."""

_MIN_RAMP_RATE_FRACTION = 0.2
"""Las rampas usan entre el 20 % y el 100 % de la pendiente maxima."""

N_RUN_STREAMS = 3
"""Flujos de aleatoriedad por run: carga, ruido de medicion y fallos."""


class FaultAxis(StrEnum):
    """Eje al que pertenece un fallo (D3: ortogonales)."""

    NONE = "none"
    INSTRUMENT = "instrument"
    PROCESS = "process"


class FaultSeverity(StrEnum):
    """Severidad estatica del TDD 2.2."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FaultLevel(StrEnum):
    """Nivel de magnitud de un fallo de proceso (bajo, medio, alto).

    El valor es la clave de `fault_tiers:` en configs/plant_parameters.yaml.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ScenarioKind(StrEnum):
    """Un escenario por run (D3): operacion normal y los 8 tipos del TDD 2.2.

    El valor es el nombre del directorio `fault_type=<valor>` del dataset.
    """

    NORMAL = "normal"
    DRIFT = "drift"
    STUCK = "stuck"
    NOISE_SPIKE = "noise_spike"
    BIAS = "bias"
    INTERMITTENT = "intermittent"
    CORE_TEMP_EXCURSION = "core_temp_excursion"
    COOLANT_FLOW_REDUCTION = "coolant_flow_reduction"
    VOID_FORMATION = "void_formation"

    @property
    def axis(self) -> FaultAxis:
        """Return the axis this scenario perturbs (NONE for normal operation)."""
        return FAULT_AXIS[self]

    @property
    def severity(self) -> FaultSeverity | None:
        """Return the static TDD severity, or None for normal operation."""
        return FAULT_SEVERITY.get(self)


FAULT_AXIS: Mapping[ScenarioKind, FaultAxis] = MappingProxyType(
    {
        ScenarioKind.NORMAL: FaultAxis.NONE,
        ScenarioKind.DRIFT: FaultAxis.INSTRUMENT,
        ScenarioKind.STUCK: FaultAxis.INSTRUMENT,
        ScenarioKind.NOISE_SPIKE: FaultAxis.INSTRUMENT,
        ScenarioKind.BIAS: FaultAxis.INSTRUMENT,
        ScenarioKind.INTERMITTENT: FaultAxis.INSTRUMENT,
        ScenarioKind.CORE_TEMP_EXCURSION: FaultAxis.PROCESS,
        ScenarioKind.COOLANT_FLOW_REDUCTION: FaultAxis.PROCESS,
        ScenarioKind.VOID_FORMATION: FaultAxis.PROCESS,
    }
)
"""Eje de cada escenario (TDD 2.2: 5 de instrumento, 3 de proceso)."""

FAULT_SEVERITY: Mapping[ScenarioKind, FaultSeverity] = MappingProxyType(
    {
        ScenarioKind.DRIFT: FaultSeverity.HIGH,
        ScenarioKind.STUCK: FaultSeverity.CRITICAL,
        ScenarioKind.NOISE_SPIKE: FaultSeverity.LOW,
        ScenarioKind.BIAS: FaultSeverity.HIGH,
        ScenarioKind.INTERMITTENT: FaultSeverity.MEDIUM,
        ScenarioKind.CORE_TEMP_EXCURSION: FaultSeverity.CRITICAL,
        ScenarioKind.COOLANT_FLOW_REDUCTION: FaultSeverity.HIGH,
        ScenarioKind.VOID_FORMATION: FaultSeverity.CRITICAL,
    }
)
"""Severidad estatica de TDD 2.2 (D10). Unica fuente: nada mas la redefine."""


@dataclass(frozen=True)
class LoadProfile:
    """Programa de carga de la operacion normal.

    Una traza lineal a tramos: carga constante `warmup_steps` pasos, y despues
    ciclos de (mantener un tiempo aleatorio, rampar a una carga objetivo aleatoria
    con una pendiente aleatoria <= la maxima). Se evalua en los instantes de
    muestreo, de modo que la pendiente entre dos muestras consecutivas tambien
    queda acotada por `max_ramp_percent_per_min`.

    Attributes:
        load_min: Carga minima (fraccion del nominal).
        load_max: Carga maxima.
        max_ramp_percent_per_min: Pendiente maxima [% del nominal por minuto].
        hold_minutes_min: Minimo de tiempo a carga constante entre rampas [min].
        hold_minutes_max: Maximo de tiempo a carga constante entre rampas [min].
        warmup_steps: Pasos iniciales a carga constante.

    Raises:
        ValueError: If the limits are inconsistent.
    """

    load_min: float
    load_max: float
    max_ramp_percent_per_min: float
    hold_minutes_min: float
    hold_minutes_max: float
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        """Validate the profile limits."""
        if not 0.0 < self.load_min < self.load_max:
            raise ValueError(
                f"LoadProfile needs 0 < load_min < load_max, got {self.load_min} and "
                f"{self.load_max}."
            )
        if not (math.isfinite(self.max_ramp_percent_per_min) and self.max_ramp_percent_per_min > 0):
            raise ValueError(
                f"LoadProfile.max_ramp_percent_per_min must be > 0, got "
                f"{self.max_ramp_percent_per_min}."
            )
        if not 0.0 < self.hold_minutes_min <= self.hold_minutes_max:
            raise ValueError(
                f"LoadProfile needs 0 < hold_minutes_min <= hold_minutes_max, got "
                f"{self.hold_minutes_min} and {self.hold_minutes_max}."
            )
        if self.warmup_steps < 0:
            raise ValueError(f"LoadProfile.warmup_steps must be >= 0, got {self.warmup_steps}.")

    @classmethod
    def from_params(cls, params: SimulationParams) -> LoadProfile:
        """Build the profile from the simulation: section.

        Args:
            params: Resolved simulation parameters.

        Returns:
            The LoadProfile with the warm-up taken from `warmup_fraction`.
        """
        return cls(
            load_min=params.load_min_fraction,
            load_max=params.load_max_fraction,
            max_ramp_percent_per_min=params.max_ramp_percent_per_min,
            hold_minutes_min=params.load_hold_minutes_min,
            hold_minutes_max=params.load_hold_minutes_max,
            warmup_steps=params.warmup_steps,
        )

    @property
    def max_ramp_fraction_per_s(self) -> float:
        """Return the maximum slope as a fraction of nominal per second."""
        return self.max_ramp_percent_per_min / 100.0 / 60.0

    def sample(self, rng: np.random.Generator, n_steps: int, dt_s: float) -> NDArray[np.float64]:
        """Return the load at each of n_steps sampling instants.

        Args:
            rng: Generador del run (flujo de carga). Es lo unico que da aleatoriedad.
            n_steps: Muestras a emitir (al menos 2).
            dt_s: Cadencia de muestreo [s].

        Returns:
            Array of shape (n_steps,) with values in [load_min, load_max]; the
            first `warmup_steps` samples are equal.

        Raises:
            ValueError: If n_steps < 2 or dt_s <= 0.
        """
        if n_steps < 2:
            raise ValueError(f"a load profile needs at least 2 samples, got {n_steps}.")
        if not dt_s > 0.0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}.")
        horizon = (n_steps - 1) * dt_s
        current = float(rng.uniform(self.load_min, self.load_max))
        t = self.warmup_steps * dt_s
        knots_t = [0.0, t]
        knots_l = [current, current]
        while t < horizon:
            t += float(rng.uniform(self.hold_minutes_min, self.hold_minutes_max)) * 60.0
            knots_t.append(t)
            knots_l.append(current)
            target = float(rng.uniform(self.load_min, self.load_max))
            rate = self.max_ramp_fraction_per_s * float(
                rng.uniform(_MIN_RAMP_RATE_FRACTION, 1.0)
            )
            t += max(abs(target - current) / rate, _MIN_RAMP_S)
            knots_t.append(t)
            knots_l.append(target)
            current = target
        times = np.arange(n_steps, dtype=np.float64) * dt_s
        loads: NDArray[np.float64] = np.interp(times, knots_t, knots_l)
        return loads


@dataclass(frozen=True)
class ProcessFaultSpec:
    """Envolvente temporal de un fallo de proceso dentro de un run.

    Todo se expresa en PASOS de muestreo desde `injected_step`. La magnitud
    relativa del fallo (`envelope`) sube linealmente durante `onset_steps`, se
    mantiene `hold_steps` y vuelve a cero linealmente durante `recovery_steps`.
    Si `hold_steps` es None el fallo persiste hasta el final del run y no hay
    recuperacion.

    La ventana etiquetada (`is_active`) va de `injected_step` hasta
    `injected_step + onset + hold + recovery`, ambos incluidos: el fallo se
    etiqueta en TODOS los sensores desde su inicio, aunque la magnitud sea aun
    cero en esa primera muestra.

    Attributes:
        level: Nivel de magnitud (clave de `fault_tiers:`).
        injected_step: Indice de la primera muestra del fallo (`injected_at`).
        onset_steps: Pasos de la rampa de entrada (0 = entra de golpe).
        hold_steps: Pasos a magnitud plena; None = persiste hasta el final.
        recovery_steps: Pasos de la rampa de salida (0 = termina de golpe).

    Raises:
        ValueError: If a step count is negative, or a recovery is requested
            for a fault that persists.
    """

    level: FaultLevel
    injected_step: int
    onset_steps: int = 0
    hold_steps: int | None = None
    recovery_steps: int = 0

    def __post_init__(self) -> None:
        """Validate the timeline."""
        if self.injected_step < 0:
            raise ValueError(
                f"ProcessFaultSpec.injected_step must be >= 0, got {self.injected_step}."
            )
        if self.onset_steps < 0:
            raise ValueError(f"ProcessFaultSpec.onset_steps must be >= 0, got {self.onset_steps}.")
        if self.hold_steps is not None and self.hold_steps < 0:
            raise ValueError(f"ProcessFaultSpec.hold_steps must be >= 0, got {self.hold_steps}.")
        if self.recovery_steps < 0:
            raise ValueError(
                f"ProcessFaultSpec.recovery_steps must be >= 0, got {self.recovery_steps}."
            )
        if self.hold_steps is None and self.recovery_steps > 0:
            raise ValueError(
                "ProcessFaultSpec: a persistent fault (hold_steps=None) cannot recover."
            )

    @property
    def last_active_step(self) -> int | None:
        """Return the last labelled step, or None if the fault persists to the end."""
        if self.hold_steps is None:
            return None
        return self.injected_step + self.onset_steps + self.hold_steps + self.recovery_steps

    def is_active(self, step: int) -> bool:
        """Return whether a step belongs to the labelled window.

        Args:
            step: Sample index.

        Returns:
            True from `injected_step` to `last_active_step` (or to the end).
        """
        if step < self.injected_step:
            return False
        last = self.last_active_step
        return last is None or step <= last

    def envelope(self, step: float) -> float:
        """Return the relative magnitude of the fault at a (fractional) step.

        Args:
            step: Position in samples, measured from the start of the run. It
                may be fractional (the simulator evaluates it at sub-steps).

        Returns:
            0 before `injected_step`, then the onset ramp, 1 while held and the
            recovery ramp back to 0.
        """
        t = step - self.injected_step
        if t < 0.0:
            return 0.0
        if t < self.onset_steps:
            return t / self.onset_steps
        if self.hold_steps is None:
            return 1.0
        t_hold_end = self.onset_steps + self.hold_steps
        if t <= t_hold_end:
            return 1.0
        if self.recovery_steps > 0 and t < t_hold_end + self.recovery_steps:
            return 1.0 - (t - t_hold_end) / self.recovery_steps
        return 0.0


@dataclass(frozen=True)
class RunPlan:
    """Plan de un run: que escenario, cuantos pasos, cuando y con que semilla.

    El plan guarda la semilla como `(entropy, spawn_key)` y NO como un
    `SeedSequence` vivo: `SeedSequence.spawn` es con estado, y un plan debe dar
    siempre los mismos generadores por muy a menudo que se consulte.

    Attributes:
        run_index: Posicion del run (0-based).
        run_id: Identificador estable, `sim_run_001`.
        scenario: Escenario del run.
        n_steps: Pasos del run.
        start_time: Instante (UTC) del primer paso.
        seed_entropy: Entropia de la semilla maestra.
        spawn_key: Clave de spawn del run bajo la semilla maestra.
        process_fault: Envolvente del fallo de proceso; obligatoria si el
            escenario es de proceso y prohibida en cualquier otro.

    Raises:
        ValueError: If the process-fault envelope does not match the scenario,
            or a finite window does not fit in the run.
    """

    run_index: int
    run_id: str
    scenario: ScenarioKind
    n_steps: int
    start_time: datetime
    seed_entropy: int
    spawn_key: tuple[int, ...]
    process_fault: ProcessFaultSpec | None = None

    def __post_init__(self) -> None:
        """Check that the process-fault envelope matches the scenario."""
        is_process = self.scenario.axis is FaultAxis.PROCESS
        if is_process and self.process_fault is None:
            raise ValueError(
                f"{self.run_id}: scenario '{self.scenario.value}' needs a ProcessFaultSpec."
            )
        if not is_process and self.process_fault is not None:
            raise ValueError(
                f"{self.run_id}: scenario '{self.scenario.value}' cannot carry a ProcessFaultSpec."
            )
        spec = self.process_fault
        if spec is not None:
            last = spec.last_active_step
            if spec.injected_step >= self.n_steps or (last is not None and last >= self.n_steps):
                raise ValueError(
                    f"{self.run_id}: the fault window (start {spec.injected_step}, "
                    f"end {last}) does not fit in {self.n_steps} steps."
                )

    def rngs(self) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
        """Return the (load, noise, fault) generators of this run.

        Returns:
            Three independent generators; the same three every call.
        """
        sequence = np.random.SeedSequence(self.seed_entropy, spawn_key=self.spawn_key)
        children = sequence.spawn(N_RUN_STREAMS)
        load, noise, fault = (np.random.default_rng(child) for child in children)
        return load, noise, fault

    def timestamps(self, dt_s: float) -> list[datetime]:
        """Return the instant of each step.

        Args:
            dt_s: Cadencia de muestreo [s].

        Returns:
            n_steps timezone-aware datetimes, strictly increasing.
        """
        return [self.start_time + timedelta(seconds=k * dt_s) for k in range(self.n_steps)]


def run_id_for(run_index: int) -> str:
    """Return the stable identifier of a run.

    Args:
        run_index: Posicion del run (0-based).

    Returns:
        `sim_run_001` for index 0 (three digits, widening past 999).
    """
    return f"sim_run_{run_index + 1:03d}"


def normal_run_plans(
    params: SimulationParams, seed: int | None = None, n_runs: int | None = None
) -> list[RunPlan]:
    """Return one normal-operation plan per run, with independent seeds.

    Los runs son contiguos en el tiempo (`start_time + i * duracion_del_run`), de
    modo que ningun timestamp se repite entre runs y un consumidor ordenado por
    tiempo no ve retrocesos.

    Args:
        params: Resolved simulation parameters.
        seed: Semilla maestra. Por defecto `params.openmc_seed`; la usa
            `derive_sim_spans` para observar la operacion normal con una semilla
            propia, distinta de la del dataset.
        n_runs: Numero de runs. Por defecto `params.n_runs`.

    Returns:
        The plans, scenario NORMAL, seeds from `SeedSequence(seed).spawn(n_runs)`
        (D12).

    Raises:
        ValueError: If n_runs is not positive.
    """
    master_seed = params.openmc_seed if seed is None else int(seed)
    count = params.n_runs if n_runs is None else int(n_runs)
    if count < 1:
        raise ValueError(f"n_runs must be >= 1, got {count}.")
    children = np.random.SeedSequence(master_seed).spawn(count)
    plans: list[RunPlan] = []
    for index, child in enumerate(children):
        plans.append(
            RunPlan(
                run_index=index,
                run_id=run_id_for(index),
                scenario=ScenarioKind.NORMAL,
                n_steps=params.run_length_steps,
                start_time=params.start_time + timedelta(seconds=index * params.run_duration_s),
                seed_entropy=master_seed,
                spawn_key=tuple(int(k) for k in child.spawn_key),
            )
        )
    return plans
