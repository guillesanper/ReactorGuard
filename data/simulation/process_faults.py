"""Fallos de proceso del reactor (F3-M4): excursion termica, reduccion de caudal y vacio.

Un fallo de proceso perturba las ECUACIONES del modelo (`ModelInputs`), no la
medicion: el estado verdadero cambia de verdad y los sensores lo miden con su
ruido de siempre. Por eso `corrupted` sigue en False y la etiqueta cubre todos
los sensores del paso (una planta que cambio no es un instrumento corrupto).

    core_temp_excursion      rampa de insercion de reactividad con el PI de barras
                             deshabilitado; n, T_f y T_out suben hasta que la
                             realimentacion negativa lo frena.
    coolant_flow_reduction   la demanda de bomba cae a una fraccion con constante
                             de tiempo, mas un termino oscilatorio pequeno.
    void_formation           una fuga despresuriza el primario hasta que T_c supera
                             T_sat(P); el vacio crece y realimenta la reactividad.
                             Su nivel es el margen de saturacion objetivo, no una
                             fuga en bar (`leak_for_margin`).

Cada nivel (`low`, `medium`, `high`) se fija en `fault_tiers:` de
configs/plant_parameters.yaml por un OBJETIVO MEDIBLE (pico de T_out, margen
minimo a T_sat, caudal minimo...), no a ojo: la banda esperada vive junto al
parametro y `measure_level_response` la mide sobre el modelo.

Observabilidad por cadencia: a `dt` la oscilacion de la bomba solo es observable si
su periodo es mayor que `2 dt` (Nyquist; 360 s a 180 s). Una oscilacion mas rapida
se ALIASA a un periodo aparente mayor (`aliased_period_s`): el estado verdadero la
tiene, la muestra no puede distinguirla de una lenta. Se integra por sub-pasos
para que el aliasing sea el real y no un artefacto de la rejilla.

Rango de validez (medido en F3-M4, ver Docs/physics_constraints.md): la tabla de
agua no extrapola y el modelo es de agua liquida, de modo que cada fallo tiene un
tope fuera del cual deja de ser fisico (`MAX_REACTIVITY_PCM`, `MIN_PUMP_DEMAND`,
`MIN_SATURATION_MARGIN_K`). Dentro de esos topes el modelo todavia puede rozar el
borde de la tabla de presiones en el pico de un transitorio de vacio; el limitador
de presion de `thermal_model` lo sostiene en lugar de abortar el run, y la matriz
medida cuenta las muestras en que eso ocurre (`pressure_limited_steps`).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray

from data.simulation.dataset_builder import RunSimulation, simulate_run
from data.simulation.scenario import (
    FaultAxis,
    FaultLevel,
    ProcessFaultSpec,
    RunPlan,
    ScenarioKind,
    run_id_for,
)
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.simulation_params import SimulationParams
from data.simulation.thermal_model import ModelInputs, ReactorModel, StateIndex

_LOG = logging.getLogger(__name__)

PROCESS_KINDS: tuple[ScenarioKind, ...] = (
    ScenarioKind.CORE_TEMP_EXCURSION,
    ScenarioKind.COOLANT_FLOW_REDUCTION,
    ScenarioKind.VOID_FORMATION,
)
"""Los tres escenarios del eje de proceso (TDD 2.2, filas 6 a 8)."""

PCM = 1.0e-5
"""1 pcm = 1e-5 dk/k."""

MAX_REACTIVITY_PCM = 700.0
"""Insercion maxima de un nivel [pcm]: con 800 pcm y las barras deshabilitadas la
presion llega al techo de 170 bar de la tabla (medido en F3-M4 a carga 1,0); el
limitador de presion del modelo la sostiene alli, pero un nivel que vive pegado
al limitador ya no mide la realimentacion del presurizador."""

MIN_PUMP_DEMAND = 0.45
"""Demanda minima de la bomba. A carga 1,0 el acantilado esta entre 0,41 (margen
0,7 K, vacio 0,016) y 0,40 (margen -17 K, vacio 0,30, n cae a 0,17): por debajo, el
fallo deja de ser una reduccion de caudal y pasa a ser una ebullicion, que es el
escenario de vacio. 0,45 es el ultimo valor con margen >= 3 K y sin vacio con la
oscilacion maxima (medido en F3-M4)."""

MAX_SATURATION_MARGIN_K = 5.0
"""Margen de saturacion maximo del vacio [K]: por encima el fallo no llega a la
ebullicion subenfriada (el vacio empieza 2 K antes de T_sat)."""

MIN_SATURATION_MARGIN_K = -16.0
"""Margen de saturacion minimo del vacio [K]: a -16 K y carga 0,7 la presion llega
a 61,2 bar, al borde inferior de la tabla (60 bar); por debajo el limitador de
presion del modelo la sostiene en el suelo y el nivel deja de ser fisico."""

MAX_OSCILLATION_AMPLITUDE = 0.10
"""Amplitud maxima del termino oscilatorio (fraccion de la demanda nominal)."""

_POINTS_PER_PERIOD = 6.0
"""Sub-pasos minimos por periodo de oscilacion al integrar."""

_TAU_STEPS = 4.0
"""Sub-pasos minimos por constante de tiempo de la demanda."""

_MAGNITUDE_KEYS: dict[ScenarioKind, str] = {
    ScenarioKind.CORE_TEMP_EXCURSION: "reactivity_pcm",
    ScenarioKind.COOLANT_FLOW_REDUCTION: "demand_fraction",
    ScenarioKind.VOID_FORMATION: "saturation_margin_k",
}
"""Clave del parametro de magnitud de cada tipo en `fault_tiers:`."""

_LEVEL_ORDER: tuple[FaultLevel, ...] = (FaultLevel.LOW, FaultLevel.MEDIUM, FaultLevel.HIGH)

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
"""Instante de partida de las medidas de nivel (no entra en ninguna cifra)."""

_NAME = "plant_parameters.yaml: 'fault_tiers'"


# ----------------------------------------------------------------------
# Niveles: configuracion tipada
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LevelResponse:
    """Respuesta medida del modelo a un nivel de fallo (carga constante).

    Attributes:
        t_out_peak_c: Pico de T_out (pierna caliente) [C].
        t_out_rise_k: Pico de T_out menos T_out nominal [K].
        t_fuel_peak_c: Pico de la temperatura del combustible [C].
        delta_t_peak_k: Pico de T_out - T_in (salto a traves del nucleo) [K].
        margin_min_k: Margen minimo a la saturacion T_sat(P) - T_out [K]
            (negativo si hay ebullicion segun la regla 6).
        flow_min_kg_s: Caudal total minimo [kg/s].
        n_peak: Pico de la potencia neutronica (fraccion del nominal).
        n_min: Minimo de la potencia neutronica.
        void_peak: Pico de la fraccion de vacio.
        p_min_bar: Presion minima del primario [bar].
        p_max_bar: Presion maxima del primario [bar].
        max_state_dt_dt_c_s: Maxima |dT/dt| del estado (combustible, nucleo, GV) [C/s].
    """

    t_out_peak_c: float
    t_out_rise_k: float
    t_fuel_peak_c: float
    delta_t_peak_k: float
    margin_min_k: float
    flow_min_kg_s: float
    n_peak: float
    n_min: float
    void_peak: float
    p_min_bar: float
    p_max_bar: float
    max_state_dt_dt_c_s: float


RESPONSE_METRICS: tuple[str, ...] = tuple(f.name for f in dataclasses.fields(LevelResponse))
"""Metricas que una banda objetivo puede nombrar."""


@dataclass(frozen=True)
class ResponseTarget:
    """Banda objetivo de una metrica de la respuesta de un nivel.

    Attributes:
        metric: Nombre de un campo de `LevelResponse`.
        low: Limite inferior (incluido).
        high: Limite superior (incluido).
    """

    metric: str
    low: float
    high: float

    def contains(self, value: float) -> bool:
        """Return whether a measured value lies inside the band."""
        return self.low <= value <= self.high


@dataclass(frozen=True)
class LevelSpec:
    """Un nivel de magnitud de un fallo de proceso.

    Attributes:
        magnitude: Parametro que gobierna el nivel: insercion [pcm], fraccion de
            demanda de bomba o margen de saturacion objetivo [K], segun el tipo.
        oscillation_amplitude: Amplitud del termino oscilatorio de la demanda
            (fraccion del nominal); 0 salvo en la reduccion de caudal.
        targets: Bandas objetivo medibles del nivel.
    """

    magnitude: float
    oscillation_amplitude: float
    targets: tuple[ResponseTarget, ...]


@dataclass(frozen=True)
class KindTiers:
    """Niveles y cronologia por defecto de un tipo de fallo de proceso.

    Attributes:
        kind: Escenario.
        onset_minutes: Duracion de la rampa de entrada [min] (0 si la dinamica la
            da una constante de tiempo).
        time_constant_minutes: Constante de tiempo de la demanda de bomba [min]
            (solo reduccion de caudal; 0 en los demas).
        oscillation_period_s: Periodo del termino oscilatorio [s] (solo caudal).
        hold_steps: Pasos a magnitud plena por defecto.
        recovery_steps: Pasos de recuperacion por defecto (0 = termina de golpe).
        levels: Nivel -> especificacion; siempre `low`, `medium` y `high`.
    """

    kind: ScenarioKind
    onset_minutes: float
    time_constant_minutes: float
    oscillation_period_s: float
    hold_steps: int
    recovery_steps: int
    levels: Mapping[FaultLevel, LevelSpec]

    def level(self, level: FaultLevel) -> LevelSpec:
        """Return the specification of one level."""
        return self.levels[level]

    def onset_steps(self, dt_s: float) -> int:
        """Return the onset ramp length in samples at a cadence.

        Args:
            dt_s: Cadencia de muestreo [s].

        Returns:
            0 for the flow reduction (its onset is the time constant); at least 1
            for the ramped faults.
        """
        if self.onset_minutes <= 0.0:
            return 0
        return max(1, round(self.onset_minutes * 60.0 / dt_s))

    def substeps(self, dt_s: float) -> int:
        """Return how many sub-intervals each sampling interval is integrated in.

        Solo la reduccion de caudal tiene dinamica mas rapida que la cadencia: el
        periodo de la oscilacion y la constante de tiempo se resuelven con al menos
        `_POINTS_PER_PERIOD` y `_TAU_STEPS` sub-pasos.

        Args:
            dt_s: Cadencia de muestreo [s].

        Returns:
            1 when the fault has no sub-sample dynamics.
        """
        if self.kind is not ScenarioKind.COOLANT_FLOW_REDUCTION:
            return 1
        finest = min(
            self.oscillation_period_s / _POINTS_PER_PERIOD,
            self.time_constant_minutes * 60.0 / _TAU_STEPS,
        )
        return max(1, math.ceil(dt_s / finest))


@dataclass(frozen=True)
class FaultTiers:
    """`fault_tiers:` resuelto: los tres tipos de proceso.

    Attributes:
        kinds: Escenario -> niveles y cronologia.
    """

    kinds: Mapping[ScenarioKind, KindTiers]

    def for_kind(self, kind: ScenarioKind) -> KindTiers:
        """Return the tiers of one process scenario.

        Raises:
            KeyError: If the scenario is not a process fault.
        """
        if kind not in self.kinds:
            raise KeyError(f"fault_tiers: '{kind.value}' is not a process fault.")
        return self.kinds[kind]


def _require(section: Mapping[str, Any], key: str, label: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Raises:
        KeyError: If key is missing.
    """
    if key not in section:
        raise KeyError(f"{label}: missing required key '{key}'.")
    return section[key]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    """Return value if it is a mapping.

    Raises:
        TypeError: If it is not.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{label}: must be a mapping.")
    return value


def _finite(value: Any, label: str) -> float:
    """Return value as a finite float.

    Raises:
        ValueError: If it is not a finite number.
    """
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: must be a number, got {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}: must be finite, got {value!r}.")
    return number


def _whole(value: Any, label: str, minimum: int) -> int:
    """Return value as an int >= minimum.

    Raises:
        ValueError: If it is not a whole number >= minimum.
    """
    number = _finite(value, label)
    if number != int(number) or number < minimum:
        raise ValueError(f"{label}: must be a whole number >= {minimum}, got {value!r}.")
    return int(number)


def _targets(raw: Any, label: str) -> tuple[ResponseTarget, ...]:
    """Parse an `expect:` mapping metric -> [low, high].

    Raises:
        TypeError: If the block or a band has the wrong type.
        ValueError: If a metric is unknown or a band is inverted.
    """
    block = _mapping(raw, f"{label}.expect")
    targets: list[ResponseTarget] = []
    for metric, band in block.items():
        if metric not in RESPONSE_METRICS:
            raise ValueError(
                f"{label}.expect: unknown metric '{metric}'; valid: {list(RESPONSE_METRICS)}."
            )
        if not isinstance(band, Sequence) or isinstance(band, str) or len(band) != 2:
            raise TypeError(f"{label}.expect.{metric}: must be a [low, high] pair.")
        low = _finite(band[0], f"{label}.expect.{metric}[0]")
        high = _finite(band[1], f"{label}.expect.{metric}[1]")
        if low > high:
            raise ValueError(f"{label}.expect.{metric}: low {low} exceeds high {high}.")
        targets.append(ResponseTarget(metric=metric, low=low, high=high))
    return tuple(targets)


def _check_magnitude(kind: ScenarioKind, value: float, label: str) -> None:
    """Raise if a level magnitude is outside the validity range of its fault.

    Raises:
        ValueError: If the magnitude is outside the validity range.
    """
    if kind is ScenarioKind.CORE_TEMP_EXCURSION:
        ok = 0.0 < value <= MAX_REACTIVITY_PCM
        rule = f"(0, {MAX_REACTIVITY_PCM}] pcm"
    elif kind is ScenarioKind.COOLANT_FLOW_REDUCTION:
        ok = MIN_PUMP_DEMAND <= value < 1.0
        rule = f"[{MIN_PUMP_DEMAND}, 1)"
    else:
        ok = MIN_SATURATION_MARGIN_K <= value <= MAX_SATURATION_MARGIN_K
        rule = f"[{MIN_SATURATION_MARGIN_K}, {MAX_SATURATION_MARGIN_K}] K"
    if not ok:
        raise ValueError(f"{label}: {value} is outside the validity range {rule}.")


def _check_monotone(kind: ScenarioKind, values: Sequence[float], label: str) -> None:
    """Raise unless the three levels grow in severity.

    La excursion crece con el parametro; la demanda de bomba y el margen de
    saturacion del vacio decrecen.

    Raises:
        ValueError: If low, medium and high are not in increasing severity.
    """
    ordered = list(values)
    if kind is not ScenarioKind.CORE_TEMP_EXCURSION:
        ordered = [-v for v in ordered]
    if not ordered[0] < ordered[1] < ordered[2]:
        raise ValueError(f"{label}: low, medium and high must increase in severity, got {values}.")


def _parse_kind(kind: ScenarioKind, raw: Any) -> KindTiers:
    """Parse the block of one process fault.

    Raises:
        KeyError: If a required key or level is missing.
        TypeError: If a block has the wrong type.
        ValueError: If a value is out of range.
    """
    label = f"{_NAME}.{kind.value}"
    block = _mapping(raw, label)
    flow = kind is ScenarioKind.COOLANT_FLOW_REDUCTION
    onset = _finite(_require(block, "onset_minutes", label), f"{label}.onset_minutes")
    tau = _finite(block.get("time_constant_minutes", 0.0), f"{label}.time_constant_minutes")
    period = _finite(block.get("oscillation_period_s", 0.0), f"{label}.oscillation_period_s")
    if flow:
        if onset != 0.0:
            raise ValueError(f"{label}.onset_minutes: must be 0 (the time constant is the onset).")
        if tau <= 0.0 or period <= 0.0:
            raise ValueError(
                f"{label}: time_constant_minutes and oscillation_period_s must be > 0."
            )
    elif onset <= 0.0:
        raise ValueError(f"{label}.onset_minutes: must be > 0, got {onset}.")
    levels_raw = _mapping(_require(block, "levels", label), f"{label}.levels")
    levels: dict[FaultLevel, LevelSpec] = {}
    for level in _LEVEL_ORDER:
        entry_label = f"{label}.levels.{level.value}"
        entry = _mapping(_require(levels_raw, level.value, f"{label}.levels"), entry_label)
        magnitude = _finite(
            _require(entry, _MAGNITUDE_KEYS[kind], entry_label), f"{entry_label}.magnitude"
        )
        _check_magnitude(kind, magnitude, entry_label)
        amplitude = _finite(entry.get("oscillation_amplitude", 0.0), f"{entry_label}.amplitude")
        if not 0.0 <= amplitude <= MAX_OSCILLATION_AMPLITUDE:
            raise ValueError(
                f"{entry_label}.oscillation_amplitude: must be in "
                f"[0, {MAX_OSCILLATION_AMPLITUDE}], got {amplitude}."
            )
        if amplitude > 0.0 and not flow:
            raise ValueError(f"{entry_label}: only the flow reduction oscillates.")
        levels[level] = LevelSpec(
            magnitude=magnitude,
            oscillation_amplitude=amplitude,
            targets=_targets(entry.get("expect", {}), entry_label),
        )
    _check_monotone(kind, [levels[lv].magnitude for lv in _LEVEL_ORDER], label)
    return KindTiers(
        kind=kind,
        onset_minutes=onset,
        time_constant_minutes=tau,
        oscillation_period_s=period,
        hold_steps=_whole(_require(block, "hold_steps", label), f"{label}.hold_steps", 1),
        recovery_steps=_whole(
            _require(block, "recovery_steps", label), f"{label}.recovery_steps", 0
        ),
        levels=levels,
    )


def parse_fault_tiers(raw: Mapping[str, Any]) -> FaultTiers:
    """Resolve the `fault_tiers:` block of plant_parameters.yaml.

    Args:
        raw: `PlantParams.fault_tiers` (the parsed mapping).

    Returns:
        The typed FaultTiers with the three process faults.

    Raises:
        KeyError: If a fault, a level or a required key is missing.
        TypeError: If a block has the wrong YAML type.
        ValueError: If a magnitude is outside its validity range, the levels do
            not grow in severity or a band names an unknown metric.
    """
    kinds = {kind: _parse_kind(kind, _require(raw, kind.value, _NAME)) for kind in PROCESS_KINDS}
    return FaultTiers(kinds=kinds)


# ----------------------------------------------------------------------
# Perfiles de entradas del modelo
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FaultInputs:
    """Entradas del modelo de un run con fallo de proceso.

    Attributes:
        sample: Entradas en cada muestra (longitud `n_steps`).
        fine: Entradas en la rejilla fina (`(n_steps - 1) * substeps + 1`), o None
            si el fallo no tiene dinamica mas rapida que la cadencia.
        substeps: Sub-intervalos por intervalo de muestreo.
        envelope: Magnitud relativa del fallo (0 a 1) en cada muestra.
    """

    sample: tuple[ModelInputs, ...]
    fine: tuple[ModelInputs, ...] | None
    substeps: int
    envelope: NDArray[np.float64]


def leak_for_margin(model: ReactorModel, load: float, margin_k: float) -> float:
    """Return the leak that puts the saturation margin at a target, open loop.

    El nivel del vacio se define por su OBJETIVO (el margen de saturacion
    `T_sat(P) - T_out`), no por una fuga en bar: la misma fuga produce
    consecuencias distintas a 70 y a 100 % de carga (T_out cambia 6 K) y un nivel
    "medio" que a plena carga ebulle no llegaria ni al vacio a carga parcial. La
    fuga que corresponde al objetivo se calcula sobre el equilibrio de la carga
    vigente: baja la presion hasta que `T_sat(P) = T_out + margin_k`.

    El margen REAL minimo difiere (el vacio realimenta y T_out baja): lo mide
    `measure_level_response` y lo acota la banda `expect:` del nivel.

    Args:
        model: Modelo termico (aporta el equilibrio y la tabla de agua).
        load: Carga de turbina vigente (fraccion del nominal).
        margin_k: Margen de saturacion objetivo [K]; negativo = ebullicion.

    Returns:
        The equivalent depressurization [bar], >= 0.

    Raises:
        ValueError: If the load is outside the model range or the target
            saturation temperature falls outside the water table.
    """
    state = model.steady_state(load)
    t_core = float(state[StateIndex.T_CORE])
    try:
        p_target = float(model.table.p_sat_bar(t_core + margin_k))
    except ValueError as exc:
        raise ValueError(
            f"saturation margin {margin_k} K at load {load} puts T_sat at "
            f"{t_core + margin_k:.1f} C, outside the water table: {exc}"
        ) from exc
    return max(float(state[StateIndex.PRESSURE]) - p_target, 0.0)


def _point(
    model: ReactorModel,
    tiers: KindTiers,
    level: LevelSpec,
    spec: ProcessFaultSpec,
    load: float,
    position: float,
    dt_s: float,
    rods_enabled: bool,
) -> ModelInputs:
    """Return the model inputs at a (fractional) sample position.

    Args:
        model: Modelo termico (el vacio calcula su fuga sobre el equilibrio).
        tiers: Tipo de fallo.
        level: Nivel.
        spec: Envolvente temporal.
        load: Carga de turbina en ese instante.
        position: Posicion en muestras desde el inicio del run.
        dt_s: Cadencia de muestreo [s].
        rods_enabled: Estado del PI de barras (solo la excursion lo deshabilita).

    Returns:
        The ModelInputs.
    """
    envelope = spec.envelope(position)
    if tiers.kind is ScenarioKind.CORE_TEMP_EXCURSION:
        return ModelInputs(
            load=load,
            reactivity_insertion=envelope * level.magnitude * PCM,
            rod_control_enabled=rods_enabled,
        )
    if tiers.kind is ScenarioKind.COOLANT_FLOW_REDUCTION:
        elapsed_s = max(position - spec.injected_step, 0.0) * dt_s
        tau_s = tiers.time_constant_minutes * 60.0
        drop = (1.0 - level.magnitude) * (1.0 - math.exp(-elapsed_s / tau_s))
        wobble = level.oscillation_amplitude * math.sin(
            2.0 * math.pi * elapsed_s / tiers.oscillation_period_s
        )
        return ModelInputs(load=load, pump_demand=1.0 + envelope * (wobble - drop))
    leak = envelope * leak_for_margin(model, load, level.magnitude) if envelope > 0.0 else 0.0
    return ModelInputs(load=load, leak_bar=leak)


def build_fault_inputs(
    model: ReactorModel,
    tiers: KindTiers,
    level: FaultLevel,
    spec: ProcessFaultSpec,
    loads: Sequence[float],
    dt_s: float,
) -> FaultInputs:
    """Return the model inputs of a process-fault run.

    La carga sigue el programa de la operacion normal (la planta sigue operando
    mientras falla); el fallo se superpone. Dentro de cada intervalo la carga de
    la rejilla fina es la interpolacion lineal de sus dos muestras, que es lo que
    el modelo integra en la rejilla gruesa.

    Args:
        model: Modelo termico.
        tiers: Niveles y cronologia del tipo de fallo.
        level: Nivel de magnitud.
        spec: Envolvente temporal en el run.
        loads: Carga en cada muestra (al menos 2).
        dt_s: Cadencia de muestreo [s].

    Returns:
        The FaultInputs.

    Raises:
        ValueError: If fewer than two loads are given or dt_s <= 0.
    """
    if len(loads) < 2:
        raise ValueError("a fault run needs at least 2 load samples.")
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    spec_level = tiers.level(level)
    excursion = tiers.kind is ScenarioKind.CORE_TEMP_EXCURSION
    n_steps = len(loads)

    def rods(step: int) -> bool:
        return not (excursion and spec.is_active(step))

    sample = tuple(
        _point(model, tiers, spec_level, spec, float(loads[k]), float(k), dt_s, rods(k))
        for k in range(n_steps)
    )
    substeps = tiers.substeps(dt_s)
    fine: tuple[ModelInputs, ...] | None = None
    if substeps > 1:
        points: list[ModelInputs] = []
        for k in range(n_steps - 1):
            for j in range(substeps):
                frac = j / substeps
                load = float(loads[k]) + (float(loads[k + 1]) - float(loads[k])) * frac
                points.append(
                    _point(model, tiers, spec_level, spec, load, k + frac, dt_s, rods(k))
                )
        points.append(sample[-1])
        fine = tuple(points)
    envelope = np.array([spec.envelope(float(k)) for k in range(n_steps)], dtype=np.float64)
    return FaultInputs(sample=sample, fine=fine, substeps=substeps, envelope=envelope)


def process_run_plan(
    params: SimulationParams,
    tiers: KindTiers,
    level: FaultLevel,
    *,
    run_index: int = 0,
    seed: int | None = None,
    injected_step: int | None = None,
    hold_steps: int | None = None,
    recovery_steps: int | None = None,
    persistent: bool = False,
) -> RunPlan:
    """Return the plan of one process-fault run.

    Misma semilla y mismo `run_index` que un run normal dan la MISMA traza de
    carga y el mismo flujo de ruido: la pareja normal/fallo es un contrafactico
    exacto hasta el instante de inyeccion.

    Args:
        params: Parametros de simulacion.
        tiers: Niveles y cronologia del tipo de fallo.
        level: Nivel de magnitud (queda en la envolvente del plan).
        run_index: Posicion del run (da instantes y semilla).
        seed: Semilla maestra; por defecto `params.openmc_seed`.
        injected_step: Primera muestra del fallo; por defecto el fin del
            calentamiento (`params.warmup_steps`).
        hold_steps: Pasos a magnitud plena; por defecto los de `fault_tiers`.
        recovery_steps: Pasos de recuperacion; por defecto los de `fault_tiers`.
        persistent: Si es True el fallo dura hasta el final del run, sin
            recuperacion (ignora `hold_steps` y `recovery_steps`).

    Returns:
        The RunPlan with its ProcessFaultSpec.

    Raises:
        ValueError: If run_index is negative or the window does not fit in the run.
    """
    if run_index < 0:
        raise ValueError(f"run_index must be >= 0, got {run_index}.")
    master = params.openmc_seed if seed is None else int(seed)
    child = np.random.SeedSequence(master).spawn(run_index + 1)[run_index]
    start = params.warmup_steps if injected_step is None else int(injected_step)
    hold: int | None
    recovery: int
    if persistent:
        hold, recovery = None, 0
    else:
        hold = tiers.hold_steps if hold_steps is None else int(hold_steps)
        recovery = tiers.recovery_steps if recovery_steps is None else int(recovery_steps)
    spec = ProcessFaultSpec(
        level=level,
        injected_step=start,
        onset_steps=tiers.onset_steps(params.dt_seconds),
        hold_steps=hold,
        recovery_steps=recovery,
    )
    return RunPlan(
        run_index=run_index,
        run_id=run_id_for(run_index),
        scenario=tiers.kind,
        n_steps=params.run_length_steps,
        start_time=params.start_time + timedelta(seconds=run_index * params.run_duration_s),
        seed_entropy=master,
        spawn_key=tuple(int(k) for k in child.spawn_key),
        process_fault=spec,
    )


# ----------------------------------------------------------------------
# Observabilidad por cadencia
# ----------------------------------------------------------------------


def min_observable_period_s(dt_s: float) -> float:
    """Return the shortest oscillation period a sampling cadence can represent.

    Nyquist: hacen falta al menos dos muestras por periodo.

    Args:
        dt_s: Cadencia de muestreo [s].

    Returns:
        2 * dt_s (360 s a 180 s).

    Raises:
        ValueError: If dt_s <= 0.
    """
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    return 2.0 * dt_s


def aliased_period_s(period_s: float, dt_s: float) -> float | None:
    """Return the apparent period of an oscillation seen through sampling.

    Args:
        period_s: Periodo verdadero [s].
        dt_s: Cadencia de muestreo [s].

    Returns:
        The apparent period [s] after folding the frequency into [0, 1/(2 dt)];
        None when the folded frequency is zero (a period that is a multiple of
        `dt_s` looks like a constant).

    Raises:
        ValueError: If period_s or dt_s is not strictly positive.
    """
    if not period_s > 0.0:
        raise ValueError(f"period_s must be > 0, got {period_s}.")
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    sampling = 1.0 / dt_s
    frequency = 1.0 / period_s
    folded = abs(frequency - math.floor(frequency / sampling + 0.5) * sampling)
    if folded < 1.0e-12:
        return None
    return 1.0 / folded


# ----------------------------------------------------------------------
# Respuesta medida de un nivel
# ----------------------------------------------------------------------


def response_from_run(sim: RunSimulation, model: ReactorModel, start_step: int) -> LevelResponse:
    """Summarise the true response of a run from a step on.

    Args:
        sim: Run integrado.
        model: Modelo que lo produjo (aporta T_out nominal y T_sat).
        start_step: Primera muestra considerada (el instante de inyeccion).

    Returns:
        The LevelResponse over samples `start_step` to the end.

    Raises:
        ValueError: If start_step is outside the run.
    """
    if not 0 <= start_step < sim.n_steps:
        raise ValueError(f"start_step {start_step} is outside the run of {sim.n_steps} steps.")
    observed = [
        model.observables(state, inputs)
        for state, inputs in zip(sim.states, sim.inputs, strict=True)
    ][start_step:]
    t_out = np.array([o["t_c"] for o in observed])
    margin = np.array([o["t_sat_c"] - o["t_c"] for o in observed])
    pressure = np.array([o["p_primary_bar"] for o in observed])
    power = np.array([o["power_fraction"] for o in observed])
    peak = float(t_out.max())
    return LevelResponse(
        t_out_peak_c=peak,
        t_out_rise_k=peak - model.nominal.t_out_c,
        t_fuel_peak_c=float(max(o["t_fuel"] for o in observed)),
        delta_t_peak_k=float(max(o["t_c"] - o["t_s"] for o in observed)),
        margin_min_k=float(margin.min()),
        flow_min_kg_s=float(min(o["flow_total_kg_s"] for o in observed)),
        n_peak=float(power.max()),
        n_min=float(power.min()),
        void_peak=float(max(o["void_fraction"] for o in observed)),
        p_min_bar=float(pressure.min()),
        p_max_bar=float(pressure.max()),
        max_state_dt_dt_c_s=float(
            max(d.max_temperature_rate_c_s for d in sim.diagnostics[start_step:])
        ),
    )


def measure_level_response(
    model: ReactorModel,
    catalog: SensorCatalog,
    tiers: KindTiers,
    level: FaultLevel,
    *,
    dt_s: float = 180.0,
    load: float = 1.0,
    settle_steps: int = 5,
    response_steps: int = 60,
) -> LevelResponse:
    """Measure the true response of the model to one level at a constant load.

    Es la medida con la que se fijan los niveles: carga constante (por defecto la
    nominal, la de mayor presion), el fallo persistente desde `settle_steps` y
    `response_steps` muestras de respuesta. Sin ruido.

    Args:
        model: Modelo termico.
        catalog: Catalogo de sensores (necesario para integrar el run).
        tiers: Niveles y cronologia del tipo de fallo.
        level: Nivel a medir.
        dt_s: Cadencia de muestreo [s].
        load: Carga constante de la medida.
        settle_steps: Muestras a carga constante antes de la inyeccion.
        response_steps: Muestras de respuesta tras la inyeccion.

    Returns:
        The LevelResponse.

    Raises:
        ValueError: If a step count is not positive.
    """
    if settle_steps < 1 or response_steps < 1:
        raise ValueError("settle_steps and response_steps must be >= 1.")
    n_steps = settle_steps + response_steps
    spec = ProcessFaultSpec(
        level=level,
        injected_step=settle_steps,
        onset_steps=tiers.onset_steps(dt_s),
        hold_steps=None,
        recovery_steps=0,
    )
    plan = RunPlan(
        run_index=0,
        run_id="sim_run_level",
        scenario=tiers.kind,
        n_steps=n_steps,
        start_time=_EPOCH,
        seed_entropy=0,
        spawn_key=(0,),
        process_fault=spec,
    )
    inputs = build_fault_inputs(model, tiers, level, spec, [load] * n_steps, dt_s)
    sim = simulate_run(
        model, catalog, plan, inputs.sample, dt_s, fine_inputs=inputs.fine, substeps=inputs.substeps
    )
    return response_from_run(sim, model, settle_steps)


def unmet_targets(response: LevelResponse, level: LevelSpec) -> list[str]:
    """Return the objective bands a measured response misses.

    Args:
        response: Respuesta medida.
        level: Nivel con sus bandas.

    Returns:
        One description per missed band; empty when every target is met.
    """
    missed: list[str] = []
    for target in level.targets:
        value = float(getattr(response, target.metric))
        if not target.contains(value):
            missed.append(f"{target.metric}={value:.4g} not in [{target.low}, {target.high}]")
    return missed


def is_process_scenario(kind: ScenarioKind) -> bool:
    """Return whether a scenario is one of the three process faults."""
    return kind.axis is FaultAxis.PROCESS
