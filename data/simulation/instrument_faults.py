"""Fallos de instrumento (F3-M5): deriva, bloqueo, pico, sesgo e intermitente.

Un fallo de instrumento NO toca la planta: la planta sigue siendo la normal y el
estado verdadero (`true_value`) se conserva. Lo que cambia es la lectura, y por
eso se aplica DESPUES del ruido, sobre la serie medida de un sensor (D3: el eje de
proceso y el de instrumento son ortogonales). Es la contraparte, sobre el modelo
termico, de `data/validation/fault_injector.py`, cuya semantica de manipulacion
(deriva en anchos de sobre por hora, bloqueo, pico saturado por el ADC, desvio
persistente) se reutiliza sin importarlo (D5).

    drift         rampa lineal en anchos de sobre de alarma por hora. El nivel alto
                  cae en la banda 13-17 anchos/hora que el detector de deriva de
                  Fase 2 puede ver a 180 s; el bajo es la deriva real del TDD
                  (~0,1 C/dia), invisible, y se etiqueta igual: la verdad es la
                  verdad aunque ningun detector la vea.
    stuck         congela el ultimo valor valido anterior a la ventana.
    noise_spike   una muestra desplazada +-k sigma_normal del sensor. El valor NO se
                  recorta: lo que satura es `raw_counts`, que clampa el ADC.
    bias          desvio persistente desde un instante del run hasta el final, en
                  fracciones del ancho de sobre (niveles dentro y fuera del sobre).
    intermittent  cadena de Markov de dos estados que alterna lectura valida e
                  invalida (carril `span.min` o `span.max`). `quality` sigue siendo
                  `good`: la etiqueta cubre toda la ventana y `corrupted` marca solo
                  las muestras realmente invalidas.

Las funciones `apply_*` son puras: reciben la serie, el span y el episodio y
devuelven una serie nueva. Toda aleatoriedad de planificacion (sensor, instante,
sentido, magnitud) esta ya decidida en el `InstrumentEpisode`; solo el
intermitente consume el generador de fallos del run al aplicarse.

`InstrumentFaultKind` e `InstrumentEpisode` viven en `scenario.py` (el `RunPlan`
los necesita y este modulo importa `scenario`); aqui se reexportan.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from data.schemas.sensor_spans import SensorSpan
from data.simulation.scenario import (
    INSTRUMENT_KINDS,
    FaultLevel,
    InstrumentEpisode,
    InstrumentFaultKind,
    RunPlan,
)
from data.simulation.sensor_catalog import SensorCatalog

_LOG = logging.getLogger(__name__)

__all__ = [
    "INSTRUMENT_KINDS",
    "EpisodeEffect",
    "InstrumentEffect",
    "InstrumentEpisode",
    "InstrumentFaultKind",
    "InstrumentFaultTiers",
    "InstrumentKindTiers",
    "InstrumentLevel",
    "apply_bias",
    "apply_drift",
    "apply_episode",
    "apply_instrument_faults",
    "apply_intermittent",
    "apply_noise_spike",
    "apply_stuck",
    "build_episode",
    "envelope_width",
    "parse_instrument_faults",
]

_LEVEL_ORDER: tuple[FaultLevel, ...] = (FaultLevel.LOW, FaultLevel.MEDIUM, FaultLevel.HIGH)

_NAME = "plant_parameters.yaml: 'instrument_faults'"

STUCK_DETECTOR_WINDOW = 8
"""Ventana del detector de bloqueo de Fase 2: un bloqueo debe durar mas."""

_MAGNITUDE_KEYS: dict[InstrumentFaultKind, str] = {
    InstrumentFaultKind.DRIFT: "rate_widths_per_hour",
    InstrumentFaultKind.NOISE_SPIKE: "k_sigma",
    InstrumentFaultKind.BIAS: "offset_widths",
}
"""Clave del rango de magnitud de cada tipo que la tiene."""

_LENGTH_KINDS = (
    InstrumentFaultKind.DRIFT,
    InstrumentFaultKind.STUCK,
    InstrumentFaultKind.INTERMITTENT,
)
"""Tipos cuyo nivel fija la longitud de la ventana (el sesgo dura hasta el final)."""


# ----------------------------------------------------------------------
# Niveles: configuracion tipada
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class InstrumentLevel:
    """Un nivel de magnitud de un fallo de instrumento.

    Attributes:
        magnitude_low: Extremo inferior del rango de magnitud (0 si el tipo no la
            tiene).
        magnitude_high: Extremo superior del rango.
        length_steps: Muestras de la ventana (0 si el tipo no la fija: el pico dura
            una y el sesgo hasta el final del run).
        p_enter: Probabilidad por muestra de pasar de valida a invalida.
        p_exit: Probabilidad por muestra de volver de invalida a valida.
    """

    magnitude_low: float
    magnitude_high: float
    length_steps: int
    p_enter: float = 0.0
    p_exit: float = 0.0

    @property
    def magnitude_mid(self) -> float:
        """Return the middle of the magnitude range."""
        return 0.5 * (self.magnitude_low + self.magnitude_high)

    @property
    def invalid_fraction(self) -> float:
        """Return the stationary share of invalid samples of the two-state chain."""
        total = self.p_enter + self.p_exit
        return 0.0 if total == 0.0 else self.p_enter / total


@dataclass(frozen=True)
class InstrumentKindTiers:
    """Niveles y parametros de un tipo de fallo de instrumento.

    Attributes:
        kind: Tipo.
        levels: Nivel -> especificacion; siempre `low`, `medium` y `high`.
        onset_fraction: (minimo, maximo) de la fraccion del run en que empieza un
            sesgo; (0, 0) en los demas tipos.
    """

    kind: InstrumentFaultKind
    levels: Mapping[FaultLevel, InstrumentLevel]
    onset_fraction: tuple[float, float] = (0.0, 0.0)

    def level(self, level: FaultLevel) -> InstrumentLevel:
        """Return the specification of one level."""
        return self.levels[level]

    def window_steps(self, level: FaultLevel, start_step: int, n_steps: int) -> int:
        """Return the labelled window of an episode of this kind and level.

        Args:
            level: Nivel del episodio.
            start_step: Primera muestra del episodio.
            n_steps: Muestras del run.

        Returns:
            1 for a spike, `n_steps - start_step` for a bias (persistent) and the
            configured length otherwise. NO se recorta: quien coloca el episodio
            comprueba que cabe en el run.
        """
        if self.kind is InstrumentFaultKind.NOISE_SPIKE:
            return 1
        if self.kind is InstrumentFaultKind.BIAS:
            return n_steps - start_step
        return self.levels[level].length_steps

    def mean_window_steps(self, n_steps: int, start_step: int = 0) -> float:
        """Return the mean window over the three levels (levels rotate evenly)."""
        return sum(self.window_steps(lv, start_step, n_steps) for lv in _LEVEL_ORDER) / len(
            _LEVEL_ORDER
        )


@dataclass(frozen=True)
class InstrumentFaultTiers:
    """`instrument_faults:` resuelto: los cinco tipos de instrumento.

    Attributes:
        kinds: Tipo -> niveles y parametros.
    """

    kinds: Mapping[InstrumentFaultKind, InstrumentKindTiers]

    def for_kind(self, kind: InstrumentFaultKind) -> InstrumentKindTiers:
        """Return the tiers of one instrument fault.

        Raises:
            KeyError: If the kind is not configured.
        """
        if kind not in self.kinds:
            raise KeyError(f"instrument_faults: '{kind.value}' is not configured.")
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


def _range(value: Any, label: str) -> tuple[float, float]:
    """Parse a scalar or a `[low, high]` pair into a positive range.

    Raises:
        TypeError: If value is a sequence of the wrong size.
        ValueError: If the range is inverted or not positive.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise TypeError(f"{label}: must be a number or a [low, high] pair.")
        low = _finite(value[0], f"{label}[0]")
        high = _finite(value[1], f"{label}[1]")
    else:
        low = high = _finite(value, label)
    if low <= 0.0 or high < low:
        raise ValueError(f"{label}: needs 0 < low <= high, got [{low}, {high}].")
    return low, high


def _probability(value: Any, label: str) -> float:
    """Return a probability in (0, 1].

    Raises:
        ValueError: If it is outside (0, 1].
    """
    number = _finite(value, label)
    if not 0.0 < number <= 1.0:
        raise ValueError(f"{label}: must be in (0, 1], got {number}.")
    return number


def _parse_level(kind: InstrumentFaultKind, raw: Any, label: str) -> InstrumentLevel:
    """Parse one level of one kind.

    Raises:
        KeyError: If a required key is missing.
        TypeError: If the block or a value has the wrong type.
        ValueError: If a value is out of range.
    """
    entry = _mapping(raw, label)
    low = high = 0.0
    if kind in _MAGNITUDE_KEYS:
        low, high = _range(_require(entry, _MAGNITUDE_KEYS[kind], label), f"{label}.magnitude")
    length = 0
    if kind in _LENGTH_KINDS:
        length = _whole(_require(entry, "length_steps", label), f"{label}.length_steps", 1)
        if kind is InstrumentFaultKind.STUCK and length <= STUCK_DETECTOR_WINDOW:
            raise ValueError(
                f"{label}.length_steps: {length} does not exceed the stuck detector window "
                f"of {STUCK_DETECTOR_WINDOW}."
            )
    p_enter = p_exit = 0.0
    if kind is InstrumentFaultKind.INTERMITTENT:
        p_enter = _probability(_require(entry, "p_enter", label), f"{label}.p_enter")
        p_exit = _probability(_require(entry, "p_exit", label), f"{label}.p_exit")
    return InstrumentLevel(
        magnitude_low=low,
        magnitude_high=high,
        length_steps=length,
        p_enter=p_enter,
        p_exit=p_exit,
    )


def _severity_key(kind: InstrumentFaultKind, level: InstrumentLevel) -> float:
    """Return the scalar that must grow from the low to the high level."""
    if kind is InstrumentFaultKind.STUCK:
        return float(level.length_steps)
    if kind is InstrumentFaultKind.INTERMITTENT:
        return level.invalid_fraction
    return level.magnitude_mid


def _parse_kind(kind: InstrumentFaultKind, raw: Any) -> InstrumentKindTiers:
    """Parse the block of one instrument fault.

    Raises:
        KeyError: If a required key or level is missing.
        TypeError: If a block has the wrong type.
        ValueError: If a value is out of range or the levels do not grow.
    """
    label = f"{_NAME}.{kind.value}"
    block = _mapping(raw, label)
    levels_raw = _mapping(_require(block, "levels", label), f"{label}.levels")
    levels = {
        level: _parse_level(
            kind,
            _require(levels_raw, level.value, f"{label}.levels"),
            f"{label}.levels.{level.value}",
        )
        for level in _LEVEL_ORDER
    }
    keys = [_severity_key(kind, levels[level]) for level in _LEVEL_ORDER]
    if not keys[0] < keys[1] < keys[2]:
        raise ValueError(f"{label}: low, medium and high must increase in severity, got {keys}.")
    onset = (0.0, 0.0)
    if kind is InstrumentFaultKind.BIAS:
        pair = _require(block, "onset_fraction", label)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise TypeError(f"{label}.onset_fraction: must be a [low, high] pair.")
        low = _finite(pair[0], f"{label}.onset_fraction[0]")
        high = _finite(pair[1], f"{label}.onset_fraction[1]")
        if not 0.0 < low <= high < 1.0:
            raise ValueError(f"{label}.onset_fraction: needs 0 < low <= high < 1, got {pair}.")
        onset = (low, high)
    return InstrumentKindTiers(kind=kind, levels=levels, onset_fraction=onset)


def parse_instrument_faults(raw: Mapping[str, Any]) -> InstrumentFaultTiers:
    """Resolve the `instrument_faults:` block of plant_parameters.yaml.

    Args:
        raw: `PlantParams.instrument_faults` (the parsed mapping).

    Returns:
        The typed InstrumentFaultTiers with the five instrument faults.

    Raises:
        KeyError: If a fault, a level or a required key is missing.
        TypeError: If a block has the wrong YAML type.
        ValueError: If a magnitude is out of range or the levels do not grow in
            severity.
    """
    kinds = {kind: _parse_kind(kind, _require(raw, kind.value, _NAME)) for kind in INSTRUMENT_KINDS}
    return InstrumentFaultTiers(kinds=kinds)


# ----------------------------------------------------------------------
# Manipulaciones puras
# ----------------------------------------------------------------------


def envelope_width(span: SensorSpan) -> float:
    """Return the width of the normal-operation envelope of a transmitter.

    Es la unidad en la que se expresan la deriva y el sesgo, igual que en el
    inyector de Fase 2 y en los umbrales del validador.

    Args:
        span: Span del sensor.

    Returns:
        `alarm_max - alarm_min`, always > 0.
    """
    return span.alarm_max - span.alarm_min


def _window(series: NDArray[np.float64], episode: InstrumentEpisode) -> slice:
    """Return the slice an episode covers, checked against the series.

    Raises:
        ValueError: If the episode leaves the series.
    """
    if episode.last_step >= series.shape[0]:
        raise ValueError(
            f"{episode.kind.value} episode on {episode.sensor_id} ends at step "
            f"{episode.last_step} but the series has {series.shape[0]} samples."
        )
    return slice(episode.start_step, episode.last_step + 1)


def apply_drift(
    series: NDArray[np.float64], span: SensorSpan, episode: InstrumentEpisode, dt_s: float
) -> NDArray[np.float64]:
    """Add a linear ramp of `magnitude` envelope widths per hour.

    La primera muestra de la ventana ya lleva un paso de rampa (`k + 1`), de modo
    que la lectura esta corrupta desde `injected_at`.

    Args:
        series: Serie medida del sensor (no se modifica).
        span: Span del sensor (da el ancho del sobre).
        episode: Episodio de deriva.
        dt_s: Cadencia de muestreo [s].

    Returns:
        A new series with the ramp added.

    Raises:
        ValueError: If dt_s <= 0 or the episode leaves the series.
    """
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    out = series.copy()
    window = _window(out, episode)
    per_step = episode.direction * episode.magnitude * envelope_width(span) * dt_s / 3600.0
    out[window] += per_step * np.arange(1, episode.length_steps + 1, dtype=np.float64)
    return out


def apply_stuck(series: NDArray[np.float64], episode: InstrumentEpisode) -> NDArray[np.float64]:
    """Freeze the sensor at its last valid value before the window.

    Args:
        series: Serie medida del sensor (no se modifica).
        episode: Episodio de bloqueo (`start_step >= 1`).

    Returns:
        A new series whose window repeats `series[start_step - 1]` exactly.

    Raises:
        ValueError: If the episode leaves the series.
    """
    out = series.copy()
    window = _window(out, episode)
    out[window] = series[episode.start_step - 1]
    return out


def apply_noise_spike(
    series: NDArray[np.float64], sigma: float, episode: InstrumentEpisode
) -> NDArray[np.float64]:
    """Offset one sample by +-k sigma of the sensor's normal noise.

    El valor NO se recorta al span: es `raw_counts` quien satura en el ADC
    (`SensorSpan.to_raw_counts`), igual que en `TEPFaultInjector`.

    Args:
        series: Serie medida del sensor (no se modifica).
        sigma: Sigma del ruido normal del sensor (> 0).
        episode: Episodio de pico (`magnitude` = k).

    Returns:
        A new series with the spike added at `start_step`.

    Raises:
        ValueError: If sigma <= 0 or the episode leaves the series.
    """
    if not sigma > 0.0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")
    out = series.copy()
    _window(out, episode)
    out[episode.start_step] += episode.direction * episode.magnitude * sigma
    return out


def apply_bias(
    series: NDArray[np.float64], span: SensorSpan, episode: InstrumentEpisode
) -> NDArray[np.float64]:
    """Add a persistent offset of `magnitude` envelope widths over the window.

    Args:
        series: Serie medida del sensor (no se modifica).
        span: Span del sensor (da el ancho del sobre).
        episode: Episodio de sesgo.

    Returns:
        A new series with the offset added.

    Raises:
        ValueError: If the episode leaves the series.
    """
    out = series.copy()
    window = _window(out, episode)
    out[window] += episode.direction * episode.magnitude * envelope_width(span)
    return out


def apply_intermittent(
    series: NDArray[np.float64],
    span: SensorSpan,
    episode: InstrumentEpisode,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Alternate valid and invalid readings with a two-state Markov chain.

    La ventana empieza en estado invalido (un fallo intermitente empieza con una
    perdida), de modo que `injected_at` coincide con la primera lectura corrupta.
    Una lectura invalida es un carril del ADC (`span.max` o `span.min`); la
    primera rafaga toma el carril de `direction` y las siguientes uno al azar.
    Se consumen siempre `2 * length_steps` aleatorios, vaya como vaya la cadena.

    Args:
        series: Serie medida del sensor (no se modifica).
        span: Span del sensor (da los carriles).
        episode: Episodio intermitente (`p_enter`, `p_exit`).
        rng: Generador del flujo de fallos del run.

    Returns:
        (new series, mask of the invalid samples over the whole series).

    Raises:
        ValueError: If the episode leaves the series.
    """
    out = series.copy()
    _window(out, episode)
    steps = episode.length_steps
    transitions = rng.random(steps)
    rail_draws = rng.random(steps)
    invalid = np.zeros(series.shape[0], dtype=np.bool_)
    rail = span.max if episode.direction > 0 else span.min
    state_invalid = True
    for k in range(steps):
        position = episode.start_step + k
        if k > 0:
            if state_invalid:
                state_invalid = bool(transitions[k] >= episode.p_exit)
            elif transitions[k] < episode.p_enter:
                state_invalid = True
                rail = span.max if rail_draws[k] < 0.5 else span.min
        if state_invalid:
            invalid[position] = True
            out[position] = rail
    return out, invalid


@dataclass(frozen=True)
class EpisodeEffect:
    """Efecto de un episodio sobre la serie de un sensor.

    Attributes:
        values: Serie manipulada (la serie completa).
        corrupted: Muestras cuya lectura difiere de la medida sin fallo.
    """

    values: NDArray[np.float64]
    corrupted: NDArray[np.bool_]


def apply_episode(
    series: NDArray[np.float64],
    episode: InstrumentEpisode,
    span: SensorSpan,
    sigma: float,
    dt_s: float,
    rng: np.random.Generator,
) -> EpisodeEffect:
    """Apply one episode to the measured series of its sensor.

    `corrupted` es objetivo: una muestra esta corrupta si su lectura difiere de la
    medida sin fallo (en el intermitente, si es un carril invalido).

    Args:
        series: Serie medida del sensor (no se modifica).
        episode: Episodio.
        span: Span del sensor.
        sigma: Sigma del ruido normal del sensor.
        dt_s: Cadencia de muestreo [s].
        rng: Generador del flujo de fallos del run.

    Returns:
        The EpisodeEffect.

    Raises:
        ValueError: If the episode leaves the series or a parameter is invalid.
    """
    kind = episode.kind
    if kind is InstrumentFaultKind.INTERMITTENT:
        values, corrupted = apply_intermittent(series, span, episode, rng)
        return EpisodeEffect(values=values, corrupted=corrupted)
    if kind is InstrumentFaultKind.DRIFT:
        values = apply_drift(series, span, episode, dt_s)
    elif kind is InstrumentFaultKind.STUCK:
        values = apply_stuck(series, episode)
    elif kind is InstrumentFaultKind.NOISE_SPIKE:
        values = apply_noise_spike(series, sigma, episode)
    else:
        values = apply_bias(series, span, episode)
    return EpisodeEffect(values=values, corrupted=values != series)


@dataclass(frozen=True)
class InstrumentEffect:
    """Efecto de los episodios de un run sobre la matriz medida.

    Attributes:
        values: Matriz manipulada, forma (n_steps, n_sensors).
        corrupted: Muestras cuya lectura difiere de la medida sin fallo.
    """

    values: NDArray[np.float64]
    corrupted: NDArray[np.bool_]


def apply_instrument_faults(
    measured: NDArray[np.float64],
    catalog: SensorCatalog,
    spans: Mapping[str, SensorSpan],
    plan: RunPlan,
    dt_s: float,
    rng: np.random.Generator,
) -> InstrumentEffect:
    """Apply every instrument episode of a run to its measured matrix.

    Los episodios se aplican en orden (paso inicial, sensor del catalogo): el
    orden fija el consumo del generador y por tanto la reproducibilidad.

    Args:
        measured: Matriz medida con ruido, forma (n_steps, n_sensors); no se modifica.
        catalog: Catalogo de sensores (orden de columnas y sigma normal).
        spans: Span calibrado por tag.
        plan: Plan del run (lleva los episodios).
        dt_s: Cadencia de muestreo [s].
        rng: Generador del flujo de fallos del run (`RunPlan.rngs()[2]`).

    Returns:
        The InstrumentEffect; a copy of `measured` and an all-False mask when the
        run carries no episodes.

    Raises:
        KeyError: If an episode names a sensor missing from the catalog or the spans.
        ValueError: If `measured` does not match the run or an episode leaves it.
    """
    if measured.shape != (plan.n_steps, len(catalog.sensors)):
        raise ValueError(
            f"{plan.run_id}: measured has shape {measured.shape}, expected "
            f"({plan.n_steps}, {len(catalog.sensors)})."
        )
    column = {tag: j for j, tag in enumerate(catalog.tags)}
    values = measured.copy()
    corrupted = np.zeros(measured.shape, dtype=np.bool_)
    ordered = sorted(
        plan.instrument_faults,
        key=lambda e: (e.start_step, column.get(e.sensor_id, len(column)), e.sensor_id),
    )
    for episode in ordered:
        if episode.sensor_id not in column:
            raise KeyError(f"{plan.run_id}: episode names unknown sensor '{episode.sensor_id}'.")
        if episode.sensor_id not in spans:
            raise KeyError(f"{plan.run_id}: no span for sensor '{episode.sensor_id}'.")
        j = column[episode.sensor_id]
        effect = apply_episode(
            values[:, j],
            episode,
            spans[episode.sensor_id],
            catalog.sensors[j].noise_sigma,
            dt_s,
            rng,
        )
        values[:, j] = effect.values
        corrupted[:, j] |= effect.corrupted
    return InstrumentEffect(values=values, corrupted=corrupted)


# ----------------------------------------------------------------------
# Construccion de episodios
# ----------------------------------------------------------------------


def build_episode(
    tiers: InstrumentKindTiers,
    level: FaultLevel,
    sensor_id: str,
    start_step: int,
    n_steps: int,
    rng: np.random.Generator,
) -> InstrumentEpisode:
    """Build the episode of one (kind, level) on a sensor from the tiers.

    Siempre consume dos aleatorios (sentido y magnitud), tenga o no el tipo
    magnitud: el consumo no depende del tipo ni del nivel, asi que cambiar un
    episodio no desplaza el flujo de los demas.

    Args:
        tiers: Niveles del tipo.
        level: Nivel del episodio.
        sensor_id: Sensor manipulado.
        start_step: Primera muestra del episodio.
        n_steps: Muestras del run (fija la ventana del sesgo).
        rng: Generador del planificador.

    Returns:
        The InstrumentEpisode.

    Raises:
        ValueError: If the episode does not fit in the run.
    """
    direction = 1 if rng.random() < 0.5 else -1
    draw = float(rng.random())
    spec = tiers.level(level)
    length = tiers.window_steps(level, start_step, n_steps)
    if length < 1 or start_step + length > n_steps:
        raise ValueError(
            f"{tiers.kind.value} episode at step {start_step} does not fit in {n_steps} steps."
        )
    magnitude = spec.magnitude_low + draw * (spec.magnitude_high - spec.magnitude_low)
    return InstrumentEpisode(
        sensor_id=sensor_id,
        kind=tiers.kind,
        level=level,
        start_step=start_step,
        length_steps=length,
        magnitude=magnitude,
        direction=direction,
        p_enter=spec.p_enter,
        p_exit=spec.p_exit,
    )
