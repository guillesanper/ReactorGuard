"""Constraints fisicos del PWR de referencia: reglas de TDD 5.4 y leyes de App. B.

Cada regla es una funcion pura y vectorizada sobre arrays; `evaluate_rules` las
aplica a un frame ancho de lecturas leyendo los roles de configs/sim_sensors.yaml
(no hay tags cableados aqui). La ecuacion de cada regla, sus unidades y sus
inconsistencias con el TDD estan en Docs/physics_constraints.md.

Convenciones que comparten todas las reglas:

    - Los operadores son ESTRICTOS: "igual al umbral" no es violacion.
    - Un valor NaN/None no se evalua: la regla devuelve False y la fila se cuenta
      como "no evaluada" en `RuleEvaluation.evaluable`.
    - La tabla de agua nunca extrapola: una fila fuera de su rejilla tampoco se
      evalua (en vez de abortar la evaluacion de todo un run).
    - Las reglas de relacion entre magnitudes (energia, orden, potencia/flujo,
      ebullicion, masa) trabajan sobre el valor del rol: la MEDIANA de sus canales
      validos (votacion de redundantes). Las de cota (rango de caudal, dT/dt) se
      evaluan canal a canal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from data.simulation.plant_params import PlantParams
from data.simulation.thermal_model import Diagnostics, StepResult
from data.simulation.water_properties import WaterTable

_Array = NDArray[np.float64]
_Mask = NDArray[np.bool_]

RULE_NAMES: tuple[str, ...] = (
    "energy_balance",
    "temperature_ordering",
    "max_dt_dt",
    "flow_range",
    "power_flux",
    "boiling",
    "mass_balance",
)
"""Reglas en el orden de las columnas de `evaluate_rules`.

Las seis primeras son las de TDD 5.4; `mass_balance` es la ley de masa de App. B.
"""

TDD_RULE_NAMES: tuple[str, ...] = RULE_NAMES[:6]
"""Las seis reglas de TDD 5.4."""

REQUIRED_ROLES: tuple[str, ...] = (
    "T_out",
    "T_in",
    "flow",
    "flow_hot_leg",
    "flow_cold_leg",
    "P",
    "flux",
    "P_th",
)
"""Roles de configs/sim_sensors.yaml que `evaluate_rules` necesita."""

_MW_PER_J_S = 1.0e-6
"""m [kg/s] x cp [J/(kg K)] x dT [K] x 1e-6 = potencia en MW."""

_PERCENT = 100.0


@dataclass(frozen=True)
class ConstraintLimits:
    """Umbrales y escalas de las constraints, derivados de PlantParams.

    Attributes:
        max_dt_c_per_s: Maxima derivada de temperatura admisible [C/s] (D9).
        energy_imbalance_fraction: Desbalance de energia que dispara la regla.
        pump_max_kg_s: Tope de caudal total de bomba [kg/s].
        thermal_power_mw: Potencia termica nominal P0 [MW].
        n_loops: Lazos del primario; cada canal de caudal mide total / n_loops.
        mass_balance_tolerance: |m_hl - m_cl| / m que dispara la regla de masa.
        quasi_steady_dt_avg_k_s: |dT_avg/dt| bajo el cual hay cuasi-estacionario [K/s].
        power_flux_k: Constante k de |P_th - k flux| (ambos en % del nominal).
        power_flux_threshold_percent: Umbral de |P_th - k flux| [puntos porcentuales].
        conservation_tolerance: Residuo relativo de conservacion admitido (D7).

    Raises:
        ValueError: If a value is not finite and strictly positive, or n_loops < 1.
    """

    max_dt_c_per_s: float
    energy_imbalance_fraction: float
    pump_max_kg_s: float
    thermal_power_mw: float
    n_loops: int
    mass_balance_tolerance: float
    quasi_steady_dt_avg_k_s: float
    power_flux_k: float
    power_flux_threshold_percent: float
    conservation_tolerance: float

    def __post_init__(self) -> None:
        """Validate that every limit is a finite, strictly positive number."""
        for name in (
            "max_dt_c_per_s",
            "energy_imbalance_fraction",
            "pump_max_kg_s",
            "thermal_power_mw",
            "mass_balance_tolerance",
            "quasi_steady_dt_avg_k_s",
            "power_flux_k",
            "power_flux_threshold_percent",
            "conservation_tolerance",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"ConstraintLimits.{name} must be finite and > 0, got {value}.")
        if self.n_loops < 1:
            raise ValueError(f"ConstraintLimits.n_loops must be >= 1, got {self.n_loops}.")

    @property
    def pump_max_per_loop_kg_s(self) -> float:
        """Return the pump ceiling of one flow channel [kg/s]."""
        return self.pump_max_kg_s / self.n_loops

    @classmethod
    def from_plant(cls, plant: PlantParams) -> ConstraintLimits:
        """Build the limits from a resolved PlantParams.

        Args:
            plant: Parametros de planta (su bloque `limits` aporta los umbrales).

        Returns:
            The ConstraintLimits.
        """
        limits = plant.limits
        return cls(
            max_dt_c_per_s=limits.max_dt_c_per_s,
            energy_imbalance_fraction=limits.energy_imbalance_fraction,
            pump_max_kg_s=plant.pump_max_kg_s,
            thermal_power_mw=plant.thermal_power_mw,
            n_loops=limits.n_loops,
            mass_balance_tolerance=limits.mass_balance_tolerance,
            quasi_steady_dt_avg_k_s=limits.quasi_steady_dt_avg_k_s,
            power_flux_k=limits.power_flux_k,
            power_flux_threshold_percent=limits.power_flux_threshold_percent,
            conservation_tolerance=limits.conservation_tolerance,
        )


# ----------------------------------------------------------------------
# Reglas puras sobre arrays
# ----------------------------------------------------------------------


def _as_array(values: ArrayLike) -> _Array:
    """Return values as float64 (None becomes NaN)."""
    return np.asarray(values, dtype=np.float64)


def in_table_domain(table: WaterTable, t_c: ArrayLike, p_bar: ArrayLike) -> _Mask:
    """Return where (T, P) lies on the water-table grid (NaN is outside).

    Args:
        table: Tabla de propiedades del agua.
        t_c: Temperatura [C].
        p_bar: Presion [bar].

    Returns:
        Boolean array, True where the table can be evaluated without extrapolating.
    """
    t, p = np.broadcast_arrays(_as_array(t_c), _as_array(p_bar))
    t_lo, t_hi = table.t_range_c
    p_lo, p_hi = table.p_range_bar
    mask: _Mask = (t >= t_lo) & (t <= t_hi) & (p >= p_lo) & (p <= p_hi)
    return mask


def removed_heat_mw(
    flow_total_kg_s: ArrayLike,
    t_out_c: ArrayLike,
    t_in_c: ArrayLike,
    p_bar: ArrayLike,
    table: WaterTable,
) -> _Array:
    """Return Q_removed = m cp(T_avg, P) (T_out - T_in) in MW.

    cp comes from the water table (not 4182 J/(kg K)): at the operating point it
    is ~5660 J/(kg K), so the constant of the TDD would understate Q_removed by
    ~26 %.

    Args:
        flow_total_kg_s: Caudal total del primario [kg/s].
        t_out_c: Temperatura de salida del nucleo [C].
        t_in_c: Temperatura de entrada [C].
        p_bar: Presion del primario [bar].
        table: Tabla de propiedades del agua.

    Returns:
        Heat removed [MW]; NaN where an input is NaN or (T_avg, P) is outside the
        table.
    """
    m, t_out, t_in, p = np.broadcast_arrays(
        _as_array(flow_total_kg_s), _as_array(t_out_c), _as_array(t_in_c), _as_array(p_bar)
    )
    t_avg = 0.5 * (t_out + t_in)
    ok = in_table_domain(table, t_avg, p)
    t_safe = np.where(ok, t_avg, table.t_range_c[0])
    p_safe = np.where(ok, p, table.p_range_bar[0])
    cp = np.asarray(table.cp(t_safe, p_safe), dtype=np.float64)
    heat: _Array = np.where(ok, m * cp * (t_out - t_in) * _MW_PER_J_S, np.nan)
    return heat


def energy_balance_residual(q_gen_mw: ArrayLike, q_removed_mw: ArrayLike) -> _Array:
    """Return |Q_gen - Q_removed| / Q_gen.

    Args:
        q_gen_mw: Potencia generada [MW].
        q_removed_mw: Potencia retirada por el refrigerante [MW].

    Returns:
        The relative imbalance; NaN where Q_gen <= 0 or an input is NaN.
    """
    q_gen, q_removed = np.broadcast_arrays(_as_array(q_gen_mw), _as_array(q_removed_mw))
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.abs(q_gen - q_removed) / q_gen
    result: _Array = np.where(q_gen > 0.0, residual, np.nan)
    return result


def energy_balance_violated(
    q_gen_mw: ArrayLike, q_removed_mw: ArrayLike, limit: float
) -> _Mask:
    """Return where |Q_gen - Q_removed| / Q_gen > limit (TDD 5.4, rule 1).

    Args:
        q_gen_mw: Potencia generada [MW].
        q_removed_mw: Potencia retirada por el refrigerante [MW].
        limit: Desbalance admitido (fraccion).

    Returns:
        Boolean array; False where the residual is not defined.
    """
    mask: _Mask = energy_balance_residual(q_gen_mw, q_removed_mw) > limit
    return mask


def temperature_ordering_violated(t_out_c: ArrayLike, t_in_c: ArrayLike) -> _Mask:
    """Return where T_out < T_in (TDD 5.4, rule 2).

    Args:
        t_out_c: Temperatura de salida del nucleo [C].
        t_in_c: Temperatura de entrada [C].

    Returns:
        Boolean array; False where an input is NaN or the two are equal.
    """
    mask: _Mask = _as_array(t_out_c) < _as_array(t_in_c)
    return mask


def finite_difference_rate(values: ArrayLike, dt_s: ArrayLike) -> _Array:
    """Return the backward-difference rate of change along axis 0.

    Args:
        values: Serie (n,) o canales (n, k).
        dt_s: Tiempo transcurrido desde la fila anterior [s], forma (n,); el
            primer elemento no se usa.

    Returns:
        Rate [unit/s] with the shape of values; NaN on the first row and where
        the elapsed time is not strictly positive.
    """
    v = _as_array(values)
    dt = _as_array(dt_s)
    rate = np.full(v.shape, np.nan, dtype=np.float64)
    if v.shape[0] < 2:
        return rate
    step = dt[1:].reshape((-1,) + (1,) * (v.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.diff(v, axis=0) / step
    rate[1:] = np.where(step > 0.0, delta, np.nan)
    return rate


def max_dt_dt_violated(rate_c_per_s: ArrayLike, limit_c_per_s: float) -> _Mask:
    """Return where |dT/dt| > limit (TDD 5.4, rule 3; magnitude, not sign).

    Args:
        rate_c_per_s: Derivada de temperatura [C/s].
        limit_c_per_s: Maxima derivada admisible [C/s].

    Returns:
        Boolean array; False where the rate is NaN.
    """
    mask: _Mask = np.abs(_as_array(rate_c_per_s)) > limit_c_per_s
    return mask


def flow_range_violated(flow_kg_s: ArrayLike, pump_max_kg_s: float) -> _Mask:
    """Return where flow < 0 or flow > pump_max (TDD 5.4, rule 4).

    Args:
        flow_kg_s: Caudal de un canal [kg/s].
        pump_max_kg_s: Tope de caudal del mismo canal [kg/s].

    Returns:
        Boolean array; False where the flow is NaN. A flow of exactly 0 is NOT a
        violation of this rule (strict operator).
    """
    flow = _as_array(flow_kg_s)
    mask: _Mask = (flow < 0.0) | (flow > pump_max_kg_s)
    return mask


def power_flux_violated(
    p_th_percent: ArrayLike, flux_percent: ArrayLike, k: float, threshold_percent: float
) -> _Mask:
    """Return where |P_th - k flux| > threshold (TDD 5.4, rule 5).

    Args:
        p_th_percent: Potencia termica [% del nominal].
        flux_percent: Flujo neutronico [% del nominal].
        k: Constante de proporcionalidad.
        threshold_percent: Umbral [puntos porcentuales].

    Returns:
        Boolean array; False where an input is NaN.
    """
    gap = np.abs(_as_array(p_th_percent) - k * _as_array(flux_percent))
    mask: _Mask = gap > threshold_percent
    return mask


def boiling_violated(t_out_c: ArrayLike, p_bar: ArrayLike, table: WaterTable) -> _Mask:
    """Return where T_out > T_sat(P), i.e. P < P_sat(T_out) (TDD 5.4, rule 6).

    Args:
        t_out_c: Temperatura de salida del nucleo [C].
        p_bar: Presion del primario [bar].
        table: Tabla de propiedades del agua (aporta T_sat(P)).

    Returns:
        Boolean array; False where an input is NaN or P is outside the table.
        T_out == T_sat is NOT a violation (strict operator, as in TDD 5.4).
    """
    t_out, p = np.broadcast_arrays(_as_array(t_out_c), _as_array(p_bar))
    p_lo, p_hi = table.p_range_bar
    ok = (p >= p_lo) & (p <= p_hi)
    t_sat = np.asarray(table.t_sat_c(np.where(ok, p, p_lo)), dtype=np.float64)
    mask: _Mask = ok & (t_out > t_sat)
    return mask


def mass_balance_residual(flow_hot_kg_s: ArrayLike, flow_cold_kg_s: ArrayLike) -> _Array:
    """Return |m_hl - m_cl| / m with m the mean of both legs.

    Args:
        flow_hot_kg_s: Caudal de la pierna caliente [kg/s].
        flow_cold_kg_s: Caudal de la pierna fria [kg/s].

    Returns:
        The relative mismatch; NaN where m <= 0 or an input is NaN.
    """
    hot, cold = np.broadcast_arrays(_as_array(flow_hot_kg_s), _as_array(flow_cold_kg_s))
    mean = 0.5 * (hot + cold)
    with np.errstate(divide="ignore", invalid="ignore"):
        residual = np.abs(hot - cold) / mean
    result: _Array = np.where(mean > 0.0, residual, np.nan)
    return result


def quasi_steady_mask(dt_avg_dt_k_s: ArrayLike, threshold_k_s: float) -> _Mask:
    """Return where |dT_avg/dt| < threshold (the plant is quasi-steady).

    Args:
        dt_avg_dt_k_s: Derivada de la temperatura media [K/s].
        threshold_k_s: Umbral de cuasi-estacionario [K/s].

    Returns:
        Boolean array; False where the rate is NaN.
    """
    mask: _Mask = np.abs(_as_array(dt_avg_dt_k_s)) < threshold_k_s
    return mask


def mass_balance_violated(
    flow_hot_kg_s: ArrayLike,
    flow_cold_kg_s: ArrayLike,
    quasi_steady: ArrayLike,
    tolerance: float,
) -> _Mask:
    """Return where |m_hl - m_cl| / m > tolerance in quasi-steady state (App. B).

    Args:
        flow_hot_kg_s: Caudal de la pierna caliente [kg/s].
        flow_cold_kg_s: Caudal de la pierna fria [kg/s].
        quasi_steady: Mascara de cuasi-estacionario (ver `quasi_steady_mask`).
        tolerance: Desajuste relativo admitido.

    Returns:
        Boolean array; False in transients and where the residual is undefined.
    """
    steady = np.asarray(quasi_steady, dtype=np.bool_)
    mask: _Mask = steady & (mass_balance_residual(flow_hot_kg_s, flow_cold_kg_s) > tolerance)
    return mask


# ----------------------------------------------------------------------
# Observabilidad por cadencia (C5)
# ----------------------------------------------------------------------


def min_observable_rate(dt_s: float, max_dt_c_per_s: float = 10.0) -> float:
    """Return the smallest |dT| between two samples at which the dT/dt rule fires.

    Con muestras separadas `dt_s`, la regla `|dT/dt| > max_dt_c_per_s` solo puede
    saltar si |T[k] - T[k-1]| > max_dt_c_per_s * dt_s: a 180 s y 10 C/s son 1800 K,
    mas que todo el rango del agua liquida (C5 del plan). El nombre es el del
    plan; el valor devuelto es un salto en K por muestra, no una tasa.

    Args:
        dt_s: Cadencia de muestreo [s].
        max_dt_c_per_s: Limite de la regla [C/s].

    Returns:
        The minimum observable step [K per sample].

    Raises:
        ValueError: If dt_s or max_dt_c_per_s is not strictly positive.
    """
    if not dt_s > 0.0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}.")
    if not max_dt_c_per_s > 0.0:
        raise ValueError(f"max_dt_c_per_s must be > 0, got {max_dt_c_per_s}.")
    return max_dt_c_per_s * dt_s


def max_observable_cadence_s(temperature_span_k: float, max_dt_c_per_s: float = 10.0) -> float:
    """Return the longest cadence at which the dT/dt rule can fire at all.

    Args:
        temperature_span_k: Salto maximo posible entre dos muestras [K] (p. ej.
            la anchura del rango de temperatura de la tabla de agua).
        max_dt_c_per_s: Limite de la regla [C/s].

    Returns:
        The cadence [s] above which min_observable_rate exceeds the span.

    Raises:
        ValueError: If an argument is not strictly positive.
    """
    if not temperature_span_k > 0.0:
        raise ValueError(f"temperature_span_k must be > 0, got {temperature_span_k}.")
    if not max_dt_c_per_s > 0.0:
        raise ValueError(f"max_dt_c_per_s must be > 0, got {max_dt_c_per_s}.")
    return temperature_span_k / max_dt_c_per_s


# ----------------------------------------------------------------------
# Evaluacion sobre un frame ancho
# ----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RuleEvaluation:
    """Resultado de `evaluate_rules_detailed`.

    Attributes:
        violations: Frame bool, una columna por regla de RULE_NAMES.
        evaluable: Frame bool de la misma forma; False donde la regla no pudo
            evaluarse (NaN, fuera de la tabla, denominador no positivo o, en la
            regla de masa, transitorio).
    """

    violations: pd.DataFrame
    evaluable: pd.DataFrame


@dataclass(frozen=True)
class RuleStats:
    """Conteos de una regla en un escenario.

    Attributes:
        n_rows: Filas del escenario.
        n_evaluable: Filas en que la regla pudo evaluarse.
        n_violations: Filas en que se violo.
    """

    n_rows: int
    n_evaluable: int
    n_violations: int

    @property
    def n_not_evaluated(self) -> int:
        """Return the rows the rule could not evaluate."""
        return self.n_rows - self.n_evaluable

    @property
    def rate(self) -> float:
        """Return n_violations / n_evaluable (NaN when nothing was evaluable)."""
        if self.n_evaluable == 0:
            return math.nan
        return self.n_violations / self.n_evaluable


@dataclass(frozen=True)
class RuleReport:
    """Tasa de violacion por regla y escenario.

    Attributes:
        scenarios: Escenario -> regla -> RuleStats.
    """

    scenarios: Mapping[str, Mapping[str, RuleStats]]

    def stats(self, scenario: str, rule: str) -> RuleStats:
        """Return the counts of one rule in one scenario.

        Args:
            scenario: Nombre del escenario.
            rule: Nombre de la regla (RULE_NAMES).

        Returns:
            The RuleStats.

        Raises:
            KeyError: If the scenario or the rule is unknown.
        """
        if scenario not in self.scenarios:
            raise KeyError(f"rule report: unknown scenario '{scenario}'.")
        rules = self.scenarios[scenario]
        if rule not in rules:
            raise KeyError(f"rule report: unknown rule '{rule}'.")
        return rules[rule]

    def rate(self, scenario: str, rule: str) -> float:
        """Return the violation rate of one rule in one scenario (NaN if none evaluable)."""
        return self.stats(scenario, rule).rate

    def to_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return a JSON-serialisable view (a NaN rate becomes None)."""
        out: dict[str, dict[str, dict[str, Any]]] = {}
        for scenario in sorted(self.scenarios):
            out[scenario] = {}
            for rule, stats in self.scenarios[scenario].items():
                rate = stats.rate
                out[scenario][rule] = {
                    "n_rows": stats.n_rows,
                    "n_evaluable": stats.n_evaluable,
                    "n_not_evaluated": stats.n_not_evaluated,
                    "n_violations": stats.n_violations,
                    "rate": None if math.isnan(rate) else rate,
                }
        return out


def build_rule_report(evaluations: Mapping[str, RuleEvaluation]) -> RuleReport:
    """Aggregate per-scenario evaluations into a RuleReport.

    Args:
        evaluations: Escenario -> resultado de `evaluate_rules_detailed`.

    Returns:
        The RuleReport.
    """
    scenarios: dict[str, dict[str, RuleStats]] = {}
    for name, evaluation in evaluations.items():
        scenarios[name] = {
            rule: RuleStats(
                n_rows=len(evaluation.violations),
                n_evaluable=int(evaluation.evaluable[rule].sum()),
                n_violations=int(evaluation.violations[rule].sum()),
            )
            for rule in RULE_NAMES
        }
    return RuleReport(scenarios=scenarios)


def _role_channels(
    frame: pd.DataFrame, catalog_roles: Mapping[str, Sequence[str]], role: str
) -> pd.DataFrame:
    """Return the numeric columns of a role (non-numeric cells become NaN).

    Raises:
        KeyError: If the role is missing, empty or names tags absent from frame.
    """
    if role not in catalog_roles or len(catalog_roles[role]) == 0:
        raise KeyError(f"physics constraints: role '{role}' is missing or empty.")
    tags = list(catalog_roles[role])
    missing = [tag for tag in tags if tag not in frame.columns]
    if missing:
        raise KeyError(f"physics constraints: role '{role}' needs columns {missing}.")
    return frame[tags].apply(pd.to_numeric, errors="coerce").astype(np.float64)


def _role_value(channels: pd.DataFrame) -> _Array:
    """Return the median of the valid channels per row (NaN if none is valid)."""
    return np.asarray(channels.median(axis=1, skipna=True), dtype=np.float64)


def _any_per_row(mask: _Mask) -> _Mask:
    """Reduce a (n, k) mask to (n,) with any()."""
    reduced: _Mask = np.any(mask, axis=1) if mask.ndim == 2 else mask
    return reduced


def evaluate_rules_detailed(
    wide_frame: pd.DataFrame,
    catalog_roles: Mapping[str, Sequence[str]],
    table: WaterTable,
    limits: ConstraintLimits,
) -> RuleEvaluation:
    """Apply the seven rules to a wide frame of readings.

    Args:
        wide_frame: Una fila por instante, una columna por tag, indice
            DatetimeIndex creciente (un unico run o escenario).
        catalog_roles: Rol -> tags (SensorCatalog.roles). Debe contener los de
            REQUIRED_ROLES.
        table: Tabla de propiedades del agua.
        limits: Umbrales y escalas.

    Returns:
        A RuleEvaluation with one bool column per rule in RULE_NAMES.

    Raises:
        TypeError: If the frame index is not a DatetimeIndex.
        KeyError: If a required role is missing or names a column absent from
            the frame.
    """
    if not isinstance(wide_frame.index, pd.DatetimeIndex):
        raise TypeError("physics constraints: the wide frame needs a DatetimeIndex.")
    channels = {role: _role_channels(wide_frame, catalog_roles, role) for role in REQUIRED_ROLES}
    value = {role: _role_value(frame) for role, frame in channels.items()}
    elapsed_s = np.asarray(
        wide_frame.index.to_series().diff().dt.total_seconds(), dtype=np.float64
    )

    t_out, t_in, pressure = value["T_out"], value["T_in"], value["P"]
    flux, p_th = value["flux"], value["P_th"]

    # Energia: Q_gen desde el flujo neutronico (potencia de fision), Q_removed
    # desde el refrigerante con cp de tabla.
    q_gen = limits.thermal_power_mw * flux / _PERCENT
    q_removed = removed_heat_mw(limits.n_loops * value["flow"], t_out, t_in, pressure, table)
    residual = energy_balance_residual(q_gen, q_removed)

    # Cotas por canal: dT/dt sobre las temperaturas medidas y rango de caudal.
    temperatures = pd.concat([channels["T_out"], channels["T_in"]], axis=1).to_numpy()
    rates = finite_difference_rate(temperatures, elapsed_s)
    flows = channels["flow"].to_numpy()

    # Masa: solo en cuasi-estacionario, con dT_avg/dt de las mediciones.
    rate_avg = finite_difference_rate(0.5 * (t_out + t_in), elapsed_s)
    steady = quasi_steady_mask(rate_avg, limits.quasi_steady_dt_avg_k_s)
    mass_residual = mass_balance_residual(value["flow_hot_leg"], value["flow_cold_leg"])

    violations = {
        "energy_balance": residual > limits.energy_imbalance_fraction,
        "temperature_ordering": temperature_ordering_violated(t_out, t_in),
        "max_dt_dt": _any_per_row(max_dt_dt_violated(rates, limits.max_dt_c_per_s)),
        "flow_range": _any_per_row(flow_range_violated(flows, limits.pump_max_per_loop_kg_s)),
        "power_flux": power_flux_violated(
            p_th, flux, limits.power_flux_k, limits.power_flux_threshold_percent
        ),
        "boiling": boiling_violated(t_out, pressure, table),
        "mass_balance": mass_balance_violated(
            value["flow_hot_leg"],
            value["flow_cold_leg"],
            steady,
            limits.mass_balance_tolerance,
        ),
    }
    p_lo, p_hi = table.p_range_bar
    evaluable = {
        "energy_balance": np.isfinite(residual),
        "temperature_ordering": np.isfinite(t_out) & np.isfinite(t_in),
        "max_dt_dt": _any_per_row(np.isfinite(rates)),
        "flow_range": _any_per_row(np.isfinite(flows)),
        "power_flux": np.isfinite(p_th) & np.isfinite(flux),
        "boiling": np.isfinite(t_out) & (pressure >= p_lo) & (pressure <= p_hi),
        "mass_balance": np.isfinite(mass_residual) & steady,
    }
    index = wide_frame.index
    return RuleEvaluation(
        violations=pd.DataFrame(violations, index=index, columns=list(RULE_NAMES)).astype(bool),
        evaluable=pd.DataFrame(evaluable, index=index, columns=list(RULE_NAMES)).astype(bool),
    )


def evaluate_rules(
    wide_frame: pd.DataFrame,
    catalog_roles: Mapping[str, Sequence[str]],
    table: WaterTable,
    limits: ConstraintLimits,
) -> pd.DataFrame:
    """Return the violation frame of the seven rules (see evaluate_rules_detailed).

    Args:
        wide_frame: Una fila por instante, una columna por tag, DatetimeIndex.
        catalog_roles: Rol -> tags (SensorCatalog.roles).
        table: Tabla de propiedades del agua.
        limits: Umbrales y escalas.

    Returns:
        Bool frame (same index as wide_frame), one column per rule in RULE_NAMES;
        True means violated, a row that could not be evaluated is False.

    Raises:
        TypeError: If the frame index is not a DatetimeIndex.
        KeyError: If a required role is missing or names a column absent from
            the frame.
    """
    return evaluate_rules_detailed(wide_frame, catalog_roles, table, limits).violations


# ----------------------------------------------------------------------
# Conservacion sobre el estado completo
# ----------------------------------------------------------------------


def conservation_residuals(
    diagnostics: Sequence[Diagnostics], thermal_power_mw: float
) -> _Array:
    """Return |dU/dt - (Q_gen - Q_sec)| / P0 for each instantaneous diagnostic.

    Es cero por construccion (identidad algebraica de las tres ecuaciones de
    nodo): detecta una regresion del modelo, no un error de integracion. El cierre
    significativo es `StepResult.energy_residual_rel` sobre cada intervalo.

    Args:
        diagnostics: Diagnosticos de ReactorModel.diagnostics.
        thermal_power_mw: Potencia termica nominal P0 [MW].

    Returns:
        Relative residuals, shape (len(diagnostics),).

    Raises:
        ValueError: If thermal_power_mw is not strictly positive.
    """
    if not thermal_power_mw > 0.0:
        raise ValueError(f"thermal_power_mw must be > 0, got {thermal_power_mw}.")
    residuals = [abs(d.conservation_residual_mw) / thermal_power_mw for d in diagnostics]
    return np.asarray(residuals, dtype=np.float64)


@dataclass(frozen=True)
class ConservationReport:
    """Cierre de conservacion de una traza del modelo (D7).

    Attributes:
        n_instants: Diagnosticos instantaneos evaluados.
        n_intervals: Intervalos de integracion evaluados.
        max_instant_residual: max |dU/dt - (Q_gen - Q_sec)| / P0.
        max_interval_residual: max de StepResult.energy_residual_rel.
        tolerance: Residuo admitido (ConstraintLimits.conservation_tolerance).
        n_failures: Instantes e intervalos con residuo > tolerance.
        max_primary_imbalance: max de |Q_gen - Q_core| / Q_gen del estado verdadero.
        n_primary_imbalance_violations: Instantes en que la regla 1 de TDD 5.4 se
            viola sobre el estado verdadero (un transitorio rapido puede hacerlo
            sin violar la conservacion: es la matriz de violaciones esperadas).
    """

    n_instants: int
    n_intervals: int
    max_instant_residual: float
    max_interval_residual: float
    tolerance: float
    n_failures: int
    max_primary_imbalance: float
    n_primary_imbalance_violations: int

    @property
    def compliant(self) -> bool:
        """Return True when no instant or interval exceeds the tolerance."""
        return self.n_failures == 0


def conservation_report(
    diagnostics: Sequence[Diagnostics],
    steps: Sequence[StepResult],
    limits: ConstraintLimits,
) -> ConservationReport:
    """Summarise the conservation of a model trace.

    Args:
        diagnostics: Diagnosticos instantaneos de la traza.
        steps: Resultados de integracion de la traza.
        limits: Umbrales (tolerancia y limite de la regla de energia).

    Returns:
        The ConservationReport.

    Raises:
        ValueError: If both diagnostics and steps are empty.
    """
    if len(diagnostics) == 0 and len(steps) == 0:
        raise ValueError("conservation report: the trace has no diagnostics and no steps.")
    instant = conservation_residuals(diagnostics, limits.thermal_power_mw)
    interval = np.asarray([s.energy_residual_rel for s in steps], dtype=np.float64)
    imbalance = np.asarray([d.energy_imbalance_fraction for d in diagnostics], dtype=np.float64)
    tol = limits.conservation_tolerance
    return ConservationReport(
        n_instants=int(instant.size),
        n_intervals=int(interval.size),
        max_instant_residual=float(instant.max()) if instant.size else 0.0,
        max_interval_residual=float(interval.max()) if interval.size else 0.0,
        tolerance=tol,
        n_failures=int((instant > tol).sum() + (interval > tol).sum()),
        max_primary_imbalance=float(imbalance.max()) if imbalance.size else 0.0,
        n_primary_imbalance_violations=int((imbalance > limits.energy_imbalance_fraction).sum()),
    )
