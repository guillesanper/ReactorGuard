"""Modelo fisico del PWR de referencia: cinetica puntual + tres nodos termicos.

Estado (16 variables, ver StateIndex):

    n                  potencia neutronica normalizada (1 = nominal)
    c_1..c_6           precursores normalizados c_i = lambda_i C_i Lambda / beta_i
                       (equivalente a la forma clasica; en equilibrio c_i = n)
    T_f, T_c, T_s      combustible, nucleo (salida / pierna caliente) y GV
                       (retorno / pierna fria) [C]
    P                  presion del primario [bar]
    void               fraccion de vacio
    m                  caudal total del primario [kg/s]
    z, r               integral del error y reactividad de las barras (PI)
    q                  potencia calorimetrica filtrada (fraccion de P0)

Ecuaciones (t en s, P0 en MW, C en MJ/K, m cp en MW/K):

    dn/dt  = ((rho - beta) n + beta sum_i a_i c_i) / Lambda
    dc_i/dt = lambda_i (n - c_i)
    rho    = r + a_f (T_f - T_f0) + a_m (T_avg - T_avg0) + a_v void + rho_ins
    C_f dT_f/dt = P0 n - hA (T_f - T_avg)
    C_c dT_c/dt = hA (T_f - T_avg) - m cp (T_c - T_s)
    C_s dT_s/dt = m cp (T_c - T_s) - UA load (T_avg - T_sec(load))
    T_ref(load) = T_avg0 + s_ref (load - 1),  T_sec(load) = T_ref(load) - (T_avg0 - T_sec0)
    tau_p dP/dt = clip(P_nom + k_p (T_avg - T_avg0) - leak, P_lo, P_hi) - P
    tau_v dvoid/dt = clip((T_c - T_sat(P) + delta) / w, 0, v_max) - void
    tau_b dm/dt = m_nom demand - m
    dz/dt = e, tau_r dr/dt = k_p e + k_i z - r, con e = T_ref(load) - T_avg
    tau_q dq/dt = Q_sec / P0 - q

La suma de los tres nodos termicos es P0 n - Q_sec: la energia se conserva por
construccion porque el mismo cp aparece en los dos terminos de transporte.

La carga es, por construccion, la potencia de equilibrio: UA_eff = UA load con
el salto T_avg - T_sec constante da Q_sec = load P0 y n_eq = load.

Limitador de presion (F3-M4): el objetivo de presion se acota a [P_lo, P_hi] = el
rango de presiones de la tabla de agua (60-170 bar). El techo es fisico (las
valvulas de alivio del presurizador abren a ~170 bar); el suelo es un LIMITE DE
VALIDEZ del modelo de agua liquida, no fisica. No actua en operacion normal
(P <= ~157 bar): solo evita que un transitorio de vacio o de potencia, que ya
esta fuera de lo que la tabla representa, aborte el run sacando P de la rejilla
(la tabla no extrapola). Un estado con P en el limite se reconoce por
`pressure_limits_bar`.

Propiedades congeladas por intervalo: cp y T_sat se evaluan UNA vez al inicio
de cada paso (operador de splitting documentado). `freeze_properties=False`
las evalua en cada llamada a la derivada y existe para MEDIR ese error.

El equilibrio se DERIVA, no se teclea: dados T_in0, caudal y P0 se itera
T_out0 = T_in0 + P0 / (m cp(T_avg0)), y despues hA = P0 / (T_f0 - T_avg0) y
UA = P0 / (T_avg0 - T_sec0).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from data.simulation.plant_params import (
    N_PRECURSOR_GROUPS,
    NeutronicsCoefficients,
    PlantParams,
)
from data.simulation.water_properties import WaterTable

Array = NDArray[np.float64]

_MW_PER_J_S = 1.0e-6
"""m [kg/s] x cp [J/(kg K)] x 1e-6 = capacidad de transporte en MW/K."""

_RTOL = 1.0e-8
_ATOL = 1.0e-10
_MAX_PROPS_ITERATIONS = 50
_PROPS_TOLERANCE_K = 1.0e-12
_MAX_LOAD = 1.5
_METHODS = ("LSODA", "Radau", "BDF")


class StateIndex(IntEnum):
    """Posicion de cada variable en el vector de estado."""

    N = 0
    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4
    C5 = 5
    C6 = 6
    T_FUEL = 7
    T_CORE = 8
    T_SG = 9
    PRESSURE = 10
    VOID = 11
    FLOW = 12
    ROD_INTEGRAL = 13
    ROD_REACTIVITY = 14
    CALORIMETRIC = 15


N_STATES = len(StateIndex)
STATE_NAMES: tuple[str, ...] = tuple(member.name.lower() for member in StateIndex)

_PRECURSORS = slice(StateIndex.C1, StateIndex.C6 + 1)

OBSERVABLE_NAMES: tuple[str, ...] = (
    "t_c",
    "t_s",
    "t_avg",
    "t_avg_dev_k",
    "t_fuel",
    "t_sec",
    "t_sat_c",
    "p_primary_bar",
    "p_secondary_bar",
    "flow_total_kg_s",
    "flow_hot_leg_kg_s",
    "flow_cold_leg_kg_s",
    "insurge_kg_s",
    "power_fraction",
    "q_sec_mw",
    "q_sec_fraction",
    "q_cal_fraction",
    "rod_position_percent",
    "void_fraction",
    "leak_bar",
    "load_fraction",
)
"""Magnitudes verdaderas que `observables` entrega y que el catalogo referencia."""


@dataclass(frozen=True)
class ModelInputs:
    """Entradas externas del modelo, constantes (o lineales) dentro de un paso.

    Attributes:
        load: Carga de turbina como fraccion del nominal (0-1,5).
        reactivity_insertion: Reactividad externa anadida [dk/k].
        pump_demand: Demanda de caudal de la bomba como fraccion del nominal.
        leak_bar: Despresurizacion equivalente por fuga [bar].
        rod_control_enabled: Si es False el PI de barras queda congelado.
    """

    load: float = 1.0
    reactivity_insertion: float = 0.0
    pump_demand: float = 1.0
    leak_bar: float = 0.0
    rod_control_enabled: bool = True

    def as_array(self) -> Array:
        """Return (load, insertion, demand, leak, rods_enabled) as float64."""
        return np.array(
            [
                self.load,
                self.reactivity_insertion,
                self.pump_demand,
                self.leak_bar,
                1.0 if self.rod_control_enabled else 0.0,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class NominalPoint:
    """Punto nominal derivado de los parametros de planta.

    Attributes:
        t_out_c: Temperatura de salida del nucleo [C].
        t_avg_c: Temperatura media del refrigerante [C].
        cp_j_kg_k: Calor especifico a (T_avg, P_nom) [J/(kg K)].
        ha_mw_k: Conductancia combustible-refrigerante [MW/K].
        ua_mw_k: Conductancia primario-secundario [MW/K].
        p_secondary_bar: Presion de saturacion del GV [bar].
        t_sat_c: Saturacion del primario a la presion nominal [C].
    """

    t_out_c: float
    t_avg_c: float
    cp_j_kg_k: float
    ha_mw_k: float
    ua_mw_k: float
    p_secondary_bar: float
    t_sat_c: float


@dataclass(frozen=True)
class StepResult:
    """Resultado de un paso de integracion.

    Attributes:
        y: Estado al final del intervalo.
        energy_generated_mj: Energia generada en el intervalo [MJ].
        energy_removed_mj: Energia cedida al secundario en el intervalo [MJ].
        internal_energy_change_mj: Variacion de la energia interna de los nodos [MJ].
        energy_residual_rel: |dU - (E_gen - E_sec)| / E_gen del intervalo.
        n_rhs_evaluations: Llamadas a la derivada que costo el paso.
    """

    y: Array
    energy_generated_mj: float
    energy_removed_mj: float
    internal_energy_change_mj: float
    energy_residual_rel: float
    n_rhs_evaluations: int


@dataclass(frozen=True)
class Diagnostics:
    """Balance instantaneo del estado completo (verdad terreno de las constraints).

    Attributes:
        q_gen_mw: Potencia generada P0 n [MW].
        q_fuel_to_coolant_mw: Calor combustible -> refrigerante [MW].
        q_core_removed_mw: Calor retirado del nucleo m cp (T_c - T_s) [MW].
        q_secondary_mw: Calor cedido al secundario [MW].
        du_dt_mw: Derivada de la energia interna de los nodos [MW].
        conservation_residual_mw: du_dt - (q_gen - q_secondary) [MW].
        energy_imbalance_fraction: |q_gen - q_core_removed| / q_gen.
        reactivity: Reactividad total [dk/k].
        dt_avg_dt_k_s: Derivada de T_avg [K/s].
        max_temperature_rate_c_s: max |dT/dt| entre T_f, T_c y T_s [C/s].
        cp_j_kg_k: cp usado en el transporte [J/(kg K)].
    """

    q_gen_mw: float
    q_fuel_to_coolant_mw: float
    q_core_removed_mw: float
    q_secondary_mw: float
    du_dt_mw: float
    conservation_residual_mw: float
    energy_imbalance_fraction: float
    reactivity: float
    dt_avg_dt_k_s: float
    max_temperature_rate_c_s: float
    cp_j_kg_k: float


@dataclass(frozen=True)
class _Props:
    """Propiedades del agua que la derivada trata como constantes."""

    cp_j_kg_k: float
    t_sat_c: float


class ReactorModel:
    """Modelo ODE determinista del reactor, calibrado en equilibrio.

    Args:
        plant: Parametros de planta.
        neutronics: Coeficientes neutronicos.
        table: Tabla de propiedades del agua.
        freeze_properties: Si es True (defecto) cp y T_sat se congelan al inicio
            de cada paso; si es False se evaluan en cada derivada.
        method: Metodo de solve_ivp (LSODA por defecto; Radau y BDF existen para
            contrastar LSODA).
        rtol: Tolerancia relativa del integrador.
        atol: Tolerancia absoluta del integrador.

    Raises:
        ValueError: If method is unknown, the nominal outlet is not subcooled or
            the secondary saturation point falls outside the water table.
    """

    def __init__(
        self,
        plant: PlantParams,
        neutronics: NeutronicsCoefficients,
        table: WaterTable,
        *,
        freeze_properties: bool = True,
        method: str = "LSODA",
        rtol: float = _RTOL,
        atol: float = _ATOL,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got '{method}'.")
        self.plant = plant
        self.neutronics = neutronics
        self.table = table
        self.freeze_properties = freeze_properties
        self.method = method
        self._rtol = rtol
        self._atol = atol
        self.nominal = self._derive_nominal()
        self._delta_t_sg_k = self.nominal.t_avg_c - plant.t_secondary_nominal_c
        self._p_floor_bar = float(table.p_range_bar[0])
        self._p_ceiling_bar = float(table.p_range_bar[1])

        self._beta_total = neutronics.beta_total
        self._fractions = neutronics.beta_fractions
        self._decay = neutronics.decay_constants_per_s
        if len(self._fractions) != N_PRECURSOR_GROUPS:
            raise ValueError("neutronics must define 6 precursor groups.")

    # ------------------------------------------------------------------
    # Punto nominal y equilibrio
    # ------------------------------------------------------------------

    def _derive_nominal(self) -> NominalPoint:
        """Derive T_out0, T_avg0, hA and UA so that the nominal point is exact."""
        plant = self.plant
        t_avg = plant.t_in_nominal_c
        t_out = plant.t_in_nominal_c
        cp = self.table.cp(t_avg, plant.nominal_pressure_bar)
        for _ in range(_MAX_PROPS_ITERATIONS):
            cp = self.table.cp(t_avg, plant.nominal_pressure_bar)
            delta = plant.thermal_power_mw / (plant.nominal_flow_kg_s * cp * _MW_PER_J_S)
            t_out = plant.t_in_nominal_c + delta
            new_avg = plant.t_in_nominal_c + 0.5 * delta
            converged = abs(new_avg - t_avg) < _PROPS_TOLERANCE_K
            t_avg = new_avg
            if converged:
                break
        t_sat = self.table.t_sat_c(plant.nominal_pressure_bar)
        if t_out >= t_sat:
            raise ValueError(
                f"nominal outlet {t_out:.2f} C is not subcooled (T_sat = {t_sat:.2f} C)."
            )
        p_secondary = self.table.p_sat_bar(plant.t_secondary_nominal_c)
        return NominalPoint(
            t_out_c=t_out,
            t_avg_c=t_avg,
            cp_j_kg_k=cp,
            ha_mw_k=plant.thermal_power_mw / (plant.t_fuel_nominal_c - t_avg),
            ua_mw_k=plant.thermal_power_mw / (t_avg - plant.t_secondary_nominal_c),
            p_secondary_bar=p_secondary,
            t_sat_c=t_sat,
        )

    @property
    def pressure_limits_bar(self) -> tuple[float, float]:
        """Return the (floor, ceiling) the pressure target is limited to [bar]."""
        return self._p_floor_bar, self._p_ceiling_bar

    def t_reference_c(self, load: float) -> float:
        """Return the T_avg programmed by the rod controller at a load [C]."""
        slope = self.plant.t_ref_slope_k_per_load
        return self.nominal.t_avg_c + slope * (load - 1.0)

    def t_secondary_c(self, load: float) -> float:
        """Return the secondary saturation temperature at a load [C].

        It follows the T_avg program so that the driving difference
        T_avg - T_sec stays at its nominal value (see the module docstring).
        """
        return self.t_reference_c(load) - self._delta_t_sg_k

    def steady_state(self, load: float = 1.0) -> Array:
        """Return the exact equilibrium of the equations at a given load.

        The rods hold T_avg = T_ref(load), the pump runs at nominal demand and
        there is no leak or external reactivity.

        Args:
            load: Turbine load as a fraction of nominal, in (0, 1.5].

        Returns:
            State vector with dy/dt = 0 (to machine precision).

        Raises:
            ValueError: If load is outside (0, 1.5] or the resulting state is
                outside the water table.
        """
        if not 0.0 < load <= _MAX_LOAD:
            raise ValueError(f"load must be in (0, {_MAX_LOAD}], got {load}.")
        plant, nominal, neutronics = self.plant, self.nominal, self.neutronics
        t_avg = self.t_reference_c(load)
        q_sec = nominal.ua_mw_k * load * (t_avg - self.t_secondary_c(load))
        n = q_sec / plant.thermal_power_mw
        pressure = plant.nominal_pressure_bar + plant.pressurizer_gain_bar_per_k * (
            t_avg - nominal.t_avg_c
        )
        cp = self.table.cp(t_avg, pressure)
        delta = q_sec / (plant.nominal_flow_kg_s * cp * _MW_PER_J_S)
        t_fuel = t_avg + q_sec / nominal.ha_mw_k
        rod = -(
            neutronics.alpha_fuel_per_k * (t_fuel - plant.t_fuel_nominal_c)
            + neutronics.alpha_moderator_per_k * (t_avg - nominal.t_avg_c)
        )
        y = np.empty(N_STATES, dtype=np.float64)
        y[StateIndex.N] = n
        y[_PRECURSORS] = n
        y[StateIndex.T_FUEL] = t_fuel
        y[StateIndex.T_CORE] = t_avg + 0.5 * delta
        y[StateIndex.T_SG] = t_avg - 0.5 * delta
        y[StateIndex.PRESSURE] = pressure
        y[StateIndex.VOID] = 0.0
        y[StateIndex.FLOW] = plant.nominal_flow_kg_s
        y[StateIndex.ROD_INTEGRAL] = rod / plant.rod_ki_per_k_s
        y[StateIndex.ROD_REACTIVITY] = rod
        y[StateIndex.CALORIMETRIC] = n
        return y

    # ------------------------------------------------------------------
    # Derivada
    # ------------------------------------------------------------------

    def _props(self, t_avg_c: float, pressure_bar: float) -> _Props:
        """Evaluate the water properties the derivative treats as constants."""
        return _Props(
            cp_j_kg_k=self.table.cp(t_avg_c, pressure_bar),
            t_sat_c=self.table.t_sat_c(pressure_bar),
        )

    def _derivative(
        self,
        y: list[float],
        u: tuple[float, float, float, float, float],
        props: _Props,
    ) -> list[float]:
        """Return dy/dt as a list; plain floats keep the hot path cheap.

        Args:
            y: State as a list of N_STATES floats.
            u: (load, insertion, demand, leak, rods_enabled).
            props: Water properties treated as constants.

        Returns:
            dy/dt as a list of N_STATES floats.
        """
        plant, nominal, neutronics = self.plant, self.nominal, self.neutronics
        load, insertion, demand, leak, rods_on = u
        n = y[0]
        t_fuel, t_core, t_sg = y[7], y[8], y[9]
        pressure, void, flow = y[10], y[11], y[12]
        z, rod, q_cal = y[13], y[14], y[15]
        t_avg = 0.5 * (t_core + t_sg)

        rho = (
            rod
            + neutronics.alpha_fuel_per_k * (t_fuel - plant.t_fuel_nominal_c)
            + neutronics.alpha_moderator_per_k * (t_avg - nominal.t_avg_c)
            + neutronics.alpha_void_per_fraction * void
            + insertion
        )
        delayed = 0.0
        dprec: list[float] = []
        for fraction, decay, c in zip(self._fractions, self._decay, y[1:7], strict=True):
            delayed += fraction * c
            dprec.append(decay * (n - c))
        dn = ((rho - self._beta_total) * n + self._beta_total * delayed) / (
            neutronics.generation_time_s
        )

        transport = flow * props.cp_j_kg_k * _MW_PER_J_S
        q_gen = plant.thermal_power_mw * n
        q_fuel = nominal.ha_mw_k * (t_fuel - t_avg)
        q_core = transport * (t_core - t_sg)
        t_ref = nominal.t_avg_c + plant.t_ref_slope_k_per_load * (load - 1.0)
        q_sec = nominal.ua_mw_k * load * (t_avg - (t_ref - self._delta_t_sg_k))

        p_target = min(
            max(
                plant.nominal_pressure_bar
                + plant.pressurizer_gain_bar_per_k * (t_avg - nominal.t_avg_c)
                - leak,
                self._p_floor_bar,
            ),
            self._p_ceiling_bar,
        )
        void_target = min(
            max(
                (t_core - props.t_sat_c + plant.void_onset_offset_k) / plant.void_ramp_width_k,
                0.0,
            ),
            plant.void_max_fraction,
        )
        error = t_ref - t_avg
        r_command = plant.rod_kp_per_k * error + plant.rod_ki_per_k_s * z
        return [
            dn,
            *dprec,
            (q_gen - q_fuel) / plant.capacity_fuel_mj_k,
            (q_fuel - q_core) / plant.capacity_core_mj_k,
            (q_core - q_sec) / plant.capacity_sg_mj_k,
            (p_target - pressure) / plant.tau_pressurizer_s,
            (void_target - void) / plant.tau_void_s,
            (plant.nominal_flow_kg_s * demand - flow) / plant.tau_pump_s,
            rods_on * error,
            rods_on * (r_command - rod) / plant.tau_rod_s,
            (q_sec / plant.thermal_power_mw - q_cal) / plant.tau_calorimetric_s,
        ]

    def jacobian(
        self,
        y: list[float],
        u: tuple[float, float, float, float, float],
        props: _Props,
    ) -> Array:
        """Return the analytic Jacobian of the augmented system (properties frozen).

        The last two rows belong to the energy accumulators E_gen and E_sec that
        `advance` appends to the state; they depend only on n, T_c and T_s.

        Args:
            y: State as a list of N_STATES floats.
            u: (load, insertion, demand, leak, rods_enabled).
            props: Water properties treated as constants.

        Returns:
            Matrix of shape (N_STATES + 2, N_STATES + 2), d f_i / d y_j.
        """
        plant, nominal, neutronics = self.plant, self.nominal, self.neutronics
        load, _, _, _, rods_on = u
        n, t_core, t_sg, flow = y[0], y[8], y[9], y[12]
        t_avg = 0.5 * (t_core + t_sg)
        size = N_STATES + 2
        jac = np.zeros((size, size), dtype=np.float64)

        gen_time = neutronics.generation_time_s
        rho = (
            y[14]
            + neutronics.alpha_fuel_per_k * (y[7] - plant.t_fuel_nominal_c)
            + neutronics.alpha_moderator_per_k * (t_avg - nominal.t_avg_c)
            + neutronics.alpha_void_per_fraction * y[11]
            + u[1]
        )
        jac[0, 0] = (rho - self._beta_total) / gen_time
        for i, (fraction, decay) in enumerate(zip(self._fractions, self._decay, strict=True)):
            jac[0, 1 + i] = self._beta_total * fraction / gen_time
            jac[1 + i, 0] = decay
            jac[1 + i, 1 + i] = -decay
        jac[0, 7] = n * neutronics.alpha_fuel_per_k / gen_time
        jac[0, 8] = jac[0, 9] = n * neutronics.alpha_moderator_per_k / (2.0 * gen_time)
        jac[0, 11] = n * neutronics.alpha_void_per_fraction / gen_time
        jac[0, 14] = n / gen_time

        ha, ua_eff = nominal.ha_mw_k, nominal.ua_mw_k * load
        cp_w = props.cp_j_kg_k * _MW_PER_J_S
        transport = flow * cp_w
        c_f, c_c, c_s = plant.capacity_fuel_mj_k, plant.capacity_core_mj_k, plant.capacity_sg_mj_k
        jac[7, 0] = plant.thermal_power_mw / c_f
        jac[7, 7] = -ha / c_f
        jac[7, 8] = jac[7, 9] = 0.5 * ha / c_f
        jac[8, 7] = ha / c_c
        jac[8, 8] = (-0.5 * ha - transport) / c_c
        jac[8, 9] = (-0.5 * ha + transport) / c_c
        jac[8, 12] = -cp_w * (t_core - t_sg) / c_c
        jac[9, 8] = (transport - 0.5 * ua_eff) / c_s
        jac[9, 9] = (-transport - 0.5 * ua_eff) / c_s
        jac[9, 12] = cp_w * (t_core - t_sg) / c_s

        tau_p = plant.tau_pressurizer_s
        p_raw = (
            plant.nominal_pressure_bar
            + plant.pressurizer_gain_bar_per_k * (t_avg - nominal.t_avg_c)
            - u[3]
        )
        if self._p_floor_bar < p_raw < self._p_ceiling_bar:
            jac[10, 8] = jac[10, 9] = 0.5 * plant.pressurizer_gain_bar_per_k / tau_p
        jac[10, 10] = -1.0 / tau_p

        void_arg = (t_core - props.t_sat_c + plant.void_onset_offset_k) / plant.void_ramp_width_k
        if 0.0 < void_arg < plant.void_max_fraction:
            jac[11, 8] = 1.0 / (plant.void_ramp_width_k * plant.tau_void_s)
        jac[11, 11] = -1.0 / plant.tau_void_s
        jac[12, 12] = -1.0 / plant.tau_pump_s

        jac[13, 8] = jac[13, 9] = -0.5 * rods_on
        tau_r = plant.tau_rod_s
        jac[14, 8] = jac[14, 9] = -0.5 * rods_on * plant.rod_kp_per_k / tau_r
        jac[14, 13] = rods_on * plant.rod_ki_per_k_s / tau_r
        jac[14, 14] = -rods_on / tau_r
        tau_q = plant.tau_calorimetric_s
        jac[15, 8] = jac[15, 9] = 0.5 * ua_eff / (plant.thermal_power_mw * tau_q)
        jac[15, 15] = -1.0 / tau_q

        jac[N_STATES, 0] = plant.thermal_power_mw
        jac[N_STATES + 1, 8] = jac[N_STATES + 1, 9] = 0.5 * ua_eff
        return jac

    def _rhs(self, y: Array, u: Array, props: _Props | None) -> Array:
        """Return dy/dt for state y, inputs u and (optionally) frozen props."""
        values = y.tolist()
        live = props
        if live is None:
            live = self._props(0.5 * (values[8] + values[9]), values[10])
        ul = u.tolist()
        return np.array(
            self._derivative(values, (ul[0], ul[1], ul[2], ul[3], ul[4]), live),
            dtype=np.float64,
        )

    def derivatives(self, y: Array, inputs: ModelInputs) -> Array:
        """Return dy/dt at a state with properties evaluated at that state.

        Args:
            y: State vector of length N_STATES.
            inputs: External inputs.

        Returns:
            dy/dt, same shape as y.

        Raises:
            ValueError: If y has the wrong shape or leaves the water table.
        """
        state = self._as_state(y)
        return self._rhs(state, inputs.as_array(), None)

    # ------------------------------------------------------------------
    # Integracion
    # ------------------------------------------------------------------

    @staticmethod
    def _as_state(y: Array) -> Array:
        """Return y as a float64 vector, validating its length."""
        state = np.asarray(y, dtype=np.float64)
        if state.shape != (N_STATES,):
            raise ValueError(f"state must have shape ({N_STATES},), got {state.shape}.")
        return state

    def internal_energy_mj(self, y: Array) -> float:
        """Return the energy stored in the three thermal nodes (reference 0 C) [MJ]."""
        plant = self.plant
        return float(
            plant.capacity_fuel_mj_k * y[StateIndex.T_FUEL]
            + plant.capacity_core_mj_k * y[StateIndex.T_CORE]
            + plant.capacity_sg_mj_k * y[StateIndex.T_SG]
        )

    def advance(
        self,
        y: Array,
        inputs: ModelInputs,
        dt_s: float,
        inputs_end: ModelInputs | None = None,
    ) -> StepResult:
        """Integrate the model over one interval with LSODA.

        Args:
            y: State at the start of the interval.
            inputs: External inputs at the start of the interval.
            dt_s: Interval length [s], strictly positive.
            inputs_end: Inputs at the end of the interval. When given, load,
                reactivity insertion, pump demand and leak vary linearly
                between `inputs` and `inputs_end`; the rod-control switch is
                taken from `inputs`.

        Returns:
            The StepResult with the end state and the energy closure.

        Raises:
            ValueError: If dt_s <= 0, y has the wrong shape or the state leaves
                the water table.
            RuntimeError: If the integrator fails.
        """
        if not dt_s > 0.0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}.")
        state = self._as_state(y)
        u0 = inputs.as_array()
        u1 = (inputs_end if inputs_end is not None else inputs).as_array()
        u1[4] = u0[4]
        props = (
            self._props(0.5 * (state[StateIndex.T_CORE] + state[StateIndex.T_SG]),
                        state[StateIndex.PRESSURE])
            if self.freeze_properties
            else None
        )
        p0 = self.plant.thermal_power_mw
        ua = self.nominal.ua_mw_k
        u0l = u0.tolist()
        dul = (u1 - u0).tolist()
        derivative = self._derivative

        def inputs_at(t: float) -> tuple[float, float, float, float, float]:
            frac = t / dt_s
            return (
                u0l[0] + dul[0] * frac,
                u0l[1] + dul[1] * frac,
                u0l[2] + dul[2] * frac,
                u0l[3] + dul[3] * frac,
                u0l[4],
            )

        def jac(t: float, aug: Array) -> Array:
            assert props is not None
            return self.jacobian(aug.tolist()[:N_STATES], inputs_at(t), props)

        def rhs(t: float, aug: Array) -> list[float]:
            u = inputs_at(t)
            values = aug.tolist()
            t_avg = 0.5 * (values[8] + values[9])
            live = props if props is not None else self._props(t_avg, values[10])
            out = derivative(values[:N_STATES], u, live)
            out.append(p0 * values[0])
            out.append(ua * u[0] * (t_avg - self.t_secondary_c(u[0])))
            return out

        start = np.concatenate([state, [0.0, 0.0]])
        solution = solve_ivp(
            rhs,
            (0.0, dt_s),
            start,
            method=self.method,
            t_eval=[dt_s],
            jac=jac if props is not None else None,
            rtol=self._rtol,
            atol=self._atol,
        )
        if not solution.success:
            raise RuntimeError(f"LSODA failed: {solution.message}")
        end = solution.y[:, -1]
        y_end = np.array(end[:N_STATES], dtype=np.float64)
        # El vacio es una fraccion: el integrador puede dejar ruido de redondeo
        # (p. ej. -1e-30) fuera de [0, void_max]; se acota sin tocar la energia.
        y_end[StateIndex.VOID] = min(
            max(y_end[StateIndex.VOID], 0.0), self.plant.void_max_fraction
        )
        e_gen = float(end[N_STATES])
        e_sec = float(end[N_STATES + 1])
        d_u = self.internal_energy_mj(y_end) - self.internal_energy_mj(state)
        residual = abs(d_u - (e_gen - e_sec)) / max(e_gen, np.finfo(float).tiny)
        return StepResult(
            y=y_end,
            energy_generated_mj=e_gen,
            energy_removed_mj=e_sec,
            internal_energy_change_mj=d_u,
            energy_residual_rel=float(residual),
            n_rhs_evaluations=int(solution.nfev),
        )

    # ------------------------------------------------------------------
    # Observables y diagnosticos
    # ------------------------------------------------------------------

    def observables(self, y: Array, inputs: ModelInputs) -> dict[str, float]:
        """Return the true (noise-free) magnitudes a sensor can read.

        Args:
            y: State vector.
            inputs: External inputs in force at that state.

        Returns:
            Mapping with exactly the names in OBSERVABLE_NAMES.

        Raises:
            ValueError: If y has the wrong shape or leaves the water table.
        """
        state = self._as_state(y)
        plant, nominal, neutronics = self.plant, self.nominal, self.neutronics
        t_core = float(state[StateIndex.T_CORE])
        t_sg = float(state[StateIndex.T_SG])
        t_avg = 0.5 * (t_core + t_sg)
        pressure = float(state[StateIndex.PRESSURE])
        dydt = self._rhs(state, inputs.as_array(), None)
        dt_avg = 0.5 * float(dydt[StateIndex.T_CORE] + dydt[StateIndex.T_SG])
        insurge = plant.surge_kg_s_per_k_s * dt_avg
        t_sec = self.t_secondary_c(inputs.load)
        q_sec = nominal.ua_mw_k * inputs.load * (t_avg - t_sec)
        flow = float(state[StateIndex.FLOW])
        values = {
            "t_c": t_core,
            "t_s": t_sg,
            "t_avg": t_avg,
            "t_avg_dev_k": t_avg - nominal.t_avg_c,
            "t_fuel": float(state[StateIndex.T_FUEL]),
            "t_sec": t_sec,
            "t_sat_c": self.table.t_sat_c(pressure),
            "p_primary_bar": pressure,
            "p_secondary_bar": self.table.p_sat_bar(t_sec),
            "flow_total_kg_s": flow,
            "flow_hot_leg_kg_s": flow + insurge,
            "flow_cold_leg_kg_s": flow,
            "insurge_kg_s": insurge,
            "power_fraction": float(state[StateIndex.N]),
            "q_sec_mw": q_sec,
            "q_sec_fraction": q_sec / plant.thermal_power_mw,
            "q_cal_fraction": float(state[StateIndex.CALORIMETRIC]),
            "rod_position_percent": neutronics.rod_position_nominal_percent
            + float(state[StateIndex.ROD_REACTIVITY]) / neutronics.rod_worth_per_percent,
            "void_fraction": float(state[StateIndex.VOID]),
            "leak_bar": inputs.leak_bar,
            "load_fraction": inputs.load,
        }
        return values

    def diagnostics(self, y: Array, inputs: ModelInputs) -> Diagnostics:
        """Return the instantaneous energy balance of the full state.

        Args:
            y: State vector.
            inputs: External inputs in force at that state.

        Returns:
            Diagnostics with the heat flows, the conservation residual and the
            fastest temperature rate.

        Raises:
            ValueError: If y has the wrong shape or leaves the water table.
        """
        state = self._as_state(y)
        plant, nominal, neutronics = self.plant, self.nominal, self.neutronics
        t_core = float(state[StateIndex.T_CORE])
        t_sg = float(state[StateIndex.T_SG])
        t_fuel = float(state[StateIndex.T_FUEL])
        t_avg = 0.5 * (t_core + t_sg)
        props = self._props(t_avg, float(state[StateIndex.PRESSURE]))
        dydt = self._rhs(state, inputs.as_array(), props)

        q_gen = plant.thermal_power_mw * float(state[StateIndex.N])
        q_fuel = nominal.ha_mw_k * (t_fuel - t_avg)
        q_core = float(state[StateIndex.FLOW]) * props.cp_j_kg_k * _MW_PER_J_S * (t_core - t_sg)
        q_sec = nominal.ua_mw_k * inputs.load * (t_avg - self.t_secondary_c(inputs.load))
        du_dt = float(
            plant.capacity_fuel_mj_k * dydt[StateIndex.T_FUEL]
            + plant.capacity_core_mj_k * dydt[StateIndex.T_CORE]
            + plant.capacity_sg_mj_k * dydt[StateIndex.T_SG]
        )
        rho = (
            float(state[StateIndex.ROD_REACTIVITY])
            + neutronics.alpha_fuel_per_k * (t_fuel - plant.t_fuel_nominal_c)
            + neutronics.alpha_moderator_per_k * (t_avg - nominal.t_avg_c)
            + neutronics.alpha_void_per_fraction * float(state[StateIndex.VOID])
            + inputs.reactivity_insertion
        )
        rates = np.abs(
            dydt[[StateIndex.T_FUEL, StateIndex.T_CORE, StateIndex.T_SG]]
        )
        return Diagnostics(
            q_gen_mw=q_gen,
            q_fuel_to_coolant_mw=q_fuel,
            q_core_removed_mw=q_core,
            q_secondary_mw=q_sec,
            du_dt_mw=du_dt,
            conservation_residual_mw=du_dt - (q_gen - q_sec),
            energy_imbalance_fraction=abs(q_gen - q_core) / max(q_gen, np.finfo(float).tiny),
            reactivity=rho,
            dt_avg_dt_k_s=0.5 * float(dydt[StateIndex.T_CORE] + dydt[StateIndex.T_SG]),
            max_temperature_rate_c_s=float(rates.max()),
            cp_j_kg_k=props.cp_j_kg_k,
        )

    def relative_derivative(self, y: Array, inputs: ModelInputs) -> float:
        """Return max_i |dy_i/dt| / max(|y_i|, 1) [1/s].

        Scale-free measure of how far a state is from equilibrium; the floor
        of 1 keeps the zero-valued states (void, rod reactivity) meaningful.

        Args:
            y: State vector.
            inputs: External inputs.

        Returns:
            The largest relative rate over the state vector.
        """
        state = self._as_state(y)
        rate = np.abs(self._rhs(state, inputs.as_array(), None))
        return float(np.max(rate / np.maximum(np.abs(state), 1.0)))

