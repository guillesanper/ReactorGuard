"""Acceso tipado a configs/plant_parameters.yaml y configs/neutronics_coefficients.yaml.

Mismo patron que data/generators/tep_params.py: dataclasses inmutables, una
funcion `_require` que nombra la clave ausente y un loader por fichero. Los
YAML de `configs/` son `deps` del stage `simulate` (F3-M6), de modo que cambiar
un valor invalida la reproduccion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PLANT_PARAMS_PATH = Path("configs/plant_parameters.yaml")
DEFAULT_NEUTRONICS_PATH = Path("configs/neutronics_coefficients.yaml")

N_PRECURSOR_GROUPS = 6
"""Numero de grupos de neutrones retardados de la cinetica puntual."""


@dataclass(frozen=True)
class PlantLimits:
    """Limites fisicos que consumen las constraints (F3-M2).

    Attributes:
        max_dt_c_per_s: Maxima derivada de temperatura admisible [C/s] (D9).
        energy_imbalance_fraction: Desbalance de energia que dispara la regla.
        pump_max_fraction: Tope de caudal de bomba como fraccion del nominal.
        n_loops: Lazos del primario; cada canal de caudal mide total / n_loops.
        mass_balance_tolerance: |m_hl - m_cl| / m que dispara la regla de masa.
        quasi_steady_dt_avg_k_s: |dT_avg/dt| bajo el cual rige la regla de masa [K/s].
        power_flux_k: Constante k de |P_th - k flux| (ambos en % del nominal).
        power_flux_threshold_percent: Umbral de |P_th - k flux| [puntos porcentuales].
        conservation_tolerance: Residuo relativo de conservacion de energia admitido.
    """

    max_dt_c_per_s: float
    energy_imbalance_fraction: float
    pump_max_fraction: float
    n_loops: int
    mass_balance_tolerance: float
    quasi_steady_dt_avg_k_s: float
    power_flux_k: float
    power_flux_threshold_percent: float
    conservation_tolerance: float


@dataclass(frozen=True)
class PlantParams:
    """Parametros de planta del modelo termico.

    Attributes:
        thermal_power_mw: Potencia termica nominal [MW].
        nominal_flow_kg_s: Caudal total del primario [kg/s].
        nominal_pressure_bar: Presion nominal del primario [bar].
        t_in_nominal_c: Temperatura nominal de pierna fria [C].
        t_fuel_nominal_c: Temperatura nominal del combustible [C].
        t_secondary_nominal_c: Temperatura de saturacion nominal del GV [C].
        capacity_fuel_mj_k: Capacidad termica del nodo combustible [MJ/K].
        capacity_core_mj_k: Capacidad termica del nodo nucleo [MJ/K].
        capacity_sg_mj_k: Capacidad termica del nodo GV/primario [MJ/K].
        tau_pump_s: Constante de tiempo de la bomba [s].
        tau_pressurizer_s: Constante de tiempo del presurizador [s].
        tau_void_s: Constante de tiempo del vacio [s].
        tau_calorimetric_s: Retardo de la potencia calorimetrica [s].
        pressurizer_gain_bar_per_k: Ganancia dP/dT_avg [bar/K].
        void_onset_offset_k: Subenfriamiento al que empieza el vacio [K].
        void_ramp_width_k: Anchura de la rampa del vacio [K].
        void_max_fraction: Tope del vacio [fraccion].
        t_ref_slope_k_per_load: Pendiente del programa de T_avg [K por unidad de carga].
        surge_kg_s_per_k_s: Insurge del presurizador por dT_avg/dt [kg/s por K/s].
        rod_kp_per_k: Ganancia proporcional del PI de barras [dk/k por K].
        rod_ki_per_k_s: Ganancia integral del PI de barras [dk/k por K s].
        tau_rod_s: Retardo mecanico de las barras [s].
        limits: Limites de constraints.
        fault_tiers: Niveles de magnitud de fallos de proceso (F3-M4).
        instrument_faults: Niveles de magnitud de fallos de instrumento (F3-M5).
        scenario_planner: Topes de densidad y minimos del planificador (F3-M5).
    """

    thermal_power_mw: float
    nominal_flow_kg_s: float
    nominal_pressure_bar: float
    t_in_nominal_c: float
    t_fuel_nominal_c: float
    t_secondary_nominal_c: float
    capacity_fuel_mj_k: float
    capacity_core_mj_k: float
    capacity_sg_mj_k: float
    tau_pump_s: float
    tau_pressurizer_s: float
    tau_void_s: float
    tau_calorimetric_s: float
    pressurizer_gain_bar_per_k: float
    void_onset_offset_k: float
    void_ramp_width_k: float
    void_max_fraction: float
    t_ref_slope_k_per_load: float
    surge_kg_s_per_k_s: float
    rod_kp_per_k: float
    rod_ki_per_k_s: float
    tau_rod_s: float
    limits: PlantLimits
    fault_tiers: dict[str, Any]
    instrument_faults: dict[str, Any] = field(default_factory=dict)
    scenario_planner: dict[str, Any] = field(default_factory=dict)

    @property
    def pump_max_kg_s(self) -> float:
        """Return the pump flow ceiling [kg/s]."""
        return self.limits.pump_max_fraction * self.nominal_flow_kg_s


@dataclass(frozen=True)
class NeutronicsCoefficients:
    """Coeficientes de la cinetica puntual de 6 grupos.

    Attributes:
        provenance: Origen de los valores (LITERATURE_PLACEHOLDER u OPENMC).
        beta: Fraccion de retardados por grupo.
        decay_constants_per_s: Constante de decaimiento por grupo [1/s].
        generation_time_s: Tiempo de generacion de neutrones prontos [s].
        alpha_fuel_per_k: Coeficiente Doppler [dk/k por K].
        alpha_moderator_per_k: Coeficiente del moderador [dk/k por K].
        alpha_void_per_fraction: Coeficiente de vacio [dk/k por unidad de vacio].
        rod_worth_per_percent: Worth del banco [dk/k por % de posicion].
        rod_position_nominal_percent: Posicion del banco en el punto nominal [%].
    """

    provenance: str
    beta: tuple[float, ...]
    decay_constants_per_s: tuple[float, ...]
    generation_time_s: float
    alpha_fuel_per_k: float
    alpha_moderator_per_k: float
    alpha_void_per_fraction: float
    rod_worth_per_percent: float
    rod_position_nominal_percent: float

    @property
    def beta_total(self) -> float:
        """Return the total delayed neutron fraction."""
        return float(sum(self.beta))

    @property
    def beta_fractions(self) -> tuple[float, ...]:
        """Return each group's share of the total delayed fraction (sums to 1)."""
        total = self.beta_total
        return tuple(b / total for b in self.beta)


def _require(section: Mapping[str, Any], key: str, label: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: Parsed mapping.
        key: Key that must be present.
        label: Where the section lives, e.g. "plant_parameters.yaml: 'plant'".

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"{label}: missing required key '{key}'.")
    return section[key]


def _read_document(path: str | Path, name: str) -> dict[str, Any]:
    """Read a YAML mapping from disk.

    Args:
        path: File to read.
        name: Human name of the file for error messages.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If path does not exist.
        TypeError: If the document is not a mapping.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{name} not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}
    if not isinstance(document, dict):
        raise TypeError(f"{name}: top level must be a mapping.")
    return document


def _section(document: Mapping[str, Any], key: str, name: str) -> Mapping[str, Any]:
    """Return a required mapping section of a document."""
    if key not in document:
        raise KeyError(f"{name}: missing required section '{key}:'.")
    section = document[key]
    if not isinstance(section, Mapping):
        raise TypeError(f"{name}: section '{key}:' must be a mapping.")
    return section


def _positive(label: str, value: float) -> float:
    """Return value if strictly positive.

    Raises:
        ValueError: If value <= 0.
    """
    if not value > 0.0:
        raise ValueError(f"{label} must be > 0, got {value}.")
    return value


def _positive_int(label: str, value: float) -> int:
    """Return value as an int if it is a whole number >= 1.

    Raises:
        ValueError: If value is not a whole number or is below 1.
    """
    if not (math.isfinite(value) and value >= 1.0 and value == int(value)):
        raise ValueError(f"{label} must be a whole number >= 1, got {value}.")
    return int(value)


def load_plant_params(path: str | Path = DEFAULT_PLANT_PARAMS_PATH) -> PlantParams:
    """Load and validate configs/plant_parameters.yaml.

    Args:
        path: Path to the YAML file.

    Returns:
        A fully resolved PlantParams.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError: If a section or required key is missing.
        TypeError: If a section is not a mapping.
        ValueError: If a value is outside its physical range.
    """
    name = "plant_parameters.yaml"
    document = _read_document(path, name)
    plant = _section(document, "plant", name)
    control = _section(document, "control", name)
    limits = _section(document, "limits", name)
    fault_tiers = _require(document, "fault_tiers", name)
    if not isinstance(fault_tiers, dict):
        raise TypeError(f"{name}: 'fault_tiers' must be a mapping.")
    instrument_faults = _require(document, "instrument_faults", name)
    if not isinstance(instrument_faults, dict):
        raise TypeError(f"{name}: 'instrument_faults' must be a mapping.")
    scenario_planner = _require(document, "scenario_planner", name)
    if not isinstance(scenario_planner, dict):
        raise TypeError(f"{name}: 'scenario_planner' must be a mapping.")

    def p(key: str) -> float:
        return float(_require(plant, key, f"{name}: 'plant'"))

    def c(key: str) -> float:
        return float(_require(control, key, f"{name}: 'control'"))

    def lim(key: str) -> float:
        return float(_require(limits, key, f"{name}: 'limits'"))

    params = PlantParams(
        thermal_power_mw=_positive("plant.thermal_power_mw", p("thermal_power_mw")),
        nominal_flow_kg_s=_positive("plant.nominal_flow_kg_s", p("nominal_flow_kg_s")),
        nominal_pressure_bar=_positive("plant.nominal_pressure_bar", p("nominal_pressure_bar")),
        t_in_nominal_c=p("t_in_nominal_c"),
        t_fuel_nominal_c=p("t_fuel_nominal_c"),
        t_secondary_nominal_c=p("t_secondary_nominal_c"),
        capacity_fuel_mj_k=_positive("plant.capacity_fuel_mj_k", p("capacity_fuel_mj_k")),
        capacity_core_mj_k=_positive("plant.capacity_core_mj_k", p("capacity_core_mj_k")),
        capacity_sg_mj_k=_positive("plant.capacity_sg_mj_k", p("capacity_sg_mj_k")),
        tau_pump_s=_positive("plant.tau_pump_s", p("tau_pump_s")),
        tau_pressurizer_s=_positive("plant.tau_pressurizer_s", p("tau_pressurizer_s")),
        tau_void_s=_positive("plant.tau_void_s", p("tau_void_s")),
        tau_calorimetric_s=_positive("plant.tau_calorimetric_s", p("tau_calorimetric_s")),
        pressurizer_gain_bar_per_k=p("pressurizer_gain_bar_per_k"),
        void_onset_offset_k=p("void_onset_offset_k"),
        void_ramp_width_k=_positive("plant.void_ramp_width_k", p("void_ramp_width_k")),
        void_max_fraction=_positive("plant.void_max_fraction", p("void_max_fraction")),
        t_ref_slope_k_per_load=p("t_ref_slope_k_per_load"),
        surge_kg_s_per_k_s=p("surge_kg_s_per_k_s"),
        rod_kp_per_k=c("rod_kp_per_k"),
        rod_ki_per_k_s=_positive("control.rod_ki_per_k_s", c("rod_ki_per_k_s")),
        tau_rod_s=_positive("control.tau_rod_s", c("tau_rod_s")),
        limits=PlantLimits(
            max_dt_c_per_s=_positive("limits.max_dt_c_per_s", lim("max_dt_c_per_s")),
            energy_imbalance_fraction=_positive(
                "limits.energy_imbalance_fraction", lim("energy_imbalance_fraction")
            ),
            pump_max_fraction=_positive("limits.pump_max_fraction", lim("pump_max_fraction")),
            n_loops=_positive_int("limits.n_loops", lim("n_loops")),
            mass_balance_tolerance=_positive(
                "limits.mass_balance_tolerance", lim("mass_balance_tolerance")
            ),
            quasi_steady_dt_avg_k_s=_positive(
                "limits.quasi_steady_dt_avg_k_s", lim("quasi_steady_dt_avg_k_s")
            ),
            power_flux_k=_positive("limits.power_flux_k", lim("power_flux_k")),
            power_flux_threshold_percent=_positive(
                "limits.power_flux_threshold_percent", lim("power_flux_threshold_percent")
            ),
            conservation_tolerance=_positive(
                "limits.conservation_tolerance", lim("conservation_tolerance")
            ),
        ),
        fault_tiers=dict(fault_tiers),
        instrument_faults=dict(instrument_faults),
        scenario_planner=dict(scenario_planner),
    )
    if params.void_max_fraction >= 1.0:
        raise ValueError("plant.void_max_fraction must be < 1.")
    if params.t_secondary_nominal_c >= params.t_in_nominal_c:
        raise ValueError("plant.t_secondary_nominal_c must be below t_in_nominal_c.")
    if params.t_fuel_nominal_c <= params.t_in_nominal_c:
        raise ValueError("plant.t_fuel_nominal_c must be above t_in_nominal_c.")
    return params


def load_neutronics(path: str | Path = DEFAULT_NEUTRONICS_PATH) -> NeutronicsCoefficients:
    """Load and validate configs/neutronics_coefficients.yaml.

    Args:
        path: Path to the YAML file.

    Returns:
        A fully resolved NeutronicsCoefficients.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError: If a section or required key is missing.
        TypeError: If a section is not a mapping.
        ValueError: If the group count is not 6, a value is not positive, the
            total delayed fraction is >= 1 or a reactivity coefficient is not
            negative (the model requires negative feedback).
    """
    name = "neutronics_coefficients.yaml"
    document = _read_document(path, name)
    provenance = str(_require(document, "provenance", name))
    section = _section(document, "neutronics", name)
    label = f"{name}: 'neutronics'"

    beta = tuple(float(v) for v in _require(section, "beta", label))
    decay = tuple(float(v) for v in _require(section, "decay_constants_per_s", label))
    if len(beta) != N_PRECURSOR_GROUPS or len(decay) != N_PRECURSOR_GROUPS:
        raise ValueError(
            f"{name}: beta and decay_constants_per_s must have {N_PRECURSOR_GROUPS} groups."
        )
    if any(v <= 0.0 for v in beta) or any(v <= 0.0 for v in decay):
        raise ValueError(f"{name}: beta and decay constants must be > 0.")
    if sum(beta) >= 1.0:
        raise ValueError(f"{name}: total delayed fraction must be < 1.")

    alphas = {
        key: float(_require(section, key, label))
        for key in ("alpha_fuel_per_k", "alpha_moderator_per_k", "alpha_void_per_fraction")
    }
    for key, value in alphas.items():
        if not value < 0.0:
            raise ValueError(f"{name}: {key} must be negative (negative feedback), got {value}.")

    return NeutronicsCoefficients(
        provenance=provenance,
        beta=beta,
        decay_constants_per_s=decay,
        generation_time_s=_positive(
            f"{name}: generation_time_s", float(_require(section, "generation_time_s", label))
        ),
        alpha_fuel_per_k=alphas["alpha_fuel_per_k"],
        alpha_moderator_per_k=alphas["alpha_moderator_per_k"],
        alpha_void_per_fraction=alphas["alpha_void_per_fraction"],
        rod_worth_per_percent=_positive(
            f"{name}: rod_worth_per_percent",
            float(_require(section, "rod_worth_per_percent", label)),
        ),
        rod_position_nominal_percent=float(
            _require(section, "rod_position_nominal_percent", label)
        ),
    )
