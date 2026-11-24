"""Perdidas fisicas de la PINN: balance de energia (D4-4) y de masa (D4-5).

Ambas son residuos ADIMENSIONALES al cuadrado sobre el estado fisico predicho y
desnormalizado (`ml.models.physics_state`), nunca sobre z-scores:

    energia:  ((Q_gen - Q_rem) / P0)^2
              Q_gen = P0 flux / 100                      [MW] (del flujo, no de P_th)
              Q_rem = n_loops m_in cp(T_avg, P) (t_out - t_in) 1e-6   [MW]
    masa:     ((m_out - m_in) / m_nom)^2,  m_nom = caudal nominal POR LAZO,
              solo en filas cuasi-estacionarias

cp sale de configs/water_table.yaml (no 4182 J/(kg K), C4 del plan) y entra como
COEFICIENTE sin gradiente: se interpola en numpy sobre valores separados del grafo.
La tabla nunca extrapola: una fila con (T_avg, P) fuera de la rejilla no cuenta en
la perdida de energia (ni en el numerador ni en el denominador de la media).

La forma estacionaria de la energia (sin dU/dt) deja un suelo sobre el estado
verdadero (0,15 % p99 del residuo relativo, seccion 2.3 del plan): es una perdida,
no una igualdad exacta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from data.simulation.physics_constraints import in_table_domain
from data.simulation.plant_params import PlantParams
from data.simulation.water_properties import WaterTable
from ml.models.physics_state import PhysicalState

_MW_PER_J_S = 1.0e-6
"""m [kg/s] x cp [J/(kg K)] x dT [K] x 1e-6 = potencia en MW."""

_PERCENT = 100.0


@dataclass(frozen=True)
class PhysicsConstants:
    """Escalas de planta que usan las perdidas.

    Attributes:
        thermal_power_mw: Potencia termica nominal P0 [MW].
        n_loops: Lazos del primario; cada canal de caudal mide total / n_loops.
        nominal_flow_per_loop_kg_s: m_nom, caudal nominal de un lazo [kg/s].
        quasi_steady_dt_avg_k_s: |dT_avg/dt| bajo el cual la fila cuenta en la
            perdida de masa [K/s] (el mismo umbral que la regla).

    Raises:
        ValueError: If a value is not finite and strictly positive, or n_loops < 1.
    """

    thermal_power_mw: float
    n_loops: int
    nominal_flow_per_loop_kg_s: float
    quasi_steady_dt_avg_k_s: float

    def __post_init__(self) -> None:
        """Validate that every scale is finite and strictly positive."""
        for name in ("thermal_power_mw", "nominal_flow_per_loop_kg_s", "quasi_steady_dt_avg_k_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"PhysicsConstants.{name} must be finite and > 0, got {value}.")
        if self.n_loops < 1:
            raise ValueError(f"PhysicsConstants.n_loops must be >= 1, got {self.n_loops}.")

    @classmethod
    def from_plant(cls, plant: PlantParams) -> PhysicsConstants:
        """Build the constants from configs/plant_parameters.yaml.

        Args:
            plant: Parametros de planta resueltos.

        Returns:
            The PhysicsConstants.
        """
        n_loops = plant.limits.n_loops
        return cls(
            thermal_power_mw=plant.thermal_power_mw,
            n_loops=n_loops,
            nominal_flow_per_loop_kg_s=plant.nominal_flow_kg_s / n_loops,
            quasi_steady_dt_avg_k_s=plant.limits.quasi_steady_dt_avg_k_s,
        )


@dataclass(frozen=True)
class PhysicsLossTerms:
    """Terminos fisicos de un lote.

    Attributes:
        energy: Media de ((Q_gen - Q_rem)/P0)^2 sobre las filas evaluables
            (tensor escalar con gradiente; 0 si ninguna lo es).
        mass: Media de ((m_out - m_in)/m_nom)^2 sobre las filas evaluables
            (cuasi-estacionarias y finitas).
        n_energy: Filas que entran en `energy`.
        n_mass: Filas que entran en `mass`.
    """

    energy: torch.Tensor
    mass: torch.Tensor
    n_energy: int
    n_mass: int

    def weighted(self, lambda_energy: float, lambda_mass: float) -> torch.Tensor:
        """Return lambda_energy * energy + lambda_mass * mass."""
        return lambda_energy * self.energy + lambda_mass * self.mass


def _finite_rows(*columns: torch.Tensor) -> torch.Tensor:
    """Return where every column is finite."""
    mask = torch.ones_like(columns[0], dtype=torch.bool)
    for column in columns:
        mask = mask & torch.isfinite(column)
    return mask


def _masked_mean_square(residual: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return mean(residual^2) over mask and the count (0 with gradient if empty).

    The residual must already be finite outside the mask (see `_safe`), so that no
    NaN reaches the gradient through the discarded rows.
    """
    count = int(mask.sum().item())
    squared = torch.where(mask, residual * residual, torch.zeros_like(residual))
    if count == 0:
        return squared.sum() * 0.0, 0
    return squared.sum() / count, count


def _safe(column: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return column with the rows outside mask replaced by 0 (no NaN gradient)."""
    return torch.where(mask, column, torch.zeros_like(column))


def cp_coefficient(
    table: WaterTable, t_avg_c: torch.Tensor, p_bar: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cp(T_avg, P) as a constant tensor and the in-grid mask.

    cp se interpola con la tabla (bilineal, numpy) sobre valores separados del
    grafo: es un coeficiente, no tiene gradiente. Fuera de la rejilla (o con NaN)
    no se extrapola: cp vale 0 y la mascara es False.

    Args:
        table: Tabla de propiedades del agua.
        t_avg_c: Temperatura media (batch,) [C].
        p_bar: Presion (batch,) [bar].

    Returns:
        (cp [J/(kg K)] sin gradiente, mascara booleana de rejilla), ambos (batch,).
    """
    t = t_avg_c.detach().cpu().numpy().astype(np.float64)
    p = p_bar.detach().cpu().numpy().astype(np.float64)
    ok = in_table_domain(table, t, p)
    cp = np.zeros_like(t)
    if ok.any():
        cp[ok] = table.cp(t[ok], p[ok])
    return (
        torch.as_tensor(cp, dtype=t_avg_c.dtype, device=t_avg_c.device),
        torch.as_tensor(ok, device=t_avg_c.device),
    )


def energy_balance_residual(
    state: PhysicalState, table: WaterTable, constants: PhysicsConstants
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signed (Q_gen - Q_rem) / P0 per row and where it is evaluable.

    Args:
        state: Estado fisico (normalmente el predicho, desnormalizado).
        table: Tabla de propiedades del agua.
        constants: Escalas de planta.

    Returns:
        (residuo (batch,), mascara de filas evaluables). Una fila no es evaluable
        si algun componente o la presion no es finito o si (T_avg, P) cae fuera
        de la tabla; en ella el residuo vale 0.
    """
    flux = state.component("flux")
    m_in = state.component("flow_in")
    t_in = state.component("t_in")
    t_out = state.component("t_out")
    finite = _finite_rows(flux, m_in, t_in, t_out, state.pressure_bar)
    cp, in_grid = cp_coefficient(table, state.t_avg, state.pressure_bar)
    valid = finite & in_grid
    flux, m_in = _safe(flux, valid), _safe(m_in, valid)
    t_in, t_out = _safe(t_in, valid), _safe(t_out, valid)
    q_gen = constants.thermal_power_mw * flux / _PERCENT
    q_rem = constants.n_loops * m_in * cp * (t_out - t_in) * _MW_PER_J_S
    residual = (q_gen - q_rem) / constants.thermal_power_mw
    return _safe(residual, valid), valid


def energy_balance_loss(
    state: PhysicalState, table: WaterTable, constants: PhysicsConstants
) -> torch.Tensor:
    """Return mean(((Q_gen - Q_rem) / P0)^2) over the evaluable rows (D4-4).

    Args:
        state: Estado fisico desnormalizado.
        table: Tabla de propiedades del agua.
        constants: Escalas de planta.

    Returns:
        Scalar tensor (0 with gradient if no row is evaluable).
    """
    residual, valid = energy_balance_residual(state, table, constants)
    return _masked_mean_square(residual, valid)[0]


def quasi_steady_mask(
    t_avg_from_c: torch.Tensor,
    t_avg_to_c: torch.Tensor,
    dt_s: float,
    constants: PhysicsConstants,
) -> torch.Tensor:
    """Return where |dT_avg/dt| < threshold between two MEASURED instants (D4-5).

    Como la regla de masa: diferencia finita de T_avg medida entre dos instantes
    consecutivos, operador estricto, NaN = no cuasi-estacionario. Sin gradiente
    (los argumentos se separan del grafo).

    Args:
        t_avg_from_c: T_avg medida en el instante anterior (batch,) [C].
        t_avg_to_c: T_avg medida en el instante posterior (batch,) [C].
        dt_s: Separacion entre ambos [s].
        constants: Escalas de planta (umbral).

    Returns:
        Boolean tensor (batch,).

    Raises:
        ValueError: If dt_s is not finite and strictly positive.
    """
    if not (math.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError(f"quasi_steady_mask needs dt_s > 0, got {dt_s}.")
    rate = (t_avg_to_c.detach() - t_avg_from_c.detach()) / dt_s
    return torch.isfinite(rate) & (rate.abs() < constants.quasi_steady_dt_avg_k_s)


def mass_balance_residual(
    state: PhysicalState, constants: PhysicsConstants
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the signed (m_out - m_in) / m_nom per row and where it is finite.

    Args:
        state: Estado fisico desnormalizado.
        constants: Escalas de planta.

    Returns:
        (residuo (batch,), mascara de filas finitas); el residuo vale 0 fuera.
    """
    m_in = state.component("flow_in")
    m_out = state.component("flow_out")
    finite = _finite_rows(m_in, m_out)
    residual = (_safe(m_out, finite) - _safe(m_in, finite)) / constants.nominal_flow_per_loop_kg_s
    return residual, finite


def mass_balance_loss(
    state: PhysicalState, steady: torch.Tensor, constants: PhysicsConstants
) -> torch.Tensor:
    """Return mean(((m_out - m_in) / m_nom)^2) over the quasi-steady rows (D4-5).

    Args:
        state: Estado fisico desnormalizado.
        steady: Mascara de cuasi-estacionario (batch,), de `quasi_steady_mask`.
        constants: Escalas de planta.

    Returns:
        Scalar tensor (0 with gradient if no row counts).
    """
    residual, finite = mass_balance_residual(state, constants)
    return _masked_mean_square(residual, finite & steady.to(torch.bool))[0]


def physics_loss_terms(
    state: PhysicalState,
    steady: torch.Tensor,
    table: WaterTable,
    constants: PhysicsConstants,
) -> PhysicsLossTerms:
    """Return both physics terms of a batch with their row counts.

    Args:
        state: Estado fisico predicho y desnormalizado.
        steady: Mascara de cuasi-estacionario de la perdida de masa.
        table: Tabla de propiedades del agua.
        constants: Escalas de planta.

    Returns:
        The PhysicsLossTerms.

    Raises:
        ValueError: If steady does not have one entry per row.
    """
    if steady.shape != state.values.shape[:1]:
        raise ValueError(
            f"physics loss: steady has shape {tuple(steady.shape)}, expected "
            f"({state.values.shape[0]},)."
        )
    e_residual, e_valid = energy_balance_residual(state, table, constants)
    m_residual, m_finite = mass_balance_residual(state, constants)
    energy, n_energy = _masked_mean_square(e_residual, e_valid)
    mass, n_mass = _masked_mean_square(m_residual, m_finite & steady.to(torch.bool))
    return PhysicsLossTerms(energy=energy, mass=mass, n_energy=n_energy, n_mass=n_mass)
