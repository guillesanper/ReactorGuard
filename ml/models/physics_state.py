"""Estado fisico de 5 componentes que leen las perdidas de la PINN (D4-3).

La red predice los 27 sensores normalizados (z-score). Las perdidas fisicas operan
en unidades de ingenieria (convencion 6 del plan de Fase 4), asi que primero se
desnormaliza y despues cada componente del estado se reduce a la MEDIANA de los
canales de su rol en configs/sim_sensors.yaml, igual que hacen las reglas de
`data.simulation.physics_constraints` (votacion de redundantes).

El orden de los componentes lo fija `PINN_STATE_ORDER` y no se deduce de ningun
fichero: un test de contrato lo congela. La presion no es un componente del estado
(el TDD 5.2 no la usa en las perdidas), pero cp se evalua en (T_avg, P), asi que se
reduce aparte con el rol `P`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

PINN_STATE_ORDER: tuple[str, ...] = ("flux", "flow_in", "t_in", "t_out", "flow_out")
"""Componentes del estado fisico, en el orden de las columnas de `PhysicalState.values`.

flux [%] del flujo neutronico; flow_in y flow_out [kg/s] POR LAZO en la pierna fria
(entrada al nucleo) y en la caliente (salida); t_in y t_out [C] de las piernas fria
y caliente.
"""

STATE_ROLES: Mapping[str, str] = MappingProxyType(
    {
        "flux": "flux",
        "flow_in": "flow_cold_leg",
        "t_in": "T_in",
        "t_out": "T_out",
        "flow_out": "flow_hot_leg",
    }
)
"""Componente del estado -> rol de configs/sim_sensors.yaml del que sale."""

PRESSURE_ROLE = "P"
"""Rol de la presion del primario [bar] (cp se evalua en (T_avg, P))."""


@dataclass(frozen=True)
class StateIndex:
    """Columnas de la matriz de sensores que alimentan cada componente.

    Attributes:
        tags: Orden de sensores de la matriz (el del catalogo).
        components: Indices de columna de cada componente, en el orden de
            PINN_STATE_ORDER.
        pressure: Indices de columna del rol de presion.
    """

    tags: tuple[str, ...]
    components: tuple[tuple[int, ...], ...]
    pressure: tuple[int, ...]

    def channels(self, component: str) -> tuple[int, ...]:
        """Return the column indices of a state component.

        Args:
            component: Nombre de PINN_STATE_ORDER.

        Returns:
            The column indices, in role order.

        Raises:
            KeyError: If component is not in PINN_STATE_ORDER.
        """
        if component not in PINN_STATE_ORDER:
            raise KeyError(f"physics state: unknown component '{component}'.")
        return self.components[PINN_STATE_ORDER.index(component)]


def _role_indices(
    tags: Sequence[str], roles: Mapping[str, Sequence[str]], role: str
) -> tuple[int, ...]:
    """Return the column indices of a role, raising KeyError if unusable."""
    if role not in roles or len(roles[role]) == 0:
        raise KeyError(f"physics state: role '{role}' is missing or empty.")
    position = {tag: i for i, tag in enumerate(tags)}
    missing = [tag for tag in roles[role] if tag not in position]
    if missing:
        raise KeyError(f"physics state: role '{role}' needs sensors {missing}.")
    return tuple(position[tag] for tag in roles[role])


def build_state_index(tags: Sequence[str], roles: Mapping[str, Sequence[str]]) -> StateIndex:
    """Map the catalog roles to column indices of a sensor matrix.

    Args:
        tags: Orden de columnas de la matriz (SensorCatalog.tags o el orden de
            assignment.json).
        roles: Rol -> tags (SensorCatalog.roles).

    Returns:
        The StateIndex.

    Raises:
        ValueError: If tags has duplicates.
        KeyError: If a needed role is missing, empty or names a tag absent from
            tags.
    """
    if len(set(tags)) != len(tags):
        raise ValueError("physics state: the sensor order has duplicated tags.")
    components = tuple(_role_indices(tags, roles, STATE_ROLES[name]) for name in PINN_STATE_ORDER)
    return StateIndex(
        tags=tuple(tags),
        components=components,
        pressure=_role_indices(tags, roles, PRESSURE_ROLE),
    )


def _as_row(values: Sequence[float] | torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Return a 1-D tensor with the dtype and device of like."""
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def normalize(
    x: torch.Tensor, mean: Sequence[float] | torch.Tensor, std: Sequence[float] | torch.Tensor
) -> torch.Tensor:
    """Return (x - mean) / std per column.

    Args:
        x: Matriz (batch, n_sensors) en unidades de ingenieria.
        mean: Media por sensor (ZScoreStats.mean).
        std: Desviacion por sensor (ZScoreStats.std).

    Returns:
        The z-scores, same shape as x.
    """
    return (x - _as_row(mean, x)) / _as_row(std, x)


def denormalize(
    z: torch.Tensor, mean: Sequence[float] | torch.Tensor, std: Sequence[float] | torch.Tensor
) -> torch.Tensor:
    """Return z * std + mean per column (inverse of `normalize`).

    Args:
        z: Matriz (batch, n_sensors) en z-score.
        mean: Media por sensor.
        std: Desviacion por sensor.

    Returns:
        The values in engineering units; differentiable with respect to z.
    """
    return z * _as_row(std, z) + _as_row(mean, z)


def channel_median(x: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    """Return the median of some columns per row, as numpy/pandas define it.

    Con un numero par de canales es la media de los dos centrales (`torch.median`
    devolveria el inferior y no coincidiria con las reglas). Diferenciable: el
    gradiente va a los canales centrales.

    Args:
        x: Matriz (batch, n_sensors).
        indices: Columnas del rol (al menos una).

    Returns:
        Tensor (batch,).

    Raises:
        ValueError: If indices is empty.
    """
    if len(indices) == 0:
        raise ValueError("channel_median needs at least one channel.")
    ordered, _ = torch.sort(x[:, list(indices)], dim=1)
    k = ordered.shape[1]
    if k % 2 == 1:
        return ordered[:, k // 2]
    return 0.5 * (ordered[:, k // 2 - 1] + ordered[:, k // 2])


@dataclass(frozen=True)
class PhysicalState:
    """Estado fisico por fila, en unidades de ingenieria.

    Attributes:
        values: Tensor (batch, 5) con las columnas de PINN_STATE_ORDER.
        pressure_bar: Tensor (batch,) con la presion del primario.
    """

    values: torch.Tensor
    pressure_bar: torch.Tensor

    def component(self, name: str) -> torch.Tensor:
        """Return one column of the state by name (see PINN_STATE_ORDER)."""
        if name not in PINN_STATE_ORDER:
            raise KeyError(f"physics state: unknown component '{name}'.")
        return self.values[:, PINN_STATE_ORDER.index(name)]

    @property
    def t_avg(self) -> torch.Tensor:
        """Return (t_in + t_out) / 2 [C]."""
        return 0.5 * (self.component("t_in") + self.component("t_out"))


def physical_state(x: torch.Tensor, index: StateIndex) -> PhysicalState:
    """Reduce a sensor matrix in engineering units to the physical state.

    Args:
        x: Matriz (batch, n_sensors) desnormalizada, columnas en el orden de
            index.tags.
        index: Resultado de `build_state_index`.

    Returns:
        The PhysicalState (median of the channels of each role).

    Raises:
        ValueError: If x is not 2-D with len(index.tags) columns.
    """
    if x.ndim != 2 or x.shape[1] != len(index.tags):
        raise ValueError(
            f"physical_state expects (batch, {len(index.tags)}), got {tuple(x.shape)}."
        )
    values = torch.stack([channel_median(x, cols) for cols in index.components], dim=1)
    return PhysicalState(values=values, pressure_bar=channel_median(x, index.pressure))
