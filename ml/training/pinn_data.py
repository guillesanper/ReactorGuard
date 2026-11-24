"""Pares de entrenamiento de la PINN a partir de las particiones de prepare_sim (F4-M3).

Convierte un frame ancho de `data/processed/sim_pinn/<split>.parquet` en los tensores
que consume el loop de `ml/training/train.py`:

- entrada de cada par = sensores MEDIDOS en t en z-score, seguidos del contexto de
  operacion de 4 entradas (D4-2) en z-score;
- objetivo = sensores MEDIDOS en t+1 en z-score (D4-6);
- solo los pares con `trainable_pair` (ningun instante del par lleva etiqueta, D4-6);
- mascara cuasi-estacionaria de la perdida de masa con T_avg MEDIDA en t (entrada) y
  en t+1 (objetivo), la misma que la regla de masa aplica en t+1 (hallazgo M2-3).

Contexto (D4-2), en este orden (`CONTEXT_NAMES`):

- `power_level`: mediana de los canales del rol `P_th` / 100 (fraccion del nominal);
- `power_rate`: diferencia hacia atras de `power_level` entre t-1 y t, por hora
  (fraccion/h). Solo usa el pasado, como en inferencia; el primer instante de cada run
  no tiene t-1 y vale 0;
- `hour_sin`, `hour_cos`: seno y coseno de 2 pi (segundos del dia UTC) / 86400.

Las cuatro columnas se normalizan con media y desviacion de train, igual que los
sensores (`fit_normalization`), y ambos juegos de estadisticos viajan en el
checkpoint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from ml.models.physics_loss import PhysicsConstants, quasi_steady_mask
from ml.models.physics_state import StateIndex, physical_state
from ml.training.sim_dataset import (
    ID_COLUMNS,
    fit_zscore,
    true_columns,
    value_columns,
)

CONTEXT_NAMES: tuple[str, ...] = ("power_level", "power_rate", "hour_sin", "hour_cos")
"""Entradas de contexto (D4-2), en el orden en que se concatenan tras los sensores."""

POWER_ROLE = "P_th"
"""Rol de configs/sim_sensors.yaml del que sale `power_level`."""

_SECONDS_PER_DAY = 86_400.0
_SECONDS_PER_HOUR = 3_600.0
_PERCENT = 100.0

FRAME_COLUMNS: tuple[str, ...] = (*ID_COLUMNS, "labelled", "trainable_pair")
"""Columnas no de sensor que se leen de cada particion."""


def partition_columns(tags: Sequence[str]) -> list[str]:
    """Return the parquet columns the trainer reads for a sensor order."""
    return [*FRAME_COLUMNS, *value_columns(tags), *true_columns(tags)]


def _power_indices(tags: Sequence[str], roles: Mapping[str, Sequence[str]]) -> list[int]:
    """Return the columns of the P_th role, raising KeyError if unusable."""
    if POWER_ROLE not in roles or len(roles[POWER_ROLE]) == 0:
        raise KeyError(f"pinn data: role '{POWER_ROLE}' is missing or empty.")
    position = {tag: i for i, tag in enumerate(tags)}
    missing = [tag for tag in roles[POWER_ROLE] if tag not in position]
    if missing:
        raise KeyError(f"pinn data: role '{POWER_ROLE}' needs sensors {missing}.")
    return [position[tag] for tag in roles[POWER_ROLE]]


def _check_order(frame: pd.DataFrame) -> None:
    """Raise if the instants of a run are not consecutive and in order."""
    steps = frame["timestep"].to_numpy(dtype=np.int64)
    same_run = frame["run_id"].to_numpy()[1:] == frame["run_id"].to_numpy()[:-1]
    if np.any(same_run & (steps[1:] != steps[:-1] + 1)):
        raise ValueError("pinn data: the instants of a run must be consecutive and in order.")
    runs = frame["run_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ~same_run])
    if len(set(runs[starts])) != len(starts):
        raise ValueError("pinn data: the instants of a run must be contiguous in the frame.")


def _run_starts(frame: pd.DataFrame) -> NDArray[np.bool_]:
    """Return where each row is the first instant of its run in the frame."""
    runs = frame["run_id"].to_numpy()
    first = np.ones(len(frame), dtype=np.bool_)
    first[1:] = runs[1:] != runs[:-1]
    return first


def _seconds(frame: pd.DataFrame) -> NDArray[np.float64]:
    """Return the timestamps as UTC seconds since the epoch."""
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    nanoseconds: NDArray[np.int64] = stamps.as_unit("ns").asi8
    return nanoseconds.astype(np.float64) / 1.0e9


def context_features(
    frame: pd.DataFrame, tags: Sequence[str], roles: Mapping[str, Sequence[str]]
) -> NDArray[np.float64]:
    """Return the raw operating context of every instant (D4-2).

    Args:
        frame: Frame ancho de una particion (uno o varios runs, instantes de cada run
            consecutivos y en orden).
        tags: Orden de sensores de las columnas `value:`.
        roles: Rol -> tags (SensorCatalog.roles).

    Returns:
        Array (n, 4) with the columns of CONTEXT_NAMES, in engineering units.

    Raises:
        KeyError: If the P_th role is missing or names an unknown sensor.
        ValueError: If the instants of a run are not consecutive and in order.
    """
    _check_order(frame)
    values = frame[value_columns(tags)].to_numpy(dtype=np.float64)
    power = np.median(values[:, _power_indices(tags, roles)], axis=1) / _PERCENT
    seconds = _seconds(frame)
    rate = np.zeros(len(frame), dtype=np.float64)
    if len(frame) > 1:
        dt_h = (seconds[1:] - seconds[:-1]) / _SECONDS_PER_HOUR
        rate[1:] = np.divide(
            power[1:] - power[:-1], dt_h, out=np.zeros_like(dt_h), where=dt_h > 0.0
        )
    rate[_run_starts(frame)] = 0.0
    phase = 2.0 * math.pi * np.mod(seconds, _SECONDS_PER_DAY) / _SECONDS_PER_DAY
    features: NDArray[np.float64] = np.column_stack([power, rate, np.sin(phase), np.cos(phase)])
    return features


@dataclass(frozen=True)
class NormalizationStats:
    """Media y desviacion de train de los sensores y del contexto.

    Attributes:
        tags: Orden de sensores.
        sensor_mean: Media de cada sensor medido.
        sensor_std: Desviacion poblacional de cada sensor medido.
        context_mean: Media de cada entrada de CONTEXT_NAMES.
        context_std: Desviacion poblacional de cada entrada de contexto.
    """

    tags: tuple[str, ...]
    sensor_mean: tuple[float, ...]
    sensor_std: tuple[float, ...]
    context_mean: tuple[float, ...]
    context_std: tuple[float, ...]

    @property
    def input_mean(self) -> tuple[float, ...]:
        """Return the mean of the full input row (sensors, then context)."""
        return self.sensor_mean + self.context_mean

    @property
    def input_std(self) -> tuple[float, ...]:
        """Return the deviation of the full input row (sensors, then context)."""
        return self.sensor_std + self.context_std


def fit_normalization(
    frame: pd.DataFrame, tags: Sequence[str], roles: Mapping[str, Sequence[str]]
) -> NormalizationStats:
    """Fit the z-score of sensors and context on the unlabelled instants of train.

    Los sensores usan `fit_zscore` de F4-M1 (valor medido de los instantes sin
    etiqueta); el contexto, los mismos instantes.

    Args:
        frame: Frame ancho de train.
        tags: Orden de sensores.
        roles: Rol -> tags.

    Returns:
        The NormalizationStats.

    Raises:
        ValueError: If there is no unlabelled instant or a sensor or a context
            input is constant on them.
    """
    sensors = fit_zscore(frame, tags)
    unlabelled = ~frame["labelled"].to_numpy(dtype=np.bool_)
    context = context_features(frame, tags, roles)[unlabelled]
    mean = context.mean(axis=0)
    std = context.std(axis=0)
    flat = [name for name, s in zip(CONTEXT_NAMES, std, strict=True) if not s > 0.0]
    if flat:
        raise ValueError(f"fit_normalization: context inputs {flat} are constant.")
    return NormalizationStats(
        tags=sensors.tags,
        sensor_mean=sensors.mean,
        sensor_std=sensors.std,
        context_mean=tuple(float(m) for m in mean),
        context_std=tuple(float(s) for s in std),
    )


@dataclass(frozen=True)
class PairSet:
    """Pares (t, t+1) de una particion, listos para el loop.

    Attributes:
        inputs: Tensor (n, n_sensors + 4) float32: sensores medidos en t y
            contexto, en z-score.
        targets: Tensor (n, n_sensors) float32: sensores medidos en t+1, en z-score.
        true_targets: Tensor (n, n_sensors) float32: valor verdadero en t+1 en
            z-score (solo para la linea base de ruido).
        steady: Tensor (n,) bool: par cuasi-estacionario (cuenta en la masa).
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    true_targets: torch.Tensor
    steady: torch.Tensor

    def __len__(self) -> int:
        """Return the number of pairs."""
        return int(self.inputs.shape[0])

    @property
    def n_sensors(self) -> int:
        """Return the sensors per row of the targets."""
        return int(self.targets.shape[1])


def _zscore(
    values: NDArray[np.float64], mean: Sequence[float], std: Sequence[float]
) -> torch.Tensor:
    """Return (values - mean) / std as float32 (computed in float64)."""
    z = (values - np.asarray(mean, dtype=np.float64)) / np.asarray(std, dtype=np.float64)
    return torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32))


def build_pairs(
    frame: pd.DataFrame,
    stats: NormalizationStats,
    roles: Mapping[str, Sequence[str]],
    index: StateIndex,
    constants: PhysicsConstants,
) -> PairSet:
    """Build the trainable pairs of a partition frame (D4-6, M2-3).

    Args:
        frame: Frame ancho de una particion.
        stats: Estadisticos de train.
        roles: Rol -> tags (contexto).
        index: Columnas de cada componente del estado fisico (orden de stats.tags).
        constants: Escalas de planta (umbral cuasi-estacionario).

    Returns:
        The PairSet (possibly empty).

    Raises:
        ValueError: If the instants of a run are not consecutive and in order, the
            last instant of a run is marked trainable, or index does not follow
            stats.tags.
    """
    if index.tags != stats.tags:
        raise ValueError("build_pairs: the state index does not follow the sensor order.")
    _check_order(frame)
    tags = stats.tags
    trainable = frame["trainable_pair"].to_numpy(dtype=np.bool_)
    last_of_run = np.r_[_run_starts(frame)[1:], True]
    if np.any(trainable & last_of_run):
        raise ValueError("build_pairs: the last instant of a run cannot start a pair.")
    origin = np.flatnonzero(trainable)
    target = origin + 1

    measured = frame[value_columns(tags)].to_numpy(dtype=np.float64)
    true = frame[true_columns(tags)].to_numpy(dtype=np.float64)
    context = context_features(frame, tags, roles)
    raw_inputs = np.concatenate([measured[origin], context[origin]], axis=1)

    seconds = _seconds(frame)
    dt_s = seconds[target] - seconds[origin]
    if len(origin) and not np.all(dt_s == dt_s[0]):
        raise ValueError("build_pairs: the pairs of a partition must share one cadence.")
    steady = torch.zeros(len(origin), dtype=torch.bool)
    if len(origin):
        t_from = physical_state(torch.from_numpy(measured[origin]), index).t_avg
        t_to = physical_state(torch.from_numpy(measured[target]), index).t_avg
        steady = quasi_steady_mask(t_from, t_to, float(dt_s[0]), constants)
    return PairSet(
        inputs=_zscore(raw_inputs, stats.input_mean, stats.input_std),
        targets=_zscore(measured[target], stats.sensor_mean, stats.sensor_std),
        true_targets=_zscore(true[target], stats.sensor_mean, stats.sensor_std),
        steady=steady,
    )


def persistence_mse(pairs: PairSet) -> float:
    """Return the MSE of predicting x(t+1) = x(t) in z-score (seccion 2.3 del plan)."""
    if len(pairs) == 0:
        return math.nan
    diff = pairs.inputs[:, : pairs.n_sensors].double() - pairs.targets.double()
    return float((diff * diff).mean().item())


def noise_mse(pairs: PairSet) -> float:
    """Return the MSE of the measurement noise value - true_value at t+1 in z-score."""
    if len(pairs) == 0:
        return math.nan
    diff = pairs.targets.double() - pairs.true_targets.double()
    return float((diff * diff).mean().item())
