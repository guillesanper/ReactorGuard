"""Cobertura de modos de fallo del dataset simulado (F3-M5, D6).

Un MODO de fallo es la terna (tipo, grupo de sensor, nivel de magnitud): el tipo
es uno de los 8 del TDD 2.2, el grupo es el `sensor_type` del contrato (los 6 de
`SensorType` presentes en el catalogo) y el nivel es `low`, `medium` o `high`. La
rejilla es el producto de las tres. Un modo esta CUBIERTO si lleva al menos
`k_min_readings` lecturas etiquetadas repartidas en al menos `n_min_episodes`
episodios.

El gate de D6 es doble: los 8 tipos presentes y al menos el 95 % de la rejilla
cubierta. Se mide sobre el dataset entero y tambien por particion, con la misma
regla que el stage `split` de Fase 2: el corte es TEMPORAL DENTRO DE CADA TIPO de
escenario (los primeros pasos a train, los siguientes a val y los ultimos a test;
el sobrante de la division entera va a test), de modo que un tipo con pocos runs
solo llega a test con los episodios de su ultimo tramo.

Una lectura etiquetada es la de un sensor dentro de la ventana de un episodio. El
modo de una lectura de un fallo de proceso toma el grupo del sensor que la emite:
un fallo de proceso etiqueta todos los grupos, asi que una celda de proceso queda
cubierta con un unico episodio de su nivel.

Este modulo es de produccion y solo ve frames largos (`dataset_builder`): no
importa nada de `data/validation`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from data.simulation.scenario import FAULT_KINDS, FaultLevel, RunPlan
from data.simulation.sensor_catalog import SensorCatalog

_LOG = logging.getLogger(__name__)

PARTITIONS: tuple[str, ...] = ("train", "val", "test")
"""Orden temporal de las particiones: train es el pasado, test el futuro."""

DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
"""Proporciones de `training:` en params.yaml."""

LEVELS: tuple[FaultLevel, ...] = (FaultLevel.LOW, FaultLevel.MEDIUM, FaultLevel.HIGH)

_NONE = "none"
_COLUMNS = (
    "run_id",
    "timestep",
    "sensor_type",
    "kind",
    "level",
    "episode_id",
    "corrupted",
)


@dataclass(frozen=True, order=True)
class FaultMode:
    """Una celda de la rejilla de cobertura.

    Attributes:
        kind: Tipo de fallo (valor de `ScenarioKind`).
        group: Grupo de sensor (valor de `SensorType`).
        level: Nivel de magnitud (valor de `FaultLevel`).
    """

    kind: str
    group: str
    level: str

    @property
    def label(self) -> str:
        """Return `<tipo>/<grupo>/<nivel>`."""
        return f"{self.kind}/{self.group}/{self.level}"


@dataclass(frozen=True)
class ModeStats:
    """Lo que un modo lleva en el dataset (o en una particion).

    Attributes:
        readings: Lecturas etiquetadas.
        episodes: Episodios distintos (`episode_id`).
        runs: Runs distintos.
        corrupted: Lecturas cuyo valor difiere de la medida sin fallo.
    """

    readings: int = 0
    episodes: int = 0
    runs: int = 0
    corrupted: int = 0


@dataclass(frozen=True)
class CoverageScope:
    """Cobertura de un conjunto de lecturas (el dataset o una particion).

    Attributes:
        name: `all`, `train`, `val` o `test`.
        n_steps: Pasos de tiempo del conjunto (suma sobre los tipos).
        labelled_readings: Lecturas etiquetadas.
        types_present: Tipos con al menos una lectura etiquetada.
        covered: Modos cubiertos segun (k_min, n_min).
        missing: Modos de la rejilla sin cubrir.
        stats: Estadisticos de cada modo con alguna lectura.
        grid_size: Celdas de la rejilla.
    """

    name: str
    n_steps: int
    labelled_readings: int
    types_present: tuple[str, ...]
    covered: tuple[FaultMode, ...]
    missing: tuple[FaultMode, ...]
    stats: Mapping[FaultMode, ModeStats]
    grid_size: int

    @property
    def n_types(self) -> int:
        """Return how many of the 8 fault types are present."""
        return len(self.types_present)

    @property
    def grid_fraction(self) -> float:
        """Return the covered share of the grid."""
        return len(self.covered) / self.grid_size

    def type_cells(self, kind: str) -> tuple[int, int]:
        """Return (covered, total) cells of one fault type."""
        total = sum(1 for m in (*self.covered, *self.missing) if m.kind == kind)
        covered = sum(1 for m in self.covered if m.kind == kind)
        return covered, total


@dataclass(frozen=True)
class CoverageReport:
    """Cobertura del dataset entero y por particion.

    Attributes:
        k_min_readings: Lecturas minimas para cubrir un modo.
        n_min_episodes: Episodios minimos para cubrir un modo.
        gate: Fraccion minima de la rejilla (D6).
        ratios: Proporciones (train, val, test) de las particiones.
        overall: Cobertura del dataset entero.
        partitions: Cobertura de cada particion.
    """

    k_min_readings: int
    n_min_episodes: int
    gate: float
    ratios: tuple[float, float, float]
    overall: CoverageScope
    partitions: Mapping[str, CoverageScope] = field(default_factory=dict)

    @property
    def types_ok(self) -> bool:
        """Return whether all 8 fault types are present in the dataset."""
        return self.overall.n_types == len(FAULT_KINDS)

    @property
    def grid_ok(self) -> bool:
        """Return whether the grid coverage reaches the gate."""
        return self.overall.grid_fraction >= self.gate

    @property
    def passed(self) -> bool:
        """Return whether the D6 gate (8/8 types and >= gate of the grid) holds."""
        return self.types_ok and self.grid_ok


def partition_bounds(n_steps: int, ratios: Sequence[float] = DEFAULT_RATIOS) -> tuple[int, int]:
    """Return (n_train, n_train + n_val): the first step of val and of test.

    Misma aritmetica que el stage `split` de Fase 2: division entera para train
    y val y el sobrante a test.

    Args:
        n_steps: Pasos de tiempo del tipo (todos sus runs, uno tras otro).
        ratios: Proporciones (train, val, test); suman 1.

    Returns:
        The two boundaries.

    Raises:
        ValueError: If the ratios are not three positive numbers summing to 1.
    """
    if len(ratios) != 3 or any(r <= 0.0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must be three positive numbers summing to 1, got {ratios}.")
    n_train = int(n_steps * ratios[0])
    n_val = int(n_steps * ratios[1])
    return n_train, n_train + n_val


def partition_of(index: int, n_steps: int, ratios: Sequence[float] = DEFAULT_RATIOS) -> str:
    """Return the partition of a time step of a fault type.

    Args:
        index: Posicion del paso en la linea temporal del tipo.
        n_steps: Pasos de tiempo del tipo.
        ratios: Proporciones (train, val, test).

    Returns:
        `train`, `val` or `test`.

    Raises:
        ValueError: If index is outside [0, n_steps) or the ratios are invalid.
    """
    if not 0 <= index < n_steps:
        raise ValueError(f"index {index} is outside [0, {n_steps}).")
    first_val, first_test = partition_bounds(n_steps, ratios)
    if index < first_val:
        return "train"
    return "val" if index < first_test else "test"


def grid(groups: Iterable[str]) -> list[FaultMode]:
    """Return the full grid: 8 types x groups x 3 levels.

    Args:
        groups: Grupos de sensor presentes en el catalogo.

    Returns:
        The modes, sorted.
    """
    names = sorted(set(groups))
    return sorted(
        FaultMode(kind=k.value, group=g, level=lv.value)
        for k in FAULT_KINDS
        for g in names
        for lv in LEVELS
    )


class CoverageCollector:
    """Acumula runs y calcula la cobertura del dataset y de sus particiones.

    Un run se anade como frame largo; solo se guardan sus filas etiquetadas (el
    resto del frame no hace falta), de modo que el dataset entero cabe en memoria.
    """

    def __init__(self) -> None:
        """Create an empty collector."""
        self._labelled: list[pd.DataFrame] = []
        self._runs: dict[str, tuple[str, int]] = {}
        self._groups: set[str] = set()

    def add_frame(self, frame: pd.DataFrame) -> None:
        """Add the long frame of ONE run.

        Args:
            frame: Frame largo de un run (`dataset_builder.LONG_COLUMNS`).

        Raises:
            ValueError: If the frame holds more than one run or the run was
                already added.
        """
        runs = frame["run_id"].unique()
        if len(runs) != 1:
            raise ValueError(f"add_frame needs exactly one run per frame, got {len(runs)}.")
        run_id = str(runs[0])
        if run_id in self._runs:
            raise ValueError(f"run {run_id} was already added to the coverage collector.")
        scenario = str(frame["scenario"].iloc[0])
        self._runs[run_id] = (scenario, int(frame["timestep"].max()) + 1)
        self._groups.update(str(g) for g in frame["sensor_type"].unique())
        process = frame["process_fault"].astype(str)
        instrument = frame["instrument_fault"].astype(str)
        mask = (process != _NONE) | (instrument != _NONE)
        if not mask.any():
            return
        rows = frame.loc[mask]
        kind = process[mask].where(process[mask] != _NONE, instrument[mask])
        table = pd.DataFrame(
            {
                "run_id": rows["run_id"].astype(str).to_numpy(),
                "timestep": rows["timestep"].to_numpy(),
                "sensor_type": rows["sensor_type"].astype(str).to_numpy(),
                "kind": kind.to_numpy(),
                "level": rows["fault_level"].astype(str).to_numpy(),
                "episode_id": rows["episode_id"].astype(str).to_numpy(),
                "corrupted": rows["corrupted"].to_numpy(),
            },
            columns=list(_COLUMNS),
        )
        self._labelled.append(table)

    def add_plan(self, plan: RunPlan, catalog: SensorCatalog) -> None:
        """Add the labels a RunPlan will produce, without simulating the run.

        Las etiquetas de un run dependen solo de su plan (ventana y nivel del
        fallo de proceso en todos los sensores; episodios de instrumento en su
        sensor), no de los valores simulados: la cobertura de un dataset
        planificado se mide sin integrar el modelo. Un test comprueba que da la
        misma tabla que el frame del run. `corrupted` queda en cero: solo el frame
        sabe que muestras del intermitente cayeron en estado invalido.

        Args:
            plan: Plan del run.
            catalog: Catalogo de sensores (da el grupo de cada tag).

        Raises:
            ValueError: If the run was already added.
            KeyError: If an episode names a sensor missing from the catalog.
        """
        if plan.run_id in self._runs:
            raise ValueError(f"run {plan.run_id} was already added to the coverage collector.")
        self._runs[plan.run_id] = (plan.scenario.value, plan.n_steps)
        group_of = {s.tag: s.sensor_type.value for s in catalog.sensors}
        self._groups.update(group_of.values())
        parts: list[pd.DataFrame] = []
        spec = plan.process_fault
        if spec is not None:
            last = plan.n_steps - 1 if spec.last_active_step is None else spec.last_active_step
            steps = np.arange(spec.injected_step, last + 1)
            tags = list(group_of.values())
            parts.append(
                self._rows(
                    plan.run_id,
                    np.repeat(steps, len(tags)),
                    np.tile(np.array(tags, dtype=object), len(steps)),
                    plan.scenario.value,
                    spec.level.value,
                    f"{plan.run_id}:{plan.scenario.value}",
                )
            )
        for episode in plan.instrument_faults:
            if episode.sensor_id not in group_of:
                raise KeyError(f"{plan.run_id}: unknown sensor '{episode.sensor_id}'.")
            steps = np.arange(episode.start_step, episode.last_step + 1)
            parts.append(
                self._rows(
                    plan.run_id,
                    steps,
                    np.full(len(steps), group_of[episode.sensor_id], dtype=object),
                    plan.scenario.value,
                    episode.level.value,
                    episode.episode_id(plan.run_id),
                )
            )
        self._labelled.extend(parts)

    @staticmethod
    def _rows(
        run_id: str,
        steps: NDArray[np.int64],
        groups: NDArray[np.object_],
        kind: str,
        level: str,
        episode_id: str,
    ) -> pd.DataFrame:
        """Return the label rows of one episode (no corruption information)."""
        n = len(steps)
        return pd.DataFrame(
            {
                "run_id": np.full(n, run_id, dtype=object),
                "timestep": steps,
                "sensor_type": groups,
                "kind": np.full(n, kind, dtype=object),
                "level": np.full(n, level, dtype=object),
                "episode_id": np.full(n, episode_id, dtype=object),
                "corrupted": np.zeros(n, dtype=np.bool_),
            },
            columns=list(_COLUMNS),
        )

    @property
    def n_runs(self) -> int:
        """Return the runs added so far."""
        return len(self._runs)

    def _offsets(self) -> tuple[dict[str, int], dict[str, int]]:
        """Return (run -> first index in its type's timeline, type -> total steps)."""
        offsets: dict[str, int] = {}
        totals: dict[str, int] = {}
        for run_id in sorted(self._runs, key=lambda r: (len(r), r)):
            scenario, n_steps = self._runs[run_id]
            offsets[run_id] = totals.get(scenario, 0)
            totals[scenario] = totals.get(scenario, 0) + n_steps
        return offsets, totals

    def report(
        self,
        k_min_readings: int,
        n_min_episodes: int,
        gate: float,
        ratios: Sequence[float] = DEFAULT_RATIOS,
        groups: Iterable[str] | None = None,
    ) -> CoverageReport:
        """Compute the coverage of the whole dataset and of each partition.

        Args:
            k_min_readings: Lecturas minimas para cubrir un modo.
            n_min_episodes: Episodios minimos para cubrir un modo.
            gate: Fraccion minima de la rejilla (D6).
            ratios: Proporciones (train, val, test).
            groups: Grupos de sensor de la rejilla; por defecto los vistos en los
                frames anadidos.

        Returns:
            The CoverageReport.

        Raises:
            ValueError: If no run was added or a threshold is out of range.
        """
        if not self._runs:
            raise ValueError("coverage report needs at least one run.")
        if k_min_readings < 1 or n_min_episodes < 1:
            raise ValueError("k_min_readings and n_min_episodes must be >= 1.")
        if not 0.0 < gate <= 1.0:
            raise ValueError(f"gate must be in (0, 1], got {gate}.")
        bounds_ratios = tuple(float(r) for r in ratios)
        partition_bounds(1, bounds_ratios)  # validates the ratios
        names = sorted(set(groups) if groups is not None else self._groups)
        modes = grid(names)
        table = (
            pd.concat(self._labelled, ignore_index=True)
            if self._labelled
            else pd.DataFrame(columns=list(_COLUMNS))
        )
        offsets, totals = self._offsets()
        if len(table):
            position = table["timestep"].to_numpy() + table["run_id"].map(offsets).to_numpy()
            scenario_steps = table["kind"].map(totals).to_numpy()
            first_val = (scenario_steps * bounds_ratios[0]).astype(int)
            first_test = first_val + (scenario_steps * bounds_ratios[1]).astype(int)
            part = pd.Series("train", index=table.index, dtype=object)
            part[position >= first_val] = "val"
            part[position >= first_test] = "test"
            table = table.assign(partition=part)
        else:
            table = table.assign(partition=pd.Series(dtype=object))

        def scope(name: str, subset: pd.DataFrame, n_steps: int) -> CoverageScope:
            stats: dict[FaultMode, ModeStats] = {}
            if len(subset):
                grouped = subset.groupby(["kind", "sensor_type", "level"], sort=True)
                for (kind, group, level), rows in grouped:
                    stats[FaultMode(str(kind), str(group), str(level))] = ModeStats(
                        readings=len(rows),
                        episodes=int(rows["episode_id"].nunique()),
                        runs=int(rows["run_id"].nunique()),
                        corrupted=int(rows["corrupted"].sum()),
                    )
            covered = tuple(
                m
                for m in modes
                if m in stats
                and stats[m].readings >= k_min_readings
                and stats[m].episodes >= n_min_episodes
            )
            covered_set = set(covered)
            present = tuple(sorted({str(k) for k in subset["kind"].unique()}))
            return CoverageScope(
                name=name,
                n_steps=n_steps,
                labelled_readings=len(subset),
                types_present=tuple(k for k in (x.value for x in FAULT_KINDS) if k in present),
                covered=covered,
                missing=tuple(m for m in modes if m not in covered_set),
                stats=stats,
                grid_size=len(modes),
            )

        overall = scope("all", table, sum(totals.values()))
        partitions: dict[str, CoverageScope] = {}
        for name in PARTITIONS:
            subset = table[table["partition"] == name]
            steps = 0
            for total in totals.values():
                first_val, first_test = partition_bounds(total, bounds_ratios)
                steps += {
                    "train": first_val,
                    "val": first_test - first_val,
                    "test": total - first_test,
                }[name]
            partitions[name] = scope(name, subset, steps)
        return CoverageReport(
            k_min_readings=k_min_readings,
            n_min_episodes=n_min_episodes,
            gate=gate,
            ratios=(bounds_ratios[0], bounds_ratios[1], bounds_ratios[2]),
            overall=overall,
            partitions=partitions,
        )


def coverage_report(
    frames: Iterable[pd.DataFrame],
    k_min_readings: int,
    n_min_episodes: int,
    gate: float,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    groups: Iterable[str] | None = None,
) -> CoverageReport:
    """Return the coverage of a dataset given as an iterable of run frames.

    Args:
        frames: Un frame largo por run, p. ej. `ReactorSimulator.iter_runs`.
        k_min_readings: Lecturas minimas para cubrir un modo.
        n_min_episodes: Episodios minimos para cubrir un modo.
        gate: Fraccion minima de la rejilla (D6).
        ratios: Proporciones (train, val, test).
        groups: Grupos de sensor de la rejilla (por defecto, los vistos).

    Returns:
        The CoverageReport.

    Raises:
        ValueError: If `frames` is empty or a frame holds more than one run.
    """
    collector = CoverageCollector()
    for frame in frames:
        collector.add_frame(frame)
    return collector.report(k_min_readings, n_min_episodes, gate, ratios, groups)
