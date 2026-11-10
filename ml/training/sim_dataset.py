"""Datos de entrenamiento de la PINN a partir del dataset simulado (F4-M1, D4-6 a D4-8).

Convierte los runs del stage `simulate` (formato largo, una fila por lectura) en
frames anchos con una fila por instante, el formato que consume la PINN:

- una columna `value:<tag>` y otra `true:<tag>` por sensor, en el ORDEN DEL
  CATALOGO (`configs/sim_sensors.yaml`), nunca en el del parquet;
- la identidad del instante (`run_id`, `scenario`, `timestep`, `timestamp`);
- las etiquetas agregadas por instante (`INSTANT_LABEL_COLUMNS`): si alguna
  lectura lleva fallo, la severidad maxima, el tipo por eje, el nivel maximo, las
  lecturas etiquetadas y los episodios activos;
- `trainable_pair`: si el par (t, t+1) del mismo run sirve para entrenar (D4-6:
  ninguno de los dos instantes lleva etiqueta).

Los episodios de instrumento se SOLAPAN en sensores distintos (8.160 instantes del
dataset tienen mas de uno activo, F4-M1), asi que un unico `episode_id` por instante
perderia informacion: se guardan todos, ordenados y unidos por `EPISODE_SEPARATOR`.

El reparto es POR RUN y estratificado por escenario (D4-7): los ultimos runs de
cada escenario por `run_id` van a test, los anteriores a validacion y el resto a
train. Un run entero cae en una sola particion, de modo que no hay fuga temporal
ni de episodio entre particiones. La cobertura de la rejilla de modos de D6 por
particion se informa (no es criterio de F4).

Este modulo solo lee: no importa nada de `data/validation` ni escribe en el
dataset de origen. Las salidas no llevan reloj ni rutas absolutas.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.simulation.coverage import LEVELS, PARTITIONS, CoverageCollector, CoverageScope
from data.simulation.dataset_builder import (
    MANIFEST_NAME,
    MANIFEST_VERSION,
    NO_FAULT,
    file_sha256,
    labelled_mask,
)
from data.simulation.scenario import FAULT_KINDS, FAULT_SEVERITY, FaultSeverity, ScenarioKind
from data.simulation.sensor_catalog import SensorCatalog

_LOG = logging.getLogger(__name__)

SPLITS: tuple[str, ...] = PARTITIONS
"""Particiones, en orden: train, val, test."""

ID_COLUMNS: tuple[str, ...] = ("run_id", "scenario", "timestep", "timestamp")
"""Identidad de un instante."""

INSTANT_LABEL_COLUMNS: tuple[str, ...] = (
    "labelled",
    "severity",
    "process_fault",
    "instrument_fault",
    "fault_level",
    "labelled_sensors",
    "n_episodes",
    "episode_ids",
    "trainable_pair",
)
"""Etiquetas agregadas por instante (ver `run_instants`)."""

VALUE_PREFIX = "value:"
"""Prefijo de la columna del valor medido de un sensor."""

TRUE_PREFIX = "true:"
"""Prefijo de la columna del valor verdadero de un sensor."""

EPISODE_SEPARATOR = "|"
"""Separador de los episodios activos de un instante en `episode_ids`."""

CRITICAL = FaultSeverity.CRITICAL.value
"""Severidad que cuenta para el recall CRITICAL (D4-10)."""

CRITICAL_KINDS: tuple[ScenarioKind, ...] = tuple(
    kind for kind in FAULT_KINDS if FAULT_SEVERITY[kind] is FaultSeverity.CRITICAL
)
"""Tipos CRITICAL de la tabla estatica TDD 2.2 (D4-10): bloqueado, excursion y vacio."""

ASSIGNMENT_NAME = "assignment.json"
"""Asignacion de runs a particiones y orden de sensores, junto a los parquet."""

OUTPUT_VERSION = 1
"""Version del formato de los parquet, de `assignment.json` y de las metricas."""

_READ_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "sensor_id",
    "sensor_type",
    "value",
    "timestep",
    "run_id",
    "scenario",
    "true_value",
    "corrupted",
    "process_fault",
    "process_severity",
    "instrument_fault",
    "instrument_severity",
    "episode_id",
    "fault_level",
)
"""Columnas del frame largo que hacen falta (H12: el frame entero ocupa 1,24 GB)."""

_SEVERITY_RANK: Mapping[str, int] = {
    severity.value: rank for rank, severity in enumerate(reversed(FaultSeverity))
}
"""LOW = 0 ... CRITICAL = 3."""

_LEVEL_RANK: Mapping[str, int] = {level.value: rank for rank, level in enumerate(LEVELS)}
"""low = 0, medium = 1, high = 2."""


def value_columns(tags: Iterable[str]) -> list[str]:
    """Return the measured-value column of each tag, in the given order."""
    return [f"{VALUE_PREFIX}{tag}" for tag in tags]


def true_columns(tags: Iterable[str]) -> list[str]:
    """Return the true-value column of each tag, in the given order."""
    return [f"{TRUE_PREFIX}{tag}" for tag in tags]


def instant_columns(tags: Sequence[str]) -> list[str]:
    """Return the full column order of a wide instant frame.

    Args:
        tags: Tags del catalogo, en su orden.

    Returns:
        ID_COLUMNS, then the value columns, the true-value columns and the labels.
    """
    return [*ID_COLUMNS, *value_columns(tags), *true_columns(tags), *INSTANT_LABEL_COLUMNS]


def _run_key(run_id: str) -> tuple[int, str]:
    """Return the sort key of a run id (numeric order for zero-padded ids)."""
    return len(run_id), run_id


# ----------------------------------------------------------------------
# Manifiesto del dataset de origen
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestRun:
    """Un run del manifiesto del stage `simulate`.

    Attributes:
        run_id: Identificador del run.
        scenario: Escenario del run.
        path: Ruta del parquet relativa a la raiz del dataset.
        sha256: Digest SHA-256 del parquet.
    """

    run_id: str
    scenario: ScenarioKind
    path: Path
    sha256: str


def read_manifest(source_dir: str | Path) -> tuple[ManifestRun, ...]:
    """Read the manifest of the simulated dataset.

    Args:
        source_dir: Raiz del dataset (`simulation.output_dir`).

    Returns:
        The runs, sorted by run id.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        KeyError: If a run entry lacks a required field.
        ValueError: If the manifest version is not supported, it lists no run,
            a scenario is unknown or a run id is repeated.
    """
    path = Path(source_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run the simulate stage before prepare_sim."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    version = document.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"{path}: manifest version {version!r} is not supported (expected {MANIFEST_VERSION})."
        )
    runs: list[ManifestRun] = []
    seen: set[str] = set()
    for entry in document.get("runs", []):
        for key in ("run_id", "scenario", "path", "sha256"):
            if key not in entry:
                raise KeyError(f"{path}: a run entry lacks the field '{key}'.")
        run_id = str(entry["run_id"])
        if run_id in seen:
            raise ValueError(f"{path}: run {run_id} appears twice.")
        seen.add(run_id)
        try:
            scenario = ScenarioKind(entry["scenario"])
        except ValueError as exc:
            raise ValueError(
                f"{path}: run {run_id} has unknown scenario {entry['scenario']!r}."
            ) from exc
        runs.append(
            ManifestRun(
                run_id=run_id,
                scenario=scenario,
                path=Path(str(entry["path"])),
                sha256=str(entry["sha256"]),
            )
        )
    if not runs:
        raise ValueError(f"{path}: the manifest lists no run.")
    return tuple(sorted(runs, key=lambda r: _run_key(r.run_id)))


def verified_run_path(source_dir: str | Path, run: ManifestRun) -> Path:
    """Return the parquet of a run after checking it against the manifest digest.

    Args:
        source_dir: Raiz del dataset.
        run: Entrada del manifiesto.

    Returns:
        The path of the parquet, `source_dir / run.path`.

    Raises:
        FileNotFoundError: If the parquet does not exist.
        ValueError: If its SHA-256 differs from the manifest.
    """
    path = Path(source_dir) / run.path
    if not path.exists():
        raise FileNotFoundError(f"Run {run.run_id}: parquet not found at {path}.")
    digest = file_sha256(path)
    if digest != run.sha256:
        raise ValueError(
            f"Run {run.run_id}: {path.name} has sha256 {digest}, the manifest says "
            f"{run.sha256}. El dataset no es el que describe su manifiesto."
        )
    return path


# ----------------------------------------------------------------------
# Reparto por run (D4-7)
# ----------------------------------------------------------------------


def assign_splits(
    runs: Sequence[ManifestRun], val_per_scenario: int, test_per_scenario: int
) -> dict[str, str]:
    """Assign whole runs to partitions, stratified by scenario (D4-7).

    Dentro de cada escenario los runs se ordenan por `run_id`: los
    `test_per_scenario` ultimos van a test, los `val_per_scenario` anteriores a
    validacion y el resto a train.

    Args:
        runs: Runs del dataset.
        val_per_scenario: Runs de cada escenario para validacion (>= 1).
        test_per_scenario: Runs de cada escenario para test (>= 1).

    Returns:
        Mapping run id -> partition.

    Raises:
        ValueError: If a count is below one or a scenario has too few runs to
            keep at least one run in train.
    """
    if val_per_scenario < 1 or test_per_scenario < 1:
        raise ValueError(
            f"val_per_scenario and test_per_scenario must be >= 1, got {val_per_scenario} "
            f"and {test_per_scenario}."
        )
    by_scenario: dict[ScenarioKind, list[str]] = defaultdict(list)
    for run in runs:
        by_scenario[run.scenario].append(run.run_id)
    needed = val_per_scenario + test_per_scenario + 1
    assignment: dict[str, str] = {}
    for scenario in sorted(by_scenario):
        ids = sorted(by_scenario[scenario], key=_run_key)
        if len(ids) < needed:
            raise ValueError(
                f"scenario {scenario.value} has {len(ids)} runs; the split needs at least "
                f"{needed} ({val_per_scenario} val + {test_per_scenario} test + 1 train)."
            )
        first_val = len(ids) - val_per_scenario - test_per_scenario
        first_test = len(ids) - test_per_scenario
        for position, run_id in enumerate(ids):
            if position < first_val:
                assignment[run_id] = "train"
            elif position < first_test:
                assignment[run_id] = "val"
            else:
                assignment[run_id] = "test"
    return assignment


# ----------------------------------------------------------------------
# Un run: frame largo -> frame ancho por instante
# ----------------------------------------------------------------------


def load_run_frame(path: str | Path) -> pd.DataFrame:
    """Read the columns of a run parquet that the PINN data needs.

    Args:
        path: Parquet de un run.

    Returns:
        The long frame restricted to the needed columns.
    """
    return pd.read_parquet(path, columns=list(_READ_COLUMNS))


def _axis_kind(frame: pd.DataFrame, column: str, run_id: str) -> str:
    """Return the single fault kind a run carries on one axis (NO_FAULT if none).

    Raises:
        ValueError: If the run carries more than one kind on the axis (D3).
    """
    kinds = sorted({str(k) for k in frame[column].unique()} - {NO_FAULT})
    if len(kinds) > 1:
        raise ValueError(f"run {run_id} carries several {column} kinds: {kinds}.")
    return kinds[0] if kinds else NO_FAULT


def _check_run_shape(frame: pd.DataFrame, catalog: SensorCatalog) -> tuple[str, int]:
    """Validate that a long frame is one complete run of the catalog.

    Returns:
        (run id, number of steps).

    Raises:
        ValueError: If the frame is empty, holds several runs, its sensors differ
            from the catalog, a (timestep, sensor) pair is missing or repeated,
            the time steps are not 0..n-1 or a value is not finite.
    """
    if frame.empty:
        raise ValueError("run_instants needs a non-empty frame.")
    runs = frame["run_id"].unique()
    if len(runs) != 1:
        raise ValueError(f"run_instants needs exactly one run per frame, got {len(runs)}.")
    run_id = str(runs[0])
    tags = set(catalog.tags)
    seen = {str(t) for t in frame["sensor_id"].unique()}
    if seen != tags:
        missing = sorted(tags - seen)
        extra = sorted(seen - tags)
        raise ValueError(
            f"run {run_id}: sensors differ from the catalog (missing {missing}, extra {extra})."
        )
    steps = frame["timestep"].to_numpy()
    n_steps = int(steps.max()) + 1
    if int(steps.min()) != 0 or len(frame) != n_steps * len(tags):
        raise ValueError(
            f"run {run_id}: expected {n_steps} steps x {len(tags)} sensors = "
            f"{n_steps * len(tags)} readings from step 0, got {len(frame)} readings."
        )
    if frame.duplicated(["timestep", "sensor_id"]).any():
        raise ValueError(f"run {run_id}: a (timestep, sensor) reading is repeated.")
    for column in ("value", "true_value"):
        if not np.isfinite(frame[column].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"run {run_id}: column {column} holds non-finite values.")
    return run_id, n_steps


def run_instants(frame: pd.DataFrame, catalog: SensorCatalog) -> pd.DataFrame:
    """Turn the long frame of ONE run into its wide frame, one row per instant.

    Args:
        frame: Frame largo de un run (al menos las columnas de `load_run_frame`).
        catalog: Catalogo de sensores; fija el orden de las columnas.

    Returns:
        DataFrame with `instant_columns(catalog.tags)`, one row per time step in
        order. `severity`, `fault_level` and `episode_ids` are null on unlabelled
        instants; `process_fault` and `instrument_fault` are NO_FAULT there.

    Raises:
        ValueError: If the frame is not one complete run of the catalog or it
            carries more than one fault kind on an axis.
    """
    run_id, n_steps = _check_run_shape(frame, catalog)
    tags = list(catalog.tags)
    steps = pd.RangeIndex(n_steps, name="timestep")

    measured = frame.pivot(index="timestep", columns="sensor_id", values="value")
    true = frame.pivot(index="timestep", columns="sensor_id", values="true_value")
    timestamps = frame.drop_duplicates("timestep").set_index("timestep")["timestamp"]

    labelled_rows = frame.loc[labelled_mask(frame)]
    by_step = labelled_rows.groupby("timestep")
    labelled_sensors = by_step.size().reindex(steps, fill_value=0).to_numpy(dtype=np.int64)
    labelled = labelled_sensors > 0

    severity_rank = (
        labelled_rows["process_severity"]
        .fillna(labelled_rows["instrument_severity"])
        .astype(str)
        .map(_SEVERITY_RANK)
    )
    level_rank = labelled_rows["fault_level"].astype(str).map(_LEVEL_RANK)
    severity_names = {rank: name for name, rank in _SEVERITY_RANK.items()}
    level_names = {rank: name for name, rank in _LEVEL_RANK.items()}
    severity = (
        severity_rank.groupby(labelled_rows["timestep"]).max().reindex(steps).map(severity_names)
    )
    level = level_rank.groupby(labelled_rows["timestep"]).max().reindex(steps).map(level_names)

    process_kind = _axis_kind(frame, "process_fault", run_id)
    instrument_kind = _axis_kind(frame, "instrument_fault", run_id)
    process_steps = labelled_rows.loc[
        labelled_rows["process_fault"].astype(str) != NO_FAULT, "timestep"
    ].unique()
    instrument_steps = labelled_rows.loc[
        labelled_rows["instrument_fault"].astype(str) != NO_FAULT, "timestep"
    ].unique()
    process = np.full(n_steps, NO_FAULT, dtype=object)
    process[process_steps] = process_kind
    instrument = np.full(n_steps, NO_FAULT, dtype=object)
    instrument[instrument_steps] = instrument_kind

    episodes = (
        labelled_rows[["timestep", "episode_id"]]
        .astype({"episode_id": str})
        .drop_duplicates()
        .sort_values(["timestep", "episode_id"])
        .groupby("timestep")["episode_id"]
    )
    n_episodes = episodes.size().reindex(steps, fill_value=0).to_numpy(dtype=np.int64)
    episode_ids = episodes.agg(EPISODE_SEPARATOR.join).reindex(steps)

    trainable = np.zeros(n_steps, dtype=np.bool_)
    trainable[:-1] = ~labelled[:-1] & ~labelled[1:]

    columns: dict[str, Any] = {
        "run_id": np.full(n_steps, run_id, dtype=object),
        "scenario": np.full(n_steps, str(frame["scenario"].iloc[0]), dtype=object),
        "timestep": steps.to_numpy(dtype=np.int64),
        "timestamp": timestamps.reindex(steps).array,
    }
    for tag, name in zip(tags, value_columns(tags), strict=True):
        columns[name] = measured[tag].to_numpy(dtype=np.float64)
    for tag, name in zip(tags, true_columns(tags), strict=True):
        columns[name] = true[tag].to_numpy(dtype=np.float64)
    columns.update(
        {
            "labelled": labelled,
            "severity": severity.astype(object).where(severity.notna(), None).to_numpy(),
            "process_fault": process,
            "instrument_fault": instrument,
            "fault_level": level.astype(object).where(level.notna(), None).to_numpy(),
            "labelled_sensors": labelled_sensors,
            "n_episodes": n_episodes,
            "episode_ids": episode_ids.astype(object).where(episode_ids.notna(), None).to_numpy(),
            "trainable_pair": trainable,
        }
    )
    return pd.DataFrame(columns, columns=instant_columns(tags))


def critical_readings(frame: pd.DataFrame) -> int:
    """Return how many readings of a long frame carry the CRITICAL severity."""
    severity = frame["process_severity"].fillna(frame["instrument_severity"])
    return int((severity.astype(str) == CRITICAL).sum())


# ----------------------------------------------------------------------
# Dataset preparado
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class SplitSummary:
    """Cifras de una particion.

    Attributes:
        runs: Runs de la particion.
        instants: Instantes (filas del frame ancho).
        labelled_instants: Instantes con alguna lectura etiquetada.
        critical_instants: Instantes cuya severidad maxima es CRITICAL (D4-10).
        critical_readings: Lecturas CRITICAL.
        trainable_pairs: Pares (t, t+1) sin etiqueta en ninguno de los dos (D4-6).
        covered_modes: Modos de la rejilla de D6 cubiertos (informativo).
        grid_size: Celdas de la rejilla.
    """

    runs: int
    instants: int
    labelled_instants: int
    critical_instants: int
    critical_readings: int
    trainable_pairs: int
    covered_modes: int
    grid_size: int

    @property
    def coverage_fraction(self) -> float:
        """Return the covered share of the D6 grid."""
        return self.covered_modes / self.grid_size


@dataclass(frozen=True)
class PreparedDataset:
    """Los datos de la PINN, repartidos por run.

    Attributes:
        tags: Orden de sensores (el del catalogo).
        runs: Runs del manifiesto, por run id.
        assignment: Run id -> particion.
        frames: Particion -> frame ancho (runs en orden de run id).
        coverage: Particion -> cobertura de la rejilla de D6.
        critical_readings: Particion -> lecturas CRITICAL.
        val_per_scenario: Runs de validacion por escenario.
        test_per_scenario: Runs de test por escenario.
        k_min_readings: Lecturas minimas para cubrir un modo (D6).
        n_min_episodes: Episodios minimos para cubrir un modo (D6).
    """

    tags: tuple[str, ...]
    runs: tuple[ManifestRun, ...]
    assignment: Mapping[str, str]
    frames: Mapping[str, pd.DataFrame]
    coverage: Mapping[str, CoverageScope]
    critical_readings: Mapping[str, int]
    val_per_scenario: int
    test_per_scenario: int
    k_min_readings: int
    n_min_episodes: int

    def summary(self, split: str) -> SplitSummary:
        """Return the figures of one partition.

        Args:
            split: `train`, `val` o `test`.

        Returns:
            The SplitSummary.

        Raises:
            KeyError: If the partition does not exist.
        """
        if split not in self.frames:
            raise KeyError(f"unknown split '{split}'; expected one of {SPLITS}.")
        frame = self.frames[split]
        scope = self.coverage[split]
        return SplitSummary(
            runs=sum(1 for s in self.assignment.values() if s == split),
            instants=len(frame),
            labelled_instants=int(frame["labelled"].sum()),
            critical_instants=int((frame["severity"] == CRITICAL).sum()),
            critical_readings=int(self.critical_readings[split]),
            trainable_pairs=int(frame["trainable_pair"].sum()),
            covered_modes=len(scope.covered),
            grid_size=scope.grid_size,
        )

    def metrics(self) -> dict[str, Any]:
        """Return the metrics document of the stage (no clock, no paths)."""
        splits: dict[str, Any] = {}
        for split in SPLITS:
            summary = self.summary(split)
            splits[split] = {
                "runs": summary.runs,
                "instants": summary.instants,
                "labelled_instants": summary.labelled_instants,
                "critical_instants": summary.critical_instants,
                "critical_readings": summary.critical_readings,
                "trainable_pairs": summary.trainable_pairs,
                "covered_modes": summary.covered_modes,
                "coverage_fraction": summary.coverage_fraction,
            }
        return {
            "version": OUTPUT_VERSION,
            "n_sensors": len(self.tags),
            "critical_kinds": [kind.value for kind in CRITICAL_KINDS],
            "coverage": {
                "grid_size": self.coverage[SPLITS[0]].grid_size,
                "k_min_readings": self.k_min_readings,
                "n_min_episodes": self.n_min_episodes,
            },
            "splits": splits,
        }

    def assignment_document(self) -> dict[str, Any]:
        """Return the run assignment and column contract written next to the parquet."""
        return {
            "version": OUTPUT_VERSION,
            "tags": list(self.tags),
            "val_runs_per_scenario": self.val_per_scenario,
            "test_runs_per_scenario": self.test_per_scenario,
            "runs": {
                run.run_id: {"scenario": run.scenario.value, "split": self.assignment[run.run_id]}
                for run in self.runs
            },
        }


def prepare_dataset(
    source_dir: str | Path,
    catalog: SensorCatalog,
    *,
    val_per_scenario: int,
    test_per_scenario: int,
    k_min_readings: int,
    n_min_episodes: int,
) -> PreparedDataset:
    """Read the simulated dataset and build the per-partition wide frames.

    Args:
        source_dir: Raiz del dataset simulado (con `manifest.json`).
        catalog: Catalogo de sensores.
        val_per_scenario: Runs de validacion por escenario.
        test_per_scenario: Runs de test por escenario.
        k_min_readings: Lecturas minimas para cubrir un modo (D6, informativo).
        n_min_episodes: Episodios minimos para cubrir un modo (D6, informativo).

    Returns:
        The PreparedDataset.

    Raises:
        FileNotFoundError: If the manifest or a run parquet is missing.
        KeyError: If the manifest lacks a field.
        ValueError: If the manifest, a digest, the split or a run frame is invalid.
    """
    runs = read_manifest(source_dir)
    assignment = assign_splits(runs, val_per_scenario, test_per_scenario)
    collectors = {split: CoverageCollector() for split in SPLITS}
    parts: dict[str, list[pd.DataFrame]] = {split: [] for split in SPLITS}
    critical: dict[str, int] = dict.fromkeys(SPLITS, 0)
    for run in runs:
        split = assignment[run.run_id]
        frame = load_run_frame(verified_run_path(source_dir, run))
        instants = run_instants(frame, catalog)
        identity = (str(instants["run_id"].iloc[0]), str(instants["scenario"].iloc[0]))
        if identity != (run.run_id, run.scenario.value):
            raise ValueError(
                f"{run.path.as_posix()}: holds run {identity[0]} ({identity[1]}), the manifest "
                f"says {run.run_id} ({run.scenario.value})."
            )
        parts[split].append(instants)
        collectors[split].add_frame(frame)
        critical[split] += critical_readings(frame)
        _LOG.info("Prepared %s (%s) -> %s", run.run_id, run.scenario.value, split)
    groups = sorted({sensor.sensor_type.value for sensor in catalog.sensors})
    coverage = {
        split: collectors[split].report(k_min_readings, n_min_episodes, 1.0, groups=groups).overall
        for split in SPLITS
    }
    frames = {split: pd.concat(parts[split], ignore_index=True) for split in SPLITS}
    return PreparedDataset(
        tags=catalog.tags,
        runs=runs,
        assignment=assignment,
        frames=frames,
        coverage=coverage,
        critical_readings=critical,
        val_per_scenario=val_per_scenario,
        test_per_scenario=test_per_scenario,
        k_min_readings=k_min_readings,
        n_min_episodes=n_min_episodes,
    )


# ----------------------------------------------------------------------
# Normalizacion z (estadisticos de train, D4-6)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ZScoreStats:
    """Media y desviacion por sensor para normalizar.

    Attributes:
        tags: Orden de sensores.
        mean: Media de cada sensor, en el orden de `tags`.
        std: Desviacion tipica poblacional de cada sensor.
    """

    tags: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]


def fit_zscore(frame: pd.DataFrame, tags: Sequence[str]) -> ZScoreStats:
    """Fit the per-sensor z-score on the unlabelled instants of a frame.

    Se ajusta sobre el valor MEDIDO de los instantes sin etiqueta: los mismos que
    entran en los pares de entrenamiento (D4-6). Debe llamarse con la particion de
    train; validacion y test se normalizan con estos estadisticos.

    Args:
        frame: Frame ancho de una particion (normalmente train).
        tags: Orden de sensores.

    Returns:
        The ZScoreStats.

    Raises:
        ValueError: If the frame has no unlabelled instant or a sensor is constant.
    """
    rows = frame.loc[~frame["labelled"].to_numpy(dtype=np.bool_), value_columns(tags)]
    if rows.empty:
        raise ValueError("fit_zscore needs at least one unlabelled instant.")
    values = rows.to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    flat = [tag for tag, s in zip(tags, std, strict=True) if not s > 0.0]
    if flat:
        raise ValueError(f"fit_zscore: sensors {flat} are constant on the unlabelled instants.")
    return ZScoreStats(
        tags=tuple(tags),
        mean=tuple(float(m) for m in mean),
        std=tuple(float(s) for s in std),
    )


# ----------------------------------------------------------------------
# Escritura
# ----------------------------------------------------------------------


def _write_json(document: Mapping[str, Any], path: Path) -> Path:
    """Write a JSON document with sorted keys and a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def write_prepared(prepared: PreparedDataset, data_dir: str | Path) -> list[Path]:
    """Write one parquet per partition and the assignment JSON.

    Args:
        prepared: Dataset preparado.
        data_dir: Directorio de salida (out de DVC).

    Returns:
        The paths written: the three parquet files, then the assignment.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for split in SPLITS:
        target = root / f"{split}.parquet"
        prepared.frames[split].to_parquet(target, index=False)
        written.append(target)
    written.append(_write_json(prepared.assignment_document(), root / ASSIGNMENT_NAME))
    return written


def write_metrics(prepared: PreparedDataset, path: str | Path) -> Path:
    """Write the metrics JSON of the stage.

    Args:
        prepared: Dataset preparado.
        path: Destino (`pinn.metrics_path`).

    Returns:
        The path written.
    """
    return _write_json(prepared.metrics(), Path(path))
