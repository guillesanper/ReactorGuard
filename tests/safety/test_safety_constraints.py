"""Safety-critical constraint tests over the simulated dataset (F3-M6).

These tests verify that the system never silently misclassifies
high-severity faults as nominal. They are run as a separate suite
to highlight their safety-critical status in CI.

Reescritos en F3-M6 contra el contrato canonico: la version anterior asertaba el
schema plano (is_anomaly, fault_type) previo al commit f130d27 y estaba en skip.
Los tres tests originales se conservan en su semantica (tasa dentro de +-2 pp,
toda lectura con fallo lleva tipo, ninguna lectura nominal lleva etiqueta) sobre
las columnas de dos ejes del TDD 2.2. Se anaden los criterios de F3-M6: solo el
eje declarado lleva etiqueta, CRITICAL por encima del minimo, conservacion bajo
tolerancia en todos los runs, matriz de violaciones esperada y determinismo.

El dataset es el del stage `simulate` (`main`) sobre una configuracion pequena
(`write_small_dataset_params`: 34 runs de 60 pasos con la tasa y los pesos de la
opcion B de D11), leido de vuelta desde los parquet escritos. Se construye dos
veces, en este proceso y con dos procesos trabajadores, y ambas deben coincidir
byte a byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.generators.reactor_simulator import main
from data.simulation.dataset_builder import (
    EXPECTED_TRUE_STATE_VIOLATIONS,
    MANIFEST_NAME,
    NO_FAULT,
    SimulatedDataset,
    file_sha256,
)
from data.simulation.physics_constraints import RULE_NAMES
from data.simulation.scenario import FAULT_SEVERITY, FaultAxis, FaultSeverity, ScenarioKind
from data.simulation.simulation_params import SimulationParams, load_simulation_params
from tests.support.sim_frames import write_small_dataset_params

RATE_TOLERANCE = 0.02
"""+-2 pp sobre `fault_injection_rate` (test original de la suite)."""


@dataclass(frozen=True)
class Build:
    """Una construccion del dataset y lo que dejo en disco."""

    params: SimulationParams
    dataset: SimulatedDataset
    frame: pd.DataFrame
    manifest: dict[str, Any]
    metrics_bytes: bytes
    manifest_bytes: bytes


def _build(work_dir: Path, workers: int) -> Build:
    params_path = write_small_dataset_params(work_dir)
    params = load_simulation_params(params_path)
    dataset = main(params_path, workers=workers)
    manifest_path = params.output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = [pd.read_parquet(params.output_dir / run["path"]) for run in manifest["runs"]]
    return Build(
        params=params,
        dataset=dataset,
        frame=pd.concat(frames, ignore_index=True),
        manifest=manifest,
        metrics_bytes=params.metrics_path.read_bytes(),
        manifest_bytes=manifest_path.read_bytes(),
    )


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Build:
    """Return the small dataset built in this process."""
    return _build(tmp_path_factory.mktemp("serial"), workers=1)


@pytest.fixture(scope="module")
def parallel(tmp_path_factory: pytest.TempPathFactory) -> Build:
    """Return the same dataset built by two spawned worker processes."""
    return _build(tmp_path_factory.mktemp("parallel"), workers=2)


def _labelled(frame: pd.DataFrame) -> pd.Series:
    return (frame["process_fault"] != NO_FAULT) | (frame["instrument_fault"] != NO_FAULT)


# ----------------------------------------------------------------------
# Tests originales (semantica conservada sobre el contrato canonico)
# ----------------------------------------------------------------------


def test_fault_injection_rate_within_tolerance(build: Build) -> None:
    """Generated labelled rate must be within +-2 pp of the configured rate."""
    actual_rate = float(_labelled(build.frame).mean())
    assert abs(actual_rate - build.params.fault_injection_rate) < RATE_TOLERANCE
    assert actual_rate == pytest.approx(build.dataset.labelled_fraction)


def test_all_faults_have_fault_type(build: Build) -> None:
    """Every labelled reading names its fault, severity, level and episode."""
    faulty = build.frame[_labelled(build.frame)]
    assert len(faulty) > 0
    kind = faulty["process_fault"].where(
        faulty["process_fault"] != NO_FAULT, faulty["instrument_fault"]
    )
    assert (kind == faulty["scenario"]).all()
    severity = faulty["process_severity"].fillna(faulty["instrument_severity"])
    assert severity.notna().all()
    assert faulty["fault_level"].notna().all()
    assert faulty["episode_id"].notna().all()
    assert faulty["injected_at"].notna().all()
    assert faulty["injected_at_step"].notna().all()


def test_nominal_readings_no_fault_label(build: Build) -> None:
    """Readings outside every fault window carry no fault label at all."""
    nominal = build.frame[~_labelled(build.frame)]
    assert len(nominal) > 0
    for column in (
        "process_severity",
        "instrument_severity",
        "fault_level",
        "episode_id",
        "injected_at",
        "injected_at_step",
    ):
        assert nominal[column].isna().all(), column
    assert not nominal["corrupted"].any()


# ----------------------------------------------------------------------
# Criterios de F3-M6
# ----------------------------------------------------------------------


def test_only_the_declared_axis_is_labelled(build: Build) -> None:
    """A fault run labels only its own axis; a normal run labels nothing."""
    frame = build.frame
    axis = frame["scenario"].map(lambda s: ScenarioKind(s).axis)
    process = frame["process_fault"] != NO_FAULT
    instrument = frame["instrument_fault"] != NO_FAULT
    assert not process[axis != FaultAxis.PROCESS].any()
    assert not instrument[axis != FaultAxis.INSTRUMENT].any()
    assert frame["process_severity"][axis != FaultAxis.PROCESS].isna().all()
    assert frame["instrument_severity"][axis != FaultAxis.INSTRUMENT].isna().all()
    normal = frame["scenario"] == ScenarioKind.NORMAL.value
    assert normal.any()
    assert not _labelled(frame)[normal].any()


def test_severity_is_the_static_tdd_table(build: Build) -> None:
    """A fault is never labelled with a lower severity than TDD 2.2 assigns it."""
    faulty = build.frame[_labelled(build.frame)]
    severity = faulty["process_severity"].fillna(faulty["instrument_severity"])
    expected = faulty["scenario"].map(lambda s: FAULT_SEVERITY[ScenarioKind(s)].value)
    assert (severity == expected).all()


def test_every_fault_type_is_present(build: Build) -> None:
    """The 8 TDD fault types reach the dataset, each with labelled readings."""
    faulty = build.frame[_labelled(build.frame)]
    assert set(faulty["scenario"]) == {kind.value for kind in FAULT_SEVERITY}
    assert build.dataset.coverage.types_ok


def test_critical_fraction_above_minimum(build: Build) -> None:
    """CRITICAL readings reach critical_min_fraction of the whole dataset."""
    faulty = build.frame[_labelled(build.frame)]
    severity = faulty["process_severity"].fillna(faulty["instrument_severity"])
    critical = int((severity == FaultSeverity.CRITICAL.value).sum())
    assert critical / len(build.frame) >= build.params.critical_min_fraction
    assert build.dataset.critical_fraction == pytest.approx(critical / len(build.frame))


def test_conservation_under_tolerance_in_every_run(build: Build) -> None:
    """Energy closes under the D7 tolerance on every instant and interval of every run."""
    tolerance = build.dataset.conservation_tolerance
    for run in build.manifest["runs"]:
        assert run["conservation_failures"] == 0, run["run_id"]
        assert run["max_interval_residual"] <= tolerance, run["run_id"]
        assert run["max_instant_residual"] <= tolerance, run["run_id"]
    assert build.dataset.conservation_compliant


def test_expected_violation_matrix(build: Build) -> None:
    """The true state violates only the rules the matrix allows for its scenario."""
    for run in build.manifest["runs"]:
        allowed = EXPECTED_TRUE_STATE_VIOLATIONS.get(ScenarioKind(run["scenario"]), frozenset())
        assert set(run["violations_true"]) == set(RULE_NAMES)
        unexpected = {r: n for r, n in run["violations_true"].items() if n and r not in allowed}
        assert unexpected == {}, run["run_id"]
    assert build.dataset.violations_as_expected


def test_true_value_untouched_by_instrument_faults(build: Build) -> None:
    """An instrument fault changes the reading, never the plant's true value."""
    frame = build.frame
    corrupted = frame[frame["corrupted"]]
    assert len(corrupted) > 0
    assert (corrupted["scenario"].map(lambda s: ScenarioKind(s).axis) == FaultAxis.INSTRUMENT).all()
    assert (corrupted["instrument_fault"] != NO_FAULT).all()


def test_manifest_matches_the_files(build: Build) -> None:
    """Each manifest entry hashes and counts the parquet it points to."""
    root = build.params.output_dir
    for run in build.manifest["runs"]:
        path = root / run["path"]
        assert not Path(run["path"]).is_absolute()
        assert file_sha256(path) == run["sha256"]
        assert len(pd.read_parquet(path, columns=["run_id"])) == run["rows"]
    assert len(build.frame) == build.manifest["total_readings"]


def test_plan_equals_frame(build: Build) -> None:
    """Every run labels exactly the readings its plan foresaw."""
    for run in build.manifest["runs"]:
        assert run["labelled"] == run["planned_labelled"], run["run_id"]


def test_determinism_across_builds_and_processes(build: Build, parallel: Build) -> None:
    """The same seed gives the same bytes, in this process or in worker processes."""
    assert build.manifest_bytes == parallel.manifest_bytes
    assert build.metrics_bytes == parallel.metrics_bytes
    assert [r["sha256"] for r in build.manifest["runs"]] == [
        r["sha256"] for r in parallel.manifest["runs"]
    ]
