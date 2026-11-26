"""Unit tests for ml.training.grid_search (rejilla de lambdas, F4-M5).

La rejilla corre sobre las particiones pequenas de `tests/support/sim_frames.py`
con la red diminuta y una rejilla de 2x2 (cada entrenamiento tarda decimas de
segundo). La regla de seleccion se prueba aparte sobre entradas sinteticas, sin
entrenar. MLflow usa un store sqlite temporal compartido por el modulo (crear el
esquema cuesta ~3 s).
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

from ml.training import grid_search as grid_module
from ml.training.grid_search import (
    GRID_NAME,
    GRID_TAG,
    ROLE_GRID,
    ROLE_SEED_REPEAT,
    GridOutcome,
    GridPoint,
    NullGridRunLogger,
    RunEntry,
    admissible,
    entry_document,
    grid_points,
    main,
    mlflow_factory,
    null_factory,
    provisional,
    repeated_points,
    run_grid,
    seed_noise,
    select,
)
from ml.training.pinn_params import PinnGridParams, load_pinn_grid_params
from tests.support.sim_frames import (
    REPO_ROOT,
    pinn_partitions,
    write_pinn_partitions,
    write_pinn_train_params,
)

# ----------------------------------------------------------------------
# Datos y rejilla compartidos
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the small partitions once per module."""
    return write_pinn_partitions(tmp_path_factory.mktemp("data") / "sim_pinn", pinn_partitions())


@pytest.fixture(scope="module")
def mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a sqlite store shared by the MLflow tests of the module."""
    store = tmp_path_factory.mktemp("mlflow") / "mlruns" / "mlflow.db"
    return f"sqlite:///{store.as_posix()}"


def _setup(work_dir: Path, data_dir: Path, **overrides: Any) -> tuple[Path, PinnGridParams]:
    path = write_pinn_train_params(work_dir, data_dir=str(data_dir), **overrides)
    return path, load_pinn_grid_params(path)


@pytest.fixture(scope="module")
def null_grid(tmp_path_factory: pytest.TempPathFactory, data_dir: Path) -> GridOutcome:
    """Run the 2x2 grid once without MLflow."""
    path, params = _setup(tmp_path_factory.mktemp("grid"), data_dir)
    return run_grid(path, params, null_factory)


# ----------------------------------------------------------------------
# Regla de seleccion sobre entradas sinteticas
# ----------------------------------------------------------------------


def _entry(
    energy: float,
    mass: float,
    val_data: float,
    *,
    seed: int = 42,
    satisfaction: float | None = 1.0,
    val_energy: float = 0.0,
    val_mass: float = 0.0,
) -> RunEntry:
    return RunEntry(
        point=GridPoint(energy, mass),
        seed=seed,
        role=ROLE_GRID if seed == 42 else ROLE_SEED_REPEAT,
        run_id=None,
        best_epoch=0,
        epochs_run=1,
        val_data=val_data,
        val_energy=val_energy,
        val_mass=val_mass,
        physics_share=0.0,
        val_satisfaction=satisfaction,
        seconds=0.0,
    )


class TestGridPoint:
    """Points have a stable label and a fixed order."""

    def test_label(self) -> None:
        """The label uses the shortest float form."""
        assert GridPoint(0.0, 1.0e4).label == "e0_m10000"
        assert GridPoint(1.0e5, 0.5).label == "e100000_m0.5"

    def test_grid_order(self, tmp_path: Path, data_dir: Path) -> None:
        """lambda_energy is the outer loop, lambda_mass the inner one."""
        _, params = _setup(tmp_path, data_dir, grid_lambda_energy=[0.0, 2.0])
        assert [p.label for p in grid_points(params)] == ["e0_m0", "e0_m1", "e2_m0", "e2_m1"]


class TestSelectionRule:
    """The written rule: admissible, min val_data, seed tolerance, physics tie-break."""

    def test_admissible_filters_by_satisfaction(self) -> None:
        """Below the threshold or without satisfaction the entry is out."""
        entries = [
            _entry(0, 0, 1.0, satisfaction=0.99),
            _entry(0, 1, 1.0, satisfaction=0.98),
            _entry(1, 0, 1.0, satisfaction=None),
        ]
        assert admissible(entries, 0.99) == entries[:1]

    def test_provisional_is_min_val_data_first_on_ties(self) -> None:
        """Ties keep the grid order."""
        entries = [_entry(0, 0, 2.0), _entry(0, 1, 1.0), _entry(1, 0, 1.0)]
        assert provisional(entries) == entries[1]
        assert provisional([]) is None

    def test_seed_noise(self) -> None:
        """Sample std over the runs of one point; None with a single run."""
        entries = [_entry(0, 0, 1.0), _entry(0, 0, 3.0, seed=43), _entry(1, 0, 5.0)]
        assert seed_noise(entries, GridPoint(0, 0)) == pytest.approx(2.0**0.5)
        assert seed_noise(entries, GridPoint(1, 0)) is None

    def test_repeated_points(self) -> None:
        """(0, 0) and the provisional point, once each, only if they are in the grid."""
        points = [GridPoint(0, 0), GridPoint(1, 0)]
        assert repeated_points(points, _entry(1, 0, 1.0)) == points
        assert repeated_points(points, _entry(0, 0, 1.0)) == points[:1]
        assert repeated_points(points, None) == points[:1]
        assert repeated_points(points[1:], None) == []

    def test_without_noise_the_min_val_data_wins(self) -> None:
        """Tolerance 0: only the minimum is a candidate."""
        grid = [_entry(0, 0, 1.0, val_energy=9.0), _entry(1, 0, 1.1)]
        selection = select(grid, [], 0.99)
        assert selection.tolerance == 0.0
        assert selection.winner == grid[0]
        assert selection.candidates == [grid[0]]

    def test_tie_break_by_physics_within_the_seed_noise(self) -> None:
        """Within the tolerance the smallest val_energy + val_mass wins."""
        grid = [
            _entry(0, 0, 1.00, val_energy=4.0, val_mass=4.0),
            _entry(1, 0, 1.05, val_energy=1.0, val_mass=2.0),
            _entry(1, 1, 1.04, val_energy=1.0, val_mass=2.0),
            _entry(2, 0, 1.50, val_energy=0.0, val_mass=0.0),
        ]
        repeats = [_entry(0, 0, 1.10, seed=43), _entry(0, 0, 1.20, seed=44)]
        selection = select(grid, repeats, 0.99)
        assert selection.tolerance == pytest.approx(0.1)
        assert selection.provisional == grid[0]
        assert selection.candidates == grid[:3]
        # Mismo val_energy + val_mass en (1, 0) y (1, 1): desempata val_data.
        assert selection.winner == grid[2]
        assert selection.n_admissible == 4

    def test_inadmissible_entries_never_win(self) -> None:
        """A better val_data below the satisfaction threshold is ignored."""
        grid = [_entry(0, 0, 0.5, satisfaction=0.5), _entry(1, 0, 1.0)]
        assert select(grid, [], 0.99).winner == grid[1]

    def test_no_admissible(self) -> None:
        """Without admissible entries there is no winner."""
        selection = select([_entry(0, 0, 1.0, satisfaction=0.1)], [], 0.99)
        assert selection.winner is None
        assert selection.provisional is None
        assert selection.n_admissible == 0
        assert selection.candidates == []

    def test_entry_document_rounds_and_maps_none(self) -> None:
        """12 significant digits; a missing satisfaction is null."""
        document = entry_document(_entry(0, 1, 1 / 3, satisfaction=None))
        assert document["val_data"] == 0.333333333333
        assert document["val_satisfaction"] is None
        assert document["point"] == "e0_m1"


# ----------------------------------------------------------------------
# Rejilla completa
# ----------------------------------------------------------------------


class TestRunGrid:
    """The 2x2 grid trains every point, repeats two and writes the manifest."""

    def test_runs_and_roles(self, null_grid: GridOutcome) -> None:
        """Four grid runs with the base seed, then the seed repeats."""
        assert [e.point.label for e in null_grid.grid] == ["e0_m0", "e0_m1", "e1_m0", "e1_m1"]
        assert {e.seed for e in null_grid.grid} == {42}
        assert {e.role for e in null_grid.grid} == {ROLE_GRID}
        repeated = repeated_points(
            [e.point for e in null_grid.grid], null_grid.selection.provisional
        )
        assert [(e.point, e.seed) for e in null_grid.repeats] == [(p, 43) for p in repeated]
        assert {e.role for e in null_grid.repeats} == {ROLE_SEED_REPEAT}

    def test_entries_are_measured(self, null_grid: GridOutcome) -> None:
        """Every run has finite losses and a val satisfaction in [0, 1]."""
        for entry in [*null_grid.grid, *null_grid.repeats]:
            assert entry.val_data > 0.0
            assert entry.val_energy >= 0.0 and entry.val_mass >= 0.0
            assert entry.val_satisfaction is not None
            assert 0.0 <= entry.val_satisfaction <= 1.0
            assert entry.run_id is None
            assert entry.seconds >= 0.0

    def test_physics_share_follows_the_lambdas(self, null_grid: GridOutcome) -> None:
        """Without physics its share is 0; with lambda > 0 it is positive."""
        shares = {e.point.label: e.physics_share for e in null_grid.grid}
        assert shares["e0_m0"] == 0.0
        assert shares["e1_m1"] > 0.0

    def test_seed_changes_the_result(self, null_grid: GridOutcome) -> None:
        """Another seed gives another val_data at (0, 0): the noise is measurable."""
        base = null_grid.grid[0]
        repeat = next(e for e in null_grid.repeats if e.point == base.point)
        assert repeat.val_data != base.val_data
        assert null_grid.selection.tolerance > 0.0

    def test_manifest(self, null_grid: GridOutcome) -> None:
        """The JSON on disk is the document, without clock or absolute paths."""
        document = json.loads(null_grid.metrics_path.read_text(encoding="utf-8"))
        assert document == null_grid.document
        assert document["n_runs"] == len(document["runs"]) == 4 + len(null_grid.repeats)
        assert document["grid"]["lambda_energy"] == [0.0, 1.0]
        assert set(document["inputs"]) == {"assignment.json", "train.parquet", "val.parquet"}
        winner = null_grid.selection.winner
        assert winner is not None
        assert document["selection"]["winner"]["point"] == winner.point.label
        text = null_grid.metrics_path.read_text(encoding="utf-8")
        assert "seconds" not in text
        assert str(null_grid.metrics_path.parent) not in text

    def test_manifest_is_byte_stable(
        self, tmp_path: Path, data_dir: Path, null_grid: GridOutcome
    ) -> None:
        """A second execution writes the same bytes."""
        path, params = _setup(tmp_path, data_dir)
        again = run_grid(path, params, null_factory)
        assert again.metrics_path.read_bytes() == null_grid.metrics_path.read_bytes()

    def test_missing_satisfaction_is_logged_as_nan(
        self, tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without evaluable instants the grid has no winner and logs NaN."""
        monkeypatch.setattr(grid_module, "val_satisfaction", lambda *args: None)
        path, params = _setup(tmp_path, data_dir, grid_lambda_energy=[0.0], grid_lambda_mass=[0.0])
        outcome = run_grid(path, params, null_factory)
        assert outcome.selection.winner is None
        assert outcome.document["selection"]["winner"] is None
        assert outcome.document["runs"][0]["val_satisfaction"] is None

    def test_missing_val_partition(self, tmp_path: Path, data_dir: Path) -> None:
        """A missing val.parquet is reported before training."""
        frames = pinn_partitions()
        del frames["val"]
        incomplete = write_pinn_partitions(tmp_path / "partial", frames)
        path, _ = _setup(tmp_path / "work", data_dir)
        val_free = load_pinn_grid_params(
            write_pinn_train_params(tmp_path / "work2", data_dir=str(incomplete))
        )
        with pytest.raises(FileNotFoundError, match="val.parquet"):
            grid_module.load_val_physics(path, val_free.train)


class _Recorder(NullGridRunLogger):
    """Logger that records how it was closed."""

    statuses: list[str] = []

    def finish(self, status: str) -> None:
        _Recorder.statuses.append(status)


class TestFailures:
    """A failing run is closed as FAILED and the error propagates."""

    def test_failed_run_is_closed(
        self, tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Training raises: the run is FAILED."""

        def boom(*args: Any) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(grid_module, "train_model", boom)
        _Recorder.statuses = []
        path, params = _setup(tmp_path, data_dir)
        with pytest.raises(RuntimeError, match="boom"):
            run_grid(path, params, lambda name: _Recorder())
        assert _Recorder.statuses == ["FAILED"]


# ----------------------------------------------------------------------
# MLflow (criterio 4)
# ----------------------------------------------------------------------


class TestMlflow:
    """Every run of the manifest is in the store with the grid tag."""

    def test_runs_in_store_match_the_manifest(
        self, tmp_path: Path, data_dir: Path, mlflow_uri: str
    ) -> None:
        """Criterion 4: store runs == manifest entries, all FINISHED and tagged."""
        from mlflow.tracking import MlflowClient

        path, params = _setup(tmp_path, data_dir, mlflow_tracking_uri=mlflow_uri)
        outcome = run_grid(path, params, mlflow_factory(mlflow_uri, "grid-test"))
        client = MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name("grid-test")
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        manifest_ids = [entry["run_id"] for entry in outcome.document["runs"]]
        assert sorted(r.info.run_id for r in runs) == sorted(manifest_ids)
        for run in runs:
            assert run.info.status == "FINISHED"
            assert run.data.tags[GRID_TAG] == GRID_NAME
            assert run.data.tags["grid_role"] in {ROLE_GRID, ROLE_SEED_REPEAT}
            assert "val_satisfaction" in run.data.metrics
        by_id = {r.info.run_id: r for r in runs}
        for entry in outcome.document["runs"]:
            logged = by_id[entry["run_id"]].data.params
            assert float(logged["lambda_energy"]) == entry["lambda_energy"]
            assert float(logged["lambda_mass"]) == entry["lambda_mass"]
            assert int(logged["seed"]) == entry["seed"]

    def test_main_and_script(
        self,
        tmp_path: Path,
        data_dir: Path,
        mlflow_uri: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """main() logs to the store of params.yaml; the script reads the working directory."""
        overrides = {
            "mlflow_tracking_uri": mlflow_uri,
            "mlflow_experiment": "grid-main",
            "grid_lambda_energy": [0.0],
            "grid_lambda_mass": [0.0],
            "max_epochs": 1,
        }
        path, params = _setup(tmp_path / "main", data_dir, **overrides)
        outcome = main(path)
        assert outcome.metrics_path == params.metrics_path
        assert all(entry["run_id"] for entry in outcome.document["runs"])

        _setup(tmp_path, data_dir, **overrides)
        monkeypatch.chdir(tmp_path)
        runpy.run_path(str(REPO_ROOT / "ml" / "training" / "grid_search.py"), run_name="__main__")
        assert (tmp_path / "metrics" / "pinn_grid.json").exists()
        assert not (tmp_path / "mlflow.db").exists()
