"""Unit tests for ml.training.prepare_sim (entry point of the prepare_sim stage).

El stage se ejecuta de punta a punta sobre un dataset sintetico en `tmp_path`, con
un params.yaml que lleva las secciones `simulation:` (rutas a las tablas
versionadas) y `pinn:` (salidas bajo `tmp_path`).
"""

from __future__ import annotations

import json
import logging
import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from data.simulation.scenario import ScenarioKind
from ml.training.prepare_sim import main
from ml.training.sim_dataset import ASSIGNMENT_NAME, SPLITS
from tests.support.sim_frames import (
    REPO_ROOT,
    LabelSpan,
    build_catalog,
    long_run_frame,
    sim_section,
    write_long_dataset,
)

_RUNS_PER_SCENARIO = 3
_N_STEPS = 6


def _write_dataset(root: Path) -> None:
    """Write three runs of normal operation and three of excursion."""
    catalog = build_catalog()
    frames = []
    for index in range(_RUNS_PER_SCENARIO):
        frames.append(
            long_run_frame(catalog, f"sim_run_{index + 1:03d}", ScenarioKind.NORMAL, _N_STEPS)
        )
        frames.append(
            long_run_frame(
                catalog,
                f"sim_run_{index + 11:03d}",
                ScenarioKind.CORE_TEMP_EXCURSION,
                _N_STEPS,
                [LabelSpan(2, 3)],
            )
        )
    write_long_dataset(root, frames)


def _write_params(work_dir: Path, **pinn_overrides: object) -> Path:
    """Write params.yaml with a simulation: section over the dataset and a pinn: section."""
    pinn: dict[str, object] = {
        "data_dir": str(work_dir / "sim_pinn"),
        "metrics_path": str(work_dir / "metrics" / "sim_pinn_metrics.json"),
        "val_runs_per_scenario": 1,
        "test_runs_per_scenario": 1,
    }
    pinn.update(pinn_overrides)
    document = {
        "simulation": sim_section(work_dir, output_dir=str(work_dir / "simulation")),
        "pinn": pinn,
    }
    path = work_dir / "params.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture()
def params_path(tmp_path: Path) -> Path:
    """Return a params file over a freshly written synthetic dataset."""
    _write_dataset(tmp_path / "simulation")
    return _write_params(tmp_path)


class TestMain:
    """main() reads the dataset, splits it by run and writes the outputs."""

    def test_writes_partitions_assignment_and_metrics(
        self, tmp_path: Path, params_path: Path
    ) -> None:
        """Three parquet files and the assignment under data_dir; metrics apart."""
        prepared = main(params_path)
        out = tmp_path / "sim_pinn"
        assert sorted(p.name for p in out.iterdir()) == sorted(
            [*(f"{s}.parquet" for s in SPLITS), ASSIGNMENT_NAME]
        )
        metrics = json.loads(
            (tmp_path / "metrics" / "sim_pinn_metrics.json").read_text(encoding="utf-8")
        )
        assert metrics == prepared.metrics()
        assert metrics["splits"]["test"]["critical_instants"] == 2

    def test_partitions_follow_the_params(self, tmp_path: Path, params_path: Path) -> None:
        """One run of each scenario per partition, read back from disk."""
        main(params_path)
        test = pd.read_parquet(tmp_path / "sim_pinn" / "test.parquet")
        assert sorted(set(test["run_id"])) == ["sim_run_003", "sim_run_013"]

    def test_coverage_thresholds_come_from_plant_parameters(self, params_path: Path) -> None:
        """k_min_readings and n_min_episodes are the D6 values of plant_parameters.yaml."""
        prepared = main(params_path)
        assert (prepared.k_min_readings, prepared.n_min_episodes) == (10, 1)

    def test_logs_one_line_per_partition(
        self, params_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The stage reports its figures."""
        with caplog.at_level(logging.INFO, logger="ml.training.prepare_sim"):
            main(params_path)
        lines = [r.getMessage() for r in caplog.records if r.name == "ml.training.prepare_sim"]
        assert sum(line.lstrip().startswith(SPLITS) for line in lines) == 3

    def test_split_needing_more_runs_fails(self, tmp_path: Path) -> None:
        """Two val runs per scenario leave no train run with three runs."""
        _write_dataset(tmp_path / "simulation")
        path = _write_params(tmp_path, val_runs_per_scenario=2)
        with pytest.raises(ValueError, match="needs at least 4"):
            main(path)
        assert not (tmp_path / "sim_pinn").exists()

    def test_missing_dataset(self, tmp_path: Path) -> None:
        """Without the simulate output the stage fails before writing."""
        path = _write_params(tmp_path)
        with pytest.raises(FileNotFoundError, match="simulate stage"):
            main(path)


class TestScriptEntryPoint:
    """`python ml/training/prepare_sim.py` reads params.yaml from the working directory."""

    def test_runs_as_a_script(
        self, tmp_path: Path, params_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The __main__ block runs the stage with the default params path."""
        monkeypatch.chdir(tmp_path)
        runpy.run_path(str(REPO_ROOT / "ml" / "training" / "prepare_sim.py"), run_name="__main__")
        assert (tmp_path / "sim_pinn" / "train.parquet").exists()
