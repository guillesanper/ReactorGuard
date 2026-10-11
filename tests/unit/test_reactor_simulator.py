"""Unit tests for data.generators.reactor_simulator (reescrito en F3-M3).

Un run simulado debe ser un frame que (1) se reconstruye como SensorReading
canonico con el mismo codigo del TEP, (2) no viola ninguna de las 7 reglas en
operacion normal, (3) es identico para la misma semilla y (4) se escribe con el
layout del TDD 4.4. La validacion offline vive aqui, no en el modulo de
produccion, que no importa nada de data/validation.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import data.generators.reactor_simulator as simulator_module
from data.generators.reactor_simulator import ReactorSimulator, cli, main
from data.generators.tep_adapter import readings_from_frame
from data.schemas.sensor_reading import SensorReading
from data.simulation.dataset_builder import LONG_COLUMNS, wide_frame_from_long
from data.simulation.physics_constraints import RULE_NAMES, ConstraintLimits, evaluate_rules
from data.simulation.scenario import ScenarioKind
from data.simulation.simulation_params import load_simulation_params
from tests.support.sim_frames import build_simulator, write_sim_params


@pytest.fixture(scope="module")
def simulator(tmp_path_factory: pytest.TempPathFactory) -> ReactorSimulator:
    """Return a simulator over 3 runs of 30 steps."""
    return build_simulator(tmp_path_factory.mktemp("sim"))


@pytest.fixture(scope="module")
def run_frame(simulator: ReactorSimulator) -> pd.DataFrame:
    """Return the first run's long frame."""
    return simulator.generate_run(simulator.plans[0])


class TestConstruction:
    """The simulator is assembled from the files the section points to."""

    def test_from_params_loads_everything(self, simulator: ReactorSimulator) -> None:
        """Model, catalog (27 sensors) and spans are loaded."""
        assert len(simulator.catalog.sensors) == 27
        assert set(simulator.catalog.tags) <= set(simulator.spans)
        assert simulator.params.plant_id == "SIM-TEST-01"

    def test_plans(self, simulator: ReactorSimulator) -> None:
        """One normal plan per run."""
        plans = simulator.plans
        assert len(plans) == 3
        assert all(p.scenario is ScenarioKind.NORMAL for p in plans)

    def test_missing_span_is_an_error(self, simulator: ReactorSimulator) -> None:
        """A sensor without span would be invisible to the validator (H7)."""
        spans = {k: v for k, v in simulator.spans.items() if k != "PT-PRZ-01"}
        with pytest.raises(KeyError, match="PT-PRZ-01"):
            ReactorSimulator(simulator.params, simulator.model, simulator.catalog, spans)

    def test_old_class_name_is_still_exported(self) -> None:
        """tests/safety/test_safety_constraints.py imports ReactorSimulator from here."""
        assert hasattr(simulator_module, "ReactorSimulator")

    def test_production_module_does_not_import_the_validator(self) -> None:
        """The simulator must not depend on data/validation (offline only)."""
        tree = ast.parse(Path(simulator_module.__file__).read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        assert not [m for m in modules if m.startswith("data.validation")]


class TestRunFrame:
    """A run frame is a canonical, rule-compliant, labelled dataset slice."""

    def test_columns(self, run_frame: pd.DataFrame) -> None:
        """The frame has the contract columns and the labels, in order."""
        assert tuple(run_frame.columns) == LONG_COLUMNS

    def test_size(self, run_frame: pd.DataFrame, simulator: ReactorSimulator) -> None:
        """30 steps x 27 sensors."""
        assert len(run_frame) == 30 * 27
        assert run_frame["run_id"].nunique() == 1
        assert run_frame["plant_id"].eq("SIM-TEST-01").all()

    def test_validates_against_the_canonical_contract(self, run_frame: pd.DataFrame) -> None:
        """Every row rebuilds as a SensorReading with the TEP's own function (H12)."""
        readings = list(readings_from_frame(run_frame))
        assert len(readings) == len(run_frame)
        assert all(isinstance(r, SensorReading) for r in readings)
        assert {r.sensor.id for r in readings} == set(run_frame["sensor_id"])

    def test_normal_operation_violates_no_rule(
        self, run_frame: pd.DataFrame, simulator: ReactorSimulator
    ) -> None:
        """Measured normal operation violates none of the seven rules."""
        limits = ConstraintLimits.from_plant(simulator.model.plant)
        wide = wide_frame_from_long(run_frame)
        violations = evaluate_rules(wide, simulator.catalog.roles, simulator.model.table, limits)
        assert list(violations.columns) == list(RULE_NAMES)
        assert int(violations.to_numpy().sum()) == 0

    def test_runs_have_distinct_loads(self, simulator: ReactorSimulator) -> None:
        """The load profile of each run comes from its own stream."""
        a = simulator.simulate(simulator.plans[0]).true_values
        b = simulator.simulate(simulator.plans[1]).true_values
        assert not np.allclose(a, b)

    def test_the_load_actually_moves(self, simulator: ReactorSimulator) -> None:
        """With short holds the run contains a load ramp (the flux changes)."""
        j = simulator.catalog.tags.index("NI-FLUX-01")
        flux = simulator.simulate(simulator.plans[0]).true_values[:, j]
        assert flux.max() - flux.min() > 1.0


class TestDeterminism:
    """The same configuration gives the same dataset."""

    def test_same_run_same_frame(self, simulator: ReactorSimulator) -> None:
        """Two builds of one run are equal frames."""
        plan = simulator.plans[1]
        assert simulator.generate_run(plan).equals(simulator.generate_run(plan))

    def test_independent_instances_agree(self, tmp_path: Path) -> None:
        """Two simulators built from the same params file produce the same run."""
        a = build_simulator(tmp_path / "a")
        b = build_simulator(tmp_path / "b")
        assert a.generate_run(a.plans[0]).equals(b.generate_run(b.plans[0]))

    def test_seed_changes_the_dataset(self, tmp_path: Path) -> None:
        """Another master seed gives other readings."""
        a = build_simulator(tmp_path / "a", openmc_seed=1)
        b = build_simulator(tmp_path / "b", openmc_seed=2)
        fa, fb = a.generate_run(a.plans[0]), b.generate_run(b.plans[0])
        assert not np.array_equal(fa["value"].to_numpy(), fb["value"].to_numpy())

    def test_run_does_not_depend_on_other_runs(self, simulator: ReactorSimulator) -> None:
        """Generating run 2 alone equals run 2 inside the whole dataset."""
        alone = simulator.generate_run(simulator.plans[2])
        whole = simulator.generate()
        inside = whole[whole["run_id"] == "sim_run_003"].reset_index(drop=True)
        assert alone.equals(inside)


class TestDataset:
    """The whole dataset concatenates runs without collisions."""

    def test_generate(self, simulator: ReactorSimulator) -> None:
        """total_steps x sensors rows, unique reading ids, unique timestamps per sensor."""
        dataset = simulator.generate()
        assert len(dataset) == simulator.params.total_steps * 27
        assert dataset["reading_id"].is_unique
        assert not dataset.duplicated(["sensor_id", "timestamp"]).any()
        assert dataset["run_id"].nunique() == 3

    def test_iter_runs_yields_plan_and_frame(self, simulator: ReactorSimulator) -> None:
        """iter_runs pairs every plan with its frame, in run order."""
        pairs = list(simulator.iter_runs())
        assert [p.run_id for p, _ in pairs] == ["sim_run_001", "sim_run_002", "sim_run_003"]
        assert all(len(f) == 30 * 27 for _, f in pairs)


class TestWrite:
    """The dataset is written with the TDD 4.4 layout."""

    def test_write_layout(self, simulator: ReactorSimulator, tmp_path: Path) -> None:
        """One parquet per run under fault_type=normal/seed=<n>/."""
        paths = simulator.write(tmp_path / "ds")
        assert [p.name for p in paths] == [
            "sim_run_001.parquet",
            "sim_run_002.parquet",
            "sim_run_003.parquet",
        ]
        assert all(p.parent == tmp_path / "ds" / "fault_type=normal" / "seed=7" for p in paths)
        assert pd.read_parquet(paths[0]).shape == (30 * 27, len(LONG_COLUMNS))

    def test_write_defaults_to_params_output_dir(self, tmp_path: Path) -> None:
        """Without an argument the files go to simulation.output_dir."""
        sim = build_simulator(tmp_path)
        paths = sim.write()
        assert all(str(p).startswith(str(sim.params.output_dir)) for p in paths)

    def test_main_writes_the_dataset(self, tmp_path: Path) -> None:
        """main(params_path) is the stage entry point."""
        params_path = write_sim_params(tmp_path)
        paths = main(params_path)
        params = load_simulation_params(params_path)
        assert len(paths) == params.n_runs
        assert all(p.exists() for p in paths)

    def test_cli_parses_params(self, tmp_path: Path) -> None:
        """The command line takes --params."""
        params_path = write_sim_params(tmp_path)
        assert len(cli(["--params", str(params_path)])) == 3

    def test_main_missing_params(self, tmp_path: Path) -> None:
        """A missing params file fails clearly."""
        with pytest.raises(FileNotFoundError):
            main(tmp_path / "nope.yaml")

    def test_main_missing_spans_file(self, tmp_path: Path) -> None:
        """A missing span table fails clearly instead of producing a blind dataset."""
        params_path = write_sim_params(tmp_path, spans_path=str(tmp_path / "nospans.yaml"))
        with pytest.raises(FileNotFoundError):
            main(params_path)


class TestRunFrameFromEditedPlan:
    """A shorter plan is honoured (the study tools rely on it)."""

    def test_n_steps_override(self, simulator: ReactorSimulator) -> None:
        """dataclasses.replace(plan, n_steps=...) changes the length of the frame."""
        plan = dataclasses.replace(simulator.plans[0], n_steps=12)
        assert len(simulator.generate_run(plan)) == 12 * 27

    def test_json_serialisable_summary(self, simulator: ReactorSimulator) -> None:
        """The plan parameters round-trip through JSON (manifest in F3-M6)."""
        plan = simulator.plans[0]
        payload = {"run_id": plan.run_id, "seed": plan.seed_entropy, "key": list(plan.spawn_key)}
        assert json.loads(json.dumps(payload))["run_id"] == "sim_run_001"
        assert yaml.safe_load(yaml.safe_dump(payload))["seed"] == 7
