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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import data.generators.reactor_simulator as simulator_module
from data.generators.reactor_simulator import ReactorSimulator, cli, main, resolve_workers
from data.generators.tep_adapter import readings_from_frame
from data.schemas.sensor_reading import SensorReading
from data.simulation.dataset_builder import (
    LONG_COLUMNS,
    MANIFEST_NAME,
    SimulatedDataset,
    file_sha256,
    wide_frame_from_long,
)
from data.simulation.physics_constraints import RULE_NAMES, ConstraintLimits, evaluate_rules
from data.simulation.scenario import ScenarioKind
from data.simulation.simulation_params import load_simulation_params
from tests.support.sim_frames import (
    SMALL_DATASET_RATE,
    build_simulator,
    write_sim_params,
    write_small_dataset_params,
)


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

    def test_main_refuses_an_unreachable_budget(self, tmp_path: Path) -> None:
        """The stage fails (no dataset) when the planner refuses the D11 budget."""
        params_path = write_sim_params(tmp_path)
        with pytest.raises(ValueError):
            main(params_path, workers=1)
        assert not (tmp_path / "out" / MANIFEST_NAME).exists()

    def test_cli_parses_params_and_workers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The command line takes --params and --workers (0 = one per CPU by default)."""
        calls: list[tuple[str, int]] = []

        def fake_main(params_path: str, *, workers: int) -> str:
            calls.append((params_path, workers))
            return "dataset"

        monkeypatch.setattr(simulator_module, "main", fake_main)
        assert cli(["--params", "p.yaml", "--workers", "3"]) == "dataset"
        assert cli(["--params", "q.yaml"]) == "dataset"
        assert calls == [("p.yaml", 3), ("q.yaml", 0)]

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


@pytest.fixture(scope="module")
def dataset_sized(tmp_path_factory: pytest.TempPathFactory) -> ReactorSimulator:
    """Return a simulator over 34 runs of 1500 steps (planned, never integrated)."""
    return build_simulator(
        tmp_path_factory.mktemp("planned"),
        n_samples=50000,
        run_length_steps=1500,
        openmc_seed=42,
    )


class TestPlannedDataset:
    """`planned` runs the D11 planner over the simulator's own configuration."""

    def test_scenario_config_is_parsed_once(self, simulator: ReactorSimulator) -> None:
        """The levels and caps are resolved lazily and cached."""
        first = simulator.scenario_config
        assert simulator.scenario_config is first
        assert first.planner.n_min_runs == 3
        assert len(first.instrument.kinds) == 5
        assert len(first.process.kinds) == 3

    def test_the_default_budget_is_refused_with_numbers(
        self, dataset_sized: ReactorSimulator
    ) -> None:
        """40 % with the D11 weights does not fit under the caps: the planner says why."""
        with pytest.raises(ValueError, match="unreachable"):
            dataset_sized.planned()

    def test_a_feasible_budget_gives_one_plan_per_run(
        self, dataset_sized: ReactorSimulator
    ) -> None:
        """The plan holds every run, with the eight types present."""
        planned = dataset_sized.planned(
            fault_injection_rate=0.12,
            severity_weights={"critical": 0.55, "high": 0.38, "medium": 0.06, "low": 0.01},
        )
        assert len(planned.plans) == dataset_sized.params.n_runs == 34
        scenarios = {p.scenario for p in planned.plans}
        assert len(scenarios) == 9
        assert abs(planned.labelled_fraction - 0.12) < 0.02

    def test_plans_property_stays_the_normal_dataset(self, dataset_sized: ReactorSimulator) -> None:
        """`plans` is still operation normal: the planned dataset is a separate call."""
        assert all(p.scenario is ScenarioKind.NORMAL for p in dataset_sized.plans)

    def test_a_planned_instrument_run_generates_a_labelled_frame(
        self, simulator: ReactorSimulator
    ) -> None:
        """generate_run applies the episodes of an instrument plan."""
        from data.simulation.scenario import (
            FaultLevel,
            InstrumentEpisode,
            InstrumentFaultKind,
            make_run_plan,
        )

        episode = InstrumentEpisode(
            "PT-PRZ-01", InstrumentFaultKind.BIAS, FaultLevel.HIGH, 10, 20, 1.2
        )
        plan = make_run_plan(simulator.params, 0, ScenarioKind.BIAS, instrument_faults=(episode,))
        frame = simulator.generate_run(plan)
        assert tuple(frame.columns) == LONG_COLUMNS
        labelled = frame[frame["instrument_fault"] == "bias"]
        assert len(labelled) == 20
        assert labelled["corrupted"].all()
        assert (labelled["sensor_id"] == "PT-PRZ-01").all()


# ----------------------------------------------------------------------
# F3-M6: el stage simulate escribe el dataset planificado
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageRun:
    """Lo que deja `main` sobre la configuracion pequena."""

    root: Path
    metrics_path: Path
    dataset: SimulatedDataset


@pytest.fixture(scope="module")
def stage_run(tmp_path_factory: pytest.TempPathFactory) -> StageRun:
    """Run the stage entry point once (34 runs of 60 steps, in this process)."""
    work = tmp_path_factory.mktemp("stage")
    params_path = write_small_dataset_params(work)
    params = load_simulation_params(params_path)
    dataset = main(params_path, workers=1)
    return StageRun(root=params.output_dir, metrics_path=params.metrics_path, dataset=dataset)


class TestStage:
    """`main` writes parquets, manifest and metrics of the planned dataset."""

    def test_layout_and_manifest(self, stage_run: StageRun) -> None:
        """One parquet per run under its scenario, each one hashed in the manifest."""
        manifest = json.loads((stage_run.root / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["n_runs"] == 34 == len(manifest["runs"])
        parquets = sorted(p.relative_to(stage_run.root).as_posix()
                          for p in stage_run.root.rglob("*.parquet"))
        assert parquets == sorted(run["path"] for run in manifest["runs"])
        for run in manifest["runs"]:
            assert run["path"] == f"fault_type={run['scenario']}/seed=42/{run['run_id']}.parquet"
            assert file_sha256(stage_run.root / run["path"]) == run["sha256"]
        assert {run["scenario"] for run in manifest["runs"]} == {k.value for k in ScenarioKind}

    def test_metrics(self, stage_run: StageRun) -> None:
        """The metrics file is the dataset summary, written next to nothing else."""
        metrics = json.loads(stage_run.metrics_path.read_text(encoding="utf-8"))
        assert metrics == json.loads(json.dumps(stage_run.dataset.metrics()))
        assert metrics["fault_injection_rate"] == SMALL_DATASET_RATE
        assert metrics["n_runs"] == 34
        assert metrics["coverage"]["types_ok"]
        assert metrics["compliance"]["conservation_compliant"]
        assert metrics["compliance"]["violations_as_expected"]
        assert metrics["plan_matches_frames"]

    def test_negative_workers_is_refused(self, tmp_path: Path) -> None:
        """workers < 0 is a usage error."""
        sim = ReactorSimulator.from_params(
            load_simulation_params(write_small_dataset_params(tmp_path))
        )
        with pytest.raises(ValueError, match="workers"):
            sim.build_dataset(tmp_path / "ds", workers=-1)

    def test_limits_are_parsed_once(self, simulator: ReactorSimulator) -> None:
        """The rule thresholds come from plant_parameters.yaml and are cached."""
        assert simulator.limits is simulator.limits
        assert simulator.limits.conservation_tolerance == 1.0e-6


class TestWorkers:
    """Worker count and submission order of the parallel stage."""

    def test_resolve_workers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """0 means one per CPU; never more workers than runs, never fewer than one."""
        monkeypatch.setattr(simulator_module.os, "cpu_count", lambda: 8)
        assert resolve_workers(0, 34) == 8
        assert resolve_workers(0, 3) == 3
        assert resolve_workers(4, 34) == 4
        assert resolve_workers(1, 0) == 1
        monkeypatch.setattr(simulator_module.os, "cpu_count", lambda: None)
        assert resolve_workers(0, 34) == 1

    def test_slow_runs_are_submitted_first(self) -> None:
        """Void first, then the other process faults, instrument faults and normal runs."""
        plans = [
            SimpleNamespace(run_id="sim_run_004", scenario=ScenarioKind.NORMAL),
            SimpleNamespace(run_id="sim_run_003", scenario=ScenarioKind.BIAS),
            SimpleNamespace(run_id="sim_run_002", scenario=ScenarioKind.COOLANT_FLOW_REDUCTION),
            SimpleNamespace(run_id="sim_run_005", scenario=ScenarioKind.CORE_TEMP_EXCURSION),
            SimpleNamespace(run_id="sim_run_001", scenario=ScenarioKind.VOID_FORMATION),
        ]
        ordered = sorted(plans, key=simulator_module._cost_key)  # type: ignore[arg-type]
        assert [p.run_id for p in ordered] == [
            "sim_run_001",
            "sim_run_002",
            "sim_run_005",
            "sim_run_003",
            "sim_run_004",
        ]
