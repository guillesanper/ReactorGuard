"""Unit tests for ml.validation.physics_check (stage evaluate, F4-M4).

Un modelo diminuto (16 neuronas, 6 epocas) se entrena una vez por modulo con
`run_training(..., NullRunLogger())` sobre las particiones pequenas de
`tests/support/sim_frames.py` (2 runs de test de 24 instantes, con un instante
etiquetado cada uno). Las reglas se prueban ademas sobre evaluaciones sinteticas para
fijar los recuentos sin depender de lo que prediga la red.
"""

from __future__ import annotations

import dataclasses
import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from data.simulation.physics_constraints import (
    RULE_NAMES,
    ConstraintLimits,
    RuleEvaluation,
    evaluate_rules_detailed,
)
from data.simulation.plant_params import load_plant_params
from data.simulation.water_properties import WaterTable
from ml.training.checkpoint import MODEL_FILE, LoadedModel, load_checkpoint
from ml.training.pinn_data import context_features
from ml.training.pinn_params import (
    PinnPhysicsParams,
    load_pinn_physics_params,
    load_pinn_train_params,
)
from ml.training.sim_dataset import value_columns
from ml.training.train import NullRunLogger, run_training
from ml.validation import physics_check
from ml.validation.physics_check import (
    SATISFACTION_THRESHOLD,
    evaluate_by_run,
    first_of_run,
    main,
    passes,
    physics_document,
    predict_next,
    run_physics_check,
    summarize,
    torch_threads,
)
from tests.support.sim_frames import (
    CONFIGS,
    REPO_ROOT,
    build_catalog,
    build_table,
    pinn_partitions,
    write_pinn_partitions,
    write_pinn_train_params,
)

ROLES = build_catalog().roles
TAGS = tuple(build_catalog().tags)


@pytest.fixture(scope="module")
def partitions() -> dict[str, pd.DataFrame]:
    """Return the small partitions (built once)."""
    return pinn_partitions()


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, partitions: dict[str, pd.DataFrame]) -> Path:
    """Write the partitions and train a tiny model once; return the params file."""
    work = tmp_path_factory.mktemp("work")
    data_dir = write_pinn_partitions(work / "sim_pinn", partitions)
    params_path = write_pinn_train_params(work, data_dir=str(data_dir))
    run_training(params_path, load_pinn_train_params(params_path), NullRunLogger())
    return params_path


@pytest.fixture(scope="module")
def loaded(trained: Path) -> LoadedModel:
    """Return the checkpoint of the tiny model."""
    return load_checkpoint(load_pinn_physics_params(trained).model_dir / MODEL_FILE)


@pytest.fixture(scope="module")
def table() -> WaterTable:
    """Return the water table of the repository."""
    return build_table()


@pytest.fixture(scope="module")
def limits() -> ConstraintLimits:
    """Return the rule limits of the repository plant."""
    return ConstraintLimits.from_plant(load_plant_params(CONFIGS / "plant_parameters.yaml"))


def _measured(frame: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.float64]]:
    return frame[value_columns(TAGS)].to_numpy(dtype=np.float64)


def _evaluation(violations: list[list[bool]], evaluable: list[list[bool]]) -> RuleEvaluation:
    index = pd.RangeIndex(len(violations))
    return RuleEvaluation(
        violations=pd.DataFrame(violations, index=index, columns=list(RULE_NAMES)),
        evaluable=pd.DataFrame(evaluable, index=index, columns=list(RULE_NAMES)),
    )


ALL = [True] * len(RULE_NAMES)
NONE = [False] * len(RULE_NAMES)


def _only(rule: str) -> list[bool]:
    return [name == rule for name in RULE_NAMES]


class TestFirstOfRun:
    """The first instant of each run is marked."""

    def test_marks_run_starts(self) -> None:
        """Two runs of two instants: rows 0 and 2."""
        frame = pd.DataFrame({"run_id": ["a", "a", "b", "b"]})
        assert first_of_run(frame).tolist() == [True, False, True, False]


class TestTorchThreads:
    """The thread count is fixed and restored."""

    def test_restores_previous_threads(self) -> None:
        """Inside the block torch uses the given threads; afterwards, the previous ones."""
        previous = torch.get_num_threads()
        with torch_threads(1):
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == previous


class TestPredictNext:
    """y(t) from the measured x(t-1) and its context, in engineering units."""

    def test_first_instant_of_each_run_has_no_prediction(
        self, loaded: LoadedModel, partitions: dict[str, pd.DataFrame]
    ) -> None:
        """NaN on run starts, finite elsewhere (labelled instants included)."""
        frame = partitions["test"]
        values = predict_next(loaded, frame, ROLES)
        assert values.shape == (len(frame), len(TAGS))
        first = first_of_run(frame)
        assert np.isnan(values[first]).all()
        assert np.isfinite(values[~first]).all()
        labelled = frame["labelled"].to_numpy(dtype=np.bool_)
        assert np.isfinite(values[labelled & ~first]).all()

    def test_matches_the_network_on_one_row(
        self, loaded: LoadedModel, partitions: dict[str, pd.DataFrame]
    ) -> None:
        """Row t = denormalize(model(z(x(t-1), context(t-1))))."""
        frame = partitions["test"]
        values = predict_next(loaded, frame, ROLES)
        stats = loaded.stats
        row = 5
        raw = np.concatenate(
            [_measured(frame)[row - 1], context_features(frame, TAGS, ROLES)[row - 1]]
        )
        z = (raw - np.asarray(stats.input_mean)) / np.asarray(stats.input_std)
        with torch.no_grad():
            out = loaded.model(torch.tensor(z, dtype=torch.float32)[None, :]).double().numpy()[0]
        expected = out * np.asarray(stats.sensor_std) + np.asarray(stats.sensor_mean)
        np.testing.assert_allclose(values[row], expected, rtol=1e-6)

    def test_predicts_in_engineering_units(
        self, loaded: LoadedModel, partitions: dict[str, pd.DataFrame]
    ) -> None:
        """The prediction is close to the measurement of the same instant, not to z = 0."""
        frame = partitions["test"]
        values = predict_next(loaded, frame, ROLES)
        first = first_of_run(frame)
        column = TAGS.index(ROLES["T_out"][0])
        measured = _measured(frame)[~first, column]
        assert np.abs(values[~first, column] - measured).max() < 5.0

    def test_runs_of_one_instant_give_no_prediction(
        self, loaded: LoadedModel, partitions: dict[str, pd.DataFrame]
    ) -> None:
        """Without t-1 in any run, everything is NaN (and the network is not called)."""
        frame = partitions["test"].iloc[[0]].reset_index(drop=True)
        assert np.isnan(predict_next(loaded, frame, ROLES)).all()


class TestEvaluateByRun:
    """The rules run per run, only on the chosen rows."""

    def test_matches_evaluate_rules_detailed_on_one_run(
        self,
        partitions: dict[str, pd.DataFrame],
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """On a single run the result is that of evaluate_rules_detailed."""
        frame = partitions["test"]
        run = frame["run_id"] == frame["run_id"].iloc[0]
        values = _measured(frame)
        result = evaluate_by_run(frame, values, run.to_numpy(), TAGS, ROLES, table, limits)
        stamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"][run], utc=True))
        wide = pd.DataFrame(values[run.to_numpy()], columns=list(TAGS), index=stamps)
        direct = evaluate_rules_detailed(wide, ROLES, table, limits)
        np.testing.assert_array_equal(
            result.evaluable[run.to_numpy()].to_numpy(), direct.evaluable.to_numpy()
        )
        np.testing.assert_array_equal(
            result.violations[run.to_numpy()].to_numpy(), direct.violations.to_numpy()
        )
        assert not result.evaluable[~run.to_numpy()].to_numpy().any()

    def test_derivatives_never_cross_runs(
        self,
        partitions: dict[str, pd.DataFrame],
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """dT/dt is not evaluable at the first row of each run, even if run 2 jumps."""
        frame = partitions["test"]
        values = _measured(frame)
        first = first_of_run(frame)
        start_2 = int(np.flatnonzero(first)[1])
        for tag in (*ROLES["T_out"], *ROLES["T_in"]):
            values[start_2:, TAGS.index(tag)] -= 5_000.0 * limits.max_dt_c_per_s
        everything = np.ones(len(frame), dtype=np.bool_)
        result = evaluate_by_run(frame, values, everything, TAGS, ROLES, table, limits)
        assert not result.evaluable["max_dt_dt"][first].any()
        assert result.evaluable["max_dt_dt"][~first].all()
        assert not result.violations["max_dt_dt"].any()


class TestSummarize:
    """Counts of satisfied, violating and non-evaluable instants."""

    def test_counts_and_satisfaction(self) -> None:
        """One run start, one violation, one instant with no evaluable rule."""
        evaluation = _evaluation(
            violations=[NONE, NONE, _only("power_flux"), NONE, _only("boiling")],
            evaluable=[NONE, ALL, ALL, NONE, [r != "boiling" for r in RULE_NAMES]],
        )
        predicted = np.array([False, True, True, True, True])
        scope = np.ones(5, dtype=np.bool_)
        block = summarize(evaluation, scope, predicted)
        assert block["n_instants"] == 5
        assert block["n_not_evaluable"] == {"first_of_run": 1, "no_rule": 1}
        assert (block["n_evaluated"], block["n_satisfied"], block["n_violating"]) == (3, 2, 1)
        assert block["satisfaction"] == pytest.approx(2 / 3)
        assert block["rules"]["power_flux"] == {
            "n_evaluable": 3,
            "n_not_evaluated": 1,
            "n_violations": 1,
            "rate": pytest.approx(1 / 3),
        }
        # A violation flagged where the rule is not evaluable never counts.
        assert block["rules"]["boiling"]["n_violations"] == 0
        assert block["rules"]["boiling"]["n_evaluable"] == 2

    def test_scope_restricts_every_count(self) -> None:
        """Only the instants of the scope are counted."""
        evaluation = _evaluation(violations=[_only("energy_balance"), NONE], evaluable=[ALL, ALL])
        block = summarize(evaluation, np.array([False, True]), np.array([True, True]))
        assert (block["n_instants"], block["n_evaluated"], block["n_violating"]) == (1, 1, 0)
        assert block["satisfaction"] == 1.0

    def test_nothing_evaluated_gives_none(self) -> None:
        """0/0 is reported as None, never NaN."""
        evaluation = _evaluation(violations=[NONE], evaluable=[NONE])
        block = summarize(evaluation, np.array([True]), np.array([True]))
        assert block["satisfaction"] is None
        assert block["rules"]["energy_balance"]["rate"] is None


class TestPasses:
    """Criterion 1 is strict (> 99 %)."""

    @pytest.mark.parametrize(
        ("satisfaction", "expected"),
        [(1.0, True), (0.9901, True), (SATISFACTION_THRESHOLD, False), (0.5, False), (None, False)],
    )
    def test_threshold(self, satisfaction: float | None, expected: bool) -> None:
        """Exactly 99 % does not pass; nothing evaluated does not pass."""
        assert passes(satisfaction) is expected


class TestPhysicsDocument:
    """Prediction and measurement summarized on the same instants."""

    def test_violating_prediction_fails_and_counts_per_scenario(
        self,
        partitions: dict[str, pd.DataFrame],
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A prediction with T_out < T_in violates temperature ordering everywhere."""
        frame = partitions["test"]
        values = _measured(frame)
        for tag in ROLES["T_out"]:
            values[:, TAGS.index(tag)] = 200.0
        values[first_of_run(frame)] = np.nan
        document = physics_document(frame, values, TAGS, ROLES, table, limits, {"x": "y"})
        every = document["prediction"]["all"]
        assert every["rules"]["temperature_ordering"]["n_violations"] == every["n_evaluated"]
        assert every["satisfaction"] == 0.0
        assert document["passed"] is False
        assert document["satisfaction"] == 0.0
        scenario = document["prediction"]["by_scenario"]["normal"]
        assert scenario["n_violating"] == every["n_evaluated"]
        assert scenario["violations"]["temperature_ordering"] == every["n_evaluated"]
        assert document["inputs"] == {"x": "y"}


class TestRunPhysicsCheck:
    """End to end on the tiny model."""

    def test_writes_the_metrics(self, trained: Path, partitions: dict[str, pd.DataFrame]) -> None:
        """Counts follow the partition; the physical traces satisfy every rule."""
        params = load_pinn_physics_params(trained)
        outcome = run_physics_check(trained, params)
        assert outcome.metrics_path == params.physics_metrics_path
        document = json.loads(outcome.metrics_path.read_text(encoding="utf-8"))
        assert document == outcome.document
        frame = partitions["test"]
        n_normal = int((~frame["labelled"]).sum())
        assert document["instants"] == {"total": len(frame), "predicted": len(frame) - 2, "runs": 2}
        assert document["split"] == "test" and document["rule_mode"] == "voted"
        for source in ("prediction", "measurement"):
            every, normal = document[source]["all"], document[source]["normal"]
            assert every["n_not_evaluable"]["first_of_run"] == 2
            assert every["n_evaluated"] == len(frame) - 2
            assert normal["n_instants"] == n_normal
            assert normal["n_evaluated"] == n_normal - 2
            # dT/dt and the mass mask need t-1 of the same source.
            assert every["rules"]["max_dt_dt"]["n_evaluable"] <= every["n_evaluated"]
        assert document["measurement"]["all"]["satisfaction"] == 1.0
        assert document["measurement"]["all"]["rules"]["max_dt_dt"]["n_evaluable"] == len(frame) - 2
        assert document["prediction"]["all"]["rules"]["max_dt_dt"]["n_evaluable"] == len(frame) - 4
        assert set(document["inputs"]) == {MODEL_FILE, "test.parquet", "assignment.json"}
        assert document["passed"] is passes(document["satisfaction"])

    def test_output_is_byte_stable_without_clock_nor_paths(self, trained: Path) -> None:
        """Two runs write the same bytes, sorted keys, no tmp path, with 1 or 2 threads."""
        params = load_pinn_physics_params(trained)
        first = run_physics_check(trained, params).metrics_path.read_bytes()
        two = dataclasses.replace(params, num_threads=2)
        second = run_physics_check(trained, two).metrics_path.read_bytes()
        assert first == second
        text = first.decode("utf-8")
        assert str(trained.parent) not in text and trained.parent.as_posix() not in text
        assert "seconds" not in text and "time" not in text
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"

    def test_main_reads_the_params_file(self, trained: Path) -> None:
        """main() is a thin wrapper over the loader and run_physics_check."""
        outcome = main(trained)
        assert outcome.metrics_path == load_pinn_physics_params(trained).physics_metrics_path

    def test_missing_test_partition(self, trained: Path, tmp_path: Path) -> None:
        """Without test.parquet the stage asks for prepare_sim."""
        params = load_pinn_physics_params(trained)
        data_dir = tmp_path / "sim_pinn"
        data_dir.mkdir()
        (data_dir / "assignment.json").write_bytes(
            (params.data_dir / "assignment.json").read_bytes()
        )
        with pytest.raises(FileNotFoundError, match="prepare_sim"):
            run_physics_check(trained, dataclasses.replace(params, data_dir=data_dir))

    def test_missing_assignment(self, trained: Path, tmp_path: Path) -> None:
        """Without assignment.json the stage asks for prepare_sim."""
        params = load_pinn_physics_params(trained)
        with pytest.raises(FileNotFoundError, match="prepare_sim"):
            run_physics_check(trained, dataclasses.replace(params, data_dir=tmp_path))

    def test_missing_checkpoint(self, trained: Path, tmp_path: Path) -> None:
        """Without model.pt the stage fails before reading data."""
        params = load_pinn_physics_params(trained)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            run_physics_check(trained, dataclasses.replace(params, model_dir=tmp_path))

    def test_sensor_order_must_agree(self, trained: Path, tmp_path: Path) -> None:
        """An assignment with another sensor order is rejected."""
        params = load_pinn_physics_params(trained)
        data_dir = tmp_path / "sim_pinn"
        data_dir.mkdir()
        document = json.loads((params.data_dir / "assignment.json").read_text(encoding="utf-8"))
        document["tags"] = list(reversed(document["tags"]))
        (data_dir / "assignment.json").write_text(json.dumps(document), encoding="utf-8")
        (data_dir / "test.parquet").write_bytes((params.data_dir / "test.parquet").read_bytes())
        with pytest.raises(ValueError, match="sensor order"):
            run_physics_check(trained, dataclasses.replace(params, data_dir=data_dir))

    def test_script_entry_point(
        self, trained: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Running the module as a script calls main() on ./params.yaml."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "params.yaml").write_bytes(trained.read_bytes())
        monkeypatch.setattr("logging.basicConfig", lambda **_: None)
        runpy.run_path(physics_check.__file__, run_name="__main__")
        assert load_pinn_physics_params(trained).physics_metrics_path.exists()


# ----------------------------------------------------------------------
# Contrato con params.yaml y dvc.yaml
# ----------------------------------------------------------------------


def _stage(name: str) -> dict[str, Any]:
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    stage: dict[str, Any] = stages[name]
    return stage


class TestRepository:
    """The versioned params.yaml and dvc.yaml agree with physics_check."""

    def test_repository_params_resolve(self) -> None:
        """The stage reads the M3 checkpoint and the prepare_sim test."""
        params = load_pinn_physics_params(REPO_ROOT / "params.yaml")
        assert params == PinnPhysicsParams(
            data_dir=Path("data/processed/sim_pinn"),
            model_dir=Path("models/pinn/candidate"),
            num_threads=4,
            physics_metrics_path=Path("metrics/pinn_physics_metrics.json"),
        )

    def test_evaluate_declares_exactly_the_keys_it_reads(self) -> None:
        """pinn.* params = loader fields; simulation.* = the three tables read."""
        stage = _stage("evaluate")
        declared = {p.split(".", 1)[1] for p in stage["params"] if p.startswith("pinn.")}
        assert declared == set(PinnPhysicsParams.__dataclass_fields__)
        simulation = {p for p in stage["params"] if p.startswith("simulation.")}
        assert simulation == {
            "simulation.sensors_path",
            "simulation.plant_parameters_path",
            "simulation.water_table_path",
        }

    def test_evaluate_metrics_and_deps(self) -> None:
        """Only metrics (no outs); deps on every module imported and on the inputs."""
        stage = _stage("evaluate")
        params = load_pinn_physics_params(REPO_ROOT / "params.yaml")
        assert stage["cmd"] == "python ml/validation/physics_check.py"
        assert "outs" not in stage
        assert stage["metrics"] == [{params.physics_metrics_path.as_posix(): {"cache": False}}]
        expected = {
            "ml/validation/physics_check.py",
            "ml/training/checkpoint.py",
            "ml/training/pinn_data.py",
            "ml/training/pinn_params.py",
            "ml/training/sim_dataset.py",
            "ml/models",
            "data/simulation",
            "configs/sim_sensors.yaml",
            "configs/plant_parameters.yaml",
            "configs/water_table.yaml",
            params.model_dir.as_posix(),
            params.data_dir.as_posix(),
        }
        assert set(stage["deps"]) == expected
