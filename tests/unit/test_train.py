"""Unit tests for ml.training.train (stage train, F4-M3).

Los entrenamientos corren sobre las particiones pequenas de
`tests/support/sim_frames.py` (6 runs de 24 instantes sobre trazas fisicas) con una
red de 16 neuronas: cada uno tarda decimas de segundo. El loop se prueba ademas
con una evaluacion guionizada para fijar la parada temprana, el curriculum y la
restauracion de la mejor epoca sin depender de la optimizacion real. MLflow usa un
store sqlite temporal compartido por el modulo (crear el esquema cuesta ~3 s).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import runpy
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import yaml

from ml.models.pinn import ReactorPINN
from ml.training import train as train_module
from ml.training.checkpoint import CARD_FILE, MODEL_FILE, load_checkpoint
from ml.training.pinn_data import CONTEXT_NAMES
from ml.training.pinn_params import PinnTrainParams, load_pinn_train_params
from ml.training.train import (
    EpochRecord,
    LossParts,
    MlflowRunLogger,
    NullRunLogger,
    TrainingData,
    curriculum_factor,
    derived_seeds,
    deterministic_torch,
    fit,
    initialize,
    load_training_data,
    main,
    mlflow_store_dir,
    run_training,
)
from tests.support.sim_frames import (
    REPO_ROOT,
    pinn_partitions,
    write_pinn_partitions,
    write_pinn_train_params,
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the small partitions once per module."""
    return write_pinn_partitions(tmp_path_factory.mktemp("data") / "sim_pinn", pinn_partitions())


@pytest.fixture(scope="module")
def mlflow_uri(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a sqlite store shared by the MLflow tests of the module."""
    store = tmp_path_factory.mktemp("mlflow") / "mlruns" / "mlflow.db"
    return f"sqlite:///{store.as_posix()}"


def _params_file(work_dir: Path, data_dir: Path, **overrides: Any) -> Path:
    return write_pinn_train_params(work_dir, data_dir=str(data_dir), **overrides)


def _setup(work_dir: Path, data_dir: Path, **overrides: Any) -> tuple[Path, PinnTrainParams]:
    path = _params_file(work_dir, data_dir, **overrides)
    return path, load_pinn_train_params(path)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingLogger:
    """RunLogger that keeps everything it receives."""

    def __init__(self) -> None:
        self.params: dict[str, str] = {}
        self.epochs: list[EpochRecord] = []
        self.summary: dict[str, float] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[Path] = []

    def log_params(self, params: Mapping[str, str]) -> None:
        self.params.update(params)

    def log_epoch(self, record: EpochRecord) -> None:
        self.epochs.append(record)

    def log_summary(self, metrics: Mapping[str, float], tags: Mapping[str, str]) -> None:
        self.summary.update(metrics)
        self.tags.update(tags)

    def log_artifacts(self, paths: Sequence[Path]) -> None:
        self.artifacts.extend(paths)


# ----------------------------------------------------------------------
# Piezas sueltas
# ----------------------------------------------------------------------


class TestCurriculum:
    """Linear ramp 0 -> 1 over curriculum_epochs."""

    @pytest.mark.parametrize(
        ("epoch", "expected"), [(0, 0.0), (25, 0.5), (49, 0.98), (50, 1.0), (80, 1.0)]
    )
    def test_ramp_of_fifty(self, epoch: int, expected: float) -> None:
        """With the default 50 epochs the target is reached at epoch 50."""
        assert curriculum_factor(epoch, 50) == pytest.approx(expected)

    def test_no_ramp(self) -> None:
        """curriculum_epochs = 0 uses the full lambdas from the start."""
        assert curriculum_factor(0, 0) == 1.0


class TestSeeds:
    """Generators derived from pinn.seed."""

    def test_deterministic_and_distinct(self) -> None:
        """The same seed gives the same list; its members differ and fit in int64."""
        seeds = derived_seeds(42, 2)
        assert seeds == derived_seeds(42, 2)
        assert seeds[0] != seeds[1]
        assert seeds != derived_seeds(43, 2)
        assert all(0 <= s < 2**63 for s in seeds)

    def test_initialize_matches_the_pytorch_default(self) -> None:
        """Same draws as nn.Linear.reset_parameters, from a Generator."""
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            reference = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3))
        model = nn.Sequential(nn.Linear(5, 4), nn.Linear(4, 3))
        initialize(model, torch.Generator().manual_seed(7))  # type: ignore[arg-type]
        for ours, theirs in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(ours, theirs, rtol=0.0, atol=0.0)

    def test_initialize_ignores_the_global_generator(self) -> None:
        """Two global seeds, one Generator seed: the same weights."""
        weights = []
        for global_seed in (1, 2):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(global_seed)
                model = ReactorPINN(3, hidden=8, n_hidden=2)
                initialize(model, torch.Generator().manual_seed(5))
            weights.append(torch.cat([p.flatten() for p in model.parameters()]))
        torch.testing.assert_close(weights[0], weights[1], rtol=0.0, atol=0.0)


class TestDeterministicTorch:
    """The global torch state is fixed for the training and restored afterwards."""

    def test_state_inside_and_after(self) -> None:
        """Seed, deterministic mode and threads inside; previous values outside."""
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        before = torch.random.get_rng_state()
        with deterministic_torch(seed=3, num_threads=1):
            assert torch.get_num_threads() == 1
            assert torch.are_deterministic_algorithms_enabled()
            first = torch.rand(2)
        with deterministic_torch(seed=3, num_threads=1):
            second = torch.rand(2)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        assert torch.get_num_threads() == threads
        assert torch.are_deterministic_algorithms_enabled() is deterministic
        assert torch.equal(torch.random.get_rng_state(), before)


class TestMlflowStoreDir:
    """The store directory comes from the sqlite URI."""

    def test_relative_and_absolute(self) -> None:
        """mlruns/mlflow.db -> mlruns; /x/y/m.db -> /x/y."""
        assert mlflow_store_dir("sqlite:///mlruns/mlflow.db") == Path("mlruns")
        assert mlflow_store_dir("sqlite:////x/y/m.db") == Path("/x/y")

    @pytest.mark.parametrize("uri", ["mlruns", "sqlite:///", "file:///tmp/m.db"])
    def test_other_uris_are_rejected(self, uri: str) -> None:
        """Only sqlite:///<path> (D4-14)."""
        with pytest.raises(ValueError, match="sqlite"):
            mlflow_store_dir(uri)


# ----------------------------------------------------------------------
# Datos del stage
# ----------------------------------------------------------------------


class TestLoadTrainingData:
    """Train and val pairs with the train z-score."""

    def test_pairs_and_digests(self, tmp_path: Path, data_dir: Path) -> None:
        """Both partitions, the catalog order and the digests of the files read."""
        path, params = _setup(tmp_path, data_dir)
        data = load_training_data(path, params)
        assert len(data.train) == len(data.val) == 42
        assert data.stats.tags[0] == "TC-CORE-01"
        assert set(data.digests) == {"assignment.json", "train.parquet", "val.parquet"}
        assert all(len(d) == 64 for d in data.digests.values())

    def test_missing_assignment(self, tmp_path: Path) -> None:
        """Without prepare_sim there is nothing to train on."""
        path, params = _setup(tmp_path, tmp_path / "nothing")
        with pytest.raises(FileNotFoundError, match="prepare_sim"):
            load_training_data(path, params)

    def test_missing_partition(self, tmp_path: Path, data_dir: Path) -> None:
        """A partition removed by hand is reported."""
        partial = tmp_path / "sim_pinn"
        partial.mkdir()
        (partial / "assignment.json").write_bytes((data_dir / "assignment.json").read_bytes())
        (partial / "train.parquet").write_bytes((data_dir / "train.parquet").read_bytes())
        path, params = _setup(tmp_path, partial)
        with pytest.raises(FileNotFoundError, match="val.parquet"):
            load_training_data(path, params)

    def test_sensor_order_must_match_the_catalog(self, tmp_path: Path, data_dir: Path) -> None:
        """prepare_sim and the catalog must agree on the columns."""
        other = tmp_path / "sim_pinn"
        other.mkdir()
        for name in ("train.parquet", "val.parquet"):
            (other / name).write_bytes((data_dir / name).read_bytes())
        document = json.loads((data_dir / "assignment.json").read_text(encoding="utf-8"))
        document["tags"] = list(reversed(document["tags"]))
        (other / "assignment.json").write_text(json.dumps(document), encoding="utf-8")
        path, params = _setup(tmp_path, other)
        with pytest.raises(ValueError, match="sensor order"):
            load_training_data(path, params)

    def test_val_without_pairs(self, tmp_path: Path) -> None:
        """A partition whose instants are all labelled cannot be validated on."""
        frames = pinn_partitions()
        frames["val"] = frames["val"].assign(labelled=True, trainable_pair=False)
        empty = write_pinn_partitions(tmp_path / "sim_pinn", frames)
        path, params = _setup(tmp_path, empty)
        with pytest.raises(ValueError, match="trainable pairs"):
            load_training_data(path, params)


# ----------------------------------------------------------------------
# Loop con evaluacion guionizada
# ----------------------------------------------------------------------


@pytest.fixture()
def scripted(
    tmp_path: Path, data_dir: Path
) -> Iterator[tuple[TrainingData, list[dict[str, torch.Tensor]]]]:
    """Return the data and a list that receives the weights seen at each evaluation."""
    path, params = _setup(tmp_path, data_dir)
    data = load_training_data(path, params)
    seen: list[dict[str, torch.Tensor]] = []
    yield data, seen


def _script(
    monkeypatch: pytest.MonkeyPatch,
    values: Sequence[float],
    seen: list[dict[str, torch.Tensor]],
) -> None:
    """Make evaluate return val_data = values[i] (physics 0) and record the weights."""
    calls = iter(values)

    def fake(model: ReactorPINN, pairs: object, physics: object) -> LossParts:
        seen.append({k: v.detach().clone() for k, v in model.state_dict().items()})
        return LossParts(data=next(calls), energy=0.0, mass=0.0)

    monkeypatch.setattr(train_module, "evaluate", fake)


def _fit(data: TrainingData, tmp_path: Path, data_dir: Path, **overrides: Any) -> Any:
    _, params = _setup(tmp_path, data_dir, **overrides)
    model = ReactorPINN(data.train.n_sensors, hidden=16, n_hidden=2, dropout=0.1)
    with deterministic_torch(params.seed, params.num_threads):
        result = fit(model, data, params, NullRunLogger())
    return model, result


class TestFitScripted:
    """Early stopping, curriculum gating and best-epoch restore."""

    def test_patience_counts_only_after_the_curriculum(
        self,
        scripted: tuple[TrainingData, list[dict[str, torch.Tensor]]],
        tmp_path: Path,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No improvement after epoch 0: epochs 1-2 (ramp) do not count, 3-4 do."""
        data, seen = scripted
        _script(monkeypatch, [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], seen)
        model, result = _fit(
            data, tmp_path, data_dir, curriculum_epochs=3, early_stopping_patience=2, max_epochs=8
        )
        assert result.epochs_run == 5
        assert result.stopped_early is True
        assert result.best_epoch == 0
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, seen[0][key], rtol=0.0, atol=0.0)

    def test_best_epoch_weights_are_restored(
        self,
        scripted: tuple[TrainingData, list[dict[str, torch.Tensor]]],
        tmp_path: Path,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The minimum of val_total wins even if later epochs are worse."""
        data, seen = scripted
        _script(monkeypatch, [3.0, 1.0, 2.0, 2.5], seen)
        model, result = _fit(
            data, tmp_path, data_dir, curriculum_epochs=0, early_stopping_patience=5, max_epochs=4
        )
        assert (result.epochs_run, result.best_epoch, result.stopped_early) == (4, 1, False)
        assert result.best.val.data == 1.0
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, seen[1][key], rtol=0.0, atol=0.0)

    def test_stopping_on_the_last_epoch_is_not_early(
        self,
        scripted: tuple[TrainingData, list[dict[str, torch.Tensor]]],
        tmp_path: Path,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Patience exhausted exactly at max_epochs is a normal end."""
        data, seen = scripted
        _script(monkeypatch, [1.0, 2.0, 2.0], seen)
        _, result = _fit(
            data, tmp_path, data_dir, curriculum_epochs=0, early_stopping_patience=2, max_epochs=3
        )
        assert (result.epochs_run, result.stopped_early) == (3, False)

    def test_learning_rate_is_reduced_on_plateau(
        self,
        scripted: tuple[TrainingData, list[dict[str, torch.Tensor]]],
        tmp_path: Path,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """lr_patience = 1, factor 0.5: two flat epochs halve the rate, never below min_lr."""
        data, seen = scripted
        _script(monkeypatch, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], seen)
        _, result = _fit(
            data,
            tmp_path,
            data_dir,
            curriculum_epochs=0,
            early_stopping_patience=10,
            max_epochs=6,
            learning_rate=1.0e-2,
            lr_patience=1,
            min_lr=4.0e-3,
        )
        assert [r.lr for r in result.history] == pytest.approx(
            [1.0e-2, 1.0e-2, 1.0e-2, 5.0e-3, 5.0e-3, 4.0e-3]
        )


# ----------------------------------------------------------------------
# Entrenamiento real (pequeno)
# ----------------------------------------------------------------------


class TestRunTraining:
    """End to end without MLflow (NullRunLogger / RecordingLogger)."""

    def test_writes_checkpoint_card_and_metrics(self, tmp_path: Path, data_dir: Path) -> None:
        """model.pt loads with weights_only and predicts; card and metrics are consistent."""
        path, params = _setup(tmp_path, data_dir)
        outcome = run_training(path, params, NullRunLogger())
        assert outcome.model_path == params.model_dir / MODEL_FILE
        assert outcome.card_path == params.model_dir / CARD_FILE
        loaded = load_checkpoint(outcome.model_path)
        x = torch.zeros(2, loaded.model.n_inputs)
        torch.testing.assert_close(loaded.model(x), outcome.model(x))

        card = json.loads(outcome.card_path.read_text(encoding="utf-8"))
        assert card["model"]["sha256"] == _digest(outcome.model_path)
        assert card["model"]["kwargs"]["hidden"] == 16
        assert card["inputs"]["context_names"] == list(CONTEXT_NAMES)
        assert card["result"]["epochs_run"] == outcome.result.epochs_run
        assert "mlflow_tracking_uri" not in card["training"]

        metrics = json.loads(outcome.metrics_path.read_text(encoding="utf-8"))
        assert metrics["pairs"] == {"train": 42, "val": 42}
        assert metrics["best_epoch"] == outcome.result.best_epoch
        assert metrics["best"]["val_data"] == pytest.approx(outcome.result.best.val.data)
        baselines = metrics["baselines"]
        assert baselines["val_data_over_persistence"] == pytest.approx(
            metrics["best"]["val_data"] / baselines["val_persistence_mse"], rel=1e-9
        )

    def test_outputs_have_no_clock_nor_absolute_paths(self, tmp_path: Path, data_dir: Path) -> None:
        """Convention 5: the card and the metrics do not mention tmp_path nor time."""
        path, params = _setup(tmp_path, data_dir)
        outcome = run_training(path, params, NullRunLogger())
        for output in (outcome.card_path, outcome.metrics_path):
            text = output.read_text(encoding="utf-8")
            assert str(tmp_path) not in text
            assert tmp_path.as_posix() not in text
            assert "seconds" not in text and "time" not in text
            assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"

    def test_card_records_the_data_dir_as_given(self, tmp_path: Path, data_dir: Path) -> None:
        """A relative data_dir stays relative in the card."""
        path, params = _setup(tmp_path, data_dir)
        model = ReactorPINN(27, hidden=4, n_hidden=1)
        relative = dataclasses.replace(params, data_dir=Path("data/processed/sim_pinn"))
        card = train_module.model_card(
            model,
            train_module.FitResult(history=[_record(0)], best_epoch=0),
            load_training_data(path, params),
            relative,
            "0" * 64,
        )
        assert card["training"]["data_dir"] == "data/processed/sim_pinn"
        assert card["model"]["n_parameters"] == sum(p.numel() for p in model.parameters())

    def test_curriculum_and_logged_epochs(self, tmp_path: Path, data_dir: Path) -> None:
        """Epoch 0 trains on data only; val_total uses the target lambdas from the start."""
        path, params = _setup(tmp_path, data_dir, max_epochs=3, early_stopping_patience=5)
        logger = RecordingLogger()
        outcome = run_training(path, params, logger)
        history = outcome.result.history
        assert [r.lambda_factor for r in history] == [0.0, 0.5, 1.0]
        assert logger.epochs == history
        first = history[0]
        assert first.train_total == first.train.data
        assert first.val_total_curriculum == first.val.data
        assert first.val_total == pytest.approx(first.val.data + first.val.energy + first.val.mass)
        assert first.physics_share == pytest.approx(
            (first.val.energy + first.val.mass) / first.val.data
        )
        assert set(first.metrics()) >= {
            "data",
            "energy",
            "mass",
            "total",
            "val_data",
            "val_energy",
            "val_mass",
            "val_total",
            "lr",
        }
        assert all(r.val.energy > 0.0 and r.val.mass > 0.0 for r in history)

    def test_logger_receives_params_summary_and_artifacts(
        self, tmp_path: Path, data_dir: Path
    ) -> None:
        """Params without the MLflow URI; the clock and the host only in the summary."""
        path, params = _setup(tmp_path, data_dir)
        logger = RecordingLogger()
        outcome = run_training(path, params, logger)
        assert "mlflow_tracking_uri" not in logger.params
        assert logger.params["seed"] == "42"
        assert logger.params["n_train_pairs"] == "42"
        assert logger.summary["train_seconds"] == pytest.approx(outcome.seconds)
        assert logger.summary["epochs_run"] == outcome.result.epochs_run
        assert logger.tags["model_sha256"] == _digest(outcome.model_path)
        assert "host" in logger.tags
        assert logger.artifacts == [outcome.card_path, outcome.metrics_path]

    def test_two_runs_give_the_same_bytes(self, tmp_path: Path, data_dir: Path) -> None:
        """H5 in process: same params -> identical model.pt, card and metrics."""
        digests = []
        for name in ("a", "b"):
            path, params = _setup(tmp_path / name, data_dir)
            outcome = run_training(path, params, NullRunLogger())
            digests.append(
                [_digest(p) for p in (outcome.model_path, outcome.card_path, outcome.metrics_path)]
            )
        assert digests[0] == digests[1]

    def test_another_seed_gives_other_weights(self, tmp_path: Path, data_dir: Path) -> None:
        """The seed reaches the initialisation."""
        digests = []
        for seed in (1, 2):
            path, params = _setup(tmp_path / str(seed), data_dir, seed=seed)
            digests.append(_digest(run_training(path, params, NullRunLogger()).model_path))
        assert digests[0] != digests[1]

    def test_training_reduces_the_validation_error(self, tmp_path: Path, data_dir: Path) -> None:
        """The best val_data beats the first epoch and the persistence baseline."""
        path, params = _setup(
            tmp_path, data_dir, max_epochs=30, early_stopping_patience=30, curriculum_epochs=5
        )
        outcome = run_training(path, params, NullRunLogger())
        metrics = json.loads(outcome.metrics_path.read_text(encoding="utf-8"))
        assert outcome.result.best.val.data < outcome.result.history[0].val.data
        assert metrics["baselines"]["val_data_over_persistence"] < 1.0


def _record(epoch: int) -> EpochRecord:
    parts = LossParts(data=1.0, energy=0.0, mass=0.0)
    return EpochRecord(
        epoch=epoch,
        lr=1.0e-3,
        lambda_factor=1.0,
        train=parts,
        train_total=1.0,
        val=parts,
        val_total=1.0,
        val_total_curriculum=1.0,
        physics_share=0.0,
    )


class TestLossParts:
    """Weighted total."""

    def test_total(self) -> None:
        """data + lambda_e energy + lambda_m mass."""
        assert LossParts(1.0, 2.0, 3.0).total(10.0, 100.0) == 321.0


# ----------------------------------------------------------------------
# MLflow y punto de entrada
# ----------------------------------------------------------------------


class TestMlflow:
    """main() logs to an explicit sqlite store and never creates mlflow.db in the cwd."""

    def test_main_logs_one_finished_run(
        self,
        tmp_path: Path,
        data_dir: Path,
        mlflow_uri: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Params, one metric point per epoch and artifact, status FINISHED."""
        from mlflow.tracking import MlflowClient

        path = _params_file(tmp_path, data_dir, mlflow_tracking_uri=mlflow_uri)
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        outcome = main(path)
        assert list(cwd.iterdir()) == []

        client = MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name("test-pinn")
        assert experiment is not None
        assert experiment.artifact_location.endswith("/mlruns/artifacts")
        runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
        run = runs[0]
        assert run.info.status == "FINISHED"
        assert run.data.params["hidden"] == "16"
        assert "mlflow_tracking_uri" not in run.data.params
        history = client.get_metric_history(run.info.run_id, "val_data")
        assert [m.step for m in history] == list(range(outcome.result.epochs_run))
        assert run.data.metrics["best_epoch"] == outcome.result.best_epoch
        names = {a.path for a in client.list_artifacts(run.info.run_id)}
        assert names == {CARD_FILE, "pinn_train_metrics.json"}

    def test_second_run_reuses_the_experiment(
        self, tmp_path: Path, data_dir: Path, mlflow_uri: str
    ) -> None:
        """The experiment is created once; each run is a new MLflow run."""
        from mlflow.tracking import MlflowClient

        first = MlflowRunLogger(mlflow_uri, "test-reuse", "a")
        second = MlflowRunLogger(mlflow_uri, "test-reuse", "b")
        first.finish("FINISHED")
        second.finish("FINISHED")
        client = MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name("test-reuse")
        assert experiment is not None
        assert len(client.search_runs([experiment.experiment_id])) == 2

    def test_failed_training_marks_the_run_failed(self, tmp_path: Path, mlflow_uri: str) -> None:
        """An exception closes the MLflow run as FAILED and propagates."""
        from mlflow.tracking import MlflowClient

        path = _params_file(
            tmp_path, tmp_path / "missing", mlflow_tracking_uri=mlflow_uri, mlflow_experiment="f"
        )
        with pytest.raises(FileNotFoundError):
            main(path)
        client = MlflowClient(tracking_uri=mlflow_uri)
        experiment = client.get_experiment_by_name("f")
        assert experiment is not None
        statuses = [r.info.status for r in client.search_runs([experiment.experiment_id])]
        assert statuses == ["FAILED"]

    def test_runs_as_a_script(
        self,
        tmp_path: Path,
        data_dir: Path,
        mlflow_uri: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`python ml/training/train.py` reads params.yaml from the working directory."""
        _params_file(tmp_path, data_dir, mlflow_tracking_uri=mlflow_uri, max_epochs=1)
        monkeypatch.chdir(tmp_path)
        runpy.run_path(str(REPO_ROOT / "ml" / "training" / "train.py"), run_name="__main__")
        assert (tmp_path / "models" / "pinn" / "candidate" / MODEL_FILE).exists()
        assert not (tmp_path / "mlflow.db").exists()


# ----------------------------------------------------------------------
# Contrato con params.yaml y dvc.yaml
# ----------------------------------------------------------------------


def _stage(name: str) -> dict[str, Any]:
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    stage: dict[str, Any] = stages[name]
    return stage


class TestRepository:
    """The versioned params.yaml, dvc.yaml and .gitignore agree with the stage."""

    def test_repository_params_resolve(self) -> None:
        """The approved values of F4-M3."""
        params = load_pinn_train_params(REPO_ROOT / "params.yaml")
        assert (params.hidden, params.n_hidden, params.dropout) == (256, 3, 0.1)
        assert (params.lambda_energy, params.lambda_mass, params.curriculum_epochs) == (
            1.0,
            1.0,
            50,
        )
        assert params.mlflow_tracking_uri == "sqlite:///mlruns/mlflow.db"
        assert params.model_dir == Path("models/pinn/candidate")

    def test_training_section_keeps_only_the_split_keys(self) -> None:
        """D4-9: the scaffold keys and the seed are gone; the split stage is untouched."""
        document = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
        assert set(document["training"]) == {
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "splits_dir",
            "metrics_path",
            "distribution_path",
        }
        split_params = {p for p in _stage("split")["params"] if p.startswith("training.")}
        assert {p.split(".", 1)[1] for p in split_params} <= set(document["training"])

    def test_train_declares_every_key_it_reads_but_the_uri(self) -> None:
        """pinn.* params = loader fields minus mlflow_tracking_uri (D4-14)."""
        stage = _stage("train")
        declared = {p.split(".", 1)[1] for p in stage["params"] if p.startswith("pinn.")}
        assert declared == set(PinnTrainParams.__dataclass_fields__) - {"mlflow_tracking_uri"}
        simulation = {p for p in stage["params"] if p.startswith("simulation.")}
        assert simulation == {
            "simulation.sensors_path",
            "simulation.plant_parameters_path",
            "simulation.water_table_path",
        }

    def test_train_outs_and_deps(self) -> None:
        """Outs where params.yaml says; deps on every module the stage imports."""
        stage = _stage("train")
        params = load_pinn_train_params(REPO_ROOT / "params.yaml")
        assert stage["cmd"] == "python ml/training/train.py"
        assert stage["outs"] == [params.model_dir.as_posix()]
        assert stage["metrics"] == [{params.train_metrics_path.as_posix(): {"cache": False}}]
        imported = {
            "ml/training/train.py",
            "ml/training/pinn_data.py",
            "ml/training/checkpoint.py",
            "ml/training/pinn_params.py",
            "ml/training/sim_dataset.py",
            "ml/models",
            "data/simulation",
            params.data_dir.as_posix(),
        }
        assert imported <= set(stage["deps"])

    def test_models_and_mlflow_store_are_ignored(self) -> None:
        """models/ (out of DVC) and mlruns/ (MLflow) never reach git."""
        lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "models/" in lines
        assert "mlruns/" in lines
        assert "mlflow.db" in lines


def test_mlflow_uri_is_never_relative_to_the_root() -> None:
    """The versioned URI points inside mlruns/, not to ./mlflow.db."""
    params = load_pinn_train_params(REPO_ROOT / "params.yaml")
    assert mlflow_store_dir(params.mlflow_tracking_uri) == Path("mlruns")
