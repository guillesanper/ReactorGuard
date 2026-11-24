"""Unit tests for ml.training.pinn_params.

Este modulo es lo que hace efectivos los `params:` de pinn: en dvc.yaml. Los tests
cubren que la seccion real de params.yaml se resuelva, que cada clave que el stage
prepare_sim declara exista y la lea el loader, y que cada valor fuera de rango falle
al cargar con un mensaje que nombre la clave.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ml.training.pinn_params import (
    DEFAULT_PARAMS_PATH,
    PinnParams,
    PinnPhysicsParams,
    PinnTrainParams,
    load_pinn_params,
    load_pinn_physics_params,
    load_pinn_train_params,
)
from tests.support.sim_frames import REPO_ROOT, pinn_train_section


def pinn_section(**overrides: Any) -> dict[str, Any]:
    """Return a valid pinn: section with some keys replaced."""
    section: dict[str, Any] = {
        "data_dir": "data/processed/sim_pinn",
        "metrics_path": "metrics/sim_pinn_metrics.json",
        "val_runs_per_scenario": 1,
        "test_runs_per_scenario": 1,
    }
    section.update(overrides)
    return section


def _write(path: Path, document: Any) -> Path:
    """Write a params file holding the given document."""
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _load(tmp_path: Path, **overrides: Any) -> PinnParams:
    """Load a pinn: section with some keys replaced."""
    return load_pinn_params(_write(tmp_path / "p.yaml", {"pinn": pinn_section(**overrides)}))


class TestLoadValidParams:
    """A valid section resolves to typed values."""

    def test_returns_typed_params(self, tmp_path: Path) -> None:
        """Paths become Path and counts int."""
        params = _load(tmp_path)
        assert params == PinnParams(
            data_dir=Path("data/processed/sim_pinn"),
            metrics_path=Path("metrics/sim_pinn_metrics.json"),
            val_runs_per_scenario=1,
            test_runs_per_scenario=1,
        )

    def test_integral_float_is_accepted(self, tmp_path: Path) -> None:
        """A YAML 2.0 is the integer 2."""
        params = _load(tmp_path, val_runs_per_scenario=2.0)
        assert params.val_runs_per_scenario == 2
        assert isinstance(params.val_runs_per_scenario, int)

    def test_min_runs_per_scenario_keeps_one_train_run(self, tmp_path: Path) -> None:
        """A scenario needs val + test + 1 runs."""
        params = _load(tmp_path, val_runs_per_scenario=2, test_runs_per_scenario=3)
        assert params.min_runs_per_scenario == 6

    def test_unrelated_sections_are_ignored(self, tmp_path: Path) -> None:
        """Only pinn: is read."""
        path = _write(tmp_path / "p.yaml", {"pinn": pinn_section(), "training": {"seed": 1}})
        assert load_pinn_params(path).test_runs_per_scenario == 1


class TestMissingInput:
    """Missing files, sections and keys fail loudly."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing params file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Params file not found"):
            load_pinn_params(tmp_path / "absent.yaml")

    def test_missing_section(self, tmp_path: Path) -> None:
        """A file without pinn: names the section."""
        path = _write(tmp_path / "p.yaml", {"training": {"seed": 42}})
        with pytest.raises(KeyError, match="'pinn:'"):
            load_pinn_params(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty file is a missing section."""
        path = tmp_path / "p.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(KeyError, match="'pinn:'"):
            load_pinn_params(path)

    def test_section_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A scalar section is rejected."""
        path = _write(tmp_path / "p.yaml", {"pinn": "data"})
        with pytest.raises(TypeError, match="must be a mapping"):
            load_pinn_params(path)

    @pytest.mark.parametrize(
        "key", ["data_dir", "metrics_path", "val_runs_per_scenario", "test_runs_per_scenario"]
    )
    def test_missing_key_names_the_key(self, tmp_path: Path, key: str) -> None:
        """The error carries the full key."""
        section = pinn_section()
        del section[key]
        path = _write(tmp_path / "p.yaml", {"pinn": section})
        with pytest.raises(KeyError, match=f"'pinn.{key}'"):
            load_pinn_params(path)


class TestValueRanges:
    """Out-of-range values fail at load time naming the key."""

    @pytest.mark.parametrize("value", [0, -1, 1.5, "two", None, True, float("inf"), [1]])
    @pytest.mark.parametrize("key", ["val_runs_per_scenario", "test_runs_per_scenario"])
    def test_counts_must_be_positive_integers(self, tmp_path: Path, key: str, value: Any) -> None:
        """Zero, negatives, fractions, booleans and non-numbers are rejected."""
        with pytest.raises(ValueError, match=f"'pinn.{key}' must be a positive integer"):
            _load(tmp_path, **{key: value})

    @pytest.mark.parametrize("value", ["", "   ", 3, None])
    @pytest.mark.parametrize("key", ["data_dir", "metrics_path"])
    def test_paths_must_be_non_empty_strings(self, tmp_path: Path, key: str, value: Any) -> None:
        """An empty or non-string path is rejected."""
        with pytest.raises(ValueError, match=f"'pinn.{key}' must be a non-empty path"):
            _load(tmp_path, **{key: value})

    def test_metrics_cannot_live_inside_the_data_dir(self, tmp_path: Path) -> None:
        """Two DVC outs cannot nest."""
        with pytest.raises(ValueError, match="no pueden anidarse"):
            _load(tmp_path, metrics_path="data/processed/sim_pinn/metrics.json")

    def test_sibling_directory_with_common_prefix_is_not_nested(self, tmp_path: Path) -> None:
        """sim_pinn_metrics is a sibling of sim_pinn, not a child."""
        params = _load(tmp_path, metrics_path="data/processed/sim_pinn_metrics.json")
        assert params.metrics_path == Path("data/processed/sim_pinn_metrics.json")


class TestRepositoryParams:
    """The versioned params.yaml and dvc.yaml agree with the loader."""

    def test_repository_section_resolves(self) -> None:
        """params.yaml carries every key the loader requires (D4-7: 1 val + 1 test run)."""
        params = load_pinn_params(REPO_ROOT / DEFAULT_PARAMS_PATH)
        assert params.data_dir == Path("data/processed/sim_pinn")
        assert params.metrics_path == Path("metrics/sim_pinn_metrics.json")
        assert (params.val_runs_per_scenario, params.test_runs_per_scenario) == (1, 1)

    def test_prepare_sim_declares_exactly_the_keys_it_reads(self) -> None:
        """Every pinn.* param of the stage is a loader field and vice versa."""
        stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
        stage = stages["prepare_sim"]
        declared = {p.split(".", 1)[1] for p in stage["params"] if p.startswith("pinn.")}
        assert declared == set(PinnParams.__dataclass_fields__)

    def test_prepare_sim_outs_match_the_params(self) -> None:
        """The stage writes where params.yaml says."""
        stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
        stage = stages["prepare_sim"]
        params = load_pinn_params(REPO_ROOT / DEFAULT_PARAMS_PATH)
        assert stage["outs"] == [params.data_dir.as_posix()]
        assert [next(iter(m)) for m in stage["metrics"]] == [params.metrics_path.as_posix()]


# ----------------------------------------------------------------------
# Stage train (F4-M3): load_pinn_train_params
# ----------------------------------------------------------------------


def _train_section(**overrides: Any) -> dict[str, Any]:
    """Return a valid pinn: section for the train stage with relative paths."""
    section = pinn_train_section(Path("work"))
    section.update(
        data_dir="data/processed/sim_pinn",
        model_dir="models/pinn/candidate",
        train_metrics_path="metrics/pinn_train_metrics.json",
        mlflow_tracking_uri="sqlite:///mlruns/mlflow.db",
    )
    section.update(overrides)
    return section


def _load_train(tmp_path: Path, **overrides: Any) -> PinnTrainParams:
    path = _write(tmp_path / "p.yaml", {"pinn": _train_section(**overrides)})
    return load_pinn_train_params(path)


class TestTrainParams:
    """The keys of the train stage resolve and fail naming the key."""

    def test_typed_values(self, tmp_path: Path) -> None:
        """Paths are Path, counts int, rates float, the URI a string."""
        params = _load_train(tmp_path, seed=0, curriculum_epochs=0, lr_patience=0.0)
        assert params.data_dir == Path("data/processed/sim_pinn")
        assert params.model_dir == Path("models/pinn/candidate")
        assert (params.seed, params.curriculum_epochs, params.lr_patience) == (0, 0, 0)
        assert isinstance(params.lr_patience, int)
        assert params.learning_rate == 1.0e-2
        assert params.mlflow_tracking_uri == "sqlite:///mlruns/mlflow.db"

    def test_prepare_sim_loader_ignores_the_train_keys(self, tmp_path: Path) -> None:
        """Both views read the same section without interfering."""
        path = _write(tmp_path / "p.yaml", {"pinn": _train_section()})
        assert load_pinn_params(path).val_runs_per_scenario == 1

    @pytest.mark.parametrize("key", sorted(PinnTrainParams.__dataclass_fields__))
    def test_missing_key_names_the_key(self, tmp_path: Path, key: str) -> None:
        """Every field is required."""
        section = _train_section()
        del section[key]
        path = _write(tmp_path / "p.yaml", {"pinn": section})
        with pytest.raises(KeyError, match=f"'pinn.{key}'"):
            load_pinn_train_params(path)

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("num_threads", 0, "positive integer"),
            ("hidden", 2.5, "positive integer"),
            ("batch_size", True, "positive integer"),
            ("seed", -1, "non-negative integer"),
            ("curriculum_epochs", 1.5, "non-negative integer"),
            ("lr_patience", "two", "non-negative integer"),
            ("seed", float("inf"), "non-negative integer"),
            ("dropout", 1.0, r"in \[0, 1\)"),
            ("dropout", -0.1, r"in \[0, 1\)"),
            ("lr_factor", 0.0, r"in \(0, 1\)"),
            ("lr_factor", 1.0, r"in \(0, 1\)"),
            ("learning_rate", 0.0, "> 0"),
            ("min_lr", -1.0e-6, "> 0"),
            ("weight_decay", -0.1, ">= 0"),
            ("lambda_energy", -1.0, ">= 0"),
            ("lambda_mass", float("nan"), "finite"),
            ("learning_rate", "fast", "a number"),
            ("learning_rate", True, "a number"),
            ("mlflow_experiment", "", "non-empty string"),
            ("mlflow_tracking_uri", 3, "non-empty string"),
            ("mlflow_tracking_uri", "mlruns", "sqlite"),
            ("mlflow_tracking_uri", "sqlite:///", "sqlite"),
            ("model_dir", "", "non-empty path"),
        ],
    )
    def test_out_of_range_values(self, tmp_path: Path, key: str, value: Any, message: str) -> None:
        """Each invalid value is rejected naming the key."""
        with pytest.raises(ValueError, match=f"'pinn.{key}'.*{message}"):
            _load_train(tmp_path, **{key: value})

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("model_dir", "data/processed/sim_pinn/model"),
            ("train_metrics_path", "models/pinn/candidate/metrics.json"),
        ],
    )
    def test_outs_cannot_nest(self, tmp_path: Path, key: str, value: str) -> None:
        """model_dir, train_metrics_path and data_dir are distinct DVC outs."""
        with pytest.raises(ValueError, match="no pueden anidarse"):
            _load_train(tmp_path, **{key: value})


# ----------------------------------------------------------------------
# Stage evaluate (F4-M4): load_pinn_physics_params
# ----------------------------------------------------------------------


def _physics_section(**overrides: Any) -> dict[str, Any]:
    """Return a valid pinn: section for physics_check with relative paths."""
    section = _train_section(physics_metrics_path="metrics/pinn_physics_metrics.json")
    section.update(overrides)
    return section


def _load_physics(tmp_path: Path, **overrides: Any) -> PinnPhysicsParams:
    path = _write(tmp_path / "p.yaml", {"pinn": _physics_section(**overrides)})
    return load_pinn_physics_params(path)


class TestPhysicsParams:
    """The keys of physics_check resolve and fail naming the key."""

    def test_typed_values(self, tmp_path: Path) -> None:
        """Paths are Path and the thread count int."""
        assert _load_physics(tmp_path, num_threads=2.0) == PinnPhysicsParams(
            data_dir=Path("data/processed/sim_pinn"),
            model_dir=Path("models/pinn/candidate"),
            num_threads=2,
            physics_metrics_path=Path("metrics/pinn_physics_metrics.json"),
        )

    def test_other_views_ignore_the_physics_key(self, tmp_path: Path) -> None:
        """prepare_sim and train read the same section without interfering."""
        path = _write(tmp_path / "p.yaml", {"pinn": _physics_section()})
        assert load_pinn_params(path).test_runs_per_scenario == 1
        assert load_pinn_train_params(path).hidden == 16

    @pytest.mark.parametrize("key", sorted(PinnPhysicsParams.__dataclass_fields__))
    def test_missing_key_names_the_key(self, tmp_path: Path, key: str) -> None:
        """Every field is required."""
        section = _physics_section()
        del section[key]
        path = _write(tmp_path / "p.yaml", {"pinn": section})
        with pytest.raises(KeyError, match=f"'pinn.{key}'"):
            load_pinn_physics_params(path)

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("num_threads", 0, "positive integer"),
            ("num_threads", 1.5, "positive integer"),
            ("physics_metrics_path", "", "non-empty path"),
            ("model_dir", 3, "non-empty path"),
        ],
    )
    def test_out_of_range_values(self, tmp_path: Path, key: str, value: Any, message: str) -> None:
        """Each invalid value is rejected naming the key."""
        with pytest.raises(ValueError, match=f"'pinn.{key}'.*{message}"):
            _load_physics(tmp_path, **{key: value})

    @pytest.mark.parametrize(
        "value", ["models/pinn/candidate/physics.json", "data/processed/sim_pinn/physics.json"]
    )
    def test_metrics_cannot_nest(self, tmp_path: Path, value: str) -> None:
        """The metrics file lives outside the model and the data outs."""
        with pytest.raises(ValueError, match="no pueden anidarse"):
            _load_physics(tmp_path, physics_metrics_path=value)
