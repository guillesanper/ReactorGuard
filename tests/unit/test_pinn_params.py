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

from ml.training.pinn_params import DEFAULT_PARAMS_PATH, PinnParams, load_pinn_params
from tests.support.sim_frames import REPO_ROOT


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
