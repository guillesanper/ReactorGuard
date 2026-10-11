"""Unit tests for data.simulation.simulation_params.

Este modulo es lo que hace efectivos los `params:` de simulation: en dvc.yaml. Los
tests cubren que la seccion real de params.yaml se resuelva completa y que cada
valor fuera de rango (tasa, pesos, longitudes, limites de carga) falle al cargar
con un mensaje que nombre la clave.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.simulation.simulation_params import (
    DEFAULT_PARAMS_PATH,
    SEVERITY_KEYS,
    SimulationParams,
    load_simulation_params,
)
from tests.support.sim_frames import REPO_ROOT, sim_section


def _write(path: Path, section: Any) -> Path:
    """Write a params file with the given simulation: section."""
    path.write_text(yaml.safe_dump({"simulation": section}), encoding="utf-8")
    return path


@pytest.fixture()
def section(tmp_path: Path) -> dict[str, Any]:
    """Return a valid simulation: section."""
    return sim_section(tmp_path)


def _load(tmp_path: Path, section: dict[str, Any], **changes: Any) -> SimulationParams:
    """Load a section with some keys replaced."""
    return load_simulation_params(_write(tmp_path / "p.yaml", {**section, **changes}))


class TestLoadValidParams:
    """A complete section resolves into a fully typed SimulationParams."""

    def test_returns_simulation_params(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """The loader returns a SimulationParams with the section's values."""
        params = _load(tmp_path, section)
        assert isinstance(params, SimulationParams)
        assert params.n_samples == 90
        assert params.fault_injection_rate == 0.40
        assert params.openmc_seed == 7
        assert params.dt_seconds == 180.0
        assert params.plant_id == "SIM-TEST-01"

    def test_paths_and_dates_are_typed(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """Paths become Path and ISO dates become date."""
        params = _load(tmp_path, section)
        assert isinstance(params.spans_path, Path)
        assert params.output_dir == Path(section["output_dir"])
        assert params.calibration_date == date(2025, 6, 1)
        assert params.last_maintenance_date == date(2025, 12, 1)

    def test_start_time_is_timezone_aware(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """start_time parses into an aware datetime."""
        start = _load(tmp_path, section).start_time
        assert isinstance(start, datetime)
        assert start.utcoffset() is not None

    def test_unquoted_yaml_values_are_accepted(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """PyYAML parses unquoted dates and timestamps into date objects."""
        path = tmp_path / "p.yaml"
        text = yaml.safe_dump({"simulation": section})
        text = text.replace("'2025-06-01'", "2025-06-01").replace(
            "'2026-01-01T00:00:00+00:00'", "2026-01-01T00:00:00+00:00"
        )
        path.write_text(text, encoding="utf-8")
        params = load_simulation_params(path)
        assert params.calibration_date == date(2025, 6, 1)
        assert params.start_time.utcoffset() is not None

    def test_severity_weights_keep_the_four_levels(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """The weights are keyed by the four severities and sum to 1."""
        weights = _load(tmp_path, section).severity_weights
        assert tuple(weights) == SEVERITY_KEYS
        assert math.isclose(sum(weights.values()), 1.0)


class TestDerivedQuantities:
    """The properties derive run counts and durations from the section."""

    def test_runs_are_rounded_up_to_whole_runs(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """n_samples that is not a multiple of the run length rounds the run count up."""
        params = _load(tmp_path, section, n_samples=100, run_length_steps=30)
        assert params.n_runs == 4
        assert params.total_steps == 120

    def test_exact_multiple_does_not_add_a_run(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """A multiple of the run length gives exactly n_samples / run_length runs."""
        params = _load(tmp_path, section, n_samples=90, run_length_steps=30)
        assert params.n_runs == 3
        assert params.total_steps == 90

    def test_warmup_and_duration(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """Warm-up steps and run duration follow from the fractions and the cadence."""
        params = _load(tmp_path, section, run_length_steps=1500, warmup_fraction=0.2)
        assert params.warmup_steps == 300
        assert params.run_duration_s == 1500 * 180.0


class TestMissingInput:
    """Absent files, sections and keys fail with a descriptive error."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing params file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_simulation_params(tmp_path / "nope.yaml")

    def test_missing_section(self, tmp_path: Path) -> None:
        """A file without simulation: raises KeyError naming the section."""
        path = tmp_path / "p.yaml"
        path.write_text(yaml.safe_dump({"tep": {}}), encoding="utf-8")
        with pytest.raises(KeyError, match="simulation"):
            load_simulation_params(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty file raises KeyError (no section)."""
        path = tmp_path / "p.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(KeyError):
            load_simulation_params(path)

    def test_section_must_be_a_mapping(self, tmp_path: Path) -> None:
        """A non-mapping section raises TypeError."""
        with pytest.raises(TypeError, match="mapping"):
            load_simulation_params(_write(tmp_path / "p.yaml", [1, 2]))

    @pytest.mark.parametrize(
        "key",
        ["n_samples", "fault_injection_rate", "start_time", "spans_path", "severity_weights",
         "load_min_fraction", "calibration_date", "drift_coefficient"],
    )
    def test_missing_key_names_the_key(
        self, tmp_path: Path, section: dict[str, Any], key: str
    ) -> None:
        """Each required key raises KeyError that names `simulation.<key>`."""
        broken = {k: v for k, v in section.items() if k != key}
        with pytest.raises(KeyError, match=f"simulation.{key}"):
            load_simulation_params(_write(tmp_path / "p.yaml", broken))


class TestValueRanges:
    """Out-of-range values are rejected at load time."""

    @pytest.mark.parametrize("rate", [0.0, 1.0, 1.5, -0.1])
    def test_rate_must_be_in_open_unit_interval(
        self, tmp_path: Path, section: dict[str, Any], rate: float
    ) -> None:
        """fault_injection_rate is in (0, 1)."""
        with pytest.raises(ValueError, match="fault_injection_rate"):
            _load(tmp_path, section, fault_injection_rate=rate)

    def test_critical_min_above_rate(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """critical_min_fraction cannot exceed fault_injection_rate."""
        with pytest.raises(ValueError, match="critical_min_fraction"):
            _load(tmp_path, section, fault_injection_rate=0.1, critical_min_fraction=0.2)

    def test_critical_min_zero(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """critical_min_fraction must be positive."""
        with pytest.raises(ValueError, match="critical_min_fraction"):
            _load(tmp_path, section, critical_min_fraction=0.0)

    def test_critical_min_equal_to_rate_is_valid(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """The bound is inclusive."""
        params = _load(tmp_path, section, fault_injection_rate=0.1, critical_min_fraction=0.1)
        assert params.critical_min_fraction == 0.1

    def test_weights_must_sum_to_one(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """Weights that do not add up to 1 are rejected."""
        weights = {"critical": 0.5, "high": 0.3, "medium": 0.25, "low": 0.15}
        with pytest.raises(ValueError, match="sum to 1"):
            _load(tmp_path, section, severity_weights=weights)

    def test_weights_need_exactly_the_four_keys(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """A missing or extra severity key is rejected."""
        with pytest.raises(ValueError, match="exactly the keys"):
            _load(tmp_path, section, severity_weights={"critical": 0.5, "high": 0.5})

    def test_weights_cannot_be_negative(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A negative weight is rejected even if the sum is 1."""
        weights = {"critical": 1.3, "high": -0.3, "medium": 0.0, "low": 0.0}
        with pytest.raises(ValueError, match=">= 0"):
            _load(tmp_path, section, severity_weights=weights)

    def test_weights_must_be_a_mapping(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A non-mapping severity_weights raises TypeError."""
        with pytest.raises(TypeError, match="severity_weights"):
            _load(tmp_path, section, severity_weights=[0.3, 0.3, 0.25, 0.15])

    @pytest.mark.parametrize("key", ["n_samples", "run_length_steps"])
    @pytest.mark.parametrize("value", [0, -5, 2.5])
    def test_counts_must_be_positive_integers(
        self, tmp_path: Path, section: dict[str, Any], key: str, value: float
    ) -> None:
        """Step counts are strictly positive integers."""
        with pytest.raises(ValueError, match=key):
            _load(tmp_path, section, **{key: value})

    def test_dt_must_be_positive(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """dt_seconds > 0."""
        with pytest.raises(ValueError, match="dt_seconds"):
            _load(tmp_path, section, dt_seconds=0)

    @pytest.mark.parametrize("value", [1.0, -0.1])
    def test_warmup_fraction_range(
        self, tmp_path: Path, section: dict[str, Any], value: float
    ) -> None:
        """warmup_fraction is in [0, 1)."""
        with pytest.raises(ValueError, match="warmup_fraction"):
            _load(tmp_path, section, warmup_fraction=value)

    def test_warmup_zero_is_valid(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A run without warm-up is allowed."""
        assert _load(tmp_path, section, warmup_fraction=0.0).warmup_steps == 0

    def test_load_limits_must_be_ordered(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """load_min_fraction must be below load_max_fraction."""
        with pytest.raises(ValueError, match="load_min_fraction"):
            _load(tmp_path, section, load_min_fraction=0.9, load_max_fraction=0.8)

    def test_load_max_is_bounded_by_the_model(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """load_max_fraction above the model ceiling is rejected."""
        with pytest.raises(ValueError, match="load_max_fraction"):
            _load(tmp_path, section, load_max_fraction=2.0)

    def test_hold_minutes_order(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """load_hold_minutes_min cannot exceed load_hold_minutes_max."""
        with pytest.raises(ValueError, match="load_hold_minutes_min"):
            _load(tmp_path, section, load_hold_minutes_min=30, load_hold_minutes_max=10)

    def test_ramp_must_be_positive(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """max_ramp_percent_per_min > 0."""
        with pytest.raises(ValueError, match="max_ramp_percent_per_min"):
            _load(tmp_path, section, max_ramp_percent_per_min=0)

    def test_non_numeric_value(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A text where a number is expected raises ValueError naming the key."""
        with pytest.raises(ValueError, match="must be a number"):
            _load(tmp_path, section, dt_seconds="fast")

    def test_non_finite_value(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """NaN and infinity are rejected."""
        with pytest.raises(ValueError, match="finite"):
            _load(tmp_path, section, drift_coefficient=float("nan"))


class TestTimeAndDates:
    """start_time and the metadata dates are validated."""

    def test_naive_start_time_is_rejected(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """A start_time without timezone raises ValueError."""
        with pytest.raises(ValueError, match="timezone"):
            _load(tmp_path, section, start_time="2026-01-01T00:00:00")

    def test_start_time_must_be_iso(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A non-ISO string raises ValueError."""
        with pytest.raises(ValueError, match="ISO-8601"):
            _load(tmp_path, section, start_time="yesterday")

    def test_start_time_type(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A number where a timestamp is expected raises TypeError."""
        with pytest.raises(TypeError, match="start_time"):
            _load(tmp_path, section, start_time=20260101)

    def test_datetime_is_reduced_to_its_date(
        self, tmp_path: Path, section: dict[str, Any]
    ) -> None:
        """A YAML timestamp where a date is expected is reduced to its date."""
        stamp = datetime(2025, 6, 1, 10, 30)
        params = _load(tmp_path, section, calibration_date=stamp)
        assert params.calibration_date == date(2025, 6, 1)

    def test_date_must_be_iso(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A malformed date string raises ValueError naming the key."""
        with pytest.raises(ValueError, match="calibration_date"):
            _load(tmp_path, section, calibration_date="June 1st")

    def test_date_type(self, tmp_path: Path, section: dict[str, Any]) -> None:
        """A number where a date is expected raises TypeError."""
        with pytest.raises(TypeError, match="last_maintenance_date"):
            _load(tmp_path, section, last_maintenance_date=20251201)


class TestRepositoryParams:
    """The simulation: section of the versioned params.yaml resolves completely."""

    def test_repository_section_resolves(self) -> None:
        """params.yaml carries every key the loader requires."""
        params = load_simulation_params(REPO_ROOT / DEFAULT_PARAMS_PATH)
        assert params.dt_seconds == 180.0
        assert params.fault_injection_rate == 0.40
        assert params.n_runs == math.ceil(params.n_samples / params.run_length_steps)

    def test_repository_paths_point_at_existing_files(self) -> None:
        """The configuration files the section names exist in the repository."""
        params = load_simulation_params(REPO_ROOT / DEFAULT_PARAMS_PATH)
        for path in (
            params.plant_parameters_path,
            params.neutronics_path,
            params.sensors_path,
            params.spans_path,
            params.water_table_path,
        ):
            assert (REPO_ROOT / path).exists(), path

    def test_spans_seed_differs_from_dataset_seed(self) -> None:
        """The span runs must not reuse the dataset's master seed."""
        from data.generators.derive_sim_spans import SPANS_SEED

        params = load_simulation_params(REPO_ROOT / DEFAULT_PARAMS_PATH)
        assert SPANS_SEED != params.openmc_seed
