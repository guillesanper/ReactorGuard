"""Unit tests for data.generators.tep_params.

Este modulo es lo que hace que los `params:` declarados en dvc.yaml sean
efectivos, asi que sus tests cubren dos cosas: que la seccion tep: real de
params.yaml se resuelva completa, y que una configuracion incompleta falle con
un mensaje que nombre la clave ausente en vez de un KeyError opaco.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest
import yaml

from data.generators.tep_params import DEFAULT_PARAMS_PATH, TEPParams, load_tep_params

_VALID_SECTION = {
    "raw_dir": "data/raw/tep",
    "processed_dir": "data/processed/tep",
    "reports_dir": "data/reports",
    "plant_id": "TEP-PLANT-01",
    "start_time": "2000-01-01T00:00:00+00:00",
    "sample_interval_minutes": 3,
    "adc_scale_max": 3000.0,
    "calibration_date": "2023-06-01",
    "last_maintenance_date": "2023-12-01",
    "drift_coefficient": 0.0001,
}


def _write_params(path: Path, section: object) -> Path:
    """Write a params file containing the given tep: section."""
    path.write_text(yaml.safe_dump({"tep": section}), encoding="utf-8")
    return path


@pytest.fixture()
def params_file(tmp_path: Path) -> Path:
    """Return a params file with a complete, valid tep: section."""
    return _write_params(tmp_path / "params.yaml", dict(_VALID_SECTION))


class TestLoadValidParams:
    """A complete section must resolve into a fully typed TEPParams."""

    def test_returns_tep_params(self, params_file: Path) -> None:
        """The loader must return a TEPParams instance."""
        assert isinstance(load_tep_params(params_file), TEPParams)

    def test_paths_are_path_objects(self, params_file: Path) -> None:
        """Directory entries must be coerced to Path."""
        params = load_tep_params(params_file)
        assert params.raw_dir == Path("data/raw/tep")
        assert params.processed_dir == Path("data/processed/tep")
        assert params.reports_dir == Path("data/reports")

    def test_start_time_is_timezone_aware(self, params_file: Path) -> None:
        """start_time must parse into an aware datetime."""
        start = load_tep_params(params_file).start_time
        assert isinstance(start, datetime)
        assert start.tzinfo is not None

    def test_dates_are_date_objects(self, params_file: Path) -> None:
        """Quoted ISO dates must be coerced to date."""
        params = load_tep_params(params_file)
        assert params.calibration_date == date(2023, 6, 1)
        assert params.last_maintenance_date == date(2023, 12, 1)

    def test_unquoted_yaml_dates_are_accepted(self, tmp_path: Path) -> None:
        """PyYAML parses unquoted ISO dates to date; those must work too."""
        section = dict(_VALID_SECTION)
        section["calibration_date"] = date(2023, 6, 1)
        path = _write_params(tmp_path / "params.yaml", section)
        assert load_tep_params(path).calibration_date == date(2023, 6, 1)

    def test_numeric_coercion(self, params_file: Path) -> None:
        """Numeric parameters must arrive with their declared types."""
        params = load_tep_params(params_file)
        assert params.sample_interval_minutes == 3
        assert params.adc_scale_max == pytest.approx(3000.0)
        assert params.drift_coefficient == pytest.approx(0.0001)

    def test_is_frozen(self, params_file: Path) -> None:
        """TEPParams must be immutable so stages cannot drift from params.yaml."""
        params = load_tep_params(params_file)
        with pytest.raises(AttributeError):
            params.plant_id = "OTHER"  # type: ignore[misc]


class TestRepositoryParamsFile:
    """The params.yaml committed to the repo must satisfy the loader."""

    def test_real_params_file_loads(self) -> None:
        """The tep: section shipped in the repo must resolve without error."""
        params = load_tep_params(DEFAULT_PARAMS_PATH)
        assert params.plant_id == "TEP-PLANT-01"
        assert params.raw_dir == Path("data/raw/tep")

    def test_processed_dir_is_not_nested_in_raw_dir(self) -> None:
        """adapt_tep's out must not live inside download_tep's out."""
        params = load_tep_params(DEFAULT_PARAMS_PATH)
        assert params.raw_dir not in params.processed_dir.parents


class TestInvalidParams:
    """Incomplete or missing configuration must fail with a named cause."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A non-existent params path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_tep_params(tmp_path / "absent.yaml")

    def test_missing_section(self, tmp_path: Path) -> None:
        """A params file without a tep: section must name the section."""
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"training": {"seed": 42}}), encoding="utf-8")
        with pytest.raises(KeyError, match="tep"):
            load_tep_params(path)

    def test_empty_file(self, tmp_path: Path) -> None:
        """An empty params file must be reported as a missing section."""
        path = tmp_path / "params.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(KeyError, match="tep"):
            load_tep_params(path)

    @pytest.mark.parametrize("missing_key", sorted(_VALID_SECTION))
    def test_each_required_key_is_enforced(
        self, tmp_path: Path, missing_key: str
    ) -> None:
        """Dropping any required key must raise a KeyError naming that key."""
        section = {k: v for k, v in _VALID_SECTION.items() if k != missing_key}
        path = _write_params(tmp_path / "params.yaml", section)
        with pytest.raises(KeyError, match=missing_key):
            load_tep_params(path)

    def test_bad_date_type(self, tmp_path: Path) -> None:
        """A non-date calibration_date must raise TypeError."""
        section = dict(_VALID_SECTION)
        section["calibration_date"] = [2023, 6, 1]
        path = _write_params(tmp_path / "params.yaml", section)
        with pytest.raises(TypeError, match="date"):
            load_tep_params(path)
