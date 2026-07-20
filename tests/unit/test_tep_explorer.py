"""Unit tests for data.generators.tep_explorer.

Se ejecutan sobre ficheros .dat sinteticos en tmp_path; ninguno depende de que
data/raw/tep este poblado. El directorio sintetico incluye un d00.dat
transpuesto, de forma que el explorer se ejercita contra la misma anomalia de
orientacion que trae el dataset real.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from data.generators.tep_explorer import explore_tep, main
from data.generators.tep_loader import N_COLUMNS

from .conftest import WriteParams


@pytest.fixture()
def reports_dir(tmp_path: Path) -> Path:
    """Return a dedicated reports directory inside tmp_path."""
    return tmp_path / "reports"


class TestReportStructure:
    """explore_tep must return a report describing every file it loaded."""

    def test_one_entry_per_file(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """Each .dat file present must appear under datasets."""
        data_dir = make_dat_dir(n_files=4, n_rows=6)
        report = explore_tep(str(data_dir), reports_dir)
        assert sorted(report["datasets"]) == ["d00", "d01", "d02", "d03"]

    def test_transposed_file_reports_normalised_shape(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """d00, stored transposed, must be reported as (n_rows, 52)."""
        data_dir = make_dat_dir(n_files=2, n_rows=6)
        report = explore_tep(str(data_dir), reports_dir)
        assert report["datasets"]["d00"]["shape"] == [6, N_COLUMNS]

    def test_statistics_cover_every_column(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """Per-file statistics must include all 52 variables."""
        data_dir = make_dat_dir(n_files=2, n_rows=6)
        report = explore_tep(str(data_dir), reports_dir)
        assert len(report["datasets"]["d01"]["statistics"]) == N_COLUMNS

    def test_statistics_expose_expected_moments(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """Each column entry must carry mean, std, min and max."""
        data_dir = make_dat_dir(n_files=2, n_rows=6)
        report = explore_tep(str(data_dir), reports_dir)
        assert set(report["datasets"]["d01"]["statistics"]["col_00"]) == {
            "mean",
            "std",
            "min",
            "max",
        }

    def test_missing_counts_are_zero_for_clean_data(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """Synthetic files have no gaps, so every missing count must be zero."""
        data_dir = make_dat_dir(n_files=2, n_rows=6)
        report = explore_tep(str(data_dir), reports_dir)
        assert set(report["datasets"]["d01"]["missing_counts"].values()) == {0}

    def test_correlation_matrix_is_square(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """The pooled correlation matrix must be 52 x 52."""
        data_dir = make_dat_dir(n_files=3, n_rows=8)
        report = explore_tep(str(data_dir), reports_dir)
        matrix = report["correlation_matrix"]
        assert len(matrix) == N_COLUMNS
        assert len(matrix["col_00"]) == N_COLUMNS


class TestArtefacts:
    """The two report artefacts must be written to the configured directory."""

    def test_creates_reports_directory(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """A missing reports directory must be created."""
        explore_tep(str(make_dat_dir()), reports_dir)
        assert reports_dir.is_dir()

    def test_writes_exploration_json(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """tep_exploration.json must be valid JSON containing the report."""
        explore_tep(str(make_dat_dir()), reports_dir)
        payload = json.loads((reports_dir / "tep_exploration.json").read_text())
        assert "datasets" in payload

    def test_writes_correlations_csv(
        self, make_dat_dir: Callable[..., Path], reports_dir: Path
    ) -> None:
        """tep_correlations.csv must be written alongside the JSON report."""
        explore_tep(str(make_dat_dir()), reports_dir)
        assert (reports_dir / "tep_correlations.csv").stat().st_size > 0

    def test_honours_the_configured_directory(
        self, make_dat_dir: Callable[..., Path], tmp_path: Path
    ) -> None:
        """Reports must land in reports_dir, not in a hardcoded location."""
        custom = tmp_path / "elsewhere" / "reports"
        explore_tep(str(make_dat_dir()), custom)
        assert (custom / "tep_exploration.json").exists()


class TestEmptyInput:
    """A directory with no TEP files must degrade gracefully."""

    def test_report_has_no_datasets(self, tmp_path: Path, reports_dir: Path) -> None:
        """An empty input directory must yield an empty datasets mapping."""
        empty = tmp_path / "empty"
        empty.mkdir()
        report = explore_tep(str(empty), reports_dir)
        assert report["datasets"] == {}

    def test_correlation_matrix_is_empty(
        self, tmp_path: Path, reports_dir: Path
    ) -> None:
        """With nothing pooled, the correlation matrix must be empty."""
        empty = tmp_path / "empty"
        empty.mkdir()
        report = explore_tep(str(empty), reports_dir)
        assert report["correlation_matrix"] == {}

    def test_json_is_still_written(self, tmp_path: Path, reports_dir: Path) -> None:
        """The JSON artefact must exist even when no input was found."""
        empty = tmp_path / "empty"
        empty.mkdir()
        explore_tep(str(empty), reports_dir)
        assert (reports_dir / "tep_exploration.json").exists()


class TestMainEntryPoint:
    """main() must take both directories from params.yaml."""

    def test_uses_configured_directories(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams,
        tmp_path: Path
    ) -> None:
        """Data must be read from raw_dir and reports written to reports_dir."""
        data_dir = make_dat_dir(n_files=2, n_rows=5)
        reports = tmp_path / "configured_reports"
        report = main(write_params(raw_dir=str(data_dir), reports_dir=str(reports)))
        assert (reports / "tep_exploration.json").exists()
        assert "d00" in report["datasets"]
