"""Unit tests for data.generators.tep_adapter.

Covers:
    - Adaptation of a single TEP row produces a valid SensorReading per column.
    - fault_type=0 maps to QualityFlag.GOOD for all readings.
    - fault_type>=1 maps to QualityFlag.SUSPECT for all readings.
    - Timestamps for a given sensor are monotonically increasing.
    - Sensor identifiers follow the expected naming convention.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from data.generators.tep_adapter import (
    _N_COLUMNS,
    _PLANT_ID,
    _SAMPLE_INTERVAL,
    TEPAdapter,
)
from data.schemas.sensor_reading import QualityFlag, SensorReading

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_ROWS = 5
_START_TIME = datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)

# One row of 52 realistic TEP values (flows ~50, pressure ~2700, temp ~120, comp ~30, pos ~50)
_SAMPLE_ROW = (
    "  0.25   3.21  42.18  18.75  31.84 22.63 2732.1 77.3 120.4 18.9"
    "  66.2  53.4 2664.0  23.8  53.2 3102.0  41.3 67.9  18.4 327.2"
    " 40.1  40.3  30.1  45.2  12.3   0.01  40.2  38.5  48.9  51.2"
    "  3.4   2.1  87.6  65.4  32.1   0.00   1.23  4.56  7.89  0.11"
    "  0.22  50.1  51.2  52.3  53.4  54.5  55.6  56.7  57.8  58.9  59.0  60.1"
)


def _write_dat_file(path: Path, n_rows: int, row_content: str) -> None:
    """Write a synthetic .dat file with n_rows identical rows."""
    with path.open("w", encoding="utf-8") as fh:
        for _ in range(n_rows):
            fh.write(row_content + "\n")


@pytest.fixture()
def normal_dat(tmp_path: Path) -> Path:
    """Return path to a synthetic TEP file representing normal operation."""
    filepath = tmp_path / "d00.dat"
    _write_dat_file(filepath, _N_ROWS, _SAMPLE_ROW)
    return filepath


@pytest.fixture()
def fault_dat(tmp_path: Path) -> Path:
    """Return path to a synthetic TEP file representing a fault scenario."""
    filepath = tmp_path / "d01.dat"
    _write_dat_file(filepath, _N_ROWS, _SAMPLE_ROW)
    return filepath


@pytest.fixture()
def adapter() -> TEPAdapter:
    """Return a TEPAdapter with a fixed start time for deterministic timestamps."""
    return TEPAdapter(start_time=_START_TIME)


# ---------------------------------------------------------------------------
# Test: valid SensorReading produced from a single file
# ---------------------------------------------------------------------------


class TestAdaptFileProducesValidReadings:
    """adapt_file must return well-formed SensorReading objects for every row."""

    def test_reading_count(self, adapter: TEPAdapter, normal_dat: Path) -> None:
        """Each row must produce exactly 52 SensorReading objects."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        assert len(readings) == _N_ROWS * _N_COLUMNS

    def test_all_are_sensor_reading_instances(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """Every element in the result must be a SensorReading instance."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        for reading in readings:
            assert isinstance(reading, SensorReading)

    def test_plant_id(self, adapter: TEPAdapter, normal_dat: Path) -> None:
        """All readings must carry the TEP plant identifier."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        assert all(r.plant_id == _PLANT_ID for r in readings)

    def test_raw_counts_within_adc_range(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """raw_counts must be within the 16-bit ADC range [0, 65535]."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        for r in readings:
            assert 0 <= r.measurement.raw_counts <= 65535

    def test_first_sensor_id_convention(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """Column 0 must produce sensor_id 'TEP-XMEAS-01'."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        first_reading = readings[0]
        assert first_reading.sensor.id == "TEP-XMEAS-01"

    def test_last_sensor_id_convention(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """Column 51 (XMV 11) must produce sensor_id 'TEP-XMV-11'."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        last_of_first_row = readings[_N_COLUMNS - 1]
        assert last_of_first_row.sensor.id == "TEP-XMV-11"


# ---------------------------------------------------------------------------
# Test: quality flag mapping
# ---------------------------------------------------------------------------


class TestQualityFlagMapping:
    """Quality flag must reflect whether the file represents normal or fault operation."""

    def test_normal_file_quality_is_good(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """fault_type=0 must yield QualityFlag.GOOD for every reading."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        for r in readings:
            assert r.measurement.quality == QualityFlag.GOOD

    def test_fault_file_quality_is_suspect(
        self, adapter: TEPAdapter, fault_dat: Path
    ) -> None:
        """fault_type=1 must yield QualityFlag.SUSPECT for every reading."""
        readings = adapter.adapt_file(str(fault_dat), fault_type=1)
        for r in readings:
            assert r.measurement.quality == QualityFlag.SUSPECT

    def test_any_nonzero_fault_type_is_suspect(
        self, adapter: TEPAdapter, fault_dat: Path
    ) -> None:
        """fault_type=21 (the highest TEP fault) must also yield SUSPECT."""
        readings = adapter.adapt_file(str(fault_dat), fault_type=21)
        for r in readings:
            assert r.measurement.quality == QualityFlag.SUSPECT

    def test_normal_readings_are_usable(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """Good-quality readings from d00 must be flagged as usable."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        assert all(r.is_usable for r in readings)

    def test_fault_readings_are_usable(
        self, adapter: TEPAdapter, fault_dat: Path
    ) -> None:
        """Suspect-quality readings are still usable (is_usable=True)."""
        readings = adapter.adapt_file(str(fault_dat), fault_type=1)
        assert all(r.is_usable for r in readings)


# ---------------------------------------------------------------------------
# Test: timestamp monotonicity
# ---------------------------------------------------------------------------


class TestTimestampMonotonicity:
    """Timestamps for a single sensor must increase strictly across rows."""

    def test_timestamps_monotonically_increasing_for_single_sensor(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """Consecutive timestamps for the same sensor_id must be strictly increasing."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)

        sensor_timestamps = [
            r.timestamp for r in readings if r.sensor.id == "TEP-XMEAS-01"
        ]

        assert len(sensor_timestamps) == _N_ROWS
        for i in range(1, len(sensor_timestamps)):
            assert sensor_timestamps[i] > sensor_timestamps[i - 1]

    def test_timestamp_interval_is_three_minutes(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """Consecutive timestamps for the same sensor must differ by exactly 3 minutes."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)

        sensor_timestamps = [
            r.timestamp for r in readings if r.sensor.id == "TEP-XMEAS-01"
        ]

        for i in range(1, len(sensor_timestamps)):
            delta = sensor_timestamps[i] - sensor_timestamps[i - 1]
            assert delta == _SAMPLE_INTERVAL

    def test_first_timestamp_equals_start_time(
        self, adapter: TEPAdapter, normal_dat: Path
    ) -> None:
        """The first reading for any sensor must have timestamp equal to start_time."""
        readings = adapter.adapt_file(str(normal_dat), fault_type=0)
        first_reading = readings[0]
        assert first_reading.timestamp == _START_TIME


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------


class TestAdaptFileErrorHandling:
    """adapt_file must raise descriptive errors for malformed input."""

    def test_raises_for_wrong_column_count(self, tmp_path: Path) -> None:
        """A file with fewer than 52 columns must raise ValueError."""
        bad_file = tmp_path / "bad.dat"
        bad_file.write_text("1.0 2.0 3.0\n4.0 5.0 6.0\n", encoding="utf-8")

        adapter = TEPAdapter()
        with pytest.raises(ValueError, match="52"):
            adapter.adapt_file(str(bad_file), fault_type=0)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """A non-existent filepath must raise FileNotFoundError."""
        adapter = TEPAdapter()
        with pytest.raises(FileNotFoundError):
            adapter.adapt_file(str(tmp_path / "nonexistent.dat"), fault_type=0)
