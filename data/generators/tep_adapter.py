"""Convert Tennessee Eastman Process (TEP) dataset rows to SensorReading schema objects.

The TEP dataset has 52 columns per sample:
    Columns 0-21  (XMEAS 1-22):  process measurements (flow, pressure, temperature)
    Columns 22-40 (XMEAS 23-41): composition/analysis (normalised mol fractions)
    Columns 41-51 (XMV 1-11):    manipulated variables (actuator positions)

Sensor identifiers follow the convention:
    TEP-XMEAS-01 through TEP-XMEAS-41  (columns 0-40)
    TEP-XMV-01   through TEP-XMV-11    (columns 41-51)
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List

import pandas as pd

from data.schemas.sensor_reading import (
    Measurement,
    MeasurementUnit,
    QualityFlag,
    SensorInfo,
    SensorLocation,
    SensorMetadata,
    SensorReading,
    SensorType,
)

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_COLUMNS = 52
_SAMPLE_INTERVAL = timedelta(minutes=3)
_PLANT_ID = "TEP-PLANT-01"
_DEFAULT_START_TIME = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_CALIBRATION_DATE = date(2023, 6, 1)
_LAST_MAINTENANCE_DATE = date(2023, 12, 1)
_DRIFT_COEFFICIENT = 0.0001

# Maximum expected TEP value used to scale floats into a 16-bit ADC range.
# TEP pressures reach ~3000 kPa; all other variables are well below this.
_ADC_SCALE_MAX = 3000.0

_FILE_NAMES: list[str] = ["d00.dat"] + [f"d{i:02d}.dat" for i in range(1, 22)]

# ---------------------------------------------------------------------------
# Static column mappings
# ---------------------------------------------------------------------------


def _build_sensor_type_map() -> dict[int, SensorType]:
    """Return a mapping from zero-based column index to SensorType.

    XMEAS 1-22 (columns 0-21): specific process measurement types.
    XMEAS 23-41 (columns 22-40): composition analysis (NORMALIZED).
    XMV 1-11 (columns 41-51): actuator positions (POSITION).
    """
    mapping: dict[int, SensorType] = {
        0: SensorType.FLOW,           # XMEAS 1:  A feed flow
        1: SensorType.FLOW,           # XMEAS 2:  D feed flow
        2: SensorType.FLOW,           # XMEAS 3:  E feed flow
        3: SensorType.FLOW,           # XMEAS 4:  A and C feed flow
        4: SensorType.FLOW,           # XMEAS 5:  Recycle flow
        5: SensorType.FLOW,           # XMEAS 6:  Reactor feed rate
        6: SensorType.PRESSURE,       # XMEAS 7:  Reactor pressure
        7: SensorType.FLOW,           # XMEAS 8:  Reactor level
        8: SensorType.THERMOCOUPLE,   # XMEAS 9:  Reactor temperature
        9: SensorType.FLOW,           # XMEAS 10: Purge rate
        10: SensorType.THERMOCOUPLE,  # XMEAS 11: Separator temperature
        11: SensorType.FLOW,          # XMEAS 12: Separator level
        12: SensorType.PRESSURE,      # XMEAS 13: Separator pressure
        13: SensorType.FLOW,          # XMEAS 14: Separator underflow
        14: SensorType.FLOW,          # XMEAS 15: Stripper level
        15: SensorType.PRESSURE,      # XMEAS 16: Stripper pressure
        16: SensorType.FLOW,          # XMEAS 17: Stripper underflow
        17: SensorType.THERMOCOUPLE,  # XMEAS 18: Stripper temperature
        18: SensorType.FLOW,          # XMEAS 19: Stripper steam flow
        19: SensorType.FLOW,          # XMEAS 20: Compressor work
        20: SensorType.THERMOCOUPLE,  # XMEAS 21: Reactor CW outlet temperature
        21: SensorType.THERMOCOUPLE,  # XMEAS 22: Separator CW outlet temperature
    }
    for col in range(22, 41):
        mapping[col] = SensorType.NORMALIZED
    for col in range(41, 52):
        mapping[col] = SensorType.POSITION
    return mapping


def _build_location_map() -> dict[int, SensorLocation]:
    """Return a mapping from zero-based column index to SensorLocation.

    Columns 0-9   (reactor measurements):    PRIMARY_LOOP
    Columns 10-21 (separator/stripper):      SECONDARY_LOOP
    Columns 22-40 (composition analysis):    CONTAINMENT
    Columns 41-51 (manipulated variables):   CORE
    """
    mapping: dict[int, SensorLocation] = {}
    for col in range(0, 10):
        mapping[col] = SensorLocation.PRIMARY_LOOP
    for col in range(10, 22):
        mapping[col] = SensorLocation.SECONDARY_LOOP
    for col in range(22, 41):
        mapping[col] = SensorLocation.CONTAINMENT
    for col in range(41, 52):
        mapping[col] = SensorLocation.CORE
    return mapping


_SENSOR_TYPE_MAP: dict[int, SensorType] = _build_sensor_type_map()
_LOCATION_MAP: dict[int, SensorLocation] = _build_location_map()

_UNIT_MAP: dict[SensorType, MeasurementUnit] = {
    SensorType.THERMOCOUPLE: MeasurementUnit.CELSIUS,
    SensorType.PRESSURE: MeasurementUnit.BAR,
    SensorType.FLOW: MeasurementUnit.KG_S,
    SensorType.NORMALIZED: MeasurementUnit.NORMALIZED,
    SensorType.POSITION: MeasurementUnit.PERCENT,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _sensor_id(col_idx: int) -> str:
    """Return the canonical sensor identifier for a given column index.

    Args:
        col_idx: Zero-based TEP column index (0-51).

    Returns:
        Sensor tag string, e.g. "TEP-XMEAS-01" or "TEP-XMV-03".
    """
    if col_idx < 41:
        return f"TEP-XMEAS-{col_idx + 1:02d}"
    return f"TEP-XMV-{col_idx - 40:02d}"


def _elevation_m(col_idx: int) -> float:
    """Return a synthetic elevation for a column, in range [-10.0, 100.0] m.

    Elevation is assigned deterministically based on column index, simulating
    sensors distributed across plant floors.

    Args:
        col_idx: Zero-based TEP column index (0-51).

    Returns:
        Elevation value in metres within [-10.0, 100.0].
    """
    return float((col_idx % 55) * 2 - 10)


def _to_raw_counts(value: float) -> int:
    """Scale a TEP process value to a 16-bit ADC count in [0, 65535].

    Uses _ADC_SCALE_MAX as the normalisation denominator. Values above
    _ADC_SCALE_MAX saturate at 65535; negative values clamp to 0.

    Args:
        value: TEP process value in engineering units.

    Returns:
        Integer ADC count in [0, 65535].
    """
    scaled = abs(value) / _ADC_SCALE_MAX * 65535.0
    return int(min(max(scaled, 0.0), 65535.0))


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------


def save_to_parquet(df: pd.DataFrame, output_dir: str) -> None:
    """Save a TEPAdapter DataFrame to fault_type-partitioned Parquet files.

    Produces one file per fault type under the directory structure:
        output_dir/fault_type=00/readings.parquet
        output_dir/fault_type=01/readings.parquet
        ...

    Args:
        df: DataFrame produced by TEPAdapter.adapt_all, which must contain a
            "fault_type" integer column.
        output_dir: Root directory for the partitioned output. Created if it
            does not exist.
    """
    out_path = Path(output_dir)
    for fault_type in sorted(df["fault_type"].unique()):
        subset = df[df["fault_type"] == fault_type].copy()
        partition_dir = out_path / f"fault_type={fault_type:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / "readings.parquet"
        subset.to_parquet(parquet_path, index=False)
        _LOG.info(
            "Saved %d readings to %s", len(subset), parquet_path
        )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class TEPAdapter:
    """Convert rows of a TEP .dat file to validated SensorReading objects.

    Each sample row in the TEP produces 52 SensorReading records, one per
    process variable. Timestamps are generated synthetically at 3-minute
    intervals starting from a configurable start time.

    Attributes:
        start_time: UTC datetime used as the origin for synthetic timestamps.
    """

    def __init__(
        self, start_time: datetime = _DEFAULT_START_TIME
    ) -> None:
        """Initialise the adapter with an optional synthetic timestamp origin.

        Args:
            start_time: UTC datetime for the first sample. Subsequent samples
                are offset by multiples of 3 minutes.
        """
        self.start_time = start_time

    def _row_to_readings(
        self,
        row: list[float],
        row_index: int,
        quality: QualityFlag,
    ) -> list[SensorReading]:
        """Convert a single TEP data row to a list of 52 SensorReading objects.

        Args:
            row: List of 52 float values for a single timestep.
            row_index: Zero-based row index used to compute the timestamp.
            quality: Quality flag applied to all readings in this row.

        Returns:
            List of 52 SensorReading instances, one per TEP variable.
        """
        timestamp = self.start_time + row_index * _SAMPLE_INTERVAL
        readings: list[SensorReading] = []

        for col_idx, value in enumerate(row):
            sensor_type = _SENSOR_TYPE_MAP[col_idx]
            reading = SensorReading(
                reading_id=uuid.uuid4(),
                timestamp=timestamp,
                plant_id=_PLANT_ID,
                sensor=SensorInfo(
                    id=_sensor_id(col_idx),
                    type=sensor_type,
                    location=_LOCATION_MAP[col_idx],
                    elevation_m=_elevation_m(col_idx),
                ),
                measurement=Measurement(
                    value=float(value),
                    unit=_UNIT_MAP[sensor_type],
                    quality=quality,
                    raw_counts=_to_raw_counts(float(value)),
                ),
                metadata=SensorMetadata(
                    calibration_date=_CALIBRATION_DATE,
                    last_maintenance=_LAST_MAINTENANCE_DATE,
                    drift_coefficient=_DRIFT_COEFFICIENT,
                ),
            )
            readings.append(reading)

        return readings

    def adapt_file(self, filepath: str, fault_type: int) -> List[SensorReading]:
        """Adapt all rows in a TEP .dat file to SensorReading objects.

        Quality is determined by fault_type:
            fault_type == 0  ->  QualityFlag.GOOD   (normal operation)
            fault_type >= 1  ->  QualityFlag.SUSPECT (fault period)

        Args:
            filepath: Path to a TEP .dat file (whitespace-delimited, no header).
            fault_type: Integer fault identifier (0 = normal, 1-21 = fault).

        Returns:
            Flat list of SensorReading objects. For a file with N rows the list
            contains N * 52 elements ordered by (row_index, col_index).

        Raises:
            ValueError: If the file does not contain exactly 52 columns.
            FileNotFoundError: If filepath does not exist.
        """
        quality = QualityFlag.GOOD if fault_type == 0 else QualityFlag.SUSPECT

        file_path = Path(filepath)
        df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
        if df.shape[1] != _N_COLUMNS:
            raise ValueError(
                f"Expected {_N_COLUMNS} columns in {file_path.name}, "
                f"got {df.shape[1]}."
            )

        all_readings: list[SensorReading] = []
        for row_index, row_values in enumerate(df.itertuples(index=False, name=None)):
            all_readings.extend(
                self._row_to_readings(list(row_values), row_index, quality)
            )

        _LOG.info(
            "Adapted %d rows x %d columns = %d readings from %s (fault_type=%d).",
            df.shape[0],
            _N_COLUMNS,
            len(all_readings),
            file_path.name,
            fault_type,
        )
        return all_readings

    def adapt_all(self, data_dir: str) -> pd.DataFrame:
        """Adapt all TEP files in data_dir and return a consolidated DataFrame.

        Each row in the returned DataFrame corresponds to one SensorReading
        (one variable at one timestep). Files that do not exist are skipped
        with a warning.

        Args:
            data_dir: Directory containing d00.dat through d21.dat.

        Returns:
            DataFrame with columns:
                reading_id, timestamp, plant_id, sensor_id, sensor_type,
                sensor_location, elevation_m, value, unit, quality,
                raw_counts, fault_type, is_usable.
        """
        data_path = Path(data_dir)
        records: list[dict] = []

        for fault_type, filename in enumerate(_FILE_NAMES):
            filepath = data_path / filename
            if not filepath.exists():
                _LOG.warning("File not found, skipping: %s", filepath)
                continue

            _LOG.info("Adapting %s (fault_type=%d).", filename, fault_type)
            readings = self.adapt_file(str(filepath), fault_type)

            for reading in readings:
                records.append(
                    {
                        "reading_id": str(reading.reading_id),
                        "timestamp": reading.timestamp,
                        "plant_id": reading.plant_id,
                        "sensor_id": reading.sensor.id,
                        "sensor_type": reading.sensor.type.value,
                        "sensor_location": reading.sensor.location.value,
                        "elevation_m": reading.sensor.elevation_m,
                        "value": reading.measurement.value,
                        "unit": reading.measurement.unit.value,
                        "quality": reading.measurement.quality.value,
                        "raw_counts": reading.measurement.raw_counts,
                        "fault_type": fault_type,
                        "is_usable": reading.is_usable,
                    }
                )

        _LOG.info("adapt_all complete: %d total readings.", len(records))
        return pd.DataFrame(records)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    adapter = TEPAdapter()
    result_df = adapter.adapt_all("data/raw/tep")
    save_to_parquet(result_df, "data/raw/tep")
    print(f"Total readings: {len(result_df):,}")
    print("Distribution by fault_type:")
    for ft, count in result_df.groupby("fault_type").size().items():
        label = "normal" if ft == 0 else f"fault_{ft:02d}"
        print(f"  fault_type={ft:2d} ({label}): {count:,} readings")
