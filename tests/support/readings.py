"""Builders of synthetic SensorReading batches for the storage tests."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta

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

_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")
_TYPES = list(SensorType)
_LOCATIONS = list(SensorLocation)
_UNITS = list(MeasurementUnit)
_QUALITIES = [QualityFlag.GOOD, QualityFlag.SUSPECT, QualityFlag.BAD]
_MAX_COUNTS = 65535


def make_reading(
    index: int,
    timestamp: datetime,
    *,
    plant_id: str = "TEP-PLANT-01",
    sensor_id: str | None = None,
) -> SensorReading:
    """Build one deterministic reading that exercises every enum and None.

    Args:
        index: Position in the batch; drives every varying field.
        timestamp: Timezone-aware timestamp of the reading.
        plant_id: Plant identifier.
        sensor_id: Sensor tag; defaults to one of 52 TEP-like tags.

    Returns:
        A reading whose reading_id is a UUID5 of (plant, index, timestamp), so
        the same arguments always give the same identifier.
    """
    missing = index % 7 == 0
    quality = QualityFlag.MISSING if missing else _QUALITIES[index % len(_QUALITIES)]
    return SensorReading(
        reading_id=uuid.uuid5(_NAMESPACE, f"{plant_id}|{index}|{timestamp.isoformat()}"),
        timestamp=timestamp,
        plant_id=plant_id,
        sensor=SensorInfo(
            id=sensor_id or f"TEP-XMEAS-{index % 52 + 1:02d}",
            type=_TYPES[index % len(_TYPES)],
            location=_LOCATIONS[index % len(_LOCATIONS)],
            elevation_m=float(index % 111) - 10.0,
        ),
        measurement=Measurement(
            value=None if missing else index * 0.37 - 5.0,
            unit=_UNITS[index % len(_UNITS)],
            quality=quality,
            raw_counts=(index * 977) % (_MAX_COUNTS + 1),
        ),
        metadata=SensorMetadata(
            calibration_date=date(2023, 6, 1) + timedelta(days=index % 5),
            last_maintenance=date(2023, 12, 1),
            drift_coefficient=0.0001 * (index % 3),
        ),
    )


def make_value_reading(
    sensor_id: str,
    timestamp: datetime,
    value: float | None,
    *,
    plant_id: str = "TEP-PLANT-01",
) -> SensorReading:
    """Build a GOOD reading of a given sensor, timestamp and value.

    Args:
        sensor_id: Sensor tag.
        timestamp: Timestamp of the reading.
        value: Measured value, or None for a reading without one.
        plant_id: Plant identifier.

    Returns:
        A reading whose reading_id is a UUID5 of (plant, sensor, timestamp), the
        way the adapter derives it, so the same arguments give the same identifier.
    """
    return SensorReading(
        reading_id=uuid.uuid5(_NAMESPACE, f"{plant_id}|{sensor_id}|{timestamp.isoformat()}"),
        timestamp=timestamp,
        plant_id=plant_id,
        sensor=SensorInfo(
            id=sensor_id,
            type=SensorType.NORMALIZED,
            location=SensorLocation.PRIMARY_LOOP,
            elevation_m=0.0,
        ),
        measurement=Measurement(
            value=value,
            unit=MeasurementUnit.NORMALIZED,
            quality=QualityFlag.GOOD,
            raw_counts=1000,
        ),
        metadata=SensorMetadata(
            calibration_date=date(2023, 6, 1),
            last_maintenance=date(2023, 12, 1),
            drift_coefficient=0.0001,
        ),
    )


def make_batch(
    count: int,
    start: datetime | None = None,
    *,
    step: timedelta = timedelta(minutes=3),
    plant_id: str = "TEP-PLANT-01",
    first_index: int = 0,
) -> list[SensorReading]:
    """Build a time-ordered batch of readings.

    Args:
        count: Number of readings.
        start: Timestamp of the first reading; defaults to 2000-01-01T00:00:00Z
            plus a microsecond, so sub-second precision is exercised.
        step: Spacing between consecutive readings.
        plant_id: Plant identifier.
        first_index: Index of the first reading, to build non-overlapping batches.

    Returns:
        The readings, ordered by timestamp.
    """
    origin = start or datetime(2000, 1, 1, tzinfo=UTC) + timedelta(microseconds=1)
    return [
        make_reading(first_index + i, origin + i * step, plant_id=plant_id)
        for i in range(count)
    ]
