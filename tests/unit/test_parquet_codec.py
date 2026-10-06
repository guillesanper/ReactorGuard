"""Tests for data/storage/parquet_codec.py."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.generators.tep_adapter import readings_from_frame
from data.schemas.sensor_reading import (
    Measurement,
    QualityFlag,
    SensorReading,
)
from data.storage import parquet_codec
from data.storage.parquet_codec import (
    READINGS_SCHEMA,
    SCHEMA_VERSION,
    bytes_to_table,
    readings_to_table,
    table_metadata,
    table_to_bytes,
    table_to_readings,
)
from tests.support.readings import make_batch, make_reading

_CREATED_AT = datetime(2024, 5, 5, 10, 0, 0, tzinfo=UTC)
_READING_COUNT = 100


def _roundtrip(readings: list[SensorReading]) -> list[SensorReading]:
    """Encode to Parquet bytes and decode again.

    Args:
        readings: Readings to round-trip.

    Returns:
        The decoded readings.
    """
    payload = table_to_bytes(readings_to_table(readings, _CREATED_AT))
    return table_to_readings(bytes_to_table(payload))


def test_roundtrip_of_100_readings_is_field_for_field_equal() -> None:
    readings = make_batch(_READING_COUNT)
    assert _roundtrip(readings) == readings


def test_roundtrip_exercises_none_and_every_enum_member() -> None:
    readings = make_batch(_READING_COUNT)
    assert any(r.measurement.value is None for r in readings)
    assert {r.measurement.quality for r in readings} == set(QualityFlag)
    decoded = _roundtrip(readings)
    assert [r.measurement.value for r in decoded] == [r.measurement.value for r in readings]
    assert [r.sensor.type for r in decoded] == [r.sensor.type for r in readings]
    assert [r.sensor.location for r in decoded] == [r.sensor.location for r in readings]
    assert [r.measurement.unit for r in decoded] == [r.measurement.unit for r in readings]


def test_roundtrip_keeps_microseconds_and_normalises_zone_to_utc() -> None:
    zone = timezone(timedelta(hours=-5))
    local = datetime(2024, 12, 31, 23, 30, 0, 123_456, tzinfo=zone)
    decoded = _roundtrip([make_reading(1, local)])[0]
    assert decoded.timestamp == local
    assert decoded.timestamp.utcoffset() == timedelta(0)
    assert decoded.timestamp.microsecond == 123_456


def test_roundtrip_keeps_extreme_raw_counts_and_dates() -> None:
    reading = make_reading(1, _CREATED_AT)
    low = reading.model_copy(
        update={"measurement": Measurement(
            value=0.0, unit=reading.measurement.unit,
            quality=QualityFlag.GOOD, raw_counts=0)}
    )
    high = reading.model_copy(
        update={"measurement": Measurement(
            value=1.0, unit=reading.measurement.unit,
            quality=QualityFlag.GOOD, raw_counts=65535)}
    )
    decoded = _roundtrip([low, high])
    assert [r.measurement.raw_counts for r in decoded] == [0, 65535]
    assert decoded[0].metadata.calibration_date == date(2023, 6, 2)


def test_empty_batch_roundtrips_to_an_empty_list() -> None:
    table = readings_to_table([], _CREATED_AT)
    assert table.num_rows == 0
    assert table_metadata(table)["record_count"] == "0"
    assert _roundtrip([]) == []


def test_schema_is_explicit_and_stable() -> None:
    table = readings_to_table(make_batch(3), _CREATED_AT)
    assert table.schema.remove_metadata().equals(READINGS_SCHEMA.remove_metadata())
    assert table.schema.field("timestamp").type == pa.timestamp("us", tz="UTC")
    assert table.schema.field("raw_counts").type == pa.uint16()
    assert table.schema.field("value").nullable
    assert table.column_names == [
        "reading_id", "timestamp", "plant_id", "sensor_id", "sensor_type",
        "sensor_location", "elevation_m", "value", "unit", "quality", "raw_counts",
        "calibration_date", "last_maintenance", "drift_coefficient",
    ]


def test_all_null_value_column_keeps_its_type() -> None:
    reading = make_reading(7, _CREATED_AT)
    assert reading.measurement.value is None
    table = bytes_to_table(table_to_bytes(readings_to_table([reading], _CREATED_AT)))
    assert table.schema.field("value").type == pa.float64()


def test_metadata_survives_the_parquet_roundtrip() -> None:
    table = bytes_to_table(table_to_bytes(readings_to_table(make_batch(5), _CREATED_AT)))
    assert table_metadata(table) == {
        "schema_version": SCHEMA_VERSION,
        "created_at": "2024-05-05T10:00:00+00:00",
        "record_count": "5",
    }


def test_created_at_is_normalised_to_utc() -> None:
    zone = timezone(timedelta(hours=2))
    table = readings_to_table([], datetime(2024, 5, 5, 12, 0, tzinfo=zone))
    assert table_metadata(table)["created_at"] == "2024-05-05T10:00:00+00:00"


def test_naive_created_at_is_rejected() -> None:
    with pytest.raises(ValueError, match="no timezone"):
        readings_to_table([], datetime(2024, 5, 5, 12, 0))


def test_naive_reading_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="no timezone"):
        readings_to_table([make_reading(1, datetime(2024, 5, 5, 12, 0))], _CREATED_AT)


def test_part_is_snappy_compressed() -> None:
    data = table_to_bytes(readings_to_table(make_batch(10), _CREATED_AT))
    column = pq.ParquetFile(pa.BufferReader(data)).metadata.row_group(0).column(0)
    assert column.compression == "SNAPPY"
    assert parquet_codec.COMPRESSION == "snappy"


def test_stored_part_feeds_readings_from_frame() -> None:
    """The columns are those readings_from_frame consumes, so no second format exists."""
    readings = make_batch(20)
    frame = bytes_to_table(table_to_bytes(readings_to_table(readings, _CREATED_AT))).to_pandas()
    first = readings[0].metadata
    rebuilt = list(
        readings_from_frame(
            frame,
            calibration_date=first.calibration_date,
            last_maintenance=first.last_maintenance,
            drift_coefficient=first.drift_coefficient,
        )
    )
    assert [r.reading_id for r in rebuilt] == [r.reading_id for r in readings]
    assert [r.timestamp for r in rebuilt] == [r.timestamp for r in readings]
    assert [r.measurement.value for r in rebuilt] == [r.measurement.value for r in readings]
    assert [r.measurement.raw_counts for r in rebuilt] == [
        r.measurement.raw_counts for r in readings
    ]


def test_foreign_schema_fails_loudly() -> None:
    table = pa.table({"reading_id": ["x"], "value": [1.0]})
    with pytest.raises(ValueError, match="does not match the readings schema"):
        table_to_readings(table)


def test_changed_column_type_fails_loudly() -> None:
    table = readings_to_table(make_batch(3), _CREATED_AT)
    index = table.schema.get_field_index("raw_counts")
    retyped = table.set_column(
        index,
        pa.field("raw_counts", pa.int64(), nullable=False),
        table.column("raw_counts").cast(pa.int64()),
    )
    with pytest.raises(ValueError, match="does not match the readings schema"):
        table_to_readings(retyped)


@pytest.mark.parametrize("version", ["0", "2", None])
def test_other_schema_version_is_rejected(version: str | None) -> None:
    table = readings_to_table(make_batch(3), _CREATED_AT)
    metadata = {} if version is None else {b"schema_version": version.encode()}
    with pytest.raises(ValueError, match="schema_version"):
        table_to_readings(table.replace_schema_metadata(metadata))


def test_garbage_bytes_are_a_value_error() -> None:
    with pytest.raises(ValueError):
        bytes_to_table(b"this is not parquet")


def test_table_metadata_of_a_table_without_any_is_empty() -> None:
    assert table_metadata(pa.table({"a": [1]})) == {}
