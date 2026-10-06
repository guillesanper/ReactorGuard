"""Parquet encoding of SensorReading batches with an explicit Arrow schema.

El esquema es explicito (no inferido de los datos) por dos razones. Una columna
`value` toda nula se inferiria como `null` en lugar de `double`, y dos partes de
la misma hora con tipos distintos romperian una lectura de rango. Y un cambio de
esquema debe fallar en voz alta al leer, no producir lecturas mal formadas.

Las columnas son las que consume `readings_from_frame`
(data/generators/tep_adapter.py) con los mismos nombres que el parquet largo del
TEP, de modo que una parte almacenada se puede cargar con pandas y pasar a esa
funcion. Se anaden tres columnas de metadatos de instrumento (calibration_date,
last_maintenance, drift_coefficient): el formato largo del adaptador las descarta
porque son constantes de la corrida, pero un almacen de lecturas arbitrarias no
puede suponerlo, y sin ellas la lectura no devolveria lo que se escribio.

No se guarda `is_usable`: es una propiedad derivada de `quality` y almacenarla
permitiria que ambas se contradijeran.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, date, datetime

import pyarrow as pa
import pyarrow.parquet as pq

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
from data.storage.layout import to_utc

SCHEMA_VERSION = "1"
COMPRESSION = "snappy"

META_SCHEMA_VERSION = b"schema_version"
META_CREATED_AT = b"created_at"
META_RECORD_COUNT = b"record_count"

READINGS_SCHEMA = pa.schema(
    [
        pa.field("reading_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("plant_id", pa.string(), nullable=False),
        pa.field("sensor_id", pa.string(), nullable=False),
        pa.field("sensor_type", pa.string(), nullable=False),
        pa.field("sensor_location", pa.string(), nullable=False),
        pa.field("elevation_m", pa.float64(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("quality", pa.string(), nullable=False),
        pa.field("raw_counts", pa.uint16(), nullable=False),
        pa.field("calibration_date", pa.date32(), nullable=False),
        pa.field("last_maintenance", pa.date32(), nullable=False),
        pa.field("drift_coefficient", pa.float64(), nullable=False),
    ]
)


def readings_to_table(readings: Sequence[SensorReading], created_at: datetime) -> pa.Table:
    """Build an Arrow table of readings with the explicit schema and file metadata.

    Args:
        readings: Readings to encode, in the order they must be stored.
        created_at: Instant stamped in the table metadata, timezone-aware.

    Returns:
        A table whose schema equals READINGS_SCHEMA plus the metadata keys
        schema_version, created_at and record_count.

    Raises:
        ValueError: If created_at or any reading timestamp has no timezone.
    """
    columns: dict[str, list[object]] = {
        "reading_id": [str(r.reading_id) for r in readings],
        "timestamp": [to_utc(r.timestamp) for r in readings],
        "plant_id": [r.plant_id for r in readings],
        "sensor_id": [r.sensor.id for r in readings],
        "sensor_type": [r.sensor.type.value for r in readings],
        "sensor_location": [r.sensor.location.value for r in readings],
        "elevation_m": [r.sensor.elevation_m for r in readings],
        "value": [r.measurement.value for r in readings],
        "unit": [r.measurement.unit.value for r in readings],
        "quality": [r.measurement.quality.value for r in readings],
        "raw_counts": [r.measurement.raw_counts for r in readings],
        "calibration_date": [r.metadata.calibration_date for r in readings],
        "last_maintenance": [r.metadata.last_maintenance for r in readings],
        "drift_coefficient": [r.metadata.drift_coefficient for r in readings],
    }
    metadata = {
        META_SCHEMA_VERSION: SCHEMA_VERSION.encode(),
        META_CREATED_AT: to_utc(created_at).isoformat().encode(),
        META_RECORD_COUNT: str(len(readings)).encode(),
    }
    schema = READINGS_SCHEMA.with_metadata(metadata)
    return pa.Table.from_pydict(columns, schema=schema)


def table_metadata(table: pa.Table) -> dict[str, str]:
    """Return the schema metadata of a table as plain strings.

    Args:
        table: A table, typically read back from a stored part.

    Returns:
        The metadata mapping; empty when the table carries none.
    """
    raw = table.schema.metadata or {}
    return {key.decode(): value.decode() for key, value in raw.items()}


def _check_schema(table: pa.Table) -> None:
    """Verify that a table matches the readings schema and version.

    Args:
        table: The table to check.

    Raises:
        ValueError: If the columns or types differ from READINGS_SCHEMA, or the
            schema_version metadata is missing or not SCHEMA_VERSION.
    """
    expected = READINGS_SCHEMA.remove_metadata()
    actual = table.schema.remove_metadata()
    if not actual.equals(expected):
        raise ValueError(
            "Stored part does not match the readings schema.\n"
            f"  expected: {expected.to_string(show_schema_metadata=False)}\n"
            f"  found:    {actual.to_string(show_schema_metadata=False)}"
        )
    version = table_metadata(table).get(META_SCHEMA_VERSION.decode())
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Stored part has schema_version {version!r}; this reader supports "
            f"{SCHEMA_VERSION!r}."
        )


def table_to_readings(table: pa.Table) -> list[SensorReading]:
    """Rebuild SensorReading objects from a stored table.

    Args:
        table: A table produced by readings_to_table (possibly after a Parquet
            round trip).

    Returns:
        One validated SensorReading per row, in row order.

    Raises:
        ValueError: If the schema or its version does not match, or a row does
            not satisfy the data contract (pydantic.ValidationError subclasses
            ValueError).
    """
    _check_schema(table)
    column = {name: table.column(name).to_pylist() for name in table.column_names}
    readings: list[SensorReading] = []
    for index in range(table.num_rows):
        timestamp: datetime = column["timestamp"][index]
        calibration: date = column["calibration_date"][index]
        maintenance: date = column["last_maintenance"][index]
        readings.append(
            SensorReading(
                reading_id=uuid.UUID(column["reading_id"][index]),
                timestamp=timestamp.astimezone(UTC),
                plant_id=column["plant_id"][index],
                sensor=SensorInfo(
                    id=column["sensor_id"][index],
                    type=SensorType(column["sensor_type"][index]),
                    location=SensorLocation(column["sensor_location"][index]),
                    elevation_m=column["elevation_m"][index],
                ),
                measurement=Measurement(
                    value=column["value"][index],
                    unit=MeasurementUnit(column["unit"][index]),
                    quality=QualityFlag(column["quality"][index]),
                    raw_counts=column["raw_counts"][index],
                ),
                metadata=SensorMetadata(
                    calibration_date=calibration,
                    last_maintenance=maintenance,
                    drift_coefficient=column["drift_coefficient"][index],
                ),
            )
        )
    return readings


def table_to_bytes(table: pa.Table) -> bytes:
    """Serialize a table to Parquet bytes (snappy), keeping its metadata.

    Args:
        table: The table to serialize.

    Returns:
        The complete Parquet file.
    """
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=COMPRESSION)  # type: ignore[no-untyped-call]
    return sink.getvalue().to_pybytes()  # type: ignore[no-any-return]


def bytes_to_table(data: bytes) -> pa.Table:
    """Parse Parquet bytes into a table.

    Args:
        data: A complete Parquet file.

    Returns:
        The decoded table, with its schema metadata.

    Raises:
        ValueError: If the bytes are not a valid Parquet file (pyarrow raises
            ArrowInvalid, a ValueError subclass).
    """
    return pq.read_table(pa.BufferReader(data))  # type: ignore[no-untyped-call]
