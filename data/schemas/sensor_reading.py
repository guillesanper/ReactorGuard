"""Pydantic v2 schema for a single reactor sensor reading (TDD section 4.2).

This module is the authoritative data contract between SCADA/Kafka producers and
all downstream consumers (ML inference, API, data validation, generators).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SensorType(StrEnum):
    """Physical sensing principle of the instrument."""

    THERMOCOUPLE = "thermocouple"
    FLUX_DETECTOR = "flux_detector"
    PRESSURE = "pressure"
    FLOW = "flow"
    POSITION = "position"
    NORMALIZED = "normalized"


class SensorLocation(StrEnum):
    """Topological zone within the reactor building where the sensor is installed."""

    PRIMARY_LOOP = "primary_loop"
    SECONDARY_LOOP = "secondary_loop"
    CORE = "core"
    CONTAINMENT = "containment"


class MeasurementUnit(StrEnum):
    """Engineering unit of the measured value."""

    CELSIUS = "celsius"
    BAR = "bar"
    KG_S = "kg_s"
    PERCENT = "percent"
    NORMALIZED = "normalized"


class QualityFlag(StrEnum):
    """Data quality status as assessed by the SCADA system."""

    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    MISSING = "missing"


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class SensorInfo(BaseModel):
    """Static descriptor of the physical sensor instrument."""

    id: str = Field(..., description="Unique instrument tag (e.g. TC-CORE-12)")
    type: SensorType = Field(..., description="Sensing principle")
    location: SensorLocation = Field(..., description="Topological zone in the plant")
    elevation_m: float = Field(
        ...,
        ge=-10.0,
        le=100.0,
        description=(
            "Sensor elevation relative to plant datum [m]. Valid range covers "
            "basement to top of reactor building."
        ),
    )


class Measurement(BaseModel):
    """Digitised process value as delivered by the SCADA analog input module."""

    value: float | None = Field(
        default=None,
        description="Engineering-unit value; may be None when quality is 'missing'.",
    )
    unit: MeasurementUnit = Field(..., description="Engineering unit of value")
    quality: QualityFlag = Field(..., description="SCADA quality status")
    raw_counts: int = Field(
        ...,
        ge=0,
        le=65535,
        description="Raw ADC counts from the 16-bit analog input card (0-65535).",
    )


class SensorMetadata(BaseModel):
    """Instrument health and calibration traceability information."""

    calibration_date: date = Field(..., description="Date of the last calibration")
    last_maintenance: date = Field(..., description="Date of the last physical maintenance")
    drift_coefficient: float = Field(
        ...,
        ge=0.0,
        description="Estimated sensor drift per day (dimensionless, must be non-negative).",
    )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class SensorReading(BaseModel):
    """Complete sensor reading record as produced by SCADA and published to Kafka.

    This is the canonical data contract for ReactorGuard (TDD section 4.2).
    All producers must emit messages conforming to this schema; all consumers
    must validate incoming bytes against it before processing.
    """

    reading_id: UUID = Field(..., description="Universally unique identifier (v4) for this record")
    timestamp: datetime = Field(..., description="UTC timestamp of the measurement (ISO 8601)")
    plant_id: str = Field(..., description="Reactor plant identifier (e.g. REACTOR-01)")
    sensor: SensorInfo
    measurement: Measurement
    metadata: SensorMetadata

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_usable(self) -> bool:
        """Return True only when the reading is safe to use for ML inference.

        A reading is considered unusable when quality is 'bad' (hardware fault)
        or 'missing' (no signal received). 'suspect' readings are still usable
        but should be treated with caution by downstream models.
        """
        return self.measurement.quality not in (QualityFlag.BAD, QualityFlag.MISSING)

    # ------------------------------------------------------------------
    # Kafka serialisation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_kafka_bytes(cls, data: bytes) -> SensorReading:
        """Deserialise a sensor reading from raw Kafka message bytes (UTF-8 JSON).

        Args:
            data: Raw bytes from a Kafka ConsumerRecord value.

        Returns:
            A validated SensorReading instance.

        Raises:
            pydantic.ValidationError: If the payload does not conform to this schema.
            json.JSONDecodeError: If the bytes are not valid JSON.
        """
        return cls.model_validate_json(data)

    def to_kafka_bytes(self) -> bytes:
        """Serialise this reading to UTF-8 JSON bytes suitable for a Kafka producer.

        UUIDs and datetimes are serialised to their standard string representations.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        return self.model_dump_json().encode("utf-8")

    # ------------------------------------------------------------------
    # ML feature extraction
    # ------------------------------------------------------------------

    def to_feature_dict(self) -> dict[str, object]:
        """Return the numeric fields required by the ML inference pipeline.

        Only scalar numeric values are included; categorical fields are excluded
        because feature engineering (encoding) is the responsibility of the
        feature store layer, not this schema.

        Returns:
            A flat dictionary with keys:
            sensor_id, timestamp_unix, value, raw_counts, drift_coefficient, elevation_m.
        """
        return {
            "sensor_id": self.sensor.id,
            "timestamp_unix": self.timestamp.timestamp(),
            "value": self.measurement.value,
            "raw_counts": self.measurement.raw_counts,
            "drift_coefficient": self.metadata.drift_coefficient,
            "elevation_m": self.sensor.elevation_m,
        }
