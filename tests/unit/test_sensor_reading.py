"""Unit tests for data.schemas.sensor_reading (TDD section 4.2).

Each test class covers one behavioural concern of SensorReading and its
nested models. All fixtures use realistic reactor instrument values.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from data.schemas.sensor_reading import (
    MeasurementUnit,
    QualityFlag,
    SensorInfo,
    SensorLocation,
    SensorMetadata,
    SensorReading,
    SensorType,
    Measurement,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_BASE_READING_ID = uuid.UUID("12345678-1234-4234-b234-123456789abc")
_BASE_TIMESTAMP = datetime(2024, 1, 15, 14, 32, 18, 1000, tzinfo=timezone.utc)


def _make_reading(**overrides) -> dict:
    """Return a minimal valid raw dict for SensorReading, with optional overrides."""
    base = {
        "reading_id": str(_BASE_READING_ID),
        "timestamp": _BASE_TIMESTAMP.isoformat(),
        "plant_id": "REACTOR-01",
        "sensor": {
            "id": "TC-CORE-12",
            "type": "thermocouple",
            "location": "core",
            "elevation_m": 3.45,
        },
        "measurement": {
            "value": 312.4,
            "unit": "celsius",
            "quality": "good",
            "raw_counts": 4092,
        },
        "metadata": {
            "calibration_date": "2024-01-01",
            "last_maintenance": "2023-12-15",
            "drift_coefficient": 0.0012,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Validation: happy paths
# ---------------------------------------------------------------------------


class TestValidReading:
    """Schema must accept fully populated, physically plausible readings."""

    def test_complete_valid_reading(self):
        """A reading with all fields within bounds must validate without error."""
        reading = SensorReading.model_validate(_make_reading())
        assert reading.plant_id == "REACTOR-01"
        assert reading.sensor.type == SensorType.THERMOCOUPLE
        assert reading.measurement.value == pytest.approx(312.4)

    def test_quality_missing_with_none_value(self):
        """quality='missing' combined with value=None must be accepted."""
        data = _make_reading()
        data["measurement"]["quality"] = "missing"
        data["measurement"]["value"] = None
        reading = SensorReading.model_validate(data)
        assert reading.measurement.quality == QualityFlag.MISSING
        assert reading.measurement.value is None

    def test_quality_missing_with_numeric_value(self):
        """quality='missing' with a numeric value must also be accepted.

        A sensor can report a stale/uncertain value while SCADA flags quality as
        missing because the signal path health check failed independently.
        """
        data = _make_reading()
        data["measurement"]["quality"] = "missing"
        data["measurement"]["value"] = 312.4
        reading = SensorReading.model_validate(data)
        assert reading.measurement.quality == QualityFlag.MISSING
        assert reading.measurement.value == pytest.approx(312.4)

    def test_all_sensor_types_accepted(self):
        """Every member of SensorType enum must pass field-level validation."""
        for sensor_type in SensorType:
            data = _make_reading()
            data["sensor"]["type"] = sensor_type.value
            reading = SensorReading.model_validate(data)
            assert reading.sensor.type == sensor_type

    def test_all_locations_accepted(self):
        """Every member of SensorLocation enum must pass field-level validation."""
        for location in SensorLocation:
            data = _make_reading()
            data["sensor"]["location"] = location.value
            reading = SensorReading.model_validate(data)
            assert reading.sensor.location == location


# ---------------------------------------------------------------------------
# Validation: constraint violations
# ---------------------------------------------------------------------------


class TestConstraintViolations:
    """Schema must reject readings that violate physical or ADC constraints."""

    def test_raw_counts_above_16bit_maximum(self):
        """raw_counts=70000 exceeds 16-bit ADC range and must raise ValidationError."""
        data = _make_reading()
        data["measurement"]["raw_counts"] = 70000
        with pytest.raises(ValidationError) as exc_info:
            SensorReading.model_validate(data)
        errors = exc_info.value.errors()
        assert any("raw_counts" in str(e["loc"]) for e in errors)

    def test_raw_counts_below_zero(self):
        """raw_counts=-1 is physically impossible for an ADC and must be rejected."""
        data = _make_reading()
        data["measurement"]["raw_counts"] = -1
        with pytest.raises(ValidationError):
            SensorReading.model_validate(data)

    def test_drift_coefficient_negative(self):
        """A negative drift coefficient is physically meaningless and must be rejected."""
        data = _make_reading()
        data["metadata"]["drift_coefficient"] = -0.001
        with pytest.raises(ValidationError) as exc_info:
            SensorReading.model_validate(data)
        errors = exc_info.value.errors()
        assert any("drift_coefficient" in str(e["loc"]) for e in errors)

    def test_elevation_below_minimum(self):
        """elevation_m below -10.0 is outside the physical building envelope."""
        data = _make_reading()
        data["sensor"]["elevation_m"] = -11.0
        with pytest.raises(ValidationError):
            SensorReading.model_validate(data)

    def test_elevation_above_maximum(self):
        """elevation_m above 100.0 is outside the physical building envelope."""
        data = _make_reading()
        data["sensor"]["elevation_m"] = 101.0
        with pytest.raises(ValidationError):
            SensorReading.model_validate(data)


# ---------------------------------------------------------------------------
# Computed property: is_usable
# ---------------------------------------------------------------------------


class TestIsUsable:
    """is_usable must correctly reflect whether a reading is safe for ML inference."""

    def test_is_usable_true_for_good_quality(self):
        """quality='good' must yield is_usable=True."""
        reading = SensorReading.model_validate(_make_reading())
        assert reading.is_usable is True

    def test_is_usable_true_for_suspect_quality(self):
        """quality='suspect' must yield is_usable=True (caution, but still usable)."""
        data = _make_reading()
        data["measurement"]["quality"] = "suspect"
        reading = SensorReading.model_validate(data)
        assert reading.is_usable is True

    def test_is_usable_false_for_bad_quality(self):
        """quality='bad' indicates a hardware fault; is_usable must be False."""
        data = _make_reading()
        data["measurement"]["quality"] = "bad"
        reading = SensorReading.model_validate(data)
        assert reading.is_usable is False

    def test_is_usable_false_for_missing_quality(self):
        """quality='missing' means no signal received; is_usable must be False."""
        data = _make_reading()
        data["measurement"]["quality"] = "missing"
        data["measurement"]["value"] = None
        reading = SensorReading.model_validate(data)
        assert reading.is_usable is False


# ---------------------------------------------------------------------------
# Kafka serialisation round-trip
# ---------------------------------------------------------------------------


class TestKafkaSerialization:
    """from_kafka_bytes and to_kafka_bytes must form a lossless round-trip."""

    def test_roundtrip_preserves_all_fields(self):
        """Deserialising the output of to_kafka_bytes must reproduce the original."""
        original = SensorReading.model_validate(_make_reading())
        kafka_bytes = original.to_kafka_bytes()

        assert isinstance(kafka_bytes, bytes)

        restored = SensorReading.from_kafka_bytes(kafka_bytes)

        assert restored.reading_id == original.reading_id
        assert restored.timestamp == original.timestamp
        assert restored.plant_id == original.plant_id
        assert restored.sensor.id == original.sensor.id
        assert restored.sensor.type == original.sensor.type
        assert restored.sensor.location == original.sensor.location
        assert restored.sensor.elevation_m == pytest.approx(original.sensor.elevation_m)
        assert restored.measurement.value == pytest.approx(original.measurement.value)
        assert restored.measurement.unit == original.measurement.unit
        assert restored.measurement.quality == original.measurement.quality
        assert restored.measurement.raw_counts == original.measurement.raw_counts
        assert restored.metadata.calibration_date == original.metadata.calibration_date
        assert restored.metadata.last_maintenance == original.metadata.last_maintenance
        assert restored.metadata.drift_coefficient == pytest.approx(
            original.metadata.drift_coefficient
        )

    def test_roundtrip_with_none_value(self):
        """Round-trip must preserve value=None without coercion."""
        data = _make_reading()
        data["measurement"]["quality"] = "missing"
        data["measurement"]["value"] = None
        original = SensorReading.model_validate(data)
        restored = SensorReading.from_kafka_bytes(original.to_kafka_bytes())
        assert restored.measurement.value is None

    def test_from_kafka_bytes_rejects_invalid_payload(self):
        """Malformed JSON bytes must raise ValidationError or JSONDecodeError."""
        with pytest.raises(Exception):
            SensorReading.from_kafka_bytes(b'{"plant_id": "REACTOR-01"}')


# ---------------------------------------------------------------------------
# ML feature extraction
# ---------------------------------------------------------------------------


class TestToFeatureDict:
    """to_feature_dict must return exactly the numeric fields needed by the ML layer."""

    _EXPECTED_KEYS = {
        "sensor_id",
        "timestamp_unix",
        "value",
        "raw_counts",
        "drift_coefficient",
        "elevation_m",
    }

    def test_returns_exactly_expected_keys(self):
        """Feature dict must contain exactly the six defined numeric fields."""
        reading = SensorReading.model_validate(_make_reading())
        feature_dict = reading.to_feature_dict()
        assert set(feature_dict.keys()) == self._EXPECTED_KEYS

    def test_sensor_id_matches_sensor_info_id(self):
        """sensor_id in feature dict must equal sensor.id."""
        reading = SensorReading.model_validate(_make_reading())
        assert reading.to_feature_dict()["sensor_id"] == "TC-CORE-12"

    def test_timestamp_unix_is_float(self):
        """timestamp_unix must be a numeric POSIX timestamp."""
        reading = SensorReading.model_validate(_make_reading())
        ts = reading.to_feature_dict()["timestamp_unix"]
        assert isinstance(ts, float)
        assert ts > 0.0

    def test_value_none_propagates_to_feature_dict(self):
        """A None value must propagate into the feature dict unchanged."""
        data = _make_reading()
        data["measurement"]["quality"] = "missing"
        data["measurement"]["value"] = None
        reading = SensorReading.model_validate(data)
        assert reading.to_feature_dict()["value"] is None

    def test_numeric_values_match_source_fields(self):
        """Each numeric field in the feature dict must match its source field."""
        reading = SensorReading.model_validate(_make_reading())
        fd = reading.to_feature_dict()
        assert fd["value"] == pytest.approx(reading.measurement.value)
        assert fd["raw_counts"] == reading.measurement.raw_counts
        assert fd["drift_coefficient"] == pytest.approx(reading.metadata.drift_coefficient)
        assert fd["elevation_m"] == pytest.approx(reading.sensor.elevation_m)
