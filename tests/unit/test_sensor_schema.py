"""Unit tests for SensorReading schema validation."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from data.schemas.sensor_reading import SensorReading


VALID_READING = {
    "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
    "reactor_id": "R-001",
    "core_power": 3000.0,
    "coolant_temp_in": 564.0,
    "coolant_temp_out": 594.0,
    "primary_pressure": 15.5,
    "coolant_flow_rate": 18000.0,
    "fuel_temp": 900.0,
    "neutron_flux_ex_core": 3.2e13,
    "steam_generator_level": 4.5,
}


def test_valid_reading_parses() -> None:
    r = SensorReading(**VALID_READING)
    assert r.reactor_id == "R-001"
    assert r.is_anomaly is False


def test_outlet_below_inlet_raises() -> None:
    bad = {**VALID_READING, "coolant_temp_out": 560.0}  # < inlet (564 K)
    with pytest.raises(ValidationError, match="Outlet temp"):
        SensorReading(**bad)


def test_negative_power_raises() -> None:
    bad = {**VALID_READING, "core_power": -100.0}
    with pytest.raises(ValidationError):
        SensorReading(**bad)
