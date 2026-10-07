"""Tests for data/validation/alert_event.py."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from data.validation.alert_event import AlertEvent, AlertSource, alert_uuid
from data.validation.sensor_fault import FaultType, SensorFault, Severity
from tests.support.readings import make_value_reading

NOW = datetime(2000, 1, 1, 12, 0, tzinfo=UTC)
SOURCE = AlertSource(topic="sensor-readings-raw", partition=3, offset=41)


def _fault(**overrides: Any) -> SensorFault:
    fields: dict[str, Any] = {
        "sensor_id": "TEP-XMV-04",
        "fault_type": FaultType.STUCK,
        "severity": Severity.MEDIUM,
        "confidence": 0.5,
        "detected_at": NOW,
        "detector": "StuckValueDetector",
        "evidence": {"run_length": 8, "window_size": 8, "value": 50.0, "tolerance": 0.0},
    }
    fields.update(overrides)
    return SensorFault(**fields)


def _event(**overrides: Any) -> AlertEvent:
    reading = make_value_reading("TEP-XMV-04", NOW, 50.0)
    return AlertEvent.from_fault(_fault(), reading, SOURCE, **overrides)


class TestAlertUuid:
    def test_the_same_fault_on_the_same_reading_gets_the_same_id(self) -> None:
        reading = make_value_reading("TEP-XMV-04", NOW, 50.0)
        first = alert_uuid(reading.reading_id, FaultType.STUCK, "StuckValueDetector")
        again = alert_uuid(reading.reading_id, FaultType.STUCK, "StuckValueDetector")
        assert first == again

    def test_each_component_of_the_key_changes_the_id(self) -> None:
        reading = make_value_reading("TEP-XMV-04", NOW, 50.0)
        other = make_value_reading("TEP-XMV-05", NOW, 50.0)
        base = alert_uuid(reading.reading_id, FaultType.STUCK, "StuckValueDetector")
        assert alert_uuid(other.reading_id, FaultType.STUCK, "StuckValueDetector") != base
        assert alert_uuid(reading.reading_id, FaultType.NOISE_SPIKE, "StuckValueDetector") != base
        assert alert_uuid(reading.reading_id, FaultType.STUCK, "RateOfChangeDetector") != base

    def test_occurrences_of_the_same_fault_are_told_apart(self) -> None:
        reading = make_value_reading("TEP-XMV-04", NOW, 50.0)
        ids = {
            alert_uuid(reading.reading_id, FaultType.DRIFT_CORRELATED, "CrossCorrelationChecker", n)
            for n in range(3)
        }
        assert len(ids) == 3

    def test_occurrence_zero_keeps_the_plain_key(self) -> None:
        reading = make_value_reading("TEP-XMV-04", NOW, 50.0)
        assert alert_uuid(
            reading.reading_id, FaultType.STUCK, "StuckValueDetector", 0
        ) == alert_uuid(reading.reading_id, FaultType.STUCK, "StuckValueDetector")


class TestFromFault:
    def test_copies_the_fault_and_the_reading_fields(self) -> None:
        event = _event(run_id="fault_type=21/loop=0")
        reading = make_value_reading("TEP-XMV-04", NOW, 50.0)
        assert event.reading_id == reading.reading_id
        assert event.alert_id == alert_uuid(
            reading.reading_id, FaultType.STUCK, "StuckValueDetector"
        )
        assert event.sensor_id == "TEP-XMV-04"
        assert event.plant_id == "TEP-PLANT-01"
        assert event.run_id == "fault_type=21/loop=0"
        assert event.fault_type is FaultType.STUCK
        assert event.severity is Severity.MEDIUM
        assert event.confidence == 0.5
        assert event.detected_at == NOW
        assert event.detector == "StuckValueDetector"
        assert event.evidence["run_length"] == 8
        assert event.source == SOURCE

    def test_run_id_is_optional(self) -> None:
        assert _event().run_id is None

    def test_evidence_is_a_copy(self) -> None:
        fault = _fault()
        event = AlertEvent.from_fault(fault, make_value_reading("TEP-XMV-04", NOW, 50.0), SOURCE)
        event.evidence["added"] = 1
        assert "added" not in fault.evidence


class TestSerialization:
    def test_round_trips_through_kafka_bytes(self) -> None:
        event = _event(run_id="fault_type=21/loop=0")
        assert AlertEvent.from_kafka_bytes(event.to_kafka_bytes()) == event

    def test_the_payload_is_json_with_string_enums_and_a_source(self) -> None:
        payload = json.loads(_event(run_id="r").to_kafka_bytes())
        assert payload["fault_type"] == "stuck"
        assert payload["severity"] == "MEDIUM"
        assert payload["source"] == {
            "topic": "sensor-readings-raw",
            "partition": 3,
            "offset": 41,
        }
        assert payload["detected_at"].startswith("2000-01-01T12:00:00")

    @pytest.mark.parametrize("payload", [b"not json", b"{}", b'{"alert_id": "x"}'])
    def test_a_malformed_payload_is_rejected(self, payload: bytes) -> None:
        with pytest.raises(ValidationError):
            AlertEvent.from_kafka_bytes(payload)


class TestValidation:
    def test_confidence_must_be_a_probability(self) -> None:
        payload = json.loads(_event().to_kafka_bytes())
        payload["confidence"] = 1.5
        with pytest.raises(ValidationError):
            AlertEvent.model_validate(payload)

    @pytest.mark.parametrize(("partition", "offset"), [(-1, 0), (0, -1)])
    def test_source_position_cannot_be_negative(self, partition: int, offset: int) -> None:
        with pytest.raises(ValidationError):
            AlertSource(topic="t", partition=partition, offset=offset)
