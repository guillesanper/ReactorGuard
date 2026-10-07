"""Alert event published to anomaly-alerts, one per detected sensor fault.

Un `SensorFault` es un objeto en memoria del validador; un `AlertEvent` es su forma
de contrato en el bus: lleva lo necesario para que un consumidor aguas abajo (el
dashboard, un gestor de alertas) actue sin consultar nada mas, incluida la posicion
de la lectura de origen en el topic crudo (`source`), y es idempotente: el mismo
fallo sobre la misma lectura produce siempre el mismo `alert_id`, de modo que un
reproceso at-least-once (D4) se puede deduplicar aguas abajo.

El contrato de `SensorReading` no se toca (D13): este esquema es propio.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from data.schemas.sensor_reading import SensorReading
from data.validation.sensor_fault import FaultType, SensorFault, Severity

ALERT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "reactorguard.anomaly-alert")


class AlertSource(BaseModel):
    """Position of the raw reading that produced an alert.

    Attributes:
        topic: Topic the reading was consumed from.
        partition: Partition of that topic.
        offset: Offset of the reading in that partition.
    """

    topic: str
    partition: int = Field(..., ge=0)
    offset: int = Field(..., ge=0)


def alert_uuid(reading_id: UUID, fault_type: FaultType, detector: str, occurrence: int = 0) -> UUID:
    """Return the deterministic identifier of an alert.

    Args:
        reading_id: Identifier of the reading the fault was found on.
        fault_type: Failure mode identified.
        detector: Class name of the detector that raised the fault.
        occurrence: Position of the fault among those of the same fault type and
            detector on the same reading. Un detector puede emitir varios fallos
            iguales sobre una lectura (CrossCorrelationChecker, uno por pareja);
            sin esto compartirian identificador y la deduplicacion descartaria
            alertas distintas. Cero (el caso normal) no entra en la clave.

    Returns:
        A UUID5 of `reading_id|fault_type|detector` (plus the occurrence when it
        is not zero).
    """
    key = f"{reading_id}|{fault_type.value}|{detector}"
    if occurrence:
        key = f"{key}|{occurrence}"
    return uuid.uuid5(ALERT_ID_NAMESPACE, key)


class AlertEvent(BaseModel):
    """One detected sensor fault, as published to the alerts topic.

    Attributes:
        alert_id: Deterministic identifier, idempotent under reprocessing.
        reading_id: Identifier of the reading the fault was found on.
        sensor_id: Instrument tag the fault is attributed to.
        plant_id: Plant the reading belongs to.
        run_id: Value of the run-id header of the reading (fault_type and loop of
            the TEP pass), or None when the message carried none.
        fault_type: Failure mode of the instrument (not the TEP process fault).
        severity: Operational urgency.
        confidence: Detector heuristic confidence in [0, 1]; not a calibrated
            probability.
        detected_at: Timestamp of the triggering reading, not wall-clock time.
        detector: Class name of the detector that raised the fault.
        evidence: Numbers that led to the detection (`SensorFault.evidence`).
        source: Topic, partition and offset of the raw reading.
    """

    alert_id: UUID
    reading_id: UUID
    sensor_id: str
    plant_id: str
    run_id: str | None = None
    fault_type: FaultType
    severity: Severity
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_at: datetime
    detector: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    source: AlertSource

    @classmethod
    def from_fault(
        cls,
        fault: SensorFault,
        reading: SensorReading,
        source: AlertSource,
        *,
        run_id: str | None = None,
        occurrence: int = 0,
    ) -> AlertEvent:
        """Build the event of a fault found on a reading.

        Args:
            fault: Fault raised by a detector.
            reading: The reading it was found on (the original, as consumed).
            source: Where the reading came from.
            run_id: Value of the run-id header of the reading, if any.
            occurrence: Position among faults with the same fault type and
                detector on this reading; see `alert_uuid`.

        Returns:
            The alert event.
        """
        return cls(
            alert_id=alert_uuid(reading.reading_id, fault.fault_type, fault.detector, occurrence),
            reading_id=reading.reading_id,
            sensor_id=fault.sensor_id,
            plant_id=reading.plant_id,
            run_id=run_id,
            fault_type=fault.fault_type,
            severity=fault.severity,
            confidence=fault.confidence,
            detected_at=fault.detected_at,
            detector=fault.detector,
            evidence=dict(fault.evidence),
            source=source,
        )

    @classmethod
    def from_kafka_bytes(cls, data: bytes) -> AlertEvent:
        """Deserialise an alert from raw Kafka message bytes (UTF-8 JSON).

        Args:
            data: Raw bytes from a Kafka record value.

        Returns:
            A validated AlertEvent.

        Raises:
            pydantic.ValidationError: If the payload does not conform to the schema
                or is not valid JSON.
        """
        return cls.model_validate_json(data)

    def to_kafka_bytes(self) -> bytes:
        """Serialise the event to UTF-8 JSON bytes for a Kafka producer.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        return self.model_dump_json().encode("utf-8")
