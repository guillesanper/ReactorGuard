"""Record of a single sensor fault as reported by a detector.

Un fallo detectado tiene que poder defenderse: quien lo recibe necesita saber no
solo QUE se detecto sino por que, con que confianza y a partir de que numeros.
De ahi que `evidence` no sea opcional ni decorativo. Un detector que marca sin
dejar evidencia es un detector imposible de calibrar en T3.6 y de auditar en
operacion.

`fault_type` describe el modo de fallo del INSTRUMENTO, y es deliberadamente
independiente del `fault_type` del dataset TEP, que describe la perturbacion del
PROCESO. Son dos ejes ortogonales: una valvula pegada (proceso) puede convivir
con un transmisor sano, y un transmisor congelado (instrumento) puede aparecer en
un proceso en operacion normal. Confundirlos convierte cualquier evaluacion del
validador en una tautologia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class FaultType(StrEnum):
    """Instrument failure mode identified by a detector."""

    STUCK = "stuck"
    """La lectura no cambia: transmisor congelado o linea de senal muerta."""

    NOISE_SPIKE = "noise_spike"
    """Salto mas rapido de lo que el proceso puede fisicamente moverse."""

    BIAS_OUT_OF_RANGE = "bias_out_of_range"
    """Valor fuera del sobre de operacion normal del transmisor."""

    DRIFT_CORRELATED = "drift_correlated"
    """Un sensor se ha desacoplado de otro con el que deberia correlacionar."""

    KALMAN_ANOMALY = "kalman_anomaly"
    """Desviacion respecto de lo que el modelo cinematico predecia."""

    SENSOR_DRIFT = "sensor_drift"
    """Deriva sostenida: la senal avanza con pendiente propia.

    Modo separado de KALMAN_ANOMALY porque lo delata otra senal del mismo filtro.
    Una deriva lineal vive en el espacio nulo del modelo de velocidad constante:
    el filtro aprende la pendiente y el residual decae a cero, asi que el residual
    NO la ve. Quien la ve es la velocidad estimada. Ver la cabecera de
    ml/features/kalman.py para los numeros medidos.
    """


class Severity(StrEnum):
    """Operational urgency of a detected fault."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class SensorFault:
    """A single fault attributed to one sensor by one detector.

    Attributes:
        sensor_id: Instrument tag the fault is attributed to.
        fault_type: Failure mode identified.
        severity: Operational urgency.
        confidence: Detector confidence in [0.0, 1.0]. These are heuristic
            ramps, not calibrated posteriors: they order faults by how far past
            its threshold a detector fired, and nothing more. T3.6 is what turns
            them into something measured.
        detected_at: Timestamp of the reading that triggered the detection, not
            wall-clock time. Replaying historical data must reproduce identical
            faults, which wall-clock stamping would break.
        detector: Class name of the detector that raised it, so a noisy detector
            can be identified without re-running the pipeline.
        evidence: Numbers that led to the detection, for explainability.

    Raises:
        ValueError: If confidence lies outside [0.0, 1.0].
    """

    sensor_id: str
    fault_type: FaultType
    severity: Severity
    confidence: float
    detected_at: datetime
    detector: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the confidence range.

        Raises:
            ValueError: If confidence is not within [0.0, 1.0].
        """
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must lie in [0.0, 1.0], got {self.confidence} for "
                f"{self.fault_type} on {self.sensor_id}."
            )

    @property
    def is_critical(self) -> bool:
        """Return whether this fault is of CRITICAL severity."""
        return self.severity is Severity.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the fault.

        Returns:
            Mapping with the enum members rendered as their string values and
            detected_at as an ISO 8601 string.
        """
        return {
            "sensor_id": self.sensor_id,
            "fault_type": self.fault_type.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "detected_at": self.detected_at.isoformat(),
            "detector": self.detector,
            "evidence": self.evidence,
        }
