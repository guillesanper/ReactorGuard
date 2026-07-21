"""Calibrated measurement span per sensor tag.

Cada entrada declara DOS rangos, porque los dos consumidores del fichero
necesitan anchos distintos y es asi como funciona la instrumentacion real: un
transmisor tiene un span calibrado (el rango que su ADC puede representar) y,
por separado y dentro de el, unos limites de alarma (el sobre de operacion
normal).

    [min, max]              span calibrado. Escalado a cuentas ADC en
                            data/generators/tep_adapter.py. Un denominador
                            global para canales heterogeneos malgasta la
                            resolucion en el extremo bajo: una composicion de
                            ~30 mol% con denominador 3000 ocupa ~655 cuentas de
                            65535, el 1% del rango.

    [alarm_min, alarm_max]  sobre de operacion normal. Deteccion de
                            fuera-de-rango en data/validation/sensor_validator.py.

Medido sobre las 550.160 lecturas: usar el sobre estrecho tambien como span del
ADC recortaria el 12,76% de las lecturas, destruyendo informacion justo en los
ficheros de fallo; usar el span ancho tambien como umbral de alarma dejaria al
detector de rango practicamente ciego. Separarlos resuelve ambos extremos sin
partir la fuente de verdad: sigue habiendo un unico fichero, declarado como
`deps` de los stages que lo consumen, y este modulo es el unico que lo lee.

El fichero se genera con data/generators/derive_sensor_spans.py, que lo deriva
exclusivamente de d00.dat (operacion normal), se commitea y se revisa a mano.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SPANS_PATH = Path("configs/sensor_spans.yaml")

ADC_MAX_COUNTS = 65535
"""Fondo de escala de la tarjeta de entrada analogica de 16 bits."""


@dataclass(frozen=True)
class SensorSpan:
    """Calibrated range and alarm envelope of a single transmitter.

    Attributes:
        sensor_id: Canonical instrument tag, e.g. "TEP-XMEAS-07".
        min: Lower end of the calibrated span, in engineering units.
        max: Upper end of the calibrated span, in engineering units.
        alarm_min: Lower end of the normal-operation envelope.
        alarm_max: Upper end of the normal-operation envelope.
        unit: Engineering unit both ranges are expressed in.
    """

    sensor_id: str
    min: float
    max: float
    alarm_min: float
    alarm_max: float
    unit: str

    @property
    def width(self) -> float:
        """Return the calibrated span width in engineering units.

        Returns:
            max - min, always strictly positive by construction.
        """
        return self.max - self.min

    def to_raw_counts(self, value: float) -> int:
        """Scale an engineering value to a 16-bit ADC count.

        Values outside the calibrated span clamp to the ends of the scale, which
        is what a real analog input card does: it cannot represent what its span
        does not cover.

        Args:
            value: Process value in the same engineering units as the span.

        Returns:
            Integer ADC count in [0, ADC_MAX_COUNTS].
        """
        scaled = (value - self.min) / self.width * ADC_MAX_COUNTS
        return int(min(max(scaled, 0.0), float(ADC_MAX_COUNTS)))

    def contains(self, value: float) -> bool:
        """Return whether a value falls inside the calibrated span.

        Args:
            value: Process value in the same engineering units as the span.

        Returns:
            True if min <= value <= max.
        """
        return self.min <= value <= self.max

    def in_alarm_envelope(self, value: float) -> bool:
        """Return whether a value falls inside the normal-operation envelope.

        This is the predicate the range detector uses. It is strictly narrower
        than contains: a value can be representable by the transmitter and still
        be an excursion out of normal operation.

        Args:
            value: Process value in the same engineering units as the span.

        Returns:
            True if alarm_min <= value <= alarm_max.
        """
        return self.alarm_min <= value <= self.alarm_max


def _parse_entry(sensor_id: str, entry: Any) -> SensorSpan:
    """Build a SensorSpan from one mapping of the YAML document.

    Args:
        sensor_id: Instrument tag acting as the key of the entry.
        entry: Mapping expected to hold min, max and unit.

    Returns:
        The parsed SensorSpan.

    Raises:
        TypeError: If entry is not a mapping.
        KeyError: If a required key is missing.
        ValueError: If the resulting span is not strictly positive.
    """
    if not isinstance(entry, dict):
        raise TypeError(
            f"sensor_spans.yaml: entry for '{sensor_id}' must be a mapping, "
            f"got {type(entry).__name__}."
        )
    for key in ("min", "max", "alarm_min", "alarm_max", "unit"):
        if key not in entry:
            raise KeyError(f"sensor_spans.yaml: '{sensor_id}' is missing key '{key}'.")

    span = SensorSpan(
        sensor_id=sensor_id,
        min=float(entry["min"]),
        max=float(entry["max"]),
        alarm_min=float(entry["alarm_min"]),
        alarm_max=float(entry["alarm_max"]),
        unit=str(entry["unit"]),
    )
    if span.width <= 0.0:
        raise ValueError(
            f"sensor_spans.yaml: '{sensor_id}' has a non-positive span "
            f"(min={span.min}, max={span.max}). A transmitter with zero span "
            "cannot be scaled or range-checked."
        )
    if span.alarm_min >= span.alarm_max:
        raise ValueError(
            f"sensor_spans.yaml: '{sensor_id}' has a non-positive alarm envelope "
            f"(alarm_min={span.alarm_min}, alarm_max={span.alarm_max})."
        )
    if span.alarm_min < span.min or span.alarm_max > span.max:
        raise ValueError(
            f"sensor_spans.yaml: '{sensor_id}' has an alarm envelope "
            f"[{span.alarm_min}, {span.alarm_max}] outside its calibrated span "
            f"[{span.min}, {span.max}]. A transmitter cannot alarm on a value it "
            "cannot represent."
        )
    return span


def load_sensor_spans(
    spans_path: str | Path = DEFAULT_SPANS_PATH,
) -> dict[str, SensorSpan]:
    """Load the calibrated span table.

    Args:
        spans_path: Path to the YAML span file.

    Returns:
        Mapping of sensor_id to its SensorSpan.

    Raises:
        FileNotFoundError: If spans_path does not exist.
        KeyError: If the sensors: section or a required key is missing.
        TypeError: If an entry is not a mapping.
        ValueError: If a span is not strictly positive.
    """
    path = Path(spans_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sensor span file not found: {path}. Generate it with "
            "data/generators/derive_sensor_spans.py."
        )

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if "sensors" not in document:
        raise KeyError(f"{path}: missing required section 'sensors:'.")

    sensors = document["sensors"]
    if not isinstance(sensors, dict):
        raise TypeError(f"{path}: 'sensors:' must be a mapping of sensor_id to span.")

    return {
        sensor_id: _parse_entry(sensor_id, entry)
        for sensor_id, entry in sensors.items()
    }
