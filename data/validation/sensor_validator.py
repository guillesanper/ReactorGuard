"""Five instrument-fault detectors and the orchestrator that runs them.

Cada detector es una Strategy independiente con estado propio por sensor. Se
mantienen separados porque detectan cosas distintas y fallan de formas distintas:
poder desactivar o recalibrar uno sin tocar los otros es lo que hace posible la
matriz de confusion por detector de T3.6.

Tres invariantes fijadas antes de escribir esto, cada una con su medicion detras:

1. NINGUN detector lee `quality` ni `is_usable`. `quality` describe la fiabilidad
   del transmisor y es justo lo que el validador esta decidiendo; leerlo como
   entrada convertiria su evaluacion en una tautologia. El validador lo ESCRIBE
   en el reading enriquecido, que es la direccion correcta del flujo.

2. El detector de rango usa `in_alarm_envelope`, no `contains`. El span calibrado
   es el rango que el ADC representa (margen 200%); el sobre de alarma es la
   operacion normal (margen 20%). Comprobar contra el span dejaria al detector
   practicamente ciego: por construccion casi nada cae fuera de el.

3. La ventana del detector de stuck es > 6. Medido sobre el dataset, las tiradas
   de valores identicos consecutivos llegan a 5-6 muestras en operacion normal
   (cuantizacion del analizador de composiciones), y a 480 en XMV-04 de d21, que
   es el unico sensor realmente congelado. Una ventana de 6 o menos convierte la
   cuantizacion en falsos positivos.

Sobre los umbrales: los valores por defecto son puntos de partida razonados, no
constantes calibradas. T3.6 es lo que los convierte en numeros medidos. Los que
dependen de la escala del canal se expresan como FRACCION del ancho del sobre de
alarma y no en unidades de ingenieria, porque un umbral absoluto compartido por
52 canales heterogeneos (caudales en kg/s, presiones en bar, composiciones en
mol%) no significa lo mismo en dos de ellos.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np

from data.schemas.sensor_reading import QualityFlag, SensorReading
from data.schemas.sensor_spans import SensorSpan
from data.validation.sensor_fault import FaultType, SensorFault, Severity
from ml.features.kalman import KalmanFilterBank

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_STUCK_WINDOW = 8
"""Lecturas identicas consecutivas que constituyen un stuck.

Por encima del suelo empirico de 6 y lo bastante cerca para detectar pronto: a
3 minutos de muestreo son 24 minutos de senal congelada.
"""

DEFAULT_STUCK_TOLERANCE_FRACTION = 0.0
"""Tolerancia para considerar dos lecturas iguales, en fracciones del sobre.

Cero significa repeticion exacta, que es lo que hace un transmisor congelado:
XMV-04 en d21 tiene nunique=1 y std=0 durante las 480 muestras. Subirlo capta
tambien transmisores casi-congelados a costa de acercarse al suelo de 5-6.
"""

DEFAULT_MAX_ENVELOPE_FRACTIONS_PER_SECOND = 0.005
"""Velocidad maxima plausible, en anchos de sobre por segundo.

A 180 s de muestreo permite recorrer 0,9 anchos de sobre entre dos muestras
consecutivas. Provisional: T3.6 lo calibra contra los saltos inyectados.
"""

DEFAULT_CORRELATION_WINDOW = 100
DEFAULT_CORRELATION_THRESHOLD = 0.3

DEFAULT_KALMAN_PROCESS_NOISE = 1e-6
"""Densidad del ruido de aceleracion Q para el detector, a cadencia TEP.

El 0.1 que trae OnlineKalmanFilter esta calibrado para dt de 1 s (sondeo SCADA).
El termino Q[0,0] escala con dt^3, asi que a los 180 s de muestreo del TEP se
dispara a 194.400, la sigma de innovacion sube a 602 y un salto de 25 unidades
queda en 0,042 sigmas: por completo invisible. Medido a 180 s: con q=1e-6 ese
mismo salto da 8,36 sigmas y las 200 muestras planas con ruido no producen ni un
falso positivo. Provisional; T3.6 lo calibra contra los saltos inyectados.
"""

DEFAULT_DRIFT_ENVELOPE_FRACTIONS_PER_HOUR = 11.0
"""Deriva maxima tolerada, en anchos de sobre por hora.

Se aplica sobre la VELOCIDAD estimada por el filtro de Kalman, no sobre su
residual: una deriva lineal sostenida vive en el espacio nulo del modelo de
velocidad constante y el residual decae a cero mientras la deriva continua.

CALIBRADO EN T3.6 sobre d00, sustituyendo al 1.0 provisional. Ese valor disparaba
en 11.969 de 25.012 muestras de operacion normal, el 48%: la velocidad
instantanea de un canal sano del TEP ya es del orden de 1 ancho de sobre por
hora. Distribucion medida sobre d00 con q=1e-6:

    p50 0,93   p90 3,44   p99 6,19   p99,9 8,33   maximo 10,13

El limite se fija justo por encima del maximo observado, lo que da cero falsos
positivos sobre d00 limpio. El margen es deliberadamente escaso porque el techo
util esta cerca: ver DRIFT_DETECTION_FLOOR_NOTE.
"""

DRIFT_DETECTION_FLOOR_NOTE = """
LIMITACION MEDIDA, no un ajuste pendiente.

A 180 s de muestreo, este detector solo ve derivas entre ~10 y ~18 anchos de
sobre por hora, y esa banda es estrecha por dos motivos independientes:

  Suelo. Por debajo de 10,13 anchos/hora la deriva es indistinguible del
  movimiento propio del proceso, que es lo que marca el maximo medido sobre d00.
  No se reduce bajando q: medido, el maximo satura en 9,13 para q <= 1e-8
  mientras los falsos positivos del residual suben del 2,1% al 6,4%. El suelo no
  es ruido del filtro, es el proceso moviendose.

  Techo. DEFAULT_MAX_ENVELOPE_FRACTIONS_PER_SECOND permite 0,9 anchos entre dos
  muestras, o sea 18 anchos/hora. Una deriva mas rapida es ademas una violacion
  de tasa en cada paso, y la reporta antes RateOfChangeDetector.

Consecuencia operativa: una deriva de instrumento lenta, que es el caso realista,
NO la detecta este mecanismo a esta cadencia. Detectarla exige un estimador de
linea base larga (comparar medias separadas por horas), que no es este filtro.
Queda declarado aqui en lugar de simularse con un umbral que no lo consigue.
"""

DEFAULT_DRIFT_MIN_STEPS = 20
"""Pasos antes de creerse la velocidad estimada.

Con la covarianza inicial deliberadamente vaga, las primeras estimaciones de
velocidad son ruido. Sin este minimo, todo sensor nuevo derivaria durante sus
primeras lecturas.
"""


def _ramped_confidence(magnitude: float, threshold: float, saturation: float) -> float:
    """Map how far past a threshold a detection lies onto [0.5, 1.0].

    Una deteccion justo en el umbral vale 0,5 y una que lo triplica vale 1,0.
    Es una rampa heuristica declarada como tal: ordena detecciones por flagrancia,
    no estima una probabilidad.

    Args:
        magnitude: Observed magnitude of the deviation.
        threshold: Value at which the detector fires.
        saturation: Multiple of the threshold at which confidence reaches 1.0.

    Returns:
        Confidence in [0.5, 1.0].
    """
    if threshold <= 0.0:
        return 1.0
    excess = (magnitude / threshold - 1.0) / max(saturation - 1.0, 1e-9)
    return float(min(1.0, 0.5 + 0.5 * max(excess, 0.0)))


class SensorFaultDetector(ABC):
    """Strategy interface shared by every detector.

    Returns a list rather than an Optional because a single reading can carry
    more than one fault of the same detector: the Kalman detector reports a
    transient and a sustained drift through different signals of the same filter.
    """

    @abstractmethod
    def check(self, reading: SensorReading, value: float) -> list[SensorFault]:
        """Evaluate one reading and return the faults it exhibits.

        Args:
            reading: Reading under evaluation.
            value: The reading's measured value, already confirmed non-None by
                the orchestrator so every detector need not re-check it.

        Returns:
            Faults detected, empty when the reading looks healthy.
        """

    @abstractmethod
    def reset(self, sensor_id: str) -> None:
        """Discard any state held for one sensor.

        Args:
            sensor_id: Instrument tag. Unknown tags must be a no-op.
        """

    @property
    def name(self) -> str:
        """Return the detector's class name, stamped onto every fault it raises."""
        return type(self).__name__


class StuckValueDetector(SensorFaultDetector):
    """Flags a transmitter whose reading has stopped changing.

    Detecta por LONGITUD DE TIRADA de valores repetidos y no por varianza de una
    ventana. La varianza se mide en unidades de ingenieria al cuadrado, asi que un
    unico umbral no significa lo mismo en un caudal y en una composicion; la
    longitud de tirada es adimensional y es ademas exactamente la magnitud sobre
    la que se midio el suelo empirico de 6.
    """

    def __init__(
        self,
        spans: dict[str, SensorSpan],
        window_size: int = DEFAULT_STUCK_WINDOW,
        tolerance_fraction: float = DEFAULT_STUCK_TOLERANCE_FRACTION,
    ) -> None:
        """Build the detector.

        Args:
            spans: Span table, used to turn the tolerance into engineering units.
            window_size: Consecutive identical readings that constitute a stuck
                sensor. Must be greater than 6, the measured run length reached
                by healthy quantised channels.
            tolerance_fraction: Fraction of the alarm envelope width within which
                two readings count as identical. Must be non-negative.

        Raises:
            ValueError: If window_size is not above the empirical floor of 6, or
                if tolerance_fraction is negative.
        """
        if window_size <= 6:
            raise ValueError(
                f"window_size must exceed 6, got {window_size}. Healthy TEP "
                "channels reach runs of 5-6 identical values through analyser "
                "quantisation; a shorter window turns that into false positives."
            )
        if tolerance_fraction < 0.0:
            raise ValueError(
                f"tolerance_fraction must be non-negative, got {tolerance_fraction}."
            )

        self.spans = spans
        self.window_size = window_size
        self.tolerance_fraction = tolerance_fraction
        self._last_value: dict[str, float] = {}
        self._run_length: dict[str, int] = {}

    def run_length(self, sensor_id: str) -> int:
        """Return the current run of repeated readings for a sensor.

        Args:
            sensor_id: Instrument tag.

        Returns:
            Length of the current run, zero if the sensor has not been seen.
        """
        return self._run_length.get(sensor_id, 0)

    def check(self, reading: SensorReading, value: float) -> list[SensorFault]:
        """Extend or break the sensor's run and flag it once it is long enough.

        Args:
            reading: Reading under evaluation.
            value: Measured value.

        Returns:
            A single stuck fault once the run reaches window_size, and on every
            reading after that while it lasts. Reporting per reading rather than
            once per episode is what lets T3.6 score precision by
            (sensor_id, timestep) instead of by episode.
        """
        sensor_id = reading.sensor.id
        tolerance = self._tolerance_for(sensor_id)

        previous = self._last_value.get(sensor_id)
        if previous is not None and abs(value - previous) <= tolerance:
            self._run_length[sensor_id] = self._run_length.get(sensor_id, 1) + 1
        else:
            self._run_length[sensor_id] = 1
        self._last_value[sensor_id] = value

        run = self._run_length[sensor_id]
        if run < self.window_size:
            return []

        ratio = run / self.window_size
        return [
            SensorFault(
                sensor_id=sensor_id,
                fault_type=FaultType.STUCK,
                severity=self._severity(ratio),
                confidence=_ramped_confidence(float(run), float(self.window_size), 3.0),
                detected_at=reading.timestamp,
                detector=self.name,
                evidence={
                    "run_length": run,
                    "window_size": self.window_size,
                    "value": value,
                    "tolerance": tolerance,
                },
            )
        ]

    def reset(self, sensor_id: str) -> None:
        """Forget the run and last value of one sensor.

        Args:
            sensor_id: Instrument tag.
        """
        self._last_value.pop(sensor_id, None)
        self._run_length.pop(sensor_id, None)

    def _tolerance_for(self, sensor_id: str) -> float:
        """Return the equality tolerance in engineering units.

        Args:
            sensor_id: Instrument tag.

        Returns:
            Absolute tolerance; zero when the sensor has no span, which reduces
            the test to exact repetition.
        """
        span = self.spans.get(sensor_id)
        if span is None or self.tolerance_fraction == 0.0:
            return 0.0
        return (span.alarm_max - span.alarm_min) * self.tolerance_fraction

    @staticmethod
    def _severity(ratio: float) -> Severity:
        """Map a run length, in multiples of the window, onto a severity."""
        if ratio >= 4.0:
            return Severity.CRITICAL
        if ratio >= 2.0:
            return Severity.HIGH
        return Severity.MEDIUM


class RateOfChangeDetector(SensorFaultDetector):
    """Flags a jump faster than the process can physically move.

    El limite se expresa en anchos de sobre por segundo, no en unidades por
    segundo. Un limite absoluto compartido por 52 canales heterogeneos seria a la
    vez inalcanzable para unos y trivial para otros; normalizado por el ancho del
    sobre, el mismo numero significa lo mismo en todos.
    """

    def __init__(
        self,
        spans: dict[str, SensorSpan],
        max_envelope_fractions_per_second: float = (
            DEFAULT_MAX_ENVELOPE_FRACTIONS_PER_SECOND
        ),
    ) -> None:
        """Build the detector.

        Args:
            spans: Span table, used to normalise the rate by envelope width.
            max_envelope_fractions_per_second: Rate above which a change is
                flagged. Must be strictly positive.

        Raises:
            ValueError: If the limit is not strictly positive.
        """
        if max_envelope_fractions_per_second <= 0.0:
            raise ValueError(
                "max_envelope_fractions_per_second must be strictly positive, "
                f"got {max_envelope_fractions_per_second}."
            )
        self.spans = spans
        self.max_rate = max_envelope_fractions_per_second
        self._last: dict[str, tuple[float, datetime]] = {}

    def check(self, reading: SensorReading, value: float) -> list[SensorFault]:
        """Compare the normalised rate of change against the limit.

        Args:
            reading: Reading under evaluation.
            value: Measured value.

        Returns:
            A noise_spike fault when the rate exceeds the limit. The first
            reading of a sensor, a sensor without a span and two readings sharing
            a timestamp all yield nothing: none of them defines a rate.
        """
        sensor_id = reading.sensor.id
        previous = self._last.get(sensor_id)
        self._last[sensor_id] = (value, reading.timestamp)

        span = self.spans.get(sensor_id)
        if previous is None or span is None:
            return []

        last_value, last_timestamp = previous
        elapsed = (reading.timestamp - last_timestamp).total_seconds()
        if elapsed <= 0.0:
            return []

        width = span.alarm_max - span.alarm_min
        rate = abs(value - last_value) / width / elapsed
        if rate <= self.max_rate:
            return []

        return [
            SensorFault(
                sensor_id=sensor_id,
                fault_type=FaultType.NOISE_SPIKE,
                severity=Severity.HIGH if rate > 5.0 * self.max_rate else Severity.MEDIUM,
                confidence=_ramped_confidence(rate, self.max_rate, 5.0),
                detected_at=reading.timestamp,
                detector=self.name,
                evidence={
                    "rate_envelope_fractions_per_second": rate,
                    "limit": self.max_rate,
                    "delta": value - last_value,
                    "elapsed_seconds": elapsed,
                },
            )
        ]

    def reset(self, sensor_id: str) -> None:
        """Forget the last reading of one sensor.

        Args:
            sensor_id: Instrument tag.
        """
        self._last.pop(sensor_id, None)


class RangeValidator(SensorFaultDetector):
    """Flags a value outside the sensor's normal-operation envelope.

    Stateless: a range check needs no history. Uses `in_alarm_envelope` and never
    `contains`, per invariant 2 in the module docstring.
    """

    def __init__(self, spans: dict[str, SensorSpan]) -> None:
        """Build the detector.

        Args:
            spans: Span table keyed by instrument tag.
        """
        self.spans = spans

    def check(self, reading: SensorReading, value: float) -> list[SensorFault]:
        """Test the value against its sensor's alarm envelope.

        Args:
            reading: Reading under evaluation.
            value: Measured value.

        Returns:
            A bias_out_of_range fault when the value sits outside the envelope.
            A value also outside the calibrated span is CRITICAL: the transmitter
            cannot even represent it, so the reading is not merely abnormal but
            untrustworthy.
        """
        sensor_id = reading.sensor.id
        span = self.spans.get(sensor_id)
        if span is None or span.in_alarm_envelope(value):
            return []

        width = span.alarm_max - span.alarm_min
        excess = max(span.alarm_min - value, value - span.alarm_max)

        if not span.contains(value):
            severity = Severity.CRITICAL
        elif excess > 0.25 * width:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

        return [
            SensorFault(
                sensor_id=sensor_id,
                fault_type=FaultType.BIAS_OUT_OF_RANGE,
                severity=severity,
                confidence=_ramped_confidence(excess + width, width, 2.0),
                detected_at=reading.timestamp,
                detector=self.name,
                evidence={
                    "value": value,
                    "alarm_min": span.alarm_min,
                    "alarm_max": span.alarm_max,
                    "excess": excess,
                    "outside_calibrated_span": not span.contains(value),
                },
            )
        ]

    def reset(self, sensor_id: str) -> None:
        """No-op: the detector holds no state.

        Args:
            sensor_id: Instrument tag, ignored.
        """


class CrossCorrelationChecker(SensorFaultDetector):
    """Flags a sensor that has decoupled from one it should track.

    Con un diccionario de correlaciones base vacio no comprueba nada, que es el
    estado por defecto: las parejas correlacionadas del TEP son una tabla de
    calibracion que aun no se ha derivado, y inventarla aqui seria cablear una
    suposicion sobre la planta.
    """

    def __init__(
        self,
        baseline_correlations: dict[tuple[str, str], float] | None = None,
        window_size: int = DEFAULT_CORRELATION_WINDOW,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    ) -> None:
        """Build the detector.

        Args:
            baseline_correlations: Expected Pearson coefficient per sensor pair.
                Empty or None disables the detector.
            window_size: Number of aligned samples the coefficient is computed
                over. Must be at least 3.
            correlation_threshold: Absolute deviation from the baseline above
                which the pair is flagged.

        Raises:
            ValueError: If window_size is below 3, or if a baseline coefficient
                lies outside [-1, 1].
        """
        if window_size < 3:
            raise ValueError(
                f"window_size must be at least 3, got {window_size}: a Pearson "
                "coefficient over two points is always +/-1."
            )
        for pair, expected in (baseline_correlations or {}).items():
            if not -1.0 <= expected <= 1.0:
                raise ValueError(
                    f"Baseline correlation for {pair} is {expected}, outside [-1, 1]."
                )

        self.baselines = dict(baseline_correlations or {})
        self.window_size = window_size
        self.correlation_threshold = correlation_threshold
        self._buffers: dict[str, deque[tuple[datetime, float]]] = {}

    def update(self, reading: SensorReading, value: float) -> None:
        """Append a reading to its sensor's rolling buffer.

        Args:
            reading: Reading to record.
            value: Measured value.
        """
        sensor_id = reading.sensor.id
        if sensor_id not in self._buffers:
            self._buffers[sensor_id] = deque(maxlen=self.window_size)
        self._buffers[sensor_id].append((reading.timestamp, value))

    def check_pair(self, id_a: str, id_b: str) -> float | None:
        """Return the Pearson coefficient of a pair over their aligned window.

        Los buffers se alinean por TIMESTAMP y no por posicion. Dos sensores
        pueden no haber llegado al mismo ritmo, y correlacionar la muestra n de
        uno con la n del otro cuando corresponden a instantes distintos produce un
        coeficiente que no significa nada.

        Args:
            id_a: First instrument tag.
            id_b: Second instrument tag.

        Returns:
            The coefficient, or None when it is undefined: fewer than three
            shared timestamps, or either series constant over the window. Un
            sensor congelado tiene varianza cero y su correlacion es NaN, no cero;
            devolver cero lo haria parecer un desacoplamiento perfecto y todo par
            que incluyera a XMV-04 en d21 se marcaria por partida doble.
        """
        buffer_a = self._buffers.get(id_a)
        buffer_b = self._buffers.get(id_b)
        if not buffer_a or not buffer_b:
            return None

        by_timestamp_b = dict(buffer_b)
        shared = [
            (value_a, by_timestamp_b[timestamp])
            for timestamp, value_a in buffer_a
            if timestamp in by_timestamp_b
        ]
        if len(shared) < 3:
            return None

        series_a = np.array([pair[0] for pair in shared], dtype=np.float64)
        series_b = np.array([pair[1] for pair in shared], dtype=np.float64)
        if series_a.std() == 0.0 or series_b.std() == 0.0:
            return None

        return float(np.corrcoef(series_a, series_b)[0, 1])

    def check(self, reading: SensorReading, value: float) -> list[SensorFault]:
        """Record the reading and re-check every baseline pair it belongs to.

        Args:
            reading: Reading under evaluation.
            value: Measured value.

        Returns:
            One drift_correlated fault per baseline pair whose observed
            coefficient has moved further than the threshold from its baseline.
        """
        self.update(reading, value)

        sensor_id = reading.sensor.id
        faults: list[SensorFault] = []
        for (id_a, id_b), expected in self.baselines.items():
            if sensor_id not in (id_a, id_b):
                continue
            observed = self.check_pair(id_a, id_b)
            if observed is None:
                continue

            deviation = abs(observed - expected)
            if deviation <= self.correlation_threshold:
                continue

            faults.append(
                SensorFault(
                    sensor_id=sensor_id,
                    fault_type=FaultType.DRIFT_CORRELATED,
                    severity=Severity.MEDIUM,
                    confidence=_ramped_confidence(
                        deviation, self.correlation_threshold, 3.0
                    ),
                    detected_at=reading.timestamp,
                    detector=self.name,
                    evidence={
                        "pair": [id_a, id_b],
                        "observed_pearson": observed,
                        "expected_pearson": expected,
                        "deviation": deviation,
                    },
                )
            )
        return faults

    def reset(self, sensor_id: str) -> None:
        """Clear the rolling buffer of one sensor.

        Args:
            sensor_id: Instrument tag.
        """
        self._buffers.pop(sensor_id, None)


class KalmanResidualDetector(SensorFaultDetector):
    """Flags deviations from a constant-velocity model, on two distinct signals.

    El residual normalizado detecta TRANSITORIOS: escalones, picos, arranques de
    deriva y cambios de pendiente. La velocidad estimada detecta la DERIVA YA
    ESTABLECIDA, que el residual no ve porque el filtro la aprende. Medido: una
    rampa de 0,5 u/paso durante 100 pasos alcanza 1,21 sigmas de residual maximo y
    no marca ni un paso, mientras la velocidad converge exactamente a la pendiente
    real. Usar solo el residual dejaria la deriva de instrumento sin detector.
    """

    def __init__(
        self,
        spans: dict[str, SensorSpan],
        k_sigma: float = 3.0,
        process_noise: float = DEFAULT_KALMAN_PROCESS_NOISE,
        observation_noise: float = 1.0,
        max_drift_envelope_fractions_per_hour: float = (
            DEFAULT_DRIFT_ENVELOPE_FRACTIONS_PER_HOUR
        ),
        min_steps_before_drift: int = DEFAULT_DRIFT_MIN_STEPS,
    ) -> None:
        """Build the detector.

        Args:
            spans: Span table, used to normalise the drift rate.
            k_sigma: Normalized residual above which a transient is flagged.
            process_noise: Q spectral density passed to every filter.
            observation_noise: R measurement variance passed to every filter.
            max_drift_envelope_fractions_per_hour: Estimated velocity above which
                a sustained drift is flagged. Must be strictly positive.
            min_steps_before_drift: Readings a sensor must accumulate before its
                velocity estimate is trusted. Must be positive.

        Raises:
            ValueError: If any argument is out of range, including those rejected
                by the underlying filter.
        """
        if max_drift_envelope_fractions_per_hour <= 0.0:
            raise ValueError(
                "max_drift_envelope_fractions_per_hour must be strictly positive, "
                f"got {max_drift_envelope_fractions_per_hour}."
            )
        if min_steps_before_drift <= 0:
            raise ValueError(
                f"min_steps_before_drift must be positive, got {min_steps_before_drift}."
            )

        self.spans = spans
        self.max_drift = max_drift_envelope_fractions_per_hour
        self.min_steps_before_drift = min_steps_before_drift
        self._bank = KalmanFilterBank(
            process_noise=process_noise,
            observation_noise=observation_noise,
            k_sigma=k_sigma,
        )
        self._steps: dict[str, int] = {}

    @property
    def k_sigma(self) -> float:
        """Return the normalized-residual threshold used by every filter."""
        return self._bank.k_sigma

    def check(self, reading: SensorReading, value: float) -> list[SensorFault]:
        """Filter the reading and report both transient and drift signals.

        Args:
            reading: Reading under evaluation.
            value: Measured value.

        Returns:
            Up to two faults: a kalman_anomaly for a transient and a sensor_drift
            for a sustained slope. They are independent and can co-occur.
        """
        sensor_id = reading.sensor.id
        result = self._bank.process(reading)
        self._steps[sensor_id] = self._steps.get(sensor_id, 0) + 1

        faults: list[SensorFault] = []

        if result.is_anomaly:
            magnitude = abs(result.normalized_residual)
            faults.append(
                SensorFault(
                    sensor_id=sensor_id,
                    fault_type=FaultType.KALMAN_ANOMALY,
                    severity=(
                        Severity.HIGH if not result.is_initialized else Severity.MEDIUM
                    ),
                    confidence=_ramped_confidence(magnitude, self.k_sigma, 3.0),
                    detected_at=reading.timestamp,
                    detector=self.name,
                    evidence={
                        "normalized_residual": result.normalized_residual,
                        "residual": result.residual,
                        "predicted_value": result.predicted_value,
                        "innovation_sigma": result.innovation_sigma,
                        "k_sigma": self.k_sigma,
                        # False aqui significa que el filtro descarto su estado:
                        # el salto fue tan grande que la historia dejo de aplicar.
                        "filter_was_initialized": result.is_initialized,
                    },
                )
            )

        drift = self._check_drift(reading, sensor_id)
        if drift is not None:
            faults.append(drift)

        return faults

    def reset(self, sensor_id: str) -> None:
        """Discard the filter state and step count of one sensor.

        Args:
            sensor_id: Instrument tag.
        """
        self._bank.reset_sensor(sensor_id)
        self._steps.pop(sensor_id, None)

    def _check_drift(self, reading: SensorReading, sensor_id: str) -> SensorFault | None:
        """Test the filter's velocity estimate against the drift limit.

        Args:
            reading: Reading under evaluation.
            sensor_id: Instrument tag.

        Returns:
            A sensor_drift fault, or None when the sensor has no span, has not
            accumulated enough history for its velocity to mean anything, or is
            drifting within the limit.
        """
        span = self.spans.get(sensor_id)
        if span is None or self._steps[sensor_id] < self.min_steps_before_drift:
            return None

        width = span.alarm_max - span.alarm_min
        velocity = self._bank.get_filter(sensor_id).velocity
        drift_rate = abs(velocity) * 3600.0 / width
        if drift_rate <= self.max_drift:
            return None

        return SensorFault(
            sensor_id=sensor_id,
            fault_type=FaultType.SENSOR_DRIFT,
            severity=Severity.HIGH if drift_rate > 5.0 * self.max_drift else Severity.MEDIUM,
            confidence=_ramped_confidence(drift_rate, self.max_drift, 5.0),
            detected_at=reading.timestamp,
            detector=self.name,
            evidence={
                "drift_envelope_fractions_per_hour": drift_rate,
                "limit": self.max_drift,
                "velocity_units_per_second": velocity,
                "steps_observed": self._steps[sensor_id],
            },
        )


class ValidationResult:
    """Verdict on a single reading.

    Attributes:
        is_valid: Whether the reading is free of detected faults.
        faults: Every fault raised, in detector order.
        enriched_reading: A copy of the reading with its quality flag set from
            the verdict. The original is never mutated.
    """

    __slots__ = ("is_valid", "faults", "enriched_reading")

    def __init__(
        self,
        is_valid: bool,
        faults: list[SensorFault],
        enriched_reading: SensorReading,
    ) -> None:
        """Build a result.

        Args:
            is_valid: Whether the reading is free of detected faults.
            faults: Faults raised by the detectors.
            enriched_reading: Reading copy carrying the resulting quality flag.
        """
        self.is_valid = is_valid
        self.faults = faults
        self.enriched_reading = enriched_reading

    @property
    def worst_severity(self) -> Severity | None:
        """Return the highest severity among the faults, or None if there are none."""
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        if not self.faults:
            return None
        return max((f.severity for f in self.faults), key=order.index)


class SensorValidator:
    """Runs every detector over a reading and consolidates their verdicts.

    El validador ESCRIBE `quality` en el reading enriquecido y no lo LEE nunca,
    ni el suyo ni el de entrada. Esa asimetria es deliberada: `quality` describe
    la fiabilidad del transmisor, que es exactamente lo que el validador esta
    decidiendo, asi que leerlo como entrada seria dar por supuesta la conclusion.
    El adaptador del TEP emite GOOD para los 22 ficheros por esa misma razon.
    """

    def __init__(
        self,
        spans: dict[str, SensorSpan],
        detectors: list[SensorFaultDetector] | None = None,
        baseline_correlations: dict[tuple[str, str], float] | None = None,
        latency_window: int = 10_000,
    ) -> None:
        """Build a validator with the five default detectors, or a custom set.

        Args:
            spans: Span table shared by the detectors that need one.
            detectors: Explicit detector list, for tests and for running a
                subset. Defaults to the five standard detectors in order.
            baseline_correlations: Passed to the default CrossCorrelationChecker.
                Ignored when `detectors` is given.
            latency_window: Number of most recent per-reading latencies kept
                for the percentiles of `get_metrics`. Acotada a proposito: en un
                servicio continuo una lista sin limite crece un float por
                lectura y nunca se libera.

        Raises:
            ValueError: If `latency_window` is not positive.
        """
        if latency_window <= 0:
            raise ValueError(f"latency_window must be positive, got {latency_window}")
        self.spans = spans
        self.detectors: list[SensorFaultDetector] = (
            detectors
            if detectors is not None
            else [
                StuckValueDetector(spans),
                RateOfChangeDetector(spans),
                RangeValidator(spans),
                CrossCorrelationChecker(baseline_correlations),
                KalmanResidualDetector(spans),
            ]
        )
        self._fault_counts: dict[str, int] = {}
        self._readings_total = 0
        self._readings_without_value = 0
        self._readings_unknown_sensor = 0
        self._latencies_seconds: deque[float] = deque(maxlen=latency_window)

    def validate(self, reading: SensorReading) -> ValidationResult:
        """Run every detector and consolidate the verdict.

        A reading carrying no value is passed through unjudged: a Kalman filter
        needs a number, a rate needs two, and imputing one would fabricate the
        evidence the verdict rests on. Absence of a value is the SCADA's own
        quality signal, which this class deliberately does not read. Such
        readings are counted so the omission stays visible in the metrics.

        Args:
            reading: Reading to evaluate.

        Returns:
            The ValidationResult, whose enriched_reading carries the quality flag
            implied by the faults found.
        """
        started = time.perf_counter()
        self._readings_total += 1

        if reading.sensor.id not in self.spans:
            self._readings_unknown_sensor += 1

        value = reading.measurement.value
        if value is None:
            self._readings_without_value += 1
            self._latencies_seconds.append(time.perf_counter() - started)
            return ValidationResult(True, [], reading)

        faults: list[SensorFault] = []
        for detector in self.detectors:
            faults.extend(detector.check(reading, value))

        for fault in faults:
            key = fault.fault_type.value
            self._fault_counts[key] = self._fault_counts.get(key, 0) + 1

        self._latencies_seconds.append(time.perf_counter() - started)
        return ValidationResult(
            is_valid=not faults,
            faults=faults,
            enriched_reading=self._enrich(reading, faults),
        )

    def reset_sensor(self, sensor_id: str) -> None:
        """Discard every detector's state for one sensor.

        Args:
            sensor_id: Instrument tag.
        """
        for detector in self.detectors:
            detector.reset(sensor_id)

    def get_metrics(self) -> dict[str, Any]:
        """Return the counters and latency summary collected so far.

        Se devuelven como datos planos en lugar de registrarse en el registro
        global de Prometheus. Una libreria que registra metricas globales al
        importarse colisiona consigo misma entre tests y ata este modulo a un
        backend concreto; el consumidor de T3.5, que es quien tendra endpoint de
        scrape, es el sitio donde estas cifras se convierten en metricas.

        Returns:
            Mapping with per-fault-type counts, reading totals and latency
            percentiles in milliseconds. Los contadores son acumulados desde
            la construccion; la latencia se calcula solo sobre las ultimas
            `latency_window` lecturas.
        """
        latencies = np.array(self._latencies_seconds, dtype=np.float64) * 1000.0
        return {
            "readings_total": self._readings_total,
            "readings_without_value": self._readings_without_value,
            "readings_unknown_sensor": self._readings_unknown_sensor,
            "faults_by_type": dict(self._fault_counts),
            "faults_total": sum(self._fault_counts.values()),
            "latency_ms": {
                "mean": float(latencies.mean()) if latencies.size else 0.0,
                "p50": float(np.percentile(latencies, 50)) if latencies.size else 0.0,
                "p99": float(np.percentile(latencies, 99)) if latencies.size else 0.0,
            },
        }

    @staticmethod
    def _enrich(reading: SensorReading, faults: list[SensorFault]) -> SensorReading:
        """Return a copy of the reading carrying the quality implied by the faults.

        Args:
            reading: Original reading, left untouched.
            faults: Faults found on it.

        Returns:
            The original object when no fault was found, so the common path
            allocates nothing; otherwise a copy flagged BAD if any fault is
            CRITICAL and SUSPECT otherwise. BAD is reserved for CRITICAL because
            it makes is_usable False and drops the reading from ML inference: a
            merely abnormal value is still evidence, an unrepresentable one is not.
        """
        if not faults:
            return reading

        quality = (
            QualityFlag.BAD
            if any(fault.is_critical for fault in faults)
            else QualityFlag.SUSPECT
        )
        return reading.model_copy(
            update={"measurement": reading.measurement.model_copy(update={"quality": quality})}
        )
