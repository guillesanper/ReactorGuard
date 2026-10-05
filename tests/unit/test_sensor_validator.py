"""Unit tests for data.validation.

La suite esta organizada por detector porque es asi como se calibran y como se
desactivan: T3.6 puntua una matriz de confusion por detector, y un test que
mezclara dos no diria cual de los dos fallo.

Tres invariantes se comprueban explicitamente aqui porque son decisiones
tomadas contra evidencia medida y su regresion seria silenciosa: la ventana de
stuck por encima del suelo empirico de 6, el uso del sobre de alarma y no del
span calibrado en el detector de rango, y que ningun detector lea `quality`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from itertools import chain
from types import SimpleNamespace

import pytest

from data.schemas.sensor_reading import QualityFlag, SensorReading
from data.schemas.sensor_spans import SensorSpan
from data.validation.sensor_fault import FaultType, SensorFault, Severity
from data.validation.sensor_validator import (
    CrossCorrelationChecker,
    KalmanResidualDetector,
    RangeValidator,
    RateOfChangeDetector,
    SensorValidator,
    StuckValueDetector,
)

_T0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
_SAMPLE = timedelta(minutes=3)
"""Periodo de muestreo del TEP: 3 minutos, 180 segundos."""

_TAG = "TEP-XMEAS-07"
_OTHER_TAG = "TEP-XMEAS-08"

# Sobre de alarma [20, 80], ancho 60, dentro de un span calibrado [0, 100].
# La franja [80, 100] es la que distingue in_alarm_envelope de contains.
_ENVELOPE_WIDTH = 60.0

_SYNTHETIC_DRIFT_LIMIT = 1.0
"""Umbral de deriva que usan los tests de rampa de este fichero.

Se declara aqui en lugar de heredar el default del detector a proposito. Estos
tests miden el MECANISMO (que la velocidad estimada delata una deriva que el
residual no ve) sobre una rampa sintetica de 5 unidades por muestra, que son
1,67 anchos de sobre por hora. El default del detector es en cambio una
constante calibrada sobre d00 en T3.6 contra el movimiento real del proceso;
atarlos significaria que recalibrar el detector rompe tests que no hablan de la
calibracion. Ver DRIFT_DETECTION_FLOOR_NOTE en sensor_validator.py.
"""


def _span(sensor_id: str = _TAG) -> SensorSpan:
    """Return a span with a 60-wide envelope inside a 100-wide calibrated span."""
    return SensorSpan(
        sensor_id=sensor_id,
        min=0.0,
        max=100.0,
        alarm_min=20.0,
        alarm_max=80.0,
        unit="bar",
    )


@pytest.fixture()
def spans() -> dict[str, SensorSpan]:
    """Return a two-sensor span table."""
    return {_TAG: _span(_TAG), _OTHER_TAG: _span(_OTHER_TAG)}


def _reading(
    value: float | None = 50.0,
    timestamp: datetime = _T0,
    sensor_id: str = _TAG,
    quality: str = "good",
) -> SensorReading:
    """Build a valid SensorReading.

    Args:
        value: Engineering value, or None for a valueless reading.
        timestamp: Measurement time.
        sensor_id: Instrument tag.
        quality: Quality flag to stamp on the input. Tests use it to prove the
            detectors ignore it.

    Returns:
        A validated SensorReading.
    """
    return SensorReading.model_validate(
        {
            "reading_id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "plant_id": "TEP-PLANT-01",
            "sensor": {
                "id": sensor_id,
                "type": "pressure",
                "location": "primary_loop",
                "elevation_m": 3.45,
            },
            "measurement": {
                "value": value,
                "unit": "bar",
                "quality": quality,
                "raw_counts": 4092,
            },
            "metadata": {
                "calibration_date": "2024-01-01",
                "last_maintenance": "2023-12-15",
                "drift_coefficient": 0.0012,
            },
        }
    )


def _run(
    detector: object,
    values: list[float],
    sensor_id: str = _TAG,
    start: datetime = _T0,
    step: timedelta = _SAMPLE,
) -> list[list[SensorFault]]:
    """Feed a series of values through a detector and collect its verdicts.

    Args:
        detector: Detector exposing check(reading, value).
        values: Measurements in chronological order.
        sensor_id: Instrument tag to stamp on every reading.
        start: Timestamp of the first reading.
        step: Interval between readings.

    Returns:
        One fault list per value, in order.
    """
    out = []
    for index, value in enumerate(values):
        reading = _reading(value, start + step * index, sensor_id)
        out.append(detector.check(reading, value))  # type: ignore[attr-defined]
    return out


# ---------------------------------------------------------------------------
# SensorFault
# ---------------------------------------------------------------------------


class TestSensorFault:
    """The fault record must be immutable, bounded and serialisable."""

    def _fault(self, **overrides: object) -> SensorFault:
        """Build a fault with sensible defaults."""
        kwargs: dict[str, object] = {
            "sensor_id": _TAG,
            "fault_type": FaultType.STUCK,
            "severity": Severity.HIGH,
            "confidence": 0.8,
            "detected_at": _T0,
            "detector": "StuckValueDetector",
            "evidence": {"run_length": 12},
        }
        kwargs.update(overrides)
        return SensorFault(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("confidence", [-0.01, 1.01])
    def test_confidence_must_be_a_probability(self, confidence: float) -> None:
        """A confidence outside [0, 1] is not orderable against the others."""
        with pytest.raises(ValueError, match="confidence"):
            self._fault(confidence=confidence)

    @pytest.mark.parametrize("confidence", [0.0, 1.0])
    def test_confidence_bounds_are_inclusive(self, confidence: float) -> None:
        """Both ends of the range must be accepted."""
        assert self._fault(confidence=confidence).confidence == confidence

    def test_is_frozen(self) -> None:
        """A fault must not be editable after the fact."""
        with pytest.raises(AttributeError):
            self._fault().severity = Severity.LOW  # type: ignore[misc]

    def test_is_critical(self) -> None:
        """is_critical must track the severity."""
        assert self._fault(severity=Severity.CRITICAL).is_critical
        assert not self._fault(severity=Severity.HIGH).is_critical

    def test_to_dict_is_json_ready(self) -> None:
        """Enums must render as strings and the timestamp as ISO 8601."""
        import json

        payload = self._fault().to_dict()
        assert payload["fault_type"] == "stuck"
        assert payload["severity"] == "HIGH"
        assert payload["detected_at"] == _T0.isoformat()
        assert json.loads(json.dumps(payload))["evidence"]["run_length"] == 12


# ---------------------------------------------------------------------------
# StuckValueDetector
# ---------------------------------------------------------------------------


class TestStuckValueDetector:
    """A frozen transmitter must be caught; a quantised healthy one must not."""

    def test_window_must_exceed_the_empirical_floor(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A window of 6 or less turns analyser quantisation into false positives.

        Medido: los canales de composicion del TEP llegan a tiradas de 5-6
        valores identicos en operacion NORMAL, incluido d00.
        """
        with pytest.raises(ValueError, match="must exceed 6"):
            StuckValueDetector(spans, window_size=6)

    def test_window_of_seven_is_accepted(self, spans: dict[str, SensorSpan]) -> None:
        """Seven is the first legal window."""
        assert StuckValueDetector(spans, window_size=7).window_size == 7

    def test_negative_tolerance_is_rejected(self, spans: dict[str, SensorSpan]) -> None:
        """A negative tolerance is not a distance."""
        with pytest.raises(ValueError, match="tolerance_fraction"):
            StuckValueDetector(spans, tolerance_fraction=-0.1)

    def test_identical_stream_is_detected(self, spans: dict[str, SensorSpan]) -> None:
        """A frozen reading must be flagged once the run reaches the window."""
        detector = StuckValueDetector(spans, window_size=8)
        results = _run(detector, [50.0] * 10)

        assert all(not faults for faults in results[:7])
        assert results[7] and results[7][0].fault_type is FaultType.STUCK

    def test_run_of_six_is_not_detected(self, spans: dict[str, SensorSpan]) -> None:
        """The measured healthy floor must stay below the threshold."""
        detector = StuckValueDetector(spans, window_size=8)
        assert not any(_run(detector, [50.0] * 6))

    def test_variable_stream_is_not_detected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A moving sensor must never be flagged as stuck."""
        detector = StuckValueDetector(spans, window_size=8)
        values = [50.0 + 0.1 * i for i in range(40)]
        assert not any(_run(detector, values))

    def test_a_single_change_breaks_the_run(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """One differing sample must restart the count, not merely pause it."""
        detector = StuckValueDetector(spans, window_size=8)
        _run(detector, [50.0] * 7)
        assert detector.run_length(_TAG) == 7

        _run(detector, [51.0], start=_T0 + _SAMPLE * 7)
        assert detector.run_length(_TAG) == 1

    def test_reports_on_every_reading_while_stuck(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Per-reading reporting is what makes the T3.6 confusion matrix possible.

        La verdad-terreno de T3.6 se puntua por (sensor_id, timestep), asi que un
        detector que avisara una vez por episodio dejaria sin marcar todos los
        timesteps siguientes del mismo episodio.
        """
        detector = StuckValueDetector(spans, window_size=8)
        results = _run(detector, [50.0] * 20)
        assert all(results[i] for i in range(7, 20))

    def test_severity_escalates_with_the_run(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A longer freeze must outrank a shorter one."""
        detector = StuckValueDetector(spans, window_size=8)
        results = _run(detector, [50.0] * 40)

        assert results[7][0].severity is Severity.MEDIUM
        assert results[16][0].severity is Severity.HIGH
        assert results[32][0].severity is Severity.CRITICAL

    def test_confidence_saturates(self, spans: dict[str, SensorSpan]) -> None:
        """A long freeze must reach full confidence, and never exceed it."""
        detector = StuckValueDetector(spans, window_size=8)
        results = _run(detector, [50.0] * 60)
        assert results[7][0].confidence == pytest.approx(0.5)
        assert results[-1][0].confidence == 1.0

    def test_evidence_carries_the_run(self, spans: dict[str, SensorSpan]) -> None:
        """The fault must be defensible without re-running the detector."""
        detector = StuckValueDetector(spans, window_size=8)
        fault = _run(detector, [50.0] * 12)[-1][0]
        assert fault.evidence["run_length"] == 12
        assert fault.evidence["value"] == 50.0
        assert fault.detector == "StuckValueDetector"

    def test_sensors_are_tracked_independently(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """One frozen sensor must not implicate another."""
        detector = StuckValueDetector(spans, window_size=8)
        for index in range(12):
            timestamp = _T0 + _SAMPLE * index
            detector.check(_reading(50.0, timestamp, _TAG), 50.0)
            moving = 50.0 + index
            detector.check(_reading(moving, timestamp, _OTHER_TAG), moving)

        assert detector.run_length(_TAG) == 12
        assert detector.run_length(_OTHER_TAG) == 1

    def test_reset_clears_the_run(self, spans: dict[str, SensorSpan]) -> None:
        """After a reset the sensor must start counting again."""
        detector = StuckValueDetector(spans, window_size=8)
        _run(detector, [50.0] * 12)
        detector.reset(_TAG)
        assert detector.run_length(_TAG) == 0

    def test_tolerance_catches_a_nearly_frozen_sensor(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A dithering transmitter must be catchable by widening the tolerance."""
        strict = StuckValueDetector(spans, window_size=8, tolerance_fraction=0.0)
        loose = StuckValueDetector(spans, window_size=8, tolerance_fraction=0.001)
        dithering = [50.0 + (0.01 if i % 2 else 0.0) for i in range(12)]

        assert not any(_run(strict, dithering))
        assert any(_run(loose, dithering))

    def test_ignores_the_input_quality_flag(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Invariant 1: the verdict must not depend on the incoming quality."""
        detector_good = StuckValueDetector(spans, window_size=8)
        detector_bad = StuckValueDetector(spans, window_size=8)

        for index in range(12):
            timestamp = _T0 + _SAMPLE * index
            detector_good.check(_reading(50.0, timestamp, quality="good"), 50.0)
            detector_bad.check(_reading(50.0, timestamp, quality="bad"), 50.0)

        assert detector_good.run_length(_TAG) == detector_bad.run_length(_TAG)


# ---------------------------------------------------------------------------
# RateOfChangeDetector
# ---------------------------------------------------------------------------


class TestRateOfChangeDetector:
    """A physically impossible jump must be caught; a normal move must not."""

    def test_non_positive_limit_is_rejected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A zero limit would flag every change."""
        with pytest.raises(ValueError, match="max_envelope_fractions_per_second"):
            RateOfChangeDetector(spans, max_envelope_fractions_per_second=0.0)

    def test_first_reading_yields_nothing(self, spans: dict[str, SensorSpan]) -> None:
        """One point defines no rate."""
        detector = RateOfChangeDetector(spans)
        assert detector.check(_reading(50.0), 50.0) == []

    def test_large_jump_is_detected(self, spans: dict[str, SensorSpan]) -> None:
        """A full-envelope jump in one sample interval must be flagged."""
        detector = RateOfChangeDetector(spans)
        results = _run(detector, [30.0, 95.0])
        assert results[1] and results[1][0].fault_type is FaultType.NOISE_SPIKE

    def test_gradual_change_is_not_detected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A slow ramp is what the process legitimately does."""
        detector = RateOfChangeDetector(spans)
        assert not any(_run(detector, [50.0 + 0.5 * i for i in range(20)]))

    def test_shared_timestamp_yields_nothing(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Zero elapsed time makes the rate undefined, not infinite."""
        detector = RateOfChangeDetector(spans)
        detector.check(_reading(30.0, _T0), 30.0)
        assert detector.check(_reading(95.0, _T0), 95.0) == []

    def test_unknown_sensor_yields_nothing(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Without a span there is no width to normalise the rate by."""
        detector = RateOfChangeDetector(spans)
        detector.check(_reading(30.0, _T0, "TEP-UNKNOWN"), 30.0)
        assert detector.check(_reading(95.0, _T0 + _SAMPLE, "TEP-UNKNOWN"), 95.0) == []

    def test_direction_does_not_matter(self, spans: dict[str, SensorSpan]) -> None:
        """A collapse is as suspect as a spike."""
        detector = RateOfChangeDetector(spans)
        assert _run(detector, [95.0, 30.0])[1]

    def test_evidence_carries_the_rate(self, spans: dict[str, SensorSpan]) -> None:
        """The rate, the limit and the elapsed time must all be recoverable."""
        detector = RateOfChangeDetector(spans)
        fault = _run(detector, [30.0, 95.0])[1][0]
        assert fault.evidence["elapsed_seconds"] == 180.0
        assert fault.evidence["delta"] == pytest.approx(65.0)
        assert fault.evidence["rate_envelope_fractions_per_second"] > fault.evidence["limit"]

    def test_a_long_gap_forgives_the_same_jump(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """The same delta over an hour is a normal move, not a spike.

        Es la razon de que el limite sea una velocidad y no una diferencia: tras
        un hueco de comunicaciones, la reanudacion no debe alarmar.
        """
        detector = RateOfChangeDetector(spans)
        detector.check(_reading(30.0, _T0), 30.0)
        assert detector.check(_reading(95.0, _T0 + timedelta(hours=1)), 95.0) == []

    def test_reset_clears_the_last_reading(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """After a reset the next reading must be treated as the first."""
        detector = RateOfChangeDetector(spans)
        detector.check(_reading(30.0, _T0), 30.0)
        detector.reset(_TAG)
        assert detector.check(_reading(95.0, _T0 + _SAMPLE), 95.0) == []


# ---------------------------------------------------------------------------
# RangeValidator
# ---------------------------------------------------------------------------


class TestRangeValidator:
    """The envelope, not the calibrated span, is the operating limit."""

    def test_value_inside_the_envelope_is_clean(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Normal operation must never be flagged."""
        assert RangeValidator(spans).check(_reading(50.0), 50.0) == []

    @pytest.mark.parametrize("value", [20.0, 80.0])
    def test_envelope_edges_are_inclusive(
        self, spans: dict[str, SensorSpan], value: float
    ) -> None:
        """A value exactly on the limit is still within it."""
        assert RangeValidator(spans).check(_reading(value), value) == []

    def test_value_outside_the_envelope_is_detected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """An excursion beyond normal operation must be flagged."""
        faults = RangeValidator(spans).check(_reading(85.0), 85.0)
        assert faults and faults[0].fault_type is FaultType.BIAS_OUT_OF_RANGE

    def test_uses_the_envelope_and_not_the_calibrated_span(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Invariant 2, stated as a test.

        85.0 esta DENTRO del span calibrado [0, 100] y FUERA del sobre [20, 80].
        Un detector que comprobase contra `contains` lo dejaria pasar, y como el
        span lleva un margen del 200% por diseno, casi nada caeria nunca fuera:
        el detector quedaria practicamente ciego.
        """
        span = spans[_TAG]
        assert span.contains(85.0)
        assert not span.in_alarm_envelope(85.0)
        assert RangeValidator(spans).check(_reading(85.0), 85.0)

    def test_value_outside_the_span_is_critical(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A value the transmitter cannot represent is untrustworthy, not just odd."""
        faults = RangeValidator(spans).check(_reading(150.0), 150.0)
        assert faults[0].severity is Severity.CRITICAL
        assert faults[0].evidence["outside_calibrated_span"] is True

    def test_severity_grows_with_the_excursion(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A small excursion must outrank nothing and underrank a large one."""
        validator = RangeValidator(spans)
        small = validator.check(_reading(85.0), 85.0)[0]
        large = validator.check(_reading(99.0), 99.0)[0]
        assert small.severity is Severity.MEDIUM
        assert large.severity is Severity.HIGH

    def test_below_the_envelope_is_detected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Excursions below the envelope count too."""
        faults = RangeValidator(spans).check(_reading(10.0), 10.0)
        assert faults and faults[0].evidence["excess"] == pytest.approx(10.0)

    def test_unknown_sensor_yields_nothing(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A tag with no span cannot be range-checked."""
        assert RangeValidator(spans).check(_reading(1e6, _T0, "TEP-UNKNOWN"), 1e6) == []

    def test_is_stateless(self, spans: dict[str, SensorSpan]) -> None:
        """History must not change the verdict on a given value."""
        validator = RangeValidator(spans)
        _run(validator, [85.0] * 10)
        validator.reset(_TAG)
        assert validator.check(_reading(50.0), 50.0) == []


# ---------------------------------------------------------------------------
# CrossCorrelationChecker
# ---------------------------------------------------------------------------


class TestCrossCorrelationChecker:
    """Pairs are only checked when a baseline says they should track."""

    def test_window_below_three_is_rejected(self) -> None:
        """Pearson over two points is always +/-1 and says nothing."""
        with pytest.raises(ValueError, match="at least 3"):
            CrossCorrelationChecker(window_size=2)

    def test_baseline_outside_the_unit_range_is_rejected(self) -> None:
        """A Pearson coefficient above 1 is not a coefficient."""
        with pytest.raises(ValueError, match="outside"):
            CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.5})

    def test_empty_baselines_check_nothing(self) -> None:
        """The default must be inert, not a guess about the plant."""
        checker = CrossCorrelationChecker()
        assert not any(_run(checker, [50.0 + i for i in range(20)]))

    def _feed_pair(
        self,
        checker: CrossCorrelationChecker,
        values_a: list[float],
        values_b: list[float],
    ) -> list[SensorFault]:
        """Feed two synchronised series and return the faults of the last step."""
        faults: list[SensorFault] = []
        for index, (value_a, value_b) in enumerate(zip(values_a, values_b, strict=True)):
            timestamp = _T0 + _SAMPLE * index
            checker.check(_reading(value_a, timestamp, _TAG), value_a)
            faults = checker.check(_reading(value_b, timestamp, _OTHER_TAG), value_b)
        return faults

    def test_correlated_pair_is_clean(self) -> None:
        """A pair still tracking its baseline must not be flagged."""
        checker = CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.0})
        values = [50.0 + i for i in range(20)]
        assert self._feed_pair(checker, values, values) == []

    def test_decoupled_pair_is_detected(self) -> None:
        """A pair that has inverted relative to its baseline must be flagged."""
        checker = CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.0})
        rising = [50.0 + i for i in range(20)]
        falling = [50.0 - i for i in range(20)]

        faults = self._feed_pair(checker, rising, falling)
        assert faults and faults[0].fault_type is FaultType.DRIFT_CORRELATED
        assert faults[0].evidence["observed_pearson"] == pytest.approx(-1.0)

    def test_constant_series_yields_no_coefficient(self) -> None:
        """A frozen sensor has zero variance: its correlation is undefined.

        Devolver 0.0 en su lugar lo haria parecer un desacoplamiento perfecto, y
        todo par que incluyera a un sensor congelado se marcaria dos veces: una
        por stuck y otra, espuria, por correlacion.
        """
        checker = CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.0})
        moving = [50.0 + i for i in range(20)]
        frozen = [50.0] * 20

        assert self._feed_pair(checker, moving, frozen) == []
        assert checker.check_pair(_TAG, _OTHER_TAG) is None

    def test_too_few_shared_samples_yields_no_coefficient(self) -> None:
        """Under three aligned points the coefficient is not computed."""
        checker = CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.0})
        self._feed_pair(checker, [50.0, 51.0], [50.0, 51.0])
        assert checker.check_pair(_TAG, _OTHER_TAG) is None

    def test_buffers_align_on_timestamps(self) -> None:
        """Samples must be paired by instant, not by position in the buffer.

        Correlacionar la muestra n de un sensor con la n de otro cuando
        corresponden a instantes distintos produce un coeficiente sin significado.
        """
        checker = CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.0})
        for index in range(20):
            timestamp = _T0 + _SAMPLE * index
            value = 50.0 + index
            checker.check(_reading(value, timestamp, _TAG), value)
            # El segundo sensor solo publica en los instantes pares.
            if index % 2 == 0:
                checker.check(_reading(value, timestamp, _OTHER_TAG), value)

        assert checker.check_pair(_TAG, _OTHER_TAG) == pytest.approx(1.0)

    def test_unknown_pair_yields_no_coefficient(self) -> None:
        """A pair with no buffered data must not raise."""
        assert CrossCorrelationChecker().check_pair("A", "B") is None

    def test_reading_outside_every_baseline_pair_is_clean(self) -> None:
        """A sensor named in no baseline pair must be checked against nothing.

        El detector solo evalua los pares que le conciernen; una lectura de un
        tercer sensor no debe disparar el par (_TAG, _OTHER_TAG).
        """
        checker = CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.0})
        assert checker.check(_reading(50.0, _T0, "TEP-XMEAS-99"), 50.0) == []

    def test_reset_clears_the_buffer(self) -> None:
        """After a reset the sensor's history must be gone."""
        checker = CrossCorrelationChecker({(_TAG, _OTHER_TAG): 1.0})
        self._feed_pair(checker, [50.0 + i for i in range(20)], [50.0 + i for i in range(20)])
        checker.reset(_TAG)
        assert checker.check_pair(_TAG, _OTHER_TAG) is None


# ---------------------------------------------------------------------------
# KalmanResidualDetector
# ---------------------------------------------------------------------------


class TestKalmanResidualDetector:
    """Two signals, two failure modes: transients and established drift."""

    def test_invalid_drift_limit_is_rejected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A non-positive drift limit would flag every sensor."""
        with pytest.raises(ValueError, match="max_drift"):
            KalmanResidualDetector(spans, max_drift_envelope_fractions_per_hour=0.0)

    def test_invalid_min_steps_is_rejected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Trusting the velocity from step zero makes every new sensor drift."""
        with pytest.raises(ValueError, match="min_steps_before_drift"):
            KalmanResidualDetector(spans, min_steps_before_drift=0)

    def test_invalid_filter_tuning_is_rejected(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Tuning errors must surface at construction, not mid-stream."""
        with pytest.raises(ValueError, match="observation_noise"):
            KalmanResidualDetector(spans, observation_noise=0.0)

    def test_steady_signal_is_clean(self, spans: dict[str, SensorSpan]) -> None:
        """A quiet sensor must produce no faults at all."""
        detector = KalmanResidualDetector(spans)
        assert not any(_run(detector, [50.0] * 40))

    def test_first_reading_is_never_flagged(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """There is nothing to compare a sensor's first reading against."""
        detector = KalmanResidualDetector(spans)
        assert detector.check(_reading(1e5), 1e5) == []

    def test_abrupt_jump_is_flagged(self, spans: dict[str, SensorSpan]) -> None:
        """A step change must raise a kalman_anomaly."""
        detector = KalmanResidualDetector(spans)
        _run(detector, [50.0] * 30)
        faults = detector.check(_reading(75.0, _T0 + _SAMPLE * 30), 75.0)

        types = [f.fault_type for f in faults]
        assert FaultType.KALMAN_ANOMALY in types

    def test_evidence_carries_the_residual(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """The normalized residual and its threshold must be recoverable."""
        detector = KalmanResidualDetector(spans)
        _run(detector, [50.0] * 30)
        fault = detector.check(_reading(75.0, _T0 + _SAMPLE * 30), 75.0)[0]

        assert abs(fault.evidence["normalized_residual"]) > fault.evidence["k_sigma"]
        assert fault.detector == "KalmanResidualDetector"

    def test_established_drift_is_flagged_by_velocity(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """The drift signal must fire where the residual has gone quiet.

        Es la consecuencia directa del hallazgo de T4.1: el residual no ve una
        deriva lineal sostenida porque el filtro le aprende la pendiente.
        """
        detector = KalmanResidualDetector(
            spans, max_drift_envelope_fractions_per_hour=_SYNTHETIC_DRIFT_LIMIT
        )
        results = _run(detector, [30.0 + 5.0 * i for i in range(30)])

        late = results[-1]
        assert any(f.fault_type is FaultType.SENSOR_DRIFT for f in late)

    def test_drift_is_not_reported_before_the_minimum_history(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A brand-new sensor must not be accused of drifting."""
        detector = KalmanResidualDetector(spans, min_steps_before_drift=20)
        results = _run(detector, [30.0 + 5.0 * i for i in range(10)])
        assert not any(
            f.fault_type is FaultType.SENSOR_DRIFT for faults in results for f in faults
        )

    def test_drift_evidence_carries_the_velocity(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """The drift rate and the limit it broke must both be recorded."""
        detector = KalmanResidualDetector(
            spans, max_drift_envelope_fractions_per_hour=_SYNTHETIC_DRIFT_LIMIT
        )
        results = _run(detector, [30.0 + 5.0 * i for i in range(30)])
        drift = next(
            f for f in results[-1] if f.fault_type is FaultType.SENSOR_DRIFT
        )
        assert drift.evidence["drift_envelope_fractions_per_hour"] > drift.evidence["limit"]

    def test_unknown_sensor_reports_no_drift(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Without a span there is no width to normalise the velocity by."""
        detector = KalmanResidualDetector(spans)
        results = _run(detector, [30.0 + 5.0 * i for i in range(30)], "TEP-UNKNOWN")
        assert not any(
            f.fault_type is FaultType.SENSOR_DRIFT for faults in results for f in faults
        )

    def test_sensors_are_independent(self, spans: dict[str, SensorSpan]) -> None:
        """A jump on one tag must not implicate another."""
        detector = KalmanResidualDetector(spans)
        for index in range(30):
            timestamp = _T0 + _SAMPLE * index
            detector.check(_reading(50.0, timestamp, _TAG), 50.0)
            detector.check(_reading(50.0, timestamp, _OTHER_TAG), 50.0)

        last = _T0 + _SAMPLE * 30
        spiked = detector.check(_reading(75.0, last, _TAG), 75.0)
        quiet = detector.check(_reading(50.0, last, _OTHER_TAG), 50.0)

        assert spiked
        assert quiet == []

    def test_reset_clears_the_filter(self, spans: dict[str, SensorSpan]) -> None:
        """After a reset the next reading must behave as the sensor's first."""
        detector = KalmanResidualDetector(spans)
        _run(detector, [50.0] * 30)
        detector.reset(_TAG)
        assert detector.check(_reading(1e5, _T0 + _SAMPLE * 31), 1e5) == []


# ---------------------------------------------------------------------------
# SensorValidator
# ---------------------------------------------------------------------------


class TestSensorValidator:
    """The orchestrator must aggregate verdicts without losing any."""

    def test_clean_reading_is_valid(self, spans: dict[str, SensorSpan]) -> None:
        """A normal reading must pass with no faults."""
        result = SensorValidator(spans).validate(_reading(50.0))
        assert result.is_valid
        assert result.faults == []

    def test_default_detector_set(self, spans: dict[str, SensorSpan]) -> None:
        """All five detectors must be wired by default."""
        names = {type(d).__name__ for d in SensorValidator(spans).detectors}
        assert names == {
            "StuckValueDetector",
            "RateOfChangeDetector",
            "RangeValidator",
            "CrossCorrelationChecker",
            "KalmanResidualDetector",
        }

    def test_stuck_and_out_of_range_are_both_reported(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Concurrent faults must not shadow one another."""
        validator = SensorValidator(spans)
        for index in range(12):
            result = validator.validate(_reading(85.0, _T0 + _SAMPLE * index))

        types = {fault.fault_type for fault in result.faults}
        assert FaultType.STUCK in types
        assert FaultType.BIAS_OUT_OF_RANGE in types

    def test_faulty_reading_is_marked_suspect(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A non-critical fault must downgrade quality to SUSPECT."""
        validator = SensorValidator(spans)
        result = validator.validate(_reading(85.0))
        assert result.enriched_reading.measurement.quality is QualityFlag.SUSPECT
        assert result.enriched_reading.is_usable

    def test_critical_fault_is_marked_bad(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A value the transmitter cannot represent must become unusable."""
        result = SensorValidator(spans).validate(_reading(150.0))
        assert result.enriched_reading.measurement.quality is QualityFlag.BAD
        assert not result.enriched_reading.is_usable

    def test_clean_reading_keeps_its_quality(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """With no faults the reading must pass through untouched."""
        reading = _reading(50.0)
        result = SensorValidator(spans).validate(reading)
        assert result.enriched_reading is reading

    def test_original_reading_is_never_mutated(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Enrichment must copy, so the input stays auditable."""
        reading = _reading(150.0)
        SensorValidator(spans).validate(reading)
        assert reading.measurement.quality is QualityFlag.GOOD

    def test_input_quality_does_not_change_the_verdict(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Invariant 1 at the orchestrator level.

        Un reading marcado BAD por el SCADA y otro marcado GOOD, con el mismo
        valor, deben producir exactamente los mismos fallos. Leer quality como
        entrada convertiria la evaluacion del validador en una tautologia.
        """
        good = SensorValidator(spans).validate(_reading(85.0, quality="good"))
        bad = SensorValidator(spans).validate(_reading(85.0, quality="bad"))

        assert [f.fault_type for f in good.faults] == [f.fault_type for f in bad.faults]

    def test_valueless_reading_passes_through(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A reading with no value cannot be judged, and must not be imputed."""
        reading = _reading(None)
        result = SensorValidator(spans).validate(reading)

        assert result.is_valid
        assert result.faults == []
        assert result.enriched_reading is reading

    def test_worst_severity(self, spans: dict[str, SensorSpan]) -> None:
        """The result must surface the highest severity it carries."""
        validator = SensorValidator(spans)
        assert validator.validate(_reading(50.0)).worst_severity is None
        assert validator.validate(_reading(150.0)).worst_severity is Severity.CRITICAL

    def test_custom_detector_set_is_honoured(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Running a subset must be possible, for per-detector scoring."""
        validator = SensorValidator(spans, detectors=[RangeValidator(spans)])
        result = validator.validate(_reading(85.0))
        assert len(result.faults) == 1
        assert result.faults[0].detector == "RangeValidator"

    def test_reset_sensor_clears_every_detector(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """A reset must reach the stateful detectors, not just the first."""
        validator = SensorValidator(spans)
        for index in range(12):
            validator.validate(_reading(50.0, _T0 + _SAMPLE * index))

        validator.reset_sensor(_TAG)
        result = validator.validate(_reading(50.0, _T0 + _SAMPLE * 12))
        assert not any(f.fault_type is FaultType.STUCK for f in result.faults)


class TestValidatorMetrics:
    """Counters must be plain data, ready for the T3.5 consumer to publish."""

    def test_counts_readings(self, spans: dict[str, SensorSpan]) -> None:
        """Every reading must be counted, faulty or not."""
        validator = SensorValidator(spans)
        for index in range(5):
            validator.validate(_reading(50.0, _T0 + _SAMPLE * index))
        assert validator.get_metrics()["readings_total"] == 5

    def test_counts_faults_by_type(self, spans: dict[str, SensorSpan]) -> None:
        """Fault counts must be broken down by type."""
        validator = SensorValidator(spans)
        for index in range(12):
            validator.validate(_reading(85.0, _T0 + _SAMPLE * index))

        metrics = validator.get_metrics()
        assert metrics["faults_by_type"]["bias_out_of_range"] == 12
        assert metrics["faults_by_type"]["stuck"] == 5
        assert metrics["faults_total"] == sum(metrics["faults_by_type"].values())

    def test_counts_valueless_readings(self, spans: dict[str, SensorSpan]) -> None:
        """Unjudged readings must stay visible instead of vanishing."""
        validator = SensorValidator(spans)
        validator.validate(_reading(None))
        validator.validate(_reading(50.0))
        assert validator.get_metrics()["readings_without_value"] == 1

    def test_counts_unknown_sensors(self, spans: dict[str, SensorSpan]) -> None:
        """Schema drift must be countable, not silent.

        Un tag sin span atraviesa casi todos los detectores sin ser evaluado. Si
        eso no se contara, una deriva de schema se manifestaria como una caida
        silenciosa de la tasa de deteccion.
        """
        validator = SensorValidator(spans)
        validator.validate(_reading(50.0, _T0, "TEP-UNKNOWN"))
        validator.validate(_reading(50.0, _T0, _TAG))
        assert validator.get_metrics()["readings_unknown_sensor"] == 1

    def test_reports_latency(self, spans: dict[str, SensorSpan]) -> None:
        """Latency percentiles must be present and finite."""
        validator = SensorValidator(spans)
        for index in range(20):
            validator.validate(_reading(50.0, _T0 + _SAMPLE * index))

        latency = validator.get_metrics()["latency_ms"]
        assert latency["p99"] >= latency["p50"] > 0.0

    def test_metrics_on_a_fresh_validator(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """An unused validator must report zeros, not crash on empty arrays."""
        metrics = SensorValidator(spans).get_metrics()
        assert metrics["readings_total"] == 0
        assert metrics["latency_ms"]["p50"] == 0.0


class TestLatencyWindow:
    """The latency history must be bounded: a continuous service never stops."""

    _WINDOW = 100

    def test_history_does_not_grow_past_the_window(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """After 3x the window the retained history must still equal the window."""
        validator = SensorValidator(spans, latency_window=self._WINDOW)
        sizes = []
        for index in range(3 * self._WINDOW):
            validator.validate(_reading(50.0, _T0 + _SAMPLE * index))
            sizes.append(len(validator._latencies_seconds))

        assert sizes[self._WINDOW - 1] == self._WINDOW
        assert sizes[2 * self._WINDOW - 1] == self._WINDOW
        assert sizes[-1] == self._WINDOW
        assert max(sizes) == self._WINDOW

    def test_counters_stay_cumulative_when_the_window_slides(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """Bounding the latency history must not bound the counters."""
        validator = SensorValidator(spans, latency_window=self._WINDOW)
        for index in range(3 * self._WINDOW):
            validator.validate(_reading(50.0, _T0 + _SAMPLE * index))

        assert validator.get_metrics()["readings_total"] == 3 * self._WINDOW

    def test_valueless_readings_are_bounded_too(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """The early-return path for unjudged readings records latency as well."""
        validator = SensorValidator(spans, latency_window=5)
        for index in range(20):
            validator.validate(_reading(None, _T0 + _SAMPLE * index))

        assert len(validator._latencies_seconds) == 5

    def test_percentiles_are_computed_over_the_recent_window(
        self, spans: dict[str, SensorSpan], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Percentiles must describe the last `window` readings, not the whole run.

        Se sustituye el reloj del modulo por uno guionizado: la lectura i-esima
        tarda (i + 1) ms, con i en [0, 300). La ventana de 100 retiene las
        lecturas 201..300 ms, de modo que los valores esperados salen a mano:
        media 250,5 ms, p50 250,5 ms y p99 = 201 + 0,99 * 99 = 299,01 ms. Si la
        ventana no descartara lo antiguo, p50 seria 150,5 ms.
        """
        latencies = [(index + 1) / 1000.0 for index in range(3 * self._WINDOW)]
        ticks = chain.from_iterable((0.0, latency) for latency in latencies)
        clock = SimpleNamespace(perf_counter=lambda: next(ticks))
        monkeypatch.setattr("data.validation.sensor_validator.time", clock)

        validator = SensorValidator(spans, latency_window=self._WINDOW)
        for index in range(3 * self._WINDOW):
            validator.validate(_reading(50.0, _T0 + _SAMPLE * index))

        latency = validator.get_metrics()["latency_ms"]
        assert latency["mean"] == pytest.approx(250.5)
        assert latency["p50"] == pytest.approx(250.5)
        assert latency["p99"] == pytest.approx(299.01)

    def test_default_window_is_ten_thousand(
        self, spans: dict[str, SensorSpan]
    ) -> None:
        """The documented default must be the one actually applied."""
        assert SensorValidator(spans)._latencies_seconds.maxlen == 10_000

    @pytest.mark.parametrize("window", [0, -1])
    def test_non_positive_window_is_rejected(
        self, spans: dict[str, SensorSpan], window: int
    ) -> None:
        """A window of zero would silently disable latency reporting."""
        with pytest.raises(ValueError, match="latency_window must be positive"):
            SensorValidator(spans, latency_window=window)
