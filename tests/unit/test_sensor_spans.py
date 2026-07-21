"""Unit tests for data.schemas.sensor_spans.

Este modulo es el unico lector del fichero de spans, y de el dependen dos cosas
que fallan en silencio si el fichero esta mal: el escalado a cuentas ADC del
adaptador y el umbral del detector de fuera-de-rango. Los tests cubren por eso
tanto la aritmetica (recorte en ambos extremos del ADC, sobre de alarma mas
estrecho que el span) como el rechazo de tablas invalidas con una causa nombrada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from data.schemas.sensor_spans import (
    ADC_MAX_COUNTS,
    SensorSpan,
    load_sensor_spans,
)

_VALID_ENTRY: dict[str, Any] = {
    "min": 0.0,
    "max": 100.0,
    "alarm_min": 20.0,
    "alarm_max": 80.0,
    "unit": "kPa",
}


def _write_spans(path: Path, sensors: object) -> Path:
    """Write a span document holding the given sensors: section.

    Args:
        path: Destination file.
        sensors: Value to place under the sensors: key.

    Returns:
        The path written to.
    """
    path.write_text(yaml.safe_dump({"sensors": sensors}), encoding="utf-8")
    return path


@pytest.fixture()
def span() -> SensorSpan:
    """Return a span of [0, 100] with an alarm envelope of [20, 80]."""
    return SensorSpan(sensor_id="TEP-XMEAS-01", **_VALID_ENTRY)


class TestSensorSpanGeometry:
    """Width, containment and the alarm envelope must agree with the ranges."""

    def test_width(self, span: SensorSpan) -> None:
        """width must be max - min."""
        assert span.width == pytest.approx(100.0)

    def test_is_frozen(self, span: SensorSpan) -> None:
        """A span must be immutable so no consumer can rescale it in place."""
        with pytest.raises(AttributeError):
            span.min = -1.0  # type: ignore[misc]

    @pytest.mark.parametrize("value", [0.0, 50.0, 100.0])
    def test_contains_inside_and_on_edges(self, span: SensorSpan, value: float) -> None:
        """The calibrated span must be inclusive of both ends."""
        assert span.contains(value)

    @pytest.mark.parametrize("value", [-0.01, 100.01])
    def test_contains_outside(self, span: SensorSpan, value: float) -> None:
        """Values beyond either end must not be contained."""
        assert not span.contains(value)

    @pytest.mark.parametrize("value", [20.0, 50.0, 80.0])
    def test_in_alarm_envelope_inside_and_on_edges(
        self, span: SensorSpan, value: float
    ) -> None:
        """The alarm envelope must be inclusive of both ends."""
        assert span.in_alarm_envelope(value)

    @pytest.mark.parametrize("value", [19.99, 80.01])
    def test_in_alarm_envelope_outside(self, span: SensorSpan, value: float) -> None:
        """Values beyond the envelope must be reported as excursions."""
        assert not span.in_alarm_envelope(value)

    @pytest.mark.parametrize("value", [5.0, 95.0])
    def test_envelope_is_strictly_narrower_than_span(
        self, span: SensorSpan, value: float
    ) -> None:
        """A value can be representable and still be out of normal operation.

        Esta es la distincion que motiva los dos rangos: el detector de rango usa
        in_alarm_envelope, no contains, y confundirlos lo dejaria ciego.
        """
        assert span.contains(value)
        assert not span.in_alarm_envelope(value)


class TestToRawCounts:
    """Scaling must span the full 16-bit range and clamp outside the span."""

    def test_span_minimum_maps_to_zero(self, span: SensorSpan) -> None:
        """The bottom of the span must be count 0."""
        assert span.to_raw_counts(0.0) == 0

    def test_span_maximum_maps_to_full_scale(self, span: SensorSpan) -> None:
        """The top of the span must be full scale."""
        assert span.to_raw_counts(100.0) == ADC_MAX_COUNTS

    def test_midpoint_maps_to_half_scale(self, span: SensorSpan) -> None:
        """The midpoint must land within one count of half scale."""
        assert span.to_raw_counts(50.0) == pytest.approx(ADC_MAX_COUNTS // 2, abs=1)

    def test_clamps_below_span(self, span: SensorSpan) -> None:
        """A value under the span must clamp to 0, as a real input card does."""
        assert span.to_raw_counts(-1000.0) == 0

    def test_clamps_above_span(self, span: SensorSpan) -> None:
        """A value over the span must clamp to full scale, not overflow."""
        assert span.to_raw_counts(1e9) == ADC_MAX_COUNTS

    def test_returns_int(self, span: SensorSpan) -> None:
        """Counts must be integers: an ADC has no fractional codes."""
        assert isinstance(span.to_raw_counts(33.3), int)

    def test_negative_span_is_scaled_not_clamped(self) -> None:
        """A span straddling zero must still scale linearly across its width."""
        bipolar = SensorSpan(
            sensor_id="TEP-XMV-01",
            min=-50.0,
            max=50.0,
            alarm_min=-10.0,
            alarm_max=10.0,
            unit="pct",
        )
        assert bipolar.to_raw_counts(-50.0) == 0
        assert bipolar.to_raw_counts(50.0) == ADC_MAX_COUNTS
        assert bipolar.to_raw_counts(0.0) == pytest.approx(ADC_MAX_COUNTS // 2, abs=1)


class TestLoadSensorSpans:
    """A well-formed table must resolve into typed spans keyed by tag."""

    def test_loads_entries(self, tmp_path: Path) -> None:
        """Every entry must become a SensorSpan keyed by its sensor_id."""
        path = _write_spans(
            tmp_path / "spans.yaml",
            {"TEP-XMEAS-01": dict(_VALID_ENTRY), "TEP-XMEAS-02": dict(_VALID_ENTRY)},
        )
        spans = load_sensor_spans(path)
        assert set(spans) == {"TEP-XMEAS-01", "TEP-XMEAS-02"}
        assert all(isinstance(s, SensorSpan) for s in spans.values())

    def test_sensor_id_is_propagated_from_the_key(self, tmp_path: Path) -> None:
        """The mapping key must become the span's sensor_id."""
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMV-04": dict(_VALID_ENTRY)})
        assert load_sensor_spans(path)["TEP-XMV-04"].sensor_id == "TEP-XMV-04"

    def test_values_are_coerced_to_float(self, tmp_path: Path) -> None:
        """Integer literals in YAML must arrive as floats."""
        entry = {"min": 0, "max": 100, "alarm_min": 20, "alarm_max": 80, "unit": "kPa"}
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": entry})
        span = load_sensor_spans(path)["TEP-XMEAS-01"]
        assert isinstance(span.min, float)
        assert isinstance(span.max, float)

    def test_accepts_an_empty_sensors_section(self, tmp_path: Path) -> None:
        """An explicit but empty sensors: mapping must yield an empty table."""
        path = _write_spans(tmp_path / "spans.yaml", {})
        assert load_sensor_spans(path) == {}

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        """The loader must accept str as well as Path."""
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": dict(_VALID_ENTRY)})
        assert len(load_sensor_spans(str(path))) == 1


class TestInvalidSpanTable:
    """A malformed table must fail loudly, naming the offending sensor."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="derive_sensor_spans"):
            load_sensor_spans(tmp_path / "absent.yaml")

    def test_missing_sensors_section(self, tmp_path: Path) -> None:
        """A document without sensors: must name the section."""
        path = tmp_path / "spans.yaml"
        path.write_text(yaml.safe_dump({"other": {}}), encoding="utf-8")
        with pytest.raises(KeyError, match="sensors"):
            load_sensor_spans(path)

    def test_empty_document(self, tmp_path: Path) -> None:
        """An empty file must be reported as a missing sensors: section."""
        path = tmp_path / "spans.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(KeyError, match="sensors"):
            load_sensor_spans(path)

    def test_sensors_section_is_not_a_mapping(self, tmp_path: Path) -> None:
        """A sensors: list instead of a mapping must raise TypeError."""
        path = _write_spans(tmp_path / "spans.yaml", ["TEP-XMEAS-01"])
        with pytest.raises(TypeError, match="sensors"):
            load_sensor_spans(path)

    def test_entry_is_not_a_mapping(self, tmp_path: Path) -> None:
        """A scalar entry must raise TypeError naming the sensor."""
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": 3000.0})
        with pytest.raises(TypeError, match="TEP-XMEAS-01"):
            load_sensor_spans(path)

    @pytest.mark.parametrize("missing_key", sorted(_VALID_ENTRY))
    def test_each_required_key_is_enforced(
        self, tmp_path: Path, missing_key: str
    ) -> None:
        """Dropping any key must raise a KeyError naming that key."""
        entry = {k: v for k, v in _VALID_ENTRY.items() if k != missing_key}
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": entry})
        with pytest.raises(KeyError, match=missing_key):
            load_sensor_spans(path)

    @pytest.mark.parametrize(
        ("min_value", "max_value"),
        [(100.0, 100.0), (100.0, 50.0)],
        ids=["zero-width", "inverted"],
    )
    def test_non_positive_span(
        self, tmp_path: Path, min_value: float, max_value: float
    ) -> None:
        """A span that is not strictly positive must raise ValueError.

        Ancho cero haria una division por cero en to_raw_counts, y un span
        invertido aceptaria en silencio un rango que no contiene nada.
        """
        entry = dict(_VALID_ENTRY, min=min_value, max=max_value)
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": entry})
        with pytest.raises(ValueError, match="non-positive span"):
            load_sensor_spans(path)

    @pytest.mark.parametrize(
        ("alarm_min", "alarm_max"),
        [(50.0, 50.0), (80.0, 20.0)],
        ids=["zero-width", "inverted"],
    )
    def test_non_positive_alarm_envelope(
        self, tmp_path: Path, alarm_min: float, alarm_max: float
    ) -> None:
        """An envelope that is not strictly positive must raise ValueError."""
        entry = dict(_VALID_ENTRY, alarm_min=alarm_min, alarm_max=alarm_max)
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": entry})
        with pytest.raises(ValueError, match="non-positive alarm envelope"):
            load_sensor_spans(path)

    @pytest.mark.parametrize(
        ("alarm_min", "alarm_max"),
        [(-10.0, 80.0), (20.0, 110.0), (-10.0, 110.0)],
        ids=["below-span", "above-span", "both-sides"],
    )
    def test_alarm_envelope_outside_the_span(
        self, tmp_path: Path, alarm_min: float, alarm_max: float
    ) -> None:
        """An envelope wider than the span must raise ValueError.

        Un transmisor no puede alarmar sobre un valor que su ADC no representa:
        el recorte de to_raw_counts lo haria inalcanzable.
        """
        entry = dict(_VALID_ENTRY, alarm_min=alarm_min, alarm_max=alarm_max)
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": entry})
        with pytest.raises(ValueError, match="outside its calibrated span"):
            load_sensor_spans(path)

    def test_envelope_may_coincide_with_the_span(self, tmp_path: Path) -> None:
        """An envelope equal to the span must be accepted.

        Es el caso que produce derive_sensor_spans.py para un canal constante en
        d00, donde ambos rangos salen del mismo ancho minimo.
        """
        entry = dict(_VALID_ENTRY, alarm_min=0.0, alarm_max=100.0)
        path = _write_spans(tmp_path / "spans.yaml", {"TEP-XMEAS-01": entry})
        assert load_sensor_spans(path)["TEP-XMEAS-01"].alarm_min == 0.0
