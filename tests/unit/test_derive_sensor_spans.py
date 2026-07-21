"""Unit tests for data.generators.derive_sensor_spans.

El deriver se ejecuta a mano y su salida se commitea, asi que no hay un `dvc
repro` que vuelva a pasar por aqui y delate una regresion. Estos tests son la
unica red: comprueban la aritmetica de los dos margenes, el caso del canal
constante en d00, el rechazo de margenes incoherentes y que lo generado sea
legible por load_sensor_spans sin retoques manuales.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data.generators.derive_sensor_spans import (
    DEFAULT_ALARM_MARGIN,
    DEFAULT_SPAN_MARGIN,
    MIN_ABSOLUTE_WIDTH,
    NORMAL_OPERATION_FILE,
    derive_spans,
    main,
    render_yaml,
)
from data.generators.tep_adapter import SENSOR_TYPE_MAP, UNIT_MAP, sensor_id
from data.generators.tep_loader import N_COLUMNS
from data.schemas.sensor_spans import load_sensor_spans

_N_ROWS = 20


def _write_normal_file(
    raw_dir: Path, values: np.ndarray | None = None, name: str = NORMAL_OPERATION_FILE
) -> Path:
    """Write a synthetic normal-operation .dat file.

    Args:
        raw_dir: Directory to write into; created if absent.
        values: Array of shape (n_rows, 52). Defaults to a per-column ramp whose
            observed range is [col, col + n_rows - 1].
        name: File name to write.

    Returns:
        Path to the written file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = np.array(
            [[float(col + row) for col in range(N_COLUMNS)] for row in range(_N_ROWS)]
        )
    lines = [" ".join(f"{v:.6f}" for v in row) for row in values]
    path = raw_dir / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def normal_file(tmp_path: Path) -> Path:
    """Return a synthetic d00.dat with a known range per column."""
    return _write_normal_file(tmp_path / "raw")


class TestDeriveSpans:
    """The derived table must cover every tag with a coherent pair of ranges."""

    def test_covers_every_tep_tag(self, normal_file: Path) -> None:
        """One entry per process variable, no more and no less."""
        spans = derive_spans(normal_file)
        assert len(spans) == N_COLUMNS
        assert set(spans) == {sensor_id(col) for col in range(N_COLUMNS)}

    def test_entry_shape(self, normal_file: Path) -> None:
        """Each entry must hold exactly the keys the loader requires."""
        entry = derive_spans(normal_file)[sensor_id(0)]
        assert set(entry) == {"min", "max", "alarm_min", "alarm_max", "unit"}

    def test_margins_are_applied_to_the_observed_range(self, normal_file: Path) -> None:
        """Both pads must be the observed width times their own margin.

        La columna 0 del fichero sintetico va de 0 a 19, ancho observado 19.
        """
        entry = derive_spans(normal_file, alarm_margin=0.2, span_margin=2.0)[sensor_id(0)]
        observed_low, observed_high, width = 0.0, float(_N_ROWS - 1), float(_N_ROWS - 1)
        assert entry["alarm_min"] == pytest.approx(observed_low - width * 0.2)
        assert entry["alarm_max"] == pytest.approx(observed_high + width * 0.2)
        assert entry["min"] == pytest.approx(observed_low - width * 2.0)
        assert entry["max"] == pytest.approx(observed_high + width * 2.0)

    def test_envelope_is_strictly_inside_the_span(self, normal_file: Path) -> None:
        """With distinct margins, every envelope must sit inside its span."""
        for tag, entry in derive_spans(normal_file).items():
            assert entry["min"] < entry["alarm_min"], tag
            assert entry["alarm_min"] < entry["alarm_max"], tag
            assert entry["alarm_max"] < entry["max"], tag

    def test_observed_range_is_inside_the_envelope(self, normal_file: Path) -> None:
        """Normal operation must never fall outside the alarm envelope.

        Es la propiedad que sostiene los 0,000% de falsos positivos sobre d00.
        """
        for col in range(N_COLUMNS):
            entry = derive_spans(normal_file)[sensor_id(col)]
            assert entry["alarm_min"] <= float(col)
            assert entry["alarm_max"] >= float(col + _N_ROWS - 1)

    def test_units_come_from_the_adapter_maps(self, normal_file: Path) -> None:
        """Units must be the adapter's, not a second source of truth."""
        spans = derive_spans(normal_file)
        for col in range(N_COLUMNS):
            expected = UNIT_MAP[SENSOR_TYPE_MAP[col]].value
            assert spans[sensor_id(col)]["unit"] == expected

    def test_equal_margins_produce_a_coincident_envelope(self, normal_file: Path) -> None:
        """With span_margin == alarm_margin both ranges must coincide."""
        entry = derive_spans(normal_file, alarm_margin=0.5, span_margin=0.5)[sensor_id(0)]
        assert entry["min"] == pytest.approx(entry["alarm_min"])
        assert entry["max"] == pytest.approx(entry["alarm_max"])

    def test_zero_margins_reduce_to_the_observed_range(self, normal_file: Path) -> None:
        """Zero margins are legal and must yield the raw observed range."""
        entry = derive_spans(normal_file, alarm_margin=0.0, span_margin=0.0)[sensor_id(0)]
        assert entry["min"] == pytest.approx(0.0)
        assert entry["max"] == pytest.approx(float(_N_ROWS - 1))

    def test_output_is_loadable_by_the_schema(self, tmp_path: Path, normal_file: Path) -> None:
        """The generated document must satisfy load_sensor_spans unedited."""
        path = tmp_path / "spans.yaml"
        path.write_text(
            render_yaml(
                derive_spans(normal_file),
                NORMAL_OPERATION_FILE,
                DEFAULT_ALARM_MARGIN,
                DEFAULT_SPAN_MARGIN,
            ),
            encoding="utf-8",
        )
        assert len(load_sensor_spans(path)) == N_COLUMNS


class TestConstantChannel:
    """A variable that does not move in d00 must still get a usable span."""

    @pytest.fixture()
    def constant_file(self, tmp_path: Path) -> Path:
        """Return a d00.dat whose column 3 is frozen at 42.0."""
        values = np.array(
            [[float(col + row) for col in range(N_COLUMNS)] for row in range(_N_ROWS)]
        )
        values[:, 3] = 42.0
        return _write_normal_file(tmp_path / "raw", values)

    def test_constant_channel_gets_the_minimum_width(self, constant_file: Path) -> None:
        """A zero-width observation must widen to MIN_ABSOLUTE_WIDTH.

        Un span de ancho cero dividiria por cero en to_raw_counts.
        """
        entry = derive_spans(constant_file)[sensor_id(3)]
        assert entry["max"] - entry["min"] == pytest.approx(MIN_ABSOLUTE_WIDTH)
        assert entry["min"] == pytest.approx(42.0 - MIN_ABSOLUTE_WIDTH / 2.0)
        assert entry["max"] == pytest.approx(42.0 + MIN_ABSOLUTE_WIDTH / 2.0)

    def test_constant_channel_survives_the_loader(
        self, tmp_path: Path, constant_file: Path
    ) -> None:
        """The minimum-width entry must be accepted by load_sensor_spans."""
        path = tmp_path / "spans.yaml"
        path.write_text(
            render_yaml(
                derive_spans(constant_file),
                NORMAL_OPERATION_FILE,
                DEFAULT_ALARM_MARGIN,
                DEFAULT_SPAN_MARGIN,
            ),
            encoding="utf-8",
        )
        span = load_sensor_spans(path)[sensor_id(3)]
        assert span.width == pytest.approx(MIN_ABSOLUTE_WIDTH)
        assert span.to_raw_counts(42.0) > 0

    def test_warns_about_the_constant_channel(
        self, constant_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The operator must be told which channel did not move."""
        with caplog.at_level("WARNING"):
            derive_spans(constant_file)
        assert sensor_id(3) in caplog.text


class TestInvalidArguments:
    """Incoherent margins must be rejected before any file is written."""

    @pytest.mark.parametrize(
        ("alarm_margin", "span_margin"),
        [(-0.1, 2.0), (0.2, -2.0)],
        ids=["negative-alarm", "negative-span"],
    )
    def test_negative_margins(
        self, normal_file: Path, alarm_margin: float, span_margin: float
    ) -> None:
        """A negative margin would invert the range; it must raise."""
        with pytest.raises(ValueError, match="non-negative"):
            derive_spans(normal_file, alarm_margin, span_margin)

    def test_span_margin_below_alarm_margin(self, normal_file: Path) -> None:
        """The envelope cannot be wider than the span it lives in."""
        with pytest.raises(ValueError, match="at least alarm_margin"):
            derive_spans(normal_file, alarm_margin=0.5, span_margin=0.2)

    def test_missing_normal_operation_file(self, tmp_path: Path) -> None:
        """An absent d00.dat must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            derive_spans(tmp_path / "absent.dat")


class TestMain:
    """The entrypoint must write a loadable table where it was told to."""

    def test_writes_a_loadable_table(self, tmp_path: Path) -> None:
        """main must produce a file that load_sensor_spans accepts."""
        raw_dir = tmp_path / "raw"
        _write_normal_file(raw_dir)
        output = tmp_path / "nested" / "sensor_spans.yaml"

        written = main(["--raw-dir", str(raw_dir), "--output", str(output)])

        assert written == output
        assert len(load_sensor_spans(output)) == N_COLUMNS

    def test_records_the_margins_in_the_header(self, tmp_path: Path) -> None:
        """The header must state the margins used, for traceability."""
        raw_dir = tmp_path / "raw"
        _write_normal_file(raw_dir)
        output = tmp_path / "sensor_spans.yaml"

        main(
            [
                "--raw-dir",
                str(raw_dir),
                "--output",
                str(output),
                "--alarm-margin",
                "0.25",
                "--span-margin",
                "1.5",
            ]
        )

        header = output.read_text(encoding="utf-8")
        assert "25%" in header
        assert "150%" in header
        assert NORMAL_OPERATION_FILE in header

    def test_custom_margins_reach_the_entries(self, tmp_path: Path) -> None:
        """The CLI margins must be the ones applied, not the defaults."""
        raw_dir = tmp_path / "raw"
        _write_normal_file(raw_dir)
        output = tmp_path / "sensor_spans.yaml"

        main(
            [
                "--raw-dir",
                str(raw_dir),
                "--output",
                str(output),
                "--alarm-margin",
                "0.0",
                "--span-margin",
                "1.0",
            ]
        )

        span = load_sensor_spans(output)[sensor_id(0)]
        assert span.alarm_min == pytest.approx(0.0)
        assert span.alarm_max == pytest.approx(float(_N_ROWS - 1))
