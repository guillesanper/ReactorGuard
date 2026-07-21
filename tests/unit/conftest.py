"""Shared fixtures for the TEP pipeline unit tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.generators.tep_adapter import SENSOR_TYPE_MAP, UNIT_MAP, sensor_id
from data.generators.tep_loader import N_COLUMNS
from data.schemas.sensor_spans import SensorSpan

WriteParams = Callable[..., Path]

# Span sintetico deliberadamente ancho: cubre con holgura los valores que genera
# make_dat_dir, de modo que ningun test de adaptacion falle por recorte del ADC
# salvo que ese sea justo lo que comprueba.
_TEST_SPAN_LIMIT = 10_000.0
_TEST_ALARM_LIMIT = 5_000.0


@pytest.fixture()
def sensor_spans() -> dict[str, SensorSpan]:
    """Return a span table covering all 52 TEP tags with wide synthetic limits."""
    return {
        sensor_id(col): SensorSpan(
            sensor_id=sensor_id(col),
            min=-_TEST_SPAN_LIMIT,
            max=_TEST_SPAN_LIMIT,
            alarm_min=-_TEST_ALARM_LIMIT,
            alarm_max=_TEST_ALARM_LIMIT,
            unit=UNIT_MAP[SENSOR_TYPE_MAP[col]].value,
        )
        for col in range(N_COLUMNS)
    }


@pytest.fixture()
def spans_file(tmp_path: Path, sensor_spans: dict[str, SensorSpan]) -> Path:
    """Write the synthetic span table to disk and return its path."""
    document = {
        "sensors": {
            tag: {
                "min": span.min,
                "max": span.max,
                "alarm_min": span.alarm_min,
                "alarm_max": span.alarm_max,
                "unit": span.unit,
            }
            for tag, span in sensor_spans.items()
        }
    }
    path = tmp_path / "sensor_spans.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture()
def write_params(tmp_path: Path, spans_file: Path) -> WriteParams:
    """Return a factory that writes a valid params.yaml into tmp_path.

    The factory accepts keyword overrides for any key of the tep: section, so a
    test can vary one parameter without restating the whole block.
    """

    def _write(**overrides: Any) -> Path:
        section: dict[str, Any] = {
            "raw_dir": str(tmp_path / "raw"),
            "processed_dir": str(tmp_path / "processed"),
            "reports_dir": str(tmp_path / "reports"),
            "plant_id": "TEP-PLANT-01",
            "start_time": "2000-01-01T00:00:00+00:00",
            "sample_interval_minutes": 3,
            "spans_path": str(spans_file),
            "calibration_date": "2023-06-01",
            "last_maintenance_date": "2023-12-01",
            "drift_coefficient": 0.0001,
        }
        section.update(overrides)
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"tep": section}), encoding="utf-8")
        return path

    return _write


@pytest.fixture()
def make_dat_dir(tmp_path: Path) -> Callable[..., Path]:
    """Return a factory that builds a directory of synthetic TEP .dat files.

    d00.dat is written transposed, as the real dataset publishes it, so any test
    using this fixture exercises the orientation-normalisation path.
    """

    def _make(n_files: int = 3, n_rows: int = 4) -> Path:
        data_dir = tmp_path / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        names = ["d00.dat"] + [f"d{i:02d}.dat" for i in range(1, n_files)]
        for name in names:
            transposed = name == "d00.dat"
            rows, cols = (
                (N_COLUMNS, n_rows) if transposed else (n_rows, N_COLUMNS)
            )
            lines = [
                " ".join(f"{(r * cols + c) % 97 + 0.5:.4f}" for c in range(cols))
                for r in range(rows)
            ]
            (data_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

        return data_dir

    return _make
