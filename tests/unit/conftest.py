"""Shared fixtures for the TEP pipeline unit tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.generators.tep_loader import N_COLUMNS

WriteParams = Callable[..., Path]


@pytest.fixture()
def write_params(tmp_path: Path) -> WriteParams:
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
            "adc_scale_max": 3000.0,
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
