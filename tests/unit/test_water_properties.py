"""Unit tests for data.simulation.water_properties.

La tabla es la unica fuente de propiedades del agua del modelo, asi que sus
tests cubren tres cosas: que la interpolacion bilineal sea exacta donde debe
(campos lineales), que fuera de la rejilla falle en vez de extrapolar, y que la
tabla versionada tenga las cifras de IAPWS-IF97 que el preflight midio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from data.simulation.water_properties import (
    DEFAULT_WATER_TABLE_PATH,
    WaterTable,
    load_water_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_T_AXIS = [200.0, 210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0]
_P_AXIS = [60.0, 70.0, 80.0, 90.0, 100.0]


def _cp(t: float, p: float) -> float:
    return 4000.0 + 10.0 * t + 2.0 * p


def _rho(t: float, p: float) -> float:
    return 1000.0 - 0.5 * t + 0.1 * p


def _h(t: float, p: float) -> float:
    return 1.0e3 * t + 5.0 * p


def _t_sat(p: float) -> float:
    return 250.0 + 0.5 * (p - 60.0)


def _document() -> dict[str, Any]:
    """Return a table document whose fields are linear in (T, P)."""
    return {
        "source": "synthetic",
        "t_c": list(_T_AXIS),
        "p_bar": list(_P_AXIS),
        "t_sat_c": [_t_sat(p) for p in _P_AXIS],
        "cp_j_kg_k": [[_cp(t, p) for t in _T_AXIS] for p in _P_AXIS],
        "rho_kg_m3": [[_rho(t, p) for t in _T_AXIS] for p in _P_AXIS],
        "h_j_kg": [[_h(t, p) for t in _T_AXIS] for p in _P_AXIS],
    }


def _write(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


@pytest.fixture()
def table(tmp_path: Path) -> WaterTable:
    """Return a table loaded from a synthetic, linear document."""
    return load_water_table(_write(tmp_path / "table.yaml", _document()))


@pytest.fixture(scope="module")
def committed() -> WaterTable:
    """Return the versioned configs/water_table.yaml."""
    return load_water_table(_REPO_ROOT / DEFAULT_WATER_TABLE_PATH)


class TestInterpolation:
    """Bilinear interpolation must be exact for fields linear in (T, P)."""

    def test_exact_at_nodes(self, table: WaterTable) -> None:
        """A node returns its stored value."""
        assert table.cp(230.0, 80.0) == pytest.approx(_cp(230.0, 80.0))
        assert table.rho(230.0, 80.0) == pytest.approx(_rho(230.0, 80.0))
        assert table.h(230.0, 80.0) == pytest.approx(_h(230.0, 80.0))

    def test_exact_inside_cells_for_linear_fields(self, table: WaterTable) -> None:
        """Between nodes a linear field is reproduced without error."""
        for t, p in ((203.7, 61.3), (255.5, 77.7), (299.9, 99.9), (214.2, 84.1)):
            assert table.cp(t, p) == pytest.approx(_cp(t, p), rel=1e-12)
            assert table.rho(t, p) == pytest.approx(_rho(t, p), rel=1e-12)
            assert table.h(t, p) == pytest.approx(_h(t, p), rel=1e-12)

    def test_grid_edges_are_valid(self, table: WaterTable) -> None:
        """The last node of each axis belongs to the grid."""
        assert table.cp(300.0, 100.0) == pytest.approx(_cp(300.0, 100.0))
        assert table.cp(200.0, 60.0) == pytest.approx(_cp(200.0, 60.0))

    def test_scalar_inputs_return_float(self, table: WaterTable) -> None:
        """Scalar inputs return a Python float, not a 0-d array."""
        assert isinstance(table.cp(250.0, 80.0), float)
        assert isinstance(table.t_sat_c(80.0), float)
        assert isinstance(table.p_sat_bar(255.0), float)

    def test_array_inputs_return_arrays(self, table: WaterTable) -> None:
        """Array inputs keep their shape."""
        t = np.array([210.0, 250.0, 290.0])
        out = table.cp(t, 80.0)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        np.testing.assert_allclose(out, [_cp(x, 80.0) for x in t])

    def test_broadcasts_temperature_and_pressure(self, table: WaterTable) -> None:
        """A column of pressures against a row of temperatures broadcasts."""
        t = np.array([[210.0, 250.0]])
        p = np.array([[65.0], [95.0]])
        out = table.rho(t, p)
        assert out.shape == (2, 2)
        assert out[1, 0] == pytest.approx(_rho(210.0, 95.0))


class TestSaturation:
    """T_sat(P) and its inverse."""

    def test_t_sat_interpolates_linearly(self, table: WaterTable) -> None:
        """T_sat between nodes follows the linear curve of the fixture."""
        assert table.t_sat_c(75.0) == pytest.approx(_t_sat(75.0))
        out = table.t_sat_c(np.array([61.0, 99.0]))
        np.testing.assert_allclose(out, [_t_sat(61.0), _t_sat(99.0)])

    def test_p_sat_is_the_inverse(self, table: WaterTable) -> None:
        """p_sat_bar undoes t_sat_c on the interior of the curve."""
        for p in (61.0, 75.0, 99.0):
            assert table.p_sat_bar(table.t_sat_c(p)) == pytest.approx(p)

    def test_p_sat_accepts_arrays(self, table: WaterTable) -> None:
        """p_sat_bar is vectorised."""
        out = table.p_sat_bar(np.array([251.0, 270.0]))
        assert out.shape == (2,)

    def test_ranges(self, table: WaterTable) -> None:
        """The covered ranges are reported."""
        assert table.t_range_c == (200.0, 300.0)
        assert table.p_range_bar == (60.0, 100.0)


class TestNeverExtrapolates:
    """Outside the grid the table raises ValueError."""

    @pytest.mark.parametrize(
        ("t", "p"),
        [(199.9, 80.0), (300.1, 80.0), (250.0, 59.9), (250.0, 100.1)],
    )
    def test_out_of_grid_raises(self, table: WaterTable, t: float, p: float) -> None:
        """Any coordinate beyond an axis end raises, for every property."""
        with pytest.raises(ValueError, match="outside the grid"):
            table.cp(t, p)
        with pytest.raises(ValueError, match="outside the grid"):
            table.rho(t, p)
        with pytest.raises(ValueError, match="outside the grid"):
            table.h(t, p)

    def test_nan_raises(self, table: WaterTable) -> None:
        """NaN is not inside the grid."""
        with pytest.raises(ValueError, match="outside the grid"):
            table.cp(float("nan"), 80.0)

    def test_one_bad_element_of_an_array_raises(self, table: WaterTable) -> None:
        """A single out-of-range element invalidates the whole call."""
        with pytest.raises(ValueError, match="outside the grid"):
            table.cp(np.array([250.0, 400.0]), 80.0)

    def test_saturation_out_of_range_raises(self, table: WaterTable) -> None:
        """T_sat and its inverse refuse pressures / temperatures off the curve."""
        with pytest.raises(ValueError, match="outside the grid"):
            table.t_sat_c(120.0)
        with pytest.raises(ValueError, match="outside the grid"):
            table.p_sat_bar(300.0)


class TestValidation:
    """Malformed tables must fail at load time."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing file names the generator."""
        with pytest.raises(FileNotFoundError, match="build_water_table"):
            load_water_table(tmp_path / "absent.yaml")

    def test_missing_key(self, tmp_path: Path) -> None:
        """A missing field names the key."""
        document = _document()
        del document["rho_kg_m3"]
        with pytest.raises(KeyError, match="rho_kg_m3"):
            load_water_table(_write(tmp_path / "t.yaml", document))

    def test_non_increasing_axis(self, tmp_path: Path) -> None:
        """Axes must be strictly increasing."""
        document = _document()
        document["t_c"] = [200.0, 200.0] + _T_AXIS[2:]
        with pytest.raises(ValueError, match="increasing"):
            load_water_table(_write(tmp_path / "t.yaml", document))

    def test_wrong_field_shape(self, tmp_path: Path) -> None:
        """A field must match (n_p, n_t)."""
        document = _document()
        document["h_j_kg"] = document["h_j_kg"][:-1]
        with pytest.raises(ValueError, match="h_j_kg"):
            load_water_table(_write(tmp_path / "t.yaml", document))

    def test_wrong_t_sat_length(self, tmp_path: Path) -> None:
        """t_sat_c must have one value per pressure node."""
        document = _document()
        document["t_sat_c"] = document["t_sat_c"][:-1]
        with pytest.raises(ValueError, match="t_sat_c_grid"):
            load_water_table(_write(tmp_path / "t.yaml", document))

    def test_source_defaults_to_empty(self, tmp_path: Path) -> None:
        """The provenance string is optional on load."""
        document = _document()
        del document["source"]
        assert load_water_table(_write(tmp_path / "t.yaml", document)).source == ""


class TestCommittedTable:
    """configs/water_table.yaml must carry the IAPWS-IF97 figures of the preflight."""

    def test_covers_the_planned_grid(self, committed: WaterTable) -> None:
        """T 200-360 C and P 60-170 bar."""
        assert committed.t_range_c == (200.0, 360.0)
        assert committed.p_range_bar == (60.0, 170.0)
        assert committed.cp_j_kg_k.shape == (23, 81)

    def test_provenance_names_the_library(self, committed: WaterTable) -> None:
        """The source records IAPWS and the library version."""
        assert "IAPWS" in committed.source
        assert "iapws" in committed.source

    def test_cp_matches_preflight_figures(self, committed: WaterTable) -> None:
        """cp at 15,5 MPa: 5243 / 5664 / 6415 J/(kg K) at 290 / 307,5 / 325 C."""
        assert committed.cp(290.0, 155.0) == pytest.approx(5243.0, rel=5e-3)
        assert committed.cp(307.5, 155.0) == pytest.approx(5664.0, rel=5e-3)
        assert committed.cp(325.0, 155.0) == pytest.approx(6415.0, rel=5e-3)

    def test_saturation_matches_preflight_figures(self, committed: WaterTable) -> None:
        """T_sat(15,5 MPa) = 344,79 C and T_sat(12 MPa) = 324,7 C."""
        assert committed.t_sat_c(155.0) == pytest.approx(344.79, abs=0.05)
        assert committed.t_sat_c(120.0) == pytest.approx(324.68, abs=0.05)

    def test_saturation_is_increasing(self, committed: WaterTable) -> None:
        """T_sat must grow with pressure."""
        assert np.all(np.diff(committed.t_sat_c_grid) > 0.0)

    def test_density_decreases_with_temperature(self, committed: WaterTable) -> None:
        """Compressed liquid gets lighter as it heats, at fixed pressure."""
        row = committed.rho_kg_m3[list(committed.p_bar).index(155.0)]
        subcooled = committed.t_c < committed.t_sat_c(155.0) - 2.0
        assert np.all(np.diff(row[subcooled]) < 0.0)

    def test_plateau_above_saturation(self, committed: WaterTable) -> None:
        """Nodes at or above T_sat hold the saturated-liquid value (no steam)."""
        i_p = list(committed.p_bar).index(100.0)
        above = committed.t_c >= committed.t_sat_c(100.0)
        values = committed.cp_j_kg_k[i_p, above]
        assert above.sum() > 2
        assert np.all(values == values[0])

    def test_inverse_saturation_at_nominal_secondary(self, committed: WaterTable) -> None:
        """The 280 C secondary sits at ~64 bar (IAPWS: 64,17 bar)."""
        assert committed.p_sat_bar(280.0) == pytest.approx(64.17, abs=0.1)
