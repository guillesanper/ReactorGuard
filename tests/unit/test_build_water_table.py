"""Unit tests for data.simulation.build_water_table.

El generador es una herramienta de desarrollo: ni el runtime ni los tests
dependen de `iapws`. Aqui se prueba con un backend analitico y con un modulo
`iapws` falso inyectado en sys.modules, y solo la comparacion de la tabla
versionada contra IAPWS real queda condicionada a que la libreria exista.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from data.simulation import build_water_table as bwt
from data.simulation.water_properties import load_water_table

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeBackend:
    """Backend analitico: campos lineales y saturacion lineal."""

    source = "fake-backend 1.0"

    @staticmethod
    def t_sat(p_bar: float) -> float:
        return 250.0 + 0.5 * (p_bar - 60.0)

    def liquid(self, t_c: float, p_bar: float) -> tuple[float, float, float]:
        return 4000.0 + 10.0 * t_c + 2.0 * p_bar, 1000.0 - 0.5 * t_c, 1.0e3 * t_c + 5.0 * p_bar

    def saturated_liquid(self, p_bar: float) -> tuple[float, float, float, float]:
        t_sat = self.t_sat(p_bar)
        return (t_sat, *self.liquid(t_sat, p_bar))


class TestMakeAxis:
    """Inclusive, evenly spaced axes."""

    def test_includes_both_ends(self) -> None:
        """200-360 C at step 2 has 81 nodes, 60-170 bar at step 5 has 23."""
        t_axis = bwt.make_axis(200.0, 360.0, 2.0)
        p_axis = bwt.make_axis(60.0, 170.0, 5.0)
        assert (len(t_axis), t_axis[0], t_axis[-1]) == (81, 200.0, 360.0)
        assert (len(p_axis), p_axis[0], p_axis[-1]) == (23, 60.0, 170.0)

    @pytest.mark.parametrize(
        ("start", "stop", "step"),
        [(0.0, 10.0, 0.0), (0.0, 10.0, -1.0), (10.0, 0.0, 1.0), (0.0, 10.0, 3.0)],
    )
    def test_invalid_axes_raise(self, start: float, stop: float, step: float) -> None:
        """Non-positive step, reversed range or a partial last step are errors."""
        with pytest.raises(ValueError):
            bwt.make_axis(start, stop, step)


class TestBuildTable:
    """The document must follow the backend and the saturation plateau rule."""

    def test_shapes(self) -> None:
        """Fields are (n_p, n_t)."""
        doc = bwt.build_table(_FakeBackend(), [200.0, 220.0, 240.0], [60.0, 80.0])
        assert np.asarray(doc["cp_j_kg_k"]).shape == (2, 3)
        assert doc["source"] == "fake-backend 1.0"

    def test_subcooled_nodes_use_the_liquid_value(self) -> None:
        """Below T_sat the node is the compressed-liquid property."""
        backend = _FakeBackend()
        doc = bwt.build_table(backend, [200.0, 240.0], [60.0])
        assert doc["cp_j_kg_k"][0][0] == pytest.approx(backend.liquid(200.0, 60.0)[0])
        assert doc["cp_j_kg_k"][0][1] == pytest.approx(backend.liquid(240.0, 60.0)[0])

    def test_nodes_at_or_above_saturation_hold_the_plateau(self) -> None:
        """T >= T_sat(P) stores the saturated liquid, including the equality case."""
        backend = _FakeBackend()
        t_sat = backend.t_sat(60.0)
        doc = bwt.build_table(backend, [t_sat - 10.0, t_sat, t_sat + 10.0], [60.0])
        saturated_cp = backend.saturated_liquid(60.0)[1]
        assert doc["cp_j_kg_k"][0][1] == pytest.approx(saturated_cp)
        assert doc["cp_j_kg_k"][0][2] == pytest.approx(saturated_cp)
        assert doc["cp_j_kg_k"][0][0] != pytest.approx(saturated_cp)
        assert doc["t_sat_c"] == [pytest.approx(t_sat)]

    def test_values_are_plain_floats(self) -> None:
        """YAML-ready: no numpy scalars."""
        doc = bwt.build_table(_FakeBackend(), [200.0], [60.0])
        assert type(doc["cp_j_kg_k"][0][0]) is float
        assert type(doc["t_sat_c"][0]) is float


class TestRenderAndRoundTrip:
    """The rendered text must load back and be deterministic."""

    def test_round_trip_through_the_loader(self, tmp_path: Path) -> None:
        """Render -> file -> load_water_table reproduces the document."""
        t_axis = bwt.make_axis(200.0, 260.0, 10.0)
        p_axis = bwt.make_axis(60.0, 100.0, 10.0)
        doc = bwt.build_table(_FakeBackend(), t_axis, p_axis)
        path = tmp_path / "table.yaml"
        path.write_text(bwt.render_table(doc), encoding="utf-8")
        table = load_water_table(path)
        assert table.source == "fake-backend 1.0"
        np.testing.assert_allclose(table.cp_j_kg_k, np.asarray(doc["cp_j_kg_k"]))
        np.testing.assert_allclose(table.t_sat_c_grid, np.asarray(doc["t_sat_c"]))

    def test_rendering_is_deterministic(self) -> None:
        """Two renders of the same document are byte-identical."""
        doc = bwt.build_table(_FakeBackend(), [200.0, 210.0], [60.0, 70.0])
        assert bwt.render_table(doc) == bwt.render_table(doc)

    def test_header_documents_the_grid_and_units(self) -> None:
        """The header names the source, grid and units, and has no clock."""
        text = bwt.render_table(bwt.build_table(_FakeBackend(), [200.0, 210.0], [60.0, 70.0]))
        assert "fake-backend 1.0" in text
        assert "J/(kg K)" in text
        assert "build_water_table" in text


class TestMain:
    """The command line entry point."""

    def test_writes_the_file(self, tmp_path: Path) -> None:
        """main writes a loadable table and returns 0."""
        out = tmp_path / "nested" / "water.yaml"
        code = bwt.main(
            ["--output", str(out), "--t-min", "200", "--t-max", "240", "--t-step", "10",
             "--p-min", "60", "--p-max", "80", "--p-step", "10"],
            backend=_FakeBackend(),
        )
        assert code == 0
        table = load_water_table(out)
        assert table.t_range_c == (200.0, 240.0)
        assert table.p_range_bar == (60.0, 80.0)

    def test_output_is_reproducible(self, tmp_path: Path) -> None:
        """Two runs produce the same bytes."""
        args = ["--t-min", "200", "--t-max", "220", "--t-step", "10",
                "--p-min", "60", "--p-max", "70", "--p-step", "10"]
        bwt.main([*args, "--output", str(tmp_path / "a.yaml")], backend=_FakeBackend())
        bwt.main([*args, "--output", str(tmp_path / "b.yaml")], backend=_FakeBackend())
        assert (tmp_path / "a.yaml").read_bytes() == (tmp_path / "b.yaml").read_bytes()

    def test_missing_iapws_returns_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without iapws the tool exits with code 2 and writes nothing."""
        monkeypatch.setitem(sys.modules, "iapws", None)
        out = tmp_path / "water.yaml"
        assert bwt.main(["--output", str(out)]) == 2
        assert not out.exists()


class TestIapwsBackend:
    """The adapter over iapws, exercised with a fake module."""

    @staticmethod
    def _install_fake(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        class _State:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)
                saturated = "x" in kwargs
                self.T = 600.0 if saturated else kwargs["T"]
                self.cp = np.float64(5.5)
                self.rho = np.float64(700.0)
                self.h = np.float64(1400.0)

        module = types.ModuleType("iapws")
        module.IAPWS97 = _State  # type: ignore[attr-defined]
        module.__version__ = "9.9.9"  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "iapws", module)
        return calls

    def test_unit_conversions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """C -> K, bar -> MPa, kJ -> J on the way in and out."""
        calls = self._install_fake(monkeypatch)
        backend = bwt.IapwsBackend()
        cp, rho, h = backend.liquid(300.0, 155.0)
        assert (cp, rho, h) == (5500.0, 700.0, 1_400_000.0)
        assert type(cp) is float
        assert calls[0]["T"] == pytest.approx(573.15)
        assert calls[0]["P"] == pytest.approx(15.5)

    def test_saturated_liquid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saturated liquid is requested with x = 0 and T is returned in C."""
        calls = self._install_fake(monkeypatch)
        t_sat, cp, rho, h = bwt.IapwsBackend().saturated_liquid(120.0)
        assert calls[0] == {"P": pytest.approx(12.0), "x": 0}
        assert t_sat == pytest.approx(600.0 - 273.15)
        assert (cp, rho, h) == (5500.0, 700.0, 1_400_000.0)

    def test_source_records_the_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The provenance string carries the library version."""
        self._install_fake(monkeypatch)
        assert bwt.IapwsBackend().source == "IAPWS-IF97 via iapws 9.9.9"

    def test_missing_library_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing dependency is explained, not a bare ImportError."""
        monkeypatch.setitem(sys.modules, "iapws", None)
        with pytest.raises(RuntimeError, match="dev extra"):
            bwt.IapwsBackend()


class TestCommittedTableAgainstIapws:
    """Contrasta configs/water_table.yaml con IAPWS real (solo si hay iapws)."""

    def test_nodes_match_iapws(self) -> None:
        """Sampled nodes of the versioned table equal IAPWS-IF97 to 1e-6 relative."""
        iapws = pytest.importorskip("iapws")
        table = load_water_table(_REPO_ROOT / "configs" / "water_table.yaml")
        rng = np.random.default_rng(7)
        for _ in range(25):
            i_t = int(rng.integers(0, table.t_c.size))
            i_p = int(rng.integers(0, table.p_bar.size))
            t_c, p_bar = float(table.t_c[i_t]), float(table.p_bar[i_p])
            if t_c >= table.t_sat_c_grid[i_p]:
                continue
            state = iapws.IAPWS97(T=t_c + 273.15, P=p_bar * 0.1)
            assert table.cp_j_kg_k[i_p, i_t] == pytest.approx(state.cp * 1000.0, rel=1e-6)
            assert table.rho_kg_m3[i_p, i_t] == pytest.approx(state.rho, rel=1e-6)
