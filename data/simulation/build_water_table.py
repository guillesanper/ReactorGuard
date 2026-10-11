"""Genera configs/water_table.yaml con IAPWS-IF97 (herramienta de desarrollo).

`iapws` es dependencia de DESARROLLO (extra `dev`): solo esta herramienta la
importa, y de forma perezosa, de modo que ni el runtime de la simulacion ni los
tests dependen de ella. La tabla resultante se commitea y es la unica fuente de
propiedades del agua del modelo (data/simulation/water_properties.py).

Uso:
    python -m data.simulation.build_water_table --output configs/water_table.yaml

Contenido de la tabla: cp, rho y h del liquido comprimido en la rejilla
T 200-360 C (paso 2) x P 60-170 bar (paso 5) y T_sat(P) por presion. En los
nodos con T >= T_sat(P) se guarda el liquido saturado (meseta), ver
water_properties.py.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

_LOG = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("configs/water_table.yaml")
DEFAULT_T_MIN_C = 200.0
DEFAULT_T_MAX_C = 360.0
DEFAULT_T_STEP_C = 2.0
DEFAULT_P_MIN_BAR = 60.0
DEFAULT_P_MAX_BAR = 170.0
DEFAULT_P_STEP_BAR = 5.0

_DECIMALS = 6
_MPA_PER_BAR = 0.1
_KELVIN = 273.15
_J_PER_KJ = 1000.0


class WaterBackend(Protocol):
    """Fuente de propiedades del agua usada para construir la tabla."""

    @property
    def source(self) -> str:
        """Return a human readable provenance string (library and version)."""
        ...

    def liquid(self, t_c: float, p_bar: float) -> tuple[float, float, float]:
        """Return (cp [J/(kg K)], rho [kg/m3], h [J/kg]) of the liquid at (T, P)."""
        ...

    def saturated_liquid(self, p_bar: float) -> tuple[float, float, float, float]:
        """Return (T_sat [C], cp, rho, h) of the saturated liquid at a pressure."""
        ...


class IapwsBackend:
    """Backend IAPWS-IF97 sobre la libreria `iapws` (importada de forma perezosa)."""

    def __init__(self) -> None:
        """Import iapws.

        Raises:
            RuntimeError: If the iapws package is not installed.
        """
        try:
            import iapws  # perezoso a proposito: dependencia de desarrollo
        except ImportError as exc:
            raise RuntimeError(
                "iapws is required to build the water table; install the dev extra "
                "(pip install -e .[dev])."
            ) from exc
        self._iapws97: Any = iapws.IAPWS97
        self._version = str(iapws.__version__)

    @property
    def source(self) -> str:
        """Return the library name and version."""
        return f"IAPWS-IF97 via iapws {self._version}"

    def liquid(self, t_c: float, p_bar: float) -> tuple[float, float, float]:
        """Return (cp, rho, h) of the compressed liquid in SI units."""
        state = self._iapws97(T=t_c + _KELVIN, P=p_bar * _MPA_PER_BAR)
        return float(state.cp) * _J_PER_KJ, float(state.rho), float(state.h) * _J_PER_KJ

    def saturated_liquid(self, p_bar: float) -> tuple[float, float, float, float]:
        """Return (T_sat, cp, rho, h) of the saturated liquid in SI units."""
        state = self._iapws97(P=p_bar * _MPA_PER_BAR, x=0)
        return (
            float(state.T) - _KELVIN,
            float(state.cp) * _J_PER_KJ,
            float(state.rho),
            float(state.h) * _J_PER_KJ,
        )


def make_axis(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive, evenly spaced axis.

    Args:
        start: First node.
        stop: Last node (included).
        step: Positive spacing.

    Returns:
        The list of nodes.

    Raises:
        ValueError: If step is not positive, stop < start or the span is not a
            whole number of steps.
    """
    if step <= 0.0 or stop < start:
        raise ValueError(f"invalid axis: start={start}, stop={stop}, step={step}.")
    count = (stop - start) / step
    if abs(count - round(count)) > 1e-9:
        raise ValueError(f"axis span {stop - start} is not a multiple of step {step}.")
    return [float(start + i * step) for i in range(int(round(count)) + 1)]


def build_table(
    backend: WaterBackend, t_axis: Sequence[float], p_axis: Sequence[float]
) -> dict[str, Any]:
    """Build the table document for a backend and a grid.

    Args:
        backend: Source of water properties.
        t_axis: Temperature nodes [C].
        p_axis: Pressure nodes [bar].

    Returns:
        A YAML-ready mapping with the axes, T_sat and the three fields
        (rows = pressure, columns = temperature).
    """
    t_sat: list[float] = []
    cp_rows: list[list[float]] = []
    rho_rows: list[list[float]] = []
    h_rows: list[list[float]] = []
    for p_bar in p_axis:
        t_sat_c, cp_sat, rho_sat, h_sat = backend.saturated_liquid(p_bar)
        t_sat.append(round(t_sat_c, _DECIMALS))
        cp_row: list[float] = []
        rho_row: list[float] = []
        h_row: list[float] = []
        for t_c in t_axis:
            if t_c >= t_sat_c:
                cp, rho, h = cp_sat, rho_sat, h_sat
            else:
                cp, rho, h = backend.liquid(t_c, p_bar)
            cp_row.append(round(cp, _DECIMALS))
            rho_row.append(round(rho, _DECIMALS))
            h_row.append(round(h, _DECIMALS))
        cp_rows.append(cp_row)
        rho_rows.append(rho_row)
        h_rows.append(h_row)
    return {
        "source": backend.source,
        "t_c": [float(t) for t in t_axis],
        "p_bar": [float(p) for p in p_axis],
        "t_sat_c": t_sat,
        "cp_j_kg_k": cp_rows,
        "rho_kg_m3": rho_rows,
        "h_j_kg": h_rows,
    }


def render_table(document: dict[str, Any]) -> str:
    """Render the table document as the text of configs/water_table.yaml.

    Args:
        document: Mapping produced by build_table.

    Returns:
        YAML text with a descriptive header; deterministic (no clock, no paths).
    """
    t_axis = document["t_c"]
    p_axis = document["p_bar"]
    header = (
        "# ==========================================================================\n"
        "# ReactorGuard - Tabla de propiedades del agua (liquido comprimido)\n"
        "#\n"
        "# GENERADO por data/simulation/build_water_table.py. Se commitea; el runtime\n"
        "# de la simulacion lee este fichero y NO importa iapws.\n"
        "#\n"
        f"#   Fuente   : {document['source']}\n"
        f"#   Rejilla  : T {t_axis[0]:g}-{t_axis[-1]:g} C ({len(t_axis)} nodos) x "
        f"P {p_axis[0]:g}-{p_axis[-1]:g} bar ({len(p_axis)} nodos)\n"
        "#   Unidades : cp [J/(kg K)], rho [kg/m3], h [J/kg], t_sat_c [C]\n"
        "#   Forma    : filas = presion (p_bar), columnas = temperatura (t_c)\n"
        "#   Meseta   : en los nodos con T >= T_sat(P) se guarda el liquido saturado\n"
        "#\n"
        "# No editar a mano: regenerar con\n"
        "#   python -m data.simulation.build_water_table --output configs/water_table.yaml\n"
        "# ==========================================================================\n\n"
    )
    body = yaml.safe_dump(document, sort_keys=False, default_flow_style=None, width=100)
    return header + body


def main(argv: Sequence[str] | None = None, backend: WaterBackend | None = None) -> int:
    """Generate the water table file.

    Args:
        argv: Command line arguments; sys.argv[1:] when None.
        backend: Property source; IapwsBackend when None (tests inject a fake).

    Returns:
        Process exit code: 0 on success, 2 when iapws is not available.
    """
    parser = argparse.ArgumentParser(description="Generate configs/water_table.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--t-min", type=float, default=DEFAULT_T_MIN_C)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX_C)
    parser.add_argument("--t-step", type=float, default=DEFAULT_T_STEP_C)
    parser.add_argument("--p-min", type=float, default=DEFAULT_P_MIN_BAR)
    parser.add_argument("--p-max", type=float, default=DEFAULT_P_MAX_BAR)
    parser.add_argument("--p-step", type=float, default=DEFAULT_P_STEP_BAR)
    args = parser.parse_args(argv)

    if backend is None:
        try:
            backend = IapwsBackend()
        except RuntimeError as exc:
            _LOG.error("%s", exc)
            return 2

    t_axis = make_axis(args.t_min, args.t_max, args.t_step)
    p_axis = make_axis(args.p_min, args.p_max, args.p_step)
    document = build_table(backend, t_axis, p_axis)
    text = render_table(document)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    _LOG.info(
        "water table written: %s (%d x %d nodes, %s)",
        args.output,
        len(p_axis),
        len(t_axis),
        backend.source,
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
