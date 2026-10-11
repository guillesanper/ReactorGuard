"""Tabla versionada de propiedades del agua del primario (liquido comprimido).

El runtime de la simulacion NO depende de `iapws`: lee la tabla que
`data/simulation/build_water_table.py` genero con IAPWS-IF97 y que se commitea
en `configs/water_table.yaml`. Asi el resultado depende solo de ficheros
versionados (determinismo) y `simulate` no arrastra una dependencia nueva.

Convenciones:
    - Unidades SI salvo T en grados Celsius y P en bar: cp en J/(kg K), rho en
      kg/m3, h en J/kg.
    - Rejilla rectangular en (P, T). Las propiedades se interpolan de forma
      bilineal; la saturacion se interpola en 1-D sobre P.
    - Fuera de la rejilla se lanza ValueError: la tabla NUNCA extrapola.
    - Por encima de la saturacion (T >= T_sat(P)) la tabla guarda el valor del
      liquido saturado (meseta). El modelo detecta la ebullicion con
      `t_sat_c`, no con estas propiedades; la meseta solo evita huecos
      numericos durante una despresurizacion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload

import numpy as np
import yaml
from numpy.typing import NDArray

DEFAULT_WATER_TABLE_PATH = Path("configs/water_table.yaml")

_Array = NDArray[np.float64]

_FIELDS = ("cp_j_kg_k", "rho_kg_m3", "h_j_kg")


@dataclass(frozen=True, eq=False)
class WaterTable:
    """Propiedades del agua en rejilla (P, T) con interpolacion bilineal.

    Attributes:
        t_c: Eje de temperatura [C], estrictamente creciente, forma (nt,).
        p_bar: Eje de presion [bar], estrictamente creciente, forma (np,).
        t_sat_c_grid: Temperatura de saturacion [C] en cada p_bar, forma (np,).
        cp_j_kg_k: Calor especifico [J/(kg K)], forma (np, nt).
        rho_kg_m3: Densidad [kg/m3], forma (np, nt).
        h_j_kg: Entalpia [J/kg], forma (np, nt).
        source: Descripcion de la procedencia (libreria y version).
    """

    t_c: NDArray[np.float64]
    p_bar: NDArray[np.float64]
    t_sat_c_grid: NDArray[np.float64]
    cp_j_kg_k: NDArray[np.float64]
    rho_kg_m3: NDArray[np.float64]
    h_j_kg: NDArray[np.float64]
    source: str = ""

    def __post_init__(self) -> None:
        """Validate axes and field shapes.

        Raises:
            ValueError: If an axis is not 1-D, strictly increasing and with at
                least two nodes, or a field does not match (np, nt).
        """
        for name, axis in (("t_c", self.t_c), ("p_bar", self.p_bar)):
            if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0.0):
                raise ValueError(f"water table: axis '{name}' must be 1-D, increasing, >= 2 nodes.")
        if self.t_sat_c_grid.shape != self.p_bar.shape:
            raise ValueError("water table: t_sat_c_grid must match p_bar.")
        shape = (self.p_bar.size, self.t_c.size)
        for name in _FIELDS:
            if getattr(self, name).shape != shape:
                raise ValueError(f"water table: field '{name}' must have shape {shape}.")

    @property
    def t_range_c(self) -> tuple[float, float]:
        """Return the (min, max) temperature covered by the grid [C]."""
        return float(self.t_c[0]), float(self.t_c[-1])

    @property
    def p_range_bar(self) -> tuple[float, float]:
        """Return the (min, max) pressure covered by the grid [bar]."""
        return float(self.p_bar[0]), float(self.p_bar[-1])

    def _check(self, name: str, value: NDArray[np.float64], axis: NDArray[np.float64]) -> None:
        """Raise ValueError if any value is NaN or outside the axis range."""
        if not np.all((value >= axis[0]) & (value <= axis[-1])):
            raise ValueError(
                f"water table: {name} outside the grid [{axis[0]:g}, {axis[-1]:g}] "
                "(the table never extrapolates)."
            )

    def _locate(
        self, axis: NDArray[np.float64], value: NDArray[np.float64]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """Return the lower node index and the fractional position in the cell."""
        idx = np.clip(np.searchsorted(axis, value, side="right") - 1, 0, axis.size - 2)
        frac = (value - axis[idx]) / (axis[idx + 1] - axis[idx])
        return idx, frac

    def _bilinear(
        self, field: NDArray[np.float64], t_c: Any, p_bar: Any
    ) -> NDArray[np.float64]:
        """Interpolate a field at (t_c, p_bar), broadcasting both inputs."""
        t, p = np.broadcast_arrays(
            np.asarray(t_c, dtype=np.float64), np.asarray(p_bar, dtype=np.float64)
        )
        self._check("temperature [C]", t, self.t_c)
        self._check("pressure [bar]", p, self.p_bar)
        it, ft = self._locate(self.t_c, t)
        ip, fp = self._locate(self.p_bar, p)
        low = field[ip, it] * (1.0 - ft) + field[ip, it + 1] * ft
        high = field[ip + 1, it] * (1.0 - ft) + field[ip + 1, it + 1] * ft
        result: NDArray[np.float64] = low * (1.0 - fp) + high * fp
        return result

    @overload
    def cp(self, t_c: float, p_bar: float) -> float: ...
    @overload
    def cp(self, t_c: _Array, p_bar: _Array | float) -> _Array: ...
    def cp(self, t_c: Any, p_bar: Any) -> Any:
        """Return the specific heat [J/(kg K)].

        Args:
            t_c: Temperature [C], scalar or array.
            p_bar: Pressure [bar], scalar or array (broadcast with t_c).

        Returns:
            cp with the shape of the broadcast inputs; float for scalars.

        Raises:
            ValueError: If any point is outside the grid or NaN.
        """
        return self._finish(self._bilinear(self.cp_j_kg_k, t_c, p_bar), t_c, p_bar)

    @overload
    def rho(self, t_c: float, p_bar: float) -> float: ...
    @overload
    def rho(self, t_c: _Array, p_bar: _Array | float) -> _Array: ...
    def rho(self, t_c: Any, p_bar: Any) -> Any:
        """Return the density [kg/m3].

        Args:
            t_c: Temperature [C], scalar or array.
            p_bar: Pressure [bar], scalar or array (broadcast with t_c).

        Returns:
            rho with the shape of the broadcast inputs; float for scalars.

        Raises:
            ValueError: If any point is outside the grid or NaN.
        """
        return self._finish(self._bilinear(self.rho_kg_m3, t_c, p_bar), t_c, p_bar)

    @overload
    def h(self, t_c: float, p_bar: float) -> float: ...
    @overload
    def h(self, t_c: _Array, p_bar: _Array | float) -> _Array: ...
    def h(self, t_c: Any, p_bar: Any) -> Any:
        """Return the specific enthalpy [J/kg].

        Args:
            t_c: Temperature [C], scalar or array.
            p_bar: Pressure [bar], scalar or array (broadcast with t_c).

        Returns:
            h with the shape of the broadcast inputs; float for scalars.

        Raises:
            ValueError: If any point is outside the grid or NaN.
        """
        return self._finish(self._bilinear(self.h_j_kg, t_c, p_bar), t_c, p_bar)

    @overload
    def t_sat_c(self, p_bar: float) -> float: ...
    @overload
    def t_sat_c(self, p_bar: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def t_sat_c(self, p_bar: Any) -> Any:
        """Return the saturation temperature [C] at a pressure.

        Args:
            p_bar: Pressure [bar], scalar or array.

        Returns:
            T_sat with the shape of p_bar; float for scalars.

        Raises:
            ValueError: If any pressure is outside the grid or NaN.
        """
        p = np.asarray(p_bar, dtype=np.float64)
        self._check("pressure [bar]", p, self.p_bar)
        result = np.interp(p, self.p_bar, self.t_sat_c_grid)
        return float(result) if np.ndim(p_bar) == 0 else result

    @overload
    def p_sat_bar(self, t_c: float) -> float: ...
    @overload
    def p_sat_bar(self, t_c: NDArray[np.float64]) -> NDArray[np.float64]: ...
    def p_sat_bar(self, t_c: Any) -> Any:
        """Return the saturation pressure [bar] at a temperature (inverse of t_sat_c).

        Args:
            t_c: Temperature [C], scalar or array.

        Returns:
            P_sat with the shape of t_c; float for scalars.

        Raises:
            ValueError: If any temperature is outside the saturation curve held
                by the table or NaN.
        """
        t = np.asarray(t_c, dtype=np.float64)
        self._check("saturation temperature [C]", t, self.t_sat_c_grid)
        result = np.interp(t, self.t_sat_c_grid, self.p_bar)
        return float(result) if np.ndim(t_c) == 0 else result

    @staticmethod
    def _finish(value: NDArray[np.float64], t_c: Any, p_bar: Any) -> Any:
        """Return a float when both inputs were scalars, the array otherwise."""
        if np.ndim(t_c) == 0 and np.ndim(p_bar) == 0:
            return float(value)
        return value


def _read_axis(document: dict[str, Any], key: str, path: Path) -> NDArray[np.float64]:
    """Return a required numeric list of the document as a float64 array."""
    if key not in document:
        raise KeyError(f"{path}: missing required key '{key}'.")
    return np.asarray(document[key], dtype=np.float64)


def load_water_table(path: str | Path = DEFAULT_WATER_TABLE_PATH) -> WaterTable:
    """Load the versioned water table.

    Args:
        path: Path to the YAML written by build_water_table.py.

    Returns:
        The WaterTable.

    Raises:
        FileNotFoundError: If path does not exist.
        KeyError: If a required key is missing.
        ValueError: If axes or fields are inconsistent.
    """
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Water table not found: {table_path}. Generate it with "
            "python -m data.simulation.build_water_table."
        )
    with table_path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    fields = {name: _read_axis(document, name, table_path) for name in _FIELDS}
    return WaterTable(
        t_c=_read_axis(document, "t_c", table_path),
        p_bar=_read_axis(document, "p_bar", table_path),
        t_sat_c_grid=_read_axis(document, "t_sat_c", table_path),
        cp_j_kg_k=fields["cp_j_kg_k"],
        rho_kg_m3=fields["rho_kg_m3"],
        h_j_kg=fields["h_j_kg"],
        source=str(document.get("source", "")),
    )
