"""Unit tests for data.generators.tep_loader.

Cubre la normalizacion de orientacion, que es el motivo por el que existe el
modulo: en el repositorio de Prof. Braatz, d00.dat se publica como (52, 500)
mientras que d01..d21 se publican como (480, 52). Cargar d00 sin transponerlo
hace fallar la validacion de 52 variables justo en el fichero de operacion
normal, que es la clase negativa de la evaluacion del validador.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from data.generators.tep_loader import N_COLUMNS, column_names, load_dat_file


def _write_matrix(path: Path, rows: int, cols: int) -> None:
    """Write a whitespace-delimited numeric matrix with a deterministic pattern."""
    lines = [
        " ".join(f"{r * cols + c:.4f}" for c in range(cols)) for r in range(rows)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestColumnNames:
    """column_names must describe the 52 TEP variables."""

    def test_length(self) -> None:
        """There must be exactly 52 names."""
        assert len(column_names()) == N_COLUMNS

    def test_zero_padded_convention(self) -> None:
        """Names must be zero-padded col_NN, first and last inclusive."""
        names = column_names()
        assert names[0] == "col_00"
        assert names[51] == "col_51"


class TestStandardOrientation:
    """Files already stored as (n_samples, 52) must pass through unchanged."""

    def test_shape_preserved(self, tmp_path: Path) -> None:
        """A (480, 52) file must load as (480, 52)."""
        path = tmp_path / "d01.dat"
        _write_matrix(path, rows=480, cols=N_COLUMNS)
        assert load_dat_file(path).shape == (480, N_COLUMNS)

    def test_columns_are_named(self, tmp_path: Path) -> None:
        """Loaded frames must carry the canonical column names."""
        path = tmp_path / "d01.dat"
        _write_matrix(path, rows=10, cols=N_COLUMNS)
        assert list(load_dat_file(path).columns) == column_names()

    def test_values_are_not_reordered(self, tmp_path: Path) -> None:
        """A non-transposed file must keep its original cell layout."""
        path = tmp_path / "d01.dat"
        _write_matrix(path, rows=3, cols=N_COLUMNS)
        df = load_dat_file(path)
        assert df.iloc[0, 0] == pytest.approx(0.0)
        assert df.iloc[1, 0] == pytest.approx(float(N_COLUMNS))


class TestTransposedOrientation:
    """Files stored as (52, n_samples) must be normalised to (n_samples, 52)."""

    def test_shape_is_normalised(self, tmp_path: Path) -> None:
        """A (52, 500) file must load as (500, 52), as d00.dat requires."""
        path = tmp_path / "d00.dat"
        _write_matrix(path, rows=N_COLUMNS, cols=500)
        assert load_dat_file(path).shape == (500, N_COLUMNS)

    def test_values_are_transposed_consistently(self, tmp_path: Path) -> None:
        """Cell (r, c) of the stored matrix must land at (c, r) after loading."""
        path = tmp_path / "d00.dat"
        _write_matrix(path, rows=N_COLUMNS, cols=5)
        df = load_dat_file(path)
        # Stored value at row 1, col 0 is 1*5+0 = 5; it must appear at (0, 1).
        assert df.iloc[0, 1] == pytest.approx(5.0)

    def test_index_is_reset(self, tmp_path: Path) -> None:
        """The transposed frame must expose a contiguous 0..n-1 index."""
        path = tmp_path / "d00.dat"
        _write_matrix(path, rows=N_COLUMNS, cols=7)
        df = load_dat_file(path)
        assert list(df.index) == list(range(7))

    def test_columns_are_named(self, tmp_path: Path) -> None:
        """Transposed frames must also carry the canonical column names."""
        path = tmp_path / "d00.dat"
        _write_matrix(path, rows=N_COLUMNS, cols=7)
        assert list(load_dat_file(path).columns) == column_names()


class TestErrorHandling:
    """Malformed input must fail loudly rather than silently mis-shaping data."""

    def test_raises_when_neither_orientation_fits(self, tmp_path: Path) -> None:
        """A matrix with 52 variables in no orientation must raise ValueError."""
        path = tmp_path / "bad.dat"
        _write_matrix(path, rows=10, cols=3)
        with pytest.raises(ValueError, match="neither orientation"):
            load_dat_file(path)

    def test_error_names_the_file(self, tmp_path: Path) -> None:
        """The error message must identify the offending file."""
        path = tmp_path / "broken.dat"
        _write_matrix(path, rows=4, cols=6)
        with pytest.raises(ValueError, match="broken.dat"):
            load_dat_file(path)

    def test_raises_for_missing_file(self, tmp_path: Path) -> None:
        """A non-existent path must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_dat_file(tmp_path / "nonexistent.dat")
