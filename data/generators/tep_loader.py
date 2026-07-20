"""Shared loader for Tennessee Eastman Process (TEP) .dat files.

Unica fuente de verdad para leer un .dat del dataset. tep_explorer.py y
tep_adapter.py delegan aqui en lugar de duplicar la lectura y la validacion.

Nota sobre la orientacion de los ficheros: en el repositorio de Prof. Braatz los
ficheros de fallo (d01.dat .. d21.dat) se publican como (n_samples, 52), pero
d00.dat se publica transpuesto, como (52, n_samples). Cargarlo tal cual produce
un DataFrame de 500 columnas y hace fallar la validacion de 52 variables. Este
modulo detecta esa orientacion y la normaliza, de modo que todos los ficheros
salen de aqui con la forma (n_samples, 52).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

_LOG = logging.getLogger(__name__)

N_COLUMNS = 52


def column_names() -> list[str]:
    """Return the canonical column names for a TEP frame.

    Returns:
        List of 52 names, col_00 through col_51.
    """
    return [f"col_{i:02d}" for i in range(N_COLUMNS)]


def load_dat_file(filepath: str | Path) -> pd.DataFrame:
    """Load a TEP whitespace-delimited .dat file with a normalised orientation.

    The file has no header row. Files stored as (52, n_samples) are transposed
    to (n_samples, 52) so that every caller sees one row per timestep and one
    column per process variable.

    Args:
        filepath: Path to the .dat file.

    Returns:
        DataFrame of shape (n_samples, 52) with columns col_00 .. col_51 and a
        contiguous RangeIndex.

    Raises:
        FileNotFoundError: If filepath does not exist.
        ValueError: If the file has 52 variables in neither orientation.
    """
    path = Path(filepath)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    if df.shape[1] != N_COLUMNS:
        if df.shape[0] == N_COLUMNS:
            _LOG.info(
                "Transposing %s: stored as (%d, %d), normalising to (%d, %d).",
                path.name,
                df.shape[0],
                df.shape[1],
                df.shape[1],
                df.shape[0],
            )
            df = df.transpose().reset_index(drop=True)
        else:
            raise ValueError(
                f"Expected {N_COLUMNS} columns in {path.name}, got {df.shape[1]} "
                f"(shape {df.shape[0]}x{df.shape[1]}; neither orientation has "
                f"{N_COLUMNS} variables)."
            )

    df.columns = pd.Index(column_names())
    return df
