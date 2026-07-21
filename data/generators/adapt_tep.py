"""Entry point for the adapt_tep DVC stage.

Lee la configuracion de la seccion tep: de params.yaml, adapta los ficheros TEP
crudos al schema canonico SensorReading y escribe el resultado como Parquet
particionado por fault_type.

La logica de adaptacion vive en tep_adapter.py; este modulo solo resuelve
parametros, orquesta y reporta. Separarlos es lo que permite que dvc.yaml
declare `params: - tep` sobre un ejecutable que realmente los lee.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.generators.tep_adapter import TEPAdapter, save_to_parquet
from data.generators.tep_params import DEFAULT_PARAMS_PATH, load_tep_params
from data.schemas.sensor_spans import load_sensor_spans

_LOG = logging.getLogger(__name__)


def summarise(df: pd.DataFrame) -> dict[int, int]:
    """Return the count of readings per fault_type.

    Args:
        df: DataFrame produced by TEPAdapter.adapt_all.

    Returns:
        Mapping of fault_type to number of readings, empty if df has no rows.
    """
    if df.empty:
        return {}
    return {int(ft): int(count) for ft, count in df.groupby("fault_type").size().items()}


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> pd.DataFrame:
    """Run the adaptation stage end to end.

    Args:
        params_path: Path to the params file holding the tep: section.

    Returns:
        The consolidated DataFrame that was written to Parquet.

    Raises:
        FileNotFoundError: If params_path or the span table does not exist.
        KeyError: If the span table does not cover every TEP sensor tag.
        ValueError: If no TEP files were found in the configured raw_dir.
    """
    params = load_tep_params(params_path)
    _LOG.info("Adapting TEP files from %s", params.raw_dir)

    spans = load_sensor_spans(params.spans_path)
    _LOG.info("Loaded %d calibrated sensor spans from %s", len(spans), params.spans_path)

    adapter = TEPAdapter.from_params(params, spans)
    df = adapter.adapt_all(str(params.raw_dir))

    if df.empty:
        raise ValueError(
            f"No TEP readings produced from {params.raw_dir}. "
            "Run the download_tep stage first."
        )

    save_to_parquet(df, str(params.processed_dir))

    _LOG.info("Total readings: %d", len(df))
    for fault_type, count in sorted(summarise(df).items()):
        label = "normal" if fault_type == 0 else f"fault_{fault_type:02d}"
        _LOG.info("  fault_type=%02d (%-10s): %d readings", fault_type, label, count)

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
