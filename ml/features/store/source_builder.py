"""Entry point for the feast_source DVC stage.

Convierte el parquet de features del stage featurize en la fuente OFFLINE de Feast.
Es un artefacto DERIVADO y SINTETICO: el TEP no tiene reloj real, asi que los
timestamps se fabrican a partir de tep.start_time y de la cadencia de 180 s.

POR QUE HACE FALTA. Feast exige que (entidad, event_timestamp) sea unico, y el
parquet de features no cumple dos cosas:

  * No tiene columna de tiempo: solo `timestep` dentro de cada run.
  * Los 22 ficheros del TEP reinician el reloj, asi que el timestep 0 del sensor
    TEP-XMEAS-01 existe en las 22 clases. Si el timestamp fuera simplemente
    start_time + timestep x 180 s, el mismo (sensor, instante) apareceria 22 veces
    y Feast serviria una fila cualquiera de ellas.

SOLUCION. Cada run se desplaza en la linea de tiempo, uno tras otro y con un hueco
(feature_store.run_gap_intervals) mayor que el TTL, de modo que la cola de un run
nunca cae dentro del TTL de la cabeza del siguiente:

    run 0:  |-- 500 intervalos --| hueco |
    run 1:                              |-- 480 intervalos --| hueco |
    run 2:                                                          |-- ...

REPRODUCIBILIDAD. created_timestamp es igual a event_timestamp, no la hora de
ejecucion. Con la hora de ejecucion cada `dvc repro` produciria bytes distintos y la
cache de DVC no se podria compartir: es el mismo defecto que tenia adapt_tep con
uuid4 y que se corrigio con uuid5. Como (sensor, event_timestamp) es unico, el
desempate por created_timestamp que hace Feast no llega a ejercerse.

ANCLA. Los timestamps parten del ano 2000, de modo que frente al reloj real estan
siempre caducados. rebase_event_timestamps() desplaza una copia hasta un ancla
reciente, que es lo que hacen las pruebas de integracion y lo que haria una carga
del almacen online en el cluster. El artefacto de DVC no se rebasa.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data.generators.tep_params import load_tep_params
from ml.features.batch_featurizer import FEATURES_FILENAME, discover_partitions
from ml.features.feature_params import DEFAULT_PARAMS_PATH, load_feature_params
from ml.features.pipeline import FeaturePipeline
from ml.features.store.feature_store_params import load_feature_store_params

_LOG = logging.getLogger(__name__)

KEY_COLUMN = "sensor_id"
TIMESTEP_COLUMN = "timestep"
FAULT_TYPE_COLUMN = "fault_type"
EVENT_TIMESTAMP_COLUMN = "event_timestamp"
CREATED_TIMESTAMP_COLUMN = "created_timestamp"


def _utc(moment: datetime) -> pd.Timestamp:
    """Return a timezone-aware UTC timestamp, treating a naive one as UTC.

    Args:
        moment: Datetime with or without tzinfo.

    Returns:
        The same instant as a UTC pandas Timestamp.
    """
    stamp = pd.Timestamp(moment)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _check_columns(
    fault_type: int, frame: pd.DataFrame, required: Iterable[str]
) -> None:
    """Check that a partition carries the key columns and every required feature.

    Args:
        fault_type: Partition being checked, for the message.
        frame: The partition's feature frame.
        required: Feature columns the FeatureView will declare.

    Raises:
        ValueError: If a column is missing.
    """
    missing = [
        column
        for column in (KEY_COLUMN, TIMESTEP_COLUMN, *required)
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"fault_type={fault_type:02d}: the feature frame lacks {missing}. "
            "Regenera features.parquet con el stage featurize: la fuente de Feast "
            "debe llevar exactamente las columnas que declara el FeatureView."
        )


def build_source_frame(
    partitions: Mapping[int, pd.DataFrame],
    origin: datetime,
    interval_seconds: int,
    run_gap_intervals: int,
    expected_sensor_count: int,
    required_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Build the single-timeline offline source from the per-fault-type features.

    Args:
        partitions: Long feature frame of each fault type, keyed by fault type.
            Cada frame lleva sensor_id y timestep mas las features.
        origin: Instant of timestep 0 of the first run (tep.start_time).
        interval_seconds: Sampling cadence (tep.sample_interval_minutes x 60).
        run_gap_intervals: Empty intervals left between consecutive runs.
        expected_sensor_count: Distinct sensors the data must contain
            (features.sensor_selection.expected_sensor_count).
        required_columns: Feature columns every partition must carry.

    Returns:
        Frame with event_timestamp, created_timestamp, sensor_id, fault_type,
        timestep and the features, ordered by (event_timestamp, sensor_id).

    Raises:
        ValueError: If there are no partitions, a partition lacks a required
            column, a timestep is negative, the sensor count differs from
            expected_sensor_count, or (sensor_id, event_timestamp) is not unique.
    """
    if not partitions:
        raise ValueError("No feature partitions to build the Feast source from.")

    origin_stamp = _utc(origin)
    pieces: list[pd.DataFrame] = []
    offset_intervals = 0

    for fault_type, frame in sorted(partitions.items()):
        _check_columns(fault_type, frame, required_columns)
        steps = frame[TIMESTEP_COLUMN].to_numpy(dtype=np.int64)
        if steps.size == 0 or steps.min() < 0:
            raise ValueError(
                f"fault_type={fault_type:02d}: timestep must be a non-empty "
                "non-negative series."
            )

        seconds = (offset_intervals + steps) * interval_seconds
        stamps = origin_stamp + pd.to_timedelta(seconds, unit="s")

        piece = frame.copy()
        piece.insert(0, EVENT_TIMESTAMP_COLUMN, stamps)
        piece.insert(1, CREATED_TIMESTAMP_COLUMN, stamps)
        if FAULT_TYPE_COLUMN not in piece.columns:
            piece[FAULT_TYPE_COLUMN] = fault_type
        pieces.append(piece)

        offset_intervals += int(steps.max()) + 1 + run_gap_intervals

    combined = pd.concat(pieces, ignore_index=True)

    sensors = int(combined[KEY_COLUMN].nunique())
    if sensors != expected_sensor_count:
        raise ValueError(
            f"The features hold {sensors} distinct sensors but "
            f"features.sensor_selection.expected_sensor_count declares "
            f"{expected_sensor_count}. Es una deriva de schema: Feast serviria un "
            "conjunto de entidades distinto del que espera el resto de la plataforma."
        )

    duplicated = int(combined.duplicated([KEY_COLUMN, EVENT_TIMESTAMP_COLUMN]).sum())
    if duplicated:
        raise ValueError(
            f"{duplicated} rows repeat a (sensor_id, event_timestamp) pair, which "
            "Feast cannot disambiguate. Revisa que (sensor_id, timestep) sea unico "
            "dentro de cada fault_type."
        )

    return combined.sort_values([EVENT_TIMESTAMP_COLUMN, KEY_COLUMN]).reset_index(
        drop=True
    )


def rebase_event_timestamps(frame: pd.DataFrame, anchor: datetime) -> pd.DataFrame:
    """Shift the timeline so its latest event lands on a given instant.

    El desplazamiento es el mismo para event_timestamp y created_timestamp y no
    altera ni el orden ni los huecos. Devuelve una copia: el artefacto de DVC no se
    modifica.

    Args:
        frame: Output of build_source_frame (or the parquet the stage wrote).
        anchor: Timezone-aware instant that the latest event is moved to.

    Returns:
        A shifted copy.

    Raises:
        ValueError: If anchor is naive, or the frame is empty.
    """
    if anchor.tzinfo is None:
        raise ValueError("anchor must be timezone-aware; a naive one is ambiguous.")
    if frame.empty:
        raise ValueError("Cannot rebase an empty frame.")

    shift = _utc(anchor) - frame[EVENT_TIMESTAMP_COLUMN].max()
    shifted = frame.copy()
    shifted[EVENT_TIMESTAMP_COLUMN] = shifted[EVENT_TIMESTAMP_COLUMN] + shift
    shifted[CREATED_TIMESTAMP_COLUMN] = shifted[CREATED_TIMESTAMP_COLUMN] + shift
    return shifted


def main(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    features_dir: str | Path | None = None,
) -> int:
    """Run the feast_source stage end to end.

    Args:
        params_path: Path to the params file.
        features_dir: Root of the feature parquet. Defaults to features.features_dir,
            que es donde escribe el stage featurize.

    Returns:
        Number of rows written.

    Raises:
        FileNotFoundError: If params_path or the feature tree does not exist.
        ValueError: If the features are malformed (see build_source_frame).
    """
    store_params = load_feature_store_params(params_path)
    feature_params = load_feature_params(params_path)
    tep_params = load_tep_params(params_path)

    root = features_dir if features_dir is not None else feature_params.features_dir
    parquet_files = discover_partitions(root, filename=FEATURES_FILENAME, producer="featurize")
    partitions = {
        fault_type: pd.read_parquet(path) for fault_type, path in parquet_files.items()
    }

    pipeline = FeaturePipeline(
        feature_params, sample_interval_seconds=store_params.sample_interval_seconds
    )
    frame = build_source_frame(
        partitions,
        origin=tep_params.start_time,
        interval_seconds=store_params.sample_interval_seconds,
        run_gap_intervals=store_params.run_gap_intervals,
        expected_sensor_count=feature_params.selection.expected_sensor_count,
        required_columns=pipeline.feature_names(),
    )

    target = store_params.source_path
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    _LOG.info(
        "feast_source complete: %d rows, %d runs, %s .. %s -> %s",
        len(frame),
        len(partitions),
        frame[EVENT_TIMESTAMP_COLUMN].min(),
        frame[EVENT_TIMESTAMP_COLUMN].max(),
        target,
    )
    return len(frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
