"""Entry point for the featurize DVC stage.

Aplica FeaturePipeline al parquet de lecturas y escribe las features con el mismo
particionado por fault_type. La logica de calculo vive en pipeline.py; este
modulo resuelve parametros, hace el pivot y reporta, igual que adapt_tep.py hace
con el adaptador.

EL PIVOT VIVE AQUI, Y POR PARTICION. El parquet de lecturas esta en formato
LARGO (una fila por sensor-timestep) y las ventanas moviles y los lags necesitan
formato ANCHO (timestep x sensor). La frontera de fault_type es la unidad natural
de ese pivot por dos motivos que se refuerzan:

  Correccion. Cada fichero del TEP es una corrida independiente que reinicia el
  reloj en start_time. Rodar una ventana a traves de la frontera haria que las
  primeras muestras de una clase heredasen historia de otra, que es fuga de
  informacion entre clases y la contaminaria antes de que el split la separe.

  Memoria. El "problema" de las 550.160 filas se disuelve al respetar esa
  frontera: cada pivot es de 500 x 52 flotantes, unos 200 KB, y las 22
  particiones nunca coexisten en memoria.

Como consecuencia pipeline.py se queda agnostico al particionado, que es lo que
le permite servir tambien al camino en linea de T3.5, donde no hay particiones ni
pivot que hacer.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from ml.features.feature_params import (
    DEFAULT_PARAMS_PATH,
    FeatureParams,
    load_feature_params,
)
from ml.features.pipeline import FeaturePipeline

_LOG = logging.getLogger(__name__)

READINGS_FILENAME = "readings.parquet"
FEATURES_FILENAME = "features.parquet"
_PARTITION_PATTERN = re.compile(r"^fault_type=(\d+)$")


def discover_partitions(
    processed_dir: str | Path,
    filename: str = READINGS_FILENAME,
    producer: str = "adapt_tep",
) -> dict[int, Path]:
    """Find the fault_type partitions of a partitioned parquet tree.

    El nombre del fichero es un argumento y no una constante porque el arbol de
    lecturas y el de features comparten disposicion: el stage split reutiliza
    esta funcion sobre features.parquet. Fijarlo dentro obligaria a duplicar la
    funcion entera para cambiar una cadena.

    Args:
        processed_dir: Root of the partitioned tree.
        filename: Parquet file expected inside each partition.
        producer: Stage that produces the tree, named in the error so the message
            apunta al remedio y no solo al sintoma.

    Returns:
        Mapping from fault type to the partition's parquet file, sorted by key.

    Raises:
        FileNotFoundError: If the root does not exist or holds no partition.
    """
    root = Path(processed_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Partition directory not found: {root}. Run the {producer} stage first."
        )

    partitions: dict[int, Path] = {}
    for entry in sorted(root.iterdir()):
        match = _PARTITION_PATTERN.match(entry.name)
        if match is None or not entry.is_dir():
            continue
        parquet = entry / filename
        if parquet.exists():
            partitions[int(match.group(1))] = parquet

    if not partitions:
        raise FileNotFoundError(
            f"No fault_type partitions with a {filename} under {root}. "
            f"Run the {producer} stage first."
        )
    return dict(sorted(partitions.items()))


def measure_sample_interval(frame: pd.DataFrame) -> float:
    """Measure the sampling cadence from the frame's own timestamps.

    Se mide en lugar de leerse de params.yaml a proposito. La cadencia decide si
    una ventana significa media hora o media jornada, y tomarla de un parametro
    que puede haber quedado desincronizado con los datos ya escritos es
    exactamente como se cuela un fallo de cadencia silencioso.

    Args:
        frame: Long-format readings frame with timestamp and timestep columns.

    Returns:
        Seconds between consecutive timesteps.

    Raises:
        ValueError: If the partition has fewer than two timesteps or the spacing
            is not uniform.
    """
    stamps = (
        frame.drop_duplicates(subset="timestep")
        .sort_values("timestep")
        .loc[:, "timestamp"]
    )
    if len(stamps) < 2:
        raise ValueError(
            f"Cannot measure the sampling interval from {len(stamps)} timestep(s)."
        )

    deltas = stamps.diff().dropna().dt.total_seconds().unique()
    if len(deltas) != 1:
        raise ValueError(
            f"Sampling interval is not uniform: found {len(deltas)} distinct gaps "
            f"({sorted(deltas)[:5]}). Every rolling window would span a different "
            "amount of time depending on where it sits."
        )
    return float(deltas[0])


def featurize_partition(
    frame: pd.DataFrame, params: FeatureParams, fault_type: int
) -> pd.DataFrame:
    """Pivot one partition to wide, featurise it and return the long result.

    Args:
        frame: Long-format readings of a single fault type.
        params: Resolved feature configuration.
        fault_type: The partition's fault type, stamped on every output row so
            the split stage can stratify by it.

    Returns:
        Long feature frame with fault_type, timestep, sensor_id and the features.

    Raises:
        ValueError: If the partition is not a uniform, contiguous run, or if the
            sensor selection does not match the declared count.
    """
    interval = measure_sample_interval(frame)

    wide = frame.pivot(index="timestep", columns="sensor_id", values="value")
    if wide.isna().to_numpy().any():
        gaps = int(wide.isna().to_numpy().sum())
        _LOG.warning(
            "fault_type=%02d: %d (timestep, sensor) cells have no reading; their "
            "rolling features will be null rather than imputed.",
            fault_type,
            gaps,
        )

    stamps = (
        frame.drop_duplicates(subset="timestep")
        .set_index("timestep")
        .sort_index()
        .loc[:, "timestamp"]
    )
    sensor_types = dict(
        zip(frame["sensor_id"], frame["sensor_type"], strict=True)
    )

    pipeline = FeaturePipeline(params, sample_interval_seconds=interval)
    features = pipeline.transform(wide, stamps, sensor_types)
    features.insert(0, "fault_type", fault_type)
    return features


def save_features(frame: pd.DataFrame, features_dir: str | Path, fault_type: int) -> Path:
    """Write one partition's features, mirroring the readings layout.

    Args:
        frame: Long feature frame of a single fault type.
        features_dir: Root of the partitioned feature tree.
        fault_type: The partition's fault type.

    Returns:
        Path of the file written.
    """
    partition = Path(features_dir) / f"fault_type={fault_type:02d}"
    partition.mkdir(parents=True, exist_ok=True)
    target = partition / FEATURES_FILENAME
    frame.to_parquet(target, index=False)
    _LOG.info("Wrote %d feature rows to %s", len(frame), target)
    return target


def main(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    readings_dir: str | Path | None = None,
) -> dict[int, int]:
    """Run the featurize stage end to end.

    Args:
        params_path: Path to the params file holding the features: section.
        readings_dir: Root of the readings parquet. Defaults to the processed_dir
            of the tep: section, which is where adapt_tep writes.

    Returns:
        Mapping from fault type to the number of feature rows written.

    Raises:
        FileNotFoundError: If params_path or the readings tree does not exist.
        ValueError: If a partition is malformed or the sensor selection fails.
    """
    params = load_feature_params(params_path)

    if readings_dir is None:
        from data.generators.tep_params import load_tep_params

        readings_dir = load_tep_params(params_path).processed_dir

    partitions = discover_partitions(readings_dir)
    _LOG.info(
        "Featurising %d partitions from %s with windows %s samples",
        len(partitions),
        readings_dir,
        list(params.window_samples),
    )

    written: dict[int, int] = {}
    for fault_type, parquet in partitions.items():
        frame = pd.read_parquet(parquet)
        features = featurize_partition(frame, params, fault_type)
        save_features(features, params.features_dir, fault_type)
        written[fault_type] = len(features)

    total = sum(written.values())
    _LOG.info("featurize complete: %d feature rows across %d partitions", total, len(written))
    return written


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
