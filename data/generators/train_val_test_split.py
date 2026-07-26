"""Entry point for the split DVC stage: train/val/test over the feature table.

El corte es TEMPORAL DENTRO DE CADA fault_type, que satisface a la vez las dos
exigencias del plan y que a primera vista parecen tirar en direcciones opuestas.

    Estratificado. Cada fault_type aporta la misma proporcion a los tres splits,
    de modo que las 22 clases estan representadas en los tres. Un corte temporal
    sobre la linea temporal GLOBAL habria puesto ficheros enteros en test, y con
    ellos clases que el modelo no habria visto jamas en entrenamiento.

    Sin fuga. Dentro de una particion, train se queda con los primeros timesteps,
    val con los siguientes y test con los ultimos. Ninguna fila de train ve una
    muestra posterior a su propio instante.

SOBRE EL SOLAPE EN LA FRONTERA. Las ventanas moviles del featurizer miran hacia
atras hasta 60 muestras, asi que las primeras filas de val llevan features
calculadas en parte sobre el periodo de train. NO es fuga y no se purga:

  - La direccion importa. Lo que corrompe una evaluacion es que el
    entrenamiento vea el FUTURO. Aqui ninguna fila de train mira mas alla de su
    instante; son las de val las que miran hacia atras, que es exactamente lo que
    tambien ocurre en inferencia real, donde el modelo dispone de su historia.

  - Purgar saldria carisimo y sin motivo. Un embargo del ancho de la ventana
    larga descartaria 60 de las 72 muestras de val en una particion de 480: el
    82% del conjunto de validacion, para corregir un sesgo que no existe.

  - CUANDO SI HABRIA QUE PURGAR: si la Fase 4 define un objetivo calculado sobre
    ventana ("hay fallo en los proximos N pasos"), las etiquetas de train
    solaparian con las features de val y entonces el embargo pasa a ser
    obligatorio. Queda anotado aqui porque es el sitio donde se decidiria.

El resto se reparte por division entera y el sobrante va a test, de modo que
ninguna muestra se pierde: la union de los tres splits es la particion entera.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.features.batch_featurizer import FEATURES_FILENAME, discover_partitions
from ml.features.feature_params import load_feature_params

_LOG = logging.getLogger(__name__)

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "training"

SPLIT_NAMES = ("train", "val", "test")
"""Orden temporal de los splits: train es el pasado, test el futuro."""


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitParams:
    """Resolved configuration of the split stage.

    Attributes:
        splits_dir: Directory holding one parquet per split.
        metrics_path: JSON consumed by `dvc metrics show`. Corto y plano a
            proposito: ver build_metrics.
        distribution_path: JSON with the full per-fault_type breakdown, kept out
            of the metrics table so that table stays readable.
        train_ratio: Share of each partition's timesteps given to train.
        val_ratio: Share given to validation.
        test_ratio: Share given to test. Recibe ademas el resto de la division
            entera, asi que su tamano real puede superar ligeramente su ratio.
    """

    splits_dir: Path
    metrics_path: Path
    distribution_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float

    @property
    def ratios(self) -> dict[str, float]:
        """Return the three ratios keyed by split name."""
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }


def load_split_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> SplitParams:
    """Load and validate the split-related keys of the training: section.

    Args:
        params_path: Path to the params file.

    Returns:
        A fully resolved SplitParams instance.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the training: section or a required key is missing.
        ValueError: If a ratio is not positive or the three do not sum to one.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    section = document[_SECTION]

    def _require(key: str) -> Any:
        if key not in section:
            raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
        return section[key]

    ratios = {name: float(_require(f"{name}_ratio")) for name in SPLIT_NAMES}
    for name, ratio in ratios.items():
        if ratio <= 0.0:
            raise ValueError(
                f"params.yaml: '{_SECTION}.{name}_ratio' must be positive, got {ratio}. "
                "Un split vacio no es un split; retira la fase si no la quieres."
            )

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"params.yaml: the three {_SECTION} ratios sum to {total}, not 1.0 "
            f"({ratios}). Una suma distinta de uno descarta muestras en silencio."
        )

    return SplitParams(
        splits_dir=Path(str(_require("splits_dir"))),
        metrics_path=Path(str(_require("metrics_path"))),
        distribution_path=Path(str(_require("distribution_path"))),
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_boundaries(n_timesteps: int, params: SplitParams) -> dict[str, range]:
    """Return the timestep range of each split within one partition.

    El sobrante de la division entera va a test y no se reparte, de modo que la
    union de los tres rangos es siempre la particion completa.

    Args:
        n_timesteps: Number of distinct timesteps in the partition.
        params: Resolved ratios.

    Returns:
        Mapping from split name to its half-open range of timesteps.

    Raises:
        ValueError: If the partition is too short to give every split at least
            one timestep.
    """
    n_train = int(n_timesteps * params.train_ratio)
    n_val = int(n_timesteps * params.val_ratio)
    n_test = n_timesteps - n_train - n_val

    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            f"A partition of {n_timesteps} timesteps cannot be split "
            f"{params.train_ratio}/{params.val_ratio}/{params.test_ratio} without "
            f"leaving a split empty (got {n_train}/{n_val}/{n_test})."
        )

    return {
        "train": range(0, n_train),
        "val": range(n_train, n_train + n_val),
        "test": range(n_train + n_val, n_timesteps),
    }


def split_partition(
    frame: pd.DataFrame, params: SplitParams
) -> dict[str, pd.DataFrame]:
    """Cut one fault_type partition into its three temporal slices.

    Args:
        frame: Long feature frame of a single fault type.
        params: Resolved ratios.

    Returns:
        Mapping from split name to its rows, each sorted by (timestep, sensor_id).

    Raises:
        ValueError: If the partition is too short to split.
    """
    timesteps = sorted(frame["timestep"].unique())
    boundaries = split_boundaries(len(timesteps), params)
    lookup = {position: step for position, step in enumerate(timesteps)}

    slices: dict[str, pd.DataFrame] = {}
    for name, positions in boundaries.items():
        wanted = {lookup[position] for position in positions}
        slices[name] = frame[frame["timestep"].isin(wanted)].sort_values(
            ["timestep", "sensor_id"], kind="stable"
        )
    return slices


def _rows_by_fault_type(frame: pd.DataFrame) -> dict[str, int]:
    """Return one split's row count per fault type.

    Args:
        frame: One assembled split.

    Returns:
        Mapping from a zero-padded fault_type label to its row count.
    """
    return {
        f"fault_type_{int(fault):02d}": int(count)
        for fault, count in sorted(frame.groupby("fault_type").size().items())
    }


def temporal_order_holds(splits: dict[str, pd.DataFrame]) -> bool:
    """Return whether train precedes val precedes test WITHIN every fault type.

    Se comprueba por clase y nunca en agregado, porque en agregado la propiedad
    NO se cumple y no puede cumplirse: d00 trae 500 timesteps y los ficheros de
    fallo 480, de modo que el train de d00 llega al timestep 349 mientras el val
    de las clases de fallo empieza en el 336. Los rangos globales se solapan
    aunque dentro de cada particion el corte sea estricto, que es lo unico que
    importa: ningun modelo entrena sobre el futuro DE SU PROPIA corrida.

    Comparar los minimos y maximos globales daria un falso negativo aqui, y con
    particiones de igual longitud daria un falso positivo capaz de esconder una
    clase mal cortada detras de las demas.

    Args:
        splits: The three assembled splits.

    Returns:
        True when the order holds for every fault type present.
    """
    for fault in sorted(splits["train"]["fault_type"].unique()):
        steps = {
            name: frame.loc[frame["fault_type"] == fault, "timestep"]
            for name, frame in splits.items()
        }
        if any(series.empty for series in steps.values()):
            return False
        if not steps["train"].max() < steps["val"].min():
            return False
        if not steps["val"].max() < steps["test"].min():
            return False
    return True


def build_metrics(
    splits: dict[str, pd.DataFrame], params: SplitParams
) -> dict[str, Any]:
    """Summarise the split into the handful of numbers `dvc metrics show` prints.

    DELIBERADAMENTE PLANO Y CORTO. `dvc metrics show` aplana el JSON anidado en
    una columna por hoja, asi que meter aqui el recuento de las 22 clases por
    cada uno de los tres splits producia 66 columnas que en un terminal no se
    leen: el comando dejaba de mostrar nada util justo por querer mostrarlo todo.
    El desglose completo va a build_distribution, en un fichero hermano tambien
    versionado.

    Las dos propiedades que el stage promete se resumen en dos banderas:
    `fault_types_in_every_split` acredita la estratificacion y
    `temporal_order_holds` el orden temporal.

    OJO con los `*_min_timestep` y `*_max_timestep`: son AGREGADOS sobre las 22
    particiones y se solapan legitimamente, porque d00 trae 500 muestras y los
    ficheros de fallo 480. En los datos reales train_max_timestep sale 349 y
    val_min_timestep 336, lo que leido en crudo parece una fuga y no lo es. Estan
    para describir el reparto, NO para verificarlo: quien verifica es
    `temporal_order_holds`, que mira clase por clase.

    Args:
        splits: The three assembled splits.
        params: Resolved ratios, reported next to what was actually achieved.

    Returns:
        A flat, JSON-serialisable summary.
    """
    total = sum(len(frame) for frame in splits.values())
    summary: dict[str, Any] = {"rows_total": total}

    classes = set()
    for name in SPLIT_NAMES:
        frame = splits[name]
        by_fault = _rows_by_fault_type(frame)
        classes.add(len(by_fault))

        summary[f"{name}_rows"] = len(frame)
        summary[f"{name}_share"] = len(frame) / total if total else 0.0
        summary[f"{name}_ratio_configured"] = params.ratios[name]
        summary[f"{name}_min_timestep"] = int(frame["timestep"].min())
        summary[f"{name}_max_timestep"] = int(frame["timestep"].max())

    summary["fault_types_per_split"] = min(classes) if classes else 0
    summary["fault_types_in_every_split"] = len(classes) == 1
    summary["temporal_order_holds"] = temporal_order_holds(splits)
    return summary


def build_distribution(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Return the full per-fault_type row count of every split.

    Es lo que hace auditable la estratificacion sin abrir un parquet de 385.000
    filas. Vive fuera del fichero de metricas para no reventar la tabla de
    `dvc metrics show`, pero se versiona igual.

    Args:
        splits: The three assembled splits.

    Returns:
        Mapping from split name to its per-fault_type counts.
    """
    return {name: _rows_by_fault_type(splits[name]) for name in SPLIT_NAMES}


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Run the split stage end to end.

    Args:
        params_path: Path to the params file holding both sections it reads.

    Returns:
        The metrics summary that was written.

    Raises:
        FileNotFoundError: If params_path or the feature tree does not exist.
        ValueError: If a partition cannot be split, or the ratios are invalid.
    """
    params = load_split_params(params_path)
    features_dir = load_feature_params(params_path).features_dir

    partitions = discover_partitions(
        features_dir, filename=FEATURES_FILENAME, producer="featurize"
    )
    _LOG.info(
        "Splitting %d partitions from %s at %.2f/%.2f/%.2f",
        len(partitions),
        features_dir,
        params.train_ratio,
        params.val_ratio,
        params.test_ratio,
    )

    collected: dict[str, list[pd.DataFrame]] = {name: [] for name in SPLIT_NAMES}
    for fault_type, parquet in partitions.items():
        frame = pd.read_parquet(parquet)
        for name, slice_ in split_partition(frame, params).items():
            collected[name].append(slice_)
        _LOG.debug("fault_type=%02d split", fault_type)

    splits = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in collected.items()
    }

    params.splits_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        target = params.splits_dir / f"{name}.parquet"
        frame.to_parquet(target, index=False)
        _LOG.info("Wrote %d rows to %s", len(frame), target)

    metrics = build_metrics(splits, params)
    params.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    params.metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _LOG.info("Wrote split metrics to %s", params.metrics_path)

    params.distribution_path.parent.mkdir(parents=True, exist_ok=True)
    params.distribution_path.write_text(
        json.dumps(build_distribution(splits), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _LOG.info("Wrote fault_type distribution to %s", params.distribution_path)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
