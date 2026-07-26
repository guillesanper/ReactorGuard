"""Typed access to the features: section of params.yaml.

Mismo motivo que data/generators/tep_params.py: si dvc.yaml declara depender de
un parametro que ningun ejecutable llega a leer, la invalidacion del stage es
aparente y no efectiva. Hasta T4.2 la seccion features: de params.yaml estaba
huerfana -- nadie la leia y ml/features/extractor.py hardcodeaba sus propias
constantes. Este modulo la convierte en configuracion real.

LAS VENTANAS SE DECLARAN EN MUESTRAS, NO EN SEGUNDOS. El TDD 5.5 las pedia en
segundos (60 s, 5 min, 1 h) porque asumia el sondeo SCADA de 1 s. A los 180 s de
muestreo del TEP, una ventana de 60 s no contiene NI UNA muestra y una de 5 min
contiene una o dos, de modo que su desviacion tipica es NaN o cero. Es el mismo
fallo de cadencia que se midio en el umbral de deriva del validador, y en un
featurizer sale mas caro: produce columnas vacias que un modelo de Fase 4
entrenaria como si fuesen senal.

Declaradas en muestras, una ventana no puede caer por debajo de una muestra por
construccion, y el numero significa lo mismo sea cual sea la cadencia de la
fuente. La equivalencia en segundos la calcula el featurizer midiendo el
intervalo real de los datos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "features"


@dataclass(frozen=True)
class SensorSelection:
    """Which sensors the featurizer is expected to find in the data.

    No enumera los tags. Enumerar los 52 del TEP seria cablear el inventario de
    una planta concreta en la configuracion, que es el error que ya se corrigio
    al retirar la lista sensor_channels del schema plano. El featurizer resuelve
    el conjunto real desde los datos y lo contrasta contra expected_sensor_count,
    de modo que una deriva de schema falla en voz alta.

    Attributes:
        include_types: Sensor types kept. Un tipo ausente de la lista se descarta.
        exclude_sensor_ids: Tags dropped regardless of their type.
        expected_sensor_count: Number of sensors the resolved set must contain.
    """

    include_types: frozenset[str]
    exclude_sensor_ids: frozenset[str]
    expected_sensor_count: int


@dataclass(frozen=True)
class FeatureParams:
    """Resolved configuration of the feature pipeline.

    Attributes:
        features_dir: Root of the fault_type-partitioned feature output.
        window_samples: Rolling window lengths, in SAMPLES, ascending.
        min_samples_per_window: Samples a window needs before its statistic is
            emitted; below it the cell is null rather than a number invented from
            one or two points.
        n_lags: Lagged copies of the value emitted per sensor.
        selection: Which sensors are expected.
        correlation_pairs: Sensor pairs whose rolling Pearson coefficient is
            emitted. Vacio por defecto, por el mismo motivo que en
            CrossCorrelationChecker: la tabla de parejas correlacionadas del TEP
            es una calibracion que nadie ha derivado, e inventarla aqui seria
            cablear una suposicion sobre la planta.
    """

    features_dir: Path
    window_samples: tuple[int, ...]
    min_samples_per_window: int
    n_lags: int
    selection: SensorSelection
    correlation_pairs: tuple[tuple[str, str], ...]

    @property
    def longest_window(self) -> int:
        """Return the longest rolling window, in samples."""
        return self.window_samples[-1]

    @property
    def shortest_window(self) -> int:
        """Return the shortest rolling window, in samples."""
        return self.window_samples[0]

    def window_seconds(self, sample_interval_seconds: float) -> dict[int, float]:
        """Return each window's span in seconds at a given cadence.

        Args:
            sample_interval_seconds: Measured spacing between samples.

        Returns:
            Mapping from window length in samples to its span in seconds.
        """
        return {
            window: window * sample_interval_seconds for window in self.window_samples
        }


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed features: mapping.
        key: Key that must be present.

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
    return section[key]


def _parse_windows(raw: Any) -> tuple[int, ...]:
    """Coerce and validate the window list.

    Args:
        raw: Value of features.window_samples.

    Returns:
        The windows as a sorted tuple of ints.

    Raises:
        TypeError: If the value is not a list.
        ValueError: If it is empty, holds a window below 2 samples, or repeats one.
    """
    if not isinstance(raw, list):
        raise TypeError(
            f"params.yaml: '{_SECTION}.window_samples' must be a list, "
            f"got {type(raw).__name__}."
        )
    if not raw:
        raise ValueError(f"params.yaml: '{_SECTION}.window_samples' must not be empty.")

    windows = tuple(sorted(int(value) for value in raw))
    if windows[0] < 2:
        raise ValueError(
            f"params.yaml: '{_SECTION}.window_samples' contains {windows[0]}, but a "
            "rolling statistic over fewer than 2 samples is not a statistic. Las "
            "ventanas se declaran en MUESTRAS, no en segundos: revisa que no sean "
            "los valores en segundos del TDD."
        )
    if len(set(windows)) != len(windows):
        raise ValueError(
            f"params.yaml: '{_SECTION}.window_samples' repeats a window: {windows}. "
            "Cada ventana genera columnas con su longitud en el nombre, de modo que "
            "un duplicado produce columnas colisionadas."
        )
    return windows


def _parse_pairs(raw: Any) -> tuple[tuple[str, str], ...]:
    """Coerce and validate the correlation pair list.

    Args:
        raw: Value of features.correlation_pairs.

    Returns:
        The pairs as a tuple of two-tuples.

    Raises:
        TypeError: If the value is not a list of two-element sequences.
        ValueError: If a pair names the same sensor twice.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(
            f"params.yaml: '{_SECTION}.correlation_pairs' must be a list, "
            f"got {type(raw).__name__}."
        )

    pairs: list[tuple[str, str]] = []
    for entry in raw:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            raise TypeError(
                f"params.yaml: every entry of '{_SECTION}.correlation_pairs' must be "
                f"a pair of sensor ids, got {entry!r}."
            )
        first, second = str(entry[0]), str(entry[1])
        if first == second:
            raise ValueError(
                f"params.yaml: '{_SECTION}.correlation_pairs' pairs '{first}' with "
                "itself, whose correlation is 1 by definition."
            )
        pairs.append((first, second))
    return tuple(pairs)


def _parse_selection(raw: Any) -> SensorSelection:
    """Build the sensor selection policy.

    Args:
        raw: Value of features.sensor_selection.

    Returns:
        The parsed SensorSelection.

    Raises:
        TypeError: If the value is not a mapping.
        KeyError: If a required key is missing.
        ValueError: If expected_sensor_count is not positive.
    """
    if not isinstance(raw, dict):
        raise TypeError(
            f"params.yaml: '{_SECTION}.sensor_selection' must be a mapping, "
            f"got {type(raw).__name__}."
        )

    expected = int(_require(raw, "expected_sensor_count"))
    if expected <= 0:
        raise ValueError(
            f"params.yaml: '{_SECTION}.sensor_selection.expected_sensor_count' must "
            f"be positive, got {expected}."
        )

    return SensorSelection(
        include_types=frozenset(str(t) for t in _require(raw, "include_types")),
        exclude_sensor_ids=frozenset(
            str(t) for t in (raw.get("exclude_sensor_ids") or [])
        ),
        expected_sensor_count=expected,
    )


def load_feature_params(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
) -> FeatureParams:
    """Load and validate the features: section of a params.yaml file.

    Args:
        params_path: Path to the params file.

    Returns:
        A fully resolved FeatureParams instance.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the features: section or a required key is missing.
        TypeError: If a value has the wrong shape.
        ValueError: If a value is out of range.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    section = document[_SECTION]

    windows = _parse_windows(_require(section, "window_samples"))
    min_samples = int(_require(section, "min_samples_per_window"))
    if min_samples < 2:
        raise ValueError(
            f"params.yaml: '{_SECTION}.min_samples_per_window' is {min_samples}; a "
            "standard deviation over a single sample is zero by construction, which "
            "would be indistinguishable from a genuinely flat channel."
        )
    if min_samples > windows[0]:
        raise ValueError(
            f"params.yaml: '{_SECTION}.min_samples_per_window' is {min_samples} but "
            f"the shortest window is {windows[0]} samples, so that window could "
            "never emit a value."
        )

    n_lags = int(_require(section, "n_lags"))
    if n_lags < 0:
        raise ValueError(
            f"params.yaml: '{_SECTION}.n_lags' must be non-negative, got {n_lags}."
        )

    return FeatureParams(
        features_dir=Path(str(_require(section, "features_dir"))),
        window_samples=windows,
        min_samples_per_window=min_samples,
        n_lags=n_lags,
        selection=_parse_selection(_require(section, "sensor_selection")),
        correlation_pairs=_parse_pairs(section.get("correlation_pairs")),
    )
