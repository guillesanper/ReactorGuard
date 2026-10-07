"""Equivalencia bit a bit de FeaturePipeline.transform frente a su version de referencia.

La tarea A de la Fase 2 sustituyo tres piezas de ml/features/pipeline.py por
versiones vectorizadas para cumplir el criterio de latencia (p99 < 100 ms por
ventana de 60x52): el bucle Kalman sensor a sensor, el rolling.apply de la
pendiente y el rolling.apply de los cruces por la media. La salida NO puede
cambiar ni un bit: featurize alimenta a split, y cualquier diferencia de redondeo
cambiaria el hash de data/processed/features y de data/processed/splits.

Este fichero conserva el codigo ANTERIOR como oraculo (`_ReferencePipeline`, copia
literal de los tres metodos y de las dos funciones auxiliares tal como estaban en
el commit b13284c) y exige igualdad exacta (`check_exact=True`) sobre ventanas
reales de d00 y d21 y sobre casos sinteticos que cubren los bordes que el dato
real no pisa: NaN, canales constantes, saltos que disparan el reinicio del filtro,
muestreo irregular, husos horarios y series mas cortas que la ventana.

No borres ni "limpies" el oraculo: es lo que hace que este test tenga dientes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from ml.features.feature_params import FeatureParams, SensorSelection
from ml.features.kalman import OnlineKalmanFilter
from ml.features.pipeline import FEATURIZER_PROCESS_NOISE, FeaturePipeline

_PROCESSED_ROOT = Path("data/processed/tep")
_NORMAL_PARTITION = _PROCESSED_ROOT / "fault_type=00" / "readings.parquet"
_STUCK_PARTITION = _PROCESSED_ROOT / "fault_type=21" / "readings.parquet"

_INTERVAL = 180.0
_T0 = datetime(2024, 3, 14, 9, 0, 0)
_TYPES = {"S-01": "flow", "S-02": "pressure", "S-03": "thermocouple", "P-01": "position"}


# ---------------------------------------------------------------------------
# Oraculo: el codigo anterior, congelado
# ---------------------------------------------------------------------------


def _slope_coefficients(window: int) -> npt.NDArray[np.float64]:
    """Return the least-squares slope weights of a window (copia de referencia).

    Args:
        window: Block length in samples.

    Returns:
        The weights, of length window.
    """
    positions = np.arange(window, dtype=np.float64)
    centred = positions - positions.mean()
    return np.asarray(centred / np.square(centred).sum())


def _least_squares_slope(block: npt.NDArray[np.float64]) -> float:
    """Return the least-squares slope of a window (copia de referencia).

    Args:
        block: Window values in sample order.

    Returns:
        The slope, 0.0 for a block too short to define one.
    """
    if block.size < 2:
        return 0.0
    return float(block @ _slope_coefficients(block.size))


def _zero_crossing_rate(block: npt.NDArray[np.float64]) -> float:
    """Return how often a window crosses its own mean (copia de referencia).

    Args:
        block: Window values.

    Returns:
        Crossings divided by the transitions available, 0.0 for a flat window.
    """
    centred = block - block.mean()
    signs = np.sign(centred)
    non_zero = signs[signs != 0.0]
    if non_zero.size < 2:
        return 0.0
    return float(np.count_nonzero(np.diff(non_zero)) / (non_zero.size - 1))


class _ReferencePipeline(FeaturePipeline):
    """FeaturePipeline con los grupos 1, 2, 3 y 5 calculados como antes."""

    def _rolling_statistics(
        self, frame: pd.DataFrame
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 1 con las cuatro estadisticas de pandas, tal como estaba.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        out: dict[str, npt.NDArray[np.float64]] = {}
        for window in self.params.window_samples:
            rolling = frame.rolling(
                window=window, min_periods=self.params.min_samples_per_window
            )
            out[f"rolling_mean_{window}"] = self._flat(rolling.mean())
            out[f"rolling_std_{window}"] = self._flat(rolling.std())
            out[f"rolling_min_{window}"] = self._flat(rolling.min())
            out[f"rolling_max_{window}"] = self._flat(rolling.max())
        return out

    def _rates(self, frame: pd.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
        """Group 2 con rolling.apply, tal como estaba.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        window = self.params.shortest_window
        slope = frame.rolling(
            window=window, min_periods=self.params.min_samples_per_window
        ).apply(_least_squares_slope, raw=True)
        return {
            "rate_of_change": self._flat(frame.diff() / self.sample_interval_seconds),
            f"slope_{window}": self._flat(slope / self.sample_interval_seconds),
        }

    def _kalman(
        self, frame: pd.DataFrame, timestamps: pd.Series
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 3 con un OnlineKalmanFilter por sensor, tal como estaba.

        Args:
            frame: Wide value frame.
            timestamps: Timestamp of every timestep.

        Returns:
            Mapping from feature name to its flattened column.
        """
        stamps = [pd.Timestamp(value).to_pydatetime() for value in timestamps]
        residual = np.empty(frame.shape, dtype=np.float64)
        sigma = np.empty(frame.shape, dtype=np.float64)
        for position, tag in enumerate(frame.columns):
            series = frame[tag].to_numpy(dtype=np.float64)
            filtered = OnlineKalmanFilter(
                process_noise=self.process_noise,
                observation_noise=self.observation_noise,
            )
            for step, (measurement, stamp) in enumerate(
                zip(series, stamps, strict=True)
            ):
                if not np.isfinite(measurement):
                    residual[step, position] = float("nan")
                    sigma[step, position] = float("nan")
                    continue
                result = filtered.step(float(measurement), stamp)
                residual[step, position] = result.normalized_residual
                sigma[step, position] = result.innovation_sigma
        return {
            "kalman_residual": np.asarray(residual.reshape(-1)),
            "kalman_uncertainty": np.asarray(sigma.reshape(-1)),
        }

    def _distribution(
        self,
        frame: pd.DataFrame,
        statistics: Mapping[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 5 con rolling.std y rolling.apply propios, tal como estaba.

        Recalcula las desviaciones en lugar de reutilizar `statistics`: es lo que
        comprueba que reutilizarlas da el mismo numero.

        Args:
            frame: Wide value frame.
            statistics: Ignored; accepted so the signature matches the pipeline.

        Returns:
            Mapping from feature name to its flattened column.
        """
        short = self.params.shortest_window
        long = self.params.longest_window
        minimum = self.params.min_samples_per_window
        short_std = frame.rolling(window=short, min_periods=minimum).std()
        long_std = frame.rolling(window=long, min_periods=minimum).std()
        ratio = short_std / long_std.replace(0.0, np.nan)
        crossings = frame.rolling(window=short, min_periods=minimum).apply(
            _zero_crossing_rate, raw=True
        )
        return {
            "variance_ratio": self._flat(ratio),
            f"zero_crossing_rate_{short}": self._flat(crossings),
        }


# ---------------------------------------------------------------------------
# Constructores de casos
# ---------------------------------------------------------------------------


def _params(windows: tuple[int, ...], min_samples: int) -> FeatureParams:
    """Build a FeatureParams over the four synthetic sensors.

    Args:
        windows: Rolling windows in samples.
        min_samples: Minimum samples per window.

    Returns:
        The parameters.
    """
    return FeatureParams(
        features_dir=Path("features"),
        window_samples=windows,
        min_samples_per_window=min_samples,
        n_lags=3,
        selection=SensorSelection(
            include_types=frozenset(_TYPES.values()),
            exclude_sensor_ids=frozenset(),
            expected_sensor_count=len(_TYPES),
        ),
        correlation_pairs=(),
    )


def _synthetic(kind: str, length: int, seed: int = 0) -> pd.DataFrame:
    """Build a wide (timestep x sensor) frame that stresses one behaviour.

    Args:
        kind: smooth, nan, constant, stuck_tail or spikes.
        length: Number of timesteps.
        seed: Seed of the generator.

    Returns:
        The wide frame indexed by timestep.
    """
    rng = np.random.default_rng(seed)
    data = np.cumsum(rng.normal(size=(length, len(_TYPES))), axis=0) * 3.0 + 2700.0
    if kind == "nan":
        data[rng.random(data.shape) < 0.12] = np.nan
    elif kind == "constant":
        data[:, 0] = 0.1
        data[:, 1] = 7.0
        data[:, 2] = np.arange(length) * 0.1
    elif kind == "stuck_tail":
        data[length // 2 :, 1] = data[length // 2, 1]
        data[: length // 3, 3] = 1.0
    elif kind == "spikes":
        data[rng.random(data.shape) < 0.06] += 5_000.0
    elif kind != "smooth":
        raise ValueError(f"Unknown kind {kind!r}.")
    columns = list(_TYPES)
    return pd.DataFrame(data, columns=columns, index=pd.RangeIndex(length, name="timestep"))


def _stamps(
    frame: pd.DataFrame, *, irregular: bool = False, tz: str | None = None
) -> pd.Series:
    """Build the timestamp series of a wide frame.

    Args:
        frame: Wide frame.
        irregular: Use random gaps (microsecond resolution, some zero) instead of
            the fixed cadence.
        tz: IANA zone to localise to, or None for naive timestamps.

    Returns:
        Timestamps sharing the frame's index.
    """
    length = len(frame)
    if irregular:
        rng = np.random.default_rng(7)
        gaps = rng.integers(0, 400_000_000, size=length)
        gaps[rng.random(length) < 0.1] = 0
        gaps[0] = 0
        offsets = np.cumsum(gaps)
        moments = [_T0 + timedelta(microseconds=int(offset)) for offset in offsets]
    else:
        moments = [_T0 + timedelta(seconds=_INTERVAL * step) for step in range(length)]
    series = pd.Series(pd.DatetimeIndex(moments), index=frame.index)
    return series.dt.tz_localize(tz) if tz else series


def _assert_identical(
    frame: pd.DataFrame,
    stamps: pd.Series,
    params: FeatureParams,
    types: dict[str, str] | None = None,
) -> None:
    """Assert transform equals the reference, bit for bit.

    Args:
        frame: Wide value frame.
        stamps: Timestamps of the frame.
        params: Feature parameters.
        types: Sensor tag to type.
    """
    sensor_types = types or _TYPES
    current = FeaturePipeline(params, sample_interval_seconds=_INTERVAL)
    reference = _ReferencePipeline(params, sample_interval_seconds=_INTERVAL)
    pd.testing.assert_frame_equal(
        current.transform(frame, stamps, sensor_types),
        reference.transform(frame, stamps, sensor_types),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Casos sinteticos
# ---------------------------------------------------------------------------


class TestSyntheticEquivalence:
    """Los bordes que el dato real del TEP no pisa."""

    @pytest.mark.parametrize("kind", ["smooth", "nan", "constant", "stuck_tail", "spikes"])
    @pytest.mark.parametrize("length", [1, 2, 3, 9, 10, 11, 60, 130])
    def test_every_behaviour_at_every_length(self, kind: str, length: int) -> None:
        """Series mas cortas, iguales y mas largas que las ventanas de params.yaml."""
        frame = _synthetic(kind, length)
        _assert_identical(frame, _stamps(frame), _params((10, 20, 60), 3))

    @pytest.mark.parametrize("windows", [(4, 6), (10, 20), (5,)])
    @pytest.mark.parametrize("min_samples", [1, 3])
    def test_other_windows_and_minimum_samples(
        self, windows: tuple[int, ...], min_samples: int
    ) -> None:
        """Nada depende de que la ventana corta sea 10 ni de que el minimo sea 3."""
        frame = _synthetic("nan", 70, seed=3)
        _assert_identical(frame, _stamps(frame), _params(windows, min_samples))

    @pytest.mark.parametrize("kind", ["smooth", "nan", "spikes"])
    def test_irregular_sampling(self, kind: str) -> None:
        """dt variable, con microsegundos y con pares de timestamps iguales."""
        frame = _synthetic(kind, 80, seed=5)
        _assert_identical(
            frame, _stamps(frame, irregular=True), _params((10, 20, 60), 3)
        )

    @pytest.mark.parametrize("tz", [None, "UTC", "Europe/Madrid"])
    def test_timezones(self, tz: str | None) -> None:
        """El huso no cambia ni un dt ni el hour_of_day."""
        frame = _synthetic("smooth", 50, seed=9)
        _assert_identical(frame, _stamps(frame, tz=tz), _params((10, 20), 3))

    def test_a_long_gap_of_missing_readings(self) -> None:
        """El filtro no avanza durante un hueco y el dt siguiente cruza todo el hueco."""
        frame = _synthetic("smooth", 60, seed=11)
        frame.iloc[20:35, 0] = np.nan
        frame.iloc[:, 2] = np.nan
        _assert_identical(frame, _stamps(frame), _params((10, 20, 60), 3))

    def test_a_column_that_starts_with_missing_readings(self) -> None:
        """El primer valor valido, no el primer paso, inicializa el filtro."""
        frame = _synthetic("smooth", 40, seed=13)
        frame.iloc[:7, 1] = np.nan
        _assert_identical(frame, _stamps(frame), _params((10, 20), 3))

    def test_non_finite_values(self) -> None:
        """Un inf se trata como ausente en el filtro, y no en las ventanas."""
        frame = _synthetic("smooth", 40, seed=15)
        frame.iloc[12, 0] = np.inf
        frame.iloc[25, 3] = -np.inf
        _assert_identical(frame, _stamps(frame), _params((10, 20), 3))

    def test_both_reject_a_timestamp_that_goes_back(self) -> None:
        """Un reloj que retrocede sigue siendo un ValueError, no un residual."""
        frame = _synthetic("smooth", 30, seed=17)
        stamps = _stamps(frame).iloc[::-1].set_axis(frame.index)
        params = _params((10, 20), 3)
        with pytest.raises(ValueError, match="precedes"):
            _ReferencePipeline(params, _INTERVAL).transform(frame, stamps, _TYPES)
        with pytest.raises(ValueError, match="precedes"):
            FeaturePipeline(params, _INTERVAL).transform(frame, stamps, _TYPES)


# ---------------------------------------------------------------------------
# Datos reales
# ---------------------------------------------------------------------------


def _real_wide(path: Path) -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    """Pivot one real TEP partition the way featurize_partition does.

    Args:
        path: Parquet of the run.

    Returns:
        The wide values, the timestamps and the sensor types.
    """
    frame = pd.read_parquet(path)
    wide = frame.pivot(index="timestep", columns="sensor_id", values="value")
    stamps = (
        frame.drop_duplicates(subset="timestep")
        .set_index("timestep")
        .sort_index()
        .loc[:, "timestamp"]
    )
    types = dict(zip(frame["sensor_id"], frame["sensor_type"], strict=True))
    return wide, stamps, types


def _real_params() -> FeatureParams:
    """Load the features: section of params.yaml.

    Returns:
        The resolved parameters.
    """
    from ml.features.feature_params import DEFAULT_PARAMS_PATH, load_feature_params

    return load_feature_params(DEFAULT_PARAMS_PATH)


_needs_tep = pytest.mark.skipif(
    not _NORMAL_PARTITION.exists() or not _STUCK_PARTITION.exists(),
    reason=(
        "El parquet adaptado del TEP no esta poblado. Generalo con "
        ".\\infra\\scripts\\Invoke-Pipeline.ps1."
    ),
)


@_needs_tep
class TestRealDataEquivalence:
    """d00 (operacion normal) y d21 (el unico sensor congelado de verdad)."""

    @pytest.mark.parametrize("path", [_NORMAL_PARTITION, _STUCK_PARTITION])
    def test_whole_partition_is_identical(self, path: Path) -> None:
        """Lo que el stage featurize escribe a disco: 500 y 480 timesteps x 52."""
        wide, stamps, types = _real_wide(path)
        _assert_identical(wide, stamps, _real_params(), types)

    @pytest.mark.parametrize("path", [_NORMAL_PARTITION, _STUCK_PARTITION])
    @pytest.mark.parametrize("start", [0, 1, 137, 420])
    def test_sixty_sample_windows_are_identical(self, path: Path, start: int) -> None:
        """La unidad del criterio 3: una ventana de longest_window x 52 sensores."""
        params = _real_params()
        wide, stamps, types = _real_wide(path)
        rows = wide.index[start : start + params.longest_window]
        _assert_identical(wide.loc[rows], stamps.loc[rows], params, types)


def test_reference_uses_the_documented_process_noise() -> None:
    """El oraculo y el pipeline comparten q: si cambia uno, el otro tambien."""
    reference = _ReferencePipeline(_params((10,), 3), _INTERVAL)
    assert reference.process_noise == FEATURIZER_PROCESS_NOISE
