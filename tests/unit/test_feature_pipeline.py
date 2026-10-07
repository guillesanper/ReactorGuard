"""Unit tests for ml.features.pipeline.

Las comprobaciones son ARITMETICAS donde se puede: una media movil de 10 muestras
se contrasta contra la media exacta de esas 10, y una pendiente sobre una rampa
perfecta contra su pendiente real. Un test que solo comprobase que la columna
existe y no es nula dejaria pasar una ventana desplazada un paso, que es
exactamente el fallo que un featurizer comete en silencio.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from ml.features.feature_params import FeatureParams, SensorSelection
from ml.features.pipeline import FeaturePipeline

_T0 = datetime(2024, 3, 14, 9, 0, 0, tzinfo=UTC)
"""Un jueves a las 09:00 UTC: hour_of_day 9, day_of_week 3."""

_INTERVAL = 180.0
_TAGS = ["SENS-01", "SENS-02", "SENS-03", "POS-01"]
_TYPES = {
    "SENS-01": "flow",
    "SENS-02": "pressure",
    "SENS-03": "thermocouple",
    "POS-01": "position",
}


def _params(
    windows: tuple[int, ...] = (10, 20),
    n_lags: int = 3,
    min_samples: int = 3,
    expected: int = 4,
    pairs: tuple[tuple[str, str], ...] = (),
    include: frozenset[str] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> FeatureParams:
    """Build a FeatureParams directly, without going through YAML.

    Args:
        windows: Rolling windows in samples.
        n_lags: Lags emitted.
        min_samples: Minimum samples per window.
        expected: Sensors the selection must resolve.
        pairs: Correlation pairs.
        include: Sensor types kept; defaults to every type in the fixture.
        exclude: Tags dropped.

    Returns:
        The parameters.
    """
    return FeatureParams(
        features_dir=Path("features"),
        window_samples=windows,
        min_samples_per_window=min_samples,
        n_lags=n_lags,
        selection=SensorSelection(
            include_types=include or frozenset(_TYPES.values()),
            exclude_sensor_ids=exclude,
            expected_sensor_count=expected,
        ),
        correlation_pairs=pairs,
    )


def _wide(values: dict[str, list[float]] | None = None, n: int = 60) -> pd.DataFrame:
    """Build a wide (timestep x sensor) frame.

    Args:
        values: Explicit series per tag. Missing tags get a smooth ramp.
        n: Number of timesteps when values is not given.

    Returns:
        The wide frame indexed by timestep.
    """
    values = values or {}
    length = len(next(iter(values.values()))) if values else n
    data = {
        tag: values.get(tag, [10.0 + index * 0.5 for index in range(length)])
        for tag in _TAGS
    }
    return pd.DataFrame(data, index=pd.RangeIndex(length, name="timestep"))


def _stamps(frame: pd.DataFrame) -> pd.Series:
    """Return the timestamp series matching a wide frame.

    Args:
        frame: Wide frame.

    Returns:
        Timestamps at the TEP cadence, sharing the frame's index.
    """
    return pd.Series(
        [_T0 + timedelta(seconds=_INTERVAL * step) for step in range(len(frame))],
        index=frame.index,
    )


def _run(
    frame: pd.DataFrame, params: FeatureParams | None = None
) -> pd.DataFrame:
    """Transform a wide frame with the default test parameters.

    Args:
        frame: Wide frame.
        params: Parameters, defaulting to _params().

    Returns:
        The long feature frame.
    """
    pipeline = FeaturePipeline(params or _params(), sample_interval_seconds=_INTERVAL)
    return pipeline.transform(frame, _stamps(frame), _TYPES)


def _column(features: pd.DataFrame, tag: str, name: str) -> pd.Series:
    """Return one sensor's feature series, indexed by timestep.

    Args:
        features: Long feature frame.
        tag: Sensor tag.
        name: Feature column.

    Returns:
        The series.
    """
    rows = features[features["sensor_id"] == tag].sort_values("timestep")
    return rows.set_index("timestep")[name]


# ---------------------------------------------------------------------------
# Shape and contract
# ---------------------------------------------------------------------------


class TestShape:
    """La forma de la salida y su correspondencia con feature_names."""

    def test_output_is_long_with_one_row_per_pair(self) -> None:
        """52 sensores no deben producir 52 columnas por feature."""
        frame = _wide(n=30)
        features = _run(frame)
        assert len(features) == 30 * len(_TAGS)

    def test_columns_match_feature_names_exactly(self) -> None:
        """feature_names is the contract Feast and the model rely on."""
        pipeline = FeaturePipeline(_params(), sample_interval_seconds=_INTERVAL)
        frame = _wide(n=30)
        features = pipeline.transform(frame, _stamps(frame), _TYPES)

        assert list(features.columns) == ["timestep", "sensor_id", *pipeline.feature_names()]

    def test_rows_are_ordered_by_timestep_then_sensor(self) -> None:
        """Order is the streaming order and must be reproducible."""
        features = _run(_wide(n=5))
        expected = [
            (step, tag) for step in range(5) for tag in sorted(_TAGS)
        ]
        assert list(zip(features["timestep"], features["sensor_id"], strict=True)) == expected

    def test_value_column_carries_the_original_reading(self) -> None:
        """The featurizer must not silently rescale what it was given."""
        frame = _wide(n=20)
        features = _run(frame)
        assert _column(features, "SENS-01", "value").tolist() == frame["SENS-01"].tolist()


# ---------------------------------------------------------------------------
# Group 1: rolling statistics
# ---------------------------------------------------------------------------


class TestRollingStatistics:
    """Aritmetica exacta de las ventanas moviles."""

    def test_rolling_mean_equals_the_exact_mean_of_the_window(self) -> None:
        """The window must cover the last 10 samples, not 9 or 11."""
        frame = _wide(n=30)
        features = _run(frame)
        series = frame["SENS-01"]

        at_20 = _column(features, "SENS-01", "rolling_mean_10").loc[20]
        assert at_20 == pytest.approx(series.iloc[11:21].mean())

    def test_rolling_std_equals_the_exact_std_of_the_window(self) -> None:
        """Same alignment check on the second moment."""
        frame = _wide(n=30)
        features = _run(frame)
        at_20 = _column(features, "SENS-01", "rolling_std_10").loc[20]
        assert at_20 == pytest.approx(frame["SENS-01"].iloc[11:21].std())

    def test_rolling_min_and_max_bracket_the_window(self) -> None:
        """The extremes must come from the same 10 samples."""
        frame = _wide(n=30)
        features = _run(frame)
        window = frame["SENS-01"].iloc[11:21]

        assert _column(features, "SENS-01", "rolling_min_10").loc[20] == pytest.approx(window.min())
        assert _column(features, "SENS-01", "rolling_max_10").loc[20] == pytest.approx(window.max())

    def test_cells_below_the_minimum_are_null_not_zero(self) -> None:
        """A statistic over two samples is not emitted as if it were solid."""
        features = _run(_wide(n=30))
        early = _column(features, "SENS-01", "rolling_std_10")
        assert pd.isna(early.loc[0])
        assert pd.isna(early.loc[1])
        assert not pd.isna(early.loc[2])

    def test_each_window_produces_its_own_columns(self) -> None:
        """Two windows must not collapse into one."""
        features = _run(_wide(n=40))
        assert "rolling_mean_10" in features.columns
        assert "rolling_mean_20" in features.columns
        assert not _column(features, "SENS-01", "rolling_mean_10").equals(
            _column(features, "SENS-01", "rolling_mean_20")
        )


# ---------------------------------------------------------------------------
# Lags
# ---------------------------------------------------------------------------


class TestLags:
    """Los lags y el rechazo del relleno hacia atras."""

    def test_lag_k_is_the_value_k_steps_earlier(self) -> None:
        """A lag that is off by one is a silent leak of the present."""
        frame = _wide(n=20)
        features = _run(frame)
        assert _column(features, "SENS-01", "lag_1").loc[10] == pytest.approx(
            frame["SENS-01"].iloc[9]
        )
        assert _column(features, "SENS-01", "lag_3").loc[10] == pytest.approx(
            frame["SENS-01"].iloc[7]
        )

    def test_early_lags_are_null_and_not_backfilled(self) -> None:
        """extractor.py rellenaba con bfill, que copia un valor del FUTURO.

        En una serie temporal eso es fuga de informacion: la fila 0 llevaria el
        valor de la fila 3 como si fuese su pasado.
        """
        features = _run(_wide(n=20))
        assert pd.isna(_column(features, "SENS-01", "lag_3").loc[0])
        assert pd.isna(_column(features, "SENS-01", "lag_3").loc[2])

    def test_lag_count_follows_the_parameter(self) -> None:
        """n_lags is configuration, not a constant."""
        features = _run(_wide(n=20), _params(n_lags=1))
        assert "lag_1" in features.columns
        assert "lag_2" not in features.columns


# ---------------------------------------------------------------------------
# Group 2: rates
# ---------------------------------------------------------------------------


class TestRates:
    """Tasas expresadas por segundo, no por muestra."""

    def test_rate_of_change_is_per_second(self) -> None:
        """A step of 0.5 units every 180 s is 0.5/180 units per second."""
        features = _run(_wide(n=20))
        assert _column(features, "SENS-01", "rate_of_change").loc[10] == pytest.approx(
            0.5 / _INTERVAL
        )

    def test_slope_of_a_perfect_ramp_equals_its_real_slope(self) -> None:
        """A least-squares slope over a straight line is that line's slope."""
        features = _run(_wide(n=30))
        assert _column(features, "SENS-01", "slope_10").loc[25] == pytest.approx(
            0.5 / _INTERVAL, rel=1e-9
        )

    def test_slope_of_a_flat_signal_is_zero(self) -> None:
        """A constant channel has no trend."""
        features = _run(_wide({"SENS-01": [7.0] * 30}))
        assert _column(features, "SENS-01", "slope_10").loc[25] == pytest.approx(0.0)

    def test_slope_sign_follows_the_direction(self) -> None:
        """A falling channel must report a negative slope."""
        falling = [100.0 - index for index in range(30)]
        features = _run(_wide({"SENS-01": falling}))
        assert _column(features, "SENS-01", "slope_10").loc[25] < 0.0

    def test_partial_windows_use_their_own_length(self) -> None:
        """Truncating the long window's weights would mis-centre the regression.

        En las primeras muestras pandas entrega bloques mas cortos que la ventana
        configurada. La pendiente de esos bloques sigue siendo exacta sobre una
        rampa perfecta, que es lo que se comprueba.
        """
        features = _run(_wide(n=30))
        assert _column(features, "SENS-01", "slope_10").loc[4] == pytest.approx(
            0.5 / _INTERVAL, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Group 3: Kalman
# ---------------------------------------------------------------------------


class TestKalmanFeatures:
    """El residual normalizado y su incertidumbre."""

    def test_residual_spikes_on_an_abrupt_jump(self) -> None:
        """The whole point of the feature."""
        series = [50.0] * 30 + [95.0] + [50.0] * 9
        features = _run(_wide({"SENS-01": series}))
        residual = _column(features, "SENS-01", "kalman_residual")

        assert abs(residual.loc[30]) > 3.0
        assert abs(residual.loc[20]) < 3.0

    def test_residual_stays_quiet_on_a_steady_signal(self) -> None:
        """A constant channel must not look anomalous."""
        features = _run(_wide({"SENS-01": [42.0] * 40}))
        residual = _column(features, "SENS-01", "kalman_residual")
        assert residual.iloc[5:].abs().max() < 3.0

    def test_uncertainty_is_positive(self) -> None:
        """An innovation sigma is a standard deviation."""
        features = _run(_wide(n=30))
        assert (_column(features, "SENS-01", "kalman_uncertainty") > 0.0).all()

    def test_filters_are_independent_per_sensor(self) -> None:
        """A jump on one tag must not raise another's residual."""
        features = _run(
            _wide({"SENS-01": [50.0] * 30 + [95.0] * 10, "SENS-02": [20.0] * 40})
        )
        assert abs(_column(features, "SENS-01", "kalman_residual").loc[30]) > 3.0
        assert abs(_column(features, "SENS-02", "kalman_residual").loc[30]) < 1.0

    def test_a_missing_reading_is_not_imputed(self) -> None:
        """Inventing the number would fabricate the evidence the residual measures."""
        series = [50.0] * 20
        series[10] = float("nan")
        features = _run(_wide({"SENS-01": series}))
        assert pd.isna(_column(features, "SENS-01", "kalman_residual").loc[10])


# ---------------------------------------------------------------------------
# Group 5: distribution
# ---------------------------------------------------------------------------


class TestDistribution:
    """Ratio de volatilidad y cruces por la media."""

    def test_variance_ratio_exceeds_one_on_a_recent_volatility_spike(self) -> None:
        """Short-term volatility above the long-term baseline is what it measures."""
        calm = [50.0 + 0.01 * (index % 2) for index in range(40)]
        volatile = [50.0 + 8.0 * (-1) ** index for index in range(10)]
        features = _run(_wide({"SENS-01": calm + volatile}))
        assert _column(features, "SENS-01", "variance_ratio").iloc[-1] > 1.0

    def test_variance_ratio_is_null_when_the_long_window_is_flat(self) -> None:
        """Dividing by zero would give inf and poison any downstream scaling."""
        features = _run(_wide({"SENS-01": [5.0] * 40}))
        assert pd.isna(_column(features, "SENS-01", "variance_ratio").iloc[-1])

    def test_zero_crossing_rate_is_zero_for_a_constant_signal(self) -> None:
        """A flat window never crosses its own mean."""
        features = _run(_wide({"SENS-01": [3.0] * 30}))
        assert _column(features, "SENS-01", "zero_crossing_rate_10").iloc[-1] == 0.0

    def test_zero_crossing_rate_is_maximal_for_an_alternating_signal(self) -> None:
        """A signal alternating each sample crosses on every transition."""
        alternating = [50.0 + 5.0 * (-1) ** index for index in range(30)]
        features = _run(_wide({"SENS-01": alternating}))
        assert _column(features, "SENS-01", "zero_crossing_rate_10").iloc[-1] == pytest.approx(1.0)

    def test_zero_crossing_rate_is_low_for_a_monotonic_ramp(self) -> None:
        """A ramp crosses its window mean once, not on every sample."""
        features = _run(_wide(n=30))
        assert _column(features, "SENS-01", "zero_crossing_rate_10").iloc[-1] < 0.3

    def test_crossings_are_counted_against_the_window_mean(self) -> None:
        """Counting against absolute zero would make the feature constant.

        Las senales del TEP viven lejos del cero: una presion de 2.700 kPa no lo
        cruza jamas, de modo que contra el cero la feature no distinguiria nada.
        """
        far_from_zero = [2700.0 + 5.0 * (-1) ** index for index in range(30)]
        features = _run(_wide({"SENS-01": far_from_zero}))
        assert _column(features, "SENS-01", "zero_crossing_rate_10").iloc[-1] > 0.9


# ---------------------------------------------------------------------------
# Group 6: fault indicators
# ---------------------------------------------------------------------------


class TestFaultIndicators:
    """La longitud de tirada como indicador de congelamiento."""

    def test_run_length_grows_while_the_value_repeats(self) -> None:
        """Ten identical readings must report a run of ten at the last one."""
        features = _run(_wide({"SENS-01": [1.0, 2.0] + [9.0] * 10 + [3.0] * 8}))
        runs = _column(features, "SENS-01", "stuck_run_length")
        assert runs.loc[11] == 10.0

    def test_run_length_resets_when_the_value_changes(self) -> None:
        """A single different reading breaks the run."""
        features = _run(_wide({"SENS-01": [9.0] * 10 + [4.0] + [9.0] * 9}))
        assert _column(features, "SENS-01", "stuck_run_length").loc[10] == 1.0

    def test_run_length_starts_at_one(self) -> None:
        """The first sample is a run of one, not of zero."""
        features = _run(_wide(n=20))
        assert _column(features, "SENS-01", "stuck_run_length").loc[0] == 1.0

    def test_a_varying_channel_keeps_a_run_of_one(self) -> None:
        """A ramp never repeats a value."""
        features = _run(_wide(n=30))
        assert (_column(features, "SENS-01", "stuck_run_length") == 1.0).all()


# ---------------------------------------------------------------------------
# Group 7: context
# ---------------------------------------------------------------------------


class TestContext:
    """Contexto operacional, identico para todos los sensores de un timestep."""

    def test_hour_and_day_come_from_the_timestamp(self) -> None:
        """_T0 is a Thursday at 09:00 UTC."""
        features = _run(_wide(n=5))
        assert _column(features, "SENS-01", "hour_of_day").loc[0] == 9.0
        assert _column(features, "SENS-01", "day_of_week").loc[0] == 3.0

    def test_the_timezone_is_not_dropped(self) -> None:
        """Converting through numpy would discard the offset silently."""
        frame = _wide(n=5)
        madrid = _stamps(frame).dt.tz_convert("Europe/Madrid")
        pipeline = FeaturePipeline(_params(), sample_interval_seconds=_INTERVAL)
        features = pipeline.transform(frame, madrid, _TYPES)

        # 09:00 UTC en marzo son las 10:00 en Madrid.
        assert _column(features, "SENS-01", "hour_of_day").loc[0] == 10.0

    def test_power_level_is_the_mean_of_the_position_sensors(self) -> None:
        """Only POS-01 carries the position type in the fixture."""
        features = _run(_wide({"POS-01": [61.0] * 20}))
        assert _column(features, "SENS-01", "power_level_pct").loc[5] == pytest.approx(61.0)

    def test_context_is_identical_across_sensors_of_a_timestep(self) -> None:
        """It describes the plant, not the instrument."""
        features = _run(_wide(n=10))
        at_step = features[features["timestep"] == 5]
        assert at_step["power_level_pct"].nunique() == 1
        assert at_step["hour_of_day"].nunique() == 1

    def test_power_level_is_null_without_position_sensors(self) -> None:
        """Averaging an empty set must not silently produce zero."""
        params = _params(include=frozenset({"flow", "pressure", "thermocouple"}), expected=3)
        features = _run(_wide(n=10), params)
        assert features["power_level_pct"].isna().all()


# ---------------------------------------------------------------------------
# Group 4: correlation
# ---------------------------------------------------------------------------


class TestCorrelation:
    """Correlacion movil de las parejas configuradas."""

    def test_no_pairs_means_no_columns(self) -> None:
        """The default state emits nothing, as in CrossCorrelationChecker."""
        features = _run(_wide(n=20))
        assert not [c for c in features.columns if c.startswith("cross_correlation")]

    def test_a_configured_pair_produces_its_column(self) -> None:
        """Two identical ramps correlate at 1."""
        params = _params(pairs=(("SENS-01", "SENS-02"),))
        features = _run(_wide(n=30), params)
        column = _column(features, "SENS-01", "cross_correlation_SENS-01__SENS-02")
        assert column.iloc[-1] == pytest.approx(1.0)

    def test_an_anticorrelated_pair_reports_minus_one(self) -> None:
        """The sign must survive."""
        rising = [float(index) for index in range(30)]
        falling = [-float(index) for index in range(30)]
        params = _params(pairs=(("SENS-01", "SENS-02"),))
        features = _run(_wide({"SENS-01": rising, "SENS-02": falling}), params)
        column = _column(features, "SENS-01", "cross_correlation_SENS-01__SENS-02")
        assert column.iloc[-1] == pytest.approx(-1.0)

    def test_an_unknown_tag_in_a_pair_is_rejected(self) -> None:
        """A typo in params.yaml must not silently emit a column of nulls."""
        params = _params(pairs=(("SENS-01", "NOPE"),))
        with pytest.raises(KeyError, match="NOPE"):
            _run(_wide(n=20), params)


# ---------------------------------------------------------------------------
# Sensor selection
# ---------------------------------------------------------------------------


class TestSensorSelection:
    """La resolucion del conjunto de sensores desde los datos."""

    def test_the_expected_count_is_enforced(self) -> None:
        """Schema drift must fail loudly, not propagate into the features."""
        pipeline = FeaturePipeline(_params(expected=99), sample_interval_seconds=_INTERVAL)
        with pytest.raises(ValueError, match="expected_sensor_count"):
            pipeline.resolve_sensors(_TYPES)

    def test_excluded_tags_are_dropped(self) -> None:
        """An excluded tag must not reach the output."""
        params = _params(exclude=frozenset({"POS-01"}), expected=3)
        features = _run(_wide(n=10), params)
        assert "POS-01" not in set(features["sensor_id"])

    def test_types_outside_the_policy_are_dropped(self) -> None:
        """include_types is what selects, not the tag prefix."""
        params = _params(include=frozenset({"flow"}), expected=1)
        features = _run(_wide(n=10), params)
        assert set(features["sensor_id"]) == {"SENS-01"}

    def test_resolution_is_sorted(self) -> None:
        """Column order must not depend on dictionary insertion order."""
        pipeline = FeaturePipeline(_params(), sample_interval_seconds=_INTERVAL)
        assert pipeline.resolve_sensors(_TYPES) == sorted(_TAGS)


# ---------------------------------------------------------------------------
# Construction and guards
# ---------------------------------------------------------------------------


class TestGuards:
    """Lo que el pipeline rechaza."""

    def test_a_non_positive_interval_is_rejected(self) -> None:
        """Every rate feature divides by it."""
        with pytest.raises(ValueError, match="sample_interval_seconds"):
            FeaturePipeline(_params(), sample_interval_seconds=0.0)

    def test_misaligned_timestamps_are_rejected(self) -> None:
        """A silent misalignment would date every feature wrongly."""
        frame = _wide(n=10)
        pipeline = FeaturePipeline(_params(), sample_interval_seconds=_INTERVAL)
        with pytest.raises(ValueError, match="same timestep index"):
            pipeline.transform(frame, _stamps(frame).iloc[:5], _TYPES)

    def test_a_missing_timestamp_is_rejected(self) -> None:
        """A NaT would date a Kalman step wrongly, so it fails instead of filtering."""
        frame = _wide(n=10)
        stamps = _stamps(frame)
        stamps.iloc[3] = pd.NaT
        pipeline = FeaturePipeline(_params(), sample_interval_seconds=_INTERVAL)
        with pytest.raises(ValueError, match="NaT"):
            pipeline.transform(frame, stamps, _TYPES)

    def test_an_empty_frame_is_rejected(self) -> None:
        """There is nothing to featurise."""
        empty = pd.DataFrame({tag: [] for tag in _TAGS}, index=pd.RangeIndex(0))
        pipeline = FeaturePipeline(_params(), sample_interval_seconds=_INTERVAL)
        with pytest.raises(ValueError, match="empty frame"):
            pipeline.transform(empty, pd.Series([], dtype="datetime64[ns, UTC]"), _TYPES)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_a_thousand_readings_take_under_a_second() -> None:
    """El objetivo de rendimiento del TDD, medido sobre el formato real.

    250 timesteps x 4 sensores son 1.000 lecturas. El presupuesto de la Fase 2
    para el pipeline de features es de 100 ms por ventana en linea; este es el
    camino por lotes, mas holgado, pero un featurizer que tarde segundos por cada
    mil lecturas no llega a las 550.160 del dataset en un tiempo razonable.
    """
    frame = _wide(n=250)
    pipeline = FeaturePipeline(_params(), sample_interval_seconds=_INTERVAL)
    stamps = _stamps(frame)

    started = time.perf_counter()
    features = pipeline.transform(frame, stamps, _TYPES)
    elapsed = time.perf_counter() - started

    assert len(features) == 1000
    assert elapsed < 1.0, f"1.000 lecturas tardaron {elapsed:.3f}s"
