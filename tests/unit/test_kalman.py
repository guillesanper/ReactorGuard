"""Unit tests for ml.features.kalman.

Un filtro de Kalman falla de forma silenciosa: no lanza excepciones, simplemente
converge mal, y aguas abajo eso se traduce en un detector que no marca nada o que
lo marca todo. Por eso los tests son de comportamiento medido (converge, detecta
un escalon, se recupera de una divergencia) y no de forma, y por eso comprueban
tambien las propiedades numericas de las que depende la deteccion: sigma finita y
positiva, covarianza simetrica, residual normalizado adimensional.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.typing as npt
import pytest

from data.schemas.sensor_reading import SensorReading
from ml.features.kalman import (
    DEFAULT_K_SIGMA,
    DEFAULT_OBSERVATION_NOISE,
    DEFAULT_PROCESS_NOISE,
    DEFAULT_RESET_SIGMA,
    INITIAL_COVARIANCE,
    KalmanFilterBank,
    KalmanResult,
    OnlineKalmanFilter,
    filter_columns,
)

_T0 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
_DT = timedelta(seconds=1)


def _make_reading(
    sensor_id: str = "TC-CORE-12",
    value: float | None = 300.0,
    timestamp: datetime = _T0,
) -> SensorReading:
    """Build a valid SensorReading around the given value.

    Args:
        sensor_id: Instrument tag.
        value: Engineering value, or None to exercise the valueless path.
        timestamp: Measurement time.

    Returns:
        A validated SensorReading.
    """
    return SensorReading.model_validate(
        {
            "reading_id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "plant_id": "REACTOR-01",
            "sensor": {
                "id": sensor_id,
                "type": "thermocouple",
                "location": "core",
                "elevation_m": 3.45,
            },
            "measurement": {
                "value": value,
                "unit": "celsius",
                "quality": "good",
                "raw_counts": 4092,
            },
            "metadata": {
                "calibration_date": "2024-01-01",
                "last_maintenance": "2023-12-15",
                "drift_coefficient": 0.0012,
            },
        }
    )


def _feed(
    filter_: OnlineKalmanFilter,
    values: list[float],
    start: datetime = _T0,
    step: timedelta = _DT,
) -> list[KalmanResult]:
    """Feed a series of values at a fixed cadence and return every result.

    Args:
        filter_: Filter to drive.
        values: Measurements in chronological order.
        start: Timestamp of the first measurement.
        step: Interval between measurements.

    Returns:
        One KalmanResult per value, in order.
    """
    return [
        filter_.step(value, start + step * index) for index, value in enumerate(values)
    ]


class TestConstruction:
    """Invalid tuning must be rejected at construction, not mid-stream."""

    def test_starts_uninitialized(self) -> None:
        """A fresh filter must not claim to hold a state."""
        assert not OnlineKalmanFilter().is_initialized

    def test_negative_process_noise(self) -> None:
        """A negative Q density is not a covariance."""
        with pytest.raises(ValueError, match="process_noise"):
            OnlineKalmanFilter(process_noise=-0.1)

    @pytest.mark.parametrize("observation_noise", [0.0, -1.0])
    def test_non_positive_observation_noise(self, observation_noise: float) -> None:
        """R must be strictly positive or the innovation variance goes singular."""
        with pytest.raises(ValueError, match="observation_noise"):
            OnlineKalmanFilter(observation_noise=observation_noise)

    @pytest.mark.parametrize("k_sigma", [0.0, -3.0])
    def test_non_positive_k_sigma(self, k_sigma: float) -> None:
        """A non-positive alarm threshold would flag every step."""
        with pytest.raises(ValueError, match="k_sigma"):
            OnlineKalmanFilter(k_sigma=k_sigma)

    def test_reset_sigma_below_k_sigma(self) -> None:
        """Resetting below the alarm threshold makes anomalies unreportable."""
        with pytest.raises(ValueError, match="reset_sigma"):
            OnlineKalmanFilter(k_sigma=3.0, reset_sigma=1.0)

    def test_reset_sigma_may_equal_k_sigma(self) -> None:
        """The degenerate but coherent case must be allowed."""
        assert OnlineKalmanFilter(k_sigma=3.0, reset_sigma=3.0).reset_sigma == 3.0


class TestFirstReading:
    """The first measurement has nothing to be predicted from."""

    def test_first_step_reports_zero_residual(self) -> None:
        """No prediction existed, so no deviation can be claimed."""
        result = OnlineKalmanFilter().step(300.0, _T0)
        assert result.residual == 0.0
        assert result.normalized_residual == 0.0

    def test_first_step_is_not_an_anomaly(self) -> None:
        """A sensor must never be flagged on the strength of its first reading."""
        assert not OnlineKalmanFilter().step(1e6, _T0).is_anomaly

    def test_first_step_reports_uninitialized(self) -> None:
        """is_initialized False marks the result as not a filtered deviation."""
        assert not OnlineKalmanFilter().step(300.0, _T0).is_initialized

    def test_first_step_seeds_the_state(self) -> None:
        """The estimate must start at the measurement, not at zero."""
        filter_ = OnlineKalmanFilter()
        filter_.step(300.0, _T0)
        assert filter_.is_initialized
        assert filter_.value == pytest.approx(300.0)

    def test_second_step_is_initialized(self) -> None:
        """From the second reading on, residuals are meaningful."""
        filter_ = OnlineKalmanFilter()
        results = _feed(filter_, [300.0, 300.1])
        assert results[1].is_initialized


class TestConvergence:
    """On a stationary noisy signal the filter must settle and stay quiet."""

    def test_residual_settles_within_two_sigma(self) -> None:
        """After 50 steps the residuals must sit inside 2 sigma.

        Es la condicion que hace utilizable el umbral de 3 sigma: si en regimen
        normal el filtro ya roza los 3 sigma, el detector vive en falso positivo.
        """
        rng = np.random.default_rng(42)
        values = [300.0 + float(rng.normal(0.0, 1.0)) for _ in range(50)]
        results = _feed(OnlineKalmanFilter(), values)

        settled = results[10:]
        assert all(abs(r.normalized_residual) < 2.0 for r in settled)

    def test_estimate_denoises_the_true_level(self) -> None:
        """The estimate must sit closer to the truth than a raw sample does.

        Se mide sobre 40 semillas y no sobre una, porque en una sola el error
        final es una variable aleatoria y cualquier tolerancia puntual seria una
        constante magica ajustada a esa semilla. El modelo de velocidad constante
        deambula sobre una senal estacionaria (su componente de velocidad absorbe
        ruido), asi que la afirmacion honesta no es que acierte el valor exacto
        sino que filtra: error medio muy por debajo del sigma del ruido.

        Medido con q=0.001 y ruido sigma=1.0: error medio 0,317, maximo 1,018.
        """
        errors = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            values = [300.0 + float(rng.normal(0.0, 1.0)) for _ in range(100)]
            results = _feed(OnlineKalmanFilter(process_noise=0.001), values)
            errors.append(abs(results[-1].corrected_value - 300.0))

        assert float(np.mean(errors)) < 0.6
        assert max(errors) < 1.5

    def test_no_anomalies_on_a_clean_signal(self) -> None:
        """A well-behaved sensor must produce zero alarms.

        Es el analogo por sensor de los 0,000% de falsos positivos sobre d00.
        """
        rng = np.random.default_rng(1234)
        values = [300.0 + float(rng.normal(0.0, 1.0)) for _ in range(200)]
        results = _feed(OnlineKalmanFilter(), values)
        assert not any(r.is_anomaly for r in results)

    def test_innovation_sigma_shrinks_as_evidence_accumulates(self) -> None:
        """Uncertainty must fall from the vague prior towards the noise floor."""
        results = _feed(OnlineKalmanFilter(), [300.0] * 30)
        assert results[1].innovation_sigma > results[-1].innovation_sigma

    def test_tracks_a_constant_ramp(self) -> None:
        """A constant-velocity signal is exactly the model; it must be tracked."""
        values = [300.0 + 2.0 * i for i in range(60)]
        results = _feed(OnlineKalmanFilter(), values)
        assert abs(results[-1].residual) < 0.5
        assert not results[-1].is_anomaly


class TestStepChange:
    """An abrupt jump is the signal the detector exists to catch."""

    def test_step_change_exceeds_k_sigma(self) -> None:
        """A 50-unit jump on a settled filter must exceed 3 sigma."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 30)
        jumped = filter_.step(350.0, _T0 + _DT * 30)

        assert abs(jumped.normalized_residual) > DEFAULT_K_SIGMA
        assert jumped.is_anomaly

    def test_residual_sign_follows_the_jump_direction(self) -> None:
        """A drop must give a negative residual, a rise a positive one."""
        rising = OnlineKalmanFilter()
        _feed(rising, [300.0] * 30)
        assert rising.step(350.0, _T0 + _DT * 30).residual > 0

        falling = OnlineKalmanFilter()
        _feed(falling, [300.0] * 30)
        assert falling.step(250.0, _T0 + _DT * 30).residual < 0

    def test_filter_recovers_after_the_jump(self) -> None:
        """Once the new level holds, the filter must stop alarming."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 30)
        after = _feed(filter_, [350.0] * 25, start=_T0 + _DT * 30)
        assert not after[-1].is_anomaly


class TestGradualDrift:
    """A linear drift lives in the model's null space. These tests pin that down.

    El spec de T4.1 esperaba que una deriva lineal hiciera crecer el residual.
    Medido, ocurre lo contrario: el modelo de velocidad constante APRENDE la
    pendiente y el residual decae a cero. Una rampa de 0,5 u/paso durante 100
    pasos (50 unidades de deriva total) da un maximo de 1,21 sigmas y cero pasos
    marcados. Lo que el filtro ve no es la deriva sino su ARRANQUE: el cambio de
    velocidad. Estos tests fijan ese comportamiento real para que el
    KalmanResidualDetector de T3.4 se disene sobre lo que el filtro hace y no
    sobre lo que se suponia que hacia.
    """

    def test_residual_peaks_at_the_onset_and_decays(self) -> None:
        """The transient is at the start of the ramp, not at its end."""
        filter_ = OnlineKalmanFilter(process_noise=0.001)
        _feed(filter_, [300.0] * 30)

        drifting = _feed(
            filter_,
            [300.0 + 0.5 * i for i in range(1, 101)],
            start=_T0 + _DT * 30,
        )

        early = float(np.mean([abs(r.residual) for r in drifting[:10]]))
        late = float(np.mean([abs(r.residual) for r in drifting[-10:]]))
        assert early > late

    def test_established_linear_drift_is_absorbed(self) -> None:
        """Once the slope is learned the residual is effectively zero.

        Esta es la limitacion, escrita como assert para que no se descubra en
        produccion: un transmisor derivando linealmente se vuelve invisible a
        este detector en cuanto el filtro engancha la pendiente.
        """
        filter_ = OnlineKalmanFilter(process_noise=0.001)
        _feed(filter_, [300.0] * 30)
        drifting = _feed(
            filter_, [300.0 + 0.5 * i for i in range(1, 101)], start=_T0 + _DT * 30
        )

        assert not any(r.is_anomaly for r in drifting)
        assert abs(drifting[-1].residual) < 1e-3
        assert filter_.velocity == pytest.approx(0.5, abs=0.01)

    def test_change_in_slope_is_visible(self) -> None:
        """What the filter does see is curvature: a slope that changes.

        Complemento del test anterior: la deteccion de deriva por Kalman existe,
        pero opera sobre el cambio de pendiente, no sobre la pendiente.
        """
        filter_ = OnlineKalmanFilter(process_noise=0.001)
        _feed(filter_, [300.0 + 0.5 * i for i in range(60)])

        value = 300.0 + 0.5 * 59
        broken = []
        for i in range(1, 11):
            value -= 3.0
            broken.append(filter_.step(value, _T0 + _DT * (59 + i)))

        assert max(abs(r.normalized_residual) for r in broken) > DEFAULT_K_SIGMA

    def test_slow_drift_does_not_trip_the_reset(self) -> None:
        """A drift must be reportable, not discarded as a discontinuity."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 30)
        drifting = _feed(
            filter_, [300.0 + 0.01 * i for i in range(1, 101)], start=_T0 + _DT * 30
        )
        assert all(r.is_initialized for r in drifting)


class TestAutoReset:
    """A wild value must not poison the state for the following minutes."""

    def test_divergent_value_triggers_the_reset(self) -> None:
        """Beyond reset_sigma the step must report an uninitialised anomaly."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 30)
        wild = filter_.step(1e6, _T0 + _DT * 30)

        assert wild.is_anomaly
        assert not wild.is_initialized
        assert abs(wild.normalized_residual) > filter_.reset_sigma

    def test_state_reseeds_on_the_divergent_value(self) -> None:
        """After the reset the estimate must sit on the new measurement."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 30)
        filter_.step(1e6, _T0 + _DT * 30)
        assert filter_.value == pytest.approx(1e6)

    def test_recovers_immediately_after_the_reset(self) -> None:
        """The step following a reset must already be quiet at the new level."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 30)
        filter_.step(1e6, _T0 + _DT * 30)
        following = _feed(filter_, [1e6] * 5, start=_T0 + _DT * 31)
        assert not any(r.is_anomaly for r in following)

    def test_moderate_anomaly_does_not_reset(self) -> None:
        """Between k_sigma and reset_sigma the state must be kept.

        Si un pico de 4 sigma reiniciase el filtro, la deteccion se perderia:
        cada anomalia borraria la referencia que permite ver la siguiente.
        """
        filter_ = OnlineKalmanFilter(k_sigma=3.0, reset_sigma=10.0)
        _feed(filter_, [300.0] * 30)

        sigma = filter_.step(300.0, _T0 + _DT * 30).innovation_sigma
        spiked = filter_.step(300.0 + 4.0 * sigma, _T0 + _DT * 31)

        assert spiked.is_anomaly
        assert spiked.is_initialized

    def test_explicit_reset_clears_the_clock(self) -> None:
        """After reset() the next step must behave as a first reading."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 10)
        filter_.reset()

        assert not filter_.is_initialized
        assert not filter_.step(500.0, _T0).is_initialized


class TestIrregularSampling:
    """Real streams do not tick on a metronome."""

    def test_irregular_intervals_are_handled(self) -> None:
        """Varying gaps must produce finite, usable results."""
        filter_ = OnlineKalmanFilter()
        timestamp = _T0
        for seconds in [1.0, 0.2, 7.5, 0.05, 60.0, 3.0]:
            timestamp += timedelta(seconds=seconds)
            result = filter_.step(300.0, timestamp)
            assert np.isfinite(result.normalized_residual)
            assert result.innovation_sigma > 0.0

    def test_zero_dt_is_allowed(self) -> None:
        """Two readings sharing a timestamp must not break the filter."""
        filter_ = OnlineKalmanFilter()
        filter_.step(300.0, _T0)
        result = filter_.step(300.5, _T0)
        assert np.isfinite(result.normalized_residual)

    def test_large_gap_widens_the_uncertainty(self) -> None:
        """A long silence must make the filter less sure, not more.

        Q crece con dt, asi que tras un hueco largo el mismo residual debe pesar
        menos en sigmas: es lo que evita alarmar por una reanudacion normal.
        """
        tight = OnlineKalmanFilter()
        _feed(tight, [300.0] * 20)
        tight_sigma = tight.step(300.0, _T0 + _DT * 20).innovation_sigma

        wide = OnlineKalmanFilter()
        _feed(wide, [300.0] * 20)
        wide_sigma = wide.step(300.0, _T0 + timedelta(hours=1)).innovation_sigma

        assert wide_sigma > tight_sigma

    def test_out_of_order_timestamp_is_rejected(self) -> None:
        """A backwards step is a caller bug; it must not be filtered silently."""
        filter_ = OnlineKalmanFilter()
        filter_.step(300.0, _T0 + _DT * 10)
        with pytest.raises(ValueError, match="precedes"):
            filter_.step(300.0, _T0)

    def test_mixing_aware_and_naive_timestamps_is_rejected(self) -> None:
        """The failure must name the cause, not surface as a TypeError."""
        filter_ = OnlineKalmanFilter()
        filter_.step(300.0, _T0)
        with pytest.raises(ValueError, match="timezone-aware and naive"):
            filter_.step(300.0, datetime(2024, 1, 15, 12, 0, 1))


class TestPredictUpdateContract:
    """The primitives must refuse to run without a state."""

    def test_predict_before_initialize(self) -> None:
        """predict() on a fresh filter must raise, not read zeros."""
        with pytest.raises(RuntimeError, match="initialize"):
            OnlineKalmanFilter().predict(1.0)

    def test_update_before_initialize(self) -> None:
        """update() on a fresh filter must raise."""
        with pytest.raises(RuntimeError, match="initialize"):
            OnlineKalmanFilter().update(300.0)

    def test_negative_dt_is_rejected(self) -> None:
        """A negative dt would produce a non-positive-definite Q."""
        filter_ = OnlineKalmanFilter()
        filter_.initialize(300.0)
        with pytest.raises(ValueError, match="non-negative"):
            filter_.predict(-1.0)

    def test_initialize_resets_the_covariance(self) -> None:
        """Re-initialising must restore the vague prior, not keep a tight one."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 30)
        filter_.initialize(500.0)
        _, sigma = filter_.predict(0.0)
        assert sigma == pytest.approx(np.sqrt(INITIAL_COVARIANCE))

    def test_predict_returns_the_projected_position(self) -> None:
        """With a known velocity the projection must be exact."""
        filter_ = OnlineKalmanFilter()
        filter_.initialize(100.0, initial_velocity=2.0)
        predicted, _ = filter_.predict(3.0)
        assert predicted == pytest.approx(106.0)

    def test_covariance_stays_symmetric(self) -> None:
        """An asymmetric P eventually yields NaN sigmas; it must not drift."""
        filter_ = OnlineKalmanFilter()
        rng = np.random.default_rng(99)
        _feed(filter_, [300.0 + float(rng.normal(0, 1)) for _ in range(300)])
        covariance = np.array(filter_.get_state()["covariance"])
        assert np.allclose(covariance, covariance.T)

    def test_velocity_is_estimated_on_a_ramp(self) -> None:
        """The second state component must recover the true rate of change."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0 + 2.0 * i for i in range(80)])
        assert filter_.velocity == pytest.approx(2.0, abs=0.2)


class TestKalmanResultShape:
    """The result must be a usable, immutable record."""

    def test_is_frozen(self) -> None:
        """No consumer may rewrite a residual after the fact."""
        result = OnlineKalmanFilter().step(300.0, _T0)
        with pytest.raises(AttributeError):
            result.residual = 1.0  # type: ignore[misc]

    def test_normalized_residual_matches_its_definition(self) -> None:
        """normalized_residual must be residual over innovation_sigma."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 20)
        result = filter_.step(305.0, _T0 + _DT * 20)
        assert result.normalized_residual == pytest.approx(
            result.residual / result.innovation_sigma
        )

    def test_residual_matches_its_definition(self) -> None:
        """residual must be measurement minus predicted_value."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 20)
        result = filter_.step(305.0, _T0 + _DT * 20)
        assert result.residual == pytest.approx(305.0 - result.predicted_value)

    def test_correction_moves_towards_the_measurement(self) -> None:
        """The corrected value must lie between prediction and measurement."""
        filter_ = OnlineKalmanFilter()
        _feed(filter_, [300.0] * 20)
        result = filter_.step(305.0, _T0 + _DT * 20)
        assert result.predicted_value < result.corrected_value <= 305.0


class TestKalmanFilterBank:
    """Sensors must be filtered independently and created on demand."""

    def test_creates_filters_lazily(self) -> None:
        """A bank must start empty and grow with the tags it sees."""
        bank = KalmanFilterBank()
        assert len(bank) == 0

        bank.process(_make_reading(sensor_id="TC-01"))
        bank.process(_make_reading(sensor_id="TC-02"))

        assert len(bank) == 2
        assert "TC-01" in bank
        assert "TC-03" not in bank

    def test_reuses_the_filter_of_a_known_sensor(self) -> None:
        """A second reading must not create a second filter."""
        bank = KalmanFilterBank()
        first = bank.get_filter("TC-01")
        assert bank.get_filter("TC-01") is first
        assert len(bank) == 1

    def test_sensors_are_independent(self) -> None:
        """A jump on one tag must not raise the residual of another.

        Es el motivo de que haya un filtro por sensor y no uno por planta.
        """
        bank = KalmanFilterBank()
        timestamp = _T0
        for _ in range(30):
            bank.process(_make_reading("TC-01", 300.0, timestamp))
            bank.process(_make_reading("TC-02", 50.0, timestamp))
            timestamp += _DT

        spiked = bank.process(_make_reading("TC-01", 900.0, timestamp))
        quiet = bank.process(_make_reading("TC-02", 50.0, timestamp))

        assert spiked.is_anomaly
        assert not quiet.is_anomaly

    def test_tuning_reaches_the_created_filters(self) -> None:
        """Bank-level tuning must not be silently dropped."""
        bank = KalmanFilterBank(process_noise=0.5, observation_noise=2.0, k_sigma=4.0)
        created = bank.get_filter("TC-01")
        assert created.process_noise == 0.5
        assert created.observation_noise == 2.0
        assert created.k_sigma == 4.0

    def test_invalid_tuning_fails_at_construction(self) -> None:
        """The bank must reject bad tuning before any reading arrives."""
        with pytest.raises(ValueError, match="observation_noise"):
            KalmanFilterBank(observation_noise=0.0)

    def test_valueless_reading_is_rejected(self) -> None:
        """A reading with no value cannot be filtered and must not be imputed."""
        bank = KalmanFilterBank()
        with pytest.raises(ValueError, match="no value"):
            bank.process(_make_reading(value=None))

    def test_reset_sensor_clears_only_that_sensor(self) -> None:
        """Resetting one tag must leave the others tracking."""
        bank = KalmanFilterBank()
        timestamp = _T0
        for _ in range(10):
            bank.process(_make_reading("TC-01", 300.0, timestamp))
            bank.process(_make_reading("TC-02", 50.0, timestamp))
            timestamp += _DT

        bank.reset_sensor("TC-01")

        assert not bank.get_filter("TC-01").is_initialized
        assert bank.get_filter("TC-02").is_initialized

    def test_reset_unknown_sensor_is_a_noop(self) -> None:
        """Resetting a tag never seen must not raise nor create a filter."""
        bank = KalmanFilterBank()
        bank.reset_sensor("TC-NEVER-SEEN")
        assert len(bank) == 0

    def test_get_all_states_is_serialisable(self) -> None:
        """States must round-trip through JSON for checkpointing."""
        import json

        bank = KalmanFilterBank()
        bank.process(_make_reading("TC-01", 300.0, _T0))
        bank.process(_make_reading("TC-02", 50.0, _T0))

        states = bank.get_all_states()
        assert set(states) == {"TC-01", "TC-02"}
        assert json.loads(json.dumps(states))["TC-01"]["is_initialized"] is True

    def test_state_records_the_last_timestamp(self) -> None:
        """The serialised clock must reflect the last reading processed."""
        bank = KalmanFilterBank()
        bank.process(_make_reading("TC-01", 300.0, _T0))
        assert bank.get_all_states()["TC-01"]["last_timestamp"] == _T0.isoformat()

    def test_fresh_filter_state_has_no_timestamp(self) -> None:
        """An untouched filter must serialise with a null clock."""
        bank = KalmanFilterBank()
        bank.get_filter("TC-01")
        assert bank.get_all_states()["TC-01"]["last_timestamp"] is None


# ---------------------------------------------------------------------------
# filter_columns: el filtro aplicado a todos los sensores a la vez
# ---------------------------------------------------------------------------

_EPOCH_US = int(_T0.timestamp()) * 1_000_000


def _micros(offsets_us: list[int]) -> npt.NDArray[np.int64]:
    """Return timestamps as integer microseconds since the epoch.

    Args:
        offsets_us: Microseconds elapsed since _T0 at each step.

    Returns:
        The timestamps, one per step.
    """
    return np.asarray([_EPOCH_US + offset for offset in offsets_us], dtype=np.int64)


def _scalar_reference(
    values: npt.NDArray[np.float64],
    timestamps_us: npt.NDArray[np.int64],
    process_noise: float,
    observation_noise: float,
    reset_sigma: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Filter every column with OnlineKalmanFilter, skipping non-finite readings.

    Es el oraculo: el filtro escalar de siempre, columna a columna.

    Args:
        values: Measurements, shape (steps, sensors).
        timestamps_us: Timestamp of every step in microseconds.
        process_noise: Spectral density q.
        observation_noise: Measurement noise variance R.
        reset_sigma: Divergence threshold.

    Returns:
        The normalized residuals and the innovation sigmas, NaN where skipped.
    """
    residual = np.full(values.shape, np.nan)
    sigma = np.full(values.shape, np.nan)
    stamps = [_T0 + timedelta(microseconds=int(us - _EPOCH_US)) for us in timestamps_us]
    for column in range(values.shape[1]):
        filtered = OnlineKalmanFilter(
            process_noise=process_noise,
            observation_noise=observation_noise,
            reset_sigma=reset_sigma,
        )
        for step in range(values.shape[0]):
            if not np.isfinite(values[step, column]):
                continue
            result = filtered.step(float(values[step, column]), stamps[step])
            residual[step, column] = result.normalized_residual
            sigma[step, column] = result.innovation_sigma
    return residual, sigma


class TestFilterColumns:
    """filter_columns debe ser el mismo filtro que OnlineKalmanFilter, bit a bit."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("irregular", [False, True])
    def test_matches_the_scalar_filter_exactly(self, seed: int, irregular: bool) -> None:
        """Ruido, huecos, saltos que disparan el reinicio y muestreo irregular."""
        rng = np.random.default_rng(seed)
        steps, width = 90, 7
        values = np.cumsum(rng.normal(size=(steps, width)), axis=0) * 2.0 + 500.0
        values[rng.random(values.shape) < 0.08] = np.nan
        values[rng.random(values.shape) < 0.03] = np.inf
        values[rng.random(values.shape) < 0.05] += 4_000.0
        if irregular:
            offsets = np.cumsum(rng.integers(0, 400_000_000, size=steps)).tolist()
        else:
            offsets = [180_000_000 * step for step in range(steps)]
        stamps = _micros(offsets)

        result = filter_columns(values, stamps, 1e-6, 1.0, 10.0)
        residual, sigma = _scalar_reference(values, stamps, 1e-6, 1.0, 10.0)

        assert np.array_equal(result.normalized_residual, residual, equal_nan=True)
        assert np.array_equal(result.innovation_sigma, sigma, equal_nan=True)

    def test_the_default_tuning_matches_the_scalar_defaults(self) -> None:
        """Sin argumentos, q, R y reset_sigma son los del filtro escalar."""
        values = np.linspace(10.0, 40.0, 30).reshape(-1, 1) + np.sin(np.arange(30))[:, None]
        stamps = _micros([1_000_000 * step for step in range(30)])
        result = filter_columns(values, stamps)
        residual, sigma = _scalar_reference(
            values, stamps, DEFAULT_PROCESS_NOISE, DEFAULT_OBSERVATION_NOISE, DEFAULT_RESET_SIGMA
        )
        assert np.array_equal(result.normalized_residual, residual, equal_nan=True)
        assert np.array_equal(result.innovation_sigma, sigma, equal_nan=True)

    def test_the_first_reading_of_a_column_has_zero_residual(self) -> None:
        """No hay nada que predecir: residual cero y sigma de la covarianza vaga."""
        values = np.array([[5.0, np.nan], [6.0, 7.0]])
        result = filter_columns(values, _micros([0, 1_000_000]), 0.1, 2.0, 10.0)

        assert result.normalized_residual[0, 0] == 0.0
        assert result.innovation_sigma[0, 0] == pytest.approx(math.sqrt(INITIAL_COVARIANCE + 2.0))
        # La segunda columna arranca en el segundo paso, no en el primero.
        assert np.isnan(result.normalized_residual[0, 1])
        assert result.normalized_residual[1, 1] == 0.0

    def test_missing_readings_are_skipped_and_not_imputed(self) -> None:
        """Una medida ausente no avanza el filtro y su celda queda en NaN."""
        values = np.array([[1.0], [np.nan], [np.inf], [1.0]])
        result = filter_columns(values, _micros([0, 1, 2, 3]))
        assert np.isnan(result.normalized_residual[1:3]).all()
        assert np.isnan(result.innovation_sigma[1:3]).all()
        assert np.isfinite(result.normalized_residual[3]).all()

    def test_a_step_without_any_finite_reading_is_a_noop(self) -> None:
        """Un paso con todas las medidas ausentes no toca el estado."""
        values = np.array([[1.0, 2.0], [np.nan, np.nan], [1.5, 2.5]])
        stamps = _micros([0, 1_000_000, 2_000_000])
        result = filter_columns(values, stamps, 0.1, 1.0, 10.0)
        residual, sigma = _scalar_reference(values, stamps, 0.1, 1.0, 10.0)
        assert np.array_equal(result.normalized_residual, residual, equal_nan=True)
        assert np.array_equal(result.innovation_sigma, sigma, equal_nan=True)

    def test_the_gap_widens_the_next_dt(self) -> None:
        """El dt de la medida que sigue a un hueco cruza todo el hueco."""
        gapped = np.array([[1.0], [np.nan], [np.nan], [4.0]])
        dense = np.array([[1.0], [4.0]])
        with_gap = filter_columns(gapped, _micros([0, 1, 2, 3_000_000]), 0.1, 1.0, 10.0)
        without = filter_columns(dense, _micros([0, 3_000_000]), 0.1, 1.0, 10.0)
        assert with_gap.normalized_residual[3, 0] == without.normalized_residual[1, 0]

    def test_a_divergent_reading_reseeds_only_its_own_column(self) -> None:
        """El reinicio de una columna no contamina a las demas."""
        flat = np.full((20, 2), 100.0)
        flat[10, 0] = 100_000.0
        result = filter_columns(flat, _micros([1_000_000 * s for s in range(20)]), 0.1, 1.0, 10.0)
        assert abs(result.normalized_residual[10, 0]) > 10.0
        assert np.all(np.abs(result.normalized_residual[:, 1]) < 1.0)

    def test_equal_timestamps_are_allowed(self) -> None:
        """Dos lecturas con el mismo instante son un dt cero, no un error."""
        values = np.array([[1.0], [1.2], [1.1]])
        result = filter_columns(values, _micros([0, 0, 1_000_000]), 0.1, 1.0, 10.0)
        assert np.isfinite(result.normalized_residual).all()

    def test_a_timestamp_that_goes_back_is_rejected(self) -> None:
        """Mismo error que el filtro escalar, y nombra la columna."""
        values = np.ones((3, 2))
        with pytest.raises(ValueError, match=r"precedes the previous one .* on column 0"):
            filter_columns(values, _micros([0, 2_000_000, 1_000_000]))

    def test_a_backwards_clock_across_a_gap_is_rejected(self) -> None:
        """El retroceso se mide contra la ultima medida valida, no contra el paso anterior."""
        values = np.array([[1.0], [np.nan], [1.0]])
        with pytest.raises(ValueError, match="precedes"):
            filter_columns(values, _micros([5_000_000, 9_000_000, 1_000_000]))

    def test_an_empty_input_gives_empty_output(self) -> None:
        """Cero pasos no es un error: no hay nada que filtrar."""
        result = filter_columns(np.empty((0, 4)), _micros([]))
        assert result.normalized_residual.shape == (0, 4)
        assert result.innovation_sigma.shape == (0, 4)

    @pytest.mark.parametrize(
        ("values", "timestamps"),
        [
            (np.ones(5), _micros([0, 1, 2, 3, 4])),
            (np.ones((5, 2)), _micros([0, 1, 2])),
            (np.ones((5, 2)), np.zeros((5, 1), dtype=np.int64)),
        ],
    )
    def test_shapes_are_validated(
        self, values: npt.NDArray[np.float64], timestamps: npt.NDArray[np.int64]
    ) -> None:
        """values es (pasos, sensores) y timestamps_us es (pasos,)."""
        with pytest.raises(ValueError, match="shape"):
            filter_columns(values, timestamps)

    @pytest.mark.parametrize(
        "tuning",
        [
            {"process_noise": -1.0},
            {"observation_noise": 0.0},
            {"reset_sigma": 1.0},
        ],
    )
    def test_invalid_tuning_is_rejected_like_the_scalar_filter(
        self, tuning: dict[str, float]
    ) -> None:
        """Los ajustes invalidos fallan con los mismos mensajes que el filtro escalar."""
        with pytest.raises(ValueError, match=r"process_noise|observation_noise|reset_sigma"):
            filter_columns(np.ones((3, 1)), _micros([0, 1, 2]), **tuning)

    def test_the_result_is_frozen(self) -> None:
        """KalmanColumns es un valor, no un contenedor mutable."""
        result = filter_columns(np.ones((2, 1)), _micros([0, 1]))
        with pytest.raises(FrozenInstanceError):
            result.normalized_residual = np.zeros((2, 1))  # type: ignore[misc]
