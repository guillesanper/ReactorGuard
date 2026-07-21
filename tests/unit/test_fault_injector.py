"""Unit tests for the synthetic instrument-fault injector.

Corren sobre un frame sintetico y no sobre el parquet del TEP: lo que se fija
aqui es el contrato del inyector (determinismo, geometria de los episodios,
correccion de las etiquetas), que tiene que valer sin que data/ este poblado. La
evaluacion sobre datos reales vive en test_validator_on_tep.py.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from data.generators.tep_adapter import (
    LOCATION_MAP,
    SENSOR_TYPE_MAP,
    UNIT_MAP,
    sensor_id,
)
from data.schemas.sensor_spans import SensorSpan
from data.validation.fault_injector import (
    DEFAULT_INJECTION_SEED,
    InjectedEpisode,
    InjectionConfig,
    InjectionKind,
    TEPFaultInjector,
)
from data.validation.sensor_fault import FaultType

_T0 = datetime(2000, 1, 1, tzinfo=UTC)
_INTERVAL = timedelta(minutes=3)
_N_TIMESTEPS = 120
"""Suficiente para dos episodios por sensor del tamano maximo configurado."""


def _frame(
    n_timesteps: int = _N_TIMESTEPS,
    n_sensors: int = 52,
    interval: timedelta = _INTERVAL,
) -> pd.DataFrame:
    """Build a synthetic long-format frame shaped like TEPAdapter.adapt_all output.

    Los valores se mantienen holgadamente dentro del sobre de alarma sintetico de
    la fixture sensor_spans, de modo que cualquier par fuera del sobre despues de
    inyectar procede de la inyeccion y no del fondo.

    Args:
        n_timesteps: Samples per sensor.
        n_sensors: Number of TEP columns to include.
        interval: Spacing between consecutive samples.

    Returns:
        The frame, ordered by (timestep, sensor_id).
    """
    records = []
    for step in range(n_timesteps):
        for col in range(n_sensors):
            sensor_type = SENSOR_TYPE_MAP[col]
            records.append(
                {
                    "reading_id": f"00000000-0000-4000-8000-{col:06d}{step:06d}",
                    "timestamp": _T0 + interval * step,
                    "timestep": step,
                    "plant_id": "TEP-PLANT-01",
                    "sensor_id": sensor_id(col),
                    "sensor_type": sensor_type.value,
                    "sensor_location": LOCATION_MAP[col].value,
                    "elevation_m": float((col % 55) * 2 - 10),
                    "value": 100.0 + col + 0.5 * step,
                    "unit": UNIT_MAP[sensor_type].value,
                    "quality": "good",
                    "raw_counts": 32768,
                    "fault_type": 0,
                    "is_usable": True,
                }
            )
    return pd.DataFrame(records)


@pytest.fixture()
def frame() -> pd.DataFrame:
    """Return the synthetic long-format frame."""
    return _frame()


# ---------------------------------------------------------------------------
# InjectedEpisode
# ---------------------------------------------------------------------------


class TestInjectedEpisode:
    """Las etiquetas que cada manipulacion justifica."""

    def test_length_counts_both_ends(self) -> None:
        """An episode spanning 10..19 lasts ten timesteps."""
        episode = InjectedEpisode("TEP-XMEAS-01", InjectionKind.FREEZE, 10, 19, 0.0)
        assert episode.length == 10

    def test_a_spike_lasts_one_sample(self) -> None:
        """Start and end coincide for a spike."""
        assert InjectedEpisode("A", InjectionKind.SPIKE, 7, 7, 2.0).length == 1

    def test_freeze_labels_stuck_from_the_second_sample(self) -> None:
        """The first sample of a freeze keeps its real value and is not yet stuck."""
        episode = InjectedEpisode("A", InjectionKind.FREEZE, 10, 14, 0.0)
        assert episode.labels(100)[FaultType.STUCK] == {11, 12, 13, 14}

    def test_freeze_labels_its_release_as_a_rapid_change(self) -> None:
        """Reconnecting with the real series is a step."""
        episode = InjectedEpisode("A", InjectionKind.FREEZE, 10, 14, 0.0)
        assert episode.labels(100)[FaultType.NOISE_SPIKE] == {15}

    def test_offset_labels_both_edges_as_rapid_changes(self) -> None:
        """A step in and a step out."""
        episode = InjectedEpisode("A", InjectionKind.OFFSET, 10, 14, 1.0)
        assert episode.labels(100)[FaultType.NOISE_SPIKE] == {10, 15}

    def test_offset_does_not_label_out_of_range_itself(self) -> None:
        """Out-of-range is derived objectively from the values, never per episode."""
        episode = InjectedEpisode("A", InjectionKind.OFFSET, 10, 14, 1.0)
        assert FaultType.BIAS_OUT_OF_RANGE not in episode.labels(100)

    def test_ramp_labels_drift_over_its_whole_window(self) -> None:
        """The sensor is drifting for as long as the ramp lasts."""
        episode = InjectedEpisode("A", InjectionKind.RAMP, 10, 13, 15.0)
        assert episode.labels(100)[FaultType.SENSOR_DRIFT] == {10, 11, 12, 13}

    def test_ramp_labels_only_its_release_as_a_rapid_change(self) -> None:
        """The slope stays under the rate limit; the drop back does not."""
        episode = InjectedEpisode("A", InjectionKind.RAMP, 10, 13, 15.0)
        assert episode.labels(100)[FaultType.NOISE_SPIKE] == {14}

    def test_kalman_covers_the_window_and_the_step_after(self) -> None:
        """The residual has reason to fire throughout and on the way out."""
        episode = InjectedEpisode("A", InjectionKind.SPIKE, 10, 10, 2.0)
        assert episode.labels(100)[FaultType.KALMAN_ANOMALY] == {10, 11}

    def test_labels_are_clipped_to_the_last_timestep(self) -> None:
        """An episode ending on the last sample has no exit edge to label."""
        episode = InjectedEpisode("A", InjectionKind.OFFSET, 8, 10, 1.0)
        labels = episode.labels(10)
        assert labels[FaultType.NOISE_SPIKE] == {8}
        assert max(labels[FaultType.KALMAN_ANOMALY]) == 10

    def test_empty_label_sets_are_dropped(self) -> None:
        """A fault type with nothing to label must not appear as an empty set."""
        episode = InjectedEpisode("A", InjectionKind.RAMP, 5, 9, 15.0)
        assert all(timesteps for timesteps in episode.labels(9).values())
        assert FaultType.NOISE_SPIKE not in episode.labels(9)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Configuraciones que el inyector debe rechazar antes de tocar nada."""

    def test_defaults_are_accepted(self, sensor_spans: dict[str, SensorSpan]) -> None:
        """The default configuration must build."""
        assert TEPFaultInjector(sensor_spans).config.episodes_per_sensor == 2

    def test_zero_sensors_per_kind_is_rejected(
        self, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        """Degrading nobody produces no ground truth."""
        with pytest.raises(ValueError, match="sensors_per_kind"):
            TEPFaultInjector(sensor_spans, InjectionConfig(sensors_per_kind=0))

    def test_zero_episodes_per_sensor_is_rejected(
        self, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        """Same reason, one level down."""
        with pytest.raises(ValueError, match="episodes_per_sensor"):
            TEPFaultInjector(sensor_spans, InjectionConfig(episodes_per_sensor=0))

    @pytest.mark.parametrize(
        ("field", "bounds"),
        [
            ("freeze_length", (0, 10)),
            ("offset_length", (20, 10)),
            ("ramp_length", (-5, 10)),
        ],
    )
    def test_inverted_or_non_positive_length_ranges_are_rejected(
        self,
        sensor_spans: dict[str, SensorSpan],
        field: str,
        bounds: tuple[int, int],
    ) -> None:
        """A length range must be non-empty and positive."""
        with pytest.raises(ValueError, match=field):
            TEPFaultInjector(sensor_spans, InjectionConfig(**{field: bounds}))

    def test_freezes_shorter_than_the_detector_window_are_rejected(
        self, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        """Injecting undetectable freezes would only depress recall."""
        with pytest.raises(ValueError, match="stuck detector window"):
            TEPFaultInjector(sensor_spans, InjectionConfig(freeze_length=(5, 30)))


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------


class TestFrameValidation:
    """Lo que el inyector exige del frame que recibe."""

    def test_missing_columns_are_named(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """A frame without value cannot be degraded."""
        with pytest.raises(KeyError, match="value"):
            TEPFaultInjector(sensor_spans).inject(frame.drop(columns=["value"]))

    def test_a_single_timestep_has_no_measurable_cadence(
        self, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        """The rate and drift thresholds need an interval to be expressed against."""
        with pytest.raises(ValueError, match="fewer than"):
            TEPFaultInjector(sensor_spans).inject(_frame(n_timesteps=1))

    def test_a_non_uniform_cadence_is_rejected(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Two different gaps make every per-second threshold ambiguous."""
        broken = frame.copy()
        late = broken["timestep"] >= 50
        broken.loc[late, "timestamp"] = broken.loc[late, "timestamp"] + timedelta(hours=1)
        with pytest.raises(ValueError, match="not uniform"):
            TEPFaultInjector(sensor_spans).inject(broken)

    def test_a_gap_in_the_timesteps_is_rejected(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """The injector addresses samples by position within a sensor."""
        holed = frame[
            ~((frame["sensor_id"] == "TEP-XMEAS-01") & (frame["timestep"] == 40))
        ]
        with pytest.raises(ValueError, match="contiguous timestep range"):
            TEPFaultInjector(sensor_spans).inject(holed)

    def test_too_few_sensors_with_a_span_is_rejected(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Every manipulation needs its own disjoint set of sensors."""
        few = {tag: sensor_spans[tag] for tag in sorted(sensor_spans)[:10]}
        with pytest.raises(ValueError, match="Injection needs"):
            TEPFaultInjector(few).inject(frame)

    def test_episodes_that_do_not_fit_are_rejected(
        self, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        """A 40-timestep freeze does not fit twice into 30 timesteps."""
        injector = TEPFaultInjector(sensor_spans)
        with pytest.raises(ValueError, match="does not fit in segment"):
            injector.inject(_frame(n_timesteps=30))


# ---------------------------------------------------------------------------
# Determinism and isolation
# ---------------------------------------------------------------------------


class TestDeterminism:
    """La verdad-terreno tiene que ser identica entre ejecuciones."""

    def test_the_same_seed_reproduces_the_episodes(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Same seed, same sensors, same windows, same magnitudes."""
        first = TEPFaultInjector(sensor_spans, seed=7).inject(frame)
        second = TEPFaultInjector(sensor_spans, seed=7).inject(frame)
        assert first.episodes == second.episodes

    def test_the_same_seed_reproduces_the_values(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """The degraded frame must be identical, not merely similarly shaped."""
        first = TEPFaultInjector(sensor_spans, seed=7).inject(frame)
        second = TEPFaultInjector(sensor_spans, seed=7).inject(frame)
        pd.testing.assert_series_equal(first.frame["value"], second.frame["value"])

    def test_a_different_seed_changes_the_episodes(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Otherwise the seed is not doing anything."""
        first = TEPFaultInjector(sensor_spans, seed=7).inject(frame)
        second = TEPFaultInjector(sensor_spans, seed=8).inject(frame)
        assert first.episodes != second.episodes

    def test_the_default_seed_is_the_documented_one(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """T3.6 measures against the default; it must not drift silently."""
        explicit = TEPFaultInjector(sensor_spans, seed=DEFAULT_INJECTION_SEED)
        assert TEPFaultInjector(sensor_spans).inject(frame).episodes == (
            explicit.inject(frame).episodes
        )

    def test_the_input_frame_is_not_mutated(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Degrading must not corrupt the caller's copy of the clean data."""
        before = frame["value"].copy()
        TEPFaultInjector(sensor_spans).inject(frame)
        pd.testing.assert_series_equal(frame["value"], before)


# ---------------------------------------------------------------------------
# What the manipulations do to the values
# ---------------------------------------------------------------------------


def _episodes_of(report: object, kind: InjectionKind) -> list[InjectedEpisode]:
    """Return the episodes of one kind from a report.

    Args:
        report: The InjectionReport.
        kind: Manipulation to filter by.

    Returns:
        The matching episodes.
    """
    return [e for e in report.episodes if e.kind is kind]  # type: ignore[attr-defined]


def _window(frame: pd.DataFrame, episode: InjectedEpisode) -> pd.Series:
    """Return the values of one episode's window, in timestep order.

    Args:
        frame: Degraded frame.
        episode: Episode to slice.

    Returns:
        The window's values.
    """
    rows = frame[
        (frame["sensor_id"] == episode.sensor_id)
        & frame["timestep"].between(episode.start_timestep, episode.end_timestep)
    ]
    return rows.sort_values("timestep")["value"]


class TestManipulations:
    """Que cada manipulacion hace lo que su nombre dice."""

    def test_every_kind_is_applied(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """All four manipulations must appear in a default run."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        assert {e.kind for e in report.episodes} == set(InjectionKind)

    def test_sensor_sets_are_disjoint_across_kinds(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """No sensor may carry two kinds, or its labels would compete."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        kinds_per_sensor: dict[str, set[InjectionKind]] = {}
        for episode in report.episodes:
            kinds_per_sensor.setdefault(episode.sensor_id, set()).add(episode.kind)
        assert all(len(kinds) == 1 for kinds in kinds_per_sensor.values())

    def test_the_episode_count_is_deterministic(
        self, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        """Segment placement means the count never depends on rejected draws."""
        config = InjectionConfig(sensors_per_kind=4, episodes_per_sensor=3)
        report = TEPFaultInjector(sensor_spans, config).inject(_frame(n_timesteps=200))
        assert len(report.episodes) == 4 * 3 * len(InjectionKind)

    def test_a_freeze_holds_a_single_value(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Every sample of the window must be identical."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        for episode in _episodes_of(report, InjectionKind.FREEZE):
            assert _window(report.frame, episode).nunique() == 1

    def test_a_freeze_holds_the_value_it_started_from(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """A latched transmitter holds its last real reading, not an invented one."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        for episode in _episodes_of(report, InjectionKind.FREEZE):
            original = frame[
                (frame["sensor_id"] == episode.sensor_id)
                & (frame["timestep"] == episode.start_timestep)
            ]["value"].iloc[0]
            assert _window(report.frame, episode).iloc[0] == pytest.approx(original)

    def test_an_offset_pushes_the_whole_window_out_of_the_envelope(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Sizing on the worst sample is what makes every sample of it a positive."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        for episode in _episodes_of(report, InjectionKind.OFFSET):
            span = sensor_spans[episode.sensor_id]
            values = _window(report.frame, episode)
            assert not values.between(span.alarm_min, span.alarm_max).any()

    def test_offsets_go_in_both_directions(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """A detector that only ever sees upward excursions is half tested."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        signs = {e.magnitude > 0 for e in _episodes_of(report, InjectionKind.OFFSET)}
        assert signs == {True, False}

    def test_a_spike_touches_exactly_one_sample(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """The sample after a spike is back on the real series."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        spikes = _episodes_of(report, InjectionKind.SPIKE)
        assert spikes
        for episode in spikes:
            assert episode.length == 1
            after = report.frame[
                (report.frame["sensor_id"] == episode.sensor_id)
                & (report.frame["timestep"] == episode.end_timestep + 1)
            ]["value"].iloc[0]
            original = frame[
                (frame["sensor_id"] == episode.sensor_id)
                & (frame["timestep"] == episode.end_timestep + 1)
            ]["value"].iloc[0]
            assert after == pytest.approx(original)

    def test_a_ramp_advances_by_its_declared_slope(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """The magnitude in envelope widths per hour must be the real slope."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        for episode in _episodes_of(report, InjectionKind.RAMP):
            span = sensor_spans[episode.sensor_id]
            width = span.alarm_max - span.alarm_min
            clean = frame[frame["sensor_id"] == episode.sensor_id].sort_values("timestep")
            degraded = report.frame[
                report.frame["sensor_id"] == episode.sensor_id
            ].sort_values("timestep")
            deviation = (
                degraded["value"].to_numpy() - clean["value"].to_numpy()
            )[episode.start_timestep : episode.end_timestep + 1]

            per_step = deviation[1] - deviation[0]
            expected = episode.magnitude * width * report.sample_interval_seconds / 3600.0
            assert per_step == pytest.approx(expected, rel=1e-9)

    def test_ramp_slopes_stay_under_the_rate_limit(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """A ramp must measure the drift detector, not the rate detector.

        El limite de tasa por defecto son 0,005 anchos de sobre por segundo, o sea
        0,9 anchos entre dos muestras a 180 s. Si la rampa por defecto lo superase,
        cada uno de sus pasos seria ademas una violacion de tasa y la matriz de
        confusion de noise_spike quedaria dominada por ella.
        """
        report = TEPFaultInjector(sensor_spans).inject(frame)
        limit_per_step = 0.005 * report.sample_interval_seconds
        for episode in _episodes_of(report, InjectionKind.RAMP):
            per_step = abs(episode.magnitude) * report.sample_interval_seconds / 3600.0
            assert per_step < limit_per_step


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


class TestGroundTruth:
    """Las etiquetas que salen del inyector."""

    def test_a_clean_frame_has_no_out_of_envelope_pairs(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Otherwise the injected positives could not be told from the background."""
        assert TEPFaultInjector(sensor_spans).out_of_envelope_pairs(frame) == set()

    def test_out_of_envelope_is_computed_from_values_and_span_alone(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """The objective label must catch a value nobody injected."""
        edited = frame.copy()
        target = (edited["sensor_id"] == "TEP-XMEAS-01") & (edited["timestep"] == 3)
        edited.loc[target, "value"] = sensor_spans["TEP-XMEAS-01"].alarm_max + 1.0

        pairs = TEPFaultInjector(sensor_spans).out_of_envelope_pairs(edited)
        assert pairs == {("TEP-XMEAS-01", 3)}

    def test_sensors_without_a_span_are_skipped(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Without an envelope there is no excursion to declare."""
        partial = {k: v for k, v in sensor_spans.items() if k != "TEP-XMEAS-01"}
        edited = frame.copy()
        target = edited["sensor_id"] == "TEP-XMEAS-01"
        edited.loc[target, "value"] = 1e9

        assert TEPFaultInjector(partial).out_of_envelope_pairs(edited) == set()

    def test_stuck_labels_match_the_freeze_windows(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Only frozen sensors may carry stuck labels."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        frozen = {e.sensor_id for e in _episodes_of(report, InjectionKind.FREEZE)}
        labelled = {sensor for sensor, _ in report.labels_for(FaultType.STUCK)}
        assert labelled == frozen

    def test_out_of_range_labels_appear_after_injection(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """The offsets must produce the objective positives they were sized for."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        assert report.labels_for(FaultType.BIAS_OUT_OF_RANGE)

    def test_labels_for_an_uninjected_type_are_empty(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Cross-correlation is not injectable and must report no positives."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        assert report.labels_for(FaultType.DRIFT_CORRELATED) == frozenset()

    def test_degraded_sensors_lists_every_touched_tag(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Its complement is the clean population a false-positive rate rests on."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        assert report.degraded_sensors() == {e.sensor_id for e in report.episodes}
        assert len(report.degraded_sensors()) == 4 * 6

    def test_the_guard_band_surrounds_every_episode(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Guard covers the window plus guard_timesteps on each side."""
        config = InjectionConfig(guard_timesteps=3)
        report = TEPFaultInjector(sensor_spans, config).inject(frame)
        for episode in report.episodes:
            for step in range(episode.start_timestep - 3, episode.end_timestep + 4):
                if 0 <= step <= _N_TIMESTEPS - 1:
                    assert (episode.sensor_id, step) in report.guard

    def test_untouched_sensors_are_outside_the_guard_band(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """The guard must not quietly swallow the clean negative class."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        guarded = {sensor for sensor, _ in report.guard}
        assert guarded == report.degraded_sensors()


# ---------------------------------------------------------------------------
# Frame consistency
# ---------------------------------------------------------------------------


class TestFrameConsistency:
    """El frame degradado tiene que seguir siendo un artefacto coherente."""

    def test_raw_counts_are_rescaled_for_degraded_sensors(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """value and raw_counts are two views of one number; both must move."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        degraded = report.frame[report.frame["sensor_id"].isin(report.degraded_sensors())]

        assert not degraded.empty
        for _, row in degraded.iterrows():
            span = sensor_spans[str(row["sensor_id"])]
            assert row["raw_counts"] == span.to_raw_counts(float(row["value"]))

    def test_untouched_sensors_keep_their_raw_counts(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Rescaling a sensor nobody degraded would rewrite data it did not own."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        intact = report.frame[~report.frame["sensor_id"].isin(report.degraded_sensors())]

        assert not intact.empty
        assert set(intact["raw_counts"]) == {32768}

    def test_a_frame_without_raw_counts_is_accepted(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """Not every consumer of the long format carries the ADC view."""
        report = TEPFaultInjector(sensor_spans).inject(frame.drop(columns=["raw_counts"]))
        assert "raw_counts" not in report.frame.columns

    def test_quality_is_left_untouched(
        self, sensor_spans: dict[str, SensorSpan], frame: pd.DataFrame
    ) -> None:
        """A degraded transmitter keeps publishing GOOD; that is the whole point."""
        report = TEPFaultInjector(sensor_spans).inject(frame)
        assert set(report.frame["quality"]) == {"good"}

    def test_the_cadence_is_measured_from_the_frame(
        self, sensor_spans: dict[str, SensorSpan]
    ) -> None:
        """A frame at a different cadence must report that cadence, not a default."""
        report = TEPFaultInjector(sensor_spans).inject(
            _frame(interval=timedelta(seconds=30))
        )
        assert report.sample_interval_seconds == 30.0
