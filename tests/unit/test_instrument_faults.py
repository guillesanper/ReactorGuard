"""Unit tests for data.simulation.instrument_faults.

Cubren la configuracion tipada de niveles (`instrument_faults:`), las cinco
manipulaciones puras sobre la serie medida (deriva, bloqueo, pico, sesgo,
intermitente), su aplicacion a un run y el frame resultante: etiquetas por
lectura, `true_value` conservado, `corrupted` objetivo y contrato canonico.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.generators.reactor_simulator import ReactorSimulator
from data.generators.tep_adapter import readings_from_frame
from data.schemas.sensor_spans import SensorSpan, load_sensor_spans
from data.simulation.dataset_builder import (
    LONG_COLUMNS,
    NO_FAULT,
    assemble_long_frame,
    measure,
)
from data.simulation.instrument_faults import (
    INSTRUMENT_KINDS,
    STUCK_DETECTOR_WINDOW,
    InstrumentFaultTiers,
    apply_bias,
    apply_drift,
    apply_episode,
    apply_instrument_faults,
    apply_intermittent,
    apply_noise_spike,
    apply_stuck,
    build_episode,
    envelope_width,
    parse_instrument_faults,
)
from data.simulation.plant_params import load_plant_params
from data.simulation.scenario import (
    FAULT_SEVERITY,
    FaultLevel,
    InstrumentEpisode,
    InstrumentFaultKind,
    ScenarioKind,
    make_run_plan,
)
from tests.support.sim_frames import CONFIGS, build_simulator

DT_S = 180.0


@pytest.fixture(scope="module")
def raw_tiers() -> dict[str, Any]:
    """Return the raw `instrument_faults:` block of the versioned plant parameters."""
    return load_plant_params(CONFIGS / "plant_parameters.yaml").instrument_faults


@pytest.fixture(scope="module")
def tiers(raw_tiers: dict[str, Any]) -> InstrumentFaultTiers:
    """Return the typed instrument tiers."""
    return parse_instrument_faults(raw_tiers)


@pytest.fixture(scope="module")
def span() -> SensorSpan:
    """Return a thermocouple span (alarm width ~9 K)."""
    return load_sensor_spans(CONFIGS / "sim_sensor_spans.yaml")["TC-HL-1"]


@pytest.fixture()
def series() -> np.ndarray:
    """Return a 60-sample measured series with a known shape."""
    rng = np.random.default_rng(11)
    return 318.0 + 0.15 * rng.standard_normal(60)


def _episode(kind: InstrumentFaultKind, **overrides: Any) -> InstrumentEpisode:
    """Return a valid episode of `kind` on TC-HL-1."""
    fields: dict[str, Any] = {
        "sensor_id": "TC-HL-1",
        "kind": kind,
        "level": FaultLevel.HIGH,
        "start_step": 20,
        "length_steps": 15,
        "magnitude": 1.0,
        "direction": 1,
        "p_enter": 0.0,
        "p_exit": 0.0,
    }
    if kind is InstrumentFaultKind.NOISE_SPIKE:
        fields["length_steps"] = 1
    if kind is InstrumentFaultKind.INTERMITTENT:
        fields.update(p_enter=0.2, p_exit=0.4, magnitude=0.0)
    if kind is InstrumentFaultKind.STUCK:
        fields["magnitude"] = 0.0
    fields.update(overrides)
    return InstrumentEpisode(**fields)


class TestTiers:
    """`instrument_faults:` resolves to typed levels or fails naming the cause."""

    def test_the_five_kinds_and_three_levels(self, tiers: InstrumentFaultTiers) -> None:
        """Every instrument kind has low, medium and high."""
        assert set(tiers.kinds) == set(INSTRUMENT_KINDS)
        for kind in INSTRUMENT_KINDS:
            assert set(tiers.for_kind(kind).levels) == set(FaultLevel)

    def test_drift_high_is_the_detectable_band_and_low_is_the_tdd_rate(
        self, tiers: InstrumentFaultTiers
    ) -> None:
        """High is the 13-17 widths/hour band of Phase 2; low is ~0.1 C/day."""
        drift = tiers.for_kind(InstrumentFaultKind.DRIFT)
        high = drift.level(FaultLevel.HIGH)
        assert (high.magnitude_low, high.magnitude_high) == (13.0, 17.0)
        low = drift.level(FaultLevel.LOW)
        # 0.1 C/day over a ~9 K envelope is ~0.0005 widths per hour.
        assert low.magnitude_mid < 0.01
        assert low.magnitude_mid < drift.level(FaultLevel.MEDIUM).magnitude_mid < 13.0

    def test_stuck_lengths_exceed_the_detector_window(self, tiers: InstrumentFaultTiers) -> None:
        """Every level freezes for longer than the 8 samples the detector needs."""
        stuck = tiers.for_kind(InstrumentFaultKind.STUCK)
        lengths = [stuck.level(lv).length_steps for lv in FaultLevel]
        assert all(n > STUCK_DETECTOR_WINDOW for n in lengths)
        assert lengths == sorted(lengths)

    def test_bias_has_levels_inside_and_outside_the_envelope(
        self, tiers: InstrumentFaultTiers
    ) -> None:
        """Low stays well inside one envelope width, high is a full width or more."""
        bias = tiers.for_kind(InstrumentFaultKind.BIAS)
        assert bias.level(FaultLevel.LOW).magnitude_high < 0.5
        assert bias.level(FaultLevel.HIGH).magnitude_low >= 1.0
        assert bias.onset_fraction == (0.25, 0.50)

    def test_intermittent_invalid_fraction_grows(self, tiers: InstrumentFaultTiers) -> None:
        """The stationary invalid share is p_enter / (p_enter + p_exit) and grows."""
        spec = tiers.for_kind(InstrumentFaultKind.INTERMITTENT)
        fractions = [spec.level(lv).invalid_fraction for lv in FaultLevel]
        assert fractions == sorted(fractions)
        low = spec.level(FaultLevel.LOW)
        assert low.invalid_fraction == pytest.approx(low.p_enter / (low.p_enter + low.p_exit))

    def test_window_steps_by_kind(self, tiers: InstrumentFaultTiers) -> None:
        """A spike lasts one sample, a bias runs to the end, the rest use the level."""
        assert (
            tiers.for_kind(InstrumentFaultKind.NOISE_SPIKE).window_steps(FaultLevel.HIGH, 100, 1500)
            == 1
        )
        assert (
            tiers.for_kind(InstrumentFaultKind.BIAS).window_steps(FaultLevel.LOW, 600, 1500) == 900
        )
        stuck = tiers.for_kind(InstrumentFaultKind.STUCK)
        assert (
            stuck.window_steps(FaultLevel.MEDIUM, 10, 1500)
            == stuck.level(FaultLevel.MEDIUM).length_steps
        )
        assert stuck.mean_window_steps(1500) == pytest.approx(
            sum(stuck.level(lv).length_steps for lv in FaultLevel) / 3
        )

    def test_unknown_kind_raises(self, tiers: InstrumentFaultTiers) -> None:
        """for_kind names the missing kind."""
        partial = InstrumentFaultTiers(kinds={})
        with pytest.raises(KeyError, match="drift"):
            partial.for_kind(InstrumentFaultKind.DRIFT)
        assert tiers.for_kind(InstrumentFaultKind.STUCK).kind is InstrumentFaultKind.STUCK

    @pytest.mark.parametrize(
        ("mutate", "error", "match"),
        [
            (lambda r: r.pop("drift"), KeyError, "drift"),
            (lambda r: r["drift"]["levels"].pop("high"), KeyError, "high"),
            (lambda r: r["drift"].pop("levels"), KeyError, "levels"),
            (lambda r: r.__setitem__("stuck", [1]), TypeError, "mapping"),
            (
                lambda r: r["stuck"]["levels"]["low"].__setitem__("length_steps", 8),
                ValueError,
                "window",
            ),
            (
                lambda r: r["stuck"]["levels"]["low"].__setitem__("length_steps", 2.5),
                ValueError,
                "whole",
            ),
            (lambda r: r["stuck"]["levels"]["low"].pop("length_steps"), KeyError, "length_steps"),
            (
                lambda r: r["noise_spike"]["levels"]["low"].__setitem__("k_sigma", [9, 3]),
                ValueError,
                "low <= high",
            ),
            (
                lambda r: r["noise_spike"]["levels"]["low"].__setitem__("k_sigma", [1, 2, 3]),
                TypeError,
                "pair",
            ),
            (
                lambda r: r["noise_spike"]["levels"]["low"].__setitem__("k_sigma", "x"),
                ValueError,
                "number",
            ),
            (
                lambda r: r["noise_spike"]["levels"]["low"].__setitem__("k_sigma", float("inf")),
                ValueError,
                "finite",
            ),
            (
                lambda r: r["noise_spike"]["levels"]["low"].__setitem__("k_sigma", -1.0),
                ValueError,
                "low <= high",
            ),
            (
                lambda r: r["intermittent"]["levels"]["low"].__setitem__("p_enter", 0.0),
                ValueError,
                "p_enter",
            ),
            (
                lambda r: r["intermittent"]["levels"]["low"].__setitem__("p_exit", 1.5),
                ValueError,
                "p_exit",
            ),
            (lambda r: r["bias"].pop("onset_fraction"), KeyError, "onset_fraction"),
            (
                lambda r: r["bias"].__setitem__("onset_fraction", [0.5, 0.2]),
                ValueError,
                "onset_fraction",
            ),
            (lambda r: r["bias"].__setitem__("onset_fraction", 0.3), TypeError, "onset_fraction"),
            (
                lambda r: r["bias"]["levels"]["medium"].__setitem__("offset_widths", 0.05),
                ValueError,
                "increase in severity",
            ),
        ],
    )
    def test_invalid_blocks(
        self,
        raw_tiers: dict[str, Any],
        mutate: Any,
        error: type[Exception],
        match: str,
    ) -> None:
        """Each malformed block fails with the key or the rule in the message."""
        broken = copy.deepcopy(raw_tiers)
        mutate(broken)
        with pytest.raises(error, match=match):
            parse_instrument_faults(broken)


class TestApplyDrift:
    """Drift adds a linear ramp in envelope widths per hour."""

    def test_slope_is_widths_per_hour(self, series: np.ndarray, span: SensorSpan) -> None:
        """The offset grows by rate x width x dt / 3600 per sample, from the first one."""
        episode = _episode(InstrumentFaultKind.DRIFT, magnitude=15.0, direction=-1)
        out = apply_drift(series, span, episode, DT_S)
        step = -15.0 * envelope_width(span) * DT_S / 3600.0
        delta = out - series
        window = slice(episode.start_step, episode.last_step + 1)
        assert np.allclose(delta[window], step * np.arange(1, episode.length_steps + 1))
        assert delta[window][0] != 0.0
        assert np.array_equal(out[: episode.start_step], series[: episode.start_step])
        assert np.array_equal(out[episode.last_step + 1 :], series[episode.last_step + 1 :])

    def test_does_not_mutate_the_input(self, series: np.ndarray, span: SensorSpan) -> None:
        """The pure function returns a new array."""
        before = series.copy()
        apply_drift(series, span, _episode(InstrumentFaultKind.DRIFT), DT_S)
        assert np.array_equal(series, before)

    def test_rejects_bad_cadence(self, series: np.ndarray, span: SensorSpan) -> None:
        """dt must be positive."""
        with pytest.raises(ValueError, match="dt_s"):
            apply_drift(series, span, _episode(InstrumentFaultKind.DRIFT), 0.0)

    def test_window_must_fit(self, series: np.ndarray, span: SensorSpan) -> None:
        """An episode past the series is an error, not a silent clip."""
        episode = _episode(InstrumentFaultKind.DRIFT, start_step=55, length_steps=10)
        with pytest.raises(ValueError, match="ends at step"):
            apply_drift(series, span, episode, DT_S)


class TestApplyStuck:
    """Stuck freezes the last valid value before the window."""

    def test_window_repeats_the_previous_sample(self, series: np.ndarray) -> None:
        """Every sample of the window equals series[start - 1], exactly."""
        episode = _episode(InstrumentFaultKind.STUCK, length_steps=20)
        out = apply_stuck(series, episode)
        window = out[episode.start_step : episode.last_step + 1]
        assert (window == series[episode.start_step - 1]).all()
        assert np.array_equal(out[: episode.start_step], series[: episode.start_step])
        assert np.array_equal(out[episode.last_step + 1 :], series[episode.last_step + 1 :])

    def test_run_of_repeats_exceeds_the_detector_window(self, series: np.ndarray) -> None:
        """The frozen run (window + the sample before) is longer than 8 identical readings."""
        episode = _episode(InstrumentFaultKind.STUCK, length_steps=STUCK_DETECTOR_WINDOW + 4)
        out = apply_stuck(series, episode)
        run = out[episode.start_step - 1 : episode.last_step + 1]
        assert len(set(run.tolist())) == 1
        assert len(run) > STUCK_DETECTOR_WINDOW

    def test_needs_a_previous_valid_sample(self) -> None:
        """A stuck episode cannot start at step 0."""
        with pytest.raises(ValueError, match="previous valid sample"):
            _episode(InstrumentFaultKind.STUCK, start_step=0)


class TestApplyNoiseSpike:
    """A spike moves one sample by k sigma and leaves the rest alone."""

    def test_one_sample_moves_by_k_sigma(self, series: np.ndarray) -> None:
        """Exactly one sample changes, by direction x k x sigma."""
        episode = _episode(InstrumentFaultKind.NOISE_SPIKE, magnitude=20.0, direction=-1)
        out = apply_noise_spike(series, 0.15, episode)
        changed = np.flatnonzero(out != series)
        assert changed.tolist() == [episode.start_step]
        assert out[episode.start_step] - series[episode.start_step] == pytest.approx(-20 * 0.15)

    def test_value_is_not_clipped_but_counts_saturate(
        self, series: np.ndarray, span: SensorSpan
    ) -> None:
        """A spike far outside the span keeps its value; raw_counts clamps at the ADC."""
        episode = _episode(InstrumentFaultKind.NOISE_SPIKE, magnitude=500.0)
        out = apply_noise_spike(series, 0.15, episode)
        spiked = float(out[episode.start_step])
        assert spiked > span.max
        assert span.to_raw_counts(spiked) == 65535

    def test_rejects_non_positive_sigma(self, series: np.ndarray) -> None:
        """sigma must be positive."""
        with pytest.raises(ValueError, match="sigma"):
            apply_noise_spike(series, 0.0, _episode(InstrumentFaultKind.NOISE_SPIKE))

    def test_a_spike_lasts_one_sample(self) -> None:
        """The episode itself refuses a longer window."""
        with pytest.raises(ValueError, match="one sample"):
            _episode(InstrumentFaultKind.NOISE_SPIKE, length_steps=3)


class TestApplyBias:
    """A bias is a persistent offset in envelope widths."""

    def test_offset_over_the_window(self, series: np.ndarray, span: SensorSpan) -> None:
        """The offset is direction x magnitude x width on every window sample."""
        episode = _episode(
            InstrumentFaultKind.BIAS, magnitude=0.4, direction=1, start_step=30, length_steps=30
        )
        out = apply_bias(series, span, episode)
        assert np.allclose(out[30:] - series[30:], 0.4 * envelope_width(span))
        assert np.array_equal(out[:30], series[:30])


class TestApplyIntermittent:
    """The two-state chain alternates valid readings and ADC rails."""

    def _long(self, **overrides: Any) -> tuple[np.ndarray, InstrumentEpisode]:
        """Return a flat 4000-sample series and a 3000-sample intermittent episode."""
        series = np.full(4000, 318.0)
        episode = _episode(
            InstrumentFaultKind.INTERMITTENT,
            start_step=10,
            length_steps=3000,
            **overrides,
        )
        return series, episode

    def test_invalid_samples_are_rails_and_valid_ones_are_untouched(self, span: SensorSpan) -> None:
        """Invalid = span.min or span.max; everything else equals the input."""
        series, episode = self._long(p_enter=0.1, p_exit=0.3)
        out, invalid = apply_intermittent(series, span, episode, np.random.default_rng(1))
        rails = {span.min, span.max}
        assert set(out[invalid].tolist()) <= rails
        assert len(set(out[invalid].tolist())) == 2
        assert (out[~invalid] == series[~invalid]).all()
        assert not invalid[: episode.start_step].any()
        assert not invalid[episode.last_step + 1 :].any()

    def test_window_starts_invalid(self, span: SensorSpan) -> None:
        """injected_at coincides with the first corrupted reading."""
        series, episode = self._long(p_enter=0.01, p_exit=0.05, direction=-1)
        out, invalid = apply_intermittent(series, span, episode, np.random.default_rng(2))
        assert invalid[episode.start_step]
        assert out[episode.start_step] == span.min

    def test_stationary_share_matches_the_chain(self, span: SensorSpan) -> None:
        """The invalid fraction converges to p_enter / (p_enter + p_exit)."""
        series, episode = self._long(p_enter=0.1, p_exit=0.3)
        _, invalid = apply_intermittent(series, span, episode, np.random.default_rng(3))
        share = invalid[episode.start_step : episode.last_step + 1].mean()
        assert share == pytest.approx(0.1 / 0.4, abs=0.04)

    def test_burst_length_matches_one_over_p_exit(self, span: SensorSpan) -> None:
        """Invalid bursts last 1 / p_exit samples on average."""
        series, episode = self._long(p_enter=0.05, p_exit=0.25)
        _, invalid = apply_intermittent(series, span, episode, np.random.default_rng(4))
        window = invalid[episode.start_step : episode.last_step + 1]
        edges = np.diff(np.concatenate(([0], window.astype(int), [0])))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        assert lengths.mean() == pytest.approx(1 / 0.25, rel=0.2)

    def test_random_consumption_is_fixed(self, span: SensorSpan) -> None:
        """2 x length draws whatever the probabilities: one episode does not shift the next."""
        series, a = self._long(p_enter=0.01, p_exit=0.9)
        _, b = self._long(p_enter=0.9, p_exit=0.01)
        rng_a, rng_b = np.random.default_rng(5), np.random.default_rng(5)
        apply_intermittent(series, span, a, rng_a)
        apply_intermittent(series, span, b, rng_b)
        assert rng_a.random() == rng_b.random()

    def test_same_generator_state_same_output(self, span: SensorSpan) -> None:
        """Deterministic given the generator."""
        series, episode = self._long(p_enter=0.1, p_exit=0.3)
        a = apply_intermittent(series, span, episode, np.random.default_rng(6))
        b = apply_intermittent(series, span, episode, np.random.default_rng(6))
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])

    def test_probabilities_are_validated(self) -> None:
        """p_enter and p_exit must be in (0, 1]."""
        with pytest.raises(ValueError, match="p_enter"):
            _episode(InstrumentFaultKind.INTERMITTENT, p_enter=0.0)
        with pytest.raises(ValueError, match="p_exit"):
            _episode(InstrumentFaultKind.INTERMITTENT, p_exit=2.0)


class TestEpisodeValidation:
    """The episode refuses windows and magnitudes that mean nothing."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"start_step": -1}, "start_step"),
            ({"length_steps": 0}, "length_steps"),
            ({"direction": 0}, "direction"),
            ({"magnitude": 0.0}, "magnitude"),
            ({"magnitude": float("nan")}, "magnitude"),
        ],
    )
    def test_invalid_drift(self, overrides: dict[str, Any], match: str) -> None:
        """Each malformed field is named."""
        with pytest.raises(ValueError, match=match):
            _episode(InstrumentFaultKind.DRIFT, **overrides)

    def test_episode_id_and_window(self) -> None:
        """The id is unique per run, kind, sensor and start; the window is inclusive."""
        episode = _episode(InstrumentFaultKind.DRIFT, start_step=5, length_steps=3)
        assert episode.episode_id("sim_run_007") == "sim_run_007:drift:TC-HL-1:5"
        assert episode.last_step == 7
        assert [episode.is_active(s) for s in (4, 5, 7, 8)] == [False, True, True, False]

    def test_kind_maps_to_its_scenario(self) -> None:
        """InstrumentFaultKind values coincide with ScenarioKind (one scenario per run)."""
        for kind in INSTRUMENT_KINDS:
            assert kind.scenario is ScenarioKind(kind.value)
            assert kind.scenario.severity is FAULT_SEVERITY[kind.scenario]


class TestApplyEpisodeAndRun:
    """`apply_episode` dispatches by kind; `apply_instrument_faults` handles a run."""

    @pytest.mark.parametrize("kind", list(INSTRUMENT_KINDS))
    def test_corrupted_is_the_objective_difference(
        self, kind: InstrumentFaultKind, series: np.ndarray, span: SensorSpan
    ) -> None:
        """corrupted == (value differs from the unfaulted measurement)."""
        episode = _episode(kind)
        effect = apply_episode(series, episode, span, 0.15, DT_S, np.random.default_rng(0))
        assert np.array_equal(effect.corrupted, effect.values != series)
        assert effect.corrupted[episode.start_step]
        assert not effect.corrupted[: episode.start_step].any()
        assert not effect.corrupted[episode.last_step + 1 :].any()

    def test_a_run_applies_its_episodes_in_order(self, tmp_path: Path) -> None:
        """Several episodes on distinct sensors; the input matrix is not mutated."""
        simulator = build_simulator(tmp_path)
        catalog, spans = simulator.catalog, simulator.spans
        base = make_run_plan(simulator.params, 0, ScenarioKind.NORMAL)
        episodes = (
            _episode(InstrumentFaultKind.STUCK, sensor_id="FT-HL-1", start_step=8, length_steps=10),
            _episode(InstrumentFaultKind.STUCK, sensor_id="TC-HL-1", start_step=6, length_steps=12),
        )
        plan = make_run_plan(simulator.params, 0, ScenarioKind.STUCK, instrument_faults=episodes)
        measured = np.random.default_rng(0).normal(100.0, 1.0, (base.n_steps, len(catalog.sensors)))
        before = measured.copy()
        effect = apply_instrument_faults(
            measured, catalog, spans, plan, DT_S, np.random.default_rng(0)
        )
        assert np.array_equal(measured, before)
        j = catalog.tags.index("FT-HL-1")
        assert (effect.values[8:18, j] == measured[7, j]).all()
        assert effect.corrupted.sum() == 22
        others = np.delete(np.arange(measured.shape[1]), [j, catalog.tags.index("TC-HL-1")])
        assert np.array_equal(effect.values[:, others], measured[:, others])

    def test_no_episodes_is_a_copy(self, tmp_path: Path) -> None:
        """A normal run comes back unchanged with an all-False mask."""
        simulator = build_simulator(tmp_path)
        plan = make_run_plan(simulator.params, 0, ScenarioKind.NORMAL)
        measured = np.ones((plan.n_steps, len(simulator.catalog.sensors)))
        effect = apply_instrument_faults(
            measured, simulator.catalog, simulator.spans, plan, DT_S, np.random.default_rng(0)
        )
        assert np.array_equal(effect.values, measured)
        assert effect.values is not measured
        assert not effect.corrupted.any()

    def test_shape_and_sensor_errors(self, tmp_path: Path) -> None:
        """A wrong matrix, an unknown sensor or a missing span fail with the cause."""
        simulator = build_simulator(tmp_path)
        n = simulator.params.run_length_steps
        plan = make_run_plan(
            simulator.params,
            0,
            ScenarioKind.STUCK,
            instrument_faults=(
                _episode(
                    InstrumentFaultKind.STUCK, sensor_id="NOPE-1", start_step=10, length_steps=8
                ),
            ),
        )
        good = np.ones((n, len(simulator.catalog.sensors)))
        with pytest.raises(ValueError, match="shape"):
            apply_instrument_faults(
                good[:5], simulator.catalog, simulator.spans, plan, DT_S, np.random.default_rng(0)
            )
        with pytest.raises(KeyError, match="NOPE-1"):
            apply_instrument_faults(
                good, simulator.catalog, simulator.spans, plan, DT_S, np.random.default_rng(0)
            )
        no_span = {k: v for k, v in simulator.spans.items() if k != "TC-HL-1"}
        plan = make_run_plan(
            simulator.params,
            0,
            ScenarioKind.STUCK,
            instrument_faults=(_episode(InstrumentFaultKind.STUCK, start_step=10, length_steps=8),),
        )
        with pytest.raises(KeyError, match="span"):
            apply_instrument_faults(
                good, simulator.catalog, no_span, plan, DT_S, np.random.default_rng(0)
            )


class TestBuildEpisode:
    """The planner's constructor turns a (kind, level, sensor, start) into an episode."""

    def test_magnitude_stays_in_the_level_range(self, tiers: InstrumentFaultTiers) -> None:
        """Drift high draws inside the 13-17 band."""
        rng = np.random.default_rng(9)
        spec = tiers.for_kind(InstrumentFaultKind.DRIFT)
        for _ in range(50):
            episode = build_episode(spec, FaultLevel.HIGH, "TC-HL-1", 400, 1500, rng)
            assert 13.0 <= episode.magnitude <= 17.0
            assert episode.length_steps == 40
            assert episode.direction in (-1, 1)

    def test_bias_runs_to_the_end_of_the_run(self, tiers: InstrumentFaultTiers) -> None:
        """A bias window is n_steps - start."""
        spec = tiers.for_kind(InstrumentFaultKind.BIAS)
        episode = build_episode(
            spec, FaultLevel.MEDIUM, "PT-PRZ-01", 500, 1500, np.random.default_rng(1)
        )
        assert episode.length_steps == 1000
        assert episode.last_step == 1499

    def test_consumption_does_not_depend_on_kind_or_level(
        self, tiers: InstrumentFaultTiers
    ) -> None:
        """Two draws per episode, so changing one episode never shifts the others."""
        states = []
        for kind, level in (
            (InstrumentFaultKind.DRIFT, FaultLevel.LOW),
            (InstrumentFaultKind.STUCK, FaultLevel.HIGH),
            (InstrumentFaultKind.INTERMITTENT, FaultLevel.MEDIUM),
        ):
            rng = np.random.default_rng(3)
            build_episode(tiers.for_kind(kind), level, "TC-HL-1", 400, 1500, rng)
            states.append(rng.random())
        assert len(set(states)) == 1

    def test_must_fit_in_the_run(self, tiers: InstrumentFaultTiers) -> None:
        """An episode that leaves the run is an error."""
        spec = tiers.for_kind(InstrumentFaultKind.DRIFT)
        with pytest.raises(ValueError, match="does not fit"):
            build_episode(spec, FaultLevel.LOW, "TC-HL-1", 1000, 1500, np.random.default_rng(0))


@pytest.fixture(scope="module")
def run_simulator(tmp_path_factory: pytest.TempPathFactory) -> ReactorSimulator:
    """Return a simulator over 30-step runs (shared: the model is loaded once)."""
    return build_simulator(tmp_path_factory.mktemp("instrument"))


def _frame(
    simulator: ReactorSimulator, kind: InstrumentFaultKind, **overrides: Any
) -> pd.DataFrame:
    """Return the long frame of a run with one episode of `kind`."""
    if kind is InstrumentFaultKind.NOISE_SPIKE:
        episode = _episode(kind, start_step=8, **overrides)
    else:
        episode = _episode(kind, start_step=8, length_steps=12, **overrides)
    plan = make_run_plan(
        simulator.params, 1, ScenarioKind(kind.value), instrument_faults=(episode,)
    )
    return simulator.generate_run(plan)


@pytest.fixture(scope="module")
def normal(run_simulator: ReactorSimulator) -> pd.DataFrame:
    """Return the normal run with the same index (the exact counterfactual)."""
    return run_simulator.generate_run(make_run_plan(run_simulator.params, 1, ScenarioKind.NORMAL))


class TestFaultedFrame:
    """The long frame of an instrument-fault run carries its labels and the truth."""

    @pytest.mark.parametrize("kind", list(INSTRUMENT_KINDS))
    def test_labels_cover_only_the_episode(
        self,
        run_simulator: ReactorSimulator,
        normal: pd.DataFrame,
        kind: InstrumentFaultKind,
    ) -> None:
        """Only the affected sensor and window carry the label, with its static severity."""
        frame = _frame(run_simulator, kind)
        assert tuple(frame.columns) == LONG_COLUMNS
        labelled = frame[frame["instrument_fault"] != NO_FAULT]
        episode_len = 1 if kind is InstrumentFaultKind.NOISE_SPIKE else 12
        assert len(labelled) == episode_len
        assert (labelled["sensor_id"] == "TC-HL-1").all()
        assert (labelled["instrument_fault"] == kind.value).all()
        assert (labelled["instrument_severity"] == FAULT_SEVERITY[kind.scenario].value).all()
        assert (labelled["fault_level"] == "high").all()
        assert (labelled["injected_at_step"] == 8).all()
        assert (labelled["timestep"].min(), labelled["timestep"].max()) == (
            8,
            8 + episode_len - 1,
        )
        stamp = frame[frame["timestep"] == 8]["timestamp"].iloc[0]
        assert (labelled["injected_at"] == stamp).all()
        assert labelled["episode_id"].nunique() == 1
        assert labelled["episode_id"].iloc[0].startswith("sim_run_002:")
        clean = frame[frame["instrument_fault"] == NO_FAULT]
        assert (clean["process_fault"] == NO_FAULT).all()
        assert clean["fault_level"].isna().all()
        assert clean["injected_at_step"].isna().all()
        assert clean["episode_id"].isna().all()
        assert (frame["process_fault"] == NO_FAULT).all()
        assert frame["process_severity"].isna().all()
        assert (frame["scenario"] == kind.value).all()

    @pytest.mark.parametrize("kind", list(INSTRUMENT_KINDS))
    def test_true_value_is_preserved_and_untouched_readings_are_identical(
        self,
        run_simulator: ReactorSimulator,
        normal: pd.DataFrame,
        kind: InstrumentFaultKind,
    ) -> None:
        """The plant is the normal one: truth identical, only the faulted readings differ."""
        frame = _frame(run_simulator, kind)
        assert np.array_equal(frame["true_value"].to_numpy(), normal["true_value"].to_numpy())
        untouched = frame["instrument_fault"] == NO_FAULT
        assert np.array_equal(
            frame.loc[untouched, "value"].to_numpy(), normal.loc[untouched, "value"].to_numpy()
        )
        assert np.array_equal(
            frame.loc[untouched, "raw_counts"].to_numpy(),
            normal.loc[untouched, "raw_counts"].to_numpy(),
        )
        assert not frame.loc[untouched, "corrupted"].any()

    @pytest.mark.parametrize("kind", list(INSTRUMENT_KINDS))
    def test_corrupted_marks_readings_that_differ(
        self,
        run_simulator: ReactorSimulator,
        normal: pd.DataFrame,
        kind: InstrumentFaultKind,
    ) -> None:
        """corrupted is exactly 'the value differs from the unfaulted measurement'."""
        frame = _frame(run_simulator, kind)
        differs = frame["value"].to_numpy() != normal["value"].to_numpy()
        assert np.array_equal(frame["corrupted"].to_numpy(), differs)
        assert frame["corrupted"].any()

    def test_intermittent_label_covers_the_window_but_corrupted_does_not(
        self, run_simulator: ReactorSimulator
    ) -> None:
        """quality stays good; the label spans the window, corrupted only the invalid ones."""
        frame = _frame(run_simulator, InstrumentFaultKind.INTERMITTENT, p_enter=0.05, p_exit=0.5)
        labelled = frame[frame["instrument_fault"] == "intermittent"]
        assert len(labelled) == 12
        assert 0 < labelled["corrupted"].sum() < 12
        assert (frame["quality"] == "good").all()
        assert frame["is_usable"].all()

    def test_spike_saturates_counts_but_not_the_value(
        self, run_simulator: ReactorSimulator
    ) -> None:
        """A huge spike keeps its engineering value and clamps raw_counts to 65535."""
        frame = _frame(run_simulator, InstrumentFaultKind.NOISE_SPIKE, magnitude=5000.0)
        row = frame[frame["instrument_fault"] == "noise_spike"].iloc[0]
        assert row["value"] > run_simulator.spans["TC-HL-1"].max
        assert row["raw_counts"] == 65535

    def test_rebuilds_canonical_readings(self, run_simulator: ReactorSimulator) -> None:
        """readings_from_frame validates every faulted row against the contract."""
        frame = _frame(run_simulator, InstrumentFaultKind.BIAS, magnitude=1.2)
        readings = list(readings_from_frame(frame))
        assert len(readings) == len(frame)
        assert all(r.is_usable for r in readings)

    def test_same_plan_same_frame(self, run_simulator: ReactorSimulator) -> None:
        """Deterministic: two generations of one plan are equal frames."""
        a = _frame(run_simulator, InstrumentFaultKind.INTERMITTENT)
        b = _frame(run_simulator, InstrumentFaultKind.INTERMITTENT)
        assert a.equals(b)

    def test_expected_labelled_readings_matches_the_frame(
        self, run_simulator: ReactorSimulator
    ) -> None:
        """The plan's count of labelled readings is the frame's."""
        episodes = (
            _episode(InstrumentFaultKind.DRIFT, sensor_id="TC-HL-1", start_step=6, length_steps=10),
            _episode(
                InstrumentFaultKind.DRIFT, sensor_id="PT-PRZ-01", start_step=9, length_steps=7
            ),
        )
        plan = make_run_plan(
            run_simulator.params, 0, ScenarioKind.DRIFT, instrument_faults=episodes
        )
        frame = run_simulator.generate_run(plan)
        n_sensors = len(run_simulator.catalog.sensors)
        labelled = int((frame["instrument_fault"] != NO_FAULT).sum())
        assert labelled == plan.expected_labelled_readings(n_sensors) == 17


class TestAssembleContract:
    """`assemble_long_frame` insists on the corruption mask for instrument runs."""

    def test_instrument_run_needs_the_mask(self, tmp_path: Path) -> None:
        """Assembling an instrument run without `corrupted` is an error."""
        simulator = build_simulator(tmp_path)
        plan = make_run_plan(
            simulator.params,
            0,
            ScenarioKind.STUCK,
            instrument_faults=(_episode(InstrumentFaultKind.STUCK, start_step=10, length_steps=8),),
        )
        sim = simulator.simulate(plan)
        measured = measure(sim, simulator.catalog, np.random.default_rng(0))
        with pytest.raises(ValueError, match="corruption mask"):
            assemble_long_frame(sim, measured, simulator.catalog, simulator.spans, "P")
        with pytest.raises(ValueError, match="corrupted has shape"):
            assemble_long_frame(
                sim,
                measured,
                simulator.catalog,
                simulator.spans,
                "P",
                corrupted=np.zeros((2, 2), dtype=np.bool_),
            )

    def test_unknown_sensor_in_an_episode(self, tmp_path: Path) -> None:
        """The labels refuse a sensor the catalog does not have."""
        simulator = build_simulator(tmp_path)
        plan = make_run_plan(
            simulator.params,
            0,
            ScenarioKind.STUCK,
            instrument_faults=(
                _episode(
                    InstrumentFaultKind.STUCK, sensor_id="NOPE-1", start_step=10, length_steps=8
                ),
            ),
        )
        sim = simulator.simulate(plan)
        measured = measure(sim, simulator.catalog, np.random.default_rng(0))
        with pytest.raises(KeyError, match="NOPE-1"):
            assemble_long_frame(
                sim,
                measured,
                simulator.catalog,
                simulator.spans,
                "P",
                corrupted=np.zeros(measured.shape, dtype=np.bool_),
            )
