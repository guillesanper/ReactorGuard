"""Unit tests for data.simulation.scenario.

Cubren la tabla unica de severidad (TDD 2.2), el programa de carga (limites de
carga y de pendiente, calentamiento constante, determinismo) y los planes de run
(semillas independientes y reproducibles, timestamps unicos entre runs).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from data.simulation.scenario import (
    FAULT_AXIS,
    FAULT_KINDS,
    FAULT_SEVERITY,
    INSTRUMENT_KINDS,
    N_RUN_STREAMS,
    FaultAxis,
    FaultLevel,
    FaultSeverity,
    InstrumentEpisode,
    InstrumentFaultKind,
    LoadProfile,
    ProcessFaultSpec,
    RunPlan,
    ScenarioKind,
    make_run_plan,
    normal_run_plans,
    run_id_for,
)
from data.simulation.simulation_params import SimulationParams, load_simulation_params
from data.validation.sensor_fault import Severity
from tests.support.sim_frames import write_sim_params


@pytest.fixture()
def params(tmp_path: Path) -> SimulationParams:
    """Return a small simulation configuration (3 runs of 30 steps)."""
    return load_simulation_params(write_sim_params(tmp_path))


@pytest.fixture()
def profile() -> LoadProfile:
    """Return a profile with short holds so a 90-minute run contains ramps."""
    return LoadProfile(
        load_min=0.70,
        load_max=1.00,
        max_ramp_percent_per_min=5.0,
        hold_minutes_min=6,
        hold_minutes_max=20,
        warmup_steps=6,
    )


class TestSeverityTable:
    """FAULT_SEVERITY is the TDD 2.2 table, and the only one."""

    def test_has_the_eight_fault_types(self) -> None:
        """Eight fault scenarios, plus normal operation that has no severity."""
        assert len(FAULT_SEVERITY) == 8
        assert ScenarioKind.NORMAL not in FAULT_SEVERITY
        assert len(ScenarioKind) == 9

    @pytest.mark.parametrize(
        ("kind", "severity"),
        [
            (ScenarioKind.DRIFT, FaultSeverity.HIGH),
            (ScenarioKind.STUCK, FaultSeverity.CRITICAL),
            (ScenarioKind.NOISE_SPIKE, FaultSeverity.LOW),
            (ScenarioKind.BIAS, FaultSeverity.HIGH),
            (ScenarioKind.INTERMITTENT, FaultSeverity.MEDIUM),
            (ScenarioKind.CORE_TEMP_EXCURSION, FaultSeverity.CRITICAL),
            (ScenarioKind.COOLANT_FLOW_REDUCTION, FaultSeverity.HIGH),
            (ScenarioKind.VOID_FORMATION, FaultSeverity.CRITICAL),
        ],
    )
    def test_matches_tdd_table(self, kind: ScenarioKind, severity: FaultSeverity) -> None:
        """Each scenario carries the severity of TDD 2.2."""
        assert FAULT_SEVERITY[kind] is severity
        assert kind.severity is severity

    def test_normal_has_no_severity(self) -> None:
        """Normal operation is unlabelled."""
        assert ScenarioKind.NORMAL.severity is None

    def test_axes(self) -> None:
        """Five instrument faults, three process faults (TDD 2.2, C1)."""
        axes = [FAULT_AXIS[k] for k in ScenarioKind if k is not ScenarioKind.NORMAL]
        assert axes.count(FaultAxis.INSTRUMENT) == 5
        assert axes.count(FaultAxis.PROCESS) == 3
        assert ScenarioKind.NORMAL.axis is FaultAxis.NONE

    def test_severity_values_match_the_validator_enum(self) -> None:
        """The labels use the same text as the validator's Severity, without importing it."""
        assert {s.value for s in FaultSeverity} == {s.value for s in Severity}

    def test_tables_are_read_only(self) -> None:
        """The severity table cannot be mutated by accident."""
        with pytest.raises(TypeError):
            FAULT_SEVERITY[ScenarioKind.NORMAL] = FaultSeverity.LOW  # type: ignore[index]

    def test_directory_names_are_unique(self) -> None:
        """The scenario value is the `fault_type=<value>` directory name."""
        values = [k.value for k in ScenarioKind]
        assert len(set(values)) == len(values)


class TestProcessFaultSpec:
    """The temporal envelope of a process fault: onset, hold, recovery, labelled window."""

    SPEC = ProcessFaultSpec(
        level=FaultLevel.HIGH, injected_step=10, onset_steps=4, hold_steps=6, recovery_steps=5
    )

    def test_envelope_is_zero_before_the_start(self) -> None:
        """Nothing happens before injected_step."""
        assert self.SPEC.envelope(0.0) == 0.0
        assert self.SPEC.envelope(9.999) == 0.0

    def test_envelope_starts_at_zero_and_reaches_one(self) -> None:
        """The onset ramps linearly from 0 at the start to 1 after onset_steps."""
        assert self.SPEC.envelope(10.0) == 0.0
        assert self.SPEC.envelope(12.0) == pytest.approx(0.5)
        assert self.SPEC.envelope(14.0) == 1.0

    def test_envelope_holds_then_recovers(self) -> None:
        """Full magnitude through the hold, then a linear ramp back to zero."""
        assert self.SPEC.envelope(20.0) == 1.0
        assert self.SPEC.envelope(22.5) == pytest.approx(0.5)
        assert self.SPEC.envelope(25.0) == 0.0
        assert self.SPEC.envelope(40.0) == 0.0

    def test_abrupt_end_without_recovery(self) -> None:
        """recovery_steps=0 ends the fault at once after the hold."""
        spec = ProcessFaultSpec(level=FaultLevel.LOW, injected_step=5, onset_steps=0, hold_steps=3)
        assert spec.envelope(5.0) == 1.0
        assert spec.envelope(8.0) == 1.0
        assert spec.envelope(8.5) == 0.0
        assert spec.last_active_step == 8

    def test_labelled_window_is_closed_on_both_ends(self) -> None:
        """is_active covers injected_step through onset + hold + recovery, inclusive."""
        assert self.SPEC.last_active_step == 25
        assert not self.SPEC.is_active(9)
        assert self.SPEC.is_active(10)
        assert self.SPEC.is_active(25)
        assert not self.SPEC.is_active(26)

    def test_persistent_fault_never_ends(self) -> None:
        """hold_steps=None keeps the fault, and the label, to the end of the run."""
        spec = ProcessFaultSpec(level=FaultLevel.MEDIUM, injected_step=7, onset_steps=2)
        assert spec.last_active_step is None
        assert spec.is_active(10_000)
        assert spec.envelope(10_000.0) == 1.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"injected_step": -1}, "injected_step"),
            ({"injected_step": 0, "onset_steps": -1}, "onset_steps"),
            ({"injected_step": 0, "hold_steps": -1}, "hold_steps"),
            ({"injected_step": 0, "hold_steps": 3, "recovery_steps": -1}, "recovery_steps"),
            ({"injected_step": 0, "recovery_steps": 2}, "persistent"),
        ],
    )
    def test_rejects_invalid_timelines(self, kwargs: dict[str, int], message: str) -> None:
        """Negative counts and a recovering persistent fault are errors."""
        with pytest.raises(ValueError, match=message):
            ProcessFaultSpec(level=FaultLevel.LOW, **kwargs)


class TestRunPlanProcessFault:
    """A plan carries a ProcessFaultSpec exactly when its scenario is a process fault."""

    def _plan(self, **overrides: object) -> RunPlan:
        base: dict[str, object] = {
            "run_index": 0,
            "run_id": "sim_run_001",
            "scenario": ScenarioKind.NORMAL,
            "n_steps": 50,
            "start_time": datetime(2026, 1, 1, tzinfo=UTC),
            "seed_entropy": 1,
            "spawn_key": (0,),
        }
        base.update(overrides)
        return RunPlan(**base)  # type: ignore[arg-type]

    def test_process_scenario_needs_a_spec(self) -> None:
        """A process scenario without its envelope is rejected."""
        with pytest.raises(ValueError, match="needs a ProcessFaultSpec"):
            self._plan(scenario=ScenarioKind.VOID_FORMATION)

    @pytest.mark.parametrize("kind", [ScenarioKind.NORMAL, ScenarioKind.STUCK])
    def test_other_scenarios_cannot_carry_one(self, kind: ScenarioKind) -> None:
        """Normal and instrument runs have no process envelope."""
        spec = ProcessFaultSpec(level=FaultLevel.LOW, injected_step=5)
        with pytest.raises(ValueError, match="cannot carry"):
            self._plan(scenario=kind, process_fault=spec)

    def test_window_must_fit_in_the_run(self) -> None:
        """A start past the end, or a finite window that overruns, is rejected."""
        late = ProcessFaultSpec(level=FaultLevel.LOW, injected_step=50)
        with pytest.raises(ValueError, match="does not fit"):
            self._plan(scenario=ScenarioKind.VOID_FORMATION, process_fault=late)
        long = ProcessFaultSpec(
            level=FaultLevel.LOW, injected_step=40, onset_steps=5, hold_steps=10, recovery_steps=0
        )
        with pytest.raises(ValueError, match="does not fit"):
            self._plan(scenario=ScenarioKind.VOID_FORMATION, process_fault=long)

    def test_valid_plan_is_accepted(self) -> None:
        """A window that fits is stored as given."""
        spec = ProcessFaultSpec(
            level=FaultLevel.MEDIUM, injected_step=10, onset_steps=2, hold_steps=5, recovery_steps=3
        )
        plan = self._plan(scenario=ScenarioKind.CORE_TEMP_EXCURSION, process_fault=spec)
        assert plan.process_fault is spec
        assert plan.scenario.severity is FaultSeverity.CRITICAL


class TestRunPlanInstrumentFaults:
    """A plan carries InstrumentEpisodes exactly when its scenario is an instrument fault."""

    def _plan(self, **overrides: object) -> RunPlan:
        base: dict[str, object] = {
            "run_index": 0,
            "run_id": "sim_run_001",
            "scenario": ScenarioKind.STUCK,
            "n_steps": 50,
            "start_time": datetime(2026, 1, 1, tzinfo=UTC),
            "seed_entropy": 1,
            "spawn_key": (0,),
        }
        base.update(overrides)
        return RunPlan(**base)  # type: ignore[arg-type]

    @staticmethod
    def _episode(
        sensor: str = "TC-HL-1",
        kind: InstrumentFaultKind = InstrumentFaultKind.STUCK,
        start: int = 10,
        length: int = 5,
    ) -> InstrumentEpisode:
        extra = {"magnitude": 3.0} if kind is not InstrumentFaultKind.STUCK else {}
        if kind is InstrumentFaultKind.INTERMITTENT:
            extra = {"p_enter": 0.1, "p_exit": 0.5}
        return InstrumentEpisode(sensor, kind, FaultLevel.LOW, start, length, **extra)  # type: ignore[arg-type]

    def test_instrument_scenario_needs_episodes(self) -> None:
        """An instrument run with no episode is rejected."""
        with pytest.raises(ValueError, match="needs at least one"):
            self._plan()

    @pytest.mark.parametrize("kind", [ScenarioKind.NORMAL, ScenarioKind.VOID_FORMATION])
    def test_other_scenarios_cannot_carry_episodes(self, kind: ScenarioKind) -> None:
        """Normal and process runs have no instrument episodes."""
        extra: dict[str, object] = {}
        if kind is ScenarioKind.VOID_FORMATION:
            extra["process_fault"] = ProcessFaultSpec(level=FaultLevel.LOW, injected_step=5)
        with pytest.raises(ValueError, match="cannot carry InstrumentEpisodes"):
            self._plan(scenario=kind, instrument_faults=(self._episode(),), **extra)

    def test_one_scenario_per_run(self) -> None:
        """A bias episode does not belong to a stuck run (D3)."""
        episode = self._episode(kind=InstrumentFaultKind.BIAS)
        with pytest.raises(ValueError, match="one scenario per run"):
            self._plan(instrument_faults=(episode,))

    def test_window_must_fit_in_the_run(self) -> None:
        """An episode that leaves the run is rejected."""
        with pytest.raises(ValueError, match="does not fit"):
            self._plan(instrument_faults=(self._episode(start=48, length=5),))

    def test_a_sensor_carries_one_episode_except_spikes(self) -> None:
        """Two stuck episodes on one sensor are rejected."""
        two = (self._episode(start=5), self._episode(start=20))
        with pytest.raises(ValueError, match="only noise spikes may repeat"):
            self._plan(instrument_faults=two)

    def test_spikes_may_repeat_two_samples_apart(self) -> None:
        """Spikes on one sensor are fine if separated by at least two samples."""
        spikes = tuple(
            self._episode(kind=InstrumentFaultKind.NOISE_SPIKE, start=s, length=1)
            for s in (5, 7, 20)
        )
        plan = self._plan(scenario=ScenarioKind.NOISE_SPIKE, instrument_faults=spikes)
        assert len(plan.instrument_faults) == 3
        close = tuple(
            self._episode(kind=InstrumentFaultKind.NOISE_SPIKE, start=s, length=1) for s in (5, 6)
        )
        with pytest.raises(ValueError, match="closer than 2 samples"):
            self._plan(scenario=ScenarioKind.NOISE_SPIKE, instrument_faults=close)

    def test_expected_labelled_readings(self) -> None:
        """Process: all sensors of the window; instrument: the episode lengths; normal: 0."""
        n = 27
        assert self._plan(instrument_faults=(self._episode(),)).expected_labelled_readings(n) == 5
        process = self._plan(
            scenario=ScenarioKind.CORE_TEMP_EXCURSION,
            process_fault=ProcessFaultSpec(
                level=FaultLevel.LOW,
                injected_step=10,
                onset_steps=2,
                hold_steps=5,
                recovery_steps=3,
            ),
        )
        assert process.expected_labelled_readings(n) == n * 11
        persistent = self._plan(
            scenario=ScenarioKind.VOID_FORMATION,
            process_fault=ProcessFaultSpec(level=FaultLevel.LOW, injected_step=40),
        )
        assert persistent.expected_labelled_readings(n) == n * 10
        assert self._plan(scenario=ScenarioKind.NORMAL).expected_labelled_readings(n) == 0
        with pytest.raises(ValueError, match="n_sensors"):
            persistent.expected_labelled_readings(0)


class TestMakeRunPlan:
    """`make_run_plan` follows the seed and time convention of the other builders."""

    def test_matches_normal_run_plans(self, params: SimulationParams) -> None:
        """Run i of any scenario shares stream, seed and slot with normal run i."""
        normal = normal_run_plans(params)
        for index, expected in enumerate(normal):
            made = make_run_plan(params, index, ScenarioKind.NORMAL)
            assert made == expected

    def test_a_faulted_run_is_the_exact_counterfactual(self, params: SimulationParams) -> None:
        """Same index: same load and noise streams as the normal run."""
        episode = InstrumentEpisode("TC-HL-1", InstrumentFaultKind.STUCK, FaultLevel.LOW, 8, 6)
        faulted = make_run_plan(params, 1, ScenarioKind.STUCK, instrument_faults=(episode,))
        normal = make_run_plan(params, 1, ScenarioKind.NORMAL)
        assert faulted.spawn_key == normal.spawn_key
        assert faulted.start_time == normal.start_time
        assert np.array_equal(faulted.rngs()[1].normal(size=4), normal.rngs()[1].normal(size=4))

    def test_custom_seed_and_negative_index(self, params: SimulationParams) -> None:
        """A custom master seed is stored; a negative index is rejected."""
        assert make_run_plan(params, 0, ScenarioKind.NORMAL, seed=5).seed_entropy == 5
        with pytest.raises(ValueError, match="run_index"):
            make_run_plan(params, -1, ScenarioKind.NORMAL)

    def test_fault_kind_tables(self) -> None:
        """Eight fault types: five on the instrument axis, three on the process axis."""
        assert len(FAULT_KINDS) == 8
        assert len(INSTRUMENT_KINDS) == 5
        assert {k for k in FAULT_KINDS if k.axis is FaultAxis.INSTRUMENT} == {
            k.scenario for k in INSTRUMENT_KINDS
        }


class TestLoadProfile:
    """The normal-operation load program is bounded and reproducible."""

    def test_loads_stay_inside_the_limits(self, profile: LoadProfile) -> None:
        """No sample leaves [load_min, load_max]."""
        loads = profile.sample(np.random.default_rng(1), 2000, 180.0)
        assert loads.shape == (2000,)
        assert loads.min() >= 0.70 - 1e-12
        assert loads.max() <= 1.00 + 1e-12

    def test_ramp_between_samples_is_bounded(self, profile: LoadProfile) -> None:
        """The slope between consecutive samples never exceeds the maximum ramp."""
        dt = 180.0
        loads = profile.sample(np.random.default_rng(2), 3000, dt)
        slope_percent_per_min = np.abs(np.diff(loads)) / dt * 60.0 * 100.0
        assert slope_percent_per_min.max() <= 5.0 + 1e-9

    def test_the_profile_actually_ramps(self, profile: LoadProfile) -> None:
        """A long profile contains both holds and ramps."""
        loads = profile.sample(np.random.default_rng(3), 3000, 180.0)
        steps = np.abs(np.diff(loads))
        assert (steps == 0).any()
        assert (steps > 1e-4).any()

    def test_warmup_is_constant(self, profile: LoadProfile) -> None:
        """The first warmup_steps samples are held at the initial load."""
        loads = profile.sample(np.random.default_rng(4), 100, 180.0)
        assert np.all(loads[: profile.warmup_steps + 1] == loads[0])

    def test_same_seed_same_profile(self, profile: LoadProfile) -> None:
        """The profile is a pure function of the generator state."""
        a = profile.sample(np.random.default_rng(5), 500, 180.0)
        b = profile.sample(np.random.default_rng(5), 500, 180.0)
        assert np.array_equal(a, b)

    def test_different_seed_different_profile(self, profile: LoadProfile) -> None:
        """Another seed gives another profile."""
        a = profile.sample(np.random.default_rng(5), 500, 180.0)
        b = profile.sample(np.random.default_rng(6), 500, 180.0)
        assert not np.array_equal(a, b)

    def test_physical_profile_is_independent_of_cadence(self) -> None:
        """The same generator gives the same load in physical time at any cadence."""
        spec = LoadProfile(0.7, 1.0, 5.0, 60, 360, warmup_steps=0)
        coarse = spec.sample(np.random.default_rng(7), 41, 180.0)  # 2 h
        fine = spec.sample(np.random.default_rng(7), 721, 10.0)  # 2 h
        assert np.allclose(coarse, fine[::18], atol=1e-12)

    def test_from_params(self, params: SimulationParams) -> None:
        """The profile takes its limits and warm-up from the simulation section."""
        built = LoadProfile.from_params(params)
        assert built.load_min == params.load_min_fraction
        assert built.max_ramp_percent_per_min == params.max_ramp_percent_per_min
        assert built.warmup_steps == params.warmup_steps
        assert math.isclose(built.max_ramp_fraction_per_s, 5.0 / 100.0 / 60.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"load_min": 0.9, "load_max": 0.8},
            {"load_min": 0.0},
            {"max_ramp_percent_per_min": 0.0},
            {"max_ramp_percent_per_min": float("nan")},
            {"hold_minutes_min": 30.0, "hold_minutes_max": 10.0},
            {"hold_minutes_min": 0.0},
            {"warmup_steps": -1},
        ],
    )
    def test_invalid_profiles_are_rejected(self, kwargs: dict[str, float]) -> None:
        """Inconsistent limits fail at construction."""
        base = {
            "load_min": 0.7,
            "load_max": 1.0,
            "max_ramp_percent_per_min": 5.0,
            "hold_minutes_min": 6.0,
            "hold_minutes_max": 20.0,
            "warmup_steps": 0,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            LoadProfile(**base)  # type: ignore[arg-type]

    def test_sample_needs_two_steps(self, profile: LoadProfile) -> None:
        """A profile of fewer than two samples is meaningless."""
        with pytest.raises(ValueError, match="at least 2"):
            profile.sample(np.random.default_rng(0), 1, 180.0)

    def test_sample_needs_positive_dt(self, profile: LoadProfile) -> None:
        """dt_s must be strictly positive."""
        with pytest.raises(ValueError, match="dt_s"):
            profile.sample(np.random.default_rng(0), 10, 0.0)


class TestRunPlans:
    """Plans give each run its own reproducible streams and a unique time slot."""

    def test_one_plan_per_run(self, params: SimulationParams) -> None:
        """normal_run_plans returns params.n_runs normal plans with stable ids."""
        plans = normal_run_plans(params)
        assert len(plans) == params.n_runs == 3
        assert [p.run_id for p in plans] == ["sim_run_001", "sim_run_002", "sim_run_003"]
        assert all(p.scenario is ScenarioKind.NORMAL for p in plans)
        assert all(p.n_steps == params.run_length_steps for p in plans)

    def test_run_id_widens_past_999(self) -> None:
        """Three digits up to 999, wider afterwards."""
        assert run_id_for(0) == "sim_run_001"
        assert run_id_for(998) == "sim_run_999"
        assert run_id_for(999) == "sim_run_1000"

    def test_runs_are_contiguous_in_time(self, params: SimulationParams) -> None:
        """Run i starts where run i-1 ends: no timestamp is shared between runs."""
        plans = normal_run_plans(params)
        for previous, current in zip(plans, plans[1:], strict=False):
            end = previous.timestamps(params.dt_seconds)[-1]
            assert current.start_time == end + timedelta(seconds=params.dt_seconds)

    def test_timestamps_are_strictly_increasing_and_aware(self, params: SimulationParams) -> None:
        """Timestamps are timezone-aware and spaced by dt."""
        stamps = normal_run_plans(params)[0].timestamps(params.dt_seconds)
        assert len(stamps) == params.run_length_steps
        gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
        assert all(gap == timedelta(seconds=180) for gap in gaps)
        assert stamps[0].utcoffset() is not None

    def test_spawn_keys_are_distinct(self, params: SimulationParams) -> None:
        """Each run descends from the master seed through a different spawn key."""
        plans = normal_run_plans(params)
        assert len({p.spawn_key for p in plans}) == len(plans)
        assert all(p.seed_entropy == params.openmc_seed for p in plans)

    def test_rngs_are_reproducible(self, params: SimulationParams) -> None:
        """Asking a plan for its generators twice gives the same streams."""
        plan = normal_run_plans(params)[0]
        first = [g.normal(size=5) for g in plan.rngs()]
        second = [g.normal(size=5) for g in plan.rngs()]
        for a, b in zip(first, second, strict=True):
            assert np.array_equal(a, b)

    def test_streams_are_independent(self, params: SimulationParams) -> None:
        """The load, noise and fault streams of a run do not coincide."""
        draws = [g.normal(size=5) for g in normal_run_plans(params)[0].rngs()]
        assert len(draws) == N_RUN_STREAMS
        assert not np.array_equal(draws[0], draws[1])
        assert not np.array_equal(draws[1], draws[2])

    def test_runs_have_different_streams(self, params: SimulationParams) -> None:
        """Two runs of the same dataset draw different noise."""
        plans = normal_run_plans(params)
        a = plans[0].rngs()[1].normal(size=5)
        b = plans[1].rngs()[1].normal(size=5)
        assert not np.array_equal(a, b)

    def test_custom_seed_and_run_count(self, params: SimulationParams) -> None:
        """The span derivation can observe other runs with its own seed."""
        own = normal_run_plans(params, seed=123, n_runs=5)
        assert len(own) == 5
        assert all(p.seed_entropy == 123 for p in own)
        default = normal_run_plans(params)
        assert own[0].rngs()[0].normal() != default[0].rngs()[0].normal()

    def test_run_count_must_be_positive(self, params: SimulationParams) -> None:
        """n_runs < 1 is rejected."""
        with pytest.raises(ValueError, match="n_runs"):
            normal_run_plans(params, n_runs=0)

    def test_plan_is_immutable(self, params: SimulationParams) -> None:
        """RunPlan is frozen."""
        plan: RunPlan = normal_run_plans(params)[0]
        with pytest.raises(AttributeError):
            plan.n_steps = 5  # type: ignore[misc]
