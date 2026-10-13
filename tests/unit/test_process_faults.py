"""Unit tests for data.simulation.process_faults (F3-M4).

Cubren la configuracion tipada de los niveles (`fault_tiers:`), las entradas del
modelo de cada fallo, la respuesta MEDIDA de cada nivel frente a su banda objetivo
(monotonia, signo de la realimentacion, rango de validez), y los runs completos:
instante de inicio exacto, severidad de la tabla TDD, conservacion durante el
transitorio y determinismo.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.generators.reactor_simulator import ReactorSimulator
from data.generators.tep_adapter import readings_from_frame
from data.simulation.dataset_builder import RunSimulation, wide_frame_from_matrix
from data.simulation.physics_constraints import (
    ConstraintLimits,
    conservation_report,
    evaluate_rules,
)
from data.simulation.plant_params import load_neutronics, load_plant_params
from data.simulation.process_faults import (
    MAX_REACTIVITY_PCM,
    MIN_PUMP_DEMAND,
    PROCESS_KINDS,
    RESPONSE_METRICS,
    FaultTiers,
    KindTiers,
    LevelResponse,
    LevelSpec,
    ResponseTarget,
    aliased_period_s,
    build_fault_inputs,
    is_process_scenario,
    leak_for_margin,
    measure_level_response,
    min_observable_period_s,
    parse_fault_tiers,
    process_run_plan,
    response_from_run,
    unmet_targets,
)
from data.simulation.scenario import (
    FAULT_SEVERITY,
    FaultLevel,
    ProcessFaultSpec,
    ScenarioKind,
    normal_run_plans,
)
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.simulation_params import SimulationParams, load_simulation_params
from data.simulation.thermal_model import ModelInputs, ReactorModel, StateIndex
from data.simulation.water_properties import load_water_table
from tests.support.sim_frames import CONFIGS, build_catalog, build_model, write_sim_params

EXCURSION = ScenarioKind.CORE_TEMP_EXCURSION
FLOW = ScenarioKind.COOLANT_FLOW_REDUCTION
VOID = ScenarioKind.VOID_FORMATION
LEVELS = (FaultLevel.LOW, FaultLevel.MEDIUM, FaultLevel.HIGH)
DT = 180.0


@pytest.fixture(scope="module")
def model() -> ReactorModel:
    """Return the reference model."""
    return build_model()


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned sensor catalog."""
    return build_catalog()


@pytest.fixture(scope="module")
def raw_tiers() -> dict[str, Any]:
    """Return the raw `fault_tiers:` block of the versioned plant parameters."""
    return load_plant_params(CONFIGS / "plant_parameters.yaml").fault_tiers


@pytest.fixture(scope="module")
def tiers(raw_tiers: dict[str, Any]) -> FaultTiers:
    """Return the parsed process-fault levels."""
    return parse_fault_tiers(raw_tiers)


@pytest.fixture(scope="module")
def params(tmp_path_factory: pytest.TempPathFactory) -> SimulationParams:
    """Return a small simulation configuration (runs of 40 steps, warm-up 10)."""
    directory = tmp_path_factory.mktemp("faults")
    path = write_sim_params(
        directory, run_length_steps=40, n_samples=80, warmup_fraction=0.25
    )
    return load_simulation_params(path)


@pytest.fixture(scope="module")
def responses(
    model: ReactorModel, catalog: SensorCatalog, tiers: FaultTiers
) -> dict[tuple[ScenarioKind, FaultLevel], LevelResponse]:
    """Return the measured response of the nine levels at nominal load."""
    return {
        (kind, level): measure_level_response(model, catalog, tiers.for_kind(kind), level)
        for kind in PROCESS_KINDS
        for level in LEVELS
    }


@pytest.fixture(scope="module")
def void_at_part_load(
    model: ReactorModel, catalog: SensorCatalog, tiers: FaultTiers
) -> dict[FaultLevel, LevelResponse]:
    """Return the three void levels measured at 70 % load."""
    kind = tiers.for_kind(VOID)
    return {
        level: measure_level_response(model, catalog, kind, level, load=0.7) for level in LEVELS
    }


def _mutated(raw: dict[str, Any], mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Return a deep copy of the raw tiers after applying a mutation."""
    clone = copy.deepcopy(raw)
    mutate(clone)
    return clone


# ----------------------------------------------------------------------
# Configuracion tipada
# ----------------------------------------------------------------------


class TestParseFaultTiers:
    """`fault_tiers:` resolves to typed levels or fails naming the cause."""

    def test_repository_block_resolves(self, tiers: FaultTiers) -> None:
        """The three process faults, each with low, medium and high."""
        assert set(tiers.kinds) == set(PROCESS_KINDS)
        for kind in PROCESS_KINDS:
            block = tiers.for_kind(kind)
            assert set(block.levels) == set(LEVELS)
            assert block.hold_steps >= 1
            assert block.recovery_steps >= 0

    def test_levels_grow_in_severity(self, tiers: FaultTiers) -> None:
        """Insertion and leak grow with the level; the demand and the margin fall."""
        pcm = [tiers.for_kind(EXCURSION).level(lv).magnitude for lv in LEVELS]
        demand = [tiers.for_kind(FLOW).level(lv).magnitude for lv in LEVELS]
        margin = [tiers.for_kind(VOID).level(lv).magnitude for lv in LEVELS]
        assert pcm == sorted(pcm)
        assert demand == sorted(demand, reverse=True)
        assert margin == sorted(margin, reverse=True)

    def test_every_level_has_a_measurable_target(self, tiers: FaultTiers) -> None:
        """No level is defined by eye: each one carries at least one band."""
        for kind in PROCESS_KINDS:
            for level in LEVELS:
                targets = tiers.for_kind(kind).level(level).targets
                assert targets
                assert all(t.metric in RESPONSE_METRICS for t in targets)

    def test_only_the_flow_fault_oscillates(self, tiers: FaultTiers) -> None:
        """The oscillatory term belongs to the pump fault alone."""
        assert all(tiers.for_kind(EXCURSION).level(lv).oscillation_amplitude == 0 for lv in LEVELS)
        assert all(tiers.for_kind(VOID).level(lv).oscillation_amplitude == 0 for lv in LEVELS)
        assert all(tiers.for_kind(FLOW).level(lv).oscillation_amplitude > 0 for lv in LEVELS)

    def test_flow_oscillation_is_faster_than_the_nyquist_period(self, tiers: FaultTiers) -> None:
        """The declared oscillation cannot be represented at the dataset cadence."""
        period = tiers.for_kind(FLOW).oscillation_period_s
        assert period < min_observable_period_s(DT)

    @pytest.mark.parametrize(
        ("name", "mutate", "error", "match"),
        [
            (
                "missing kind",
                lambda r: r.pop("void_formation"),
                KeyError,
                "void_formation",
            ),
            (
                "missing level",
                lambda r: r["core_temp_excursion"]["levels"].pop("high"),
                KeyError,
                "high",
            ),
            (
                "missing hold",
                lambda r: r["core_temp_excursion"].pop("hold_steps"),
                KeyError,
                "hold_steps",
            ),
            (
                "missing magnitude",
                lambda r: r["void_formation"]["levels"]["low"].pop("saturation_margin_k"),
                KeyError,
                "saturation_margin_k",
            ),
            (
                "kind not a mapping",
                lambda r: r.update(void_formation=[1, 2]),
                TypeError,
                "mapping",
            ),
            (
                "expect not a mapping",
                lambda r: r["void_formation"]["levels"]["low"].update(expect=[1]),
                TypeError,
                "expect",
            ),
            (
                "band wrong length",
                lambda r: r["void_formation"]["levels"]["low"].update(expect={"void_peak": [1]}),
                TypeError,
                "pair",
            ),
            (
                "insertion above the limit",
                lambda r: r["core_temp_excursion"]["levels"]["high"].update(
                    reactivity_pcm=MAX_REACTIVITY_PCM + 100
                ),
                ValueError,
                "validity range",
            ),
            (
                "demand below the limit",
                lambda r: r["coolant_flow_reduction"]["levels"]["high"].update(
                    demand_fraction=MIN_PUMP_DEMAND - 0.05
                ),
                ValueError,
                "validity range",
            ),
            (
                "margin below the limit",
                lambda r: r["void_formation"]["levels"]["high"].update(saturation_margin_k=-30.0),
                ValueError,
                "validity range",
            ),
            (
                "levels not increasing",
                lambda r: r["core_temp_excursion"]["levels"]["low"].update(reactivity_pcm=650.0),
                ValueError,
                "increase in severity",
            ),
            (
                "amplitude too large",
                lambda r: r["coolant_flow_reduction"]["levels"]["low"].update(
                    oscillation_amplitude=0.5
                ),
                ValueError,
                "oscillation_amplitude",
            ),
            (
                "oscillation outside the pump fault",
                lambda r: r["void_formation"]["levels"]["low"].update(oscillation_amplitude=0.01),
                ValueError,
                "only the flow reduction",
            ),
            (
                "flow with a ramp onset",
                lambda r: r["coolant_flow_reduction"].update(onset_minutes=5.0),
                ValueError,
                "must be 0",
            ),
            (
                "ramped fault without onset",
                lambda r: r["core_temp_excursion"].update(onset_minutes=0.0),
                ValueError,
                "must be > 0",
            ),
            (
                "flow without time constant",
                lambda r: r["coolant_flow_reduction"].update(time_constant_minutes=0.0),
                ValueError,
                "time_constant_minutes",
            ),
            (
                "fractional hold",
                lambda r: r["core_temp_excursion"].update(hold_steps=2.5),
                ValueError,
                "whole number",
            ),
            (
                "zero hold",
                lambda r: r["core_temp_excursion"].update(hold_steps=0),
                ValueError,
                "whole number",
            ),
            (
                "negative recovery",
                lambda r: r["core_temp_excursion"].update(recovery_steps=-1),
                ValueError,
                "whole number",
            ),
            (
                "unknown metric",
                lambda r: r["void_formation"]["levels"]["low"].update(expect={"colour": [0, 1]}),
                ValueError,
                "unknown metric",
            ),
            (
                "inverted band",
                lambda r: r["void_formation"]["levels"]["low"].update(
                    expect={"void_peak": [0.5, 0.1]}
                ),
                ValueError,
                "exceeds",
            ),
            (
                "text magnitude",
                lambda r: r["core_temp_excursion"]["levels"]["low"].update(reactivity_pcm="big"),
                ValueError,
                "must be a number",
            ),
            (
                "non finite magnitude",
                lambda r: r["core_temp_excursion"]["levels"]["low"].update(
                    reactivity_pcm=float("nan")
                ),
                ValueError,
                "finite",
            ),
        ],
    )
    def test_invalid_blocks_fail_with_a_named_cause(
        self,
        raw_tiers: dict[str, Any],
        name: str,
        mutate: Callable[[dict[str, Any]], None],
        error: type[Exception],
        match: str,
    ) -> None:
        """Each broken block raises the right exception and names the key."""
        with pytest.raises(error, match=match):
            parse_fault_tiers(_mutated(raw_tiers, mutate))

    def test_for_kind_rejects_non_process_scenarios(self, tiers: FaultTiers) -> None:
        """Normal operation and the instrument faults have no levels."""
        with pytest.raises(KeyError, match="not a process fault"):
            tiers.for_kind(ScenarioKind.NORMAL)
        with pytest.raises(KeyError):
            tiers.for_kind(ScenarioKind.STUCK)

    def test_is_process_scenario(self) -> None:
        """Exactly the three process faults are on the process axis."""
        assert [k for k in ScenarioKind if is_process_scenario(k)] == list(PROCESS_KINDS)


class TestKindTiers:
    """Cadence-dependent quantities of one fault type."""

    def test_ramped_faults_have_an_onset_in_steps(self, tiers: FaultTiers) -> None:
        """30 minutes at 180 s is 10 samples, and never fewer than one."""
        assert tiers.for_kind(EXCURSION).onset_steps(DT) == 10
        assert tiers.for_kind(VOID).onset_steps(DT) == 10
        assert tiers.for_kind(EXCURSION).onset_steps(3600.0) == 1

    def test_flow_reduction_has_no_ramp(self, tiers: FaultTiers) -> None:
        """Its onset is the time constant, not a ramp."""
        assert tiers.for_kind(FLOW).onset_steps(DT) == 0

    def test_only_the_flow_fault_is_sub_stepped(self, tiers: FaultTiers) -> None:
        """Four sub-steps of 45 s resolve the 300 s oscillation; the rest use one."""
        assert tiers.for_kind(FLOW).substeps(DT) == 4
        assert tiers.for_kind(EXCURSION).substeps(DT) == 1
        assert tiers.for_kind(VOID).substeps(DT) == 1

    def test_substeps_resolve_the_oscillation(self, tiers: FaultTiers) -> None:
        """At least six points per period, whatever the cadence."""
        flow = tiers.for_kind(FLOW)
        for dt in (60.0, 180.0, 600.0):
            sub_dt = dt / flow.substeps(dt)
            assert flow.oscillation_period_s / sub_dt >= 6.0 - 1e-9


class TestResponseTarget:
    """A band is inclusive on both ends."""

    def test_contains(self) -> None:
        """Edges belong to the band."""
        band = ResponseTarget(metric="n_peak", low=1.0, high=1.2)
        assert band.contains(1.0)
        assert band.contains(1.2)
        assert not band.contains(1.2001)
        assert not band.contains(0.99)


# ----------------------------------------------------------------------
# Observabilidad por cadencia
# ----------------------------------------------------------------------


class TestObservability:
    """What a 180 s cadence can and cannot see of a fast oscillation."""

    def test_nyquist_period(self) -> None:
        """Two samples per period: 360 s at 180 s."""
        assert min_observable_period_s(180.0) == 360.0

    @pytest.mark.parametrize(
        ("period", "expected"),
        [(300.0, 450.0), (120.0, 360.0), (900.0, 900.0), (360.0, 360.0)],
    )
    def test_aliased_period(self, period: float, expected: float) -> None:
        """Sampling folds a fast oscillation into a slower apparent one."""
        assert aliased_period_s(period, 180.0) == pytest.approx(expected)

    def test_multiple_of_the_cadence_looks_constant(self) -> None:
        """A period equal to the cadence (or a multiple) has no apparent oscillation."""
        assert aliased_period_s(180.0, 180.0) is None
        assert aliased_period_s(90.0, 180.0) is None

    def test_alias_matches_a_sampled_sinusoid(self) -> None:
        """The folded period is the one a sampled 300 s sinusoid actually shows."""
        k = np.arange(400)
        samples = np.sin(2.0 * np.pi * k * 180.0 / 300.0)
        spectrum = np.abs(np.fft.rfft(samples))
        frequency = np.fft.rfftfreq(k.size, d=180.0)[int(np.argmax(spectrum))]
        assert 1.0 / frequency == pytest.approx(450.0, rel=0.02)

    @pytest.mark.parametrize("call", [lambda: aliased_period_s(0.0, 180.0),
                                      lambda: aliased_period_s(300.0, 0.0),
                                      lambda: min_observable_period_s(-1.0)])
    def test_rejects_non_positive_arguments(self, call: Callable[[], object]) -> None:
        """Periods and cadences must be strictly positive."""
        with pytest.raises(ValueError, match="> 0"):
            call()


# ----------------------------------------------------------------------
# Entradas del modelo
# ----------------------------------------------------------------------


def _loads(n: int = 30, base: float = 0.9) -> list[float]:
    """Return a slowly varying load profile of n samples."""
    return [base + 0.05 * math.sin(k / 6.0) for k in range(n)]


def _nominal(load: float) -> ModelInputs:
    """Return the inputs of normal operation at a load."""
    return ModelInputs(load=load)


def _spec(**overrides: Any) -> ProcessFaultSpec:
    """Return a spec starting at step 8."""
    base: dict[str, Any] = {
        "level": FaultLevel.MEDIUM,
        "injected_step": 8,
        "onset_steps": 4,
        "hold_steps": 6,
        "recovery_steps": 4,
    }
    base.update(overrides)
    return ProcessFaultSpec(**base)


class TestExcursionInputs:
    """A reactivity ramp with the rod controller disabled."""

    def test_insertion_ramps_up_and_back_down(self, model: ReactorModel, tiers: FaultTiers) -> None:
        """Zero at the start, the level's pcm at full magnitude, zero after recovery."""
        kind = tiers.for_kind(EXCURSION)
        spec = _spec(level=FaultLevel.HIGH)
        inputs = build_fault_inputs(model, kind, FaultLevel.HIGH, spec, _loads(), DT)
        rho = np.array([u.reactivity_insertion for u in inputs.sample])
        full = kind.level(FaultLevel.HIGH).magnitude * 1e-5
        assert rho[8] == 0.0
        assert rho[12] == pytest.approx(full)
        assert rho[18] == pytest.approx(full)
        assert rho[22] == 0.0
        assert np.all(np.diff(rho[8:13]) > 0.0)
        assert np.all(np.diff(rho[18:23]) < 0.0)

    def test_rods_are_disabled_only_inside_the_window(
        self, model: ReactorModel, tiers: FaultTiers
    ) -> None:
        """The PI is frozen from the injection to the end of the recovery."""
        spec = _spec()
        kind = tiers.for_kind(EXCURSION)
        inputs = build_fault_inputs(model, kind, spec.level, spec, _loads(), DT)
        enabled = [u.rod_control_enabled for u in inputs.sample]
        assert all(enabled[:8])
        assert not any(enabled[8:23])
        assert all(enabled[23:])

    def test_start_is_exact(self, model: ReactorModel, tiers: FaultTiers) -> None:
        """Nothing differs from normal operation before injected_step; the rods do at it."""
        spec = _spec()
        loads = _loads()
        inputs = build_fault_inputs(model, tiers.for_kind(EXCURSION), spec.level, spec, loads, DT)
        assert [inputs.sample[k] for k in range(8)] == [_nominal(loads[k]) for k in range(8)]
        assert inputs.sample[8] != _nominal(loads[8])

    def test_after_recovery_inputs_are_nominal_again(
        self, model: ReactorModel, tiers: FaultTiers
    ) -> None:
        """Past the window the inputs equal those of normal operation."""
        spec = _spec()
        loads = _loads()
        inputs = build_fault_inputs(model, tiers.for_kind(EXCURSION), spec.level, spec, loads, DT)
        assert inputs.sample[25] == _nominal(loads[25])

    def test_no_fine_grid(self, model: ReactorModel, tiers: FaultTiers) -> None:
        """The excursion has no sub-sample dynamics."""
        spec = _spec()
        kind = tiers.for_kind(EXCURSION)
        inputs = build_fault_inputs(model, kind, spec.level, spec, _loads(), DT)
        assert inputs.fine is None
        assert inputs.substeps == 1


class TestFlowInputs:
    """A pump demand that decays with a time constant, plus a small oscillation."""

    def test_demand_falls_towards_the_level_fraction(
        self, model: ReactorModel, tiers: FaultTiers
    ) -> None:
        """Starts at 1, falls with 1 - exp(-t/tau) and ends near the fraction."""
        kind = tiers.for_kind(FLOW)
        spec = _spec(onset_steps=0, hold_steps=None, recovery_steps=0, level=FaultLevel.HIGH)
        inputs = build_fault_inputs(model, kind, FaultLevel.HIGH, spec, _loads(60), DT)
        demand = np.array([u.pump_demand for u in inputs.sample])
        level = kind.level(FaultLevel.HIGH)
        assert demand[8] == 1.0
        assert demand[9] < 1.0
        assert demand[-1] == pytest.approx(level.magnitude, abs=level.oscillation_amplitude + 1e-9)
        assert demand[20:].max() < demand[:8].min()

    def test_start_is_exact(self, model: ReactorModel, tiers: FaultTiers) -> None:
        """The first sample of the fault is still nominal (zero elapsed time)."""
        spec = _spec(onset_steps=0)
        loads = _loads()
        inputs = build_fault_inputs(model, tiers.for_kind(FLOW), spec.level, spec, loads, DT)
        assert [inputs.sample[k] for k in range(9)] == [_nominal(loads[k]) for k in range(9)]
        assert inputs.sample[9] != _nominal(loads[9])

    def test_fine_grid_contains_the_samples(self, model: ReactorModel, tiers: FaultTiers) -> None:
        """Every fourth fine point is a sample, and the grid has (n-1) * 4 + 1 points."""
        spec = _spec(onset_steps=0)
        loads = _loads()
        inputs = build_fault_inputs(model, tiers.for_kind(FLOW), spec.level, spec, loads, DT)
        assert inputs.fine is not None
        assert inputs.substeps == 4
        assert len(inputs.fine) == (len(loads) - 1) * 4 + 1
        for k, sample in enumerate(inputs.sample):
            assert inputs.fine[k * 4] == sample

    def test_fine_load_is_the_linear_interpolant(
        self, model: ReactorModel, tiers: FaultTiers
    ) -> None:
        """Inside an interval the load follows the straight line between its samples."""
        spec = _spec(onset_steps=0)
        loads = _loads()
        inputs = build_fault_inputs(model, tiers.for_kind(FLOW), spec.level, spec, loads, DT)
        assert inputs.fine is not None
        for j in range(4):
            expected = loads[3] + (loads[4] - loads[3]) * j / 4
            assert inputs.fine[3 * 4 + j].load == pytest.approx(expected)

    def test_oscillation_is_present_between_samples(
        self, model: ReactorModel, tiers: FaultTiers
    ) -> None:
        """The fine demand swings by more than the sampled one resolves (subtle term)."""
        spec = _spec(onset_steps=0, hold_steps=None, recovery_steps=0)
        inputs = build_fault_inputs(model, tiers.for_kind(FLOW), spec.level, spec, _loads(40), DT)
        assert inputs.fine is not None
        fine = np.array([u.pump_demand for u in inputs.fine])
        amplitude = tiers.for_kind(FLOW).level(spec.level).oscillation_amplitude
        trend = np.convolve(fine, np.ones(7) / 7, mode="same")
        assert np.abs(fine - trend)[40:-40].max() > 0.5 * amplitude


class TestVoidInputs:
    """A leak sized from the saturation-margin target."""

    def test_leak_is_zero_before_and_positive_during(
        self, model: ReactorModel, tiers: FaultTiers
    ) -> None:
        """No leak until the injection; a positive one at full magnitude."""
        spec = _spec(level=FaultLevel.MEDIUM)
        inputs = build_fault_inputs(model, tiers.for_kind(VOID), spec.level, spec, _loads(), DT)
        leak = np.array([u.leak_bar for u in inputs.sample])
        assert np.all(leak[:9] == 0.0)
        assert np.all(leak[12:19] > 0.0)
        assert leak[25] == 0.0

    def test_start_is_exact(self, model: ReactorModel, tiers: FaultTiers) -> None:
        """The first sample of the fault is nominal (the ramp starts at zero)."""
        spec = _spec()
        loads = _loads()
        inputs = build_fault_inputs(model, tiers.for_kind(VOID), spec.level, spec, loads, DT)
        assert [inputs.sample[k] for k in range(9)] == [_nominal(loads[k]) for k in range(9)]
        assert inputs.sample[9] != _nominal(loads[9])

    def test_rejects_degenerate_inputs(self, model: ReactorModel, tiers: FaultTiers) -> None:
        """At least two loads and a positive cadence."""
        kind = tiers.for_kind(VOID)
        with pytest.raises(ValueError, match="at least 2"):
            build_fault_inputs(model, kind, FaultLevel.LOW, _spec(), [0.9], DT)
        with pytest.raises(ValueError, match="dt_s"):
            build_fault_inputs(model, kind, FaultLevel.LOW, _spec(), _loads(), 0.0)


class TestLeakForMargin:
    """The leak that produces a saturation-margin target."""

    @pytest.mark.parametrize("load", [0.7, 0.85, 1.0])
    @pytest.mark.parametrize("margin", [-8.0, 0.0, 1.5, 6.0])
    def test_saturation_lands_on_the_target(
        self, model: ReactorModel, load: float, margin: float
    ) -> None:
        """T_sat(P - leak) sits `margin` K above the equilibrium T_out."""
        state = model.steady_state(load)
        leak = leak_for_margin(model, load, margin)
        pressure = float(state[StateIndex.PRESSURE]) - leak
        t_sat = float(model.table.t_sat_c(pressure))
        assert t_sat - float(state[StateIndex.T_CORE]) == pytest.approx(margin, abs=0.05)

    def test_lower_margin_needs_more_leak(self, model: ReactorModel) -> None:
        """Going deeper into boiling needs a bigger depressurization."""
        leaks = [leak_for_margin(model, 1.0, m) for m in (6.0, 1.5, 0.0, -8.0)]
        assert leaks == sorted(leaks)
        assert leaks[0] < leaks[-1]

    def test_leak_is_never_negative(self, model: ReactorModel) -> None:
        """A margin the plant already exceeds needs no leak."""
        assert leak_for_margin(model, 1.0, 30.0) == 0.0

    def test_same_level_needs_a_different_leak_at_part_load(self, model: ReactorModel) -> None:
        """That is why the level is a margin and not a number of bar."""
        full = leak_for_margin(model, 1.0, -5.0)
        part = leak_for_margin(model, 0.7, -5.0)
        assert abs(full - part) > 5.0

    def test_rejects_a_target_outside_the_table(self, model: ReactorModel) -> None:
        """The table does not extrapolate: an impossible depressurization is an error."""
        with pytest.raises(ValueError, match="outside the water table"):
            leak_for_margin(model, 1.0, -90.0)


# ----------------------------------------------------------------------
# Planes
# ----------------------------------------------------------------------


class TestProcessRunPlan:
    """The plan of a process-fault run."""

    def test_defaults_come_from_the_tiers(
        self, params: SimulationParams, tiers: FaultTiers
    ) -> None:
        """Start at the end of the warm-up, timeline from fault_tiers."""
        kind = tiers.for_kind(EXCURSION)
        plan = process_run_plan(params, kind, FaultLevel.HIGH, hold_steps=5, recovery_steps=2)
        spec = plan.process_fault
        assert spec is not None
        assert plan.scenario is EXCURSION
        assert spec.injected_step == params.warmup_steps == 10
        assert spec.onset_steps == kind.onset_steps(params.dt_seconds)
        assert spec.level is FaultLevel.HIGH
        assert plan.n_steps == params.run_length_steps

    def test_same_seed_as_the_normal_run_of_that_index(
        self, params: SimulationParams, tiers: FaultTiers
    ) -> None:
        """The (fault, normal) pair is an exact counterfactual until the injection."""
        normal = normal_run_plans(params)
        for index in (0, 1):
            plan = process_run_plan(
                params, tiers.for_kind(VOID), FaultLevel.LOW, run_index=index, persistent=True
            )
            assert plan.spawn_key == normal[index].spawn_key
            assert plan.seed_entropy == normal[index].seed_entropy
            assert plan.start_time == normal[index].start_time
            assert plan.run_id == normal[index].run_id

    def test_persistent_fault_has_no_recovery(
        self, params: SimulationParams, tiers: FaultTiers
    ) -> None:
        """persistent=True lasts to the end of the run."""
        plan = process_run_plan(params, tiers.for_kind(FLOW), FaultLevel.LOW, persistent=True)
        assert plan.process_fault is not None
        assert plan.process_fault.hold_steps is None
        assert plan.process_fault.recovery_steps == 0

    def test_explicit_timeline_overrides_the_defaults(
        self, params: SimulationParams, tiers: FaultTiers
    ) -> None:
        """Start, hold and recovery can be set per run."""
        plan = process_run_plan(
            params,
            tiers.for_kind(EXCURSION),
            FaultLevel.LOW,
            injected_step=5,
            hold_steps=7,
            recovery_steps=2,
            seed=99,
        )
        spec = plan.process_fault
        assert spec is not None
        assert (spec.injected_step, spec.hold_steps, spec.recovery_steps) == (5, 7, 2)
        assert plan.seed_entropy == 99

    def test_window_must_fit_in_the_run(self, params: SimulationParams, tiers: FaultTiers) -> None:
        """The default 120-step hold does not fit a 40-step run."""
        with pytest.raises(ValueError, match="does not fit"):
            process_run_plan(params, tiers.for_kind(EXCURSION), FaultLevel.LOW)

    def test_rejects_a_negative_index(self, params: SimulationParams, tiers: FaultTiers) -> None:
        """run_index is a position."""
        with pytest.raises(ValueError, match="run_index"):
            process_run_plan(
                params, tiers.for_kind(VOID), FaultLevel.LOW, run_index=-1, persistent=True
            )


# ----------------------------------------------------------------------
# Respuesta medida de cada nivel
# ----------------------------------------------------------------------


class TestLevelBands:
    """Every level reaches the objective that defines it."""

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    @pytest.mark.parametrize("level", LEVELS)
    def test_measured_response_is_inside_the_bands(
        self,
        responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse],
        tiers: FaultTiers,
        kind: ScenarioKind,
        level: FaultLevel,
    ) -> None:
        """unmet_targets is empty: the level is where fault_tiers says it is."""
        spec = tiers.for_kind(kind).level(level)
        assert unmet_targets(responses[(kind, level)], spec) == []

    @pytest.mark.parametrize("level", LEVELS)
    def test_void_bands_hold_at_part_load(
        self,
        void_at_part_load: dict[FaultLevel, LevelResponse],
        tiers: FaultTiers,
        level: FaultLevel,
    ) -> None:
        """The void level means the same at 70 % load (it is defined by the margin)."""
        spec = tiers.for_kind(VOID).level(level)
        assert unmet_targets(void_at_part_load[level], spec) == []

    def test_unmet_targets_reports_the_miss(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse]
    ) -> None:
        """A level measured against another level's bands is reported, with the number."""
        wrong = LevelSpec(
            magnitude=0.0,
            oscillation_amplitude=0.0,
            targets=(ResponseTarget("n_peak", 2.0, 3.0),),
        )
        missed = unmet_targets(responses[(EXCURSION, FaultLevel.LOW)], wrong)
        assert len(missed) == 1
        assert "n_peak" in missed[0]


class TestMonotonicResponse:
    """A higher level is a bigger fault, in the metric that defines it."""

    def test_excursion_grows(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse]
    ) -> None:
        """Power, fuel and outlet temperature rise with the inserted reactivity."""
        r = [responses[(EXCURSION, lv)] for lv in LEVELS]
        for metric in ("n_peak", "t_fuel_peak_c", "t_out_peak_c", "p_max_bar"):
            values = [getattr(x, metric) for x in r]
            assert values == sorted(values), metric
        margins = [x.margin_min_k for x in r]
        assert margins == sorted(margins, reverse=True)

    def test_flow_reduction_grows(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse]
    ) -> None:
        """Less flow, hotter outlet, bigger core delta-T, thinner margin."""
        r = [responses[(FLOW, lv)] for lv in LEVELS]
        assert [x.flow_min_kg_s for x in r] == sorted((x.flow_min_kg_s for x in r), reverse=True)
        assert [x.t_out_peak_c for x in r] == sorted(x.t_out_peak_c for x in r)
        assert [x.delta_t_peak_k for x in r] == sorted(x.delta_t_peak_k for x in r)
        assert [x.margin_min_k for x in r] == sorted((x.margin_min_k for x in r), reverse=True)

    def test_void_grows(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse]
    ) -> None:
        """More void, lower pressure, less power."""
        r = [responses[(VOID, lv)] for lv in LEVELS]
        assert [x.void_peak for x in r] == sorted(x.void_peak for x in r)
        assert [x.p_min_bar for x in r] == sorted((x.p_min_bar for x in r), reverse=True)
        assert [x.n_min for x in r] == sorted((x.n_min for x in r), reverse=True)

    def test_void_is_monotone_at_part_load_too(
        self, void_at_part_load: dict[FaultLevel, LevelResponse]
    ) -> None:
        """The ordering does not depend on the load."""
        r = [void_at_part_load[lv] for lv in LEVELS]
        assert [x.void_peak for x in r] == sorted(x.void_peak for x in r)
        assert [x.margin_min_k for x in r] == sorted((x.margin_min_k for x in r), reverse=True)


class TestFeedbackSigns:
    """The signature of each fault has the physical sign."""

    def test_excursion_raises_power_and_temperatures(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse], model: ReactorModel
    ) -> None:
        """n, T_f and T_out rise; pressure follows the average temperature."""
        r = responses[(EXCURSION, FaultLevel.HIGH)]
        assert r.n_peak > 1.0
        assert r.t_fuel_peak_c > model.plant.t_fuel_nominal_c
        assert r.t_out_rise_k > 0.0
        assert r.p_max_bar > model.plant.nominal_pressure_bar
        assert r.void_peak == pytest.approx(0.0, abs=1e-9)

    def test_flow_reduction_cuts_flow_and_raises_delta_t(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse], model: ReactorModel
    ) -> None:
        """The flow falls, T_out and the core delta-T rise, the power barely moves."""
        r = responses[(FLOW, FaultLevel.HIGH)]
        nominal_delta = model.nominal.t_out_c - model.plant.t_in_nominal_c
        assert r.flow_min_kg_s < model.plant.nominal_flow_kg_s
        assert r.t_out_rise_k > 0.0
        assert r.delta_t_peak_k > nominal_delta
        assert r.n_peak == pytest.approx(1.0, abs=0.02)

    def test_void_feedback_lowers_the_power(
        self,
        catalog: SensorCatalog,
        tiers: FaultTiers,
        model: ReactorModel,
        responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse],
    ) -> None:
        """The void coefficient is negative: without it the power does not collapse."""
        with_feedback = responses[(VOID, FaultLevel.MEDIUM)]
        neutronics = load_neutronics(CONFIGS / "neutronics_coefficients.yaml")
        no_void = dataclasses.replace(neutronics, alpha_void_per_fraction=0.0)
        counterfactual = ReactorModel(
            model.plant, no_void, load_water_table(CONFIGS / "water_table.yaml")
        )
        reference = measure_level_response(
            counterfactual, catalog, tiers.for_kind(VOID), FaultLevel.MEDIUM
        )
        assert with_feedback.void_peak > 0.1
        assert with_feedback.n_min < 0.5
        assert reference.n_min > with_feedback.n_min + 0.3

    def test_void_depressurizes(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse], model: ReactorModel
    ) -> None:
        """Pressure falls below nominal and the margin to saturation goes negative."""
        r = responses[(VOID, FaultLevel.MEDIUM)]
        assert r.p_min_bar < model.plant.nominal_pressure_bar - 20.0
        assert r.margin_min_k < 0.0

    def test_low_void_level_does_not_boil(
        self, responses: dict[tuple[ScenarioKind, FaultLevel], LevelResponse]
    ) -> None:
        """Incipient void: voids appear, but T_out stays below T_sat (rule 6 silent)."""
        r = responses[(VOID, FaultLevel.LOW)]
        assert r.void_peak > 0.0
        assert r.margin_min_k > 0.0


class TestValidityRange:
    """The limits that keep each fault inside the water table."""

    def test_largest_valid_insertion_stays_in_the_table(
        self, model: ReactorModel, catalog: SensorCatalog, tiers: FaultTiers
    ) -> None:
        """MAX_REACTIVITY_PCM at nominal load stays clear of the pressure ceiling."""
        kind = tiers.for_kind(EXCURSION)
        edge = dataclasses.replace(
            kind,
            levels={FaultLevel.LOW: LevelSpec(MAX_REACTIVITY_PCM, 0.0, ())},
        )
        response = measure_level_response(model, catalog, edge, FaultLevel.LOW)
        assert response.p_max_bar < model.pressure_limits_bar[1] - 1.0

    def test_insertion_past_the_limit_pins_the_pressure(
        self, model: ReactorModel, catalog: SensorCatalog, tiers: FaultTiers
    ) -> None:
        """800 pcm reaches the 170 bar ceiling: the limiter holds it instead of aborting."""
        kind = tiers.for_kind(EXCURSION)
        beyond = dataclasses.replace(
            kind,
            levels={FaultLevel.LOW: LevelSpec(MAX_REACTIVITY_PCM + 100.0, 0.0, ())},
        )
        response = measure_level_response(model, catalog, beyond, FaultLevel.LOW)
        assert response.p_max_bar == pytest.approx(model.pressure_limits_bar[1], abs=1.0)
        assert response.p_max_bar <= model.pressure_limits_bar[1] + 1e-9

    def test_lowest_valid_demand_does_not_boil(
        self, model: ReactorModel, catalog: SensorCatalog, tiers: FaultTiers
    ) -> None:
        """MIN_PUMP_DEMAND with the largest oscillation: no void, margin above zero."""
        kind = tiers.for_kind(FLOW)
        edge = dataclasses.replace(
            kind,
            levels={FaultLevel.LOW: LevelSpec(MIN_PUMP_DEMAND, 0.03, ())},
        )
        response = measure_level_response(model, catalog, edge, FaultLevel.LOW)
        assert response.void_peak == pytest.approx(0.0, abs=1e-9)
        assert response.margin_min_k > 0.0

    def test_demand_past_the_cliff_boils(
        self, model: ReactorModel, catalog: SensorCatalog, tiers: FaultTiers
    ) -> None:
        """At 0,40 the pump fault has become a boiling fault: the reason for the limit."""
        kind = tiers.for_kind(FLOW)
        beyond = dataclasses.replace(kind, levels={FaultLevel.LOW: LevelSpec(0.40, 0.0, ())})
        response = measure_level_response(model, catalog, beyond, FaultLevel.LOW)
        assert response.margin_min_k < 0.0
        assert response.void_peak > 0.1

    def test_measure_rejects_non_positive_step_counts(
        self, model: ReactorModel, catalog: SensorCatalog, tiers: FaultTiers
    ) -> None:
        """settle_steps and response_steps must be at least 1."""
        kind = tiers.for_kind(EXCURSION)
        with pytest.raises(ValueError, match=">= 1"):
            measure_level_response(model, catalog, kind, FaultLevel.LOW, settle_steps=0)

    def test_response_from_run_rejects_a_start_outside_the_run(
        self,
        model: ReactorModel,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
    ) -> None:
        """The first considered sample must exist."""
        sim, _ = runs[EXCURSION]
        with pytest.raises(ValueError, match="outside the run"):
            response_from_run(sim, model, sim.n_steps)
        with pytest.raises(ValueError, match="outside the run"):
            response_from_run(sim, model, -1)


# ----------------------------------------------------------------------
# Runs completos
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulator(params: SimulationParams) -> ReactorSimulator:
    """Return a simulator over the small configuration."""
    return ReactorSimulator.from_params(params)


@pytest.fixture(scope="module")
def runs(
    simulator: ReactorSimulator,
) -> dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]]:
    """Return the integrated run and the long frame of each process fault (medium level)."""
    out: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]] = {}
    for kind in PROCESS_KINDS:
        plan = simulator.process_plan(kind, FaultLevel.MEDIUM, hold_steps=8, recovery_steps=4)
        out[kind] = (simulator.simulate(plan), simulator.generate_run(plan))
    return out


class TestProcessRuns:
    """A process-fault run end to end: labels, start, conservation, determinism."""

    INJECTED = 10
    """Fin del calentamiento: 40 pasos x 0,25."""

    @staticmethod
    def _last(sim: RunSimulation) -> int:
        """Return the last labelled step of a run (its window is finite in these runs)."""
        assert sim.plan.process_fault is not None
        last = sim.plan.process_fault.last_active_step
        assert last is not None
        return last

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_labels_start_exactly_at_the_injection(
        self, runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]], kind: ScenarioKind
    ) -> None:
        """The first labelled step is injected_step, and injected_at is its timestamp."""
        sim, frame = runs[kind]
        labelled = frame[frame["process_fault"] != "none"]
        assert labelled["timestep"].min() == self.INJECTED
        assert (labelled["injected_at"] == pd.Timestamp(sim.timestamps[self.INJECTED])).all()
        assert labelled["timestep"].max() == self._last(sim)

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_severity_is_the_tdd_table(
        self, runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]], kind: ScenarioKind
    ) -> None:
        """process_severity equals FAULT_SEVERITY, and corrupted stays False."""
        _, frame = runs[kind]
        labelled = frame[frame["process_fault"] != "none"]
        assert set(labelled["process_severity"]) == {FAULT_SEVERITY[kind].value}
        assert set(labelled["process_fault"]) == {kind.value}
        assert not frame["corrupted"].any()
        assert (frame["scenario"] == kind.value).all()

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_every_sensor_is_labelled_in_the_window(
        self,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
        catalog: SensorCatalog,
        kind: ScenarioKind,
    ) -> None:
        """All sensors of a step in the window carry the label (it is the plant)."""
        sim, frame = runs[kind]
        per_step = frame[frame["process_fault"] != "none"].groupby("timestep").size()
        assert (per_step == len(catalog.sensors)).all()
        assert len(per_step) == self._last(sim) - self.INJECTED + 1

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_before_the_injection_it_is_the_normal_run(
        self,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
        simulator: ReactorSimulator,
        params: SimulationParams,
        kind: ScenarioKind,
    ) -> None:
        """Same seed and index: the samples before the start match the normal run."""
        _, frame = runs[kind]
        normal = simulator.generate_run(normal_run_plans(params)[0])
        before = frame["timestep"] < self.INJECTED
        assert np.allclose(
            frame.loc[before, "true_value"].to_numpy(),
            normal.loc[before, "true_value"].to_numpy(),
            rtol=1e-6,
        )
        assert np.allclose(
            frame.loc[before, "value"].to_numpy(),
            normal.loc[before, "value"].to_numpy(),
            rtol=1e-6,
        )

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_the_fault_changes_the_state(
        self,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
        simulator: ReactorSimulator,
        params: SimulationParams,
        kind: ScenarioKind,
    ) -> None:
        """After the start the true values leave those of the normal run."""
        sim, frame = runs[kind]
        normal = simulator.generate_run(normal_run_plans(params)[0])
        after = frame["timestep"].between(self.INJECTED + 4, self._last(sim))
        gap = np.abs(
            frame.loc[after, "true_value"].to_numpy() - normal.loc[after, "true_value"].to_numpy()
        )
        assert gap.max() > 1.0

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_energy_is_conserved_through_the_transient(
        self,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
        model: ReactorModel,
        kind: ScenarioKind,
    ) -> None:
        """Conservation holds on the full state even where the rules are violated."""
        sim, _ = runs[kind]
        report = conservation_report(
            sim.diagnostics, sim.steps, ConstraintLimits.from_plant(model.plant)
        )
        assert report.compliant
        assert report.max_interval_residual < 1e-6

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_no_rule_is_violated_before_the_injection(
        self,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
        model: ReactorModel,
        catalog: SensorCatalog,
        kind: ScenarioKind,
    ) -> None:
        """The window before injected_at is normal operation (true values)."""
        sim, _ = runs[kind]
        wide = wide_frame_from_matrix(sim.true_values, catalog, sim.timestamps)
        violations = evaluate_rules(
            wide.iloc[: self.INJECTED],
            catalog.roles,
            model.table,
            ConstraintLimits.from_plant(model.plant),
        )
        assert not violations.to_numpy().any()

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_after_recovery_the_inputs_are_nominal(
        self,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
        kind: ScenarioKind,
    ) -> None:
        """Past the window the model gets no fault input."""
        sim, _ = runs[kind]
        last = sim.inputs[self._last(sim) + 2]
        assert last.leak_bar == 0.0
        assert last.reactivity_insertion == 0.0
        assert last.pump_demand == 1.0
        assert last.rod_control_enabled

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_frame_rebuilds_canonical_readings(
        self, runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]], kind: ScenarioKind
    ) -> None:
        """The labelled frame validates against the SensorReading contract."""
        _, frame = runs[kind]
        assert len(list(readings_from_frame(frame))) == len(frame)

    @pytest.mark.parametrize("kind", PROCESS_KINDS)
    def test_is_deterministic(
        self,
        runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]],
        simulator: ReactorSimulator,
        kind: ScenarioKind,
    ) -> None:
        """The same plan gives the same frame bit for bit."""
        sim, frame = runs[kind]
        again = simulator.generate_run(sim.plan)
        assert frame.equals(again)

    def test_flow_run_is_sub_stepped(
        self, runs: dict[ScenarioKind, tuple[RunSimulation, pd.DataFrame]]
    ) -> None:
        """The pump run integrates four sub-intervals per sample and still closes energy."""
        sim, _ = runs[FLOW]
        assert sim.max_energy_residual < 1e-6
        assert len(sim.steps) == sim.plan.n_steps - 1

    def test_fault_tiers_are_parsed_once(self, simulator: ReactorSimulator) -> None:
        """The simulator caches the parsed levels."""
        assert simulator.fault_tiers is simulator.fault_tiers
        assert isinstance(simulator.fault_tiers.for_kind(VOID), KindTiers)

    def test_process_plan_rejects_non_process_scenarios(self, simulator: ReactorSimulator) -> None:
        """Only the three process faults have levels."""
        with pytest.raises(KeyError):
            simulator.process_plan(ScenarioKind.NORMAL, FaultLevel.LOW, persistent=True)
