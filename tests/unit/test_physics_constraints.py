"""Unit tests for data.simulation.physics_constraints.

Cubren lo que prometen las reglas: operadores estrictos en el umbral, NaN y
fuera-de-tabla sin evaluar (y contados aparte), roles leidos del catalogo,
cp de tabla y no 4182, y verificacion contra el modelo de F3-M1: el estado
verdadero (equilibrio y rampas normales) no viola ninguna regla y conserva la
energia.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from data.simulation.physics_constraints import (
    REQUIRED_ROLES,
    RULE_NAMES,
    TDD_RULE_NAMES,
    ConstraintLimits,
    RuleStats,
    boiling_violated,
    build_rule_report,
    conservation_report,
    conservation_residuals,
    energy_balance_residual,
    energy_balance_violated,
    evaluate_rules,
    evaluate_rules_detailed,
    finite_difference_rate,
    flow_range_violated,
    in_table_domain,
    mass_balance_residual,
    mass_balance_violated,
    max_dt_dt_violated,
    max_observable_cadence_s,
    min_observable_rate,
    power_flux_violated,
    quasi_steady_mask,
    removed_heat_mw,
    rule_margins,
    temperature_ordering_violated,
)
from data.simulation.plant_params import load_plant_params
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.thermal_model import ModelInputs, ReactorModel
from data.simulation.water_properties import WaterTable
from tests.support.sim_frames import (
    CONFIGS,
    DEFAULT_DT_S,
    Trace,
    build_catalog,
    build_model,
    build_table,
    ramp_loads,
    run_load_profile,
    wide_frame,
)


@pytest.fixture(scope="module")
def model() -> ReactorModel:
    """Return the reference model on the versioned configuration."""
    return build_model()


@pytest.fixture(scope="module")
def table() -> WaterTable:
    """Return the versioned water table."""
    return build_table()


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned sensor catalog."""
    return build_catalog()


@pytest.fixture(scope="module")
def limits() -> ConstraintLimits:
    """Return the limits of the versioned configuration."""
    return ConstraintLimits.from_plant(load_plant_params(CONFIGS / "plant_parameters.yaml"))


@pytest.fixture(scope="module")
def equilibrium(model: ReactorModel, catalog: SensorCatalog) -> pd.DataFrame:
    """Return a noise-free wide frame of 6 samples at nominal equilibrium."""
    trace = run_load_profile(model, [1.0] * 6)
    return wide_frame(trace, model, catalog)


@pytest.fixture(scope="module")
def ramp_trace(model: ReactorModel) -> Trace:
    """Return a 70 -> 100 % ramp (5 %/min at 180 s is 2 steps; this one is 1 %/min)."""
    return run_load_profile(model, ramp_loads(0.7, 1.0, 10, hold_steps=3))


def _limits(base: ConstraintLimits, **changes: Any) -> ConstraintLimits:
    return replace(base, **changes)


def _hit(mask: Any) -> list[bool]:
    return np.atleast_1d(np.asarray(mask)).tolist()


class TestConstraintLimits:
    """ConstraintLimits is derived from PlantParams and validated on its own."""

    def test_from_plant_copies_the_configuration(self, limits: ConstraintLimits) -> None:
        """Every field comes from plant_parameters.yaml, nothing is typed here."""
        plant = load_plant_params(CONFIGS / "plant_parameters.yaml")
        assert limits.max_dt_c_per_s == plant.limits.max_dt_c_per_s
        assert limits.energy_imbalance_fraction == plant.limits.energy_imbalance_fraction
        assert limits.pump_max_kg_s == plant.pump_max_kg_s
        assert limits.thermal_power_mw == plant.thermal_power_mw
        assert limits.n_loops == plant.limits.n_loops
        assert limits.mass_balance_tolerance == plant.limits.mass_balance_tolerance

    def test_channel_ceiling_is_total_over_loops(self, limits: ConstraintLimits) -> None:
        """A channel measures total / n_loops: 19800 / 4 = 4950 kg/s."""
        assert limits.pump_max_per_loop_kg_s == pytest.approx(4950.0)

    def test_is_frozen(self, limits: ConstraintLimits) -> None:
        """The limits are immutable."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            limits.max_dt_c_per_s = 1.0  # type: ignore[misc]

    @pytest.mark.parametrize(
        "name",
        [
            "max_dt_c_per_s",
            "energy_imbalance_fraction",
            "pump_max_kg_s",
            "thermal_power_mw",
            "mass_balance_tolerance",
            "quasi_steady_dt_avg_k_s",
            "power_flux_k",
            "power_flux_threshold_percent",
            "conservation_tolerance",
        ],
    )
    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_rejects_non_positive_or_non_finite(
        self, limits: ConstraintLimits, name: str, bad: float
    ) -> None:
        """A zero, negative, NaN or infinite limit is a configuration error."""
        with pytest.raises(ValueError, match=name):
            _limits(limits, **{name: bad})

    def test_rejects_zero_loops(self, limits: ConstraintLimits) -> None:
        """n_loops divides the flow, so it cannot be 0."""
        with pytest.raises(ValueError, match="n_loops"):
            _limits(limits, n_loops=0)

    def test_catalog_flow_gain_matches_loop_count(
        self, catalog: SensorCatalog, limits: ConstraintLimits
    ) -> None:
        """The flow sensors read total / n_loops: gain 1 / n_loops, or the energy rule drifts."""
        for tag in catalog.roles["flow"]:
            assert catalog.by_tag(tag).gain == pytest.approx(1.0 / limits.n_loops)

    def test_catalog_provides_every_required_role(self, catalog: SensorCatalog) -> None:
        """evaluate_rules needs exactly these roles from configs/sim_sensors.yaml."""
        assert set(REQUIRED_ROLES) <= set(catalog.roles)

    def test_rule_names_put_the_tdd_rules_first(self) -> None:
        """Six rules of TDD 5.4 then the App. B mass law."""
        assert RULE_NAMES[:6] == TDD_RULE_NAMES
        assert RULE_NAMES[6] == "mass_balance"
        assert len(set(RULE_NAMES)) == 7


class TestEnergyBalance:
    """TDD 5.4 rule 1: |Q_gen - Q_removed| / Q_gen > 5 %, with cp from the table."""

    def test_removed_heat_uses_table_cp_not_4182(self, table: WaterTable) -> None:
        """m cp (T_out - T_in) with cp(T_avg, P) from the table."""
        t_out, t_in, p, m = 320.7, 291.0, 155.0, 18000.0
        cp = table.cp(0.5 * (t_out + t_in), p)
        expected = m * cp * (t_out - t_in) * 1e-6
        assert float(removed_heat_mw(m, t_out, t_in, p, table)) == pytest.approx(expected)
        assert cp > 5000.0  # 4182 would understate Q_removed by more than 20 %
        assert expected > 1.2 * m * 4182.0 * (t_out - t_in) * 1e-6

    def test_nominal_point_removes_the_nominal_power(
        self, model: ReactorModel, table: WaterTable
    ) -> None:
        """At nominal equilibrium the table cp closes 3000 MW to the table's own precision."""
        nominal = model.nominal
        q = removed_heat_mw(18000.0, nominal.t_out_c, 291.0, 155.0, table)
        assert float(q) == pytest.approx(3000.0, rel=1e-9)

    def test_removed_heat_is_nan_outside_the_table(self, table: WaterTable) -> None:
        """The table never extrapolates: a point off the grid is not evaluated."""
        q = removed_heat_mw([18000.0, 18000.0, 18000.0], [320.0, 400.0, 320.0],
                            [290.0, 390.0, 290.0], [155.0, 155.0, 30.0], table)
        assert np.isfinite(q[0])
        assert np.isnan(q[1])  # T_avg = 395 C is above 360 C
        assert np.isnan(q[2])  # 30 bar is below 60 bar

    def test_residual_arithmetic(self) -> None:
        """|100 - 90| / 100 = 0.10."""
        assert float(energy_balance_residual(100.0, 90.0)) == pytest.approx(0.10)

    def test_residual_is_nan_without_generation(self) -> None:
        """Q_gen <= 0 or NaN has no relative imbalance."""
        res = energy_balance_residual([0.0, -5.0, np.nan, 100.0], [10.0, 10.0, 10.0, np.nan])
        assert np.isnan(res).all()

    def test_equal_to_the_limit_is_not_a_violation(self) -> None:
        """Strict operator: exactly 5 % passes, a hair above fails."""
        assert _hit(energy_balance_violated(100.0, 95.0, 0.05)) == [False]
        assert _hit(energy_balance_violated(100.0, 94.9, 0.05)) == [True]
        assert _hit(energy_balance_violated(100.0, 105.1, 0.05)) == [True]

    def test_nan_is_not_evaluated(self) -> None:
        """NaN never reports a violation."""
        assert _hit(energy_balance_violated([np.nan, 100.0], [90.0, np.nan], 0.05)) == [
            False,
            False,
        ]


class TestTemperatureOrdering:
    """TDD 5.4 rule 2: T_out < T_in."""

    def test_equal_temperatures_are_not_a_violation(self) -> None:
        """T_out == T_in is not strictly less."""
        assert _hit(temperature_ordering_violated(300.0, 300.0)) == [False]

    def test_outlet_below_inlet_is_a_violation(self) -> None:
        """T_out < T_in by any margin."""
        assert _hit(temperature_ordering_violated(299.999, 300.0)) == [True]

    def test_nan_is_not_evaluated(self) -> None:
        """NaN on either side is not a violation."""
        assert _hit(temperature_ordering_violated([np.nan, 300.0], [300.0, np.nan])) == [
            False,
            False,
        ]


class TestMaxTemperatureRate:
    """TDD 5.4 rule 3 on the magnitude of the rate, and its finite-difference input."""

    def test_limit_is_strict_and_symmetric(self) -> None:
        """10 C/s passes; 10.000001 fails in either direction."""
        rates = np.array([10.0, 10.000001, -10.000001, -10.0, 0.0, np.nan])
        assert _hit(max_dt_dt_violated(rates, 10.0)) == [False, True, True, False, False, False]

    def test_finite_difference_of_a_series(self) -> None:
        """First sample has no predecessor; the rest are dT / dt."""
        rate = finite_difference_rate([300.0, 303.0, 303.0, 297.0], [np.nan, 180.0, 180.0, 60.0])
        assert np.isnan(rate[0])
        assert rate[1:].tolist() == pytest.approx([3.0 / 180.0, 0.0, -0.1])

    def test_finite_difference_of_channels(self) -> None:
        """A (n, k) block is differenced column by column."""
        values = np.array([[1.0, 10.0], [2.0, 30.0], [4.0, 30.0]])
        rate = finite_difference_rate(values, [np.nan, 1.0, 2.0])
        assert rate.shape == (3, 2)
        assert np.isnan(rate[0]).all()
        assert rate[1:].tolist() == [[1.0, 20.0], [1.0, 0.0]]

    def test_non_positive_elapsed_time_is_not_evaluated(self) -> None:
        """A repeated or reversed timestamp yields NaN, not inf."""
        rate = finite_difference_rate([1.0, 2.0, 3.0], [np.nan, 0.0, -1.0])
        assert np.isnan(rate).all()

    def test_single_sample_has_no_rate(self) -> None:
        """One row cannot be differenced."""
        assert np.isnan(finite_difference_rate([300.0], [np.nan])).all()


class TestFlowRange:
    """TDD 5.4 rule 4: flow < 0 or flow > pump_max."""

    def test_zero_is_not_a_violation_of_this_rule(self) -> None:
        """Strict operator: a stopped pump is caught by the energy rule, not this one."""
        assert _hit(flow_range_violated(0.0, 4950.0)) == [False]

    def test_negative_flow_is_a_violation(self) -> None:
        """Reverse flow is impossible."""
        assert _hit(flow_range_violated([-0.001, -100.0], 4950.0)) == [True, True]

    def test_ceiling_is_strict(self) -> None:
        """Exactly pump_max passes; above it fails."""
        assert _hit(flow_range_violated([4950.0, 4950.001], 4950.0)) == [False, True]

    def test_nan_is_not_evaluated(self) -> None:
        """NaN is neither negative nor above the ceiling."""
        assert _hit(flow_range_violated(np.nan, 4950.0)) == [False]


class TestPowerFlux:
    """TDD 5.4 rule 5: |P_th - k flux| > threshold."""

    def test_threshold_is_strict(self) -> None:
        """A gap equal to the threshold passes."""
        assert _hit(power_flux_violated([104.5, 104.6, 95.4], 100.0, 1.0, 4.5)) == [
            False,
            True,
            True,
        ]

    def test_k_scales_the_flux(self) -> None:
        """With k = 2 the expected thermal power is twice the flux."""
        assert _hit(power_flux_violated(200.0, 100.0, 2.0, 1.0)) == [False]
        assert _hit(power_flux_violated(100.0, 100.0, 2.0, 1.0)) == [True]

    def test_nan_is_not_evaluated(self) -> None:
        """NaN on either side is not a violation."""
        assert _hit(power_flux_violated([np.nan, 100.0], [100.0, np.nan], 1.0, 4.5)) == [
            False,
            False,
        ]


class TestBoiling:
    """TDD 5.4 rule 6: P < P_sat(T), written as T_out > T_sat(P)."""

    def test_saturation_is_the_edge(self, table: WaterTable) -> None:
        """T_sat - eps and T_sat itself pass; T_sat + eps is boiling."""
        t_sat = table.t_sat_c(155.0)
        eps = 1e-9
        t = np.array([t_sat - eps, t_sat, t_sat + eps])
        assert _hit(boiling_violated(t, 155.0, table)) == [False, False, True]

    def test_nominal_point_is_subcooled(self, table: WaterTable) -> None:
        """T_out 320.7 C is 24 K under saturation at 155 bar."""
        assert _hit(boiling_violated(320.69, 155.0, table)) == [False]

    def test_depressurisation_triggers_it(self, table: WaterTable) -> None:
        """At 100 bar T_sat is ~311 C: the nominal outlet would boil."""
        assert _hit(boiling_violated(320.69, 100.0, table)) == [True]

    def test_equivalent_to_p_below_p_sat(self, table: WaterTable) -> None:
        """T > T_sat(P) and P < P_sat(T) agree on a grid of states."""
        t = np.linspace(280.0, 345.0, 30)
        p = np.linspace(100.0, 160.0, 30)
        tt, pp = np.meshgrid(t, p)
        by_temperature = boiling_violated(tt, pp, table)
        by_pressure = pp < table.p_sat_bar(np.clip(tt, *table.t_sat_c_grid[[0, -1]]))
        inside = (tt >= table.t_sat_c_grid[0]) & (tt <= table.t_sat_c_grid[-1])
        assert np.array_equal(by_temperature[inside], by_pressure[inside])

    def test_pressure_outside_the_table_is_not_evaluated(self, table: WaterTable) -> None:
        """The table never extrapolates; NaN is not evaluated either."""
        assert _hit(boiling_violated([400.0, 400.0], [30.0, np.nan], table)) == [False, False]

    def test_vectorised_shape(self, table: WaterTable) -> None:
        """Output follows the broadcast shape of the inputs."""
        out = boiling_violated(np.full((3, 4), 320.0), np.full((3, 4), 155.0), table)
        assert out.shape == (3, 4)
        assert out.dtype == np.bool_


class TestMassBalance:
    """App. B mass law: |m_hl - m_cl| / m > tol, only in quasi-steady state."""

    def test_residual_arithmetic(self) -> None:
        """|4600 - 4400| / 4500 = 0.0444."""
        assert float(mass_balance_residual(4600.0, 4400.0)) == pytest.approx(200.0 / 4500.0)

    def test_residual_is_nan_without_flow(self) -> None:
        """A stopped loop has no relative mismatch."""
        assert np.isnan(mass_balance_residual([0.0, -1.0, np.nan], [0.0, -1.0, 1.0])).all()

    def test_transient_suppresses_the_rule(self) -> None:
        """The same mismatch is a violation at steady state and not in a transient."""
        hot, cold, tol = 4800.0, 4400.0, 0.05
        assert _hit(mass_balance_violated(hot, cold, True, tol)) == [True]
        assert _hit(mass_balance_violated(hot, cold, False, tol)) == [False]

    def test_tolerance_is_strict(self) -> None:
        """A mismatch equal to the tolerance passes."""
        assert _hit(mass_balance_violated(1050.0, 950.0, True, 0.1)) == [False]
        assert _hit(mass_balance_violated(1051.0, 949.0, True, 0.1)) == [True]

    def test_quasi_steady_mask_is_strict_and_symmetric(self) -> None:
        """|dT_avg/dt| < threshold; equal is not quasi-steady; NaN is not either."""
        rates = np.array([0.0, 0.0199, -0.0199, 0.02, -0.02, np.nan])
        assert _hit(quasi_steady_mask(rates, 0.02)) == [True, True, True, False, False, False]


class TestObservabilityByCadence:
    """C5: the dT/dt rule cannot fire between samples that are far apart."""

    def test_180_seconds_is_1800_kelvin(self) -> None:
        """10 C/s x 180 s."""
        assert min_observable_rate(180.0) == 1800.0

    def test_scales_with_the_cadence_and_the_limit(self) -> None:
        """The step is linear in both."""
        assert min_observable_rate(1.0) == 10.0
        assert min_observable_rate(60.0, 50.0) == 3000.0

    def test_exceeds_the_water_table_span_at_180_seconds(self, table: WaterTable) -> None:
        """No liquid-water state differs from another by 1800 K."""
        t_lo, t_hi = table.t_range_c
        assert min_observable_rate(180.0) > t_hi - t_lo

    def test_cadence_that_can_ever_fire(self) -> None:
        """A 160 K span is crossed at 10 C/s in 16 s."""
        assert max_observable_cadence_s(160.0) == 16.0

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_non_positive_arguments(self, bad: float) -> None:
        """Cadence, limit and span must be strictly positive."""
        with pytest.raises(ValueError, match="dt_s"):
            min_observable_rate(bad)
        with pytest.raises(ValueError, match="max_dt_c_per_s"):
            min_observable_rate(180.0, bad)
        with pytest.raises(ValueError, match="temperature_span_k"):
            max_observable_cadence_s(bad)
        with pytest.raises(ValueError, match="max_dt_c_per_s"):
            max_observable_cadence_s(160.0, bad)


class TestDomain:
    """in_table_domain guards every table lookup."""

    def test_grid_edges_are_inside(self, table: WaterTable) -> None:
        """Inclusive at the edges, outside beyond them, NaN outside."""
        t_lo, t_hi = table.t_range_c
        p_lo, p_hi = table.p_range_bar
        t = np.array([t_lo, t_hi, t_hi + 0.1, 300.0, np.nan])
        p = np.array([p_lo, p_hi, 100.0, p_lo - 0.1, 100.0])
        assert in_table_domain(table, t, p).tolist() == [True, True, False, False, False]


class TestEvaluateRulesOnTheTrueState:
    """The F3-M1 model satisfies every rule with no noise (D7)."""

    def test_equilibrium_violates_nothing(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Nominal equilibrium: seven rules, zero violations, everything evaluable."""
        out = evaluate_rules_detailed(equilibrium, catalog.roles, table, limits)
        assert not out.violations.to_numpy().any()
        # The first row has no predecessor, so dT/dt and the mass gate cannot apply.
        assert out.evaluable.drop(index=equilibrium.index[0]).to_numpy().all()

    def test_load_ramp_violates_nothing(
        self,
        model: ReactorModel,
        ramp_trace: Trace,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A 1 %/min ramp from 70 to 100 %: no rule fires on the true state."""
        frame = wide_frame(ramp_trace, model, catalog)
        assert not evaluate_rules(frame, catalog.roles, table, limits).to_numpy().any()

    def test_standard_5_percent_per_minute_ramp_violates_nothing(
        self,
        model: ReactorModel,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """The normal load-following ramp is inside every threshold (the calibration case)."""
        trace = run_load_profile(model, ramp_loads(0.7, 1.0, 2, hold_steps=4))
        frame = wide_frame(trace, model, catalog)
        assert not evaluate_rules(frame, catalog.roles, table, limits).to_numpy().any()

    def test_output_is_a_bool_frame_with_the_rule_columns(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Vectorised: one bool column per rule, indexed like the input."""
        out = evaluate_rules(equilibrium, catalog.roles, table, limits)
        assert list(out.columns) == list(RULE_NAMES)
        assert out.index.equals(equilibrium.index)
        assert (out.dtypes == np.dtype("bool")).all()

    def test_does_not_modify_its_input(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """The wide frame is read-only for the rules."""
        before = equilibrium.copy()
        evaluate_rules(equilibrium, catalog.roles, table, limits)
        pd.testing.assert_frame_equal(equilibrium, before)

    def test_noise_at_the_initial_sigma_violates_nothing(
        self,
        model: ReactorModel,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Contract F3-M3 recalibrates against: initial catalog sigma, no violations."""
        trace = run_load_profile(model, [1.0] * 400)
        rng = np.random.default_rng(np.random.SeedSequence(20261011))
        frame = wide_frame(trace, model, catalog, rng=rng)
        out = evaluate_rules(frame, catalog.roles, table, limits)
        assert not out.to_numpy().any()


class TestEvaluateRulesOnInjectedFaults:
    """Each rule fires on the frame manipulation it is meant for."""

    def _eval(
        self,
        frame: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> pd.DataFrame:
        return evaluate_rules(frame, catalog.roles, table, limits)

    def test_swapped_legs_break_the_ordering(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Hot-leg reading colder than the cold leg."""
        frame = equilibrium.copy()
        frame[["TC-HL-1", "TC-HL-2"]] = equilibrium[["TC-CL-1", "TC-CL-2"]].to_numpy()
        frame[["TC-CL-1", "TC-CL-2"]] = equilibrium[["TC-HL-1", "TC-HL-2"]].to_numpy()
        assert self._eval(frame, catalog, table, limits)["temperature_ordering"].all()

    def test_hot_outlet_boils(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """T_out 30 K up crosses T_sat(155 bar) = 344.8 C."""
        frame = equilibrium.copy()
        frame[["TC-HL-1", "TC-HL-2"]] += 30.0
        assert self._eval(frame, catalog, table, limits)["boiling"].all()

    def test_negative_flow_on_one_channel(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Channel-wise bound fires; the median of 4 channels keeps the energy rule quiet."""
        frame = equilibrium.copy()
        frame["FT-CL-1"] = -10.0
        out = self._eval(frame, catalog, table, limits)
        assert out["flow_range"].all()
        assert not out["energy_balance"].any()

    def test_flow_above_the_pump_ceiling(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A channel above 4950 kg/s exceeds 110 % of nominal."""
        frame = equilibrium.copy()
        frame["FT-HL-2"] = 5000.0
        assert self._eval(frame, catalog, table, limits)["flow_range"].all()

    def test_temperature_jump_between_samples(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """+2000 K in one 180 s sample is 11 C/s up and 11 C/s down; row 0 has no rate."""
        frame = equilibrium.copy()
        frame.iloc[2, frame.columns.get_loc("TC-CL-1")] += 2000.0
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert out.violations["max_dt_dt"].tolist() == [False, False, True, True, False, False]
        assert not out.evaluable["max_dt_dt"].iloc[0]

    def test_flux_loss_breaks_energy_and_power_flux(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """All three flux channels read half: Q_gen = 1500 MW against 3000 MW removed."""
        frame = equilibrium.copy()
        frame[["NI-FLUX-01", "NI-FLUX-02", "NI-FLUX-03"]] *= 0.5
        out = self._eval(frame, catalog, table, limits)
        assert out["energy_balance"].all()
        assert out["power_flux"].all()

    def test_thermal_power_channel_off(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """The calorimetric channel alone disagreeing with the flux fires rule 5 only."""
        frame = equilibrium.copy()
        frame["PWR-TH-01"] = 50.0
        out = self._eval(frame, catalog, table, limits)
        assert out["power_flux"].all()
        assert not out["energy_balance"].any()

    def test_hot_leg_flow_mismatch(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Both hot-leg channels 20 % above the cold leg at steady state."""
        frame = equilibrium.copy()
        frame[["FT-HL-1", "FT-HL-2"]] *= 1.2
        assert self._eval(frame, catalog, table, limits)["mass_balance"].iloc[1:].all()

    def test_mass_rule_is_off_in_a_transient(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A 20 K jump in T_avg between samples (0.11 K/s) disables the mass rule there."""
        frame = equilibrium.copy()
        frame[["TC-HL-1", "TC-HL-2", "TC-CL-1", "TC-CL-2"]] += np.array(
            [0.0, 0.0, 20.0, 20.0, 20.0, 20.0]
        )[:, None]
        frame[["FT-HL-1", "FT-HL-2"]] *= 1.2
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert not out.evaluable["mass_balance"].iloc[2]
        assert not out.violations["mass_balance"].iloc[2]
        assert out.violations["mass_balance"].iloc[3]

    def test_a_single_flux_channel_is_voted_out(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Relation rules read the median of the role: one dead channel of three is masked."""
        frame = equilibrium.copy()
        frame["NI-FLUX-01"] = 0.0
        out = self._eval(frame, catalog, table, limits)
        assert not out[["energy_balance", "power_flux"]].to_numpy().any()


class TestNotEvaluated:
    """NaN, text, out-of-table values and bad clocks are counted apart, never violated."""

    def test_nan_pressure_blocks_boiling_and_energy(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """No valid P channel: the rules that need P are not evaluated on that row."""
        frame = equilibrium.copy()
        frame.iloc[1, frame.columns.get_indexer(["PT-PRZ-01", "PT-PRZ-02"])] = np.nan
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert not out.evaluable["boiling"].iloc[1]
        assert not out.evaluable["energy_balance"].iloc[1]
        assert out.evaluable["temperature_ordering"].iloc[1]
        assert not out.violations.iloc[1].any()

    def test_one_nan_channel_still_evaluates(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """The median skips a NaN channel: the other one carries the role."""
        frame = equilibrium.copy()
        frame["TC-HL-1"] = np.nan
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert out.evaluable["temperature_ordering"].all()
        assert not out.violations.to_numpy().any()

    def test_none_and_text_cells_are_not_evaluated(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """An object column with None and a string behaves as NaN."""
        frame = equilibrium.astype(object)
        frame.iloc[0, frame.columns.get_indexer(["TC-HL-1", "TC-HL-2"])] = [None, "n/a"]
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert not out.evaluable["temperature_ordering"].iloc[0]
        assert out.evaluable["temperature_ordering"].iloc[1:].all()

    def test_pressure_below_the_table_is_not_evaluated(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """40 bar is under the 60 bar floor of the grid: no extrapolation, no verdict."""
        frame = equilibrium.copy()
        frame[["PT-PRZ-01", "PT-PRZ-02"]] = 40.0
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert not out.evaluable[["boiling", "energy_balance"]].to_numpy().any()
        assert not out.violations[["boiling", "energy_balance"]].to_numpy().any()

    def test_repeated_timestamp_has_no_rate(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A duplicated timestamp makes dT/dt undefined on that row, not infinite."""
        frame = equilibrium.copy()
        frame.index = pd.DatetimeIndex(
            [frame.index[0], frame.index[1], frame.index[1], *frame.index[3:]]
        )
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert not out.evaluable["max_dt_dt"].iloc[2]

    def test_timezone_aware_index_is_accepted(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Elapsed time does not depend on the index being naive."""
        frame = equilibrium.copy()
        frame.index = frame.index.tz_localize("UTC")
        out = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        assert out.evaluable["max_dt_dt"].iloc[1:].all()

    def test_frame_needs_a_datetime_index(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Without a clock there is no dT/dt."""
        with pytest.raises(TypeError, match="DatetimeIndex"):
            evaluate_rules(equilibrium.reset_index(drop=True), catalog.roles, table, limits)

    def test_missing_role_is_named(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A catalogue without a required role fails loudly."""
        roles = {k: v for k, v in catalog.roles.items() if k != "P"}
        with pytest.raises(KeyError, match="role 'P'"):
            evaluate_rules(equilibrium, roles, table, limits)

    def test_empty_role_is_named(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A role with no tags is as bad as a missing one."""
        roles = {**catalog.roles, "flux": ()}
        with pytest.raises(KeyError, match="role 'flux'"):
            evaluate_rules(equilibrium, roles, table, limits)

    def test_missing_column_is_named(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A role tag absent from the frame is reported with its tag."""
        with pytest.raises(KeyError, match="PT-PRZ-02"):
            evaluate_rules(equilibrium.drop(columns=["PT-PRZ-02"]), catalog.roles, table, limits)


class TestRuleReport:
    """Per-rule, per-scenario violation rates with the not-evaluated rows apart."""

    def test_counts_rates_and_not_evaluated(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A faulted scenario shows its rate; the clean one shows zero."""
        faulty = equilibrium.copy()
        faulty["FT-CL-1"] = -10.0
        faulty.iloc[0, faulty.columns.get_indexer(["PT-PRZ-01", "PT-PRZ-02"])] = np.nan
        report = build_rule_report(
            {
                "normal": evaluate_rules_detailed(equilibrium, catalog.roles, table, limits),
                "negative_flow": evaluate_rules_detailed(faulty, catalog.roles, table, limits),
            }
        )
        assert report.rate("normal", "flow_range") == 0.0
        assert report.rate("negative_flow", "flow_range") == 1.0
        stats = report.stats("negative_flow", "boiling")
        assert (stats.n_rows, stats.n_evaluable, stats.n_not_evaluated) == (6, 5, 1)
        assert stats.n_violations == 0

    def test_rate_is_nan_when_nothing_was_evaluable(self) -> None:
        """0 / 0 is undefined, not 0."""
        stats = RuleStats(n_rows=3, n_evaluable=0, n_violations=0)
        assert math.isnan(stats.rate)
        assert stats.n_not_evaluated == 3

    def test_to_dict_is_json_ready(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Sorted scenarios, every rule present, plain python numbers."""
        import json

        out = build_rule_report(
            {"b": evaluate_rules_detailed(equilibrium, catalog.roles, table, limits),
             "a": evaluate_rules_detailed(equilibrium.iloc[:1], catalog.roles, table, limits)}
        ).to_dict()
        assert list(out) == ["a", "b"]
        assert set(out["a"]) == set(RULE_NAMES)
        assert out["a"]["max_dt_dt"]["rate"] is None  # one row: no rate, nothing evaluable
        assert out["b"]["energy_balance"]["rate"] == 0.0
        json.dumps(out)

    def test_unknown_scenario_and_rule(
        self, equilibrium: pd.DataFrame, catalog: SensorCatalog, table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Typos are errors, not zeros."""
        report = build_rule_report(
            {"normal": evaluate_rules_detailed(equilibrium, catalog.roles, table, limits)}
        )
        with pytest.raises(KeyError, match="scenario 'x'"):
            report.rate("x", "boiling")
        with pytest.raises(KeyError, match="rule 'y'"):
            report.rate("normal", "y")


class TestConservation:
    """D7: energy conservation on the full state, with the 5.4 imbalance reported apart."""

    def test_ramp_trace_closes_the_energy_budget(
        self, ramp_trace: Trace, limits: ConstraintLimits
    ) -> None:
        """Instantaneous residual ~ machine zero, interval residual far below tolerance."""
        report = conservation_report(ramp_trace.diagnostics, ramp_trace.steps, limits)
        assert report.compliant
        assert report.n_instants == len(ramp_trace.diagnostics)
        assert report.n_intervals == len(ramp_trace.steps)
        assert report.max_instant_residual < 1e-12
        assert report.max_interval_residual < 1e-9
        assert report.n_primary_imbalance_violations == 0
        assert report.max_primary_imbalance < limits.energy_imbalance_fraction

    def test_residuals_are_relative_to_nominal_power(
        self, ramp_trace: Trace, limits: ConstraintLimits
    ) -> None:
        """An injected 30 MW mismatch is 1 % of 3000 MW."""
        broken = replace(ramp_trace.diagnostics[0], conservation_residual_mw=-30.0)
        res = conservation_residuals([broken], limits.thermal_power_mw)
        assert res.tolist() == pytest.approx([0.01])

    def test_failures_are_counted_in_instants_and_intervals(
        self, ramp_trace: Trace, limits: ConstraintLimits
    ) -> None:
        """A tolerance below the solver closure turns the trace non-compliant."""
        tight = replace(limits, conservation_tolerance=1e-30)
        report = conservation_report(ramp_trace.diagnostics, ramp_trace.steps, tight)
        assert not report.compliant
        assert report.n_failures >= len(ramp_trace.steps)

    def test_fast_transient_conserves_energy_but_breaks_rule_1(
        self, model: ReactorModel, limits: ConstraintLimits
    ) -> None:
        """Pump trip: the 5.4 imbalance is violated on the true state, conservation holds."""
        y = model.steady_state(1.0)
        inputs = ModelInputs(pump_demand=0.0)
        diagnostics, steps = [model.diagnostics(y, inputs)], []
        for _ in range(4):
            step = model.advance(y, inputs, DEFAULT_DT_S)
            y = step.y
            steps.append(step)
            diagnostics.append(model.diagnostics(y, inputs))
        report = conservation_report(diagnostics, steps, limits)
        assert report.compliant
        assert report.n_primary_imbalance_violations > 0
        assert report.max_primary_imbalance > limits.energy_imbalance_fraction

    def test_only_steps_or_only_diagnostics(
        self, ramp_trace: Trace, limits: ConstraintLimits
    ) -> None:
        """Either half of the trace is enough; the other counts as zero."""
        only_steps = conservation_report([], ramp_trace.steps, limits)
        assert only_steps.n_instants == 0
        assert only_steps.max_instant_residual == 0.0
        assert only_steps.max_primary_imbalance == 0.0
        only_diag = conservation_report(ramp_trace.diagnostics, [], limits)
        assert only_diag.n_intervals == 0
        assert only_diag.max_interval_residual == 0.0

    def test_empty_trace_is_an_error(self, limits: ConstraintLimits) -> None:
        """Nothing to certify is not 'compliant'."""
        with pytest.raises(ValueError, match="no diagnostics and no steps"):
            conservation_report([], [], limits)

    def test_power_must_be_positive(self, ramp_trace: Trace) -> None:
        """P0 normalises the residual."""
        with pytest.raises(ValueError, match="thermal_power_mw"):
            conservation_residuals(ramp_trace.diagnostics, 0.0)


class TestRuleMargins:
    """rule_margins is the signed distance to the threshold each rule compares (F3-M3)."""

    def _faulty(self, equilibrium: pd.DataFrame, catalog: SensorCatalog) -> pd.DataFrame:
        """Return a frame whose rows 1-5 each break a different rule."""
        bad = equilibrium.copy()
        roles = {k: list(v) for k, v in catalog.roles.items()}
        bad.loc[bad.index[1], roles["T_out"]] = 400.0  # boiling (and a huge energy residual)
        bad.loc[bad.index[2], roles["flow"]] = -5.0  # negative flow
        bad.loc[bad.index[3], roles["flow"][0]] = 6000.0  # above the per-channel ceiling
        bad.loc[bad.index[4], roles["T_in"]] = 340.0  # T_out < T_in
        bad.loc[bad.index[5], roles["P_th"]] = 50.0  # power/flux gap
        return bad

    def test_equilibrium_is_inside_every_rule(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """The true state has positive margins; the first row has no rate (NaN)."""
        margins = rule_margins(equilibrium, catalog.roles, table, limits)
        assert list(margins.columns) == list(RULE_NAMES)
        assert margins.index.equals(equilibrium.index)
        assert np.isnan(margins["max_dt_dt"].iloc[0])
        assert np.isnan(margins["mass_balance"].iloc[0])
        assert (margins.iloc[1:] > 0.0).all().all()

    def test_negative_margin_is_exactly_a_violation(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Where a rule is evaluable, margin < 0 coincides with its violation."""
        frame = self._faulty(equilibrium, catalog)
        evaluation = evaluate_rules_detailed(frame, catalog.roles, table, limits)
        margins = rule_margins(frame, catalog.roles, table, limits)
        for rule in RULE_NAMES:
            evaluable = evaluation.evaluable[rule].to_numpy()
            negative = (margins[rule].to_numpy() < 0.0) & evaluable
            assert np.array_equal(negative, evaluation.violations[rule].to_numpy()), rule
        assert evaluation.violations.to_numpy().any(axis=0).sum() >= 4

    def test_units_per_rule(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Margins are in the unit of each rule (K, kg/s, pp, fraction)."""
        roles = {k: list(v) for k, v in catalog.roles.items()}
        row = equilibrium.iloc[[2]]
        margins = rule_margins(equilibrium, roles, table, limits).iloc[2]
        t_out = float(row[roles["T_out"]].median(axis=1).iloc[0])
        t_in = float(row[roles["T_in"]].median(axis=1).iloc[0])
        pressure = float(row[roles["P"]].median(axis=1).iloc[0])
        flow = row[roles["flow"]].to_numpy()
        assert margins["temperature_ordering"] == pytest.approx(t_out - t_in)
        assert margins["boiling"] == pytest.approx(float(table.t_sat_c(pressure)) - t_out)
        expected_flow = np.minimum(flow, limits.pump_max_per_loop_kg_s - flow).min()
        assert margins["flow_range"] == pytest.approx(expected_flow)
        gap = abs(
            float(row[roles["P_th"]].iloc[0, 0])
            - limits.power_flux_k * float(row[roles["flux"]].median(axis=1).iloc[0])
        )
        assert margins["power_flux"] == pytest.approx(limits.power_flux_threshold_percent - gap)
        energy = limits.energy_imbalance_fraction
        assert margins["energy_balance"] == pytest.approx(energy, abs=1e-3)

    def test_not_evaluable_rows_are_nan(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """NaN readings and a pressure outside the table leave the margin undefined."""
        frame = equilibrium.copy()
        frame.loc[frame.index[2], list(catalog.roles["flow"])] = np.nan
        frame.loc[frame.index[3], list(catalog.roles["P"])] = 10.0  # below the table (60 bar)
        margins = rule_margins(frame, catalog.roles, table, limits)
        assert np.isnan(margins["flow_range"].iloc[2])
        assert np.isnan(margins["energy_balance"].iloc[2])
        assert np.isnan(margins["boiling"].iloc[3])

    def test_requires_a_datetime_index(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """Like evaluate_rules, margins need a DatetimeIndex."""
        with pytest.raises(TypeError, match="DatetimeIndex"):
            rule_margins(equilibrium.reset_index(drop=True), catalog.roles, table, limits)

    def test_requires_the_roles(
        self,
        equilibrium: pd.DataFrame,
        catalog: SensorCatalog,
        table: WaterTable,
        limits: ConstraintLimits,
    ) -> None:
        """A missing role is a KeyError, as in evaluate_rules."""
        roles = {k: v for k, v in catalog.roles.items() if k != "flux"}
        with pytest.raises(KeyError, match="flux"):
            rule_margins(equilibrium, roles, table, limits)
