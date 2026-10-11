"""Unit tests for data.simulation.thermal_model.

Cubren lo que el modelo promete: equilibrio nominal DERIVADO (no tecleado),
conservacion de energia por construccion, signos de realimentacion, acuerdo
entre integradores, determinismo y rechazo explicito de lo que sale de la
tabla de agua.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from data.simulation import thermal_model
from data.simulation.plant_params import (
    NeutronicsCoefficients,
    load_neutronics,
    load_plant_params,
)
from data.simulation.thermal_model import (
    N_STATES,
    OBSERVABLE_NAMES,
    STATE_NAMES,
    ModelInputs,
    ReactorModel,
    StateIndex,
)
from data.simulation.water_properties import load_water_table

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _REPO_ROOT / "configs"
DT = 180.0


def _build(**kwargs: Any) -> ReactorModel:
    return ReactorModel(
        load_plant_params(_CONFIGS / "plant_parameters.yaml"),
        load_neutronics(_CONFIGS / "neutronics_coefficients.yaml"),
        load_water_table(_CONFIGS / "water_table.yaml"),
        **kwargs,
    )


@pytest.fixture(scope="module")
def model() -> ReactorModel:
    """Return the model on the versioned configuration."""
    return _build()


@pytest.fixture(scope="module")
def nominal(model: ReactorModel) -> np.ndarray:
    """Return the nominal equilibrium state."""
    return model.steady_state()


def _rel_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0)))


def _run(model: ReactorModel, y: np.ndarray, inputs: ModelInputs, steps: int) -> np.ndarray:
    for _ in range(steps):
        y = model.advance(y, inputs, DT).y
    return y


class TestStateLayout:
    """The state vector is addressed by name, never by position."""

    def test_sixteen_states(self) -> None:
        """6 precursors + n + 3 temperatures + P + void + flow + PI (2) + calorimetric."""
        assert N_STATES == 16
        assert len(STATE_NAMES) == N_STATES

    def test_names_follow_the_enum(self) -> None:
        """STATE_NAMES is the lower-cased enum in order."""
        assert STATE_NAMES[StateIndex.T_CORE] == "t_core"
        assert STATE_NAMES[0] == "n"


class TestNominalPoint:
    """The nominal equilibrium is derived from the plant parameters."""

    def test_outlet_closes_the_power_balance(self, model: ReactorModel) -> None:
        """P0 = m cp(T_avg0, P) (T_out0 - T_in0) with the table cp."""
        plant, nom = model.plant, model.nominal
        cp = model.table.cp(nom.t_avg_c, plant.nominal_pressure_bar)
        removed = plant.nominal_flow_kg_s * cp * (nom.t_out_c - plant.t_in_nominal_c) * 1e-6
        assert removed == pytest.approx(plant.thermal_power_mw, rel=1e-12)

    def test_average_is_the_midpoint(self, model: ReactorModel) -> None:
        """T_avg0 = (T_in0 + T_out0) / 2."""
        nom = model.nominal
        assert nom.t_avg_c == pytest.approx(0.5 * (model.plant.t_in_nominal_c + nom.t_out_c))

    def test_conductances_are_derived_not_typed(self, model: ReactorModel) -> None:
        """hA and UA make the nominal point an exact equilibrium."""
        plant, nom = model.plant, model.nominal
        assert nom.ha_mw_k * (plant.t_fuel_nominal_c - nom.t_avg_c) == pytest.approx(
            plant.thermal_power_mw, rel=1e-12
        )
        assert nom.ua_mw_k * (nom.t_avg_c - plant.t_secondary_nominal_c) == pytest.approx(
            plant.thermal_power_mw, rel=1e-12
        )

    def test_outlet_is_subcooled_with_margin(self, model: ReactorModel) -> None:
        """T_sat - T_out0 is about 24 K at 15,5 MPa."""
        assert 20.0 < model.nominal.t_sat_c - model.nominal.t_out_c < 30.0

    def test_iapws_scale_of_the_balance(self, model: ReactorModel) -> None:
        """With IAPWS cp the outlet is ~320,7 C (cp = 4182 would give ~ 331 C)."""
        assert model.nominal.t_out_c == pytest.approx(320.7, abs=0.2)

    def test_secondary_pressure_comes_from_the_table(self, model: ReactorModel) -> None:
        """280 C saturation is ~64 bar."""
        assert model.nominal.p_secondary_bar == pytest.approx(64.2, abs=0.2)

    def test_boiling_nominal_outlet_is_rejected(self) -> None:
        """A cold leg so hot that the outlet would boil is not a PWR."""
        plant = dataclasses.replace(
            load_plant_params(_CONFIGS / "plant_parameters.yaml"), t_in_nominal_c=330.0
        )
        with pytest.raises(ValueError, match="not subcooled"):
            ReactorModel(
                plant,
                load_neutronics(_CONFIGS / "neutronics_coefficients.yaml"),
                load_water_table(_CONFIGS / "water_table.yaml"),
            )

    def test_unknown_integrator_is_rejected(self) -> None:
        """Only LSODA, Radau and BDF are accepted."""
        with pytest.raises(ValueError, match="method"):
            _build(method="RK45")

    def test_six_precursor_groups_are_required(self) -> None:
        """The kinetics assume exactly 6 delayed groups."""
        real = load_neutronics(_CONFIGS / "neutronics_coefficients.yaml")
        two = NeutronicsCoefficients(
            provenance="X",
            beta=real.beta[:2],
            decay_constants_per_s=real.decay_constants_per_s[:2],
            generation_time_s=real.generation_time_s,
            alpha_fuel_per_k=real.alpha_fuel_per_k,
            alpha_moderator_per_k=real.alpha_moderator_per_k,
            alpha_void_per_fraction=real.alpha_void_per_fraction,
            rod_worth_per_percent=real.rod_worth_per_percent,
            rod_position_nominal_percent=real.rod_position_nominal_percent,
        )
        with pytest.raises(ValueError, match="6 precursor"):
            ReactorModel(
                load_plant_params(_CONFIGS / "plant_parameters.yaml"),
                two,
                load_water_table(_CONFIGS / "water_table.yaml"),
            )


class TestSteadyState:
    """steady_state is an exact fixed point of the equations at any load."""

    @pytest.mark.parametrize("load", [0.3, 0.5, 0.7, 0.85, 1.0, 1.2])
    def test_derivative_vanishes(self, model: ReactorModel, load: float) -> None:
        """max |dy/dt| / max(|y|, 1) is at machine precision."""
        y = model.steady_state(load)
        assert model.relative_derivative(y, ModelInputs(load=load)) < 1e-11

    @pytest.mark.parametrize("load", [0.5, 0.7, 1.0])
    def test_equilibrium_power_equals_load(self, model: ReactorModel, load: float) -> None:
        """By construction n_eq = load and Q_sec = load x P0."""
        y = model.steady_state(load)
        assert y[StateIndex.N] == pytest.approx(load, rel=1e-12)
        diag = model.diagnostics(y, ModelInputs(load=load))
        assert diag.q_secondary_mw == pytest.approx(load * model.plant.thermal_power_mw)

    def test_average_temperature_sits_on_the_program(self, model: ReactorModel) -> None:
        """The rods hold T_avg = T_ref(load)."""
        for load in (0.7, 1.0):
            y = model.steady_state(load)
            t_avg = 0.5 * (y[StateIndex.T_CORE] + y[StateIndex.T_SG])
            assert t_avg == pytest.approx(model.t_reference_c(load))

    def test_total_reactivity_is_zero(self, model: ReactorModel) -> None:
        """Rod reactivity cancels the Doppler and moderator terms."""
        for load in (0.6, 1.0):
            diag = model.diagnostics(model.steady_state(load), ModelInputs(load=load))
            assert abs(diag.reactivity) < 1e-12

    def test_precursors_equal_power(self, model: ReactorModel) -> None:
        """Normalised precursors equal n at equilibrium."""
        y = model.steady_state(0.8)
        assert np.allclose(y[1:7], y[StateIndex.N])

    def test_nominal_state_values(self, nominal: np.ndarray, model: ReactorModel) -> None:
        """n = 1, rods at 0, fuel at T_f0, no void, nominal flow and pressure."""
        assert nominal[StateIndex.N] == 1.0
        assert nominal[StateIndex.T_FUEL] == pytest.approx(model.plant.t_fuel_nominal_c)
        assert nominal[StateIndex.T_SG] == pytest.approx(model.plant.t_in_nominal_c)
        assert nominal[StateIndex.PRESSURE] == pytest.approx(155.0)
        assert nominal[StateIndex.VOID] == 0.0
        assert nominal[StateIndex.FLOW] == 18000.0
        assert abs(nominal[StateIndex.ROD_REACTIVITY]) < 1e-15

    @pytest.mark.parametrize("load", [0.0, -0.1, 1.6])
    def test_invalid_load_is_rejected(self, model: ReactorModel, load: float) -> None:
        """Load must lie in (0, 1,5]."""
        with pytest.raises(ValueError, match="load"):
            model.steady_state(load)


class TestDynamicEquilibrium:
    """The derived equilibrium is what the dynamics converge to."""

    def test_nominal_holds_for_24_hours(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """24 h at nominal: n = 1 +- 1e-6 and max |dy/dt| relative < 1e-9."""
        y = _run(model, nominal, ModelInputs(), 480)
        assert abs(y[StateIndex.N] - 1.0) < 1e-6
        assert model.relative_derivative(y, ModelInputs()) < 1e-9

    def test_perturbation_relaxes_back(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """A hot fuel, low pressure and low flow relax to the equilibrium."""
        y = nominal.copy()
        y[StateIndex.T_FUEL] += 20.0
        y[StateIndex.PRESSURE] -= 3.0
        y[StateIndex.FLOW] *= 0.95
        y = _run(model, y, ModelInputs(), 240)
        assert _rel_distance(y, nominal) < 1e-6

    def test_load_step_relaxes_to_the_part_load_equilibrium(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """A step to 80 % load ends at steady_state(0.8)."""
        y = _run(model, nominal, ModelInputs(load=0.8), 160)
        assert _rel_distance(y, model.steady_state(0.8)) < 1e-6

    def test_rod_controller_limits_the_temperature_excursion(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """With rods enabled a +100 pcm insertion ends near n = 1; without, n climbs."""
        insertion = 1.0e-3
        with_rods = _run(model, nominal, ModelInputs(reactivity_insertion=insertion), 20)
        without = _run(
            model,
            nominal,
            ModelInputs(reactivity_insertion=insertion, rod_control_enabled=False),
            20,
        )
        assert abs(with_rods[StateIndex.N] - 1.0) < 0.02
        assert without[StateIndex.N] > 1.03


class TestFeedbackSigns:
    """Negative temperature and void feedback, positive external reactivity."""

    @pytest.mark.parametrize(
        ("index", "delta"),
        [
            (StateIndex.T_FUEL, 10.0),
            (StateIndex.T_CORE, 5.0),
            (StateIndex.T_SG, 5.0),
            (StateIndex.VOID, 0.05),
        ],
    )
    def test_heating_or_voiding_reduces_power(
        self,
        model: ReactorModel,
        nominal: np.ndarray,
        index: StateIndex,
        delta: float,
    ) -> None:
        """+T_f, +T_avg or +void makes dn/dt negative at equilibrium."""
        y = nominal.copy()
        y[index] += delta
        inputs = ModelInputs(rod_control_enabled=False)
        assert model.derivatives(y, inputs)[StateIndex.N] < 0.0

    def test_external_reactivity_raises_power(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """A positive insertion makes dn/dt positive."""
        inputs = ModelInputs(reactivity_insertion=2.0e-4, rod_control_enabled=False)
        assert model.derivatives(nominal, inputs)[StateIndex.N] > 0.0

    def test_integrated_response_to_a_hot_fuel(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """One second after +10 K on the fuel, n is below 1."""
        y = nominal.copy()
        y[StateIndex.T_FUEL] += 10.0
        out = model.advance(y, ModelInputs(rod_control_enabled=False), 1.0).y
        assert out[StateIndex.N] < 1.0

    def test_void_feedback_pushes_power_down(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """With void present and rods frozen the power ends below the no-void case."""
        y = nominal.copy()
        y[StateIndex.VOID] = 0.05
        inputs = ModelInputs(rod_control_enabled=False)
        voided = model.advance(y, inputs, 30.0).y[StateIndex.N]
        reference = model.advance(nominal, inputs, 30.0).y[StateIndex.N]
        assert voided < reference


class TestEnergyClosure:
    """Energy is conserved by construction and the closure is measured."""

    def test_closure_on_a_load_ramp(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """Residual < 1e-6 on every interval of a 100 -> 70 % ramp."""
        y = nominal
        steps = 20
        for k in range(steps):
            start = ModelInputs(load=1.0 - 0.3 * k / steps)
            end = ModelInputs(load=1.0 - 0.3 * (k + 1) / steps)
            result = model.advance(y, start, DT, end)
            assert result.energy_residual_rel < 1e-6
            y = result.y

    def test_energy_balance_at_equilibrium(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """At equilibrium E_gen = E_sec = P0 dt and dU = 0."""
        result = model.advance(nominal, ModelInputs(), DT)
        expected = model.plant.thermal_power_mw * DT
        assert result.energy_generated_mj == pytest.approx(expected, rel=1e-9)
        assert result.energy_removed_mj == pytest.approx(expected, rel=1e-9)
        assert abs(result.internal_energy_change_mj) < 1e-3

    def test_stored_energy_accounts_for_a_power_excess(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """More generation than removal shows up as dU = E_gen - E_sec."""
        y = nominal.copy()
        y[StateIndex.N] = 1.05
        y[1:7] = 1.05
        result = model.advance(y, ModelInputs(rod_control_enabled=False), 20.0)
        gained = result.energy_generated_mj - result.energy_removed_mj
        assert result.internal_energy_change_mj == pytest.approx(gained, rel=1e-6)

    def test_instantaneous_residual_is_zero_at_random_states(self, model: ReactorModel) -> None:
        """du/dt = Q_gen - Q_sec at arbitrary (non-equilibrium) states."""
        rng = np.random.default_rng(3)
        base = model.steady_state()
        for _ in range(8):
            y = base * (1.0 + 0.01 * rng.standard_normal(N_STATES))
            y[StateIndex.VOID] = 0.0
            diag = model.diagnostics(y, ModelInputs(load=0.9))
            assert abs(diag.conservation_residual_mw) < 1e-9

    def test_node_equations_cancel_symbolically(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """sympy: the three node equations add up to P0 n - UA (T_avg - T_sec)."""
        sympy = pytest.importorskip("sympy")
        n, p0, ha, t_f, t_avg = sympy.symbols("n P0 hA T_f T_avg")
        transport, t_c, t_s, ua, t_sec = sympy.symbols("K T_c T_s UA T_sec")
        fuel = p0 * n - ha * (t_f - t_avg)
        core = ha * (t_f - t_avg) - transport * (t_c - t_s)
        steam_generator = transport * (t_c - t_s) - ua * (t_avg - t_sec)
        total = sympy.simplify(fuel + core + steam_generator)
        assert sympy.simplify(total - (p0 * n - ua * (t_avg - t_sec))) == 0

        # El mismo resultado, evaluado con el codigo en un estado fuera de equilibrio.
        y = nominal.copy()
        y[StateIndex.N] = 0.93
        y[StateIndex.T_FUEL] += 4.0
        inputs = ModelInputs(load=0.9)
        diag = model.diagnostics(y, inputs)
        t_avg_value = 0.5 * (y[StateIndex.T_CORE] + y[StateIndex.T_SG])
        symbolic = total.subs(
            {
                n: y[StateIndex.N],
                p0: model.plant.thermal_power_mw,
                ha: model.nominal.ha_mw_k,
                t_f: y[StateIndex.T_FUEL],
                t_avg: t_avg_value,
                transport: 1.0,
                t_c: y[StateIndex.T_CORE],
                t_s: y[StateIndex.T_SG],
                ua: model.nominal.ua_mw_k * inputs.load,
                t_sec: model.t_secondary_c(inputs.load),
            }
        )
        assert diag.du_dt_mw == pytest.approx(float(symbolic))


class TestPropertyFreezing:
    """Frozen properties are a documented operator splitting, and it is measured."""

    def test_frozen_and_live_trajectories_agree(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """Over a ramp the two modes differ by well under a hundredth of a kelvin."""
        live = _build(freeze_properties=False)
        y_frozen = y_live = nominal
        for k in range(10):
            start = ModelInputs(load=1.0 - 0.03 * k)
            end = ModelInputs(load=1.0 - 0.03 * (k + 1))
            y_frozen = model.advance(y_frozen, start, DT, end).y
            result = live.advance(y_live, start, DT, end)
            y_live = result.y
            assert result.energy_residual_rel < 1e-6
        assert np.max(np.abs(y_frozen[7:10] - y_live[7:10])) < 1e-2
        assert abs(y_frozen[StateIndex.N] - y_live[StateIndex.N]) < 1e-5

    def test_live_mode_reports_its_flag(self) -> None:
        """The mode is inspectable."""
        assert _build(freeze_properties=False).freeze_properties is False


class TestIntegratorAgreement:
    """LSODA must agree with an implicit Runge-Kutta reference."""

    def test_lsoda_matches_radau(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """Relative difference below 1e-5 over a short ramp."""
        radau = _build(method="Radau")
        y_l = y_r = nominal
        for k in range(5):
            start = ModelInputs(load=1.0 - 0.04 * k)
            end = ModelInputs(load=1.0 - 0.04 * (k + 1))
            y_l = model.advance(y_l, start, DT, end).y
            y_r = radau.advance(y_r, start, DT, end).y
        assert _rel_distance(y_l, y_r) < 1e-5

    def test_bdf_is_available(self, nominal: np.ndarray) -> None:
        """BDF runs and conserves energy too."""
        result = _build(method="BDF").advance(nominal, ModelInputs(load=0.95), 60.0)
        assert result.energy_residual_rel < 1e-5


class TestInputs:
    """ModelInputs and the controller switch."""

    def test_as_array_order(self) -> None:
        """(load, insertion, demand, leak, rods_enabled)."""
        arr = ModelInputs(0.9, 1e-4, 0.8, 2.0, False).as_array()
        assert arr.tolist() == [0.9, 1e-4, 0.8, 2.0, 0.0]

    def test_frozen_controller_holds_its_states(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """With rods disabled z and r do not move, even under a disturbance."""
        inputs = ModelInputs(load=0.9, rod_control_enabled=False)
        d = model.derivatives(nominal, inputs)
        assert d[StateIndex.ROD_INTEGRAL] == 0.0
        assert d[StateIndex.ROD_REACTIVITY] == 0.0
        out = model.advance(nominal, inputs, DT).y
        assert out[StateIndex.ROD_INTEGRAL] == pytest.approx(0.0, abs=1e-12)
        assert out[StateIndex.ROD_REACTIVITY] == pytest.approx(0.0, abs=1e-15)

    def test_constant_end_inputs_equal_no_end_inputs(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """inputs_end == inputs reproduces the piecewise-constant step."""
        inputs = ModelInputs(load=0.9)
        a = model.advance(nominal, inputs, DT).y
        b = model.advance(nominal, inputs, DT, ModelInputs(load=0.9)).y
        assert np.array_equal(a, b)

    def test_linear_ramp_differs_from_a_step(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """A ramp 1.0 -> 0.8 inside the interval ends above the step to 0.8."""
        step = model.advance(nominal, ModelInputs(load=0.8), DT).y
        ramp = model.advance(nominal, ModelInputs(load=1.0), DT, ModelInputs(load=0.8)).y
        assert ramp[StateIndex.N] > step[StateIndex.N]

    def test_rod_switch_comes_from_the_start_inputs(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """inputs_end cannot re-enable a controller disabled at the start."""
        start = ModelInputs(load=0.9, rod_control_enabled=False)
        end = ModelInputs(load=0.9, rod_control_enabled=True)
        out = model.advance(nominal, start, DT, end).y
        assert out[StateIndex.ROD_REACTIVITY] == pytest.approx(0.0, abs=1e-15)

    def test_pump_demand_lowers_flow(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """A 70 % demand drives the flow toward 70 % with the pump time constant."""
        y = model.advance(nominal, ModelInputs(pump_demand=0.7), 30.0).y
        assert y[StateIndex.FLOW] < nominal[StateIndex.FLOW]
        assert y[StateIndex.FLOW] > 0.7 * nominal[StateIndex.FLOW]

    def test_leak_lowers_pressure(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """A leak depressurises the primary toward P_nom - leak."""
        y = _run(model, nominal, ModelInputs(leak_bar=10.0), 10)
        assert y[StateIndex.PRESSURE] == pytest.approx(145.0, abs=0.5)


class TestVoid:
    """Boiling is flagged through T_sat(P), and void is capped."""

    def test_no_void_while_subcooled(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """A 5 bar leak keeps the core subcooled: void stays at integrator-noise level."""
        y = _run(model, nominal, ModelInputs(leak_bar=5.0), 10)
        assert 0.0 <= y[StateIndex.VOID] < 1e-12

    def test_void_forms_when_the_core_outlet_reaches_saturation(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """A 60 bar leak drops T_sat below T_c: void grows and is capped."""
        y = nominal
        for _ in range(12):
            y = model.advance(y, ModelInputs(leak_bar=60.0), 30.0).y
        assert y[StateIndex.VOID] > 0.0
        assert y[StateIndex.VOID] <= model.plant.void_max_fraction


class TestObservables:
    """The true magnitudes a sensor can read."""

    def test_names_are_exactly_the_published_tuple(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """observables returns OBSERVABLE_NAMES, no more and no less."""
        values = model.observables(nominal, ModelInputs())
        assert tuple(values) == OBSERVABLE_NAMES

    def test_nominal_values(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """Equilibrium readings: T_out0, T_in0, 155 bar, n = 1, rods at 50 %."""
        v = model.observables(nominal, ModelInputs())
        assert v["t_c"] == pytest.approx(model.nominal.t_out_c)
        assert v["t_s"] == pytest.approx(291.0)
        assert v["t_avg_dev_k"] == pytest.approx(0.0, abs=1e-12)
        assert v["p_primary_bar"] == pytest.approx(155.0)
        assert v["power_fraction"] == 1.0
        assert v["q_sec_fraction"] == pytest.approx(1.0)
        assert v["q_cal_fraction"] == pytest.approx(1.0)
        assert v["rod_position_percent"] == pytest.approx(50.0)
        assert v["p_secondary_bar"] == pytest.approx(model.nominal.p_secondary_bar)
        assert v["t_sat_c"] == pytest.approx(model.nominal.t_sat_c)

    def test_no_surge_at_equilibrium(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """Hot and cold leg carry the same flow when T_avg is constant."""
        v = model.observables(nominal, ModelInputs())
        assert v["insurge_kg_s"] == pytest.approx(0.0, abs=1e-6)
        assert v["flow_hot_leg_kg_s"] == pytest.approx(v["flow_cold_leg_kg_s"])

    def test_surge_has_the_sign_of_the_heating(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """Heating pushes water into the pressurizer through the hot leg.

        Two seconds after a +100 pcm insertion T_avg is still rising (later the
        rods bring it back and the surge reverses).
        """
        y = model.advance(nominal, ModelInputs(reactivity_insertion=1e-3), 2.0).y
        v = model.observables(y, ModelInputs(reactivity_insertion=1e-3))
        assert v["insurge_kg_s"] > 0.0
        assert v["flow_hot_leg_kg_s"] > v["flow_cold_leg_kg_s"]

    def test_inputs_are_echoed(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """Load and leak are observables (the containment tap reads the leak)."""
        v = model.observables(nominal, ModelInputs(load=0.9, leak_bar=2.0))
        assert v["load_fraction"] == 0.9
        assert v["leak_bar"] == 2.0

    def test_secondary_follows_the_program(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """T_sec = T_ref - (T_avg0 - T_sec0)."""
        v = model.observables(nominal, ModelInputs(load=0.7))
        assert v["t_sec"] == pytest.approx(model.t_reference_c(0.7) - model.nominal.t_avg_c + 280.0)


class TestDiagnostics:
    """The instantaneous balance of the full state."""

    def test_equilibrium_balance(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """Q_gen = Q_core = Q_sec = P0 and nothing is stored or imbalanced."""
        d = model.diagnostics(nominal, ModelInputs())
        p0 = model.plant.thermal_power_mw
        assert d.q_gen_mw == pytest.approx(p0)
        assert d.q_core_removed_mw == pytest.approx(p0)
        assert d.q_secondary_mw == pytest.approx(p0)
        assert d.q_fuel_to_coolant_mw == pytest.approx(p0)
        assert abs(d.du_dt_mw) < 1e-9
        assert d.energy_imbalance_fraction < 1e-12
        assert d.max_temperature_rate_c_s < 1e-12

    def test_cp_is_the_table_value(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """The diagnostic cp is the one the transport term used."""
        d = model.diagnostics(nominal, ModelInputs())
        assert d.cp_j_kg_k == pytest.approx(model.nominal.cp_j_kg_k)

    def test_transient_rates_respect_the_physical_limit(self, model: ReactorModel) -> None:
        """Even a 30 % load step keeps |dT/dt| under the 10 C/s limit (D9)."""
        y = model.steady_state()
        d = model.diagnostics(y, ModelInputs(load=0.7))
        assert d.max_temperature_rate_c_s < model.plant.limits.max_dt_c_per_s

    def test_imbalance_appears_during_a_transient(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """Core removal differs from generation while energy is being stored."""
        y = nominal.copy()
        y[StateIndex.N] = 1.1
        d = model.diagnostics(y, ModelInputs(rod_control_enabled=False))
        assert d.energy_imbalance_fraction == pytest.approx(0.1 / 1.1, rel=1e-6)
        assert d.du_dt_mw > 0.0

    def test_relative_derivative_detects_disequilibrium(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """A perturbed state has a clearly non-zero relative derivative."""
        y = nominal.copy()
        y[StateIndex.T_FUEL] += 50.0
        assert model.relative_derivative(y, ModelInputs()) > 1e-3


class TestJacobian:
    """The analytic Jacobian handed to LSODA must match finite differences."""

    def test_matches_central_differences(self, model: ReactorModel, nominal: np.ndarray) -> None:
        """Relative error below 1e-5 at perturbed states, void inside its ramp."""
        rng = np.random.default_rng(1)
        plant = model.plant
        u = (0.9, 2.0e-4, 0.95, 3.0, 1.0)
        for _ in range(3):
            y = nominal * (1.0 + 0.01 * rng.standard_normal(N_STATES))
            y[StateIndex.VOID] = 0.1 * rng.random()
            y[StateIndex.T_CORE] = 346.0 + rng.random()
            props = thermal_model._Props(
                cp_j_kg_k=model.table.cp(0.5 * (y[8] + y[9]), y[10]),
                t_sat_c=model.table.t_sat_c(y[10]),
            )

            def f(v: np.ndarray, props: Any = props) -> np.ndarray:
                d = model._derivative(v[:N_STATES].tolist(), u, props)
                t_avg = 0.5 * (v[8] + v[9])
                d.append(plant.thermal_power_mw * v[0])
                d.append(model.nominal.ua_mw_k * u[0] * (t_avg - model.t_secondary_c(u[0])))
                return np.array(d)

            aug = np.concatenate([y, [0.0, 0.0]])
            numeric = np.zeros((N_STATES + 2, N_STATES + 2))
            for j in range(N_STATES + 2):
                h = 1e-6 * max(abs(aug[j]), 1.0)
                e = np.zeros(N_STATES + 2)
                e[j] = h
                numeric[:, j] = (f(aug + e) - f(aug - e)) / (2.0 * h)
            analytic = model.jacobian(y.tolist(), u, props)
            scale = np.maximum(np.abs(numeric), np.abs(analytic)).max(axis=0, keepdims=True)
            assert np.max(np.abs(analytic - numeric) / np.maximum(scale, 1e-12)) < 1e-5


class TestValidation:
    """Bad arguments and failures are explicit."""

    @pytest.mark.parametrize("dt", [0.0, -1.0])
    def test_non_positive_interval(
        self, model: ReactorModel, nominal: np.ndarray, dt: float
    ) -> None:
        """dt_s must be strictly positive."""
        with pytest.raises(ValueError, match="dt_s"):
            model.advance(nominal, ModelInputs(), dt)

    def test_wrong_state_shape(self, model: ReactorModel) -> None:
        """A vector of the wrong length is rejected by every entry point."""
        bad = np.zeros(N_STATES - 1)
        with pytest.raises(ValueError, match="shape"):
            model.advance(bad, ModelInputs(), DT)
        with pytest.raises(ValueError, match="shape"):
            model.derivatives(bad, ModelInputs())
        with pytest.raises(ValueError, match="shape"):
            model.observables(bad, ModelInputs())
        with pytest.raises(ValueError, match="shape"):
            model.diagnostics(bad, ModelInputs())

    def test_pressure_outside_the_table_is_refused(
        self, model: ReactorModel, nominal: np.ndarray
    ) -> None:
        """The model never silently extrapolates the water properties."""
        y = nominal.copy()
        y[StateIndex.PRESSURE] = 20.0
        with pytest.raises(ValueError, match="outside the grid"):
            model.advance(y, ModelInputs(), DT)

    def test_integrator_failure_is_reported(
        self, model: ReactorModel, nominal: np.ndarray, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed solve_ivp becomes a RuntimeError, not a corrupt state."""
        monkeypatch.setattr(
            thermal_model,
            "solve_ivp",
            lambda *a, **k: SimpleNamespace(success=False, message="boom"),
        )
        with pytest.raises(RuntimeError, match="boom"):
            model.advance(nominal, ModelInputs(), DT)


_DETERMINISM_SCRIPT = """
import hashlib
import numpy as np
from data.simulation.plant_params import load_neutronics, load_plant_params
from data.simulation.thermal_model import ModelInputs, ReactorModel
from data.simulation.water_properties import load_water_table
m = ReactorModel(load_plant_params(), load_neutronics(), load_water_table())
y = m.steady_state(0.9)
out = [y]
for k in range(12):
    start = ModelInputs(load=0.9 - 0.01 * k)
    end = ModelInputs(load=0.9 - 0.01 * (k + 1))
    y = m.advance(y, start, 180.0, end).y
    out.append(y)
print(hashlib.sha256(np.array(out).tobytes()).hexdigest())
"""


def _trace_hash(model: ReactorModel) -> str:
    y = model.steady_state(0.9)
    trace = [y]
    for k in range(12):
        y = model.advance(
            y, ModelInputs(load=0.9 - 0.01 * k), DT, ModelInputs(load=0.9 - 0.01 * (k + 1))
        ).y
        trace.append(y)
    return hashlib.sha256(np.array(trace).tobytes()).hexdigest()


class TestDeterminism:
    """Same inputs, same bytes: within a process and across processes."""

    def test_two_models_produce_identical_traces(self) -> None:
        """Two independent models give byte-identical trajectories."""
        assert _trace_hash(_build()) == _trace_hash(_build())

    def test_separate_process_produces_the_same_trace(self, model: ReactorModel) -> None:
        """A fresh interpreter reproduces the in-process hash."""
        completed = subprocess.run(  # noqa: S603 - interprete propio, script fijo
            [sys.executable, "-c", _DETERMINISM_SCRIPT],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout.strip() == _trace_hash(model)
