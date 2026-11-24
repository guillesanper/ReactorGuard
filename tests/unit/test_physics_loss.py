"""Unit tests for ml.models.physics_loss (perdidas de energia y masa, D4-4 y D4-5).

Cubren las ecuaciones con cifras a mano (Q_gen del flujo, Q_rem con el caudal de
ENTRADA y cp de tabla), cp como coeficiente sin gradiente y sin extrapolar (la fila
fuera de la tabla no cuenta), la mascara de cuasi-estacionario (estricta, NaN fuera),
que la perdida es ~0 sobre el estado verdadero estacionario del modelo de referencia
y que su gradiente apunta a reducir el desbalance, tambien a traves de la
desnormalizacion y de la mediana por rol.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from data.simulation.physics_constraints import removed_heat_mw
from data.simulation.plant_params import load_plant_params
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.water_properties import WaterTable
from ml.models.physics_loss import (
    PhysicsConstants,
    PhysicsLossTerms,
    cp_coefficient,
    energy_balance_loss,
    energy_balance_residual,
    mass_balance_loss,
    mass_balance_residual,
    physics_loss_terms,
    quasi_steady_mask,
)
from ml.models.physics_state import (
    PhysicalState,
    build_state_index,
    denormalize,
    normalize,
    physical_state,
)
from tests.support.sim_frames import (
    CONFIGS,
    build_catalog,
    build_model,
    build_table,
    ramp_loads,
    run_load_profile,
    wide_frame,
)

DT_S = 180.0


@pytest.fixture(scope="module")
def constants() -> PhysicsConstants:
    return PhysicsConstants.from_plant(load_plant_params(CONFIGS / "plant_parameters.yaml"))


@pytest.fixture(scope="module")
def table() -> WaterTable:
    return build_table()


def make_state(rows: list[list[float]], pressure: list[float] | None = None) -> PhysicalState:
    """Return a PhysicalState from rows [flux, flow_in, t_in, t_out, flow_out]."""
    values = torch.tensor(rows, dtype=torch.float64)
    p = torch.tensor(pressure or [155.0] * len(rows), dtype=torch.float64)
    return PhysicalState(values=values, pressure_bar=p)


def true_frame(loads: list[float]) -> tuple[pd.DataFrame, SensorCatalog]:
    model, catalog = build_model(), build_catalog()
    return wide_frame(run_load_profile(model, loads), model, catalog), catalog


class TestConstants:
    def test_constants_come_from_the_plant_parameters(self, constants: PhysicsConstants) -> None:
        assert constants.thermal_power_mw == 3000.0
        assert constants.n_loops == 4
        assert constants.nominal_flow_per_loop_kg_s == 4500.0
        assert constants.quasi_steady_dt_avg_k_s == 0.02

    @pytest.mark.parametrize(
        "field", ["thermal_power_mw", "nominal_flow_per_loop_kg_s", "quasi_steady_dt_avg_k_s"]
    )
    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
    def test_non_positive_or_non_finite_scales_are_rejected(
        self, constants: PhysicsConstants, field: str, value: float
    ) -> None:
        kwargs = {
            "thermal_power_mw": 3000.0,
            "n_loops": 4,
            "nominal_flow_per_loop_kg_s": 4500.0,
            "quasi_steady_dt_avg_k_s": 0.02,
            field: value,
        }
        with pytest.raises(ValueError, match=field):
            PhysicsConstants(**kwargs)  # type: ignore[arg-type]

    def test_zero_loops_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_loops"):
            PhysicsConstants(3000.0, 0, 4500.0, 0.02)


class TestCpCoefficient:
    def test_cp_is_the_table_value_and_carries_no_gradient(self, table: WaterTable) -> None:
        t = torch.tensor([300.0, 305.5], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([155.0, 150.0], dtype=torch.float64)
        cp, ok = cp_coefficient(table, t, p)
        assert ok.tolist() == [True, True]
        np.testing.assert_allclose(cp.numpy(), table.cp(np.array([300.0, 305.5]), [155.0, 150.0]))
        assert not cp.requires_grad
        assert cp.dtype == torch.float64

    def test_outside_the_grid_or_nan_is_not_extrapolated(self, table: WaterTable) -> None:
        t_hi = table.t_range_c[1]
        t = torch.tensor([t_hi + 1.0, 300.0, math.nan, 300.0])
        p = torch.tensor([155.0, table.p_range_bar[0] - 1.0, 155.0, 155.0])
        cp, ok = cp_coefficient(table, t, p)
        assert ok.tolist() == [False, False, False, True]
        assert cp[:3].tolist() == [0.0, 0.0, 0.0]
        assert cp[3].item() > 4182.0  # cp de tabla, no la constante del TDD (C4)

    def test_an_all_outside_batch_returns_zeros(self, table: WaterTable) -> None:
        cp, ok = cp_coefficient(table, torch.tensor([1.0e4]), torch.tensor([155.0]))
        assert cp.tolist() == [0.0]
        assert ok.tolist() == [False]


class TestEnergyResidual:
    def test_residual_by_hand_uses_flux_inlet_flow_and_table_cp(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        state = make_state([[90.0, 4400.0, 290.0, 322.0, 4600.0]])
        residual, valid = energy_balance_residual(state, table, constants)
        cp = table.cp(306.0, 155.0)
        q_gen = 3000.0 * 90.0 / 100.0
        q_rem = 4 * 4400.0 * cp * 32.0 * 1e-6
        assert valid.tolist() == [True]
        assert residual.item() == pytest.approx((q_gen - q_rem) / 3000.0, rel=1e-12)

    def test_residual_matches_removed_heat_of_the_constraints(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        rows = [[100.0, 4500.0, 291.0, 325.0, 4500.0], [70.0, 4510.0, 293.0, 316.0, 4490.0]]
        state = make_state(rows, [155.0, 154.0])
        residual, _ = energy_balance_residual(state, table, constants)
        arr = np.array(rows)
        q_rem = removed_heat_mw(4 * arr[:, 1], arr[:, 3], arr[:, 2], [155.0, 154.0], table)
        expected = (3000.0 * arr[:, 0] / 100.0 - q_rem) / 3000.0
        np.testing.assert_allclose(residual.numpy(), expected, rtol=1e-12)

    def test_outlet_flow_does_not_enter_the_energy_balance(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        a = make_state([[100.0, 4500.0, 291.0, 325.0, 4500.0]])
        b = make_state([[100.0, 4500.0, 291.0, 325.0, 9000.0]])
        assert energy_balance_loss(a, table, constants).item() == pytest.approx(
            energy_balance_loss(b, table, constants).item(), rel=0.0, abs=0.0
        )


class TestEnergyLoss:
    @pytest.mark.parametrize("load", [1.0, 0.7])
    def test_loss_is_about_zero_on_the_true_steady_state(
        self, table: WaterTable, constants: PhysicsConstants, load: float
    ) -> None:
        frame, catalog = true_frame([load] * 5)
        index = build_state_index(catalog.tags, catalog.roles)
        state = physical_state(torch.tensor(frame.to_numpy()), index)
        residual, valid = energy_balance_residual(state, table, constants)
        assert bool(valid.all())
        # Suelo de la forma estacionaria: 0,0075 % p50 sobre true_value (seccion 2.3).
        assert float(residual.abs().max()) < 1.0e-4
        assert energy_balance_loss(state, table, constants).item() < 1.0e-8

    def test_a_row_outside_the_table_does_not_count(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        inside = [[90.0, 4400.0, 290.0, 322.0, 4600.0], [100.0, 4500.0, 291.0, 325.0, 4500.0]]
        outside = [[100.0, 4500.0, 291.0, 325.0, 4500.0]]
        p_out = table.p_range_bar[1] + 10.0
        mixed = make_state(inside + outside, [155.0, 155.0, p_out])
        terms = physics_loss_terms(mixed, torch.ones(3, dtype=torch.bool), table, constants)
        assert terms.n_energy == 2
        expected = energy_balance_loss(make_state(inside), table, constants)
        assert terms.energy.item() == pytest.approx(expected.item(), rel=1e-12)

    def test_no_evaluable_row_gives_zero_with_a_gradient(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        values = torch.tensor([[100.0, 4500.0, 291.0, 325.0, 4500.0]], requires_grad=True)
        state = PhysicalState(values=values, pressure_bar=torch.tensor([1000.0]))
        loss = energy_balance_loss(state, table, constants)
        assert loss.item() == 0.0
        loss.backward()
        assert values.grad is not None
        assert values.grad.abs().sum().item() == 0.0

    def test_a_nan_row_does_not_poison_the_gradient(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        rows = [[90.0, 4400.0, 290.0, 322.0, 4600.0], [math.nan, 4500.0, 291.0, 325.0, 4500.0]]
        values = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
        state = PhysicalState(values=values, pressure_bar=torch.tensor([155.0, 155.0]))
        loss = energy_balance_loss(state, table, constants)
        loss.backward()
        assert math.isfinite(loss.item()) and loss.item() > 0.0
        assert values.grad is not None
        assert torch.isfinite(values.grad).all()
        assert values.grad[1].abs().sum().item() == 0.0

    def test_gradient_points_to_reducing_the_imbalance(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        # Flujo por encima del calor retirado: Q_gen > Q_rem.
        values = torch.tensor(
            [[125.0, 4500.0, 291.0, 325.0, 4500.0]], dtype=torch.float64, requires_grad=True
        )
        state = PhysicalState(
            values=values, pressure_bar=torch.tensor([155.0], dtype=torch.float64)
        )
        residual, _ = energy_balance_residual(state, table, constants)
        assert residual.item() > 0.0
        loss = energy_balance_loss(state, table, constants)
        loss.backward()
        assert values.grad is not None
        grad = values.grad[0]
        assert grad[0] > 0.0  # bajar el flujo
        assert grad[1] < 0.0  # subir el caudal de entrada
        assert grad[2] > 0.0  # bajar t_in (sube el salto)
        assert grad[3] < 0.0  # subir t_out
        assert grad[4] == 0.0  # el caudal de salida no entra
        with torch.no_grad():
            stepped = values - 1.0e-6 * values.grad / values.grad.norm() * values.norm()
        after = energy_balance_loss(
            PhysicalState(values=stepped, pressure_bar=state.pressure_bar), table, constants
        )
        assert after.item() < loss.item()


class TestQuasiSteadyMask:
    def test_the_threshold_is_strict_and_nan_is_not_steady(
        self, constants: PhysicsConstants
    ) -> None:
        limit = constants.quasi_steady_dt_avg_k_s * DT_S
        before = torch.tensor([300.0, 300.0, 300.0, math.nan, 300.0])
        after = torch.tensor([300.0, 300.0 + limit, 300.0 + 0.5 * limit, 300.0, 300.0 - 2 * limit])
        mask = quasi_steady_mask(before, after, DT_S, constants)
        assert mask.tolist() == [True, False, True, False, False]
        assert mask.dtype == torch.bool

    def test_the_mask_carries_no_gradient(self, constants: PhysicsConstants) -> None:
        a = torch.tensor([300.0], requires_grad=True)
        assert not quasi_steady_mask(a, a + 0.1, DT_S, constants).requires_grad

    @pytest.mark.parametrize("dt_s", [0.0, -180.0, math.nan])
    def test_a_non_positive_step_is_rejected(
        self, constants: PhysicsConstants, dt_s: float
    ) -> None:
        with pytest.raises(ValueError, match="dt_s"):
            quasi_steady_mask(torch.zeros(1), torch.zeros(1), dt_s, constants)

    def test_measured_ramp_matches_the_rule_mask(self, constants: PhysicsConstants) -> None:
        frame, catalog = true_frame(ramp_loads(0.7, 1.0, 10, hold_steps=3))
        index = build_state_index(catalog.tags, catalog.roles)
        t_avg = physical_state(torch.tensor(frame.to_numpy()), index).t_avg
        mask = quasi_steady_mask(t_avg[:-1], t_avg[1:], DT_S, constants)
        rate = np.diff(t_avg.numpy()) / DT_S
        assert mask.tolist() == (np.abs(rate) < 0.02).tolist()


class TestMassLoss:
    def test_residual_by_hand_is_normalised_by_the_nominal_loop_flow(
        self, constants: PhysicsConstants
    ) -> None:
        state = make_state([[100.0, 4400.0, 290.0, 322.0, 4490.0]])
        residual, finite = mass_balance_residual(state, constants)
        assert finite.tolist() == [True]
        assert residual.item() == pytest.approx(90.0 / 4500.0, rel=1e-12)

    def test_only_quasi_steady_rows_count(self, constants: PhysicsConstants) -> None:
        state = make_state(
            [[100.0, 4400.0, 290.0, 322.0, 4490.0], [100.0, 4400.0, 290.0, 322.0, 4850.0]]
        )
        steady = torch.tensor([True, False])
        loss = mass_balance_loss(state, steady, constants)
        assert loss.item() == pytest.approx((90.0 / 4500.0) ** 2, rel=1e-12)
        assert mass_balance_loss(state, torch.tensor([False, False]), constants).item() == 0.0

    def test_loss_is_about_zero_on_the_true_steady_state(self, constants: PhysicsConstants) -> None:
        frame, catalog = true_frame([1.0] * 5)
        index = build_state_index(catalog.tags, catalog.roles)
        state = physical_state(torch.tensor(frame.to_numpy()), index)
        steady = torch.ones(len(frame), dtype=torch.bool)
        assert mass_balance_loss(state, steady, constants).item() < 1.0e-20

    def test_gradient_pulls_both_legs_together(self, constants: PhysicsConstants) -> None:
        values = torch.tensor(
            [[100.0, 4400.0, 290.0, 322.0, 4490.0]], dtype=torch.float64, requires_grad=True
        )
        state = PhysicalState(values=values, pressure_bar=torch.tensor([155.0]))
        mass_balance_loss(state, torch.tensor([True]), constants).backward()
        assert values.grad is not None
        grad = values.grad[0]
        assert grad[1] < 0.0 < grad[4]
        assert grad[0] == grad[2] == grad[3] == 0.0

    def test_a_nan_leg_does_not_count(self, constants: PhysicsConstants) -> None:
        state = make_state([[100.0, math.nan, 290.0, 322.0, 4490.0]])
        _, finite = mass_balance_residual(state, constants)
        assert finite.tolist() == [False]
        assert mass_balance_loss(state, torch.tensor([True]), constants).item() == 0.0


class TestPhysicsLossTerms:
    def test_terms_and_counts_of_a_batch(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        state = make_state(
            [[90.0, 4400.0, 290.0, 322.0, 4600.0], [100.0, 4500.0, 291.0, 325.0, 4560.0]]
        )
        steady = torch.tensor([False, True])
        terms = physics_loss_terms(state, steady, table, constants)
        assert isinstance(terms, PhysicsLossTerms)
        assert (terms.n_energy, terms.n_mass) == (2, 1)
        assert terms.energy.item() == pytest.approx(
            energy_balance_loss(state, table, constants).item(), rel=1e-12
        )
        assert terms.mass.item() == pytest.approx((60.0 / 4500.0) ** 2, rel=1e-12)
        weighted = terms.weighted(lambda_energy=2.0, lambda_mass=0.5)
        assert weighted.item() == pytest.approx(
            2.0 * terms.energy.item() + 0.5 * terms.mass.item(), rel=1e-12
        )

    def test_a_steady_mask_of_the_wrong_length_is_rejected(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        state = make_state([[90.0, 4400.0, 290.0, 322.0, 4600.0]])
        with pytest.raises(ValueError, match="steady"):
            physics_loss_terms(state, torch.ones(2, dtype=torch.bool), table, constants)

    def test_gradient_flows_from_the_loss_to_the_normalised_prediction(
        self, table: WaterTable, constants: PhysicsConstants
    ) -> None:
        frame, catalog = true_frame([1.0] * 4)
        x = torch.tensor(frame.to_numpy())
        mean = x.mean(dim=0)
        std = torch.full_like(mean, 2.0)
        index = build_state_index(catalog.tags, catalog.roles)
        z = normalize(x, mean, std).clone()
        flux_cols = list(index.channels("flux"))
        z[:, flux_cols] += 5.0  # +10 % de flujo: desbalance de energia
        z.requires_grad_(True)
        state = physical_state(denormalize(z, mean, std), index)
        steady = torch.ones(len(frame), dtype=torch.bool)
        terms = physics_loss_terms(state, steady, table, constants)
        assert terms.energy.item() > 1.0e-3
        terms.weighted(1.0, 1.0).backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()
        assert (z.grad[:, flux_cols].sum(dim=1) > 0.0).all()
