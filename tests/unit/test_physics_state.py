"""Unit tests for ml.models.physics_state (estado fisico de la PINN, D4-3).

Cubren el contrato del orden del estado, el mapeo de roles del catalogo versionado
a indices, la desnormalizacion (inversa de la normalizacion y diferenciable), la
mediana por rol con la misma definicion que las reglas (media de los dos centrales
con un numero par de canales) y la coincidencia con la mediana por rol de las
constraints (pandas, `skipna`) sobre un frame del modelo de referencia.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.models.physics_state import (
    PINN_STATE_ORDER,
    PRESSURE_ROLE,
    STATE_ROLES,
    PhysicalState,
    build_state_index,
    channel_median,
    denormalize,
    normalize,
    physical_state,
)
from tests.support.sim_frames import build_catalog, build_model, run_load_profile, wide_frame


class TestContract:
    def test_state_order_is_frozen(self) -> None:
        # Contrato de D4-3: cambiar el orden rompe checkpoints y perdidas.
        assert PINN_STATE_ORDER == ("flux", "flow_in", "t_in", "t_out", "flow_out")

    def test_each_component_reads_the_role_of_the_plan(self) -> None:
        assert dict(STATE_ROLES) == {
            "flux": "flux",
            "flow_in": "flow_cold_leg",
            "t_in": "T_in",
            "t_out": "T_out",
            "flow_out": "flow_hot_leg",
        }
        assert tuple(STATE_ROLES) == PINN_STATE_ORDER
        assert PRESSURE_ROLE == "P"

    def test_the_versioned_catalog_maps_to_the_expected_tags(self) -> None:
        catalog = build_catalog()
        index = build_state_index(catalog.tags, catalog.roles)
        tags = catalog.tags

        def named(cols: tuple[int, ...]) -> list[str]:
            return [tags[i] for i in cols]

        assert named(index.channels("flux")) == ["NI-FLUX-01", "NI-FLUX-02", "NI-FLUX-03"]
        assert named(index.channels("flow_in")) == ["FT-CL-1", "FT-CL-2"]
        assert named(index.channels("t_in")) == ["TC-CL-1", "TC-CL-2"]
        assert named(index.channels("t_out")) == ["TC-HL-1", "TC-HL-2"]
        assert named(index.channels("flow_out")) == ["FT-HL-1", "FT-HL-2"]
        assert named(index.pressure) == ["PT-PRZ-01", "PT-PRZ-02"]
        assert index.tags == tags


class TestBuildStateIndex:
    ROLES = {
        "flux": ["F"],
        "flow_cold_leg": ["MC"],
        "T_in": ["TI"],
        "T_out": ["TO"],
        "flow_hot_leg": ["MH"],
        "P": ["P"],
    }

    def test_indices_follow_the_given_column_order(self) -> None:
        index = build_state_index(["P", "TO", "TI", "MH", "MC", "F"], self.ROLES)
        assert index.components == ((5,), (4,), (2,), (1,), (3,))
        assert index.pressure == (0,)

    def test_a_missing_role_is_reported(self) -> None:
        roles = {k: v for k, v in self.ROLES.items() if k != "T_out"}
        with pytest.raises(KeyError, match="'T_out' is missing"):
            build_state_index(["P", "TO", "TI", "MH", "MC", "F"], roles)

    def test_an_empty_role_is_reported(self) -> None:
        with pytest.raises(KeyError, match="'P' is missing or empty"):
            build_state_index(["TO", "TI", "MH", "MC", "F"], {**self.ROLES, "P": []})

    def test_a_tag_absent_from_the_matrix_is_reported(self) -> None:
        with pytest.raises(KeyError, match=r"needs sensors \['MH'\]"):
            build_state_index(["P", "TO", "TI", "MC", "F"], self.ROLES)

    def test_duplicated_tags_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicated"):
            build_state_index(["P", "P", "TO", "TI", "MH", "MC", "F"], self.ROLES)

    def test_an_unknown_component_is_rejected(self) -> None:
        index = build_state_index(["P", "TO", "TI", "MH", "MC", "F"], self.ROLES)
        with pytest.raises(KeyError, match="unknown component"):
            index.channels("pressure")


class TestNormalization:
    def test_denormalize_inverts_normalize(self) -> None:
        x = torch.tensor([[300.0, 4500.0], [310.0, 4400.0]], dtype=torch.float64)
        mean, std = [305.0, 4450.0], [5.0, 50.0]
        z = normalize(x, mean, std)
        assert torch.allclose(z, torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=torch.float64))
        assert torch.allclose(denormalize(z, mean, std), x)

    def test_denormalize_keeps_dtype_and_gradient(self) -> None:
        z = torch.zeros(1, 2, dtype=torch.float32, requires_grad=True)
        out = denormalize(z, torch.tensor([1.0, 2.0]), (3.0, 4.0))
        assert out.dtype == torch.float32
        out.sum().backward()
        assert z.grad is not None
        assert z.grad.tolist() == [[3.0, 4.0]]


class TestChannelMedian:
    def test_odd_count_takes_the_central_channel(self) -> None:
        x = torch.tensor([[3.0, 1.0, 2.0, 9.0]])
        assert channel_median(x, [0, 1, 2]).item() == 2.0

    def test_even_count_averages_the_two_central_channels_like_pandas(self) -> None:
        x = torch.tensor([[4.0, 1.0, 10.0, 2.0]])
        assert channel_median(x, [0, 1, 2, 3]).item() == 3.0
        assert channel_median(x, [0, 1]).item() == 2.5

    def test_gradient_goes_to_the_central_channels(self) -> None:
        x = torch.tensor([[1.0, 5.0, 3.0]], requires_grad=True)
        channel_median(x, [0, 1, 2]).sum().backward()
        assert x.grad is not None
        assert x.grad.tolist() == [[0.0, 0.0, 1.0]]

    def test_no_channel_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            channel_median(torch.zeros(1, 2), [])


class TestPhysicalState:
    def test_matches_the_role_value_of_the_constraints_on_a_noisy_frame(self) -> None:
        model, catalog = build_model(), build_catalog()
        frame = wide_frame(
            run_load_profile(model, [1.0] * 5), model, catalog, rng=np.random.default_rng(3)
        )
        index = build_state_index(catalog.tags, catalog.roles)
        state = physical_state(torch.tensor(frame.to_numpy()), index)
        for name in PINN_STATE_ORDER:
            expected = frame[list(catalog.roles[STATE_ROLES[name]])].median(axis=1).to_numpy()
            np.testing.assert_allclose(state.component(name).numpy(), expected, rtol=1e-12)
        expected_p = frame[list(catalog.roles[PRESSURE_ROLE])].median(axis=1).to_numpy()
        np.testing.assert_allclose(state.pressure_bar.numpy(), expected_p, rtol=1e-12)

    def test_values_follow_the_state_order_and_t_avg_is_the_mean(self) -> None:
        index = build_state_index(
            ["F", "MC", "TI", "TO", "MH", "P"],
            {
                "flux": ["F"],
                "flow_cold_leg": ["MC"],
                "T_in": ["TI"],
                "T_out": ["TO"],
                "flow_hot_leg": ["MH"],
                "P": ["P"],
            },
        )
        x = torch.tensor([[100.0, 4500.0, 290.0, 320.0, 4501.0, 155.0]])
        state = physical_state(x, index)
        assert isinstance(state, PhysicalState)
        assert state.values.tolist() == [[100.0, 4500.0, 290.0, 320.0, 4501.0]]
        assert state.pressure_bar.tolist() == [155.0]
        assert state.t_avg.tolist() == [305.0]
        with pytest.raises(KeyError, match="unknown component"):
            state.component("P")

    def test_a_matrix_of_the_wrong_width_is_rejected(self) -> None:
        catalog = build_catalog()
        index = build_state_index(catalog.tags, catalog.roles)
        with pytest.raises(ValueError, match="expects"):
            physical_state(torch.zeros(2, len(catalog.tags) - 1), index)
