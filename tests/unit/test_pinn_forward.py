"""Unit tests for ml.models.pinn (MLP del TDD 5.2, F4-M2).

Cubren la arquitectura (capas, numero de parametros del preflight, dropout entre
capas ocultas con claves estables), la validacion de argumentos y de la forma de
entrada, los argumentos primitivos del checkpoint y el determinismo con semilla.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from ml.models.pinn import N_CONTEXT, ReactorPINN

N_SENSORS = 27


class TestArchitecture:
    def test_the_tdd_network_has_the_parameter_count_of_the_preflight(self) -> None:
        model = ReactorPINN(N_SENSORS)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == 146_715  # seccion 2.4 del plan: 31 -> 256 Tanh x3 -> 27

    def test_layers_follow_the_tdd_with_dropout_between_hidden_layers(self) -> None:
        model = ReactorPINN(N_SENSORS, hidden=8, n_hidden=3, dropout=0.1)
        kinds = [type(layer) for layer in model.net]
        assert kinds == [
            nn.Linear, nn.Tanh,
            nn.Dropout, nn.Linear, nn.Tanh,
            nn.Dropout, nn.Linear, nn.Tanh,
            nn.Linear,
        ]  # fmt: skip
        first, last = model.net[0], model.net[-1]
        assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
        assert first.in_features == N_SENSORS + N_CONTEXT
        assert last.out_features == N_SENSORS

    def test_a_single_hidden_layer_has_no_dropout(self) -> None:
        model = ReactorPINN(5, n_context=0, hidden=4, n_hidden=1)
        assert [type(layer) for layer in model.net] == [nn.Linear, nn.Tanh, nn.Linear]

    def test_state_dict_keys_do_not_depend_on_dropout(self) -> None:
        with_dropout = ReactorPINN(N_SENSORS, hidden=8, dropout=0.3).state_dict().keys()
        without = ReactorPINN(N_SENSORS, hidden=8, dropout=0.0).state_dict().keys()
        assert list(with_dropout) == list(without)

    def test_model_kwargs_rebuild_an_identical_network(self) -> None:
        torch.manual_seed(0)
        model = ReactorPINN(N_SENSORS, n_context=2, hidden=16, n_hidden=2, dropout=0.2)
        kwargs = model.model_kwargs()
        assert kwargs == {
            "n_sensors": N_SENSORS,
            "n_context": 2,
            "hidden": 16,
            "n_hidden": 2,
            "dropout": 0.2,
        }
        clone = ReactorPINN(**kwargs)
        clone.load_state_dict(model.state_dict())
        x = torch.randn(3, model.n_inputs)
        model.eval()
        clone.eval()
        assert torch.equal(model(x), clone(x))


class TestForward:
    def test_output_has_one_column_per_sensor(self) -> None:
        model = ReactorPINN(N_SENSORS, hidden=32)
        y = model(torch.randn(4, N_SENSORS + N_CONTEXT))
        assert y.shape == (4, N_SENSORS)
        assert torch.isfinite(y).all()

    @pytest.mark.parametrize("shape", [(4, N_SENSORS), (2, 3, N_SENSORS + N_CONTEXT), (5,)])
    def test_a_wrong_input_shape_is_rejected(self, shape: tuple[int, ...]) -> None:
        model = ReactorPINN(N_SENSORS, hidden=8)
        with pytest.raises(ValueError, match="expects"):
            model(torch.randn(*shape))

    def test_gradient_reaches_every_parameter(self) -> None:
        model = ReactorPINN(N_SENSORS, hidden=8)
        model(torch.randn(6, model.n_inputs)).pow(2).mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    def test_eval_mode_is_deterministic_and_train_mode_applies_dropout(self) -> None:
        torch.manual_seed(1)
        model = ReactorPINN(N_SENSORS, hidden=64, dropout=0.5)
        x = torch.randn(8, model.n_inputs)
        model.eval()
        assert torch.equal(model(x), model(x))
        model.train()
        assert not torch.equal(model(x), model(x))

    def test_the_same_seed_gives_the_same_weights(self) -> None:
        torch.manual_seed(42)
        first = ReactorPINN(N_SENSORS, hidden=16).state_dict()
        torch.manual_seed(42)
        second = ReactorPINN(N_SENSORS, hidden=16).state_dict()
        assert all(torch.equal(first[key], second[key]) for key in first)


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [{"n_sensors": 0}, {"n_sensors": 3, "hidden": 0}, {"n_sensors": 3, "n_hidden": 0}],
    )
    def test_sizes_below_one_are_rejected(self, kwargs: dict[str, int]) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            ReactorPINN(**kwargs)

    def test_a_negative_context_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_context"):
            ReactorPINN(3, n_context=-1)

    @pytest.mark.parametrize("dropout", [-0.1, 1.0])
    def test_dropout_outside_zero_one_is_rejected(self, dropout: float) -> None:
        with pytest.raises(ValueError, match="dropout"):
            ReactorPINN(3, dropout=dropout)
