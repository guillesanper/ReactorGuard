"""Unit tests for PINN model forward pass (shape and output range checks)."""

import pytest
import torch

from ml.models.pinn import ReactorPINN


@pytest.fixture
def model() -> ReactorPINN:
    return ReactorPINN(input_size=8, hidden_size=64, n_layers=2, dropout=0.0)


def test_output_shapes(model: ReactorPINN) -> None:
    batch, seq_len, n_sensors = 4, 20, 8
    x = torch.randn(batch, seq_len, n_sensors)
    recon, score = model(x)
    assert recon.shape == (batch, seq_len, n_sensors)
    assert score.shape == (batch, seq_len, 1)


def test_anomaly_score_in_range(model: ReactorPINN) -> None:
    x = torch.randn(2, 10, 8)
    _, score = model(x)
    assert score.min() >= 0.0
    assert score.max() <= 1.0


def test_physics_residual_nonneg(model: ReactorPINN) -> None:
    x = torch.randn(2, 10, 8)
    recon, _ = model(x)
    residual = model.physics_residual(x, recon)
    assert float(residual) >= 0.0
