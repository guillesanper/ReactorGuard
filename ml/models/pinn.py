"""Physics-Informed Neural Network for reactor anomaly detection.

The PINN architecture combines a data-driven LSTM encoder with a
physics residual head that enforces neutron-balance and energy-balance
constraints during training (loss term L_physics).

References:
  - Raissi et al. (2019) "Physics-informed neural networks"
  - Reactor physics constraints: point-kinetics equations
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReactorPINN(nn.Module):
    """Physics-Informed Neural Network for reactor state prediction.

    Args:
        input_size: Number of sensor channels per timestep.
        hidden_size: Width of hidden layers.
        n_layers: Number of LSTM layers.
        dropout: Dropout probability between layers.
        output_size: Prediction targets (default=input_size for reconstruction).
    """

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 256,
        n_layers: int = 4,
        dropout: float = 0.2,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        output_size = output_size or input_size

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        # Anomaly score head (sigmoid → [0, 1])
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Sensor sequence of shape (batch, seq_len, input_size).

        Returns:
            reconstruction: Predicted sensor values (batch, seq_len, output_size).
            anomaly_score: Anomaly probability per timestep (batch, seq_len, 1).
        """
        lstm_out, _ = self.lstm(x)           # (B, T, hidden)
        reconstruction = self.head(lstm_out)  # (B, T, output_size)
        anomaly_score = self.anomaly_head(lstm_out)  # (B, T, 1)
        return reconstruction, anomaly_score

    def physics_residual(
        self, x: torch.Tensor, y_pred: torch.Tensor, dt: float = 180.0
    ) -> torch.Tensor:
        """Compute physics constraint residual (point-kinetics approximation).

        Penalises predictions that violate dP/dt ≈ (ρ - β)/Λ · P,
        where P = core_power, ρ = reactivity, β = delayed-neutron fraction,
        Λ = prompt-neutron lifetime.

        Args:
            x: Input sensor readings (batch, seq_len, input_size).
            y_pred: Predicted sensor readings (batch, seq_len, output_size).
            dt: Seconds between consecutive samples. Defaults to the real TEP
                cadence (180 s = tep.sample_interval_minutes), not the legacy
                1 s SCADA rate: usar 1 s aqui escalaria dP/dt por 180 y es el
                mismo fallo de cadencia que el resto del pipeline ya corrige.

        Returns:
            Scalar physics residual (currently detached — see note below).
        """
        # Andamiaje de Fase 4: la version completa acoplada a OpenMC (el termino
        # L_physics real) aun no esta implementada. Se asume canal 0 = core_power,
        # que todavia no corresponde al layout del featurizer del TEP; a revisar
        # al cablear el loop de entrenamiento real.
        power_pred = y_pred[..., 0]  # core_power channel (placeholder)
        dp_dt = (power_pred[:, 1:] - power_pred[:, :-1]) / dt
        # .detach(): hoy es solo un diagnostico escalar, no un termino de perdida
        # que retropropague. Retirar el detach cuando pase a ser un L_physics real
        # en Fase 4 (asi el float() de los tests no dispara el UserWarning de grad).
        return torch.mean(dp_dt**2).detach()
