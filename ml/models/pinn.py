"""Red de la PINN de ReactorGuard: el MLP del TDD 5.2 (D4-1, D4-2).

Predice el estado de los sensores en el paso siguiente (t + 180 s) a partir del
estado medido en t y de un contexto de operacion de 4 entradas (D4-2:
`power_level`, `dP_th/dt` normalizada, `sin` y `cos` de la hora del dia). Entrada y
salida van en z-score por sensor; el residuo prediccion - medicion es la senal de
anomalia. Las perdidas fisicas viven en `ml.models.physics_loss` y operan sobre la
prediccion desnormalizada, no aqui.

Arquitectura (TDD 5.2, con dropout entre capas ocultas por el riesgo de
sobreajuste de Plan_Fases):

    Linear(n_sensors + n_context, hidden) -> Tanh
    [-> Dropout -> Linear(hidden, hidden) -> Tanh] x (n_hidden - 1)
    -> Linear(hidden, n_sensors)

Sustituye al LSTM del andamiaje (H1 del plan): mismo nombre de clase, otra red.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

N_CONTEXT = 4
"""Entradas de contexto de operacion (D4-2)."""


class ReactorPINN(nn.Module):
    """MLP de prediccion del paso siguiente.

    Args:
        n_sensors: Sensores de entrada y de salida (27 en la simulacion).
        n_context: Entradas de contexto concatenadas tras los sensores.
        hidden: Anchura de las capas ocultas.
        n_hidden: Numero de capas ocultas (cada una seguida de Tanh).
        dropout: Probabilidad de dropout entre capas ocultas. Con 0 el modulo
            sigue presente (identidad), de modo que las claves del state_dict no
            dependen del valor.

    Raises:
        ValueError: If a size is < 1 (n_context < 0) or dropout is not in [0, 1).
    """

    def __init__(
        self,
        n_sensors: int,
        n_context: int = N_CONTEXT,
        hidden: int = 256,
        n_hidden: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_sensors < 1 or hidden < 1 or n_hidden < 1:
            raise ValueError(
                f"ReactorPINN: n_sensors, hidden and n_hidden must be >= 1, got "
                f"{n_sensors}, {hidden}, {n_hidden}."
            )
        if n_context < 0:
            raise ValueError(f"ReactorPINN: n_context must be >= 0, got {n_context}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"ReactorPINN: dropout must be in [0, 1), got {dropout}.")
        self.n_sensors = n_sensors
        self.n_context = n_context
        self.hidden = hidden
        self.n_hidden = n_hidden
        self.dropout = dropout

        layers: list[nn.Module] = [nn.Linear(n_sensors + n_context, hidden), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, n_sensors))
        self.net = nn.Sequential(*layers)

    @property
    def n_inputs(self) -> int:
        """Return the input width (n_sensors + n_context)."""
        return self.n_sensors + self.n_context

    def model_kwargs(self) -> dict[str, Any]:
        """Return the constructor arguments as primitives (for the checkpoint)."""
        return {
            "n_sensors": self.n_sensors,
            "n_context": self.n_context,
            "hidden": self.hidden,
            "n_hidden": self.n_hidden,
            "dropout": self.dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the normalised sensor state of the next step.

        Args:
            x: Tensor (batch, n_sensors + n_context): sensores en z-score en el
                orden del catalogo seguidos del contexto.

        Returns:
            Tensor (batch, n_sensors) con la prediccion en z-score.

        Raises:
            ValueError: If x is not 2-D with n_inputs columns.
        """
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError(f"ReactorPINN expects (batch, {self.n_inputs}), got {tuple(x.shape)}.")
        out: torch.Tensor = self.net(x)
        return out
