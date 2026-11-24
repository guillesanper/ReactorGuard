"""PINN training entry point.

Usage:
    python -m ml.training.train --params params.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml

from ml.features.feature_params import load_feature_params
from ml.models.pinn import ReactorPINN

logger = logging.getLogger(__name__)


def load_params(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def build_model(params: dict[str, Any], input_size: int) -> ReactorPINN:
    # Andamiaje hasta F4-M3: solo se adapta a la firma del MLP de F4-M2 (D4-1). Las
    # claves de training: se retiran en M3 junto con la reescritura (D4-9).
    t = params["training"]
    return ReactorPINN(
        n_sensors=input_size,
        hidden=t["hidden_size"],
        n_hidden=t["n_layers"],
        dropout=t["dropout"],
    )


def train(params_path: str = "params.yaml") -> None:
    params = load_params(params_path)
    t = params["training"]

    torch.manual_seed(t["seed"])

    # input_size deriva del contrato de features, no de una lista de canales
    # cableada: features.sensor_channels se retiro de params.yaml en favor de
    # features.sensor_selection (ver ml/features/feature_params.py), asi que el
    # ancho de entrada es el numero de sensores que el featurizer debe resolver.
    feature_params = load_feature_params(params_path)
    input_size = feature_params.selection.expected_sensor_count
    model = build_model(params, input_size)
    _optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )

    _physics_lambda: float = t["physics_lambda"]

    mlflow.set_experiment("reactorguard-pinn")
    with mlflow.start_run():
        mlflow.log_params(t)

        # TODO: load DataLoaders from ml/features or Feast
        # TODO: implement training loop with early stopping
        # TODO: log metrics, save model artifact

        logger.info("Training loop not yet implemented — skeleton only.")
        mlflow.log_metric("placeholder_loss", 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train(args.params)
