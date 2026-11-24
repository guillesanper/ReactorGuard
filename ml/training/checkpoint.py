"""Checkpoint de la PINN: contrato, escritura y carga con `weights_only=True` (F4-M3).

El checkpoint (`model.pt`) es un dict de tensores y primitivos (convencion 5 del
plan): `torch.load(..., weights_only=True)` lo carga sin ejecutar codigo arbitrario.
Lleva todo lo que hace falta para predecir sin volver a leer train:

- `contract_version`: version de este contrato (`CONTRACT_VERSION`);
- `model_kwargs`: argumentos de `ReactorPINN` (primitivos);
- `state_dict`: pesos;
- `sensor_tags` y `context_names`: orden de las columnas de entrada y de salida;
- `sensor_mean`, `sensor_std`, `context_mean`, `context_std`: z-score de train.

Sin reloj ni rutas. `torch.save` es estable en bytes para el mismo contenido y el
mismo nombre de fichero (medido en F4-M3: las entradas del zip llevan fecha fija de
1980, `.data/serialization_id` depende del contenido y el prefijo interno sale del
nombre del fichero), de modo que no hace falta un serializador propio.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ml.models.pinn import ReactorPINN
from ml.training.pinn_data import CONTEXT_NAMES, NormalizationStats

CONTRACT_VERSION = 1
"""Version del contenido de model.pt; se sube si cambia cualquier clave."""

MODEL_FILE = "model.pt"
"""Nombre del checkpoint dentro de `pinn.model_dir`."""

CARD_FILE = "model_card.json"
"""Nombre de la ficha del modelo dentro de `pinn.model_dir`."""

_KEYS: frozenset[str] = frozenset(
    {
        "contract_version",
        "model_kwargs",
        "state_dict",
        "sensor_tags",
        "context_names",
        "sensor_mean",
        "sensor_std",
        "context_mean",
        "context_std",
    }
)


@dataclass(frozen=True)
class LoadedModel:
    """Modelo cargado de un checkpoint, en modo evaluacion.

    Attributes:
        model: La red con los pesos del checkpoint.
        stats: Orden de sensores y z-score de train.
    """

    model: ReactorPINN
    stats: NormalizationStats


def checkpoint_document(model: ReactorPINN, stats: NormalizationStats) -> dict[str, Any]:
    """Return the checkpoint dict (tensors and primitives only).

    Args:
        model: Red entrenada.
        stats: Estadisticos de train con el orden de sensores.

    Returns:
        The document that `save_checkpoint` writes.

    Raises:
        ValueError: If the model width does not match the sensors and context.
    """
    if model.n_sensors != len(stats.tags) or model.n_context != len(CONTEXT_NAMES):
        raise ValueError(
            f"checkpoint: model is {model.n_sensors}+{model.n_context}, stats describe "
            f"{len(stats.tags)}+{len(CONTEXT_NAMES)}."
        )
    weights = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
    return {
        "contract_version": CONTRACT_VERSION,
        "model_kwargs": dict(model.model_kwargs()),
        "state_dict": weights,
        "sensor_tags": list(stats.tags),
        "context_names": list(CONTEXT_NAMES),
        "sensor_mean": list(stats.sensor_mean),
        "sensor_std": list(stats.sensor_std),
        "context_mean": list(stats.context_mean),
        "context_std": list(stats.context_std),
    }


def save_checkpoint(path: str | Path, model: ReactorPINN, stats: NormalizationStats) -> Path:
    """Write model.pt.

    Args:
        path: Destino (normalmente `<model_dir>/model.pt`).
        model: Red entrenada.
        stats: Estadisticos de train.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_document(model, stats), target)
    return target


def _floats(document: Mapping[str, Any], key: str, size: int) -> tuple[float, ...]:
    """Return a list of floats of the given size, raising ValueError otherwise."""
    raw = document[key]
    if not isinstance(raw, list) or len(raw) != size:
        raise ValueError(f"checkpoint: '{key}' must be a list of {size} numbers.")
    return tuple(float(v) for v in raw)


def load_checkpoint(path: str | Path) -> LoadedModel:
    """Load model.pt with `weights_only=True` and rebuild the network.

    Args:
        path: Checkpoint escrito por `save_checkpoint`.

    Returns:
        The LoadedModel, with the network in eval mode.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the document does not follow this contract.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    document = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(document, dict) or set(document) != _KEYS:
        raise ValueError(f"checkpoint: {source.name} does not follow the PINN contract.")
    if document["contract_version"] != CONTRACT_VERSION:
        raise ValueError(
            f"checkpoint: contract version {document['contract_version']!r}, expected "
            f"{CONTRACT_VERSION}."
        )
    if tuple(document["context_names"]) != CONTEXT_NAMES:
        raise ValueError(f"checkpoint: context {document['context_names']!r} != {CONTEXT_NAMES}.")
    tags = tuple(str(tag) for tag in document["sensor_tags"])
    stats = NormalizationStats(
        tags=tags,
        sensor_mean=_floats(document, "sensor_mean", len(tags)),
        sensor_std=_floats(document, "sensor_std", len(tags)),
        context_mean=_floats(document, "context_mean", len(CONTEXT_NAMES)),
        context_std=_floats(document, "context_std", len(CONTEXT_NAMES)),
    )
    model = ReactorPINN(**document["model_kwargs"])
    model.load_state_dict(document["state_dict"])
    if model.n_sensors != len(tags):
        raise ValueError(f"checkpoint: model has {model.n_sensors} sensors, tags {len(tags)}.")
    model.eval()
    return LoadedModel(model=model, stats=stats)


def write_json(document: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON document with sorted keys, no NaN and a trailing newline.

    Args:
        document: Contenido (sin reloj ni rutas absolutas).
        path: Destino.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.write_text(text, encoding="utf-8", newline="\n")
    return target
