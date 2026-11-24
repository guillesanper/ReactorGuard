"""Stage train: entrenamiento de la PINN sobre la simulacion (F4-M3, T6.2-T6.4).

Lee las particiones de `prepare_sim` (`pinn.data_dir`), entrena el MLP del TDD 5.2
(D4-1) sobre los pares sin etiqueta (D4-6) y escribe en `pinn.model_dir` el
checkpoint (`model.pt`) y la ficha (`model_card.json`), mas las metricas en
`pinn.train_metrics_path`. Cada epoca se registra en MLflow (D4-14).

Perdida por lote:

    total = data + f(epoca) (lambda_energy energy + lambda_mass mass)

con `data` el MSE en z-score frente al valor medido de t+1, `energy` y `mass` los
terminos de `ml.models.physics_loss` sobre la prediccion DESNORMALIZADA (convencion
6), y `f` la rampa lineal del curriculum: 0 en la epoca 0 y 1 desde la epoca
`curriculum_epochs`.

Adam con ReduceLROnPlateau. La cifra que vigilan el planificador de lr, la parada
temprana y la eleccion de la mejor epoca es la perdida total de val con los lambdas
OBJETIVO (`val_total`): asi es comparable entre epocas aunque la rampa cambie los
pesos con los que se entrena. La paciencia solo cuenta epocas a partir del final del
curriculum. Al terminar se restauran los pesos de la mejor epoca.

Aleatoriedad (convencion 4): la inicializacion y el barajado usan Generators
sembrados desde `pinn.seed`; `torch.manual_seed` solo se fija aqui, en el punto de
entrada del entrenamiento (`deterministic_torch`), porque el dropout lee el
generador global. Los algoritmos deterministas y los hilos se fijan y se restauran
al salir.

Salidas sin reloj ni rutas absolutas (convencion 5): el tiempo del entrenamiento,
el host y las versiones van solo a MLflow.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data.simulation.dataset_builder import file_sha256
from data.simulation.plant_params import load_plant_params
from data.simulation.sensor_catalog import load_sensor_catalog
from data.simulation.simulation_params import load_simulation_params
from data.simulation.water_properties import WaterTable, load_water_table
from ml.models.physics_loss import PhysicsConstants, physics_loss_terms
from ml.models.physics_state import StateIndex, build_state_index, denormalize, physical_state
from ml.models.pinn import ReactorPINN
from ml.training.checkpoint import (
    CARD_FILE,
    CONTRACT_VERSION,
    MODEL_FILE,
    save_checkpoint,
    write_json,
)
from ml.training.pinn_data import (
    CONTEXT_NAMES,
    NormalizationStats,
    PairSet,
    build_pairs,
    fit_normalization,
    noise_mse,
    partition_columns,
    persistence_mse,
)
from ml.training.pinn_params import (
    DEFAULT_PARAMS_PATH,
    MLFLOW_URI_SCHEME,
    PinnTrainParams,
    load_pinn_train_params,
)
from ml.training.sim_dataset import ASSIGNMENT_NAME

_LOG = logging.getLogger(__name__)

METRICS_VERSION = 1
"""Version del JSON de metricas del stage."""

FIT_SPLITS: tuple[str, ...] = ("train", "val")
"""Particiones que lee el entrenamiento; test queda para F4-M4 y F4-M6."""

MLFLOW_ARTIFACTS_DIR = "artifacts"
"""Subdirectorio de artefactos junto a la base sqlite (D4-14: mlruns/artifacts)."""


# ----------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingData:
    """Todo lo que el loop necesita, ya resuelto.

    Attributes:
        train: Pares de train.
        val: Pares de validacion.
        stats: z-score de train y orden de sensores.
        index: Columnas de cada componente del estado fisico.
        table: Tabla de propiedades del agua (cp).
        constants: Escalas de planta.
        digests: sha256 de los ficheros de datos leidos (por nombre de fichero).
    """

    train: PairSet
    val: PairSet
    stats: NormalizationStats
    index: StateIndex
    table: WaterTable
    constants: PhysicsConstants
    digests: Mapping[str, str]


def _read_tags(data_dir: Path) -> tuple[str, ...]:
    """Return the sensor order written by prepare_sim."""
    path = data_dir / ASSIGNMENT_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path.as_posix()} not found: run the prepare_sim stage "
            "(.\\infra\\scripts\\Invoke-Pipeline.ps1 -Stage prepare_sim)."
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(tag) for tag in document["tags"])


def load_training_data(params_path: str | Path, params: PinnTrainParams) -> TrainingData:
    """Read train and val, fit the z-score on train and build the pairs.

    Args:
        params_path: Fichero de parametros (secciones pinn: y simulation:).
        params: Seccion pinn: resuelta.

    Returns:
        The TrainingData.

    Raises:
        FileNotFoundError: If a partition, the assignment or a table is missing.
        ValueError: If the sensor order of the data does not match the catalog
            or the data are malformed.
    """
    simulation = load_simulation_params(params_path)
    catalog = load_sensor_catalog(simulation.sensors_path)
    tags = _read_tags(params.data_dir)
    if tags != tuple(catalog.tags):
        raise ValueError("train: the sensor order of prepare_sim does not match the catalog.")
    frames: dict[str, pd.DataFrame] = {}
    digests = {ASSIGNMENT_NAME: file_sha256(params.data_dir / ASSIGNMENT_NAME)}
    for split in FIT_SPLITS:
        path = params.data_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path.as_posix()} not found: run the prepare_sim stage.")
        frames[split] = pd.read_parquet(path, columns=partition_columns(tags))
        digests[path.name] = file_sha256(path)

    roles = catalog.roles
    stats = fit_normalization(frames["train"], tags, roles)
    index = build_state_index(tags, roles)
    constants = PhysicsConstants.from_plant(load_plant_params(simulation.plant_parameters_path))
    pairs = {
        split: build_pairs(frames[split], stats, roles, index, constants) for split in FIT_SPLITS
    }
    if len(pairs["train"]) == 0 or len(pairs["val"]) == 0:
        raise ValueError(
            f"train: needs trainable pairs in train and val, got {len(pairs['train'])} and "
            f"{len(pairs['val'])}."
        )
    return TrainingData(
        train=pairs["train"],
        val=pairs["val"],
        stats=stats,
        index=index,
        table=load_water_table(simulation.water_table_path),
        constants=constants,
        digests=digests,
    )


# ----------------------------------------------------------------------
# Aleatoriedad e inicializacion
# ----------------------------------------------------------------------


@contextmanager
def deterministic_torch(seed: int, num_threads: int) -> Iterator[None]:
    """Fix the global torch state for one training and restore it afterwards.

    Fija `torch.manual_seed(seed)` (lo usa el dropout), los algoritmos deterministas
    y los hilos intra-op. El generador global, el modo determinista y los hilos
    previos se restauran al salir, de modo que entrenar no altera el proceso.

    Args:
        seed: Semilla del generador global.
        num_threads: Hilos intra-op.

    Yields:
        None.
    """
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        torch.set_num_threads(num_threads)
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(deterministic)
            torch.set_num_threads(threads)


def derived_seeds(seed: int, count: int) -> list[int]:
    """Return independent 63-bit seeds derived from one seed (numpy SeedSequence).

    Args:
        seed: `pinn.seed`.
        count: Semillas pedidas.

    Returns:
        count integers, the same for the same seed.
    """
    states = np.random.SeedSequence(seed).generate_state(count, dtype=np.uint64)
    return [int(s >> np.uint64(1)) for s in states]


def initialize(model: ReactorPINN, generator: torch.Generator) -> None:
    """Re-initialise every Linear layer as PyTorch does, drawing from a Generator.

    Reproduce `nn.Linear.reset_parameters` (Kaiming uniforme con a = sqrt(5) y sesgo
    uniforme en +-1/sqrt(fan_in)) sin tocar el generador global.

    Args:
        model: Red recien construida.
        generator: Generator sembrado desde `pinn.seed`.
    """
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)
            bound = 1.0 / math.sqrt(layer.in_features)
            nn.init.uniform_(layer.bias, -bound, bound, generator=generator)


def curriculum_factor(epoch: int, curriculum_epochs: int) -> float:
    """Return the fraction of the target lambdas used in an epoch (linear ramp).

    Args:
        epoch: Epoca (desde 0).
        curriculum_epochs: Epocas de la rampa; 0 = lambdas enteros desde el inicio.

    Returns:
        min(1, epoch / curriculum_epochs).
    """
    if curriculum_epochs == 0:
        return 1.0
    return min(1.0, epoch / curriculum_epochs)


# ----------------------------------------------------------------------
# Perdidas y evaluacion
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LossParts:
    """Terminos de una evaluacion, como floats.

    Attributes:
        data: MSE en z-score.
        energy: Termino de energia (media sobre filas evaluables).
        mass: Termino de masa (media sobre filas cuasi-estacionarias).
    """

    data: float
    energy: float
    mass: float

    def total(self, lambda_energy: float, lambda_mass: float) -> float:
        """Return data + lambda_energy energy + lambda_mass mass."""
        return self.data + lambda_energy * self.energy + lambda_mass * self.mass


@dataclass(frozen=True)
class _Physics:
    """Lo que necesitan los terminos fisicos (sin estado)."""

    stats: NormalizationStats
    index: StateIndex
    table: WaterTable
    constants: PhysicsConstants


def _terms(
    prediction: torch.Tensor, target: torch.Tensor, steady: torch.Tensor, physics: _Physics
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
    """Return (data, energy, n_energy, mass, n_mass) of a batch, with gradient."""
    data = torch.mean((prediction - target) ** 2)
    engineering = denormalize(prediction, physics.stats.sensor_mean, physics.stats.sensor_std)
    terms = physics_loss_terms(
        physical_state(engineering, physics.index), steady, physics.table, physics.constants
    )
    return data, terms.energy, terms.n_energy, terms.mass, terms.n_mass


def evaluate(model: ReactorPINN, pairs: PairSet, physics: _Physics) -> LossParts:
    """Return the loss terms of a whole partition in eval mode (no dropout, no grad).

    Args:
        model: Red.
        pairs: Pares de la particion.
        physics: Estadisticos, indices, tabla y constantes.

    Returns:
        The LossParts.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        prediction = model(pairs.inputs)
        data, energy, _, mass, _ = _terms(prediction, pairs.targets, pairs.steady, physics)
    model.train(was_training)
    return LossParts(data=float(data.item()), energy=float(energy.item()), mass=float(mass.item()))


@dataclass(frozen=True)
class EpochRecord:
    """Lo que se registra de una epoca.

    Attributes:
        epoch: Epoca (desde 0).
        lr: Tasa de aprendizaje usada en la epoca.
        lambda_factor: Fraccion de los lambdas objetivo (curriculum).
        train: Terminos de train (media de los lotes ponderada por filas).
        train_total: Perdida total de train con los lambdas de la epoca.
        val: Terminos de val (modo evaluacion).
        val_total: Perdida total de val con los lambdas OBJETIVO (la que se vigila).
        val_total_curriculum: Perdida total de val con los lambdas de la epoca.
        physics_share: (lambda_e energy + lambda_m mass) / data de val con los
            lambdas objetivo (peso relativo de la fisica, hallazgo M2-1).
    """

    epoch: int
    lr: float
    lambda_factor: float
    train: LossParts
    train_total: float
    val: LossParts
    val_total: float
    val_total_curriculum: float
    physics_share: float

    def metrics(self) -> dict[str, float]:
        """Return the flat metric names logged to MLflow for this epoch."""
        return {
            "data": self.train.data,
            "energy": self.train.energy,
            "mass": self.train.mass,
            "total": self.train_total,
            "val_data": self.val.data,
            "val_energy": self.val.energy,
            "val_mass": self.val.mass,
            "val_total": self.val_total,
            "val_total_curriculum": self.val_total_curriculum,
            "physics_share": self.physics_share,
            "lambda_factor": self.lambda_factor,
            "lr": self.lr,
        }


class RunLogger(Protocol):
    """Destino del registro de un entrenamiento (MLflow en produccion)."""

    def log_params(self, params: Mapping[str, str]) -> None:
        """Record the parameters of the run."""

    def log_epoch(self, record: EpochRecord) -> None:
        """Record the metrics of one epoch."""

    def log_summary(self, metrics: Mapping[str, float], tags: Mapping[str, str]) -> None:
        """Record the final metrics and tags of the run."""

    def log_artifacts(self, paths: Sequence[Path]) -> None:
        """Attach files to the run."""


class NullRunLogger:
    """RunLogger que no registra nada (tests y uso sin MLflow)."""

    def log_params(self, params: Mapping[str, str]) -> None:
        """Ignore the parameters."""

    def log_epoch(self, record: EpochRecord) -> None:
        """Ignore the epoch."""

    def log_summary(self, metrics: Mapping[str, float], tags: Mapping[str, str]) -> None:
        """Ignore the summary."""

    def log_artifacts(self, paths: Sequence[Path]) -> None:
        """Ignore the artifacts."""


@dataclass
class FitResult:
    """Resultado del loop.

    Attributes:
        history: Un EpochRecord por epoca ejecutada.
        best_epoch: Epoca cuyos pesos se restauraron.
        stopped_early: Si paro por paciencia antes de max_epochs.
    """

    history: list[EpochRecord] = field(default_factory=list)
    best_epoch: int = -1
    stopped_early: bool = False

    @property
    def epochs_run(self) -> int:
        """Return the number of epochs executed."""
        return len(self.history)

    @property
    def best(self) -> EpochRecord:
        """Return the record of the best epoch."""
        return self.history[self.best_epoch]


def _train_epoch(
    model: ReactorPINN,
    optimizer: torch.optim.Optimizer,
    pairs: PairSet,
    physics: _Physics,
    lambdas: tuple[float, float],
    batch_size: int,
    generator: torch.Generator,
) -> LossParts:
    """Run one epoch over shuffled batches and return the row-weighted terms."""
    model.train()
    order = torch.randperm(len(pairs), generator=generator)
    data_sum = energy_sum = mass_sum = 0.0
    n_energy_sum = n_mass_sum = 0
    for start in range(0, len(pairs), batch_size):
        batch = order[start : start + batch_size]
        prediction = model(pairs.inputs[batch])
        data, energy, n_energy, mass, n_mass = _terms(
            prediction, pairs.targets[batch], pairs.steady[batch], physics
        )
        loss = data + lambdas[0] * energy + lambdas[1] * mass
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()
        data_sum += float(data.item()) * len(batch)
        energy_sum += float(energy.item()) * n_energy
        mass_sum += float(mass.item()) * n_mass
        n_energy_sum += n_energy
        n_mass_sum += n_mass
    return LossParts(
        data=data_sum / len(pairs),
        energy=energy_sum / n_energy_sum if n_energy_sum else 0.0,
        mass=mass_sum / n_mass_sum if n_mass_sum else 0.0,
    )


def fit(
    model: ReactorPINN, data: TrainingData, params: PinnTrainParams, logger: RunLogger
) -> FitResult:
    """Train the network and leave in it the weights of the best epoch.

    Debe llamarse dentro de `deterministic_torch` para que el dropout sea
    reproducible; el barajado usa su propio Generator.

    Args:
        model: Red ya inicializada.
        data: Datos resueltos.
        params: Hiperparametros.
        logger: Destino del registro por epoca.

    Returns:
        The FitResult.
    """
    physics = _Physics(data.stats, data.index, data.table, data.constants)
    shuffle = torch.Generator().manual_seed(derived_seeds(params.seed, 2)[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=params.lr_factor,
        patience=params.lr_patience,
        min_lr=params.min_lr,
    )
    result = FitResult()
    best_value = math.inf
    best_state: dict[str, torch.Tensor] = {}
    bad_epochs = 0
    for epoch in range(params.max_epochs):
        factor = curriculum_factor(epoch, params.curriculum_epochs)
        lambdas = (factor * params.lambda_energy, factor * params.lambda_mass)
        lr = float(optimizer.param_groups[0]["lr"])
        train = _train_epoch(
            model, optimizer, data.train, physics, lambdas, params.batch_size, shuffle
        )
        val = evaluate(model, data.val, physics)
        val_total = val.total(params.lambda_energy, params.lambda_mass)
        record = EpochRecord(
            epoch=epoch,
            lr=lr,
            lambda_factor=factor,
            train=train,
            train_total=train.total(*lambdas),
            val=val,
            val_total=val_total,
            val_total_curriculum=val.total(*lambdas),
            physics_share=(val_total - val.data) / val.data if val.data > 0.0 else 0.0,
        )
        result.history.append(record)
        logger.log_epoch(record)
        scheduler.step(val_total)
        if val_total < best_value:
            best_value = val_total
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            result.best_epoch = epoch
            bad_epochs = 0
        elif epoch >= params.curriculum_epochs:
            bad_epochs += 1
        _LOG.debug("epoch %d: val_total %.6g (best %d)", epoch, val_total, result.best_epoch)
        if bad_epochs >= params.early_stopping_patience:
            result.stopped_early = epoch + 1 < params.max_epochs
            break
    model.load_state_dict(best_state)
    return result


# ----------------------------------------------------------------------
# MLflow
# ----------------------------------------------------------------------


def mlflow_store_dir(tracking_uri: str) -> Path:
    """Return the directory that holds the sqlite file of a tracking URI.

    Raises:
        ValueError: If the URI is not a sqlite:///<path> URI.
    """
    if not tracking_uri.startswith(MLFLOW_URI_SCHEME) or tracking_uri == MLFLOW_URI_SCHEME:
        raise ValueError(f"MLflow: expected a {MLFLOW_URI_SCHEME}<path> URI, got {tracking_uri!r}.")
    return Path(tracking_uri[len(MLFLOW_URI_SCHEME) :]).parent


class MlflowRunLogger:
    """RunLogger sobre un MlflowClient con URI explicita (D4-14).

    Crea el directorio del store y, si no existe, el experimento con los artefactos
    en `<store>/artifacts`. No toca el estado global de `mlflow` (sin
    `set_tracking_uri`), de modo que nunca crea `mlflow.db` en el directorio de
    trabajo.

    Args:
        tracking_uri: URI sqlite del store.
        experiment: Nombre del experimento.
        run_name: Nombre del run.
    """

    def __init__(self, tracking_uri: str, experiment: str, run_name: str) -> None:
        from mlflow.tracking import MlflowClient

        store = mlflow_store_dir(tracking_uri)
        store.mkdir(parents=True, exist_ok=True)
        self._client = MlflowClient(tracking_uri=tracking_uri)
        found = self._client.get_experiment_by_name(experiment)
        if found is None:
            artifacts = (store / MLFLOW_ARTIFACTS_DIR).resolve().as_uri()
            experiment_id = self._client.create_experiment(experiment, artifact_location=artifacts)
        else:
            experiment_id = found.experiment_id
        self.run_id: str = self._client.create_run(experiment_id, run_name=run_name).info.run_id

    def log_params(self, params: Mapping[str, str]) -> None:
        """Record the parameters of the run."""
        from mlflow.entities import Param

        # mlflow.entities.Param y RunTag no anotan __init__ (no-untyped-call).
        entries = [Param(k, v) for k, v in params.items()]  # type: ignore[no-untyped-call]
        self._client.log_batch(self.run_id, params=entries)

    def log_epoch(self, record: EpochRecord) -> None:
        """Record the metrics of one epoch with step = epoch."""
        from mlflow.entities import Metric

        stamp = int(time.time() * 1000)
        metrics = [
            Metric(name, value, stamp, record.epoch) for name, value in record.metrics().items()
        ]
        self._client.log_batch(self.run_id, metrics=metrics)

    def log_summary(self, metrics: Mapping[str, float], tags: Mapping[str, str]) -> None:
        """Record the final metrics and tags of the run."""
        from mlflow.entities import Metric, RunTag

        stamp = int(time.time() * 1000)
        self._client.log_batch(
            self.run_id,
            metrics=[Metric(k, v, stamp, 0) for k, v in metrics.items()],
            tags=[RunTag(k, v) for k, v in tags.items()],  # type: ignore[no-untyped-call]
        )

    def log_artifacts(self, paths: Sequence[Path]) -> None:
        """Attach files to the run."""
        for path in paths:
            self._client.log_artifact(self.run_id, str(path))

    def finish(self, status: str) -> None:
        """Close the run with a terminal status (FINISHED or FAILED)."""
        self._client.set_terminated(self.run_id, status=status)


# ----------------------------------------------------------------------
# Salidas
# ----------------------------------------------------------------------


def _round(value: float) -> float:
    """Return a float with 12 significant digits (estable entre plataformas cercanas)."""
    return float(f"{value:.12g}")


def _parts(parts: LossParts, prefix: str) -> dict[str, float]:
    """Return the terms of a LossParts under a prefix."""
    return {
        f"{prefix}data": parts.data,
        f"{prefix}energy": parts.energy,
        f"{prefix}mass": parts.mass,
    }


def hyperparameters(params: PinnTrainParams) -> dict[str, Any]:
    """Return the training parameters as primitives, without MLflow or output paths."""
    raw = asdict(params)
    for key in ("mlflow_tracking_uri", "mlflow_experiment", "model_dir", "train_metrics_path"):
        raw.pop(key)
    raw["data_dir"] = params.data_dir.as_posix()
    return raw


def metrics_document(
    result: FitResult, data: TrainingData, params: PinnTrainParams, n_parameters: int
) -> dict[str, Any]:
    """Return the metrics JSON of the stage (sin reloj ni rutas).

    Args:
        result: Resultado del loop.
        data: Datos (pares y lineas base).
        params: Hiperparametros.
        n_parameters: Parametros entrenables de la red.

    Returns:
        The document written to `pinn.train_metrics_path`.
    """
    best = result.best
    persistence = persistence_mse(data.val)
    noise = noise_mse(data.val)
    return {
        "version": METRICS_VERSION,
        "pairs": {"train": len(data.train), "val": len(data.val)},
        "steady_pairs": {
            "train": int(data.train.steady.sum().item()),
            "val": int(data.val.steady.sum().item()),
        },
        "n_parameters": n_parameters,
        "epochs_run": result.epochs_run,
        "best_epoch": best.epoch,
        "stopped_early": result.stopped_early,
        "lambdas": {"energy": params.lambda_energy, "mass": params.lambda_mass},
        "best": {
            key: _round(value)
            for key, value in {
                **_parts(best.train, "train_"),
                **_parts(best.val, "val_"),
                "val_total": best.val_total,
                "physics_share": best.physics_share,
                "lr": best.lr,
            }.items()
        },
        "baselines": {
            "val_persistence_mse": _round(persistence),
            "val_noise_mse": _round(noise),
            "val_data_over_persistence": _round(best.val.data / persistence),
            "val_data_over_noise": _round(best.val.data / noise),
        },
    }


def model_card(
    model: ReactorPINN,
    result: FitResult,
    data: TrainingData,
    params: PinnTrainParams,
    model_sha256: str,
) -> dict[str, Any]:
    """Return model_card.json (sin reloj ni rutas absolutas).

    Args:
        model: Red con los pesos guardados.
        result: Resultado del loop.
        data: Datos (orden de columnas y huellas de los ficheros).
        params: Hiperparametros.
        model_sha256: sha256 de model.pt.

    Returns:
        The card document.
    """
    best = result.best
    return {
        "contract_version": CONTRACT_VERSION,
        "model": {
            "class": "ml.models.pinn.ReactorPINN",
            "kwargs": dict(model.model_kwargs()),
            "n_parameters": sum(p.numel() for p in model.parameters()),
            "file": MODEL_FILE,
            "sha256": model_sha256,
        },
        "inputs": {"sensor_tags": list(data.stats.tags), "context_names": list(CONTEXT_NAMES)},
        "objective": "value medido de t+1 en z-score (D4-6); mascara de masa (x(t), y(t+1))",
        "training": hyperparameters(params),
        "data": {
            "sha256": dict(sorted(data.digests.items())),
            "pairs": {"train": len(data.train), "val": len(data.val)},
        },
        "result": {
            "epochs_run": result.epochs_run,
            "best_epoch": best.epoch,
            "stopped_early": result.stopped_early,
            "val_data": _round(best.val.data),
            "val_energy": _round(best.val.energy),
            "val_mass": _round(best.val.mass),
            "val_total": _round(best.val_total),
        },
        "framework": {"torch": torch.__version__.split("+")[0]},
    }


@dataclass(frozen=True)
class TrainingOutcome:
    """Lo que deja una ejecucion del stage.

    Attributes:
        model: Red con los pesos de la mejor epoca.
        result: Historia y mejor epoca.
        model_path: model.pt escrito.
        card_path: model_card.json escrito.
        metrics_path: JSON de metricas escrito.
        seconds: Tiempo de pared del loop (solo informativo).
    """

    model: ReactorPINN
    result: FitResult
    model_path: Path
    card_path: Path
    metrics_path: Path
    seconds: float


def run_training(
    params_path: str | Path, params: PinnTrainParams, logger: RunLogger
) -> TrainingOutcome:
    """Train the PINN and write the checkpoint, the card and the metrics.

    Args:
        params_path: Fichero de parametros (secciones pinn: y simulation:).
        params: Seccion pinn: resuelta.
        logger: Destino del registro (MlflowRunLogger o NullRunLogger).

    Returns:
        The TrainingOutcome.
    """
    data = load_training_data(params_path, params)
    init_seed = derived_seeds(params.seed, 2)[0]
    with deterministic_torch(params.seed, params.num_threads):
        model = ReactorPINN(
            n_sensors=data.train.n_sensors,
            n_context=len(CONTEXT_NAMES),
            hidden=params.hidden,
            n_hidden=params.n_hidden,
            dropout=params.dropout,
        )
        initialize(model, torch.Generator().manual_seed(init_seed))
        n_parameters = sum(p.numel() for p in model.parameters())
        logger.log_params(
            {
                **{k: str(v) for k, v in hyperparameters(params).items()},
                "n_parameters": str(n_parameters),
                "n_train_pairs": str(len(data.train)),
                "n_val_pairs": str(len(data.val)),
            }
        )
        start = time.perf_counter()
        result = fit(model, data, params, logger)
        seconds = time.perf_counter() - start
    model.eval()

    model_path = save_checkpoint(params.model_dir / MODEL_FILE, model, data.stats)
    sha = file_sha256(model_path)
    card = model_card(model, result, data, params, sha)
    card_path = write_json(card, params.model_dir / CARD_FILE)
    metrics = metrics_document(result, data, params, n_parameters)
    metrics_path = write_json(metrics, params.train_metrics_path)

    best = result.best
    logger.log_summary(
        {
            **{f"best_{k}": v for k, v in best.metrics().items()},
            "best_epoch": float(best.epoch),
            "epochs_run": float(result.epochs_run),
            "train_seconds": seconds,
            "val_persistence_mse": metrics["baselines"]["val_persistence_mse"],
            "val_noise_mse": metrics["baselines"]["val_noise_mse"],
        },
        {
            "model_sha256": sha,
            "stopped_early": str(result.stopped_early),
            "torch_version": torch.__version__,
            "host": platform.node(),
            "platform": platform.platform(),
        },
    )
    logger.log_artifacts([card_path, metrics_path])
    _LOG.info(
        "Trained %d epochs in %.1f s (best %d%s): val_data %.5f (persistence %.5f, noise %.5f), "
        "val_energy %.3g, val_mass %.3g",
        result.epochs_run,
        seconds,
        best.epoch,
        ", stopped early" if result.stopped_early else "",
        best.val.data,
        metrics["baselines"]["val_persistence_mse"],
        metrics["baselines"]["val_noise_mse"],
        best.val.energy,
        best.val.mass,
    )
    return TrainingOutcome(
        model=model,
        result=result,
        model_path=model_path,
        card_path=card_path,
        metrics_path=metrics_path,
        seconds=seconds,
    )


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> TrainingOutcome:
    """Run the train stage end to end, logging to MLflow.

    Args:
        params_path: Fichero de parametros.

    Returns:
        The TrainingOutcome.
    """
    params = load_pinn_train_params(params_path)
    logger = MlflowRunLogger(params.mlflow_tracking_uri, params.mlflow_experiment, "train")
    try:
        outcome = run_training(params_path, params, logger)
    except BaseException:
        logger.finish("FAILED")
        raise
    logger.finish("FINISHED")
    return outcome


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
