"""Model serving: loads PINN + MAPIE wrapper for online inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mapie.regression import MapieRegressor

from ml.models.pinn import ReactorPINN

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    anomaly_score: float          # [0, 1]
    is_anomaly: bool
    interval_low: float
    interval_high: float
    confidence_level: float


class ReactorPredictor:
    """Wraps PINN + MAPIE for production inference.

    Args:
        model_path: Path to saved PINN checkpoint (.pt).
        confidence_level: Conformal prediction confidence (e.g. 0.95).
        anomaly_threshold: Score threshold above which we flag anomaly.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_level: float = 0.95,
        anomaly_threshold: float = 0.5,
    ) -> None:
        self.confidence_level = confidence_level
        self.anomaly_threshold = anomaly_threshold
        self._model: ReactorPINN | None = None
        self._mapie: MapieRegressor | None = None

        if model_path.exists():
            self._load(model_path)
        else:
            logger.warning("Model checkpoint not found at %s — predictor not ready.", model_path)

    def _load(self, model_path: Path) -> None:
        # weights_only=True usa el unpickler restringido de PyTorch: solo tensores
        # y tipos primitivos, nunca ejecución de pickle arbitrario. El checkpoint
        # llega desde un bucket GCS; si esa cadena de custodia se compromete, un
        # torch.load con el default inseguro seria RCE en el pod de inferencia
        # safety-critical. El checkpoint solo contiene model_kwargs (primitivos)
        # y model_state (tensores), ambos compatibles con weights_only=True.
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self._model = ReactorPINN(**checkpoint["model_kwargs"])
        self._model.load_state_dict(checkpoint["model_state"])
        self._model.eval()
        logger.info("PINN loaded from %s", model_path)

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def predict(self, features: np.ndarray) -> PredictionResult:
        """Run inference on a single feature vector.

        Args:
            features: 1-D or 2-D numpy array of extracted features.

        Returns:
            PredictionResult with anomaly score and interval.
        """
        if not self.is_ready:
            raise RuntimeError("Predictor not loaded — call _load() first.")

        assert self._model is not None  # guaranteed by is_ready check above
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            _, anomaly_score_t = self._model(x)

        score = float(anomaly_score_t.squeeze())

        # TODO(MAPIE, Fase 6): sustituir este intervalo stub por el intervalo
        # conforme calibrado de MAPIE. Al implementarlo, el umbral definitivo
        # debe leerse de serving.anomaly_threshold en params.yaml (3.0 sigma
        # sobre el intervalo nominal), NO del default 0.5 de este constructor,
        # que solo tiene sentido como placeholder sobre el score en [0, 1].
        #
        # Mientras sea stub, el intervalo se acota a [0, 1] para que el score y
        # sus limites vivan en el mismo dominio [0, 1] del sigmoid: sin el clamp
        # interval_low sale negativo para score < 0.5 e interval_high es 1.0
        # constante, un intervalo que no es defendible ni como placeholder.
        margin = 1.0 - score
        interval_low = max(0.0, score - margin)
        interval_high = min(1.0, score + margin)
        return PredictionResult(
            anomaly_score=score,
            is_anomaly=score >= self.anomaly_threshold,
            interval_low=interval_low,
            interval_high=interval_high,
            confidence_level=self.confidence_level,
        )
