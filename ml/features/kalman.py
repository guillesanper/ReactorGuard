"""Online Kalman filtering of sensor signals, one filter per instrument tag.

El filtro estima el estado "verdadero" de un sensor a partir de lecturas
ruidosas. Lo que interesa aguas abajo no es tanto el estado estimado como el
RESIDUAL: la diferencia entre lo que el sensor mide y lo que el modelo predecia
que iba a medir. Normalizado por su propia incertidumbre, ese residual es una
senal de anomalia adimensional y comparable entre canales heterogeneos, que es
justo lo que hace falta cuando conviven caudales, presiones y composiciones.

Este modulo existe separado a proposito: lo consumen dos sitios (el feature
`kalman_residual` de ml/features/pipeline.py y el KalmanResidualDetector de
data/validation/sensor_validator.py) y mantener dos implementaciones del mismo
filtro seria garantizar que divergen.

Modelo: cinematica de orden 1, estado x = [posicion, velocidad]^T.

    Prediccion
        x_pred = F x
        P_pred = F P F^T + Q

        F = [[1, dt],
             [0,  1]]        la velocidad persiste, la posicion avanza con ella

        Q = q * [[dt^3/3, dt^2/2],
                 [dt^2/2, dt   ]]   ruido blanco en la aceleracion, integrado

    Correccion (observacion escalar: solo se mide la posicion)
        H = [[1, 0]]
        y = z - H x_pred                       innovacion (el residual)
        S = H P_pred H^T + R                   varianza de la innovacion
        K = P_pred H^T S^-1                    ganancia de Kalman
        x = x_pred + K y
        P = (I - K H) P_pred

El residual normalizado y / sqrt(S) es, bajo el modelo, de media cero y varianza
uno; por eso el umbral de anomalia se expresa en sigmas y no en unidades de
ingenieria.

DOS SENALES, NO UNA. Una deriva lineal sostenida vive en el espacio nulo de este
modelo: el filtro aprende la pendiente y el residual decae a cero. Medido, una
rampa de 0,5 u/paso durante 100 pasos (50 unidades de deriva acumulada) alcanza
un maximo de 1,21 sigmas y no marca ni un solo paso, mientras la velocidad
estimada converge exactamente a 0,5. Por tanto el residual detecta transitorios
(escalones, picos, arranques y cambios de pendiente) y es `velocity` quien
detecta la deriva establecida. El KalmanResidualDetector de T3.4 consume las dos:
`normalized_residual` contra k_sigma, y `velocity` contra su propio umbral. Ver
tests/unit/test_kalman.py::TestGradualDrift, que fija ambos comportamientos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from data.schemas.sensor_reading import SensorReading

DEFAULT_PROCESS_NOISE = 0.1
"""Densidad espectral del ruido de aceleracion, el parametro q de Q.

Gobierna cuanto se fia el filtro del modelo cinematico frente a la medida. Subirlo
hace al filtro mas agil y menos sensible (sigue mejor los cambios reales, marca
menos anomalias); bajarlo lo hace mas rigido y mas sensible.
"""

DEFAULT_OBSERVATION_NOISE = 1.0
"""Varianza del ruido del transmisor, el escalar R.

En unidades de ingenieria al cuadrado. El valor por defecto es un marcador
razonable, no una medida: calibrarlo por canal es trabajo de T3.6.
"""

DEFAULT_K_SIGMA = 3.0
"""Residual normalizado a partir del cual una lectura se marca como anomala."""

DEFAULT_RESET_SIGMA = 10.0
"""Residual normalizado a partir del cual el filtro descarta su estado.

Sin esto, un unico valor aberrante envenena el estado y el filtro tarda decenas
de pasos en recuperarse, marcando como anomalas lecturas que son normales. Al
superar este umbral se reinicia sobre la medida actual: se pierde la historia,
que es exactamente lo que se quiere cuando la historia ya no describe la senal.
"""

INITIAL_COVARIANCE = 1000.0
"""Diagonal de P al inicializar: incertidumbre alta y deliberadamente vaga.

Hace que las primeras medidas pesen casi todo frente al estado inicial, de modo
que el filtro converja rapido en lugar de arrastrar un arranque arbitrario.
"""


@dataclass(frozen=True)
class KalmanResult:
    """Outcome of a single filter step.

    Attributes:
        predicted_value: Position the filter expected before seeing the
            measurement.
        corrected_value: Position estimate after incorporating the measurement.
        residual: measurement - predicted_value, in engineering units.
        innovation_sigma: Standard deviation of the residual under the model,
            sqrt(S). Always strictly positive.
        normalized_residual: residual / innovation_sigma, in sigmas. This is the
            dimensionless anomaly signal.
        is_anomaly: Whether abs(normalized_residual) exceeded k_sigma.
        is_initialized: Whether the filter held a usable state before this step.
            False on the first reading of a sensor and on any step that tripped
            the auto-reset, where the residual describes a discontinuity rather
            than a filtered deviation. Consumers that average residuals should
            skip those steps.
    """

    predicted_value: float
    corrected_value: float
    residual: float
    innovation_sigma: float
    normalized_residual: float
    is_anomaly: bool
    is_initialized: bool


class OnlineKalmanFilter:
    """Constant-velocity Kalman filter fed one measurement at a time.

    The filter carries its own clock: each step computes dt from the timestamp
    of the previous one, so irregular sampling is handled by construction rather
    than assumed away.
    """

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        observation_noise: float = DEFAULT_OBSERVATION_NOISE,
        k_sigma: float = DEFAULT_K_SIGMA,
        reset_sigma: float = DEFAULT_RESET_SIGMA,
    ) -> None:
        """Build an uninitialised filter.

        Args:
            process_noise: Spectral density q of the acceleration noise in Q.
                Must be non-negative.
            observation_noise: Measurement noise variance R. Must be strictly
                positive: a sensor with zero noise would make the innovation
                variance singular whenever the state is also certain.
            k_sigma: Normalized residual above which a step is flagged anomalous.
                Must be strictly positive.
            reset_sigma: Normalized residual above which the filter discards its
                state. Must be at least k_sigma, otherwise the filter would
                reset before it could ever report an anomaly.

        Raises:
            ValueError: If any argument violates the constraints above.
        """
        if process_noise < 0.0:
            raise ValueError(f"process_noise must be non-negative, got {process_noise}.")
        if observation_noise <= 0.0:
            raise ValueError(
                f"observation_noise must be strictly positive, got {observation_noise}. "
                "A noiseless sensor makes the innovation variance singular."
            )
        if k_sigma <= 0.0:
            raise ValueError(f"k_sigma must be strictly positive, got {k_sigma}.")
        if reset_sigma < k_sigma:
            raise ValueError(
                f"reset_sigma ({reset_sigma}) must be at least k_sigma ({k_sigma}): "
                "a filter that resets below its alarm threshold can never report one."
            )

        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.k_sigma = k_sigma
        self.reset_sigma = reset_sigma

        self._state: npt.NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self._covariance: npt.NDArray[np.float64] = np.eye(2, dtype=np.float64)
        self._last_timestamp: datetime | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Return whether the filter holds a usable state."""
        return self._initialized

    @property
    def value(self) -> float:
        """Return the current position estimate."""
        return float(self._state[0])

    @property
    def velocity(self) -> float:
        """Return the current velocity estimate, in units per second."""
        return float(self._state[1])

    # ------------------------------------------------------------------
    # Filter cycle
    # ------------------------------------------------------------------

    def initialize(self, initial_value: float, initial_velocity: float = 0.0) -> None:
        """Seed the state and reset the covariance to its vague prior.

        Args:
            initial_value: Starting position, normally the first measurement.
            initial_velocity: Starting velocity. Zero is the honest default: a
                single sample carries no information about a rate of change.
        """
        self._state = np.array([initial_value, initial_velocity], dtype=np.float64)
        self._covariance = np.eye(2, dtype=np.float64) * INITIAL_COVARIANCE
        self._initialized = True

    def predict(self, dt: float) -> tuple[float, float]:
        """Advance the state by dt seconds.

        Applies x <- F x and P <- F P F^T + Q, mutating the filter so that a
        following update() corrects the predicted state.

        Args:
            dt: Elapsed time in seconds. Must be non-negative. Zero is allowed
                and degenerates to the identity transition, which is the correct
                behaviour for two readings sharing a timestamp.

        Returns:
            Tuple of (predicted position, standard deviation of that position).

        Raises:
            RuntimeError: If the filter has not been initialised.
            ValueError: If dt is negative.
        """
        if not self._initialized:
            raise RuntimeError(
                "predict() called before initialize(). Use step(), which "
                "initialises on the first measurement."
            )
        if dt < 0.0:
            raise ValueError(
                f"dt must be non-negative, got {dt}. A negative dt means the "
                "readings arrived out of order; reorder them before filtering."
            )

        transition = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        process_covariance = self.process_noise * np.array(
            [[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]], dtype=np.float64
        )

        self._state = transition @ self._state
        self._covariance = (
            transition @ self._covariance @ transition.T + process_covariance
        )

        return float(self._state[0]), float(math.sqrt(self._covariance[0, 0]))

    def update(self, measurement: float) -> KalmanResult:
        """Correct the predicted state with a measurement.

        Args:
            measurement: Observed position in engineering units.

        Returns:
            The KalmanResult of this correction. is_initialized is always True
            here; the first-reading and auto-reset cases are handled by step().

        Raises:
            RuntimeError: If the filter has not been initialised.
        """
        if not self._initialized:
            raise RuntimeError(
                "update() called before initialize(). Use step(), which "
                "initialises on the first measurement."
            )

        predicted_value = float(self._state[0])
        residual = measurement - predicted_value

        innovation_variance = float(self._covariance[0, 0]) + self.observation_noise
        innovation_sigma = math.sqrt(innovation_variance)
        normalized_residual = residual / innovation_sigma

        gain = self._covariance[:, 0] / innovation_variance
        self._state = self._state + gain * residual

        identity = np.eye(2, dtype=np.float64)
        observation = np.array([[1.0, 0.0]], dtype=np.float64)
        self._covariance = (identity - np.outer(gain, observation)) @ self._covariance
        # La forma estandar de la actualizacion de covarianza pierde simetria por
        # error de redondeo a lo largo de miles de pasos, y una P asimetrica
        # puede volverse no definida positiva y producir sigmas NaN. Simetrizar
        # cuesta cuatro operaciones y elimina esa deriva.
        self._covariance = (self._covariance + self._covariance.T) / 2.0

        return KalmanResult(
            predicted_value=predicted_value,
            corrected_value=float(self._state[0]),
            residual=residual,
            innovation_sigma=innovation_sigma,
            normalized_residual=normalized_residual,
            is_anomaly=abs(normalized_residual) > self.k_sigma,
            is_initialized=True,
        )

    def step(self, measurement: float, timestamp: datetime) -> KalmanResult:
        """Run one full predict-update cycle against the filter's own clock.

        Three cases are folded in here:

        - First reading of the sensor: the filter initialises on the measurement
          and returns a zero residual, because there was nothing to predict from.
        - Normal reading: predict by the elapsed dt, then correct.
        - Divergent reading (abs(normalized_residual) > reset_sigma): the state
          is discarded and reseeded on the measurement. The step is still
          reported as an anomaly, but with is_initialized False, so a consumer
          can tell a filtered deviation from a discontinuity.

        Args:
            measurement: Observed value in engineering units.
            timestamp: Time of the measurement. Must be consistently timezone
                aware or naive across calls on the same filter.

        Returns:
            The KalmanResult of this step.

        Raises:
            ValueError: If timestamp mixes aware and naive datetimes with the
                previous one, or if it precedes the previous timestamp.
        """
        if not self._initialized or self._last_timestamp is None:
            self.initialize(measurement)
            self._last_timestamp = timestamp
            return KalmanResult(
                predicted_value=measurement,
                corrected_value=measurement,
                residual=0.0,
                innovation_sigma=math.sqrt(INITIAL_COVARIANCE + self.observation_noise),
                normalized_residual=0.0,
                is_anomaly=False,
                is_initialized=False,
            )

        dt = self._elapsed_seconds(timestamp)
        self.predict(dt)
        result = self.update(measurement)
        self._last_timestamp = timestamp

        if abs(result.normalized_residual) > self.reset_sigma:
            self.initialize(measurement)
            return KalmanResult(
                predicted_value=result.predicted_value,
                corrected_value=measurement,
                residual=result.residual,
                innovation_sigma=result.innovation_sigma,
                normalized_residual=result.normalized_residual,
                is_anomaly=True,
                is_initialized=False,
            )

        return result

    def reset(self) -> None:
        """Discard the state entirely, including the clock.

        The next step() behaves as the first reading of a fresh sensor.
        """
        self._state = np.zeros(2, dtype=np.float64)
        self._covariance = np.eye(2, dtype=np.float64)
        self._last_timestamp = None
        self._initialized = False

    def get_state(self) -> dict[str, Any]:
        """Return the filter state in JSON-serialisable form.

        Returns:
            Mapping with the state vector, the covariance as nested lists, the
            last timestamp as an ISO 8601 string (or None) and the
            initialisation flag.
        """
        return {
            "state": self._state.tolist(),
            "covariance": self._covariance.tolist(),
            "last_timestamp": (
                self._last_timestamp.isoformat() if self._last_timestamp else None
            ),
            "is_initialized": self._initialized,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _elapsed_seconds(self, timestamp: datetime) -> float:
        """Return the seconds elapsed since the previous step.

        Args:
            timestamp: Time of the incoming measurement.

        Returns:
            Non-negative elapsed time in seconds.

        Raises:
            ValueError: If timestamp mixes awareness with the stored one, or if
                it precedes it.
        """
        previous = self._last_timestamp
        assert previous is not None  # noqa: S101 - guarded by the caller

        if (previous.tzinfo is None) != (timestamp.tzinfo is None):
            raise ValueError(
                "Cannot mix timezone-aware and naive timestamps on the same "
                f"filter: previous={previous!r}, incoming={timestamp!r}."
            )

        dt = (timestamp - previous).total_seconds()
        if dt < 0.0:
            raise ValueError(
                f"Timestamp {timestamp!r} precedes the previous one {previous!r}. "
                "Readings must be filtered in chronological order."
            )
        return dt


class KalmanFilterBank:
    """One OnlineKalmanFilter per sensor tag, created on first sight.

    A plant has as many signals as tags, and they are neither synchronised nor
    comparable, so they cannot share a filter. The bank creates filters lazily
    rather than requiring the caller to enumerate the tag inventory up front,
    which is what keeps it usable against a stream whose schema can grow.
    """

    def __init__(
        self,
        process_noise: float = DEFAULT_PROCESS_NOISE,
        observation_noise: float = DEFAULT_OBSERVATION_NOISE,
        k_sigma: float = DEFAULT_K_SIGMA,
        reset_sigma: float = DEFAULT_RESET_SIGMA,
    ) -> None:
        """Build an empty bank.

        Args:
            process_noise: Passed to every filter the bank creates.
            observation_noise: Passed to every filter the bank creates.
            k_sigma: Passed to every filter the bank creates.
            reset_sigma: Passed to every filter the bank creates.

        Raises:
            ValueError: If any argument is rejected by OnlineKalmanFilter.
        """
        # Construir un filtro descartable valida los parametros aqui y no en el
        # primer reading, donde el fallo llegaria a mitad de un stream.
        OnlineKalmanFilter(process_noise, observation_noise, k_sigma, reset_sigma)

        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.k_sigma = k_sigma
        self.reset_sigma = reset_sigma
        self._filters: dict[str, OnlineKalmanFilter] = {}

    def __len__(self) -> int:
        """Return the number of sensors the bank currently tracks."""
        return len(self._filters)

    def __contains__(self, sensor_id: str) -> bool:
        """Return whether a filter already exists for a sensor tag."""
        return sensor_id in self._filters

    def get_filter(self, sensor_id: str) -> OnlineKalmanFilter:
        """Return the filter for a tag, creating it if this is its first reading.

        Args:
            sensor_id: Instrument tag.

        Returns:
            The filter owned by the bank for that tag.
        """
        if sensor_id not in self._filters:
            self._filters[sensor_id] = OnlineKalmanFilter(
                process_noise=self.process_noise,
                observation_noise=self.observation_noise,
                k_sigma=self.k_sigma,
                reset_sigma=self.reset_sigma,
            )
        return self._filters[sensor_id]

    def process(self, reading: SensorReading) -> KalmanResult:
        """Filter one reading through its sensor's filter.

        Args:
            reading: Canonical sensor reading.

        Returns:
            The KalmanResult of that sensor's step.

        Raises:
            ValueError: If the reading carries no value. A Kalman filter needs a
                number; absence of one is not a small residual, it is no
                observation at all, and silently substituting a value would
                fabricate evidence. Note this is a check on the value itself,
                not on the quality flag: no detector reads quality.
        """
        value = reading.measurement.value
        if value is None:
            raise ValueError(
                f"Reading {reading.reading_id} for sensor {reading.sensor.id} has "
                "no value. Filter callers must drop valueless readings rather "
                "than impute them."
            )
        return self.get_filter(reading.sensor.id).step(value, reading.timestamp)

    def reset_sensor(self, sensor_id: str) -> None:
        """Discard the state of one sensor's filter.

        Args:
            sensor_id: Instrument tag. Unknown tags are a no-op: there is no
                state to discard, which is the requested end state anyway.
        """
        if sensor_id in self._filters:
            self._filters[sensor_id].reset()

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Return every filter state in JSON-serialisable form.

        Returns:
            Mapping of sensor_id to the mapping produced by
            OnlineKalmanFilter.get_state.
        """
        return {
            sensor_id: filter_.get_state()
            for sensor_id, filter_ in self._filters.items()
        }
