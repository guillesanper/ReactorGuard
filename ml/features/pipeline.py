"""Feature engineering over a wide (timestep x sensor) frame, per TDD 5.5.

Sustituye a ml/features/extractor.py, que quedo superado: hardcodeaba ocho
canales de un schema plano que ya no existe, ignoraba params.yaml y usaba la API
fillna(method=) eliminada en pandas 2.x, de modo que no era ejecutable tal cual.

TRES DESVIACIONES DEL TDD 5.5, cada una con su motivo medido.

1. LAS VENTANAS VAN EN MUESTRAS, no en segundos. El TDD pedia 60 s, 5 min y 1 h
   asumiendo el sondeo SCADA de 1 s; a los 180 s del TEP la de 60 s no contiene
   ni una muestra. Ver la cabecera de ml/features/feature_params.py.

2. LA SALIDA ES LARGA, una fila por (timestep, sensor_id), y los nombres de las
   features NO llevan el tag dentro. El TDD los nombraba rolling_mean_{sensor},
   que en formato ancho da 52 columnas por feature y obliga a reescribir el
   modelo cada vez que la planta cambia de instrumentacion. En formato largo la
   fila identifica al sensor, el numero de columnas no depende del inventario, y
   ademas es la forma que Feast espera: su sensor_entity tiene join_key
   sensor_id, o sea una fila por sensor.

3. EL INDICADOR DE STUCK ES LA LONGITUD DE TIRADA, no la varianza movil. Es la
   misma decision que en StuckValueDetector y por el mismo motivo: la varianza se
   mide en unidades de ingenieria al cuadrado y no es comparable entre un caudal
   en kg/s y una composicion en mol%, mientras que la longitud de tirada es
   adimensional. Ademas es estrictamente mas informativa para detectar un
   congelamiento: distingue ocho muestras identicas de ocho muestras casi iguales,
   que es justo la distincion que la varianza pierde.

El pipeline NO llama al validador. featurize consume el parquet adaptado tal
cual. Acoplarlos haria que recalibrar un umbral del validador invalidase todas
las features, y quien escribe `quality` es el consumidor de T3.5, sobre el
stream, no este stage por lotes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pandas as pd

from ml.features.feature_params import FeatureParams
from ml.features.kalman import (
    DEFAULT_OBSERVATION_NOISE,
    OnlineKalmanFilter,
)

FEATURIZER_PROCESS_NOISE = 1e-6
"""Densidad del ruido de aceleracion q para las features de Kalman.

El mismo valor que usa el KalmanResidualDetector y por la misma razon: el 0.1
por defecto de OnlineKalmanFilter esta calibrado para dt de 1 s, y como Q[0,0]
escala con dt^3, a los 180 s del TEP la sigma de innovacion se dispara y el
residual normalizado deja de discriminar. Compartir el valor con el detector no
es casualidad: si el feature kalman_residual y el detector se calculasen con
filtros distintos, el modelo de Fase 4 aprenderia sobre una senal que el
detector nunca ve.
"""

_POSITION_TYPE = "position"
"""Tipo de sensor cuyo promedio define power_level_pct (las XMV del TEP)."""


@lru_cache(maxsize=64)
def _slope_coefficients(window: int) -> npt.NDArray[np.float64]:
    """Return the fixed weights whose dot product is a least-squares slope.

    Para una ventana de posiciones x = 0..w-1, la pendiente de minimos cuadrados
    es sum((x - xbar) * y) / sum((x - xbar)^2), que es LINEAL en y: un producto
    escalar con pesos que solo dependen de w. Precalcularlos evita ajustar una
    regresion por cada posicion de la ventana.

    Se cachean por longitud porque al principio de la serie pandas entrega
    ventanas mas cortas que la configurada, y truncar los pesos de la ventana
    larga daria una regresion mal centrada en lugar de la de ese bloque.

    Args:
        window: Block length in samples. Must be at least 2.

    Returns:
        The weights, of length window.
    """
    positions = np.arange(window, dtype=np.float64)
    centred = positions - positions.mean()
    return np.asarray(centred / np.square(centred).sum())


def _least_squares_slope(block: npt.NDArray[np.float64]) -> float:
    """Return the least-squares slope of a window, in units per sample.

    Args:
        block: Window values in sample order.

    Returns:
        The slope, 0.0 for a block too short to define one.
    """
    if block.size < 2:
        return 0.0
    return float(block @ _slope_coefficients(block.size))


class FeaturePipeline:
    """Turns a wide value frame into the long feature table of TDD 5.5.

    Attributes:
        params: Resolved feature configuration.
        sample_interval_seconds: Measured cadence, used to express every rate in
            units per second regardless of how often the source samples.
    """

    def __init__(
        self,
        params: FeatureParams,
        sample_interval_seconds: float,
        process_noise: float = FEATURIZER_PROCESS_NOISE,
        observation_noise: float = DEFAULT_OBSERVATION_NOISE,
    ) -> None:
        """Build the pipeline.

        Args:
            params: Resolved features: section of params.yaml.
            sample_interval_seconds: Spacing between consecutive samples. Must be
                strictly positive.
            process_noise: Q spectral density of the Kalman features.
            observation_noise: R measurement variance of the Kalman features.

        Raises:
            ValueError: If the sample interval is not strictly positive.
        """
        if sample_interval_seconds <= 0.0:
            raise ValueError(
                "sample_interval_seconds must be strictly positive, got "
                f"{sample_interval_seconds}. Toda feature de tasa se divide por el."
            )
        self.params = params
        self.sample_interval_seconds = sample_interval_seconds
        self.process_noise = process_noise
        self.observation_noise = observation_noise

    # ------------------------------------------------------------------
    # Sensor selection
    # ------------------------------------------------------------------

    def resolve_sensors(self, sensor_types: Mapping[str, str]) -> list[str]:
        """Resolve the sensor set from the data and check it against the contract.

        No enumera tags: los descubre. Lo que se declara en params.yaml es cuantos
        deberia haber, de modo que perder un transmisor o ganar uno falla en voz
        alta en lugar de propagarse en silencio a las features y de ahi al modelo.

        Args:
            sensor_types: Sensor tag to its type, as present in the data.

        Returns:
            The selected tags, sorted.

        Raises:
            ValueError: If the resolved set does not hold expected_sensor_count
                sensors.
        """
        selection = self.params.selection
        resolved = sorted(
            tag
            for tag, sensor_type in sensor_types.items()
            if sensor_type in selection.include_types
            and tag not in selection.exclude_sensor_ids
        )

        if len(resolved) != selection.expected_sensor_count:
            dropped = sorted(set(sensor_types) - set(resolved))
            raise ValueError(
                f"Sensor selection resolved {len(resolved)} sensors but "
                f"features.sensor_selection.expected_sensor_count declares "
                f"{selection.expected_sensor_count}. Los datos traen "
                f"{len(sensor_types)} tags y la politica descarto {len(dropped)}"
                f"{': ' + ', '.join(dropped[:5]) if dropped else ''}. Es una deriva "
                "de schema: actualiza expected_sensor_count si es intencionada."
            )
        return resolved

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def feature_names(self) -> list[str]:
        """Return every feature column, in the order transform emits them.

        Returns:
            The column names, excluding the timestep and sensor_id keys.
        """
        names = ["value"]
        for window in self.params.window_samples:
            names += [
                f"rolling_mean_{window}",
                f"rolling_std_{window}",
                f"rolling_min_{window}",
                f"rolling_max_{window}",
            ]
        names += [f"lag_{lag}" for lag in range(1, self.params.n_lags + 1)]
        names += [
            "rate_of_change",
            f"slope_{self.params.shortest_window}",
            "kalman_residual",
            "kalman_uncertainty",
            "variance_ratio",
            f"zero_crossing_rate_{self.params.shortest_window}",
            "stuck_run_length",
            "hour_of_day",
            "day_of_week",
            "power_level_pct",
        ]
        names += [
            f"cross_correlation_{first}__{second}"
            for first, second in self.params.correlation_pairs
        ]
        return names

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        values: pd.DataFrame,
        timestamps: pd.Series,
        sensor_types: Mapping[str, str],
    ) -> pd.DataFrame:
        """Compute every feature group over one contiguous run of samples.

        La entrada debe ser UNA tirada continua de un mismo origen. Las ventanas
        moviles y los lags miran hacia atras, de modo que concatenar dos ficheros
        del TEP y transformarlos juntos haria que las primeras muestras del
        segundo heredaran historia del primero, que es una fuga de informacion
        entre clases. batch_featurizer respeta esa frontera transformando cada
        particion fault_type por separado.

        Args:
            values: Wide frame indexed by timestep, one column per sensor tag.
            timestamps: Timestamp of every timestep, sharing the values index.
            sensor_types: Sensor tag to its type, for the selection check and for
                resolving which tags define power_level_pct.

        Returns:
            Long frame with timestep, sensor_id and one column per feature,
            ordered by (timestep, sensor_id).

        Raises:
            ValueError: If the frames do not align, or if the sensor selection
                does not match the declared count.
        """
        if not values.index.equals(timestamps.index):
            raise ValueError(
                "values and timestamps must share the same timestep index; got "
                f"{len(values.index)} and {len(timestamps.index)} entries."
            )
        if values.empty:
            raise ValueError("Cannot featurise an empty frame.")

        selected = self.resolve_sensors(sensor_types)
        frame = values.loc[:, selected].astype(np.float64)

        columns: dict[str, npt.NDArray[np.float64]] = {"value": self._flat(frame)}
        columns.update(self._rolling_statistics(frame))
        columns.update(self._lags(frame))
        columns.update(self._rates(frame))
        columns.update(self._kalman(frame, timestamps))
        columns.update(self._distribution(frame))
        columns.update(self._fault_indicators(frame))
        columns.update(self._context(frame, timestamps, sensor_types, selected))
        columns.update(self._correlations(frame))

        index = pd.MultiIndex.from_product(
            [frame.index, frame.columns], names=["timestep", "sensor_id"]
        )
        long = pd.DataFrame(columns, index=index).reset_index()
        return long.loc[:, ["timestep", "sensor_id", *self.feature_names()]]

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    @staticmethod
    def _flat(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Flatten a wide frame in (timestep, sensor) row-major order.

        Es el mismo orden que produce MultiIndex.from_product sobre (index,
        columns), que es lo que alinea cada columna de features con su fila.

        Args:
            frame: Wide frame.

        Returns:
            The values as a 1-D array.
        """
        return np.asarray(frame.to_numpy(dtype=np.float64).reshape(-1))

    def _rolling_statistics(
        self, frame: pd.DataFrame
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 1: rolling mean, std, min and max over each configured window.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        out: dict[str, npt.NDArray[np.float64]] = {}
        for window in self.params.window_samples:
            rolling = frame.rolling(
                window=window, min_periods=self.params.min_samples_per_window
            )
            out[f"rolling_mean_{window}"] = self._flat(rolling.mean())
            out[f"rolling_std_{window}"] = self._flat(rolling.std())
            out[f"rolling_min_{window}"] = self._flat(rolling.min())
            out[f"rolling_max_{window}"] = self._flat(rolling.max())
        return out

    def _lags(self, frame: pd.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
        """Lagged copies of the value.

        Los primeros lags salen nulos y se dejan nulos. extractor.py los rellenaba
        con fillna(method="bfill"), que copia hacia atras un valor del FUTURO: en
        una serie temporal eso es fuga de informacion, y ademas la API que usaba
        desaparecio en pandas 2.x.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        return {
            f"lag_{lag}": self._flat(frame.shift(lag))
            for lag in range(1, self.params.n_lags + 1)
        }

    def _rates(self, frame: pd.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
        """Group 2: first difference and least-squares slope, both per second.

        Se expresan en unidades por SEGUNDO y no por muestra para que el numero
        signifique lo mismo si algun dia entra una fuente a otra cadencia.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        window = self.params.shortest_window
        slope = frame.rolling(
            window=window, min_periods=self.params.min_samples_per_window
        ).apply(_least_squares_slope, raw=True)

        return {
            "rate_of_change": self._flat(frame.diff() / self.sample_interval_seconds),
            f"slope_{window}": self._flat(slope / self.sample_interval_seconds),
        }

    def _kalman(
        self, frame: pd.DataFrame, timestamps: pd.Series
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 3: normalized residual and innovation sigma, one filter per tag.

        Se filtra columna a columna con OnlineKalmanFilter en lugar de con
        KalmanFilterBank porque el banco consume SensorReading y aqui se parte de
        un frame: construir 26.000 objetos pydantic por particion solo para
        volver a extraerles el numero seria trabajo puro de traduccion.

        Args:
            frame: Wide value frame.
            timestamps: Timestamp of every timestep.

        Returns:
            Mapping from feature name to its flattened column.
        """
        stamps = [pd.Timestamp(value).to_pydatetime() for value in timestamps]
        residual = np.empty(frame.shape, dtype=np.float64)
        sigma = np.empty(frame.shape, dtype=np.float64)

        for position, tag in enumerate(frame.columns):
            series = frame[tag].to_numpy(dtype=np.float64)
            filtered = OnlineKalmanFilter(
                process_noise=self.process_noise,
                observation_noise=self.observation_noise,
            )
            for step, (measurement, stamp) in enumerate(
                zip(series, stamps, strict=True)
            ):
                residual[step, position], sigma[step, position] = self._filter_step(
                    filtered, measurement, stamp
                )

        return {
            "kalman_residual": np.asarray(residual.reshape(-1)),
            "kalman_uncertainty": np.asarray(sigma.reshape(-1)),
        }

    @staticmethod
    def _filter_step(
        filtered: OnlineKalmanFilter, measurement: float, stamp: datetime
    ) -> tuple[float, float]:
        """Advance one filter by one sample.

        Una lectura ausente no se imputa: se pasa como nula y el filtro no avanza.
        Inventar el numero fabricaria justo la evidencia que el residual mide.

        Args:
            filtered: The sensor's filter.
            measurement: Measured value, possibly NaN.
            stamp: Timestamp of the sample.

        Returns:
            The normalized residual and the innovation sigma, both NaN when the
            measurement is absent.
        """
        if not np.isfinite(measurement):
            return float("nan"), float("nan")
        result = filtered.step(float(measurement), stamp)
        return result.normalized_residual, result.innovation_sigma

    def _distribution(self, frame: pd.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
        """Group 5: short-to-long volatility ratio and zero-crossing rate.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        short = self.params.shortest_window
        long = self.params.longest_window
        minimum = self.params.min_samples_per_window

        short_std = frame.rolling(window=short, min_periods=minimum).std()
        long_std = frame.rolling(window=long, min_periods=minimum).std()
        # Un canal plano en la ventana larga tiene volatilidad indefinida, no
        # infinita: dividir por cero daria inf y envenenaria cualquier escalado.
        ratio = short_std / long_std.replace(0.0, np.nan)

        crossings = frame.rolling(window=short, min_periods=minimum).apply(
            self._zero_crossing_rate, raw=True
        )

        return {
            "variance_ratio": self._flat(ratio),
            f"zero_crossing_rate_{short}": self._flat(crossings),
        }

    @staticmethod
    def _zero_crossing_rate(block: npt.NDArray[np.float64]) -> float:
        """Return how often a window crosses its own mean, normalised to [0, 1].

        Se cuenta contra la MEDIA de la ventana y no contra el cero absoluto: las
        senales del TEP viven lejos del cero (una presion de 2.700 kPa no cruza
        el cero jamas) y contra el cero la feature seria constante.

        Args:
            block: Window values.

        Returns:
            Crossings divided by the transitions available, 0.0 for a flat window.
        """
        centred = block - block.mean()
        signs = np.sign(centred)
        non_zero = signs[signs != 0.0]
        if non_zero.size < 2:
            return 0.0
        return float(np.count_nonzero(np.diff(non_zero)) / (non_zero.size - 1))

    def _fault_indicators(
        self, frame: pd.DataFrame
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 6: length of the current run of identical readings.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        runs = np.column_stack(
            [
                self._run_lengths(frame[tag].to_numpy(dtype=np.float64))
                for tag in frame.columns
            ]
        )
        return {"stuck_run_length": np.asarray(runs.reshape(-1))}

    @staticmethod
    def _run_lengths(series: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return, per sample, how long the current run of equal values is.

        Args:
            series: One sensor's values in sample order.

        Returns:
            Run length at each position, starting at 1.
        """
        repeated = np.zeros(series.size, dtype=bool)
        repeated[1:] = series[1:] == series[:-1]

        positions = np.arange(series.size)
        # El ultimo indice en que la tirada se rompio, propagado hacia delante.
        last_break = np.maximum.accumulate(np.where(~repeated, positions, 0))
        return np.asarray((positions - last_break + 1).astype(np.float64))

    def _context(
        self,
        frame: pd.DataFrame,
        timestamps: pd.Series,
        sensor_types: Mapping[str, str],
        selected: list[str],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 7: operational context, identical for every sensor of a timestep.

        Args:
            frame: Wide value frame.
            timestamps: Timestamp of every timestep.
            sensor_types: Sensor tag to its type.
            selected: The resolved sensor set.

        Returns:
            Mapping from feature name to its flattened column.
        """
        # DatetimeIndex y no Series.to_numpy(): numpy no tiene tipo con zona
        # horaria, asi que convertir por ahi la descarta en silencio y hour_of_day
        # saldria desplazado para cualquier fuente que no publique en UTC.
        moments = pd.DatetimeIndex(timestamps)
        positions = [tag for tag in selected if sensor_types[tag] == _POSITION_TYPE]
        power = (
            frame.loc[:, positions].mean(axis=1)
            if positions
            else pd.Series(np.nan, index=frame.index)
        )

        width = frame.shape[1]
        return {
            "hour_of_day": np.repeat(
                np.asarray(moments.hour, dtype=np.float64), width
            ),
            "day_of_week": np.repeat(
                np.asarray(moments.dayofweek, dtype=np.float64), width
            ),
            "power_level_pct": np.repeat(power.to_numpy(np.float64), width),
        }

    def _correlations(self, frame: pd.DataFrame) -> dict[str, npt.NDArray[np.float64]]:
        """Group 4: rolling Pearson coefficient of the configured pairs.

        Con correlation_pairs vacio, que es el estado por defecto, no emite nada.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.

        Raises:
            KeyError: If a configured pair names a sensor the selection dropped.
        """
        window = self.params.shortest_window
        width = frame.shape[1]
        out: dict[str, npt.NDArray[np.float64]] = {}

        for first, second in self.params.correlation_pairs:
            missing = [tag for tag in (first, second) if tag not in frame.columns]
            if missing:
                raise KeyError(
                    f"features.correlation_pairs names {missing}, absent from the "
                    "resolved sensor set. Corrige la pareja o la politica de "
                    "seleccion."
                )
            coefficient = (
                frame[first]
                .rolling(window=window, min_periods=self.params.min_samples_per_window)
                .corr(frame[second])
            )
            out[f"cross_correlation_{first}__{second}"] = np.repeat(
                coefficient.to_numpy(np.float64), width
            )
        return out
