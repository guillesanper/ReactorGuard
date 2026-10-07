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

from collections.abc import Callable, Mapping
from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pandas.api.indexers import BaseIndexer

from ml.features.feature_params import FeatureParams
from ml.features.kalman import (
    DEFAULT_OBSERVATION_NOISE,
    filter_columns,
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


@lru_cache(maxsize=64)
def _trailing_bounds(
    window: int, steps: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the [start, end) bounds of every trailing window of a series.

    Son las mismas cotas que calcula pandas para una ventana entera (cierre por
    la derecha, sin centrar): la ventana de la posicion i abarca las ultimas
    window muestras hasta i incluida, mas corta al principio.

    Args:
        window: Window length in samples.
        steps: Series length.

    Returns:
        The start and end positions of each window.
    """
    end = np.arange(1, steps + 1, dtype=np.int64)
    return np.maximum(end - window, 0), end


class _TrailingWindow(BaseIndexer):  # type: ignore[misc]  # pandas no publica stubs
    """Window indexer that hands pandas precomputed bounds.

    Para una ventana entera pandas recalcula las cotas (y recorta con np.clip)
    una vez por COLUMNA y por llamada: con 52 sensores, tres ventanas y dos
    estadisticas son 312 recalculos identicos por ventana de datos, casi la mitad
    del coste de rolling.mean y rolling.std. Las cotas solo dependen de la
    longitud de la ventana y de la serie, de modo que se calculan una vez. Es la
    API publica de pandas para ventanas personalizadas y los nucleos que se
    ejecutan son los mismos, con lo que el resultado no cambia.
    """

    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the cached bounds of a series of num_values samples.

        Args:
            num_values: Series length.
            min_periods: Ignored; pandas applies it itself.
            center: Ignored; windows are always trailing.
            closed: Ignored; windows are always closed on the right.
            step: Ignored; every position is evaluated.

        Returns:
            The start and end positions of each window.
        """
        return _trailing_bounds(self.window_size, num_values)


def _window_counts(present: npt.NDArray[np.bool_], window: int) -> npt.NDArray[np.int64]:
    """Count the observations inside the trailing window of every position.

    Args:
        present: Boolean array (sensors, steps), True where a value exists.
        window: Window length in samples.

    Returns:
        Array (sensors, steps) with how many values each trailing window holds.
        Las primeras posiciones tienen ventanas mas cortas que window.
    """
    cumulative = np.zeros((present.shape[0], present.shape[1] + 1), dtype=np.int64)
    np.cumsum(present, axis=1, out=cumulative[:, 1:])
    ends = np.arange(1, present.shape[1] + 1)
    return np.asarray(cumulative[:, ends] - cumulative[:, np.maximum(ends - window, 0)])


def _rolling_reduce(
    values: npt.NDArray[np.float64],
    window: int,
    min_periods: int,
    reduce: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Apply a window reducer along time for every column, without a Python loop.

    Sustituye a DataFrame.rolling(...).apply(func, raw=True), que llamaba a func
    una vez por ventana y por sensor (3.120 llamadas en una ventana de 60x52). Aqui
    las ventanas completas son una vista sin copia (sliding_window_view) y se
    reducen todas de una vez; solo las window - 1 primeras posiciones, que tienen
    ventanas mas cortas, se reducen una a una. La semantica es la de pandas: el
    bloque que recibe el reductor INCLUYE los NaN, y la celda sale NaN si la
    ventana tiene menos de min_periods valores no nulos.

    Args:
        values: Array (steps, sensors) in time order. Infinities count as absent.
        window: Window length in samples.
        min_periods: Minimum non-NaN values for a cell to be emitted.
        reduce: Maps an array (sensors, windows, length) of windows to an array
            (sensors, windows). Debe tratar cada ventana de forma independiente.

    Returns:
        Array (steps, sensors) with the reduction of each trailing window.
    """
    series = _window_series(values)
    steps = series.shape[1]
    out = np.full(series.shape, np.nan, dtype=np.float64)
    for end in range(min(window - 1, steps)):
        out[:, end] = reduce(series[:, None, : end + 1])[:, 0]
    if steps >= window:
        out[:, window - 1 :] = reduce(sliding_window_view(series, window, axis=1))
    out[_window_counts(~np.isnan(series), window) < min_periods] = np.nan
    return np.ascontiguousarray(out.T)


def _window_series(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Lay values out the way a window operation consumes them.

    Contiguo por sensor: cada ventana es un tramo contiguo de memoria, igual que
    el bloque que pandas entregaba a func, y de eso depende el orden de suma de
    numpy y por tanto que el resultado coincida bit a bit con el anterior. pandas
    trata +-inf como ausente en toda operacion de ventana (los cambia a NaN antes
    de contar y de operar); se replica para no divergir.

    Args:
        values: Array (steps, sensors) in time order.

    Returns:
        Contiguous array (sensors, steps) with infinities replaced by NaN.
    """
    return np.ascontiguousarray(np.where(np.isinf(values.T), np.nan, values.T))


def _rolling_extreme(
    values: npt.NDArray[np.float64],
    window: int,
    min_periods: int,
    ufunc: np.ufunc,
) -> npt.NDArray[np.float64]:
    """Return the rolling minimum or maximum of every column.

    Minimo y maximo no redondean: el resultado es uno de los valores de la
    ventana, asi que cualquier algoritmo correcto da el mismo numero que
    DataFrame.rolling().min() y .max() sin tener que replicar su cola monotona.
    fmin y fmax ignoran los NaN como pandas. Las ventanas parciales del principio
    son extremos acumulados, y se calculan de una vez.

    Args:
        values: Array (steps, sensors) in time order. Infinities count as absent.
        window: Window length in samples.
        min_periods: Minimum non-NaN values for a cell to be emitted.
        ufunc: np.fmin for the minimum or np.fmax for the maximum.

    Returns:
        Array (steps, sensors) with the extreme of each trailing window.
    """
    series = _window_series(values)
    steps = series.shape[1]
    out = np.full(series.shape, np.nan, dtype=np.float64)
    lead = min(window - 1, steps)
    out[:, :lead] = ufunc.accumulate(series[:, :lead], axis=1)
    if steps >= window:
        out[:, window - 1 :] = ufunc.reduce(
            sliding_window_view(series, window, axis=1), axis=-1
        )
    out[_window_counts(~np.isnan(series), window) < min_periods] = np.nan
    return np.ascontiguousarray(out.T)


def _slope_of_windows(windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the least-squares slope of every window, in units per sample.

    Se hace con matmul por lotes y no con windows @ pesos porque solo el primero
    llama al producto escalar (ddot) que usaba el codigo anterior sobre cada
    bloque; el segundo usa gemv, que suma en otro orden y difiere en el ultimo bit.

    Args:
        windows: Array (sensors, count, length) of windows in sample order.

    Returns:
        Array (sensors, count) of slopes, 0.0 for windows too short to define one.
    """
    length = windows.shape[-1]
    if length < 2:
        return np.zeros(windows.shape[:-1], dtype=np.float64)
    weights = _slope_coefficients(length)[:, None]
    return np.asarray((windows[:, :, None, :] @ weights)[:, :, 0, 0])


def _zero_crossing_of_windows(windows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return how often each window crosses its own mean, normalised to [0, 1].

    Se cuenta contra la MEDIA de la ventana y no contra el cero absoluto: las
    senales del TEP viven lejos del cero (una presion de 2.700 kPa no cruza el
    cero jamas) y contra el cero la feature seria constante.

    Cada cruce es un signo distinto del ultimo signo no nulo anterior dentro de la
    ventana. La media se calcula sobre el eje contiguo para que numpy use la
    misma suma por pares que usaba block.mean(): un canal congelado depende de
    ello, porque la media de diez 0,1 identicos puede no ser exactamente 0,1 y
    entonces el signo de (valor - media) deja de ser cero.

    Args:
        windows: Array (sensors, count, length) of windows in sample order.

    Returns:
        Array (sensors, count) of crossings divided by the transitions available,
        0.0 for a flat window.
    """
    length = windows.shape[-1]
    signs = np.sign(windows - windows.mean(axis=-1, keepdims=True))
    non_zero = signs != 0.0
    # Ultimo indice con signo no nulo hasta cada posicion, y el de la posicion previa.
    last = np.maximum.accumulate(np.where(non_zero, np.arange(length), -1), axis=-1)
    previous = np.concatenate(
        [np.full(last.shape[:-1] + (1,), -1, dtype=last.dtype), last[..., :-1]], axis=-1
    )
    previous_sign = np.take_along_axis(signs, np.maximum(previous, 0), axis=-1)
    crossings = np.count_nonzero(
        non_zero & (previous >= 0) & (signs != previous_sign), axis=-1
    )
    transitions = np.count_nonzero(non_zero, axis=-1) - 1
    return np.asarray(
        np.where(transitions >= 1, crossings / np.maximum(transitions, 1), 0.0)
    )


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
        statistics = self._rolling_statistics(frame)
        columns.update(statistics)
        columns.update(self._lags(frame))
        columns.update(self._rates(frame))
        columns.update(self._kalman(frame, timestamps))
        columns.update(self._distribution(frame, statistics))
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

        La media y la desviacion se dejan a pandas: su algoritmo es incremental
        (suma compensada que entra y sale), de modo que el ultimo bit de cada
        celda depende de la historia de la ventana y solo su propio nucleo lo
        reproduce; se le dan las cotas ya calculadas (_TrailingWindow). El minimo
        y el maximo, que no redondean, se calculan con numpy.

        Args:
            frame: Wide value frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        minimum = self.params.min_samples_per_window
        values = frame.to_numpy(dtype=np.float64)
        out: dict[str, npt.NDArray[np.float64]] = {}
        for window in self.params.window_samples:
            rolling = frame.rolling(
                window=_TrailingWindow(window_size=window), min_periods=minimum
            )
            out[f"rolling_mean_{window}"] = self._flat(rolling.mean())
            out[f"rolling_std_{window}"] = self._flat(rolling.std())
            out[f"rolling_min_{window}"] = np.asarray(
                _rolling_extreme(values, window, minimum, np.fmin).reshape(-1)
            )
            out[f"rolling_max_{window}"] = np.asarray(
                _rolling_extreme(values, window, minimum, np.fmax).reshape(-1)
            )
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
        slope = _rolling_reduce(
            frame.to_numpy(dtype=np.float64),
            window,
            self.params.min_samples_per_window,
            _slope_of_windows,
        )

        return {
            "rate_of_change": self._flat(frame.diff() / self.sample_interval_seconds),
            f"slope_{window}": np.asarray(
                (slope / self.sample_interval_seconds).reshape(-1)
            ),
        }

    def _kalman(
        self, frame: pd.DataFrame, timestamps: pd.Series
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 3: normalized residual and innovation sigma, one filter per tag.

        Se filtran los sensores a la vez con filter_columns, que es el mismo
        filtro que OnlineKalmanFilter con salida identica bit a bit, y no con
        KalmanFilterBank porque el banco consume SensorReading y aqui se parte de
        un frame. El bucle sensor a sensor (24.960 llamadas a step() por ventana
        de 60x52) era mas de la mitad de los 230 ms medidos por transform().

        Una lectura ausente no se imputa: el filtro no avanza. Inventar el numero
        fabricaria justo la evidencia que el residual mide.

        Args:
            frame: Wide value frame.
            timestamps: Timestamp of every timestep.

        Returns:
            Mapping from feature name to its flattened column.

        Raises:
            ValueError: If a timestamp is missing or precedes the previous one.
        """
        moments = pd.DatetimeIndex(timestamps)
        if moments.hasnans:
            raise ValueError("timestamps must not contain missing values (NaT).")
        # asi8 es el instante UTC aunque haya huso, y en microsegundos reproduce
        # exactamente timedelta.total_seconds() del camino anterior.
        microseconds = np.asarray(moments.as_unit("us").asi8, dtype=np.int64)

        filtered = filter_columns(
            frame.to_numpy(dtype=np.float64),
            microseconds,
            process_noise=self.process_noise,
            observation_noise=self.observation_noise,
        )
        return {
            "kalman_residual": np.asarray(filtered.normalized_residual.reshape(-1)),
            "kalman_uncertainty": np.asarray(filtered.innovation_sigma.reshape(-1)),
        }

    def _distribution(
        self,
        frame: pd.DataFrame,
        statistics: Mapping[str, npt.NDArray[np.float64]],
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Group 5: short-to-long volatility ratio and zero-crossing rate.

        La razon reutiliza las desviaciones rolling_std_{corta} y rolling_std_{larga}
        del grupo 1 en lugar de recalcularlas: son la misma llamada de pandas con
        los mismos argumentos (la ventana corta y la larga siempre estan entre
        window_samples), asi que el resultado es el mismo y se ahorra casi una
        cuarta parte del tiempo de transform().

        Args:
            frame: Wide value frame.
            statistics: Output of _rolling_statistics over the same frame.

        Returns:
            Mapping from feature name to its flattened column.
        """
        short = self.params.shortest_window
        long = self.params.longest_window

        short_std = statistics[f"rolling_std_{short}"]
        long_std = statistics[f"rolling_std_{long}"]
        # Un canal plano en la ventana larga tiene volatilidad indefinida, no
        # infinita: dividir por cero daria inf y envenenaria cualquier escalado.
        ratio = short_std / np.where(long_std == 0.0, np.nan, long_std)

        crossings = _rolling_reduce(
            frame.to_numpy(dtype=np.float64),
            short,
            self.params.min_samples_per_window,
            _zero_crossing_of_windows,
        )

        return {
            "variance_ratio": np.asarray(ratio),
            f"zero_crossing_rate_{short}": np.asarray(crossings.reshape(-1)),
        }

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
