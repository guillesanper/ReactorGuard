"""Deterministic injection of synthetic instrument faults into an adapted TEP frame.

Existe porque el TEP no trae verdad-terreno de fallo de INSTRUMENTO. Sus 21
ficheros de fallo describen perturbaciones del PROCESO, que es un eje ortogonal
al que valida data/validation/sensor_validator.py. Medido sobre el dataset, el
unico transmisor realmente congelado es TEP-XMV-04 en d21: 480 pares
(sensor_id, timestep) positivos frente a 550.160 totales, el 0,087%. Un gate de
precision construido sobre esa unica clase positiva mide sobre todo el ruido del
muestreo. De ahi la inyeccion: se parte de d00 (operacion normal), se degradan
sensores y ventanas CONOCIDOS, y el resultado es una verdad-terreno de miles de
pares con la que la matriz de confusion significa algo.

Separacion deliberada entre lo que se HACE a la senal y lo que un detector
deberia LLAMARLO. `InjectionKind` nombra la manipulacion (congelar, desplazar,
pinchar, rampar); `FaultType` nombra el modo de fallo que un detector reporta.
Mezclarlos convertiria la evaluacion en una tautologia: la verdad-terreno se
estaria definiendo con el vocabulario del detector que se quiere medir.

Una misma manipulacion produce varias etiquetas legitimas, porque una es varias
cosas a la vez. Un escalon fuera del sobre es un fuera-de-rango mientras dura Y
un cambio rapido en sus dos flancos. Etiquetar solo lo primero haria que el
detector de tasa acertase y se le contase como falso positivo. Las etiquetas
derivadas se declaran por tipo de manipulacion en `InjectedEpisode.labels`.

Una etiqueta se calcula de forma OBJETIVA y no por episodio: la de
`BIAS_OUT_OF_RANGE` es, por definicion, "el valor cae fuera del sobre de alarma
de su transmisor". Se evalua sobre el frame ya inyectado y contra el span, sin
mirar ningun umbral de detector, de modo que tambien recoge las excursiones que
una rampa provoca al final de su recorrido. Las demas etiquetas no admiten una
definicion asi sin circularidad: "stuck" medido como "tirada de N repeticiones"
usaria el propio parametro del detector como verdad.

El campo `quality` NO se toca al inyectar. Un transmisor que se congela sigue
publicando GOOD; que la degradacion sea invisible en los metadatos es justo la
condicion bajo la que el validador tiene que trabajar, y coincide con que el
validador ESCRIBE quality y nunca la lee.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from data.schemas.sensor_spans import SensorSpan
from data.validation.sensor_fault import FaultType

DEFAULT_INJECTION_SEED = 20260721
"""Semilla por defecto del generador.

Fija por contrato: la verdad-terreno de T3.6 tiene que ser identica entre
ejecuciones, o el gate de precision mediria una muestra distinta cada vez.
"""


class InjectionKind(StrEnum):
    """Manipulation applied to a signal, named for what it does to the values."""

    FREEZE = "freeze"
    """La senal se latch-ea en su ultimo valor: transmisor congelado."""

    OFFSET = "offset"
    """Se suma un desplazamiento constante que saca la senal del sobre de alarma."""

    SPIKE = "spike"
    """Una unica muestra salta y la siguiente vuelve a la serie real."""

    RAMP = "ramp"
    """Se acumula una pendiente propia sobre la serie real: deriva de instrumento."""


@dataclass(frozen=True)
class InjectedEpisode:
    """One contiguous manipulation of one sensor, with the labels it justifies.

    Attributes:
        sensor_id: Instrument tag that was degraded.
        kind: Manipulation applied.
        start_timestep: First timestep affected, inclusive.
        end_timestep: Last timestep affected, inclusive. Equal to start for a
            SPIKE, which lasts one sample.
        magnitude: Size of the manipulation in alarm-envelope widths. Its meaning
            depends on kind: the applied offset for OFFSET and SPIKE, the slope in
            envelope widths per hour for RAMP, and 0.0 for FREEZE, which adds
            nothing to the signal.
    """

    sensor_id: str
    kind: InjectionKind
    start_timestep: int
    end_timestep: int
    magnitude: float

    @property
    def length(self) -> int:
        """Return the number of timesteps the episode spans, both ends included."""
        return self.end_timestep - self.start_timestep + 1

    def labels(self, last_timestep: int) -> dict[FaultType, set[int]]:
        """Return the timesteps this episode justifies labelling, per fault type.

        BIAS_OUT_OF_RANGE nunca aparece aqui: se deriva de los valores finales
        contra el sobre de alarma, que es una definicion objetiva y no una
        consecuencia esperada de la manipulacion.

        KALMAN_ANOMALY cubre el episodio entero mas el paso siguiente, para
        cualquier tipo. El detector de Kalman es la senal de anomalia de proposito
        general del validador: mientras dura una degradacion inyectada, y en el
        paso en que la senal vuelve, tiene motivo para disparar.

        Args:
            last_timestep: Highest timestep present in the frame. Las etiquetas
                se recortan a el, porque un episodio que termina en la ultima
                muestra no tiene flanco de salida que etiquetar.

        Returns:
            Mapping from fault type to the set of timesteps labelled positive.
            Solo contiene las claves con al menos un timestep.
        """
        start, end = self.start_timestep, self.end_timestep
        after = end + 1

        labels: dict[FaultType, set[int]] = {
            FaultType.KALMAN_ANOMALY: set(range(start, min(after, last_timestep) + 1))
        }

        if self.kind is InjectionKind.FREEZE:
            # El primer paso de la ventana conserva su valor real; congelado esta
            # a partir del siguiente, que es el primero que repite.
            labels[FaultType.STUCK] = set(range(start + 1, end + 1))
            # El flanco de salida es un escalon: la senal reengancha con la serie
            # real. Se etiqueta aunque la serie real pueda no haberse movido mucho
            # en la ventana: el evento de instrumento existe, y que el detector de
            # tasa no lo alcance es un falso negativo suyo, no un error de la
            # verdad-terreno.
            labels[FaultType.NOISE_SPIKE] = {after}
        elif self.kind is InjectionKind.OFFSET:
            labels[FaultType.NOISE_SPIKE] = {start, after}
        elif self.kind is InjectionKind.SPIKE:
            labels[FaultType.NOISE_SPIKE] = {start, after}
        elif self.kind is InjectionKind.RAMP:
            labels[FaultType.SENSOR_DRIFT] = set(range(start, end + 1))
            # La pendiente por muestra se elige por debajo del limite de tasa, de
            # modo que la rampa mide al detector de deriva y no al de tasa. Su
            # SALIDA si es un escalon: la senal cae de golpe toda la desviacion
            # acumulada, que son varios anchos de sobre en una sola muestra.
            labels[FaultType.NOISE_SPIKE] = {after}

        return {
            fault_type: {t for t in timesteps if 0 <= t <= last_timestep}
            for fault_type, timesteps in labels.items()
            if any(0 <= t <= last_timestep for t in timesteps)
        }


@dataclass(frozen=True)
class InjectionConfig:
    """How much of each manipulation to apply, and how large.

    Todas las magnitudes dependientes de escala se expresan en anchos del sobre de
    alarma y no en unidades de ingenieria, por la misma razon que los umbrales del
    validador: un numero absoluto compartido por 52 canales heterogeneos no
    significa lo mismo en dos de ellos.

    Attributes:
        sensors_per_kind: Sensores degradados con cada manipulacion. Los conjuntos
            son disjuntos, de modo que ningun sensor sufre dos tipos distintos y
            las etiquetas nunca compiten sobre el mismo par.
        episodes_per_sensor: Episodios por sensor degradado, repartidos uno por
            segmento igual de la linea temporal para que no se solapen.
        freeze_length: Rango [min, max] de duracion de un congelamiento, en
            timesteps. El minimo supera la ventana del detector de stuck, que es 8.
        offset_length: Rango de duracion de un desplazamiento, en timesteps.
        ramp_length: Rango de duracion de una rampa, en timesteps.
        offset_margin: Cuanto sobrepasa el desplazamiento el limite de alarma, en
            anchos de sobre. Se aplica sobre el extremo mas desfavorable de la
            ventana, de modo que TODAS sus muestras quedan fuera del sobre.
        spike_magnitude: Rango de la altura de un pico, en anchos de sobre.
        ramp_rate_per_hour: Rango de la pendiente de una rampa, en anchos de sobre
            por hora. El rango por defecto cae dentro de la unica banda que el
            detector de deriva puede ver a cadencia TEP: por encima de los 10,13
            anchos/hora del movimiento propio del proceso en d00, y por debajo de
            los 18 que ya constituyen una violacion de tasa en cada muestra. Ver
            DRIFT_DETECTION_FLOOR_NOTE en sensor_validator.py. Inyectar mas lento
            mediria un detector ciego por construccion; mas rapido mediria al
            detector de tasa creyendo medir al de deriva.
        guard_timesteps: Margen libre a cada lado del segmento donde se coloca un
            episodio, para que el flanco de salida de uno no caiga sobre el de
            entrada del siguiente.
    """

    sensors_per_kind: int = 6
    episodes_per_sensor: int = 2
    freeze_length: tuple[int, int] = (12, 40)
    offset_length: tuple[int, int] = (15, 40)
    ramp_length: tuple[int, int] = (25, 45)
    offset_margin: float = 0.35
    spike_magnitude: tuple[float, float] = (1.5, 4.0)
    ramp_rate_per_hour: tuple[float, float] = (13.0, 17.0)
    guard_timesteps: int = 3


@dataclass(frozen=True)
class InjectionReport:
    """The degraded frame together with the ground truth it carries.

    Attributes:
        frame: Copy of the input frame with the manipulated values and their
            recomputed raw_counts. Ordenado por (sensor_id, timestep).
        episodes: Every episode applied, in the order it was applied.
        labels: Positive (sensor_id, timestep) pairs per fault type.
        guard: Pairs adjacent to an episode, excluded from the negative class.
            Alrededor de una degradacion la senal esta genuinamente perturbada y
            QUE detector dispara no esta determinado: el flanco de un escalon es a
            la vez un cambio rapido, un transitorio de velocidad y una anomalia
            de residual. Contar esas casillas como falsos positivos de todos los
            detectores menos uno mediria el reparto de atribucion, no la
            deteccion. Se excluyen de la clase negativa; NO se excluyen de la
            positiva, de modo que un par etiquetado sigue exigiendo su deteccion.
        sample_interval_seconds: Cadencia medida sobre los timestamps del frame,
            no supuesta. Los umbrales del validador que dependen del tiempo se
            calibran contra este numero.
    """

    frame: pd.DataFrame
    episodes: tuple[InjectedEpisode, ...]
    labels: dict[FaultType, frozenset[tuple[str, int]]]
    guard: frozenset[tuple[str, int]]
    sample_interval_seconds: float

    def labels_for(self, fault_type: FaultType) -> frozenset[tuple[str, int]]:
        """Return the positive pairs of one fault type.

        Args:
            fault_type: Instrument failure mode.

        Returns:
            The labelled (sensor_id, timestep) pairs, empty when the injection
            produced none of that type.
        """
        return self.labels.get(fault_type, frozenset())

    def degraded_sensors(self) -> frozenset[str]:
        """Return every sensor touched by at least one episode."""
        return frozenset(episode.sensor_id for episode in self.episodes)


def _sample_interval_seconds(frame: pd.DataFrame) -> float:
    """Measure the sampling cadence from the frame's own timestamps.

    Se mide en lugar de asumirse porque los umbrales que dependen del tiempo
    (tasa de cambio, deriva por hora) cambian de significado con la cadencia, y
    la del TEP son 180 s, no el segundo de un sondeo SCADA.

    Args:
        frame: Long-format frame with timestamp and timestep columns.

    Returns:
        Seconds between consecutive timesteps.

    Raises:
        ValueError: If the frame has fewer than two distinct timesteps, or if the
            spacing is not uniform.
    """
    per_timestep = (
        frame.drop_duplicates(subset="timestep")
        .sort_values("timestep")
        .loc[:, "timestamp"]
    )
    if len(per_timestep) < 2:
        raise ValueError(
            "Cannot measure the sampling interval from a frame with fewer than "
            f"two timesteps (got {len(per_timestep)})."
        )

    deltas = per_timestep.diff().dropna().dt.total_seconds().unique()
    if len(deltas) != 1:
        raise ValueError(
            f"Sampling interval is not uniform: found {len(deltas)} distinct gaps "
            f"({sorted(deltas)[:5]}). The rate and drift thresholds assume a "
            "single cadence."
        )
    return float(deltas[0])


class TEPFaultInjector:
    """Degrade chosen sensors of an adapted TEP frame and record the ground truth.

    El generador se construye con una semilla fija y toda decision aleatoria
    (que sensores, donde y cuanto) sale de el, de modo que dos ejecuciones
    producen exactamente el mismo frame y las mismas etiquetas.
    """

    def __init__(
        self,
        spans: dict[str, SensorSpan],
        config: InjectionConfig | None = None,
        seed: int = DEFAULT_INJECTION_SEED,
    ) -> None:
        """Build the injector.

        Args:
            spans: Span table. Cada sensor degradado necesita el suyo, porque las
                magnitudes se expresan en anchos de su sobre de alarma.
            config: Sizing of the injection. Defaults to InjectionConfig().
            seed: Seed of the random generator.

        Raises:
            ValueError: If the configuration is internally inconsistent.
        """
        config = config or InjectionConfig()
        if config.sensors_per_kind <= 0:
            raise ValueError(
                f"sensors_per_kind must be positive, got {config.sensors_per_kind}."
            )
        if config.episodes_per_sensor <= 0:
            raise ValueError(
                f"episodes_per_sensor must be positive, got {config.episodes_per_sensor}."
            )
        for name, bounds in (
            ("freeze_length", config.freeze_length),
            ("offset_length", config.offset_length),
            ("ramp_length", config.ramp_length),
        ):
            if bounds[0] <= 0 or bounds[1] < bounds[0]:
                raise ValueError(
                    f"{name} must be a non-empty range of positive lengths, got {bounds}."
                )
        if config.freeze_length[0] <= 8:
            raise ValueError(
                f"freeze_length starts at {config.freeze_length[0]}, which does not "
                "exceed the stuck detector window of 8. Shorter freezes are "
                "undetectable by construction and would only depress recall."
            )

        self.spans = spans
        self.config = config
        self.seed = seed

    def inject(self, frame: pd.DataFrame) -> InjectionReport:
        """Degrade a copy of the frame and return it with its ground truth.

        Args:
            frame: Long-format frame as produced by TEPAdapter.adapt_all, with at
                least sensor_id, timestep, timestamp and value columns.

        Returns:
            The InjectionReport holding the degraded frame, the episodes applied
            and the positive pairs per fault type.

        Raises:
            KeyError: If a required column is missing.
            ValueError: If the timesteps of a sensor are not the contiguous range
                0..n-1, if the span table does not cover enough sensors, or if the
                configured episodes do not fit in the available timeline.
        """
        required = {"sensor_id", "timestep", "timestamp", "value"}
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(
                f"Frame is missing required columns: {sorted(missing)}. Expected the "
                "long format produced by TEPAdapter.adapt_all."
            )

        interval = _sample_interval_seconds(frame)
        working = frame.sort_values(
            ["sensor_id", "timestep"], kind="stable"
        ).reset_index(drop=True)

        n_timesteps = int(working["timestep"].nunique())
        last_timestep = n_timesteps - 1
        positions = {
            str(sensor): np.asarray(index)
            for sensor, index in working.groupby("sensor_id").indices.items()
        }
        for sensor, index in positions.items():
            steps = working["timestep"].to_numpy()[index]
            if not np.array_equal(steps, np.arange(n_timesteps)):
                raise ValueError(
                    f"Sensor {sensor} does not carry the contiguous timestep range "
                    f"0..{last_timestep}. The injector addresses samples by "
                    "position within a sensor, which that assumption underpins."
                )

        rng = np.random.default_rng(self.seed)
        targets = self._choose_targets(sorted(positions), rng)

        values = working["value"].to_numpy(dtype=np.float64).copy()
        episodes: list[InjectedEpisode] = []
        for kind, sensors in targets.items():
            for sensor in sensors:
                for window in self._windows(kind, n_timesteps, rng):
                    episodes.append(
                        self._apply(kind, sensor, window, positions[sensor], values, interval, rng)
                    )

        working["value"] = values
        self._recompute_raw_counts(working, {episode.sensor_id for episode in episodes})

        return InjectionReport(
            frame=working,
            episodes=tuple(episodes),
            labels=self._collect_labels(episodes, working, last_timestep),
            guard=self._guard_pairs(episodes, last_timestep),
            sample_interval_seconds=interval,
        )

    def _guard_pairs(
        self, episodes: list[InjectedEpisode], last_timestep: int
    ) -> frozenset[tuple[str, int]]:
        """Return the pairs adjacent to an episode, excluded from the negative class.

        Args:
            episodes: Episodes applied.
            last_timestep: Highest timestep present.

        Returns:
            The (sensor_id, timestep) pairs within guard_timesteps of any episode,
            its own window included.
        """
        guard = self.config.guard_timesteps
        pairs: set[tuple[str, int]] = set()
        for episode in episodes:
            low = max(0, episode.start_timestep - guard)
            high = min(last_timestep, episode.end_timestep + guard)
            pairs.update((episode.sensor_id, step) for step in range(low, high + 1))
        return frozenset(pairs)

    # ------------------------------------------------------------------
    # Target and window selection
    # ------------------------------------------------------------------

    def _choose_targets(
        self, sensors: list[str], rng: np.random.Generator
    ) -> dict[InjectionKind, list[str]]:
        """Assign a disjoint set of sensors to every manipulation.

        Args:
            sensors: Sorted tags present in the frame.
            rng: Seeded generator.

        Returns:
            Mapping from manipulation to the sensors it will degrade.

        Raises:
            ValueError: If the frame does not hold enough sensors with a span to
                cover every manipulation.
        """
        eligible = [sensor for sensor in sensors if sensor in self.spans]
        needed = self.config.sensors_per_kind * len(InjectionKind)
        if len(eligible) < needed:
            raise ValueError(
                f"Injection needs {needed} sensors with a span "
                f"({self.config.sensors_per_kind} per manipulation across "
                f"{len(InjectionKind)} kinds) but only {len(eligible)} of the "
                f"{len(sensors)} sensors in the frame have one."
            )

        shuffled = [eligible[i] for i in rng.permutation(len(eligible))]
        targets: dict[InjectionKind, list[str]] = {}
        for offset, kind in enumerate(InjectionKind):
            start = offset * self.config.sensors_per_kind
            targets[kind] = sorted(
                shuffled[start : start + self.config.sensors_per_kind]
            )
        return targets

    def _windows(
        self, kind: InjectionKind, n_timesteps: int, rng: np.random.Generator
    ) -> list[tuple[int, int]]:
        """Place the episodes of one sensor, one per equal segment of the timeline.

        Repartir por segmentos en lugar de sortear libremente y rechazar solapes
        hace el numero de episodios determinista: no depende de cuantos intentos
        salieron mal.

        Args:
            kind: Manipulation, which fixes the length distribution.
            n_timesteps: Length of the timeline.
            rng: Seeded generator.

        Returns:
            List of (start, end) timestep pairs, both ends inclusive.

        Raises:
            ValueError: If an episode of the configured length does not fit in its
                segment once the guard margins are taken.
        """
        guard = self.config.guard_timesteps
        segment = n_timesteps // self.config.episodes_per_sensor
        windows: list[tuple[int, int]] = []

        for index in range(self.config.episodes_per_sensor):
            length = int(rng.integers(*self._length_bounds(kind)))
            earliest = index * segment + guard
            latest = (index + 1) * segment - guard - length
            if latest < earliest:
                raise ValueError(
                    f"A {kind.value} episode of {length} timesteps does not fit in "
                    f"segment {index} of {self.config.episodes_per_sensor} over "
                    f"{n_timesteps} timesteps with a guard of {guard}. Shorten the "
                    "episodes or place fewer per sensor."
                )
            start = int(rng.integers(earliest, latest + 1))
            windows.append((start, start + length - 1))

        return windows

    def _length_bounds(self, kind: InjectionKind) -> tuple[int, int]:
        """Return the half-open length range of a manipulation, for rng.integers.

        Args:
            kind: Manipulation.

        Returns:
            (low, high_exclusive) episode length in timesteps.
        """
        bounds = {
            InjectionKind.FREEZE: self.config.freeze_length,
            InjectionKind.OFFSET: self.config.offset_length,
            InjectionKind.RAMP: self.config.ramp_length,
            InjectionKind.SPIKE: (1, 1),
        }[kind]
        return bounds[0], bounds[1] + 1

    # ------------------------------------------------------------------
    # Value manipulation
    # ------------------------------------------------------------------

    def _apply(
        self,
        kind: InjectionKind,
        sensor_id: str,
        window: tuple[int, int],
        sensor_positions: np.ndarray,
        values: np.ndarray,
        interval_seconds: float,
        rng: np.random.Generator,
    ) -> InjectedEpisode:
        """Degrade one window of one sensor in place and describe what was done.

        Args:
            kind: Manipulation to apply.
            sensor_id: Instrument tag.
            window: (start, end) timesteps, both inclusive.
            sensor_positions: Row positions of this sensor in timestep order.
            values: The frame's value column, mutated in place.
            interval_seconds: Measured sampling cadence.
            rng: Seeded generator.

        Returns:
            The episode record, with its magnitude in alarm-envelope widths.
        """
        start, end = window
        rows = sensor_positions[start : end + 1]
        width = self.spans[sensor_id].alarm_max - self.spans[sensor_id].alarm_min
        direction = 1.0 if rng.random() < 0.5 else -1.0

        if kind is InjectionKind.FREEZE:
            values[rows] = values[rows[0]]
            magnitude = 0.0
        elif kind is InjectionKind.OFFSET:
            offset = self._offset_for(sensor_id, values[rows], width, direction)
            values[rows] += offset
            magnitude = offset / width
        elif kind is InjectionKind.SPIKE:
            offset = direction * float(rng.uniform(*self.config.spike_magnitude)) * width
            values[rows] += offset
            magnitude = offset / width
        else:
            rate_per_hour = direction * float(rng.uniform(*self.config.ramp_rate_per_hour))
            per_step = rate_per_hour * width * interval_seconds / 3600.0
            values[rows] += per_step * np.arange(1, len(rows) + 1, dtype=np.float64)
            magnitude = rate_per_hour

        return InjectedEpisode(
            sensor_id=sensor_id,
            kind=kind,
            start_timestep=start,
            end_timestep=end,
            magnitude=float(magnitude),
        )

    def _offset_for(
        self, sensor_id: str, window: np.ndarray, width: float, direction: float
    ) -> float:
        """Return the offset that pushes an entire window out of the alarm envelope.

        Se calcula contra el extremo mas desfavorable de la ventana y no contra su
        primer valor: un desplazamiento dimensionado sobre una sola muestra dejaria
        dentro del sobre a las que se mueven en sentido contrario, y esas quedarian
        etiquetadas como fuera de rango sin estarlo.

        Args:
            sensor_id: Instrument tag.
            window: Values of the window before the offset.
            width: Alarm envelope width.
            direction: +1 to push above alarm_max, -1 to push below alarm_min.

        Returns:
            The offset in engineering units.
        """
        span = self.spans[sensor_id]
        margin = self.config.offset_margin * width
        if direction > 0.0:
            return float(span.alarm_max - window.min() + margin)
        return float(span.alarm_min - window.max() - margin)

    def _recompute_raw_counts(self, frame: pd.DataFrame, sensors: set[str]) -> None:
        """Rescale raw_counts of the degraded sensors in place.

        El frame largo lleva value y raw_counts como dos vistas del mismo numero.
        Cambiar una y dejar la otra produciria un artefacto internamente
        incoherente, y el clamp del ADC forma parte de lo que se esta simulando:
        un transmisor desplazado fuera de su span calibrado satura, no informa un
        valor imposible.

        Args:
            frame: Degraded frame, mutated in place. Sin columna raw_counts no
                hace nada, porque no todo consumidor la necesita.
            sensors: Tags whose values changed.
        """
        if "raw_counts" not in frame.columns or not sensors:
            return

        counts = frame["raw_counts"].to_numpy().copy()
        values = frame["value"].to_numpy(dtype=np.float64)
        column = frame["sensor_id"].to_numpy()
        for position in np.flatnonzero(np.isin(column, list(sensors))):
            span = self.spans[str(column[position])]
            counts[position] = span.to_raw_counts(float(values[position]))
        frame["raw_counts"] = counts

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------

    def _collect_labels(
        self,
        episodes: list[InjectedEpisode],
        frame: pd.DataFrame,
        last_timestep: int,
    ) -> dict[FaultType, frozenset[tuple[str, int]]]:
        """Merge the per-episode labels with the objective out-of-range label.

        Args:
            episodes: Episodes applied.
            frame: Degraded frame.
            last_timestep: Highest timestep present.

        Returns:
            Mapping from fault type to its positive (sensor_id, timestep) pairs.
        """
        collected: dict[FaultType, set[tuple[str, int]]] = {}
        for episode in episodes:
            for fault_type, timesteps in episode.labels(last_timestep).items():
                bucket = collected.setdefault(fault_type, set())
                bucket.update((episode.sensor_id, step) for step in timesteps)

        out_of_range = self.out_of_envelope_pairs(frame)
        if out_of_range:
            collected.setdefault(FaultType.BIAS_OUT_OF_RANGE, set()).update(out_of_range)

        return {
            fault_type: frozenset(pairs) for fault_type, pairs in collected.items()
        }

    def out_of_envelope_pairs(self, frame: pd.DataFrame) -> set[tuple[str, int]]:
        """Return every pair whose value lies outside its sensor's alarm envelope.

        Definicion objetiva de la clase positiva de BIAS_OUT_OF_RANGE: no depende
        de ningun umbral de detector, solo del valor y del span. Es la unica de las
        etiquetas que admite una definicion asi; "stuck" medido como una tirada de
        N repeticiones usaria el propio parametro del detector como verdad.

        Args:
            frame: Any long-format frame with sensor_id, timestep and value.

        Returns:
            The out-of-envelope (sensor_id, timestep) pairs. Sensors without a
            span are skipped: sin sobre no hay excursion que declarar.
        """
        pairs: set[tuple[str, int]] = set()
        for sensor, group in frame.groupby("sensor_id"):
            span = self.spans.get(str(sensor))
            if span is None:
                continue
            values = group["value"].to_numpy(dtype=np.float64)
            outside = (values < span.alarm_min) | (values > span.alarm_max)
            steps = group["timestep"].to_numpy()[outside]
            pairs.update((str(sensor), int(step)) for step in steps)
        return pairs
