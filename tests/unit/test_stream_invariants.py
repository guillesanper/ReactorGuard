"""End-to-end invariants of the streaming path, over the real TEP d21 run.

Protege el criterio 2 de la Fase 2 (precision del detector de stuck) en el
DESPLIEGUE y no solo en el validador: el unico transmisor congelado de verdad del
dataset (TEP-XMV-04 en d21, 480 muestras) solo se detecta si el streamer, el broker y
el consumidor conservan el orden por sensor. La validacion offline
(test_validator_on_tep.py) detecta 473 de esos 480 pares (se pierde la cabecera de la
ventana de 8 muestras). El camino por streaming debe dar EXACTAMENTE esa cifra con 1,
2 y 3 consumidores repartiendose las 12 particiones del topic crudo.

El camino completo es el real salvo el broker: `TEPStreamer` emite d21 con la
cabecera run-id y la clave `sensor_id`, `InMemoryBroker` reparte con el murmur2 de
kafka-python, y varios `ValidationConsumer` (cada uno con su `SensorValidator` y un
conjunto disjunto de particiones) validan con la tabla de spans calibrada.

El test ESPEJO emite lo mismo SIN clave (round-robin, lo que haria un productor mal
configurado). Demuestra que el test tiene dientes: sin la clave el sensor congelado
deja de detectarse sin ningun error ni log, que es exactamente el fallo silencioso
que la clave `sensor_id` (D1) existe para evitar.

Si el parquet adaptado no esta poblado los tests se SALTAN, no fallan; el script de
verificacion distingue skipped de passed.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data.generators.reading_source import RunFrame, run_name
from data.generators.tep_adapter import readings_from_frame
from data.generators.tep_streamer import TEPStreamer
from data.generators.tep_streamer_params import StreamerParams, StreamMode, load_streamer_params
from data.schemas.sensor_reading import SensorReading
from data.schemas.sensor_spans import DEFAULT_SPANS_PATH, SensorSpan, load_sensor_spans
from data.streaming import metrics as metric_names
from data.streaming.metrics import RESET_REASONS, StreamMetrics
from data.streaming.streaming_params import StreamingParams, load_streaming_params
from data.streaming.transport import Header
from data.validation.alert_event import AlertEvent
from data.validation.sensor_fault import FaultType
from data.validation.sensor_validator import DEFAULT_STUCK_WINDOW, SensorValidator
from data.validation.validation_consumer import ValidationConsumer
from tests.support.in_memory_broker import InMemoryBroker
from tests.support.in_memory_consumer import InMemoryConsumer

PARAMS_PATH = Path("params.yaml")
_STUCK_PARTITION = Path("data/processed/tep/fault_type=21/readings.parquet")
_FROZEN_TAG = "TEP-XMV-04"
_N_PARTITIONS = 12
"""Particiones del topic crudo (k8s/base/kafka/kafka-topics.yaml)."""

Pair = tuple[str, datetime]
"""(sensor_id, timestamp) de una lectura: un par (sensor, timestep) del TEP."""

FaultKey = tuple[str, str, str, datetime]
"""(sensor_id, fault_type, detector, timestamp) de una alerta."""

pytestmark = pytest.mark.skipif(
    not _STUCK_PARTITION.exists(),
    reason=(
        "El parquet adaptado del TEP no esta poblado. Generalo con "
        ".\\infra\\scripts\\Invoke-Pipeline.ps1 antes de medir las invariantes."
    ),
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _SingleRunSource:
    """A ReadingSource that yields one already-loaded run."""

    def __init__(self, run_id: str, frame: pd.DataFrame) -> None:
        self._run = (run_id, frame)

    def runs(self) -> Iterator[RunFrame]:
        return iter([self._run])


class _KeylessProducer:
    """A producer that forgets the key, as a misconfigured one would."""

    def __init__(self, inner: InMemoryBroker) -> None:
        self._inner = inner

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[Header] = (),
    ) -> None:
        del key
        self._inner.send(topic, None, value, headers)

    def flush(self, timeout: float | None = None) -> None:
        self._inner.flush(timeout)

    def close(self, timeout: float | None = None) -> None:
        self._inner.close(timeout)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fault_key(
    sensor: str, fault_type: FaultType, detector: str, detected_at: datetime
) -> FaultKey:
    """Build the identity of an alert, independent of where it was raised.

    Args:
        sensor: Instrument tag.
        fault_type: Failure mode.
        detector: Class name of the detector.
        detected_at: Timestamp of the triggering reading.

    Returns:
        A hashable key; two runs agree on an alert when they agree on this key.
    """
    return (sensor, fault_type.value, detector, detected_at)


def split_partitions(consumers: int) -> list[list[int]]:
    """Share the partitions of the raw topic among a number of consumers.

    Args:
        consumers: Number of consumers of the group.

    Returns:
        One disjoint list of partition numbers per consumer, covering all of them.
    """
    return [list(range(index, _N_PARTITIONS, consumers)) for index in range(consumers)]


def stream_run(
    frame: pd.DataFrame,
    streamer_params: StreamerParams,
    streaming: StreamingParams,
    *,
    keyed: bool,
) -> InMemoryBroker:
    """Emit one run through the real streamer into a fresh in-memory broker.

    Args:
        frame: Long-format frame of the run (d21).
        streamer_params: Streamer configuration (FAST, no loop).
        streaming: Topic names.
        keyed: True for the production behaviour (key = sensor_id); False for the
            mirror, which drops the key so the broker deals messages round-robin.

    Returns:
        The broker holding the raw topic.
    """
    broker = InMemoryBroker(_N_PARTITIONS)
    producer = broker if keyed else _KeylessProducer(broker)
    streamer = TEPStreamer(
        producer,
        _SingleRunSource(run_name(21), frame),
        streamer_params,
        StreamMetrics(),
        topic=streaming.raw_topic,
    )
    streamer.run()
    return broker


@dataclasses.dataclass(frozen=True)
class GroupResult:
    """What a consumer group concluded about the stream.

    Attributes:
        stuck: Pairs flagged as stuck, per sensor.
        faults: Every alert raised, whatever the detector.
        resets: Detector state resets by reason, summed over the consumers.
        validated: Readings published to the validated topic.
    """

    stuck: dict[str, set[Pair]]
    faults: set[FaultKey]
    resets: dict[str, float]
    validated: int


def consume_group(
    raw: InMemoryBroker,
    spans: dict[str, SensorSpan],
    streaming: StreamingParams,
    consumers: int,
) -> GroupResult:
    """Drain the raw topic with a group of consumers owning disjoint partitions.

    Args:
        raw: Broker holding the raw topic.
        spans: Calibrated span table.
        streaming: Topic names and batch sizes.
        consumers: Number of consumers in the group.

    Returns:
        The stuck pairs and every alert found, the state resets and the validated count.
    """
    out = InMemoryBroker(_N_PARTITIONS)
    group: list[tuple[ValidationConsumer, StreamMetrics]] = []
    for partitions in split_partitions(consumers):
        metrics = StreamMetrics()
        consumer = InMemoryConsumer(raw, partitions=partitions)
        sut = ValidationConsumer(
            consumer, out, SensorValidator(spans), streaming, metrics, warmup_seconds=None
        )
        consumer.subscribe([streaming.raw_topic], sut)
        group.append((sut, metrics))

    while sum([sut.run_once() for sut, _ in group]) > 0:
        pass

    stuck: dict[str, set[Pair]] = defaultdict(set)
    faults: set[FaultKey] = set()
    for message in out.sent:
        if message.topic != streaming.alerts_topic:
            continue
        alert = AlertEvent.from_kafka_bytes(message.value)
        faults.add(fault_key(alert.sensor_id, alert.fault_type, alert.detector, alert.detected_at))
        if alert.fault_type is FaultType.STUCK:
            stuck[alert.sensor_id].add((alert.sensor_id, alert.detected_at))
    resets = {
        reason: sum(
            metrics.registry.get_sample_value(
                metric_names.DETECTOR_STATE_RESETS, {"reason": reason}
            )
            or 0.0
            for _, metrics in group
        )
        for reason in RESET_REASONS
    }
    validated = sum(1 for m in out.sent if m.topic == streaming.validated_topic)
    return GroupResult(dict(stuck), faults, resets, validated)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spans() -> dict[str, SensorSpan]:
    """Load the committed, calibrated span table."""
    return load_sensor_spans(DEFAULT_SPANS_PATH)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Return d21 straight from the adapted parquet."""
    return pd.read_parquet(_STUCK_PARTITION)


@pytest.fixture(scope="module")
def streaming() -> StreamingParams:
    """Load the streaming section of params.yaml."""
    return load_streaming_params(PARAMS_PATH)


@pytest.fixture(scope="module")
def streamer_params() -> StreamerParams:
    """Load the streamer section, forced to FAST mode and a single pass."""
    loaded = load_streamer_params(PARAMS_PATH)
    return dataclasses.replace(loaded, mode=StreamMode.FAST, loop=False)


@pytest.fixture(scope="module")
def keyed_broker(
    frame: pd.DataFrame, streamer_params: StreamerParams, streaming: StreamingParams
) -> InMemoryBroker:
    """Raw topic emitted the way production does it (key = sensor_id)."""
    return stream_run(frame, streamer_params, streaming, keyed=True)


@pytest.fixture(scope="module")
def keyless_broker(
    frame: pd.DataFrame, streamer_params: StreamerParams, streaming: StreamingParams
) -> InMemoryBroker:
    """Raw topic emitted without a key (round-robin over the 12 partitions)."""
    return stream_run(frame, streamer_params, streaming, keyed=False)


@pytest.fixture(scope="module")
def offline_faults(frame: pd.DataFrame, spans: dict[str, SensorSpan]) -> set[FaultKey]:
    """Alerts of the validator run in-process over d21, in (timestep, sensor) order.

    Es la referencia: lo mismo que mide test_validator_on_tep.py, sin broker.
    """
    validator = SensorValidator(spans)
    faults: set[FaultKey] = set()
    ordered = frame.sort_values(["timestep", "sensor_id"], kind="stable")
    for reading in readings_from_frame(ordered):
        for fault in validator.validate(reading).faults:
            faults.add(
                fault_key(fault.sensor_id, fault.fault_type, fault.detector, fault.detected_at)
            )
    return faults


@pytest.fixture(scope="module")
def expected_pairs(frame: pd.DataFrame) -> int:
    """Pairs the stuck detector can flag on the frozen transmitter.

    Se pierde solo la cabecera de la ventana: 480 muestras menos 7 = 473, la cifra de
    validator_metrics_tep.json.
    """
    return int(frame["timestep"].nunique()) - (DEFAULT_STUCK_WINDOW - 1)


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


class TestOfflineReference:
    """La cifra que el camino por streaming tiene que reproducir."""

    def test_offline_finds_473_pairs_on_the_frozen_transmitter_only(
        self, offline_faults: set[FaultKey], expected_pairs: int
    ) -> None:
        """The in-process reference matches the published validator metric."""
        stuck = [key for key in offline_faults if key[1] == FaultType.STUCK.value]
        assert expected_pairs == 473
        assert {key[0] for key in stuck} == {_FROZEN_TAG}
        assert len(stuck) == expected_pairs


# ---------------------------------------------------------------------------
# D1: the key keeps every sensor in one partition, in order
# ---------------------------------------------------------------------------


class TestKeyedPartitioning:
    """La clave sensor_id deja cada sensor en una sola particion y en orden."""

    def test_every_sensor_lives_in_exactly_one_partition(
        self, keyed_broker: InMemoryBroker, streaming: StreamingParams, frame: pd.DataFrame
    ) -> None:
        """No sensor is split across partitions."""
        homes: dict[str, set[int]] = defaultdict(set)
        for partition, records in keyed_broker.partitions_of(streaming.raw_topic).items():
            for record in records:
                assert record.key is not None
                homes[record.key.decode("utf-8")].add(partition)
        assert len(homes) == frame["sensor_id"].nunique()
        assert all(len(partitions) == 1 for partitions in homes.values())

    def test_each_sensor_is_in_timestamp_order_with_all_its_readings(
        self, keyed_broker: InMemoryBroker, streaming: StreamingParams, frame: pd.DataFrame
    ) -> None:
        """Offset order inside a partition is the emission order of the sensor."""
        steps = int(frame["timestep"].nunique())
        stamps: dict[str, list[datetime]] = defaultdict(list)
        for records in keyed_broker.partitions_of(streaming.raw_topic).values():
            for record in records:
                assert record.value is not None
                reading = SensorReading.from_kafka_bytes(record.value)
                stamps[reading.sensor.id].append(reading.timestamp)
        assert all(len(series) == steps for series in stamps.values())
        assert all(series == sorted(series) for series in stamps.values())

    def test_messages_carry_the_run_id_header(
        self, keyed_broker: InMemoryBroker, streaming: StreamingParams
    ) -> None:
        """Every message names its run, so the consumer can tell passes apart (D2)."""
        records = [
            record
            for partition in keyed_broker.partitions_of(streaming.raw_topic).values()
            for record in partition
        ]
        assert {record.header("run-id") for record in records} == {b"fault_type=21/loop=0"}


# ---------------------------------------------------------------------------
# The invariant: the group finds what the offline validator finds
# ---------------------------------------------------------------------------


class TestStuckDetectionThroughTheStream:
    """El criterio 2 en el despliegue: 473 pares con 1, 2 y 3 consumidores."""

    @pytest.mark.parametrize("consumers", [1, 2, 3])
    def test_group_detects_exactly_the_offline_pairs(
        self,
        keyed_broker: InMemoryBroker,
        spans: dict[str, SensorSpan],
        streaming: StreamingParams,
        offline_faults: set[FaultKey],
        expected_pairs: int,
        consumers: int,
    ) -> None:
        """Splitting the partitions among consumers must not change the detection.

        Se compara el conjunto ENTERO de alertas (todos los detectores) y no solo el
        stuck: el stuck de una senal constante sobrevive a casi cualquier reparto,
        pero el Kalman y la tasa de cambio dependen del orden y del timestamp.
        """
        result = consume_group(keyed_broker, spans, streaming, consumers)

        assert len(result.stuck[_FROZEN_TAG]) == expected_pairs
        assert set(result.stuck) == {_FROZEN_TAG}
        assert result.faults == offline_faults

    @pytest.mark.parametrize("consumers", [1, 3])
    def test_no_reading_is_lost_and_no_state_is_reset(
        self,
        keyed_broker: InMemoryBroker,
        spans: dict[str, SensorSpan],
        streaming: StreamingParams,
        frame: pd.DataFrame,
        consumers: int,
    ) -> None:
        """Every reading comes out validated and an ordered stream resets nothing.

        Un reinicio de estado en un stream continuo y ordenado seria un falso
        retroceso de timestamp o un cambio de run espurio: borraria la tirada de
        stuck del sensor.
        """
        result = consume_group(keyed_broker, spans, streaming, consumers)

        assert result.validated == len(frame)
        assert result.resets == {reason: 0.0 for reason in RESET_REASONS}


# ---------------------------------------------------------------------------
# The mirror: without the key the invariant breaks, silently
# ---------------------------------------------------------------------------


class TestKeylessMirror:
    """El mismo escenario sin clave: debe fallar, y fallar sin ningun error.

    MEDIDO, y distinto de lo que se supuso: sin clave el stuck de XMV-04 NO desaparece.
    Con 52 sensores y 12 particiones, el reparto round-robin manda cada sensor a solo 3
    particiones (52 mod 12 = 4) y una senal CONSTANTE sigue acumulando tirada en cada
    una; se pierden unas 7 detecciones por flujo adicional (459 de 473 con 3
    consumidores). Lo que la clave protege es el orden: sin ella el timestamp retrocede
    y los detectores con memoria (Kalman, tasa de cambio) ven una serie distinta. De
    ahi que el espejo se compare contra TODAS las alertas.
    """

    @pytest.mark.parametrize("consumers", [1, 3])
    def test_round_robin_changes_what_is_detected_without_any_error(
        self,
        keyless_broker: InMemoryBroker,
        spans: dict[str, SensorSpan],
        streaming: StreamingParams,
        frame: pd.DataFrame,
        offline_faults: set[FaultKey],
        expected_pairs: int,
        consumers: int,
    ) -> None:
        """Without the sensor_id key the detections differ from the offline ones.

        Sin excepcion y con todas las lecturas validadas: lo unico que cambia es el
        conjunto de alertas.
        """
        result = consume_group(keyless_broker, spans, streaming, consumers)

        assert result.validated == len(frame)
        assert len(result.stuck.get(_FROZEN_TAG, set())) < expected_pairs
        assert result.faults != offline_faults
        assert len(offline_faults - result.faults) > 0
        assert len(result.faults - offline_faults) > 0

    def test_round_robin_with_one_consumer_breaks_timestamp_order(
        self,
        keyless_broker: InMemoryBroker,
        spans: dict[str, SensorSpan],
        streaming: StreamingParams,
    ) -> None:
        """A single consumer reads the partitions one after another and sees time go back."""
        result = consume_group(keyless_broker, spans, streaming, 1)

        assert result.resets["time_regression"] > 0
