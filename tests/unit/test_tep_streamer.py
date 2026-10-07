"""Tests for data/generators/tep_streamer.py.

Reloj y sleeper inyectados: ningun test duerme de verdad (salvo el que comprueba que
stop() despierta al sleeper por defecto). Los fallos del productor se simulan con
fakes en memoria. El invariante D1 (clave = sensor_id, todas las lecturas de un
sensor en una particion y en orden) se comprueba contra el reparto murmur2 real.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.generators.tep_adapter import readings_from_frame
from data.generators.tep_streamer import TEPStreamer, format_run_id, timestep_slices
from data.generators.tep_streamer_params import StreamerParams, StreamMode, load_streamer_params
from data.schemas.sensor_reading import SensorReading
from data.streaming.errors import PublishError
from data.streaming.metrics import (
    STREAM_MESSAGES_PRODUCED,
    STREAM_PRODUCE_ERRORS,
    STREAM_PRODUCE_LATENCY,
    StreamMetrics,
)
from data.streaming.transport import HEADER_RUN_ID, Header, MessageProducer
from tests.support.in_memory_broker import InMemoryBroker
from tests.support.tep_frames import make_run_frame

TOPIC = "sensor-readings-raw"
SENSORS = 52
REAL_FAULT_21 = Path("data/processed/tep/fault_type=21/readings.parquet")


class FakeClock:
    """Monotonic clock advanced only by the fake sleeper and the fake producer."""

    def __init__(self, overshoot: float = 0.0) -> None:
        self.now = 1000.0
        self.overshoot = overshoot
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds + self.overshoot


class FakeProducer:
    """Records every call, can advance a clock and can be told to fail."""

    def __init__(
        self,
        clock: FakeClock | None = None,
        per_message_s: float = 0.0,
        flush_s: float = 0.0,
    ) -> None:
        self.clock = clock
        self.per_message_s = per_message_s
        self.flush_s = flush_s
        self.sent: list[tuple[str, bytes | None, bytes, tuple[Header, ...]]] = []
        self.sent_at_flush: list[int] = []
        self.flush_timeouts: list[float | None] = []
        self.on_send: Callable[[int], None] | None = None
        self.send_error_at: int | None = None
        self.flush_error: PublishError | None = None

    def send(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: Sequence[Header] = (),
    ) -> None:
        if self.send_error_at is not None and len(self.sent) >= self.send_error_at:
            raise PublishError("queue full")
        self.sent.append((topic, key, value, tuple(headers)))
        if self.clock is not None:
            self.clock.now += self.per_message_s
        if self.on_send is not None:
            self.on_send(len(self.sent))

    def flush(self, timeout: float | None = None) -> None:
        self.flush_timeouts.append(timeout)
        self.sent_at_flush.append(len(self.sent))
        if self.clock is not None:
            self.clock.now += self.flush_s
        if self.flush_error is not None:
            raise self.flush_error

    def close(self, timeout: float | None = None) -> None:
        del timeout


class ListSource:
    """ReadingSource over frames held in memory; counts how often it is iterated."""

    def __init__(self, runs: list[tuple[str, pd.DataFrame]]) -> None:
        self._runs = runs
        self.calls = 0

    def runs(self) -> Iterator[tuple[str, pd.DataFrame]]:
        self.calls += 1
        return iter(self._runs)


def _params(**changes: Any) -> StreamerParams:
    return dataclasses.replace(load_streamer_params(), **changes)


def _run(fault_type: int, timesteps: int) -> tuple[str, pd.DataFrame]:
    return f"fault_type={fault_type:02d}", make_run_frame(fault_type, timesteps)


def _decode(producer: FakeProducer) -> list[SensorReading]:
    return [SensorReading.from_kafka_bytes(value) for _, _, value, _ in producer.sent]


def _streamer(
    source: ListSource,
    producer: MessageProducer,
    params: StreamerParams | None = None,
    *,
    clock: FakeClock | None = None,
    metrics: StreamMetrics | None = None,
    on_tick: Callable[[], None] | None = None,
    flush_timeout_s: float | None = None,
) -> TEPStreamer:
    fake_clock = clock if clock is not None else FakeClock()
    return TEPStreamer(
        producer,
        source,
        params if params is not None else _params(),
        metrics if metrics is not None else StreamMetrics(),
        topic=TOPIC,
        flush_timeout_s=flush_timeout_s,
        clock=fake_clock,
        sleeper=fake_clock.sleep,
        on_tick=on_tick,
    )


class TestTimestepSlices:
    def test_splits_contiguous_timesteps(self) -> None:
        frame = make_run_frame(0, 4)
        slices = list(timestep_slices(frame))
        assert [len(piece) for piece in slices] == [SENSORS] * 4
        assert [int(piece["timestep"].iloc[0]) for piece in slices] == [0, 1, 2, 3]
        assert pd.concat(slices).equals(frame)

    def test_uneven_groups_keep_their_size(self) -> None:
        frame = make_run_frame(0, 3).drop(index=[0, 1, 2])
        assert [len(piece) for piece in timestep_slices(frame)] == [SENSORS - 3, SENSORS, SENSORS]

    def test_empty_frame_yields_nothing(self) -> None:
        assert list(timestep_slices(make_run_frame(0, 1).iloc[0:0])) == []

    def test_missing_column_raises(self) -> None:
        with pytest.raises(KeyError, match="timestep"):
            list(timestep_slices(make_run_frame(0, 1).drop(columns=["timestep"])))

    def test_timesteps_going_backwards_raise(self) -> None:
        frame = make_run_frame(0, 3).iloc[::-1]
        with pytest.raises(ValueError, match="non-decreasing"):
            list(timestep_slices(frame))


class TestFormatRunId:
    def test_combines_run_and_loop(self) -> None:
        assert format_run_id("fault_type=21", 0) == "fault_type=21/loop=0"
        assert format_run_id("fault_type=00", 3) == "fault_type=00/loop=3"


class TestEmission:
    def test_n_rows_become_n_messages_on_the_raw_topic(self) -> None:
        run = _run(0, 5)
        producer = FakeProducer()
        _streamer(ListSource([run]), producer).run()
        assert len(producer.sent) == len(run[1]) == 5 * SENSORS
        assert {topic for topic, *_ in producer.sent} == {TOPIC}

    def test_key_is_the_sensor_id_in_every_message(self) -> None:
        producer = FakeProducer()
        _streamer(ListSource([_run(0, 3)]), producer).run()
        readings = _decode(producer)
        assert all(key == reading.sensor.id.encode("utf-8")
                   for (_, key, _, _), reading in zip(producer.sent, readings, strict=True))
        assert len({key for _, key, _, _ in producer.sent}) == SENSORS

    def test_every_message_carries_the_run_id_header(self) -> None:
        producer = FakeProducer()
        _streamer(ListSource([_run(7, 2)]), producer).run()
        expected = ((HEADER_RUN_ID, b"fault_type=07/loop=0"),)
        assert all(headers == expected for _, _, _, headers in producer.sent)

    def test_messages_are_the_frame_rows_in_order(self) -> None:
        run = _run(3, 4)
        producer = FakeProducer()
        _streamer(ListSource([run]), producer).run()
        expected = [str(reading.reading_id) for reading in readings_from_frame(run[1])]
        assert [str(reading.reading_id) for reading in _decode(producer)] == expected

    def test_a_timestep_is_complete_before_the_next_one_starts(self) -> None:
        producer = FakeProducer()
        _streamer(ListSource([_run(0, 6)]), producer).run()
        stamps = [reading.timestamp for reading in _decode(producer)]
        for start in range(0, len(stamps), SENSORS):
            assert len(set(stamps[start : start + SENSORS])) == 1
        assert stamps == sorted(stamps)

    def test_runs_are_streamed_one_after_another_never_interleaved(self) -> None:
        producer = FakeProducer()
        _streamer(ListSource([_run(0, 3), _run(1, 3), _run(2, 3)]), producer).run()
        run_ids = [headers[0][1] for _, _, _, headers in producer.sent]
        collapsed = [run_ids[0]]
        for run_id in run_ids[1:]:
            if run_id != collapsed[-1]:
                collapsed.append(run_id)
        assert collapsed == [
            b"fault_type=00/loop=0",
            b"fault_type=01/loop=0",
            b"fault_type=02/loop=0",
        ]

    def test_the_timestamp_clock_restarts_on_every_run(self) -> None:
        producer = FakeProducer()
        _streamer(ListSource([_run(0, 2), _run(1, 2)]), producer).run()
        stamps = [reading.timestamp for reading in _decode(producer)]
        assert stamps[4 * SENSORS // 2] == stamps[0]

    def test_the_metadata_comes_from_the_params(self) -> None:
        producer = FakeProducer()
        _streamer(ListSource([_run(0, 1)]), producer, _params(drift_coefficient=0.25)).run()
        assert {reading.metadata.drift_coefficient for reading in _decode(producer)} == {0.25}

    def test_an_invalid_row_publishes_nothing_of_its_timestep(self) -> None:
        frame = make_run_frame(0, 2)
        frame.loc[SENSORS + 30, "quality"] = "bogus"
        producer = FakeProducer()
        streamer = _streamer(ListSource([("fault_type=00", frame)]), producer)
        with pytest.raises(ValueError, match="bogus"):
            streamer.run()
        assert len(producer.sent) == SENSORS

    def test_run_can_only_be_called_once(self) -> None:
        streamer = _streamer(ListSource([_run(0, 1)]), FakeProducer())
        streamer.run()
        with pytest.raises(RuntimeError, match="only be called once"):
            streamer.run()

    def test_a_source_without_timesteps_is_an_error_even_with_loop(self) -> None:
        empty = make_run_frame(0, 1).iloc[0:0]
        for runs in ([], [("fault_type=00", empty)]):
            streamer = _streamer(ListSource(runs), FakeProducer(), _params(loop=True))
            with pytest.raises(ValueError, match="no timesteps"):
                streamer.run()


class TestFastMode:
    def test_never_sleeps(self) -> None:
        clock = FakeClock()
        _streamer(ListSource([_run(0, 20)]), FakeProducer(), _params(mode=StreamMode.FAST),
                  clock=clock).run()
        assert clock.sleeps == []

    def test_flushes_every_n_timesteps_and_at_the_end_of_the_run(self) -> None:
        producer = FakeProducer()
        params = _params(mode=StreamMode.FAST, flush_every_timesteps=3)
        _streamer(ListSource([_run(0, 7)]), producer, params).run()
        assert producer.sent_at_flush == [3 * SENSORS, 6 * SENSORS, 7 * SENSORS]

    def test_flushes_at_the_end_of_every_run(self) -> None:
        producer = FakeProducer()
        params = _params(mode=StreamMode.FAST, flush_every_timesteps=1000)
        _streamer(ListSource([_run(0, 2), _run(1, 2)]), producer, params).run()
        assert producer.sent_at_flush == [2 * SENSORS, 4 * SENSORS]

    def test_the_final_flush_covers_everything_and_gets_the_timeout(self) -> None:
        producer = FakeProducer()
        _streamer(ListSource([_run(0, 2)]), producer, flush_timeout_s=12.5).run()
        assert producer.sent_at_flush[-1] == len(producer.sent)
        assert set(producer.flush_timeouts) == {12.5}


class TestRealtimeMode:
    def test_sleeps_exactly_interval_over_speed_between_batches(self) -> None:
        clock = FakeClock()
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=60.0, wait_slice_s=30.0)
        _streamer(ListSource([_run(0, 4)]), FakeProducer(), params, clock=clock).run()
        assert clock.sleeps == [3.0, 3.0, 3.0]

    def test_the_interval_is_the_tep_cadence_of_180_seconds(self) -> None:
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=1.0, wait_slice_s=1000.0)
        assert params.step_interval_s == 180.0
        clock = FakeClock()
        _streamer(ListSource([_run(0, 3)]), FakeProducer(), params, clock=clock).run()
        assert clock.sleeps == [180.0, 180.0]

    def test_the_first_batch_leaves_without_waiting(self) -> None:
        clock = FakeClock()
        producer = FakeProducer(clock=clock)
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=60.0)
        _streamer(ListSource([_run(0, 1)]), producer, params, clock=clock).run()
        assert clock.sleeps == []
        assert len(producer.sent) == SENSORS

    def test_a_long_wait_is_cut_in_slices_that_add_up_to_the_interval(self) -> None:
        clock = FakeClock()
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=1.0, wait_slice_s=30.0)
        _streamer(ListSource([_run(0, 3)]), FakeProducer(), params, clock=clock).run()
        assert clock.sleeps == [30.0] * 12
        assert sum(clock.sleeps) == 2 * 180.0

    def test_an_interval_that_is_not_representable_adds_no_spurious_sleep(self) -> None:
        clock = FakeClock()
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=7.0, wait_slice_s=1000.0)
        _streamer(ListSource([_run(0, 6)]), FakeProducer(), params, clock=clock).run()
        assert len(clock.sleeps) == 5
        assert sum(clock.sleeps) == pytest.approx(5 * 180.0 / 7.0)

    def test_each_batch_is_flushed_before_the_wait(self) -> None:
        producer = FakeProducer()
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=60.0)
        _streamer(ListSource([_run(0, 4)]), producer, params).run()
        assert producer.sent_at_flush == [SENSORS * step for step in (1, 2, 3, 4)]

    def test_overshoot_and_processing_time_do_not_accumulate_as_drift(self) -> None:
        interval = 10.0
        timesteps = 20
        clock = FakeClock(overshoot=0.3)
        producer = FakeProducer(clock=clock, per_message_s=0.001, flush_s=0.05)
        params = _params(
            mode=StreamMode.REALTIME, speed_multiplier=180.0 / interval, wait_slice_s=1000.0
        )
        start = clock.now
        _streamer(ListSource([_run(0, timesteps)]), producer, params, clock=clock).run()
        elapsed = clock.now - start
        # Con `sleep(interval)` tras cada lote el total seria timesteps * (interval + 0,4):
        # unos 8 s de deriva. Con plazos absolutos solo queda el sobresueo del ultimo.
        assert (timesteps - 1) * interval <= elapsed < (timesteps - 1) * interval + 0.5

    def test_a_late_batch_is_caught_up_without_sleeping(self) -> None:
        clock = FakeClock()
        producer = FakeProducer(clock=clock, per_message_s=0.1)
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=60.0)
        streamer = _streamer(ListSource([_run(0, 4)]), producer, params, clock=clock)
        streamer.run()
        assert clock.sleeps == []
        assert streamer.get_metrics()["late_timesteps"] == 3.0

    def test_the_heartbeat_ticks_on_every_slice_and_every_batch(self) -> None:
        ticks: list[int] = []
        clock = FakeClock()
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=1.0, wait_slice_s=30.0)
        _streamer(
            ListSource([_run(0, 3)]), FakeProducer(), params, clock=clock,
            on_tick=lambda: ticks.append(1),
        ).run()
        assert len(ticks) == 12 + 3

    def test_no_wait_is_longer_than_the_slice(self) -> None:
        clock = FakeClock()
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=0.5, wait_slice_s=30.0)
        _streamer(ListSource([_run(0, 2)]), FakeProducer(), params, clock=clock).run()
        assert max(clock.sleeps) == 30.0
        assert sum(clock.sleeps) == 360.0


class TestLoop:
    def test_loop_increments_the_loop_number_of_the_run_id(self) -> None:
        producer = FakeProducer()
        streamer_holder: list[TEPStreamer] = []

        def stop_during_third_run(count: int) -> None:
            if count == 4 * SENSORS + 1:
                streamer_holder[0].stop()

        producer.on_send = stop_during_third_run
        source = ListSource([_run(0, 2), _run(1, 2)])
        streamer = _streamer(source, producer, _params(loop=True))
        streamer_holder.append(streamer)
        streamer.run()
        sequence: list[bytes] = []
        for _, _, _, headers in producer.sent:
            if not sequence or sequence[-1] != headers[0][1]:
                sequence.append(headers[0][1])
        assert sequence == [
            b"fault_type=00/loop=0",
            b"fault_type=01/loop=0",
            b"fault_type=00/loop=1",
        ]
        assert source.calls == 2

    def test_without_loop_the_source_is_read_once(self) -> None:
        source = ListSource([_run(0, 2)])
        _streamer(source, FakeProducer(), _params(loop=False)).run()
        assert source.calls == 1


class TestStop:
    def test_stop_before_run_sends_nothing(self) -> None:
        producer = FakeProducer()
        streamer = _streamer(ListSource([_run(0, 5)]), producer)
        streamer.stop()
        streamer.run()
        assert producer.sent == []
        assert producer.sent_at_flush == []

    def test_stop_in_fast_mode_finishes_the_timestep_in_progress(self) -> None:
        producer = FakeProducer()
        holder: list[TEPStreamer] = []
        producer.on_send = lambda count: holder[0].stop() if count == 10 else None
        streamer = _streamer(ListSource([_run(0, 5)]), producer)
        holder.append(streamer)
        streamer.run()
        assert len(producer.sent) == SENSORS
        assert producer.sent_at_flush == [SENSORS]

    def test_stop_interrupts_a_realtime_wait(self) -> None:
        clock = FakeClock()
        producer = FakeProducer()
        holder: list[TEPStreamer] = []

        def sleep_then_stop(seconds: float) -> None:
            clock.sleep(seconds)
            if len(clock.sleeps) == 2:
                holder[0].stop()

        params = _params(mode=StreamMode.REALTIME, speed_multiplier=1.0, wait_slice_s=30.0)
        streamer = TEPStreamer(
            producer, ListSource([_run(0, 10)]), params, StreamMetrics(),
            topic=TOPIC, clock=clock, sleeper=sleep_then_stop,
        )
        holder.append(streamer)
        streamer.run()
        assert len(clock.sleeps) == 2
        assert len(producer.sent) == SENSORS
        assert producer.sent_at_flush[-1] == SENSORS

    def test_stop_wakes_the_default_sleeper_immediately(self) -> None:
        producer = FakeProducer()
        first_batch_sent = threading.Event()
        producer.on_send = lambda count: first_batch_sent.set() if count == SENSORS else None
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=1.0, wait_slice_s=30.0)
        streamer = TEPStreamer(
            producer, ListSource([_run(0, 5)]), params, StreamMetrics(), topic=TOPIC
        )
        worker = threading.Thread(target=streamer.run)
        worker.start()
        assert first_batch_sent.wait(timeout=10.0)
        streamer.stop()
        worker.join(timeout=10.0)
        assert not worker.is_alive()
        assert len(producer.sent) == SENSORS


class TestPublishErrors:
    def test_a_failed_send_propagates_and_is_counted(self) -> None:
        producer = FakeProducer()
        producer.send_error_at = 9
        metrics = StreamMetrics()
        streamer = _streamer(ListSource([_run(0, 3)]), producer, metrics=metrics)
        with pytest.raises(PublishError, match="queue full"):
            streamer.run()
        assert len(producer.sent) == 9
        assert producer.sent_at_flush == []
        assert streamer.get_metrics()["errors"] == 1.0
        assert streamer.get_metrics()["messages"] == 9.0
        assert metrics.registry.get_sample_value(STREAM_PRODUCE_ERRORS) == 1.0
        assert metrics.registry.get_sample_value(
            STREAM_MESSAGES_PRODUCED, {"topic": TOPIC}
        ) == 9.0

    def test_a_failed_flush_propagates_and_streaming_stops(self) -> None:
        producer = FakeProducer()
        producer.flush_error = PublishError("1 of 52 messages failed delivery")
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=60.0)
        streamer = _streamer(ListSource([_run(0, 5)]), producer, params)
        with pytest.raises(PublishError, match="failed delivery"):
            streamer.run()
        assert len(producer.sent) == SENSORS
        assert streamer.get_metrics()["errors"] == 1.0
        assert streamer.get_metrics()["flushes"] == 0.0

    def test_a_failed_final_flush_propagates(self) -> None:
        producer = FakeProducer()
        producer.flush_error = PublishError("timeout")
        streamer = _streamer(ListSource([_run(0, 2)]), producer, _params(mode=StreamMode.FAST))
        with pytest.raises(PublishError, match="timeout"):
            streamer.run()
        assert len(producer.sent) == 2 * SENSORS


class TestMetrics:
    def test_counters_before_running_are_zero(self) -> None:
        metrics = _streamer(ListSource([_run(0, 1)]), FakeProducer()).get_metrics()
        assert set(metrics.values()) == {0.0}

    def test_flat_dict_reports_messages_bytes_latency_and_rate(self) -> None:
        clock = FakeClock()
        producer = FakeProducer(clock=clock, per_message_s=0.01, flush_s=0.2)
        streamer = _streamer(ListSource([_run(0, 4)]), producer, clock=clock)
        streamer.run()
        report = streamer.get_metrics()
        assert report["messages"] == 4 * SENSORS
        assert report["bytes"] == sum(len(value) for _, _, value, _ in producer.sent)
        assert report["errors"] == 0.0
        assert report["timesteps"] == 4.0
        assert report["flushes"] == 1.0
        assert report["late_timesteps"] == 0.0
        assert report["elapsed_s"] == pytest.approx(4 * SENSORS * 0.01 + 0.2)
        assert report["mean_flush_latency_s"] == pytest.approx(4 * SENSORS * 0.01 + 0.2)
        assert report["messages_per_second"] == pytest.approx(4 * SENSORS / report["elapsed_s"])
        assert all(isinstance(value, float) for value in report.values())

    def test_prometheus_metrics_are_fed(self) -> None:
        clock = FakeClock()
        producer = FakeProducer(clock=clock, flush_s=0.5)
        metrics = StreamMetrics()
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=60.0)
        _streamer(ListSource([_run(0, 3)]), producer, params, clock=clock, metrics=metrics).run()
        registry = metrics.registry
        assert registry.get_sample_value(
            STREAM_MESSAGES_PRODUCED, {"topic": TOPIC}
        ) == 3 * SENSORS
        assert registry.get_sample_value(f"{STREAM_PRODUCE_LATENCY}_count") == 3.0
        assert registry.get_sample_value(f"{STREAM_PRODUCE_LATENCY}_sum") == pytest.approx(1.5)
        assert registry.get_sample_value(STREAM_PRODUCE_ERRORS) == 0.0

    def test_mean_latency_after_several_flushes(self) -> None:
        clock = FakeClock()
        producer = FakeProducer(clock=clock, flush_s=0.4)
        params = _params(mode=StreamMode.REALTIME, speed_multiplier=60.0)
        streamer = _streamer(ListSource([_run(0, 5)]), producer, params, clock=clock)
        streamer.run()
        assert streamer.get_metrics()["flushes"] == 5.0
        assert streamer.get_metrics()["mean_flush_latency_s"] == pytest.approx(0.4)

    def test_metrics_can_be_read_while_running(self) -> None:
        clock = FakeClock()
        producer = FakeProducer(clock=clock, per_message_s=0.01)
        seen: list[float] = []
        holder: list[TEPStreamer] = []
        producer.on_send = lambda count: (
            seen.append(holder[0].get_metrics()["elapsed_s"]) if count == SENSORS else None
        )
        streamer = _streamer(ListSource([_run(0, 2)]), producer, clock=clock)
        holder.append(streamer)
        streamer.run()
        assert seen == [pytest.approx(SENSORS * 0.01)]


class TestPartitioningInvariantD1:
    """D1: every reading of a sensor lands in one partition, in order."""

    PARTITIONS = 12

    def _stream(self, broker: InMemoryBroker, runs: list[tuple[str, pd.DataFrame]]) -> None:
        _streamer(ListSource(runs), broker, _params(mode=StreamMode.FAST)).run()

    def _by_sensor(self, broker: InMemoryBroker) -> dict[str, set[int]]:
        placement: dict[str, set[int]] = defaultdict(set)
        for partition, records in broker.partitions_of(TOPIC).items():
            for record in records:
                placement[SensorReading.from_kafka_bytes(record.value or b"").sensor.id].add(
                    partition
                )
        return placement

    def test_every_sensor_lands_in_exactly_one_partition(self) -> None:
        broker = InMemoryBroker(self.PARTITIONS)
        self._stream(broker, [_run(0, 30), _run(1, 30), _run(2, 30)])
        placement = self._by_sensor(broker)
        assert len(placement) == SENSORS
        assert all(len(partitions) == 1 for partitions in placement.values())

    def test_the_sensors_are_spread_over_the_partitions(self) -> None:
        broker = InMemoryBroker(self.PARTITIONS)
        self._stream(broker, [_run(0, 5)])
        assert len(broker.partitions_of(TOPIC)) > 1

    def test_each_sensor_keeps_run_order_and_timestamp_order(self) -> None:
        broker = InMemoryBroker(self.PARTITIONS)
        self._stream(broker, [_run(0, 30), _run(1, 30), _run(2, 30)])
        for records in broker.partitions_of(TOPIC).values():
            offsets = [record.offset for record in records]
            assert offsets == list(range(len(records)))
            last: dict[tuple[str, bytes | None], pd.Timestamp] = {}
            runs_seen: dict[str, list[bytes]] = defaultdict(list)
            for record in records:
                reading = SensorReading.from_kafka_bytes(record.value or b"")
                run_id = record.header(HEADER_RUN_ID) or b""
                sensor = reading.sensor.id
                if not runs_seen[sensor] or runs_seen[sensor][-1] != run_id:
                    runs_seen[sensor].append(run_id)
                previous = last.get((sensor, run_id))
                if previous is not None:
                    assert reading.timestamp > previous
                last[(sensor, run_id)] = pd.Timestamp(reading.timestamp)
            assert all(order == sorted(order) for order in runs_seen.values())
            assert all(len(order) == 3 for order in runs_seen.values())

    def test_without_a_key_the_sensors_scatter_so_the_test_has_teeth(self) -> None:
        class KeylessProducer(InMemoryBroker):
            def send(
                self,
                topic: str,
                key: bytes | None,
                value: bytes,
                headers: Sequence[Header] = (),
            ) -> None:
                super().send(topic, None, value, headers)

        broker = KeylessProducer(self.PARTITIONS)
        self._stream(broker, [_run(0, 30)])
        assert all(len(partitions) > 1 for partitions in self._by_sensor(broker).values())

    @pytest.mark.skipif(
        not REAL_FAULT_21.exists(), reason="data/processed/tep is not populated (run adapt_tep)"
    )
    def test_the_only_real_stuck_sensor_stays_in_one_partition(self) -> None:
        frame = pd.read_parquet(REAL_FAULT_21)
        broker = InMemoryBroker(self.PARTITIONS)
        self._stream(broker, [("fault_type=21", frame)])
        stuck = [
            (partition, SensorReading.from_kafka_bytes(record.value or b""))
            for partition, records in broker.partitions_of(TOPIC).items()
            for record in records
            if record.key == b"TEP-XMV-04"
        ]
        assert len(stuck) == 480
        assert len({partition for partition, _ in stuck}) == 1
        assert len({reading.measurement.value for _, reading in stuck}) == 1
