"""Tests for the pure parts of the Phase 2 benchmarks (tests/integration/benchmark_*.py).

Los benchmarks necesitan Kafka o varios minutos de CPU y no corren en la suite
unitaria; lo que si se comprueba aqui es lo que Verify-Phase2.ps1 va a leer: el
esquema del informe, el sentido de la comparacion con el umbral y las funciones que
convierten medidas en cifras (ventanas de tasa, percentiles, bucle de produccion).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data.streaming.errors import PublishError
from data.streaming.kafka_settings import KafkaConnectionSettings
from data.streaming.streaming_params import load_streaming_params
from tests.integration import benchmark_feature_latency as feature_bench
from tests.integration import benchmark_kafka as kafka_bench
from tests.integration import benchmark_report as report_tools
from tests.support.in_memory_broker import InMemoryBroker


class FakeClock:
    """A clock that advances a fixed step every time it is read."""

    def __init__(self, step: float) -> None:
        self._now = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self._now
        self._now += self._step
        return value


def _report(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "criterion": "kafka_throughput",
        "environment": "local",
        "value": 100.0,
        "unit": "messages_per_second",
        "threshold": 50.0,
        "direction": "at_least",
        "details": {"k": 1},
        "commit": "abc1234",
        "measured_at": "2026-10-07T12:00:00Z",
    }
    arguments.update(overrides)
    return report_tools.build_report(**arguments)


class TestReportFormat:
    """El esquema que lee Verify-Phase2.ps1."""

    def test_report_has_every_documented_field(self) -> None:
        """The document carries exactly the schema of the module docstring."""
        assert set(_report()) == {
            "schema_version", "criterion", "environment", "measured_at", "commit",
            "value", "unit", "threshold", "direction", "passed", "details",
        }

    def test_values_are_recorded_as_given(self) -> None:
        document = _report()
        assert document["schema_version"] == report_tools.SCHEMA_VERSION
        assert document["environment"] == "local"
        assert document["commit"] == "abc1234"
        assert document["measured_at"] == "2026-10-07T12:00:00Z"

    @pytest.mark.parametrize(
        ("value", "threshold", "direction", "expected"),
        [
            (50.0, 50.0, "at_least", True),
            (49.9, 50.0, "at_least", False),
            (100.0, 100.0, "at_most", True),
            (100.1, 100.0, "at_most", False),
        ],
    )
    def test_direction_decides_how_the_threshold_is_read(
        self, value: float, threshold: float, direction: str, expected: bool
    ) -> None:
        """Throughput must reach the threshold, latency must not exceed it."""
        assert report_tools.meets_threshold(value, threshold, direction) is expected
        document = _report(value=value, threshold=threshold, direction=direction)
        assert document["passed"] is expected

    def test_unknown_direction_is_refused(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            report_tools.meets_threshold(1.0, 1.0, "sideways")

    def test_unknown_environment_is_refused(self) -> None:
        """A typo must not silently become a cluster measurement."""
        with pytest.raises(ValueError, match="environment"):
            _report(environment="prod")

    def test_defaults_resolve_commit_and_time(self) -> None:
        document = report_tools.build_report(
            criterion="c", environment="cluster", value=1.0, unit="u", threshold=1.0,
            direction="at_least", details={},
        )
        assert document["measured_at"].endswith("Z")
        assert isinstance(document["commit"], str)
        assert document["commit"]

    def test_write_report_creates_the_directory_and_round_trips(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "result.json"
        report_tools.write_report(target, _report())
        assert json.loads(target.read_text(encoding="utf-8")) == _report()

    def test_commit_is_unknown_outside_a_repository(self, tmp_path: Path) -> None:
        assert report_tools.git_commit(tmp_path) == "unknown"

    def test_commit_is_unknown_when_git_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _missing(*args: Any, **kwargs: Any) -> Any:
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", _missing)
        assert report_tools.git_commit() == "unknown"


class TestWindowedRates:
    def test_rate_per_closed_window(self) -> None:
        marks = [(0.5, 500), (1.0, 1100), (1.5, 1500), (2.0, 2300)]
        assert kafka_bench.windowed_rates(marks, window_s=1.0) == [1100.0, 1200.0]

    def test_trailing_partial_window_is_dropped(self) -> None:
        assert kafka_bench.windowed_rates([(0.4, 400), (0.8, 900)], window_s=1.0) == []


class TestRunPhase:
    """El bucle de produccion cronometrado."""

    def _payloads(self) -> list[kafka_bench.Payload]:
        return [(b"s1", b"aaaa"), (b"s2", b"bbbbbb")]

    def test_counts_only_confirmed_messages_and_stops_on_time(self) -> None:
        broker = InMemoryBroker(12)
        result = kafka_bench.run_phase(
            broker, "t", self._payloads(), (("run-id", b"x"),),
            duration_s=0.05, flush_every=10, flush_timeout_s=1.0, clock=FakeClock(0.01),
        )
        assert result.messages % 10 == 0
        assert result.messages == len(broker.sent)
        assert result.flushes == result.messages // 10
        assert result.payload_bytes == sum(len(m.value) for m in broker.sent)
        assert result.elapsed_s >= 0.05

    def test_messages_cycle_through_the_payloads_with_header(self) -> None:
        broker = InMemoryBroker(12)
        kafka_bench.run_phase(
            broker, "t", self._payloads(), (("run-id", b"x"),),
            duration_s=0.01, flush_every=4, flush_timeout_s=None, clock=FakeClock(0.01),
        )
        assert [m.key for m in broker.sent[:4]] == [b"s1", b"s2", b"s1", b"s2"]
        assert all(m.headers == (("run-id", b"x"),) for m in broker.sent)

    def test_rates_and_megabytes(self) -> None:
        result = kafka_bench.ThroughputResult(
            messages=2000, payload_bytes=2 * 1024 * 1024, elapsed_s=2.0, flushes=2,
            window_rates=(1000.0,),
        )
        assert result.messages_per_second == 1000.0
        assert result.megabytes_per_second == 1.0

    def test_zero_elapsed_gives_zero_rates(self) -> None:
        result = kafka_bench.ThroughputResult(0, 0, 0.0, 0, ())
        assert result.messages_per_second == 0.0
        assert result.megabytes_per_second == 0.0

    def test_failed_flush_is_not_a_measurement(self) -> None:
        broker = InMemoryBroker(12)
        broker.fail_flush_with = PublishError("Simulated delivery failure.")
        with pytest.raises(PublishError):
            kafka_bench.run_phase(
                broker, "t", self._payloads(), (), duration_s=1.0, flush_every=2,
                flush_timeout_s=None, clock=FakeClock(0.01),
            )

    @pytest.mark.parametrize(("duration", "flush_every"), [(0.0, 10), (1.0, 0)])
    def test_non_positive_parameters_are_refused(self, duration: float, flush_every: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            kafka_bench.run_phase(
                InMemoryBroker(12), "t", self._payloads(), (), duration_s=duration,
                flush_every=flush_every, flush_timeout_s=None,
            )

    def test_empty_payloads_are_refused(self) -> None:
        with pytest.raises(ValueError, match="payloads"):
            kafka_bench.run_phase(
                InMemoryBroker(12), "t", [], (), duration_s=1.0, flush_every=1,
                flush_timeout_s=None,
            )


class TestKafkaReport:
    def test_report_is_a_local_below_threshold_figure(self) -> None:
        result = kafka_bench.ThroughputResult(
            messages=140_000, payload_bytes=57_540_000, elapsed_s=10.0, flushes=27,
            window_rates=(13_000.0, 15_000.0, 14_000.0),
        )
        streaming = load_streaming_params(Path("params.yaml"))
        document = kafka_bench.make_report(
            result, environment="local", streaming=streaming, details={"topic": "t"}
        )
        assert document["criterion"] == "kafka_throughput"
        assert document["value"] == 14_000.0
        assert document["threshold"] == 50_000.0
        assert document["passed"] is False
        assert document["details"]["window_rates"] == {
            "windows": 3, "min": 13_000.0, "median": 14_000.0, "max": 15_000.0,
        }
        assert document["details"]["producer"]["compression_type"] == streaming.compression_type
        assert document["details"]["topic"] == "t"

    def test_no_windows_reports_none(self) -> None:
        result = kafka_bench.ThroughputResult(10, 10, 1.0, 1, ())
        streaming = load_streaming_params(Path("params.yaml"))
        document = kafka_bench.make_report(
            result, environment="cluster", streaming=streaming, details={}
        )
        assert document["details"]["window_rates"]["min"] is None
        assert document["environment"] == "cluster"

    def test_payloads_need_the_parquet(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Invoke-Pipeline"):
            kafka_bench.build_payloads(tmp_path / "missing.parquet", Path("params.yaml"))

    def test_topic_name_is_required_unless_created(self) -> None:
        settings = KafkaConnectionSettings(bootstrap_servers=("127.0.0.1:9092",))
        with pytest.raises(ValueError, match="topic name"), kafka_bench.benchmark_topic(
            settings, None, create=False
        ):
            pass

    def test_existing_topic_is_used_as_is(self) -> None:
        settings = KafkaConnectionSettings(bootstrap_servers=("127.0.0.1:9092",))
        with kafka_bench.benchmark_topic(settings, "sensor-readings-raw", create=False) as topic:
            assert topic == "sensor-readings-raw"

    def test_defaults_of_the_command_line(self) -> None:
        args = kafka_bench.parse_args([])
        assert args.environment == "local"
        assert args.output == kafka_bench.DEFAULT_OUTPUT
        assert args.create_topic is False
        assert args.fail_below_threshold is False


class TestFeatureLatencyTools:
    def test_percentile_interpolates(self) -> None:
        assert feature_bench.percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
        assert feature_bench.percentile([5.0], 99.0) == 5.0
        assert feature_bench.percentile([1.0, 3.0], 100.0) == 3.0

    def test_percentile_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            feature_bench.percentile([], 50.0)
        with pytest.raises(ValueError, match="within"):
            feature_bench.percentile([1.0], 101.0)

    def test_summary_orders_before_ranking(self) -> None:
        summary = feature_bench.summarize([4.0, 1.0, 3.0, 2.0])
        assert summary["count"] == 4.0
        assert summary["p50_ms"] == 2.5
        assert summary["max_ms"] == 4.0
        assert summary["mean_ms"] == 2.5
        assert summary["p99_ms"] == pytest.approx(3.97)

    def _frame(self, steps: int) -> pd.DataFrame:
        rows = [
            {"timestep": step, "sensor_id": tag, "value": float(step),
             "timestamp": pd.Timestamp("2000-01-01") + pd.Timedelta(minutes=3 * step)}
            for step in range(steps)
            for tag in ("A", "B")
        ]
        return pd.DataFrame(rows)

    def test_sliding_windows_step_by_one(self) -> None:
        windows = list(feature_bench.sliding_windows(self._frame(6), 4))
        assert len(windows) == 3
        values, stamps = windows[1]
        assert list(values.index) == [1, 2, 3, 4]
        assert list(stamps.index) == [1, 2, 3, 4]
        assert list(values.columns) == ["A", "B"]

    def test_run_shorter_than_a_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="fewer than the window"):
            list(feature_bench.sliding_windows(self._frame(3), 4))

    def test_time_transform_discards_warmup_and_times_each_call(self) -> None:
        calls: list[int] = []

        class _Pipeline:
            def transform(self, values: pd.DataFrame, stamps: pd.Series, types: Any) -> None:
                calls.append(len(values))

        windows = list(feature_bench.sliding_windows(self._frame(6), 4))
        latencies = feature_bench.time_transform(
            _Pipeline(), windows, {}, repeats=2, warmup_calls=5  # type: ignore[arg-type]
        )
        assert len(latencies) == 6
        assert len(calls) == 11
        assert all(value >= 0.0 for value in latencies)

    def test_time_transform_needs_windows_and_repeats(self) -> None:
        with pytest.raises(ValueError, match="windows"):
            feature_bench.time_transform(
                None, [], {}, repeats=1, warmup_calls=0  # type: ignore[arg-type]
            )

    def test_defaults_of_the_command_line(self) -> None:
        args = feature_bench.parse_args([])
        assert args.load_workers == 0
        assert args.environment == "local"
        assert args.output == feature_bench.DEFAULT_OUTPUT
