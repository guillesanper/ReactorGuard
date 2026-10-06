"""Tests for data/streaming/streaming_params.py."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.generators.tep_params import load_tep_params
from data.streaming.streaming_params import load_streaming_params

WriteSection = Callable[..., Path]


@pytest.fixture()
def write_section(tmp_path: Path) -> WriteSection:
    """Return a factory writing a valid streaming: section with overrides."""

    def _write(**overrides: Any) -> Path:
        section: dict[str, Any] = {
            "raw_topic": "raw",
            "validated_topic": "validated",
            "alerts_topic": "alerts",
            "consumer_group": "group",
            "poll_timeout_ms": 1000,
            "max_poll_records": 500,
            "auto_offset_reset": "earliest",
            "isolation_level": "read_uncommitted",
            "commit_retries": 3,
            "commit_timeout_ms": 30000,
            "close_timeout_ms": 5000,
            "max_poll_interval_ms": 300000,
            "compression_type": "lz4",
            "batch_linger_ms": 10,
            "producer_batch_bytes": 131072,
            "max_in_flight_requests": 5,
            "request_timeout_ms": 30000,
            "delivery_timeout_ms": 120000,
            "flush_timeout_s": 30.0,
            "bind_host": "127.0.0.1",
            "health_port": 8000,
            "metrics_port": 9090,
            "staleness_seconds": 540,
        }
        section.update(overrides)
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"streaming": section}), encoding="utf-8")
        return path

    return _write


class TestLoad:
    def test_loads_every_field(self, write_section: WriteSection) -> None:
        params = load_streaming_params(write_section())
        assert params.raw_topic == "raw"
        assert params.consumer_group == "group"
        assert params.max_poll_records == 500
        assert params.compression_type == "lz4"
        assert params.flush_timeout_s == 30.0
        assert params.health_port == 8000
        assert params.metrics_port == 9090

    def test_is_frozen(self, write_section: WriteSection) -> None:
        params = load_streaming_params(write_section())
        with pytest.raises(AttributeError):
            params.raw_topic = "other"  # type: ignore[misc]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_streaming_params(tmp_path / "absent.yaml")

    def test_missing_section(self, tmp_path: Path) -> None:
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"tep": {}}), encoding="utf-8")
        with pytest.raises(KeyError, match="streaming:"):
            load_streaming_params(path)

    def test_empty_file_reports_missing_section(self, tmp_path: Path) -> None:
        path = tmp_path / "params.yaml"
        path.write_text("", encoding="utf-8")
        with pytest.raises(KeyError, match="streaming:"):
            load_streaming_params(path)

    def test_missing_key_is_named(self, tmp_path: Path, write_section: WriteSection) -> None:
        path = write_section()
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        del document["streaming"]["flush_timeout_s"]
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        with pytest.raises(KeyError, match=r"streaming\.flush_timeout_s"):
            load_streaming_params(path)


class TestValidation:
    @pytest.mark.parametrize(
        "key",
        [
            "poll_timeout_ms",
            "max_poll_records",
            "commit_timeout_ms",
            "close_timeout_ms",
            "max_poll_interval_ms",
            "producer_batch_bytes",
            "max_in_flight_requests",
            "request_timeout_ms",
            "delivery_timeout_ms",
        ],
    )
    def test_positive_int_rejects_zero(self, write_section: WriteSection, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            load_streaming_params(write_section(**{key: 0}))

    @pytest.mark.parametrize("key", ["commit_retries", "batch_linger_ms"])
    def test_non_negative_int_rejects_negative(
        self, write_section: WriteSection, key: str
    ) -> None:
        with pytest.raises(ValueError, match=key):
            load_streaming_params(write_section(**{key: -1}))

    def test_zero_retries_and_linger_are_valid(self, write_section: WriteSection) -> None:
        params = load_streaming_params(write_section(commit_retries=0, batch_linger_ms=0))
        assert params.commit_retries == 0
        assert params.batch_linger_ms == 0

    @pytest.mark.parametrize("key", ["flush_timeout_s", "staleness_seconds"])
    def test_positive_float_rejects_zero(self, write_section: WriteSection, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            load_streaming_params(write_section(**{key: 0}))

    @pytest.mark.parametrize("key", ["raw_topic", "consumer_group", "bind_host"])
    def test_blank_string_rejected(self, write_section: WriteSection, key: str) -> None:
        with pytest.raises(ValueError, match=key):
            load_streaming_params(write_section(**{key: "  "}))

    @pytest.mark.parametrize("port", [0, 65536, -5])
    def test_port_out_of_range(self, write_section: WriteSection, port: int) -> None:
        with pytest.raises(ValueError, match="health_port"):
            load_streaming_params(write_section(health_port=port))

    def test_ports_must_differ(self, write_section: WriteSection) -> None:
        with pytest.raises(ValueError, match="must differ"):
            load_streaming_params(write_section(health_port=9090, metrics_port=9090))

    def test_topics_must_be_distinct(self, write_section: WriteSection) -> None:
        with pytest.raises(ValueError, match="distinct"):
            load_streaming_params(write_section(alerts_topic="raw"))

    def test_bad_auto_offset_reset(self, write_section: WriteSection) -> None:
        with pytest.raises(ValueError, match="auto_offset_reset"):
            load_streaming_params(write_section(auto_offset_reset="middle"))

    def test_bad_isolation_level(self, write_section: WriteSection) -> None:
        with pytest.raises(ValueError, match="isolation_level"):
            load_streaming_params(write_section(isolation_level="serializable"))

    @pytest.mark.parametrize("raw", [None, "none", "NONE"])
    def test_compression_disabled(self, write_section: WriteSection, raw: Any) -> None:
        assert load_streaming_params(write_section(compression_type=raw)).compression_type is None

    def test_unknown_compression(self, write_section: WriteSection) -> None:
        with pytest.raises(ValueError, match="compression_type"):
            load_streaming_params(write_section(compression_type="brotli"))

    def test_in_flight_above_ordering_limit(self, write_section: WriteSection) -> None:
        with pytest.raises(ValueError, match="ordering"):
            load_streaming_params(write_section(max_in_flight_requests=6))

    def test_delivery_must_cover_linger_and_request(self, write_section: WriteSection) -> None:
        with pytest.raises(ValueError, match="delivery_timeout_ms"):
            load_streaming_params(
                write_section(
                    batch_linger_ms=10, request_timeout_ms=30000, delivery_timeout_ms=30009
                )
            )

    def test_delivery_exactly_linger_plus_request_is_valid(
        self, write_section: WriteSection
    ) -> None:
        params = load_streaming_params(
            write_section(batch_linger_ms=10, request_timeout_ms=30000, delivery_timeout_ms=30010)
        )
        assert params.delivery_timeout_ms == 30010

    def test_worst_case_batch_must_fit_poll_interval(self, write_section: WriteSection) -> None:
        # 30 s de flush + 4 * 30 s de commit = 150 s; con 150 s de intervalo no cabe.
        with pytest.raises(ValueError, match="max_poll_interval_ms"):
            load_streaming_params(write_section(max_poll_interval_ms=150000))

    def test_worst_case_property(self, write_section: WriteSection) -> None:
        params = load_streaming_params(write_section())
        assert params.worst_case_batch_seconds == pytest.approx(150.0)


class TestRealParamsFile:
    """Invariantes de la seccion streaming: del params.yaml del repo."""

    def test_real_section_loads(self) -> None:
        params = load_streaming_params()
        assert params.consumer_group == "sensor-validator"
        assert params.compression_type == "lz4"

    def test_staleness_tolerates_two_sample_intervals(self) -> None:
        # El silencio legitimo mas largo es la espera de 180 s del streamer en
        # realtime. Con menos de dos intervalos /health reiniciaria el pod en
        # cada espera o al perder un solo lote.
        interval_s = load_tep_params().sample_interval_minutes * 60
        assert load_streaming_params().staleness_seconds >= 2 * interval_s

    def test_worst_case_batch_below_poll_interval_with_margin(self) -> None:
        params = load_streaming_params()
        assert params.worst_case_batch_seconds * 1000 <= params.max_poll_interval_ms / 2
