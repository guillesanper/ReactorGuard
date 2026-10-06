"""Tests for data/streaming/metrics.py."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

from data.streaming import metrics as metrics_module
from data.streaming.metrics import (
    BATCH_SIZE_BUCKETS,
    LATENCY_BUCKETS,
    METRIC_NAMES,
    RESET_REASONS,
    StreamMetrics,
)


def _family_names(stream_metrics: StreamMetrics) -> set[str]:
    return {family.name for family in stream_metrics.registry.collect()}


class TestNames:
    def test_every_name_is_prefixed_and_unique(self) -> None:
        assert len(set(METRIC_NAMES)) == len(METRIC_NAMES)
        assert all(name.startswith("reactorguard_") for name in METRIC_NAMES)

    def test_plan_names_are_all_present(self) -> None:
        expected = {
            "reactorguard_stream_messages_produced_total",
            "reactorguard_stream_produce_errors_total",
            "reactorguard_stream_produce_latency_seconds",
            "reactorguard_readings_consumed_total",
            "reactorguard_readings_validated_total",
            "reactorguard_sensor_faults_total",
            "reactorguard_alerts_published_total",
            "reactorguard_poison_messages_total",
            "reactorguard_validation_latency_seconds",
            "reactorguard_batch_size",
            "reactorguard_consumer_lag",
            "reactorguard_consumer_rebalances_total",
            "reactorguard_detector_state_resets_total",
            "reactorguard_commit_failures_total",
            "reactorguard_consumer_heartbeat_timestamp_seconds",
        }
        assert set(METRIC_NAMES) == expected

    def test_constants_match_registered_metrics(self) -> None:
        # prometheus_client registra los contadores sin el sufijo _total.
        registered = _family_names(StreamMetrics())
        assert {name.removesuffix("_total") for name in METRIC_NAMES} == registered

    def test_every_module_constant_is_listed(self) -> None:
        constants = {
            value
            for key, value in vars(metrics_module).items()
            if key.isupper() and isinstance(value, str) and value.startswith("reactorguard_")
        }
        assert constants == set(METRIC_NAMES)


class TestBuckets:
    def test_latency_buckets_span_50_us_to_250_ms(self) -> None:
        assert LATENCY_BUCKETS[0] == pytest.approx(50e-6)
        assert LATENCY_BUCKETS[-1] == pytest.approx(0.25)
        assert list(LATENCY_BUCKETS) == sorted(LATENCY_BUCKETS)

    def test_validator_latencies_land_in_distinct_buckets(self) -> None:
        # Medido: p50 0,066 ms y p99 0,23 ms. Deben quedar en buckets distintos
        # dentro del rango, no los dos en el primero ni en +Inf.
        def bucket_of(value: float) -> float:
            return next(edge for edge in LATENCY_BUCKETS if value <= edge)

        assert bucket_of(0.000066) < bucket_of(0.00023) < LATENCY_BUCKETS[-1]

    def test_batch_size_buckets_cover_poll_ceiling(self) -> None:
        assert BATCH_SIZE_BUCKETS[-1] >= 500
        assert 52 in BATCH_SIZE_BUCKETS  # un timestep completo del TEP

    def test_reset_reasons(self) -> None:
        assert RESET_REASONS == ("run_change", "time_regression", "rebalance")


class TestIsolation:
    def test_two_instances_do_not_collide(self) -> None:
        first, second = StreamMetrics(), StreamMetrics()
        first.readings_consumed.inc(3)
        assert first.registry.get_sample_value("reactorguard_readings_consumed_total") == 3.0
        assert second.registry.get_sample_value("reactorguard_readings_consumed_total") == 0.0

    def test_explicit_registry_is_used(self) -> None:
        registry = CollectorRegistry()
        assert StreamMetrics(registry).registry is registry

    def test_same_registry_twice_is_rejected(self) -> None:
        registry = CollectorRegistry()
        StreamMetrics(registry)
        with pytest.raises(ValueError, match="Duplicated"):
            StreamMetrics(registry)


class TestRecording:
    def test_labelled_counters(self) -> None:
        metrics = StreamMetrics()
        metrics.messages_produced.labels(topic="raw").inc(52)
        metrics.readings_validated.labels(quality="SUSPECT").inc()
        metrics.sensor_faults.labels(fault_type="stuck").inc(2)
        metrics.alerts_published.labels(severity="LOW").inc()
        metrics.detector_state_resets.labels(reason="rebalance").inc()
        get = metrics.registry.get_sample_value
        assert get("reactorguard_stream_messages_produced_total", {"topic": "raw"}) == 52.0
        assert get("reactorguard_readings_validated_total", {"quality": "SUSPECT"}) == 1.0
        assert get("reactorguard_sensor_faults_total", {"fault_type": "stuck"}) == 2.0
        assert get("reactorguard_alerts_published_total", {"severity": "LOW"}) == 1.0
        assert get("reactorguard_detector_state_resets_total", {"reason": "rebalance"}) == 1.0

    def test_unlabelled_counters_and_gauges(self) -> None:
        metrics = StreamMetrics()
        metrics.produce_errors.inc()
        metrics.poison_messages.inc(2)
        metrics.consumer_rebalances.inc()
        metrics.commit_failures.inc(4)
        metrics.consumer_lag.labels(partition="3").set(17)
        metrics.consumer_heartbeat.set(1234.5)
        get = metrics.registry.get_sample_value
        assert get("reactorguard_stream_produce_errors_total") == 1.0
        assert get("reactorguard_poison_messages_total") == 2.0
        assert get("reactorguard_consumer_rebalances_total") == 1.0
        assert get("reactorguard_commit_failures_total") == 4.0
        assert get("reactorguard_consumer_lag", {"partition": "3"}) == 17.0
        assert get("reactorguard_consumer_heartbeat_timestamp_seconds") == 1234.5

    def test_histograms_observe(self) -> None:
        metrics = StreamMetrics()
        metrics.validation_latency.observe(0.000066)
        metrics.produce_latency.observe(0.002)
        metrics.batch_size.observe(52)
        get = metrics.registry.get_sample_value
        assert get("reactorguard_validation_latency_seconds_count") == 1.0
        assert get("reactorguard_stream_produce_latency_seconds_count") == 1.0
        assert get("reactorguard_batch_size_count") == 1.0
        first_edge = {"le": str(LATENCY_BUCKETS[0])}  # prometheus_client escribe 5e-05
        assert get("reactorguard_validation_latency_seconds_bucket", {"le": "0.0001"}) == 1.0
        assert get("reactorguard_validation_latency_seconds_bucket", first_edge) == 0.0

    def test_render_exposes_recorded_values(self) -> None:
        metrics = StreamMetrics()
        metrics.readings_consumed.inc(7)
        text = metrics.render().decode("utf-8")
        assert "reactorguard_readings_consumed_total 7.0" in text
        assert text.isascii()
