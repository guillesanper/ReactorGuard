"""Tests for data/streaming/observability.py (ServiceHttpServer and Heartbeat)."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from http.server import ThreadingHTTPServer
from typing import Any

import pytest

from data.streaming import observability
from data.streaming.metrics import StreamMetrics
from data.streaming.observability import Heartbeat, ServiceHttpServer


def _get(port: int, path: str) -> tuple[int, str, str]:
    """Return (status, content type, body) of a GET, including 4xx and 5xx."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:  # noqa: S310
            return response.status, response.headers["Content-Type"], response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers["Content-Type"], exc.read().decode()


def _server(
    metrics: StreamMetrics,
    liveness: Callable[[], bool] = lambda: True,
    readiness: Callable[[], bool] = lambda: True,
) -> ServiceHttpServer:
    return ServiceHttpServer(
        host="127.0.0.1",
        health_port=0,
        metrics_port=0,
        metrics=metrics,
        liveness=liveness,
        readiness=readiness,
    )


@pytest.fixture()
def metrics() -> StreamMetrics:
    return StreamMetrics()


@pytest.fixture()
def running(metrics: StreamMetrics) -> Iterator[ServiceHttpServer]:
    server = _server(metrics)
    server.start()
    yield server
    server.stop()


class TestEndpoints:
    def test_health_and_ready_ok(self, running: ServiceHttpServer) -> None:
        assert _get(running.health_port, "/health")[::2] == (200, "ok\n")
        assert _get(running.health_port, "/ready")[::2] == (200, "ok\n")

    def test_ports_are_distinct_and_ephemeral(self, running: ServiceHttpServer) -> None:
        assert running.health_port != running.metrics_port
        assert running.health_port > 0

    def test_not_alive_returns_503(self, metrics: StreamMetrics) -> None:
        server = _server(metrics, liveness=lambda: False)
        server.start()
        try:
            assert _get(server.health_port, "/health")[::2] == (503, "unavailable\n")
            assert _get(server.health_port, "/ready")[0] == 200
        finally:
            server.stop()

    def test_not_ready_returns_503(self, metrics: StreamMetrics) -> None:
        server = _server(metrics, readiness=lambda: False)
        server.start()
        try:
            assert _get(server.health_port, "/ready")[::2] == (503, "unavailable\n")
            assert _get(server.health_port, "/health")[0] == 200
        finally:
            server.stop()

    def test_failing_check_counts_as_unavailable(self, metrics: StreamMetrics) -> None:
        def explode() -> bool:
            raise RuntimeError("check broke")

        server = _server(metrics, readiness=explode)
        server.start()
        try:
            assert _get(server.health_port, "/ready")[0] == 503
        finally:
            server.stop()

    def test_probe_follows_state_changes(self, metrics: StreamMetrics) -> None:
        state = {"ready": False}
        server = _server(metrics, readiness=lambda: state["ready"])
        server.start()
        try:
            assert _get(server.health_port, "/ready")[0] == 503
            state["ready"] = True
            assert _get(server.health_port, "/ready")[0] == 200
        finally:
            server.stop()

    def test_metrics_endpoint_exposes_the_registry(
        self, running: ServiceHttpServer, metrics: StreamMetrics
    ) -> None:
        metrics.readings_consumed.inc(5)
        status, content_type, body = _get(running.metrics_port, "/metrics")
        assert status == 200
        assert content_type.startswith("text/plain")
        assert "reactorguard_readings_consumed_total 5.0" in body

    def test_query_string_is_ignored(self, running: ServiceHttpServer) -> None:
        assert _get(running.health_port, "/health?verbose=1")[0] == 200

    def test_unknown_path_is_404(self, running: ServiceHttpServer) -> None:
        assert _get(running.health_port, "/nope")[0] == 404
        assert _get(running.metrics_port, "/nope")[0] == 404

    def test_endpoints_are_not_mixed_between_ports(self, running: ServiceHttpServer) -> None:
        assert _get(running.health_port, "/metrics")[0] == 404
        assert _get(running.metrics_port, "/health")[0] == 404


class TestLifecycle:
    def test_ports_before_start_raise(self, metrics: StreamMetrics) -> None:
        server = _server(metrics)
        with pytest.raises(RuntimeError, match="not been started"):
            _ = server.health_port
        with pytest.raises(RuntimeError, match="not been started"):
            _ = server.metrics_port

    def test_double_start_raises(self, running: ServiceHttpServer) -> None:
        with pytest.raises(RuntimeError, match="already running"):
            running.start()

    def test_stop_releases_the_ports_and_is_idempotent(self, metrics: StreamMetrics) -> None:
        server = _server(metrics)
        server.start()
        health_port, metrics_port = server.health_port, server.metrics_port
        server.stop()
        server.stop()
        for port in (health_port, metrics_port):
            with pytest.raises(OSError), socket.create_connection(("127.0.0.1", port), timeout=1):
                pass

    def test_can_restart_after_stop(self, metrics: StreamMetrics) -> None:
        server = _server(metrics)
        server.start()
        server.stop()
        server.start()
        try:
            assert _get(server.health_port, "/health")[0] == 200
        finally:
            server.stop()

    def test_failed_metrics_bind_releases_the_health_port(
        self, metrics: StreamMetrics, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created: list[ThreadingHTTPServer] = []

        def flaky(address: tuple[str, int], handler: Any) -> ThreadingHTTPServer:
            if created:
                raise OSError("port in use")
            server = ThreadingHTTPServer(address, handler)
            created.append(server)
            return server

        monkeypatch.setattr(observability, "ThreadingHTTPServer", flaky)
        server = _server(metrics)
        with pytest.raises(OSError, match="port in use"):
            server.start()
        assert created[0].socket.fileno() == -1  # server_close() ya se llamo
        with pytest.raises(RuntimeError, match="not been started"):
            _ = server.health_port


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class TestHeartbeat:
    def test_alive_from_construction(self) -> None:
        clock = _Clock()
        assert Heartbeat(60.0, clock=clock).is_alive()

    def test_goes_stale_after_the_limit(self) -> None:
        clock = _Clock()
        heartbeat = Heartbeat(60.0, clock=clock)
        clock.now += 60.0
        assert heartbeat.is_alive()  # exactamente en el limite aun cuenta
        clock.now += 0.1
        assert not heartbeat.is_alive()
        assert heartbeat.age_seconds() == pytest.approx(60.1)

    def test_beat_revives(self) -> None:
        clock = _Clock()
        heartbeat = Heartbeat(60.0, clock=clock)
        clock.now += 100.0
        assert not heartbeat.is_alive()
        heartbeat.beat()
        assert heartbeat.is_alive()
        assert heartbeat.age_seconds() == 0.0

    def test_beat_stamps_the_gauge(self) -> None:
        clock = _Clock(500.0)
        metrics = StreamMetrics()
        heartbeat = Heartbeat(60.0, metrics=metrics, clock=clock)
        gauge = "reactorguard_consumer_heartbeat_timestamp_seconds"
        assert metrics.registry.get_sample_value(gauge) == 500.0
        clock.now = 650.0
        heartbeat.beat()
        assert metrics.registry.get_sample_value(gauge) == 650.0

    @pytest.mark.parametrize("staleness", [0, -1.0])
    def test_staleness_must_be_positive(self, staleness: float) -> None:
        with pytest.raises(ValueError, match="staleness_seconds"):
            Heartbeat(staleness)

    def test_default_clock_is_wall_time(self) -> None:
        assert Heartbeat(60.0).is_alive()
