"""HTTP endpoints of a streaming service: health, readiness and metrics.

Dos servidores `http.server` en hilos demonio (D10): /health y /ready en un
puerto (sondas de Kubernetes) y /metrics en otro (Prometheus). Separarlos permite
que una NetworkPolicy abra el puerto de metricas solo a observability sin exponer
las sondas, y a la inversa. El bind es configurable: el servicio lo toma de
params.yaml (0.0.0.0 en el pod) y los tests usan 127.0.0.1 con puerto efimero.

/health es vivacidad: responde segun un latido reciente (Heartbeat). /ready es
disponibilidad: responde segun un callable del servicio (particiones asignadas,
productor listo). Un fallo del callable cuenta como no listo, nunca como error 500.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from prometheus_client import CONTENT_TYPE_LATEST

from data.streaming.metrics import StreamMetrics

_LOG = logging.getLogger(__name__)

_OK = 200
_UNAVAILABLE = 503
_NOT_FOUND = 404
_TEXT = "text/plain; charset=utf-8"
_SHUTDOWN_JOIN_SECONDS = 5.0


class Heartbeat:
    """Records the last sign of life of a service loop.

    Se crea con un latido inicial (el arranque) para que el tiempo sin latido se
    cuente desde que el servicio existe y no desde un instante indefinido.
    """

    def __init__(
        self,
        staleness_seconds: float,
        *,
        metrics: StreamMetrics | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Start the heartbeat.

        Args:
            staleness_seconds: Age after which the service counts as not alive.
            metrics: When given, every beat also stamps the heartbeat gauge.
            clock: Wall-clock source in seconds since the epoch.

        Raises:
            ValueError: If staleness_seconds is not positive.
        """
        if staleness_seconds <= 0:
            raise ValueError(f"staleness_seconds must be > 0, got {staleness_seconds}.")
        self._staleness_seconds = staleness_seconds
        self._metrics = metrics
        self._clock = clock
        self._lock = threading.Lock()
        self._last = clock()
        if metrics is not None:
            metrics.consumer_heartbeat.set(self._last)

    def beat(self) -> None:
        """Record that the service loop is still running."""
        now = self._clock()
        with self._lock:
            self._last = now
        if self._metrics is not None:
            self._metrics.consumer_heartbeat.set(now)

    def age_seconds(self) -> float:
        """Return the seconds elapsed since the last beat.

        Returns:
            The age of the latest heartbeat.
        """
        with self._lock:
            last = self._last
        return self._clock() - last

    def is_alive(self) -> bool:
        """Return whether the last beat is recent enough.

        Returns:
            True when the heartbeat age does not exceed the staleness limit.
        """
        return self.age_seconds() <= self._staleness_seconds


def _make_handler(
    routes: dict[str, Callable[[], tuple[int, str, bytes]]],
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to a set of GET routes.

    Args:
        routes: Maps a path to a callable returning (status, content type, body).

    Returns:
        A BaseHTTPRequestHandler subclass serving those routes.
    """

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802  (nombre fijado por http.server)
            route = routes.get(self.path.split("?", 1)[0])
            if route is None:
                status, content_type, body = _NOT_FOUND, _TEXT, b"not found\n"
            else:
                status, content_type, body = route()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            _LOG.debug("http %s - %s", self.address_string(), format % args)

    return _Handler


def _probe(check: Callable[[], bool], name: str) -> Callable[[], tuple[int, str, bytes]]:
    """Wrap a boolean check as a probe route.

    Args:
        check: Returns True when the service passes the probe.
        name: Probe name used in logs.

    Returns:
        A route callable. An exception in check counts as failing the probe.
    """

    def _route() -> tuple[int, str, bytes]:
        try:
            ok = check()
        except Exception:
            _LOG.exception("%s check raised; reporting not ok", name)
            ok = False
        if ok:
            return _OK, _TEXT, b"ok\n"
        return _UNAVAILABLE, _TEXT, b"unavailable\n"

    return _route


class ServiceHttpServer:
    """Serves /health and /ready on one port and /metrics on another."""

    def __init__(
        self,
        *,
        host: str,
        health_port: int,
        metrics_port: int,
        metrics: StreamMetrics,
        liveness: Callable[[], bool],
        readiness: Callable[[], bool],
    ) -> None:
        """Prepare the servers without binding any socket.

        Args:
            host: Interface to bind both servers to.
            health_port: Port for /health and /ready; 0 picks a free port.
            metrics_port: Port for /metrics; 0 picks a free port.
            metrics: Metrics whose registry /metrics exposes.
            liveness: Returns True while the service loop is alive.
            readiness: Returns True while the service can do useful work.
        """
        self._host = host
        self._health_port = health_port
        self._metrics_port = metrics_port
        self._metrics = metrics
        self._liveness = liveness
        self._readiness = readiness
        self._servers: list[ThreadingHTTPServer] = []
        self._threads: list[threading.Thread] = []
        self._health_server: ThreadingHTTPServer | None = None
        self._metrics_server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        """Bind both ports and start serving in background threads.

        Raises:
            RuntimeError: If the servers are already running.
            OSError: If a port cannot be bound.
        """
        if self._servers:
            raise RuntimeError("ServiceHttpServer is already running.")

        def _metrics_route() -> tuple[int, str, bytes]:
            return _OK, CONTENT_TYPE_LATEST, self._metrics.render()

        health_routes = {
            "/health": _probe(self._liveness, "liveness"),
            "/ready": _probe(self._readiness, "readiness"),
        }
        self._health_server = ThreadingHTTPServer(
            (self._host, self._health_port), _make_handler(health_routes)
        )
        try:
            self._metrics_server = ThreadingHTTPServer(
                (self._host, self._metrics_port), _make_handler({"/metrics": _metrics_route})
            )
        except OSError:
            self._health_server.server_close()
            self._health_server = None
            raise
        self._servers = [self._health_server, self._metrics_server]
        for server in self._servers:
            server.daemon_threads = True
            thread = threading.Thread(
                target=server.serve_forever, name="reactorguard-http", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        _LOG.info(
            "HTTP endpoints up: health/ready on %s:%d, metrics on %s:%d",
            self._host,
            self.health_port,
            self._host,
            self.metrics_port,
        )

    @property
    def health_port(self) -> int:
        """Return the bound port of /health and /ready.

        Raises:
            RuntimeError: If the server has not been started.
        """
        if self._health_server is None:
            raise RuntimeError("ServiceHttpServer has not been started.")
        return int(self._health_server.server_address[1])

    @property
    def metrics_port(self) -> int:
        """Return the bound port of /metrics.

        Raises:
            RuntimeError: If the server has not been started.
        """
        if self._metrics_server is None:
            raise RuntimeError("ServiceHttpServer has not been started.")
        return int(self._metrics_server.server_address[1])

    def stop(self) -> None:
        """Stop serving and release both ports. Safe to call more than once."""
        for server in self._servers:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=_SHUTDOWN_JOIN_SECONDS)
        self._servers = []
        self._threads = []
        self._health_server = None
        self._metrics_server = None
