"""Cooperative shutdown on SIGTERM / SIGINT.

Kubernetes envia SIGTERM y espera terminationGracePeriodSeconds antes de matar el
pod. El manejador solo activa un threading.Event: el bucle del servicio comprueba
el evento entre lotes, termina el lote en curso, hace flush y commit y cierra. No
se registra nada desde el manejador (logging toma locks y la senal puede llegar con
uno tomado), el servicio consulta `signal_name` despues y lo registra en su hilo.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Sequence
from types import FrameType, TracebackType
from typing import Any


class GracefulShutdown:
    """Turns termination signals into a flag the service loop polls."""

    def __init__(
        self, signals: Sequence[signal.Signals] = (signal.SIGTERM, signal.SIGINT)
    ) -> None:
        """Prepare the handler without installing it.

        Args:
            signals: Signals that request a shutdown.
        """
        self._signals = tuple(signals)
        self._event = threading.Event()
        self._signal_number: int | None = None
        self._previous: dict[signal.Signals, Any] = {}

    def install(self) -> None:
        """Register the handler for every configured signal.

        Raises:
            ValueError: If called outside the main thread (Python restriction on
                signal handlers).
        """
        for sig in self._signals:
            self._previous[sig] = signal.signal(sig, self._handle)

    def restore(self) -> None:
        """Put back the handlers that were active before install()."""
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        self._previous = {}

    def request(self) -> None:
        """Ask for a shutdown programmatically, as a signal would."""
        self._event.set()

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler: record the signal and wake the loop.

        Args:
            signum: Number of the received signal.
            frame: Interrupted stack frame; unused.
        """
        del frame
        if self._signal_number is None:
            self._signal_number = signum
        self._event.set()

    @property
    def is_requested(self) -> bool:
        """Return whether a shutdown has been requested."""
        return self._event.is_set()

    @property
    def signal_name(self) -> str | None:
        """Return the name of the first signal received, or None."""
        if self._signal_number is None:
            return None
        return signal.Signals(self._signal_number).name

    def wait(self, timeout: float | None = None) -> bool:
        """Block until a shutdown is requested or the timeout expires.

        Args:
            timeout: Maximum seconds to wait; None waits indefinitely.

        Returns:
            True if a shutdown was requested, False on timeout.
        """
        return self._event.wait(timeout)

    def __enter__(self) -> GracefulShutdown:
        """Install the handlers.

        Returns:
            This instance.
        """
        self.install()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the previous handlers.

        Args:
            exc_type: Exception type, if the block raised.
            exc: Exception instance, if the block raised.
            traceback: Traceback, if the block raised.
        """
        self.restore()
