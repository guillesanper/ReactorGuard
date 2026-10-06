"""Tests for data/streaming/lifecycle.py."""

from __future__ import annotations

import signal
import threading
import time

import pytest

from data.streaming.lifecycle import GracefulShutdown


def _deliver(sig: signal.Signals) -> None:
    """Invoke the currently installed Python handler as the interpreter would."""
    handler = signal.getsignal(sig)
    assert callable(handler)
    handler(sig, None)


class TestState:
    def test_initially_not_requested(self) -> None:
        shutdown = GracefulShutdown()
        assert not shutdown.is_requested
        assert shutdown.signal_name is None

    def test_request_sets_the_flag_without_a_signal(self) -> None:
        shutdown = GracefulShutdown()
        shutdown.request()
        assert shutdown.is_requested
        assert shutdown.signal_name is None

    def test_wait_times_out_when_not_requested(self) -> None:
        assert GracefulShutdown().wait(timeout=0.01) is False

    def test_wait_returns_true_once_requested(self) -> None:
        shutdown = GracefulShutdown()
        shutdown.request()
        assert shutdown.wait(timeout=0.01) is True

    def test_wait_wakes_when_another_thread_requests(self) -> None:
        shutdown = GracefulShutdown()
        timer = threading.Timer(0.05, shutdown.request)
        timer.start()
        started = time.monotonic()
        try:
            assert shutdown.wait(timeout=5.0) is True
        finally:
            timer.join()
        assert time.monotonic() - started < 4.0


class TestSignals:
    def test_install_and_restore_swap_the_handlers(self) -> None:
        before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        shutdown = GracefulShutdown()
        shutdown.install()
        try:
            for sig in before:
                assert signal.getsignal(sig) != before[sig]
        finally:
            shutdown.restore()
        assert {sig: signal.getsignal(sig) for sig in before} == before

    @pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
    def test_signal_requests_shutdown_and_names_itself(self, sig: signal.Signals) -> None:
        with GracefulShutdown() as shutdown:
            assert not shutdown.is_requested
            _deliver(sig)
            assert shutdown.is_requested
            assert shutdown.signal_name == sig.name

    def test_first_signal_wins(self) -> None:
        with GracefulShutdown() as shutdown:
            _deliver(signal.SIGTERM)
            _deliver(signal.SIGINT)
            assert shutdown.signal_name == "SIGTERM"

    def test_only_configured_signals_are_handled(self) -> None:
        before_term = signal.getsignal(signal.SIGTERM)
        with GracefulShutdown(signals=(signal.SIGINT,)):
            assert signal.getsignal(signal.SIGTERM) == before_term

    def test_context_manager_restores_on_error(self) -> None:
        before = signal.getsignal(signal.SIGINT)
        with pytest.raises(RuntimeError), GracefulShutdown():
            raise RuntimeError("loop crashed")
        assert signal.getsignal(signal.SIGINT) == before

    def test_restore_without_install_is_a_noop(self) -> None:
        GracefulShutdown().restore()

    def test_install_outside_main_thread_raises(self) -> None:
        errors: list[BaseException] = []

        def attempt() -> None:
            try:
                GracefulShutdown().install()
            except ValueError as exc:
                errors.append(exc)

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join()
        assert len(errors) == 1
