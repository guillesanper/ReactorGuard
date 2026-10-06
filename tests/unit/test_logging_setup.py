"""Tests for data/streaming/logging_setup.py."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import pytest

from data.streaming.logging_setup import JsonFormatter, configure_logging


@pytest.fixture()
def restore_root_logger() -> Iterator[None]:
    """Undo whatever configure_logging does to the root logger."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


def _record(message: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord("svc.module", logging.WARNING, __file__, 1, message, (), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


class TestJsonFormatter:
    def test_core_fields(self) -> None:
        payload = json.loads(JsonFormatter().format(_record("lag high")))
        assert payload["level"] == "WARNING"
        assert payload["logger"] == "svc.module"
        assert payload["msg"] == "lag high"
        assert payload["ts"].endswith("+00:00")
        assert "T" in payload["ts"]

    def test_single_line(self) -> None:
        assert "\n" not in JsonFormatter().format(_record("a\nb"))

    def test_lazy_formatting_arguments_are_applied(self) -> None:
        record = logging.LogRecord("n", logging.INFO, __file__, 1, "%d of %d", (3, 52), None)
        assert json.loads(JsonFormatter().format(record))["msg"] == "3 of 52"

    def test_output_is_ascii(self) -> None:
        message = "temperatura " + chr(0xB0) + "C " + chr(0x2013) + " ok"
        line = JsonFormatter().format(_record(message))
        assert line.isascii()
        assert json.loads(line)["msg"] == message

    def test_extra_fields_become_keys(self) -> None:
        payload = json.loads(JsonFormatter().format(_record(partition=3, offset=42)))
        assert payload["partition"] == 3
        assert payload["offset"] == 42

    def test_extra_colliding_with_core_keys_is_prefixed(self) -> None:
        payload = json.loads(JsonFormatter().format(_record("real", msg_id="x", level="fake")))
        assert payload["level"] == "WARNING"
        assert payload["extra_level"] == "fake"
        assert payload["msg"] == "real"

    def test_non_serializable_extra_is_stringified(self) -> None:
        payload = json.loads(JsonFormatter().format(_record(thing=object)))
        assert "object" in payload["thing"]

    def test_exception_is_included(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "n", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
            )
        payload = json.loads(JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exc"]
        assert "exc" not in json.loads(JsonFormatter().format(_record()))


@pytest.mark.usefixtures("restore_root_logger")
class TestConfigureLogging:
    def test_installs_json_handler_and_level(self) -> None:
        stream = io.StringIO()
        configure_logging("DEBUG", stream)
        logging.getLogger("svc").debug("started", extra={"port": 8000})
        payload = json.loads(stream.getvalue().strip().splitlines()[-1])
        assert payload["msg"] == "started"
        assert payload["port"] == 8000
        assert logging.getLogger().level == logging.DEBUG

    def test_is_idempotent(self) -> None:
        first, second = io.StringIO(), io.StringIO()
        configure_logging("INFO", first)
        configure_logging("INFO", second)
        logging.getLogger("svc").info("once")
        assert first.getvalue() == ""
        assert len(second.getvalue().strip().splitlines()) == 1

    def test_does_not_remove_foreign_handlers(self) -> None:
        foreign = logging.NullHandler()
        logging.getLogger().addHandler(foreign)
        configure_logging("INFO", io.StringIO())
        assert foreign in logging.getLogger().handlers

    def test_level_is_case_insensitive(self) -> None:
        configure_logging("warning", io.StringIO())
        assert logging.getLogger().level == logging.WARNING

    def test_unknown_level(self) -> None:
        with pytest.raises(ValueError, match="Unknown log level"):
            configure_logging("LOUD", io.StringIO())

    def test_defaults_to_stderr(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging("INFO")
        logging.getLogger("svc").info("to stderr")
        assert "to stderr" in capsys.readouterr().err
