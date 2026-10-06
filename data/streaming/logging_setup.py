"""Structured JSON logging for the streaming services.

Una linea JSON por registro, en ASCII (ensure_ascii escapa cualquier caracter no
ASCII), para que el agregador del cluster pueda indexar los campos sin parsear
texto libre. Los campos pasados con `extra=` se vuelcan como claves propias.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import IO, Any

# Atributos que logging.LogRecord define por si mismo; todo lo demas viene de extra=.
_STANDARD_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}
_RESERVED_KEYS = frozenset({"ts", "level", "logger", "msg", "exc"})
_HANDLER_MARKER = "_reactorguard_json_handler"


class JsonFormatter(logging.Formatter):
    """Format log records as single-line ASCII JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a record.

        Args:
            record: The record to format.

        Returns:
            One JSON object with ts, level, logger, msg, the extra fields and,
            when present, exc with the formatted traceback.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRIBUTES or key.startswith("_"):
                continue
            payload[f"extra_{key}" if key in _RESERVED_KEYS else key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, default=str)


def configure_logging(level: str = "INFO", stream: IO[str] | None = None) -> None:
    """Install the JSON formatter on the root logger.

    Es idempotente: una segunda llamada sustituye el manejador instalado por la
    primera en lugar de duplicar cada linea.

    Args:
        level: Root log level name.
        stream: Destination stream. Defaults to sys.stderr.

    Raises:
        ValueError: If level is not a valid logging level name.
    """
    numeric = logging.getLevelName(level.upper())
    if not isinstance(numeric, int):
        raise ValueError(f"Unknown log level '{level}'.")

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    root.setLevel(numeric)
