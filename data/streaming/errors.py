"""Excepciones operativas del streaming.

El resto del repo usa solo excepciones builtin. Aqui hay fallos operativos que el
llamador debe distinguir: un PublishError impide hacer commit del offset, un
CommitError obliga a reintentar o abortar, un ConfigurationError impide arrancar.
Mapear las excepciones de kafka-python a estas evita que los llamadores dependan
de la libreria.
"""

from __future__ import annotations


class StreamingError(Exception):
    """Base class for every operational failure of the streaming layer."""


class PublishError(StreamingError):
    """A message could not be delivered to the broker.

    Tras un PublishError el llamador NO debe hacer commit del offset del lote que
    origino el envio: hacerlo perderia mensajes (semantica at-least-once, D4).
    """


class CommitError(StreamingError):
    """A consumer offset commit failed or timed out."""


class ConfigurationError(StreamingError):
    """Connection or runtime settings are missing, inconsistent or unusable."""
