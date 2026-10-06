"""Kafka connection settings, read from the environment and validated.

Endpoints y credenciales viven solo en variables de entorno (12-factor): nunca en
params.yaml ni en el repo. En el cluster los tres KafkaUser son `authentication:
tls`, asi que el camino normal es SSL con certificado de cliente (mTLS, 9093); en
local es PLAINTEXT. SCRAM/PLAIN se admiten para el listener 9092.

La validacion es deliberadamente estricta: una configuracion de seguridad
incoherente (por ejemplo ficheros TLS con PLAINTEXT) no se ignora en silencio,
porque el cliente se conectaria sin cifrar creyendo lo contrario.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.streaming.errors import ConfigurationError

ENV_BOOTSTRAP = "KAFKA_BOOTSTRAP"
ENV_SECURITY_PROTOCOL = "KAFKA_SECURITY_PROTOCOL"
ENV_SSL_CAFILE = "KAFKA_SSL_CAFILE"
ENV_SSL_CERTFILE = "KAFKA_SSL_CERTFILE"
ENV_SSL_KEYFILE = "KAFKA_SSL_KEYFILE"
ENV_SSL_KEY_PASSWORD = "KAFKA_SSL_KEY_PASSWORD"  # noqa: S105  (nombre de variable, no un secreto)
ENV_SSL_CHECK_HOSTNAME = "KAFKA_SSL_CHECK_HOSTNAME"
ENV_SASL_MECHANISM = "KAFKA_SASL_MECHANISM"
ENV_SASL_USERNAME = "KAFKA_SASL_USERNAME"
ENV_SASL_PASSWORD = "KAFKA_SASL_PASSWORD"  # noqa: S105  (nombre de variable, no un secreto)

_PROTOCOLS = frozenset({"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"})
_SSL_PROTOCOLS = frozenset({"SSL", "SASL_SSL"})
_SASL_PROTOCOLS = frozenset({"SASL_PLAINTEXT", "SASL_SSL"})
_SASL_MECHANISMS = frozenset({"PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512"})
_TRUE_VALUES = frozenset({"1", "true", "yes"})
_FALSE_VALUES = frozenset({"0", "false", "no"})
_MAX_PORT = 65535
_REDACTED = "***"


@dataclass(frozen=True, repr=False)
class KafkaConnectionSettings:
    """Validated connection and security settings for a Kafka client.

    Attributes:
        bootstrap_servers: Broker addresses as host:port.
        security_protocol: PLAINTEXT, SSL, SASL_PLAINTEXT or SASL_SSL.
        ssl_cafile: CA bundle used to verify the brokers.
        ssl_certfile: Client certificate (mTLS).
        ssl_keyfile: Client private key (mTLS).
        ssl_key_password: Passphrase of the client key, if encrypted.
        ssl_check_hostname: Whether the broker certificate hostname is verified.
        sasl_mechanism: PLAIN, SCRAM-SHA-256 or SCRAM-SHA-512.
        sasl_username: SASL user name.
        sasl_password: SASL password.
    """

    bootstrap_servers: tuple[str, ...]
    security_protocol: str = "PLAINTEXT"
    ssl_cafile: Path | None = None
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None
    ssl_key_password: str | None = None
    ssl_check_hostname: bool = True
    sasl_mechanism: str | None = None
    sasl_username: str | None = None
    sasl_password: str | None = None

    def __post_init__(self) -> None:
        """Validate the settings as a whole.

        Raises:
            ConfigurationError: If the combination of fields is unusable.
        """
        _validate_bootstrap(self.bootstrap_servers)
        if self.security_protocol not in _PROTOCOLS:
            raise ConfigurationError(
                f"Unsupported security protocol '{self.security_protocol}'; "
                f"expected one of {sorted(_PROTOCOLS)}."
            )
        self._validate_ssl()
        self._validate_sasl()

    def _validate_ssl(self) -> None:
        """Check the TLS fields against the protocol.

        Raises:
            ConfigurationError: If TLS fields are missing, unreadable, half
                specified, or present under a protocol that ignores them.
        """
        ssl_fields = (
            self.ssl_cafile,
            self.ssl_certfile,
            self.ssl_keyfile,
            self.ssl_key_password,
        )
        if self.security_protocol not in _SSL_PROTOCOLS:
            if any(field is not None for field in ssl_fields):
                raise ConfigurationError(
                    f"TLS settings were given but security protocol is "
                    f"'{self.security_protocol}'; they would be ignored and the "
                    "connection would not be encrypted."
                )
            return
        if self.ssl_cafile is None:
            raise ConfigurationError(
                f"Security protocol '{self.security_protocol}' requires {ENV_SSL_CAFILE}."
            )
        if (self.ssl_certfile is None) != (self.ssl_keyfile is None):
            raise ConfigurationError(
                f"{ENV_SSL_CERTFILE} and {ENV_SSL_KEYFILE} must be given together (mTLS)."
            )
        for label, path in (
            (ENV_SSL_CAFILE, self.ssl_cafile),
            (ENV_SSL_CERTFILE, self.ssl_certfile),
            (ENV_SSL_KEYFILE, self.ssl_keyfile),
        ):
            if path is not None and not path.is_file():
                raise ConfigurationError(f"{label} points to a missing file: {path}")

    def _validate_sasl(self) -> None:
        """Check the SASL fields against the protocol.

        Raises:
            ConfigurationError: If SASL fields are missing, unsupported, or
                present under a protocol that ignores them.
        """
        sasl_fields = (self.sasl_mechanism, self.sasl_username, self.sasl_password)
        if self.security_protocol not in _SASL_PROTOCOLS:
            if any(field is not None for field in sasl_fields):
                raise ConfigurationError(
                    f"SASL settings were given but security protocol is "
                    f"'{self.security_protocol}'; they would be ignored."
                )
            return
        if self.sasl_mechanism not in _SASL_MECHANISMS:
            raise ConfigurationError(
                f"{ENV_SASL_MECHANISM} must be one of {sorted(_SASL_MECHANISMS)}, "
                f"got '{self.sasl_mechanism}'."
            )
        if not self.sasl_username or not self.sasl_password:
            raise ConfigurationError(
                f"{ENV_SASL_USERNAME} and {ENV_SASL_PASSWORD} are required for SASL."
            )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> KafkaConnectionSettings:
        """Build the settings from KAFKA_* environment variables.

        Args:
            env: Variable mapping to read. Defaults to os.environ.

        Returns:
            Validated settings.

        Raises:
            ConfigurationError: If KAFKA_BOOTSTRAP is absent or a value is
                malformed or inconsistent.
        """
        source = os.environ if env is None else env
        bootstrap = source.get(ENV_BOOTSTRAP, "").strip()
        if not bootstrap:
            raise ConfigurationError(f"{ENV_BOOTSTRAP} is required (comma-separated host:port).")
        servers = tuple(item.strip() for item in bootstrap.split(",") if item.strip())

        return cls(
            bootstrap_servers=servers,
            security_protocol=source.get(ENV_SECURITY_PROTOCOL, "PLAINTEXT").strip().upper(),
            ssl_cafile=_optional_path(source, ENV_SSL_CAFILE),
            ssl_certfile=_optional_path(source, ENV_SSL_CERTFILE),
            ssl_keyfile=_optional_path(source, ENV_SSL_KEYFILE),
            ssl_key_password=_optional_text(source, ENV_SSL_KEY_PASSWORD),
            ssl_check_hostname=_parse_bool(source, ENV_SSL_CHECK_HOSTNAME, default=True),
            sasl_mechanism=_optional_upper(source, ENV_SASL_MECHANISM),
            sasl_username=_optional_text(source, ENV_SASL_USERNAME),
            sasl_password=_optional_text(source, ENV_SASL_PASSWORD),
        )

    def to_client_config(self) -> dict[str, Any]:
        """Return the kafka-python keyword arguments for these settings.

        Returns:
            A dict valid for both KafkaProducer and KafkaConsumer. Only keys that
            apply to the security protocol are included.
        """
        config: dict[str, Any] = {
            "bootstrap_servers": list(self.bootstrap_servers),
            "security_protocol": self.security_protocol,
        }
        if self.security_protocol in _SSL_PROTOCOLS:
            config["ssl_check_hostname"] = self.ssl_check_hostname
            config["ssl_cafile"] = str(self.ssl_cafile)
            if self.ssl_certfile is not None and self.ssl_keyfile is not None:
                config["ssl_certfile"] = str(self.ssl_certfile)
                config["ssl_keyfile"] = str(self.ssl_keyfile)
            if self.ssl_key_password is not None:
                config["ssl_password"] = self.ssl_key_password
        if self.security_protocol in _SASL_PROTOCOLS:
            config["sasl_mechanism"] = self.sasl_mechanism
            config["sasl_plain_username"] = self.sasl_username
            config["sasl_plain_password"] = self.sasl_password
        return config

    def __repr__(self) -> str:
        """Return a representation with every secret replaced by a marker.

        Returns:
            A string safe to log: key passphrase and SASL password are redacted
            and the SASL user name is kept.
        """
        key_password = _REDACTED if self.ssl_key_password is not None else None
        sasl_password = _REDACTED if self.sasl_password is not None else None
        return (
            f"KafkaConnectionSettings(bootstrap_servers={self.bootstrap_servers!r}, "
            f"security_protocol={self.security_protocol!r}, ssl_cafile={self.ssl_cafile!r}, "
            f"ssl_certfile={self.ssl_certfile!r}, ssl_keyfile={self.ssl_keyfile!r}, "
            f"ssl_key_password={key_password!r}, "
            f"ssl_check_hostname={self.ssl_check_hostname!r}, "
            f"sasl_mechanism={self.sasl_mechanism!r}, sasl_username={self.sasl_username!r}, "
            f"sasl_password={sasl_password!r})"
        )


def _validate_bootstrap(servers: tuple[str, ...]) -> None:
    """Check that every bootstrap address is a host:port pair.

    Args:
        servers: The addresses to check.

    Raises:
        ConfigurationError: If the list is empty or an entry is malformed.
    """
    if not servers:
        raise ConfigurationError("At least one bootstrap server is required.")
    for server in servers:
        host, sep, port = server.rpartition(":")
        if not sep or not host or not port.isdigit() or not 1 <= int(port) <= _MAX_PORT:
            raise ConfigurationError(f"Bootstrap server '{server}' is not a valid host:port.")


def _optional_text(source: Mapping[str, str], name: str) -> str | None:
    """Read an optional variable, treating blank as absent.

    Args:
        source: Variable mapping.
        name: Variable name.

    Returns:
        The value, or None when unset or blank.
    """
    value = source.get(name)
    if value is None or not value.strip():
        return None
    return value


def _optional_upper(source: Mapping[str, str], name: str) -> str | None:
    """Read an optional variable and upper-case it.

    Args:
        source: Variable mapping.
        name: Variable name.

    Returns:
        The stripped upper-case value, or None when unset or blank.
    """
    value = _optional_text(source, name)
    return None if value is None else value.strip().upper()


def _optional_path(source: Mapping[str, str], name: str) -> Path | None:
    """Read an optional variable as a path.

    Args:
        source: Variable mapping.
        name: Variable name.

    Returns:
        The path, or None when unset or blank.
    """
    value = _optional_text(source, name)
    return None if value is None else Path(value.strip())


def _parse_bool(source: Mapping[str, str], name: str, *, default: bool) -> bool:
    """Read a boolean variable.

    Args:
        source: Variable mapping.
        name: Variable name.
        default: Value used when the variable is unset or blank.

    Returns:
        The parsed boolean.

    Raises:
        ConfigurationError: If the value is not a recognised boolean.
    """
    value = _optional_text(source, name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{name} must be true or false, got '{value}'.")
