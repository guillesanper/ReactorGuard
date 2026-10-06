"""Tests for data/streaming/kafka_settings.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.streaming.errors import ConfigurationError
from data.streaming.kafka_settings import (
    ENV_BOOTSTRAP,
    KafkaConnectionSettings,
)


@pytest.fixture()
def tls_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create empty CA, certificate and key files."""
    ca, cert, key = tmp_path / "ca.crt", tmp_path / "user.crt", tmp_path / "user.key"
    for path in (ca, cert, key):
        path.write_text("x", encoding="utf-8")
    return ca, cert, key


class TestPlaintext:
    def test_from_env_minimal(self) -> None:
        settings = KafkaConnectionSettings.from_env({ENV_BOOTSTRAP: "localhost:9092"})
        assert settings.bootstrap_servers == ("localhost:9092",)
        assert settings.security_protocol == "PLAINTEXT"
        assert settings.to_client_config() == {
            "bootstrap_servers": ["localhost:9092"],
            "security_protocol": "PLAINTEXT",
        }

    def test_multiple_servers_are_split_and_stripped(self) -> None:
        settings = KafkaConnectionSettings.from_env({ENV_BOOTSTRAP: " a:9092 , b:9092 ,"})
        assert settings.bootstrap_servers == ("a:9092", "b:9092")

    def test_reads_process_environment_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_BOOTSTRAP, "env-host:9092")
        assert KafkaConnectionSettings.from_env().bootstrap_servers == ("env-host:9092",)

    @pytest.mark.parametrize("value", ["", "   ", ","])
    def test_bootstrap_required(self, value: str) -> None:
        with pytest.raises(ConfigurationError, match="KAFKA_BOOTSTRAP|bootstrap"):
            KafkaConnectionSettings.from_env({ENV_BOOTSTRAP: value})

    def test_bootstrap_absent(self) -> None:
        with pytest.raises(ConfigurationError, match="KAFKA_BOOTSTRAP"):
            KafkaConnectionSettings.from_env({})

    @pytest.mark.parametrize(
        "server", ["localhost", "localhost:", ":9092", "h:abc", "h:0", "h:70000"]
    )
    def test_malformed_bootstrap(self, server: str) -> None:
        with pytest.raises(ConfigurationError, match="host:port"):
            KafkaConnectionSettings(bootstrap_servers=(server,))

    def test_unknown_protocol(self) -> None:
        with pytest.raises(ConfigurationError, match="Unsupported"):
            KafkaConnectionSettings(bootstrap_servers=("h:9092",), security_protocol="TLS")

    def test_tls_settings_under_plaintext_are_rejected(
        self, tls_files: tuple[Path, Path, Path]
    ) -> None:
        with pytest.raises(ConfigurationError, match="not be encrypted"):
            KafkaConnectionSettings(bootstrap_servers=("h:9092",), ssl_cafile=tls_files[0])

    def test_sasl_settings_under_plaintext_are_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="SASL settings"):
            KafkaConnectionSettings(bootstrap_servers=("h:9092",), sasl_username="u")


class TestSsl:
    def test_ssl_without_ca_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="KAFKA_SSL_CAFILE"):
            KafkaConnectionSettings.from_env(
                {ENV_BOOTSTRAP: "h:9093", "KAFKA_SECURITY_PROTOCOL": "SSL"}
            )

    def test_ssl_with_missing_files_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="missing file"):
            KafkaConnectionSettings.from_env(
                {
                    ENV_BOOTSTRAP: "h:9093",
                    "KAFKA_SECURITY_PROTOCOL": "ssl",
                    "KAFKA_SSL_CAFILE": str(tmp_path / "absent-ca.crt"),
                }
            )

    def test_missing_client_cert_file(self, tls_files: tuple[Path, Path, Path]) -> None:
        ca, cert, key = tls_files
        cert.unlink()
        with pytest.raises(ConfigurationError, match="KAFKA_SSL_CERTFILE"):
            KafkaConnectionSettings(
                bootstrap_servers=("h:9093",),
                security_protocol="SSL",
                ssl_cafile=ca,
                ssl_certfile=cert,
                ssl_keyfile=key,
            )

    def test_cert_without_key_is_rejected(self, tls_files: tuple[Path, Path, Path]) -> None:
        ca, cert, _ = tls_files
        with pytest.raises(ConfigurationError, match="together"):
            KafkaConnectionSettings(
                bootstrap_servers=("h:9093",),
                security_protocol="SSL",
                ssl_cafile=ca,
                ssl_certfile=cert,
            )

    def test_mtls_from_env(self, tls_files: tuple[Path, Path, Path]) -> None:
        ca, cert, key = tls_files
        settings = KafkaConnectionSettings.from_env(
            {
                ENV_BOOTSTRAP: "kafka:9093",
                "KAFKA_SECURITY_PROTOCOL": "ssl",
                "KAFKA_SSL_CAFILE": str(ca),
                "KAFKA_SSL_CERTFILE": str(cert),
                "KAFKA_SSL_KEYFILE": str(key),
                "KAFKA_SSL_KEY_PASSWORD": "pass-phrase",
            }
        )
        assert settings.to_client_config() == {
            "bootstrap_servers": ["kafka:9093"],
            "security_protocol": "SSL",
            "ssl_check_hostname": True,
            "ssl_cafile": str(ca),
            "ssl_certfile": str(cert),
            "ssl_keyfile": str(key),
            "ssl_password": "pass-phrase",
        }

    def test_server_auth_only_tls_omits_client_cert(
        self, tls_files: tuple[Path, Path, Path]
    ) -> None:
        settings = KafkaConnectionSettings(
            bootstrap_servers=("h:9093",), security_protocol="SSL", ssl_cafile=tls_files[0]
        )
        config = settings.to_client_config()
        assert "ssl_certfile" not in config
        assert "ssl_password" not in config

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("true", True), ("1", True), ("YES", True), ("false", False), ("0", False), ("no", False)],
    )
    def test_check_hostname_parsing(
        self, tls_files: tuple[Path, Path, Path], raw: str, expected: bool
    ) -> None:
        settings = KafkaConnectionSettings.from_env(
            {
                ENV_BOOTSTRAP: "h:9093",
                "KAFKA_SECURITY_PROTOCOL": "SSL",
                "KAFKA_SSL_CAFILE": str(tls_files[0]),
                "KAFKA_SSL_CHECK_HOSTNAME": raw,
            }
        )
        assert settings.ssl_check_hostname is expected
        assert settings.to_client_config()["ssl_check_hostname"] is expected

    def test_invalid_check_hostname(self, tls_files: tuple[Path, Path, Path]) -> None:
        with pytest.raises(ConfigurationError, match="true or false"):
            KafkaConnectionSettings.from_env(
                {
                    ENV_BOOTSTRAP: "h:9093",
                    "KAFKA_SECURITY_PROTOCOL": "SSL",
                    "KAFKA_SSL_CAFILE": str(tls_files[0]),
                    "KAFKA_SSL_CHECK_HOSTNAME": "maybe",
                }
            )


class TestSasl:
    def _env(self, **extra: str) -> dict[str, str]:
        env = {
            ENV_BOOTSTRAP: "h:9092",
            "KAFKA_SECURITY_PROTOCOL": "SASL_PLAINTEXT",
            "KAFKA_SASL_MECHANISM": "scram-sha-512",
            "KAFKA_SASL_USERNAME": "svc",
            "KAFKA_SASL_PASSWORD": "s3cret",
        }
        env.update(extra)
        return env

    def test_scram_from_env(self) -> None:
        config = KafkaConnectionSettings.from_env(self._env()).to_client_config()
        assert config["sasl_mechanism"] == "SCRAM-SHA-512"
        assert config["sasl_plain_username"] == "svc"
        assert config["sasl_plain_password"] == "s3cret"
        assert "ssl_cafile" not in config

    def test_unsupported_mechanism(self) -> None:
        with pytest.raises(ConfigurationError, match="KAFKA_SASL_MECHANISM"):
            KafkaConnectionSettings.from_env(self._env(KAFKA_SASL_MECHANISM="GSSAPI"))

    def test_missing_mechanism(self) -> None:
        env = self._env()
        del env["KAFKA_SASL_MECHANISM"]
        with pytest.raises(ConfigurationError, match="KAFKA_SASL_MECHANISM"):
            KafkaConnectionSettings.from_env(env)

    def test_missing_credentials(self) -> None:
        env = self._env()
        del env["KAFKA_SASL_PASSWORD"]
        with pytest.raises(ConfigurationError, match="required for SASL"):
            KafkaConnectionSettings.from_env(env)

    def test_sasl_ssl_needs_ca(self) -> None:
        with pytest.raises(ConfigurationError, match="KAFKA_SSL_CAFILE"):
            KafkaConnectionSettings.from_env(self._env(KAFKA_SECURITY_PROTOCOL="SASL_SSL"))

    def test_sasl_ssl_complete(self, tls_files: tuple[Path, Path, Path]) -> None:
        settings = KafkaConnectionSettings.from_env(
            self._env(KAFKA_SECURITY_PROTOCOL="SASL_SSL", KAFKA_SSL_CAFILE=str(tls_files[0]))
        )
        config = settings.to_client_config()
        assert config["security_protocol"] == "SASL_SSL"
        assert config["ssl_cafile"] == str(tls_files[0])
        assert config["sasl_plain_username"] == "svc"


class TestRepr:
    def test_credentials_are_absent_from_repr_and_str(
        self, tls_files: tuple[Path, Path, Path]
    ) -> None:
        settings = KafkaConnectionSettings(
            bootstrap_servers=("h:9093",),
            security_protocol="SASL_SSL",
            ssl_cafile=tls_files[0],
            ssl_key_password="key-pass-123",
            sasl_mechanism="PLAIN",
            sasl_username="svc-user",
            sasl_password="hunter2-secret",
        )
        for text in (repr(settings), str(settings), f"{settings}", f"{settings!r}"):
            assert "key-pass-123" not in text
            assert "hunter2-secret" not in text
            assert "***" in text
            assert "svc-user" in text

    def test_repr_without_secrets_shows_none(self) -> None:
        text = repr(KafkaConnectionSettings(bootstrap_servers=("h:9092",)))
        assert "ssl_key_password=None" in text
        assert "sasl_password=None" in text
        assert "***" not in text

    def test_settings_are_frozen(self) -> None:
        settings = KafkaConnectionSettings(bootstrap_servers=("h:9092",))
        with pytest.raises(AttributeError):
            settings.security_protocol = "SSL"  # type: ignore[misc]
