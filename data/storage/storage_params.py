"""Typed access to the storage: section of params.yaml.

Mismo patron que data/streaming/streaming_params.py: dataclass inmutable, `_require`
para claves obligatorias y validacion en la carga.

Los nombres de bucket NO son parametros: se derivan de bucket_prefix y env con la
misma plantilla que infra/terraform/modules/storage/main.tf
(`<prefix>-<proposito>-<env>`). Duplicar los cuatro nombres a mano en params.yaml
serian cuatro copias que pueden divergir de Terraform sin que nada lo note hasta
recibir un 404 en el cluster; derivarlos deja dos valores (prefijo y entorno), y
un test los contrasta con el .tf.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")
_SECTION = "storage"

# Proposito de cada bucket, tal y como Terraform lo codifica en el nombre.
RAW_PURPOSE = "data-raw"
PROCESSED_PURPOSE = "data-processed"
MODELS_PURPOSE = "models"
MLFLOW_PURPOSE = "mlflow"

# Subconjunto estricto de las reglas de nombres de GCS: minusculas, digitos y
# guiones, empezando y acabando en alfanumerico, de 3 a 63 caracteres. GCS admite
# ademas puntos y guiones bajos; el proyecto no los usa.
_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_BUCKET_MIN_LEN = 3
_BUCKET_MAX_LEN = 63


@dataclass(frozen=True)
class StorageParams:
    """Resolved configuration of the storage client.

    Attributes:
        bucket_prefix: Prefix shared by every bucket (Terraform locals.bucket_prefix).
        env: Environment suffix of every bucket (Terraform locals.env).
        local_root: Root directory of the on-disk backend (LocalCache).
        read_workers: Threads used to fetch the parts of a range read.
        request_timeout_s: Timeout of one operation against the blob store.
    """

    bucket_prefix: str
    env: str
    local_root: Path
    read_workers: int
    request_timeout_s: float

    def bucket_name(self, purpose: str) -> str:
        """Return the bucket name for a purpose, following the Terraform template.

        Args:
            purpose: Bucket purpose, e.g. "data-raw".

        Returns:
            The name `<bucket_prefix>-<purpose>-<env>`.
        """
        return f"{self.bucket_prefix}-{purpose}-{self.env}"

    @property
    def raw_bucket(self) -> str:
        """Return the bucket holding raw sensor readings."""
        return self.bucket_name(RAW_PURPOSE)

    @property
    def processed_bucket(self) -> str:
        """Return the bucket holding features and dataset splits."""
        return self.bucket_name(PROCESSED_PURPOSE)

    @property
    def models_bucket(self) -> str:
        """Return the bucket holding serialized models."""
        return self.bucket_name(MODELS_PURPOSE)

    @property
    def mlflow_bucket(self) -> str:
        """Return the bucket holding MLflow artefacts."""
        return self.bucket_name(MLFLOW_PURPOSE)


def _require(section: dict[str, Any], key: str) -> Any:
    """Return section[key], raising a descriptive error when absent.

    Args:
        section: The parsed storage: mapping.
        key: Key that must be present.

    Returns:
        The raw value associated with key.

    Raises:
        KeyError: If key is missing from the section.
    """
    if key not in section:
        raise KeyError(f"params.yaml: missing required key '{_SECTION}.{key}'.")
    return section[key]


def _label(section: dict[str, Any], key: str) -> str:
    """Read a string usable as a component of a GCS bucket name.

    Args:
        section: The parsed storage: mapping.
        key: Key to read.

    Returns:
        The value.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not lowercase alphanumerics and hyphens.
    """
    value = str(_require(section, key))
    if not _LABEL_RE.match(value):
        raise ValueError(
            f"params.yaml: '{_SECTION}.{key}' must be lowercase letters, digits and "
            f"hyphens, starting and ending with a letter or digit, got '{value}'."
        )
    return value


def _positive_int(section: dict[str, Any], key: str) -> int:
    """Read an integer that must be strictly positive.

    Args:
        section: The parsed storage: mapping.
        key: Key to read.

    Returns:
        The value as an int.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not positive.
    """
    value = int(_require(section, key))
    if value <= 0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be > 0, got {value}.")
    return value


def _positive_float(section: dict[str, Any], key: str) -> float:
    """Read a float that must be strictly positive.

    Args:
        section: The parsed storage: mapping.
        key: Key to read.

    Returns:
        The value as a float.

    Raises:
        KeyError: If key is missing.
        ValueError: If the value is not positive.
    """
    value = float(_require(section, key))
    if value <= 0:
        raise ValueError(f"params.yaml: '{_SECTION}.{key}' must be > 0, got {value}.")
    return value


def _validate_bucket_names(params: StorageParams) -> None:
    """Check that every derived bucket name is a legal GCS name.

    Args:
        params: The configuration to check.

    Raises:
        ValueError: If a derived name is shorter than 3 or longer than 63 characters.
    """
    for purpose in (RAW_PURPOSE, PROCESSED_PURPOSE, MODELS_PURPOSE, MLFLOW_PURPOSE):
        name = params.bucket_name(purpose)
        if not _BUCKET_MIN_LEN <= len(name) <= _BUCKET_MAX_LEN:
            raise ValueError(
                f"params.yaml: derived bucket name '{name}' has {len(name)} characters; "
                f"GCS requires {_BUCKET_MIN_LEN}..{_BUCKET_MAX_LEN}. "
                f"Shorten '{_SECTION}.bucket_prefix' or '{_SECTION}.env'."
            )


def load_storage_params(params_path: str | Path = DEFAULT_PARAMS_PATH) -> StorageParams:
    """Load and validate the storage: section of a params.yaml file.

    Args:
        params_path: Path to the params file. Defaults to params.yaml at the
            repository root.

    Returns:
        A fully resolved StorageParams instance.

    Raises:
        FileNotFoundError: If params_path does not exist.
        KeyError: If the storage: section or a required key is missing.
        ValueError: If a value is out of range or yields an illegal bucket name.
    """
    path = Path(params_path)
    if not path.exists():
        raise FileNotFoundError(f"Params file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        document = yaml.safe_load(fh) or {}

    if _SECTION not in document:
        raise KeyError(f"params.yaml: missing required section '{_SECTION}:'.")
    section = document[_SECTION]

    params = StorageParams(
        bucket_prefix=_label(section, "bucket_prefix"),
        env=_label(section, "env"),
        local_root=Path(str(_require(section, "local_root"))),
        read_workers=_positive_int(section, "read_workers"),
        request_timeout_s=_positive_float(section, "request_timeout_s"),
    )
    _validate_bucket_names(params)
    return params
