"""Tests for data/storage/storage_params.py.

El contraste con Terraform no repite ningun nombre a mano: extrae de
infra/terraform/modules/storage/main.tf la plantilla de cada bucket y de
environments/dev/main.tf el entorno, y exige que los nombres derivados coincidan.
Si alguien renombra un bucket en Terraform sin tocar params.yaml, falla aqui y no
con un 404 en el cluster.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from data.storage.storage_params import (
    StorageParams,
    load_storage_params,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TERRAFORM_STORAGE = REPO_ROOT / "infra" / "terraform" / "modules" / "storage" / "main.tf"
TERRAFORM_DEV = REPO_ROOT / "infra" / "terraform" / "environments" / "dev" / "main.tf"

# StorageParams expone un bucket por proposito; la clave es la etiqueta del
# `resource "google_storage_bucket"` de Terraform que lo crea.
_BUCKET_ATTRIBUTES = {
    "data_raw": "raw_bucket",
    "data_processed": "processed_bucket",
    "models": "models_bucket",
    "mlflow": "mlflow_bucket",
}


def _write(tmp_path: Path, section: dict[str, object] | None) -> Path:
    """Write a params.yaml holding only a storage: section.

    Args:
        tmp_path: Directory to write into.
        section: Content of the storage: mapping, or None to omit the section.

    Returns:
        Path of the written file.
    """
    path = tmp_path / "params.yaml"
    document = {} if section is None else {"storage": section}
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _valid_section() -> dict[str, object]:
    """Return a complete, valid storage: mapping."""
    return {
        "bucket_prefix": "reactorguard",
        "env": "dev",
        "local_root": "somewhere/store",
        "read_workers": 4,
        "request_timeout_s": 30.0,
    }


def test_loads_the_repository_params() -> None:
    params = load_storage_params(REPO_ROOT / "params.yaml")
    assert params.bucket_prefix == "reactorguard"
    assert params.env == "dev"
    assert params.local_root == Path("data/processed/local_store")
    assert params.read_workers > 0
    assert params.request_timeout_s > 0


def test_derived_bucket_names_follow_the_terraform_template() -> None:
    params = StorageParams("reactorguard", "dev", Path("x"), 4, 30.0)
    assert params.raw_bucket == "reactorguard-data-raw-dev"
    assert params.processed_bucket == "reactorguard-data-processed-dev"
    assert params.models_bucket == "reactorguard-models-dev"
    assert params.mlflow_bucket == "reactorguard-mlflow-dev"


def _terraform_bucket_templates() -> dict[str, str]:
    """Map each Terraform bucket resource label to its name template.

    Returns:
        For example {"data_raw": "${local.bucket_prefix}-data-raw-${var.env}"}.
    """
    text = TERRAFORM_STORAGE.read_text(encoding="utf-8")
    pattern = re.compile(
        r'resource\s+"google_storage_bucket"\s+"(\w+)"\s*\{[^}]*?name\s*=\s*"([^"]+)"',
        re.DOTALL,
    )
    return dict(pattern.findall(text))


def _terraform_local(path: Path, name: str) -> str:
    """Extract a string value from a Terraform locals block.

    Args:
        path: Terraform file.
        name: Local value name.

    Returns:
        The quoted value assigned to the local.
    """
    match = re.search(rf'^\s*{name}\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.M)
    assert match is not None, f"{name} not found in {path}"
    return match.group(1)


def test_terraform_declares_exactly_the_buckets_params_knows() -> None:
    assert set(_terraform_bucket_templates()) == set(_BUCKET_ATTRIBUTES)


def test_params_buckets_match_terraform() -> None:
    params = load_storage_params(REPO_ROOT / "params.yaml")
    prefix = _terraform_local(TERRAFORM_STORAGE, "bucket_prefix")
    env = _terraform_local(TERRAFORM_DEV, "env")
    assert params.bucket_prefix == prefix
    assert params.env == env
    for label, template in _terraform_bucket_templates().items():
        expected = template.replace("${local.bucket_prefix}", prefix).replace("${var.env}", env)
        assert getattr(params, _BUCKET_ATTRIBUTES[label]) == expected


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_storage_params(tmp_path / "nope.yaml")


def test_missing_section(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="section 'storage:'"):
        load_storage_params(_write(tmp_path, None))


def test_empty_document_is_a_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(KeyError, match="storage"):
        load_storage_params(path)


@pytest.mark.parametrize(
    "key", ["bucket_prefix", "env", "local_root", "read_workers", "request_timeout_s"]
)
def test_each_key_is_required(tmp_path: Path, key: str) -> None:
    section = _valid_section()
    del section[key]
    with pytest.raises(KeyError, match=f"storage.{key}"):
        load_storage_params(_write(tmp_path, section))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("bucket_prefix", "Reactor"),
        ("bucket_prefix", "-lead"),
        ("bucket_prefix", "trail-"),
        ("bucket_prefix", "with_underscore"),
        ("bucket_prefix", ""),
        ("env", "DEV"),
        ("env", "a b"),
    ],
)
def test_labels_must_be_bucket_safe(tmp_path: Path, key: str, value: str) -> None:
    section = _valid_section()
    section[key] = value
    with pytest.raises(ValueError, match=f"storage.{key}"):
        load_storage_params(_write(tmp_path, section))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("read_workers", 0),
        ("read_workers", -1),
        ("request_timeout_s", 0),
        ("request_timeout_s", -2.5),
    ],
)
def test_numbers_must_be_positive(tmp_path: Path, key: str, value: float) -> None:
    section = _valid_section()
    section[key] = value
    with pytest.raises(ValueError, match=f"storage.{key}"):
        load_storage_params(_write(tmp_path, section))


def test_derived_name_longer_than_63_characters_is_rejected(tmp_path: Path) -> None:
    section = _valid_section()
    section["bucket_prefix"] = "p" * 50
    with pytest.raises(ValueError, match="derived bucket name"):
        load_storage_params(_write(tmp_path, section))


def test_params_are_immutable() -> None:
    params = StorageParams("reactorguard", "dev", Path("x"), 4, 30.0)
    with pytest.raises(AttributeError):
        params.env = "prod"  # type: ignore[misc]
