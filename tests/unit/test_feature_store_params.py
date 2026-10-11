"""Tests for ml/features/store/feature_store_params.py and the Feast YAML configs.

Los ficheros feature_store.*.yaml no son un modulo, pero llevan valores que deben
coincidir con params.yaml, con Terraform (nombre del bucket del registro) y con el
manifiesto de Redis. Se contrastan aqui para que un cambio en uno de ellos falle en
la suite y no en el cluster.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.storage.storage_params import load_storage_params
from ml.features.store.feature_store_params import (
    BACKEND_REDIS,
    BACKEND_SQLITE,
    MIN_TTL_INTERVALS,
    FeatureStoreParams,
    load_feature_store_params,
)

from .conftest import WriteParams

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PARAMS = REPO_ROOT / "params.yaml"
REDIS_MANIFEST = REPO_ROOT / "k8s" / "base" / "ingestion" / "redis.yaml"


def _valid_section() -> dict[str, Any]:
    """Return a complete, valid feature_store: mapping."""
    return {
        "project": "reactorguard",
        "ttl_intervals": 3,
        "run_gap_intervals": 10,
        "online_backend": "sqlite",
        "source_path": "somewhere/source.parquet",
        "local_config_path": "cfg/local.yaml",
        "cluster_config_path": "cfg/cluster.yaml",
    }


def _write(write_params: WriteParams, **overrides: Any) -> Path:
    """Write a params.yaml with a tep: section and a feature_store: section.

    Args:
        write_params: Factory of the shared fixture (writes the tep: section).
        **overrides: Replacements for keys of the feature_store: mapping. A value
            of None removes the key.

    Returns:
        Path of the file.
    """
    path = write_params()
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    section = _valid_section()
    for key, value in overrides.items():
        if value is None:
            section.pop(key)
        else:
            section[key] = value
    document["feature_store"] = section
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _real() -> FeatureStoreParams:
    """Load the repository's params.yaml."""
    return load_feature_store_params(REAL_PARAMS)


# ---------------------------------------------------------------------------
# TTL calibrated to the TEP cadence
# ---------------------------------------------------------------------------


def test_ttl_is_540_seconds_at_the_tep_cadence() -> None:
    params = _real()
    assert params.sample_interval_seconds == 180
    assert params.ttl_seconds == 540


def test_ttl_exceeds_the_sampling_interval_with_slack() -> None:
    # Por que importa: con el TTL de 30 s del prompt original y una muestra cada
    # 180 s, la feature es vigente 30 s y nula los 150 s restantes:
    #     150 / 180 = 83 % del tiempo nula (Handoff_Paso4.md:463-475).
    # El servicio no fallaria, recibiria nulos. Hace falta TTL > intervalo, y con
    # margen para tolerar al menos una lectura perdida (>= 2 intervalos).
    params = _real()
    assert params.ttl_seconds > params.sample_interval_seconds
    assert params.ttl_seconds >= 2 * params.sample_interval_seconds


def test_the_83_percent_calculation_holds() -> None:
    interval, old_ttl = 180, 30
    assert (interval - old_ttl) / interval == pytest.approx(0.8333, abs=1e-4)


def test_ttl_follows_the_cadence_not_a_literal(write_params: WriteParams) -> None:
    path = write_params(sample_interval_minutes=5)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["feature_store"] = _valid_section()
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    params = load_feature_store_params(path)
    assert params.sample_interval_seconds == 300
    assert params.ttl_seconds == 900


@pytest.mark.parametrize("intervals", [0, 1, -3])
def test_ttl_below_two_intervals_is_rejected(
    write_params: WriteParams, intervals: int
) -> None:
    path = _write(write_params, ttl_intervals=intervals, run_gap_intervals=10)
    with pytest.raises(ValueError, match="ttl_intervals"):
        load_feature_store_params(path)


def test_minimum_ttl_constant_matches_the_rule() -> None:
    assert MIN_TTL_INTERVALS == 2


@pytest.mark.parametrize("bad", ["3", 2.5, True, None])
def test_ttl_intervals_must_be_an_integer(write_params: WriteParams, bad: object) -> None:
    path = _write(write_params, ttl_intervals=bad)
    if bad is None:
        with pytest.raises(KeyError, match="ttl_intervals"):
            load_feature_store_params(path)
    else:
        with pytest.raises(TypeError, match="ttl_intervals"):
            load_feature_store_params(path)


# ---------------------------------------------------------------------------
# Run gap, backend, project
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gap", [1, 3])
def test_run_gap_must_exceed_the_ttl(write_params: WriteParams, gap: int) -> None:
    path = _write(write_params, ttl_intervals=3, run_gap_intervals=gap)
    with pytest.raises(ValueError, match="run_gap_intervals"):
        load_feature_store_params(path)


def test_real_run_gap_exceeds_the_ttl() -> None:
    params = _real()
    assert params.run_gap_intervals > params.ttl_intervals


def test_unknown_backend_is_rejected(write_params: WriteParams) -> None:
    path = _write(write_params, online_backend="dynamodb")
    with pytest.raises(ValueError, match="online_backend"):
        load_feature_store_params(path)


@pytest.mark.parametrize("project", ["Reactor", "1guard", "reactor-guard", ""])
def test_invalid_project_name_is_rejected(write_params: WriteParams, project: str) -> None:
    path = _write(write_params, project=project)
    with pytest.raises(ValueError, match="project"):
        load_feature_store_params(path)


def test_config_path_follows_the_backend(write_params: WriteParams) -> None:
    sqlite = load_feature_store_params(_write(write_params, online_backend=BACKEND_SQLITE))
    redis = load_feature_store_params(_write(write_params, online_backend=BACKEND_REDIS))
    assert sqlite.config_path == Path("cfg/local.yaml")
    assert redis.config_path == Path("cfg/cluster.yaml")


# ---------------------------------------------------------------------------
# Loader failures
# ---------------------------------------------------------------------------


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_feature_store_params(tmp_path / "absent.yaml")


def test_missing_section_is_reported(write_params: WriteParams) -> None:
    with pytest.raises(KeyError, match="feature_store"):
        load_feature_store_params(write_params())


def test_empty_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(KeyError, match="feature_store"):
        load_feature_store_params(path)


@pytest.mark.parametrize(
    "key",
    [
        "project",
        "run_gap_intervals",
        "online_backend",
        "source_path",
        "local_config_path",
        "cluster_config_path",
    ],
)
def test_every_required_key_is_enforced(write_params: WriteParams, key: str) -> None:
    path = _write(write_params, **{key: None})
    with pytest.raises(KeyError, match=key):
        load_feature_store_params(path)


def test_zero_cadence_is_rejected(write_params: WriteParams) -> None:
    path = write_params(sample_interval_minutes=0)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["feature_store"] = _valid_section()
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="sample_interval_minutes"):
        load_feature_store_params(path)


# ---------------------------------------------------------------------------
# The YAML configs agree with params.yaml, Terraform and the Redis manifest
# ---------------------------------------------------------------------------


def _config(name: str) -> dict[str, Any]:
    """Parse one of the feature_store.*.yaml files.

    Args:
        name: "local_config_path" or "cluster_config_path" attribute name.

    Returns:
        The parsed document.
    """
    params = _real()
    path = REPO_ROOT / getattr(params, name)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_real_params_point_at_existing_configs() -> None:
    params = _real()
    assert (REPO_ROOT / params.local_config_path).is_file()
    assert (REPO_ROOT / params.cluster_config_path).is_file()


def test_both_configs_use_the_params_project() -> None:
    project = _real().project
    assert _config("local_config_path")["project"] == project
    assert _config("cluster_config_path")["project"] == project


def test_local_config_lives_under_data_processed_with_relative_paths() -> None:
    config = _config("local_config_path")
    assert config["online_store"]["type"] == "sqlite"
    for path in (config["registry"], config["online_store"]["path"]):
        # Una ruta absoluta de Windows se interpreta como URL con esquema "c".
        assert not Path(path).is_absolute()
        assert ":" not in path
        assert path.startswith("data/processed/feast/")


def test_cluster_key_ttl_equals_the_feature_view_ttl() -> None:
    config = _config("cluster_config_path")
    assert config["online_store"]["type"] == "redis"
    assert config["online_store"]["key_ttl_seconds"] == _real().ttl_seconds


def test_cluster_registry_is_in_the_processed_bucket() -> None:
    bucket = load_storage_params(REAL_PARAMS).processed_bucket
    registry = _config("cluster_config_path")["registry"]
    assert registry.startswith(f"gs://{bucket}/")
    assert registry.endswith("registry.pb")


def test_cluster_redis_endpoint_matches_the_manifest() -> None:
    manifest = REDIS_MANIFEST.read_text(encoding="utf-8")
    documents = [doc for doc in yaml.safe_load_all(manifest) if doc]
    service = next(doc for doc in documents if doc["kind"] == "Service")
    port = service["spec"]["ports"][0]["port"]
    host = f"{service['metadata']['name']}.{service['metadata']['namespace']}"

    connection = _config("cluster_config_path")["online_store"]["connection_string"]
    assert connection.startswith(f"{host}:{port},")


def test_cluster_password_comes_from_the_environment_never_the_file() -> None:
    connection = _config("cluster_config_path")["online_store"]["connection_string"]
    assert re.search(r"password=\$\{REDIS_PASSWORD\}$", connection)


def test_cluster_password_variable_matches_the_redis_secret() -> None:
    manifest = REDIS_MANIFEST.read_text(encoding="utf-8")
    assert "name: REDIS_PASSWORD" in manifest
    assert "name: redis-credentials" in manifest
