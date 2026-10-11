"""Unit tests for ml/features/store/feast_client.py.

Usan Feast de verdad contra un SQLite en tmp_path (sin fakeredis, sin Docker).

El hallazgo que justifica el modulo, medido en Feast 0.64: el TTL del FeatureView no
caduca nada en la lectura online. test_feast_itself_serves_a_row_older_than_the_ttl
lo deja fijado: si una version futura de Feast empezara a caducar por su cuenta, ese
test falla y avisa de que la capa de FeastClient ya no es la unica barrera.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from feast import FeatureStore
from feast.repo_config import load_repo_config
from redis.exceptions import ConnectionError as RedisConnectionError

from ml.features.feature_params import FeatureParams, load_feature_params
from ml.features.pipeline import FeaturePipeline
from ml.features.store.feast_client import (
    FeastClient,
    OnlineFeatures,
    _clean,
    _utc_now,
    assert_placeholders_resolved,
)
from ml.features.store.feature_store_params import (
    FeatureStoreParams,
    load_feature_store_params,
)
from ml.features.store.feature_views import SENSOR_FEATURES_VIEW
from tests.support.feast_store import (
    CONFIG_NAME,
    INTERVAL_SECONDS,
    build_store,
    local_config_text,
    make_params,
    synthetic_source_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PARAMS = REPO_ROOT / "params.yaml"
CLUSTER_CONFIG = REPO_ROOT / "ml" / "features" / "store" / "feature_store.cluster.yaml"

LAST_EVENT = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
SENSORS = ("S-01", "S-02", "S-03")
TTL = 540


class _Fixture:
    """Bundle of the materialized store and what produced it."""

    def __init__(
        self,
        root: Path,
        store: FeatureStore,
        params: FeatureStoreParams,
        names: list[str],
    ) -> None:
        self.root = root
        self.store = store
        self.params = params
        self.names = names

    def client(self, seconds_after_last_event: float) -> FeastClient:
        """Return a client whose clock is a given time after the latest row.

        Args:
            seconds_after_last_event: Age of the latest row at read time.

        Returns:
            A client with a frozen clock.
        """
        now = LAST_EVENT + timedelta(seconds=seconds_after_last_event)
        return FeastClient(self.store, TTL, clock=lambda: now)


@pytest.fixture(scope="module")
def feature_params() -> FeatureParams:
    """Return the repository's features: section."""
    return load_feature_params(REAL_PARAMS)


@pytest.fixture(scope="module")
def materialized(
    tmp_path_factory: pytest.TempPathFactory, feature_params: FeatureParams
) -> Iterator[_Fixture]:
    """Build one materialized SQLite store shared by the read-only tests.

    S-03 lleva NaN en dos features de la ultima fila, como el calentamiento de una
    ventana movil.
    """
    root = tmp_path_factory.mktemp("feast_client")
    params = make_params("client_test")
    names = FeaturePipeline(feature_params, INTERVAL_SECONDS).feature_names()
    frame = synthetic_source_frame(
        names,
        SENSORS,
        LAST_EVENT,
        nan_cells=[("S-03", "rolling_std_60"), ("S-03", "lag_5")],
    )
    store = build_store(
        root, params, frame, LAST_EVENT - timedelta(hours=1), LAST_EVENT, feature_params
    )
    yield _Fixture(root, store, params, names)


# ---------------------------------------------------------------------------
# Fresh reads
# ---------------------------------------------------------------------------


def test_fresh_sensors_return_all_28_features(materialized: _Fixture) -> None:
    result = materialized.client(60).get_online_features(["S-01", "S-02"])

    assert result.complete
    assert result.fresh_sensors == ("S-01", "S-02")
    assert result.stale_sensors == ()
    assert result.missing_sensors == ()
    assert len(materialized.names) == 28
    for sensor in ("S-01", "S-02"):
        assert set(result.values[sensor]) == set(materialized.names)
        assert all(value is not None for value in result.values[sensor].values())


def test_the_latest_row_is_served_not_an_earlier_one(materialized: _Fixture) -> None:
    result = materialized.client(60).get_online_features(["S-01"])
    # Ultima fila del sensor 0: 1000 x 1 + indice de feature ("value" es el 0).
    assert result.values["S-01"]["value"] == pytest.approx(1000.0)
    assert result.event_timestamps["S-01"] == LAST_EVENT


def test_values_follow_the_requested_order_and_sensor(materialized: _Fixture) -> None:
    result = materialized.client(60).get_online_features(["S-02", "S-01"])
    assert list(result.values) == ["S-02", "S-01"]
    assert result.values["S-02"]["value"] == pytest.approx(2000.0)
    assert result.values["S-01"]["value"] == pytest.approx(1000.0)


def test_elapsed_time_is_measured(materialized: _Fixture) -> None:
    result = materialized.client(60).get_online_features(list(SENSORS))
    assert result.elapsed_ms > 0.0


def test_nan_cells_become_none_without_marking_the_sensor_stale(
    materialized: _Fixture,
) -> None:
    result = materialized.client(60).get_online_features(["S-03"])
    values = result.values["S-03"]

    assert values["rolling_std_60"] is None
    assert values["lag_5"] is None
    assert values["value"] is not None
    assert result.fresh_sensors == ("S-03",)


# ---------------------------------------------------------------------------
# Freshness: the guard Feast does not provide
# ---------------------------------------------------------------------------


def test_feast_itself_serves_a_row_older_than_the_ttl(materialized: _Fixture) -> None:
    # Fija el hallazgo medido: el almacen devuelve el valor aunque el FeatureView
    # tenga TTL de 540 s y la fila tenga 26 anos respecto al reloj real.
    response = materialized.store.get_online_features(
        features=materialized.store.get_feature_service("sensor_analysis"),
        entity_rows=[{"sensor_id": "S-01"}],
    ).to_dict()
    assert response["value"] == [pytest.approx(1000.0)]
    view = materialized.store.get_feature_view(SENSOR_FEATURES_VIEW)
    assert view.ttl == timedelta(seconds=TTL)
    assert datetime.now(UTC) - LAST_EVENT > view.ttl


def test_a_row_exactly_at_the_ttl_is_still_fresh(materialized: _Fixture) -> None:
    result = materialized.client(TTL).get_online_features(["S-01"])
    assert result.complete


def test_a_row_one_second_past_the_ttl_is_stale(materialized: _Fixture) -> None:
    result = materialized.client(TTL + 1).get_online_features(["S-01", "S-02"])

    assert result.stale_sensors == ("S-01", "S-02")
    assert result.fresh_sensors == ()
    assert not result.complete


def test_stale_sensors_get_null_features_but_keep_their_timestamp(
    materialized: _Fixture,
) -> None:
    result = materialized.client(TTL + 1).get_online_features(["S-01"])

    assert set(result.values["S-01"]) == set(materialized.names)
    assert all(value is None for value in result.values["S-01"].values())
    assert result.event_timestamps["S-01"] == LAST_EVENT


def test_staleness_is_logged(materialized: _Fixture, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="ml.features.store.feast_client"):
        materialized.client(TTL + 1).get_online_features(["S-01"])
    assert "1 stale" in caplog.text


def test_three_missed_samples_is_the_tolerance(materialized: _Fixture) -> None:
    # TTL = 3 intervalos: tras dos muestras perdidas (360 s) aun se sirve; tras la
    # tercera (540 s) es el limite; la cuarta (720 s) ya caduca.
    assert materialized.client(2 * INTERVAL_SECONDS).get_online_features(["S-01"]).complete
    assert materialized.client(3 * INTERVAL_SECONDS).get_online_features(["S-01"]).complete
    assert not materialized.client(4 * INTERVAL_SECONDS).get_online_features(["S-01"]).complete


# ---------------------------------------------------------------------------
# Missing sensors
# ---------------------------------------------------------------------------


def test_an_unknown_sensor_is_missing_not_stale(materialized: _Fixture) -> None:
    result = materialized.client(60).get_online_features(["S-01", "S-99"])

    assert result.missing_sensors == ("S-99",)
    assert result.stale_sensors == ()
    assert result.fresh_sensors == ("S-01",)
    assert result.event_timestamps["S-99"] is None
    assert all(value is None for value in result.values["S-99"].values())
    assert not result.complete


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [[], [""], ["S-01", " "], ["S-01", "S-01"]])
def test_invalid_sensor_lists_are_rejected(materialized: _Fixture, bad: list[str]) -> None:
    with pytest.raises(ValueError, match="sensor_ids"):
        materialized.client(60).get_online_features(bad)


@pytest.mark.parametrize("ttl", [0, -5])
def test_a_non_positive_ttl_is_rejected(materialized: _Fixture, ttl: int) -> None:
    with pytest.raises(ValueError, match="ttl_seconds"):
        FeastClient(materialized.store, ttl)


# ---------------------------------------------------------------------------
# Infrastructure failures are errors, not nulls
# ---------------------------------------------------------------------------


class _BrokenStore:
    """Stand-in for a FeatureStore whose backend raises on read."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    def get_feature_service(self, name: str) -> object:
        """Return a placeholder service."""
        return object()

    def get_online_features(self, **_: Any) -> Any:
        """Raise the configured error."""
        raise self._error


@pytest.mark.parametrize(
    "error",
    [
        RedisConnectionError("redis down"),
        sqlite3.OperationalError("database is locked"),
        OSError("network unreachable"),
    ],
)
def test_backend_failures_become_connection_errors(error: BaseException) -> None:
    client = FeastClient(cast(FeatureStore, _BrokenStore(error)), TTL)
    with pytest.raises(ConnectionError, match="unreachable"):
        client.get_online_features(["S-01"])


def test_unrelated_errors_are_not_swallowed() -> None:
    client = FeastClient(cast(FeatureStore, _BrokenStore(KeyError("bug"))), TTL)
    with pytest.raises(KeyError):
        client.get_online_features(["S-01"])


def test_an_unregistered_service_asks_for_feast_apply(
    tmp_path: Path, feature_params: FeatureParams
) -> None:
    (tmp_path / CONFIG_NAME).write_text(local_config_text("empty_registry"), encoding="utf-8")
    store = FeatureStore(repo_path=str(tmp_path), fs_yaml_file=tmp_path / CONFIG_NAME)

    with pytest.raises(LookupError, match="feast apply"):
        FeastClient(store, TTL).get_online_features(["S-01"])


# ---------------------------------------------------------------------------
# from_params
# ---------------------------------------------------------------------------


def test_from_params_reads_through_the_configured_backend(materialized: _Fixture) -> None:
    client = FeastClient.from_params(
        materialized.params,
        repo_path=materialized.root,
        clock=lambda: LAST_EVENT + timedelta(seconds=30),
    )
    assert client.ttl_seconds == TTL
    assert client.get_online_features(["S-01"]).complete


def test_from_params_defaults_to_the_working_directory(
    materialized: _Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(materialized.root)
    client = FeastClient.from_params(
        materialized.params, clock=lambda: LAST_EVENT + timedelta(seconds=30)
    )
    assert client.get_online_features(["S-02"]).complete


def test_from_params_with_an_absolute_config_path(materialized: _Fixture) -> None:
    absolute = FeatureStoreParams(
        **{
            **materialized.params.__dict__,
            "local_config_path": materialized.root / CONFIG_NAME,
        }
    )
    client = FeastClient.from_params(absolute, repo_path=materialized.root)
    assert client.ttl_seconds == TTL


def test_from_params_fails_if_the_config_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FeastClient.from_params(make_params(), repo_path=tmp_path)


def test_from_params_for_redis_requires_the_password_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    params = load_feature_store_params(REAL_PARAMS)
    redis_params = FeatureStoreParams(**{**params.__dict__, "online_backend": "redis"})

    with pytest.raises(ValueError, match="REDIS_PASSWORD"):
        FeastClient.from_params(redis_params, repo_path=REPO_ROOT)


# ---------------------------------------------------------------------------
# Placeholders in the YAML
# ---------------------------------------------------------------------------


def test_unset_placeholder_is_reported_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SOME_UNSET_VAR", raising=False)
    config = tmp_path / "feature_store.yaml"
    config.write_text("key: ${SOME_UNSET_VAR}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SOME_UNSET_VAR"):
        assert_placeholders_resolved(config)


def test_set_placeholder_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_SET_VAR", "value")
    config = tmp_path / "feature_store.yaml"
    config.write_text("key: ${SOME_SET_VAR}\n", encoding="utf-8")
    assert_placeholders_resolved(config)


def test_placeholders_in_comments_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("COMMENTED_VAR", raising=False)
    config = tmp_path / "feature_store.yaml"
    config.write_text(
        "# la variable ${COMMENTED_VAR} se documenta aqui\nkey: 1  # y ${COMMENTED_VAR}\n",
        encoding="utf-8",
    )
    assert_placeholders_resolved(config)


def test_missing_config_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assert_placeholders_resolved(tmp_path / "absent.yaml")


def test_cluster_config_is_rejected_without_the_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="REDIS_PASSWORD"):
        assert_placeholders_resolved(CLUSTER_CONFIG)


def test_cluster_config_is_accepted_by_feast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_PASSWORD", "s3cret")
    assert_placeholders_resolved(CLUSTER_CONFIG)

    config = load_repo_config(REPO_ROOT, CLUSTER_CONFIG)

    online = config.online_store
    assert online.type == "redis"
    assert online.connection_string == (
        "redis-service.reactorguard-ingestion:6379,password=s3cret"
    )
    assert online.key_ttl_seconds == load_feature_store_params(REAL_PARAMS).ttl_seconds
    assert config.project == "reactorguard"


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------


def test_online_features_helpers() -> None:
    result = OnlineFeatures(
        values={"A": {"x": 1.0}, "B": {"x": None}, "C": {"x": None}},
        event_timestamps={"A": LAST_EVENT, "B": LAST_EVENT, "C": None},
        stale_sensors=("B",),
        missing_sensors=("C",),
        elapsed_ms=1.0,
    )
    assert result.fresh_sensors == ("A",)
    assert not result.complete


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), (float("nan"), None), (3, 3.0), (2.5, 2.5)],
)
def test_clean_maps_nan_and_none_to_none(raw: object, expected: float | None) -> None:
    assert _clean(raw) == expected


def test_default_clock_is_timezone_aware_utc() -> None:
    now = _utc_now()
    assert now.tzinfo is not None
    assert abs((datetime.now(UTC) - now).total_seconds()) < 5
