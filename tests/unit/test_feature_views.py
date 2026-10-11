"""Unit tests for ml/features/store/feature_views.py.

Lo central es que los campos del FeatureView se generan desde
FeaturePipeline.feature_names(): cambiar una ventana en params.yaml no puede dejar la
tabla de Feast desincronizada con las columnas que escribe el featurizer.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from feast.types import Float64

from ml.features.feature_params import FeatureParams, load_feature_params
from ml.features.pipeline import FeaturePipeline
from ml.features.store.entities import sensor_entity
from ml.features.store.feature_store_params import load_feature_store_params
from ml.features.store.feature_views import (
    SENSOR_ANALYSIS_SERVICE,
    SENSOR_FEATURES_VIEW,
    apply_definitions,
    build_definitions,
    build_source,
    feature_fields,
)
from ml.features.store.source_builder import (
    CREATED_TIMESTAMP_COLUMN,
    EVENT_TIMESTAMP_COLUMN,
)
from tests.support.feast_store import (
    INTERVAL_SECONDS,
    build_store,
    make_params,
    synthetic_source_frame,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PARAMS = REPO_ROOT / "params.yaml"


@pytest.fixture(scope="module")
def feature_params() -> FeatureParams:
    """Return the repository's features: section (windows [10, 20, 60], 5 lags)."""
    return load_feature_params(REAL_PARAMS)


def _names(feature_params: FeatureParams) -> list[str]:
    """Return the feature columns the featurizer would write."""
    return FeaturePipeline(feature_params, INTERVAL_SECONDS).feature_names()


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def test_real_params_generate_28_fields(feature_params: FeatureParams) -> None:
    fields = feature_fields(feature_params, INTERVAL_SECONDS)
    assert len(fields) == 28


def test_fields_match_the_pipeline_columns_in_order(feature_params: FeatureParams) -> None:
    fields = feature_fields(feature_params, INTERVAL_SECONDS)
    assert [field.name for field in fields] == _names(feature_params)


def test_every_field_is_float64_and_names_are_unique(feature_params: FeatureParams) -> None:
    fields = feature_fields(feature_params, INTERVAL_SECONDS)
    assert {field.dtype for field in fields} == {Float64}
    assert len({field.name for field in fields}) == len(fields)


def test_changing_a_window_changes_the_fields(feature_params: FeatureParams) -> None:
    changed = replace(feature_params, window_samples=(5, 15))
    names = [field.name for field in feature_fields(changed, INTERVAL_SECONDS)]
    assert "rolling_mean_5" in names
    assert "rolling_mean_15" in names
    assert "rolling_mean_60" not in names
    assert names == _names(changed)


def test_changing_the_lag_count_changes_the_fields(feature_params: FeatureParams) -> None:
    changed = replace(feature_params, n_lags=2)
    names = [field.name for field in feature_fields(changed, INTERVAL_SECONDS)]
    assert "lag_2" in names
    assert "lag_3" not in names


def test_key_columns_are_not_declared_as_features(feature_params: FeatureParams) -> None:
    names = {field.name for field in feature_fields(feature_params, INTERVAL_SECONDS)}
    assert names.isdisjoint({"sensor_id", "timestep", "fault_type", EVENT_TIMESTAMP_COLUMN})


# ---------------------------------------------------------------------------
# Source and definitions
# ---------------------------------------------------------------------------


def test_source_uses_the_synthetic_timeline_columns() -> None:
    source = build_source(load_feature_store_params(REAL_PARAMS))
    assert source.timestamp_field == EVENT_TIMESTAMP_COLUMN
    assert source.created_timestamp_column == CREATED_TIMESTAMP_COLUMN


def test_source_path_has_forward_slashes() -> None:
    # Feast trata la ruta como URL: una ruta de Windows con barra invertida (o con
    # unidad) no funciona.
    source = build_source(load_feature_store_params(REAL_PARAMS))
    assert "\\" not in source.path
    assert source.path == "data/processed/feast_source/sensor_features.parquet"


def test_definitions_have_one_view_with_the_540_second_ttl(
    feature_params: FeatureParams,
) -> None:
    definitions = build_definitions(load_feature_store_params(REAL_PARAMS), feature_params)

    assert definitions.view.name == SENSOR_FEATURES_VIEW == "sensor_features"
    assert definitions.view.ttl == timedelta(seconds=540)
    assert definitions.view.online is True
    assert definitions.entity is sensor_entity
    assert [field.name for field in definitions.view.features] == _names(feature_params)
    assert definitions.view.tags["cadence_seconds"] == "180"


def test_definitions_ttl_follows_the_params(feature_params: FeatureParams) -> None:
    params = replace(make_params(), ttl_intervals=4)
    definitions = build_definitions(params, feature_params)
    assert definitions.view.ttl == timedelta(seconds=4 * INTERVAL_SECONDS)


def test_feature_service_wraps_the_view(feature_params: FeatureParams) -> None:
    definitions = build_definitions(load_feature_store_params(REAL_PARAMS), feature_params)
    assert definitions.service.name == SENSOR_ANALYSIS_SERVICE == "sensor_analysis"
    assert len(definitions.service.feature_view_projections) == 1
    assert definitions.service.feature_view_projections[0].name == SENSOR_FEATURES_VIEW


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


def test_apply_registers_entity_view_and_service(
    tmp_path: Path, feature_params: FeatureParams
) -> None:
    last = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    frame = synthetic_source_frame(_names(feature_params), ["S-01", "S-02"], last)
    store = build_store(
        tmp_path, make_params("views_test"), frame, last - timedelta(hours=1), last, feature_params
    )

    assert [entity.name for entity in store.list_entities()] == ["sensor"]
    assert [view.name for view in store.list_feature_views()] == ["sensor_features"]
    assert [service.name for service in store.list_feature_services()] == ["sensor_analysis"]


def test_apply_is_idempotent(tmp_path: Path, feature_params: FeatureParams) -> None:
    last = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    frame = synthetic_source_frame(_names(feature_params), ["S-01"], last)
    params = make_params("views_idem")
    store = build_store(tmp_path, params, frame, last - timedelta(hours=1), last, feature_params)

    apply_definitions(store, build_definitions(params, feature_params))

    assert len(store.list_feature_views()) == 1
    assert len(store.list_feature_services()) == 1
