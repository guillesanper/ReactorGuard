"""Feast FeatureView and FeatureService of the sensor features.

UN SOLO FeatureView. El diseno original pedia varios, con cuatro TTL distintos
(1 h, 30 s, 1 min, 5 min) pensados para una cadencia de 1 s. Con una unica
cadencia de 180 s no hay nada que justifique TTL distintos: todas las features de
una fila se calculan a la vez, con el mismo timestamp, y caducan a la vez.

LOS CAMPOS SE GENERAN, NO SE ENUMERAN. Los Field salen de
FeaturePipeline.feature_names(), la misma fuente que usa el featurizer para escribir
las columnas. Si alguien cambia features.window_samples o features.n_lags en
params.yaml, los campos del FeatureView cambian con ellos; una lista escrita a mano
quedaria desincronizada con la tabla y Feast leeria columnas que no existen.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float64

from ml.features.feature_params import FeatureParams
from ml.features.pipeline import FeaturePipeline
from ml.features.store.entities import sensor_entity
from ml.features.store.feature_store_params import FeatureStoreParams
from ml.features.store.source_builder import (
    CREATED_TIMESTAMP_COLUMN,
    EVENT_TIMESTAMP_COLUMN,
)

SENSOR_FEATURES_VIEW = "sensor_features"
SENSOR_ANALYSIS_SERVICE = "sensor_analysis"


@dataclass(frozen=True)
class FeatureDefinitions:
    """The Feast objects that `feast apply` registers.

    Attributes:
        entity: The sensor entity.
        source: The offline file source.
        view: The sensor_features FeatureView.
        service: The sensor_analysis FeatureService.
    """

    entity: Entity
    source: FileSource
    view: FeatureView
    service: FeatureService


def feature_fields(feature_params: FeatureParams, sample_interval_seconds: int) -> list[Field]:
    """Generate one Float64 Field per feature column of the pipeline.

    Args:
        feature_params: Resolved features: section of params.yaml.
        sample_interval_seconds: TEP cadence, needed to build the pipeline (no
            afecta a los nombres).

    Returns:
        The fields, in the order the featurizer emits the columns.
    """
    pipeline = FeaturePipeline(feature_params, sample_interval_seconds=sample_interval_seconds)
    return [Field(name=name, dtype=Float64) for name in pipeline.feature_names()]


def build_source(store_params: FeatureStoreParams) -> FileSource:
    """Build the offline parquet source.

    La ruta se emite con barras normales: Feast la trata como URL y una ruta
    absoluta de Windows (C:\\...) se interpreta como esquema "c".

    Args:
        store_params: Resolved feature_store: section.

    Returns:
        A FileSource over feast_source's parquet.
    """
    return FileSource(
        name=f"{SENSOR_FEATURES_VIEW}_source",
        path=Path(store_params.source_path).as_posix(),
        timestamp_field=EVENT_TIMESTAMP_COLUMN,
        created_timestamp_column=CREATED_TIMESTAMP_COLUMN,
    )


def build_definitions(
    store_params: FeatureStoreParams, feature_params: FeatureParams
) -> FeatureDefinitions:
    """Build the entity, source, FeatureView and FeatureService.

    Args:
        store_params: Resolved feature_store: section.
        feature_params: Resolved features: section.

    Returns:
        The definitions to register with `apply_definitions`.
    """
    source = build_source(store_params)
    view = FeatureView(
        name=SENSOR_FEATURES_VIEW,
        entities=[sensor_entity],
        ttl=timedelta(seconds=store_params.ttl_seconds),
        schema=feature_fields(feature_params, store_params.sample_interval_seconds),
        source=source,
        online=True,
        tags={"cadence_seconds": str(store_params.sample_interval_seconds)},
        description="Features por sensor y muestra (TDD 5.5), cadencia TEP de 180 s.",
    )
    service = FeatureService(name=SENSOR_ANALYSIS_SERVICE, features=[view])
    return FeatureDefinitions(entity=sensor_entity, source=source, view=view, service=service)


def apply_definitions(store: FeatureStore, definitions: FeatureDefinitions) -> None:
    """Register the definitions in the store's registry (`feast apply`).

    Args:
        store: The feature store to register into.
        definitions: Objects built by build_definitions.
    """
    store.apply([definitions.entity, definitions.view, definitions.service])
