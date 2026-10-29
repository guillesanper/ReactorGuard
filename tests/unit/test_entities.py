"""Unit tests for ml/features/store/entities.py."""

from __future__ import annotations

from feast import ValueType

import ml.features.store.entities as entities
from ml.features.store.entities import (
    SENSOR_ENTITY_NAME,
    SENSOR_JOIN_KEY,
    sensor_entity,
)


def test_sensor_entity_joins_on_sensor_id() -> None:
    assert sensor_entity.name == SENSOR_ENTITY_NAME == "sensor"
    assert sensor_entity.join_key == SENSOR_JOIN_KEY == "sensor_id"


def test_sensor_entity_is_keyed_by_string_tags() -> None:
    assert sensor_entity.value_type == ValueType.STRING


def test_there_is_no_plant_entity() -> None:
    # power_level_pct, hour_of_day y day_of_week ya viajan en cada fila larga
    # (decision 10 de Docs/Despliegue_GCP.md): una segunda entidad obligaria a enviar dos claves
    # para recuperar datos que ya estan en la primera.
    assert not hasattr(entities, "plant_entity")
