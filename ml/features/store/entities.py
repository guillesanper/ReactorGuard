"""Feast entities of the ReactorGuard feature store.

Una sola entidad: el sensor. Se descarta la `plant_entity` del diseno original
porque los atributos de planta que habria aportado (power_level_pct, hour_of_day,
day_of_week) ya viajan en cada fila larga del featurizer, repetidos por sensor
(decision 10 de Docs/Despliegue_GCP.md). Una segunda entidad obligaria a cada consulta online a
enviar dos claves para recuperar datos que ya estan en la primera.
"""

from __future__ import annotations

from feast import Entity, ValueType

SENSOR_ENTITY_NAME = "sensor"
SENSOR_JOIN_KEY = "sensor_id"

sensor_entity = Entity(
    name=SENSOR_ENTITY_NAME,
    join_keys=[SENSOR_JOIN_KEY],
    value_type=ValueType.STRING,
    description="Un transmisor de la planta, identificado por su tag (TEP-XMEAS-01, ...).",
)
