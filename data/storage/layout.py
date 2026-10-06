"""Object-key layout of the sensor-readings store (funciones puras, sin E/S).

Las lecturas se particionan por planta y hora UTC en estilo hive:

    plant=<id>/year=YYYY/month=MM/day=DD/hour=HH/part-<id>.parquet

La hora es la unidad porque es la granularidad mas fina que un lector de rango
necesita: un rango de dos horas toca exactamente dos prefijos. A la cadencia del
TEP (180 s) una hora son 20 muestras x 52 sensores = 1.040 lecturas.

Todo se calcula en UTC. El contrato de datos declara el timestamp en UTC, y una
particion por hora local seria ambigua en el cambio de horario. Un timestamp sin
zona se rechaza en lugar de suponerlo UTC: colocar una lectura en la hora
equivocada no produce ningun error, solo un rango de lectura que no la encuentra.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime, timedelta

PART_PREFIX = "part-"
PART_SUFFIX = ".parquet"
_PART_ID_LENGTH = 12
_MAX_FAULT_TYPE = 99

# Un identificador de planta forma parte de una clave de objeto y de un valor de
# particion hive: no puede llevar "/", "=" ni empezar por punto (".." escaparia
# del prefijo en el backend local).
_PLANT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def to_utc(timestamp: datetime) -> datetime:
    """Return the timestamp expressed in UTC.

    Args:
        timestamp: A timezone-aware datetime in any zone.

    Returns:
        The same instant with tzinfo set to UTC.

    Raises:
        ValueError: If the timestamp has no timezone.
    """
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(
            f"Timestamp {timestamp.isoformat()} has no timezone; the data contract "
            "requires UTC and the store will not guess the partition of a naive time."
        )
    return timestamp.astimezone(UTC)


def validate_plant_id(plant_id: str) -> str:
    """Check that a plant id is safe to embed in an object key.

    Args:
        plant_id: Plant identifier of a reading.

    Returns:
        The same identifier, unchanged.

    Raises:
        ValueError: If it is empty, starts with a non-alphanumeric character or
            contains characters other than letters, digits, ".", "_" and "-".
    """
    if not _PLANT_ID_RE.match(plant_id):
        raise ValueError(
            f"Invalid plant_id '{plant_id}': it must start with a letter or digit and "
            "contain only letters, digits, '.', '_' and '-'."
        )
    return plant_id


def readings_prefix(plant_id: str, timestamp: datetime) -> str:
    """Return the partition prefix holding the readings of one plant-hour.

    Args:
        plant_id: Plant identifier.
        timestamp: Timezone-aware instant; truncated to its UTC hour.

    Returns:
        A prefix ending in "/", e.g. "plant=P1/year=2024/month=03/day=09/hour=07/".

    Raises:
        ValueError: If plant_id is not key-safe or timestamp has no timezone.
    """
    moment = to_utc(timestamp)
    return (
        f"plant={validate_plant_id(plant_id)}/year={moment.year:04d}/"
        f"month={moment.month:02d}/day={moment.day:02d}/hour={moment.hour:02d}/"
    )


def readings_object(prefix: str, part_id: str) -> str:
    """Return the object key of one part inside a partition prefix.

    Args:
        prefix: Partition prefix, as returned by readings_prefix.
        part_id: Part identifier, as returned by part_id.

    Returns:
        The full object key.
    """
    return f"{prefix}{PART_PREFIX}{part_id}{PART_SUFFIX}"


def part_id(first_reading_id: str, last_reading_id: str, count: int) -> str:
    """Return the deterministic identifier of a part.

    El nombre sale del contenido (primer y ultimo reading_id y numero de
    lecturas), no de un reloj ni de un aleatorio. Asi un reintento at-least-once
    del mismo lote escribe el MISMO objeto: la escritura condicional lo ve como
    ya existente y no hay duplicado ni sobrescritura (ingestion-sa solo tiene
    objectCreator y no puede sobrescribir). SHA-1 se usa como huella, no como
    primitiva de seguridad.

    Args:
        first_reading_id: reading_id of the first reading of the part.
        last_reading_id: reading_id of the last reading of the part.
        count: Number of readings in the part.

    Returns:
        The first 12 hexadecimal characters of the SHA-1 of the three values.
    """
    digest = hashlib.sha1(
        f"{first_reading_id}|{last_reading_id}|{count}".encode(),
        usedforsecurity=False,
    )
    return digest.hexdigest()[:_PART_ID_LENGTH]


def is_part_key(key: str) -> bool:
    """Tell whether an object key names a readings part.

    Args:
        key: Object key.

    Returns:
        True if the final path segment is `part-<id>.parquet`.
    """
    name = key.rsplit("/", 1)[-1]
    return name.startswith(PART_PREFIX) and name.endswith(PART_SUFFIX)


def hour_prefixes(plant_id: str, start: datetime, end: datetime) -> list[str]:
    """Return the partition prefixes that a half-open range [start, end) touches.

    Args:
        plant_id: Plant identifier.
        start: Inclusive lower bound, timezone-aware.
        end: Exclusive upper bound, timezone-aware.

    Returns:
        One prefix per UTC hour overlapping the range, in chronological order.
        Empty when start equals end.

    Raises:
        ValueError: If a bound has no timezone, plant_id is not key-safe, or
            start is later than end.
    """
    start_utc = to_utc(start)
    end_utc = to_utc(end)
    validate_plant_id(plant_id)
    if start_utc > end_utc:
        raise ValueError(
            f"Range start {start_utc.isoformat()} is later than end {end_utc.isoformat()}."
        )
    hour = start_utc.replace(minute=0, second=0, microsecond=0)
    prefixes: list[str] = []
    while hour < end_utc:
        prefixes.append(readings_prefix(plant_id, hour))
        hour += timedelta(hours=1)
    return prefixes


def features_object(fault_type: int) -> str:
    """Return the object key of the feature table of one fault type.

    Conserva el layout de DVC (`data/processed/features/fault_type=NN/`) bajo el
    bucket data-processed, para que `gsutil rsync` y este cliente hablen del mismo
    arbol. Las ventanas de features se miden en muestras y viven como columnas del
    fichero, no como parte de la ruta.

    Args:
        fault_type: TEP fault identifier, 0 (normal) to 99.

    Returns:
        The key `features/fault_type=NN/features.parquet`.

    Raises:
        ValueError: If fault_type is outside 0..99.
    """
    if not 0 <= fault_type <= _MAX_FAULT_TYPE:
        raise ValueError(f"fault_type must be in 0..{_MAX_FAULT_TYPE}, got {fault_type}.")
    return f"features/fault_type={fault_type:02d}/features.parquet"
