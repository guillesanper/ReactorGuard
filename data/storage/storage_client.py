"""Storage client for sensor readings: hourly partitioning over a BlobStore.

`StorageClient` contiene toda la logica (agrupar por hora, nombrar partes,
codificar, listar un rango, descartar duplicados) y solo conoce el puerto
`BlobStore`. `GCSStorageClient` y `LocalCache` son subclases finas que unicamente
eligen el almacen, de modo que lo que se prueba en disco es lo que corre contra GCS.

Hoy ningun proceso del pipeline llama a `write_sensor_readings`: este modulo es el
cliente que desbloquea el sumidero de archivo de las lecturas y la fuente del
streamer en el cluster. Queda fuera de alcance, a proposito, la escritura de
simulaciones y de modelos (Fases 3 y 4): no tienen productor ni consumidor, y
ingestion-sa tampoco puede escribir en el bucket de modelos.

Semantica de escritura (at-least-once, D4): una escritura que falla a medias se
reintenta ENTERA. Las partes ya escritas se reconocen por su nombre determinista y
no se duplican; las lecturas que un reintento reparta en lotes con otras fronteras
pueden repetirse entre partes, y la lectura las descarta por reading_id.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from data.schemas.sensor_reading import SensorReading
from data.storage.blob_store import BlobStore, GCSBlobStore, LocalBlobStore, default_gcs_client
from data.storage.layout import (
    hour_prefixes,
    is_part_key,
    part_id,
    readings_object,
    readings_prefix,
    to_utc,
)
from data.storage.parquet_codec import (
    bytes_to_table,
    readings_to_table,
    table_to_bytes,
    table_to_readings,
)
from data.storage.storage_params import StorageParams

_LOG = logging.getLogger(__name__)

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    """Return the current instant as an aware UTC datetime."""
    return datetime.now(UTC)


class StorageBackend(StrEnum):
    """Backends selectable through get_storage_client."""

    GCS = "gcs"
    LOCAL = "local"


@dataclass(frozen=True)
class WriteResult:
    """Outcome of one write_sensor_readings call.

    Attributes:
        readings: Number of readings handed to the call.
        created: Keys of the parts this call created.
        existing: Keys of the parts that were already stored (idempotent retry).
    """

    readings: int
    created: tuple[str, ...]
    existing: tuple[str, ...]


class StorageClient:
    """Reads and writes sensor readings partitioned by plant and UTC hour."""

    def __init__(
        self,
        store: BlobStore,
        params: StorageParams,
        *,
        clock: Clock = _utc_now,
    ) -> None:
        """Bind the client to a blob store.

        Args:
            store: Backend holding the objects.
            params: Storage parameters (read parallelism).
            clock: Source of the created_at metadata stamp. Tests inject a fixed
                instant.
        """
        self._store = store
        self._params = params
        self._clock = clock

    def write_sensor_readings(self, readings: Sequence[SensorReading]) -> WriteResult:
        """Store a batch of readings, one part per plant-hour it touches.

        Dentro de cada particion se conserva el orden de entrada. El nombre de la
        parte sale de (primer reading_id, ultimo, numero de lecturas), por lo que
        reenviar el mismo lote es un no-op.

        Args:
            readings: Readings to store. May span several plants and hours.

        Returns:
            Which parts were created and which already existed.

        Raises:
            ValueError: If a reading has a naive timestamp or an unsafe plant_id.
            google.api_core.exceptions.GoogleAPICallError: On a GCS failure other
                than "already exists". Parts written before the failure stay; the
                caller retries the whole batch.
            OSError: If the local file system rejects a write.
        """
        groups: dict[str, list[SensorReading]] = {}
        for reading in readings:
            prefix = readings_prefix(reading.plant_id, reading.timestamp)
            groups.setdefault(prefix, []).append(reading)

        created: list[str] = []
        existing: list[str] = []
        created_at = self._clock()
        for prefix, group in groups.items():
            name = part_id(str(group[0].reading_id), str(group[-1].reading_id), len(group))
            key = readings_object(prefix, name)
            payload = table_to_bytes(readings_to_table(group, created_at))
            (created if self._store.put_if_absent(key, payload) else existing).append(key)

        _LOG.info(
            "Stored %d readings: %d new parts, %d already present",
            len(readings),
            len(created),
            len(existing),
        )
        return WriteResult(
            readings=len(readings), created=tuple(created), existing=tuple(existing)
        )

    def read_sensor_readings(
        self, plant_id: str, start: datetime, end: datetime
    ) -> list[SensorReading]:
        """Read the readings of one plant in the half-open range [start, end).

        Solo se listan los prefijos horarios que el rango toca, y las partes se
        descargan con un pool de `read_workers` hilos. El coste es proporcional al
        numero de horas del rango: un rango de meses implica miles de listados.

        Args:
            plant_id: Plant identifier.
            start: Inclusive lower bound, timezone-aware.
            end: Exclusive upper bound, timezone-aware.

        Returns:
            The readings ordered by timestamp (stable, so ties keep storage
            order), without duplicates by reading_id.

        Raises:
            ValueError: If a bound has no timezone, start is after end, plant_id
                is unsafe, or a stored part is corrupt or has another schema.
            FileNotFoundError: If a listed part disappears before it is read.
        """
        prefixes = hour_prefixes(plant_id, start, end)
        if not prefixes:
            return []
        start_utc = to_utc(start)
        end_utc = to_utc(end)

        with ThreadPoolExecutor(max_workers=self._params.read_workers) as pool:
            listings = pool.map(self._store.list_keys, prefixes)
            keys = [key for listing in listings for key in listing if is_part_key(key)]
            parts = list(pool.map(self._load_part, keys))

        seen: set[str] = set()
        selected: list[SensorReading] = []
        for part in parts:
            for reading in part:
                if not start_utc <= reading.timestamp < end_utc:
                    continue
                identifier = str(reading.reading_id)
                if identifier in seen:
                    continue
                seen.add(identifier)
                selected.append(reading)
        selected.sort(key=lambda reading: reading.timestamp)
        _LOG.info(
            "Read %d readings of %s from %d parts in %d hourly partitions",
            len(selected),
            plant_id,
            len(keys),
            len(prefixes),
        )
        return selected

    def _load_part(self, key: str) -> list[SensorReading]:
        """Download and decode one part.

        Args:
            key: Object key of the part.

        Returns:
            The readings it holds.

        Raises:
            ValueError: If the object is not a valid part, naming the key.
            FileNotFoundError: If the object does not exist.
        """
        try:
            return table_to_readings(bytes_to_table(self._store.get(key)))
        except ValueError as exc:
            raise ValueError(f"Part '{key}' is corrupt or incompatible: {exc}") from exc


class GCSStorageClient(StorageClient):
    """StorageClient over the raw-data GCS bucket."""

    def __init__(
        self,
        params: StorageParams,
        *,
        bucket_name: str | None = None,
        client_factory: Callable[[], Any] = default_gcs_client,
        clock: Clock = _utc_now,
    ) -> None:
        """Bind to a GCS bucket.

        Args:
            params: Storage parameters.
            bucket_name: Bucket to use; defaults to the raw-data bucket derived
                from params.
            client_factory: Builds the google-cloud-storage client. Tests inject
                an in-memory double.
            clock: Source of the created_at stamp.
        """
        store = GCSBlobStore(
            bucket_name or params.raw_bucket,
            timeout_s=params.request_timeout_s,
            client_factory=client_factory,
        )
        super().__init__(store, params, clock=clock)


class LocalCache(StorageClient):
    """StorageClient over a local directory, with the same interface as GCS."""

    def __init__(
        self,
        params: StorageParams,
        *,
        root: str | Path | None = None,
        clock: Clock = _utc_now,
    ) -> None:
        """Bind to a directory.

        Args:
            params: Storage parameters.
            root: Directory to use; defaults to params.local_root.
            clock: Source of the created_at stamp.
        """
        super().__init__(LocalBlobStore(root or params.local_root), params, clock=clock)


def get_storage_client(
    backend: str | StorageBackend,
    params: StorageParams,
    *,
    clock: Clock = _utc_now,
) -> StorageClient:
    """Build the storage client of a backend.

    Args:
        backend: "gcs" or "local".
        params: Storage parameters.
        clock: Source of the created_at stamp.

    Returns:
        A GCSStorageClient or a LocalCache.

    Raises:
        ValueError: If backend is not a known value.
    """
    try:
        selected = StorageBackend(backend)
    except ValueError:
        allowed = sorted(member.value for member in StorageBackend)
        raise ValueError(
            f"Unknown storage backend {backend!r}; expected one of {allowed}."
        ) from None
    if selected is StorageBackend.GCS:
        return GCSStorageClient(params, clock=clock)
    return LocalCache(params, clock=clock)
