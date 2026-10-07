"""Sources of long-format TEP readings for the streamer.

`ReadingSource` es el puerto: entrega los runs del TEP uno tras otro como pares
(run_id, DataFrame) en formato largo, el mismo que escribe adapt_tep. El streamer
no sabe de donde vienen. Dos adaptadores:

- `ParquetDirSource`: `<root>/fault_type=NN/readings.parquet` en disco, el out de
  DVC del stage adapt_tep.
- `StorageSource`: el mismo arbol dentro de un `BlobStore` (bucket data-raw), tal y
  como lo deja `gsutil rsync data/processed/tep gs://<bucket>/tep/` (no hay remoto
  DVC). Se apoya en BlobStore y no en StorageClient a proposito: StorageClient lee el
  layout horario hive, y los 22 runs del TEP comparten linea temporal (todos desde
  2000-01-01), asi que caerian en las mismas particiones horarias y el run no se
  podria reconstruir.

Los runs se entregan ordenados por numero de fault_type y UNO A UNO: nunca se
entrelazan (romperia la monotonia del timestamp por sensor) y solo uno esta en
memoria a la vez.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import pandas as pd

from data.storage.blob_store import BlobStore

_LOG = logging.getLogger(__name__)

RunFrame = tuple[str, pd.DataFrame]

READINGS_FILE = "readings.parquet"
_RUN_DIR_RE = re.compile(r"fault_type=(\d+)")


def run_name(fault_type: int) -> str:
    """Return the canonical run identifier of a fault type.

    Args:
        fault_type: Integer fault identifier (0 = normal, 1-21 = fault).

    Returns:
        The identifier, e.g. "fault_type=07", the same spelling as the partition.
    """
    return f"fault_type={fault_type:02d}"


class ReadingSource(Protocol):
    """Provides the TEP runs in order, one long-format frame per run."""

    def runs(self) -> Iterator[RunFrame]:
        """Iterate the runs, ordered by fault type.

        Se puede llamar de nuevo para otra pasada (modo loop). Los errores de
        descubrimiento (origen inexistente o vacio) se lanzan en esta llamada, no
        al consumir el primer elemento.

        Returns:
            An iterator of (run_id, frame) pairs.

        Raises:
            FileNotFoundError: If the origin does not exist or holds no run.
            ValueError: If a run cannot be decoded.
        """
        ...


def _decode(label: str, origin: io.BytesIO | Path) -> pd.DataFrame:
    """Read one parquet file, naming it in any decoding error.

    Args:
        label: Name of the file, for the error message.
        origin: A path or an in-memory buffer holding the parquet file.

    Returns:
        The decoded frame.

    Raises:
        ValueError: If the content is not a readable parquet file.
    """
    try:
        return pd.read_parquet(origin)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{label} is not a readable parquet file: {exc}") from exc


class ParquetDirSource:
    """Runs stored as `<root>/fault_type=NN/readings.parquet` on disk."""

    def __init__(self, root: str | Path) -> None:
        """Bind the source to a directory.

        Args:
            root: Directory holding the fault_type partitions.
        """
        self._root = Path(root)

    def runs(self) -> Iterator[RunFrame]:
        """Iterate the runs found under the root.

        Returns:
            An iterator of (run_id, frame) pairs, ordered by fault type.

        Raises:
            FileNotFoundError: If the root does not exist or holds no run.
            ValueError: If a partition is not a readable parquet file.
        """
        if not self._root.is_dir():
            raise FileNotFoundError(f"TEP readings directory not found: {self._root}")
        found: list[tuple[int, Path]] = []
        for path in self._root.glob(f"fault_type=*/{READINGS_FILE}"):
            match = _RUN_DIR_RE.fullmatch(path.parent.name)
            if match is not None:
                found.append((int(match.group(1)), path))
        if not found:
            raise FileNotFoundError(
                f"No fault_type=NN/{READINGS_FILE} under {self._root}. "
                "Run the adapt_tep stage with .\\infra\\scripts\\Invoke-Pipeline.ps1."
            )
        found.sort()
        return self._iterate(found)

    def _iterate(self, found: list[tuple[int, Path]]) -> Iterator[RunFrame]:
        """Yield the frames lazily.

        Args:
            found: (fault_type, path) pairs, already ordered.

        Yields:
            The (run_id, frame) pairs.
        """
        for fault_type, path in found:
            frame = _decode(str(path), path)
            _LOG.info("Loaded %d readings from %s", len(frame), path)
            yield run_name(fault_type), frame


class StorageSource:
    """Runs stored as `<prefix>/fault_type=NN/readings.parquet` in a BlobStore."""

    def __init__(self, store: BlobStore, prefix: str) -> None:
        """Bind the source to a store and an object prefix.

        Args:
            store: Backend holding the objects (GCS bucket or local directory).
            prefix: Object prefix of the tree, e.g. "tep". May be empty.
        """
        self._store = store
        self._prefix = prefix.strip("/")
        base = re.escape(self._prefix + "/") if self._prefix else ""
        self._key_re = re.compile(rf"{base}fault_type=(\d+)/{re.escape(READINGS_FILE)}")

    def runs(self) -> Iterator[RunFrame]:
        """Iterate the runs found under the prefix.

        Returns:
            An iterator of (run_id, frame) pairs, ordered by fault type.

        Raises:
            FileNotFoundError: If no run is stored under the prefix.
            ValueError: If an object is not a readable parquet file.
        """
        listing_prefix = f"{self._prefix}/fault_type=" if self._prefix else "fault_type="
        found: list[tuple[int, str]] = []
        for key in self._store.list_keys(listing_prefix):
            match = self._key_re.fullmatch(key)
            if match is not None:
                found.append((int(match.group(1)), key))
        if not found:
            raise FileNotFoundError(
                f"No '{listing_prefix}NN/{READINGS_FILE}' objects in the store. Upload the "
                "adapted readings with: gsutil -m rsync -r data/processed/tep "
                "gs://<raw bucket>/<prefix>/"
            )
        found.sort()
        return self._iterate(found)

    def _iterate(self, found: list[tuple[int, str]]) -> Iterator[RunFrame]:
        """Download and decode the objects lazily.

        Args:
            found: (fault_type, key) pairs, already ordered.

        Yields:
            The (run_id, frame) pairs.

        Raises:
            FileNotFoundError: If a listed object disappears before it is read.
        """
        for fault_type, key in found:
            frame = _decode(key, io.BytesIO(self._store.get(key)))
            _LOG.info("Loaded %d readings from object %s", len(frame), key)
            yield run_name(fault_type), frame
