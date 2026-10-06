"""Blob stores: the port of the storage client and its two adapters.

`BlobStore` es el puerto (D11): cuatro operaciones sobre claves de objeto. Sus
adaptadores son GCSBlobStore (google-cloud-storage) y LocalBlobStore (disco), con
el MISMO contrato, de modo que la logica de particionado de StorageClient se
prueba sin red ni credenciales y se ejecuta igual en el cluster.

El contrato de escritura es "crear si no existe" y nunca sobrescribe. No es una
preferencia: ingestion-sa solo tiene el rol objectCreator en data-raw, asi que
sobrescribir devolveria 403, y un reintento at-least-once del mismo lote
reescribe el mismo objeto. En GCS eso se expresa con `if_generation_match=0`
(la escritura solo prospera si el objeto no existe) y el 412 resultante significa
"ya estaba", es decir, exito idempotente.

Los errores de la libreria que no son "ya existia" ni "no existe" (403, 5xx,
timeouts) se propagan sin envolver: el llamador no puede hacer nada util con una
excepcion propia que no distinga mas que la original.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from google.api_core import exceptions as gexc

_LOG = logging.getLogger(__name__)

_CONTENT_TYPE = "application/octet-stream"
_TMP_DIR = ".tmp"


class BlobStore(Protocol):
    """Minimal object-store interface the storage client depends on.

    `list_keys` se llama asi y no `list` porque un metodo `list` dentro de la
    clase oculta el builtin en las anotaciones `list[str]` del propio cuerpo.
    """

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Store data under key unless an object already exists there.

        Args:
            key: Object key.
            data: Object content.

        Returns:
            True if the object was created, False if it already existed (in which
            case the stored content is left untouched).
        """
        ...

    def get(self, key: str) -> bytes:
        """Return the content of an object.

        Args:
            key: Object key.

        Returns:
            The stored bytes.

        Raises:
            FileNotFoundError: If no object exists under key.
        """
        ...

    def list_keys(self, prefix: str) -> list[str]:
        """List the keys that start with a prefix.

        Args:
            prefix: Raw string prefix (not necessarily ending in "/").

        Returns:
            Matching keys in lexicographic order.
        """
        ...

    def exists(self, key: str) -> bool:
        """Tell whether an object exists.

        Args:
            key: Object key.

        Returns:
            True if an object is stored under key.
        """
        ...


def default_gcs_client() -> Any:
    """Create the real google-cloud-storage client from ambient credentials.

    Returns:
        A `google.cloud.storage.Client`. Construction does not touch the network;
        credentials come from Application Default Credentials (Workload Identity
        in the cluster) or STORAGE_EMULATOR_HOST for an emulator.
    """
    from google.cloud import storage

    return storage.Client()


class GCSBlobStore:
    """BlobStore backed by one Google Cloud Storage bucket."""

    def __init__(
        self,
        bucket_name: str,
        *,
        timeout_s: float,
        client_factory: Callable[[], Any] = default_gcs_client,
    ) -> None:
        """Bind the store to a bucket.

        Args:
            bucket_name: Name of an existing bucket.
            timeout_s: Timeout of every request against GCS.
            client_factory: Builds the underlying client. Tests inject an
                in-memory double; production uses the default.
        """
        self._bucket_name = bucket_name
        self._timeout_s = timeout_s
        self._client = client_factory()
        self._bucket = self._client.bucket(bucket_name)

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Create the object only if it does not exist (if_generation_match=0).

        Args:
            key: Object key.
            data: Object content.

        Returns:
            True if created, False if the precondition failed (HTTP 412), i.e.
            the object was already there.

        Raises:
            google.api_core.exceptions.GoogleAPICallError: On any other API
                failure (403 without objectCreator, 5xx after the library's
                own retries, timeouts).
        """
        blob = self._bucket.blob(key)
        try:
            blob.upload_from_string(
                data,
                content_type=_CONTENT_TYPE,
                if_generation_match=0,
                timeout=self._timeout_s,
            )
        except gexc.PreconditionFailed:
            _LOG.debug("Object gs://%s/%s already exists", self._bucket_name, key)
            return False
        return True

    def get(self, key: str) -> bytes:
        """Download an object.

        Args:
            key: Object key.

        Returns:
            The stored bytes.

        Raises:
            FileNotFoundError: If the object does not exist (HTTP 404).
            google.api_core.exceptions.GoogleAPICallError: On any other failure.
        """
        blob = self._bucket.blob(key)
        try:
            content: bytes = blob.download_as_bytes(timeout=self._timeout_s)
        except gexc.NotFound as exc:
            raise FileNotFoundError(f"gs://{self._bucket_name}/{key} does not exist.") from exc
        return content

    def list_keys(self, prefix: str) -> list[str]:
        """List the object names that start with a prefix.

        Args:
            prefix: Raw string prefix.

        Returns:
            Matching names in lexicographic order.
        """
        blobs = self._client.list_blobs(
            self._bucket_name, prefix=prefix, timeout=self._timeout_s
        )
        return sorted(str(blob.name) for blob in blobs)

    def exists(self, key: str) -> bool:
        """Tell whether an object exists.

        Args:
            key: Object key.

        Returns:
            True if the object exists.
        """
        return bool(self._bucket.blob(key).exists(timeout=self._timeout_s))


class LocalBlobStore:
    """BlobStore backed by a directory tree, with the same contract as GCS."""

    def __init__(self, root: str | Path) -> None:
        """Bind the store to a root directory, creating it if needed.

        Args:
            root: Directory that plays the role of the bucket.
        """
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._resolved_root = self._root.resolve()

    def _path(self, key: str) -> Path:
        """Map an object key to a path, refusing anything outside the root.

        Args:
            key: Object key.

        Returns:
            The path under the root where the object lives.

        Raises:
            ValueError: If the key is empty, absolute, contains backslashes,
                drive letters, or "." / ".." segments, or uses the reserved
                top-level directory for temporary files.
        """
        parts = PurePosixPath(key).parts
        illegal = (
            not key
            or key.startswith("/")
            or "\\" in key
            or ":" in key
            or any(part in {"", ".", ".."} for part in key.split("/"))
            or parts[0] == _TMP_DIR
        )
        if illegal:
            raise ValueError(f"Illegal object key {key!r}.")
        path = (self._root / Path(*parts)).resolve()
        if self._resolved_root not in path.parents:
            raise ValueError(f"Object key {key!r} escapes the store root.")
        return path

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Create the object atomically unless it already exists.

        Se escribe a un fichero temporal dentro del propio arbol y se publica con
        `os.link`, que falla con FileExistsError si el destino existe. Asi el
        objeto aparece completo o no aparece, y dos escritores concurrentes de la
        misma clave no pueden pisarse: es el equivalente local de if_generation_match=0.

        Args:
            key: Object key.
            data: Object content.

        Returns:
            True if created, False if it already existed.

        Raises:
            ValueError: If the key is illegal.
            OSError: If the file system rejects the write.
        """
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = self._root / _TMP_DIR
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / f"{uuid.uuid4().hex}.tmp"
        try:
            with tmp_path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(tmp_path, target)
            except FileExistsError:
                return False
            return True
        finally:
            tmp_path.unlink(missing_ok=True)

    def get(self, key: str) -> bytes:
        """Read an object.

        Args:
            key: Object key.

        Returns:
            The stored bytes.

        Raises:
            ValueError: If the key is illegal.
            FileNotFoundError: If the object does not exist.
        """
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist.")
        return path.read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        """List the object keys that start with a prefix.

        Args:
            prefix: Raw string prefix, with the same semantics as GCS: a plain
                string match on the whole key, not a directory listing.

        Returns:
            Matching keys, in lexicographic order, using "/" separators.

        Raises:
            ValueError: If the prefix contains ".." segments or backslashes.
        """
        if "\\" in prefix or ".." in prefix.split("/"):
            raise ValueError(f"Illegal prefix {prefix!r}.")
        base = self._root
        directory = prefix.rpartition("/")[0]
        if directory:
            base = base / directory
        if not base.is_dir():
            return []
        keys = (
            path.relative_to(self._root).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        )
        return sorted(
            key for key in keys if key.startswith(prefix) and not key.startswith(f"{_TMP_DIR}/")
        )

    def exists(self, key: str) -> bool:
        """Tell whether an object exists.

        Args:
            key: Object key.

        Returns:
            True if the object exists.

        Raises:
            ValueError: If the key is illegal.
        """
        return self._path(key).is_file()
