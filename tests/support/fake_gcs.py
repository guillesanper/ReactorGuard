"""In-memory double of the slice of google.cloud.storage that GCSBlobStore uses.

No usa red ni credenciales. Reproduce las dos semanticas de las que depende el
cliente de almacenamiento: `if_generation_match=0` falla con 412
(`PreconditionFailed`) si el objeto existe, y descargar un objeto inexistente
falla con 404 (`NotFound`). Lanza las excepciones REALES de `google.api_core`, no
sustitutos, de modo que los `except` del adaptador se ejercitan tal cual.

Registra cada llamada con sus kwargs para que los tests comprueben que existen en
la firma real de la libreria (un nombre mal escrito fallaria aqui y no en el cluster).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from google.api_core import exceptions as gexc


class FakeBlob:
    """Double of google.cloud.storage.Blob."""

    def __init__(self, bucket: FakeBucket, name: str) -> None:
        """Create a handle to an object that may or may not exist.

        Args:
            bucket: Owning fake bucket.
            name: Object name.
        """
        self.bucket = bucket
        self.name = name

    def upload_from_string(
        self,
        data: bytes,
        content_type: str = "text/plain",
        if_generation_match: int | None = None,
        timeout: float | None = None,
    ) -> None:
        """Store the object, enforcing a generation-0 precondition when requested.

        Args:
            data: Object content.
            content_type: Declared media type.
            if_generation_match: 0 means "only if the object does not exist".
            timeout: Request timeout, recorded only.

        Raises:
            PreconditionFailed: If if_generation_match is 0 and the object exists.
        """
        self.bucket.record(
            "upload_from_string",
            name=self.name,
            content_type=content_type,
            if_generation_match=if_generation_match,
            timeout=timeout,
        )
        if self.bucket.client.upload_error is not None:
            raise self.bucket.client.upload_error
        if if_generation_match == 0 and self.name in self.bucket.objects:
            raise gexc.PreconditionFailed(
                f"conditionNotMet: object {self.name} already exists"
            )
        self.bucket.objects[self.name] = data

    def download_as_bytes(self, timeout: float | None = None) -> bytes:
        """Return the object content.

        Args:
            timeout: Request timeout, recorded only.

        Returns:
            The stored bytes.

        Raises:
            NotFound: If the object does not exist.
        """
        self.bucket.record("download_as_bytes", name=self.name, timeout=timeout)
        if self.name not in self.bucket.objects:
            raise gexc.NotFound(f"No such object: {self.name}")
        return self.bucket.objects[self.name]

    def exists(self, timeout: float | None = None) -> bool:
        """Tell whether the object exists.

        Args:
            timeout: Request timeout, recorded only.

        Returns:
            True if the object is stored.
        """
        self.bucket.record("exists", name=self.name, timeout=timeout)
        return self.name in self.bucket.objects


class FakeBucket:
    """Double of google.cloud.storage.Bucket, owning an in-memory object map."""

    def __init__(self, client: FakeGcsClient, name: str) -> None:
        """Create the bucket.

        Args:
            client: Owning fake client.
            name: Bucket name.
        """
        self.client = client
        self.name = name
        self.objects: dict[str, bytes] = {}

    def blob(self, name: str) -> FakeBlob:
        """Return a handle to an object.

        Args:
            name: Object name.

        Returns:
            A FakeBlob.
        """
        return FakeBlob(self, name)

    def record(self, method: str, **kwargs: Any) -> None:
        """Append a call to the client's log.

        Args:
            method: Name of the invoked method.
            **kwargs: Keyword arguments the caller passed.
        """
        self.client.calls.append((method, kwargs))


class FakeGcsClient:
    """Double of google.cloud.storage.Client."""

    def __init__(self) -> None:
        """Create an empty client with no buckets."""
        self.buckets: dict[str, FakeBucket] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        # Si se asigna, toda subida falla con esta excepcion (p. ej. un 403).
        self.upload_error: Exception | None = None

    def bucket(self, name: str) -> FakeBucket:
        """Return the bucket with that name, creating it on first use.

        Args:
            name: Bucket name.

        Returns:
            The FakeBucket.
        """
        return self.buckets.setdefault(name, FakeBucket(self, name))

    def list_blobs(
        self,
        bucket_or_name: str,
        prefix: str | None = None,
        timeout: float | None = None,
    ) -> Iterator[FakeBlob]:
        """Iterate the objects whose name starts with a prefix, in name order.

        Args:
            bucket_or_name: Bucket name.
            prefix: Name prefix filter.
            timeout: Request timeout, recorded only.

        Yields:
            One FakeBlob per matching object.
        """
        bucket = self.bucket(bucket_or_name)
        self.calls.append(("list_blobs", {"prefix": prefix, "timeout": timeout}))
        for name in sorted(bucket.objects):
            if prefix is None or name.startswith(prefix):
                yield FakeBlob(bucket, name)
