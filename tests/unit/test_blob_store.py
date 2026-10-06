"""Tests for data/storage/blob_store.py.

El contrato (crear si no existe, nunca sobrescribir, listar por prefijo textual) se
ejecuta contra los DOS adaptadores con las mismas aserciones: es lo que garantiza
que lo probado en disco se comporta igual contra GCS. El doble de GCS lanza el
412 y el 404 reales de google.api_core.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import pytest
from google.api_core import exceptions as gexc
from google.cloud import storage

from data.storage.blob_store import (
    BlobStore,
    GCSBlobStore,
    LocalBlobStore,
    default_gcs_client,
)
from tests.support.fake_gcs import FakeGcsClient

_BUCKET = "reactorguard-data-raw-dev"
_TIMEOUT = 12.5

StoreFactory = Callable[[], BlobStore]


@pytest.fixture(params=["local", "gcs"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> BlobStore:
    """Return each adapter in turn, so every contract test runs against both."""
    if request.param == "local":
        return LocalBlobStore(tmp_path / "store")
    return GCSBlobStore(_BUCKET, timeout_s=_TIMEOUT, client_factory=FakeGcsClient)


# --------------------------------------------------------------------------- contrato


def test_put_creates_then_reports_existing(store: BlobStore) -> None:
    assert store.put_if_absent("a/b/part-1.parquet", b"first") is True
    assert store.put_if_absent("a/b/part-1.parquet", b"second") is False


def test_put_never_overwrites(store: BlobStore) -> None:
    store.put_if_absent("a/obj", b"first")
    store.put_if_absent("a/obj", b"second")
    assert store.get("a/obj") == b"first"


def test_get_returns_exact_bytes(store: BlobStore) -> None:
    payload = bytes(range(256)) * 4
    store.put_if_absent("bin", payload)
    assert store.get("bin") == payload


def test_get_missing_raises_file_not_found(store: BlobStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.get("does/not/exist")


def test_exists(store: BlobStore) -> None:
    assert store.exists("k") is False
    store.put_if_absent("k", b"x")
    assert store.exists("k") is True


def test_list_keys_is_sorted_and_filtered_by_prefix(store: BlobStore) -> None:
    for key in ["p/2/b", "p/1/b", "p/1/a", "q/1/a", "p/10/a"]:
        store.put_if_absent(key, b"x")
    assert store.list_keys("p/1/") == ["p/1/a", "p/1/b"]
    assert store.list_keys("p/") == ["p/1/a", "p/1/b", "p/10/a", "p/2/b"]
    assert store.list_keys("") == ["p/1/a", "p/1/b", "p/10/a", "p/2/b", "q/1/a"]


def test_list_keys_prefix_is_a_plain_string_match(store: BlobStore) -> None:
    """Like GCS, 'p/1' matches 'p/10/...' because it is not a directory listing."""
    for key in ["p/1/a", "p/10/a", "p/2/a"]:
        store.put_if_absent(key, b"x")
    assert store.list_keys("p/1") == ["p/1/a", "p/10/a"]


def test_list_keys_of_an_unknown_prefix_is_empty(store: BlobStore) -> None:
    store.put_if_absent("p/1/a", b"x")
    assert store.list_keys("nothing/here/") == []


def test_hive_style_keys_roundtrip(store: BlobStore) -> None:
    key = "plant=P1/year=2024/month=03/day=09/hour=07/part-abc123.parquet"
    store.put_if_absent(key, b"x")
    assert store.list_keys("plant=P1/year=2024/month=03/day=09/hour=07/") == [key]


def test_empty_payload_is_a_valid_object(store: BlobStore) -> None:
    assert store.put_if_absent("empty", b"") is True
    assert store.get("empty") == b""


# ------------------------------------------------------------------------- LocalBlobStore


@pytest.mark.parametrize(
    "key",
    ["", "/abs", "../escape", "a/../../escape", "a//b", "a/./b", "a\\b", "C:/x", ".tmp/x"],
)
def test_local_rejects_illegal_keys(tmp_path: Path, key: str) -> None:
    local = LocalBlobStore(tmp_path)
    with pytest.raises(ValueError, match="Illegal object key"):
        local.put_if_absent(key, b"x")
    with pytest.raises(ValueError, match="Illegal object key"):
        local.get(key)
    with pytest.raises(ValueError, match="Illegal object key"):
        local.exists(key)


@pytest.mark.parametrize("prefix", ["../x", "a/../b", "a\\b"])
def test_local_rejects_illegal_prefixes(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(ValueError, match="Illegal prefix"):
        LocalBlobStore(tmp_path).list_keys(prefix)


def test_local_refuses_a_symlink_that_escapes_the_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside.mkdir()
    local = LocalBlobStore(root)
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("this system does not allow creating symlinks")
    with pytest.raises(ValueError, match="escapes the store root"):
        local.put_if_absent("link/obj", b"x")
    assert list(outside.iterdir()) == []


def test_local_creates_its_root(tmp_path: Path) -> None:
    root = tmp_path / "deep" / "root"
    LocalBlobStore(root)
    assert root.is_dir()


def test_local_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    local = LocalBlobStore(tmp_path)
    local.put_if_absent("k", b"x")
    local.put_if_absent("k", b"y")  # el destino existe: el temporal tambien se limpia
    assert list((tmp_path / ".tmp").iterdir()) == []


def test_local_listing_never_exposes_temporary_files(tmp_path: Path) -> None:
    local = LocalBlobStore(tmp_path)
    (tmp_path / ".tmp").mkdir()
    (tmp_path / ".tmp" / "orphan.tmp").write_bytes(b"x")
    local.put_if_absent("k", b"x")
    assert local.list_keys("") == ["k"]


def test_local_state_survives_a_new_instance(tmp_path: Path) -> None:
    LocalBlobStore(tmp_path).put_if_absent("k", b"x")
    assert LocalBlobStore(tmp_path).get("k") == b"x"


def test_local_get_of_a_directory_is_file_not_found(tmp_path: Path) -> None:
    local = LocalBlobStore(tmp_path)
    local.put_if_absent("dir/obj", b"x")
    with pytest.raises(FileNotFoundError):
        local.get("dir")
    assert local.exists("dir") is False


# --------------------------------------------------------------------------- GCSBlobStore


def _gcs() -> tuple[GCSBlobStore, FakeGcsClient]:
    """Build a GCSBlobStore over a fake client and return both."""
    client = FakeGcsClient()
    return GCSBlobStore(_BUCKET, timeout_s=_TIMEOUT, client_factory=lambda: client), client


def test_gcs_upload_is_conditional_on_generation_zero() -> None:
    gcs, client = _gcs()
    gcs.put_if_absent("k", b"x")
    method, kwargs = client.calls[0]
    assert method == "upload_from_string"
    assert kwargs["if_generation_match"] == 0
    assert kwargs["timeout"] == _TIMEOUT


def test_gcs_reproduces_412_as_already_existing() -> None:
    gcs, client = _gcs()
    assert gcs.put_if_absent("k", b"first") is True
    assert gcs.put_if_absent("k", b"second") is False
    assert client.bucket(_BUCKET).objects["k"] == b"first"
    uploads = [kwargs for method, kwargs in client.calls if method == "upload_from_string"]
    assert len(uploads) == 2


def test_gcs_other_api_errors_propagate() -> None:
    gcs, client = _gcs()
    client.upload_error = gexc.Forbidden("ingestion-sa lacks permission")
    with pytest.raises(gexc.Forbidden):
        gcs.put_if_absent("k", b"x")


def test_gcs_get_maps_404_to_file_not_found() -> None:
    gcs, _ = _gcs()
    with pytest.raises(FileNotFoundError, match=f"gs://{_BUCKET}/missing"):
        gcs.get("missing")


def test_gcs_passes_the_timeout_to_every_operation() -> None:
    gcs, client = _gcs()
    gcs.put_if_absent("p/k", b"x")
    gcs.get("p/k")
    gcs.exists("p/k")
    gcs.list_keys("p/")
    assert {method for method, _ in client.calls} == {
        "upload_from_string", "download_as_bytes", "exists", "list_blobs",
    }
    assert all(kwargs["timeout"] == _TIMEOUT for _, kwargs in client.calls)


def test_gcs_lists_with_the_prefix() -> None:
    gcs, client = _gcs()
    gcs.put_if_absent("p/k", b"x")
    gcs.list_keys("p/")
    assert ("list_blobs", {"prefix": "p/", "timeout": _TIMEOUT}) in client.calls


def test_gcs_uses_the_named_bucket() -> None:
    gcs, client = _gcs()
    gcs.put_if_absent("k", b"x")
    assert set(client.buckets) == {_BUCKET}


def test_gcs_calls_use_keyword_names_of_the_real_library() -> None:
    """A misspelled kwarg would fail in the cluster; check the real signatures."""
    gcs, client = _gcs()
    gcs.put_if_absent("p/k", b"x")
    gcs.get("p/k")
    gcs.exists("p/k")
    gcs.list_keys("p/")
    real = {
        "upload_from_string": storage.Blob.upload_from_string,
        "download_as_bytes": storage.Blob.download_as_bytes,
        "exists": storage.Blob.exists,
        "list_blobs": storage.Client.list_blobs,
    }
    for method, kwargs in client.calls:
        accepted = set(inspect.signature(real[method]).parameters)
        assert set(kwargs) - {"name"} <= accepted, (method, kwargs)


def test_default_client_builds_without_network_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an emulator host the real client is anonymous: no credentials, no call."""
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://localhost:9")
    client = default_gcs_client()
    assert isinstance(client, storage.Client)
    gcs = GCSBlobStore(_BUCKET, timeout_s=_TIMEOUT)
    assert gcs is not None
