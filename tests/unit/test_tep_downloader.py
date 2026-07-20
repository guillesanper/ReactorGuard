"""Unit tests for data.generators.tep_downloader.

Ningun test toca la red: requests.get se sustituye por un doble que sirve
contenido en memoria y cuenta invocaciones. Eso permite verificar las dos
propiedades que importan del downloader (idempotencia por MD5 y reintento con
backoff) de forma determinista y sin depender de GitHub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import requests

from data.generators import tep_downloader
from data.generators.tep_downloader import (
    _CHECKSUM_FILENAME,
    _FILE_NAMES,
    _compute_md5,
    download_tep,
    main,
)

from .conftest import WriteParams


class _FakeResponse:
    """Minimal stand-in for a streamed requests.Response."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {"content-length": str(len(payload))}

    def raise_for_status(self) -> None:
        """Successful responses raise nothing."""

    def iter_content(self, chunk_size: int) -> Any:
        """Yield the payload in chunk_size slices."""
        for start in range(0, len(self._payload), chunk_size):
            yield self._payload[start : start + chunk_size]


class _Recorder:
    """Callable replacement for requests.get that records the URLs requested."""

    def __init__(self, payload: bytes = b"1.0 2.0\n3.0 4.0\n") -> None:
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str, **_kwargs: Any) -> _FakeResponse:
        self.urls.append(url)
        return _FakeResponse(self.payload)


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Patch requests.get in the downloader module and return the recorder."""
    rec = _Recorder()
    monkeypatch.setattr(tep_downloader.requests, "get", rec)
    return rec


class TestDownloadAllFiles:
    """A first run must fetch every TEP file and record its checksum."""

    def test_writes_all_22_files(self, tmp_path: Path, recorder: _Recorder) -> None:
        """All 22 .dat files must exist after a successful run."""
        download_tep(str(tmp_path))
        for name in _FILE_NAMES:
            assert (tmp_path / name).exists(), name

    def test_requests_one_url_per_file(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """Exactly one HTTP request per file must be issued."""
        download_tep(str(tmp_path))
        assert len(recorder.urls) == len(_FILE_NAMES)

    def test_creates_missing_output_directory(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """A non-existent output directory must be created."""
        target = tmp_path / "nested" / "tep"
        download_tep(str(target))
        assert target.is_dir()

    def test_writes_checksum_manifest(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """The manifest must record an MD5 for every file."""
        download_tep(str(tmp_path))
        manifest = json.loads((tmp_path / _CHECKSUM_FILENAME).read_text())
        assert set(manifest) == set(_FILE_NAMES)

    def test_checksums_match_file_contents(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """Recorded checksums must match the bytes actually on disk."""
        download_tep(str(tmp_path))
        manifest = json.loads((tmp_path / _CHECKSUM_FILENAME).read_text())
        assert manifest["d00.dat"] == _compute_md5(tmp_path / "d00.dat")


class TestIdempotency:
    """Re-running must not re-fetch files whose checksum still matches."""

    def test_second_run_issues_no_requests(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """A clean second run must skip every file."""
        download_tep(str(tmp_path))
        recorder.urls.clear()
        download_tep(str(tmp_path))
        assert recorder.urls == []

    def test_corrupted_file_is_refetched(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """A file whose contents no longer match its checksum must be re-fetched."""
        download_tep(str(tmp_path))
        (tmp_path / "d05.dat").write_bytes(b"corrupted")
        recorder.urls.clear()
        download_tep(str(tmp_path))
        assert [u.rsplit("/", 1)[-1] for u in recorder.urls] == ["d05.dat"]

    def test_corrupted_file_is_restored(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """The re-fetched file must end up with the correct contents again."""
        download_tep(str(tmp_path))
        (tmp_path / "d05.dat").write_bytes(b"corrupted")
        download_tep(str(tmp_path))
        assert (tmp_path / "d05.dat").read_bytes() == recorder.payload

    def test_file_present_without_manifest_is_refetched(
        self, tmp_path: Path, recorder: _Recorder
    ) -> None:
        """An untracked pre-existing file must be re-downloaded, not trusted."""
        (tmp_path / "d00.dat").write_bytes(recorder.payload)
        download_tep(str(tmp_path))
        assert "d00.dat" in [u.rsplit("/", 1)[-1] for u in recorder.urls]


class TestRetryBehaviour:
    """Transient failures must be retried; permanent ones must surface."""

    def test_retries_then_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A single transient failure must not abort the download."""
        monkeypatch.setattr(tep_downloader.time, "sleep", lambda _s: None)
        attempts = {"n": 0}

        def flaky(url: str, **_kwargs: Any) -> _FakeResponse:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise requests.ConnectionError("transient")
            return _FakeResponse(b"1.0 2.0\n")

        monkeypatch.setattr(tep_downloader.requests, "get", flaky)
        download_tep(str(tmp_path))
        assert (tmp_path / "d00.dat").exists()

    def test_raises_after_exhausting_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Persistent failures must raise RuntimeError naming the attempt count."""
        monkeypatch.setattr(tep_downloader.time, "sleep", lambda _s: None)

        def always_fails(url: str, **_kwargs: Any) -> _FakeResponse:
            raise requests.ConnectionError("down")

        monkeypatch.setattr(tep_downloader.requests, "get", always_fails)
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            download_tep(str(tmp_path))


class TestMainEntryPoint:
    """main() must resolve its output directory from params.yaml."""

    def test_downloads_into_configured_raw_dir(
        self, tmp_path: Path, recorder: _Recorder, write_params: WriteParams
    ) -> None:
        """The raw_dir from the params file must be used as the destination."""
        raw_dir = tmp_path / "configured_raw"
        main(write_params(raw_dir=str(raw_dir)))
        assert (raw_dir / "d00.dat").exists()
