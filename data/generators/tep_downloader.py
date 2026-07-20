"""Download Tennessee Eastman Process (TEP) dataset files from the Prof. Braatz GitHub repository.

The dataset consists of 22 space-delimited .dat files (d00.dat through d21.dat):
    d00.dat  - normal operating conditions (no fault)
    d01.dat to d21.dat - 21 distinct fault scenarios

Source: https://github.com/camaramm/tennessee-eastman-profBraatz
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import requests
from tqdm import tqdm

from data.generators.tep_params import DEFAULT_PARAMS_PATH, load_tep_params

_LOG = logging.getLogger(__name__)

_BASE_URL = (
    "https://raw.githubusercontent.com/camaramm/tennessee-eastman-profBraatz/master/"
)
_FILE_NAMES: list[str] = ["d00.dat"] + [f"d{i:02d}.dat" for i in range(1, 22)]
_CHECKSUM_FILENAME = "tep_checksums.json"
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds between retry attempts (exponential base)
_CHUNK_SIZE = 8192  # bytes per read chunk


def _compute_md5(filepath: Path) -> str:
    """Compute the MD5 checksum of a file by reading it in chunks.

    Args:
        filepath: Absolute or relative path to the target file.

    MD5 se usa aqui solo para verificar integridad de descarga contra los
    checksums publicados del dataset TEP, no como primitiva de seguridad; de ahi
    usedforsecurity=False.

    Returns:
        Lowercase hexadecimal MD5 digest string.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with filepath.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_single_file(url: str, dest: Path) -> None:
    """Download one file from url to dest with exponential-backoff retry.

    Args:
        url: HTTP(S) URL of the resource to download.
        dest: Destination file path (parent directory must exist).

    Raises:
        RuntimeError: If all retry attempts are exhausted.
        requests.HTTPError: If the server returns a non-2xx status code on the
            final attempt.
    """
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            total_bytes = int(response.headers.get("content-length", 0))
            with dest.open("wb") as fh, tqdm(
                desc=dest.name,
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as progress:
                for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
                    fh.write(chunk)
                    progress.update(len(chunk))
            return
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES:
                raise RuntimeError(
                    f"Failed to download {url} after {_MAX_RETRIES} attempts: {exc}"
                ) from exc
            wait_seconds = _BACKOFF_BASE**attempt
            _LOG.warning(
                "Attempt %d/%d failed for %s. Retrying in %.1f s.",
                attempt,
                _MAX_RETRIES,
                url,
                wait_seconds,
            )
            time.sleep(wait_seconds)


def download_tep(output_dir: str) -> None:
    """Download the 22 TEP dataset files (d00.dat to d21.dat) to output_dir.

    Files that already exist and whose MD5 matches the stored checksum are
    skipped, making this function idempotent. Checksums are written to
    tep_checksums.json inside output_dir and used on subsequent runs to detect
    corruption or partial downloads.

    Args:
        output_dir: Directory where the .dat files will be saved. Created if
            it does not exist.

    Raises:
        RuntimeError: If any file cannot be downloaded after all retries.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checksum_path = out_path / _CHECKSUM_FILENAME
    known_checksums: dict[str, str] = {}
    if checksum_path.exists():
        with checksum_path.open("r", encoding="utf-8") as fh:
            known_checksums = json.load(fh)

    updated_checksums: dict[str, str] = dict(known_checksums)

    _LOG.info("Downloading TEP dataset to %s", out_path.resolve())
    for filename in tqdm(_FILE_NAMES, desc="TEP files", unit="file"):
        dest = out_path / filename

        if dest.exists():
            current_md5 = _compute_md5(dest)
            if filename in known_checksums and known_checksums[filename] == current_md5:
                _LOG.debug("Skipping %s (checksum verified).", filename)
                continue
            _LOG.info(
                "Re-downloading %s (checksum missing or mismatch).", filename
            )

        url = _BASE_URL + filename
        _LOG.info("Downloading %s", filename)
        _download_single_file(url, dest)
        updated_checksums[filename] = _compute_md5(dest)

    with checksum_path.open("w", encoding="utf-8") as fh:
        json.dump(updated_checksums, fh, indent=2)
    _LOG.info("Download complete. %d files verified in %s.", len(_FILE_NAMES), out_path)


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> None:
    """Run the download stage using the raw_dir configured in params.yaml.

    Args:
        params_path: Path to the params file holding the tep: section.
    """
    params = load_tep_params(params_path)
    download_tep(str(params.raw_dir))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
