"""Tests for data/generators/reading_source.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.generators.reading_source import (
    ParquetDirSource,
    StorageSource,
    run_name,
)
from data.storage.blob_store import GCSBlobStore, LocalBlobStore
from tests.support.fake_gcs import FakeGcsClient
from tests.support.tep_frames import make_run_frame, write_partition


def _frame(fault_type: int, n_timesteps: int = 2) -> pd.DataFrame:
    return make_run_frame(fault_type, n_timesteps)


def _upload(store: LocalBlobStore, key: str, frame: pd.DataFrame, tmp_path: Path) -> None:
    scratch = tmp_path / "scratch.parquet"
    frame.to_parquet(scratch, index=False)
    store.put_if_absent(key, scratch.read_bytes())


class TestRunName:
    def test_zero_pads_to_two_digits(self) -> None:
        assert run_name(0) == "fault_type=00"
        assert run_name(21) == "fault_type=21"


class TestParquetDirSource:
    def test_yields_runs_in_numeric_order_with_their_frames(self, tmp_path: Path) -> None:
        for fault_type in (10, 2, 0):
            write_partition(tmp_path, fault_type, _frame(fault_type))
        runs = list(ParquetDirSource(tmp_path).runs())
        assert [name for name, _ in runs] == ["fault_type=00", "fault_type=02", "fault_type=10"]
        assert [int(frame["fault_type"].iloc[0]) for _, frame in runs] == [0, 2, 10]
        assert all(len(frame) == 2 * 52 for _, frame in runs)

    def test_unpadded_directory_names_still_sort_numerically(self, tmp_path: Path) -> None:
        for number in (10, 2):
            directory = tmp_path / f"fault_type={number}"
            directory.mkdir()
            _frame(number).to_parquet(directory / "readings.parquet", index=False)
        names = [name for name, _ in ParquetDirSource(tmp_path).runs()]
        assert names == ["fault_type=02", "fault_type=10"]

    def test_ignores_directories_that_are_not_runs(self, tmp_path: Path) -> None:
        write_partition(tmp_path, 1, _frame(1))
        (tmp_path / "fault_type=abc").mkdir()
        (tmp_path / "fault_type=abc" / "readings.parquet").write_bytes(b"junk")
        (tmp_path / "notes").mkdir()
        assert [name for name, _ in ParquetDirSource(tmp_path).runs()] == ["fault_type=01"]

    def test_frames_are_loaded_one_at_a_time(self, tmp_path: Path) -> None:
        write_partition(tmp_path, 0, _frame(0))
        write_partition(tmp_path, 1, _frame(1))
        write_partition(tmp_path, 2, _frame(2))
        (tmp_path / "fault_type=02" / "readings.parquet").write_bytes(b"junk")
        iterator = ParquetDirSource(tmp_path).runs()
        assert next(iterator)[0] == "fault_type=00"
        assert next(iterator)[0] == "fault_type=01"
        with pytest.raises(ValueError, match="fault_type=02"):
            next(iterator)

    def test_can_be_iterated_again(self, tmp_path: Path) -> None:
        write_partition(tmp_path, 0, _frame(0))
        source = ParquetDirSource(tmp_path)
        assert len(list(source.runs())) == len(list(source.runs())) == 1

    def test_missing_directory_fails_when_runs_is_called(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            ParquetDirSource(tmp_path / "absent").runs()

    def test_directory_without_runs_fails_with_a_hint(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Invoke-Pipeline"):
            ParquetDirSource(tmp_path).runs()

    def test_corrupt_parquet_names_the_file(self, tmp_path: Path) -> None:
        path = write_partition(tmp_path, 3, _frame(3))
        path.write_bytes(b"not parquet")
        with pytest.raises(ValueError, match=r"fault_type=03"):
            list(ParquetDirSource(tmp_path).runs())


class TestStorageSource:
    def test_reads_the_dvc_layout_under_the_prefix(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        for fault_type in (1, 0):
            _upload(store, f"tep/fault_type={fault_type:02d}/readings.parquet",
                    _frame(fault_type), tmp_path)
        runs = list(StorageSource(store, "tep").runs())
        assert [name for name, _ in runs] == ["fault_type=00", "fault_type=01"]
        assert [int(frame["fault_type"].iloc[0]) for _, frame in runs] == [0, 1]

    def test_slashes_around_the_prefix_are_ignored(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        _upload(store, "tep/fault_type=00/readings.parquet", _frame(0), tmp_path)
        assert len(list(StorageSource(store, "/tep/").runs())) == 1

    def test_empty_prefix_reads_from_the_bucket_root(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        _upload(store, "fault_type=05/readings.parquet", _frame(5), tmp_path)
        assert [name for name, _ in StorageSource(store, "").runs()] == ["fault_type=05"]

    def test_ignores_other_objects_and_other_prefixes(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        _upload(store, "tep/fault_type=00/readings.parquet", _frame(0), tmp_path)
        _upload(store, "other/fault_type=01/readings.parquet", _frame(1), tmp_path)
        store.put_if_absent("tep/fault_type=02/notes.txt", b"x")
        store.put_if_absent("tep/fault_type=xx/readings.parquet", b"x")
        store.put_if_absent("tep/readings.parquet", b"x")
        assert [name for name, _ in StorageSource(store, "tep").runs()] == ["fault_type=00"]

    def test_nothing_stored_fails_with_the_upload_command(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        with pytest.raises(FileNotFoundError, match="gsutil"):
            StorageSource(store, "tep").runs()

    def test_corrupt_object_names_its_key(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        store.put_if_absent("tep/fault_type=00/readings.parquet", b"not parquet")
        with pytest.raises(ValueError, match="tep/fault_type=00/readings.parquet"):
            list(StorageSource(store, "tep").runs())

    def test_object_removed_between_listing_and_reading_raises(self, tmp_path: Path) -> None:
        store = LocalBlobStore(tmp_path / "bucket")
        _upload(store, "tep/fault_type=00/readings.parquet", _frame(0), tmp_path)
        iterator = StorageSource(store, "tep").runs()
        (tmp_path / "bucket" / "tep" / "fault_type=00" / "readings.parquet").unlink()
        with pytest.raises(FileNotFoundError):
            next(iterator)

    def test_works_over_the_gcs_store(self, tmp_path: Path) -> None:
        local = LocalBlobStore(tmp_path / "staging")
        _upload(local, "tep/fault_type=00/readings.parquet", _frame(0), tmp_path)
        payload = local.get("tep/fault_type=00/readings.parquet")

        client = FakeGcsClient()
        gcs = GCSBlobStore(
            "reactorguard-data-raw-dev", timeout_s=5.0, client_factory=lambda: client
        )
        gcs.put_if_absent("tep/fault_type=00/readings.parquet", payload)

        runs = list(StorageSource(gcs, "tep").runs())
        assert [name for name, _ in runs] == ["fault_type=00"]
        pd.testing.assert_frame_equal(runs[0][1], _frame(0))
