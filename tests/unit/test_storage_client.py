"""Tests for data/storage/storage_client.py."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from data.schemas.sensor_reading import SensorReading
from data.storage.blob_store import BlobStore, LocalBlobStore
from data.storage.layout import readings_prefix
from data.storage.parquet_codec import table_metadata
from data.storage.storage_client import (
    GCSStorageClient,
    LocalCache,
    StorageBackend,
    StorageClient,
    WriteResult,
    get_storage_client,
)
from data.storage.storage_params import StorageParams
from tests.support.fake_gcs import FakeGcsClient
from tests.support.readings import make_batch, make_reading

_PLANT = "TEP-PLANT-01"
_HOUR = timedelta(hours=1)
_T0 = datetime(2024, 3, 9, 10, 0, tzinfo=UTC)
_FIXED_NOW = datetime(2024, 5, 5, 8, 0, tzinfo=UTC)
_READING_COUNT = 100


def _now() -> datetime:
    """Return the fixed instant the clients are stamped with."""
    return _FIXED_NOW


@pytest.fixture()
def params(tmp_path: Path) -> StorageParams:
    """Return storage parameters rooted in a temporary directory."""
    return StorageParams("reactorguard", "dev", tmp_path / "local", 4, 30.0)


class SpyStore:
    """BlobStore wrapper that records every call, to count partitions touched."""

    def __init__(self, inner: BlobStore) -> None:
        """Wrap a store.

        Args:
            inner: The store to delegate to.
        """
        self._inner = inner
        self.listed: list[str] = []
        self.fetched: list[str] = []
        self.put_calls = 0

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Delegate and count."""
        self.put_calls += 1
        return self._inner.put_if_absent(key, data)

    def get(self, key: str) -> bytes:
        """Delegate and record the key."""
        self.fetched.append(key)
        return self._inner.get(key)

    def list_keys(self, prefix: str) -> list[str]:
        """Delegate and record the prefix."""
        self.listed.append(prefix)
        return self._inner.list_keys(prefix)

    def exists(self, key: str) -> bool:
        """Delegate."""
        return self._inner.exists(key)


class FlakyStore(SpyStore):
    """Store whose put_if_absent fails on a chosen call, then works."""

    def __init__(self, inner: BlobStore, fail_on_call: int) -> None:
        """Wrap a store.

        Args:
            inner: The store to delegate to.
            fail_on_call: 1-based put call that raises once.
        """
        super().__init__(inner)
        self._fail_on_call = fail_on_call

    def put_if_absent(self, key: str, data: bytes) -> bool:
        """Raise on the configured call, delegate otherwise."""
        if self.put_calls + 1 == self._fail_on_call:
            self.put_calls += 1
            raise OSError("simulated network failure")
        return super().put_if_absent(key, data)


@pytest.fixture(params=["local", "gcs"])
def client(request: pytest.FixtureRequest, params: StorageParams) -> StorageClient:
    """Return each concrete client in turn, so behaviour tests cover both."""
    if request.param == "local":
        return LocalCache(params, clock=_now)
    return GCSStorageClient(params, client_factory=FakeGcsClient, clock=_now)


def _spied(params: StorageParams, tmp_path: Path) -> tuple[StorageClient, SpyStore]:
    """Build a client over a spied local store.

    Args:
        params: Storage parameters.
        tmp_path: Directory for the store.

    Returns:
        The client and its spy.
    """
    spy = SpyStore(LocalBlobStore(tmp_path / "spied"))
    return StorageClient(spy, params, clock=_now), spy


# ---------------------------------------------------------------- roundtrip and parity


def test_roundtrip_of_100_readings(client: StorageClient) -> None:
    readings = make_batch(_READING_COUNT, _T0)
    client.write_sensor_readings(readings)
    read = client.read_sensor_readings(_PLANT, _T0, _T0 + 24 * _HOUR)
    assert read == readings


def test_roundtrip_of_100_readings_by_local_cache(params: StorageParams) -> None:
    cache = LocalCache(params, clock=_now)
    readings = make_batch(_READING_COUNT)
    result = cache.write_sensor_readings(readings)
    assert result.readings == _READING_COUNT
    start = datetime(2000, 1, 1, tzinfo=UTC)
    read = cache.read_sensor_readings(_PLANT, start, start + 24 * _HOUR)
    assert read == readings
    assert any(r.measurement.value is None for r in read)


def test_gcs_and_local_produce_identical_keys_and_results(params: StorageParams) -> None:
    local = LocalCache(params, clock=_now)
    fake = FakeGcsClient()
    gcs = GCSStorageClient(params, client_factory=lambda: fake, clock=_now)
    readings = make_batch(_READING_COUNT, _T0 - timedelta(minutes=30))
    local_result = local.write_sensor_readings(readings)
    gcs_result = gcs.write_sensor_readings(readings)
    assert local_result == gcs_result
    assert sorted(fake.bucket(params.raw_bucket).objects) == sorted(local_result.created)
    window = (_PLANT, _T0 - _HOUR, _T0 + 6 * _HOUR)
    assert local.read_sensor_readings(*window) == gcs.read_sensor_readings(*window)


def test_gcs_client_targets_the_raw_bucket_by_default(params: StorageParams) -> None:
    fake = FakeGcsClient()
    GCSStorageClient(params, client_factory=lambda: fake).write_sensor_readings(
        make_batch(3, _T0)
    )
    assert set(fake.buckets) == {"reactorguard-data-raw-dev"}


def test_gcs_client_accepts_another_bucket(params: StorageParams) -> None:
    fake = FakeGcsClient()
    gcs = GCSStorageClient(params, bucket_name="other-bucket", client_factory=lambda: fake)
    gcs.write_sensor_readings(make_batch(3, _T0))
    assert set(fake.buckets) == {"other-bucket"}


def test_local_cache_root_override(params: StorageParams, tmp_path: Path) -> None:
    root = tmp_path / "elsewhere"
    LocalCache(params, root=root).write_sensor_readings(make_batch(3, _T0))
    assert any(root.rglob("*.parquet"))
    assert not params.local_root.exists()


# -------------------------------------------------------------------------- layout


def test_keys_follow_the_hive_layout(client: StorageClient) -> None:
    result = client.write_sensor_readings(make_batch(5, _T0))
    (key,) = result.created
    assert key.startswith("plant=TEP-PLANT-01/year=2024/month=03/day=09/hour=10/part-")
    assert key.endswith(".parquet")


def test_batch_spanning_hours_writes_one_part_per_hour(client: StorageClient) -> None:
    readings = make_batch(60, _T0 - timedelta(minutes=30))  # 3 min * 60 = 3 h
    result = client.write_sensor_readings(readings)
    hours = [key.split("/")[4] for key in result.created]
    assert hours == ["hour=09", "hour=10", "hour=11", "hour=12"]


def test_batch_crossing_the_year_boundary(client: StorageClient) -> None:
    start = datetime(2024, 12, 31, 23, 30, tzinfo=UTC)
    readings = make_batch(20, start)
    result = client.write_sensor_readings(readings)
    assert [key.split("/")[1:4] for key in result.created] == [
        ["year=2024", "month=12", "day=31"],
        ["year=2025", "month=01", "day=01"],
    ]
    assert client.read_sensor_readings(_PLANT, start, start + 2 * _HOUR) == readings


def test_readings_in_another_zone_land_in_their_utc_partition(client: StorageClient) -> None:
    zone = timezone(timedelta(hours=-5))
    reading = make_reading(1, datetime(2024, 12, 31, 23, 30, tzinfo=zone))
    (key,) = client.write_sensor_readings([reading]).created
    assert "year=2025/month=01/day=01/hour=04" in key
    start = datetime(2025, 1, 1, 4, 0, tzinfo=UTC)
    assert client.read_sensor_readings(_PLANT, start, start + _HOUR) == [reading]


def test_plants_are_isolated(client: StorageClient) -> None:
    mine = make_batch(5, _T0, plant_id="P-A")
    theirs = make_batch(5, _T0, plant_id="P-B", first_index=100)
    client.write_sensor_readings(mine + theirs)
    assert client.read_sensor_readings("P-A", _T0, _T0 + _HOUR) == mine
    assert client.read_sensor_readings("P-B", _T0, _T0 + _HOUR) == theirs


# ------------------------------------------------------------------- idempotency


def test_second_write_is_a_no_op(client: StorageClient) -> None:
    readings = make_batch(_READING_COUNT, _T0)
    first = client.write_sensor_readings(readings)
    second = client.write_sensor_readings(readings)
    assert first.created and not first.existing
    assert second.created == ()
    assert second.existing == first.created
    assert client.read_sensor_readings(_PLANT, _T0, _T0 + 24 * _HOUR) == readings


def test_second_write_leaves_a_single_object_per_hour(params: StorageParams) -> None:
    local = LocalCache(params, clock=_now)
    readings = make_batch(20, _T0)
    local.write_sensor_readings(readings)
    local.write_sensor_readings(readings)
    objects = [p for p in params.local_root.rglob("*.parquet")]
    assert len(objects) == 1


def test_part_name_is_deterministic_across_clients(params: StorageParams, tmp_path: Path) -> None:
    readings = make_batch(20, _T0)
    one = LocalCache(params, root=tmp_path / "one", clock=_now).write_sensor_readings(readings)
    two = LocalCache(params, root=tmp_path / "two", clock=_now).write_sensor_readings(readings)
    assert one.created == two.created


def test_retry_after_a_partial_failure_completes_without_duplicates(
    params: StorageParams, tmp_path: Path
) -> None:
    readings = make_batch(60, _T0)  # 3 horas: tres partes
    flaky = FlakyStore(LocalBlobStore(tmp_path / "flaky"), fail_on_call=2)
    client = StorageClient(flaky, params, clock=_now)
    with pytest.raises(OSError, match="simulated"):
        client.write_sensor_readings(readings)
    retry = client.write_sensor_readings(readings)
    assert len(retry.existing) == 1  # la parte escrita antes del fallo
    assert len(retry.created) == 2
    assert client.read_sensor_readings(_PLANT, _T0, _T0 + 6 * _HOUR) == readings


def test_overlapping_batches_with_other_boundaries_are_deduplicated_on_read(
    client: StorageClient,
) -> None:
    readings = make_batch(20, _T0)
    client.write_sensor_readings(readings[:12])
    client.write_sensor_readings(readings[5:])  # reintento tras un crash: otra frontera
    assert client.read_sensor_readings(_PLANT, _T0, _T0 + _HOUR) == readings


def test_empty_write_stores_nothing(params: StorageParams, tmp_path: Path) -> None:
    client, spy = _spied(params, tmp_path)
    result = client.write_sensor_readings([])
    assert result == WriteResult(readings=0, created=(), existing=())
    assert spy.put_calls == 0


def test_created_at_metadata_comes_from_the_injected_clock(
    params: StorageParams, tmp_path: Path
) -> None:
    store = LocalBlobStore(tmp_path / "meta")
    client = StorageClient(store, params, clock=_now)
    (key,) = client.write_sensor_readings(make_batch(4, _T0)).created
    from data.storage.parquet_codec import bytes_to_table

    metadata = table_metadata(bytes_to_table(store.get(key)))
    assert metadata["created_at"] == _FIXED_NOW.isoformat()
    assert metadata["record_count"] == "4"
    assert metadata["schema_version"] == "1"


def test_default_clock_stamps_a_real_utc_instant(params: StorageParams, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "clock")
    client = StorageClient(store, params)
    before = datetime.now(UTC)
    (key,) = client.write_sensor_readings(make_batch(2, _T0)).created
    from data.storage.parquet_codec import bytes_to_table

    stamp = datetime.fromisoformat(table_metadata(bytes_to_table(store.get(key)))["created_at"])
    assert before <= stamp <= datetime.now(UTC)


# ----------------------------------------------------------------------- range reads


def test_two_hour_range_touches_exactly_two_partitions(
    params: StorageParams, tmp_path: Path
) -> None:
    client, spy = _spied(params, tmp_path)
    client.write_sensor_readings(make_batch(100, _T0 - 2 * _HOUR))  # 5 horas de datos
    spy.listed.clear()
    client.read_sensor_readings(_PLANT, _T0, _T0 + 2 * _HOUR)
    assert spy.listed == [
        readings_prefix(_PLANT, _T0),
        readings_prefix(_PLANT, _T0 + _HOUR),
    ]
    assert len(spy.fetched) == 2  # una parte por particion, ninguna fuera del rango


def test_range_is_half_open(client: StorageClient) -> None:
    readings = make_batch(40, _T0)  # cada 3 min
    client.write_sensor_readings(readings)
    start = readings[5].timestamp
    end = readings[15].timestamp
    selected = client.read_sensor_readings(_PLANT, start, end)
    assert selected == readings[5:15]


def test_range_filters_inside_a_partition(client: StorageClient) -> None:
    readings = make_batch(20, _T0)
    client.write_sensor_readings(readings)
    window = client.read_sensor_readings(
        _PLANT, _T0 + timedelta(minutes=30), _T0 + timedelta(minutes=45)
    )
    assert window == [r for r in readings if timedelta(minutes=30) <= r.timestamp - _T0
                      < timedelta(minutes=45)]
    assert len(window) == 5


def test_range_with_no_data_is_empty(client: StorageClient) -> None:
    assert client.read_sensor_readings(_PLANT, _T0, _T0 + _HOUR) == []


def test_empty_range_reads_nothing_and_lists_nothing(
    params: StorageParams, tmp_path: Path
) -> None:
    client, spy = _spied(params, tmp_path)
    assert client.read_sensor_readings(_PLANT, _T0, _T0) == []
    assert spy.listed == []


def test_result_is_ordered_by_timestamp_even_if_parts_arrive_out_of_order(
    client: StorageClient,
) -> None:
    readings = make_batch(20, _T0)
    # Seis partes de la misma hora, escritas en orden inverso. Se listan por nombre
    # (un hash), de modo que solo la ordenacion por timestamp las devuelve bien.
    for chunk in reversed([readings[i : i + 4] for i in range(0, 20, 4)]):
        client.write_sensor_readings(chunk)
    assert client.read_sensor_readings(_PLANT, _T0, _T0 + _HOUR) == readings


def test_range_read_with_a_naive_bound_is_rejected(client: StorageClient) -> None:
    with pytest.raises(ValueError, match="no timezone"):
        client.read_sensor_readings(_PLANT, datetime(2024, 3, 9, 10), _T0 + _HOUR)


def test_reversed_range_is_rejected(client: StorageClient) -> None:
    with pytest.raises(ValueError, match="later than end"):
        client.read_sensor_readings(_PLANT, _T0 + _HOUR, _T0)


def test_non_part_objects_in_a_partition_are_ignored(
    params: StorageParams, tmp_path: Path
) -> None:
    store = LocalBlobStore(tmp_path / "noise")
    client = StorageClient(store, params, clock=_now)
    readings = make_batch(5, _T0)
    client.write_sensor_readings(readings)
    store.put_if_absent(readings_prefix(_PLANT, _T0) + "_SUCCESS", b"")
    assert client.read_sensor_readings(_PLANT, _T0, _T0 + _HOUR) == readings


def test_corrupt_part_names_its_key(params: StorageParams, tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "corrupt")
    client = StorageClient(store, params, clock=_now)
    key = readings_prefix(_PLANT, _T0) + "part-deadbeef0000.parquet"
    store.put_if_absent(key, b"definitely not parquet")
    with pytest.raises(ValueError, match="part-deadbeef0000.parquet"):
        client.read_sensor_readings(_PLANT, _T0, _T0 + _HOUR)


def test_part_with_a_foreign_schema_is_rejected(params: StorageParams, tmp_path: Path) -> None:
    import pyarrow as pa

    from data.storage.parquet_codec import table_to_bytes

    store = LocalBlobStore(tmp_path / "foreign")
    client = StorageClient(store, params, clock=_now)
    key = readings_prefix(_PLANT, _T0) + "part-aaaaaaaaaaaa.parquet"
    store.put_if_absent(key, table_to_bytes(pa.table({"x": [1]})))
    with pytest.raises(ValueError, match="incompatible"):
        client.read_sensor_readings(_PLANT, _T0, _T0 + _HOUR)


def test_reads_use_the_configured_worker_count(params: StorageParams, tmp_path: Path) -> None:
    single = StorageParams(
        params.bucket_prefix, params.env, params.local_root, 1, params.request_timeout_s
    )
    client = StorageClient(LocalBlobStore(tmp_path / "w1"), single, clock=_now)
    readings = make_batch(60, _T0)
    client.write_sensor_readings(readings)
    assert client.read_sensor_readings(_PLANT, _T0, _T0 + 6 * _HOUR) == readings


# ------------------------------------------------------------------------ write guards


def test_naive_reading_timestamp_is_rejected_before_anything_is_written(
    params: StorageParams, tmp_path: Path
) -> None:
    client, spy = _spied(params, tmp_path)
    batch: list[SensorReading] = [make_reading(1, _T0), make_reading(2, datetime(2024, 1, 1))]
    with pytest.raises(ValueError, match="no timezone"):
        client.write_sensor_readings(batch)
    assert spy.put_calls == 0


def test_unsafe_plant_id_is_rejected(client: StorageClient) -> None:
    with pytest.raises(ValueError, match="Invalid plant_id"):
        client.write_sensor_readings([make_reading(1, _T0, plant_id="../escape")])


# ----------------------------------------------------------------------------- factory


def test_factory_builds_a_local_cache(params: StorageParams) -> None:
    built = get_storage_client("local", params, clock=_now)
    assert isinstance(built, LocalCache)
    assert params.local_root.is_dir()


def test_factory_accepts_the_enum(params: StorageParams) -> None:
    assert isinstance(get_storage_client(StorageBackend.LOCAL, params), LocalCache)


def test_factory_builds_a_gcs_client_without_credentials_using_an_emulator_host(
    params: StorageParams, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://localhost:9")
    assert isinstance(get_storage_client("gcs", params), GCSStorageClient)


@pytest.mark.parametrize("backend", ["s3", "", "GCS "])
def test_factory_rejects_unknown_backends(params: StorageParams, backend: str) -> None:
    with pytest.raises(ValueError, match="Unknown storage backend"):
        get_storage_client(backend, params)


def test_subclasses_share_the_storage_client_interface(params: StorageParams) -> None:
    factories: list[Callable[[], StorageClient]] = [
        lambda: LocalCache(params),
        lambda: GCSStorageClient(params, client_factory=FakeGcsClient),
    ]
    for build in factories:
        instance = build()
        assert isinstance(instance, StorageClient)
        assert callable(instance.write_sensor_readings)
        assert callable(instance.read_sensor_readings)
