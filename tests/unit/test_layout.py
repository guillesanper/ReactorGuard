"""Tests for data/storage/layout.py."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from data.storage import layout

_MADRID_WINTER = timezone(timedelta(hours=1))
_NEW_YORK_WINTER = timezone(timedelta(hours=-5))
_MAX_MICROSECOND = 999_999


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (
            datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC),
            "plant=P1/year=2000/month=01/day=01/hour=00/",
        ),
        (
            datetime(2024, 3, 9, 7, 59, 59, _MAX_MICROSECOND, tzinfo=UTC),
            "plant=P1/year=2024/month=03/day=09/hour=07/",
        ),
        # Cambio de hora: el ultimo microsegundo de la 07 (caso anterior) y el primero de la 08.
        (
            datetime(2024, 3, 9, 8, 0, 0, tzinfo=UTC),
            "plant=P1/year=2024/month=03/day=09/hour=08/",
        ),
        # Cambio de dia.
        (
            datetime(2024, 3, 9, 23, 59, 59, tzinfo=UTC),
            "plant=P1/year=2024/month=03/day=09/hour=23/",
        ),
        (
            datetime(2024, 3, 10, 0, 0, 0, tzinfo=UTC),
            "plant=P1/year=2024/month=03/day=10/hour=00/",
        ),
        # Cambio de mes y de ano.
        (
            datetime(2024, 1, 31, 23, 30, tzinfo=UTC),
            "plant=P1/year=2024/month=01/day=31/hour=23/",
        ),
        (
            datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC),
            "plant=P1/year=2024/month=12/day=31/hour=23/",
        ),
        (
            datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
            "plant=P1/year=2025/month=01/day=01/hour=00/",
        ),
        # Bisiesto.
        (
            datetime(2024, 2, 29, 12, 0, tzinfo=UTC),
            "plant=P1/year=2024/month=02/day=29/hour=12/",
        ),
    ],
)
def test_readings_prefix_utc(timestamp: datetime, expected: str) -> None:
    assert layout.readings_prefix("P1", timestamp) == expected


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        # 23:30 en Nueva York del 31-dic es 04:30 UTC del 1-ene: cruza el ano.
        (
            datetime(2024, 12, 31, 23, 30, tzinfo=_NEW_YORK_WINTER),
            "plant=P1/year=2025/month=01/day=01/hour=04/",
        ),
        # 00:30 en Madrid del 1-ene es 23:30 UTC del 31-dic: cruza el ano hacia atras.
        (
            datetime(2025, 1, 1, 0, 30, tzinfo=_MADRID_WINTER),
            "plant=P1/year=2024/month=12/day=31/hour=23/",
        ),
        # Offset no entero de horas: 05:45 +05:45 son las 00:00 UTC exactas.
        (
            datetime(2024, 6, 1, 5, 45, tzinfo=timezone(timedelta(hours=5, minutes=45))),
            "plant=P1/year=2024/month=06/day=01/hour=00/",
        ),
    ],
)
def test_readings_prefix_converts_other_zones_to_utc(timestamp: datetime, expected: str) -> None:
    assert layout.readings_prefix("P1", timestamp) == expected


def test_same_instant_in_two_zones_shares_a_partition() -> None:
    utc = datetime(2024, 5, 5, 10, 15, tzinfo=UTC)
    madrid = utc.astimezone(timezone(timedelta(hours=2)))
    assert layout.readings_prefix("P1", utc) == layout.readings_prefix("P1", madrid)


def test_naive_timestamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="no timezone"):
        layout.readings_prefix("P1", datetime(2024, 1, 1, 0, 0))


@pytest.mark.parametrize("plant_id", ["", ".", "..", "a/b", "a=b", "-x", ".hidden", "a b", "a\\b"])
def test_unsafe_plant_ids_are_rejected(plant_id: str) -> None:
    with pytest.raises(ValueError, match="Invalid plant_id"):
        layout.readings_prefix(plant_id, datetime(2024, 1, 1, tzinfo=UTC))


@pytest.mark.parametrize("plant_id", ["TEP-PLANT-01", "reactor.1", "R_2", "9"])
def test_valid_plant_ids_are_kept(plant_id: str) -> None:
    assert layout.validate_plant_id(plant_id) == plant_id


def test_readings_object_joins_prefix_and_part() -> None:
    prefix = "plant=P1/year=2024/month=03/day=09/hour=07/"
    assert layout.readings_object(prefix, "abc123") == f"{prefix}part-abc123.parquet"


def test_part_id_is_deterministic_and_twelve_hex_characters() -> None:
    first = layout.part_id("id-a", "id-z", 100)
    assert first == layout.part_id("id-a", "id-z", 100)
    assert len(first) == 12
    assert all(char in "0123456789abcdef" for char in first)


@pytest.mark.parametrize(
    "other",
    [("id-b", "id-z", 100), ("id-a", "id-y", 100), ("id-a", "id-z", 101), ("id-z", "id-a", 100)],
)
def test_part_id_depends_on_each_ingredient(other: tuple[str, str, int]) -> None:
    assert layout.part_id("id-a", "id-z", 100) != layout.part_id(*other)


def test_part_id_matches_the_documented_recipe() -> None:
    import hashlib

    expected = hashlib.sha1(b"a|b|3", usedforsecurity=False).hexdigest()[:12]
    assert layout.part_id("a", "b", 3) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("plant=P1/year=2024/month=03/day=09/hour=07/part-abc.parquet", True),
        ("part-abc.parquet", True),
        ("plant=P1/year=2024/month=03/day=09/hour=07/_SUCCESS", False),
        ("plant=P1/year=2024/month=03/day=09/hour=07/part-abc.parquet.tmp", False),
        ("plant=P1/year=2024/month=03/day=09/hour=07/other.parquet", False),
        ("part-dir/notes.txt", False),
    ],
)
def test_is_part_key(key: str, expected: bool) -> None:
    assert layout.is_part_key(key) is expected


def test_two_hour_range_touches_exactly_two_prefixes() -> None:
    start = datetime(2024, 3, 9, 10, 0, tzinfo=UTC)
    prefixes = layout.hour_prefixes("P1", start, start + timedelta(hours=2))
    assert prefixes == [
        "plant=P1/year=2024/month=03/day=09/hour=10/",
        "plant=P1/year=2024/month=03/day=09/hour=11/",
    ]


@pytest.mark.parametrize(
    ("start", "end", "count"),
    [
        (datetime(2024, 3, 9, 10, 30, tzinfo=UTC), datetime(2024, 3, 9, 12, 0, tzinfo=UTC), 2),
        (datetime(2024, 3, 9, 10, 30, tzinfo=UTC), datetime(2024, 3, 9, 12, 30, tzinfo=UTC), 3),
        (datetime(2024, 3, 9, 10, 0, tzinfo=UTC), datetime(2024, 3, 9, 10, 1, tzinfo=UTC), 1),
        (datetime(2024, 3, 9, 10, 59, tzinfo=UTC), datetime(2024, 3, 9, 11, 1, tzinfo=UTC), 2),
        (datetime(2024, 3, 9, 0, 0, tzinfo=UTC), datetime(2024, 3, 10, 0, 0, tzinfo=UTC), 24),
    ],
)
def test_hour_prefix_count(start: datetime, end: datetime, count: int) -> None:
    assert len(layout.hour_prefixes("P1", start, end)) == count


def test_range_crossing_the_year_boundary() -> None:
    prefixes = layout.hour_prefixes(
        "P1",
        datetime(2024, 12, 31, 23, 0, tzinfo=UTC),
        datetime(2025, 1, 1, 1, 0, tzinfo=UTC),
    )
    assert prefixes == [
        "plant=P1/year=2024/month=12/day=31/hour=23/",
        "plant=P1/year=2025/month=01/day=01/hour=00/",
    ]


def test_range_given_in_another_zone_is_converted() -> None:
    prefixes = layout.hour_prefixes(
        "P1",
        datetime(2025, 1, 1, 1, 0, tzinfo=_MADRID_WINTER),
        datetime(2025, 1, 1, 2, 0, tzinfo=_MADRID_WINTER),
    )
    assert prefixes == ["plant=P1/year=2025/month=01/day=01/hour=00/"]


def test_empty_range_has_no_prefixes() -> None:
    instant = datetime(2024, 3, 9, 10, 0, tzinfo=UTC)
    assert layout.hour_prefixes("P1", instant, instant) == []


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="later than end"):
        layout.hour_prefixes(
            "P1", datetime(2024, 3, 9, 12, tzinfo=UTC), datetime(2024, 3, 9, 10, tzinfo=UTC)
        )


def test_range_with_naive_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="no timezone"):
        layout.hour_prefixes("P1", datetime(2024, 3, 9, 10), datetime(2024, 3, 9, 12, tzinfo=UTC))


def test_range_with_unsafe_plant_is_rejected() -> None:
    instant = datetime(2024, 3, 9, 10, tzinfo=UTC)
    with pytest.raises(ValueError, match="Invalid plant_id"):
        layout.hour_prefixes("../x", instant, instant + timedelta(hours=1))


@pytest.mark.parametrize(
    ("fault_type", "expected"),
    [
        (0, "features/fault_type=00/features.parquet"),
        (7, "features/fault_type=07/features.parquet"),
        (21, "features/fault_type=21/features.parquet"),
        (99, "features/fault_type=99/features.parquet"),
    ],
)
def test_features_object(fault_type: int, expected: str) -> None:
    assert layout.features_object(fault_type) == expected


@pytest.mark.parametrize("fault_type", [-1, 100])
def test_features_object_rejects_out_of_range(fault_type: int) -> None:
    with pytest.raises(ValueError, match="fault_type"):
        layout.features_object(fault_type)
