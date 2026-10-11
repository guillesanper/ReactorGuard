"""Unit tests for ml/features/store/source_builder.py (feast_source stage).

Lo que se protege aqui es la unicidad de (sensor_id, event_timestamp): los 22 ficheros
del TEP reinician el reloj, y si dos runs compartieran eje temporal Feast serviria una
fila cualquiera de ellos. El resto de pruebas (determinismo, fallo ruidoso ante una
deriva de sensores) protege la reproducibilidad del artefacto de DVC.

Corren sobre arboles sinteticos en tmp_path; la unica prueba que lee los datos reales
es de solo lectura y se salta si data/processed/features no esta poblado.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.features.batch_featurizer import FEATURES_FILENAME, discover_partitions
from ml.features.feature_params import load_feature_params
from ml.features.pipeline import FeaturePipeline
from ml.features.store.feature_store_params import load_feature_store_params
from ml.features.store.source_builder import (
    CREATED_TIMESTAMP_COLUMN,
    EVENT_TIMESTAMP_COLUMN,
    build_source_frame,
    main,
    rebase_event_timestamps,
)

from .conftest import WriteParams

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_PARAMS = REPO_ROOT / "params.yaml"
REAL_FEATURES = REPO_ROOT / "data" / "processed" / "features"

_ORIGIN = datetime(2000, 1, 1, tzinfo=UTC)
_INTERVAL = 180
_GAP = 5
_SENSORS = ("S-01", "S-02", "S-03", "S-04")
_FEATURES = ("value", "rolling_mean_3")


def _partition(fault_type: int, steps: int, columns: tuple[str, ...] = _FEATURES) -> pd.DataFrame:
    """Build the long feature frame of one synthetic run.

    Args:
        fault_type: Fault type stamped on the rows.
        steps: Number of timesteps.
        columns: Feature columns to include.

    Returns:
        A frame with fault_type, timestep, sensor_id and the feature columns.
    """
    rows = [
        {"fault_type": fault_type, "timestep": step, "sensor_id": sensor}
        | {column: float(step + index) for index, column in enumerate(columns)}
        for step in range(steps)
        for sensor in _SENSORS
    ]
    return pd.DataFrame(rows)


def _build(
    partitions: dict[int, pd.DataFrame] | None = None, **overrides: Any
) -> pd.DataFrame:
    """Call build_source_frame with the synthetic defaults.

    Args:
        partitions: Frames to use; three runs of 6, 4 and 4 steps by default.
        **overrides: Replacements for the keyword arguments.

    Returns:
        The source frame.
    """
    if partitions is None:
        partitions = {0: _partition(0, 6), 1: _partition(1, 4), 2: _partition(2, 4)}
    arguments: dict[str, Any] = {
        "origin": _ORIGIN,
        "interval_seconds": _INTERVAL,
        "run_gap_intervals": _GAP,
        "expected_sensor_count": len(_SENSORS),
        "required_columns": _FEATURES,
    }
    arguments.update(overrides)
    return build_source_frame(partitions, **arguments)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_sensor_and_timestamp_pairs_are_unique_across_runs() -> None:
    frame = _build()
    assert not frame.duplicated(["sensor_id", EVENT_TIMESTAMP_COLUMN]).any()
    assert len(frame) == (6 + 4 + 4) * len(_SENSORS)


def test_the_same_timestep_of_two_runs_lands_at_different_instants() -> None:
    # Es exactamente el fallo que se evita: el timestep 0 de un sensor existe en
    # las tres clases y, sin desplazar, compartiria instante.
    frame = _build()
    firsts = frame[(frame["sensor_id"] == "S-01") & (frame["timestep"] == 0)]
    assert len(firsts) == 3
    assert firsts[EVENT_TIMESTAMP_COLUMN].is_unique


def test_first_run_starts_at_the_origin_and_follows_the_cadence() -> None:
    frame = _build()
    run0 = frame[frame["fault_type"] == 0]
    sensor = run0[run0["sensor_id"] == "S-01"].sort_values("timestep")
    expected = [_ORIGIN + timedelta(seconds=_INTERVAL * step) for step in range(6)]
    assert list(sensor[EVENT_TIMESTAMP_COLUMN]) == expected


def test_each_run_starts_after_the_previous_one_plus_the_gap() -> None:
    frame = _build()
    starts = frame.groupby("fault_type")[EVENT_TIMESTAMP_COLUMN].min()
    # run 0 ocupa 6 intervalos, run 1 ocupa 4: 6 + gap y 6 + gap + 4 + gap.
    assert starts[1] == _ORIGIN + timedelta(seconds=_INTERVAL * (6 + _GAP))
    assert starts[2] == _ORIGIN + timedelta(seconds=_INTERVAL * (6 + _GAP + 4 + _GAP))


def test_gap_between_runs_exceeds_the_ttl() -> None:
    # La cola de un run debe quedar fuera del TTL (3 intervalos) de la cabeza del
    # siguiente, o un join historico mezclaria dos clases.
    frame = _build()
    ends = frame.groupby("fault_type")[EVENT_TIMESTAMP_COLUMN].max()
    starts = frame.groupby("fault_type")[EVENT_TIMESTAMP_COLUMN].min()
    ttl = timedelta(seconds=3 * _INTERVAL)
    for previous, following in ((0, 1), (1, 2)):
        assert starts[following] - ends[previous] > ttl


def test_runs_are_ordered_by_fault_type_not_by_dict_order() -> None:
    frame = _build({2: _partition(2, 4), 0: _partition(0, 6), 1: _partition(1, 4)})
    starts = frame.groupby("fault_type")[EVENT_TIMESTAMP_COLUMN].min()
    assert starts.is_monotonic_increasing


def test_output_is_sorted_by_event_time_then_sensor() -> None:
    frame = _build()
    keys = list(zip(frame[EVENT_TIMESTAMP_COLUMN], frame["sensor_id"], strict=True))
    assert keys == sorted(keys)


def test_a_run_with_non_zero_first_timestep_still_cannot_overlap_the_next() -> None:
    late = _partition(0, 4)
    late["timestep"] += 10
    frame = _build({0: late, 1: _partition(1, 4)})
    assert not frame.duplicated(["sensor_id", EVENT_TIMESTAMP_COLUMN]).any()
    starts = frame.groupby("fault_type")[EVENT_TIMESTAMP_COLUMN].min()
    # La longitud del run es max(timestep) + 1 = 14: el siguiente arranca en 14 + gap.
    assert starts[1] == _ORIGIN + timedelta(seconds=_INTERVAL * (14 + _GAP))


# ---------------------------------------------------------------------------
# Columns, types, determinism
# ---------------------------------------------------------------------------


def test_created_timestamp_equals_event_timestamp_not_the_wall_clock() -> None:
    frame = _build()
    assert (frame[CREATED_TIMESTAMP_COLUMN] == frame[EVENT_TIMESTAMP_COLUMN]).all()
    assert frame[CREATED_TIMESTAMP_COLUMN].max() < pd.Timestamp("2001-01-01", tz="UTC")


def test_timestamps_are_timezone_aware_utc() -> None:
    frame = _build()
    assert str(frame[EVENT_TIMESTAMP_COLUMN].dt.tz) == "UTC"
    assert str(frame[CREATED_TIMESTAMP_COLUMN].dt.tz) == "UTC"


def test_a_naive_origin_is_taken_as_utc() -> None:
    frame = _build(origin=datetime(2000, 1, 1))  # noqa: DTZ001
    assert frame[EVENT_TIMESTAMP_COLUMN].min() == pd.Timestamp("2000-01-01", tz="UTC")


def test_a_non_utc_origin_is_converted() -> None:
    plus_two = datetime(2000, 1, 1, 2, tzinfo=timezone(timedelta(hours=2)))
    frame = _build(origin=plus_two)
    assert frame[EVENT_TIMESTAMP_COLUMN].min() == pd.Timestamp("2000-01-01", tz="UTC")


def test_provenance_and_feature_columns_are_kept() -> None:
    frame = _build()
    assert {"sensor_id", "fault_type", "timestep", *_FEATURES} <= set(frame.columns)
    assert list(frame.columns[:2]) == [EVENT_TIMESTAMP_COLUMN, CREATED_TIMESTAMP_COLUMN]


def test_fault_type_is_added_when_the_frame_lacks_it() -> None:
    bare = {
        fault_type: _partition(fault_type, 3).drop(columns="fault_type")
        for fault_type in (0, 1)
    }
    frame = _build(bare)
    assert sorted(frame["fault_type"].unique()) == [0, 1]


def test_build_is_deterministic(tmp_path: Path) -> None:
    first = _build()
    second = _build()
    pd.testing.assert_frame_equal(first, second)

    first.to_parquet(tmp_path / "a.parquet", index=False)
    second.to_parquet(tmp_path / "b.parquet", index=False)
    assert (tmp_path / "a.parquet").read_bytes() == (tmp_path / "b.parquet").read_bytes()


def test_inputs_are_not_mutated() -> None:
    partitions = {0: _partition(0, 4), 1: _partition(1, 4)}
    snapshot = {key: frame.copy() for key, frame in partitions.items()}
    _build(partitions)
    for key, frame in partitions.items():
        pd.testing.assert_frame_equal(frame, snapshot[key])


# ---------------------------------------------------------------------------
# Loud failures
# ---------------------------------------------------------------------------


def test_sensor_count_mismatch_fails_loudly() -> None:
    with pytest.raises(ValueError, match="expected_sensor_count"):
        _build(expected_sensor_count=len(_SENSORS) + 1)


def test_no_partitions_fail_loudly() -> None:
    with pytest.raises(ValueError, match="No feature partitions"):
        build_source_frame({}, _ORIGIN, _INTERVAL, _GAP, 1)


def test_a_missing_feature_column_names_the_column() -> None:
    thin = {0: _partition(0, 3, columns=("value",))}
    with pytest.raises(ValueError, match="rolling_mean_3"):
        _build(thin)


def test_a_missing_key_column_fails() -> None:
    broken = {0: _partition(0, 3).drop(columns="sensor_id")}
    with pytest.raises(ValueError, match="sensor_id"):
        _build(broken, expected_sensor_count=1)


def test_negative_timestep_fails() -> None:
    frame = _partition(0, 3)
    frame.loc[0, "timestep"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        _build({0: frame})


def test_empty_partition_fails() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _build({0: _partition(0, 3).iloc[0:0]})


def test_repeated_sensor_timestep_inside_a_run_fails() -> None:
    frame = _partition(0, 3)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="repeat"):
        _build({0: frame})


# ---------------------------------------------------------------------------
# Rebase
# ---------------------------------------------------------------------------


def test_rebase_moves_the_latest_event_onto_the_anchor() -> None:
    frame = _build()
    anchor = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
    rebased = rebase_event_timestamps(frame, anchor)
    assert rebased[EVENT_TIMESTAMP_COLUMN].max() == pd.Timestamp(anchor)
    assert (rebased[CREATED_TIMESTAMP_COLUMN] == rebased[EVENT_TIMESTAMP_COLUMN]).all()


def test_rebase_preserves_spacing_and_uniqueness() -> None:
    frame = _build()
    rebased = rebase_event_timestamps(frame, datetime(2026, 1, 1, tzinfo=UTC))
    original = frame[EVENT_TIMESTAMP_COLUMN].diff().dropna().reset_index(drop=True)
    shifted = rebased[EVENT_TIMESTAMP_COLUMN].diff().dropna().reset_index(drop=True)
    pd.testing.assert_series_equal(original, shifted)
    assert not rebased.duplicated(["sensor_id", EVENT_TIMESTAMP_COLUMN]).any()


def test_rebase_returns_a_copy() -> None:
    frame = _build()
    before = frame[EVENT_TIMESTAMP_COLUMN].copy()
    rebase_event_timestamps(frame, datetime(2026, 1, 1, tzinfo=UTC))
    pd.testing.assert_series_equal(frame[EVENT_TIMESTAMP_COLUMN], before)


def test_rebase_rejects_a_naive_anchor() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        rebase_event_timestamps(_build(), datetime(2026, 1, 1))  # noqa: DTZ001


def test_rebase_rejects_an_empty_frame() -> None:
    with pytest.raises(ValueError, match="empty"):
        rebase_event_timestamps(_build().iloc[0:0], datetime(2026, 1, 1, tzinfo=UTC))


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _write_stage_inputs(tmp_path: Path, write_params: WriteParams) -> tuple[Path, Path]:
    """Write a params.yaml and a synthetic feature tree for main().

    Args:
        tmp_path: Directory to write into.
        write_params: Factory of the shared fixture (writes the tep: section).

    Returns:
        The params path and the feature tree root.
    """
    params_path = write_params()
    document = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    features_dir = tmp_path / "features"
    document["features"] = {
        "features_dir": str(features_dir),
        "window_samples": [3, 4],
        "min_samples_per_window": 2,
        "n_lags": 1,
        "correlation_pairs": [],
        "sensor_selection": {
            "include_types": ["flow"],
            "exclude_sensor_ids": [],
            "expected_sensor_count": len(_SENSORS),
        },
    }
    document["feature_store"] = {
        "project": "reactorguard",
        "ttl_intervals": 3,
        "run_gap_intervals": 10,
        "online_backend": "sqlite",
        "source_path": str(tmp_path / "out" / "source.parquet"),
        "local_config_path": "unused/local.yaml",
        "cluster_config_path": "unused/cluster.yaml",
    }
    params_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    feature_params = load_feature_params(params_path)
    names = FeaturePipeline(feature_params, sample_interval_seconds=_INTERVAL).feature_names()
    rng = np.random.default_rng(0)
    for fault_type, steps in ((0, 8), (1, 5)):
        frame = _partition(fault_type, steps, columns=())
        for name in names:
            frame[name] = rng.normal(size=len(frame))
        target = features_dir / f"fault_type={fault_type:02d}" / FEATURES_FILENAME
        target.parent.mkdir(parents=True)
        frame.to_parquet(target, index=False)
    return params_path, features_dir


def test_main_writes_the_source_parquet(tmp_path: Path, write_params: WriteParams) -> None:
    params_path, _ = _write_stage_inputs(tmp_path, write_params)

    rows = main(params_path)

    target = load_feature_store_params(params_path).source_path
    written = pd.read_parquet(target)
    assert rows == len(written) == (8 + 5) * len(_SENSORS)
    assert not written.duplicated(["sensor_id", EVENT_TIMESTAMP_COLUMN]).any()


def test_main_accepts_an_explicit_features_dir(tmp_path: Path, write_params: WriteParams) -> None:
    params_path, features_dir = _write_stage_inputs(tmp_path, write_params)
    assert main(params_path, features_dir=features_dir) == (8 + 5) * len(_SENSORS)


def test_main_fails_when_the_features_stage_has_not_run(
    tmp_path: Path, write_params: WriteParams
) -> None:
    params_path, features_dir = _write_stage_inputs(tmp_path, write_params)
    for parquet in features_dir.rglob(FEATURES_FILENAME):
        parquet.unlink()
    with pytest.raises(FileNotFoundError, match="featurize"):
        main(params_path)


def test_main_fails_loudly_on_a_sensor_count_drift(
    tmp_path: Path, write_params: WriteParams
) -> None:
    params_path, _ = _write_stage_inputs(tmp_path, write_params)
    document = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    document["features"]["sensor_selection"]["expected_sensor_count"] = 52
    params_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_sensor_count"):
        main(params_path)


# ---------------------------------------------------------------------------
# dvc.yaml declares what the stage reads and writes
# ---------------------------------------------------------------------------


def _stage() -> dict[str, Any]:
    """Return the feast_source stage of the repository's dvc.yaml."""
    document = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))
    stage = document["stages"]["feast_source"]
    assert isinstance(stage, dict)
    return stage


def test_dvc_stage_writes_the_path_that_params_declares() -> None:
    store = load_feature_store_params(REAL_PARAMS)
    assert _stage()["outs"] == [store.source_path.as_posix()]


def test_dvc_stage_declares_every_param_the_builder_reads() -> None:
    declared = set(_stage()["params"])
    required = {
        "tep.start_time",
        "tep.sample_interval_minutes",
        "features.features_dir",
        "features.sensor_selection",
        "feature_store.source_path",
        "feature_store.run_gap_intervals",
    }
    assert required <= declared


def test_dvc_stage_depends_on_its_code_and_on_the_featurize_output() -> None:
    deps = _stage()["deps"]
    assert "data/processed/features" in deps
    assert "ml/features/store/source_builder.py" in deps
    for dep in deps:
        if dep.endswith(".py"):
            assert (REPO_ROOT / dep).is_file(), dep


# ---------------------------------------------------------------------------
# Real data (read only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not REAL_FEATURES.is_dir(), reason="data/processed/features is not populated"
)
def test_real_features_yield_a_unique_timeline_for_all_52_sensors() -> None:
    feature_params = load_feature_params(REAL_PARAMS)
    store = load_feature_store_params(REAL_PARAMS)
    partitions = {
        fault_type: pd.read_parquet(path)
        for fault_type, path in discover_partitions(
            REAL_FEATURES, filename=FEATURES_FILENAME
        ).items()
    }
    pipeline = FeaturePipeline(
        feature_params, sample_interval_seconds=store.sample_interval_seconds
    )

    frame = build_source_frame(
        partitions,
        origin=datetime(2000, 1, 1, tzinfo=UTC),
        interval_seconds=store.sample_interval_seconds,
        run_gap_intervals=store.run_gap_intervals,
        expected_sensor_count=feature_params.selection.expected_sensor_count,
        required_columns=pipeline.feature_names(),
    )

    assert frame["sensor_id"].nunique() == 52
    assert frame["fault_type"].nunique() == 22
    assert len(frame) == sum(len(part) for part in partitions.values())
    assert not frame.duplicated(["sensor_id", EVENT_TIMESTAMP_COLUMN]).any()
