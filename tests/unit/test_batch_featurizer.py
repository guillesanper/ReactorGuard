"""Unit tests for ml.features.batch_featurizer.

Corren sobre un arbol de parquet sintetico en tmp_path, de modo que la suite no
depende de que data/processed/tep este poblado ni escribe en el.

Lo que mas importa aqui es la frontera de particion: el featurizer debe tratar
cada fault_type por separado, porque cada fichero del TEP reinicia el reloj y
rodar una ventana a traves de la frontera contaminaria una clase con la historia
de otra.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from ml.features.batch_featurizer import (
    FEATURES_FILENAME,
    READINGS_FILENAME,
    discover_partitions,
    featurize_partition,
    main,
    measure_sample_interval,
    save_features,
)
from ml.features.feature_params import FeatureParams, SensorSelection, load_feature_params

_T0 = datetime(2000, 1, 1, tzinfo=UTC)
_INTERVAL = timedelta(minutes=3)
_TAGS = {"SENS-01": "flow", "SENS-02": "pressure", "POS-01": "position"}


def _readings(n_timesteps: int = 40, interval: timedelta = _INTERVAL) -> pd.DataFrame:
    """Build a long-format readings frame for the synthetic sensor set.

    Args:
        n_timesteps: Samples per sensor.
        interval: Spacing between samples.

    Returns:
        The long frame.
    """
    records = []
    for step in range(n_timesteps):
        for offset, (tag, sensor_type) in enumerate(_TAGS.items()):
            records.append(
                {
                    "timestamp": _T0 + interval * step,
                    "timestep": step,
                    "sensor_id": tag,
                    "sensor_type": sensor_type,
                    "value": 10.0 + offset + 0.5 * step,
                }
            )
    return pd.DataFrame(records)


def _params(features_dir: Path, expected: int = 3) -> FeatureParams:
    """Build feature parameters matching the synthetic sensor set.

    Args:
        features_dir: Output root.
        expected: Sensors the selection must resolve.

    Returns:
        The parameters.
    """
    return FeatureParams(
        features_dir=features_dir,
        window_samples=(10, 20),
        min_samples_per_window=3,
        n_lags=2,
        selection=SensorSelection(
            include_types=frozenset(_TAGS.values()),
            exclude_sensor_ids=frozenset(),
            expected_sensor_count=expected,
        ),
        correlation_pairs=(),
    )


@pytest.fixture()
def readings_tree(tmp_path: Path) -> Path:
    """Write a two-partition readings tree and return its root."""
    root = tmp_path / "processed" / "tep"
    for fault_type in (0, 7):
        partition = root / f"fault_type={fault_type:02d}"
        partition.mkdir(parents=True)
        _readings().to_parquet(partition / READINGS_FILENAME, index=False)
    return root


# ---------------------------------------------------------------------------
# Partition discovery
# ---------------------------------------------------------------------------


class TestDiscoverPartitions:
    """El descubrimiento del arbol de lecturas."""

    def test_finds_every_partition_sorted(self, readings_tree: Path) -> None:
        """Order must be deterministic so the log reads the same every run."""
        found = discover_partitions(readings_tree)
        assert list(found) == [0, 7]

    def test_missing_root_is_reported(self, tmp_path: Path) -> None:
        """The message must point at the stage that produces it."""
        with pytest.raises(FileNotFoundError, match="adapt_tep"):
            discover_partitions(tmp_path / "absent")

    def test_a_root_without_partitions_is_reported(self, tmp_path: Path) -> None:
        """An empty tree is not silently featurised into nothing."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No fault_type partitions"):
            discover_partitions(empty)

    def test_directories_that_are_not_partitions_are_ignored(
        self, readings_tree: Path
    ) -> None:
        """A stray directory must not become a fault type."""
        (readings_tree / "_tmp").mkdir()
        assert list(discover_partitions(readings_tree)) == [0, 7]

    def test_a_partition_without_its_parquet_is_skipped(
        self, readings_tree: Path
    ) -> None:
        """A half-written partition must not be picked up."""
        (readings_tree / "fault_type=13").mkdir()
        assert list(discover_partitions(readings_tree)) == [0, 7]


# ---------------------------------------------------------------------------
# Cadence
# ---------------------------------------------------------------------------


class TestMeasureSampleInterval:
    """La cadencia se mide de los datos, no se lee de params.yaml."""

    def test_measures_the_tep_cadence(self) -> None:
        """Three minutes between samples is 180 seconds."""
        assert measure_sample_interval(_readings()) == 180.0

    def test_measures_a_different_cadence(self) -> None:
        """The number must come from the data, not from a constant."""
        assert measure_sample_interval(
            _readings(interval=timedelta(seconds=30))
        ) == 30.0

    def test_a_single_timestep_is_rejected(self) -> None:
        """One sample defines no interval."""
        with pytest.raises(ValueError, match="Cannot measure"):
            measure_sample_interval(_readings(n_timesteps=1))

    def test_a_non_uniform_cadence_is_rejected(self) -> None:
        """Every rolling window would span a different amount of time."""
        frame = _readings()
        late = frame["timestep"] >= 20
        frame.loc[late, "timestamp"] = frame.loc[late, "timestamp"] + timedelta(hours=2)
        with pytest.raises(ValueError, match="not uniform"):
            measure_sample_interval(frame)


# ---------------------------------------------------------------------------
# Featurising one partition
# ---------------------------------------------------------------------------


class TestFeaturizePartition:
    """El pivot y la transformacion de una particion."""

    def test_output_has_one_row_per_pair(self, tmp_path: Path) -> None:
        """40 timesteps x 3 sensors."""
        features = featurize_partition(_readings(), _params(tmp_path), fault_type=7)
        assert len(features) == 40 * 3

    def test_fault_type_is_stamped_first(self, tmp_path: Path) -> None:
        """The split stage stratifies by it, so it must travel with the features."""
        features = featurize_partition(_readings(), _params(tmp_path), fault_type=7)
        assert features.columns[0] == "fault_type"
        assert set(features["fault_type"]) == {7}

    def test_the_keys_follow_the_fault_type(self, tmp_path: Path) -> None:
        """(fault_type, timestep, sensor_id) is the primary key of the long table."""
        features = featurize_partition(_readings(), _params(tmp_path), fault_type=0)
        assert list(features.columns[:3]) == ["fault_type", "timestep", "sensor_id"]
        assert not features.duplicated(subset=["timestep", "sensor_id"]).any()

    def test_features_are_actually_computed(self, tmp_path: Path) -> None:
        """A ramp of 0.5 units every 180 s must show up as its rate."""
        features = featurize_partition(_readings(), _params(tmp_path), fault_type=0)
        row = features[(features["sensor_id"] == "SENS-01") & (features["timestep"] == 20)]
        assert row["rate_of_change"].iloc[0] == pytest.approx(0.5 / 180.0)

    def test_schema_drift_is_rejected(self, tmp_path: Path) -> None:
        """A partition with an unexpected sensor count must fail loudly."""
        with pytest.raises(ValueError, match="expected_sensor_count"):
            featurize_partition(_readings(), _params(tmp_path, expected=52), fault_type=0)

    def test_a_gap_in_the_readings_warns_and_leaves_nulls(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A missing cell must not be imputed behind the operator's back."""
        frame = _readings()
        holed = frame[~((frame["sensor_id"] == "SENS-01") & (frame["timestep"] == 15))]

        with caplog.at_level("WARNING"):
            features = featurize_partition(holed, _params(tmp_path), fault_type=0)

        assert "no reading" in caplog.text
        row = features[(features["sensor_id"] == "SENS-01") & (features["timestep"] == 15)]
        assert pd.isna(row["value"].iloc[0])


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class TestSaveFeatures:
    """La escritura del arbol de features."""

    def test_layout_mirrors_the_readings(self, tmp_path: Path) -> None:
        """Same partition scheme, so the two trees can be joined by path."""
        features = featurize_partition(_readings(), _params(tmp_path), fault_type=3)
        written = save_features(features, tmp_path / "features", fault_type=3)

        assert written == tmp_path / "features" / "fault_type=03" / FEATURES_FILENAME
        assert written.exists()

    def test_the_written_file_round_trips(self, tmp_path: Path) -> None:
        """Parquet must preserve the frame the pipeline produced."""
        features = featurize_partition(_readings(), _params(tmp_path), fault_type=3)
        written = save_features(features, tmp_path / "features", fault_type=3)

        pd.testing.assert_frame_equal(pd.read_parquet(written), features)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _write_params(tmp_path: Path, features_dir: Path, **overrides: Any) -> Path:
    """Write a params.yaml holding both sections the stage reads.

    Args:
        tmp_path: Temporary directory.
        features_dir: Output root for the features.
        **overrides: Keys replaced in the features: section.

    Returns:
        Path of the params file.
    """
    section: dict[str, Any] = {
        "features_dir": str(features_dir),
        "window_samples": [10, 20],
        "min_samples_per_window": 3,
        "n_lags": 2,
        "sensor_selection": {
            "include_types": list(_TAGS.values()),
            "exclude_sensor_ids": [],
            "expected_sensor_count": 3,
        },
        "correlation_pairs": [],
    }
    section.update(overrides)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"features": section}), encoding="utf-8")
    return path


class TestMain:
    """El stage completo."""

    def test_every_partition_is_written(
        self, tmp_path: Path, readings_tree: Path
    ) -> None:
        """Two partitions in, two partitions out."""
        features_dir = tmp_path / "features"
        params_path = _write_params(tmp_path, features_dir)

        written = main(params_path, readings_dir=readings_tree)

        assert written == {0: 120, 7: 120}
        for fault_type in (0, 7):
            assert (
                features_dir / f"fault_type={fault_type:02d}" / FEATURES_FILENAME
            ).exists()

    def test_partitions_are_featurised_independently(
        self, tmp_path: Path, readings_tree: Path
    ) -> None:
        """The window must not carry history across the fault_type boundary.

        Si el featurizer concatenase las particiones, el timestep 0 de la segunda
        heredaria la ventana de la primera y su rolling_mean no seria nulo. Que lo
        sea es lo que demuestra que cada corrida arranca en frio.
        """
        features_dir = tmp_path / "features"
        main(_write_params(tmp_path, features_dir), readings_dir=readings_tree)

        second = pd.read_parquet(features_dir / "fault_type=07" / FEATURES_FILENAME)
        first_row = second[(second["timestep"] == 0) & (second["sensor_id"] == "SENS-01")]
        assert pd.isna(first_row["rolling_mean_10"].iloc[0])
        assert pd.isna(first_row["lag_1"].iloc[0])

    def test_a_missing_params_file_is_reported(self, tmp_path: Path) -> None:
        """The stage must not run on defaults it invented."""
        with pytest.raises(FileNotFoundError):
            main(tmp_path / "absent.yaml", readings_dir=tmp_path)

    def test_the_repository_params_configure_this_stage(self) -> None:
        """The committed params.yaml must be loadable by the featurizer.

        Sin esto, un error en params.yaml solo se descubriria al correr dvc repro.
        """
        params = load_feature_params()
        assert params.features_dir == Path("data/processed/features")


class TestPartitionDiscoveryIsReusable:
    """El arbol de features comparte disposicion con el de lecturas.

    El stage split reutiliza discover_partitions sobre features.parquet, de modo
    que el nombre del fichero y el stage que lo produce son argumentos: un
    mensaje de error que dijese "run adapt_tep" ante un arbol de features vacio
    apuntaria al remedio equivocado.
    """

    def test_a_different_filename_is_honoured(self, tmp_path: Path) -> None:
        """features.parquet must be discoverable with the same function."""
        partition = tmp_path / "fault_type=03"
        partition.mkdir(parents=True)
        _readings().to_parquet(partition / FEATURES_FILENAME, index=False)

        found = discover_partitions(tmp_path, filename=FEATURES_FILENAME)
        assert list(found) == [3]

    def test_the_error_names_the_producing_stage(self, tmp_path: Path) -> None:
        """Pointing at adapt_tep for a missing feature tree would misdirect."""
        empty = tmp_path / "features"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="featurize"):
            discover_partitions(empty, filename=FEATURES_FILENAME, producer="featurize")

    def test_the_readings_filename_stays_the_default(self, readings_tree: Path) -> None:
        """The featurize stage must keep working without passing the argument."""
        assert list(discover_partitions(readings_tree)) == [0, 7]
