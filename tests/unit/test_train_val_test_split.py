"""Unit tests for data.generators.train_val_test_split.

Las dos propiedades que un split de series temporales tiene que cumplir y que un
test de "el fichero existe" dejaria pasar: que NINGUN timestep aparezca en dos
splits, y que el maximo de train sea estrictamente menor que el minimo de val en
CADA fault_type. Un corte que se equivoque en cualquiera de las dos entrena
sobre el futuro y produce metricas de Fase 4 que no significan nada.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from data.generators.train_val_test_split import (
    SPLIT_NAMES,
    SplitParams,
    build_distribution,
    build_metrics,
    load_split_params,
    main,
    split_boundaries,
    split_partition,
    temporal_order_holds,
)
from ml.features.batch_featurizer import FEATURES_FILENAME

_FAULT_TYPES = (0, 1, 7)
_TAGS = ("SENS-01", "SENS-02")
_N_TIMESTEPS = 40


def _params(tmp_path: Path, **overrides: float) -> SplitParams:
    """Build split parameters pointing into a temporary directory.

    Args:
        tmp_path: Temporary directory.
        **overrides: Ratio overrides.

    Returns:
        The parameters.
    """
    ratios = {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15}
    ratios.update(overrides)
    return SplitParams(
        splits_dir=tmp_path / "splits",
        metrics_path=tmp_path / "metrics" / "split_metrics.json",
        distribution_path=tmp_path / "metrics" / "split_distribution.json",
        **ratios,  # type: ignore[arg-type]
    )


def _features(fault_type: int, n_timesteps: int = _N_TIMESTEPS) -> pd.DataFrame:
    """Build a long feature frame for one fault type.

    Args:
        fault_type: Partition identifier.
        n_timesteps: Samples per sensor.

    Returns:
        The frame.
    """
    return pd.DataFrame(
        [
            {
                "fault_type": fault_type,
                "timestep": step,
                "sensor_id": tag,
                "value": 10.0 + offset + step,
            }
            for step in range(n_timesteps)
            for offset, tag in enumerate(_TAGS)
        ]
    )


@pytest.fixture()
def feature_tree(tmp_path: Path) -> Path:
    """Write a three-partition feature tree and return its root."""
    root = tmp_path / "features"
    for fault_type in _FAULT_TYPES:
        partition = root / f"fault_type={fault_type:02d}"
        partition.mkdir(parents=True)
        _features(fault_type).to_parquet(partition / FEATURES_FILENAME, index=False)
    return root


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


class TestSplitBoundaries:
    """La aritmetica del corte."""

    def test_ranges_are_contiguous_and_ordered(self, tmp_path: Path) -> None:
        """train precedes val precedes test, with no gap between them."""
        bounds = split_boundaries(100, _params(tmp_path))
        assert bounds["train"] == range(0, 70)
        assert bounds["val"] == range(70, 85)
        assert bounds["test"] == range(85, 100)

    def test_no_timestep_is_lost_to_integer_division(self, tmp_path: Path) -> None:
        """The remainder goes to test rather than being silently dropped."""
        bounds = split_boundaries(37, _params(tmp_path))
        covered = sorted(step for r in bounds.values() for step in r)
        assert covered == list(range(37))

    def test_the_remainder_lands_in_test(self, tmp_path: Path) -> None:
        """37 * 0.7 = 25.9 -> 25 train, 37 * 0.15 = 5.55 -> 5 val, 7 test."""
        bounds = split_boundaries(37, _params(tmp_path))
        assert len(bounds["train"]) == 25
        assert len(bounds["val"]) == 5
        assert len(bounds["test"]) == 7

    def test_a_partition_too_short_is_rejected(self, tmp_path: Path) -> None:
        """Five timesteps cannot fill three splits at 70/15/15."""
        with pytest.raises(ValueError, match="cannot be split"):
            split_boundaries(5, _params(tmp_path))

    def test_the_ranges_never_overlap(self, tmp_path: Path) -> None:
        """The property the whole stage rests on."""
        for size in (20, 37, 100, 480, 500):
            bounds = split_boundaries(size, _params(tmp_path))
            steps = [step for r in bounds.values() for step in r]
            assert len(steps) == len(set(steps)) == size


# ---------------------------------------------------------------------------
# Splitting one partition
# ---------------------------------------------------------------------------


class TestSplitPartition:
    """El corte de una particion concreta."""

    def test_every_row_lands_in_exactly_one_split(self, tmp_path: Path) -> None:
        """No row may be duplicated or dropped."""
        frame = _features(0)
        slices = split_partition(frame, _params(tmp_path))
        assert sum(len(s) for s in slices.values()) == len(frame)

    def test_train_strictly_precedes_val_and_test(self, tmp_path: Path) -> None:
        """La comprobacion que impide entrenar sobre el futuro."""
        slices = split_partition(_features(0), _params(tmp_path))
        assert slices["train"]["timestep"].max() < slices["val"]["timestep"].min()
        assert slices["val"]["timestep"].max() < slices["test"]["timestep"].min()

    def test_no_timestep_is_shared_between_splits(self, tmp_path: Path) -> None:
        """Sharing one would leak the same instant into two roles."""
        slices = split_partition(_features(0), _params(tmp_path))
        seen: set[int] = set()
        for frame in slices.values():
            steps = set(frame["timestep"])
            assert not (steps & seen)
            seen |= steps

    def test_every_sensor_survives_in_every_split(self, tmp_path: Path) -> None:
        """The cut is temporal, so it must not drop instruments."""
        slices = split_partition(_features(0), _params(tmp_path))
        for frame in slices.values():
            assert set(frame["sensor_id"]) == set(_TAGS)

    def test_rows_are_ordered_within_a_split(self, tmp_path: Path) -> None:
        """A deterministic order keeps the parquet reproducible."""
        train = split_partition(_features(0), _params(tmp_path))["train"]
        assert train["timestep"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def _write_params(tmp_path: Path, features_dir: Path, **overrides: Any) -> Path:
    """Write a params.yaml holding the two sections the stage reads.

    Args:
        tmp_path: Temporary directory.
        features_dir: Root of the feature tree.
        **overrides: Keys replaced in the training: section.

    Returns:
        Path of the params file.
    """
    training: dict[str, Any] = {
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "splits_dir": str(tmp_path / "splits"),
        "metrics_path": str(tmp_path / "metrics" / "split_metrics.json"),
        "distribution_path": str(tmp_path / "metrics" / "split_distribution.json"),
    }
    training.update(overrides)
    document = {
        "training": training,
        "features": {
            "features_dir": str(features_dir),
            "window_samples": [10, 20],
            "min_samples_per_window": 3,
            "n_lags": 2,
            "sensor_selection": {
                "include_types": ["flow"],
                "exclude_sensor_ids": [],
                "expected_sensor_count": 2,
            },
            "correlation_pairs": [],
        },
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


class TestLoadSplitParams:
    """La validacion de los ratios."""

    def test_loads_a_valid_section(self, tmp_path: Path, feature_tree: Path) -> None:
        """The happy path resolves every field."""
        params = load_split_params(_write_params(tmp_path, feature_tree))
        assert params.train_ratio == 0.70
        assert params.splits_dir == tmp_path / "splits"

    def test_ratios_that_do_not_sum_to_one_are_rejected(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """A different sum discards samples silently."""
        path = _write_params(tmp_path, feature_tree, test_ratio=0.30)
        with pytest.raises(ValueError, match="sum to"):
            load_split_params(path)

    def test_a_non_positive_ratio_is_rejected(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """An empty split is not a split."""
        path = _write_params(tmp_path, feature_tree, val_ratio=0.0, test_ratio=0.30)
        with pytest.raises(ValueError, match="val_ratio"):
            load_split_params(path)

    def test_a_missing_key_is_named(self, tmp_path: Path, feature_tree: Path) -> None:
        """A missing splits_dir must be reported by name."""
        path = _write_params(tmp_path, feature_tree)
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        del document["training"]["splits_dir"]
        path.write_text(yaml.safe_dump(document), encoding="utf-8")

        with pytest.raises(KeyError, match="splits_dir"):
            load_split_params(path)

    def test_a_missing_file_is_reported(self, tmp_path: Path) -> None:
        """The stage must not run on invented defaults."""
        with pytest.raises(FileNotFoundError):
            load_split_params(tmp_path / "absent.yaml")

    def test_the_repository_params_load(self) -> None:
        """A broken committed params.yaml would fail the stage, not a test."""
        params = load_split_params()
        assert params.splits_dir == Path("data/processed/splits")
        assert params.metrics_path == Path("metrics/split_metrics.json")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestMain:
    """El stage completo."""

    def test_writes_one_parquet_per_split(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """Three splits, three files."""
        main(_write_params(tmp_path, feature_tree))
        for name in SPLIT_NAMES:
            assert (tmp_path / "splits" / f"{name}.parquet").exists()

    def test_every_fault_type_reaches_every_split(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """La estratificacion. Un corte temporal GLOBAL habria dejado clases
        enteras fuera de train, y el modelo nunca las habria visto."""
        main(_write_params(tmp_path, feature_tree))
        for name in SPLIT_NAMES:
            frame = pd.read_parquet(tmp_path / "splits" / f"{name}.parquet")
            assert set(frame["fault_type"]) == set(_FAULT_TYPES)

    def test_temporal_order_holds_within_every_fault_type(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """Comprobado POR CLASE, no en agregado.

        En agregado la comprobacion pasaria aunque una particion estuviese mal
        cortada, porque las otras la taparian.
        """
        main(_write_params(tmp_path, feature_tree))
        frames = {
            name: pd.read_parquet(tmp_path / "splits" / f"{name}.parquet")
            for name in SPLIT_NAMES
        }
        for fault_type in _FAULT_TYPES:
            per_class = {
                name: frame[frame["fault_type"] == fault_type]
                for name, frame in frames.items()
            }
            assert per_class["train"]["timestep"].max() < per_class["val"]["timestep"].min()
            assert per_class["val"]["timestep"].max() < per_class["test"]["timestep"].min()

    def test_no_row_is_lost_or_duplicated(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """The union of the splits must be the whole feature table."""
        metrics = main(_write_params(tmp_path, feature_tree))
        expected = len(_FAULT_TYPES) * _N_TIMESTEPS * len(_TAGS)
        assert metrics["rows_total"] == expected

        keys = pd.concat(
            [
                pd.read_parquet(tmp_path / "splits" / f"{name}.parquet")
                for name in SPLIT_NAMES
            ]
        )[["fault_type", "timestep", "sensor_id"]]
        assert not keys.duplicated().any()
        assert len(keys) == expected

    def test_metrics_are_written_and_valid_json(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """dvc metrics show reads this file."""
        main(_write_params(tmp_path, feature_tree))
        written = json.loads(
            (tmp_path / "metrics" / "split_metrics.json").read_text(encoding="utf-8")
        )
        assert written["fault_types_per_split"] == len(_FAULT_TYPES)
        assert written["fault_types_in_every_split"] is True
        assert all(f"{name}_rows" in written for name in SPLIT_NAMES)

    def test_the_achieved_share_tracks_the_configured_ratio(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """Reporting both is what makes the drift from integer division visible."""
        metrics = main(_write_params(tmp_path, feature_tree))
        assert metrics["train_ratio_configured"] == 0.70
        assert abs(metrics["train_share"] - 0.70) < 0.05


class TestTemporalOrderIsCheckedPerClass:
    """La bandera que acredita el orden temporal mira clase por clase.

    En agregado la propiedad NO se cumple sobre los datos reales y no puede: d00
    trae 500 timesteps y los ficheros de fallo 480, asi que el train de d00 llega
    al 349 mientras el val de las clases de fallo empieza en el 336. Comparar
    minimos y maximos globales daria un falso negativo ahi, y con particiones de
    igual longitud daria un falso positivo capaz de tapar una clase mal cortada.
    """

    def test_it_holds_for_a_correct_split(self, tmp_path: Path) -> None:
        """The happy path over several partitions."""
        params = _params(tmp_path)
        collected: dict[str, list[pd.DataFrame]] = {n: [] for n in SPLIT_NAMES}
        for fault_type in _FAULT_TYPES:
            for name, slice_ in split_partition(_features(fault_type), params).items():
                collected[name].append(slice_)
        splits = {n: pd.concat(f, ignore_index=True) for n, f in collected.items()}

        assert temporal_order_holds(splits) is True

    def test_partitions_of_different_lengths_still_pass(self, tmp_path: Path) -> None:
        """Reproduce la geometria real: 500 timesteps en d00, 480 en el resto.

        Los rangos globales se solapan y aun asi el orden es correcto en cada
        clase, que es exactamente el caso que una comprobacion agregada suspende.
        """
        params = _params(tmp_path)
        collected: dict[str, list[pd.DataFrame]] = {n: [] for n in SPLIT_NAMES}
        for fault_type, length in ((0, 500), (1, 480), (7, 480)):
            frame = _features(fault_type, n_timesteps=length)
            for name, slice_ in split_partition(frame, params).items():
                collected[name].append(slice_)
        splits = {n: pd.concat(f, ignore_index=True) for n, f in collected.items()}

        assert splits["train"]["timestep"].max() > splits["val"]["timestep"].min()
        assert temporal_order_holds(splits) is True

    def test_a_single_mis_cut_class_is_caught(self, tmp_path: Path) -> None:
        """Una clase mal cortada no puede quedar tapada por las demas."""
        params = _params(tmp_path)
        collected: dict[str, list[pd.DataFrame]] = {n: [] for n in SPLIT_NAMES}
        for fault_type in _FAULT_TYPES:
            for name, slice_ in split_partition(_features(fault_type), params).items():
                collected[name].append(slice_)
        splits = {n: pd.concat(f, ignore_index=True) for n, f in collected.items()}

        # Se cuela en val una muestra temprana de una sola clase.
        intruder = splits["train"][
            (splits["train"]["fault_type"] == 1) & (splits["train"]["timestep"] == 0)
        ]
        splits["val"] = pd.concat([splits["val"], intruder], ignore_index=True)

        assert temporal_order_holds(splits) is False

    def test_a_missing_class_in_a_split_is_caught(self, tmp_path: Path) -> None:
        """A class absent from a split cannot be said to be ordered."""
        params = _params(tmp_path)
        splits = split_partition(_features(0), params)
        splits["val"] = splits["val"].iloc[0:0]

        assert temporal_order_holds(splits) is False

    def test_the_flag_reaches_the_metrics(self, tmp_path: Path) -> None:
        """It is the number a reader should trust, so it must be reported."""
        splits = split_partition(_features(0), _params(tmp_path))
        assert build_metrics(splits, _params(tmp_path))["temporal_order_holds"] is True


class TestBuildMetrics:
    """El resumen que consume dvc metrics show."""

    def test_shares_sum_to_one(self, tmp_path: Path) -> None:
        """Otherwise rows went missing between the split and the report."""
        splits = split_partition(_features(0), _params(tmp_path))
        metrics = build_metrics(splits, _params(tmp_path))
        shares = [metrics[f"{name}_share"] for name in SPLIT_NAMES]
        assert sum(shares) == pytest.approx(1.0)

    def test_the_fault_type_distribution_is_reported(self, tmp_path: Path) -> None:
        """It is what makes the stratification auditable from the metrics alone."""
        splits = split_partition(_features(7), _params(tmp_path))
        assert "fault_type_07" in build_distribution(splits)["train"]

    def test_timestep_ranges_are_reported_per_split(self, tmp_path: Path) -> None:
        """A reader must be able to see the temporal cut without opening the parquet."""
        splits = split_partition(_features(0), _params(tmp_path))
        metrics = build_metrics(splits, _params(tmp_path))
        assert metrics["train_max_timestep"] < metrics["val_min_timestep"]


class TestMetricsStayReadable:
    """`dvc metrics show` aplana el JSON: el fichero de metricas debe ser corto.

    Con el desglose de las 22 clases por cada uno de los tres splits dentro, el
    comando imprimia 66 columnas y en un terminal no se leia nada. El desglose
    vive ahora en un fichero hermano, tambien versionado.
    """

    def test_the_metrics_file_stays_flat(self, tmp_path: Path) -> None:
        """A nested value would become a column family when flattened."""
        splits = split_partition(_features(0), _params(tmp_path))
        metrics = build_metrics(splits, _params(tmp_path))
        assert all(not isinstance(value, dict) for value in metrics.values())

    def test_the_metrics_file_stays_small(self, tmp_path: Path) -> None:
        """A readable table is the whole point of the split."""
        splits = split_partition(_features(0), _params(tmp_path))
        assert len(build_metrics(splits, _params(tmp_path))) <= 20

    def test_the_headline_metrics_prove_both_promises(self, tmp_path: Path) -> None:
        """Stratification and temporal order must be checkable without the parquet."""
        splits = split_partition(_features(0), _params(tmp_path))
        metrics = build_metrics(splits, _params(tmp_path))

        assert metrics["fault_types_in_every_split"] is True
        assert metrics["temporal_order_holds"] is True

    def test_the_distribution_covers_every_split(self, tmp_path: Path) -> None:
        """The detail must not lose a split on the way out."""
        splits = split_partition(_features(0), _params(tmp_path))
        assert set(build_distribution(splits)) == set(SPLIT_NAMES)

    def test_the_distribution_file_is_written(
        self, tmp_path: Path, feature_tree: Path
    ) -> None:
        """It is a declared out of the stage, so it must exist after a run."""
        main(_write_params(tmp_path, feature_tree))
        written = json.loads(
            (tmp_path / "metrics" / "split_distribution.json").read_text(encoding="utf-8")
        )
        assert set(written["train"]) == {
            f"fault_type_{ft:02d}" for ft in _FAULT_TYPES
        }
