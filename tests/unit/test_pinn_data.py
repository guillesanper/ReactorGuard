"""Unit tests for ml.training.pinn_data (pares, contexto y z-score del stage train).

Los frames salen de `tests/support/sim_frames.py`: trazas del modelo de referencia
con ruido de medida, en el formato ancho de prepare_sim, de modo que las perdidas
fisicas y la mascara cuasi-estacionaria son evaluables.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from data.simulation.plant_params import load_plant_params
from data.simulation.sensor_catalog import SensorCatalog
from ml.models.physics_loss import PhysicsConstants
from ml.models.physics_state import StateIndex, build_state_index
from ml.training.pinn_data import (
    CONTEXT_NAMES,
    FRAME_COLUMNS,
    NormalizationStats,
    PairSet,
    build_pairs,
    context_features,
    fit_normalization,
    noise_mse,
    partition_columns,
    persistence_mse,
)
from tests.support.sim_frames import CONFIGS, build_catalog, pinn_partitions


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    """Return the small train/val/test frames (built once per module)."""
    return pinn_partitions()


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned catalog."""
    return build_catalog()


@pytest.fixture(scope="module")
def index(catalog: SensorCatalog) -> StateIndex:
    """Return the state index of the catalog order."""
    return build_state_index(catalog.tags, catalog.roles)


@pytest.fixture(scope="module")
def constants() -> PhysicsConstants:
    """Return the plant scales."""
    return PhysicsConstants.from_plant(load_plant_params(CONFIGS / "plant_parameters.yaml"))


@pytest.fixture(scope="module")
def stats(frames: dict[str, pd.DataFrame], catalog: SensorCatalog) -> NormalizationStats:
    """Return the z-score fitted on train."""
    return fit_normalization(frames["train"], catalog.tags, catalog.roles)


def _pairs(
    frame: pd.DataFrame,
    stats: NormalizationStats,
    catalog: SensorCatalog,
    index: StateIndex,
    constants: PhysicsConstants,
) -> PairSet:
    return build_pairs(frame, stats, catalog.roles, index, constants)


class TestPartitionColumns:
    """The trainer reads only what it needs."""

    def test_identity_labels_then_values_then_true(self, catalog: SensorCatalog) -> None:
        """Columns follow FRAME_COLUMNS, then value: and true: in catalog order."""
        columns = partition_columns(catalog.tags)
        assert columns[: len(FRAME_COLUMNS)] == list(FRAME_COLUMNS)
        assert columns[len(FRAME_COLUMNS)] == f"value:{catalog.tags[0]}"
        assert columns[-1] == f"true:{catalog.tags[-1]}"
        assert len(columns) == len(FRAME_COLUMNS) + 2 * len(catalog.tags)


class TestContextFeatures:
    """D4-2: power level, its backward rate and the hour of the day."""

    def test_shape_and_order(self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog) -> None:
        """One row per instant, the four CONTEXT_NAMES columns."""
        context = context_features(frames["train"], catalog.tags, catalog.roles)
        assert context.shape == (len(frames["train"]), len(CONTEXT_NAMES))
        assert CONTEXT_NAMES == ("power_level", "power_rate", "hour_sin", "hour_cos")

    def test_power_level_is_p_th_over_100(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """The P_th role has one channel: power_level = PWR-TH-01 / 100."""
        frame = frames["train"]
        context = context_features(frame, catalog.tags, catalog.roles)
        expected = frame["value:PWR-TH-01"].to_numpy() / 100.0
        np.testing.assert_allclose(context[:, 0], expected)

    def test_power_rate_is_a_backward_difference_per_hour(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """rate(t) = (level(t) - level(t-1)) / 0.05 h, and 0 at the first instant of a run."""
        frame = frames["train"]
        context = context_features(frame, catalog.tags, catalog.roles)
        level = context[:, 0]
        first = np.flatnonzero(frame["timestep"].to_numpy() == 0)
        assert list(first) == [0, 24]
        assert context[first, 1].tolist() == [0.0, 0.0]
        np.testing.assert_allclose(context[1:24, 1], (level[1:24] - level[:23]) / 0.05)

    def test_rate_uses_only_the_past(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """Changing a future instant does not change the context of earlier instants."""
        frame = frames["train"].copy()
        before = context_features(frame, catalog.tags, catalog.roles)
        frame.loc[10, "value:PWR-TH-01"] += 50.0
        after = context_features(frame, catalog.tags, catalog.roles)
        np.testing.assert_array_equal(before[:10], after[:10])
        assert not np.array_equal(before[10:12], after[10:12])

    def test_hour_is_a_point_on_the_unit_circle(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """06:00 UTC -> (sin, cos) = (1, 0); 18:00 -> (-1, 0)."""
        frame = frames["train"].iloc[:2].copy()
        frame["timestamp"] = pd.to_datetime(
            ["2026-01-01T06:00:00Z", "2026-01-01T18:00:00Z"], utc=True
        )
        frame["timestep"] = [0, 1]
        context = context_features(frame, catalog.tags, catalog.roles)
        np.testing.assert_allclose(context[:, 2], [1.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(context[:, 3], [0.0, 0.0], atol=1e-12)

    def test_single_instant(self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog) -> None:
        """A one-row frame has rate 0."""
        context = context_features(frames["train"].iloc[:1], catalog.tags, catalog.roles)
        assert context[0, 1] == 0.0

    def test_missing_power_role(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """Without P_th the context cannot be computed."""
        roles = {k: v for k, v in catalog.roles.items() if k != "P_th"}
        with pytest.raises(KeyError, match="P_th"):
            context_features(frames["train"], catalog.tags, roles)

    def test_power_role_naming_an_unknown_sensor(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """A role that names a sensor outside the order is rejected."""
        roles = {**catalog.roles, "P_th": ("NOPE",)}
        with pytest.raises(KeyError, match="NOPE"):
            context_features(frames["train"], catalog.tags, roles)

    def test_unordered_run_is_rejected(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """Instants of a run must be consecutive."""
        frame = frames["train"].iloc[[0, 2, 1]]
        with pytest.raises(ValueError, match="consecutive"):
            context_features(frame, catalog.tags, catalog.roles)

    def test_split_run_is_rejected(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """The instants of a run cannot be interleaved with another run."""
        frame = frames["train"]
        mixed = pd.concat([frame.iloc[:2], frame.iloc[24:26], frame.iloc[2:4]])
        with pytest.raises(ValueError, match="contiguous"):
            context_features(mixed, catalog.tags, catalog.roles)


class TestFitNormalization:
    """z-score of train, sensors and context, on unlabelled instants."""

    def test_sensor_statistics_are_those_of_fit_zscore(
        self, frames: dict[str, pd.DataFrame], stats: NormalizationStats
    ) -> None:
        """Mean and population std of the measured values of unlabelled instants."""
        frame = frames["train"]
        rows = frame.loc[~frame["labelled"], "value:TC-HL-1"]
        index = stats.tags.index("TC-HL-1")
        assert stats.sensor_mean[index] == pytest.approx(rows.mean())
        assert stats.sensor_std[index] == pytest.approx(rows.std(ddof=0))

    def test_context_statistics_skip_labelled_instants(
        self,
        frames: dict[str, pd.DataFrame],
        catalog: SensorCatalog,
        stats: NormalizationStats,
    ) -> None:
        """The context is averaged over the same instants as the sensors."""
        frame = frames["train"]
        context = context_features(frame, catalog.tags, catalog.roles)[~frame["labelled"]]
        np.testing.assert_allclose(stats.context_mean, context.mean(axis=0))
        np.testing.assert_allclose(stats.context_std, context.std(axis=0))

    def test_input_statistics_concatenate_sensors_and_context(
        self, stats: NormalizationStats
    ) -> None:
        """The input row is sensors then context."""
        assert stats.input_mean == stats.sensor_mean + stats.context_mean
        assert stats.input_std == stats.sensor_std + stats.context_std
        assert len(stats.input_mean) == len(stats.tags) + len(CONTEXT_NAMES)

    def test_constant_context_is_rejected(
        self, frames: dict[str, pd.DataFrame], catalog: SensorCatalog
    ) -> None:
        """A context input without variation cannot be normalised."""
        frame = frames["train"].copy()
        frame["timestamp"] = pd.Timestamp("2026-01-01T00:00:00Z")
        with pytest.raises(ValueError, match="hour_sin"):
            fit_normalization(frame, catalog.tags, catalog.roles)


class TestBuildPairs:
    """D4-6 pairs and the M2-3 mass mask."""

    def test_only_trainable_pairs(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """One pair per trainable_pair row; the labelled instant removes two pairs per run."""
        pairs = _pairs(frames["val"], stats, catalog, index, constants)
        assert len(pairs) == int(frames["val"]["trainable_pair"].sum()) == 2 * (23 - 2)
        assert pairs.inputs.shape == (len(pairs), len(catalog.tags) + len(CONTEXT_NAMES))
        assert pairs.targets.shape == pairs.true_targets.shape == (len(pairs), len(catalog.tags))
        assert pairs.inputs.dtype == pairs.targets.dtype == torch.float32
        assert pairs.n_sensors == len(catalog.tags)

    def test_input_is_t_and_target_is_t_plus_one(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """The first pair maps the measured instant 0 to the measured instant 1."""
        frame = frames["train"]
        pairs = _pairs(frame, stats, catalog, index, constants)
        k = stats.tags.index("TC-CL-1")
        mean, std = stats.sensor_mean[k], stats.sensor_std[k]
        assert pairs.inputs[0, k].item() == pytest.approx(
            (frame.loc[0, "value:TC-CL-1"] - mean) / std, rel=1e-5
        )
        assert pairs.targets[0, k].item() == pytest.approx(
            (frame.loc[1, "value:TC-CL-1"] - mean) / std, rel=1e-5
        )
        assert pairs.true_targets[0, k].item() == pytest.approx(
            (frame.loc[1, "true:TC-CL-1"] - mean) / std, rel=1e-5
        )

    def test_context_is_normalised(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """The context columns of train are z-scores with train statistics."""
        frame = frames["train"]
        pairs = _pairs(frame, stats, catalog, index, constants)
        raw = context_features(frame, catalog.tags, catalog.roles)[frame["trainable_pair"]]
        expected = (raw - np.array(stats.context_mean)) / np.array(stats.context_std)
        context = pairs.inputs[:, len(catalog.tags) :].numpy()
        np.testing.assert_allclose(context, expected, rtol=1e-5)

    def test_steady_mask_uses_measured_t_and_t_plus_one(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """A jump of T_avg between t and t+1 marks the pair as not quasi-steady (M2-3)."""
        frame = frames["train"].copy()
        for tag in ("TC-HL-1", "TC-HL-2", "TC-CL-1", "TC-CL-2"):
            frame.loc[3, f"value:{tag}"] += 20.0
        pairs = _pairs(frame, stats, catalog, index, constants)
        steady = pairs.steady.tolist()
        assert steady[2] is False and steady[3] is False
        assert steady[0] is True and steady[5] is True

    def test_empty_partition(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """Without trainable pairs the set is empty and the baselines are NaN."""
        frame = frames["val"].assign(trainable_pair=False)
        pairs = _pairs(frame, stats, catalog, index, constants)
        assert len(pairs) == 0
        assert pairs.steady.shape == (0,)
        assert math.isnan(persistence_mse(pairs))
        assert math.isnan(noise_mse(pairs))

    def test_last_instant_of_a_run_cannot_start_a_pair(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """A malformed trainable flag at the end of a run is an error, not a cross-run pair."""
        frame = frames["val"].copy()
        frame.loc[23, "trainable_pair"] = True
        with pytest.raises(ValueError, match="last instant"):
            _pairs(frame, stats, catalog, index, constants)

    def test_mixed_cadence_is_rejected(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """The mass mask needs one dt per partition."""
        frame = frames["val"].copy()
        frame.loc[1:23, "timestamp"] = frame.loc[1:23, "timestamp"] + pd.Timedelta(seconds=60)
        with pytest.raises(ValueError, match="cadence"):
            _pairs(frame, stats, catalog, index, constants)

    def test_index_must_follow_the_sensor_order(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        constants: PhysicsConstants,
    ) -> None:
        """A state index built on another order would read the wrong columns."""
        reordered = build_state_index(tuple(reversed(catalog.tags)), catalog.roles)
        with pytest.raises(ValueError, match="sensor order"):
            build_pairs(frames["val"], stats, catalog.roles, reordered, constants)


class TestBaselines:
    """Persistence and noise references (seccion 2.3 del plan)."""

    def test_persistence_and_noise(
        self,
        frames: dict[str, pd.DataFrame],
        stats: NormalizationStats,
        catalog: SensorCatalog,
        index: StateIndex,
        constants: PhysicsConstants,
    ) -> None:
        """Both are the mean squared z-differences of their definition."""
        pairs = _pairs(frames["val"], stats, catalog, index, constants)
        n = len(catalog.tags)
        persistence = ((pairs.inputs[:, :n] - pairs.targets) ** 2).double().mean().item()
        noise = ((pairs.targets - pairs.true_targets) ** 2).double().mean().item()
        assert persistence_mse(pairs) == pytest.approx(persistence, rel=1e-6)
        assert noise_mse(pairs) == pytest.approx(noise, rel=1e-6)
        assert 0.0 < noise_mse(pairs) < persistence_mse(pairs)
