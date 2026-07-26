"""Unit tests for ml.features.feature_params.

Lo que se fija aqui es sobre todo lo que el loader RECHAZA. La seccion features:
estuvo huerfana hasta T4.2 y sus valores venian en segundos de un TDD escrito
para una cadencia de 1 s; el guardarrail que impide que vuelvan a colarse es
justamente la validacion, no el camino feliz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from ml.features.feature_params import (
    FeatureParams,
    SensorSelection,
    load_feature_params,
)


def _section(**overrides: Any) -> dict[str, Any]:
    """Return a valid features: section with optional overrides.

    Args:
        **overrides: Keys to replace in the default section.

    Returns:
        The section as a mapping.
    """
    section: dict[str, Any] = {
        "features_dir": "data/processed/features",
        "window_samples": [10, 20, 60],
        "min_samples_per_window": 3,
        "n_lags": 5,
        "sensor_selection": {
            "include_types": ["flow", "pressure", "thermocouple", "normalized", "position"],
            "exclude_sensor_ids": [],
            "expected_sensor_count": 52,
        },
        "correlation_pairs": [],
    }
    section.update(overrides)
    return section


@pytest.fixture()
def write_features(tmp_path: Path):
    """Return a factory writing a params.yaml with a features: section."""

    def _write(**overrides: Any) -> Path:
        path = tmp_path / "params.yaml"
        path.write_text(
            yaml.safe_dump({"features": _section(**overrides)}), encoding="utf-8"
        )
        return path

    return _write


class TestHappyPath:
    """La carga de una seccion valida."""

    def test_loads_every_field(self, write_features) -> None:
        """A well-formed section must resolve into a FeatureParams."""
        params = load_feature_params(write_features())

        assert isinstance(params, FeatureParams)
        assert params.features_dir == Path("data/processed/features")
        assert params.window_samples == (10, 20, 60)
        assert params.n_lags == 5
        assert params.correlation_pairs == ()

    def test_windows_are_sorted(self, write_features) -> None:
        """Order in the file must not change the meaning of shortest and longest."""
        params = load_feature_params(write_features(window_samples=[60, 10, 20]))
        assert params.window_samples == (10, 20, 60)
        assert params.shortest_window == 10
        assert params.longest_window == 60

    def test_window_seconds_uses_the_measured_cadence(self, write_features) -> None:
        """The same window means different spans at different cadences."""
        params = load_feature_params(write_features())
        assert params.window_seconds(180.0) == {10: 1800.0, 20: 3600.0, 60: 10800.0}
        assert params.window_seconds(1.0) == {10: 10.0, 20: 20.0, 60: 60.0}

    def test_selection_is_parsed(self, write_features) -> None:
        """The selection policy must arrive as sets, not as lists."""
        params = load_feature_params(write_features())
        assert isinstance(params.selection, SensorSelection)
        assert "flow" in params.selection.include_types
        assert params.selection.expected_sensor_count == 52

    def test_correlation_pairs_are_parsed(self, write_features) -> None:
        """A configured pair must arrive as a tuple of two tags."""
        params = load_feature_params(
            write_features(correlation_pairs=[["TEP-XMEAS-01", "TEP-XMEAS-02"]])
        )
        assert params.correlation_pairs == (("TEP-XMEAS-01", "TEP-XMEAS-02"),)

    def test_missing_correlation_pairs_defaults_to_empty(self, tmp_path: Path) -> None:
        """The key is optional; absent means the group emits nothing."""
        section = _section()
        del section["correlation_pairs"]
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"features": section}), encoding="utf-8")

        assert load_feature_params(path).correlation_pairs == ()


class TestRejections:
    """Lo que el loader tiene que rechazar."""

    def test_missing_file(self, tmp_path: Path) -> None:
        """An absent params file must say so."""
        with pytest.raises(FileNotFoundError):
            load_feature_params(tmp_path / "nope.yaml")

    def test_missing_section(self, tmp_path: Path) -> None:
        """A params file without features: must name the section."""
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"tep": {}}), encoding="utf-8")
        with pytest.raises(KeyError, match="features"):
            load_feature_params(path)

    @pytest.mark.parametrize(
        "key",
        ["features_dir", "window_samples", "min_samples_per_window", "n_lags",
         "sensor_selection"],
    )
    def test_each_required_key_is_named(self, tmp_path: Path, key: str) -> None:
        """A missing key must be reported by name, not as a KeyError of pandas."""
        section = _section()
        del section[key]
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump({"features": section}), encoding="utf-8")

        with pytest.raises(KeyError, match=key):
            load_feature_params(path)

    def test_windows_in_seconds_are_caught(self, write_features) -> None:
        """The whole point: a sub-sample window must not load.

        [10, 30, 60, 300] eran los window_sizes en SEGUNDOS del params.yaml
        anterior. Con cadencia de 180 s los tres primeros no llegan a una muestra.
        Aqui se rechaza el caso generico: cualquier ventana por debajo de 2.
        """
        with pytest.raises(ValueError, match="MUESTRAS"):
            load_feature_params(write_features(window_samples=[1, 30, 60, 300]))

    def test_empty_window_list_is_rejected(self, write_features) -> None:
        """Without a window there are no rolling features to compute."""
        with pytest.raises(ValueError, match="must not be empty"):
            load_feature_params(write_features(window_samples=[]))

    def test_non_list_window_is_rejected(self, write_features) -> None:
        """A scalar where a list belongs must be named as such."""
        with pytest.raises(TypeError, match="window_samples"):
            load_feature_params(write_features(window_samples=60))

    def test_duplicate_windows_are_rejected(self, write_features) -> None:
        """Two identical windows would produce colliding column names."""
        with pytest.raises(ValueError, match="repeats a window"):
            load_feature_params(write_features(window_samples=[10, 10, 20]))

    def test_min_samples_below_two_is_rejected(self, write_features) -> None:
        """A std over one sample is zero, indistinguishable from a flat channel."""
        with pytest.raises(ValueError, match="single sample"):
            load_feature_params(write_features(min_samples_per_window=1))

    def test_min_samples_above_the_shortest_window_is_rejected(
        self, write_features
    ) -> None:
        """That window could never emit a value."""
        with pytest.raises(ValueError, match="could never emit"):
            load_feature_params(
                write_features(window_samples=[5, 20], min_samples_per_window=8)
            )

    def test_negative_lags_are_rejected(self, write_features) -> None:
        """A negative lag would look into the future."""
        with pytest.raises(ValueError, match="n_lags"):
            load_feature_params(write_features(n_lags=-1))

    def test_non_mapping_selection_is_rejected(self, write_features) -> None:
        """The selection policy must be a mapping."""
        with pytest.raises(TypeError, match="sensor_selection"):
            load_feature_params(write_features(sensor_selection=["flow"]))

    def test_missing_expected_count_is_rejected(self, write_features) -> None:
        """Without it there is no schema-drift check at all."""
        with pytest.raises(KeyError, match="expected_sensor_count"):
            load_feature_params(
                write_features(sensor_selection={"include_types": ["flow"]})
            )

    def test_non_positive_expected_count_is_rejected(self, write_features) -> None:
        """Expecting zero sensors makes the check vacuous."""
        with pytest.raises(ValueError, match="expected_sensor_count"):
            load_feature_params(
                write_features(
                    sensor_selection={
                        "include_types": ["flow"],
                        "expected_sensor_count": 0,
                    }
                )
            )

    def test_non_list_correlation_pairs_is_rejected(self, write_features) -> None:
        """A mapping where a list belongs must be named."""
        with pytest.raises(TypeError, match="correlation_pairs"):
            load_feature_params(write_features(correlation_pairs={"a": "b"}))

    def test_malformed_pair_is_rejected(self, write_features) -> None:
        """A pair needs exactly two tags."""
        with pytest.raises(TypeError, match="pair of sensor ids"):
            load_feature_params(write_features(correlation_pairs=[["only-one"]]))

    def test_self_pair_is_rejected(self, write_features) -> None:
        """A sensor correlates with itself at 1 by definition."""
        with pytest.raises(ValueError, match="with\n?\\s*itself|itself"):
            load_feature_params(write_features(correlation_pairs=[["A", "A"]]))


class TestCommittedParams:
    """El params.yaml del repositorio tiene que cargar."""

    def test_the_repository_params_load(self) -> None:
        """A broken committed params.yaml would fail the featurize stage, not a test."""
        params = load_feature_params()
        assert params.window_samples
        assert params.selection.expected_sensor_count == 52

    def test_the_repository_windows_are_meaningful_at_tep_cadence(self) -> None:
        """Every committed window must span at least two TEP samples."""
        params = load_feature_params()
        spans = params.window_seconds(180.0)
        assert min(spans.values()) >= 2 * 180.0
