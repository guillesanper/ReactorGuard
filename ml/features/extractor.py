"""Feature extraction pipeline for reactor sensor streams."""

from __future__ import annotations

import numpy as np
import pandas as pd


SENSOR_CHANNELS = [
    "core_power",
    "coolant_temp_in",
    "coolant_temp_out",
    "primary_pressure",
    "coolant_flow_rate",
    "fuel_temp",
    "neutron_flux_ex_core",
    "steam_generator_level",
]

WINDOW_SIZES = [10, 30, 60, 300]  # seconds; overridden by params.yaml


class FeatureExtractor:
    """Extracts rolling-window statistical features from raw sensor readings.

    Args:
        window_sizes: List of rolling window sizes in seconds.
        n_lags: Number of lag features per sensor channel.
        dt_seconds: Sampling interval in seconds.
    """

    def __init__(
        self,
        window_sizes: list[int] = WINDOW_SIZES,
        n_lags: int = 5,
        dt_seconds: float = 1.0,
    ) -> None:
        self.window_sizes = window_sizes
        self.n_lags = n_lags
        self.dt_seconds = dt_seconds

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features for a sensor dataframe.

        Args:
            df: DataFrame with columns matching SENSOR_CHANNELS and a
                DatetimeIndex (or integer index at dt_seconds cadence).

        Returns:
            Feature DataFrame (same index, many more columns).
        """
        features = df[SENSOR_CHANNELS].copy()

        for channel in SENSOR_CHANNELS:
            series = df[channel]

            # Rolling statistics
            for w in self.window_sizes:
                w_rows = int(w / self.dt_seconds)
                roll = series.rolling(window=w_rows, min_periods=1)
                features[f"{channel}_mean_{w}s"] = roll.mean()
                features[f"{channel}_std_{w}s"] = roll.std().fillna(0.0)
                features[f"{channel}_min_{w}s"] = roll.min()
                features[f"{channel}_max_{w}s"] = roll.max()

            # Lag features
            for lag in range(1, self.n_lags + 1):
                features[f"{channel}_lag{lag}"] = series.shift(lag).fillna(method="bfill")

            # Rate of change (first derivative approximation)
            features[f"{channel}_delta"] = series.diff().fillna(0.0)

        return features

    @property
    def feature_names(self) -> list[str]:
        """Return expected feature column names (useful for SHAP)."""
        names: list[str] = list(SENSOR_CHANNELS)
        for ch in SENSOR_CHANNELS:
            for w in self.window_sizes:
                names += [f"{ch}_mean_{w}s", f"{ch}_std_{w}s",
                          f"{ch}_min_{w}s", f"{ch}_max_{w}s"]
            for lag in range(1, self.n_lags + 1):
                names.append(f"{ch}_lag{lag}")
            names.append(f"{ch}_delta")
        return names
