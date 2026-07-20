"""Explore and report statistics for the Tennessee Eastman Process (TEP) dataset.

Produces two artefacts:
    data/reports/tep_exploration.json  - per-file statistics and missing-value counts
    data/reports/tep_correlations.csv  - Pearson correlation matrix over all pooled samples
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from data.generators.tep_loader import load_dat_file
from data.generators.tep_params import DEFAULT_PARAMS_PATH, load_tep_params

_LOG = logging.getLogger(__name__)

_FILE_NAMES: list[str] = ["d00.dat"] + [f"d{i:02d}.dat" for i in range(1, 22)]
_REPORTS_DIR = Path("data/reports")


def _compute_dataset_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute shape, per-column descriptive statistics, and missing-value counts.

    Args:
        df: DataFrame with TEP data (52 numeric columns).

    Returns:
        Dictionary containing:
            shape: [n_rows, n_cols]
            missing_counts: {col_name: n_missing_values}
            statistics: {col_name: {mean, std, min, max}}
    """
    description = df.describe()
    missing_counts = df.isnull().sum()

    stats: dict[str, dict[str, float]] = {}
    for col in df.columns:
        stats[col] = {
            "mean": float(description.loc["mean", col]),
            "std": float(description.loc["std", col]),
            "min": float(description.loc["min", col]),
            "max": float(description.loc["max", col]),
        }

    return {
        "shape": list(df.shape),
        "missing_counts": {col: int(missing_counts[col]) for col in df.columns},
        "statistics": stats,
    }


def explore_tep(data_dir: str, reports_dir: str | Path = _REPORTS_DIR) -> dict[str, Any]:
    """Load all TEP files and produce a structured exploration report.

    Iterates over d00.dat through d21.dat, computes per-file statistics, then
    pools all samples to build a global Pearson correlation matrix.

    Writes:
        data/reports/tep_exploration.json  - full report dictionary
        data/reports/tep_correlations.csv  - (52 x 52) correlation matrix

    Args:
        data_dir: Directory containing the TEP .dat files.
        reports_dir: Directory where the two report artefacts are written.
            Created if it does not exist.

    Returns:
        Report dictionary with keys:
            datasets: {file_key: {shape, missing_counts, statistics}}
            correlation_matrix: {col: {col: pearson_r}}
    """
    data_path = Path(data_dir)
    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"datasets": {}}
    pooled_frames: list[pd.DataFrame] = []

    for filename in _FILE_NAMES:
        filepath = data_path / filename
        if not filepath.exists():
            _LOG.warning("File not found, skipping: %s", filepath)
            continue

        _LOG.info("Loading %s", filename)
        df = load_dat_file(filepath)
        fault_key = filename.replace(".dat", "")
        report["datasets"][fault_key] = _compute_dataset_stats(df)
        pooled_frames.append(df)

    if pooled_frames:
        pooled = pd.concat(pooled_frames, ignore_index=True)
        corr_matrix = pooled.corr(method="pearson")

        report["correlation_matrix"] = {
            row: {col: float(val) for col, val in row_data.items()}
            for row, row_data in corr_matrix.to_dict().items()
        }

        corr_path = reports_path / "tep_correlations.csv"
        corr_matrix.to_csv(corr_path)
        _LOG.info("Saved correlation matrix to %s", corr_path)
    else:
        _LOG.warning("No TEP files found in %s; report will be empty.", data_dir)
        report["correlation_matrix"] = {}

    exploration_path = reports_path / "tep_exploration.json"
    with exploration_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _LOG.info("Saved exploration report to %s", exploration_path)

    return report


def main(params_path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Run the exploration stage using the paths configured in params.yaml.

    Args:
        params_path: Path to the params file holding the tep: section.

    Returns:
        The exploration report dictionary.
    """
    params = load_tep_params(params_path)
    return explore_tep(str(params.raw_dir), params.reports_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
