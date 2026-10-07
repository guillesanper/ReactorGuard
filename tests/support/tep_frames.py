"""Builders of synthetic long-format TEP frames for the streamer tests.

Reproducen las columnas que escribe `TEPAdapter.adapt_all` (mismos ids de sensor,
tipos, unidades y orden por timestep y columna), de modo que `readings_from_frame`
los lee igual que al parquet real, sin depender de que `data/processed/tep` exista.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.generators.tep_adapter import (
    LOCATION_MAP,
    SENSOR_TYPE_MAP,
    UNIT_MAP,
    reading_uuid,
    sensor_id,
)
from data.generators.tep_loader import N_COLUMNS

PLANT_ID = "TEP-PLANT-01"
START = datetime(2000, 1, 1, tzinfo=UTC)
SAMPLE_INTERVAL = timedelta(minutes=3)


def make_run_frame(
    fault_type: int,
    n_timesteps: int,
    *,
    stuck_sensor: str | None = None,
) -> pd.DataFrame:
    """Build the long-format frame of one TEP run.

    Args:
        fault_type: Fault identifier of the run (0 = normal).
        n_timesteps: Number of timesteps; the frame has 52 rows per timestep.
        stuck_sensor: Sensor whose value stays constant, as in the frozen XMV-04
            of fault 21. Every other value varies with the timestep.

    Returns:
        A frame with the columns of adapt_tep, ordered by (timestep, column).
    """
    rows: list[dict[str, object]] = []
    for step in range(n_timesteps):
        timestamp = START + step * SAMPLE_INTERVAL
        for col in range(N_COLUMNS):
            tag = sensor_id(col)
            sensor_type = SENSOR_TYPE_MAP[col]
            value = 50.0 if tag == stuck_sensor else 50.0 + col + 0.01 * step
            rows.append(
                {
                    "reading_id": str(reading_uuid(PLANT_ID, fault_type, tag, timestamp)),
                    "timestamp": timestamp,
                    "timestep": step,
                    "plant_id": PLANT_ID,
                    "sensor_id": tag,
                    "sensor_type": sensor_type.value,
                    "sensor_location": LOCATION_MAP[col].value,
                    "elevation_m": float((col % 55) * 2 - 10),
                    "value": value,
                    "unit": UNIT_MAP[sensor_type].value,
                    "quality": "good",
                    "raw_counts": 1000 + col,
                    "fault_type": fault_type,
                    "is_usable": True,
                }
            )
    return pd.DataFrame(rows)


def write_partition(root: Path, fault_type: int, frame: pd.DataFrame) -> Path:
    """Write a frame as `<root>/fault_type=NN/readings.parquet`.

    Args:
        root: Root of the partitioned tree.
        fault_type: Fault identifier naming the partition.
        frame: Frame to store.

    Returns:
        The path of the parquet file.
    """
    directory = root / f"fault_type={fault_type:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "readings.parquet"
    frame.to_parquet(path, index=False)
    return path
