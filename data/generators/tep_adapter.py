"""Convert Tennessee Eastman Process (TEP) dataset rows to SensorReading schema objects.

The TEP dataset has 52 columns per sample:
    Columns 0-21  (XMEAS 1-22):  process measurements (flow, pressure, temperature)
    Columns 22-40 (XMEAS 23-41): composition/analysis (normalised mol fractions)
    Columns 41-51 (XMV 1-11):    manipulated variables (actuator positions)

Sensor identifiers follow the convention:
    TEP-XMEAS-01 through TEP-XMEAS-41  (columns 0-40)
    TEP-XMV-01   through TEP-XMV-11    (columns 41-51)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from data.generators.tep_loader import N_COLUMNS, load_dat_file
from data.generators.tep_params import TEPParams
from data.schemas.sensor_reading import (
    Measurement,
    MeasurementUnit,
    QualityFlag,
    SensorInfo,
    SensorLocation,
    SensorMetadata,
    SensorReading,
    SensorType,
)
from data.schemas.sensor_spans import SensorSpan

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reexportado desde tep_loader, que es donde vive la validacion de forma.
_N_COLUMNS = N_COLUMNS
_SAMPLE_INTERVAL = timedelta(minutes=3)
_PLANT_ID = "TEP-PLANT-01"
_DEFAULT_START_TIME = datetime(2000, 1, 1, 0, 0, 0, tzinfo=UTC)
_CALIBRATION_DATE = date(2023, 6, 1)
_LAST_MAINTENANCE_DATE = date(2023, 12, 1)
_DRIFT_COEFFICIENT = 0.0001

_FILE_NAMES: list[str] = ["d00.dat"] + [f"d{i:02d}.dat" for i in range(1, 22)]

# ---------------------------------------------------------------------------
# Static column mappings
# ---------------------------------------------------------------------------


def _build_sensor_type_map() -> dict[int, SensorType]:
    """Return a mapping from zero-based column index to SensorType.

    XMEAS 1-22 (columns 0-21): specific process measurement types.
    XMEAS 23-41 (columns 22-40): composition analysis (NORMALIZED).
    XMV 1-11 (columns 41-51): actuator positions (POSITION).
    """
    mapping: dict[int, SensorType] = {
        0: SensorType.FLOW,           # XMEAS 1:  A feed flow
        1: SensorType.FLOW,           # XMEAS 2:  D feed flow
        2: SensorType.FLOW,           # XMEAS 3:  E feed flow
        3: SensorType.FLOW,           # XMEAS 4:  A and C feed flow
        4: SensorType.FLOW,           # XMEAS 5:  Recycle flow
        5: SensorType.FLOW,           # XMEAS 6:  Reactor feed rate
        6: SensorType.PRESSURE,       # XMEAS 7:  Reactor pressure
        7: SensorType.FLOW,           # XMEAS 8:  Reactor level
        8: SensorType.THERMOCOUPLE,   # XMEAS 9:  Reactor temperature
        9: SensorType.FLOW,           # XMEAS 10: Purge rate
        10: SensorType.THERMOCOUPLE,  # XMEAS 11: Separator temperature
        11: SensorType.FLOW,          # XMEAS 12: Separator level
        12: SensorType.PRESSURE,      # XMEAS 13: Separator pressure
        13: SensorType.FLOW,          # XMEAS 14: Separator underflow
        14: SensorType.FLOW,          # XMEAS 15: Stripper level
        15: SensorType.PRESSURE,      # XMEAS 16: Stripper pressure
        16: SensorType.FLOW,          # XMEAS 17: Stripper underflow
        17: SensorType.THERMOCOUPLE,  # XMEAS 18: Stripper temperature
        18: SensorType.FLOW,          # XMEAS 19: Stripper steam flow
        19: SensorType.FLOW,          # XMEAS 20: Compressor work
        20: SensorType.THERMOCOUPLE,  # XMEAS 21: Reactor CW outlet temperature
        21: SensorType.THERMOCOUPLE,  # XMEAS 22: Separator CW outlet temperature
    }
    for col in range(22, 41):
        mapping[col] = SensorType.NORMALIZED
    for col in range(41, 52):
        mapping[col] = SensorType.POSITION
    return mapping


def _build_location_map() -> dict[int, SensorLocation]:
    """Return a mapping from zero-based column index to SensorLocation.

    Columns 0-9   (reactor measurements):    PRIMARY_LOOP
    Columns 10-21 (separator/stripper):      SECONDARY_LOOP
    Columns 22-40 (composition analysis):    CONTAINMENT
    Columns 41-51 (manipulated variables):   CORE
    """
    mapping: dict[int, SensorLocation] = {}
    for col in range(0, 10):
        mapping[col] = SensorLocation.PRIMARY_LOOP
    for col in range(10, 22):
        mapping[col] = SensorLocation.SECONDARY_LOOP
    for col in range(22, 41):
        mapping[col] = SensorLocation.CONTAINMENT
    for col in range(41, 52):
        mapping[col] = SensorLocation.CORE
    return mapping


# Publicos: son el contrato de identidad de las 52 columnas del TEP, compartido
# con data/generators/derive_sensor_spans.py, que necesita el tipo y la unidad de
# cada canal para etiquetar los spans que deriva.
SENSOR_TYPE_MAP: dict[int, SensorType] = _build_sensor_type_map()
LOCATION_MAP: dict[int, SensorLocation] = _build_location_map()

UNIT_MAP: dict[SensorType, MeasurementUnit] = {
    SensorType.THERMOCOUPLE: MeasurementUnit.CELSIUS,
    SensorType.PRESSURE: MeasurementUnit.BAR,
    SensorType.FLOW: MeasurementUnit.KG_S,
    SensorType.NORMALIZED: MeasurementUnit.NORMALIZED,
    SensorType.POSITION: MeasurementUnit.PERCENT,
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def sensor_id(col_idx: int) -> str:
    """Return the canonical sensor identifier for a given column index.

    Args:
        col_idx: Zero-based TEP column index (0-51).

    Returns:
        Sensor tag string, e.g. "TEP-XMEAS-01" or "TEP-XMV-03".
    """
    if col_idx < 41:
        return f"TEP-XMEAS-{col_idx + 1:02d}"
    return f"TEP-XMV-{col_idx - 40:02d}"


def _elevation_m(col_idx: int) -> float:
    """Return a synthetic elevation for a column, in range [-10.0, 100.0] m.

    Elevation is assigned deterministically based on column index, simulating
    sensors distributed across plant floors.

    Args:
        col_idx: Zero-based TEP column index (0-51).

    Returns:
        Elevation value in metres within [-10.0, 100.0].
    """
    return float((col_idx % 55) * 2 - 10)


def _to_raw_counts(value: float, span: SensorSpan) -> int:
    """Scale a TEP process value to a 16-bit ADC count using its calibrated span.

    El escalado es (value - min) / (max - min) * 65535 con clamp en ambos
    extremos. Un denominador global comun a los 52 canales saturaba el 5,86% de
    las lecturas en XMEAS-02, -03, -07 y -16, y a la vez dejaba las variables de
    composicion ocupando el 1% del rango; el span por transmisor resuelve ambos
    extremos a la vez.

    Args:
        value: TEP process value in engineering units.
        span: Calibrated span of the sensor this value belongs to.

    Returns:
        Integer ADC count in [0, 65535].
    """
    return span.to_raw_counts(value)


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------


def save_to_parquet(df: pd.DataFrame, output_dir: str) -> None:
    """Save a TEPAdapter DataFrame to fault_type-partitioned Parquet files.

    Produces one file per fault type under the directory structure:
        output_dir/fault_type=00/readings.parquet
        output_dir/fault_type=01/readings.parquet
        ...

    Args:
        df: DataFrame produced by TEPAdapter.adapt_all, which must contain a
            "fault_type" integer column.
        output_dir: Root directory for the partitioned output. Created if it
            does not exist.
    """
    out_path = Path(output_dir)
    for fault_type in sorted(df["fault_type"].unique()):
        subset = df[df["fault_type"] == fault_type].copy()
        partition_dir = out_path / f"fault_type={fault_type:02d}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = partition_dir / "readings.parquet"
        subset.to_parquet(parquet_path, index=False)
        _LOG.info(
            "Saved %d readings to %s", len(subset), parquet_path
        )


def readings_from_frame(
    frame: pd.DataFrame,
    calibration_date: date = _CALIBRATION_DATE,
    last_maintenance: date = _LAST_MAINTENANCE_DATE,
    drift_coefficient: float = _DRIFT_COEFFICIENT,
) -> Iterator[SensorReading]:
    """Rebuild SensorReading objects from the long-format frame.

    Inverso de TEPAdapter.adapt_all, y vive aqui porque es este modulo el que
    define ese formato largo. Lo necesita todo consumidor que trabaje desde el
    parquet en lugar de desde los .dat: el validador opera sobre el contrato
    SensorReading, no sobre filas de un DataFrame.

    Las lecturas se emiten en el orden de las filas del frame. El orden importa:
    los detectores del validador tienen estado por sensor y lo construyen segun
    van llegando las lecturas, de modo que reordenar el frame cambia lo que ven.

    Los tres campos de SensorMetadata no viajan en el formato largo, porque son
    constantes de la corrida y no propiedades de la lectura. Se reinyectan desde
    los argumentos, cuyos valores por defecto son los mismos que usa el adaptador.
    Ningun detector los lee.

    Args:
        frame: Long-format frame with the columns produced by adapt_all.
        calibration_date: Calibration date stamped on every reading.
        last_maintenance: Maintenance date stamped on every reading.
        drift_coefficient: Drift coefficient stamped on every reading.

    Yields:
        One validated SensorReading per row, in row order.

    Raises:
        KeyError: If the frame is missing a column the schema requires.
        pydantic.ValidationError: If a row does not satisfy the contract.
    """
    required = {
        "reading_id",
        "timestamp",
        "plant_id",
        "sensor_id",
        "sensor_type",
        "sensor_location",
        "elevation_m",
        "value",
        "unit",
        "quality",
        "raw_counts",
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"Frame is missing columns required to rebuild readings: {sorted(missing)}."
        )

    metadata = SensorMetadata(
        calibration_date=calibration_date,
        last_maintenance=last_maintenance,
        drift_coefficient=drift_coefficient,
    )

    for row in frame.itertuples(index=False):
        yield SensorReading(
            reading_id=uuid.UUID(str(row.reading_id)),
            timestamp=pd.Timestamp(row.timestamp).to_pydatetime(),
            plant_id=str(row.plant_id),
            sensor=SensorInfo(
                id=str(row.sensor_id),
                type=SensorType(row.sensor_type),
                location=SensorLocation(row.sensor_location),
                elevation_m=float(row.elevation_m),
            ),
            measurement=Measurement(
                value=None if pd.isna(row.value) else float(row.value),
                unit=MeasurementUnit(row.unit),
                quality=QualityFlag(row.quality),
                raw_counts=int(row.raw_counts),
            ),
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------


class TEPAdapter:
    """Convert rows of a TEP .dat file to validated SensorReading objects.

    Each sample row in the TEP produces 52 SensorReading records, one per
    process variable. Timestamps are generated synthetically at 3-minute
    intervals starting from a configurable start time.

    Todos los valores configurables llegan por constructor para que los
    parametros declarados en params.yaml sean efectivos; los valores por defecto
    reproducen el comportamiento anterior a la parametrizacion.

    Attributes:
        spans: Calibrated span per sensor_id, used to scale raw_counts.
        start_time: UTC datetime used as the origin for synthetic timestamps.
        sample_interval: Spacing between consecutive samples.
        plant_id: Plant identifier stamped on every reading.
        calibration_date: Calibration date recorded in SensorMetadata.
        last_maintenance: Maintenance date recorded in SensorMetadata.
        drift_coefficient: Drift coefficient recorded in SensorMetadata.
    """

    def __init__(
        self,
        spans: dict[str, SensorSpan],
        start_time: datetime = _DEFAULT_START_TIME,
        sample_interval: timedelta = _SAMPLE_INTERVAL,
        plant_id: str = _PLANT_ID,
        calibration_date: date = _CALIBRATION_DATE,
        last_maintenance: date = _LAST_MAINTENANCE_DATE,
        drift_coefficient: float = _DRIFT_COEFFICIENT,
    ) -> None:
        """Initialise the adapter.

        Args:
            spans: Calibrated span per sensor_id, as loaded from
                configs/sensor_spans.yaml. Must cover all 52 TEP tags.
            start_time: UTC datetime for the first sample. Subsequent samples
                are offset by multiples of sample_interval.
            sample_interval: Spacing between consecutive TEP samples.
            plant_id: Plant identifier stamped on every reading.
            calibration_date: Calibration date recorded in SensorMetadata.
            last_maintenance: Maintenance date recorded in SensorMetadata.
            drift_coefficient: Drift coefficient recorded in SensorMetadata.

        Raises:
            KeyError: If spans does not cover every TEP sensor tag.
        """
        missing = [
            sensor_id(col) for col in range(_N_COLUMNS) if sensor_id(col) not in spans
        ]
        if missing:
            raise KeyError(
                f"Sensor span table is missing {len(missing)} TEP tags: "
                f"{', '.join(missing[:5])}"
                f"{'...' if len(missing) > 5 else ''}. "
                "Regenerate it with data/generators/derive_sensor_spans.py."
            )

        self.spans = spans
        self.start_time = start_time
        self.sample_interval = sample_interval
        self.plant_id = plant_id
        self.calibration_date = calibration_date
        self.last_maintenance = last_maintenance
        self.drift_coefficient = drift_coefficient

    @classmethod
    def from_params(cls, params: TEPParams, spans: dict[str, SensorSpan]) -> TEPAdapter:
        """Build an adapter from the tep: section of params.yaml.

        Args:
            params: Resolved parameters.
            spans: Calibrated span per sensor_id.

        Returns:
            A TEPAdapter configured from params.

        Raises:
            KeyError: If spans does not cover every TEP sensor tag.
        """
        return cls(
            spans=spans,
            start_time=params.start_time,
            sample_interval=timedelta(minutes=params.sample_interval_minutes),
            plant_id=params.plant_id,
            calibration_date=params.calibration_date,
            last_maintenance=params.last_maintenance_date,
            drift_coefficient=params.drift_coefficient,
        )

    def _row_to_readings(
        self,
        row: list[float],
        row_index: int,
    ) -> list[SensorReading]:
        """Convert a single TEP data row to a list of 52 SensorReading objects.

        Args:
            row: List of 52 float values for a single timestep.
            row_index: Zero-based row index used to compute the timestamp.

        Returns:
            List of 52 SensorReading instances, one per TEP variable.
        """
        timestamp = self.start_time + row_index * self.sample_interval
        readings: list[SensorReading] = []

        for col_idx, value in enumerate(row):
            sensor_type = SENSOR_TYPE_MAP[col_idx]
            tag = sensor_id(col_idx)
            reading = SensorReading(
                reading_id=uuid.uuid4(),
                timestamp=timestamp,
                plant_id=self.plant_id,
                sensor=SensorInfo(
                    id=tag,
                    type=sensor_type,
                    location=LOCATION_MAP[col_idx],
                    elevation_m=_elevation_m(col_idx),
                ),
                measurement=Measurement(
                    value=float(value),
                    unit=UNIT_MAP[sensor_type],
                    quality=QualityFlag.GOOD,
                    raw_counts=_to_raw_counts(float(value), self.spans[tag]),
                ),
                metadata=SensorMetadata(
                    calibration_date=self.calibration_date,
                    last_maintenance=self.last_maintenance,
                    drift_coefficient=self.drift_coefficient,
                ),
            )
            readings.append(reading)

        return readings

    def adapt_file(self, filepath: str, fault_type: int) -> list[SensorReading]:
        """Adapt all rows in a TEP .dat file to SensorReading objects.

        Todas las lecturas se emiten con QualityFlag.GOOD, incluidas las de los
        ficheros de fallo. QualityFlag describe la fiabilidad del transmisor;
        los fallos del TEP son perturbaciones de proceso, que se registran en
        fault_type. Son campos ortogonales: marcar SUSPECT por fault_type>=1
        confundia "la planta esta perturbada" con "el instrumento no es fiable",
        y convertia cualquier evaluacion del validador contra fault_type en una
        tautologia. El dataset no aporta informacion de salud de instrumento, de
        modo que GOOD es la unica lectura honesta del campo.

        Args:
            filepath: Path to a TEP .dat file (whitespace-delimited, no header).
            fault_type: Integer fault identifier (0 = normal, 1-21 = fault).
                Recorded on the output rows; it does not affect quality.

        Returns:
            Flat list of SensorReading objects. For a file with N rows the list
            contains N * 52 elements ordered by (row_index, col_index).

        Raises:
            ValueError: If the file does not contain exactly 52 columns.
            FileNotFoundError: If filepath does not exist.
        """
        file_path = Path(filepath)
        df = load_dat_file(file_path)

        all_readings: list[SensorReading] = []
        for row_index, row_values in enumerate(df.itertuples(index=False, name=None)):
            all_readings.extend(self._row_to_readings(list(row_values), row_index))

        _LOG.info(
            "Adapted %d rows x %d columns = %d readings from %s (fault_type=%d).",
            df.shape[0],
            _N_COLUMNS,
            len(all_readings),
            file_path.name,
            fault_type,
        )
        return all_readings

    def adapt_all(self, data_dir: str) -> pd.DataFrame:
        """Adapt all TEP files in data_dir and return a consolidated DataFrame.

        Each row in the returned DataFrame corresponds to one SensorReading
        (one variable at one timestep). Files that do not exist are skipped
        with a warning.

        Args:
            data_dir: Directory containing d00.dat through d21.dat.

        Returns:
            DataFrame with columns:
                reading_id, timestamp, timestep, plant_id, sensor_id,
                sensor_type, sensor_location, elevation_m, value, unit,
                quality, raw_counts, fault_type, is_usable.

            timestep is the zero-based sample index within its source file. It
            is redundant with timestamp inside one file, but every file restarts
            the clock at start_time, so (fault_type, timestep, sensor_id) is the
            primary key of the long-format table and the index the validation
            confusion matrix and the wide-format pivot are built on.
        """
        data_path = Path(data_dir)
        records: list[dict[str, object]] = []

        for fault_type, filename in enumerate(_FILE_NAMES):
            filepath = data_path / filename
            if not filepath.exists():
                _LOG.warning("File not found, skipping: %s", filepath)
                continue

            _LOG.info("Adapting %s (fault_type=%d).", filename, fault_type)
            readings = self.adapt_file(str(filepath), fault_type)

            for offset, reading in enumerate(readings):
                records.append(
                    {
                        "reading_id": str(reading.reading_id),
                        "timestamp": reading.timestamp,
                        "timestep": offset // _N_COLUMNS,
                        "plant_id": reading.plant_id,
                        "sensor_id": reading.sensor.id,
                        "sensor_type": reading.sensor.type.value,
                        "sensor_location": reading.sensor.location.value,
                        "elevation_m": reading.sensor.elevation_m,
                        "value": reading.measurement.value,
                        "unit": reading.measurement.unit.value,
                        "quality": reading.measurement.quality.value,
                        "raw_counts": reading.measurement.raw_counts,
                        "fault_type": fault_type,
                        "is_usable": reading.is_usable,
                    }
                )

        _LOG.info("adapt_all complete: %d total readings.", len(records))
        return pd.DataFrame(records)

# El punto de entrada ejecutable vive en data/generators/adapt_tep.py, que lee
# las rutas y la configuracion de params.yaml. Este modulo queda como libreria.
