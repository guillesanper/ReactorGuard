"""Unit tests for data.generators.adapt_tep and TEPAdapter.adapt_all.

Cubren la ruta que ejecuta el stage adapt_tep de dvc.yaml: resolver params,
recorrer el directorio crudo, construir el DataFrame consolidado y escribir el
Parquet particionado por fault_type.

Un test verifica explicitamente que los parametros de params.yaml llegan al
resultado: es la propiedad que justifica declarar `params:` en dvc.yaml, y sin
ella la invalidacion de stages seria decorativa.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data.generators.adapt_tep import main, summarise
from data.generators.tep_adapter import TEPAdapter, save_to_parquet
from data.generators.tep_loader import N_COLUMNS
from data.generators.tep_params import load_tep_params
from data.schemas.sensor_spans import SensorSpan, load_sensor_spans

from .conftest import WriteParams

Spans = dict[str, SensorSpan]

_EXPECTED_COLUMNS = {
    "reading_id",
    "timestamp",
    "timestep",
    "plant_id",
    "sensor_id",
    "sensor_type",
    "sensor_location",
    "elevation_m",
    "value",
    "unit",
    "quality",
    "raw_counts",
    "fault_type",
    "is_usable",
}


class TestAdaptAll:
    """adapt_all must consolidate every file in the directory into one frame."""

    def test_row_count(
        self, make_dat_dir: Callable[..., Path], sensor_spans: Spans
    ) -> None:
        """Rows must equal n_files * n_rows * 52."""
        data_dir = make_dat_dir(n_files=3, n_rows=4)
        df = TEPAdapter(sensor_spans).adapt_all(str(data_dir))
        assert len(df) == 3 * 4 * N_COLUMNS

    def test_columns(
        self, make_dat_dir: Callable[..., Path], sensor_spans: Spans
    ) -> None:
        """The frame must expose the documented column set."""
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=2, n_rows=3)))
        assert set(df.columns) == _EXPECTED_COLUMNS

    def test_fault_types_are_indexed_by_filename(
        self, make_dat_dir: Callable[..., Path], sensor_spans: Spans
    ) -> None:
        """d00 must map to fault_type 0 and d01 to 1."""
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=3, n_rows=3)))
        assert sorted(df.fault_type.unique()) == [0, 1, 2]

    def test_transposed_d00_yields_full_row_count(
        self, make_dat_dir: Callable[..., Path], sensor_spans: Spans
    ) -> None:
        """The transposed d00 must contribute n_rows * 52 readings, not 52 * 52."""
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=2, n_rows=7)))
        assert len(df[df.fault_type == 0]) == 7 * N_COLUMNS

    def test_quality_is_good_regardless_of_fault_type(
        self, make_dat_dir: Callable[..., Path], sensor_spans: Spans
    ) -> None:
        """quality must be independent of fault_type.

        QualityFlag describe la fiabilidad del transmisor y fault_type la
        perturbacion del proceso. Son ortogonales: derivar quality de fault_type
        haria que cualquier evaluacion del validador contra fault_type midiese
        una tautologia.
        """
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=3, n_rows=3)))
        assert set(df.quality) == {"good"}
        assert df.is_usable.all()

    def test_timestep_restarts_per_file(
        self, make_dat_dir: Callable[..., Path], sensor_spans: Spans
    ) -> None:
        """timestep must be the zero-based sample index within each source file."""
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=2, n_rows=4)))
        for fault_type in (0, 1):
            subset = df[df.fault_type == fault_type]
            assert sorted(subset.timestep.unique()) == [0, 1, 2, 3]
            assert len(subset[subset.timestep == 0]) == N_COLUMNS

    def test_missing_files_are_skipped(
        self, tmp_path: Path, sensor_spans: Spans
    ) -> None:
        """A directory with no TEP files must produce an empty frame."""
        empty = tmp_path / "empty"
        empty.mkdir()
        assert TEPAdapter(sensor_spans).adapt_all(str(empty)).empty


class TestSaveToParquet:
    """save_to_parquet must produce one partition directory per fault type."""

    def test_creates_one_partition_per_fault_type(
        self, make_dat_dir: Callable[..., Path], tmp_path: Path,
        sensor_spans: Spans
    ) -> None:
        """Partition directories must be zero-padded fault_type=NN."""
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=3, n_rows=3)))
        out = tmp_path / "processed"
        save_to_parquet(df, str(out))
        assert sorted(p.name for p in out.iterdir()) == [
            "fault_type=00",
            "fault_type=01",
            "fault_type=02",
        ]

    def test_partition_contents_round_trip(
        self, make_dat_dir: Callable[..., Path], tmp_path: Path,
        sensor_spans: Spans
    ) -> None:
        """A partition must read back with only its own fault_type."""
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=2, n_rows=3)))
        out = tmp_path / "processed"
        save_to_parquet(df, str(out))
        part = pd.read_parquet(out / "fault_type=01" / "readings.parquet")
        assert set(part.fault_type) == {1}
        assert len(part) == 3 * N_COLUMNS


class TestSummarise:
    """summarise must count readings per fault type."""

    def test_counts_per_fault_type(
        self, make_dat_dir: Callable[..., Path], sensor_spans: Spans
    ) -> None:
        """Each fault type must map to its reading count."""
        df = TEPAdapter(sensor_spans).adapt_all(str(make_dat_dir(n_files=2, n_rows=3)))
        assert summarise(df) == {0: 3 * N_COLUMNS, 1: 3 * N_COLUMNS}

    def test_empty_frame_yields_empty_mapping(self) -> None:
        """An empty frame must summarise to an empty dict, not raise."""
        assert summarise(pd.DataFrame(columns=["fault_type"])) == {}


class TestMainEntryPoint:
    """main() must drive the whole stage from params.yaml alone."""

    def test_writes_partitions_to_processed_dir(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams,
        tmp_path: Path
    ) -> None:
        """Parquet output must land under the configured processed_dir."""
        data_dir = make_dat_dir(n_files=2, n_rows=3)
        processed = tmp_path / "configured_processed"
        main(write_params(raw_dir=str(data_dir), processed_dir=str(processed)))
        assert (processed / "fault_type=00" / "readings.parquet").exists()

    def test_returns_the_written_frame(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams
    ) -> None:
        """main must return the consolidated frame it persisted."""
        data_dir = make_dat_dir(n_files=2, n_rows=3)
        df = main(write_params(raw_dir=str(data_dir)))
        assert len(df) == 2 * 3 * N_COLUMNS

    def test_empty_raw_dir_raises(
        self, write_params: WriteParams, tmp_path: Path
    ) -> None:
        """An unpopulated raw_dir must fail with a message pointing at download."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="download_tep"):
            main(write_params(raw_dir=str(empty)))


class TestParamsReachTheOutput:
    """Values from params.yaml must be observable in the adapted readings.

    Sin esta propiedad, declarar `params:` en dvc.yaml invalidaria stages ante
    cambios que no alteran el resultado: reproducibilidad aparente.
    """

    def test_plant_id_is_applied(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams
    ) -> None:
        """A custom plant_id must appear on every reading."""
        data_dir = make_dat_dir(n_files=2, n_rows=3)
        df = main(write_params(raw_dir=str(data_dir), plant_id="TEP-PLANT-99"))
        assert set(df.plant_id) == {"TEP-PLANT-99"}

    def test_sample_interval_is_applied(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams
    ) -> None:
        """A custom sampling interval must set the timestamp spacing."""
        data_dir = make_dat_dir(n_files=2, n_rows=3)
        df = main(
            write_params(raw_dir=str(data_dir), sample_interval_minutes=10)
        )
        stamps = sorted(df[df.sensor_id == "TEP-XMEAS-01"].timestamp.unique())
        assert stamps[1] - stamps[0] == timedelta(minutes=10)

    def test_narrowing_the_span_changes_raw_counts(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams,
        sensor_spans: Spans, tmp_path: Path
    ) -> None:
        """A narrower calibrated span must push the same values to higher counts."""
        import yaml

        data_dir = make_dat_dir(n_files=2, n_rows=3)
        wide = main(write_params(raw_dir=str(data_dir)))

        narrow_path = tmp_path / "narrow_spans.yaml"
        narrow_path.write_text(
            yaml.safe_dump(
                {
                    "sensors": {
                        tag: {
                            "min": -200.0,
                            "max": 200.0,
                            "alarm_min": -100.0,
                            "alarm_max": 100.0,
                            "unit": span.unit,
                        }
                        for tag, span in sensor_spans.items()
                    }
                }
            ),
            encoding="utf-8",
        )
        narrow = main(
            write_params(raw_dir=str(data_dir), spans_path=str(narrow_path))
        )
        assert narrow.raw_counts.sum() > wide.raw_counts.sum()

    def test_missing_span_for_a_tag_raises(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams,
        sensor_spans: Spans, tmp_path: Path
    ) -> None:
        """An incomplete span table must fail loudly, not silently skip a sensor."""
        import yaml

        partial = dict(sensor_spans)
        partial.pop("TEP-XMV-11")
        partial_path = tmp_path / "partial_spans.yaml"
        partial_path.write_text(
            yaml.safe_dump(
                {
                    "sensors": {
                        tag: {
                            "min": span.min,
                            "max": span.max,
                            "alarm_min": span.alarm_min,
                            "alarm_max": span.alarm_max,
                            "unit": span.unit,
                        }
                        for tag, span in partial.items()
                    }
                }
            ),
            encoding="utf-8",
        )
        data_dir = make_dat_dir(n_files=2, n_rows=3)
        with pytest.raises(KeyError, match="TEP-XMV-11"):
            main(write_params(raw_dir=str(data_dir), spans_path=str(partial_path)))

    def test_start_time_is_applied(
        self, make_dat_dir: Callable[..., Path], write_params: WriteParams
    ) -> None:
        """A custom start_time must become the first timestamp."""
        data_dir = make_dat_dir(n_files=2, n_rows=3)
        df = main(
            write_params(
                raw_dir=str(data_dir), start_time="2020-05-04T12:00:00+00:00"
            )
        )
        assert min(df.timestamp).isoformat().startswith("2020-05-04T12:00:00")


class TestFromParams:
    """TEPAdapter.from_params must map every parameter onto the adapter."""

    def test_all_fields_are_mapped(
        self, write_params: WriteParams, sensor_spans: Spans
    ) -> None:
        """Each tep: key must reach its corresponding adapter attribute."""
        params = load_tep_params(
            write_params(
                plant_id="TEP-PLANT-77",
                sample_interval_minutes=15,
                calibration_date="2021-01-02",
                last_maintenance_date="2021-03-04",
                drift_coefficient=0.005,
            )
        )
        adapter = TEPAdapter.from_params(params, sensor_spans)

        assert adapter.plant_id == "TEP-PLANT-77"
        assert adapter.sample_interval == timedelta(minutes=15)
        assert adapter.calibration_date == date(2021, 1, 2)
        assert adapter.last_maintenance == date(2021, 3, 4)
        assert adapter.drift_coefficient == pytest.approx(0.005)

    def test_spans_path_is_loaded(self, write_params: WriteParams) -> None:
        """The spans_path parameter must resolve to a full 52-tag span table."""
        params = load_tep_params(write_params())
        spans = load_sensor_spans(params.spans_path)
        assert len(spans) == N_COLUMNS
        assert spans["TEP-XMEAS-01"].alarm_min > spans["TEP-XMEAS-01"].min
