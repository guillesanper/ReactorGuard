"""Tests for data/generators/tep_streamer_params.py."""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.generators.tep_streamer_params import (
    ENV_DATA_SOURCE,
    ENV_SPEED_MULTIPLIER,
    ENV_STREAM_MODE,
    DataSource,
    StreamerParams,
    StreamMode,
    apply_env_overrides,
    load_streamer_params,
    validate_against_streaming,
)
from data.streaming.streaming_params import load_streaming_params

_REAL_PARAMS = Path("params.yaml")
WriteDoc = Callable[[dict[str, Any]], Path]


@pytest.fixture()
def document() -> dict[str, Any]:
    """Return the real params.yaml as a mutable dict."""
    loaded: dict[str, Any] = yaml.safe_load(_REAL_PARAMS.read_text(encoding="utf-8"))
    return copy.deepcopy(loaded)


@pytest.fixture()
def write_doc(tmp_path: Path) -> WriteDoc:
    """Return a function that writes a params document into tmp_path."""

    def _write(doc: dict[str, Any]) -> Path:
        path = tmp_path / "params.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture()
def params() -> StreamerParams:
    """Return the streamer parameters of the repository."""
    return load_streamer_params(_REAL_PARAMS)


class TestLoading:
    def test_real_params_resolve(self, params: StreamerParams) -> None:
        assert params.mode is StreamMode.FAST
        assert params.speed_multiplier == 1.0
        assert params.loop is False
        assert params.data_source is DataSource.PARQUET
        assert params.storage_prefix == "tep"
        assert params.client_id == "tep-streamer"

    def test_interval_comes_from_the_tep_section_not_a_literal(
        self, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        assert load_streamer_params(_REAL_PARAMS).sample_interval_s == 180.0
        document["tep"]["sample_interval_minutes"] = 5
        assert load_streamer_params(write_doc(document)).sample_interval_s == 300.0

    def test_directory_and_metadata_come_from_the_tep_section(
        self, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        document["tep"]["processed_dir"] = "elsewhere/tep"
        document["tep"]["drift_coefficient"] = 0.5
        loaded = load_streamer_params(write_doc(document))
        assert loaded.parquet_dir == Path("elsewhere/tep")
        assert loaded.drift_coefficient == 0.5
        assert loaded.calibration_date.isoformat() == "2023-06-01"
        assert loaded.last_maintenance.isoformat() == "2023-12-01"

    def test_step_interval_divides_by_speed(
        self, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        document["streamer"]["speed_multiplier"] = 60
        assert load_streamer_params(write_doc(document)).step_interval_s == 3.0

    def test_storage_prefix_is_stripped_of_slashes(
        self, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        document["streamer"]["storage_prefix"] = "/tep/"
        assert load_streamer_params(write_doc(document)).storage_prefix == "tep"

    def test_enum_values_are_case_insensitive(
        self, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        document["streamer"]["mode"] = "REALTIME"
        document["streamer"]["data_source"] = "GCS"
        loaded = load_streamer_params(write_doc(document))
        assert loaded.mode is StreamMode.REALTIME
        assert loaded.data_source is DataSource.GCS

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_streamer_params(tmp_path / "absent.yaml")

    def test_missing_section_raises(self, document: dict[str, Any], write_doc: WriteDoc) -> None:
        del document["streamer"]
        with pytest.raises(KeyError, match="streamer"):
            load_streamer_params(write_doc(document))

    @pytest.mark.parametrize(
        "key",
        [
            "mode",
            "speed_multiplier",
            "loop",
            "data_source",
            "storage_prefix",
            "flush_every_timesteps",
            "wait_slice_s",
            "client_id",
        ],
    )
    def test_missing_key_raises(
        self, key: str, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        del document["streamer"][key]
        with pytest.raises(KeyError, match=f"streamer.{key}"):
            load_streamer_params(write_doc(document))


class TestValidation:
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("mode", "turbo"),
            ("data_source", "ftp"),
            ("speed_multiplier", 0),
            ("speed_multiplier", -2),
            ("speed_multiplier", "fast"),
            ("speed_multiplier", float("nan")),
            ("speed_multiplier", float("inf")),
            ("loop", "false"),
            ("loop", 0),
            ("flush_every_timesteps", 0),
            ("flush_every_timesteps", -5),
            ("flush_every_timesteps", True),
            ("flush_every_timesteps", 1.5),
            ("wait_slice_s", 0),
            ("wait_slice_s", -1.0),
            ("client_id", "  "),
        ],
    )
    def test_invalid_value_raises(
        self, key: str, value: object, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        document["streamer"][key] = value
        with pytest.raises(ValueError, match=key):
            load_streamer_params(write_doc(document))

    def test_non_positive_tep_interval_raises(
        self, document: dict[str, Any], write_doc: WriteDoc
    ) -> None:
        document["tep"]["sample_interval_minutes"] = 0
        with pytest.raises(ValueError, match="sample_interval_minutes"):
            load_streamer_params(write_doc(document))


class TestEnvOverrides:
    def test_no_variables_changes_nothing(self, params: StreamerParams) -> None:
        assert apply_env_overrides(params, {}) == params

    def test_each_variable_overrides_its_field(self, params: StreamerParams) -> None:
        result = apply_env_overrides(
            params,
            {
                ENV_STREAM_MODE: "realtime",
                ENV_SPEED_MULTIPLIER: "30",
                ENV_DATA_SOURCE: "gcs",
            },
        )
        assert result.mode is StreamMode.REALTIME
        assert result.speed_multiplier == 30.0
        assert result.data_source is DataSource.GCS
        assert result.client_id == params.client_id

    def test_values_are_case_and_space_insensitive(self, params: StreamerParams) -> None:
        result = apply_env_overrides(params, {ENV_STREAM_MODE: "  RealTime "})
        assert result.mode is StreamMode.REALTIME

    def test_blank_variables_count_as_unset(self, params: StreamerParams) -> None:
        env = {ENV_STREAM_MODE: " ", ENV_SPEED_MULTIPLIER: "", ENV_DATA_SOURCE: "  "}
        assert apply_env_overrides(params, env) == params

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            (ENV_STREAM_MODE, "warp"),
            (ENV_SPEED_MULTIPLIER, "0"),
            (ENV_SPEED_MULTIPLIER, "-1"),
            (ENV_SPEED_MULTIPLIER, "nan"),
            (ENV_SPEED_MULTIPLIER, "quick"),
            (ENV_DATA_SOURCE, "s3"),
        ],
    )
    def test_invalid_variable_raises_naming_it(
        self, name: str, value: str, params: StreamerParams
    ) -> None:
        with pytest.raises(ValueError, match=name):
            apply_env_overrides(params, {name: value})

    def test_reads_os_environ_by_default(
        self, params: StreamerParams, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_STREAM_MODE, "realtime")
        assert apply_env_overrides(params).mode is StreamMode.REALTIME


class TestAgainstStreaming:
    def test_real_values_leave_room_for_heartbeats(self, params: StreamerParams) -> None:
        streaming = load_streaming_params()
        validate_against_streaming(params, streaming)
        assert params.wait_slice_s * 2 <= streaming.staleness_seconds

    def test_a_slice_longer_than_half_the_staleness_is_rejected(
        self, params: StreamerParams
    ) -> None:
        streaming = load_streaming_params()
        too_long = StreamerParams(
            **{**params.__dict__, "wait_slice_s": streaming.staleness_seconds}
        )
        with pytest.raises(ValueError, match="heartbeats"):
            validate_against_streaming(too_long, streaming)
