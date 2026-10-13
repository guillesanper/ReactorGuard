"""Unit tests for data.simulation.dataset_builder.

Cubren la cadena observables -> ruido -> cuentas ADC -> filas: orden y tipos de
las columnas, identificadores deterministas con run_id, escalado con el span, y
que el frame se reconstruya como SensorReading con el mismo codigo del TEP (H12).
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.generators.tep_adapter import READING_ID_NAMESPACE, reading_uuid, readings_from_frame
from data.schemas.sensor_reading import QualityFlag, SensorReading
from data.schemas.sensor_spans import SensorSpan, load_sensor_spans
from data.simulation.dataset_builder import (
    CONTRACT_COLUMNS,
    LABEL_COLUMNS,
    LONG_COLUMNS,
    NO_FAULT,
    RunSimulation,
    assemble_long_frame,
    build_run_frame,
    measure,
    noise_sigmas,
    raw_counts_matrix,
    run_parquet_path,
    sim_reading_uuid,
    simulate_run,
    wide_frame_from_long,
    wide_frame_from_matrix,
    write_run_parquet,
)
from data.simulation.scenario import (
    FaultLevel,
    ProcessFaultSpec,
    RunPlan,
    ScenarioKind,
    normal_run_plans,
)
from data.simulation.sensor_catalog import SensorCatalog
from data.simulation.simulation_params import SimulationParams, load_simulation_params
from data.simulation.thermal_model import ModelInputs, ReactorModel
from tests.support.sim_frames import (
    SIM_SPANS_PATH,
    build_catalog,
    build_model,
    write_sim_params,
)

PLANT = "SIM-TEST-01"


@pytest.fixture(scope="module")
def model() -> ReactorModel:
    """Return the reference model."""
    return build_model()


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned sensor catalog."""
    return build_catalog()


@pytest.fixture(scope="module")
def spans() -> dict[str, SensorSpan]:
    """Return the versioned simulation span table."""
    return load_sensor_spans(SIM_SPANS_PATH)


@pytest.fixture(scope="module")
def params(tmp_path_factory: pytest.TempPathFactory) -> SimulationParams:
    """Return a small simulation configuration."""
    return load_simulation_params(write_sim_params(tmp_path_factory.mktemp("sim")))


@pytest.fixture(scope="module")
def plan(params: SimulationParams) -> RunPlan:
    """Return the first run's plan."""
    return normal_run_plans(params)[0]


@pytest.fixture(scope="module")
def sim(model: ReactorModel, catalog: SensorCatalog, plan: RunPlan) -> RunSimulation:
    """Return an integrated run with a ramp from 100 % to 80 %."""
    loads = np.concatenate([np.full(10, 1.0), np.linspace(1.0, 0.8, 11)[1:], np.full(10, 0.8)])
    inputs = [ModelInputs(load=float(x)) for x in loads]
    return simulate_run(model, catalog, plan, inputs, 180.0)


@pytest.fixture(scope="module")
def frame(
    sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
) -> pd.DataFrame:
    """Return the long frame of the integrated run with noise."""
    noise_rng = np.random.default_rng(11)
    return build_run_frame(sim, catalog, spans, PLANT, noise_rng)


class TestReadingUuid:
    """The identifier is a pure function of the reading's identity, run included."""

    def test_is_deterministic(self) -> None:
        """The same identity gives the same UUID."""
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        assert sim_reading_uuid(PLANT, "sim_run_001", "TC-HL-1", stamp) == sim_reading_uuid(
            PLANT, "sim_run_001", "TC-HL-1", stamp
        )

    def test_run_id_enters_the_key(self) -> None:
        """Two runs sharing a plant, tag and instant do not collide."""
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        a = sim_reading_uuid(PLANT, "sim_run_001", "TC-HL-1", stamp)
        b = sim_reading_uuid(PLANT, "sim_run_002", "TC-HL-1", stamp)
        assert a != b

    def test_each_component_changes_the_id(self) -> None:
        """Plant, tag and instant all matter."""
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        base = sim_reading_uuid(PLANT, "r", "T", stamp)
        assert sim_reading_uuid("OTHER", "r", "T", stamp) != base
        assert sim_reading_uuid(PLANT, "r", "U", stamp) != base
        assert sim_reading_uuid(PLANT, "r", "T", datetime(2026, 1, 1, 0, 3, tzinfo=UTC)) != base

    def test_shares_the_tep_namespace(self) -> None:
        """The UUID is version 5 under the same namespace as the TEP."""
        value = sim_reading_uuid(PLANT, "r", "T", datetime(2026, 1, 1, tzinfo=UTC))
        assert value.version == 5
        assert value == uuid.uuid5(
            READING_ID_NAMESPACE, f"{PLANT}|r|T|{datetime(2026, 1, 1, tzinfo=UTC).isoformat()}"
        )

    def test_differs_from_a_tep_identifier(self) -> None:
        """With a real run id a simulation identifier cannot equal a TEP one.

        The key of a TEP reading ends its fault type as two digits (`00`..`21`), so a
        run id of the form `sim_run_NNN` keeps the two key spaces disjoint.
        """
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        tep = {reading_uuid(PLANT, ft, "T", stamp) for ft in range(22)}
        assert sim_reading_uuid(PLANT, "sim_run_001", "T", stamp) not in tep


class TestSimulateRun:
    """A run is integrated from the equilibrium of its first load."""

    def test_shapes(self, sim: RunSimulation, catalog: SensorCatalog) -> None:
        """One row of true values per sample and one column per sensor."""
        assert sim.n_steps == 30
        assert sim.true_values.shape == (30, len(catalog.sensors))
        assert sim.states.shape[0] == 30
        assert len(sim.steps) == 29
        assert len(sim.diagnostics) == 30
        assert len(sim.timestamps) == 30

    def test_starts_in_equilibrium(
        self, sim: RunSimulation, model: ReactorModel, catalog: SensorCatalog
    ) -> None:
        """The first sample is the steady state of the first load."""
        expected = catalog.true_values(
            model.observables(model.steady_state(1.0), ModelInputs(load=1.0))
        )
        assert np.allclose(sim.true_values[0], [expected[t] for t in catalog.tags])

    def test_energy_closes(self, sim: RunSimulation) -> None:
        """The interval residual stays at integrator precision."""
        assert sim.max_energy_residual < 1e-9

    def test_load_step_moves_the_flux(self, sim: RunSimulation, catalog: SensorCatalog) -> None:
        """Dropping the load to 80 % lowers the neutron flux channel."""
        j = catalog.tags.index("NI-FLUX-01")
        assert sim.true_values[-1, j] < sim.true_values[0, j] - 10.0

    def test_no_steps_gives_zero_residual(self, sim: RunSimulation) -> None:
        """The residual summary of a run without intervals is zero."""
        assert dataclasses.replace(sim, steps=()).max_energy_residual == 0.0

    def test_length_mismatch(
        self, model: ReactorModel, catalog: SensorCatalog, plan: RunPlan
    ) -> None:
        """The number of input samples must equal plan.n_steps."""
        with pytest.raises(ValueError, match="input samples"):
            simulate_run(model, catalog, plan, [ModelInputs()] * 5, 180.0)

    def test_needs_two_steps(self, model: ReactorModel, catalog: SensorCatalog) -> None:
        """A one-sample run has no interval to integrate."""
        one = RunPlan(0, "r", ScenarioKind.NORMAL, 1, datetime(2026, 1, 1, tzinfo=UTC), 1, (0,))
        with pytest.raises(ValueError, match="at least 2"):
            simulate_run(model, catalog, one, [ModelInputs()], 180.0)

    def test_dt_must_be_positive(
        self, model: ReactorModel, catalog: SensorCatalog, plan: RunPlan
    ) -> None:
        """dt_s <= 0 is rejected."""
        with pytest.raises(ValueError, match="dt_s"):
            simulate_run(model, catalog, plan, [ModelInputs()] * plan.n_steps, 0.0)


class TestMeasure:
    """Noise is white, per sensor, and reproducible from the run's generator."""

    def test_is_reproducible(self, sim: RunSimulation, catalog: SensorCatalog) -> None:
        """The same generator state gives the same measurement."""
        a = measure(sim, catalog, np.random.default_rng(5))
        b = measure(sim, catalog, np.random.default_rng(5))
        assert np.array_equal(a, b)

    def test_noise_has_the_catalog_sigma(self, sim: RunSimulation, catalog: SensorCatalog) -> None:
        """The empirical sigma of each channel is the catalog's, within sampling error."""
        long = dataclasses.replace(sim, true_values=np.zeros((20000, len(catalog.sensors))))
        noise = measure(long, catalog, np.random.default_rng(1))
        assert np.allclose(noise.std(axis=0), noise_sigmas(catalog), rtol=0.05)
        assert np.abs(noise.mean(axis=0)).max() < 0.05 * noise_sigmas(catalog).max()

    def test_zero_noise_mean_equals_truth(self, sim: RunSimulation, catalog: SensorCatalog) -> None:
        """Averaging many noisy draws recovers the true values."""
        reps = np.mean(
            [measure(sim, catalog, np.random.default_rng(s)) for s in range(200)], axis=0
        )
        assert np.allclose(reps, sim.true_values, atol=5 * noise_sigmas(catalog) / np.sqrt(200))


class TestRawCounts:
    """ADC counts come from SensorSpan.to_raw_counts, value by value."""

    def test_matches_the_span_method(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """Every cell equals the scalar span method."""
        counts = raw_counts_matrix(sim.true_values, catalog, spans)
        for j, tag in enumerate(catalog.tags):
            expected = [spans[tag].to_raw_counts(float(v)) for v in sim.true_values[:, j]]
            assert counts[:, j].tolist() == expected

    def test_clamps_outside_the_span(
        self, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """Values beyond the calibrated span saturate at 0 and 65535."""
        wild = np.zeros((2, len(catalog.sensors)))
        wild[0] = [spans[t].min - 1e6 for t in catalog.tags]
        wild[1] = [spans[t].max + 1e6 for t in catalog.tags]
        counts = raw_counts_matrix(wild, catalog, spans)
        assert (counts[0] == 0).all()
        assert (counts[1] == 65535).all()

    def test_missing_span_raises(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """A catalog sensor without span is an error, not a silently blind sensor (H7)."""
        partial = {k: v for k, v in spans.items() if k != "TC-HL-1"}
        with pytest.raises(KeyError, match="TC-HL-1"):
            raw_counts_matrix(sim.true_values, catalog, partial)

    def test_non_finite_values_raise(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """NaN cannot be scaled to an ADC count."""
        bad = sim.true_values.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            raw_counts_matrix(bad, catalog, spans)


class TestLongFrame:
    """The long frame carries the contract columns and the ground-truth labels."""

    def test_column_order(self, frame: pd.DataFrame) -> None:
        """Columns are the 11 contract ones followed by the labels."""
        assert tuple(frame.columns) == LONG_COLUMNS
        assert LONG_COLUMNS[: len(CONTRACT_COLUMNS)] == CONTRACT_COLUMNS
        assert LONG_COLUMNS[len(CONTRACT_COLUMNS) :] == LABEL_COLUMNS

    def test_row_count_and_order(self, frame: pd.DataFrame, catalog: SensorCatalog) -> None:
        """One row per (step, sensor), step-major, sensors in catalog order."""
        assert len(frame) == 30 * len(catalog.sensors)
        first_step = frame[frame["timestep"] == 0]
        assert first_step["sensor_id"].tolist() == list(catalog.tags)
        assert frame["timestep"].is_monotonic_increasing

    def test_reading_ids_are_unique_and_match_the_function(
        self, frame: pd.DataFrame, plan: RunPlan
    ) -> None:
        """reading_id is unique per row and equals sim_reading_uuid."""
        assert frame["reading_id"].is_unique
        row = frame.iloc[40]
        expected = sim_reading_uuid(
            PLANT, plan.run_id, row["sensor_id"], row["timestamp"].to_pydatetime()
        )
        assert row["reading_id"] == str(expected)

    def test_no_fault_labels(self, frame: pd.DataFrame) -> None:
        """Normal operation: no fault on either axis, no severity, no corruption."""
        assert (frame["process_fault"] == NO_FAULT).all()
        assert (frame["instrument_fault"] == NO_FAULT).all()
        assert frame["process_severity"].isna().all()
        assert frame["instrument_severity"].isna().all()
        assert frame["injected_at"].isna().all()
        assert frame["episode_id"].isna().all()
        assert not frame["corrupted"].any()
        assert (frame["scenario"] == "normal").all()

    def test_quality_and_usability(self, frame: pd.DataFrame) -> None:
        """quality is always good and is_usable True (the validator writes quality)."""
        assert (frame["quality"] == QualityFlag.GOOD.value).all()
        assert frame["is_usable"].all()

    def test_true_value_is_preserved(self, frame: pd.DataFrame, sim: RunSimulation) -> None:
        """true_value is the noise-free value and differs from the measurement."""
        assert np.array_equal(frame["true_value"].to_numpy(), sim.true_values.reshape(-1))
        assert (frame["value"] != frame["true_value"]).all()

    def test_dtypes(self, frame: pd.DataFrame) -> None:
        """Timestamps are tz-aware, counts are integers in the 16-bit range."""
        assert str(frame["timestamp"].dtype) == "datetime64[ns, UTC]"
        assert str(frame["injected_at"].dtype) == "datetime64[ns, UTC]"
        assert frame["raw_counts"].between(0, 65535).all()
        assert pd.api.types.is_integer_dtype(frame["raw_counts"])

    def test_timestamps_follow_the_cadence(self, frame: pd.DataFrame) -> None:
        """Samples are 180 s apart."""
        stamps = frame.drop_duplicates("timestep")["timestamp"]
        assert (stamps.diff().dropna() == pd.Timedelta(seconds=180)).all()

    def test_rebuilds_sensor_readings(self, frame: pd.DataFrame) -> None:
        """readings_from_frame validates every row against the canonical contract."""
        readings = list(readings_from_frame(frame))
        assert len(readings) == len(frame)
        assert all(isinstance(r, SensorReading) for r in readings)
        assert all(r.is_usable for r in readings)
        assert all(r.plant_id == PLANT for r in readings)

    def test_rejects_instrument_scenarios(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """Instrument-fault labels arrive with F3-M5: until then the builder refuses them."""
        faulty = dataclasses.replace(
            sim, plan=dataclasses.replace(sim.plan, scenario=ScenarioKind.STUCK)
        )
        with pytest.raises(NotImplementedError, match="stuck"):
            assemble_long_frame(faulty, sim.true_values, catalog, spans, PLANT)

    def test_rejects_wrong_shape(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """The measured matrix must match the run."""
        with pytest.raises(ValueError, match="shape"):
            assemble_long_frame(sim, sim.true_values[:5], catalog, spans, PLANT)

    def test_same_noise_stream_same_frame(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """Two builds with the same generator state are equal frames."""
        a = build_run_frame(sim, catalog, spans, PLANT, np.random.default_rng(3))
        b = build_run_frame(sim, catalog, spans, PLANT, np.random.default_rng(3))
        assert a.equals(b)


def _process_sim(
    sim: RunSimulation,
    kind: ScenarioKind,
    spec: ProcessFaultSpec,
) -> RunSimulation:
    """Return the run `sim` re-labelled as a process-fault run (labels come from the plan)."""
    plan = dataclasses.replace(sim.plan, scenario=kind, process_fault=spec)
    return dataclasses.replace(sim, plan=plan)


class TestProcessLabels:
    """A process fault labels EVERY sensor of each step in its window, from the start."""

    SPEC = ProcessFaultSpec(
        level=FaultLevel.MEDIUM, injected_step=10, onset_steps=2, hold_steps=5, recovery_steps=3
    )

    @pytest.fixture()
    def labelled(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> pd.DataFrame:
        """Return the long frame of an excursion run with the window 10..20."""
        faulty = _process_sim(sim, ScenarioKind.CORE_TEMP_EXCURSION, self.SPEC)
        return assemble_long_frame(faulty, sim.true_values, catalog, spans, PLANT)

    def test_window_covers_every_sensor_of_each_step(
        self, labelled: pd.DataFrame, catalog: SensorCatalog
    ) -> None:
        """Steps 10 to 20 inclusive carry the label on all sensors; the rest none."""
        active = labelled["timestep"].between(10, 20)
        assert (labelled.loc[active, "process_fault"] == "core_temp_excursion").all()
        assert (labelled.loc[~active, "process_fault"] == NO_FAULT).all()
        assert active.sum() == 11 * len(catalog.sensors)

    def test_severity_is_the_tdd_table(self, labelled: pd.DataFrame) -> None:
        """The severity inside the window is the static TDD 2.2 value."""
        active = labelled["timestep"].between(10, 20)
        assert (labelled.loc[active, "process_severity"] == "CRITICAL").all()
        assert labelled.loc[~active, "process_severity"].isna().all()

    def test_injected_at_is_the_exact_start(
        self, labelled: pd.DataFrame, sim: RunSimulation
    ) -> None:
        """injected_at is the timestamp of step 10 on every labelled row, NaT elsewhere."""
        active = labelled["timestep"].between(10, 20)
        assert (labelled.loc[active, "injected_at"] == pd.Timestamp(sim.timestamps[10])).all()
        assert labelled.loc[~active, "injected_at"].isna().all()
        first = labelled[labelled["process_fault"] != NO_FAULT]["timestep"].min()
        assert first == self.SPEC.injected_step

    def test_episode_id_and_corruption(self, labelled: pd.DataFrame) -> None:
        """One episode id per run; a process fault never marks a reading corrupted."""
        active = labelled["process_fault"] != NO_FAULT
        assert (labelled.loc[active, "episode_id"] == "sim_run_001:core_temp_excursion").all()
        assert labelled.loc[~active, "episode_id"].isna().all()
        assert not labelled["corrupted"].any()

    def test_instrument_axis_untouched(self, labelled: pd.DataFrame) -> None:
        """The instrument axis stays at none."""
        assert (labelled["instrument_fault"] == NO_FAULT).all()
        assert labelled["instrument_severity"].isna().all()

    def test_persistent_fault_labels_to_the_end(
        self, sim: RunSimulation, catalog: SensorCatalog, spans: dict[str, SensorSpan]
    ) -> None:
        """hold_steps=None keeps the label to the last step of the run."""
        spec = ProcessFaultSpec(level=FaultLevel.LOW, injected_step=20, onset_steps=1)
        faulty = _process_sim(sim, ScenarioKind.COOLANT_FLOW_REDUCTION, spec)
        frame = assemble_long_frame(faulty, sim.true_values, catalog, spans, PLANT)
        active = frame["process_fault"] != NO_FAULT
        assert frame.loc[active, "timestep"].min() == 20
        assert frame.loc[active, "timestep"].max() == sim.n_steps - 1
        assert (frame.loc[active, "process_severity"] == "HIGH").all()

    def test_labelled_frame_rebuilds_readings(self, labelled: pd.DataFrame) -> None:
        """The labelled frame still validates against the canonical contract."""
        readings = list(readings_from_frame(labelled))
        assert len(readings) == len(labelled)


class TestSubsteps:
    """Sub-stepped integration reproduces the coarse grid and validates its inputs."""

    @pytest.fixture()
    def constant(self) -> list[ModelInputs]:
        """Return 30 samples of a constant 90 % load."""
        return [ModelInputs(load=0.9)] * 30

    def test_matches_the_coarse_grid_on_equal_inputs(
        self,
        model: ReactorModel,
        catalog: SensorCatalog,
        plan: RunPlan,
        constant: list[ModelInputs],
    ) -> None:
        """Splitting each interval in three leaves the states where they were."""
        coarse = simulate_run(model, catalog, plan, constant, 180.0)
        fine = [ModelInputs(load=0.9)] * (29 * 3 + 1)
        split = simulate_run(model, catalog, plan, constant, 180.0, fine_inputs=fine, substeps=3)
        assert np.allclose(coarse.states, split.states, rtol=1e-9, atol=1e-9)
        assert split.max_energy_residual < 1e-6

    def test_energy_closes_over_the_whole_interval(
        self,
        model: ReactorModel,
        catalog: SensorCatalog,
        plan: RunPlan,
        constant: list[ModelInputs],
    ) -> None:
        """Each StepResult is one sampling interval, not one sub-interval."""
        fine = [ModelInputs(load=0.9)] * (29 * 4 + 1)
        sim_fine = simulate_run(model, catalog, plan, constant, 180.0, fine_inputs=fine, substeps=4)
        assert len(sim_fine.steps) == 29
        assert all(step.energy_residual_rel < 1e-6 for step in sim_fine.steps)

    @pytest.mark.parametrize("substeps", [0, -1])
    def test_rejects_non_positive_substeps(
        self,
        model: ReactorModel,
        catalog: SensorCatalog,
        plan: RunPlan,
        constant: list[ModelInputs],
        substeps: int,
    ) -> None:
        """substeps must be at least 1."""
        with pytest.raises(ValueError, match="substeps"):
            simulate_run(model, catalog, plan, constant, 180.0, substeps=substeps)

    def test_requires_the_fine_grid(
        self,
        model: ReactorModel,
        catalog: SensorCatalog,
        plan: RunPlan,
        constant: list[ModelInputs],
    ) -> None:
        """substeps > 1 without fine inputs, or with the wrong length, is an error."""
        with pytest.raises(ValueError, match="fine inputs"):
            simulate_run(model, catalog, plan, constant, 180.0, substeps=2)
        with pytest.raises(ValueError, match="fine inputs"):
            simulate_run(
                model, catalog, plan, constant, 180.0, fine_inputs=constant[:5], substeps=2
            )


class TestWideFrames:
    """The wide views are what the rules consume."""

    def test_wide_from_long_recovers_values(
        self, frame: pd.DataFrame, catalog: SensorCatalog
    ) -> None:
        """Pivoting the long frame restores one column per sensor, in catalog order."""
        wide = wide_frame_from_long(frame)
        assert list(wide.columns) == list(catalog.tags)
        assert isinstance(wide.index, pd.DatetimeIndex)
        assert wide.shape == (30, len(catalog.sensors))
        first = frame[frame["timestep"] == 0].set_index("sensor_id")["value"]
        assert np.allclose(wide.iloc[0].to_numpy(), first[list(catalog.tags)].to_numpy())

    def test_wide_from_long_true_value(self, frame: pd.DataFrame) -> None:
        """The pivoted column can be the noise-free value."""
        wide = wide_frame_from_long(frame, "true_value")
        assert wide.shape[0] == 30

    def test_wide_from_long_rejects_several_runs(self, frame: pd.DataFrame) -> None:
        """A frame holding two runs must be pivoted run by run."""
        other = frame.copy()
        other["run_id"] = "sim_run_002"
        with pytest.raises(ValueError, match="one run"):
            wide_frame_from_long(pd.concat([frame, other], ignore_index=True))

    def test_wide_from_matrix(self, sim: RunSimulation, catalog: SensorCatalog) -> None:
        """The matrix view carries the timestamps and tags."""
        wide = wide_frame_from_matrix(sim.true_values, catalog, sim.timestamps)
        assert wide.shape == sim.true_values.shape
        assert list(wide.columns) == list(catalog.tags)
        assert wide.index[1] - wide.index[0] == pd.Timedelta(seconds=180)

    def test_wide_from_matrix_shape_mismatch(
        self, sim: RunSimulation, catalog: SensorCatalog
    ) -> None:
        """Values and timestamps must agree."""
        with pytest.raises(ValueError, match="expected"):
            wide_frame_from_matrix(sim.true_values[:3], catalog, sim.timestamps)


class TestParquet:
    """The dataset layout is TDD 4.4 and the file round-trips."""

    def test_path_layout(self, tmp_path: Path) -> None:
        """fault_type=<scenario>/seed=<n>/<run_id>.parquet under the root."""
        path = run_parquet_path(tmp_path, ScenarioKind.STUCK, 42, "sim_run_001")
        assert path == tmp_path / "fault_type=stuck" / "seed=42" / "sim_run_001.parquet"

    def test_round_trip(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        """Writing creates directories and reading returns the same frame."""
        path = write_run_parquet(frame, tmp_path / "a" / "b" / "run.parquet")
        assert path.exists()
        back = pd.read_parquet(path)
        assert back.equals(frame)

    def test_bytes_are_reproducible(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        """The same frame written twice gives identical bytes."""
        a = write_run_parquet(frame, tmp_path / "a.parquet").read_bytes()
        b = write_run_parquet(frame, tmp_path / "b.parquet").read_bytes()
        assert a == b
