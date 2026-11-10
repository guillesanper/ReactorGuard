"""Unit tests for ml.training.sim_dataset (F4-M1).

Cubren el contrato de columnas (orden del catalogo, no del parquet), las etiquetas
agregadas por instante (episodios solapados, severidad y nivel maximos, pares de
entrenamiento de D4-6), el reparto por run de D4-7, la verificacion del manifiesto
y la escritura determinista. Los datos son frames largos sinteticos de
`tests/support/sim_frames.py`, sin simulador.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.simulation.dataset_builder import MANIFEST_NAME
from data.simulation.scenario import ScenarioKind
from data.simulation.sensor_catalog import SensorCatalog
from ml.training.sim_dataset import (
    ASSIGNMENT_NAME,
    CRITICAL,
    CRITICAL_KINDS,
    EPISODE_SEPARATOR,
    ID_COLUMNS,
    INSTANT_LABEL_COLUMNS,
    SPLITS,
    ManifestRun,
    PreparedDataset,
    assign_splits,
    critical_readings,
    fit_zscore,
    instant_columns,
    prepare_dataset,
    read_manifest,
    run_instants,
    true_columns,
    value_columns,
    verified_run_path,
    write_metrics,
    write_prepared,
)
from tests.support.sim_frames import (
    TRUE_OFFSET,
    LabelSpan,
    build_catalog,
    long_run_frame,
    synthetic_value,
    write_long_dataset,
)

N_STEPS = 8


@pytest.fixture(scope="module")
def catalog() -> SensorCatalog:
    """Return the versioned sensor catalog (27 sensors)."""
    return build_catalog()


def _run(run_id: str, scenario: ScenarioKind) -> ManifestRun:
    """Return a manifest entry with a dummy path and digest."""
    return ManifestRun(run_id, scenario, Path(f"{run_id}.parquet"), "0" * 64)


# ----------------------------------------------------------------------
# Contrato de columnas
# ----------------------------------------------------------------------


class TestColumnContract:
    """The wide frame has a fixed column order taken from the catalog."""

    def test_instant_columns_order(self) -> None:
        """Identity, values, true values, then labels."""
        tags = ["B", "A"]
        assert instant_columns(tags) == [
            *ID_COLUMNS,
            "value:B",
            "value:A",
            "true:B",
            "true:A",
            *INSTANT_LABEL_COLUMNS,
        ]

    def test_value_and_true_columns_keep_the_given_order(self) -> None:
        """The prefixes are applied tag by tag."""
        assert value_columns(["X", "Y"]) == ["value:X", "value:Y"]
        assert true_columns(["X", "Y"]) == ["true:X", "true:Y"]

    def test_critical_kinds_follow_the_static_severity_table(self) -> None:
        """D4-10: stuck, excursion and void are CRITICAL; drift is not."""
        assert set(CRITICAL_KINDS) == {
            ScenarioKind.STUCK,
            ScenarioKind.CORE_TEMP_EXCURSION,
            ScenarioKind.VOID_FORMATION,
        }
        assert ScenarioKind.DRIFT not in CRITICAL_KINDS


# ----------------------------------------------------------------------
# Manifiesto
# ----------------------------------------------------------------------


class TestReadManifest:
    """The manifest of the simulate stage is parsed and validated."""

    def _write(self, root: Path, document: object) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / MANIFEST_NAME).write_text(json.dumps(document), encoding="utf-8")
        return root

    def _entry(self, run_id: str, scenario: str = "normal") -> dict[str, str]:
        return {"run_id": run_id, "scenario": scenario, "path": f"{run_id}.parquet", "sha256": "a"}

    def test_runs_are_sorted_numerically(self, tmp_path: Path) -> None:
        """run_10 comes after run_9 even though it sorts before it as text."""
        root = self._write(
            tmp_path, {"version": 1, "runs": [self._entry("run_10"), self._entry("run_9")]}
        )
        runs = read_manifest(root)
        assert [r.run_id for r in runs] == ["run_9", "run_10"]
        assert runs[0] == ManifestRun("run_9", ScenarioKind.NORMAL, Path("run_9.parquet"), "a")

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """Without a manifest the simulate stage has not run."""
        with pytest.raises(FileNotFoundError, match="simulate stage"):
            read_manifest(tmp_path)

    def test_unsupported_version(self, tmp_path: Path) -> None:
        """Only the current manifest version is read."""
        root = self._write(tmp_path, {"version": 99, "runs": [self._entry("r1")]})
        with pytest.raises(ValueError, match="version 99"):
            read_manifest(root)

    @pytest.mark.parametrize("field", ["run_id", "scenario", "path", "sha256"])
    def test_missing_field(self, tmp_path: Path, field: str) -> None:
        """Every run entry carries the four fields."""
        entry = self._entry("r1")
        del entry[field]
        root = self._write(tmp_path, {"version": 1, "runs": [entry]})
        with pytest.raises(KeyError, match=f"'{field}'"):
            read_manifest(root)

    def test_repeated_run(self, tmp_path: Path) -> None:
        """A run id cannot appear twice."""
        root = self._write(tmp_path, {"version": 1, "runs": [self._entry("r1")] * 2})
        with pytest.raises(ValueError, match="appears twice"):
            read_manifest(root)

    def test_unknown_scenario(self, tmp_path: Path) -> None:
        """A scenario outside ScenarioKind is rejected."""
        root = self._write(tmp_path, {"version": 1, "runs": [self._entry("r1", "meltdown")]})
        with pytest.raises(ValueError, match="unknown scenario 'meltdown'"):
            read_manifest(root)

    def test_empty_manifest(self, tmp_path: Path) -> None:
        """A manifest without runs is an error, not an empty dataset."""
        root = self._write(tmp_path, {"version": 1, "runs": []})
        with pytest.raises(ValueError, match="lists no run"):
            read_manifest(root)


class TestVerifiedRunPath:
    """Each parquet is checked against the manifest digest."""

    def test_matching_digest_returns_the_path(self, tmp_path: Path, catalog: SensorCatalog) -> None:
        """An untouched dataset verifies."""
        frame = long_run_frame(catalog, "sim_run_001", ScenarioKind.NORMAL, 3)
        write_long_dataset(tmp_path, [frame])
        (run,) = read_manifest(tmp_path)
        assert verified_run_path(tmp_path, run) == tmp_path / run.path

    def test_missing_parquet(self, tmp_path: Path) -> None:
        """A run listed but absent fails."""
        with pytest.raises(FileNotFoundError, match="parquet not found"):
            verified_run_path(tmp_path, _run("r1", ScenarioKind.NORMAL))

    def test_modified_parquet(self, tmp_path: Path, catalog: SensorCatalog) -> None:
        """A parquet that changed after the manifest was written fails."""
        frame = long_run_frame(catalog, "sim_run_001", ScenarioKind.NORMAL, 3)
        write_long_dataset(tmp_path, [frame])
        (run,) = read_manifest(tmp_path)
        frame.assign(value=frame["value"] + 1.0).to_parquet(tmp_path / run.path, index=False)
        with pytest.raises(ValueError, match="no es el que describe su manifiesto"):
            verified_run_path(tmp_path, run)


# ----------------------------------------------------------------------
# Reparto por run (D4-7)
# ----------------------------------------------------------------------


class TestAssignSplits:
    """Whole runs go to one partition, stratified by scenario."""

    def test_last_run_to_test_and_penultimate_to_val(self) -> None:
        """Per scenario: train..., val, test in run-id order."""
        runs = [_run(f"sim_run_00{i}", ScenarioKind.STUCK) for i in (3, 1, 2)]
        assert assign_splits(runs, 1, 1) == {
            "sim_run_001": "train",
            "sim_run_002": "val",
            "sim_run_003": "test",
        }

    def test_four_normal_runs_split_two_one_one(self) -> None:
        """The normal scenario of the dataset gives 2/1/1."""
        runs = [_run(f"r{i}", ScenarioKind.NORMAL) for i in range(4)]
        assignment = assign_splits(runs, 1, 1)
        assert [assignment[f"r{i}"] for i in range(4)] == ["train", "train", "val", "test"]

    def test_scenarios_are_split_independently(self) -> None:
        """Interleaved run ids of two scenarios do not mix."""
        runs = [
            _run("r1", ScenarioKind.DRIFT),
            _run("r2", ScenarioKind.BIAS),
            _run("r3", ScenarioKind.DRIFT),
            _run("r4", ScenarioKind.BIAS),
            _run("r5", ScenarioKind.DRIFT),
            _run("r6", ScenarioKind.BIAS),
        ]
        assignment = assign_splits(runs, 1, 1)
        assert assignment == {
            "r1": "train",
            "r3": "val",
            "r5": "test",
            "r2": "train",
            "r4": "val",
            "r6": "test",
        }

    def test_numeric_order_of_run_ids(self) -> None:
        """r10 is the last run, not r9."""
        runs = [_run(f"r{i}", ScenarioKind.NORMAL) for i in (8, 9, 10)]
        assert assign_splits(runs, 1, 1)["r10"] == "test"

    def test_larger_counts(self) -> None:
        """Two val and two test runs leave the rest in train."""
        runs = [_run(f"r{i}", ScenarioKind.NORMAL) for i in range(6)]
        assignment = assign_splits(runs, 2, 2)
        assert [assignment[f"r{i}"] for i in range(6)] == [
            "train",
            "train",
            "val",
            "val",
            "test",
            "test",
        ]

    def test_every_partition_gets_every_scenario(self) -> None:
        """Stratification: each scenario reaches train, val and test."""
        runs = [_run(f"{k.value}_{i}", k) for k in ScenarioKind for i in range(3)]
        assignment = assign_splits(runs, 1, 1)
        for kind in ScenarioKind:
            splits = {assignment[f"{kind.value}_{i}"] for i in range(3)}
            assert splits == set(SPLITS)

    def test_too_few_runs_in_a_scenario(self) -> None:
        """Train must keep at least one run of each scenario."""
        runs = [_run("r1", ScenarioKind.VOID_FORMATION), _run("r2", ScenarioKind.VOID_FORMATION)]
        with pytest.raises(
            ValueError, match="void_formation has 2 runs; the split needs at least 3"
        ):
            assign_splits(runs, 1, 1)

    @pytest.mark.parametrize(("val", "test"), [(0, 1), (1, 0), (-1, 1)])
    def test_counts_below_one(self, val: int, test: int) -> None:
        """Every partition gets at least one run per scenario."""
        runs = [_run(f"r{i}", ScenarioKind.NORMAL) for i in range(4)]
        with pytest.raises(ValueError, match="must be >= 1"):
            assign_splits(runs, val, test)


# ----------------------------------------------------------------------
# Un run: frame largo -> instantes
# ----------------------------------------------------------------------


class TestRunInstantsValues:
    """Values land in the catalog-ordered columns, one row per step."""

    def test_one_row_per_step_in_catalog_order(self, catalog: SensorCatalog) -> None:
        """The frame has the full column contract and N_STEPS rows."""
        frame = long_run_frame(catalog, "r1", ScenarioKind.NORMAL, N_STEPS)
        instants = run_instants(frame, catalog)
        assert list(instants.columns) == instant_columns(catalog.tags)
        assert instants["timestep"].tolist() == list(range(N_STEPS))

    def test_order_comes_from_the_catalog_not_the_parquet(self, catalog: SensorCatalog) -> None:
        """Shuffled rows give the same wide frame."""
        frame = long_run_frame(catalog, "r1", ScenarioKind.NORMAL, N_STEPS)
        shuffled = frame.sample(frac=1.0, random_state=0)
        pd.testing.assert_frame_equal(run_instants(shuffled, catalog), run_instants(frame, catalog))

    def test_measured_and_true_values(self, catalog: SensorCatalog) -> None:
        """value:<tag> holds the measured value and true:<tag> the true one."""
        frame = long_run_frame(catalog, "r1", ScenarioKind.NORMAL, N_STEPS)
        instants = run_instants(frame, catalog)
        for index, tag in enumerate(catalog.tags):
            expected = [synthetic_value(index, step) for step in range(N_STEPS)]
            np.testing.assert_allclose(instants[f"value:{tag}"], expected)
            np.testing.assert_allclose(instants[f"true:{tag}"], np.array(expected) - TRUE_OFFSET)

    def test_identity_and_timestamps(self, catalog: SensorCatalog) -> None:
        """run_id, scenario and the UTC timestamp of each step are kept."""
        frame = long_run_frame(catalog, "sim_run_007", ScenarioKind.CORE_TEMP_EXCURSION, 3)
        instants = run_instants(frame, catalog)
        assert set(instants["run_id"]) == {"sim_run_007"}
        assert set(instants["scenario"]) == {"core_temp_excursion"}
        assert str(instants["timestamp"].dt.tz) == "UTC"
        assert instants["timestamp"].diff().dropna().dt.total_seconds().tolist() == [180.0, 180.0]


class TestRunInstantsLabels:
    """Labels are aggregated per instant."""

    def test_unlabelled_run(self, catalog: SensorCatalog) -> None:
        """Normal operation: no label, every pair but the last is trainable."""
        frame = long_run_frame(catalog, "r1", ScenarioKind.NORMAL, N_STEPS)
        instants = run_instants(frame, catalog)
        assert not instants["labelled"].any()
        assert instants["severity"].isna().all()
        assert instants["fault_level"].isna().all()
        assert instants["episode_ids"].isna().all()
        assert set(instants["process_fault"]) == {"none"}
        assert set(instants["instrument_fault"]) == {"none"}
        assert instants["n_episodes"].tolist() == [0] * N_STEPS
        assert instants["trainable_pair"].tolist() == [True] * (N_STEPS - 1) + [False]

    def test_process_fault_labels_every_sensor(self, catalog: SensorCatalog) -> None:
        """A void episode labels the instant on the process axis as CRITICAL."""
        frame = long_run_frame(
            catalog, "r1", ScenarioKind.VOID_FORMATION, N_STEPS, [LabelSpan(3, 4, "medium")]
        )
        instants = run_instants(frame, catalog).set_index("timestep")
        assert instants["labelled"].tolist() == [False] * 3 + [True] * 2 + [False] * 3
        assert instants.loc[3, "labelled_sensors"] == len(catalog.tags)
        assert instants.loc[3, "severity"] == CRITICAL
        assert instants.loc[3, "process_fault"] == "void_formation"
        assert instants.loc[3, "instrument_fault"] == "none"
        assert instants.loc[4, "fault_level"] == "medium"
        assert instants.loc[4, "episode_ids"] == "r1:ep0"
        assert instants.loc[2, "process_fault"] == "none"

    def test_trainable_pair_needs_both_instants_unlabelled(self, catalog: SensorCatalog) -> None:
        """D4-6: (t, t+1) is excluded if either instant carries a label."""
        frame = long_run_frame(
            catalog, "r1", ScenarioKind.CORE_TEMP_EXCURSION, N_STEPS, [LabelSpan(3, 4)]
        )
        instants = run_instants(frame, catalog)
        assert instants["trainable_pair"].tolist() == [
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
        ]

    def test_overlapping_instrument_episodes(self, catalog: SensorCatalog) -> None:
        """Two episodes on different sensors keep both ids and the maximum level."""
        first, second = catalog.tags[0], catalog.tags[5]
        spans = [
            LabelSpan(1, 4, "low", (first,), "b"),
            LabelSpan(3, 6, "high", (second,), "a"),
        ]
        frame = long_run_frame(catalog, "r1", ScenarioKind.DRIFT, N_STEPS, spans)
        instants = run_instants(frame, catalog).set_index("timestep")
        assert instants.loc[3, "n_episodes"] == 2
        assert instants.loc[3, "episode_ids"] == EPISODE_SEPARATOR.join(["r1:a", "r1:b"])
        assert instants.loc[3, "fault_level"] == "high"
        assert instants.loc[3, "labelled_sensors"] == 2
        assert instants.loc[2, "fault_level"] == "low"
        assert instants.loc[2, "n_episodes"] == 1
        assert instants.loc[3, "instrument_fault"] == "drift"
        assert instants.loc[3, "process_fault"] == "none"
        assert instants.loc[3, "severity"] == "HIGH"
        assert instants.loc[7, "n_episodes"] == 0

    def test_critical_readings_count(self, catalog: SensorCatalog) -> None:
        """Only readings with the CRITICAL severity count."""
        spans = [LabelSpan(0, 1, tags=(catalog.tags[0], catalog.tags[1]))]
        stuck = long_run_frame(catalog, "r1", ScenarioKind.STUCK, N_STEPS, spans)
        drift = long_run_frame(catalog, "r2", ScenarioKind.DRIFT, N_STEPS, spans)
        assert critical_readings(stuck) == 4
        assert critical_readings(drift) == 0


class TestRunInstantsRejects:
    """A frame that is not one complete run of the catalog fails."""

    def _frame(self, catalog: SensorCatalog) -> pd.DataFrame:
        return long_run_frame(catalog, "r1", ScenarioKind.NORMAL, 4)

    def test_empty_frame(self, catalog: SensorCatalog) -> None:
        """An empty frame is not a run."""
        with pytest.raises(ValueError, match="non-empty"):
            run_instants(self._frame(catalog).iloc[:0], catalog)

    def test_two_runs(self, catalog: SensorCatalog) -> None:
        """One run per call."""
        other = long_run_frame(catalog, "r2", ScenarioKind.NORMAL, 4)
        with pytest.raises(ValueError, match="exactly one run"):
            run_instants(pd.concat([self._frame(catalog), other]), catalog)

    def test_missing_sensor(self, catalog: SensorCatalog) -> None:
        """A catalog sensor absent from the run is named."""
        frame = self._frame(catalog)
        tag = catalog.tags[2]
        with pytest.raises(ValueError, match=f"missing \\['{tag}'\\]"):
            run_instants(frame[frame["sensor_id"] != tag], catalog)

    def test_extra_sensor(self, catalog: SensorCatalog) -> None:
        """A sensor outside the catalog is named."""
        frame = self._frame(catalog)
        renamed = frame.assign(
            sensor_id=frame["sensor_id"].replace(catalog.tags[0], "XX-UNKNOWN-1")
        )
        with pytest.raises(ValueError, match="extra \\['XX-UNKNOWN-1'\\]"):
            run_instants(renamed, catalog)

    def test_missing_reading(self, catalog: SensorCatalog) -> None:
        """A (step, sensor) pair absent breaks the step x sensor count."""
        with pytest.raises(ValueError, match="readings from step 0"):
            run_instants(self._frame(catalog).iloc[1:], catalog)

    def test_run_not_starting_at_step_zero(self, catalog: SensorCatalog) -> None:
        """Steps are 0..n-1."""
        frame = self._frame(catalog)
        with pytest.raises(ValueError, match="readings from step 0"):
            run_instants(frame.assign(timestep=frame["timestep"] + 1), catalog)

    def test_repeated_reading(self, catalog: SensorCatalog) -> None:
        """A duplicated reading replacing a missing one is caught."""
        frame = self._frame(catalog)
        broken = pd.concat([frame.iloc[1:], frame.iloc[[2]]], ignore_index=True)
        with pytest.raises(ValueError, match="repeated"):
            run_instants(broken, catalog)

    @pytest.mark.parametrize("column", ["value", "true_value"])
    def test_non_finite_value(self, catalog: SensorCatalog, column: str) -> None:
        """NaN or infinite readings never reach the PINN."""
        frame = self._frame(catalog)
        frame.loc[5, column] = np.nan
        with pytest.raises(ValueError, match=f"column {column} holds non-finite"):
            run_instants(frame, catalog)

    def test_two_kinds_on_one_axis(self, catalog: SensorCatalog) -> None:
        """D3: a run carries one fault kind per axis."""
        frame = long_run_frame(catalog, "r1", ScenarioKind.DRIFT, 4, [LabelSpan(0, 3)])
        frame.loc[0, "instrument_fault"] = "bias"
        with pytest.raises(ValueError, match="several instrument_fault kinds"):
            run_instants(frame, catalog)


# ----------------------------------------------------------------------
# Dataset completo
# ----------------------------------------------------------------------

_SCENARIO_RUNS: dict[ScenarioKind, int] = {
    ScenarioKind.NORMAL: 4,
    ScenarioKind.STUCK: 3,
    ScenarioKind.CORE_TEMP_EXCURSION: 3,
    ScenarioKind.DRIFT: 3,
}
"""Runs por escenario del dataset sintetico (13 runs)."""


def _dataset_frames(catalog: SensorCatalog) -> list[pd.DataFrame]:
    """Return 13 runs: faults labelled at steps 2..3 (stuck on one sensor)."""
    frames: list[pd.DataFrame] = []
    index = 1
    for scenario, n_runs in _SCENARIO_RUNS.items():
        for _ in range(n_runs):
            spans: list[LabelSpan] = []
            if scenario is ScenarioKind.STUCK:
                spans = [LabelSpan(2, 3, "low", (catalog.tags[0],))]
            elif scenario is not ScenarioKind.NORMAL:
                spans = [LabelSpan(2, 3, "high")]
            frames.append(long_run_frame(catalog, f"sim_run_{index:03d}", scenario, N_STEPS, spans))
            index += 1
    return frames


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory, catalog: SensorCatalog) -> Path:
    """Write the synthetic dataset once for the module."""
    root = tmp_path_factory.mktemp("simulation")
    write_long_dataset(root, _dataset_frames(catalog))
    return root


@pytest.fixture(scope="module")
def prepared(dataset_dir: Path, catalog: SensorCatalog) -> PreparedDataset:
    """Prepare the synthetic dataset with 1 val and 1 test run per scenario."""
    return prepare_dataset(
        dataset_dir,
        catalog,
        val_per_scenario=1,
        test_per_scenario=1,
        k_min_readings=1,
        n_min_episodes=1,
    )


class TestPrepareDataset:
    """The dataset is split by run and summarised per partition."""

    def test_assignment(self, prepared: PreparedDataset) -> None:
        """Normal 2/1/1 and 1/1/1 for each fault scenario."""
        by_split = {s: sorted(r for r, p in prepared.assignment.items() if p == s) for s in SPLITS}
        assert by_split == {
            "train": ["sim_run_001", "sim_run_002", "sim_run_005", "sim_run_008", "sim_run_011"],
            "val": ["sim_run_003", "sim_run_006", "sim_run_009", "sim_run_012"],
            "test": ["sim_run_004", "sim_run_007", "sim_run_010", "sim_run_013"],
        }

    def test_frames_hold_whole_runs_in_order(self, prepared: PreparedDataset) -> None:
        """Each partition is the concatenation of its runs, in run order."""
        for split in SPLITS:
            frame = prepared.frames[split]
            runs = [r for r, p in prepared.assignment.items() if p == split]
            assert list(dict.fromkeys(frame["run_id"])) == sorted(runs)
            assert len(frame) == len(runs) * N_STEPS
            assert frame.index.tolist() == list(range(len(frame)))

    def test_no_run_in_two_partitions(self, prepared: PreparedDataset) -> None:
        """No temporal or episode leakage across partitions."""
        seen = [set(prepared.frames[s]["run_id"]) for s in SPLITS]
        assert not (seen[0] & seen[1]) and not (seen[0] & seen[2]) and not (seen[1] & seen[2])

    def test_summary_counts(self, prepared: PreparedDataset) -> None:
        """Counts follow the spans: steps 2..3 labelled in each fault run."""
        train = prepared.summary("train")
        # train: 2 normal, 1 stuck, 1 excursion, 1 drift.
        assert (train.runs, train.instants) == (5, 5 * N_STEPS)
        assert train.labelled_instants == 3 * 2
        # stuck (2 instants) + excursion (2 instants); drift is HIGH.
        assert train.critical_instants == 4
        # stuck: 2 readings on one sensor; excursion: 2 steps x every sensor.
        assert train.critical_readings == 2 + 2 * len(prepared.tags)
        # normal: 7 pairs each; fault runs: 7 - 3 excluded pairs.
        assert train.trainable_pairs == 2 * 7 + 3 * 4
        assert train.grid_size == 144

    def test_coverage_is_informative(self, prepared: PreparedDataset) -> None:
        """Process faults cover their level for every sensor group; stuck one cell."""
        summary = prepared.summary("val")
        groups = len({s.sensor_type for s in build_catalog().sensors})
        # excursion high: one cell per group; drift high: one per group; stuck low: one.
        assert summary.covered_modes == 2 * groups + 1
        assert summary.coverage_fraction == pytest.approx(summary.covered_modes / 144)

    def test_unknown_split(self, prepared: PreparedDataset) -> None:
        """Only train, val and test exist."""
        with pytest.raises(KeyError, match="unknown split 'holdout'"):
            prepared.summary("holdout")

    def test_metrics_document(self, prepared: PreparedDataset) -> None:
        """Flat per-partition figures, the CRITICAL kinds and the coverage rule."""
        metrics = prepared.metrics()
        assert metrics["version"] == 1
        assert metrics["n_sensors"] == len(prepared.tags)
        assert metrics["critical_kinds"] == ["stuck", "core_temp_excursion", "void_formation"]
        assert metrics["coverage"] == {"grid_size": 144, "k_min_readings": 1, "n_min_episodes": 1}
        assert set(metrics["splits"]) == set(SPLITS)
        assert metrics["splits"]["train"]["critical_instants"] == 4
        json.dumps(metrics, allow_nan=False)

    def test_assignment_document(self, prepared: PreparedDataset) -> None:
        """Run -> scenario and split, plus the column order."""
        document = prepared.assignment_document()
        assert document["tags"] == list(build_catalog().tags)
        assert document["runs"]["sim_run_004"] == {"scenario": "normal", "split": "test"}
        assert (document["val_runs_per_scenario"], document["test_runs_per_scenario"]) == (1, 1)
        assert len(document["runs"]) == 13

    def test_parquet_holding_another_run(self, tmp_path: Path, catalog: SensorCatalog) -> None:
        """The manifest entry and the parquet must name the same run."""
        frames = _dataset_frames(catalog)
        manifest = write_long_dataset(tmp_path, frames)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["runs"][0]["run_id"] = "sim_run_000"
        manifest.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match="holds run sim_run_001 \\(normal\\), the manifest"):
            prepare_dataset(
                tmp_path,
                catalog,
                val_per_scenario=1,
                test_per_scenario=1,
                k_min_readings=1,
                n_min_episodes=1,
            )


# ----------------------------------------------------------------------
# Normalizacion z
# ----------------------------------------------------------------------


class TestFitZscore:
    """The z-score is fitted on unlabelled instants only."""

    def test_labelled_instants_are_ignored(self, catalog: SensorCatalog) -> None:
        """A huge value on a labelled instant does not move the statistics."""
        frame = long_run_frame(catalog, "r1", ScenarioKind.DRIFT, N_STEPS, [LabelSpan(2, 3)])
        instants = run_instants(frame, catalog)
        instants.loc[instants["labelled"], "value:" + catalog.tags[0]] = 1.0e9
        stats = fit_zscore(instants, catalog.tags)
        steps = [s for s in range(N_STEPS) if s not in (2, 3)]
        expected = np.array([synthetic_value(0, s) for s in steps])
        assert stats.tags == catalog.tags
        assert stats.mean[0] == pytest.approx(expected.mean())
        assert stats.std[0] == pytest.approx(expected.std())

    def test_constant_sensor(self, catalog: SensorCatalog) -> None:
        """A sensor without variance cannot be normalised."""
        instants = run_instants(long_run_frame(catalog, "r1", ScenarioKind.NORMAL, 4), catalog)
        instants["value:" + catalog.tags[3]] = 7.0
        with pytest.raises(ValueError, match=catalog.tags[3]):
            fit_zscore(instants, catalog.tags)

    def test_no_unlabelled_instant(self, catalog: SensorCatalog) -> None:
        """A partition without healthy instants cannot be fitted."""
        frame = long_run_frame(catalog, "r1", ScenarioKind.DRIFT, 4, [LabelSpan(0, 3)])
        with pytest.raises(ValueError, match="at least one unlabelled"):
            fit_zscore(run_instants(frame, catalog), catalog.tags)


# ----------------------------------------------------------------------
# Escritura
# ----------------------------------------------------------------------


class TestWrite:
    """Outputs round-trip and are byte-for-byte reproducible."""

    def test_written_files(self, tmp_path: Path, prepared: PreparedDataset) -> None:
        """Three parquet files and the assignment."""
        written = write_prepared(prepared, tmp_path / "out")
        assert [p.name for p in written] == [
            "train.parquet",
            "val.parquet",
            "test.parquet",
            ASSIGNMENT_NAME,
        ]

    def test_parquet_round_trip(self, tmp_path: Path, prepared: PreparedDataset) -> None:
        """Reading a partition back gives the same frame, dtypes included."""
        write_prepared(prepared, tmp_path)
        for split in SPLITS:
            back = pd.read_parquet(tmp_path / f"{split}.parquet")
            pd.testing.assert_frame_equal(back, prepared.frames[split])

    def test_two_writes_give_the_same_bytes(
        self, tmp_path: Path, prepared: PreparedDataset
    ) -> None:
        """No clock or path in any output."""
        first = write_prepared(prepared, tmp_path / "a")
        second = write_prepared(prepared, tmp_path / "b")
        for a, b in zip(first, second, strict=True):
            assert a.read_bytes() == b.read_bytes()

    def test_metrics_json(self, tmp_path: Path, prepared: PreparedDataset) -> None:
        """Sorted keys, trailing newline, same content as metrics()."""
        path = write_metrics(prepared, tmp_path / "m" / "metrics.json")
        text = path.read_text(encoding="utf-8")
        assert text.endswith("}\n")
        assert json.loads(text) == prepared.metrics()
        assert str(tmp_path) not in text
